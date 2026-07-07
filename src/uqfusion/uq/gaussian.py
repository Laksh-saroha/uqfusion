"""Gaussian σ² head + β-NLL loss inside Ultralytics (O2 — scope §6.2, §7.1; plan B1).

Design (decision D1, option (c) of scope §7.2): the σ² branch (`cv4`) is ADDED
alongside the untouched DFL box path. Conversion mutates an already-loaded
`Detect` head in place — new `cv4` branch, then a class swap to
`GaussianDetect` — so pretrained weights, head-structure variants (`legacy`
flag), and checkpoint pickling stay consistent without touching a line of
Ultralytics code. Verified against the pinned ultralytics 8.4.90 internals.

Gradient policy (plan B1; answer A4-11's "keep only if it doesn't hurt"):
by default the σ branch reads DETACHED neck features and the NLL sees a
DETACHED μ, so the deterministic detector receives gradients bit-identical to
baseline — detection parity holds by construction, not by hope. Both detaches
are config flags (`gaussian:` block) so the Phase 2 ablation can relax them.

Warm-up (scope §6.2 / R10, R11): the NLL term's weight is 0 for
`warmup_epochs`, then ramps linearly over `ramp_epochs`. With μ detached, the
NLL is cv4's only gradient source, so zero weight IS the σ-freeze — no
parameter-group surgery needed. (Weight decay still nudges cv4 during warm-up;
negligible and noted.)

σ parameterization: per-anchor log σ² over the four DFL regression targets —
LTRB distances in stride units (matching `bbox2dist`), converted to pixels at
inference by multiplying exp(½·logvar) with the anchor's stride.
"""

from __future__ import annotations

import torch
import torch.nn as nn
from ultralytics.nn.modules import Detect
from ultralytics.nn.modules.conv import Conv
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils.loss import v8DetectionLoss
from ultralytics.utils.tal import bbox2dist

LOGVAR_MIN, LOGVAR_MAX = -12.0, 8.0  # clamp in stride-unit log-variance space

DEFAULT_GAUSSIAN_CFG = {
    "beta": 0.5,                    # β-NLL exponent (R10); 0 = plain NLL
    "nll_gain": 1.0,                # weight of the NLL term in the summed loss
    "warmup_epochs": 5,             # μ trains alone first (σ frozen via zero NLL weight)
    "ramp_epochs": 5,               # linear NLL-weight ramp after warm-up
    "sigma_detach_features": True,  # cv4 reads detached neck features
    "nll_detach_mu": True,          # NLL trains σ only; μ stays owned by DFL/CIoU
}


class GaussianDetect(Detect):
    """Detect head with a parallel per-coordinate log-variance branch (cv4).

    Instances are produced by `convert_to_gaussian` via in-place class swap;
    this class deliberately defines no __init__ of its own.
    """

    output_sigma = False           # inference: append σ (pixels) after class scores
    output_dfl_unc = False         # inference: also append DFL-distribution std (pixels) — §7.2 ablation row
    sigma_detach_features = True   # gradient isolation of the σ branch from the trunk

    def forward_head(self, x, box_head=None, cls_head=None):
        """Standard head outputs plus `logvars` (bs, 4, anchors) on the one-to-many path."""
        preds = Detect.forward_head(self, x, box_head=box_head, cls_head=cls_head)
        if preds and getattr(self, "cv4", None) is not None and box_head is self.cv2:
            bs = x[0].shape[0]
            xs = [xi.detach() for xi in x] if self.sigma_detach_features else x
            preds["logvars"] = torch.cat(
                [self.cv4[i](xs[i]).view(bs, 4, -1) for i in range(self.nl)], dim=2
            )
        return preds

    def _inference(self, x):
        """Standard decoded output; σ channels appended only when output_sigma is set.

        Off by default so the stock predictor/validator (which infer nc from the
        channel count) keep working during training-time validation. The UQ
        inference path (uqfusion.uq.infer) enables it and passes nc explicitly.
        """
        y = Detect._inference(self, x)
        if self.output_sigma and "logvars" in x:
            sigma = (0.5 * x["logvars"].clamp(LOGVAR_MIN, LOGVAR_MAX)).exp() * self.strides  # px, LTRB
            y = torch.cat((y, sigma), dim=1)
        if self.output_dfl_unc and self.reg_max > 1:
            # DFL-derived uncertainty (scope §7.2 option (a), GFLv2/R2): std of the
            # per-coordinate bin distribution — same trained model, zero extra params.
            bs, _, a = x["boxes"].shape
            probs = x["boxes"].view(bs, 4, self.reg_max, a).softmax(2)
            bins = torch.arange(self.reg_max, device=probs.device, dtype=probs.dtype).view(1, 1, -1, 1)
            mean = (probs * bins).sum(2, keepdim=True)
            dfl_std = ((probs * (bins - mean) ** 2).sum(2)).clamp_min(0).sqrt() * self.strides  # px, LTRB
            y = torch.cat((y, dfl_std), dim=1)
        return y


class GaussianDetectionLoss(v8DetectionLoss):
    """box/cls/dfl (untouched, incl. target assignment) + β-NLL over LTRB distances.

    Returns a 4-vector (box, cls, dfl, nll); the trainer sums it for backward
    and logs the items under 4 loss names (GaussianTrainer.get_validator).
    """

    def __init__(self, model):
        super().__init__(model)
        cfg = {**DEFAULT_GAUSSIAN_CFG, **getattr(model, "gaussian_cfg", {})}
        self.beta = float(cfg["beta"])
        self.nll_gain = float(cfg["nll_gain"])
        self.warmup_epochs = int(cfg["warmup_epochs"])
        self.ramp_epochs = int(cfg["ramp_epochs"])
        self.nll_detach_mu = bool(cfg["nll_detach_mu"])
        self.epoch = 0  # kept current by GaussianTrainer's on_train_epoch_start callback

    def nll_weight(self) -> float:
        """0 during warm-up, then a linear ramp to 1 (scope §6.2 warm-up fix)."""
        if self.epoch < self.warmup_epochs:
            return 0.0
        if self.ramp_epochs <= 0:
            return 1.0
        return min(1.0, (self.epoch - self.warmup_epochs + 1) / self.ramp_epochs)

    def gaussian_nll(self, preds, fg_mask, target_bboxes, anchor_points, stride_tensor):
        """β-NLL over foreground anchors, fp32 regardless of AMP.

        μ = DFL expectation (same decode as bbox_decode, pre-dist2bbox); target =
        clamped bbox2dist exactly as the DFL loss uses; both in stride units.
        """
        logvars = preds.get("logvars")
        if logvars is None or not fg_mask.any():
            return torch.zeros((), device=self.device)

        pred_distri = preds["boxes"].permute(0, 2, 1)  # (bs, A, 4*reg_max)
        b, a, c = pred_distri.shape
        mu = pred_distri.view(b, a, 4, c // 4).softmax(3).matmul(self.proj.type(pred_distri.dtype))
        target_ltrb = bbox2dist(anchor_points, target_bboxes / stride_tensor, self.reg_max - 1)

        mu = mu[fg_mask].float()
        logvar = logvars.permute(0, 2, 1)[fg_mask].float().clamp(LOGVAR_MIN, LOGVAR_MAX)
        target = target_ltrb[fg_mask].float()
        if self.nll_detach_mu:
            mu = mu.detach()

        var = logvar.exp()
        nll = 0.5 * ((target - mu) ** 2 / var + logvar)
        if self.beta > 0:
            nll = nll * var.detach().pow(self.beta)  # β-NLL: stop-gradient sample weight (R10)
        return nll.mean()

    def loss(self, preds, batch):
        batch_size = preds["boxes"].shape[0]
        assigned, loss3, _ = self.get_assigned_targets_and_loss(preds, batch)
        fg_mask, _, target_bboxes, anchor_points, stride_tensor = assigned
        nll = self.gaussian_nll(preds, fg_mask, target_bboxes, anchor_points, stride_tensor)
        nll = self.nll_weight() * self.nll_gain * nll
        loss4 = torch.cat((loss3, nll.reshape(1)))
        return loss4 * batch_size, loss4.detach()


class GaussianDetectionModel(DetectionModel):
    """DetectionModel whose criterion is the Gaussian loss. Produced by class swap
    in `convert_to_gaussian`; survives checkpoint pickling because the class is
    importable from the installed uqfusion package."""

    def init_criterion(self):
        return GaussianDetectionLoss(self)


def convert_to_gaussian(model: DetectionModel, gaussian_cfg: dict | None = None) -> DetectionModel:
    """In-place conversion of a standard DetectionModel: add cv4, swap classes.

    Idempotent (resume-safe). Raises on non-plain-Detect heads (v10Detect,
    YOLO26, ...) — plan B1's conditionals apply there and need their own path.
    """
    head = model.model[-1]
    cfg = {**DEFAULT_GAUSSIAN_CFG, **(gaussian_cfg or {})}

    if isinstance(head, GaussianDetect):  # already converted (resume / reload)
        model.__class__ = GaussianDetectionModel
        model.gaussian_cfg = {**cfg, **getattr(model, "gaussian_cfg", {})}
        return model
    if type(head) is not Detect:
        raise TypeError(
            f"convert_to_gaussian expects a plain Detect head, got {type(head).__name__} — "
            "for non-DFL/end2end heads the plan B1 conditional design applies (not implemented yet)"
        )
    if getattr(head, "end2end", False):
        raise NotImplementedError("end2end (one2one) heads are out of scope for the σ branch (plan B1)")

    ch = tuple(seq[0].conv.in_channels for seq in head.cv2)   # neck feature widths per level
    c4 = head.cv2[0][0].conv.out_channels                     # mirror cv2's hidden width
    cv4 = nn.ModuleList(
        nn.Sequential(Conv(x, c4, 3), Conv(c4, c4, 3), nn.Conv2d(c4, 4, 1)) for x in ch
    )
    for seq in cv4:
        seq[-1].bias.data.zero_()  # logvar starts at 0 -> σ = 1 stride unit: sane, non-degenerate
    device = next(head.parameters()).device
    head.cv4 = cv4.to(device)

    head.__class__ = GaussianDetect
    head.output_sigma = False
    head.sigma_detach_features = bool(cfg["sigma_detach_features"])

    model.__class__ = GaussianDetectionModel
    model.gaussian_cfg = cfg
    if getattr(model, "criterion", None) is not None:
        model.criterion = None  # force re-init with the Gaussian loss
    return model


def _register_safe_globals() -> None:
    """Allow-list our classes for torch weights_only checkpoint loading (torch>=2.6
    default). Registered at import so every path that touches uqfusion.uq —
    training resume, validation, UQ inference — can unpickle our checkpoints."""
    try:
        torch.serialization.add_safe_globals([GaussianDetect, GaussianDetectionModel])
    except Exception:  # noqa: BLE001 - older torch; loading falls back to ultralytics handling
        pass


_register_safe_globals()
