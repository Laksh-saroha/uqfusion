"""MC-Dropout baseline (O2 comparison — scope §9.2, R5; plan B6-3).

Ultralytics detect heads have no dropout, so MC-Dropout is NOT free on YOLO:
dropout must be inserted and the model RETRAINED — a different deterministic
model whose own detection row must be reported (its standard training
results.csv provides exactly that, since validation runs with dropout off).

Pre-fixed protocol (plan B6-3, decided before any real run):
  placement = one Dropout2d before the final 1x1 conv of each head branch
  (box and cls, every level); p and T from config
  (`baselines.mc_dropout`, defaults p=0.15, T=10).

**Deviation from B6-3, forced 2026-08-23 (handoff §1.5).** B6-3 names `cv2` and
`cv3`, which is correct only for a conventional Detect head. YOLO26's head is
end-to-end: `Detect.forward` runs it twice and inference decodes from
`preds["one2one"]`, discarding the one2many `cv2`/`cv3` outputs entirely. Dropout
placed there is trained but NEVER EXECUTED at inference — T stochastic passes come
out identical, epistemic variance is exactly zero, and nothing raises. It also
diverged both VIS runs: the noise perturbs the shared trunk's gradients while the
scored branch gets none of the regularisation, so `val/cls_loss` explodes (2.2 →
72) while `train/cls_loss` stays flat, at the same epoch under two seeds.

So on end2end heads the rule is applied to `one2one_cv2`/`one2one_cv3` — the same
branch, selected by the same criterion ("whichever produces the final
detections"), and the same choice `gaussian._sigma_box_head` already makes for the
σ² branch. See `deployed_head_branches`.

Inference: T stochastic passes (dropout re-enabled in eval), merged by the
shared clustering protocol (uq/clustering.py) into the standard record schema.
`PlainPredictor` forwards the real `nn.Module`, so arming it works; note that
Ultralytics' own `val()` goes through `AutoBackend`, which rebuilds the model from
the checkpoint path and silently discards any in-memory arming.

Gate: `scripts/smoke_mc_e2e.py` — CPU, seconds, no dataset. Run it before any GPU
time. Counting dropout layers is NOT evidence that they execute.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

from uqfusion.uq.clustering import cluster_records
from uqfusion.uq.infer import PlainPredictor


def _detect_head(model):
    """The Detect head, whether given a DetectionModel or the head itself."""
    inner = getattr(model, "model", None)
    if inner is not None and len(inner) and hasattr(inner[-1], "nl"):
        return inner[-1]
    return model


def deployed_head_branches(head) -> tuple[str, ...]:
    """Names of the head branches whose outputs actually reach inference.

    End2end heads (YOLO26) run Detect twice and decode from `preds["one2one"]`;
    the one2many `cv2`/`cv3` outputs are discarded. Dropout there is trained but
    never executed, so MC-Dropout yields exactly zero variance while every
    aggregate check still passes (handoff §1.5). Same criterion, and same answer,
    as `gaussian._sigma_box_head`.
    """
    if getattr(head, "end2end", False):
        return ("one2one_cv2", "one2one_cv3")
    return ("cv2", "cv3")


class MCDropoutConv2d(nn.Conv2d):
    """A head's final conv, with MC-Dropout applied to its own input.

    **Why a subclass and not an inserted layer (2026-08-31).** The original
    implementation rebuilt each branch as
    `nn.Sequential(*children[:-1], nn.Dropout2d(p), children[-1])`. That renumbers
    the final conv -- `one2one_cv2.0.2.weight` becomes `one2one_cv2.0.3.weight` --
    so every deployed-branch key in the state_dict shifts by one.

    Ultralytics' *training-time* load (`DetectionTrainer.get_model` ->
    `BaseModel.load`) builds a plain model from yaml and matches the checkpoint
    **by key name**, intersecting silently. Reloading an MC checkpoint therefore
    dropped exactly 12 tensors -- the box (`cv2`) and class (`cv3`) output convs at
    all three scales, 62,460 parameters. 0.28% of the model, and 100% of the output
    layer: it arrived randomly initialised. That decapitated `mc_vis_seed0_ft`
    (parent 0.24667 -> 0.15123) and `mc_ir_seed0_ft` (0.12973 -> 0.11894). The
    inference path was never affected -- `YOLO()` restores the pickled module, so
    no key matching happens -- which is why it surfaced only in the fine-tunes.

    Subclassing keeps `weight` and `bias` at exactly the indices a plain model has,
    so an MC checkpoint stays structurally interchangeable with a stock one and
    every load path -- training reload, resume, eval -- works.

    `mc_armed` exists because MC inference needs dropout active while the model is
    in eval mode; the old code achieved that by calling `.train()` on the Dropout
    modules, which has no equivalent here.
    """

    mc_p: float = 0.15
    mc_armed: bool = False

    def forward(self, x):
        if self.training or self.mc_armed:
            x = F.dropout2d(x, self.mc_p, training=True)
        return super().forward(x)


def insert_head_dropout(model, p: float = 0.15):
    """Arm the last conv of each DEPLOYED head branch with MC-Dropout. Idempotent.

    Does not change the module tree -- see `MCDropoutConv2d` for why that matters.
    """
    head = _detect_head(model)
    for branch_name in deployed_head_branches(head):
        branch = getattr(head, branch_name, None)
        if branch is None:
            continue
        for seq in branch:
            conv = seq[-1]
            if not isinstance(conv, nn.Conv2d):
                raise TypeError(
                    f"{branch_name}: expected the branch to end in nn.Conv2d, got "
                    f"{type(conv).__name__}. The dropout placement rule (B6-3) is "
                    "defined relative to that final conv."
                )
            conv.__class__ = MCDropoutConv2d  # same C layout; parameters untouched
            conv.mc_p = float(p)
    return model


def enable_mc_dropout(model) -> int:
    """Re-enable dropout layers after model.eval(); returns how many were armed.

    Raises unless at least one armed layer sits on the branch that reaches
    inference. The old guard only checked that *some* dropout existed anywhere,
    which passed for six months of a model whose every dropout layer was on the
    discarded branch — the failure that cost two VIS runs (handoff §1.5).
    """
    head = _detect_head(model)

    def _armable(mod):
        # MCDropoutConv2d is the current placement; nn.Dropout* is the pre-2026-08-31
        # inserted-layer form, still carried by mc_vis_seed0 / mc_ir_seed0, whose
        # detection training was sound and which must keep working unretrained.
        return isinstance(mod, (MCDropoutConv2d, nn.Dropout, nn.Dropout2d))

    on_path = 0
    for branch_name in deployed_head_branches(head):
        branch = getattr(head, branch_name, None)
        if branch is None:
            continue
        on_path += sum(1 for m in branch.modules() if _armable(m))

    n = 0
    for m in model.modules():
        if isinstance(m, MCDropoutConv2d):
            m.mc_armed = True
            n += 1
        elif isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
            n += 1

    if n == 0:
        raise RuntimeError("no dropout layers found — was this model trained via MCDropoutTrainer?")
    if on_path == 0:
        raise RuntimeError(
            f"{n} dropout layer(s) found, but none on {deployed_head_branches(head)} — the "
            "branch that produces the final detections. They will never execute at inference "
            "and every MC pass would be identical (epistemic variance exactly zero). This is "
            "the 2026-08-23 defect; retrain with the current insert_head_dropout."
        )
    return n


class MCDropoutTrainer(DetectionTrainer):
    """Standard detection training on the dropout-augmented head (3 loss items)."""

    dropout_p: float = 0.15

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        return insert_head_dropout(model, self.dropout_p)


def make_mc_trainer(p: float) -> type[MCDropoutTrainer]:
    return type("ConfiguredMCDropoutTrainer", (MCDropoutTrainer,), {"dropout_p": float(p)})


def train_mc_dropout(
    cfg: dict,
    data_yaml: str | Path,
    variant: str,
    seed: int = 0,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    run_name: str | None = None,
    weights=None,
    callbacks=None,
    train_overrides=None,
):
    """Train the MC-Dropout model with the shared config. Returns (best_weights, run_dir).

    `weights` starts the run from an explicit checkpoint instead of the variant's COCO
    weights (a fine-tune continuation stage). `callbacks` is registered on the model
    before training (queue heartbeat/pause wiring, same shape as `uq.train_gaussian`'s).
    Resume (same run's own `last.pt`) takes priority over both.
    """
    from ultralytics import YOLO

    from uqfusion.bench.grid import resolve_device
    from uqfusion.uq.variants import build_variant, is_variant

    b = cfg["benchmark"]
    p = float((cfg.get("baselines") or {}).get("mc_dropout", {}).get("p", 0.15))
    out_root = Path(cfg["paths"]["outputs_root"]) / "mc_dropout"
    name = run_name or f"mc_{variant}_seed{seed}"
    run_dir = out_root / name

    last_ckpt = run_dir / "weights" / "last.pt"
    resuming = last_ckpt.is_file()

    if resuming:
        model = YOLO(str(last_ckpt))
    elif weights is not None:
        if not Path(weights).is_file():
            raise FileNotFoundError(f"start weights not found: {weights}")
        model = YOLO(str(weights))
    elif is_variant(variant):
        model, transfer_pct = build_variant(variant)
        print(f"[mc-dropout] {name}: variant {variant}, {transfer_pct:.1f}% seeded")
    else:
        model = YOLO(f"{variant}.pt" if b.get("pretrained", True) else f"{variant}.yaml")

    for event, fns in (callbacks or {}).items():
        for fn in fns:
            model.add_callback(event, fn)

    if resuming:
        # Same rationale as uq/train_gaussian.py: on resume Ultralytics reloads every
        # hyperparameter from the checkpoint's own args.yaml, so nothing else is passed.
        model.train(trainer=make_mc_trainer(p), resume=True)
    else:
        kwargs = dict(
            data=str(data_yaml),
            epochs=epochs if epochs is not None else b["epochs"],
            imgsz=imgsz if imgsz is not None else b["imgsz"],
            batch=batch if batch is not None else b["batch"],
            seed=seed, deterministic=b["deterministic"], optimizer=b.get("optimizer", "auto"),
            patience=b["patience"], amp=b["amp"],
            workers=workers if workers is not None else b["workers"],
            device=resolve_device(cfg), project=str(out_root), name=name, exist_ok=True, verbose=True,
        )
        kwargs.update(train_overrides or {})
        model.train(trainer=make_mc_trainer(p), **kwargs)
    best = run_dir / "weights" / "best.pt"
    return (best if best.is_file() else run_dir / "weights" / "last.pt"), run_dir


class MCDropoutPredictor:
    """T stochastic passes -> shared clustering -> standard record (+n_support)."""

    def __init__(self, weights, T: int = 10, device="cpu", imgsz=640, conf=0.25, iou=0.7, seed: int = 0):
        self.base = PlainPredictor(weights, device=device, imgsz=imgsz, conf=conf, iou=iou, want_features=True)
        self.T = int(T)
        self.seed = int(seed)
        self.n_armed = enable_mc_dropout(self.base.model)

    def __call__(self, image) -> dict:
        torch.manual_seed(self.seed)  # reproducible pass set per image
        return cluster_records([self.base(image) for _ in range(self.T)])
