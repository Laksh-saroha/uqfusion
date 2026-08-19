"""Inference producing the project-wide per-frame record schema.

All uncertainty sources (Gaussian head, MC-Dropout, Deep Ensemble) emit the
SAME record, so every downstream consumer — calibration metrics, reliability,
fusion, caches — has one codepath (plan B6-4's "one protocol, not a knob"):

    boxes_xyxy (N,4) px | conf (N,) | cls (N,) int | sigma_ltrb (N,4) px
    [dfl_sigma_ltrb (N,4) px — Gaussian source only, §7.2 ablation row]
    feat (D,) pooled neck features | image_hw (h, w) | n_support (N,) [sampled sources]

The stock DetectionPredictor infers class count from the channel dimension
(nc=0 for detect), so σ channels would be misread as classes — this module
drives the model directly and passes nc explicitly to NMS; σ rides through NMS
as the extra trailing columns (ultralytics.utils.nms contract, pinned 8.4.90).

On end2end heads (YOLO26) there is no NMS: the head's own `postprocess` has already
done top-k selection and returns (bs, k, 6 + extra). Both layouts put σ at columns
6:10 and DFL-derived σ at 10:14, so everything downstream of `det` is shared.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from uqfusion.uq.gaussian import GaussianDetect, GaussianDetectionModel


def _register_safe_globals() -> None:
    """Allow-list our classes for torch weights_only checkpoint loading."""
    try:
        torch.serialization.add_safe_globals([GaussianDetect, GaussianDetectionModel])
    except Exception:  # noqa: BLE001 - older torch or already registered
        pass


def load_model(weights: str | Path, device: str = "cpu"):
    """Load any detection checkpoint (standard or Gaussian). Returns (nn model, names)."""
    _register_safe_globals()
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    model = yolo.model.to(device).eval()
    names = yolo.names if isinstance(yolo.names, dict) else dict(enumerate(yolo.names))
    return model, names


class PlainPredictor:
    """Single-pass inference on any detection checkpoint, emitting the record schema.

    `enable_sigma=True` requires a GaussianDetect head and adds sigma_ltrb +
    dfl_sigma_ltrb from the same pass. Without it, sigma_ltrb is absent —
    sampling-based wrappers (MC-Dropout/Ensemble) fill it via clustering.
    """

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
        enable_sigma: bool = False,
        want_features: bool = True,
    ):
        from ultralytics.data.augment import LetterBox

        self.model, self.names = load_model(weights, device)
        head = self.model.model[-1]
        self.has_sigma = False
        if enable_sigma:
            if not isinstance(head, GaussianDetect):
                raise TypeError(f"{weights} has no GaussianDetect head (got {type(head).__name__})")
            head.output_sigma = True
            head.output_dfl_unc = True  # §7.2 ablation row rides in the same pass (D18)
            self.has_sigma = True

        self.nc = len(self.names)
        self.end2end = bool(getattr(head, "end2end", False))
        if self.end2end:
            head.max_det = max_det  # the head owns selection; there is no NMS to cap it
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self._letterbox = LetterBox((imgsz, imgsz), auto=False)
        self._feats: torch.Tensor | None = None
        if want_features:
            head.register_forward_pre_hook(self._capture)

    def _capture(self, module, args) -> None:
        x = args[0]
        self._feats = torch.cat([xi.float().mean(dim=(2, 3)) for xi in x], dim=1)

    def _read(self, image) -> np.ndarray:
        import cv2

        if isinstance(image, (str, Path)):
            im0 = cv2.imread(str(image))
            if im0 is None:
                raise FileNotFoundError(f"could not read image: {image}")
            return im0
        return image

    def __call__(self, image) -> dict:
        from ultralytics.utils import nms, ops

        im0 = self._read(image)
        h0, w0 = im0.shape[:2]
        im = self._letterbox(image=im0)
        im = im[..., ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        t = torch.from_numpy(np.ascontiguousarray(im)).float().div_(255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(t)
        y = out[0] if isinstance(out, (tuple, list)) else out

        if self.end2end:
            # GaussianDetect.postprocess already ran top-k and gathered σ with the
            # boxes; rows are [x1 y1 x2 y2 | conf | cls | σ...] in letterboxed px.
            det = y[0]
            det = det[det[:, 4] >= self.conf]
        else:
            det = nms.non_max_suppression(y, self.conf, self.iou, nc=self.nc, max_det=self.max_det)[0]

        boxes = det[:, :4].clone()
        if boxes.shape[0]:
            boxes = ops.scale_boxes(t.shape[2:], boxes, im0.shape)
        gain = min(t.shape[2] / h0, t.shape[3] / w0)

        record = {
            "boxes_xyxy": boxes.cpu().numpy(),
            "conf": det[:, 4].cpu().numpy(),
            "cls": det[:, 5].cpu().numpy().astype(int),
            "image_hw": (h0, w0),
        }
        if self._feats is not None:
            record["feat"] = self._feats[0].cpu().numpy()
        if self.has_sigma:
            record["sigma_ltrb"] = (det[:, 6:10] / gain).cpu().numpy()
            if det.shape[1] >= 14:
                record["dfl_sigma_ltrb"] = (det[:, 10:14] / gain).cpu().numpy()
        return record


class UQPredictor(PlainPredictor):
    """Gaussian-head inference: σ + DFL-derived σ + features, one forward pass."""

    def __init__(self, weights, device="cpu", imgsz=640, conf=0.25, iou=0.7, max_det=300):
        super().__init__(
            weights, device=device, imgsz=imgsz, conf=conf, iou=iou, max_det=max_det,
            enable_sigma=True, want_features=True,
        )


def load_uq_model(weights: str | Path, device: str = "cpu"):
    """Back-compat helper: load a Gaussian checkpoint with σ output enabled."""
    model, names = load_model(weights, device)
    head = model.model[-1]
    if not isinstance(head, GaussianDetect):
        raise TypeError(f"{weights} does not contain a GaussianDetect head (got {type(head).__name__})")
    head.output_sigma = True
    head.output_dfl_unc = True
    return model, names
