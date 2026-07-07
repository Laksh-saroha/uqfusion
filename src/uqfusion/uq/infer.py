"""UQ inference: per-detection σ + frame-level features in one forward pass.

The stock DetectionPredictor infers class count from the channel dimension
(nc=0 for the detect task), so σ channels would be misread as class scores.
This wrapper therefore drives the model directly: letterbox → forward (with
`output_sigma` enabled) → NMS with EXPLICIT nc (σ rides through NMS as the
extra trailing columns, ultralytics.utils.nms contract) → rescale boxes and σ
to original pixels. A forward pre-hook on the head captures the neck feature
maps and pools them into one frame vector for the Mahalanobis scorer (O3) —
same pass, no extra cost (scope §6.1's "two signals tapped in one pass").

This is also the producer for the cached-predictions design (plan B5-2): one
record per frame with boxes/conf/cls/σ/feature, so every gate-level ablation
downstream is CPU post-processing.
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
    except Exception:  # noqa: BLE001 - older torch or already registered; loading may still work
        pass


def load_uq_model(weights: str | Path, device: str = "cpu"):
    """Load a Gaussian-head checkpoint and enable σ output. Returns (nn model, names dict)."""
    _register_safe_globals()
    from ultralytics import YOLO

    yolo = YOLO(str(weights))
    model = yolo.model.to(device).eval()
    head = model.model[-1]
    if not isinstance(head, GaussianDetect):
        raise TypeError(f"{weights} does not contain a GaussianDetect head (got {type(head).__name__})")
    head.output_sigma = True
    head.output_dfl_unc = True  # §7.2 ablation row rides in the same pass (decision D1)
    names = yolo.names if isinstance(yolo.names, dict) else dict(enumerate(yolo.names))
    return model, names


class UQPredictor:
    """Single-image UQ inference. Returns one record dict per frame:

    boxes_xyxy (N,4) px | conf (N,) | cls (N,) | sigma_ltrb (N,4) px | feat (D,)
    """

    def __init__(
        self,
        weights: str | Path,
        device: str = "cpu",
        imgsz: int = 640,
        conf: float = 0.25,
        iou: float = 0.7,
        max_det: int = 300,
    ):
        from ultralytics.data.augment import LetterBox

        self.model, self.names = load_uq_model(weights, device)
        self.nc = len(self.names)
        self.device = device
        self.imgsz = imgsz
        self.conf = conf
        self.iou = iou
        self.max_det = max_det
        self._letterbox = LetterBox((imgsz, imgsz), auto=False)
        self._feats: torch.Tensor | None = None
        # Pre-hook on the head captures the neck feature maps (the head's inputs)
        # in the same forward pass; mean-pool each level -> one frame vector.
        head = self.model.model[-1]
        head.register_forward_pre_hook(self._capture)

    def _capture(self, module, args) -> None:
        x = args[0]
        self._feats = torch.cat([xi.float().mean(dim=(2, 3)) for xi in x], dim=1)

    def __call__(self, image) -> dict:
        """image: path or BGR ndarray (cv2 convention, matching training I/O)."""
        import cv2
        from ultralytics.utils import nms, ops

        if isinstance(image, (str, Path)):
            im0 = cv2.imread(str(image))
            if im0 is None:
                raise FileNotFoundError(f"could not read image: {image}")
        else:
            im0 = image
        h0, w0 = im0.shape[:2]

        im = self._letterbox(image=im0)
        im = im[..., ::-1].transpose(2, 0, 1)  # BGR -> RGB, HWC -> CHW
        t = torch.from_numpy(np.ascontiguousarray(im)).float().div_(255.0).unsqueeze(0).to(self.device)

        with torch.no_grad():
            out = self.model(t)
        y = out[0] if isinstance(out, (tuple, list)) else out  # (1, 4+nc+4, A)

        det = nms.non_max_suppression(
            y, self.conf, self.iou, nc=self.nc, max_det=self.max_det
        )[0]  # (N, 6+extras): x1,y1,x2,y2,conf,cls, σ_ltrb[4], dfl_std_ltrb[4 if enabled]

        boxes = det[:, :4].clone()
        if boxes.shape[0]:
            boxes = ops.scale_boxes(t.shape[2:], boxes, im0.shape)
        gain = min(t.shape[2] / h0, t.shape[3] / w0)  # letterbox scale factor
        sigma = det[:, 6:10] / gain  # lengths rescale by 1/gain (no pad offset)
        dfl_sigma = det[:, 10:14] / gain if det.shape[1] >= 14 else None

        record = {
            "boxes_xyxy": boxes.cpu().numpy(),
            "conf": det[:, 4].cpu().numpy(),
            "cls": det[:, 5].cpu().numpy().astype(int),
            "sigma_ltrb": sigma.cpu().numpy(),
            "feat": self._feats[0].cpu().numpy(),
            "image_hw": (h0, w0),
        }
        if dfl_sigma is not None:
            record["dfl_sigma_ltrb"] = dfl_sigma.cpu().numpy()
        return record
