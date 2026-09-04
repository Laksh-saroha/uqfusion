"""Idea I3 -- build the two sources of INDEPENDENT box estimates that exist.

`probe_within_modality.py` shows the mechanism `sigma_weighted_fusion` exists for
-- inverse-variance coordinate averaging -- has never run on a multi-member
cluster, because `iou_thr` is 0.85 and `single_passthrough` skips WBF on
one-stream frames. It also shows that clustering the detector's OWN duplicates is
negative: they are not independent estimates, they are the same error twice.

Two sources of genuinely independent estimates:

  **(a) the one2many branch.** `Detect.forward` returns `(y, preds)` in eval and
  `preds["one2many"]` is the raw one2many head output -- it is computed on every
  forward pass and thrown away. YOLO26's o2o branch is trained for NMS-free
  deployment and emits 8.9 boxes/frame; o2m + NMS is the higher-recall path.
  NOTE: sigma rides on `one2one_cv2` only (`gaussian.py`), so o2m boxes carry NO
  valid sigma -- the cache fills it with the frame's o2o median and stamps
  `sigma_valid: False` in the meta. Any arm that weights o2m boxes by sigma is
  reading a placeholder.

  **(b) test-time augmentation.** hflip and two scales. The estimates are
  pixel-exactly co-registered, so C4's registration-residual explanation for why
  merging fails is absent by construction -- which is what makes this the clean
  test of whether merging fails for the DEEPER reason (the best member is already
  the best estimate).

Writes to `runs/cache_tta/` and `runs/cache_o2m/` -- NEW directories.

Usage:
    python scripts/build_tta_o2m.py --mode tta o2m
    python scripts/build_tta_o2m.py --mode tta --limit 40 --out-suffix _smoke
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.bench.grid import _git_commit          # noqa: E402
from uqfusion.config import load_config              # noqa: E402
from uqfusion.uq.gaussian import GaussianDetect      # noqa: E402
from uqfusion.uq.infer import load_model             # noqa: E402


class MultiPredictor:
    """One model, several views. Emits the project record schema.

    `views` are (name, scale, flip). Boxes from each view are mapped back to the
    original image frame and concatenated with a `view` column, so a downstream
    probe can either pool them or cluster them.
    """

    def __init__(self, weights, device="cuda:0", imgsz=640, conf=0.001, iou=0.7,
                 max_det=300, views=(("id", 1.0, False),)):
        from ultralytics.data.augment import LetterBox

        self.model, self.names = load_model(weights, device)
        head = self.model.model[-1]
        if not isinstance(head, GaussianDetect):
            raise TypeError(f"{weights} has no GaussianDetect head")
        head.output_sigma = True
        head.output_dfl_unc = True
        self.head = head
        self.nc = len(self.names)
        self.end2end = bool(getattr(head, "end2end", False))
        if self.end2end:
            head.max_det = max_det
        self.device, self.imgsz, self.conf, self.iou, self.max_det = device, imgsz, conf, iou, max_det
        self.views = list(views)
        self._lb = {}
        for _n, s, _f in self.views:
            k = int(round(imgsz * s))
            self._lb[k] = LetterBox((k, k), auto=False)
        self._feats = None
        head.register_forward_pre_hook(self._capture)

    def _capture(self, module, args):
        x = args[0]
        self._feats = torch.cat([xi.float().mean(dim=(2, 3)) for xi in x], dim=1)

    def _one_view(self, im0, size, flip):
        from ultralytics.utils import nms, ops

        im = im0[:, ::-1] if flip else im0
        im = self._lb[size](image=im)
        im = im[..., ::-1].transpose(2, 0, 1)
        t = torch.from_numpy(np.ascontiguousarray(im)).float().div_(255.0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            out = self.model(t)
        y = out[0] if isinstance(out, (tuple, list)) else out
        raw = out[1] if isinstance(out, (tuple, list)) and len(out) > 1 else None
        if self.end2end:
            det = y[0]
            det = det[det[:, 4] >= self.conf]
        else:
            det = nms.non_max_suppression(y, self.conf, self.iou, nc=self.nc,
                                          max_det=self.max_det)[0]
        boxes = det[:, :4].clone()
        if boxes.shape[0]:
            boxes = ops.scale_boxes(t.shape[2:], boxes, im0.shape)
        sig = det[:, 6:10].clone() if det.shape[1] >= 10 else torch.zeros((len(det), 4), device=det.device)
        gain = min(t.shape[2] / im0.shape[0], t.shape[3] / im0.shape[1])
        if gain > 0:
            sig = sig / gain
        b = boxes.cpu().numpy().reshape(-1, 4)
        if flip and len(b):
            w = im0.shape[1]
            b = np.column_stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]])
            sg = sig.cpu().numpy().reshape(-1, 4)
            sg = sg[:, [2, 1, 0, 3]]
        else:
            sg = sig.cpu().numpy().reshape(-1, 4)
        return (b, det[:, 4].cpu().numpy(), det[:, 5].cpu().numpy().astype(int), sg, raw)

    def o2m(self, im0, raw):
        """Decode + NMS the one2many branch that eval mode computes and discards."""
        from ultralytics.utils import nms, ops

        if raw is None or not isinstance(raw, dict) or "one2many" not in raw:
            return None
        y = self.head._inference(raw["one2many"])
        det = nms.non_max_suppression(y, self.conf, self.iou, nc=self.nc,
                                      max_det=self.max_det)[0]
        boxes = det[:, :4].clone()
        k = int(round(self.imgsz))
        if boxes.shape[0]:
            boxes = ops.scale_boxes((k, k), boxes, im0.shape)
        return (boxes.cpu().numpy().reshape(-1, 4), det[:, 4].cpu().numpy(),
                det[:, 5].cpu().numpy().astype(int))

    def __call__(self, image, want_o2m=False):
        import cv2

        im0 = cv2.imread(str(image))
        if im0 is None:
            raise FileNotFoundError(f"could not read image: {image}")
        B, C, K, S, V = [], [], [], [], []
        raw0 = None
        for vi, (name, s, f) in enumerate(self.views):
            b, c, k, sg, raw = self._one_view(im0, int(round(self.imgsz * s)), f)
            if vi == 0:
                raw0 = raw
                feat = self._feats[0].cpu().numpy() if self._feats is not None else np.zeros(1)
            B.append(b); C.append(c); K.append(k); S.append(sg); V.append(np.full(len(b), vi))
        if want_o2m:
            got = self.o2m(im0, raw0)
            if got is None:
                raise RuntimeError("one2many branch not exposed by this head; "
                                   "eval-mode Detect.forward must return (y, preds)")
            b, c, k = got
            # Sigma rides on one2one only -- fill with the o2o median, and the meta
            # stamps sigma_valid False so nothing reads this as a measurement.
            fill = np.median(S[0]) if len(S[0]) else 1.0
            B = [b]; C = [c]; K = [k]; S = [np.full((len(b), 4), fill)]; V = [np.full(len(b), 0)]
        return {"boxes_xyxy": np.concatenate(B).reshape(-1, 4).astype(np.float64),
                "conf": np.concatenate(C).astype(np.float64),
                "cls": np.concatenate(K).astype(int),
                "sigma_ltrb": np.concatenate(S).reshape(-1, 4).astype(np.float64),
                "view": np.concatenate(V).astype(int),
                "feat": feat, "image_hw": im0.shape[:2]}


VIEWS = {
    "tta": (("id", 1.0, False), ("flip", 1.0, True), ("s0.8", 0.8, False), ("s1.25", 1.25, False)),
    "o2m": (("id", 1.0, False),),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", nargs="+", default=["tta", "o2m"], choices=["tta", "o2m"])
    ap.add_argument("--weights", default="runs/full_scale/gauss_vis_seed0/weights/best.pt")
    ap.add_argument("--images-list", default="runs/derived/paired_val_vis.txt")
    ap.add_argument("--out-suffix", default="")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    images = [Path(l.strip()) for l in (ROOT / args.images_list).read_text(
        encoding="utf-8").splitlines() if l.strip()]
    if args.limit:
        images = images[: args.limit]
    dev = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"[tta] {len(images)} frames on {dev}")

    rc = 0
    for mode in args.mode:
        out = ROOT / f"runs/cache_{mode}" / f"gauss_vis_paired_clean{args.out_suffix}.pkl"
        if out.is_file():
            print(f"[skip] {out}")
            continue
        out.parent.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        pred = MultiPredictor(str(ROOT / args.weights), device=dev, imgsz=args.imgsz,
                              conf=args.conf, views=VIEWS[mode])
        recs = []
        try:
            for i, img in enumerate(images):
                r = pred(img, want_o2m=(mode == "o2m"))
                r["image_path"] = str(img)
                recs.append(r)
                if (i + 1) % 200 == 0:
                    print(f"[{mode}] {i + 1}/{len(images)}  "
                          f"{np.mean([len(x['conf']) for x in recs]):.1f} bx/fr")
        except Exception as e:  # noqa: BLE001 - one mode failing must not kill the other
            print(f"[FAIL] {mode}: {type(e).__name__}: {e}")
            rc = 1
            continue
        meta = {"source": "gaussian", "weights": [str(args.weights)], "mode": mode,
                "views": [v[0] for v in VIEWS[mode]], "imgsz": args.imgsz, "conf": args.conf,
                "corrupt": None, "images_list": args.images_list,
                "sigma_valid": mode != "o2m", "n_frames": len(recs),
                "git_commit": _git_commit()}
        with open(out, "wb") as fh:
            pickle.dump({"meta": meta, "records": recs}, fh)
        print(f"[ok] {out}  {len(recs)} frames, "
              f"{np.mean([len(x['conf']) for x in recs]):.1f} bx/fr, {time.time() - t0:.0f}s")
    return rc


if __name__ == "__main__":
    sys.exit(main())
