"""Batched inference vs batch 1: how fast, how different (2026-10-10).

`PlainPredictor.predict_batch` runs several frames in one forward pass. The forward pass
is not bit-identical to batch 1 (cuDNN picks kernels per batch shape), so before any cache
is built batched this measures, on real frames and a real Phase 3 checkpoint:

1. `predict_batch([x])` against `__call__(x)`: must be bit-identical (same ops, batch 1);
2. for each batch size B: per-detection |d conf|, |d box|, |d sigma| on matched detections,
   detections present in one run only, max |d feat|, and the mAP@50-95 / ship AP@50-95
   difference against batch 1 over the same frames;
3. frames/s and peak GPU memory per B.

    py -3.13 scripts/check_batched_inference.py --n 512 --batches 1 8 16 32
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.matching import load_gt, local_ap50_95     # noqa: E402
from uqfusion.uq.infer import UQPredictor                      # noqa: E402

PV = ROOT / "runs/derived/paired_val_vis.txt"


def iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    tl = np.maximum(a[:, None, :2], b[None, :, :2])
    br = np.minimum(a[:, None, 2:], b[None, :, 2:])
    inter = np.prod(np.clip(br - tl, 0, None), axis=2)
    area = lambda x: np.prod(x[:, 2:] - x[:, :2], axis=1)
    return inter / (area(a)[:, None] + area(b)[None, :] - inter + 1e-9)


def compare(ref: list[dict], got: list[dict]) -> dict:
    dconf, dbox, dsig, dfeat, unmatched, total, exact = [], [], [], [], 0, 0, 0
    for r, g in zip(ref, got):
        dfeat.append(float(np.max(np.abs(r["feat"] - g["feat"]))))
        same = all(r[k].shape == g[k].shape and np.array_equal(r[k], g[k])
                   for k in ("boxes_xyxy", "conf", "cls", "sigma_ltrb"))
        exact += same
        total += len(r["conf"])
        if not len(r["conf"]) or not len(g["conf"]):
            unmatched += abs(len(r["conf"]) - len(g["conf"]))
            continue
        m = iou(r["boxes_xyxy"], g["boxes_xyxy"])
        m[r["cls"][:, None] != g["cls"][None, :]] = 0
        j = m.argmax(1)
        ok = m[np.arange(len(j)), j] > 0.99
        unmatched += int((~ok).sum()) + max(0, len(g["conf"]) - int(ok.sum()))
        dconf += np.abs(r["conf"][ok] - g["conf"][j[ok]]).tolist()
        dbox += np.abs(r["boxes_xyxy"][ok] - g["boxes_xyxy"][j[ok]]).max(1).tolist()
        dsig += np.abs(r["sigma_ltrb"][ok] - g["sigma_ltrb"][j[ok]]).max(1).tolist()
    q = lambda v: {"max": float(np.max(v)) if v else 0.0, "p99": float(np.quantile(v, 0.99)) if v else 0.0}
    return {"frames_bit_identical": exact, "detections": total, "unmatched": unmatched,
            "d_conf": q(dconf), "d_box_px": q(dbox), "d_sigma_px": q(dsig), "d_feat_max": float(max(dfeat))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", default=str(ROOT / "runs/phase3_stage2/p3_vis_seed0/weights/best.pt"))
    ap.add_argument("--n", type=int, default=512)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 8, 16, 32])
    ap.add_argument("--out", default=str(ROOT / "docs/eval/corruption_v2/batched_inference.json"))
    args = ap.parse_args()

    paths = [ln.strip() for ln in PV.read_text(encoding="utf-8").splitlines() if ln.strip()]
    idx = np.linspace(0, len(paths) - 1, args.n).astype(int)        # day and night, every run
    paths = [paths[i] for i in idx]
    frames = [cv2.imread(p) for p in paths]
    p = UQPredictor(args.weights, device=0, imgsz=640, conf=0.001)
    for f in frames[:4]:
        p(f)                                                        # warm-up

    torch.cuda.synchronize()
    t = time.perf_counter()
    ref = [p(f) for f in frames]
    torch.cuda.synchronize()
    ref_fps = len(frames) / (time.perf_counter() - t)
    gts = [load_gt(pp, r["image_hw"]) for pp, r in zip(paths, ref)]
    for r, pp in zip(ref, paths):
        r["image_path"] = pp
    base = local_ap50_95(ref, gts)

    out = {"weights": args.weights, "n_frames": len(frames), "torch": torch.__version__,
           "cudnn_benchmark": torch.backends.cudnn.benchmark, "gpu": torch.cuda.get_device_name(0),
           "call_fps": ref_fps, "batches": {}}
    print(f"__call__ (batch 1): {ref_fps:.1f} frames/s, mAP50-95 {base['map50_95']:.6f}")
    for b in args.batches:
        p.predict_batch(frames[:b])                                 # warm-up this shape
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        t = time.perf_counter()
        got = []
        for i in range(0, len(frames), b):
            got += p.predict_batch(frames[i:i + b])
        torch.cuda.synchronize()
        fps = len(frames) / (time.perf_counter() - t)
        for r, pp in zip(got, paths):
            r["image_path"] = pp
        m = local_ap50_95(got, gts)
        c = compare(ref, got)
        c.update(fps=fps, speedup=fps / ref_fps, peak_mem_gb=torch.cuda.max_memory_allocated() / 2**30,
                 d_map50_95=m["map50_95"] - base["map50_95"],
                 d_ship_ap50_95=m["per_class"][0]["ap50_95"] - base["per_class"][0]["ap50_95"])
        out["batches"][b] = c
        print(f"B={b:3d}: {fps:6.1f} frames/s ({c['speedup']:.2f}x), peak {c['peak_mem_gb']:.2f} GB | "
              f"bit-identical frames {c['frames_bit_identical']}/{len(frames)}, unmatched {c['unmatched']}/{c['detections']}, "
              f"max|dconf| {c['d_conf']['max']:.2e}, max|dbox| {c['d_box_px']['max']:.2e} px, "
              f"max|dsigma| {c['d_sigma_px']['max']:.2e} px, max|dfeat| {c['d_feat_max']:.2e}, "
              f"d mAP {c['d_map50_95']:+.2e}, d ship AP {c['d_ship_ap50_95']:+.2e}", flush=True)
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
