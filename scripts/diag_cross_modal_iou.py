"""How often do VIS and IR boxes actually meet? — the null-result explanation for A1.

sigma-weighted WBF (`eval_sigma_wbf.py`) changed pooled mAP by +/-0.0001. That is
either "sigma carries no useful information" or "sigma never got the chance to
act", and those have opposite implications for the paper. This separates them.

Inverse-variance averaging only does anything inside a cluster of TWO OR MORE
boxes. A cluster forms when a VIS box and an IR box overlap at IoU > `iou_thr`.
So the question is purely geometric and needs no detector: after mapping IR boxes
through the per-run homography, what fraction of VIS detections have ANY IR
detection above each threshold?

This is measured on detections, not labels, and involves no fitting, so it is
descriptive for every run including the held-out one.

Usage:
    python scripts/diag_cross_modal_iou.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.uq.fusion import apply_homography

THRESHOLDS = (0.85, 0.70, 0.55, 0.40, 0.25, 0.10)


def pairwise_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """IoU of every row of `a` against every row of `b` -> (len(a), len(b))."""
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / np.clip(area_a + area_b - inter, 1e-9, None)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-dir", default="runs/cache")
    parser.add_argument("--homography", default="runs/derived/homography_ir_to_vis.json")
    parser.add_argument("--condition", default="clean")
    parser.add_argument("--conf", type=float, default=0.0,
                        help="ignore detections below this confidence (0 = every cached box)")
    parser.add_argument("--out", default="runs/eval/cross_modal_iou.md")
    args = parser.parse_args()
    load_config(args.config)

    cache_dir = Path(args.cache_dir)
    h_by_run = {k: np.asarray(v["H_ir_canvas_to_vis_canvas"], dtype=float) for k, v in
                json.loads(Path(args.homography).read_text(encoding="utf-8"))["runs"].items()}
    vis_recs, _ = load_cache(cache_dir / f"gauss_vis_paired_{args.condition}.pkl")
    ir_recs, _ = load_cache(cache_dir / "gauss_ir_paired_clean.pkl")
    runs = np.asarray([Path(r["image_path"]).parent.name for r in vis_recs])

    best = []          # per VIS box: best IoU against any IR box in that frame
    per_run = {}
    n_vis = n_ir = 0
    for rv, ri, run in zip(vis_recs, ir_recs, runs):
        v = np.asarray(rv["boxes_xyxy"], dtype=float).reshape(-1, 4)
        i = apply_homography(np.asarray(ri["boxes_xyxy"], dtype=float).reshape(-1, 4), h_by_run[run])
        if args.conf > 0:
            v = v[np.asarray(rv["conf"], dtype=float) >= args.conf]
            i = i[np.asarray(ri["conf"], dtype=float) >= args.conf]
        n_vis += len(v)
        n_ir += len(i)
        if not len(v):
            continue
        m = pairwise_iou(v, i).max(axis=1) if len(i) else np.zeros(len(v))
        best.append(m)
        per_run.setdefault(run, []).append(m)

    best = np.concatenate(best) if best else np.zeros(0)
    print(f"[xmodal] condition={args.condition}  VIS boxes {n_vis:,}  IR boxes {n_ir:,}")
    print(f"[xmodal] per-VIS-box best IoU against any IR box, over {len(best):,} VIS detections")

    lines = [f"# Cross-modal box agreement — why A1 is a no-op at `iou_thr` 0.85", "",
             f"Condition `{args.condition}`, {n_vis:,} VIS detections and {n_ir:,} IR detections "
             f"over {len(vis_recs):,} paired frames, IR mapped through the per-run homography. "
             f"For each VIS box: the best IoU against any IR box in the same frame.", "",
             "Inverse-variance averaging can only act inside a cluster of 2+ boxes, so this is "
             "the ceiling on how often sigma-weighted WBF has anything to do at all.", "",
             "| WBF iou_thr | VIS boxes with an IR partner above it | share |", "|---|---|---|"]
    for thr in THRESHOLDS:
        n = int((best > thr).sum())
        lines.append(f"| {thr:.2f} | {n:,} | **{n / max(len(best), 1):.2%}** |")
        print(f"[xmodal]   IoU > {thr:.2f}: {n:>7,}  ({n / max(len(best), 1):6.2%})")

    lines += ["", "| percentile of best-IoU | value |", "|---|---|"]
    for q in (50, 75, 90, 95, 99):
        lines.append(f"| p{q} | {np.percentile(best, q):.4f} |")

    lines += ["", "## By run", "", "| run | VIS boxes | share with IR partner at IoU > 0.85 | > 0.55 |",
              "|---|---|---|---|"]
    for run in sorted(per_run):
        m = np.concatenate(per_run[run])
        lines.append(f"| {run} | {len(m):,} | {(m > 0.85).mean():.2%} | {(m > 0.55).mean():.2%} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[xmodal] -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
