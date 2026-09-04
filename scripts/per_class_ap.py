"""Per-class AP from a prediction cache (decision D-3, handoff 2026-08-19 §4.2).

mAP is macro-averaged over classes, so a class that collapses to ~0 AP is
invisible in the headline number. This splits a cache by class and reports each
class's AP separately, plus GT and detection counts, so "buoys are invisible in
640x512 thermal" can be distinguished from "ship AP is also 0.07".

Reuses `matching.map50_95` on class-filtered records/GT rather than
reimplementing AP, so the numbers are produced by the same code as every other
mAP in the project.

Usage:
    python scripts/per_class_ap.py --cache runs/cache/gauss_ir_paired_clean.pkl \
                                   --cache runs/cache/gauss_vis_paired_clean.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.matching import load_gt, map50_95

NAMES = {0: "ship", 1: "buoy"}


def per_class(records: list[dict]) -> tuple[dict, dict]:
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in records]
    overall = map50_95(records, gts)
    classes = sorted({int(c) for g in gts for c in g["cls"]})
    out = {}
    for c in classes:
        rec_c, gt_c = [], []
        for r, g in zip(records, gts):
            m = np.asarray(r["cls"], dtype=int) == c
            rec_c.append({"boxes_xyxy": np.asarray(r["boxes_xyxy"]).reshape(-1, 4)[m],
                          "conf": np.asarray(r["conf"])[m],
                          "cls": np.asarray(r["cls"], dtype=int)[m]})
            gm = np.asarray(g["cls"], dtype=int) == c
            gt_c.append({"boxes_xyxy": g["boxes_xyxy"][gm], "cls": g["cls"][gm]})
        res = map50_95(rec_c, gt_c)
        out[c] = {**res,
                  "n_gt": int(sum(int((np.asarray(g["cls"], dtype=int) == c).sum()) for g in gts)),
                  "n_det": int(sum(len(r["conf"]) for r in rec_c))}
    return overall, out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache", action="append", required=True)
    parser.add_argument("--out", default="runs/eval/per_class_ap.md")
    args = parser.parse_args()
    load_config(args.config)

    lines = ["| cache | class | AP@50-95 | AP@50 | GT boxes | detections |", "|---|---|---:|---:|---:|---:|"]
    for path in args.cache:
        records, _ = load_cache(path)
        overall, per = per_class(records)
        tag = Path(path).stem
        print(f"\n{tag}: macro mAP@50-95 {overall['map50_95']:.5f}  mAP@50 {overall['map50']:.5f}")
        lines.append(f"| {tag} | **all (macro)** | {overall['map50_95']:.5f} | {overall['map50']:.5f} | | |")
        for c, r in per.items():
            name = NAMES.get(c, str(c))
            print(f"  {name:6s} AP@50-95 {r['map50_95']:.5f}  AP@50 {r['map50']:.5f}  "
                  f"GT {r['n_gt']:6d}  dets {r['n_det']:7d}")
            lines.append(f"| {tag} | {name} | {r['map50_95']:.5f} | {r['map50']:.5f} | {r['n_gt']} | {r['n_det']} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"\n[per-class] -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
