"""Idea I4 screen -- is the AP loss concentrated in small objects?

`image_hw` is (640, 640): the Pohang trees on disk were pre-resized, and
`scripts/prepare_pohang_fullres.py` + `scripts/prep_fullres_lists.py` exist and
have never been run. Re-prepping at 960 or 1280 is a full re-prep plus a retrain
-- the most expensive idea on the list -- so it needs a screen that can refuse it.

The screen: split GT by area (COCO's small/medium/large) and report, per bin,
the recall ceiling and the oracle re-ranking headroom. Three outcomes:

  * loss concentrated in SMALL, and their recall ceiling is the binding term
    -> resolution is the lever, because pixels-on-target is what a small box
    lacks and no re-ranking creates them;
  * loss spread evenly, or concentrated where recall is already high
    -> resolution is NOT the lever and the headroom is ordering, which
    `fit_rerank.py` addresses at a fraction of the cost;
  * small objects barely exist in this dataset -> the question is moot.

Detections are assigned to a bin by their MATCHED GT's area where they match,
and by their own area where they do not, so an FP is charged to the scale it
claims to be.

Usage:
    python scripts/probe_ap_by_size.py --cache-dir runs/cache_m
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import (IOU_LEVELS, ROOT, day_night, fmt, gts_for,  # noqa: E402
                           load_records, md_table, sgn, subsample, write_md)

sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import NL, _ap_from_sorted   # noqa: E402
from uqfusion.eval.matching import iou_matrix, tp_matrix  # noqa: E402

# COCO's convention, on the 640x640 canvas these caches were produced at.
BINS = (("small", 0.0, 32.0 ** 2), ("medium", 32.0 ** 2, 96.0 ** 2),
        ("large", 96.0 ** 2, np.inf))
CLS_NAMES = {0: "ship", 1: "buoy"}


def area(b):
    b = np.asarray(b, float).reshape(-1, 4)
    return np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--vis-cache", default=None)
    ap.add_argument("--out", default="runs/eval/ap_by_size.md")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    p = Path(args.vis_cache) if args.vis_cache else Path(args.cache_dir) / "gauss_vis_paired_clean.pkl"
    vis, meta = load_records(ROOT / p)
    if args.limit:
        vis = subsample(vis, args.limit)
    gts = gts_for(vis)
    _runs, day, _night = day_night(vis)

    # Pool detections with a size tag and a GT-count-per-bin denominator.
    tp, conf, cls, dbin, = [], [], [], []
    ngt: dict[tuple[int, str], int] = {}
    gt_area_all = []
    for i in day:
        r, g = vis[i], gts[i]
        ga = area(g["boxes_xyxy"])
        gt_area_all.append(ga)
        for c, a in zip(g["cls"], ga):
            for name, lo, hi in BINS:
                if lo <= a < hi:
                    ngt[(int(c), name)] = ngt.get((int(c), name), 0) + 1
        b = np.asarray(r["boxes_xyxy"], float).reshape(-1, 4)
        if not len(b):
            continue
        t = tp_matrix(r, g)
        # Charge a detection to its matched GT's scale; an unmatched one to its own.
        da = area(b)
        if len(g["boxes_xyxy"]):
            m = iou_matrix(b, g["boxes_xyxy"])
            m = np.where(np.asarray(r["cls"]).astype(int)[:, None] == g["cls"][None, :], m, 0.0)
            j = m.argmax(1)
            hit = m[np.arange(len(b)), j] >= 0.5
            da = np.where(hit, ga[j] if len(ga) else da, da)
        tags = np.full(len(b), "large", dtype=object)
        for name, lo, hi in BINS:
            tags[(da >= lo) & (da < hi)] = name
        tp.append(t); conf.append(np.asarray(r["conf"], float))
        cls.append(np.asarray(r["cls"]).astype(int)); dbin.append(tags)
    tp = np.concatenate(tp) if tp else np.zeros((0, NL), bool)
    conf = np.concatenate(conf) if len(conf) else np.zeros(0)
    cls = np.concatenate(cls) if len(cls) else np.zeros(0, int)
    dbin = np.concatenate(dbin) if len(dbin) else np.zeros(0, dtype=object)
    gt_area_all = np.concatenate(gt_area_all) if gt_area_all else np.zeros(0)

    secs = [f"Source: `{p}`  \nWeights: `{meta.get('weights')}`  imgsz {meta.get('imgsz')}  \n"
            f"{len(day)} day frames, {len(conf)} detections, {len(gt_area_all)} GT boxes.  \n"
            f"GT area percentiles (px^2 on the 640 canvas): "
            + ", ".join(f"p{q}={np.percentile(gt_area_all, q):.0f}" for q in (5, 25, 50, 75, 95))
            + f"  \nGT side length median: {np.sqrt(np.median(gt_area_all)):.1f} px."]

    rows = []
    for c in sorted({int(x) for x in cls} | {k[0] for k in ngt}):
        for name, _lo, _hi in BINS:
            n = ngt.get((c, name), 0)
            if n == 0:
                continue
            m = (cls == c) & (dbin == name)
            if not m.any():
                rows.append([f"{c} {CLS_NAMES.get(c, '?')}", name, n, 0, "--", "--", "--", "--", "--"])
                continue
            t, cf = tp[m], conf[m]
            a = _ap_from_sorted(t[np.argsort(-cf)], n)
            o = np.empty(NL)
            for k in range(NL):
                o[k] = _ap_from_sorted(t[np.lexsort((-cf, -t[:, k].astype(float)))], n)[k]
            rec = t.sum(0) / n
            rows.append([f"{c} {CLS_NAMES.get(c, '?')}", name, n, int(m.sum()),
                         fmt(a[0]), fmt(a.mean()), fmt(o.mean()), sgn(o.mean() - a.mean()),
                         fmt(rec[0], 3)])
    secs.append("## 1. AP and headroom by object scale\n\n"
                "`rec@50` is the recall CEILING -- the share of GT this bin's detections "
                "cover at all, at any confidence. A low ceiling cannot be fixed by "
                "re-ranking; it needs more pixels on target or a better detector.\n\n"
                + md_table(["class", "size", "n_gt", "n_pred", "AP50", "AP50-95", "oracle",
                            "headroom", "rec@50"], rows))

    # Where is the *total* AP loss? Weight each bin by its share of GT.
    tot = sum(ngt.values())
    share = []
    for name, _lo, _hi in BINS:
        n = sum(v for k, v in ngt.items() if k[1] == name)
        share.append([name, n, fmt(n / max(tot, 1), 3)])
    secs.append("## 2. GT mass by scale\n\n"
                "A bin with a terrible AP and 2% of the GT cannot be the lever.\n\n"
                + md_table(["size", "n_gt", "share of GT"], share))

    secs.append(
        "## 3. How to read this\n\n"
        "* **Loss concentrated in `small`, with a low `rec@50` there, and `small` "
        "carrying real GT mass** -> resolution is the lever. Run "
        "`scripts/prepare_pohang_fullres.py` and retrain (idea I4, expensive).\n"
        "* **Headroom spread evenly, or largest where `rec@50` is already high** -> the "
        "gap is ORDERING, not pixels. `scripts/fit_rerank.py` (I1) addresses the same "
        "AP for a CPU-minutes cost, and I4 should not be launched.\n"
        "* **`small` holds little GT mass** -> the question is moot at this canvas size.")
    secs.append(f"---\n\n_Generated by `scripts/probe_ap_by_size.py` in {time.time() - t0:.1f}s._")
    write_md(args.out, "AP by object scale -- the resolution screen (I4)", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
