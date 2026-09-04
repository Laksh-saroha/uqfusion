"""Ideas I3/I9 -- the duplicate population, and whether merging or suppressing helps.

Two facts the published numbers never exercised:

  1. **46.6% of VIS day boxes have a same-class sibling at IoU >= 0.6.** `iou_thr`
     is 0.85, so none of them cluster; and `single_passthrough` returns a
     one-stream frame BEFORE WBF is called at all. So `sigma_weighted_fusion`'s
     inverse-variance coordinate averaging -- the mechanism this project is named
     for -- has never run on a multi-member cluster in any published number.

  2. Running it is NEGATIVE. Measured 2026-09-01: mAP50 rises (duplicates removed)
     while mAP50-95 falls, at every `iou_thr` from 0.5 to 0.8, with sigma-weighting
     worth +-0.0002 against plain averaging.

     > C4 concluded that cross-modal merging fails because the IR box is worse
     > localised and the registration residual makes it worse still. That
     > explanation was too specific. Merging fails because the BEST MEMBER IS
     > ALREADY THE BEST ESTIMATE, and any average is a move away from it. Two
     > boxes from the same detector on the same frame, in perfect registration,
     > lose just as much.

This script re-measures both under leave-one-run-out with a bootstrap, and adds
the arms the ad-hoc pilot did not have: soft-NMS, and choosing the surviving
member by sigma rather than confidence. It is also the reference measurement for
I9 -- if merging is off everywhere, that should be a stated decision and not an
accident of the value 0.85.

Usage:
    python scripts/probe_within_modality.py --cache-dir runs/cache_m
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import (ROOT, ap_of, day_night, fmt, gts_for, load_records,  # noqa: E402
                           loro_folds, md_table, sgn, subsample, write_md)

sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.matching import iou_matrix                     # noqa: E402
from uqfusion.uq.fusion import sigma_weighted_fusion              # noqa: E402

NORM = np.array([640.0, 640.0, 640.0, 640.0])


def merge_wbf(rec, iou_thr, use_sigma):
    b = np.asarray(rec["boxes_xyxy"], float).reshape(-1, 4)
    if not len(b):
        return dict(rec)
    hw = rec.get("image_hw", (640, 640))
    norm = np.array([hw[1], hw[0], hw[1], hw[0]], float)
    s = np.asarray(rec["sigma_ltrb"], float).reshape(-1, 4) / norm
    fb, fs, fl = sigma_weighted_fusion(
        [np.clip(b / norm, 0.0, 1.0).tolist()], [np.asarray(rec["conf"], float).tolist()],
        [np.asarray(rec["cls"], float).tolist()], [s], [1.0],
        iou_thr=iou_thr, use_sigma=use_sigma)
    return {**rec, "boxes_xyxy": np.asarray(fb) * norm, "conf": np.asarray(fs),
            "cls": np.asarray(fl).astype(int)}


def suppress(rec, iou_thr, key="conf", soft=False, sigma_nms=0.5):
    """Greedy NMS. `key` picks the survivor; `soft` decays instead of deleting."""
    b = np.asarray(rec["boxes_xyxy"], float).reshape(-1, 4)
    if not len(b):
        return dict(rec)
    c = np.asarray(rec["conf"], float).copy()
    cl = np.asarray(rec["cls"]).astype(int)
    sg = np.asarray(rec["sigma_ltrb"], float).reshape(-1, 4).mean(1)
    rank = c if key == "conf" else c / (1.0 + sg)
    keep_all = []
    for cc in np.unique(cl):
        m = np.flatnonzero(cl == cc)
        order = m[np.argsort(-rank[m])].tolist()
        sel = []
        while order:
            i = order.pop(0)
            sel.append(i)
            if not order:
                break
            iou = iou_matrix(b[i:i + 1], b[order])[0]
            if soft:
                c[order] = c[order] * np.exp(-(iou ** 2) / sigma_nms)
                order = [o for o, v in zip(order, c[order]) if v > 1e-4]
            else:
                order = [o for o, v in zip(order, iou) if v < iou_thr]
        keep_all += sel
    k = np.array(sorted(keep_all), int)
    return {**rec, "boxes_xyxy": b[k], "conf": c[k], "cls": cl[k]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--vis-cache", default=None)
    ap.add_argument("--out", default="runs/eval/within_modality.md")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    p = Path(args.vis_cache) if args.vis_cache else Path(args.cache_dir) / "gauss_vis_paired_clean.pkl"
    vis, meta = load_records(ROOT / p)
    if args.limit:
        vis = subsample(vis, args.limit)
    gts = gts_for(vis)
    runs, day, _n = day_night(vis)
    base_parts = frame_parts(vis, gts)
    base = ap_of(vis, gts, sel=day)
    secs = [f"Source: `{p}`  \nWeights: `{meta.get('weights')}`  \n{len(day)} day frames.  \n"
            f"Baseline (shipped: no within-modality merge) mAP50-95 "
            f"**{fmt(base['map50_95'])}**, mAP50 {fmt(base['map50'])}, "
            f"{np.mean([len(vis[i]['conf']) for i in day]):.1f} boxes/frame."]

    # ---- 1. the duplicate census -----------------------------------------
    rows = []
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        dup = tot = 0
        for i in day:
            b = np.asarray(vis[i]["boxes_xyxy"], float).reshape(-1, 4)
            cl = np.asarray(vis[i]["cls"]).astype(int)
            for cc in np.unique(cl):
                bb = b[cl == cc]
                tot += len(bb)
                if len(bb) > 1:
                    M = iou_matrix(bb, bb)
                    np.fill_diagonal(M, 0.0)
                    dup += int((M.max(1) >= thr).sum())
        rows.append([f"{thr:.2f}", dup, tot, fmt(dup / max(tot, 1), 3)])
    secs.append("## 1. Duplicate census -- boxes with a same-class sibling\n\n"
                "`iou_thr` is 0.85, so everything below that line never forms a WBF "
                "cluster, and `single_passthrough` skips WBF entirely on one-stream "
                "frames.\n\n"
                + md_table(["sibling IoU >=", "boxes with a sibling", "total", "share"], rows))

    # ---- 2. merge (the mechanism the project is named for) ---------------
    arms = []
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        for us in (False, True):
            arms.append((f"WBF merge @{thr:.2f}" + (" sigma-weighted" if us else " plain"),
                         lambda r, t=thr, u=us: merge_wbf(r, t, u)))
    # ---- 3. suppress (keep the best member, delete the rest) -------------
    for thr in (0.5, 0.6, 0.7, 0.8, 0.9):
        arms.append((f"NMS @{thr:.2f} by conf", lambda r, t=thr: suppress(r, t, "conf")))
    for thr in (0.6, 0.7, 0.8):
        arms.append((f"NMS @{thr:.2f} by conf/(1+sigma)",
                     lambda r, t=thr: suppress(r, t, "sigma")))
    arms.append(("soft-NMS sigma 0.5", lambda r: suppress(r, 0.0, "conf", soft=True)))

    rows = []
    for name, fn in arms:
        out = [fn(r) for r in vis]
        parts = frame_parts(out, gts)
        a = ap_of(out, gts, sel=day)
        if args.n_boot:
            bs = bootstrap_delta(parts, base_parts, sel=day, n_boot=args.n_boot, seed=0)
            lo, hi = bs["ci_lo"], bs["ci_hi"]
        else:
            lo = hi = float("nan")
        # leave-one-run-out spread, so a gain that lives in one run is visible
        per_run = [ap_of(out, gts, sel=te)["map50_95"] - ap_of(vis, gts, sel=te)["map50_95"]
                   for _h, _tr, te in loro_folds(runs, day)]
        rows.append([name, fmt(a["map50"]), fmt(a["map50_95"]), sgn(a["map50_95"] - base["map50_95"]),
                     f"[{sgn(lo, 4)}, {sgn(hi, 4)}]" if np.isfinite(lo) else "--",
                     sgn(min(per_run)) + " / " + sgn(max(per_run)) if per_run else "--",
                     f"{np.mean([len(o['conf']) for o in [out[i] for i in day]]):.1f}"])
    secs.append("## 2. Merge vs suppress, day frames\n\n"
                f"Paired bootstrap n={args.n_boot} against the shipped baseline; "
                "`worst/best run` is the leave-one-run-out spread of the same delta.\n\n"
                + md_table(["arm", "mAP50", "mAP50-95", "delta", "95% CI",
                            "worst/best run", "bx/fr"], rows))

    secs.append(
        "## 3. How to read this\n\n"
        "If `mAP50` rises while `mAP50-95` falls, the arm is trading localisation for "
        "precision: it removes duplicate false positives (which AP@50 rewards) and "
        "drags surviving coordinates (which AP@75+ punishes). If sigma-weighted and "
        "plain merging differ by less than the CI, sigma is not the deciding factor "
        "and the inverse-variance argument does not apply here.\n\n"
        "**For I9:** whichever `iou_thr` wins should become an explicit merge "
        "parameter separate from `support_iou`, so that 'merging is off' is a decision "
        "with a number behind it rather than a side effect of 0.85.")
    secs.append(f"---\n\n_Generated by `scripts/probe_within_modality.py` in "
                f"{time.time() - t0:.1f}s._")
    write_md(args.out, "Within-modality merging and suppression (I3/I9)", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
