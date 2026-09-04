"""Idea I6 -- buoy is half the macro metric and nothing has ever touched it.

`mAP` macro-averages over classes. Buoy carries **600 GT boxes against ship's
10,663** -- 5% of the objects and 50% of the number, so a point of buoy AP is
worth 18x a point of ship AP per GT box. And:

  * IR is `nc=1` (ship only), so the IR stream can never supply a buoy;
  * every gate constant -- the capability weights, `cap_ir_scale`, the veto axes,
    `support_iou`/`support_gamma`, `ir_nms` -- is class-agnostic;
  * every headline table in the project reports SHIP AP, which is blind to all
    of it. C6 found `class_veto` worth "exactly zero" measured that way.

So the class the metric rewards most is the one no lever addresses. This script
prices the class-conditional versions of the two levers that are actually live.

Three sections:

  1. **Where the macro number really lives** -- per-class AP, oracle headroom and
     recall ceiling, and what one point of each class is worth to the macro.
  2. **Class-conditional `support`.** The adopted term boosts a VIS box when IR
     overlaps it. For buoy that term is structurally dead -- IR emits no buoys --
     so a single global `support_gamma` is fitted on ship and applied to a class
     it cannot help. Sweeps gamma per class.
  3. **The buoy-only ceiling.** What the whole system would score if buoy AP were
     oracle-re-ranked and ship left alone, and vice versa -- which says which
     class the next unit of effort should go to.

Usage:
    python scripts/sweep_per_class.py --cache-dir runs/cache_m
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import (ROOT, day_night, fmt, gts_for, load_records,  # noqa: E402
                           md_table, oracle_curves, sgn, subsample, write_md)

sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import ap_from_parts, frame_parts     # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402
from uqfusion.eval.matching import iou_matrix                      # noqa: E402

CLS_NAMES = {0: "ship", 1: "buoy"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/per_class_levers.md")
    ap.add_argument("--gammas", type=float, nargs="+", default=[0.0, 0.5, 1.0, 2.0])
    ap.add_argument("--conditions", nargs="+", default=["clean"])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    vis, meta = load_records(ROOT / args.cache_dir / "gauss_vis_paired_clean.pkl")
    if args.limit:
        vis = subsample(vis, args.limit)
    gts = gts_for(vis)
    runs, day, _n = day_night(vis)
    parts = frame_parts(vis, gts)
    secs = [f"Source: `{args.cache_dir}`  \nWeights: `{meta.get('weights')}`  \n"
            f"{len(day)} day frames."]

    # ---- 1. where the macro number lives ---------------------------------
    A, O, per_class, recall = oracle_curves(parts, day)
    n_cls = max(len(per_class), 1)
    rows = []
    for c, d in sorted(per_class.items()):
        rows.append([f"{c} {CLS_NAMES.get(c, '?')}", d["n_gt"],
                     fmt(d["n_gt"] / max(sum(x['n_gt'] for x in per_class.values()), 1), 3),
                     fmt(1.0 / n_cls, 3), d["n_pred"], fmt(d["actual"][0]),
                     fmt(d["actual"].mean()), fmt(d["oracle"].mean()),
                     sgn(d["oracle"].mean() - d["actual"].mean()),
                     fmt(recall[c][0], 3), fmt(recall[c][5], 3)])
    secs.append("## 1. Where the macro metric actually spends\n\n"
                "`share of GT` is how much of the data the class is; `weight in mAP` is "
                "how much of the number it is. The gap between those two columns is the "
                "whole point of this script.\n\n"
                + md_table(["class", "n_gt", "share of GT", "weight in mAP", "n_pred",
                            "AP50", "AP50-95", "oracle", "headroom", "rec@50", "rec@75"],
                           rows))

    # ---- 2. class-conditional support ------------------------------------
    ctx = load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                       conditions=tuple(args.conditions), verbose=False)
    cday = np.flatnonzero(~np.isin(ctx.runs, NIGHT_RUNS))
    base = run_systems(ctx, args.conditions[0])
    bparts = frame_parts(base["fused_gated"], ctx.gts)
    b = ap_from_parts(bparts, sel=cday)
    rows = [["adopted (gamma %g, all classes)" % ctx.support_gamma,
             fmt(b["map50_95"]), "--"]
            + [fmt(b["per_class"].get(c, {}).get("ap50_95", float("nan")))
               for c in sorted(CLS_NAMES)]]
    for g in args.gammas:
        res = run_systems(ctx, args.conditions[0], support_gamma=g)
        p = frame_parts(res["fused_gated"], ctx.gts)
        a = ap_from_parts(p, sel=cday)
        rows.append([f"support_gamma {g:g} (all classes)", fmt(a["map50_95"]),
                     sgn(a["map50_95"] - b["map50_95"])]
                    + [fmt(a["per_class"].get(c, {}).get("ap50_95", float("nan")))
                       for c in sorted(CLS_NAMES)])
    secs.append("## 2. `support_gamma` and the class it cannot help\n\n"
                "IR is `nc=1`, so a buoy box can never have cross-modal support. If the "
                "buoy column is flat across every gamma while ship moves, the term is "
                "being fitted on one class and charged to both -- and the right form is "
                "per class, not global.\n\n"
                + md_table(["arm", "macro mAP50-95", "delta"]
                           + [f"AP {CLS_NAMES[c]}" for c in sorted(CLS_NAMES)], rows))

    # ---- 3. per-class oracle: where should the next unit of effort go? ----
    rows = []
    for target in sorted(per_class):
        mixed = 0.0
        for c, d in per_class.items():
            mixed += (d["oracle"].mean() if c == target else d["actual"].mean())
        mixed /= n_cls
        rows.append([f"oracle-rerank {CLS_NAMES.get(target, target)} only, other class left alone",
                     fmt(mixed), sgn(mixed - A.mean())])
    rows.append(["oracle-rerank both", fmt(O.mean()), sgn(O.mean() - A.mean())])
    secs.append("## 3. Which class is the next unit of effort worth spending on?\n\n"
                "Macro mAP if ONE class were perfectly re-ranked and the other left "
                "exactly as it is. The larger number is the class where re-ranking "
                "work pays.\n\n"
                + md_table(["arm", "macro mAP50-95", "delta"], rows))

    secs.append("## 4. Decision rule\n\n"
                "* **Buoy's oracle-only row beats ship's** -> the next re-ranker (I1) "
                "should be fitted and selected per class, not pooled, and buoy deserves "
                "its own feature set. A pooled fit is dominated by ship's 10,663 boxes "
                "and will optimise the class that matters least per unit of metric.\n"
                "* **Buoy's `AP` column is flat across every `support_gamma`** -> the "
                "adopted term is a ship-only lever wearing a global name. Split it.\n"
                "* **Buoy's recall ceiling is the binding term** -> no re-ranking helps; "
                "that is a detector or resolution problem (I4) on small objects.")
    secs.append(f"---\n\n_Generated by `scripts/sweep_per_class.py` in {time.time() - t0:.1f}s._")
    write_md(args.out, "Per-class levers -- the half of the metric nothing touches (I6)", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
