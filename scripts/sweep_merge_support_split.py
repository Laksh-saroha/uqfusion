"""Idea I9 -- one `iou_thr` is doing two incompatible jobs.

`iou_thr` = 0.85 currently governs BOTH:

  * cross-modal merging, which C4/C5 showed must not happen (80x more geometric
    agreement, AP falls, every alignment arm negative held-out); and
  * within-modality clustering, which `probe_within_modality.py` measures as
    negative too.

At 0.85 both are nearly off -- but that is a side effect of one number chosen for
one of the two jobs, not a decision about either. `support_iou` (0.30) and
`support_gamma` (0.5) are already separate parameters, so the split only needs to
be MEASURED: sweep the merge threshold and the support threshold independently
and report the surface.

The expected result is that merging is off at every value and support carries the
whole gain, which would mean the architecture is a score-modulated concatenation
and should say so. This script is what turns that from an inference into a table.

Usage:
    python scripts/sweep_merge_support_split.py --cache-dir runs/cache_m
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import fmt, md_table, sgn, write_md   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import (NIGHT_RUNS, TEST_RUNS, TUNE_RUNS, load_context,   # noqa: E402
                               run_systems)

CELLS = [("clean", None), ("clean", "glare_s2"), ("blur_s3", None), ("rain_s2", None)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/merge_support_split.md")
    ap.add_argument("--merge-ious", type=float, nargs="+", default=[0.55, 0.70, 0.85, 0.95])
    ap.add_argument("--support-ious", type=float, nargs="+", default=[0.0, 0.10, 0.30, 0.55])
    ap.add_argument("--support-gammas", type=float, nargs="+", default=[0.5, 1.0])
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--cells", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    cells = CELLS[: args.cells] if args.cells else CELLS
    conds = sorted({c for c, _ in cells})
    secs = []

    # Baseline: exactly the shipped configuration.
    ctx0 = load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                        conditions=tuple(conds), verbose=True)
    day = np.flatnonzero(~np.isin(ctx0.runs, NIGHT_RUNS))
    night = np.flatnonzero(np.isin(ctx0.runs, NIGHT_RUNS))
    tune = np.flatnonzero(np.isin(ctx0.runs, TUNE_RUNS))
    test = np.flatnonzero(np.isin(ctx0.runs, TEST_RUNS))
    base_parts = frame_parts(run_systems(ctx0, "clean")["fused_gated"], ctx0.gts)
    base_day = ap_from_parts(base_parts, sel=day)["map50_95"]
    secs.append(f"Shipped configuration: merge `iou_thr` {ctx0.iou_thr}, "
                f"`support_iou` {ctx0.support_iou}, `support_gamma` {ctx0.support_gamma}.  \n"
                f"Baseline clean/clean day mAP50-95 **{fmt(base_day)}**.  \n"
                f"Frames: day {len(day)}, night {len(night)}, TUNE {len(tune)}, "
                f"TEST {len(test)}.")

    # ---- 1. the 2-D surface on the clean cell ----------------------------
    rows = []
    for miou in args.merge_ious:
        ctx = load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                           conditions=tuple(conds), iou_thr=miou, verbose=False)
        for siou in args.support_ious:
            for g in (args.support_gammas if siou > 0 else [0.0]):
                res = run_systems(ctx, "clean", support_iou=siou, support_gamma=g)
                parts = frame_parts(res["fused_gated"], ctx.gts)
                d = ap_from_parts(parts, sel=day)["map50_95"]
                rows.append([f"{miou:.2f}", f"{siou:.2f}", f"{g:g}", fmt(d),
                             sgn(d - base_day),
                             fmt(ap_from_parts(parts, sel=tune)["map50_95"]),
                             fmt(ap_from_parts(parts, sel=test)["map50_95"])])
    secs.append("## 1. Merge threshold x support threshold, clean cell\n\n"
                "`merge iou` is WBF's clustering threshold (the coordinate-moving one); "
                "`support iou` is the score-only confirmation that never touches "
                "coordinates. They have never been varied independently.\n\n"
                + md_table(["merge iou", "support iou", "gamma", "day", "delta day",
                            "TUNE", "TEST"], rows))

    # ---- 2. does the best split hold across cells? -----------------------
    rows = []
    for miou in args.merge_ious:
        ctx = load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                           conditions=tuple(conds), iou_thr=miou, verbose=False)
        for vis_cond, ir_cond in cells:
            if ir_cond is not None:
                ctx_c = load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                                     conditions=tuple(conds), iou_thr=miou,
                                     ir_condition=ir_cond, verbose=False)
            else:
                ctx_c = ctx
            res = run_systems(ctx_c, vis_cond,
                              support_iou=ctx0.support_iou, support_gamma=ctx0.support_gamma)
            parts = frame_parts(res["fused_gated"], ctx_c.gts)
            base_res = run_systems(
                load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                             conditions=tuple(conds), ir_condition=ir_cond, verbose=False),
                vis_cond)
            bparts = frame_parts(base_res["fused_gated"], ctx_c.gts)
            dd = (ap_from_parts(parts, sel=day)["map50_95"]
                  - ap_from_parts(bparts, sel=day)["map50_95"])
            dn = (ap_from_parts(parts, sel=night)["map50_95"]
                  - ap_from_parts(bparts, sel=night)["map50_95"])
            bs = bootstrap_delta(parts, bparts, sel=day, n_boot=args.n_boot, seed=0) \
                if args.n_boot else None
            rows.append([f"{miou:.2f}", f"{vis_cond}/{ir_cond or 'clean'}", sgn(dd),
                         f"[{sgn(bs['ci_lo'])}, {sgn(bs['ci_hi'])}]" if bs else "--", sgn(dn)])
    secs.append("## 2. Merge threshold across cells, support held at the adopted value\n\n"
                "Delta against the shipped `iou_thr` 0.85 on the same cell.\n\n"
                + md_table(["merge iou", "cell", "delta day", "95% CI", "delta night"], rows))

    secs.append("## 3. Decision rule\n\n"
                "If no `merge iou` beats 0.95 (i.e. merging fully disabled), the honest "
                "description of this architecture is **score-modulated concatenation**: "
                "two detection lists pooled, with the cross-modal term re-weighting one of "
                "them, and no coordinate ever combined. That should be named in the "
                "preset as a separate `merge_iou` parameter set to 'off' rather than left "
                "implicit in the value 0.85 -- so that a future detector swap re-prices it "
                "deliberately, the way D2 wishes the veil veto had been.")
    secs.append(f"---\n\n_Generated by `scripts/sweep_merge_support_split.py` in "
                f"{time.time() - t0:.1f}s._")
    write_md(args.out, "Merge threshold vs support threshold (I9)", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
