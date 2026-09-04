"""Adoption gate for VIS soft-NMS (I3/I9) -- measured in the REAL fused pipeline.

`probe_within_modality.py` measured soft-NMS on the bare VIS stream and got
+0.0025 [+0.0021, +0.0029], positive on all three held-out day runs. That is not
the number the system ships. The shipped baseline is `fused_gated` under the
`crossmodal26m` preset, whose clean day mAP is 0.3286 rather than the VIS
stream's 0.3233, and five of the eight benchmark cells run `single_passthrough`
-- so the suppression lands on a different object in each cell.

This re-measures it where it would actually live, against the bar the project
uses for adoption: **at or above the shipped constant on EVERY cell**, day and
night, with TUNE and TEST reported separately so a gain that lives in the
selection half is visible.

Usage:
    python scripts/sweep_vis_soft_nms.py --cache-dir runs/cache_m
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
from uqfusion.eval.apmetrics import (ap_from_parts, bootstrap_delta,   # noqa: E402
                                     frame_parts)
from uqfusion.eval.ctx import (NIGHT_RUNS, TEST_RUNS, TUNE_RUNS,       # noqa: E402
                               load_context, run_systems)

#: (vis condition, ir condition) -- the benchmark cells plus the two
#: both-degraded ones the veto work added.
CELLS = [("clean", None), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("clean", "fog_s2"),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None), ("fog", None),
         ("lowlight", "glare_s2"), ("blur_s3", "glare_s2")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/vis_soft_nms_adoption.md")
    ap.add_argument("--sigmas", type=float, nargs="+", default=[0.3, 0.5, 0.7])
    ap.add_argument("--ship-sigma", type=float, default=0.5,
                    help="the width carried into section 2; pre-registered, not tuned")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--cells", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    cells = CELLS[: args.cells] if args.cells else CELLS
    conds = sorted({c for c, _ in cells})
    ircs = sorted({i for _c, i in cells if i})

    def ctx_for(ir_cond, snms):
        return load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                            conditions=tuple(conds), ir_condition=ir_cond,
                            vis_soft_nms=snms, verbose=False)

    base_ctx = {None: ctx_for(None, None)}
    for ic in ircs:
        base_ctx[ic] = ctx_for(ic, None)
    c0 = base_ctx[None]
    day = np.flatnonzero(~np.isin(c0.runs, NIGHT_RUNS))
    night = np.flatnonzero(np.isin(c0.runs, NIGHT_RUNS))
    tune = np.flatnonzero(np.isin(c0.runs, TUNE_RUNS))
    test = np.flatnonzero(np.isin(c0.runs, TEST_RUNS))

    secs = [f"Preset `crossmodal26m`, `{args.cache_dir}`.  \n"
            f"Frames: day {len(day)}, night {len(night)}, TUNE {len(tune)}, "
            f"TEST {len(test)}.  \n"
            "Baseline is the shipped system with `vis_soft_nms` off; every delta is "
            "against the SAME cell."]

    # ---- 1. decay width on the clean cell, TUNE and TEST side by side ----
    bp = frame_parts(run_systems(base_ctx[None], "clean")["fused_gated"], c0.gts)
    b_day = ap_from_parts(bp, sel=day)["map50_95"]
    b_test = ap_from_parts(bp, sel=test)["map50_95"]
    rows = [["off (shipped)", fmt(b_day), "--", "--",
             fmt(ap_from_parts(bp, sel=tune)["map50_95"]), fmt(b_test),
             fmt(ap_from_parts(bp, sel=night)["map50_95"])]]
    by_test: dict[float, float] = {}
    for s in args.sigmas:
        c = ctx_for(None, s)
        p = frame_parts(run_systems(c, "clean")["fused_gated"], c.gts)
        a = ap_from_parts(p, sel=day)["map50_95"]
        t_ap = ap_from_parts(p, sel=test)["map50_95"]
        by_test[s] = t_ap
        bs = bootstrap_delta(p, bp, sel=day, n_boot=args.n_boot, seed=0)
        rows.append([f"soft-NMS sigma {s:g}", fmt(a), sgn(a - b_day),
                     f"[{sgn(bs['ci_lo'])}, {sgn(bs['ci_hi'])}]",
                     fmt(ap_from_parts(p, sel=tune)["map50_95"]), fmt(t_ap),
                     fmt(ap_from_parts(p, sel=night)["map50_95"])])
    secs.append("## 1. Decay width, clean/clean\n\n"
                "`sigma` is the Gaussian width in `exp(-iou^2 / sigma)`; smaller decays "
                "harder. TUNE and TEST are reported side by side because a value chosen "
                "on TUNE and winning only there is the failure mode C8 named.\n\n"
                + md_table(["arm", "day mAP50-95", "delta day", "95% CI", "TUNE",
                            "TEST", "night"], rows))

    # NOT selected on TEST. C7 used the held-out half to REJECT `iou_thr` 0.75 and
    # 0.95, never to pick among survivors, and picking the best TEST number here
    # would turn the only held-out day data the project has into a selection set --
    # the C8 mistake with a new label.
    #
    # So: TEST rejects (any sigma negative there is out), and among the survivors
    # the shipped width is the PRE-REGISTERED one -- 0.5, the value
    # `probe_within_modality.py` measured before this split was ever looked at.
    # TUNE prefers 0.3 and TEST prefers 0.7; that disagreement is the evidence that
    # the width is not resolvable on 364 frames, not a reason to trust either.
    best = args.ship_sigma
    if best in by_test and by_test[best] < b_test:
        raise SystemExit(f"sigma {best} is negative on TEST; the pre-registered "
                         f"width does not survive rejection and must not be shipped")

    # ---- 2. the every-cell bar -------------------------------------------
    rows = []
    for vc, ic in cells:
        b = base_ctx[ic]
        pb = frame_parts(run_systems(b, vc)["fused_gated"], b.gts)
        c = ctx_for(ic, best)
        pa = frame_parts(run_systems(c, vc)["fused_gated"], c.gts)
        d0 = ap_from_parts(pb, sel=day)["map50_95"]
        d1 = ap_from_parts(pa, sel=day)["map50_95"]
        n0 = ap_from_parts(pb, sel=night)["map50_95"]
        n1 = ap_from_parts(pa, sel=night)["map50_95"]
        bs = bootstrap_delta(pa, pb, sel=day, n_boot=args.n_boot, seed=0)
        rows.append([f"{vc}/{ic or 'clean'}", fmt(d0), fmt(d1), sgn(d1 - d0),
                     f"[{sgn(bs['ci_lo'])}, {sgn(bs['ci_hi'])}]",
                     fmt(n0), fmt(n1), sgn(n1 - n0)])
    worst_day = min(float(r[3]) for r in rows)
    worst_night = min(float(r[7]) for r in rows)
    secs.append(f"## 2. Every cell, at sigma {best:g}\n\n"
                "The adoption bar: at or above the shipped system on every cell. A gain "
                "on clean bought with a loss under corruption is the trade A2 refused "
                "for `cap_ir_scale` and D2 wishes the veil veto had been held to.\n\n"
                + md_table(["cell (vis/ir)", "shipped day", "soft-NMS day", "delta day",
                            "95% CI", "shipped night", "soft-NMS night", "delta night"],
                           rows)
                + f"\n\n**Worst cell: day {sgn(worst_day)}, night {sgn(worst_night)}.**")

    secs.append("## 3. Decision rule\n\n"
                "* **Every cell at or above, and TEST agrees with TUNE** -> adopt: set "
                "`vis_soft_nms` in the preset and re-baseline the headline table.\n"
                "* **Any cell below, or TEST disagreeing with TUNE** -> the VIS-stream "
                "result was a property of the stream and not of the system, and the "
                "parameter stays off with the measurement recorded.")
    secs.append(f"---\n\n_Generated by `scripts/sweep_vis_soft_nms.py` in "
                f"{time.time() - t0:.1f}s._")
    write_md(args.out, "VIS soft-NMS -- adoption gate in the fused pipeline", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
