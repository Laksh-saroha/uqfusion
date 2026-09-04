"""Idea I7 -- `cap_ir_scale` is a constant where the gate already knows the answer.

A2 re-swept the ratio on 26m and found x8/x16 marginally BETTER on clean and
WORSE on `clean x IR glare_s2` / `blur_s2`, so x4 was kept as the smallest value
reaching the best worst-cell. That tie is not a tuning limit -- it is a single
constant being asked to serve two regimes at once:

    a healthy IR deserves more weight than x4 gives it,
    a damaged IR deserves less,

and the system already computes, per frame, which regime it is in. `ir_ok` and
the IR health terms exist and are consulted for the veto; they are not consulted
for the weight. This measures what conditioning the scale on them is worth.

Implementation note: the scale is a scalar on `ctx.cap_ir`, so a per-frame
version is produced by running the full system twice -- once at each scale -- and
splicing the per-frame fused records by the gate flag. That is exactly equivalent
to a per-frame scale and needs no change under `src/`, which keeps every published
number reproducible while this is being priced.

Usage:
    python scripts/sweep_cap_ir_gated.py --cache-dir runs/cache_m
"""

from __future__ import annotations

import argparse
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import fmt, md_table, sgn, write_md   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems              # noqa: E402

#: (VIS condition, IR condition) -- the extended grid of `final_26m_grid_v2`.
CELLS = [("clean", None), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("clean", "fog_s2"),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None),
         ("lowlight", "glare_s2"), ("blur_s3", "glare_s2"), ("fog", None)]


def splice(a_records, b_records, use_b: np.ndarray):
    """Frame i from B where `use_b[i]`, else from A."""
    return [b if flag else a for a, b, flag in zip(a_records, b_records, use_b)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/cap_ir_gated.md")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 4.0, 8.0, 16.0])
    ap.add_argument("--healthy-scale", type=float, default=16.0)
    ap.add_argument("--damaged-scale", type=float, default=1.0)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--cells", type=int, default=None, help="first N cells (preflight)")
    args = ap.parse_args()
    t0 = time.time()

    cells = CELLS[: args.cells] if args.cells else CELLS
    conds = sorted({c for c, _ in cells})
    secs, rows = [], []
    fixed = {}

    for scale in args.scales:
        for vis_cond, ir_cond in cells:
            ctx = load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                               conditions=tuple(conds), ir_condition=ir_cond,
                               cap_ir_scale=scale, verbose=False)
            day = np.flatnonzero(~np.isin(ctx.runs, NIGHT_RUNS))
            night = np.flatnonzero(np.isin(ctx.runs, NIGHT_RUNS))
            res = run_systems(ctx, vis_cond)
            parts = frame_parts(res["fused_gated"], ctx.gts)
            fixed[(scale, vis_cond, ir_cond)] = (parts, ctx, res, day, night)
            rows.append([f"x{scale:g}", f"{vis_cond}/{ir_cond or 'clean'}",
                         fmt(ap_from_parts(parts, sel=day)["map50_95"]),
                         fmt(ap_from_parts(parts, sel=night)["map50_95"])])
    secs.append("## 1. Constant `cap_ir_scale`, every cell\n\n"
                "The tie A2 recorded: larger scales help where IR is intact and hurt "
                "where it is damaged.\n\n"
                + md_table(["scale", "cell", "day", "night"], rows))

    # ---- 2. the gated version --------------------------------------------
    rows = []
    for vis_cond, ir_cond in cells:
        key_h = (args.healthy_scale, vis_cond, ir_cond)
        key_d = (args.damaged_scale, vis_cond, ir_cond)
        key_a = (4.0, vis_cond, ir_cond)
        if key_h not in fixed or key_d not in fixed or key_a not in fixed:
            continue
        (_ph, ctx, res_h, day, night) = fixed[key_h]
        (_pd, _c, res_d, _d, _n) = fixed[key_d]
        (pa, _c2, _ra, _d2, _n2) = fixed[key_a]
        # The gate's own IR-health verdict. `veto_ir` is the frame-level flag the
        # system already computes; where it is set, IR is the stream the gate does
        # not trust, and that is exactly where the weight should fall back.
        vir = np.asarray(res_h["veto_ir"], bool)
        ir_bad = vir if vir.any() else np.zeros(len(ctx.gts), bool)
        gated = splice(res_h["fused_gated"], res_d["fused_gated"], ir_bad)
        pg = frame_parts(gated, ctx.gts)
        d_day = (ap_from_parts(pg, sel=day)["map50_95"]
                 - ap_from_parts(pa, sel=day)["map50_95"])
        d_night = (ap_from_parts(pg, sel=night)["map50_95"]
                   - ap_from_parts(pa, sel=night)["map50_95"])
        bs = bootstrap_delta(pg, pa, sel=day, n_boot=args.n_boot, seed=0) if args.n_boot else None
        rows.append([f"{vis_cond}/{ir_cond or 'clean'}", fmt(ir_bad.mean(), 3),
                     fmt(ap_from_parts(pg, sel=day)["map50_95"]), sgn(d_day),
                     f"[{sgn(bs['ci_lo'])}, {sgn(bs['ci_hi'])}]" if bs else "--",
                     sgn(d_night)])
    secs.append(f"## 2. Gate-conditional: x{args.healthy_scale:g} where the gate trusts IR, "
                f"x{args.damaged_scale:g} where it does not\n\n"
                f"Delta is against the adopted constant x4. `IR vetoed` is the share of "
                f"frames taking the damaged branch -- **if it is 0.000 the arm is "
                f"identical to the constant by construction and the cell says nothing**.\n\n"
                + md_table(["cell", "IR vetoed", "gated day", "delta day", "95% CI",
                            "delta night"], rows))

    secs.append("## 3. Decision rule\n\n"
                "Adopt only if the gated arm is at or above the x4 constant on **every** "
                "cell, including the corrupted-IR ones that priced x4 in the first place. "
                "A gain on clean bought with a loss on `clean x IR fog_s2` is the same "
                "trade A2 already refused.\n\n"
                "If `IR vetoed` is near zero everywhere, the gate's IR-health flag is not "
                "firing on these cells and the idea needs a different conditioning signal "
                "(`ir_d2`, the multivariate health term) rather than a different scale.")
    secs.append(f"---\n\n_Generated by `scripts/sweep_cap_ir_gated.py` in "
                f"{time.time() - t0:.1f}s._")
    write_md(args.out, "Gate-conditional cap_ir_scale (I7)", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
