"""B-3: Fixed-GT risk-coverage (AURC) for `R_sys` -- TODO-2026-08-20 SS B-3 [was SS0.6].

`run_cheap_fixes.py` SS4 built a risk-coverage table by dropping the
lowest-`R_sys` frames and recomputing mAP over whatever frames remained. That
recomputes the AP denominator (total GT boxes) from the SURVIVING frames every
time the coverage changes, so a "risk" drop can come from the covered subset
containing an easier mix of GT rather than from `R_sys` doing anything -- the
non-monotonicity that made abstention look like a net negative may be an
artifact of a shifting yardstick, not a property of the signal.

Fixed-GT metric (`apmetrics.ap_from_parts(..., gt_sel=...)`, added for this
script): the GT denominator is always the FULL frame set for the condition,
regardless of coverage. Predictions from abstained frames simply stop
contributing true positives -- their GT boxes become permanently missed
detections instead of vanishing from the count. It is monotonicity-honest: risk
can only move because of what `R_sys` chose to drop, not because the yardstick
itself moved.

**Corrected 2026-09-10 (R-A4/F12): this is NOT "the standard selective-risk
definition", as this docstring claimed until now.** Standard selective risk
conditions on the ACCEPTED set -- it is the loss averaged over what the system
chose to answer, so the denominator shrinks with coverage by construction. The
fixed-GT quantity here holds the FULL-condition GT count in the denominator
regardless of coverage, which makes rejected frames count as permanent misses.
That is a different estimand: "how much of all the work did the system get
done", not "how good is the system on what it accepted". Both are legitimate and
both are already reported below, side by side, with a random-rejection control
-- only the LABEL was wrong, and a reader who took the claim at face value would
have compared these numbers against published selective-risk figures that
condition differently. Same defect family as F14: a name that does not pin a
computation.

For each condition, both risk curves (shifting-denominator vs fixed-GT) are
reported against a coverage grid, together with a random-order control (mean
over 20 shuffles) so "AURC(R_sys) < AURC(random)" has something to be measured
against. AURC here is the trapezoidal integral of risk over coverage from 1.0
down to 0.1 -- note this is a DIFFERENT computation from
`uqfusion.eval.metrics.sparsification`'s `aurc`, which is a 20-point grid MEAN
over per-detection 1-IoU risk. Two functions, one name, different quantities;
they must never be quoted against each other. See `metrics.AURC_INTEGRATION`.

CPU-only, cached predictions (~1-2 h with the bootstrap; the core sweep alone
is a couple of minutes). Usage:
    python scripts/eval_risk_coverage_fixed_gt.py
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems  # noqa: E402

CONDITIONS = ("clean", "fog", "lowlight", "glare")
COVERAGE_GRID = np.round(np.arange(1.0, 0.09, -0.05), 2)   # 1.00, 0.95, ..., 0.10
N_RANDOM = 20


def risk_curve(parts: list[dict], order: np.ndarray, fixed_gt: bool) -> list[float]:
    n = len(parts)
    all_idx = np.arange(n)
    out = []
    for cov in COVERAGE_GRID:
        k = max(int(round(cov * n)), 1)
        covered = order[:k]
        gt_sel = all_idx if fixed_gt else covered
        ap = ap_from_parts(parts, sel=covered, gt_sel=gt_sel)["map50_95"]
        out.append(1.0 - ap)
    return out


def aurc(risks: list[float]) -> float:
    return float(np.trapz(risks[::-1], COVERAGE_GRID[::-1]))   # ascending x for trapz


def main() -> int:
    t0 = time.time()
    ctx = load_context(conditions=CONDITIONS, verbose=True)
    rng = np.random.default_rng(0)

    rows, curves = [], {}
    for cond in CONDITIONS:
        res = run_systems(ctx, cond)
        parts = frame_parts(res["fused_gated"], ctx.gts)
        rsys = np.asarray(res["R_sys"])
        n = len(parts)
        order_rsys = np.argsort(-rsys)   # most-reliable-first

        shifting = risk_curve(parts, order_rsys, fixed_gt=False)
        fixed = risk_curve(parts, order_rsys, fixed_gt=True)

        random_curves = []
        for s in range(N_RANDOM):
            order_rand = rng.permutation(n)
            random_curves.append(risk_curve(parts, order_rand, fixed_gt=True))
        random_mean = np.mean(random_curves, axis=0).tolist()

        mono_fixed = bool(np.all(np.diff(fixed) >= -1e-9))
        mono_shift = bool(np.all(np.diff(shifting) >= -1e-9))

        curves[cond] = {"shifting": shifting, "fixed": fixed, "random": random_mean}
        rows.append({
            "condition": cond, "n_frames": n,
            "aurc_shifting": aurc(shifting), "aurc_fixed": aurc(fixed), "aurc_random": aurc(random_mean),
            "monotone_shifting": mono_shift, "monotone_fixed": mono_fixed,
        })
        print(f"[b3] {cond:10s} AURC shifting={aurc(shifting):.4f} (monotone={mono_shift})  "
              f"fixed-GT={aurc(fixed):.4f} (monotone={mono_fixed})  random={aurc(random_mean):.4f}  "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- report ----------------------------------------------------------
    L = ["# B-3 -- fixed-GT risk-coverage (AURC) for `R_sys`", "",
         f"{len(ctx.gts)} paired frames, coverage grid {list(COVERAGE_GRID)}, "
         f"{N_RANDOM}-shuffle random-order control (fixed-GT denominator). "
         "Lower AURC is better: less risk (1 - mAP) for the same coverage.", "",
         "## 1. AURC summary", "",
         "| condition | frames | AURC (shifting denom, old) | monotone? | "
         "AURC (fixed-GT, new) | monotone? | AURC (random order) | R_sys beats random? |",
         "|---|---:|---:|---|---:|---|---:|---|"]
    for r in rows:
        beats = "YES" if r["aurc_fixed"] < r["aurc_random"] else "no"
        L.append(f"| {r['condition']} | {r['n_frames']} | {r['aurc_shifting']:.4f} | "
                 f"{'yes' if r['monotone_shifting'] else 'NO'} | {r['aurc_fixed']:.4f} | "
                 f"{'yes' if r['monotone_fixed'] else 'NO'} | {r['aurc_random']:.4f} | {beats} |")

    for cond in CONDITIONS:
        c = curves[cond]
        L += ["", f"## Risk vs coverage -- `{cond}`", "",
              "| coverage | risk (shifting denom) | risk (fixed-GT) | risk (random order, fixed-GT) |",
              "|---:|---:|---:|---:|"]
        for i, cov in enumerate(COVERAGE_GRID):
            L.append(f"| {cov:.0%} | {c['shifting'][i]:.4f} | {c['fixed'][i]:.4f} | {c['random'][i]:.4f} |")

    n_fixed_useful = sum(r["aurc_fixed"] < r["aurc_random"] for r in rows)
    n_monotone_fixed = sum(r["monotone_fixed"] for r in rows)
    n_monotone_shift = sum(r["monotone_shifting"] for r in rows)
    L += ["", "## 2. Reading", "",
          f"Fixed-GT risk is monotone (non-increasing as coverage rises) on "
          f"{n_monotone_fixed}/{len(rows)} conditions, vs {n_monotone_shift}/{len(rows)} "
          f"under the old shifting-denominator method. `R_sys` beats a random abstention "
          f"order (lower fixed-GT AURC) on {n_fixed_useful}/{len(rows)} conditions -- "
          + ("record abstain as a real, if modest, positive on those; the previous "
             "negative reading was measuring the denominator, not the signal."
             if n_fixed_useful > 0 else
             "R_sys does not select better frames than random even once the denominator "
             "is held fixed, so 'abstain as a negative' survives this correction.")]

    out_md = ROOT / "runs/eval/x_risk_coverage_fixed_gt.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps({
        "coverage_grid": COVERAGE_GRID.tolist(), "rows": rows, "curves": curves,
    }, indent=2), encoding="utf-8")
    print(f"\n[b3] wrote {out_md} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
