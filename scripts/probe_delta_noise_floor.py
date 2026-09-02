"""The noise floor for a *paired delta*, which is what a gate actually reads.

`runs/eval/metric_noise_floor.md` §3 reported `2 x sd(AP)` as a
"smallest-resolvable macro effect". **That is wrong, and this script exists to
replace it.** `sd(AP)` is the uncertainty in the metric's LEVEL — how well the
absolute 0.3286 is known from 1,200 frames. A gate never asks that. It compares
two systems **on the same frames**, so the resample is common to both arms and
almost all of the level uncertainty cancels. Using the unpaired sd as the bar
would demand ~0.012 of any arm and reject everything this project has ever
shipped, including the +0.0106 crossmodal gate.

So the floor is measured the way the comparison is actually made:

1. **Paired frame bootstrap** — resample frame indices ONCE per replicate and
   score both arms on that same resample, then take the sd of the difference.
   `bootstrap_delta` already does exactly this; it is called here rather than
   reimplemented, so the numbers are produced by the same code the gates use.
2. **Corruption draw** — sd of the delta across the 4 draws, which is the axis
   §4.5 opened and the one that killed the single-draw bar.

VIS soft-NMS at sigma 0.5 is the probe arm, purely because it is a real,
already-measured perturbation of the shipped system with a known small effect.
Nothing here re-opens its adoption: it was rejected in §4.6 and stays rejected.
Only the SPREAD of its delta is used.

The comparison that matters is `paired` vs `unpaired` in section 1. If pairing
buys an order of magnitude, then §3 of the previous file overstates the bar by
that factor and the corruption draw — not the frame count — is the binding
constraint, exactly as §4.5 concluded.

Day frames only, to match the per-class file (buoy has no night GT).

Usage:
    python scripts/probe_delta_noise_floor.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import fmt, md_table, sgn, write_md          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import (ap_weighted, bootstrap_delta,   # noqa: E402
                                     frame_parts, presort)
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402

CELLS = [("clean", None), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("clean", "fog_s2"),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None), ("fog", None),
         ("lowlight", "glare_s2"), ("blur_s3", "glare_s2")]

DRAWS = [("1/7 (shipped)", ROOT / "runs/cache_m"),
         ("901/911", ROOT / "runs/cache_m_draw901"),
         ("902/912", ROOT / "runs/cache_m_draw902"),
         ("903/913", ROOT / "runs/cache_m_draw903")]

SIGMA = 0.5
SHIP, BUOY = 0, 1


def cell_name(vc, ic) -> str:
    return f"{vc}/{ic or 'clean'}"


def contexts(cache_dir: Path):
    conds = sorted({c for c, _ in CELLS})
    ircs = sorted({i for _c, i in CELLS if i})

    def mk(ic, snms):
        return load_context(preset="crossmodal26m", cache_dir=str(cache_dir),
                            conditions=tuple(conds), ir_condition=ic,
                            vis_soft_nms=snms, verbose=False)
    base = {ic: mk(ic, None) for ic in [None] + ircs}
    arm = {ic: mk(ic, SIGMA) for ic in [None] + ircs}
    return base, arm


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/eval/delta_noise_floor.md")
    args = ap.parse_args()
    t0 = time.time()

    # ---- paired vs unpaired, on the shipped draw --------------------------
    base, arm = contexts(DRAWS[0][1])
    c0 = base[None]
    day = np.flatnonzero(~np.isin(c0.runs, NIGHT_RUNS))

    rows = []
    for vc, ic in CELLS:
        print(f"[paired] {cell_name(vc, ic)}", flush=True)
        pb = frame_parts(run_systems(base[ic], vc)["fused_gated"], base[ic].gts)
        pa = frame_parts(run_systems(arm[ic], vc)["fused_gated"], arm[ic].gts)

        # paired: bootstrap_delta resamples once and scores both arms on it
        bs = bootstrap_delta(pa, pb, sel=day, n_boot=args.n_boot, seed=0)
        paired_sd = (bs["ci_hi"] - bs["ci_lo"]) / (2 * 1.96)

        # unpaired: independent resamples of each arm, differenced
        pre_a, pre_b = presort(pa, sel=day), presort(pb, sel=day)
        n = pre_a["n_frames"]
        ra = np.random.default_rng(1)
        rb = np.random.default_rng(2)
        p = np.full(n, 1.0 / n)
        diffs = np.empty(args.n_boot)
        for t in range(args.n_boot):
            diffs[t] = (ap_weighted(pre_a, ra.multinomial(n, p))["map50_95"]
                        - ap_weighted(pre_b, rb.multinomial(n, p))["map50_95"])
        unpaired_sd = float(np.std(diffs, ddof=1))

        rows.append([cell_name(vc, ic), sgn(bs["delta"]), fmt(paired_sd),
                     fmt(unpaired_sd),
                     f"{unpaired_sd / paired_sd:.0f}x" if paired_sd > 0 else "--"])

    # ---- draw-to-draw sd of the same delta --------------------------------
    per_draw = {}
    for label, cd in DRAWS:
        if not cd.is_dir():
            continue
        print(f"[draw] {label}", flush=True)
        b, a = contexts(cd)
        d = np.flatnonzero(~np.isin(b[None].runs, NIGHT_RUNS))
        for vc, ic in CELLS:
            qb = frame_parts(run_systems(b[ic], vc)["fused_gated"], b[ic].gts)
            qa = frame_parts(run_systems(a[ic], vc)["fused_gated"], a[ic].gts)
            preb, prea = presort(qb, sel=d), presort(qa, sel=d)
            delta = (ap_weighted(prea)["map50_95"] - ap_weighted(preb)["map50_95"])
            per_draw.setdefault((vc, ic), []).append(delta)

    floor_rows = []
    for i, (vc, ic) in enumerate(CELLS):
        arr = np.asarray(per_draw[(vc, ic)])
        dsd = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")
        psd = float(rows[i][2])
        tot = float(np.hypot(dsd, psd))
        floor_rows.append([cell_name(vc, ic), fmt(dsd), fmt(psd), fmt(tot),
                           fmt(2 * tot),
                           "draw" if dsd > psd else "frames"])

    secs = [
        "Replaces §3 of `runs/eval/metric_noise_floor.md`, which reported "
        "`2 x sd(AP)` as a resolvable-effect bar. That figure is the uncertainty in "
        "the metric's **level**, not in a **paired delta**, and using it as a bar "
        "would reject every result this project has shipped.  \n"
        f"Probe arm: `vis_soft_nms` {SIGMA} — chosen only because it is a real, "
        "already-measured perturbation. Its adoption stays rejected (§4.6). Day "
        "frames only.",

        "## 1. Pairing is the whole ballgame\n\n"
        "`paired` resamples frames once and scores both arms on that resample — what "
        "`bootstrap_delta` does and what every gate in this project reads. "
        "`unpaired` resamples the two arms independently and differences them, which "
        "is what §3 of the previous file effectively assumed.\n\n"
        + md_table(["cell (vis/ir)", "delta", "sd paired", "sd unpaired",
                    "ratio"], rows),

        "## 2. The real noise floor for a gated delta\n\n"
        "Corruption draw and paired frame bootstrap, added in quadrature. "
        "`binding` names the larger of the two — the one worth spending on.\n\n"
        + md_table(["cell (vis/ir)", "sd draw", "sd paired boot", "total",
                    "2×total", "binding"], floor_rows),

        f"---\n\n_Generated by `scripts/probe_delta_noise_floor.py` in "
        f"{time.time() - t0:.1f}s._",
    ]
    write_md(args.out, "The noise floor for a paired delta", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
