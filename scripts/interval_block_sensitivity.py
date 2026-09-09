"""R-A3: how much too narrow are this project's intervals?

Every headline interval here resamples individual frames from 10 Hz recordings.
Consecutive frames are nearly the same picture, so the iid frame bootstrap treats
~1,032 highly dependent observations as ~1,032 independent ones and the resulting
standard error is too small. This measures by how much, by re-running the same paired
bootstrap with a contiguous **block** as the resampling unit and sweeping the block
length.

The number that matters is the ratio ``se(L) / se(L=1)``. If it plateaus at 2, every
published interval in this project is about half as wide as it should be, and any
margin defended at "just outside the CI" needs re-reading.

Usage:
    python scripts/interval_block_sensitivity.py --out runs/eval/interval_block_sensitivity.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import ROOT, fmt, md_table, write_md  # noqa: E402

from uqfusion.eval.apmetrics import bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.blockboot import block_bootstrap_delta, run_ids, run_slices  # noqa: E402
from uqfusion.eval.cache import load_cache  # noqa: E402
from uqfusion.eval.matching import load_gt  # noqa: E402

# 10 Hz recordings: L=10 is one second of video, L=100 is ten seconds.
BLOCKS = (1, 2, 5, 10, 20, 50, 100, 200)

ARMS = {
    "sigma": "sigma_vis_seed0_nightfull.pkl",
    "mc": "mc_vis_nightfull.pkl",
    "ens": "ens_vis_nightfull.pkl",
}


def load(fname: str):
    obj = load_cache(ROOT / "runs/cache_uqslice" / fname)
    recs = obj[0] if isinstance(obj, tuple) else obj["records"]
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in recs]
    return recs, gts, [r["image_path"] for r in recs]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/interval_block_sensitivity.md")
    ap.add_argument("--boot", type=int, default=1000)
    args = ap.parse_args()

    parts, paths = {}, None
    for k, f in ARMS.items():
        recs, gts, p = load(f)
        parts[k] = frame_parts(recs, gts)
        paths = p
    per_run = run_slices(run_ids(paths))
    # A moving-block bootstrap is only valid while the block is SHORT relative to the
    # series. The shortest run here decides that, and it decides it for every run,
    # because a run shorter than L admits almost one block start: the resample becomes
    # near-deterministic and the variance COLLAPSES. Blocks past this bound are still
    # computed and shown, marked invalid, because seeing the ratio fall again is the
    # evidence for the bound.
    min_run = min(len(v) for v in per_run.values())
    valid_max = max(1, min_run // 5)

    rows, ratios, prov = [], {}, None
    t0 = time.time()
    for a, b in itertools.combinations(ARMS, 2):
        base_se = None
        for L in BLOCKS:
            r = block_bootstrap_delta(parts[a], parts[b], paths, L,
                                      n_boot=args.boot, seed=0)
            if prov is None:
                prov = {k: r[k] for k in
                        ("resampling_unit", "covers", "does_not_cover", "runs", "n_runs")}
            if base_se is None:
                base_se = r["se"]
            ratio = r["se"] / base_se if base_se else float("nan")
            ratios[(f"{a}-{b}", L)] = ratio
            rows.append([f"{a} − {b}", str(L), fmt(r["delta"], 6), fmt(r["se"], 6),
                         f"[{fmt(r['ci_lo'], 6)}, {fmt(r['ci_hi'], 6)}]",
                         fmt(r["ci_hi"] - r["ci_lo"], 6), f"{ratio:.2f}×",
                         "yes" if r["spans_zero"] else "**no**",
                         "yes" if L <= valid_max else "**no — variance collapse**"])
            print(f"[{a}-{b} L={L:3d}] se {r['se']:.6f}  ratio {ratio:.2f}x  "
                  f"({time.time() - t0:.0f}s)")

    # the existing global frame bootstrap, for reference: it also lets run composition
    # vary, which L=1 here deliberately does not.
    glob = []
    for a, b in itertools.combinations(ARMS, 2):
        g = bootstrap_delta(parts[a], parts[b], n_boot=args.boot, seed=0)
        glob.append([f"{a} − {b}", fmt(g["delta"], 6), fmt(g["se"], 6),
                     f"[{fmt(g['ci_lo'], 6)}, {fmt(g['ci_hi'], 6)}]"])

    # A moving-block bootstrap is only valid while the block is SHORT relative to the
    # series. `pohang03` holds 117 frames, so at L=200 every run shorter than L admits
    # exactly one block start, the resample becomes near-deterministic, and the variance
    # COLLAPSES -- which shows up as the ratio falling again rather than plateauing.
    # That is an artefact of the estimator, not a property of the data, so the headline
    # is taken from the largest block that is still valid.
    valid = [L for L in BLOCKS if L <= valid_max]
    pairs = sorted({k[0] for k in ratios})
    best_L = valid[-1]
    med = float(np.median([ratios[(p, best_L)] for p in pairs]))
    worst = max(ratios[(p, best_L)] for p in pairs)
    invalid = [L for L in BLOCKS if L > valid_max]
    # Where the curve actually turns over, so the write-up names the real block rather
    # than assuming the collapse begins at the first invalid L (it does not -- L=50 and
    # L=100 are still rising here, they are simply no longer trustworthy).
    med_by_L = {L: float(np.median([ratios[(p, L)] for p in pairs])) for L in BLOCKS}
    peak_L = max(med_by_L, key=med_by_L.get)
    peak_med = med_by_L[peak_L]
    tail_med = med_by_L[BLOCKS[-1]]

    secs = [
        "Produced by `scripts/interval_block_sensitivity.py` for R-A3 "
        "(`docs/TODO-2026-09-09-architecture-review.md`, finding F04). The resampling "
        "unit changes from a single frame to a contiguous block of frames drawn within "
        "a run, because consecutive frames of a 10 Hz recording are not independent "
        "observations and an iid frame bootstrap therefore reports an interval that is "
        "too narrow.",

        f"**Headline: this project's intervals are roughly {med:.1f}× too narrow, and "
        f"that is a lower bound.** At a one-second block (L=10) the standard error is "
        f"already {np.median([ratios[(p, 10)] for p in pairs]):.2f}× the iid value; at "
        f"the largest defensible block (L={best_L}, {best_L / 10:.0f}s of video) it is "
        f"{med:.2f}× across pairs, worst {worst:.2f}×. The curve is still **rising** "
        "there, so the true inflation is larger — this data cannot say how much larger, "
        "for the reason below.",

        f"**Why the sweep stops at L={best_L}, and why the larger rows must not be "
        f"quoted.** A moving-block bootstrap needs a block short relative to the series. "
        f"The shortest run here is **{min_run} frames** (`pohang03`): at L={valid_max * 2} "
        f"it offers {min_run - valid_max * 2 + 1} block starts, and at L={BLOCKS[-1]} it "
        f"offers **one**, so the resample becomes near-deterministic and the variance "
        f"**collapses**. The median ratio does keep rising to L={peak_L} "
        f"({peak_med:.2f}×) before falling to {tail_med:.2f}× at L={BLOCKS[-1]} — but "
        f"everything past L={valid_max} (shortest run / 5) is already unreliable in "
        f"*either* direction, so the rise there is no more quotable than the fall. The "
        f"collapse is shown because it is the evidence for the bound, not because it is "
        f"a result. Honest summary: the inflation is **at least {med:.1f}×**, and this "
        f"dataset's short runs prevent measuring where it levels off.",

        "## Sensitivity to block length\n\n"
        + md_table(["comparison", "block L", "delta", "se", "95% CI", "CI width",
                    "se vs L=1", "spans zero", "valid"], rows)
        + "\n\n`L=1` is an iid frame bootstrap **within run**, so it isolates block "
        "length as the only thing changing. 10 Hz means L=10 is one second of video "
        f"and L=100 is ten seconds. The **valid** column marks L <= {valid_max} "
        f"(shortest run {min_run} / 5); beyond that the variance collapses and the "
        "ratio is an artefact.",

        "## The existing global frame bootstrap, for reference\n\n"
        + md_table(["comparison", "delta", "se", "95% CI"], glob)
        + "\n\nThis is what `apmetrics.bootstrap_delta` produces today. It differs from "
        "`L=1` above by also letting run composition vary, so the two are not identical "
        "baselines.",

        "## What these intervals cover, and what they cannot\n\n"
        f"* **Resampling unit:** {prov['resampling_unit']}.\n"
        f"* **Covers:** {prov['covers']}.\n"
        + "".join(f"* **Does not cover:** {d}.\n" for d in prov["does_not_cover"])
        + f"\nRuns present: `{prov['runs']}` ({prov['n_runs']} runs).\n\n"
        "**Night has exactly one run.** No block length, and no number of draws, makes "
        "a between-night-run interval estimable from this data. Every night interval "
        "this project reports is conditional on `pohang01` and must be stated that way.",

        "## On sign-flip fractions\n\n"
        "`blockboot` reports `sign_flip_fraction` and keeps `p_sign_flip` only as a "
        "deprecated alias. It is **not a p-value and not the probability that a "
        "hypothesis is true** — it is a descriptive property of the resampling "
        "distribution under this scheme, and the old name invited exactly the reading "
        "R-A3 says to stop making.",
    ]
    out = write_md(args.out, "Interval sensitivity to block length", secs)
    Path(str(out).replace(".md", ".json")).write_text(json.dumps(
        {"ratios": {f"{k[0]}|L{k[1]}": v for k, v in ratios.items()},
         "blocks": list(BLOCKS), "n_boot": args.boot,
         "provenance": prov}, indent=1, default=str), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
