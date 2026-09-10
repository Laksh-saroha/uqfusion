"""Replace the IR ladder's "statistically indistinguishable" with an interval.

R-F3 / F15. `docs/ir-benchmark-closed-2026-09-01.md` closed the IR architecture
queue at 44/93 runs on the strength of a one-way ANOVA, F(12, 26) = 1.037,
p = 0.45, and called thirteen architectures "statistically indistinguishable".

**A non-significant test is not an equivalence test.** p = 0.45 says the design
failed to detect a difference; it says nothing about how large a difference the
design *could* have detected. The review asks for the honest version: declare a
tolerable delta and estimate an interval for it.

This script does three things and asserts no verdict the data cannot carry:

  1. reproduces the published ANOVA from the compiled per-seed table, so the
     number being criticised is the number in the file;
  2. computes the detectable effect the design actually had (the minimum
     between-variant spread this n would reject the null for, at 80% power);
  3. reports a Tukey-HSD *simultaneous* interval on the largest observed
     pairwise gap, and prints, for a ladder of candidate deltas, whether that
     interval falls inside it -- i.e. which equivalence claims the data
     supports and which it does not.

The delta ladder is printed for every value, not picked after the fact: the
point is to show the reader where the boundary sits, not to find the one
threshold that yields a satisfying verdict.

    python scripts/ir_equivalence_interval.py --out docs/eval/ir_equivalence_<date>.md

Reports; gates nothing.
"""

from __future__ import annotations

import argparse
import csv
import io
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

DEFAULT_CSV = "phase1_benchmark/compiled/ir_benchmark_stride4_variants.csv"
SEED_COLS = ("seed0", "seed1", "seed2")

# Candidate tolerable deltas, in absolute mAP50-95. Printed as a ladder so the
# boundary is visible rather than chosen. The first two are the project's paired
# 2-sigma noise floor (runs/eval/metric_noise_floor.md); the rest bracket the
# observed IR scale, where every variant mean sits in 0.061-0.073.
DELTAS = (0.0014, 0.0031, 0.0050, 0.0100, 0.0150, 0.0200)

ALPHA = 0.05
POWER = 0.80


def load(csv_path: Path) -> tuple[list[str], list[np.ndarray]]:
    names: list[str] = []
    groups: list[np.ndarray] = []
    with io.open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            vals = [row[c].strip() for c in SEED_COLS]
            if any(v == "" for v in vals):
                continue  # incomplete variant: excluded, and counted below
            names.append(row["variant"])
            groups.append(np.array([float(v) for v in vals], dtype=float))
    return names, groups


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--out", default=None, help="write a markdown report here (NEW file)")
    args = ap.parse_args(argv)

    csv_path = ROOT / args.csv
    names, groups = load(csv_path)
    k = len(groups)
    n = groups[0].size
    df_within = sum(g.size for g in groups) - k
    if k < 2:
        print("[refuse] fewer than two complete variants", file=sys.stderr)
        return 2

    means = np.array([g.mean() for g in groups])
    # ddof=1 per group; the pooled within-variant variance is the ANOVA error term.
    ss_within = float(sum(((g - g.mean()) ** 2).sum() for g in groups))
    ms_within = ss_within / df_within
    s_pooled = float(np.sqrt(ms_within))

    F, p = stats.f_oneway(*groups)
    order = np.argsort(means)
    lo_i, hi_i = int(order[0]), int(order[-1])
    gap = float(means[hi_i] - means[lo_i])

    # Tukey HSD simultaneous CI on the largest observed pairwise difference.
    q = stats.studentized_range.ppf(1 - ALPHA, k, df_within)
    hsd_half = float(q * np.sqrt(ms_within / n))
    hsd_lo, hsd_hi = gap - hsd_half, gap + hsd_half

    # Plain two-sample interval for the same pair, ignoring multiplicity, so the
    # cost of looking at all pairs is visible rather than implicit.
    t_crit = stats.t.ppf(1 - ALPHA / 2, df_within)
    pair_half = float(t_crit * np.sqrt(2 * ms_within / n))

    # What spread this design could have detected: solve for the between-variant
    # effect size f at which the F test reaches POWER, then express it as the
    # equivalent max-min spread of a two-point mean configuration.
    f_crit = stats.f.ppf(1 - ALPHA, k - 1, df_within)
    lam_lo, lam_hi = 0.0, 400.0
    for _ in range(200):
        lam = 0.5 * (lam_lo + lam_hi)
        if stats.ncf.sf(f_crit, k - 1, df_within, lam) < POWER:
            lam_lo = lam
        else:
            lam_hi = lam
    lam = 0.5 * (lam_lo + lam_hi)
    # lambda = n * sum((mu_i - mubar)^2) / sigma^2; for the least-favourable
    # configuration (one group high, one low, rest at the mean) sum of squares
    # is spread^2 / 2, so spread = sqrt(2 * lambda / n) * sigma.
    mde_spread = float(np.sqrt(2.0 * lam / n) * s_pooled)

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit(f"source: {args.csv}")
    emit(f"complete variants (n={n} seeds): {k}    within-variant df: {df_within}")
    emit(f"excluded (incomplete seed set): {', '.join(_incomplete(csv_path)) or 'none'}")
    emit()
    emit(f"one-way ANOVA        F({k-1}, {df_within}) = {F:.3f}   p = {p:.3f}")
    emit(f"pooled within sd     {s_pooled:.5f}")
    emit(f"variant means        {means.min():.5f} .. {means.max():.5f}")
    emit(f"largest gap          {names[hi_i]} - {names[lo_i]} = {gap:.5f}")
    emit()
    emit("The question the ANOVA does NOT answer: how big a difference could this")
    emit("design have found?")
    emit(f"  minimum detectable spread, alpha={ALPHA}, power={POWER:.0%}: {mde_spread:.5f}")
    emit(f"  observed spread is {gap / mde_spread:.2f}x that.")
    emit()
    emit(f"Tukey HSD {1-ALPHA:.0%} simultaneous CI on the largest gap:")
    emit(f"  [{hsd_lo:+.5f}, {hsd_hi:+.5f}]   (half-width {hsd_half:.5f}, q = {q:.3f})")
    emit(f"Same pair without a multiplicity correction:")
    emit(f"  [{gap - pair_half:+.5f}, {gap + pair_half:+.5f}]   (half-width {pair_half:.5f})")
    emit()
    emit(f"{'delta':>8}  {'equivalence supported?':<24}  basis")
    for d in DELTAS:
        ok = hsd_hi < d
        verdict = "YES - CI inside +/-delta" if ok else "NO  - CI exceeds delta"
        emit(f"{d:>8.4f}  {verdict:<24}  upper bound {hsd_hi:.5f} vs {d:.4f}")
    emit()
    emit("Read: equivalence is supported only for deltas ABOVE the CI upper bound.")
    emit(f"At every delta at or below {hsd_hi:.4f} the data cannot rule out a real")
    emit("architecture effect. It never showed there was none.")

    if args.out:
        from _ideas_common import write_md  # noqa: E402

        body = "\n".join(
            [
                "```",
                *lines,
                "```",
            ]
        )
        write_md(
            args.out,
            "IR architecture ladder - equivalence interval, not a null",
            [
                "Regenerate: `python scripts/ir_equivalence_interval.py --out <new path>`.",
                body,
            ],
        )
    return 0


def _incomplete(csv_path: Path) -> list[str]:
    out: list[str] = []
    with io.open(csv_path, encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if any(row[c].strip() == "" for c in SEED_COLS):
                out.append(f"{row['variant']}(n={row['n_seeds']})")
    return out


if __name__ == "__main__":
    raise SystemExit(main())
