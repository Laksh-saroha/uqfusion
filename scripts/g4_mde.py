"""G4 -- the minimum detectable effect for each Stage 2/3 comparison, before it is run.

Pre-registered at `docs/prereg-phase3-retrain-2026-09-10.md` §3 gate G4: "For each
comparison in §5 and §6, compute the minimum detectable effect at alpha = 0.05,
power = 0.80 ... **Any arm whose MDE exceeds the effect it is meant to resolve is either
given more seeds or cut from the design here, before it is run.**"

This is R-F3 applied forwards. `ir_equivalence_interval.py` does the same arithmetic
backwards -- it asks what a finished design could have detected. G4 asks it in advance,
where the answer can still change the design.

**The variance that matters here is between-SEED, not between-draw.** Two different noise
sources have been measured on this project and they are not interchangeable:

  * `runs/eval/metric_noise_floor.md` measures **evaluation** noise -- corruption draws and
    frame bootstrap over one fixed checkpoint. Its paired 2-sigma band, 0.0014-0.0031, is
    what §8's floor is built from.
  * Stage 2 compares **separately trained** arms. Each seed is a different optimisation
    trajectory, so its spread includes everything the evaluation noise includes *and* the
    training run's own variability. It is the larger number, and using the evaluation
    figure here would understate the MDE and pass arms that cannot resolve anything.

So sigma is estimated from the per-seed benchmark tables, where the same variant was
trained at several seeds and the within-variant spread is exactly the quantity wanted.
Those runs are stride-4 and a different architecture from Stage 2's, which is stated as an
assumption rather than hidden: it is the only multi-seed training-variance estimate this
project has, and G4 has to be answered before Stage 2 produces a better one.

Reports; gates nothing directly. The cut decision is the operator's, recorded per §9's
BUDGET-CUT outcome.
"""
from __future__ import annotations

import argparse
import csv
import math
import sys
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[1]

ALPHA = 0.05
POWER = 0.80

#: Per-seed tables. Within-variant spread across seeds is the between-seed sigma.
TABLES = {
    "vis": "phase1_benchmark/compiled/vis_benchmark_stride4_variants.csv",
    "ir": "phase1_benchmark/compiled/ir_benchmark_stride4_variants.csv",
}
SEED_COLS = ("seed0", "seed1", "seed2")

#: §8's four reporting floors, verdict taken at 0.0060.
FLOORS = (0.0014, 0.0031, 0.0060, 0.0100)
VERDICT_FLOOR = 0.0060

SEED_LADDER = (3, 5, 7, 10, 15)


def pooled_seed_sd(csv_path: Path) -> tuple[float, int, int]:
    """Pooled within-variant SD across seeds, plus (variants, total runs).

    Pooling rather than averaging the per-variant SDs: each variant contributes its own
    sum of squares and its own degrees of freedom, which is the right weighting when
    every variant has few seeds.
    """
    ss = 0.0
    df = 0
    n_var = n_run = 0
    with csv_path.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            vals = [float(row[c]) for c in SEED_COLS
                    if row.get(c) not in (None, "", "nan")]
            if len(vals) < 2:
                continue
            a = np.array(vals)
            ss += float(((a - a.mean()) ** 2).sum())
            df += len(a) - 1
            n_var += 1
            n_run += len(a)
    if df == 0:
        raise SystemExit(f"no multi-seed rows in {csv_path}")
    return math.sqrt(ss / df), n_var, n_run


def mde_two_arm(sd: float, n: int, alpha: float = ALPHA, power: float = POWER) -> float:
    """Smallest true difference two n-seed arms would detect at `power`.

    Two-sample t, equal n, two-sided. Solved on the noncentral t rather than the normal
    approximation, which is optimistic at the seed counts this project can afford.
    """
    df = 2 * n - 2
    tcrit = stats.t.ppf(1 - alpha / 2, df)
    se = sd * math.sqrt(2.0 / n)

    def attained(delta: float) -> float:
        nc = delta / se
        return (stats.nct.sf(tcrit, df, nc) + stats.nct.cdf(-tcrit, df, nc))

    lo, hi = 0.0, se * 10 or 1.0
    while attained(hi) < power:
        hi *= 2
        if hi > 1e6:
            return float("inf")
    for _ in range(200):
        mid = (lo + hi) / 2
        if attained(mid) < power:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="write a markdown report here too")
    args = ap.parse_args()

    lines: list[str] = []

    def emit(s: str = "") -> None:
        print(s)
        lines.append(s)

    emit("# G4 -- minimum detectable effect, computed before the arms are run")
    emit()
    emit(f"alpha = {ALPHA}, power = {POWER:.0%}, two-sided two-sample t, equal seeds per arm.")
    emit(f"Verdict floor (§8): **{VERDICT_FLOOR}**.")
    emit()

    verdicts = {}
    for arm, rel in TABLES.items():
        p = ROOT / rel
        if not p.is_file():
            emit(f"## {arm.upper()} -- MISSING table {rel}")
            emit()
            continue
        sd, n_var, n_run = pooled_seed_sd(p)
        emit(f"## {arm.upper()}")
        emit()
        emit(f"Between-seed sigma pooled from `{rel}`: **{sd:.5f}** "
             f"({n_var} variants, {n_run} runs, {n_run - n_var} df).")
        emit()
        emit("| seeds/arm | MDE | resolves 0.0060? |")
        emit("|---:|---:|---|")
        for n in SEED_LADDER:
            m = mde_two_arm(sd, n)
            ok = "yes" if m <= VERDICT_FLOOR else "**no**"
            emit(f"| {n} | {m:.5f} | {ok} |")
            if n == 5:
                verdicts[arm] = (sd, m)
        emit()
        emit("MDE against each of §8's four reporting floors, at 5 seeds:")
        emit()
        emit("| floor | MDE(5) | detectable |")
        emit("|---:|---:|---|")
        m5 = mde_two_arm(sd, 5)
        for f in FLOORS:
            emit(f"| {f} | {m5:.5f} | {'yes' if m5 <= f else '**no**'} |")
        emit()

    emit("## Which comparisons this governs, and which it does not")
    emit()
    emit("This MDE is for **two independently trained arms** -- different checkpoints, so")
    emit("each arm carries its own training-trajectory noise and the comparison is unpaired.")
    emit("That is the right model for §5's Stage 2 questions (does the retrained detector")
    emit("differ from what is deployed) and for any arm-vs-arm claim over fresh checkpoints.")
    emit()
    emit("**It is the wrong model for §6's mechanism arms and must not be used to cut them.**")
    emit("S0-S8 vary the fusion mechanism over the *same* checkpoints and the *same* frames:")
    emit("S1 real sigma against S3 shuffled sigma is one cache scored two ways. Seed variance")
    emit("is common to both sides and cancels, so their resolution is set by §8's moving-block")
    emit("bootstrap over frames, not by seed count -- which is why §8 specifies that interval")
    emit("and the 0.0060 floor rather than a power calculation. Applying the number above to")
    emit("them would cut arms that are in fact well resolved.")
    emit()
    emit("A paired between-seed sigma for the mechanism arms is not estimable from these")
    emit("tables, which hold one number per run and no per-frame scores. It does not need to")
    emit("be: the block bootstrap measures it directly when §6 runs.")
    emit()
    emit("## Verdict")
    emit()
    for arm, (sd, m5) in verdicts.items():
        if m5 <= VERDICT_FLOOR:
            emit(f"* **{arm.upper()}: 5 seeds resolve the floor** "
                 f"(MDE {m5:.5f} <= {VERDICT_FLOOR}).")
        else:
            need = next((n for n in range(5, 61)
                         if mde_two_arm(sd, n) <= VERDICT_FLOOR), None)
            emit(f"* **{arm.upper()}: 5 seeds do NOT resolve the floor** "
                 f"(MDE {m5:.5f} > {VERDICT_FLOOR}). "
                 + (f"Would need **{need} seeds/arm**." if need
                    else "No affordable seed count resolves it."))
    emit()
    emit("Per G4, an arm whose MDE exceeds the effect it exists to resolve is given more "
         "seeds or cut here, before it runs -- §9's BUDGET-CUT.")
    emit()
    emit("**Caveats on sigma, both of which cut against over-reading the seed counts above.**")
    emit("The VIS estimate rests on 2 variants and 4 degrees of freedom; it is the noisiest")
    emit("input in this calculation and its implied seed count should be read as an order of")
    emit("magnitude, not a target. Both estimates come from stride-4 runs on other")
    emit("architectures, at a different frame count from Stage 2's, and between-seed spread")
    emit("is not guaranteed to transfer across either. They are nevertheless the only")
    emit("multi-seed training-variance measurements this project holds, and G4 has to be")
    emit("answered before Stage 2 produces a better one -- so the honest reading is that")
    emit("**5 seeds are very unlikely to resolve 0.0060 for a trained-arm comparison**, not")
    emit("that 19 is the precise number.")

    if args.out:
        o = ROOT / args.out
        o.parent.mkdir(parents=True, exist_ok=True)
        o.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(f"\nwrote {o}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
