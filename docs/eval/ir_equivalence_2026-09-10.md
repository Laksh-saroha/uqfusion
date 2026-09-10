# IR architecture ladder - equivalence interval, not a null

Regenerate: `python scripts/ir_equivalence_interval.py --out <new path>`.

```
source: phase1_benchmark/compiled/ir_benchmark_stride4_variants.csv
complete variants (n=3 seeds): 13    within-variant df: 26
excluded (incomplete seed set): yolov9c(n=1), yolov10s(n=1), yolo26n(n=2)

one-way ANOVA        F(12, 26) = 1.037   p = 0.447
pooled within sd     0.00508
variant means        0.06124 .. 0.07316
largest gap          yolo26l - yolov10n = 0.01193

The question the ANOVA does NOT answer: how big a difference could this
design have found?
  minimum detectable spread, alpha=0.05, power=80%: 0.02067
  observed spread is 0.58x that.

Tukey HSD 95% simultaneous CI on the largest gap:
  [-0.00314, +0.02700]   (half-width 0.01507, q = 5.139)
Same pair without a multiplicity correction:
  [+0.00340, +0.02045]   (half-width 0.00852)

   delta  equivalence supported?    basis
  0.0014  NO  - CI exceeds delta    upper bound 0.02700 vs 0.0014
  0.0031  NO  - CI exceeds delta    upper bound 0.02700 vs 0.0031
  0.0050  NO  - CI exceeds delta    upper bound 0.02700 vs 0.0050
  0.0100  NO  - CI exceeds delta    upper bound 0.02700 vs 0.0100
  0.0150  NO  - CI exceeds delta    upper bound 0.02700 vs 0.0150
  0.0200  NO  - CI exceeds delta    upper bound 0.02700 vs 0.0200

Read: equivalence is supported only for deltas ABOVE the CI upper bound.
At every delta at or below 0.0270 the data cannot rule out a real
architecture effect. It never showed there was none.
```

## Provenance

Written automatically (R-E1/F14). A result that records a preset *name* cannot say which system produced it -- `preset="crossmodal"` named three different systems on 2026-09-01. These are values.

| field | value |
|---|---|
| `written_utc` | `2026-09-10T08:23:45+00:00` |
| `git_head` | `11a10e572d3138929aa7a2189bd8861ae8fd9afc` |
| `git_dirty` | `False` |
| `dirty_sha256` | -- |
| `git_branch` | `fusion-uq-phase3` |
| `ap_convention` | `local-linear-interp` |
| `missing_class_policy` | `drop` |
| `sort_kind` | `stable` |
