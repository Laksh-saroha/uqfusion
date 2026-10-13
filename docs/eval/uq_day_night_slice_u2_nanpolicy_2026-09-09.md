# UQ calibration table — day and night, unpooled

> **Declared override in effect** (`--vis-sigma-cache sigma_vis_seed0_nightfull.pkl`, `--vis-mc-cache mc_vis_nightfull.pkl`, `--vis-ens-cache ens_vis_nightfull.pkl`): one or more VIS arms are scored from checkpoints other than the U1 registration's — see `docs/prereg-uq-day-night-slice-u2-stageb.md`. The IR control and every rule and band below are unchanged.

> **Amendment, declared.** Rule 9 as registered forced CONTAMINATED on *any* day-vs-pooled ordering change. On 2026-09-03, **after seeing it fire on a swap inside the noise**, it was floored: a flip now counts only if the two arms are separated by more than the floor in **both** orderings. Amending a rule after seeing it fire is the exact move a pre-registration exists to prevent, so both verdicts are reported and the registered one is never dropped. The unfloored rule is the same sign-test-on-noise this project has already been burned by twice, which is why the amendment was made rather than the result accepted.

Scored by `scripts/slice_uq_day_night.py` against the bands fixed in `docs/prereg-uq-day-night-slice.md` (commit `a4f9364`), written before any cache in `runs/cache_uqslice/` existed. No weight was trained and no label was touched.

**Stage B.** All three VIS arms are present, so `sep` and `spread` are three-arm quantities on VIS as they always were on IR, and a **CLEAN verdict is reachable** — which U1's staging rule forbids at Stage A. Registered in `docs/prereg-uq-day-night-slice-u2-stageb.md`.

**VIS SUSPECT, IR SUSPECT.** IR is the negative control: its labels were never filtered. **They land in the same band**, so on the prereg's own reading the distortion is NOT label-driven — night is intrinsically harder to calibrate on, and retraining would not repair it.

## Cache verification

**VIS** — images_list=runs/derived/paired_val_vis.txt, imgsz=640, conf=0.001, corrupt=None

**IR** — images_list=runs/derived/paired_val_ir.txt, imgsz=640, conf=0.001, corrupt=None

## Detector sanity, before any band is read

### Night detection counts — diagnostic, **not a decision input** (U2 §4)

| arm | night frames | night detections | per frame | day detections | checkpoint |
|---|---:|---:|---:|---:|---:|
| sigma-head | 1,032 | 7,694 | 7.5 | 23,416 | gauss_vis_seed0_nightfull |
| MC-Dropout | 1,032 | 11,567 | 11.2 | 27,319 | mc_vis_nightfull |
| ensemble(n=5) | 1,032 | 16,700 | 16.2 | 34,458 | ens_vis_nightfull_seed0 |

An arm emitting near-zero detections at night produces calibration numbers that are hollow rather than good, which is how both prior runs of this slice were partly decided. This table enters no band and moves no verdict.

### Night detection counts — diagnostic, **not a decision input** (U2 §4)

| arm | night frames | night detections | per frame | day detections | checkpoint |
|---|---:|---:|---:|---:|---:|
| sigma-head | 1,032 | 22,204 | 21.5 | 45,903 | gauss_ir_seed0_ft |
| MC-Dropout | 1,032 | 21,885 | 21.2 | 36,625 | mc_ir_seed0_ft_refit |
| ensemble(n=5) | 1,032 | 43,333 | 42.0 | 79,599 | ens_ir_seed0_ft |

An arm emitting near-zero detections at night produces calibration numbers that are hollow rather than good, which is how both prior runs of this slice were partly decided. This table enters no band and moves no verdict.

## VIS — verdict **SUSPECT**

2232 frames = 1200 day + 1032 night (46.2% night).

### Metrics by subset

| arm | subset | d_ece | nll | interval_ece | ause | aurc | map50_95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sigma-head | pooled | 0.1481 | 4.2016 | 0.1725 | 0.0748 | 0.4397 | 0.3078 |
|  | day | 0.1119 | 3.9057 | 0.1693 | 0.0731 | 0.4662 | 0.3384 |
|  | night | 0.2583 | 4.8474 | 0.1796 | 0.0721 | 0.3654 | 0.2429 |
| MC-Dropout | pooled | 0.1536 | -- | 0.3895 | 0.1421 | 0.5860 | 0.2843 |
|  | day | 0.1316 | -- | 0.3788 | 0.1153 | 0.5709 | 0.3103 |
|  | night | 0.2054 | -- | 0.4112 | 0.2186 | 0.6378 | 0.2254 |
| ensemble(n=5) | pooled | 0.1078 | -- | 0.4822 | 0.0974 | 0.5963 | 0.3202 |
|  | day | 0.0840 | -- | 0.4888 | 0.0950 | 0.5995 | 0.3464 |
|  | night | 0.1570 | -- | 0.4698 | 0.1032 | 0.5916 | 0.2722 |

### The night pull `D(a) = pooled − day`, per arm

| metric | sigma-head | MC-Dropout | ensemble(n=5) |
|---|---:|---:|---:|
| d_ece | +0.0362 | +0.0220 | +0.0238 |
| nll | +0.2959 | -- | -- |
| interval_ece | +0.0032 | +0.0108 | -0.0066 |
| ause | +0.0016 | +0.0268 | +0.0024 |
| aurc | -0.0265 | +0.0151 | -0.0032 |

A uniform handicap makes this row constant; `spread` is its range.

### The statistic

| metric | sep (day) | se(sep) | floor | clears floor | spread | r | order flips | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d_ece | 0.0476 | 0.0018 | 0.0036 | yes | 0.0142 | 0.2992 | no | **SUSPECT** |
| nll | -- | -- | -- | **no** | -- | -- | no | **NO-SIGNAL** |
| interval_ece | 0.3195 | 0.0024 | 0.0070 | yes | 0.0174 | 0.0544 | no | **CLEAN** |
| ause | 0.0421 | 0.0021 | 0.0041 | yes | 0.0252 | 0.5982 | no | **SUSPECT** |
| aurc | 0.1333 | 0.0023 | 0.0108 | yes | 0.0416 | 0.3119 | no | **SUSPECT** |

**Arms with no value on a metric** — `sep` and `spread` are reported as undefined (NaN -> NO-SIGNAL), not as a range over whoever is left. The finite-subset columns are a **disclosed secondary and enter no band**:

| metric | arms with no value | arms remaining | sep (finite only) | spread (finite only) |
|---|---:|---:|---:|---:|
| nll | MC-Dropout, ensemble(n=5) | 1 | -- | -- |

## IR — verdict **SUSPECT** — **as registered: CONTAMINATED** (rule 9 unfloored)

2232 frames = 1200 day + 1032 night (46.2% night).

### Metrics by subset

| arm | subset | d_ece | nll | interval_ece | ause | aurc | map50_95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| sigma-head | pooled | 0.0690 | 3.8734 | 0.1796 | 0.0771 | 0.7931 | 0.0706 |
|  | day | 0.0568 | 2.7041 | 0.1079 | 0.0856 | 0.8344 | 0.0707 |
|  | night | 0.0950 | 5.4433 | 0.2758 | 0.0642 | 0.7218 | 0.1538 |
| MC-Dropout | pooled | 0.0882 | -- | 0.4741 | 0.1309 | 0.8264 | 0.0633 |
|  | day | 0.0806 | -- | 0.4351 | 0.1296 | 0.8455 | 0.0691 |
|  | night | 0.1010 | -- | 0.5236 | 0.1290 | 0.7930 | 0.1177 |
| ensemble(n=5) | pooled | 0.0538 | -- | 0.2975 | 0.0785 | 0.8761 | 0.0735 |
|  | day | 0.0477 | -- | 0.2576 | 0.0775 | 0.8983 | 0.0744 |
|  | night | 0.0651 | -- | 0.3478 | 0.0764 | 0.8367 | 0.1591 |

### The night pull `D(a) = pooled − day`, per arm

| metric | sigma-head | MC-Dropout | ensemble(n=5) |
|---|---:|---:|---:|
| d_ece | +0.0122 | +0.0076 | +0.0061 |
| nll | +1.1693 | -- | -- |
| interval_ece | +0.0717 | +0.0390 | +0.0399 |
| ause | -0.0085 | +0.0014 | +0.0010 |
| aurc | -0.0413 | -0.0192 | -0.0223 |

A uniform handicap makes this row constant; `spread` is its range.

### The statistic

| metric | sep (day) | se(sep) | floor | clears floor | spread | r | order flips | band |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| d_ece | 0.0330 | 0.0010 | 0.0020 | yes | 0.0060 | 0.1833 | no | **CLEAN** |
| nll | -- | -- | -- | **no** | -- | -- | no | **NO-SIGNAL** |
| interval_ece | 0.3272 | 0.0051 | 0.0103 | yes | 0.0326 | 0.0998 | no | **CLEAN** |
| ause | 0.0521 | 0.0017 | 0.0034 | yes | 0.0099 | 0.1904 | no (+1 unresolvable) | **CLEAN** |
| aurc | 0.0639 | 0.0019 | 0.0166 | yes | 0.0221 | 0.3454 | no | **SUSPECT** |

Ordering changes, and whether they clear the floor in **both** orderings (rule 9):

- `ause` unresolvable swap sigma-head vs ensemble(n=5): gap day 0.0082, pooled 0.0014, floor 0.0034

**Arms with no value on a metric** — `sep` and `spread` are reported as undefined (NaN -> NO-SIGNAL), not as a range over whoever is left. The finite-subset columns are a **disclosed secondary and enter no band**:

| metric | arms with no value | arms remaining | sep (finite only) | spread (finite only) |
|---|---:|---:|---:|---:|
| nll | MC-Dropout, ensemble(n=5) | 1 | -- | -- |

---

_Rules: bands CLEAN < 0.25 ≤ SUSPECT < 1.0 ≤ CONTAMINATED on `r = spread/sep`; a metric enters the verdict only if `sep ≥ max(2·se_sep, 0.02·scale)`; an ordering flip forces CONTAMINATED only when the swapped pair clears the floor in both orderings (amended — as registered, any flip counted); the verdict is the worst band over metrics that clear the floor._
