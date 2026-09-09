# AP convention parity — local vs official COCO

Produced by `scripts/ap_convention_parity.py` for R-A1 (`docs/TODO-2026-09-09-architecture-review.md`, finding F03). The local AP interpolates the precision envelope linearly; official COCO samples it at the first attained recall at or above each threshold. They are different metrics, and this measures the distance between them on real caches rather than on hand-built examples.

Config is pinned by `uqfusion.eval.cocoparity.TASK_CONFIG`: the same 0.50:0.05:0.95 IoU sweep, a single all-inclusive area range, no crowd or ignore regions, stable score sort, and **`maxDets` taken from the data rather than COCO's default of 100** — caches are built at `conf 0.001` and a 100-cap would truncate the low-confidence tail, turning a detection cap into what looks like an interpolation gap.

## Absolute AP — the convention is visible here

| arm | subset | local | COCO | local − COCO |
|---|---:|---:|---:|---:|
| VIS sigma-head | all | 0.30781575 | 0.30820109 | -0.00038535 |
| VIS sigma-head | day | 0.33842800 | 0.33880857 | -0.00038056 |
| VIS sigma-head | night | 0.24289897 | 0.24290502 | -0.00000606 |
| VIS MC-Dropout | all | 0.28425836 | 0.28441577 | -0.00015741 |
| VIS MC-Dropout | day | 0.31030716 | 0.31046374 | -0.00015658 |
| VIS MC-Dropout | night | 0.22539629 | 0.22540233 | -0.00000603 |
| VIS ensemble | all | 0.32015232 | 0.32059475 | -0.00044243 |
| VIS ensemble | day | 0.34635687 | 0.34678945 | -0.00043258 |
| VIS ensemble | night | 0.27222674 | 0.27223275 | -0.00000601 |
| IR sigma-head | all | 0.07059267 | 0.07059323 | -0.00000056 |
| IR sigma-head | day | 0.07067698 | 0.07067769 | -0.00000071 |
| IR sigma-head | night | 0.15378955 | 0.15379139 | -0.00000185 |
| IR MC-Dropout | all | 0.06328361 | 0.06328399 | -0.00000038 |
| IR MC-Dropout | day | 0.06908312 | 0.06908348 | -0.00000036 |
| IR MC-Dropout | night | 0.11769886 | 0.11770012 | -0.00000126 |
| IR ensemble | all | 0.07352604 | 0.07352627 | -0.00000024 |
| IR ensemble | day | 0.07439967 | 0.07440085 | -0.00000118 |
| IR ensemble | night | 0.15908489 | 0.15908574 | -0.00000086 |

The local metric reads **systematically low**, which is the direction the review predicted. It is not uniform: the gap is far larger on `day` than on `night`, consistently across arms. That pattern is reported as observed and is **not explained here**.

## Deltas — the convention largely cancels

| comparison | subset | local Δ | COCO Δ | disagreement |
|---|---:|---:|---:|---:|
| VIS sigma-head − VIS MC-Dropout | all | 0.02355739 | 0.02378532 | -0.00022793 |
| VIS sigma-head − VIS ensemble | all | -0.01233658 | -0.01239366 | 0.00005708 |
| VIS MC-Dropout − VIS ensemble | all | -0.03589396 | -0.03617898 | 0.00028501 |
| IR sigma-head − IR MC-Dropout | all | 0.00730906 | 0.00730924 | -0.00000018 |
| IR sigma-head − IR ensemble | all | -0.00293337 | -0.00293304 | -0.00000032 |
| IR MC-Dropout − IR ensemble | all | -0.01024242 | -0.01024228 | -0.00000014 |
| VIS sigma-head − VIS MC-Dropout | day | 0.02812084 | 0.02834483 | -0.00022399 |
| VIS sigma-head − VIS ensemble | day | -0.00792886 | -0.00798088 | 0.00005202 |
| VIS MC-Dropout − VIS ensemble | day | -0.03604971 | -0.03632571 | 0.00027601 |
| IR sigma-head − IR MC-Dropout | day | 0.00159386 | 0.00159421 | -0.00000035 |
| IR sigma-head − IR ensemble | day | -0.00372269 | -0.00372316 | 0.00000047 |
| IR MC-Dropout − IR ensemble | day | -0.00531655 | -0.00531737 | 0.00000082 |
| VIS sigma-head − VIS MC-Dropout | night | 0.01750267 | 0.01750270 | -0.00000002 |
| VIS sigma-head − VIS ensemble | night | -0.02932777 | -0.02932773 | -0.00000004 |
| VIS MC-Dropout − VIS ensemble | night | -0.04683044 | -0.04683042 | -0.00000002 |
| IR sigma-head − IR MC-Dropout | night | 0.03609068 | 0.03609127 | -0.00000059 |
| IR sigma-head − IR ensemble | night | -0.00529534 | -0.00529435 | -0.00000099 |
| IR MC-Dropout − IR ensemble | night | -0.04138602 | -0.04138562 | -0.00000040 |

Every comparison scores both arms under the same convention, so a systematic offset subtracts out. What survives is the *second-order* part, and that is what the last column measures.

## What this settles, and what it does not

**Worst delta disagreement across every pair and subset: 0.00028501.** The measured paired 2-sigma noise floor is 0.0014–0.0031 (`runs/eval/metric_noise_floor.md`), so the convention cannot flip a decision whose margin clears that floor — it is 5× smaller than the lower bound.

**It does not license quoting a local AP as a COCO AP.** They differ by more than the noise floor on the small hand-built cases (−0.0033 and −0.0050, reproduced exactly in `scripts/smoke_cocoparity.py` cases 2 and 3), so a published absolute number must name its convention.

**It does not cover margins below the noise floor.** This project has decided things on margins far smaller than the disagreement measured here — soft-NMS was rejected at −1.03e-5 (`runs/eval/snms_gate_draw_avg.md`). A convention disagreement of ~2e-4 is more than an order of magnitude larger than that. Such decisions were already unsound for the noise reason on record; this adds a second, independent reason and does not rescue any of them.

**Ultralytics is a third convention.** `docs/phase1-experimental-record.md` records a ~0.034 mAP difference between ultralytics versions on identical weights — two orders above everything here. Phase 1 ultralytics numbers still must not be compared directly against custom fusion AP.
