# Change-impact table — adopted decisions under the corrected metrics

Produced by `scripts/change_impact_table.py` for R-A5 (`docs/TODO-2026-09-09-architecture-review.md`). Two measurements from earlier in workstream A are applied to every recorded decision that was defended by an interval excluding zero.

**The two corrections.** R-A1: the AP-convention gap survives into a delta at most **0.00029** (`runs/eval/ap_convention_parity.md`), applied here as a systematic band on the point estimate. R-A3: the iid frame bootstrap on 10 Hz video is **1.95× too narrow** as a lower bound (`runs/eval/interval_block_sensitivity_v3.md`), applied by widening each recorded interval about its own delta. The second dominates.

## Result: 16 of 53 significant findings become indeterminate

| source | family | item | delta | recorded 95% CI | corrected band |
|---|---:|---:|---:|---:|---:|
| final_system_crossmodal | gated vs visible_only | clean / ship | 0.003399 | [0.001738, 0.005624] | [-0.000125, 0.008023] |
| final_system_crossmodal | D27 ablation: with_maha | clean/day / macro | -0.000151 | [-0.000298, -0.000001] | [-0.000722, 0.000427] |
| final_system_crossmodal | D27 ablation: with_maha | clean/day / ship | -0.000272 | [-0.000350, -0.000167] | [-0.000709, 0.000218] |
| final_system_crossmodal | D27 ablation: no_veto | clean/night / macro | -0.002245 | [-0.002934, -0.001168] | [-0.003874, 0.000140] |
| final_system_crossmodal | D27 ablation: no_veto | clean/night / ship | -0.002245 | [-0.002934, -0.001168] | [-0.003874, 0.000140] |
| final_system_crossmodal | D27 ablation: no_veto | fog/day / macro | -0.002091 | [-0.002736, -0.000813] | [-0.003633, 0.000687] |
| final_system_crossmodal | D27 ablation: no_veto | fog/day / ship | -0.004588 | [-0.005540, -0.002373] | [-0.006730, 0.000017] |
| final_system_crossmodal | D27 ablation: no_veto | lowlight/night / macro | -0.002200 | [-0.002892, -0.001123] | [-0.003835, 0.000185] |
| final_system_crossmodal | D27 ablation: no_veto | lowlight/night / ship | -0.002200 | [-0.002892, -0.001123] | [-0.003835, 0.000185] |
| final_system | gated vs visible_only | clean / ship | 0.003126 | [0.001447, 0.005391] | [-0.000432, 0.007829] |
| final_system | gated vs visible_only | glare / macro | 0.002235 | [0.000381, 0.004236] | [-0.001665, 0.006422] |
| final_system | gated vs visible_only | glare / ship | 0.003585 | [0.001528, 0.005183] | [-0.000711, 0.006987] |
| final_system | D27 ablation: no_maha | clean/day / macro | 0.000193 | [0.000070, 0.000317] | [-0.000332, 0.000719] |
| final_system | D27 ablation: no_maha | clean/day / ship | 0.000324 | [0.000244, 0.000401] | [-0.000117, 0.000760] |
| final_system | D27 ablation: cap_only | clean/day / macro | 0.000152 | [0.000000, 0.000298] | [-0.000428, 0.000722] |
| final_system | D27 ablation: cap_only | clean/day / ship | 0.000273 | [0.000168, 0.000350] | [-0.000218, 0.000708] |

These are not refuted. They are **no longer supported by the interval that was used to defend them**, which is a different and weaker statement: the effect may well be real, but this data and this resampling scheme cannot establish it at 95%.

## Per-claim roll-up — does the claim survive, or only some of its cells?

| claim | cells defended by an interval | still supported | now indeterminate | reading |
|---|---:|---:|---:|---:|
| D27 ablation: cap_only | 6 | 4 | 2 | holds on the larger cells (smallest surviving |delta| 0.002288) |
| D27 ablation: no_maha | 6 | 4 | 2 | holds on the larger cells (smallest surviving |delta| 0.002304) |
| D27 ablation: no_veto | 18 | 12 | 6 | holds on the larger cells (smallest surviving |delta| 0.003702) |
| D27 ablation: photometric_veto | 2 | 2 | 0 | all hold |
| D27 ablation: with_maha | 6 | 4 | 2 | holds on the larger cells (smallest surviving |delta| 0.002288) |
| gated vs visible_only | 14 | 10 | 4 | holds on the larger cells (smallest surviving |delta| 0.002120) |
| veil veto repair | 1 | 1 | 0 | all hold |

A claim can lose cells and still stand: what breaks first is always the smallest effect, and several of these families were defended across a range of magnitudes. Read this table before the row-level one.

## Every decision that was defended by a non-zero-spanning interval

| source | family | item | delta | recorded 95% CI | corrected band | verdict |
|---|---:|---:|---:|---:|---:|---:|
| final_system | D27 ablation: cap_only | clean/day / macro | 0.000152 | [0.000000, 0.000298] | [-0.000428, 0.000722] | **INDETERMINATE** |
| final_system | D27 ablation: cap_only | clean/day / ship | 0.000273 | [0.000168, 0.000350] | [-0.000218, 0.000708] | **INDETERMINATE** |
| final_system | D27 ablation: no_maha | clean/day / macro | 0.000193 | [0.000070, 0.000317] | [-0.000332, 0.000719] | **INDETERMINATE** |
| final_system | D27 ablation: no_maha | clean/day / ship | 0.000324 | [0.000244, 0.000401] | [-0.000117, 0.000760] | **INDETERMINATE** |
| final_system_crossmodal | D27 ablation: no_veto | clean/night / macro | -0.002245 | [-0.002934, -0.001168] | [-0.003874, 0.000140] | **INDETERMINATE** |
| final_system_crossmodal | D27 ablation: no_veto | clean/night / ship | -0.002245 | [-0.002934, -0.001168] | [-0.003874, 0.000140] | **INDETERMINATE** |
| final_system_crossmodal | D27 ablation: no_veto | fog/day / macro | -0.002091 | [-0.002736, -0.000813] | [-0.003633, 0.000687] | **INDETERMINATE** |
| final_system_crossmodal | D27 ablation: no_veto | fog/day / ship | -0.004588 | [-0.005540, -0.002373] | [-0.006730, 0.000017] | **INDETERMINATE** |
| final_system_crossmodal | D27 ablation: no_veto | lowlight/night / macro | -0.002200 | [-0.002892, -0.001123] | [-0.003835, 0.000185] | **INDETERMINATE** |
| final_system_crossmodal | D27 ablation: no_veto | lowlight/night / ship | -0.002200 | [-0.002892, -0.001123] | [-0.003835, 0.000185] | **INDETERMINATE** |
| final_system_crossmodal | D27 ablation: with_maha | clean/day / macro | -0.000151 | [-0.000298, -0.000001] | [-0.000722, 0.000427] | **INDETERMINATE** |
| final_system_crossmodal | D27 ablation: with_maha | clean/day / ship | -0.000272 | [-0.000350, -0.000167] | [-0.000709, 0.000218] | **INDETERMINATE** |
| final_system_crossmodal | gated vs visible_only | clean / ship | 0.003399 | [0.001738, 0.005624] | [-0.000125, 0.008023] | **INDETERMINATE** |
| final_system | gated vs visible_only | clean / ship | 0.003126 | [0.001447, 0.005391] | [-0.000432, 0.007829] | **INDETERMINATE** |
| final_system | gated vs visible_only | glare / macro | 0.002235 | [0.000381, 0.004236] | [-0.001665, 0.006422] | **INDETERMINATE** |
| final_system | gated vs visible_only | glare / ship | 0.003585 | [0.001528, 0.005183] | [-0.000711, 0.006987] | **INDETERMINATE** |
| final_system | D27 ablation: cap_only | fog/night / macro | -0.002826 | [-0.003641, -0.002081] | [-0.004699, -0.001089] | SURVIVES |
| final_system | D27 ablation: cap_only | fog/night / ship | -0.002826 | [-0.003641, -0.002081] | [-0.004699, -0.001089] | SURVIVES |
| final_system | D27 ablation: cap_only | glare/day / macro | 0.002288 | [0.001398, 0.002867] | [0.000267, 0.003702] | SURVIVES |
| final_system | D27 ablation: cap_only | glare/day / ship | 0.003248 | [0.002372, 0.004077] | [0.001254, 0.005149] | SURVIVES |
| final_system | D27 ablation: no_maha | fog/night / macro | -0.002824 | [-0.003636, -0.002079] | [-0.004693, -0.001086] | SURVIVES |
| final_system | D27 ablation: no_maha | fog/night / ship | -0.002824 | [-0.003636, -0.002079] | [-0.004693, -0.001086] | SURVIVES |
| final_system | D27 ablation: no_maha | glare/day / macro | 0.002304 | [0.001425, 0.002877] | [0.000305, 0.003707] | SURVIVES |
| final_system | D27 ablation: no_maha | glare/day / ship | 0.003265 | [0.002409, 0.004093] | [0.001312, 0.005165] | SURVIVES |
| final_system | D27 ablation: no_veto | clean/night / macro | -0.005078 | [-0.006006, -0.003607] | [-0.007173, -0.001925] | SURVIVES |
| final_system | D27 ablation: no_veto | clean/night / ship | -0.005078 | [-0.006006, -0.003607] | [-0.007173, -0.001925] | SURVIVES |
| final_system_crossmodal | D27 ablation: no_veto | fog/night / macro | -0.004831 | [-0.005795, -0.003803] | [-0.006996, -0.002542] | SURVIVES |
| final_system | D27 ablation: no_veto | fog/night / macro | -0.003702 | [-0.004615, -0.002221] | [-0.005768, -0.000530] | SURVIVES |
| final_system_crossmodal | D27 ablation: no_veto | fog/night / ship | -0.004831 | [-0.005795, -0.003803] | [-0.006996, -0.002542] | SURVIVES |
| final_system | D27 ablation: no_veto | fog/night / ship | -0.003702 | [-0.004615, -0.002221] | [-0.005768, -0.000530] | SURVIVES |
| final_system_crossmodal | D27 ablation: no_veto | glare/night / macro | -0.005384 | [-0.006449, -0.004205] | [-0.007747, -0.002801] | SURVIVES |
| final_system | D27 ablation: no_veto | glare/night / macro | -0.017289 | [-0.018589, -0.014932] | [-0.020108, -0.012408] | SURVIVES |
| final_system_crossmodal | D27 ablation: no_veto | glare/night / ship | -0.005384 | [-0.006449, -0.004205] | [-0.007747, -0.002801] | SURVIVES |
| final_system | D27 ablation: no_veto | glare/night / ship | -0.017289 | [-0.018589, -0.014932] | [-0.020108, -0.012408] | SURVIVES |
| final_system | D27 ablation: no_veto | lowlight/night / macro | -0.003814 | [-0.004729, -0.002445] | [-0.005884, -0.000859] | SURVIVES |
| final_system | D27 ablation: no_veto | lowlight/night / ship | -0.003814 | [-0.004729, -0.002445] | [-0.005884, -0.000859] | SURVIVES |
| final_system_crossmodal | D27 ablation: photometric_veto | lowlight/day / macro | -0.010182 | [-0.011648, -0.008728] | [-0.013325, -0.007061] | SURVIVES |
| final_system_crossmodal | D27 ablation: photometric_veto | lowlight/day / ship | -0.020371 | [-0.023303, -0.017470] | [-0.026372, -0.014430] | SURVIVES |
| final_system_crossmodal | D27 ablation: with_maha | glare/day / macro | -0.002288 | [-0.002867, -0.001398] | [-0.003702, -0.000267] | SURVIVES |
| final_system_crossmodal | D27 ablation: with_maha | glare/day / ship | -0.003248 | [-0.004077, -0.002372] | [-0.005149, -0.001254] | SURVIVES |
| final_system_crossmodal | D27 ablation: with_maha | lowlight/day / macro | -0.011390 | [-0.012856, -0.009702] | [-0.014533, -0.007814] | SURVIVES |
| final_system_crossmodal | D27 ablation: with_maha | lowlight/day / ship | -0.022766 | [-0.025697, -0.019409] | [-0.028765, -0.015934] | SURVIVES |
| final_system_crossmodal | gated vs visible_only | fog / macro | 0.008032 | [0.006521, 0.009725] | [0.004800, 0.011619] | SURVIVES |
| final_system | gated vs visible_only | fog / macro | 0.005476 | [0.004157, 0.007295] | [0.002618, 0.009309] | SURVIVES |
| final_system_crossmodal | gated vs visible_only | fog / ship | 0.015745 | [0.012720, 0.018847] | [0.009560, 0.022079] | SURVIVES |
| final_system | gated vs visible_only | fog / ship | 0.010614 | [0.007953, 0.014068] | [0.005140, 0.017634] | SURVIVES |
| final_system_crossmodal | gated vs visible_only | glare / macro | 0.004523 | [0.002612, 0.006172] | [0.000511, 0.008024] | SURVIVES |
| final_system_crossmodal | gated vs visible_only | glare / ship | 0.006833 | [0.004934, 0.008177] | [0.002845, 0.009738] | SURVIVES |
| final_system_crossmodal | gated vs visible_only | lowlight / macro | 0.002120 | [0.001345, 0.002865] | [0.000324, 0.003857] | SURVIVES |
| final_system | gated vs visible_only | lowlight / macro | -0.008614 | [-0.010422, -0.007262] | [-0.012425, -0.005692] | SURVIVES |
| final_system_crossmodal | gated vs visible_only | lowlight / ship | 0.003489 | [0.001915, 0.004889] | [0.000136, 0.006505] | SURVIVES |
| final_system | gated vs visible_only | lowlight / ship | -0.017990 | [-0.021635, -0.015264] | [-0.025383, -0.012389] | SURVIVES |
| veil_veto_repair_26m | veil veto repair | day | 0.071603 | [0.065677, 0.078431] | [0.059763, 0.085203] | SURVIVES |

## Decisions this table cannot re-adjudicate

Three of the five families R-A5 names were not decided by a bootstrap interval, so widening one says nothing about them. They need their own treatment:

* **The inherited-constants reprice** (`runs/eval/reprice_constants.md`) uses a *worse-somewhere* rule over per-cell deltas with a margin built from a draw SD and a bootstrap SD. The bootstrap half of that margin is understated by the R-A3 factor, so the margins are too tight, but the rule is a sign count and not an interval — re-deciding it means re-running it, not rescaling it. Its three verdicts (`cap_ir_scale`, `iou_thr`, veil repair all STAND) are **not** revisited here.
* **The soft-NMS reject** (`runs/eval/snms_gate_draw_avg.md`) is a draw-averaged sign count decided at −1.03e-5 on 4 draws. R-A3 does not touch it: its noise is corruption-draw noise, not frame-resampling noise. It was already unsound for the reason `project-gate-magnitude-floor` records — a measured margin with no absolute floor degenerates to a sign test on ~1e-5.
* **The crossmodal gate cells** are partly covered above through `final_system_crossmodal`, but the tuning sweeps (`runs/eval/crossmodal_tuning_*.json`) selected constants by comparing cell means without intervals at all. Selection under a too-narrow interval is a different failure from adoption under one, and is R-B2's territory.

## Re-adjudicated, not re-scored — the limit of this table

Every row here takes a **recorded** delta and interval and widens it. The 1.95× factor was measured on VIS UQ-arm mAP deltas over `paired_val_vis.txt`; transporting it to fusion cells assumes a comparable dependence structure. That is reasonable — same frames, same four runs, same 10 Hz cadence — but it is an **assumption, not a measurement**, and a full re-score would re-run each decision's own bootstrap through `blockboot.block_bootstrap_delta` on its own caches.

That is now mechanical rather than hard, and it is the right next step for any row this table puts in doubt. It is also the only way to settle rows where the corrected band lands close to zero, since the transported factor is a lower bound and the true inflation for those cells is unknown.
