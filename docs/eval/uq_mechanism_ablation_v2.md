# R-D1 - does predicted uncertainty improve the fusion?

Pre-registered in [`docs/prereg-uq-mechanism-ablation.md`](../../docs/prereg-uq-mechanism-ablation.md) (`a8f087c`, amendment 1 `f713961`) **before this ran**. The decision rule below was fixed there; this report only applies it.

## The finding this exists because of

Read off the resolved context at run time, not quoted from a document:

| term | value |
|---|---|
| `mu_d_vis` | `1000000000.0` |
| `lam_vis` | `0.0` |
| `mu_d_ir` | `1000000000.0` |
| `lam_ir` | `0.0` |
| `sigma_score_alpha_default` | `0.0` |

With `mu_d` at 1e9 and `lam` at 0, the Mahalanobis soft weight is inert and the fusion weight is a constant. No predicted uncertainty reaches the fusion decision in the shipped preset.

## Arms

| arm | what | AP clean | AP fog | AP lowlight | AP glare |
|---|---|---:|---:|---:|---:|
| **S0** | shipped (no sigma anywhere) | 0.256524 | 0.021794 | 0.029046 | 0.214990 |
| **S1** | real sigma, coordinates | 0.256436 | 0.021794 | 0.029045 | 0.214917 |
| **S2** | constant sigma, coordinates | 0.256524 | 0.021794 | 0.029046 | 0.214990 |
| **S3** | shuffled within frame, coordinates | 0.257002 | 0.021794 | 0.029047 | 0.214512 |
| **S4** | shuffled across cache, coordinates | 0.256286 | 0.021794 | 0.029046 | 0.214937 |
| **S5** | real sigma, score | 0.239857 | 0.023389 | 0.026967 | 0.204097 |
| **S6** | constant sigma, score | 0.253015 | 0.020309 | 0.025744 | 0.210148 |
| **S7** | shuffled within frame, score | 0.244277 | 0.019429 | 0.024972 | 0.203582 |

## Primary comparisons — the verdict rests on these


### S1 − S3 · coordinate path · **NULL**

| condition | delta | se | 95% CI (block L=20) | excludes zero |
|---|---:|---:|---|---|
| clean | -0.000566 | 0.000228 | [-0.000893, -0.000041] | **yes** |
| fog | +0.000000 | 0.000000 | [+0.000000, +0.000000] | no |
| lowlight | -0.000001 | 0.000003 | [-0.000004, +0.000008] | no |
| glare | +0.000406 | 0.000231 | [-0.000123, +0.000764] | no |

Conditions meeting both criteria, by floor: 0.0014: 0/4, 0.0031: 0/4, **0.0060: 0/4**, 0.0100: 0/4 — the verdict is taken at 0.0060 (needs 3/4).

### S5 − S7 · score path · **NULL**

| condition | delta | se | 95% CI (block L=20) | excludes zero |
|---|---:|---:|---|---|
| clean | -0.004420 | 0.004676 | [-0.011030, +0.005829] | no |
| fog | +0.003960 | 0.000624 | [+0.002691, +0.005183] | **yes** |
| lowlight | +0.001995 | 0.000443 | [+0.000944, +0.002684] | **yes** |
| glare | +0.000514 | 0.002530 | [-0.004569, +0.004948] | no |

Conditions meeting both criteria, by floor: 0.0014: 2/4, 0.0031: 1/4, **0.0060: 0/4**, 0.0100: 0/4 — the verdict is taken at 0.0060 (needs 3/4).

## Secondary comparisons — context, not verdict


**S1 − S4** · coordinate: per-box vs between-frame signal

| condition | delta | 95% CI | excludes zero |
|---|---:|---|---|
| clean | +0.000150 | [-0.000505, +0.000800] | no |
| fog | +0.000000 | [+0.000000, +0.000000] | no |
| lowlight | -0.000000 | [-0.000001, +0.000005] | no |
| glare | -0.000020 | [-0.000522, +0.000320] | no |

**S1 − S2** · coordinate: real vs constant (expected inert)

| condition | delta | 95% CI | excludes zero |
|---|---:|---|---|
| clean | -0.000089 | [-0.000335, -0.000005] | **yes** |
| fog | +0.000000 | [+0.000000, +0.000000] | no |
| lowlight | -0.000000 | [-0.000001, +0.000005] | no |
| glare | -0.000073 | [-0.000278, +0.000030] | no |

**S5 − S6** · score: real vs constant

| condition | delta | 95% CI | excludes zero |
|---|---:|---|---|
| clean | -0.013157 | [-0.022630, -0.000678] | **yes** |
| fog | +0.003079 | [+0.001774, +0.003849] | **yes** |
| lowlight | +0.001222 | [+0.000159, +0.001686] | **yes** |
| glare | -0.006051 | [-0.012570, +0.000803] | no |

**S1 − S0** · coordinate: does enabling the mechanism at all move it

| condition | delta | 95% CI | excludes zero |
|---|---:|---|---|
| clean | -0.000089 | [-0.000335, -0.000005] | **yes** |
| fog | +0.000000 | [+0.000000, +0.000000] | no |
| lowlight | -0.000000 | [-0.000001, +0.000005] | no |
| glare | -0.000073 | [-0.000278, +0.000030] | no |

## Was the control actually applied?

A permutation that silently did nothing would be indistinguishable from a null result, so the audit is published rather than assumed.

| arm/stream | mode | sigma rows | rows moved | singletons |
|---|---|---:|---:|---:|
| S0/ir | none | 54259 | 0 | 0 |
| S0/vis | none | 19899 | 0 | 0 |
| S1/ir | real | 54259 | 0 | 0 |
| S1/vis | real | 19899 | 0 | 0 |
| S2/ir | const | 54259 | 0 | 0 |
| S2/vis | const | 19899 | 0 | 0 |
| S3/ir | shuf_frame | 54259 | 52077 | 11 |
| S3/vis | shuf_frame | 19899 | 18666 | 5 |
| S4/ir | shuf_cache | 54259 | 54259 | 0 |
| S4/vis | shuf_cache | 19899 | 19897 | 0 |
| S5/ir | real | 54259 | 0 | 0 |
| S5/vis | real | 19899 | 0 | 0 |
| S6/ir | const | 54259 | 0 | 0 |
| S6/vis | const | 19899 | 0 | 0 |
| S7/ir | shuf_frame | 54259 | 52077 | 11 |
| S7/vis | shuf_frame | 19899 | 18666 | 5 |

## Provenance

Written automatically (R-E1/F14). A result that records a preset *name* cannot say which system produced it -- `preset="crossmodal"` named three different systems on 2026-09-01. These are values.

| field | value |
|---|---|
| `written_utc` | `2026-09-10T06:55:57+00:00` |
| `git_head` | `6d7e56e9e0aa1c1111c3513a198f18c8b63b6a5e` |
| `git_dirty` | `True` |
| `dirty_sha256` | `3adb938a3d6523dd` |
| `git_branch` | `fusion-uq-phase3` |
| `ap_convention` | `local-linear-interp` |
| `missing_class_policy` | `drop` |
| `sort_kind` | `stable` |
| `preset` | `crossmodal` |
| `preset_resolved` | `crossmodal` |
| `role` | `develop` |
| `ir_nms` | `0.7` |
| `cap_ir_scale` | `4.0` |
| `cap_vis` | `0.3232903212523053` |
| `cap_ir` | `0.0023961514901125443` |
| `iou_thr` | `0.85` |
| `veto` | `0.5` |
| `veto_rule` | `gini+ir_night` |
| `veto_filter` | `['dilate', 15]` |
| `veto_health_mode` | `night_arm` |
| `single_passthrough` | `True` |
| `cap_note` | `capability prior over fit frames` |
| `load_context_inputs` | `{'conditions': ['clean', 'fog', 'lowlight', 'glare'], 'cache_dir': 'A:\\Uncertain\\runs\\cache_m', 'manifest': 'runs/derived/paired_val_manifest.csv', 'homography': 'runs/derived/homography_ir_to_vis.json', 'constants': 'runs/eval/reliability_constants.json', 'brightness_constants': 'runs/eval/brightness_constants.json', 'bright_dir': 'runs/derived/brightness', 'iou_thr': 0.85, 'veto': 0.5, 'capability_sel': 'fit', 'role': 'develop', 'bright_soft': False, 'veto_filter': ['dilate', 15], 'veil_filter': ['majority', 15], 'tau_lap': 508.6742858886719, 'preset': 'crossmodal', 'ir_condition': None, 'ir_nms': 0.7, 'vis_soft_nms': None, 'cap_ir_scale': 4.0, 'structure_dir': 'runs/derived/structure', 'structure_constants': 'runs/eval/structure_constants.json', 'ir_bright': None, 'veto_keep_cls': [], 'config': None}` |
| `alpha` | `1.0` |
| `block_len` | `20` |
| `n_boot` | `1000` |
| `seed` | `0` |
| `adopted_floor` | `0.006` |
| `cache_dir` | `runs/cache_m` |

**The working tree was dirty when this ran.** `git_head` alone does not identify the source that produced these numbers; `dirty_sha256` is a hash of `git diff HEAD` and is the part that does.
