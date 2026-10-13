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
| **S0** | shipped (no sigma anywhere) | 0.265175 | 0.046345 | 0.031479 | 0.223106 |
| **S1** | real sigma, coordinates | 0.265088 | 0.046340 | 0.031478 | 0.223042 |
| **S2** | constant sigma, coordinates | 0.265175 | 0.046345 | 0.031479 | 0.223106 |
| **S3** | shuffled within frame, coordinates | 0.265666 | 0.046349 | 0.031479 | 0.222645 |
| **S4** | shuffled across cache, coordinates | 0.264891 | 0.046347 | 0.031478 | 0.223066 |
| **S5** | real sigma, score | 0.242044 | 0.044793 | 0.027346 | 0.206402 |
| **S6** | constant sigma, score | 0.256185 | 0.042119 | 0.026773 | 0.213245 |
| **S7** | shuffled within frame, score | 0.246275 | 0.038637 | 0.025624 | 0.205679 |

## Primary comparisons — the verdict rests on these


### S1 − S3 · coordinate path · **NULL**

| condition | delta | se | 95% CI (block L=20) | excludes zero |
|---|---:|---:|---|---|
| clean | -0.000578 | 0.000226 | [-0.000879, -0.000036] | **yes** |
| fog | -0.000009 | 0.000030 | [-0.000039, +0.000090] | no |
| lowlight | -0.000001 | 0.000003 | [-0.000005, +0.000010] | no |
| glare | +0.000397 | 0.000232 | [-0.000124, +0.000761] | no |

Conditions meeting both criteria, by floor: 0.0014: 0/4, 0.0031: 0/4, **0.0060: 0/4**, 0.0100: 0/4 — the verdict is taken at 0.0060 (needs 3/4).

### S5 − S7 · score path · **NULL**

| condition | delta | se | 95% CI (block L=20) | excludes zero |
|---|---:|---:|---|---|
| clean | -0.004232 | 0.004498 | [-0.010648, +0.005676] | no |
| fog | +0.006156 | 0.000787 | [+0.003934, +0.007107] | **yes** |
| lowlight | +0.001722 | 0.000472 | [+0.000592, +0.002397] | **yes** |
| glare | +0.000724 | 0.002467 | [-0.004205, +0.005273] | no |

Conditions meeting both criteria, by floor: 0.0014: 2/4, 0.0031: 1/4, **0.0060: 1/4**, 0.0100: 0/4 — the verdict is taken at 0.0060 (needs 3/4).

## Secondary comparisons — context, not verdict


**S1 − S4** · coordinate: per-box vs between-frame signal

| condition | delta | 95% CI | excludes zero |
|---|---:|---|---|
| clean | +0.000197 | [-0.000433, +0.000841] | no |
| fog | -0.000007 | [-0.000298, +0.000047] | no |
| lowlight | +0.000000 | [-0.000001, +0.000006] | no |
| glare | -0.000024 | [-0.000523, +0.000319] | no |

**S1 − S2** · coordinate: real vs constant (expected inert)

| condition | delta | 95% CI | excludes zero |
|---|---:|---|---|
| clean | -0.000087 | [-0.000318, -0.000007] | **yes** |
| fog | -0.000005 | [-0.000020, +0.000026] | no |
| lowlight | -0.000000 | [-0.000001, +0.000005] | no |
| glare | -0.000064 | [-0.000266, +0.000030] | no |

**S5 − S6** · score: real vs constant

| condition | delta | 95% CI | excludes zero |
|---|---:|---|---|
| clean | -0.014141 | [-0.023552, -0.001021] | **yes** |
| fog | +0.002674 | [+0.001067, +0.004000] | **yes** |
| lowlight | +0.000573 | [-0.000654, +0.001241] | no |
| glare | -0.006843 | [-0.012964, +0.000357] | no |

**S1 − S0** · coordinate: does enabling the mechanism at all move it

| condition | delta | 95% CI | excludes zero |
|---|---:|---|---|
| clean | -0.000087 | [-0.000318, -0.000007] | **yes** |
| fog | -0.000005 | [-0.000020, +0.000026] | no |
| lowlight | -0.000000 | [-0.000001, +0.000005] | no |
| glare | -0.000064 | [-0.000266, +0.000030] | no |

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
| `written_utc` | `2026-09-27T11:00:41+00:00` |
| `git_head` | `d9b24f4c32c600cef64b11203209ffde2d8757f7` |
| `git_dirty` | `False` |
| `dirty_sha256` | -- |
| `git_branch` | `fusion-uq-phase3` |
| `ap_convention` | `local-linear-interp` |
| `missing_class_policy` | `drop` |
| `sort_kind` | `stable` |
| `preset` | `crossmodal26m` |
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
