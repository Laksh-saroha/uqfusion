# Phase 3 corrupted cells — does fused ≥ max(VIS, IR) survive the retrain?

Logged in `docs/exposure-ledger-2026-09-09.md` §7 (2026-09-27) before it ran. **No pohang04 frame is read.** Ship AP (class 0, AP50-95, local convention), preset `crossmodal26m`, five Phase 3 systems, VIS draws 941–944, caches `runs/cache_p3dev/seed{k}/draw{v}_{v+10}/`, statistics `runs/derived_p3dev/`. Day and night never pooled. **Adopts nothing.**

## Per cell

AP is the mean over seeds of the draw-averaged value. Deltas carry the between-seed 95% t-interval (df 4) on draw-averaged per-seed deltas, the decision interval. **Fails** = a delta ≤ −0.0060 with its interval entirely below zero (fixed before the run).

| cell | VIS only | IR only | fused | fused − VIS | fused − IR | VIS veto rate | fused ≥ max |
|---|---:|---:|---:|---|---|---:|---|
| fog/day | 0.0601 | 0.0218 | **0.0659** | +0.0059 [+0.0036, +0.0082] | +0.0442 [+0.0347, +0.0536] | 0.000 | holds |
| fog/night | 0.0003 | 0.0687 | **0.0687** | +0.0685 [+0.0618, +0.0751] | +0.0000 [+0.0000, +0.0000] | 1.000 | holds |
| lowlight/day | 0.0531 | 0.0218 | **0.0607** | +0.0076 [+0.0039, +0.0112] | +0.0389 [+0.0337, +0.0441] | 0.000 | holds |
| lowlight/night | 0.0005 | 0.0687 | **0.0687** | +0.0682 [+0.0610, +0.0755] | +0.0000 [+0.0000, +0.0000] | 1.000 | holds |
| glare/day | 0.2833 | 0.0218 | **0.2940** | +0.0107 [+0.0079, +0.0135] | +0.2722 [+0.2630, +0.2814] | 0.000 | holds |
| glare/night | 0.1559 | 0.0687 | **0.0687** | -0.0872 [-0.1047, -0.0697] | -0.0000 [-0.0000, -0.0000] | 0.998 | **FAILS** |

## Noise

| cell | between-seed sd (fused) | mean within-seed draw sd (fused) |
|---|---:|---:|
| fog/day | 0.0125 | 0.0012 |
| fog/night | 0.0057 | 0.0000 |
| lowlight/day | 0.0046 | 0.0007 |
| lowlight/night | 0.0057 | 0.0000 |
| glare/day | 0.0090 | 0.0010 |
| glare/night | 0.0057 | 0.0001 |

The clean cells are in `docs/eval/p3_night_check_2026-09-27.md` (clean day holds, +0.0081; clean night fails, −0.1847).

## Provenance

Written automatically (R-E1/F14). A result that records a preset *name* cannot say which system produced it -- `preset="crossmodal"` named three different systems on 2026-09-01. These are values.

| field | value |
|---|---|
| `written_utc` | `2026-09-27T12:06:13+00:00` |
| `git_head` | `503fa0fb71e3642f9d7ba647aae10b01f93c5bd0` |
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
| `cap_vis` | `0.3166696188481059` |
| `cap_ir` | `0.002786760636891604` |
| `iou_thr` | `0.85` |
| `veto` | `0.5` |
| `veto_rule` | `gini+ir_night` |
| `veto_filter` | `['dilate', 15]` |
| `veto_health_mode` | `night_arm` |
| `single_passthrough` | `True` |
| `cap_note` | `capability prior over fit frames` |
| `load_context_inputs` | `{'conditions': ['fog', 'lowlight', 'glare'], 'cache_dir': 'A:\\Uncertain\\runs\\cache_p3dev\\seed0\\draw941_951', 'manifest': 'runs/derived/paired_val_manifest.csv', 'homography': 'runs/derived/homography_ir_to_vis.json', 'constants': 'runs/eval/reliability_constants.json', 'brightness_constants': 'runs/eval/brightness_constants.json', 'bright_dir': 'runs/derived_p3dev/brightness/draw941_951', 'iou_thr': 0.85, 'veto': 0.5, 'capability_sel': 'fit', 'role': 'develop', 'bright_soft': False, 'veto_filter': ['dilate', 15], 'veil_filter': ['majority', 15], 'tau_lap': 508.6742858886719, 'preset': 'crossmodal', 'ir_condition': None, 'ir_nms': 0.7, 'vis_soft_nms': None, 'cap_ir_scale': 4.0, 'structure_dir': 'runs/derived_p3dev/structure/draw941_951', 'structure_constants': 'runs/eval/structure_constants.json', 'ir_bright': None, 'veto_keep_cls': [], 'config': None}` |
| `seeds` | `[0, 1, 2, 3, 4]` |
| `draws` | `[941, 942, 943, 944]` |
| `conditions` | `['fog', 'lowlight', 'glare']` |
| `caches` | `runs/cache_p3dev/seed{k}/draw{v}_{v+10}` |
