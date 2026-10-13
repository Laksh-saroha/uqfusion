# Phase 3 corrupted cells — does fused ≥ max(VIS, IR) survive the retrain?

Logged in `docs/exposure-ledger-2026-09-09.md` §7 (2026-09-27) before it ran. **No pohang04 frame is read.** Ship AP (class 0, AP50-95, local convention), preset `crossmodal26m`, five Phase 3 systems, VIS conditions `fog`, `lowlight`, `glare`, VIS draws 941–944, caches `runs/cache_p3dev_v2/seed{k}/draw{v}_{v+10}/`, statistics `runs/derived_p3dev_v2/`. Day and night never pooled. **Adopts nothing.**

## Per cell

AP is the mean over seeds of the draw-averaged value. Deltas carry the between-seed 95% t-interval (df 4) on draw-averaged per-seed deltas, the decision interval. **Fails** = a delta ≤ −0.0060 with its interval entirely below zero (fixed before the run).

| cell | VIS only | IR only | fused | fused − VIS | fused − IR | VIS veto rate | fused ≥ max |
|---|---:|---:|---:|---|---|---:|---|
| fog/day | 0.0106 | 0.0218 | **0.0239** | +0.0133 [+0.0065, +0.0200] | +0.0021 [-0.0022, +0.0064] | 0.000 | holds |
| fog/night | 0.0439 | 0.0687 | **0.0687** | +0.0248 [+0.0172, +0.0324] | -0.0000 [-0.0004, +0.0004] | 0.864 | holds |
| lowlight/day | 0.1719 | 0.0218 | **0.1790** | +0.0070 [+0.0047, +0.0094] | +0.1572 [+0.1354, +0.1789] | 0.000 | holds |
| lowlight/night | 0.2390 | 0.0687 | **0.0687** | -0.1703 [-0.1862, -0.1543] | +0.0000 [+0.0000, +0.0000] | 1.000 | **FAILS** |
| glare/day | 0.1581 | 0.0218 | **0.1701** | +0.0120 [+0.0084, +0.0156] | +0.1483 [+0.1366, +0.1600] | 0.000 | holds |
| glare/night | 0.0833 | 0.0687 | **0.0687** | -0.0146 [-0.0268, -0.0024] | +0.0000 [+0.0000, +0.0000] | 1.000 | **FAILS** |

## Noise

| cell | between-seed sd (fused) | mean within-seed draw sd (fused) |
|---|---:|---:|
| fog/day | 0.0037 | 0.0016 |
| fog/night | 0.0059 | 0.0001 |
| lowlight/day | 0.0211 | 0.0013 |
| lowlight/night | 0.0057 | 0.0000 |
| glare/day | 0.0121 | 0.0017 |
| glare/night | 0.0057 | 0.0000 |

The clean cells are in `docs/eval/p3_night_check_2026-09-27.md` (clean day holds, +0.0081; clean night fails, −0.1847).

## Provenance

Written automatically (R-E1/F14). A result that records a preset *name* cannot say which system produced it -- `preset="crossmodal"` named three different systems on 2026-09-01. These are values.

| field | value |
|---|---|
| `written_utc` | `2026-10-09T22:22:20+00:00` |
| `git_head` | `f131fb2f0bb6e543b2d0c173bdb2d0daebfc205a` |
| `git_dirty` | `True` |
| `dirty_sha256` | `6994a8937387c465` |
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
| `load_context_inputs` | `{'conditions': ['fog', 'lowlight', 'glare'], 'cache_dir': 'A:\\Uncertain\\runs\\cache_p3dev_v2\\seed0\\draw941_951', 'manifest': 'runs/derived/paired_val_manifest.csv', 'homography': 'runs/derived/homography_ir_to_vis.json', 'constants': 'runs/eval/reliability_constants.json', 'brightness_constants': 'runs/eval/brightness_constants.json', 'bright_dir': 'runs/derived_p3dev_v2/brightness/draw941_951', 'iou_thr': 0.85, 'veto': 0.5, 'capability_sel': 'fit', 'role': 'develop', 'bright_soft': False, 'veto_filter': ['dilate', 15], 'veil_filter': ['majority', 15], 'tau_lap': 508.6742858886719, 'preset': 'crossmodal', 'ir_condition': None, 'ir_nms': 0.7, 'vis_soft_nms': None, 'cap_ir_scale': 4.0, 'structure_dir': 'runs/derived_p3dev_v2/structure/draw941_951', 'structure_constants': 'runs/eval/structure_constants.json', 'ir_bright': None, 'veto_keep_cls': [], 'config': None}` |
| `seeds` | `[0, 1, 2, 3, 4]` |
| `draws` | `[941, 942, 943, 944]` |
| `conditions` | `['fog', 'lowlight', 'glare']` |
| `caches` | `runs/cache_p3dev_v2/seed{k}/draw{v}_{v+10}` |

**The working tree was dirty when this ran.** `git_head` alone does not identify the source that produced these numbers; `dirty_sha256` is a hash of `git diff HEAD` and is the part that does.
