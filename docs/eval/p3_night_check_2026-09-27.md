# Phase 3 night check — does fused ≥ max(VIS, IR) survive the retrain?

Logged in `docs/exposure-ledger-2026-09-09.md` §7 (2026-09-27) before it ran. **No pohang04 frame is read.** Ship AP (class 0, AP50-95, local convention), `clean/clean`, preset `crossmodal26m`, five Phase 3 systems, caches `runs/cache_p3/seed{k}/`. Day and night are never pooled. `off` = the veto's night arm removed (V1's OFF arm). **Adopts nothing.**

## Verdict on the paper's claim (fixed before the run)

* **day**: fused ≥ max(VIS, IR) is not refuted — `on−vis` +0.0081, `on−ir` +0.3348
* **night**: fused ≥ max(VIS, IR) **FAILS** — `on−vis` -0.1847, `on−ir` +0.0000

Fails = a seed-mean delta ≤ −0.0060 with its between-seed 95% t-interval entirely below zero.

## Ship AP per arm

| slice | arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | **mean** | sd |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| day | `vis` | 0.3507 | 0.3391 | 0.3543 | 0.3544 | 0.3439 | **0.3485** | 0.0068 |
| day | `ir` | 0.0223 | 0.0152 | 0.0218 | 0.0178 | 0.0319 | **0.0218** | 0.0064 |
| day | `on` | 0.3580 | 0.3431 | 0.3657 | 0.3637 | 0.3526 | **0.3566** | 0.0091 |
| day | `off` | 0.3580 | 0.3431 | 0.3657 | 0.3637 | 0.3526 | **0.3566** | 0.0091 |
| night | `vis` | 0.2431 | 0.2360 | 0.2644 | 0.2585 | 0.2654 | **0.2535** | 0.0132 |
| night | `ir` | 0.0646 | 0.0754 | 0.0656 | 0.0636 | 0.0745 | **0.0687** | 0.0057 |
| night | `on` | 0.0646 | 0.0754 | 0.0656 | 0.0636 | 0.0745 | **0.0687** | 0.0057 |
| night | `off` | 0.2797 | 0.2690 | 0.2940 | 0.2933 | 0.3115 | **0.2895** | 0.0161 |

## Deltas

Seed CI = between-seed 95% t-interval (df 4), the decision interval. Boot CI = paired moving-block bootstrap over frames (L = 20, seed 0), **evaluation noise only**. Last column: below zero by more than each floor 0.0014 / 0.0031 / 0.0060 / 0.0100 with the seed CI clear of zero.

| slice | delta | mean | seed CI | boot CI | floors |
|---|---|---:|---|---|---|
| day | `on-vis` | **+0.0081** | [+0.0047, +0.0115] | [+0.0030, +0.0136] | · · · · |
| day | `on-ir` | **+0.3348** | [+0.3216, +0.3480] | [+0.2974, +0.3819] | · · · · |
| day | `off-on` | **+0.0000** | [+0.0000, +0.0000] | [+0.0000, +0.0000] | · · · · |
| day | `off-vis` | **+0.0081** | [+0.0047, +0.0115] | [+0.0030, +0.0136] | · · · · |
| night | `on-vis` | **-0.1847** | [-0.2039, -0.1655] | [-0.1972, -0.1728] | Y Y Y Y |
| night | `on-ir` | **+0.0000** | [+0.0000, +0.0000] | [+0.0000, +0.0000] | · · · · |
| night | `off-on` | **+0.2208** | [+0.1995, +0.2420] | [+0.2014, +0.2395] | · · · · |
| night | `off-vis` | **+0.0360** | [+0.0284, +0.0437] | [+0.0269, +0.0449] | · · · · |

## Veto rates (VIS dropped)

| slice | frames | `on` per seed | `off` per seed |
|---|---:|---|---|
| day | 1200 | 0.000 0.000 0.000 0.000 0.000 | 0.000 0.000 0.000 0.000 0.000 |
| night | 1032 | 1.000 1.000 1.000 1.000 1.000 | 0.000 0.000 0.000 0.000 0.000 |

## Provenance

Written automatically (R-E1/F14). A result that records a preset *name* cannot say which system produced it -- `preset="crossmodal"` named three different systems on 2026-09-01. These are values.

| field | value |
|---|---|
| `written_utc` | `2026-09-27T10:16:01+00:00` |
| `git_head` | `8cbe999f06a1b280061b6dad99e8eb943d52133d` |
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
| `load_context_inputs` | `{'conditions': ['clean'], 'cache_dir': 'A:\\Uncertain\\runs\\cache_p3\\seed0', 'manifest': 'runs/derived/paired_val_manifest.csv', 'homography': 'runs/derived/homography_ir_to_vis.json', 'constants': 'runs/eval/reliability_constants.json', 'brightness_constants': 'runs/eval/brightness_constants.json', 'bright_dir': 'runs/derived/brightness', 'iou_thr': 0.85, 'veto': 0.5, 'capability_sel': 'fit', 'role': 'develop', 'bright_soft': False, 'veto_filter': ['dilate', 15], 'veil_filter': ['majority', 15], 'tau_lap': 508.6742858886719, 'preset': 'crossmodal', 'ir_condition': None, 'ir_nms': 0.7, 'vis_soft_nms': None, 'cap_ir_scale': 4.0, 'structure_dir': 'runs/derived/structure', 'structure_constants': 'runs/eval/structure_constants.json', 'ir_bright': None, 'veto_keep_cls': [], 'config': None}` |
| `seeds` | `[0, 1, 2, 3, 4]` |
| `cell` | `clean` |
| `block_len` | `20` |
| `n_boot` | `1000` |
| `bootstrap_seed` | `0` |
| `caches` | `['runs/cache_p3/seed0', 'runs/cache_p3/seed1', 'runs/cache_p3/seed2', 'runs/cache_p3/seed3', 'runs/cache_p3/seed4']` |
