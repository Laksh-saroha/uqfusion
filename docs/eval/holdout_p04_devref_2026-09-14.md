# pohang04 holdout — development reference (A9.3)

Executes `docs/prereg-phase3-retrain-2026-09-10.md` Amendment 9 §A9.3 and §A9.5 item 2. **No pohang04 frame is read.** Fused ship AP, `clean/clean`, preset `crossmodal26m`, five Phase 3 systems (VIS seed k + IR seed k), caches `runs/cache_p3/seed{k}/`.

## Reference — **AP_ref = 0.2898** (pohang02+pohang03)

Fixed from here. The single look scores pohang04 as `AP_p04` (seed mean, block bootstrap seed 1) and declares HOLDOUT-GAP iff `AP_ref − AP_p04 ≥ 0.0060` **and** the unpaired 95% interval of that difference, built replicate by replicate against the 1,000 replicates stored in the JSON beside this file, lies above zero.

## Both development groups

| group | frames | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | **seed mean** | 95% CI | seed sd |
|---|---:|---:|---:|---:|---:|---:|---:|---|---:|
| pohang00 | 836 | 0.3954 | 0.3907 | 0.4134 | 0.3965 | 0.3813 | **0.3955** | [0.3352, 0.4827] | 0.0117 |
| pohang02+pohang03 | 364 | 0.2900 | 0.2650 | 0.2840 | 0.3084 | 0.3015 | **0.2898** | [0.2580, 0.3175] | 0.0168 |

Spread between the groups: **0.1057** (A9.3 was chosen on a pre-Phase-3 figure of 0.033; this is the Phase 3 value). The seed sd describes training variance and is not a decision input.

## Provenance

Written automatically (R-E1/F14). A result that records a preset *name* cannot say which system produced it -- `preset="crossmodal"` named three different systems on 2026-09-01. These are values.

| field | value |
|---|---|
| `written_utc` | `2026-09-14T11:22:18+00:00` |
| `git_head` | `babceb6c8c6ed5596fd6e079160487d67245659b` |
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
