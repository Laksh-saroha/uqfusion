# pohang04 — the single look

The single look (§7.2), under Amendments 8 and 9, at freeze commit `85a07c18655981f13339c2c2822f77265ced17f6`. 12482 pohang04 day pairs; fused ship AP; five Phase 3 systems; block bootstrap L=20, n_boot=1000, seed 1. **Not to be re-run.**

## Verdict (`clean/clean`, A9.3) — **NO-GAP**

| | value |
|---|---|
| AP_ref (pohang02+pohang03) | 0.2898 |
| AP_p04 (seed mean) | 0.2682 |
| D = AP_ref − AP_p04 | +0.0216 |
| D 95% CI (unpaired, replicate-wise) | [-0.0120, +0.0502] |

Gap at each floor: 0.0014: no, 0.0031: no, 0.006: no, 0.01: no — verdict taken at 0.006.



## All 11 cells — values and intervals (only `clean/clean` carries a verdict)

| cell (VIS/IR) | seed-mean AP | 95% CI |
|---|---:|---|
| clean/clean | 0.2682 | [0.2576, 0.2793] |
| clean/glare_s2 | 0.2668 | [0.2569, 0.2777] |
| clean/blur_s2 | 0.2671 | [0.2569, 0.2777] |
| clean/noise_s2 | 0.2692 | [0.2590, 0.2798] |
| clean/fog_s2 | 0.2691 | [0.2585, 0.2802] |
| blur_s3/clean | 0.0418 | [0.0390, 0.0448] |
| noise_s2/clean | 0.0114 | [0.0095, 0.0134] |
| rain_s2/clean | 0.1850 | [0.1744, 0.1956] |
| fog/clean | 0.0580 | [0.0529, 0.0634] |
| lowlight/glare_s2 | 0.0121 | [0.0108, 0.0139] |
| blur_s3/glare_s2 | 0.0407 | [0.0379, 0.0437] |

## Per system (and draw)

- **clean/clean**: seed0/clean 0.2789, seed1/clean 0.2696, seed2/clean 0.2655, seed3/clean 0.2646, seed4/clean 0.2625
- **clean/glare_s2**: seed0/draw941_951 0.2783, seed0/draw942_952 0.2784, seed0/draw943_953 0.2782, seed0/draw944_954 0.2783, seed1/draw941_951 0.2681, seed1/draw942_952 0.2684, seed1/draw943_953 0.2673, seed1/draw944_954 0.2687, seed2/draw941_951 0.2639, seed2/draw942_952 0.2643, seed2/draw943_953 0.2642, seed2/draw944_954 0.2642, seed3/draw941_951 0.2640, seed3/draw942_952 0.2641, seed3/draw943_953 0.2638, seed3/draw944_954 0.2647, seed4/draw941_951 0.2597, seed4/draw942_952 0.2598, seed4/draw943_953 0.2578, seed4/draw944_954 0.2596
- **clean/blur_s2**: seed0/draw941_951 0.2763, seed0/draw942_952 0.2765, seed0/draw943_953 0.2763, seed0/draw944_954 0.2763, seed1/draw941_951 0.2701, seed1/draw942_952 0.2701, seed1/draw943_953 0.2703, seed1/draw944_954 0.2701, seed2/draw941_951 0.2663, seed2/draw942_952 0.2659, seed2/draw943_953 0.2656, seed2/draw944_954 0.2656, seed3/draw941_951 0.2636, seed3/draw942_952 0.2631, seed3/draw943_953 0.2635, seed3/draw944_954 0.2637, seed4/draw941_951 0.2590, seed4/draw942_952 0.2603, seed4/draw943_953 0.2599, seed4/draw944_954 0.2602
- **clean/noise_s2**: seed0/draw941_951 0.2743, seed0/draw942_952 0.2755, seed0/draw943_953 0.2736, seed0/draw944_954 0.2753, seed1/draw941_951 0.2709, seed1/draw942_952 0.2718, seed1/draw943_953 0.2694, seed1/draw944_954 0.2702, seed2/draw941_951 0.2676, seed2/draw942_952 0.2676, seed2/draw943_953 0.2674, seed2/draw944_954 0.2678, seed3/draw941_951 0.2699, seed3/draw942_952 0.2693, seed3/draw943_953 0.2696, seed3/draw944_954 0.2700, seed4/draw941_951 0.2619, seed4/draw942_952 0.2616, seed4/draw943_953 0.2653, seed4/draw944_954 0.2656
- **clean/fog_s2**: seed0/draw941_951 0.2811, seed0/draw942_952 0.2791, seed0/draw943_953 0.2810, seed0/draw944_954 0.2804, seed1/draw941_951 0.2668, seed1/draw942_952 0.2654, seed1/draw943_953 0.2654, seed1/draw944_954 0.2671, seed2/draw941_951 0.2686, seed2/draw942_952 0.2702, seed2/draw943_953 0.2678, seed2/draw944_954 0.2701, seed3/draw941_951 0.2673, seed3/draw942_952 0.2675, seed3/draw943_953 0.2668, seed3/draw944_954 0.2672, seed4/draw941_951 0.2625, seed4/draw942_952 0.2624, seed4/draw943_953 0.2630, seed4/draw944_954 0.2626
- **blur_s3/clean**: seed0/draw941_951 0.0448, seed0/draw942_952 0.0476, seed0/draw943_953 0.0447, seed0/draw944_954 0.0465, seed1/draw941_951 0.0347, seed1/draw942_952 0.0350, seed1/draw943_953 0.0346, seed1/draw944_954 0.0339, seed2/draw941_951 0.0427, seed2/draw942_952 0.0450, seed2/draw943_953 0.0435, seed2/draw944_954 0.0437, seed3/draw941_951 0.0408, seed3/draw942_952 0.0429, seed3/draw943_953 0.0409, seed3/draw944_954 0.0396, seed4/draw941_951 0.0425, seed4/draw942_952 0.0454, seed4/draw943_953 0.0432, seed4/draw944_954 0.0440
- **noise_s2/clean**: seed0/draw941_951 0.0114, seed0/draw942_952 0.0114, seed0/draw943_953 0.0114, seed0/draw944_954 0.0113, seed1/draw941_951 0.0149, seed1/draw942_952 0.0151, seed1/draw943_953 0.0149, seed1/draw944_954 0.0143, seed2/draw941_951 0.0084, seed2/draw942_952 0.0085, seed2/draw943_953 0.0084, seed2/draw944_954 0.0085, seed3/draw941_951 0.0072, seed3/draw942_952 0.0070, seed3/draw943_953 0.0070, seed3/draw944_954 0.0070, seed4/draw941_951 0.0147, seed4/draw942_952 0.0159, seed4/draw943_953 0.0157, seed4/draw944_954 0.0148
- **rain_s2/clean**: seed0/draw941_951 0.2053, seed0/draw942_952 0.2021, seed0/draw943_953 0.2042, seed0/draw944_954 0.2049, seed1/draw941_951 0.1938, seed1/draw942_952 0.1930, seed1/draw943_953 0.1946, seed1/draw944_954 0.1955, seed2/draw941_951 0.1772, seed2/draw942_952 0.1780, seed2/draw943_953 0.1784, seed2/draw944_954 0.1788, seed3/draw941_951 0.1735, seed3/draw942_952 0.1725, seed3/draw943_953 0.1736, seed3/draw944_954 0.1740, seed4/draw941_951 0.1754, seed4/draw942_952 0.1740, seed4/draw943_953 0.1749, seed4/draw944_954 0.1759
- **fog/clean**: seed0/draw941_951 0.0498, seed0/draw942_952 0.0527, seed0/draw943_953 0.0510, seed0/draw944_954 0.0524, seed1/draw941_951 0.0338, seed1/draw942_952 0.0357, seed1/draw943_953 0.0343, seed1/draw944_954 0.0361, seed2/draw941_951 0.0722, seed2/draw942_952 0.0726, seed2/draw943_953 0.0703, seed2/draw944_954 0.0714, seed3/draw941_951 0.0546, seed3/draw942_952 0.0551, seed3/draw943_953 0.0561, seed3/draw944_954 0.0555, seed4/draw941_951 0.0755, seed4/draw942_952 0.0770, seed4/draw943_953 0.0765, seed4/draw944_954 0.0773
- **lowlight/glare_s2**: seed0/draw941_951 0.0099, seed0/draw942_952 0.0106, seed0/draw943_953 0.0097, seed0/draw944_954 0.0104, seed1/draw941_951 0.0138, seed1/draw942_952 0.0136, seed1/draw943_953 0.0127, seed1/draw944_954 0.0141, seed2/draw941_951 0.0115, seed2/draw942_952 0.0089, seed2/draw943_953 0.0097, seed2/draw944_954 0.0138, seed3/draw941_951 0.0125, seed3/draw942_952 0.0112, seed3/draw943_953 0.0103, seed3/draw944_954 0.0109, seed4/draw941_951 0.0166, seed4/draw942_952 0.0120, seed4/draw943_953 0.0138, seed4/draw944_954 0.0154
- **blur_s3/glare_s2**: seed0/draw941_951 0.0438, seed0/draw942_952 0.0465, seed0/draw943_953 0.0431, seed0/draw944_954 0.0454, seed1/draw941_951 0.0339, seed1/draw942_952 0.0337, seed1/draw943_953 0.0337, seed1/draw944_954 0.0333, seed2/draw941_951 0.0413, seed2/draw942_952 0.0437, seed2/draw943_953 0.0427, seed2/draw944_954 0.0425, seed3/draw941_951 0.0397, seed3/draw942_952 0.0419, seed3/draw943_953 0.0399, seed3/draw944_954 0.0392, seed4/draw941_951 0.0410, seed4/draw942_952 0.0439, seed4/draw943_953 0.0417, seed4/draw944_954 0.0424

## Provenance

Written automatically (R-E1/F14). A result that records a preset *name* cannot say which system produced it -- `preset="crossmodal"` named three different systems on 2026-09-01. These are values.

| field | value |
|---|---|
| `written_utc` | `2026-09-20T23:46:08+00:00` |
| `git_head` | `85a07c18655981f13339c2c2822f77265ced17f6` |
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
| `draws` | `[941, 942, 943, 944]` |
| `block_len` | `20` |
| `n_boot` | `1000` |
| `bootstrap_seed` | `1` |
| `freeze_commit` | `85a07c18655981f13339c2c2822f77265ced17f6` |
