# pohang04 corrupted cells under corruption v2 (second exposure, disclosed)

**A disclosed second exposure of pohang04** (decided 2026-10-10; `docs/prereg-p04-v2-rescore-2026-10-10.md`). The corrupted cells of the single look re-scored with corruption v2 (`docs/eval/corruption_v2/README.md`); everything else is the look's: five Phase 3 systems, preset `crossmodal26m`, development calibration injected, 12482 day pairs, fused ship AP against VIS GT, draws VIS 941–944 / IR 951–954, moving-block bootstrap L=20, n_boot=1000, seed 1. Descriptive only. The `clean/clean` verdict (NO-GAP, D = +0.0216) involves no corruption and is not re-scored.

## Cells

| cell (VIS/IR) | v2 AP | v2 95% CI | v1 AP (look) | v2 − v1 / note |
|---|---:|---|---:|---|
| clean/clean (verdict, not re-scored) | 0.2682 | [0.2576, 0.2793] | 0.2682 | — |
| clean/glare_s2 | — | — | 0.2668 | IR glare not modelled in v2 |
| clean/blur_s2 | 0.2673 | [0.2570, 0.2779] | 0.2671 | +0.0002 |
| clean/noise_s2 | 0.2635 | [0.2542, 0.2738] | 0.2692 | -0.0057 |
| clean/fog_s2 | 0.2666 | [0.2565, 0.2771] | 0.2691 | -0.0025 |
| blur_s3/clean | 0.0419 | [0.0389, 0.0448] | 0.0418 | +0.0001 |
| noise_s2/clean | 0.0123 | [0.0097, 0.0142] | 0.0114 | +0.0009 |
| rain_s2/clean | 0.2209 | [0.2107, 0.2324] | 0.1850 | +0.0359 |
| fog/clean | 0.0445 | [0.0393, 0.0505] | 0.0580 | -0.0136 |
| lowlight/glare_s2 | — | — | 0.0121 | IR glare not modelled in v2; see lowlight/clean |
| blur_s3/glare_s2 | — | — | 0.0407 | IR glare not modelled in v2; see blur_s3/clean |
| lowlight/clean (new; replaces lowlight/glare_s2) | 0.2134 | [0.2040, 0.2237] | — | — |

## Per system and draw

- **clean/blur_s2**: seed0/draw941_951 0.2763, seed0/draw942_952 0.2761, seed0/draw943_953 0.2761, seed0/draw944_954 0.2761, seed1/draw941_951 0.2704, seed1/draw942_952 0.2704, seed1/draw943_953 0.2704, seed1/draw944_954 0.2705, seed2/draw941_951 0.2664, seed2/draw942_952 0.2663, seed2/draw943_953 0.2658, seed2/draw944_954 0.2659, seed3/draw941_951 0.2645, seed3/draw942_952 0.2640, seed3/draw943_953 0.2643, seed3/draw944_954 0.2647, seed4/draw941_951 0.2588, seed4/draw942_952 0.2600, seed4/draw943_953 0.2597, seed4/draw944_954 0.2598
- **clean/noise_s2**: seed0/draw941_951 0.2754, seed0/draw942_952 0.2755, seed0/draw943_953 0.2768, seed0/draw944_954 0.2749, seed1/draw941_951 0.2649, seed1/draw942_952 0.2656, seed1/draw943_953 0.2658, seed1/draw944_954 0.2671, seed2/draw941_951 0.2621, seed2/draw942_952 0.2621, seed2/draw943_953 0.2621, seed2/draw944_954 0.2622, seed3/draw941_951 0.2617, seed3/draw942_952 0.2619, seed3/draw943_953 0.2619, seed3/draw944_954 0.2620, seed4/draw941_951 0.2528, seed4/draw942_952 0.2516, seed4/draw943_953 0.2523, seed4/draw944_954 0.2513
- **clean/fog_s2**: seed0/draw941_951 0.2765, seed0/draw942_952 0.2763, seed0/draw943_953 0.2765, seed0/draw944_954 0.2765, seed1/draw941_951 0.2682, seed1/draw942_952 0.2682, seed1/draw943_953 0.2682, seed1/draw944_954 0.2680, seed2/draw941_951 0.2640, seed2/draw942_952 0.2640, seed2/draw943_953 0.2639, seed2/draw944_954 0.2636, seed3/draw941_951 0.2633, seed3/draw942_952 0.2635, seed3/draw943_953 0.2635, seed3/draw944_954 0.2634, seed4/draw941_951 0.2611, seed4/draw942_952 0.2611, seed4/draw943_953 0.2613, seed4/draw944_954 0.2611
- **blur_s3/clean**: seed0/draw941_951 0.0450, seed0/draw942_952 0.0459, seed0/draw943_953 0.0436, seed0/draw944_954 0.0458, seed1/draw941_951 0.0360, seed1/draw942_952 0.0357, seed1/draw943_953 0.0358, seed1/draw944_954 0.0355, seed2/draw941_951 0.0431, seed2/draw942_952 0.0450, seed2/draw943_953 0.0436, seed2/draw944_954 0.0457, seed3/draw941_951 0.0399, seed3/draw942_952 0.0422, seed3/draw943_953 0.0407, seed3/draw944_954 0.0390, seed4/draw941_951 0.0423, seed4/draw942_952 0.0441, seed4/draw943_953 0.0439, seed4/draw944_954 0.0442
- **noise_s2/clean**: seed0/draw941_951 0.0123, seed0/draw942_952 0.0115, seed0/draw943_953 0.0115, seed0/draw944_954 0.0116, seed1/draw941_951 0.0150, seed1/draw942_952 0.0158, seed1/draw943_953 0.0137, seed1/draw944_954 0.0138, seed2/draw941_951 0.0084, seed2/draw942_952 0.0084, seed2/draw943_953 0.0084, seed2/draw944_954 0.0084, seed3/draw941_951 0.0149, seed3/draw942_952 0.0144, seed3/draw943_953 0.0139, seed3/draw944_954 0.0140, seed4/draw941_951 0.0119, seed4/draw942_952 0.0127, seed4/draw943_953 0.0106, seed4/draw944_954 0.0152
- **rain_s2/clean**: seed0/draw941_951 0.2320, seed0/draw942_952 0.2309, seed0/draw943_953 0.2314, seed0/draw944_954 0.2330, seed1/draw941_951 0.2194, seed1/draw942_952 0.2190, seed1/draw943_953 0.2207, seed1/draw944_954 0.2198, seed2/draw941_951 0.2187, seed2/draw942_952 0.2193, seed2/draw943_953 0.2200, seed2/draw944_954 0.2178, seed3/draw941_951 0.2178, seed3/draw942_952 0.2158, seed3/draw943_953 0.2170, seed3/draw944_954 0.2179, seed4/draw941_951 0.2157, seed4/draw942_952 0.2165, seed4/draw943_953 0.2168, seed4/draw944_954 0.2178
- **fog/clean**: seed0/draw941_951 0.0537, seed0/draw942_952 0.0534, seed0/draw943_953 0.0533, seed0/draw944_954 0.0530, seed1/draw941_951 0.0441, seed1/draw942_952 0.0419, seed1/draw943_953 0.0438, seed1/draw944_954 0.0424, seed2/draw941_951 0.0410, seed2/draw942_952 0.0418, seed2/draw943_953 0.0428, seed2/draw944_954 0.0419, seed3/draw941_951 0.0396, seed3/draw942_952 0.0393, seed3/draw943_953 0.0397, seed3/draw944_954 0.0393, seed4/draw941_951 0.0451, seed4/draw942_952 0.0444, seed4/draw943_953 0.0447, seed4/draw944_954 0.0439
- **lowlight/clean**: seed0/draw941_951 0.2301, seed0/draw942_952 0.2292, seed0/draw943_953 0.2297, seed0/draw944_954 0.2284, seed1/draw941_951 0.1933, seed1/draw942_952 0.1911, seed1/draw943_953 0.1921, seed1/draw944_954 0.1914, seed2/draw941_951 0.2151, seed2/draw942_952 0.2149, seed2/draw943_953 0.2142, seed2/draw944_954 0.2161, seed3/draw941_951 0.2159, seed3/draw942_952 0.2160, seed3/draw943_953 0.2164, seed3/draw944_954 0.2160, seed4/draw941_951 0.2159, seed4/draw942_952 0.2137, seed4/draw943_953 0.2146, seed4/draw944_954 0.2138

Caches built at inference batch 4 (`build_cache_multi.py --batch`); the look's caches were batch 1. Measured batch-4 drift on a Phase 3 checkpoint: |d ship AP| ~1e-5 (`docs/eval/corruption_v2/batched_inference.json`). Gate statistics for the corrupted streams were computed in the build pass under the GPU interpreter (numpy/opencv versions in each JSON); the look's clean statistics are reused. The two interpreters agree to ≤ 6.3e-7 relative on float statistics and exactly on percentile statistics.

## Provenance

Written automatically (R-E1/F14). A result that records a preset *name* cannot say which system produced it -- `preset="crossmodal"` named three different systems on 2026-09-01. These are values.

| field | value |
|---|---|
| `written_utc` | `2026-10-10T13:56:06+00:00` |
| `git_head` | `f131fb2f0bb6e543b2d0c173bdb2d0daebfc205a` |
| `git_dirty` | `True` |
| `dirty_sha256` | `07fa13fb3349f0fd` |
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
| `corrupt_version` | `v2` |
| `infer_batch` | `4` |

**The working tree was dirty when this ran.** `git_head` alone does not identify the source that produced these numbers; `dirty_sha256` is a hash of `git diff HEAD` and is the part that does.
