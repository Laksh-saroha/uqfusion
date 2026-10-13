# Phase 1 robustness — Table 1 mAP vs curve-peak, per run

Campaign: **main**

| run | Table 1 mAP50-95 | curve peak | delta | best ep | last ep | gap | stop reason | admissible |
|---|---|---|---|---|---|---|---|---|
| main_yolo12s_seed0 | 0.27198 | 0.27033 | +0.00165 | 11 | 31 | 20 | early_stop | yes |
| main_yolo12s_seed1 | 0.29089 | 0.28857 | +0.00232 | 18 | 38 | 20 | early_stop | yes |
| main_yolo12s_seed2 | 0.27189 | 0.27045 | +0.00144 | 12 | 32 | 20 | early_stop | yes |
| main_yolo12m_seed0 | 0.29219 | 0.29008 | +0.00211 | 26 | 46 | 20 | early_stop | yes |
| main_yolo12m_seed1 | 0.29424 | 0.29197 | +0.00227 | 41 | 61 | 20 | early_stop | yes |
| main_yolo12m_seed2 | 0.28547 | 0.28362 | +0.00185 | 25 | 45 | 20 | early_stop | yes |
| main_yolo12l_seed0 | 0.27997 | 0.27793 | +0.00204 | 14 | 34 | 20 | early_stop | yes |
| main_yolo12l_seed1 | 0.28303 | 0.28081 | +0.00222 | 30 | 50 | 20 | early_stop | yes |
| main_yolo12l_seed2 | 0.29788 | 0.29550 | +0.00238 | 14 | 34 | 20 | early_stop | yes |
| main_yolo12x_seed0 | 0.30414 | 0.30152 | +0.00262 | 9 | 29 | 20 | shared_dir_past_patience | yes |
| main_yolo12x_seed1 | 0.29548 | 0.29430 | +0.00118 | 22 | 36 | 14 | shared_dir_before_patience | no |
| main_yolo12x_seed2 | 0.29719 | 0.29500 | +0.00219 | 4 | 24 | 20 | early_stop | yes |
| main_yolo26n_seed0 | 0.24734 | 0.24701 | +0.00033 | 16 | 36 | 20 | early_stop | yes |
| main_yolo26n_seed1 | 0.25802 | 0.25775 | +0.00027 | 23 | 43 | 20 | early_stop | yes |
| main_yolo26n_seed2 | 0.25656 | 0.25651 | +0.00005 | 16 | 36 | 20 | early_stop | yes |
| main_yolo26s_seed0 | 0.28635 | 0.28608 | +0.00027 | 11 | 31 | 20 | early_stop | yes |
| main_yolo26s_seed1 | 0.27780 | 0.27735 | +0.00045 | 11 | 31 | 20 | early_stop | yes |
| main_yolo26s_seed2 | 0.27971 | 0.27936 | +0.00035 | 22 | 42 | 20 | early_stop | yes |
| main_yolo26m_seed0 | 0.30607 | 0.34474 | -0.03867 | 9 | 29 | 20 | early_stop | yes [CURVE ON 8.4.7 SCALE] |
| main_yolo26m_seed1 | 0.29626 | 0.29586 | +0.00040 | 6 | 26 | 20 | early_stop | yes |
| main_yolo26m_seed2 | 0.30244 | 0.30201 | +0.00043 | 10 | 33 | 23 | killed_past_patience | yes |
| main_yolo26l_seed0 | 0.30253 | 0.30229 | +0.00024 | 8 | 28 | 20 | early_stop | yes |
| main_yolo26l_seed1 | 0.29743 | 0.29710 | +0.00033 | 8 | 28 | 20 | early_stop_after_resume | yes |
| main_yolo26l_seed2 | 0.29947 | 0.29910 | +0.00037 | 6 | 26 | 20 | killed_past_patience | yes |
| main_yolo26x_seed0 | 0.30479 | 0.30406 | +0.00073 | 8 | 28 | 20 | early_stop | yes |
| main_yolo26x_seed1 | 0.30690 | 0.30700 | -0.00010 | 12 | 32 | 20 | early_stop | yes |
| main_yolo26x_seed2 | 0.30294 | 0.30270 | +0.00024 | 30 | 50 | 20 | early_stop_after_resume | yes |

Explicit-val minus curve-peak across 26 runs: mean +0.00110, sd 0.00092, range -0.00010..+0.00262

### Offset by family

| family | n | mean delta |
|---|---|---|
| `yolo12*` | 12 | +0.00202 |
| `yolo26*` | 14 | +0.00031 |

### Seed-means, ALL rows

| variant | n | Table 1 mean | curve-peak mean |
|---|---|---|---|
| yolo26x | 3 | 0.3049 | 0.3046 |
| yolo26l | 3 | 0.2998 | 0.2995 |
| yolo26m | 2 | 0.2994 | 0.2989 |
| yolo12x | 3 | 0.2989 | 0.2969 |
| yolo12m | 3 | 0.2906 | 0.2886 |
| yolo12l | 3 | 0.2870 | 0.2847 |
| yolo26s | 3 | 0.2813 | 0.2809 |
| yolo12s | 3 | 0.2783 | 0.2765 |
| yolo26n | 3 | 0.2540 | 0.2538 |

ranking identical under both metrics: **True**

### Seed-means, ADMISSIBLE rows only

| variant | n | Table 1 mean | curve-peak mean |
|---|---|---|---|
| yolo26x | 3 | 0.3049 | 0.3046 |
| yolo12x | 2 | 0.3007 | 0.2983 |
| yolo26l | 3 | 0.2998 | 0.2995 |
| yolo26m | 2 | 0.2994 | 0.2989 |
| yolo12m | 3 | 0.2906 | 0.2886 |
| yolo12l | 3 | 0.2870 | 0.2847 |
| yolo26s | 3 | 0.2813 | 0.2809 |
| yolo12s | 3 | 0.2783 | 0.2765 |
| yolo26n | 3 | 0.2540 | 0.2538 |

ranking identical under both metrics: **False**
- Table 1 order: yolo26x > yolo12x > yolo26l > yolo26m > yolo12m > yolo12l > yolo26s > yolo12s > yolo26n
- curve-peak order: yolo26x > yolo26l > yolo26m > yolo12x > yolo12m > yolo12l > yolo26s > yolo12s > yolo26n

