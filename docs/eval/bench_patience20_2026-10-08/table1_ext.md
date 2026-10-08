# Table 1 — VIS benchmark at patience 20 (ep25 grid + extensions)

All 93 runs finished; 31 families.

**14 of the 93 finished runs are author-reported (AR):** their extension logs are not in the repo, so their stop is not replayed and each scores its ep25 best. See Notes.

## Ranking (families with every seed finished)

| # | Family | Seeds | mAP@50–95 | mAP@50 | Precision | Recall | ep25 mAP@50–95 | Δ vs ep25 | Improved | Best epoch | Batch | Host |
|---:|---|:-:|---|---|---|---|---|---:|:-:|---:|---|---|
| 1 | `yolo26x` | 3/3 (1 AR) | 0.2666 ± 0.0040 | 0.668 ± 0.033 | 0.739 ± 0.010 | 0.659 ± 0.015 | 0.2666 ± 0.0040 | +0.0000 | 0/3 | 14 | 16/4 | LSLP1, author-reported, dgxanode01 |
| 2 | `yolo26l` | 3/3 | 0.2664 ± 0.0070 | 0.677 ± 0.003 | 0.763 ± 0.018 | 0.678 ± 0.015 | 0.2655 ± 0.0056 | +0.0010 | 1/3 | 20 | 8 | LSLP1 |
| 3 | `yolo26m` | 3/3 | 0.2626 ± 0.0109 | 0.672 ± 0.026 | 0.759 ± 0.037 | 0.665 ± 0.007 | 0.2576 ± 0.0056 | +0.0051 | 1/3 | 19 | 16 | LSLP1 |
| 4 | `yolov10x` | 3/3 (2 AR) | 0.2524 ± 0.0051 | 0.642 ± 0.010 | 0.744 ± 0.016 | 0.617 ± 0.005 | 0.2524 ± 0.0051 | +0.0000 | 0/3 | 16 | 4 | LSLP1, author-reported |
| 5 | `yolo12x` | 3/3 (2 AR) | 0.2523 ± 0.0091 | 0.641 ± 0.017 | 0.748 ± 0.019 | 0.641 ± 0.020 | 0.2519 ± 0.0088 | +0.0004 | 1/3 | 21 | 16 | author-reported, dgxanode01 |
| 6 | `yolo26s` | 3/3 | 0.2502 ± 0.0090 | 0.665 ± 0.007 | 0.752 ± 0.024 | 0.665 ± 0.002 | 0.2472 ± 0.0050 | +0.0030 | 1/3 | 27 | 16 | LSLP1 |
| 7 | `yolov10b` | 3/3 | 0.2476 ± 0.0051 | 0.623 ± 0.015 | 0.735 ± 0.011 | 0.605 ± 0.034 | 0.2476 ± 0.0051 | +0.0000 | 0/3 | 20 | 8 | LSLP1 |
| 8 | `yolov8l` | 3/3 | 0.2470 ± 0.0033 | 0.634 ± 0.006 | 0.747 ± 0.012 | 0.617 ± 0.020 | 0.2470 ± 0.0033 | +0.0000 | 0/3 | 11 | 16 | LSLP1 |
| 9 | `yolo11x` | 3/3 (2 AR) | 0.2466 ± 0.0078 | 0.650 ± 0.022 | 0.737 ± 0.032 | 0.645 ± 0.012 | 0.2466 ± 0.0078 | +0.0000 | 0/3 | 16 | 8 | LSLP1, author-reported |
| 10 | `yolov8x` | 3/3 | 0.2457 ± 0.0039 | 0.619 ± 0.014 | 0.731 ± 0.029 | 0.615 ± 0.025 | 0.2384 ± 0.0027 | +0.0073 | 3/3 | 40 | 16 | dgxanode01 |
| 11 | `yolo12l` | 3/3 (2 AR) | 0.2444 ± 0.0061 | 0.633 ± 0.016 | 0.732 ± 0.025 | 0.624 ± 0.011 | 0.2444 ± 0.0061 | +0.0000 | 0/3 | 15 | 8 | LSLP1, author-reported |
| 12 | `yolov10l` | 3/3 (2 AR) | 0.2444 ± 0.0078 | 0.625 ± 0.007 | 0.738 ± 0.020 | 0.626 ± 0.007 | 0.2434 ± 0.0060 | +0.0010 | 1/3 | 31 | 8 | LSLP1, author-reported |
| 13 | `yolo11m` | 3/3 | 0.2427 ± 0.0038 | 0.645 ± 0.025 | 0.744 ± 0.010 | 0.635 ± 0.024 | 0.2427 ± 0.0038 | +0.0000 | 0/3 | 16 | 16 | LSLP1 |
| 14 | `yolo11l` | 3/3 | 0.2414 ± 0.0092 | 0.631 ± 0.010 | 0.719 ± 0.027 | 0.634 ± 0.010 | 0.2406 ± 0.0089 | +0.0008 | 1/3 | 19 | 8 | LSLP1 |
| 15 | `yolo26n` | 3/3 | 0.2384 ± 0.0064 | 0.653 ± 0.016 | 0.762 ± 0.014 | 0.642 ± 0.015 | 0.2107 ± 0.0049 | +0.0276 | 3/3 | 63 | 16 | LSLP1, dgxanode01 |
| 16 | `yolov9c` | 3/3 | 0.2377 ± 0.0067 | 0.608 ± 0.018 | 0.720 ± 0.008 | 0.626 ± 0.014 | 0.2298 ± 0.0014 | +0.0079 | 2/3 | 43 | 16/8 | LSLP1 |
| 17 | `yolov9e` | 3/3 (3 AR) | 0.2376 ± 0.0081 | 0.600 ± 0.031 | 0.700 ± 0.024 | 0.624 ± 0.004 | 0.2376 ± 0.0081 | +0.0000 | 0/3 | 17 |  | author-reported |
| 18 | `yolov9m` | 3/3 | 0.2371 ± 0.0117 | 0.630 ± 0.025 | 0.732 ± 0.010 | 0.607 ± 0.016 | 0.2371 ± 0.0117 | +0.0000 | 0/3 | 9 | 16 | LSLP1 |
| 19 | `yolov10m` | 3/3 | 0.2366 ± 0.0128 | 0.611 ± 0.018 | 0.709 ± 0.009 | 0.602 ± 0.014 | 0.2340 ± 0.0090 | +0.0026 | 1/3 | 39 | 16 | LSLP1 |
| 20 | `yolo12m` | 3/3 | 0.2366 ± 0.0075 | 0.616 ± 0.024 | 0.723 ± 0.016 | 0.623 ± 0.006 | 0.2346 ± 0.0079 | +0.0019 | 2/3 | 28 | 8 | LSLP1 |
| 21 | `yolov8m` | 3/3 | 0.2331 ± 0.0039 | 0.612 ± 0.027 | 0.739 ± 0.019 | 0.603 ± 0.010 | 0.2331 ± 0.0039 | +0.0000 | 0/3 | 16 | 16 | LSLP1 |
| 22 | `yolo12s` | 3/3 | 0.2228 ± 0.0079 | 0.613 ± 0.026 | 0.715 ± 0.018 | 0.603 ± 0.009 | 0.2228 ± 0.0079 | +0.0000 | 0/3 | 13 | 16 | LSLP1 |
| 23 | `yolov8s` | 3/3 | 0.2202 ± 0.0022 | 0.605 ± 0.005 | 0.721 ± 0.006 | 0.613 ± 0.005 | 0.2202 ± 0.0022 | +0.0000 | 0/3 | 13 | 16 | LSLP1, dgxanode01 |
| 24 | `yolo11s` | 3/3 | 0.2200 ± 0.0022 | 0.601 ± 0.017 | 0.728 ± 0.021 | 0.611 ± 0.022 | 0.2200 ± 0.0022 | +0.0000 | 0/3 | 15 | 16 | LSLP1 |
| 25 | `yolov10s` | 3/3 | 0.2130 ± 0.0101 | 0.585 ± 0.036 | 0.705 ± 0.035 | 0.592 ± 0.015 | 0.2130 ± 0.0101 | +0.0000 | 0/3 | 12 | 16 | LSLP1 |
| 26 | `yolov9s` | 3/3 | 0.2052 ± 0.0069 | 0.573 ± 0.030 | 0.705 ± 0.027 | 0.598 ± 0.032 | 0.2007 ± 0.0070 | +0.0045 | 1/3 | 28 | 16 | LSLP1 |
| 27 | `yolo12n` | 3/3 | 0.1943 ± 0.0054 | 0.562 ± 0.016 | 0.691 ± 0.004 | 0.557 ± 0.024 | 0.1943 ± 0.0054 | +0.0000 | 0/3 | 13 | 16 | LSLP1 |
| 28 | `yolo11n` | 3/3 | 0.1935 ± 0.0057 | 0.561 ± 0.018 | 0.665 ± 0.018 | 0.559 ± 0.013 | 0.1935 ± 0.0057 | +0.0000 | 0/3 | 9 | 16 | LSLP1 |
| 29 | `yolov10n` | 3/3 | 0.1913 ± 0.0055 | 0.554 ± 0.013 | 0.677 ± 0.029 | 0.552 ± 0.008 | 0.1913 ± 0.0055 | +0.0000 | 0/3 | 14 | 16 | LSLP1 |
| 30 | `yolov8n` | 3/3 | 0.1833 ± 0.0028 | 0.547 ± 0.009 | 0.694 ± 0.008 | 0.560 ± 0.011 | 0.1833 ± 0.0028 | +0.0000 | 0/3 | 12 | 16 | LSLP1 |
| 31 | `yolov9t` | 3/3 | 0.1777 ± 0.0113 | 0.496 ± 0.021 | 0.660 ± 0.021 | 0.528 ± 0.039 | 0.1732 ± 0.0054 | +0.0045 | 1/3 | 33 | 16 | dgxanode01 |

## Reading

* **Leader `yolo26x`** 0.2666; next `yolo26l` at 0.2664, a margin of 0.0002 against seed sds of 0.0040 and 0.0070.
* **Not separable from the leader** (gap below the larger of the two seed sds, the Phase 1 convention; a screen, not a test): `yolo26l`, `yolo26m`.
* **Top-5 spread** 0.0143; median seed sd across families 0.0067.
* **Ranking vs ep25:** Spearman ρ = 0.967 over 31 families; top 5 at ep25 `yolo26x`, `yolo26l`, `yolo26m`, `yolov10x`, `yolo12x`; at patience 20 `yolo26x`, `yolo26l`, `yolo26m`, `yolov10x`, `yolo12x`.
* **Moved ≥ 3 places:** `yolov8x` 14→10, `yolo26n` 25→15, `yolov9c` 20→16, `yolo12m` 17→20.
* **Runs that beat their ep25 best:** 19 of 93.
* **AR runs in the top 5:** `yolo26x` 1/3, `yolov10x` 2/3, `yolo12x` 2/3.

## Notes

* **Score:** best-epoch row over base + extension, earliest maximum, the checkpoint EarlyStopping keeps. The epoch is chosen on the same val split it is reported on (as in Phase 1), so absolute values carry a small selection optimism shared by every family. *Best epoch* counts from base epoch 1.
* **Data:** `data_vis_stride4.yaml`, restored labels (train ledger `8ed69b5974ed`). The val split spans pohang00–04, pohang01 night included. These are the training-time val logs, not a new evaluation and not a holdout claim.
* **Recipe deviations, local runs** (authorized 2026-09-27): batch planned per family, the largest of 16/8/4/2 under 90% of the 12 GB card (80% from 2026-10-06, after `yolov9c` at 16 spilled to system RAM), with nbs=64 kept, so the effective batch is 64 throughout; loader workers 6 (server 2). Batch and host are per family above, per run in `table1_ext_runs.csv`.
* **Excluded:** `vis_bench_yolov8s_seed2` (diverged in the base grid (DIVERGENCE-ALARM.txt)). `yolov8s` reports seeds 0, 1, 3.
* **Replay / rows:** every finished run except the AR ones replays its stopper exactly; rows are contiguous except where noted below.
* **Author-reported (AR), recorded 2026-10-08:** the author reports that each of these runs reached patience 20 with no extension epoch above its ep25 best. Their extension logs are not in the repo, so the stop is not replayed; the score, best epoch, precision, recall and mAP50 are the ep25 best row from the server's base log, and *Ext epochs* is where the stopper fires in that case. Runs: `yolo11x_seed1`, `yolo11x_seed2`, `yolo12l_seed1`, `yolo12l_seed2`, `yolo12x_seed1`, `yolo12x_seed2`, `yolo26x_seed2`, `yolov10l_seed1`, `yolov10l_seed2`, `yolov10x_seed1`, `yolov10x_seed2`, `yolov9e_seed0`, `yolov9e_seed1`, `yolov9e_seed2`.
* `yolov9e_seed0`: author-reported 2026-10-08: patience 20 reached at ext epoch 7, no new best; 3 of 7 ext epochs logged here, none above the base.
* `yolo12x_seed0`: results.csv ends at ext epoch 21, the run at 22 (last.pt); stop = best 2 + 20, so the missing rows hold no new best.
* `yolo26x_seed1`: base best at epoch 5, gap 20 >= patience 20: final at ep25, no extension epoch.
* `yolov9c_seed1`: batch 16 -> 8 from ext epoch 44 (batch 16 spilled 4.2 GB to system RAM (epochs 14 -> 22-30 min); re-probed at FIT_FRACTION 0.80 on 2026-10-06, user-approved); nbs 64 keeps the effective batch at 64.
