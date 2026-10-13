# Pre-registration: does σ's within-detector re-ranking gain replicate on the five Phase 3 detectors?

Written 2026-10-09, **before the run**. No re-ranker has been fit or scored on any Phase 3 cache: no re-ranker record in `runs/eval/` or `docs/eval/` names `cache_p3`.

## 1. What is being tested, and why

Draft 2's title now reads "Predicted Uncertainty Is Useful Within a Detector, Not Between Visible and Infrared Streams" (commit `b7a33bf`). Its "useful" half rests on one number, measured on 2026-10-09 (`docs/eval/class_set_audit_2026-10-08.md` §8):

* **What it measures:** σ's paired increment in the out-of-fold VIS re-ranker, ship AP, 1,200 paired day frames: **+0.0050 [+0.0038, +0.0064]**.
* **The two arms:** the 4-feature arm `conf, log_conf, conf_over_frame_median, sigma_mean_norm` minus the same arm without `sigma_mean_norm`, both monotone, λ 0.30.

That number has three weaknesses:

* It was not pre-registered.
* It comes from one checkpoint, `runs/full_scale/gauss_vis_seed0`.
* λ 0.30 was chosen by the 2026-09-02 macro sweep for the 4-feature arm.

This document fixes, before any Phase 3 number exists, a test of whether the increment replicates on five independently trained VIS detectors.

**What this replicates across, and what it does not.** It replicates across detectors: different seeds, trained on restored night labels with pohang04 removed (`docs/prereg-phase3-retrain-2026-09-10.md`). It does not replicate across data. The 1,200 day frames are the same, and pohang04 is spent and is not scored.

## 2. Fixed in advance

* **Detectors:** the five Phase 3 VIS checkpoints `runs/phase3_stage2/p3_vis_seed{0..4}/weights/best.pt`, through their existing caches `runs/cache_p3/seed{k}/gauss_vis_paired_clean.pkl` (2,232 frames each, `sigma_ltrb` present). All five are scored. None may be excluded, re-trained or replaced.
* **Frames:** the 1,200 paired day frames. Leave-one-run-out folds: pohang00 (836), pohang02 (247), pohang03 (117).
* **Arms:** A = 4 features (with σ); B = 3 features, the same prefix without `sigma_mean_norm`. Both monotone, both λ 0.30. The re-ranker, its features, its fitting (`fit_one`, `random_state` 0) and the score rule `conf^(1−λ) · predIoU^λ` are unchanged.
* **Metric:** ship AP (class 0), local AP. The re-ranker is fit on both classes' boxes, as in the original.
* **Statistic per seed:** the paired out-of-fold delta AP(A) − AP(B), with a moving-block bootstrap interval: L = 20, n_boot 1000, seed 0.
* **Command per seed,** at the commit that registers this file. It differs from the original only in `--cache-dir`, `--ir-cache` and `--out`:

  ```
  py -3.13 scripts/fit_rerank.py --cache-dir runs/cache_p3/seed{k} --ir-cache runs/cache_p3/seed{k}/gauss_ir_paired_clean.pkl --cls 0 --n-features 3 4 --block-len 20 --ci-arm 4 yes 0.30 --ci-vs 3 yes 0.30 --out runs/eval/rerank_p3_seed{k}_sigma_increment.md
  ```

  `--ir-cache` only adds the two cross-modal features, which sit after index 18 and enter neither arm.
* **Pre-run check (already-exposed data):** the same command on `runs/cache_m` must reproduce +0.004960 [+0.003762, +0.006392]. If it does not, the run stops, and the code is fixed under an amendment before any Phase 3 seed is scored.

## 3. Decision rule

A seed **passes at floor F** if its increment is ≥ F **and** the lower bound of its block interval is > 0.

| outcome | condition | meaning |
|---|---|---|
| **REPLICATES** | ≥ 4 of 5 seeds pass at **0.0060**, the Phase 3 floor of record | σ's within-detector gain is beyond the floor every registered verdict in the paper uses |
| **REPLICATES AT SHIP FLOOR** | not the above, and ≥ 4 of 5 seeds pass at **0.0047**, the measured ship-AP floor (`docs/eval/delta_noise_floor_ship_2026-10-08.md`) | the gain is beyond the measured ship-AP noise, not beyond the floor of record |
| **FAILS** | neither | the gain does not replicate at a floor the paper recognises |

Both floors were measured before the original +0.0050 existed: 0.0060 on 2026-09-10 and 0.0047 on 2026-10-08. The original itself clears 0.0047 but not 0.0060. The two-level rule is stated now so that neither floor is chosen after seeing a Phase 3 number.

Reported descriptively, without changing the outcome:

* per-seed pass counts at 0, 0.0031, 0.0047, 0.0060 and 0.0100;
* the seed mean and range of the increment;
* each seed's arm A against no re-ranking.

The full λ sweep the script prints is also descriptive. No other λ, feature count or monotonicity setting may enter the verdict.

## 4. Consequences for the paper, fixed now

* **REPLICATES:**
  * The title stays.
  * The abstract, §1, contribution 1, §6.4, Table L and §8 report the original as unregistered and this replication as registered, with the count k/5 and the floor.
* **REPLICATES AT SHIP FLOOR:**
  * The title stays.
  * The same sections say "beyond the measured ship-AP floor (0.0047), not the 0.0060 floor of record".
* **FAILS:**
  * The title and abstract revert to the 2026-10-08 framing: "Predicted Uncertainty Is Informative but Does Not Improve Visible–Infrared Maritime Detection".
  * §1, contribution 1, §6.4 and §8 revert to match.
  * The +0.0050 is reported as a single-checkpoint observation that did not replicate on the Phase 3 detectors.

Under any outcome:

* No seed is re-run, added or dropped after the numbers are seen.
* The rule is not amended after the numbers are seen.
* Any follow-up needs its own pre-registration.

## 5. Exposure

Development data only, logged in `docs/exposure-ledger-2026-09-09.md` §7 before the run. The Phase 3 VIS caches have been inspected before, through the fused cells (Table 3b) and the night and corrupted-cell checks. Never by the re-ranker. **No pohang04 frame is scored.** This adopts nothing into the shipped system.
