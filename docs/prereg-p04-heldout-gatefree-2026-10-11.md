# Pre-registration: the gate-free systems on the held-out run (pohang04), 2026-10-11

Written before any pohang04 number for these systems exists and before any new pohang04 cache is
built. Logged in `docs/exposure-ledger-2026-09-09.md` at the same time.

**Why now.** On 2026-10-11 Laksh decided that pohang04 is scored whenever a result needs held-out
validation. The development verdicts of `docs/prereg-no-retrain-2026-10-10.md` (Table 8 of the paper)
have none. This is the third scoring of pohang04: the single look (2026-09-20, `crossmodal26m`,
clean/clean verdict) and the corruption-v2 re-score of its corrupted cells (2026-10-10) came first.
`runs/holdout_p04/LOOK_TAKEN.json`, its mirror and the look's verdict are not touched.

## Systems

All five Phase 3 systems, assembled as the look assembles them (`holdout_p04_look.system_context`:
`crossmodal26m` machinery, development calibration injected, nothing fitted on pohang04).

| arm | definition (identical to development) | role |
|---|---|---|
| `shipped` | `crossmodal26m` as is | reference |
| `union` | E4: the veto switched off (`veto=None`) | **primary** |
| `full_tta` | E8: five checkpoints per stream, each also on the horizontally flipped frame, merged by plain WBF at IoU 0.55; IR NMS 0.70 per view; IR records replaced by their NLM-restored records where the IR merge-health statistic fires (`ir_d2 > ir_bound`); no veto | **primary** |
| `ens5_union` | E5 decomposition: five checkpoints, no flips, no restoration, no veto | descriptive |
| `lex` | E9: every VIS ship box above every IR ship box | descriptive |

Every arm's VIS-only and IR-only streams are scored too (descriptive, for the claim below).

**Calibration.** Single-checkpoint arms use each system's development capability prior (the look's
injection). Ensemble arms use the prior `seed_ensemble.py` computes on development for the same arm
(its clean VIS and IR streams on the development fit frames), injected unchanged. The restoration flag
is a frame statistic of the IR image (seed-independent), read from pohang04's own statistics files, as
the shipped gate reads it.

## Cells, statistic, interval

* Cells: `clean/clean` (**primary**) and the eight corruption-v2 cells of the 2026-10-10 re-score:
  clean/blur_s2, clean/noise_s2, clean/fog_s2, blur_s3/clean, noise_s2/clean, rain_s2/clean, fog/clean,
  lowlight/clean; draws VIS 941–944 with IR 951–954, the re-score's caches. pohang04 has no night frame.
* Statistic: ship AP (class 0, AP50:95, local linear-interpolation AP) against the VIS ground truth, mean
  over the five systems, and for corrupted cells the mean over the four draws of that.
* Paired Δ = arm − `shipped`. Interval: the 2.5th and 97.5th percentiles of 1,000 moving-block bootstrap
  replicates (L = 20, seed 1), one resample scoring every arm, system and draw on the same frames.
* Check: `shipped` recomputed here must reproduce the look's clean/clean (0.2682) and the re-score's
  eight cells to the fourth decimal, or nothing is reported until the difference is explained.

## Predictions and decision rules

* **P-U, `union` is non-inferior by day.** PASS iff no cell has Δ ≤ −0.0014 with its interval below
  zero. With no night frame, `union` differs from `shipped` only where the veto fires by day, so no
  improvement is predicted and none is required.
* **P-F, `full_tta` replicates.** REPLICATES iff (a) no cell has Δ ≤ −0.0014 with its interval below
  zero, (b) at least one cell has Δ ≥ +0.0060 with its interval above zero, and (c) the primary cell
  clean/clean has Δ ≥ +0.0060 with its interval above zero. Otherwise DOES NOT REPLICATE. (Development:
  clean day +0.0235.)
* Descriptive, no verdict: the claim fused − max(VIS, IR) per arm and cell (a failure is ≤ −0.0060 with
  the interval below zero), `ens5_union` and `lex` deltas, per-system and per-draw values.

## Order and partial reports

1. clean/clean for every arm (the flipped and restored clean caches are built first).
2. The eight corrupted cells for `shipped`, `union`, `lex`, `ens5_union` (existing caches).
3. The eight corrupted cells for `full_tta`, once its caches are built.

A report written before stage 3 completes is labelled partial; P-U and P-F are decided on all nine cells.

## New caches

Under `runs/holdout_p04/tta/`, outside `runs/cache*` as the look's trees are. For each of the five
checkpoints: the horizontally flipped frame for VIS clean and the five VIS conditions × four draws and
for IR clean and the three IR conditions × four draws (boxes mapped back), and the NLM-restored IR frame
on the frames whose IR flag fires (clean and three conditions × four draws). Corruption is replayed
with the same v2 code and seeds as the re-score. Inference uses `predict_batch` at batch 8; development
flips were built at batch 1, and batching moves ship AP by about 1e-5 (measured at batch 4,
`docs/eval/corruption_v2/batched_inference.json`). Disclosed, not corrected.

**No development number changes, and no constant is fitted on pohang04.**

## Amendment 1 (10:48 AM, before any pohang04 cache is built): precision and throughput

The new pohang04 views (flips and NLM-restored IR) are computed in **fp16** (autocast), with
channels-last weights and cuDNN autotuning, one builder process, batch 8, one letterbox per frame shared
by the five checkpoints. Measured before the switch (`runs/holdout_p04/tta/bench_*.json`): fp32 runs at
74 detector passes/s, fp16 at 134–146; 2–4 concurrent processes and batch 16/32 were slower. Priced on
development before use (`runs/holdout_p04/tta/fp16_devcheck.json`, clean frames, all five checkpoints):
fp16 flip views against the development fp32 flip caches move the `full_tta` stream's ship AP by +0.00015
(VIS day), −0.00032 (VIS night), +0.00013 and +0.00008 (IR day, night); a single flip view moves by at
most 0.0015 (mean 0.00028). The unflipped views are the existing fp32 pohang04 caches. The benchmark
runs decoded pohang04 frames for timing only; no detection from them was kept or scored.

## Amendment 2 (5:05 PM, before any `full_tta_lex` number exists on pohang04): `full_tta_lex`, descriptive

Added at Laksh's request. `full_tta_lex` is E9 on top of E5 + E8, built as on development
(`seed_ensemble.ARMS["full_tta_lex"]`):
* the `full_tta` VIS and IR streams, unchanged;
* no merge: every VIS ship box ranks above every IR ship box, with IR boxes mapped to the VIS canvas
  (`gate_zoo.build_lex_records`).

Development (fresh draws 945–947, `docs/eval/no_retrain/ensembles_fresh.md`): IMPROVES yes, 57 cells
better, 2 claim failures, macro 0.0894 → 0.1803.

*What was seen first.*
* `full_tta`'s clean/clean streams on pohang04: fused 0.2925, VIS 0.2946, IR 0.0162.
* Every cell's `ens5_union` VIS stream (the `full_tta` VIS stream without its flipped views).

Appending boxes below every other box never lowers AP. So `full_tta_lex` is at least its VIS stream, and
its clean/clean value is at least 0.2946 (Δ vs `shipped` ≥ +0.0264). The corrupted cells are bounded
the same way, up to the flipped views' contribution. A gain verdict against `shipped` would test
little, so **no verdict is registered**.

No `full_tta` fused value of a corrupted cell has been displayed. Draws 941–942 are scored on disk;
their values were not read.

*Reported, descriptive.* For each of the nine cells, on the same replicates:
* Δ vs `shipped`;
* Δ vs `full_tta` (paired): does the lex ordering beat the WBF merge on pohang04?
* the claim, fused − max(VIS, IR).

Same cells, statistic and interval as above. P-U and P-F are unchanged.
