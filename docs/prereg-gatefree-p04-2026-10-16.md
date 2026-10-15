# Pre-registration: gate-free BN adaptation (GF31) on the held-out run (pohang04), the thirteenth scoring, 2026-10-16

Written after GF31 PASSED development (`docs/eval/ttbn/gatefree_dev_2026-10-16.md`) and before any GF31 detection exists
on pohang04. Logged in `docs/exposure-ledger-2026-09-09.md` at the same time. It is the thirteenth scoring of pohang04.
Arm and constants as registered in `docs/prereg-gatefree-ttbn-2026-10-16.md`, unchanged:
* BN-mix with the median hinge on the first 31 of 130 BatchNorm layers;
* N = 16, n = 8, non-overlapping 8-frame blocks of the stream's frame list, every block, no gate;
* D0 from `runs/novel/ttbn_gatefree/calib.pt` (sha256 `7318e8a8e1f59a7e…`, training frames only).

Code: `scripts/ttbn_gatefree.py` (`check-fp16`, `build-p04`, `score-p04`, `boot-p04`, `report-p04`), committed before
this file.

## Cells and statistic

The nine registered held-out cells of Table 3, the same as the gate ablation's pohang04 scoring:
* VIS/IR: clean/clean (draw-free); clean/blur_s2, clean/noise_s2, clean/fog_s2; blur_s3/clean, noise_s2/clean,
  rain_s2/clean, fog/clean, lowlight/clean.
* Draws 941–944 for the corrupted cells; systems 0–4.

The statistic is unchanged:
* Ship AP against the VIS ground truth, mean over the five systems within a draw, then over the four draws.
* Δ paired.
* 95% interval: the 2.5th and 97.5th percentiles of the gate-free scoring's 1,000 moving-block replicates (L = 20,
  seed 1).
* Worse: Δ ≤ −0.0014 with interval below zero. Better: Δ ≥ +0.0060 with interval above zero.

References, all existing presorts, none recomputed:
* `union` (E4);
* IG-BN (N4's registered presorts);
* ungated BN-mix (the gate ablation's `union_ungated`).

**Precision.** fp16 autocast (`p04_heldout.setup_fp16`) iff a development check passes, as N4's:
* GF31 in fp16 on draw 942, VIS fog, fog_s1 and lowlight, IR clean, day frames, five systems, against the registered
  fp32 development scores;
* it passes iff each cell's `union` ship AP moves by at most 0.0010;
* otherwise pohang04 is built in fp32.
* The check is development-only and its numbers are reported.

## Predictions

* **Q1 (no harm and a gain against `union`): REPLICATES** iff no cell is worse and at least one is better.
  * Predicted: REPLICATES.
  * Expected better cells: noise_s2, fog, lowlight.
  * Expected gain on the four clean-VIS cells: between 0 and ungated's +0.020–0.022.
    pohang04's clean frames sit outside the training distribution, so the hinge should fire on part of them.
* **Q2 (against IG-BN, stated expectation).**
  * Not worse than IG-BN on the four clean-VIS cells, nor on blur_s3 or rain_s2, where IG-BN fires on few frames.
  * Better than IG-BN on at least one of the four clean-VIS cells.
  * May be worse than IG-BN on lowlight and noise_s2, where IG-BN fires on 85% of frames: at development the shallow
    hinge kept 82% of IG-BN's low-light gain.
* **Q3 (against ungated BN-mix, stated expectation).**
  * No cell better than ungated: pohang04 is all daylight, where full-network adaptation does best.
  * Worse than ungated in the noise_s2, fog and clean-VIS cells.

## If Q1 fails

* GF31 DOES NOT REPLICATE; it is reported as such.
* No re-reading, no change of constants, no rescoring under another rule.
* Q2 and Q3 are reported as held or failed, with counts.

## Not done here

No development constant is changed. No PAPER.md edit in this session.
