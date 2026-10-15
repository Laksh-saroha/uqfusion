# Pre-registration: gate-free BatchNorm adaptation (shallow + hinge, GF), development draws 942–944, 2026-10-16

Written at 12:10 AM on 10-16, before any gate-free arm has a detection or a score on draws 942–944. Logged in
`docs/exposure-ledger-2026-09-09.md` at the same time. Screen that chose the arm: `docs/eval/ttbn/gatefree_screen_2026-10-15.md`
(draw 941 only, unregistered). Code: `scripts/ttbn_gatefree.py` as of `fcd3ebf`, plus the scoring/verdict commands
added for this registration (they change no arithmetic of the arms). Development only: no pohang04 frame is read or
scored under this registration.

**Why.** The gate ablation (2026-10-14) showed ungated BN-mix loses on clean and night frames, while a reviewer argued
the thermal gate looks optional. The screen found an adaptation that needs no gate at all. This registration tests
whether its draw-941 behaviour holds on the verdict draws.

## Arms

Both arms use the five Phase 3 VIS checkpoints, non-overlapping blocks of 8 consecutive frames of the stream's frame
list (IG-BN's grid), fp32, and E4 `union` with the plain IR stream. **Every block is adapted; no gate.**

* **GF31 (primary): BN-mix with the median hinge on the first 31 of 130 BatchNorm layers.**
  * Layer i < 31 normalises with μ = (1 − a) μ_src + a μ_b and σ² likewise.
    * μ_src, σ²_src: the model's running statistics.
    * μ_b, σ²_b: the block's biased statistics at that layer during its own pass.
    * a = n/(N + n) · max(0, 1 − D0_i/D_i), with N = 16 and n = 8.
    * D_i: mean over channels of the symmetric KL between the block's and the source's channel Gaussians.
    * D0_i: per layer and per checkpoint, the median of D_i over 240 clean training blocks (120 day, 120 night).
  * Layers 31–129 use the running statistics unchanged.
* **GF21 (secondary):** the same with the first 21 layers.
* **Frozen constants.** k ∈ {31, 21}, N = 16, n = 8, and D0 from `runs/novel/ttbn_gatefree/calib.pt`.
  * The file is sha256 `7318e8a8e1f59a7e…`, built 2026-10-15 from training-split frames only.
  * None of those frames is in the val, test or development lists.
  * k and the median quantile were chosen on draw 941 (disclosed).
* **References.**
  * `union`: E4, plain VIS.
  * IG-BN: N4 as registered, i.e. the gate ablation's complete BN-mix streams (N4's stored blocks plus the
    complement) selected by N4's gate.
  * IG-BN must reproduce N4's registered development `union_ttbn` scores exactly (checked, max |diff| reported).

## Statistic

As N4 and the gate ablation:
* **Cells.** 5 VIS conditions (clean, fog, fog_s1, lowlight, glare) × 6 IR conditions (clean, fog_s1–s3,
  noise_s2–s3) × {day, night} = 60 cells.
* **Draws.** VIS 942–944 / IR 952–954.
* **Per system k.** Δ_k = mean over the three draws of ship AP(arm) − ship AP(reference).
* **Cell value.** The mean over the five systems, with a 95% t-interval (df 4).
* **Worse:** Δ ≤ −0.0014 with interval below zero.
* **Better:** Δ ≥ +0.0060 with interval above zero.

## Predictions (GF31; GF21 is scored the same way, descriptively)

* **P1 (no harm, the registered D-H rule):** no cell is worse than `union`.
* **P1b (gain, D-G):** at least one cell is better than `union`.
* **GF31 PASSES development iff P1 and P1b.**
* **P2 (where IR corruption switches IG-BN off):** GF31 is better than IG-BN, by the "better" rule against IG-BN, in
  all ten of these day cells:
  * fog_s1 with IR clean, fog_s1, fog_s2, fog_s3, noise_s2, noise_s3 (6 cells);
  * lowlight with IR fog_s2, fog_s3, noise_s2, noise_s3 (4 cells).
  * P2 HOLDS iff all ten are better; otherwise it reports the count.
* **P3 (night, stated expectation).** Draw 941 put clean night at −0.0021 and low-light night at −0.0022 to −0.0037,
  with intervals about ±0.005 wide. Averaging three draws narrowed the gate ablation's intervals only slightly, because
  system-to-system spread dominates. So we expect:
  * (a) every night cell's point estimate in [−0.0050, +0.0050] for clean and low-light VIS;
  * (b) no night cell meeting the "worse" rule.
  * Expected verdict P1 PASS; we put real probability (about one in three) on a low-light night cell reaching "worse".
  * P3 HOLDS iff (a) and (b).
* **P4 (descriptive, expected losses).** GF31 is below IG-BN where IG-BN's gate fires on every frame: lowlight/clean/day
  (draw 941: −0.0129) and fog/clean/day (−0.0044). These are reported, not counted as failures.

## If a prediction fails

* If P1 fails in any cell, at night or by day, **GF31 FAILS development** and is reported as FAIL.
  * No re-reading as "noise" or "within margin".
  * No change to k, D0, N, the hinge or the cells.
  * No pohang04 scoring of GF31 without a new registration that names the failure.
* A failed P2 or P3 is reported as failed, with the counts. It does not change the P1 verdict.
* GF21 cannot replace GF31 after the fact. If GF21 passes and GF31 fails, that is reported as an observation only.

## Not done here

No pohang04 frame. No PAPER.md change. No new constant fitted on 942–944.
