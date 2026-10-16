# Pre-registration: shallow BN-mix (GF31) and ungated BN-mix on the pohang01 night test, 2026-10-16

Written at 7:38 PM on 10-16, before any adapted detection exists on these frames. Logged in
`docs/exposure-ledger-2026-09-09.md` at the same time. Code: `scripts/gatefree_night_p01.py` (this commit), which
reuses `p01_nighttest.py` (frames, systems, presorts, replicates) and `ttbn_gatefree.py` (GF31's registered
arithmetic) unchanged.

**Why.** Shallow BN-mix is the only one of the paper's three answers that adapts at night, and its night evidence is
development only (clean and low-light night cells −0.0046 to −0.0021, every interval crossing zero). IG-BN's gate and
the clock gate were checked on the night test's frames (`docs/eval/ttbn/gate_night_p01_2026-10-14.md`); shallow BN-mix
was not. Ungated BN-mix, the ablation's evidence that whole-network adaptation needs a day–night signal (clean night
−0.0195 on development), has also never been scored on night frames outside development.

## Frames, systems, arms

* **Frames.** The night test's 1,109 pohang01 test-split pairs (`runs/night_p01test/pairs_vis.txt`, capture order),
  VIS and IR clean. They were scored once before, on 2026-10-11 (`docs/prereg-p01-nighttest-2026-10-11.md`: `union`,
  `shipped`, `lex`, `ens5_union`; Amendment 1: `full_tta`, `full_tta_lex`), and their gate statistics were measured on
  2026-10-14 without labels. This is their second registered scoring.
* **Calibration.** GF31's D₀ (`runs/novel/ttbn_gatefree/calib.pt`, sha256 `7318e8a8…`) was built from training-split
  blocks only; the 1,629 pohang01 test-split frames are in the excluded list (`runs/derived/vis_test_p04out.txt`).
* **Blocks.** Non-overlapping blocks of 8 consecutive entries of the pair list: 139 blocks, of which 138 are
  contiguous in frame number (gaps of at most 2) and one spans the jump between the test split's two stretches.
* **Systems.** The five Phase 3 systems, assembled exactly as the night test assembled them; fused by `union` (E4,
  veto off) with the plain IR stream.
* **Arms (fp32, as development; every block adapted, no gate).**
  * **GF31 (primary):** the registered shallow BN-mix: first 31 of 130 BatchNorm layers, weight
    n/(N + n) · max(0, 1 − D₀/D), N = 16, n = 8.
  * **mix16 (secondary):** ungated BN-mix on every layer, N = 16.
* **References.** `union` as stored by the night test. IG-BN equals `union` on every pair here (its gate fires on no
  pair with IR clean), and so does the clock gate (sun below the horizon on every pair).

## Statistic

* Ship AP (class 0, AP50:95, local linear interpolation) against the VIS labels, mean over the five systems.
* Δ = arm − `union`, paired.
* 95% interval: 2.5th–97.5th percentiles of the night test's own 1,000 moving-block replicates (L = 20, seed 1,
  `runs/night_p01test/boot_weights.npy`), the same replicates that scored `union`.
* Worse: Δ ≤ −0.0014 with the interval below zero. Better: Δ ≥ +0.0060 with the interval above zero (§3.4).

## Predictions

The night test's level differs from development's: `union` scores 0.0506 here against 0.2895 on development's clean
night, and night AP swings between this run's 52-second episodes. So directions, not sizes, are expected to transfer.

* **N1 (primary, GF31 at night):** GF31 is not worse than `union`. PASS iff the worse rule is not met.
* **N2 (secondary, ungated BN-mix at night):** mix16 is worse than `union` by the worse rule. Development lost 0.0195
  at a level of 0.29; scaled to this test's level the loss would be about 0.0035, so we put real probability (about
  one in three) on N2 failing for lack of size rather than of direction. HOLDS iff the worse rule is met.
* **N3 (stated expectation):** GF31's Δ lies in [−0.0050, +0.0050].
* **Descriptive:** each arm's adapted VIS stream against the plain VIS stream; per-system values.

## If a prediction fails

* N1 fails: shallow BN-mix is reported as worse than `union` at night on unseen frames, in the paper's results and in
  Table 23's caveat, and the recommendation for it is revised. No change of constants, rule or frames.
* N2 fails: reported as failed, with its Δ; the development ablation stands as the evidence for the day–night term,
  and the paper says the night test did not reproduce it.
* N3 is reported as held or failed.

## Not done here

No pohang04 frame. No corrupted night frames (the night test registered clean frames only). No change to any constant.
