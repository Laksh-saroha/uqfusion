# Pre-registration: N4 on top of `full_tta` (N4F), 2026-10-11

Written (5:28 PM) before any N4F detection exists. Logged in `docs/exposure-ledger-2026-09-09.md` at the same time.
Base methods:
* `full_tta`: E8 on E5, `docs/prereg-no-retrain-2026-10-10.md`. Development verdict on draws 945–947; on
  pohang04, registered as P-F of `docs/prereg-p04-heldout-gatefree-2026-10-11.md`, still being scored.
* N4: `docs/prereg-ttbn-2026-10-11.md`. Passed development; pohang04 test still being scored.

Code: `scripts/ttbn_fulltta.py`.

## Method (`full_tta_ttbn`)

`full_tta` exactly (no veto, `full_tta`'s development capability prior), except on frames where N4's gate
fires (`q_vis < 1` & IR reads day & `ir_d2 ≤ 222.7`). There, each of the ten VIS views is replaced by its
BN-mix-adapted record:
* each checkpoint's unflipped view, by N4's adapted record (N = 16, blocks of 8);
* each checkpoint's flipped view, by the record of the same BN-mix applied to the flipped block (the
  same 8 frames, each mirrored), mapped back to the original frame.

The views are then merged as `full_tta` merges them (plain WBF, IoU 0.55). IR is unchanged.

**Precision.** The adapted views are computed in fp16 with fused statistics, as N4's pohang04 build. N4's
fp16 development check priced this: `union_ttbn` moved by at most 0.0010. The unadapted views are the
existing caches. The Δ below therefore includes a precision term of that order, disclosed rather than
corrected.

## Development (draws VIS 945–947 / IR 955–957)

These are the draws on which `full_tta` was scored for the no-retrain verdict; their flipped and restored
caches exist only there. This is their second use, for a different question.

**This departs from the ledger entry of 7:25 AM**, which declared these draws spent: "a further registration
needs draws no system here has been scored on."
* Why: building fresh draws for `full_tta` (flipped and restored caches for every checkpoint and condition)
  costs hours of GPU.
* What it costs: N4F was not tuned on these draws, but `full_tta` was judged on them. A pass here is
  therefore weaker evidence than a pass on unseen draws, and the pohang04 test below is the deciding
  one. N4's own development test, on 942–944, was held to the same "never scored with this method"
  standard.

* **Cells:** N4's 60 (VIS clean, fog, fog_s1, lowlight, glare × IR clean, fog_s1–s3, noise_s2–s3 × day,
  night).
* **Statistic:** ship AP. Δ = `full_tta_ttbn` − `full_tta`, mean over the three draws. Both arms are
  ensembles of all five checkpoints, identical for every system, so there is no system-level spread.
* **Interval:** 2.5th and 97.5th percentiles of 1,000 moving-block replicates (L = 20, seed 1, blocks
  within runs, `holdout_p04_v2_rescore.resample_weights`). They are drawn over the slice's frames, and one
  resample scores both arms and all three draws.
* **Check:** `full_tta` recomputed here must equal its no-retrain value in every cell
  (`docs/eval/no_retrain/ensembles_fresh.json`) to the fourth decimal. If not, nothing is reported until
  the difference is explained.
* **F-H:** no cell has Δ ≤ −0.0014 with its interval below zero.
* **F-G:** at least one cell has Δ ≥ +0.0060 with its interval above zero.
* N4F passes development iff F-H and F-G.

## pohang04 (only if N4F passes development)

* **Cells:** N4's pohang04 cells. Where the gate fires on no frame, Δ = 0 exactly.
* **Adapted views:**
  * Unflipped: the adapted records N4's pohang04 build already holds.
  * Flipped: new records, built for the same blocks.
* **Statistic and interval:** Δ = `full_tta_ttbn` − `full_tta`, mean over the four draws, on the gate-free
  scoring's 1,000 replicates (`runs/holdout_p04/gatefree/boot_weights.npy`).
* **P-FT, REPLICATES** iff no cell has Δ ≤ −0.0014 with its interval below zero, and at least one cell
  has Δ ≥ +0.0060 with its interval above zero.

Nothing is fitted on pohang04, and no other verdict changes.
