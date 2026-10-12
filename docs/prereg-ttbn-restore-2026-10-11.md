# Pre-registration: N4ae, IR-vouched auto-exposure on dark gated frames, 2026-10-11

Written at 8:40 PM, before any restored-frame detection exists on development draws 942–944 or on pohang04.
(First stamped 8:58 PM by mistake. The time and the glare line below were corrected at 8:47 PM, still before
any restored-frame detection.)
Logged in `docs/exposure-ledger-2026-09-09.md` at the same time. Base method: N4 (`docs/prereg-ttbn-2026-10-11.md`).
Code: `scripts/ttbn_restore.py`.

## Why

N4 adapts the VIS detector's normalisation where VIS reports itself unhealthy and IR vouches that the frame is
in daylight and healthy. A classical image restoration applied on the same gate is the other way to repair the
input. Image-only restorations (`scripts/restore.py`, 2026-10-10) had failed under the shipped gate because that
gate fires on only 9% of fogged day frames. N4's gate fires on all of them.

## What was seen before writing

* **Draw 941, the screen draw; systems 0 and 1; day frames; IR clean; Δ against `union`.**
  * Restorations on N4's gate (`scratchpad screen_restore_gated.py`): auto-exposure on low light
    +0.0953/+0.0965, where N4 gives +0.0635/+0.0767. Dark-channel dehaze on fog +0.0134/+0.0153 and on
    fog_s1 +0.0176/+0.0129, where N4 gives +0.0087/+0.0082 and +0.0097/+0.0061. Bilateral + dehaze lost
    up to 0.0616 on low light.
  * The per-frame split used here (`screen_restore_sel.py`): linear luminance below 0.18 holds on 100% of
    low-light day frames and on 0% of fog, fog_s1 and glare day frames.
  * Merging the restored record with N4's adapted one (plain WBF) was below the restored record alone on
    low light.
  * Glare, under N4g's gate (N4's gate fired on none of draw 941's glare day frames): dehaze gained
    +0.0057/+0.0163 where N4g's adaptation gained +0.0170/+0.0267. Glare lies outside N4ae's gate.
* N4's development verdict and its pohang04 partials (P-T): low light +0.0186 over `union` on pohang04.
* N4c, N4c2, N4g and N4F material, and all gate-free pohang04 numbers.
* No restored-frame detection on draws 942–947 or on pohang04.

Development has no VIS noise, rain or blur condition, but pohang04's cells do. Dehaze on those frames is
untested, so dehaze is not part of the registered method (see the descriptive arm below).

## Method

* **Gate:** N4's (`q_vis < 1` AND IR reads day AND `ir_d2 ≤ 222.7`), unchanged.
* **Dark:** the VIS frame's mean linear luminance over its content rows is below 0.18. Computed on the frame
  as the detector receives it, by `restore.gate_feats`. This is exactly where auto-exposure changes the
  image: its target is 0.18, and it only brightens. It is label-free and seed-free.
* **VIS stream of N4ae**, frame by frame:
  * gated and dark: the record of the auto-exposed frame (`restore.restore(im, "ae", ...)`: linear-light
    gain to a mean of 0.18, at most ×8), detected by the same checkpoint;
  * gated and not dark: N4's adapted record, unchanged;
  * not gated: the plain record.
* **Fusion:** E4 `union` (veto off), as N4. IR is untouched.
* **Precision:** restored-frame detections in fp16 autocast with channels_last, batch 8, as N4's pohang04
  build and the screen.

## Development verdict (draws VIS 942–944 / IR 952–954, five systems, N4's 60 cells)

Δ = `union_n4ae` − `union`, mean over the three draws per system, then over the five systems; 95% t-interval
(df 4), as N4.
* **A-H (no harm):** no cell has Δ ≤ −0.0014 with its interval below zero.
* **A-G (gain):** at least one cell has Δ ≥ +0.0060 with its interval above zero.
* **N4ae passes iff A-H and A-G.**
* Descriptive: `union_n4ae` − `union_ttbn` (N4's development scores), and the share of gated frames that
  are dark.

**Descriptive arm `sel` (development only, no verdict):** as N4ae, except that gated frames that are not dark
get the dark-channel-dehazed frame's record (`restore.restore(im, "dcp", ...)`) instead of N4's adapted one.
Reported against `union` and against N4.

## pohang04 test (only if N4ae passes development)

* **Cells, statistic and replicates:** P-T's (`docs/prereg-ttbn-2026-10-11.md`). These are the registered
  pohang04 cells where N4's gate fires, ship AP as the mean over the five systems and the four draws, and the
  gate-free scoring's 1,000 moving-block replicates.
* Δ = `union_n4ae` − `union`, paired.
* **P-A, REPLICATES** iff no cell has Δ ≤ −0.0014 with its interval below zero AND at least one cell has
  Δ ≥ +0.0060 with its interval above zero. Otherwise DOES NOT REPLICATE.
* Descriptive: Δ against `union_ttbn` (N4) on the same replicates.
* Only dark gated frames need new detections. Every other frame reuses N4's pohang04 records.

No constant here is fitted on development draws 942–944 or on pohang04. N4's verdicts are unchanged.
