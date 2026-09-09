# Amendment — U1 re-run on a night-capable VIS detector

**Written 2026-09-04, before the new cache is built** — same pre-commit discipline as
`docs/prereg-uq-day-night-slice.md` (`a4f9364`).

## Why an amendment, not a re-run

U1 scored the VIS sigma-head arm from `runs/full_scale/gauss_vis_seed0_ft/`, trained on
night-emptied labels. Its Stage A verdict was **VIS CLEAN**, but hollow: that checkpoint
emits **2 detections across 1,032 night frames**
(`docs/night-veto-axis-closed-2026-09-04.md` §2). A detector that cannot see at night
cannot show label contamination at night either — CLEAN measured nothing.

`runs/full_scale/gauss_vis_nightfull/` is trained on restored night labels
(`b92739202127`) and is the only local VIS checkpoint that detects at night. Asking U1's
question against it is a **different question** — a different detector, per the standing
rule that a detector swap needs its own declared line — not a re-run hoping for a
friendlier number.

## What changes

* **Changes:** VIS sigma-head arm checkpoint `gauss_vis_seed0_ft` → `gauss_vis_nightfull`.
  New cache `runs/cache_uqslice/sigma_vis_nightfull.pkl`, built with the identical U1
  rule-1 command (`--images-list runs/derived/paired_val_vis.txt --imgsz 640 --conf 0.001`)
  so `verify()`'s cross-arm settings check still applies.
* **Unchanged:** VIS MC-Dropout arm (`mc_vis.pkl`, still `mc_vis_seed0_ft_refit` — also
  trained on emptied night labels, not re-run), IR control, the five decision metrics,
  bootstrap, magnitude floor, bands, hard ordering-flip trigger, aggregation rule. This is
  Stage A (two VIS arms) — **it can return CONTAMINATED but not CLEAN**, per U1's staging
  rule, since only one of two arms changed.
* **Output:** new report `runs/eval/uq_day_night_slice_nightfull.md`; the original
  `runs/eval/uq_day_night_slice.md` is not touched.

## Disclosed weakness, carried forward

`gauss_vis_nightfull` is **one seed, early-stopped at epoch 11 of 100** with LR still
annealing (`docs/handoff-2026-09-04.md` §7). Any verdict inherits that: it asks whether
the *hollow-CLEAN* reason has evaporated, not whether a fully-trained night-capable VIS
arm calibrates well. CONTAMINATED or SUSPECT is real evidence the question is live; CLEAN
still leaves "at full training" untested.

## Consequences, fixed in advance

Same as U1's: **no outcome licenses a retrain.** If this reopens the VIS-ensemble
build-vs-drop decision (`docs/handoff-2026-09-04.md` §2), that is decided on its own terms
— GPU cost, publication scope — not automatically by this verdict.
