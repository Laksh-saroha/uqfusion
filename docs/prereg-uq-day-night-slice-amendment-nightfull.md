# Amendment — U1 re-run on a night-capable VIS detector

**Written 2026-09-04, before the new cache is built.** Committed ahead of the run,
same discipline as `docs/prereg-uq-day-night-slice.md` (`a4f9364`).

## Why this is an amendment, not a re-run

`docs/prereg-uq-day-night-slice.md` (U1) scored the VIS sigma-head arm from
`runs/full_scale/gauss_vis_seed0_ft/`, a checkpoint trained on the night-emptied
labels. Its Stage A verdict was **VIS CLEAN**, but the reason is hollow: that
checkpoint emits **2 detections across 1,032 night frames**
(`docs/night-veto-axis-closed-2026-09-04.md` §2) — there was never enough night
signal present to be miscalibrated. A detector that cannot see at night cannot
show label contamination at night either; CLEAN measured nothing.

`runs/full_scale/gauss_vis_nightfull/` is trained on the restored night labels
(`b92739202127`) and is the only local VIS checkpoint that detects at night. Asking
U1's question against it is a **different question** — a different detector, per
this project's standing rule that a detector swap needs its own declared line — not
a re-run hoping for a friendlier number. Nothing else about U1 changes.

## What changes, and what does not

* **Changes:** the VIS sigma-head arm's checkpoint —
  `gauss_vis_seed0_ft` → `gauss_vis_nightfull`. New cache, new filename
  (`runs/cache_uqslice/sigma_vis_nightfull.pkl`), built with the identical command
  in U1 rule 1 (`--images-list runs/derived/paired_val_vis.txt --imgsz 640
  --conf 0.001`) so `verify()`'s cross-arm settings check still applies unchanged.
* **Does not change:** the VIS MC-Dropout arm (`mc_vis.pkl`, still
  `mc_vis_seed0_ft_refit` — also trained on emptied night labels, not re-run here),
  the IR control, the five decision metrics, the bootstrap, the magnitude floor, the
  bands, the hard ordering-flip trigger, or the aggregation rule. This is Stage A
  (two VIS arms), same as the original — **it can return CONTAMINATED but not
  CLEAN**, per U1's own staging rule, since only one of the two arms changed.
* **Output:** a new report, `runs/eval/uq_day_night_slice_nightfull.md` — the
  original `runs/eval/uq_day_night_slice.md` is not touched or overwritten.

## Disclosed weakness, carried forward

`gauss_vis_nightfull` is **one seed, early-stopped at epoch 11 of 100** with the
learning rate still annealing (`docs/handoff-2026-09-04.md` §7). Any verdict from
this run inherits that: it asks whether the *hollow-CLEAN* reason has evaporated,
not whether a fully-trained night-capable VIS arm would calibrate well. A
CONTAMINATED or SUSPECT verdict here is real evidence the question is live; a CLEAN
verdict would still leave "at full training" untested.

## Consequences, fixed in advance

Same as U1's: **no outcome of this document licenses a retrain.** If this reopens
the VIS-ensemble build-vs-drop decision (`docs/handoff-2026-09-04.md` §2), that
decision is made on its own terms — GPU cost, publication scope — not automatically
by this verdict.
