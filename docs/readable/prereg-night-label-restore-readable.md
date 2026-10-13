# Pre-registration — the night label restore (I5)

**Written 2026-09-02, before any label is changed or any weight trained.**

## What is being changed

`filter_night_boxes.py --cut-dark pohang01:100 --execute` emptied **17,502 VIS TRAIN label
files** — all of `pohang01`, which is entirely night — deleting **132,688 boxes**. `val`
and `test` were never touched and are not touched now. IR labels were never touched (0
`.pre_visfilter` backups under `infrared/`).

The cut was **frame-level**: a dark frame lost every box in it, whatever that box's own
photometric score. `runs/visfilter/box_scores.csv` holds the per-box audit at the committed
thresholds (`--t-int 45 --t-grad 8 --t-contrast 10`, frame gate `--dark-median 40`):

| population | boxes | fate under the frame-level cut |
|---|---:|---|
| scored, FLAGGED (all three tests fail) | 38,135 | deleted — correctly |
| scored, not flagged | 82,694 | deleted — **not** by its own score |
| never scored (frame median 40–100) | 11,859 | deleted — never tested at all |
| total | 132,688 | |

`audit_night_restore.py` found flagged and kept populations **overlap** on two of three
axes (`box_mean` flagged p90 4.86 vs kept p10 3.18; `contrast` 2.95 vs 0.37; only `grad`
separates), firing the audit's first branch.

**The change:** restore all 132,688, then re-drop only the 38,135 the per-box test flags.
Net **+94,553 boxes** back into VIS training. Mechanically `--restore` then a per-box
`--execute` at the same committed thresholds. Backups are written once and never clobbered
(`filter_night_boxes.py:617`), so the original pre-filter state survives both operations.

## Why the fused benchmark cannot be the endpoint

`veto_vis` fires on **100% of night frames** (measured, `crossmodal26m`, clean). At night
the shipped system is `ir_only` by switch, not by fusion, so **the fused night number cannot
move no matter what VIS learns**; scoring on it would manufacture a meaningless null.

Endpoint is the **VIS stream alone on night val**: 2,068 frames, 16,179 GT boxes, labels
never filtered. VIS currently recovers 0.0000 of it.

## The rule (fixed before the run)

1. **Training.** Fine-tune from `runs/full_scale/gauss_vis_seed0/weights/best.pt`, same
   `data_vis_stride2.yaml`, `imgsz` 640, batch 16, 25 epochs, patience 10, into the **new**
   dir `runs/full_scale/gauss_vis_nightrestore/`. Nothing under `gauss_vis_seed0/` is written.
2. **Primary endpoint.** VIS-only `mAP@50-95` on the 2,068 night val frames, via the
   project's own `map50_95`, against the same metric on the existing checkpoint. Bands:
   * **DEAD** — < 0.005. The filter was right, night VIS is physically blind; correct move
     is the audit's second branch — stop treating night as a fusion problem and state it is
     single-sensor by physics, not omission.
   * **WEAK** — 0.005–0.02. Boxes were learnable but the sensor is marginal. Record; do not
     touch the veto.
   * **ALIVE** — ≥ 0.02. Night VIS is real, and the 100% night veto becomes a live question
     needing its own pre-registration.
3. **Guard — day must not regress.** VIS-only `mAP@50-95` on the 1,200 paired day frames
   measured for both checkpoints as a *paired* delta with its own bootstrap sd. If the day
   delta is worse than `−max(2 × sd_paired, 0.002)`, the restore is **REJECTED** whatever
   night does. The 0.002 absolute floor is included because §4.10 showed twice that a
   margin alone degenerates into a sign test.
4. **TEST is not consulted.** No parameter is selected here; a single pre-specified change
   leaves nothing for a held-out split to choose among.
5. **Reported but NOT decision inputs:** fused benchmark numbers (structurally pinned, see
   above), per-class splits, night `mAP@50`, training curves.

## Out of scope

Changing `veto_vis`, changing any preset, or re-baselining the headline. Shipped
`crossmodal26m` numbers keep reproducing from `runs/full_scale/gauss_vis_seed0/`, which this
run does not write. An **ALIVE** verdict licenses a *new pre-registration* about the night
veto — not a veto change in this run.

## The design's weakness, stated in advance

This is a **fine-tune from a checkpoint trained on empty night labels**, so the
initialisation already encodes "night frames contain nothing" — a real prior to unlearn in
25 epochs. Consequence accepted now: **ALIVE** is strong evidence, having overcome that
prior. **DEAD** is **provisional** — it cannot distinguish "night VIS is blind" from "25
epochs could not undo the initialisation". Settling a DEAD verdict needs a from-scratch run
(~13.7 h at the measured rate), which this document does not authorise.

## Side effect to record

The VIS train label content hash will change, diverging from the copy on `dgxanode01`
(recorded hash `287b11c50b5a`). The new hash is written to the manifest and must be quoted
before any server run mixes the two.
