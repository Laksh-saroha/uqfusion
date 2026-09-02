# Pre-registration — the night label restore (I5)

**Written 2026-09-02, before any label is changed or any weight is trained.**
Committed ahead of the run so the rule is verifiable in git history.

## What is being changed

`filter_night_boxes.py --cut-dark pohang01:100 --execute` emptied **17,502 VIS
TRAIN label files** — all of `pohang01`, which is entirely night — deleting
**132,688 boxes**. `val` and `test` were never touched and are not touched now.
IR labels were never touched (0 `.pre_visfilter` backups under `infrared/`).

The cut was **frame-level**: a dark frame lost every box in it, whatever that
box's own photometric score. `runs/visfilter/box_scores.csv` holds the per-box
audit at the committed thresholds (`--t-int 45 --t-grad 8 --t-contrast 10`,
frame gate `--dark-median 40`):

| population | boxes | fate under the frame-level cut |
|---|---:|---|
| scored, FLAGGED (all three tests fail) | 38,135 | deleted — correctly |
| scored, not flagged | 82,694 | deleted — **not** by its own score |
| never scored (frame median 40–100) | 11,859 | deleted — never tested at all |
| total | 132,688 | |

`audit_night_restore.py` found the flagged and kept populations **overlap** on two
of three axes (`box_mean` flagged p90 4.86 vs kept p10 3.18; `contrast` 2.95 vs
0.37; only `grad` separates), which fires the audit's first branch.

**The change:** restore all 132,688, then re-drop only the 38,135 the per-box test
actually flags. Net **+94,553 boxes** returned to VIS training. Mechanically
`--restore` followed by a per-box `--execute` at the same committed thresholds.
Backups are written once and never clobbered (`filter_night_boxes.py:617`), so
the original pre-filter state survives both operations.

## Why the fused benchmark cannot be the endpoint

`veto_vis` fires on **100% of night frames** (measured, `crossmodal26m`, clean).
At night the shipped system is `ir_only` by switch, not by fusion. **The fused
night number therefore cannot move no matter what the VIS detector learns**, and
scoring this experiment on it would manufacture a null that means nothing.

The endpoint is the **VIS stream alone on night val**: 2,068 frames carrying
16,179 GT boxes, labels never filtered. VIS currently recovers 0.0000 of it.

## The rule

Fixed before the run, not to be edited afterwards:

1. **Training.** Fine-tune from `runs/full_scale/gauss_vis_seed0/weights/best.pt`,
   same `data_vis_stride2.yaml`, same `imgsz` 640, same batch 16, 25 epochs,
   patience 10, into the **new** directory `runs/full_scale/gauss_vis_nightrestore/`.
   Nothing under `runs/full_scale/gauss_vis_seed0/` is written.
2. **Primary endpoint.** VIS-only `mAP@50-95` on the 2,068 night val frames,
   measured with the project's own `map50_95`, against the same metric on the
   existing checkpoint. Three pre-registered bands:
   * **DEAD** — night VIS < 0.005. The filter was right, night VIS is physically
     blind, and the correct architectural move is the audit's second branch: stop
     treating night as a fusion problem and state that it is single-sensor by
     physics rather than by omission.
   * **WEAK** — 0.005 to 0.02. The boxes were learnable but the sensor is
     marginal. Record and do not touch the veto.
   * **ALIVE** — ≥ 0.02. Night VIS is real, and the 100% night veto becomes a
     live question that needs its own pre-registration.
3. **Guard — day must not regress.** Adding 94,553 night boxes can degrade the day
   model. VIS-only `mAP@50-95` on the 1,200 paired day frames is measured for both
   checkpoints as a *paired* delta with its own bootstrap sd. If the day delta is
   worse than `−max(2 × sd_paired, 0.002)`, the restore is **REJECTED** whatever
   the night number does. The absolute floor of 0.002 is included because §4.10
   showed twice that a margin alone degenerates into a sign test.
4. **TEST is not consulted.** No parameter is selected here; this is a single
   pre-specified change, so there is nothing for a held-out split to choose among.
5. **Reported but NOT decision inputs:** the fused benchmark numbers (structurally
   pinned, see above), per-class splits, night `mAP@50`, and the training curves.

## What is explicitly out of scope

Changing `veto_vis`, changing any preset, or re-baselining the headline. The
shipped `crossmodal26m` numbers keep reproducing from
`runs/full_scale/gauss_vis_seed0/`, which this run does not write. An **ALIVE**
verdict licenses a *new pre-registration* about the night veto — not a veto change
in this run.

## The weakness of this design, stated in advance

This is a **fine-tune from a checkpoint trained on empty night labels**, so the
initialisation already encodes "night frames contain nothing". That is a real
prior to unlearn in 25 epochs.

The consequence to accept now: an **ALIVE** verdict is strong evidence, because it
had to overcome that prior. A **DEAD** verdict is **provisional** — it cannot
cleanly distinguish "night VIS is blind" from "25 epochs of fine-tuning could not
undo the initialisation". Settling a DEAD verdict would need a from-scratch run
(~13.7 h at the measured rate), and this document does not authorise one.

## Side effect to record

The VIS train label content hash will change, diverging from the copy on
`dgxanode01` (recorded hash `287b11c50b5a`). The new hash is written to the
manifest and must be quoted before any server run mixes the two.
