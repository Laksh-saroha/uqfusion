# Phase 2 laptop architecture test — 2026-08-18

Run on the 640 dataset, `yolo26s`, VIS at **stride 5** (19,256 train frames), IR at
stride 2 (11,640). Epochs 100 / patience 20 — full-scale schedule, reduced model.
Batch 24 / workers 8, measured by `scripts/tune_batch.py` on the RTX 4080 Laptop.

## Results so far

| run | epochs | best ep | mAP50-95 | note |
|---|---|---|---|---|
| `gauss_vis_seed0` | 54 | 34 | **0.25049** | σ head, first full-scale Gaussian run |
| `gauss_vis_seed0_ft` | 10 | 8 | 0.25053 | mosaic-off stage, corrected LR |
| `gauss_vis_seed0_ft_hilr` | 10 | 7 | 0.24513 | mosaic-off stage, **wrong** LR — kept as evidence |
| `parity_vis_seed0` | 68 | 48 | **0.26124** | no σ |

**The σ head works.** NLL pinned at 0 for the 5 warm-up epochs, active and monotone
from epoch 6 (0 → −0.169 by epoch 10, −0.143 at the end), while `box_loss` fell
straight through the activation with no discontinuity (1.760 → 1.673 → 1.594 → 0.989).

## Four defects found and fixed

1. **σ weights were silently discarded on every resume.** `DetectionTrainer.get_model`
   builds a plain detector then calls `BaseModel.load`, which intersects the checkpoint
   against a model with no `cv4` yet — so every σ key fell out and the branch restarted
   from init while the detector carried on. Fixed by converting *before* loading.
   Gated by `scripts/smoke_resume.py`, which reproduces the old order to prove the
   hazard is real (it would have moved σ by max |Δ| 0.482).

2. **Early stopping forgot its best epoch on resume.** Ultralytics rebuilds
   `EarlyStopping` and `resume_training` never restores it, so a paused run would train
   past the rule un-paused runs obeyed. Restored from the run's own `results.csv`.

3. **A cosmetic progress write could kill a trainer.** `os.replace` onto `live.json`
   fails with `WinError 5` whenever another process has it open; the dashboard polls at
   0.5 Hz. It killed `gauss_vis_seed0` seven minutes in. Writers now retry then write in
   place, and every queue callback is wrapped so only `PauseRequested` can reach the
   training loop. `smoke_queue.py` now hammers the files concurrently (7.8M reads).

4. **`cv4` initialisation advanced the global RNG**, offsetting every subsequent draw.
   Now built inside `torch.random.fork_rng()`.

Two `TypeError: got multiple values for keyword argument` bugs (`patience`, then
`optimizer`) reached real runs because `smoke_queue.py` used a *thinner* override set
than the queue ships. The smoke now uses the production recipe verbatim.

## OPEN: §12.1 parity is not bit-identical, and the reason is not known

D17 claims the deterministic detector trains bit-identically to its baseline. At the
gradient level it does — `smoke_gaussian_e2e.py` still measures max |Δ| = 0.00e+00 on
fixed inputs. **End-to-end over a training run, it does not.**

Evidence chain, in order:

- σ and parity arms differ from **epoch 1**, inside the warm-up where the NLL weight is
  exactly 0 and σ cannot contribute a gradient. `lr/pg0` matches exactly.
- Training is **fully deterministic**: the same arm run twice gives max |Δ| = 0.000e+00.
  So the difference is systematic, not noise.
- The cause was *partly* RNG: Python's `random` state (which Ultralytics augments with)
  diverged during trainer construction, while numpy and torch matched. Model build,
  conversion and weight load are all clean in isolation — the consuming call was not
  located.
- `reseed_at_train_start` now re-seeds all three generators after the dataloaders exist
  and before the first augmentation draw. **This fixed the data stream**: both arms now
  read byte-identical batches (verified by `train_batch*.jpg` hashes; they differed before).
- **Despite identical data, identical config and deterministic execution, the arms still
  differ by max |Δ| 3.9e-2 in early-epoch losses.** The residual is in the optimisation
  step, not the inputs. Not located.

`scripts/smoke_parity.py` encodes this as a gate. **It is currently RED**, deliberately.

### Ruled out (measured, not argued)

| candidate | verdict | evidence |
|---|---|---|
| Nondeterministic training | **ruled out** | same arm twice, max \|Δ\| = 0.000e+00 |
| Different data / augmentation | **ruled out** (after fix) | `train_batch*.jpg` byte-identical across arms |
| Different config | **ruled out** | `args.yaml` diff empty |
| Wrong assigner topk in our E2E loss | **ruled out** | ours is `tal_topk=7, tal_topk2=1`, matching stock exactly |
| σ gradients leaking via global grad-norm clipping | **ruled out** | cv4 grad norm is exactly 0.000000e+00 through warm-up, so it cannot move the global norm |
| AMP step-skipping differing between arms | **ruled out** | first 6 optimiser steps show identical nan/inf scale-ramp in both arms |

Prime remaining suspect: floating-point non-associativity from the extra ops in the
graph (cuDNN algorithm/reduction-order changes under autocast). Note the regime is
unusually sensitive — measured detector grad norms are 786 and 284 against
`max_norm=10`, so clipping is active on essentially every step, which will amplify any
small numerical difference. Not confirmed.

### Consequence for the measured numbers

The VIS pair reads σ 0.25049 vs parity 0.26124, i.e. −0.01075, which is 2.4x the Phase 1
seed sd and looks like "the σ branch costs 0.011 mAP". **That reading is not supported.**
`parity_vis_seed0` was additionally interrupted and restarted three times during
unrelated debugging, and each resume restarts the augmentation stream; it also trained
68 epochs against 54. §12.1 is **unanswered**, not failed.

### Recommendation

Until the residual is found, §12.1 cannot be a bit-identity claim on this stack. Either
locate the systematic difference in the optimiser step, or re-specify §12.1 as a
multi-seed statistical comparison ("within seed noise") and state the noise floor.
The A4-11 condition can still be met that way, but the wording in D17 ("bit-identically
by construction") overstates what has been demonstrated end-to-end.

## Mosaic (OQ: §7.2 close_mosaic)

`close_mosaic: 10` fires at epoch `epochs - close_mosaic` = 90. Runs early-stop at
26–54. **Mosaic has never been closed in this project, including all 93 Phase 1 rows** —
every model trained entirely on mosaicked frames and was scored on clean ones.

Option (C), a second stage continuing from `best.pt` with `mosaic=0`, was implemented
and measured. Result: **neutral** (+0.00004 mAP50-95, −0.00066 fitness), far below seed
noise. The first attempt lost 0.0054 purely to leaving `optimizer: auto`, which restarts
a fresh schedule at full `lr0`; the corrected stage reconstructs epochs 90→100 of the
schedule (`AdamW`, `lr0 1.67e-4`, `lrf 0.1`).

**Option (C) structurally cannot answer the mosaic question.** Restarting training
reinitialises the EMA with `updates = 0`, so ~3 of 10 epochs are spent recovering from a
bookkeeping artefact. It measures "restart training with mosaic off", not "finish
training with mosaic off". **Option (B)** — a patience-keyed callback inside a single
run, no LR or EMA discontinuity — is the construction that can.

Note the motivation was never mAP: σ is fitted to the spread of whatever distribution it
is shown. The `gauss_vis_seed0` / `_ft` pair is the checkpoint pair to run D-ECE,
interval coverage and NLL over — same detection accuracy, different σ training
distribution.
