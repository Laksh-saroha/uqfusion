# The three UQ arms, compiled (2026-09-01)

All values recomputed from `results.csv` directly, so the `run_queue.py` off-by-one
on `best_epoch` (fixed 2026-08-24, still present in runs predating the server sync)
does not touch them. VIS = `data_vis_stride2.yaml`, IR = `data_ir_shiponly.yaml`,
10 epochs, seed 0, `_ft` = fine-tune from the arm's own parent.

## 0. What the refit was for

`insert_head_dropout` renumbered the final conv in each head branch, so a training
reload matched state-dict keys by name and silently dropped the 12 output-conv
tensors (62,460 params = 0.28% of the model, 100% of the output layer). Fixed by
making dropout an `nn.Conv2d` subclass, so keys never move.

Confirmed live in this queue: `Transferred 768/768` (VIS) and `828/828` (IR).
The old path transferred 756/768.

IR is the clean before/after, because a run with the deployed head intact survives:

| IR run | best | epoch-mean | dropout placement | note |
|---|---:|---:|---|---|
| `mc_ir_seed0` (parent) | 0.11351 | 0.09589 | one2one | |
| `mc_ir_seed0_ft_broken-20260823` | 0.12871 | 0.12506 | **one2many (discarded)** | correctly named |
| `mc_ir_seed0_ft` (rebuilt) | 0.11894 | 0.11513 | one2one | decapitated |
| `mc_ir_seed0_ft_refit` | **0.12916** | 0.12147 | one2one, `MCDropoutConv2d` | **fixed** |

**Two distinct defects, and they must not be conflated.** The 2026-08-23 defect put
all six dropout layers on `cv2`/`cv3` — the one2many branch YOLO26 computes and then
discards — so dropout never executed at inference, T passes came out bit-identical,
and epistemic variance was exactly zero. That run is useless as an MC arm and the
`_broken-20260823` name is accurate. Its mAP is *high* precisely because it is
effectively a plain detector: its deployed head was never touched by dropout, so the
reload had no shifted keys to drop.

The 2026-08-31 defect is the opposite shape. Placement was correct (one2one), but the
inserted module renumbered the deployed keys, so the training reload dropped the 12
output convs: 0.12973 -> 0.11894, with MC variance working the whole time.

That makes `_broken-20260823` the right reference for *this* repair even though it is
the wrong model for the paper: it is the only IR run whose deployed head loaded whole.
The refit lands +0.0005 from it and +0.0102 above the decapitated rebuild, which is
the decapitation defect measured and closed.

> Verified from the checkpoints, not inferred: dropout modules on one2one vs one2many
> are 0/6 for `_broken-20260823`, 6/0 for the rebuild, 6/0 for the refit.

## 1. The arms (VIS)

The MC arm ran on the laptop; σ-head and ensemble ran on the server. The control
(`ens_vis_seed0_ft_control`) is seed 0 of the ensemble re-run on the laptop, and
exists only to measure that difference:

| | server | laptop (control) | machine term |
|---|---:|---:|---:|
| best-epoch | 0.24711 | 0.25031 | **+0.00320** |
| epoch-mean | 0.24013 | 0.24573 | **+0.00559** |

Both peaked at epoch 6; the trajectories are otherwise unrelated (per-epoch delta
ranges over [-0.0064, +0.0197]). There is no scalar offset that transports a
trajectory — only the summary statistics can be anchored.

Anchored to server-equivalent by subtracting the machine term from the MC arm:

**By best-epoch (the record's convention):**

| arm | server-equiv | vs ensemble |
|---|---:|---:|
| MC-Dropout | 0.25668 | **+4.53 sd** |
| σ-head | 0.24750 | +0.21 sd |
| deep ensemble (n=5) | 0.24706 ± 0.00212 | — |

**By epoch-mean:**

| arm | server-equiv | vs ensemble |
|---|---:|---:|
| deep ensemble (n=5) | 0.23826 ± 0.00219 | — |
| σ-head | 0.23563 | −1.20 sd |
| MC-Dropout | 0.23170 | **−3.00 sd** |

**The ranking inverts.** MC is either the best arm by 4.5 sd or the worst by 3.0 sd
depending on the estimator, and nothing else about the run changes.

## 2. Why the epoch-mean is the one to believe

`best.pt` is `max` over 10 noisy validations, so it rewards the noisiest run:

| run | epoch sd | best-epoch peak vs its own neighbours | slope |
|---|---:|---:|---:|
| ens control | 0.00350 | +0.00315 | −0.0002/ep |
| σ-head | 0.00635 | — | +0.0003/ep |
| ensemble (mean of 5) | 0.00654 | — | — |
| **MC refit** | **0.01056** | **+0.01920** | **−0.0022/ep** |

MC's epoch-4 peak stands +0.0192 above both neighbours — ~2σ of its own noise, a
single-epoch spike — after which it declines monotonically to 0.22584. Its slope is
10× the control's. The mechanism is straightforward: p=0.15 dropout in the head is a
real regularizer, and fine-tuning under it degrades the deterministic (disarmed)
detector. So the defensible claim is that **MC-Dropout costs accuracy**, which is the
opposite of what best-fitness reports.

## 3. The selection-noise problem, which is not MC-specific

Within-run epoch sd is **0.0035–0.0106**. Between-seed sd of best-fitness is
**0.00212**. The noise inside each number is 2–5× the spread we would report as
uncertainty across seeds. Every arm in the record is a `max` over epochs, so every
arm carries this. The σ-head/ensemble gap of 0.00044 is far inside it.

## 4. Decision (D31, 2026-09-01)

**Checkpoint selection stays best-epoch**, and the epoch-mean is reported beside it
for every arm. Full rationale in `docs/D31-checkpoint-selection-2026-09-01.md`; the
short version is that best-epoch is the convention, every existing server number
already is one, and for the pre-2026-09-01 runs `best.pt` is the only checkpoint
still on disk. It was chosen knowing it flatters MC — the arm this project has the
least interest in flattering — which is the safer direction to be wrong in.

**The arms are not ranked on mAP.** On the estimator that survives scrutiny the
sigma-head and the ensemble are indistinguishable (-1.2 sd, n=1 against n=5) and MC
is behind; on best-epoch MC leads. Reporting one of those and not the other would be
the actual dishonesty, so the paper reports both and draws no ranking. The arm
comparison is carried by the calibration metrics (d-ECE, NLL, AUSE, AURC), where the
separations are large relative to their noise.

`mc_vis_seed0_ft_server` runs with `save_period = 1`, so all ten of its epoch
checkpoints are kept and any future change of rule is a recomputation rather than a
retrain.
