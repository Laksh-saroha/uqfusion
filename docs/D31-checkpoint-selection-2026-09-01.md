# D31 — checkpoint selection stays best-epoch; both estimators reported

**Decision (2026-09-01).** The reported checkpoint for every UQ arm remains Ultralytics'
`best.pt` (highest validation mAP@50-95). The 10-epoch mean is computed and reported
alongside. No arm is ranked on mAP alone.

## Why the question came up

`max` over 10 noisy validation scores is a selection, and pays the largest bonus to the
noisiest run. Measured on the two runs sharing a machine, seed, and config:

| run | epoch sd | best | best vs its own neighbours | slope |
|---|---:|---:|---:|---:|
| `ens_vis_seed0_ft_control` | 0.00350 | 0.25031 (ep6) | +0.0032 | -0.0002/ep |
| `mc_vis_seed0_ft_refit` | **0.01056** | 0.25988 (ep4) | **+0.0192** | **-0.0022/ep** |

MC's epoch-4 peak stands +0.0192 above both neighbours (0.23961, 0.24175) — ~2σ of its own
epoch noise — then declines monotonically to 0.22584. So MC reads **+0.0096 better** than
the control by best-epoch and **-0.0084 worse** by 10-epoch mean. Same run, opposite
conclusion, and the gap is mostly a function of which arm is noisier.

Not MC-specific: within-run epoch sd across all arms is 0.0035–0.0106, while between-seed
sd of best-fitness is **0.00212** — selection noise inside each headline number is 2–5× the
spread we'd report as its uncertainty. The sigma-head/ensemble gap of 0.00044 sits well
inside it.

## Why best-epoch anyway

1. It is the convention; deviating needs a stronger reason than "a different rule flatters
   a different arm".
2. Every existing server number (sigma-head 0.24750, five ensemble members 0.24706 ±
   0.00212) already *is* best-epoch. Switching would mean re-deriving all of them, and for
   pre-2026-09-01 runs only `best.pt` survives on disk — per-epoch checkpoints are gone.
3. The rule was chosen knowing it favours MC, the arm this project would otherwise have an
   interest in flattering. Being wrong in the direction that helps the competitor is safer.

## What the paper must say

Report best-epoch **and** epoch-mean for every arm, and state the arms are
indistinguishable on mAP once selection noise is accounted for. No ranking. The arm
comparison is carried by calibration metrics (d-ECE, NLL, AUSE, AURC), where separations
are large relative to their noise. The rule was fixed *after* both estimators were
computed, so concealing either would be the actual problem.

## Consequence for running work

`mc_vis_seed0_ft_server` runs with `train_overrides.save_period = 1`, keeping all ten epoch
checkpoints — a future rule change is a recomputation, not a retrain, for that run.

Supersedes nothing. Related: `docs/uq-arms-compiled-2026-09-01.md`, D27 (veto-only gate),
D28/A-1 (IR nc=1).
