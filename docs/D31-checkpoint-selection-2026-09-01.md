# D31 — checkpoint selection stays best-epoch, and both estimators get reported

**Decision (2026-09-01).** The reported checkpoint for every UQ arm remains
Ultralytics' `best.pt`, i.e. **the epoch with the highest validation mAP@50-95**.
The 10-epoch mean is computed and reported alongside it. No arm is ranked on
mAP alone.

## Why the question came up

`max` over 10 noisy validation scores is a selection, and it pays the largest
bonus to the noisiest run. Measured on the two runs that share a machine, a seed,
and a config:

| run | epoch sd | best | best vs its own neighbours | slope |
|---|---:|---:|---:|---:|
| `ens_vis_seed0_ft_control` | 0.00350 | 0.25031 (ep6) | +0.0032 | -0.0002/ep |
| `mc_vis_seed0_ft_refit` | **0.01056** | 0.25988 (ep4) | **+0.0192** | **-0.0022/ep** |

MC's epoch-4 peak stands +0.0192 above both neighbours (0.23961, 0.24175) — about
2 sigma of its own epoch noise — and the run then declines monotonically to
0.22584. So MC reads **+0.0096 better** than the control by best-epoch and
**-0.0084 worse** by the 10-epoch mean. Same run, opposite conclusion, and the
gap is mostly a function of which arm happens to be noisier.

This is not MC-specific. Within-run epoch sd across all arms is 0.0035-0.0106,
while the between-seed sd of best-fitness is **0.00212** — the selection noise
inside each headline number is 2-5x the spread we would report as its uncertainty.
The sigma-head/ensemble gap of 0.00044 sits well inside it.

## Why best-epoch anyway

1. It is the convention, and deviating from it needs a stronger reason than
   "a different rule flatters a different arm".
2. Every existing server number (sigma-head 0.24750, the five ensemble members
   0.24706 +/- 0.00212) already *is* best-epoch. Switching the rule would mean
   re-deriving all of them, and for the pre-2026-09-01 runs only `best.pt`
   survives on disk — the per-epoch checkpoints are gone.
3. The rule was chosen knowing it favours MC, which is the arm this project would
   otherwise have an interest in flattering. Choosing the rule that helps the
   competitor is the safer direction to be wrong in.

## What the paper must therefore say

Report best-epoch **and** the epoch-mean for every arm, and state that the arms
are indistinguishable on mAP once selection noise is accounted for. Do not present
a ranking. The arm comparison is carried by the calibration metrics (d-ECE, NLL,
AUSE, AURC), where the separations are large relative to their noise.

Stating both is what keeps this honest: the rule was fixed *after* both estimators
had been computed, so concealing either one would be the actual problem.

## Consequence for the running work

`mc_vis_seed0_ft_server` runs with `train_overrides.save_period = 1`, so all ten
of its epoch checkpoints are kept. That makes any future change of rule a
recomputation rather than a retrain, for that run at least.

Supersedes nothing. Related: `docs/uq-arms-compiled-2026-09-01.md` (the measured
arms), D27 (veto-only gate), D28/A-1 (IR nc=1).
