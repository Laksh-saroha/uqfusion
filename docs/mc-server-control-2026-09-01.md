# Same recipe, two machines: the best-epoch estimator flips the answer (2026-09-01)

`mc_vis_seed0_ft_server` on dgxanode01, run to compare against the laptop's
`mc_vis_seed0_ft_refit`. Same seed, same recipe, same data, 10 epochs each.
Server run: started 04:50:42 UTC, finished 07:08:33 UTC, 794.3 s/epoch, status
`done`, queue idle.

Read via the Jupyter Contents API from Chrome (`runs/queue_mc_server/state.json`,
`runs/mc_dropout/mc_vis_seed0_ft_server/results.csv`).

## 1. The two curves

mAP50-95 by epoch:

| epoch | laptop | server | server − laptop |
|---:|---:|---:|---:|
| 1 | 0.24685 | 0.24230 | −0.00455 |
| 2 | 0.23175 | 0.24091 | +0.00916 |
| 3 | 0.23961 | 0.23778 | −0.00183 |
| **4** | **0.25988** | **0.25104** | −0.00884 |
| 5 | 0.24175 | 0.24256 | +0.00081 |
| 6 | 0.23956 | 0.24804 | +0.00848 |
| 7 | 0.22854 | 0.23991 | +0.01137 |
| 8 | 0.23228 | 0.24258 | +0.01030 |
| 9 | 0.22689 | 0.24110 | +0.01421 |
| 10 | 0.22584 | 0.23639 | +0.01055 |

| | laptop | server |
|---|---:|---:|
| best | **0.25988** (ep4) | **0.25104** (ep4) |
| epoch-mean | 0.23730 | **0.24226** |
| epoch sd | **0.01056** | 0.00440 |
| best − mean | **+0.02258** | +0.00878 |
| trend slope | **−0.00216**/epoch | −0.00034/epoch |
| final epoch | 0.22584 | 0.23639 |

## 2. The finding

**The two machines disagree about which is better, and the sign depends entirely
on the estimator.**

- By **best-epoch**, the laptop wins by **+0.00884**.
- By **epoch-mean**, the server wins by **+0.00497**.

Same seed, same recipe, same data. The only thing that changed is the hardware.

The mechanism is visible in the table: **the laptop's epoch-to-epoch scatter is
2.4× the server's** (sd 0.01056 vs 0.00440). `max` over 10 noisy validations
rewards the noisiest run, so the laptop's *selection premium* — best minus mean —
is **+0.02258 against the server's +0.00878, a factor of 2.6**. The laptop does
not train a better model; it produces a wider spread for `max` to pick from.

This is the selection-bias mechanism flagged when the arms were first compiled,
now measured directly rather than argued from first principles.

## 3. There is no stable machine offset

Per-epoch `server − laptop`: mean **+0.00497**, sd **0.00789**, range
**[−0.00884, +0.01421]**, **7 epochs positive / 3 negative**.

The delta is not a constant that can be subtracted out. It is comparable in size
to the between-arm differences the three-arm comparison was trying to resolve.
This confirms the earlier correction to the record — an apparent stable offset
from a single epoch-1 observation was an artifact of n=1.

**Consequence: best-epoch numbers measured on different machines are not
comparable.** The compiled three-arm result had arms trained on different boxes,
which is precisely why this single-machine control was run. It was the right call.

## 4. What reproduced and what did not

**Location reproduced.** Both runs peak at **epoch 4**. Given hardware
nondeterminism (cuDNN autotune, atomic reduction order, different GPU), identical
seeds do *not* guarantee identical trajectories across boxes, so the peak
surviving a hardware change makes it more likely to be a real feature of the
trajectory — the LR schedule and data order — than a lucky validation.

**Magnitude did not.** The peak stands **+0.02258** above the mean on the laptop
and only **+0.00878** on the server. The best-epoch comparison depends entirely
on the magnitude, and the magnitude is the half that failed to reproduce.

This tempers, without overturning, the earlier reading of the ep4 spike as noise:
it is a real bump in the trajectory whose *height* is largely machine-dependent.

**The decay signature also weakened.** The laptop's −0.00216/epoch decline — cited
as evidence the MC arm degrades with training, against the control's
−0.0002/epoch — is **−0.00034/epoch** on the server, an order of magnitude
smaller and essentially flat. That specific piece of evidence for "MC declines"
does not survive the machine change.

## 5. Bearing on the open decisions

**Decision 1 (how to report the three arms).** The recommendation stands and is
now measured rather than argued: **do not rank the arms on best-epoch mAP.** The
estimator alone flips the laptop/server ordering, the machine contributes
±0.008–0.014 per epoch with no stable offset, and the selection premium varies by
2.6× across boxes. Report epoch-mean with the spread, and state that the arms
were not all trained on one machine.

**Decision 2 (best-epoch, settled).** Retaining best-epoch for *checkpoint
selection* is unaffected — that is a within-run choice and this is a
between-run comparability problem. The two questions were correctly separated
in `docs/D31-checkpoint-selection-2026-09-01.md`; this run supports keeping them
separate.

## 6. Caveats

- **n = 1 pair.** One laptop run against one server run. Hardware and
  fixed-seed nondeterminism are confounded and cannot be separated here; what is
  measured is "same seed, different box", which is the operationally relevant
  quantity but not a clean hardware effect.
- 10 epochs is short. `patience` was 20 and never engaged, so neither run
  converged; these are early-training curves and the decay slopes in particular
  should not be extrapolated.
- Only the VIS MC arm was re-run. The ensemble and σ-head arms remain as
  originally measured.
