# σ re-ranker replication on the five Phase 3 VIS detectors: FAILS

**Run 2026-10-09 under `docs/prereg-sigma-rerank-replication-2026-10-09.md`** (registered at `e2e66d8`, before any Phase 3 re-ranker number existed). Development data. **No pohang04 frame was scored.** Nothing is adopted.

## Verdict

**FAILS.** σ's paired increment clears 0.0060 with an interval above zero on 2 of 5 seeds, and 0.0047 on 3 of 5. The rule needs 4 of 5 at either floor.

**Descriptively, the sign replicates.** The increment is positive on all five seeds, every interval is above zero, and the mean is +0.0053, against the original +0.0050. What fails is the magnitude. Under the registered consequence (§4 of the prereg), the paper's title and framing revert to 2026-10-08, "informative, not useful". The +0.0050 is reported as a single-checkpoint observation whose registered replication failed.

## Pre-run check

On the original checkpoint (`runs/cache_m`), the registered command reproduced +0.004960 [+0.003762, +0.006392] exactly, and the fixed arm +0.006342 [+0.003042, +0.009411] as well (`runs/eval/rerank_prerun_check_2026-10-09.md`).

## Per seed

The command is as registered: `py -3.13 scripts/fit_rerank.py --cache-dir runs/cache_p3/seed{k} --ir-cache runs/cache_p3/seed{k}/gauss_ir_paired_clean.pkl --cls 0 --n-features 3 4 --block-len 20 --ci-arm 4 yes 0.30 --ci-vs 3 yes 0.30`. Raw output: `runs/eval/rerank_p3_seed{k}_sigma_increment.md` (git-ignored).

The table reports ship AP (local AP), out of fold, over 1,200 paired day frames. The arms are monotone with λ 0.30. Intervals are moving-block, L = 20, n 1000, seed 0.

| seed | VIS baseline | σ increment (4 − 3 features) | ≥ 0.0047 | ≥ 0.0060 | re-ranker (4 features) vs none |
|---|---:|---:|:---:|:---:|---:|
| 0 | 0.3507 | +0.0045 [+0.0023, +0.0080] | no | no | +0.0025 [−0.0031, +0.0068] |
| 1 | 0.3391 | +0.0052 [+0.0033, +0.0071] | yes | no | +0.0054 [+0.0028, +0.0078] |
| 2 | 0.3543 | +0.0065 [+0.0047, +0.0090] | yes | yes | +0.0029 [−0.0001, +0.0057] |
| 3 | 0.3544 | +0.0038 [+0.0026, +0.0053] | no | no | +0.0029 [−0.0007, +0.0067] |
| 4 | 0.3439 | +0.0062 [+0.0033, +0.0091] | yes | yes | +0.0101 [+0.0057, +0.0136] |
| original (`runs/cache_m`, unregistered) | 0.3686 | +0.0050 [+0.0038, +0.0064] | yes | no | +0.0063 [+0.0030, +0.0094] |

Pass counts, descriptive:

| floor | 0 | 0.0031 | 0.0047 | 0.0060 | 0.0100 |
|---|---|---|---|---|---|
| seeds passing | 5/5 | 5/5 | 3/5 | 2/5 | 0/5 |

The increment's mean is +0.0053 and its range +0.0038 to +0.0065.

## What this means

* **σ carries a real but small within-detector ranking signal.** Learned jointly with confidence, it adds +0.0038 to +0.0065 ship AP to a VIS re-ranker on every detector tested. It is not reliably beyond the floor the paper uses for its claims, and this paper does not call it useful.
* **The whole re-ranker is weaker on the Phase 3 detectors.** Against no re-ranking, its interval is above zero on 2 of 5 seeds, against the original checkpoint's +0.0063.
* **No re-run, no added seed, no other λ.** The λ sweep each seed's file prints is descriptive. Any follow-up needs its own pre-registration.
