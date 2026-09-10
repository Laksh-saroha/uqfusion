# Metric contracts — what the UQ suite actually measures

**R-A4, review finding F12.** Written 2026-09-10. The charge is that the metric
suite's short names oversell their computations, in the same way R-A1's
`map50_95` claimed to be COCO and R-B2's `TEST_RUNS` claimed to be a test set.
Six specific claims were made. **All six reproduce**; one of them reproduces with
a different mechanism than the review gives, and one turns out to be a naming
defect over a computation that was already correct and already double-reported.

Everything below was measured on `runs/cache_uqslice/sigma_vis_seed0_nightfull.pkl`
(31,110 detections, 19,133 GT boxes) and on the paired val set through
`run_systems`, **before** any code was changed. Nothing published moves: `aurc`,
`ause`, `d_ece`, `nll`, `interval_ece` and `map50_95` all keep their exact
historical values, verified bit-for-bit after the edits.

---

## 1. What was measured

| # | Review's claim | Verdict | Evidence |
|---|---|---|---|
| 1 | D-ECE conditions on confidence only | **TRUE** | binning is on `conf` alone; nothing else enters |
| 2 | AUSE/AURC are ranking-only | **TRUE** | ×100 on every sigma leaves both bit-identical |
| 3 | NLL / interval-ECE are TP-only | **TRUE** | 13,779 of 31,110 detections (44.29%) |
| 4 | AURC is not an integral | **TRUE** | grid mean 0.4397 vs trapezoid 0.4182, Δ0.0215 |
| 5 | "standard selective-risk definition" is wrong | **TRUE, label only** | both curves were already reported |
| 6 | fused scores exceed 1 | **TRUE, wrong mechanism** | max 1.753204; cause is self-agreement, not support |

## 2. Claim 2 — AUSE and AURC read order, never magnitude

Multiplying **every** sigma in the cache by 100:

| metric | before | after |
|---|---|---|
| `ause` | 0.0747863360 | 0.0747863360 — **bit-identical** |
| `aurc` | 0.4397123625 | 0.4397123625 — **bit-identical** |
| `aurc_trapz` | 0.4182321651 | 0.4182321651 — **bit-identical** |
| `nll` | 4.201611 | 5.208694 |
| `interval_ece` | 0.172533 | 0.188100 |

A monotone rescale is inert by construction — `sparsification` sorts by `u` and
reads risk off the order. So a good AUSE says the uncertainty **orders** errors
well; it is no evidence the sigmas are the right **size**, and an arm can win on
AUSE while its intervals cover nothing. `RANKING_METRICS` in
`src/uqfusion/eval/metrics.py` now records this, and
`scripts/smoke_metric_contracts.py` case 2 pins it — including the control that a
rank-*reversing* map does move the number (0.0630 → 0.3569), without which the
invariance test would prove nothing.

## 3. Claim 3 — the UQ subset, and the denominator that was never published

`gaussian_nll` and `coverage_interval_ece` drop every row whose `err_edges` is
not finite, which is every false positive: an unmatched detection has no GT edge
to have erred against. So both describe the TP subset — **and that subset differs
between detectors.** A higher-recall arm admits harder, later matches its rival
never made, and is then scored on a different population.

On the measured cache: `n_detections` 31,110, `n_tp_detections` 13,779,
`tp_share` 0.4429, `n_gt` 19,133, `recall_tp_over_gt` 0.7202.

`summarize_cache` now publishes all five, plus `uq_subset` and
`dece_conditioning`. Comparing two arms' NLL without them is not a like-for-like
comparison, and until now the denominator was left to be assumed equal.

## 4. Claim 4 — `aurc` is a grid mean, in three separate ways

1. **A plain mean of 20 points, not a trapezoidal integral.** 0.4397 vs 0.4182 on
   the same points — a gap of **0.0215**, roughly 7–15× the 0.0014–0.0031 paired
   noise floor (`runs/eval/metric_noise_floor.md`). Not interchangeable.
2. **Coverage never reaches 0.** `fracs` stops at 0.95, so the curve runs
   retention 1.00 → 0.05 and the deep-rejection tail is absent.
3. **Keep counts are integer-rounded**, so realised coverage is not exactly
   `1 - frac` (max gap 0.000502 on a 997-row subset).

`sparsification` now returns `aurc_trapz` and the realised `coverage` grid beside
the historical `aurc`, so the gap is visible instead of inferred. **The `aurc` key
keeps its exact value** — this is a disclosure, not a re-score.

### 4b. Ties

R-A2's tie argument applies here too: an unstable sort makes tied uncertainties
resolve arbitrarily, and the curve is read off that order. Measured effect on this
cache: **0.000e+00** — `u` is 99.8232% unique. Both argsorts are now
`kind="stable"` for reproducibility, not because it was biting.

## 5. Claim 5 — the label was wrong; the computation was not

`scripts/eval_risk_coverage_fixed_gt.py` described its fixed-GT quantity as *"the
standard selective-risk definition"*. It is not. Standard selective risk
conditions on the **accepted** set, so its denominator shrinks with coverage by
construction. The fixed-GT quantity holds the full-condition GT count regardless
of coverage, making rejected frames permanent misses. Different estimand: *"how
much of all the work got done"* rather than *"how good is the system on what it
accepted"*.

Both are legitimate, and the script **already reported both curves side by side
with a random-rejection control**. Only the name was wrong — but a reader taking
it at face value would have compared these numbers against published
selective-risk figures that condition differently.

The same file also owns a second `aurc()` which really is trapezoidal, over
frame-level mAP risk rather than per-detection 1−IoU. **Two functions, one name,
different quantities.** Both docstrings now say so.

## 6. Claim 6 — scores above 1, and why the stated cause is not the cause

Reproduced through the real `run_systems` path:

| preset | max fused score | detections > 1 | where |
|---|---:|---|---|
| `crossmodal` | **1.753204** | 271 / 422,598 (0.0641%) | clean 149, glare 122 |
| `adopted` | **1.713590** | 226 / 495,497 (0.0456%) | clean 130, glare 96 |

The review attributes this to cross-stream support (its own example: 1.47015).
**Measured, that is not what is happening.** `support_gamma` resolves to **0.0 in
both shipped presets** — read off the resolved context, not the source — so the
term is switched off. And cross-modal agreement *cannot* overflow: with weights
summing to 1, a two-stream cluster scores `w_v·s_v + w_i·s_i ≤ max(s) ≤ 1`.

What overflows is WBF's `k = min(n_models, n_cluster)` counting cluster
**members**. Two overlapping boxes from the *same* stream are priced as a
confirmation, so the score becomes `w_stream · (s1 + s2)`, which passes 1 as soon
as a confident duplicate pair lands on a stream holding nearly all the weight —
and on the clean cell `w_vis` is 0.9930.

**Ablation, not argument.** On the clean cell:

| configuration | max | over 1 |
|---|---:|---:|
| defaults (shipped) | 1.753204 | 149 / 119,393 |
| `consensus_distinct=True` | 0.970650 | **0** |
| `consensus_beta=0.0` | 0.970650 | **0** |

So this is exactly the defect the `consensus_distinct` flag was written to fix —
self-agreement priced as cross-modal agreement — surfacing in the score range.
It is live because that flag is **off by default**.

**Why it has never shown up in a result, and why it still matters.** AP is
rank-based and reads only the ordering, so no published number is affected.
Anything that reads a score as P(object) is a different story: `d_ece` bins into
[0,1], so scores above 1 pile into the top bin against a precision that cannot
exceed 1. **No such reading happens today** — `d_ece` runs only on single-stream
UQ caches, and all of them top out at 0.979101 with `conf_out_of_unit_range = 0`.
This is a latent trap, not a live corruption. `summarize_cache` now counts
out-of-range scores so it cannot become one silently.

Fixing it is a **system change, not a disclosure**: `consensus_distinct=True`
alters fused scores on every frame and would require a re-score under a
pre-registration. Not done here, and deliberately so.

## 7. What changed

* `src/uqfusion/eval/metrics.py` — declared contracts `DECE_CONDITIONING`,
  `UQ_SUBSET`, `AURC_GRID`, `AURC_INTEGRATION`, `RANKING_METRICS`, each carrying
  its measured numbers; `sparsification` gains stable sorts, realised `coverage`
  and `aurc_trapz`; `summarize_cache` publishes the TP denominators and
  `conf_out_of_unit_range`.
* `src/uqfusion/uq/fusion.py` — the out-of-unit-range disclosure, with the
  measured mechanism and the ablation that identifies it.
* `scripts/eval_risk_coverage_fixed_gt.py` — selective-risk label corrected; the
  two-`aurc()` collision named.
* `scripts/smoke_metric_contracts.py` — **new**, 8 cases, always-on acceptance
  gate so none of the above can drift back silently.

Consumers checked: `scripts/evaluate_uq.py` prints a fixed `COLUMNS` list and
`scripts/slice_uq_day_night.py` filters `sparsification` to `("ause", "aurc")`, so
the new keys are additive and no existing report changes shape.

## 8. What this does not fix

* **Claim 6 is disclosed, not repaired.** The adopted system still prices
  same-stream duplicates as confirmations. Repair needs a prereg and a re-score.
* **D-ECE is still confidence-only.** Nothing here makes it conditional on scale,
  class or scene; a detector can score 0 and be badly miscalibrated on small
  objects.
* **`aurc` is still the reported number.** `aurc_trapz` is published beside it,
  but no table has been switched over, because switching would move recorded
  values by ~0.02 with no decision depending on it.
* **The TP-subset problem is reported, not solved.** Publishing the denominators
  makes an unfair NLL comparison visible; it does not make it fair.
