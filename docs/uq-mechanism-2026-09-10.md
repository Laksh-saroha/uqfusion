# Does predicted uncertainty improve the fusion? — R-D1 / F09

**Pre-registered** in [`prereg-uq-mechanism-ablation.md`](prereg-uq-mechanism-ablation.md)
(`a8f087c`, amendment 1 `f713961`) **before the ablation was built**. The decision rule
below was fixed there; this report applies it and does not revise it.

**Verdict: NULL on both primary comparisons.** The pre-registration named NULL as the
outcome the existing evidence pointed to and declared it publishable in advance. It is.

Raw artifact: `runs/eval/uq_mechanism_ablation_v2.md`, reproduced in
[`eval/uq_mechanism_ablation_v2.md`](eval/uq_mechanism_ablation_v2.md). Run twice; **every
delta identical to the last digit**.

---

## 1. The finding that made this necessary

Read off the resolved context at run time, for `preset="crossmodal"`:

| term | value | consequence |
|---|---|---|
| `mu_d` (both modalities) | `1e9` | the Mahalanobis soft weight is inert |
| `lam` (both modalities) | `0.0` | as above |
| `sigma_weighted` | `False` | box sigma does not move a coordinate |
| `sigma_score_alpha` | `0.0` | box sigma does not move a score |

`r_frame_vis` and `r_frame_ir` are **exactly 1.0000 on every frame of every condition**,
and `w_vis` is a **single constant 0.9930** across all 2,232 frames × 4 conditions. The
shipped fusion weight does not vary. What varies is the hard veto, which is driven by
image statistics.

So the honest description of the shipped system is **image-statistic sensor rejection
plus constant-weight aggregation**. This ablation asks the one question that follows:
supplied with real predicted uncertainty, does the fusion do better than with fake
uncertainty of the same shape?

## 2. Results

AP is `gated_fusion` ship AP (`local_ap50_95`). Predictions and every fusion option are
fixed; only the `sigma_ltrb` array varies.

| arm | what | clean | fog | lowlight | glare |
|---|---|---:|---:|---:|---:|
| S0 | shipped (no sigma anywhere) | 0.256524 | 0.021794 | 0.029046 | 0.214990 |
| S1 | real sigma, coordinates | 0.256436 | 0.021794 | 0.029045 | 0.214917 |
| S2 | constant sigma, coordinates | 0.256524 | 0.021794 | 0.029046 | 0.214990 |
| S3 | shuffled in frame, coordinates | 0.257002 | 0.021794 | 0.029047 | 0.214512 |
| S4 | shuffled across cache, coordinates | 0.256286 | 0.021794 | 0.029046 | 0.214937 |
| S5 | real sigma, score | 0.239857 | 0.023389 | 0.026967 | 0.204097 |
| S6 | constant sigma, score | 0.253015 | 0.020309 | 0.025744 | 0.210148 |
| S7 | shuffled in frame, score | 0.244277 | 0.019429 | 0.024972 | 0.203582 |

**S1 − S3, the coordinate path — real sigma vs the same sigmas on the wrong boxes:**

| condition | delta | 95% CI (block L=20) | excludes zero |
|---|---:|---|---|
| clean | **−0.000566** | [−0.000893, −0.000041] | **yes — and negative** |
| fog | +0.000000 | [+0.000000, +0.000000] | no |
| lowlight | −0.000001 | [−0.000004, +0.000008] | no |
| glare | +0.000406 | [−0.000123, +0.000764] | no |

Conditions meeting both criteria at the adopted floor of 0.0060: **0 of 4** (needs 3).
At every reported floor — 0.0014, 0.0031, 0.0060, 0.0100 — the count is **0 of 4**, so
the verdict does not hinge on the floor choice.

**S5 − S7, the score path:** fog +0.003960 [+0.002691, +0.005183] and lowlight
+0.001995 [+0.000944, +0.002684] both exclude zero and favour real sigma; clean
−0.004420 and glare +0.000514 span zero. **0 of 4** at 0.0060. Also NULL — but see §4,
because these two positives are not measuring the fusion.

## 3. The most direct number in the table

**S1 − S0**, real inverse-variance weighting against the shipped system: clean
−0.000089, fog +0.000000, lowlight −0.000000, glare −0.000073.

Turning on the mechanism this project is named for, with real uncertainties, changes
essentially nothing. That is not a subtle statistical call.

**S1 − S2**, real vs constant sigma, is likewise ≤ 0.000089 everywhere — as the
pre-registration predicted analytically, since inverse-variance averaging with equal
sigmas reduces to the score-weighted mean stock WBF already computes. The
pre-registration called this control "nearly vacuous" before seeing it, and it was.

## 4. What the two score-path positives actually measure — and it is not fusion

Fog produced `delta = 0.000000` with a **zero-width CI** on every coordinate comparison.
That is not a tie; it is a vacuum. Measured veto rates:

| condition | VIS vetoed | both streams alive |
|---|---:|---:|
| clean | 46.2% | 53.8% |
| **fog** | **100.0%** | **0.0%** |
| lowlight | 46.2% | 53.8% |
| glare | 46.2% | 53.8% |

Per run: `pohang01` (all 1,032 night frames) is **100% vetoed**; `pohang00`, `pohang02`
and `pohang03` are **0% vetoed**. So 46.2% is exactly the night run, and **on fog VIS is
vetoed on every frame** — no two-stream fusion occurs there at all.

`sigma_score_alpha` is nonetheless applied inside the `single_passthrough` branch of
`fuse_detections`, which returns the surviving stream after re-scoring it by
`(median_u / u) ** alpha`. Therefore:

* the **coordinate** path acts only where two streams cluster — **1,200 day frames**, and
  **zero frames on fog**;
* the **score** path acts on every frame, including single-stream ones, where it is
  re-ranking one detector's own detections.

**The two positives sit exactly where no fusion happens.** Fog is 100% single-stream, and
lowlight's fusing frames are the ones where the coordinate path measured −0.000001. So
the only place real sigma shows a consistent positive sign is **within-stream
re-ranking**, which is a different and much weaker claim than sensor fusion — and it is
still below the floor.

Read together: **where uncertainty could influence the fusion it does nothing; where it
shows any signal it is not fusing anything.**

## 5. Two results that point the wrong way

* **S1 − S3 on clean is −0.000566 with a CI excluding zero.** Shuffled sigma beats real
  sigma. At ten times below the floor this is a real *sign*, not a real *effect*, and it
  is reported because suppressing an inconvenient significant sign is exactly the
  practice this backlog exists to correct.
* **S5 − S6 on clean is −0.013157** [−0.022630, −0.000678], excluding zero — the largest
  magnitude anywhere in the table. Real-sigma score re-ranking is **worse than constant**
  on clean, and S5 costs 0.0167 against the shipped system (0.239857 vs 0.256524).
  Sigma-based re-scoring is actively harmful on the clean cell.

## 6. A defect in my own decision rule

The pre-registration required 3 of 4 conditions. With fog structurally incapable of
showing a coordinate-path effect, a POSITIVE verdict would have needed **all three**
informative conditions. That is stricter than intended.

It biases **against** POSITIVE, so it cannot have manufactured this NULL — but it is a
flaw, and recording it now is cheaper than discovering it later. **Any follow-up
pre-registration must count only conditions where fusion actually occurs.**

## 7. What this does and does not establish

**Does:** on the day frames of clean, lowlight and glare — every frame where two-stream
fusion actually happens — real predicted uncertainty does not beat shuffled uncertainty
in the fusion, at any floor, on either mechanism.

**Does not:**

* **Nothing about fog**, which could not be tested.
* **Nothing about whether the sigmas are well calibrated.** A sigma can be an excellent
  localization-error scale and still be useless to *this* fusion, whose weights are
  constant and whose clusters are rare (at `iou_thr` 0.85 only ~0.05% of VIS boxes have
  an IR partner — `project-fusion-mechanism`). That is Q3's question, measured separately.
* **Nothing generalisable.** Development data, one recording, 10 Hz, with the temporal
  dependence R-A3 quantified. Every frame here has been inspected
  (`exposure-ledger-2026-09-09.md`).
* **Nothing about `adopted`**, which has a live Mahalanobis term and is a different system.

## 8. What follows

Per the pre-registration: **there is no fourth run.** The response to NULL is to narrow
the claim, not to search for a threshold that reverses it.

Concretely, the thesis claim should become **image-statistic sensor selection**, which is
measured, works, and is the actual mechanism — with the Gaussian head evaluated on its
own terms as a learned conditional localization-error scale (R-C3's renaming), not as a
fusion input. The UQ table then answers "are these uncertainties any good?" (Q3)
independently of "do they improve the fusion?", which this answers: not here.

A follow-up worth pre-registering separately, and **not** run here: the score path's
within-stream re-ranking showed positive signs on both single-stream conditions. That is
a real, narrow, testable claim about uncertainty-aware NMS/ranking on one detector — a
different paper's paragraph, and it needs its own registration with a rule that counts
single-stream frames deliberately rather than by accident.
