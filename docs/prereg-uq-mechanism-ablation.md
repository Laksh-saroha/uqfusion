# Pre-registration — does predicted uncertainty improve the fusion? (R-D1 / F09)

**Written 2026-09-10, before any ablation cache is built or any arm is scored.**
Committed ahead of the run so the rule is verifiable in git history.

## 1. The claim under test

The project is named for uncertainty quantification in sensor fusion. The review's
finding F09 is that the **shipped system does not use predicted uncertainty to make
the fusion decision at all**.

That is now measured, not argued. Reading the resolved context for
`preset="crossmodal"` and `preset="crossmodal26m"`:

| term | value in the shipped preset | consequence |
|---|---|---|
| `mu_d` (both modalities) | `1e9` | the Mahalanobis soft weight is inert |
| `lam` (both modalities) | `0.0` | as above |
| `sigma_weighted` | `False` (default) | box sigma does not move a coordinate |
| `sigma_score_alpha` | `0.0` | box sigma does not move a score |

And the consequence, measured over all four conditions on the paired val set:

| condition | `r_frame_vis` | `r_frame_ir` | `w_vis` | distinct `w_vis` values |
|---|---|---|---|---:|
| clean | 1.0000–1.0000 | 1.0000–1.0000 | 0.9930 | **1** |
| fog | 1.0000–1.0000 | 1.0000–1.0000 | 0.9930 | **1** |
| lowlight | 1.0000–1.0000 | 1.0000–1.0000 | 0.9930 | **1** |
| glare | 1.0000–1.0000 | 1.0000–1.0000 | 0.9930 | **1** |

**The fusion weight is a single constant, 0.9930, on every one of 2,232 frames in
every condition.** Nothing about a frame changes it. What does vary per frame is the
hard veto, which is driven by image statistics (`grad_gini`, `ir_p05`), not by any
predicted uncertainty.

For contrast, `preset="adopted"` — the older system — has `mu_d` 71.07 / 72.52 and
`lam` 1.02 / 0.71, and its `w_vis` genuinely varies (fog 0.0031–0.5041, lowlight
0.0153–0.5782). The soft term was live there and was deliberately disabled in
`crossmodal` because it was **the mechanism of the lowlight/day loss** — the
docstring at `ctx.py` says so and the number is +0.0215 on that cell.

So the honest description of `crossmodal26m` today is: **image-statistic sensor
rejection plus constant-weight aggregation.** Gaussian training may shape the
checkpoints, but shaping a checkpoint is not using predicted uncertainty to decide
the fusion.

**This pre-registration does not test whether that system is good.** It is measured
and it works. It tests one narrower question: *does supplying real predicted
uncertainty to the fusion beat supplying fake uncertainty of the same shape?*

## 2. Why a constant control is not enough, and what the real control is

The obvious control is constant sigma. It is necessary but nearly vacuous:
inverse-variance averaging with all sigmas equal **reduces analytically** to the
score-weighted mean that stock WBF already computes. A tie against constant sigma is
the expected result, not evidence.

**The control that carries the weight is SHUFFLED sigma**: the same multiset of
sigma values, permuted so each is attached to the wrong box. It preserves the
distribution, the scale, the per-frame statistics and the numerical conditioning, and
destroys only the one property under test — that a box's sigma describes *that box*.

If real sigma does not beat shuffled sigma, then predicted uncertainty is not adding
information to the fusion, whatever else it may be doing.

## 3. Arms

Predictions and fusion options are held **fixed**. Only the `sigma_ltrb` array
supplied to `fuse_detections` varies. Boxes, confidences, classes, features, the
gate, the veto, the capability prior, `iou_thr`, and every other constant are
untouched, and the caches are the ones already on disk (`runs/cache_m`).

| arm | sigma supplied | mechanism enabled |
|---|---|---|
| **S0** shipped | — | none (baseline; the current system) |
| **S1** real, coordinates | as cached | `sigma_weighted=True` |
| **S2** constant, coordinates | every sigma = the cache median | `sigma_weighted=True` |
| **S3** shuffled, coordinates | cached values, permuted within frame | `sigma_weighted=True` |
| **S4** shuffled, coordinates | cached values, permuted across the cache | `sigma_weighted=True` |
| **S5** real, score | as cached | `sigma_score_alpha=α*` |
| **S6** constant, score | median | `sigma_score_alpha=α*` |
| **S7** shuffled, score | permuted within frame | `sigma_score_alpha=α*` |

`α*` is fixed at **1.0** before the run. It is not tuned: tuning it on the same cells
we then report would reintroduce exactly the selection problem R-B2 is about. If
α = 1.0 is inert, that is a reported result, not a reason to search.

Both permutations are seeded (`seed=0`) and the permutation is recorded in the run
manifest, so every arm is reproducible bit-for-bit.

**S3 vs S4 is itself informative.** Within-frame shuffling keeps each frame's sigma
scale; across-cache shuffling does not. If S1 beats S3 but not S4, what the sigma
carries is a between-frame quality signal, not a per-box one — a different and
weaker claim, and one worth being able to state separately.

## 4. Cells and metric

Scored on the paired val set through `run_systems`, all four conditions
(`clean`, `fog`, `lowlight`, `glare`), reporting **`gated_fusion` ship AP
(`local_ap50_95`)** — the metric every comparable table in this project uses.

Naive 0.5/0.5 fusion is carried as the fixed-weight control it has always been and
must not be touched by any arm.

## 5. Decision rule, fixed before the numbers exist

The paired 2σ noise floor is **0.0014–0.0031**
(`runs/eval/metric_noise_floor.md`). Per `project-gate-magnitude-floor`, a measured
margin alone degenerates to a sign test at these magnitudes, so both a statistical
and an absolute criterion must be met.

**UQ is judged to improve the fusion only if, on at least three of the four
conditions:**

1. `AP(S1) − AP(S3) ≥ 0.0031` (the upper bound of the floor), **and**
2. the paired moving-block bootstrap CI for that delta excludes zero at n_boot 1000,
   using `blockboot` — not the IID bootstrap, because the frames are 10 Hz video
   (R-A3, 1.95× interval inflation).

The same rule is applied independently to the score path (S5 vs S7).

**Counts are reported at three floors — 0.0014, 0.0031, 0.0100 — not just the
adopted one**, so a verdict that hinges on the floor choice is visible as such.

## 6. Outcomes, all three declared in advance

* **POSITIVE** — the rule above is met. The claim "predicted uncertainty improves the
  fusion" is supported for the specific mechanism that won, on this recording, and
  the shipped preset should be re-examined for adopting it.
* **NULL** — S1 ties S3 within the floor. Then predicted uncertainty adds no
  information to the fusion decision here, and the thesis claim must narrow to what
  is actually demonstrated: image-statistic sensor selection, with the Gaussian head
  evaluated on its own terms (localization-error scale quality) rather than as a
  fusion input. **This is the outcome the current evidence points to, and it is a
  publishable result, not a failure.**
* **NEGATIVE** — S1 is worse than S3 beyond the floor. Then the sigmas are
  anti-informative for fusion and the reason has to be found before anything else is
  claimed.

**There is no fourth run.** If the result is NULL, the response is to narrow the
claim, not to search for a mechanism or a threshold that makes it positive. Any
follow-up requires a new pre-registration naming what changed and why.

## 7. What this cannot settle

* **It is scored on development data.** Every frame here has been inspected
  (`docs/exposure-ledger-2026-09-09.md`); there is no untouched test set and there
  never was. A positive result would be a development-set result and must be labelled
  one. This is Q1's subject and it is not resolved by this run.
* **It tests the fusion, not the sigmas.** A NULL result says the sigmas do not help
  *this* fusion; it does not say they are badly calibrated. That is the UQ table's
  question (Q3) and it is measured separately.
* **One recording.** Four runs of one harbour dataset, 10 Hz, with the temporal
  dependence R-A3 quantified. No claim generalises beyond it.
* **`crossmodal` only.** `adopted` has a live Mahalanobis term and is a different
  system; it is not an arm here.

## 8. Provenance

Caches: `runs/cache_m/gauss_{vis,ir}_paired_*.pkl`, unchanged. Context:
`preset="crossmodal"`, `cache_dir="runs/cache_m"`. Every arm records the full
`eval.identity.system_identity()` block including the permutation seed, and the
report is written to a NEW filename per project rule.
