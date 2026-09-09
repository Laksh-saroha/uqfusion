# TODO 2026-09-09 — external architecture review: repair backlog

**Status: OPEN, nothing started.** Source: `architecture-review-2026-09-09.md` (external review,
delivered 2026-09-09, inspected revision `1886258` = our current HEAD). The reviewer worked from a
source-only checkout — no dataset, no working venv — so every finding is source-level, analytical,
or drawn from our own docs. No GPU experiment in it was re-run.

This file is the actionable backlog. It does **not** replace the review; read the review for the
evidence and the reasoning. Task IDs map to the review's finding numbers (`F01`–`F20`).

**Conventions.** `P1` = resolve before using affected results as confirmatory evidence.
`P2` = material but not blocking. Size is `S` (≤1 day), `M` (2–4 days), `L` (>1 week) and is rough
except where grounded in a measured rate. "Confirmed here" means I checked it against the working
tree on 2026-09-09; "reviewer's claim" means it is not yet independently checked.

---

## 0. What I verified before writing this

All six of the load-bearing P1 claims I spot-checked hold at source level.

| Finding | Check | Result |
|---|---|---|
| F03 | `apmetrics.py:74`, `matching.py:169` | Both use `np.interp` linear interpolation of the precision envelope. COCO samples at the first attained recall via `searchsorted`. **Confirmed.** |
| F04 | `apmetrics.py:133` vs `:79` | `presort` freezes `classes` from the *full* parts; `ap_weighted:177` iterates that frozen set; `ap_from_parts:79` derives its class set from the resample. The two paths diverge when a resample drops a class. **Confirmed.** |
| F01 | `fit_structure_gate.py:132` | `ir_thr = (ir_v[ir_fit].max() + ir_v[ir_night].min()) / 2.0` — the night side is in the formula, and the file's own metadata key calls it `heldout_night_min`. **Confirmed.** |
| F02 | `ctx.py:251` | `capability_sel: str \| None = "fit"` is still the default. **Confirmed.** |
| F09 | `ctx.py:557-558` | `replace(c_vis, mu_d=1e9, lam=0.0)` on both streams. **Confirmed** (and already acknowledged in `rebaseline-proposal-2026-09-04.md`). |
| F13 | `audit.py:102`, `:92` | `report["ok"]` flips only on `duplicates` or `temporal_violations`; `missing_splits` is printed at `:111` but never gates the verdict. The proximity check uses `train_like={"train"}` / `eval_like={"val","test"}`, so a crossing requires one side to be `train` — **val-vs-test adjacency is structurally unreachable**. **Confirmed, both halves.** |

Not yet checked here: F06 (clipping coupling — analytical, plausible from the installed trainer),
F07/F08 (estimator semantics — argument, not defect), F10/F11 (geometry), F14 (cache identity),
F18 (queue locking), F19 (timing/env), F20 (system identity).

---

## 1. The reordering: why the evaluator comes first

The review says the AP-convention difference (0.0033–0.0050 on its two toy rankings) does not prove
any real result flips, and it is right that this needs rescoring to settle. But it had no dataset,
so it could not put that number next to ours.

We measured our own noise floor: **real 2σ is 0.0014–0.0031** paired
([`runs/eval/metric_noise_floor.md`](../runs/eval/metric_noise_floor.md)), and several adopted
margins sit far below it — soft-NMS was rejected at **−1.03e-5**
([`runs/eval/snms_gate_draw_avg.md`](../runs/eval/snms_gate_draw_avg.md)) — and the draw-noise work
already established that single-draw gates cannot resolve ±0.001 on cells scoring ~0.027.

A systematic metric bias of the same order as the noise floor, or larger, sits underneath **every**
adoption decision made with that evaluator. That makes F03+F04 not a documentation cleanup but the
gate on whether the existing decision record means anything — and the repair is CPU-only, so it
contends with neither the idle laptop nor the server grid.

**Consequence for sequencing:** workstream A precedes everything. The change-impact table (R-A5) is
the artifact the rest of the backlog waits on.

---

## 2. Workstream A — evaluator and statistics

**P1. Blocks: every table, every adoption decision, the UQ arms comparison, the 93-run grid readout.**

### R-A1 — official-AP parity harness (F03) · P1 · M · **HARNESS DONE 2026-09-09 (`a3182ba`); (a)/(b) call open**
Stand up `pycocotools` as the authoritative evaluator behind an explicit task config (IoU sweep,
area ranges, maxDets, ignore policy, stable score sort). Decide per table whether we (a) replace the
local AP, or (b) keep it for historical continuity under an honest name (`custom_ap_linear_interp`)
and publish an official-AP companion column. Recommendation: (a) for everything forward-looking,
(b) only where re-scoring is impossible.

*Acceptance:* fixtures covering imperfect recall, duplicate recall values, tied scores, missing
classes, no predictions, wrong classes, per-image detection caps, and image resampling. Ultralytics'
own convention is a **third** convention — do not treat "matches Ultralytics" as "matches COCO".

**Built and measured 2026-09-09.** `src/uqfusion/eval/cocoparity.py` runs COCOeval behind an
explicit `TASK_CONFIG` (IoU sweep, one open area range, no crowd/ignore, stable score sort, and
`maxDets` from the data rather than COCO's 100 — a 100-cap would truncate a `conf 0.001` cache and
disguise a detection cap as an interpolation gap). All eight acceptance cases pass in
`scripts/smoke_cocoparity.py`; the detection-cap case was vacuous on first write and was rebuilt so
capping drops AP 0.1667 → 0.0. `matching.py`'s argsorts are pinned stable to match.

**The measurement (`runs/eval/ap_convention_parity.md`, 6 arms × 3 subsets):** local AP reads
systematically LOW by −0.00039 (VIS) to −6e-7 (IR) in absolute terms, but the offset largely cancels
in a delta because both arms share the convention. **Worst delta disagreement 0.00028501 — 5× below
the 0.0014–0.0031 paired 2σ noise floor.** So F03 cannot flip a decision whose margin clears the
floor, and it does *not* rescue decisions made below it (soft-NMS rejected at −1.03e-5 is 20× smaller
than this disagreement). Unexplained and reported as observed: the gap is much larger on day than
night, consistently across arms.

**Open:** the (a) replace / (b) rename-and-companion decision. Recommendation on the measurement
above is **(b)** — deltas are safe at 5× margin, so re-scoring everything buys little, but every
published *absolute* number must name its convention.

### R-A2 — bootstrap fast/reference equivalence (F04) · P1 · S · **DONE 2026-09-09 (`a4cf208`)**
Declare a missing-class policy (drop the class from the macro mean, or score it 0 — the review's
counterexample is 1.0 vs 0.5 on the same resample) and make `presort`/`ap_weighted` and
`ap_from_parts` agree under it. Replace default unstable `argsort` with a specified stable sort so
tie semantics are reproducible.

*Acceptance:* randomized small-fixture equivalence between the fast and literal paths, including
sparse-class resamples.

**Closed 2026-09-09.** Policy declared as `MISSING_CLASS_POLICY = "drop"` (COCO: a class with
no GT is undefined, not zero) and `SORT_KIND = "stable"`, both module constants in
`src/uqfusion/eval/apmetrics.py` with the reasoning in their docstrings. The divergence was
**reproduced before it was fixed** — reference 1.0000 vs fast 0.5000 on a two-class fixture,
the review's own counterexample on our code. `_score` returned 0.0 for an undefined per-class
request and now returns NaN; `bootstrap_delta` excludes those draws and reports
`n_undefined`/`n_effective`. Acceptance met by `smoke_apmetrics.py` sections E (200 randomized
sparse-class draws, with a guard asserting ≥10 actually dropped a class so it cannot pass
vacuously) and F (tie determinism); A–D unchanged and still passing.

**Blast radius: nothing published moves, measured not assumed.** Buoy GT sits in 153 of the
2,232 paired val frames, so P(a draw deletes all of them) = 1.5e-69 — the missing-class bug
bites only on small subsets (per-cell tables, per-run slices, LORO folds). The tie effect is
0.0067 mAP on a deliberately tie-dense synthetic fixture but **2.7e-7 on a real cache**, since
99.9% of 31,110 real confidence values are unique. This does not reduce the case for R-A1/R-A5:
the AP *interpolation convention* (F03) is untouched by this task and is the one with a
plausibly material effect.

### R-A3 — dependence-aware intervals (F04) · P1 · M · **DONE 2026-09-09 (`e2ba483`)**
Frame-level resampling of a 10 Hz recording does not give 1,032 independent observations. Move
headline intervals to **paired contiguous-block** resampling within runs, with a block-length
sensitivity table, and report run-level effects separately. State explicitly that with one night run
the between-run night component **cannot** be estimated.

*Acceptance:* every reported interval names its resampling unit and the randomness it covers. Stop
presenting bootstrap sign-flip fractions as p-values or as posterior probabilities of the hypothesis.

**Closed 2026-09-09.** `src/uqfusion/eval/blockboot.py` adds a paired moving-block bootstrap drawn
within runs, and `describe()` returns the resampling unit and the uncovered randomness with every
interval. `bootstrap_delta` keeps working for continuity, with a docstring that now says its
interval is the too-narrow one; `sign_flip_fraction` replaces `p_sign_flip`, which survives only as
a deprecated alias.

**The number: intervals are ~1.9× too narrow, as a LOWER BOUND**
([`runs/eval/interval_block_sensitivity_v3.md`](../runs/eval/interval_block_sensitivity_v3.md)).
se/se(L=1) is 1.61× at a one-second block and 1.95× (worst 1.99×) at L=20, and the curve is still
rising there. Unlike R-A1/R-A2 this **does** move things: the 0.0014–0.0031 noise floor was itself
computed with the iid bootstrap, so it is understated by the same factor, and every margin defended
as "just outside the CI" needs re-reading.

**Bound on what is measurable here.** `pohang03` holds 117 frames, so blocks past shortest-run/5 = 23
collapse the variance (at L=200 that run admits exactly one block start). Larger rows are shown
marked invalid — the collapse is the evidence for the bound, not a result. Where the inflation levels
off cannot be measured on this dataset.

**Not estimable at any block length:** night is entirely `pohang01`, so the between-night-run
component does not exist in this data. Every night interval is conditional on that one recording.
Training-seed variance is a separate component this does not cover either.

### R-A4 — metric contract audit (F12) · P1 · M
Rename and re-scope the metric suite to what it actually measures:
- confidence-only detection ECE ≠ multidimensional calibration conditional on location/scale;
- error/uncertainty correlation is **ranking**, not magnitude (×100 on every sigma preserves it and
  destroys coverage);
- TP-only NLL/coverage compares different selected subsets across detectors with different recall —
  publish TP counts, recall, and the denominator beside every UQ number;
- AURC as implemented is a 20-point mean to 95% removal with integer rounding, not an exact integral;
- `eval_risk_coverage_fixed_gt.py:14` calls its fixed-full-GT denominator "the standard selective
  risk definition" — standard selective risk conditions on the accepted set. Report **both**, named
  correctly, with a random-rejection control.

Also: cross-stream support can exceed 1 (reviewer's local example: 1.47015), so support scores must
not be presented as probabilities without recalibration.

### R-A5 — **re-score existing caches; change-impact table** (F03/F04/F12) · P1 · M · **FIRST PASS DONE 2026-09-09 (`f8c6b7e`)**
The deliverable everything else waits on. Re-score every cached prediction set under the corrected
evaluator and dependence-aware intervals, then publish: which claimed gains, rankings, and adoption
decisions survive, which move, and which invert. Cover at minimum the veil-veto reprice, the
inherited-constants reprice, the soft-NMS reject, the crossmodal gate cells, and the D27 ablations.

*Blocked on:* R-A1, R-A2, R-A3. *Then unblocks:* everything in C, D, and the UQ table.

**First pass 2026-09-09** — [`runs/eval/change_impact_v3.md`](../runs/eval/change_impact_v3.md).
**16 of 53 findings that were defended by a zero-excluding interval no longer are.** Not refuted —
the effects may be real — but the interval used to defend them no longer supports them.

| claim | cells | still supported | reading |
|---|---:|---:|---|
| `no_veto` ("the veto is load-bearing") | 18 | 12 | stands, down to \|delta\| 0.0037 |
| gated vs `visible_only` | 14 | 10 | **which** ones matters — see below |
| `with_maha` / `no_maha` / `cap_only` | 6 each | 4 each | clean/day cells lost; were razor-thin already |
| `photometric_veto`, veil veto repair | 2, 1 | all | hold |

**The one to read carefully:** the shipped `crossmodal` preset keeps glare (+0.0045 macro, +0.0068
ship) while the older 2026-08-20 D27 system loses it (+0.0022 → [−0.0017, +0.0064]). The recorded
headline *"glare/day beats both single streams (+0.0022 [+0.0004, +0.0042])"* is the **old** system's
number and is now indeterminate; what survives is the shipped system's larger effect.

**Also newly indeterminate in both systems:** gated vs `visible_only` clean/ship. That is the very
instance quoted inside `matching.map50_95`'s docstring — *"ship AP +0.0031 CI [+0.0015, +0.0052], not
spanning zero"* — to show macro dilution hiding a real ship gain. The structural argument is
untouched; the empirical instance cited for it no longer clears the bar.

**Remaining work.** This RE-ADJUDICATES recorded intervals; it does not RE-SCORE. The 1.95× factor
was measured on VIS UQ-arm deltas and transported to fusion cells, which assumes a comparable
dependence structure (same frames, same runs, same cadence — reasonable, but an assumption). A full
re-score through `blockboot.block_bootstrap_delta` on each decision's own caches is mechanical now
and is the right next step for any row in doubt. Three named families cannot be re-adjudicated this
way at all and are excluded rather than fudged: the inherited-constants reprice (a *worse-somewhere*
sign rule), the soft-NMS reject (draw noise, not frame-resampling noise), and the crossmodal tuning
sweeps (selection without intervals — R-B2's territory).

### R-A6 — decision-rule statistics (F16) · P2 · S
The draw-averaged rules mix draw SD, bootstrap uncertainty and fixed floors in quadrature. Those
target different variance components; a quadrature sum is not automatically an interval. A four-draw
SD and the SD of the four-draw mean are different quantities. Specify the estimand first (mean change
over scenes / corruption draws / trained models), then pick the estimator and a practical margin.
This is the formal version of what `project-gate-magnitude-floor` already says.

---

## 3. Workstream B — data, split and label integrity

### R-B1 — gate constants are fit on evaluation night data (F01) · P1 · M
`fit_structure_gate.py:132` sets `ir_thr` from a midpoint that includes the night minimum, and the IR
health fit (`:224`) estimates mean, SDs, covariance and the max-distance bound from **all** clean
paired IR frames including night. Its reported 0% clean outlier rate under the max bound is an
in-sample construction, not a false-alarm measurement. The p99 authority bound has a different
in-sample interpretation and must not be conflated with the max bound.

*Fix:* a genuine calibration set with healthy day **and** night; freeze the gate; score a different
block/run. Store calibration frame IDs alongside every fitted constant. With one night run, use
time-separated calibration/test blocks with guard intervals and limit the claim to this recording.

*Acceptance:* the test manifest has zero intersection with every fitting manifest; changing any test
image leaves the fitted constants byte-identical; the clean false-alarm rate is measured outside the
bound fit.

*Blocked on:* the partition decision (§11 Q1).

### R-B2 — the development set has become the test set (F02) · P1 · M
`capability_sel` defaults to `"fit"`, so the capability prior uses the larger repeatedly-inspected
set unless overridden. The 2026-09-01 log §147 discusses keeping support IoU 0.30 after 0.55 scored
poorly **on TEST** — a test that rejects candidates is participating in selection.

This is independent confirmation of what `project-benchmark-holdout` already records: there was never
held-out day data. New corruption seeds on the same scenes do not supply new scene-level test data,
and later preregistration improves auditability without erasing prior exposure (Cawley & Talbot 2010).

*Fix:* label all repeatedly-inspected substrates as development data; nested leave-one-run-out for
tuning with every fitted component inside the fold; reserve one frozen end-to-end evaluation for a
genuinely untouched release. Change the `capability_sel` default or make it a required argument.

*Acceptance:* the final scoring command cannot fit constants or select checkpoints. Do not describe
night held out from the **gate** as night held out from **detector training** — D6-rev explicitly
permits night training frames.

### R-B3 — annotation releases (F05) · P1 · M
Publish three immutable, separately named annotation releases: **original**, **frame-filtered
historical** (132,688 boxes removed from 17,502 files), **reviewed restoration** (94,553 restored,
38,135 still excluded). The current tree is not the untouched original release.

Audit the remaining 38,135 per-box exclusions by human inspection, stratified by visibility, size and
run. Where an annotation is genuinely unassessable, represent it as an **ignore/censored region**
rather than silently converting it to a negative — deleting a visible object's box while keeping its
image teaches the detector that object is background. This applies equally to any future IR
thermal-crossover label filter.

*Acceptance:* labels cannot change during training (start and end content hash agree); every
checkpoint names its label release; night-cut and restored-label results live in different tables; a
preregistered INCONCLUSIVE is never reported as evidence for the old physical premise.

Already consistent with `project-night-restore-verdict` and
[`prereg-night-label-restore.md`](prereg-night-label-restore.md) — the review adds the ignore-region
recommendation and the three-release requirement.

### R-B4 — split audit hardening (F13) · P1 · S
Confirmed defects: `missing_splits` does not gate `ok`; val-vs-test proximity is unreachable by
construction; runs with no parseable ordinals never enter `by_run`; partial ordinal coverage is a
warning only; the median filename-derived interval is assumed to equal the acquisition period; exact
path/stem matching cannot detect duplicate image *content* under different names.

*Fix:* make the required split policy explicit per operation; fail on missing/empty required splits,
missing timestamps, and proximity across **every** relevant split pair; use real timestamps or an
explicitly supplied period; add sampled content/near-duplicate hashing; audit paired modalities
jointly.

*Acceptance:* adversarial fixtures **fail** — missing test, no ordinals, adjacent val/test blocks,
renamed duplicate images, cross-modal pairs split inconsistently. Emit an audit artifact bound to the
exact data release and training recipe.

### R-B5 — strict GT loading (F14) · P1 · S
`matching.py:29` silently treats a missing label file as an empty (background-only) frame and skips
malformed short lines. A wholly missing label path can therefore evaluate as a valid dataset.
Explicit empty labels are legitimate; missing files are not the same thing.

*Fix:* strict parsing plus a declared allow-empty manifest.

---

## 4. Workstream C — training contract and UQ semantics

### R-C1 — detector parity is not guaranteed by detaching (F06) · P1 · M
`gaussian.py:194` correctly removes the direct gradient paths, but the inherited Ultralytics trainer
(`trainer.py:785`) clips **all** parameters together at norm 10 before the step. The reviewer's
analytical counterexample: a detector gradient of 6 with an independent sigma gradient of 100 becomes
**0.598923** after shared clipping. Shared AMP overflow / step-skipping is a second coupling path.

The 2026-08-18 parity investigation's exclusion of sigma clipping concerns **zero-NLL warm-up**, when
sigma gradients are zero — that does not establish parity once NLL is active, and conversely
active-NLL clipping does not explain the earlier warm-up discrepancy.

*Decide the contract:* either (a) strict parity — isolate sigma optimization including its clipping
and any skip decision, or (b) drop the bit-identity promise and predefine a non-inferiority margin
across matched seeds.

*Acceptance:* compare detector parameters after **full optimizer steps** in warm-up, ramp and active
NLL, with AMP on/off and across resume. Note a zero NLL weight is not a literal freeze — BatchNorm
buffers and weight decay on the sigma branch need separate treatment.

**Remove the "bit-identical by construction" wording from the D17/D26 record until (a) or (b) lands.**

### R-C2 — MC/ensemble sigma is not the Gaussian head's sigma (F07) · P1 · M
`clustering.py` produces a confidence-weighted mean box, an **unweighted** coordinate SD among
matched predictions, a support-penalized confidence, and a box-size sigma for singletons. That is
principally *disagreement among surviving detections*. It omits the within-member variance term:

    Var(Y|x) = E_m[Var(Y|x,m)] + Var_m(E[Y|x,m])

Deep Ensembles combines both; we use deterministic members and report only the second term, then
compare it against the Gaussian head's residual-scale prediction as if they were the same quantity.
Two identical-but-wrong detections give sigma 0 — NaN NLL under the repaired evaluator
([`runs/eval/nll_floor.md`](../runs/eval/nll_floor.md)) and zero coverage. IoU clustering further
conditions the estimate on detection survival, agreement, and the 0.55 association threshold. M=5 is
one ensemble replicate, not five.

*Decide the estimand:* **disagreement ranking** (then evaluate it honestly beside recall and
calibration on matched detection subsets) or **predictive likelihood** (then use probabilistic
members with total-variance aggregation, a declared mixture likelihood, or a residual component fit
on independent calibration data for every arm).

*Acceptance:* report degenerate-sigma frequency, association support, TP counts, recall and common-TP
comparisons beside NLL. Distinguish MC-deterministic, MC-stochastic, member, and ensemble accuracy.
Do not label a failed predictive model "no signal".

**This gates the UQ table** — see §11 Q3.

### R-C3 — name the Gaussian output correctly (F08) · P2 · S
With a fixed mean predictor the NLL optimum is `sigma²(x) = Var(Y|x) + (E[Y|x] − mu_fixed(x))²`, so
systematic detector bias enters the fitted scale. The diagonal four-edge form omits edge dependence
and cannot certify a joint 4D region. Beta-NLL is a training option, not a calibration guarantee, and
the detached-mean design already removes the mean/variance interaction that motivates it.

*Fix:* call it a **learned conditional localization-error scale** unless a stronger decomposition is
tested. Compare beta=0 vs 0.5 on development data; consider post-hoc scale calibration. Report
coverage by edge, object size, modality, day/night and uncertainty quantile. Keep DFL-vs-Gaussian
comparisons inside a DFL-capable model — YOLO12-vs-YOLO26 changes far more than the variance source.

### R-C4 — sigma-branch numerics disclosure · P2 · S
Report log-variance clamp saturation frequency and sensitivity. Stability and non-degenerate sigma
are not calibration.

---

## 5. Workstream D — fusion, geometry, mechanism

### R-D1 — the shipped preset does not test the UQ mechanism (F09) · P1 · M
`ctx.py:557-558` makes the learned Mahalanobis soft weight and the box-sigma soft weight inert;
`sigma_weighted` and sigma-in-score are off by default. `crossmodal26m` is therefore image-statistic
sensor rejection + capability-weighted aggregation + cross-stream score support. Gaussian training
may shape the checkpoints, but that is not *using predicted uncertainty to make the fusion decision*.

*Fix:* separate two claims — (a) localization-uncertainty quality and its compute cost; (b)
image-statistic-guided sensor selection. To claim UQ improves fusion, run an ablation on **fixed
predictions and fixed fusion options**, varying only the supplied uncertainty, with **constant** and
**shuffled** uncertainty controls. Remove dead reference-cache/scorer requirements from the inference
path once compatibility is handled explicitly.

*Acceptance:* a system manifest enumerates every active component; a constant-UQ system is never
presented as evidence for learned uncertainty weighting.

Consistent with what `project-crossmodal-gate` and
[`rebaseline-proposal-2026-09-04.md`](rebaseline-proposal-2026-09-04.md) §58 already say — the review
raises it from a footnote to a claim-validity problem.

### R-D2 — covariance is not transformed with the boxes (F10) · P2 · S
`fusion.py:278` maps IR corners into the VIS plane but then reads the original IR `sigma_ltrb` and
divides by the VIS canvas scale, with a comment asserting the result is in VIS pixels. A scale-2
transform must map 1 px SD to 2 px; rotation, projective mapping and corner-envelope selection need
more than a scalar. The inverse-variance path is off in the default preset, so this does **not**
invalidate default `crossmodal26m` AP — it invalidates any claim that `sigma_weighted` is verified.
The path also returns boxes/scores/classes with **no fused sigma**, against `scope.md:114`.

*Fix:* make coordinate frame and covariance representation explicit in prediction records; propagate
covariance by Jacobian or sampling; define fused uncertainty under a stated dependence model; either
emit tested fused uncertainty or declare it unavailable in the schema.

### R-D3 — signed projective division (F10) · P2 · S
`fusion.py:34` clips the projective denominator to a positive epsilon, so `H` and `−H` — the same
projective map — produce different coordinates. Latent, not currently triggered by the stored
normalized calibration. *Fix:* validate the signed division and reject points near the horizon rather
than silently clipping.

*Acceptance fixtures:* identity, translation, scaling, sign-equivalent `H`, rotation, horizon;
uncertainty transform checked against Monte Carlo samples.

### R-D4 — pairing and correspondence (F11) · P2 · M
Registration residuals are several pixels with ~10 px within-run horizontal swings, and cross-modal
partners at merge IoU 0.85 are rare — `project-fusion-mechanism` records 0.05%. A single homography
is exact only for a plane or pure rotation; parallax with separated cameras makes distant-scene use
an approximation whose error must be measured. Thermal and visible box corners need not mark the same
physical points, so RANSAC on box corners is not automatically sound calibration. `iralign.py:17`
claims the median offset survives a **majority** of wrong correspondences — false in general.

Also: VIS-plane evaluation scores agreement with **VIS annotations**, so IR-only physical objects
count as false positives. That is a legitimate defined task but is not "detect every maritime
obstacle", and a naive union of independently annotated boxes double-counts.

*Fix:* measure residuals by range proxy/size/location/run/timestamp delta; keep a one-to-one
timestamped pair manifest with a declared max time offset; build a small manually reconciled
cross-modal subset with identity and visibility before using union-label AP as a headline; keep
native-plane detector AP and VIS-plane system AP separate.

### R-D5 — abstention contract (F12 / D3) · P2 · S
`ctx.py:846` builds `R_sys_gate` from hand-shaped novelty ratios and exports an abstain flag, but the
system never actually abstains. Ratios in [0,1] are not probabilities and 0.5 is not a calibrated
boundary. Consistent with `project-ir-night-switch-safety` ("abstain is a flag, not an override").
*Fix:* call it an advisory reject flag, or define real consumer behaviour and costs.

---

## 6. Workstream E — system identity, caches, ops

### R-E1 — experiment manifest and cache identity (F14) · P1 · M
`cache.py` stores records + caller metadata + frame count + git HEAD; `load_cache:55` validates
nothing; `fusion_eval.py:185` checks equal **lengths**, not pair identities, so reordered or
cross-substrate caches are accepted. `grid.py:40`'s `split_fingerprint` hashes frame **names**, so a
label edit leaves it unchanged (reviewer reproduced: fingerprint stays `896841440b53`). Git HEAD
misses uncommitted source. The grid's completed-run lookup omits the recipe, so changing epochs,
imgsz, initial weights or overrides can silently skip a different experiment.

*Fix:* one immutable, hashed experiment manifest containing ordered pair IDs, data + annotation
release hashes, checkpoint content hash, preprocessing, dimensions, class map, inference branch,
thresholds, MC seed/T/p, ensemble member hashes, clustering policy, corruption parameters, software
versions, and source revision + dirty hash. Validate on every cache load and every resume.

*Acceptance:* changing any scientifically relevant input changes the experiment ID; reordered pairs,
stale constants, missing labels, mismatched classes and a different checkpoint all fail **before**
metrics run; label hashes at training start/end and at caching/evaluation agree.

Extends the ledger already built (`scripts/label_hash_ledger.py`, `runs/label_hash_ledger.csv`) from
labels to the whole recipe.

### R-E2 — queue ownership and durable state (F18) · P2 · M
`cmd_run:783` writes its PID without acquiring exclusive ownership, so two runners can start on one
directory — exactly the race that cost us the 2026-09-06 session
([`handoff-2026-09-06-redo31.md`](handoff-2026-09-06-redo31.md) §4). `write_json:105` uses a shared
deterministic temp filename and falls back to in-place overwrite, so concurrent writers can lose
updates or tear durable state. Failed/skipped runs are terminal by design, so a restart is not a
retry policy.

*Fix:* exclusive OS-level lock before loading or mutating durable state; a device lease if queues
share a GPU; separate best-effort telemetry from durable events (SQLite, or atomic uniquely-named
writes with recovery); bind state to run/checkpoint IDs; distinguish retryable infrastructure errors
from scientific divergence and deliberate cancellation.

*Acceptance:* a second runner fails **before** training; interrupted writes recover the last valid
state; resumed best-epoch/fitness/checkpoint identities agree. Note the new resume fix (`1886258`)
combines queue-state best-epoch with checkpoint best-fitness — that consistency now matters.

### R-E3 — full-system timing (F19) · P2 · M
`bench/fps.py` times a single model's `predict()`. That is backbone selection, not two-stream system
latency with Gaussian output, image statistics, feature extraction, homography, fusion and MC/ensemble
aggregation. "One pass" means one pass **per detector**.

*Fix:* benchmark synchronized end-to-end pairs on the declared target device — mean/p50/p95/p99,
peak memory, warm-up, batch size, input dims, active options — reporting detector-only and
full-system separately.

### R-E4 — causal filtering (F19) · P2 · S
`hysteresis.py:97` centres a 15-observation window, so future observation 7 changes output 0 — up to
0.7 s lookahead at 10 Hz. This violates the D14 per-frame headline convention. **The cross-modal rule
does not use this filter**, so it applies to the historical/`adopted` protocol only. *Fix:* use a
causal hold/release filter for any streaming claim, or disclose the buffering delay.

### R-E5 — environment reproducibility (F19 / D13) · P2 · S
The reviewer's checkout had a `.venv` pointing at a missing former-user Python and an absent dataset
tree — a local limitation, not evidence our training env is broken, but it does show the environment
is not portable. The promised full `requirements.lock.txt` is absent. Several UQ libraries in
`requirements.txt` appear only in smoke imports while core metrics are locally implemented.

*Fix:* per-platform tested locks recording CPU/CUDA index and runtime/driver versions; move machine
paths into local profiles and validate before launch (extends D30); drop unused research deps from
the minimal eval environment.

*Acceptance:* a fresh machine installs, loads a versioned artifact, runs a small real-data fixture and
reproduces the declared metric.

### R-E6 — one authoritative current architecture (F20) · P1 · M
`load_context:251` defaults to the older `adopted` preset; the `crossmodal26m_snms` comment calls
soft-NMS adopted while [`experiment-log-2026-09-02.md`](experiment-log-2026-09-02.md):1012 says do
not adopt; [`docs/eval/final_system_2026-09-01.md`](eval/final_system_2026-09-01.md) actually freezes
the **August** small-model system despite its filename. Callers can evaluate different systems
unknowingly through defaults, inherited constants and historical cache paths.

*Fix:* one current-architecture document generated from or checked against the R-E1 manifest;
explicit dated IDs for historical systems; callers must choose one. Record active mechanisms,
checkpoints, label release, class map, calibration release, preprocessing, corruption recipe and
evaluator version. Keep historical docs intact with supersession links.

*Acceptance:* two commands claiming to evaluate the same system print the same manifest hash;
incompatible presets/checkpoints/constants fail early; tests cover output-branch execution,
sigma-index alignment, save/load, resume, and disabled-option identity under the pinned Ultralytics
version.

---

## 7. Workstream F — claims, wording, publication

### R-F1 — the broad novelty claim is false (F17) · **P1 for publication** · S
The URF progress report says existing visible/IR fusion is static and never conditioned on live
reliability. [UA-CMDet](https://github.com/SunYM2020/UA-CMDet) (2022) combines uncertainty-aware
cross-modal learning with illumination-aware NMS **at inference**; see also *Uncertainty-Aware
Cross-Modality Fusion for Visible-Infrared Object Detection*, DICTA 2024. Learned attention is not
"static" merely because its parameters are frozen — the attention values depend on the input. A
single-pass Gaussian localization head is also prior art (Gaussian YOLOv3, ICCV 2019).

*Fix:* replace categorical novelty language with a comparison table — domain, sensor types,
uncertainty target, inference-time adaptation, calibration evaluation, registration assumptions,
compute. The defensible contribution is a controlled maritime uncertainty study plus a lightweight
interpretable sensor-selection baseline **including its failure cases**.

**This has the only external deadline in the backlog and does not depend on any repair above.**

### R-F2 — bibliography corrections · P2 · S
D-FINE is ICLR 2025, not arXiv-only. RT-DETR is CVPR 2024. Pohang (2023 sensor dataset) and PoLaRIS
(2024 preprint, ICRA 2025) are **distinct releases** — update the undated tracker. MassMIND's seven
segmentation categories are not our ship/buoy taxonomy and need a documented instance/class mapping.
MIT Marine Perception is visible 12 fps / IR 30 fps, CC BY-NC-SA 4.0 — different rates require
explicit pairing, and the source does not verify our claimed manual annotation subset. SMD has
separate visible/IR material, not synchronized calibrated pairs.

### R-F3 — historical ranking hygiene (F15) · P2 · M
66 pilot rows merged with 27 main rows despite an unrecoverable different pilot split; mixed batch
sizes; the 8.4.7-vs-8.4.90 AP offset; interrupted runs; a resolved directory collision. Pooling
cannot make those conditions identical.

*Fix:* separate pilot and main campaigns, class sets, annotation releases, metric versions and
schedules in all tables; treat the old pooled ranking as exploratory; run a small confirmatory
comparison of the selected model against the strongest relevant controls **on the deployment recipe**.
A non-significant ANOVA (p≈0.45) in the IR closure is not an equivalence test, and "within one seed
SD" is a selection heuristic, not indistinguishability — if equivalence matters, define a tolerable
delta and estimate an interval.

*Acceptance:* each table row resolves to one manifest and a compatible metric; do not describe the
latest 93-cell campaign as complete based on an older campaign's count.

### R-F4 — stop promoting local nulls to universal limits (F16) · P2 · S
Rename these rows to what they tested:

| Current inference | What the experiment permits |
|---|---|
| A logistic/MLP gate upper-bounds the features | One learned baseline with a chosen target, model class, regularization and protocol |
| A GT-informed coordinate-ascent oracle sets the ceiling | A local optimum in a restricted action space; must at least contain the baseline as feasible |
| Temporal support lift ≈1, so tracking cannot help | That feature/threshold/dataset gave little marginal signal; motion, identity and other association rules untested |
| A fixed union's recall bounds all future improvement | It bounds re-ranking of those candidates under that matching rule |
| A single batch/resolution comparison isolates the cause | Seeds, head init, shuffling, schedule and transfer fraction confound it |
| An optional fine-tune checkpoint cannot hurt | It enlarges the selection space on repeatedly-inspected validation data |
| A predetermined AP floor sits above single-seed noise | Requires measured variability for that comparison; preregistration makes a rule auditable, not calibrated |

Keep every negative result — restate them as **conditional** findings.

### R-F5 — factual corrections in the record · P2 · S
See §9.

---

## 8. Decision-register deltas (D1–D31)

Only the entries the review asks us to change. Entries not listed are endorsed as-is.

| D | Change required | Task |
|---|---|---|
| D1 | Non-degradation and calibrated usefulness are conditions, not established properties | R-C1, R-C2, R-C3 |
| D2 | Cite the evidential critique alongside the original; claim no clean epistemic decomposition | R-F2 |
| D3 | An advisory novelty flag is not calibrated abstention; define consumer behaviour and costs | R-D5 |
| D4 | Rename "upper bound" → learned baseline; nested fitting | R-F4 |
| D5 | A formula can still use test data; repeated development selection is still selection | R-B1, R-B2 |
| D6-rev | Audit must test all split boundaries, real timestamps, cross-modal consistency | R-B4 |
| D7 | Keep VIS coordinates; propagate uncertainty; separate VIS-reference AP from physical coverage | R-D2, R-D4 |
| D8 | One-SD ties are not equivalence | R-F3 |
| D9 | Narrow "detector-agnostic" to the implementations actually demonstrated | R-F1 |
| D10 | No cross-dataset generalization claim without a documented independent release | R-F2 |
| D13 | Lock transitive deps and platform builds | R-E5 |
| D14 | Centred filters violate the per-frame headline; document the cross-modal path separately | R-E4 |
| D15 | Early stopping before mosaic closure changes the effective recipe — validate the actual schedule; do not claim full annealing from a 100-epoch maximum | R-F3 |
| D16 | Bind results to content hashes, not YAML paths | R-E1 |
| D17 | **Remove the bit-identity guarantee** until full-step verification | R-C1 |
| D18 | Unavailable on YOLO26's `reg_max=1`; raw DFL spread still needs calibration evidence | R-C3 |
| D19 | Head-only dropout is a restricted approximation; show T-convergence and p-sensitivity | R-C2 |
| D20 | Justify mean/variance weighting, support semantics and the arbitrary singleton scale; report association sensitivity | R-C2 |
| D21 | Official-AP parity; accepted-set vs full-system risk; TP-selection disclosure | R-A1, R-A4 |
| D22 | One ensemble is one replicate; member-0 features are not ensemble-level epistemic evidence | R-C2 |
| D23 | New seeds do not establish scene/family generalization — hash transforms, hold out scenes and families separately | R-B2 |
| D24 | Winner-only replication does not estimate ranking uncertainty | R-F3 |
| D25 | Upsampling prepared 640 images cannot recover native VIS detail | R-F3 |
| D26 | Verify the deployed branch per pinned version rather than assuming future YOLO26 defaults match | R-E6 |
| D27 | Preserve with dated IDs; noncausal filtering, low merge frequency and night-label changes constrain its claims | R-E4, R-E6 |
| D28 | P2 vs P2-feature initialization differs; IR class removal narrows coverage | R-F3 |
| D29 | Also share annotation release, training stage, deployment branch, scoring policy and comparable predictive semantics | R-C2, R-B3 |
| D30 | Move machine paths into local profiles and validate before launch | R-E5 |
| D31 | Select on validation, score **once** on test; best-validation AP is not unbiased; epoch SD is not uncertainty over trained models | R-B2 |

Component-level notes to fold into the architecture doc: three-channel IR expansion does not create
three independent measurements; min–max 16→8-bit is a pixel contract, not radiometry (store raw
values and mapping parameters); prepared-640 irreversibly removes small-object detail; the capability
prior is an empirical ranking prior, not a probability or a sensor characteristic;
multiplicative/min/geometric reliability combination is heuristic until calibrated to a target; IR
dedup NMS means the nominal NMS-free detector now has added NMS — keep a raw-IR control; **"cold
scratch" should read "cold start from COCO" everywhere**; the six corruption families' cell count is
not a maritime operating-frequency distribution; conformal coverage should wait until calibration/test
isolation is fixed, since exchangeability does not survive correlated video.

---

## 9. Factual corrections to make in the record (R-F5)

| Where | Claim | Correction |
|---|---|---|
| `reliability.py` comment | "a product lets a confident OOD score dilute a brightness alarm" | False for reliabilities in [0,1]: `a*b <= min(a,b)`; 0.2×0.8 = 0.16 < 0.2. Justify `min` empirically without the reversed algebra |
| Fusion notes | A failed stream always halves the survivor's WBF scores | Depends on weights, member counts and implementation — cite the actual formula and the tested config |
| Veto rationale | Every retained low-score FP hurts AP | False: after full recall an FP tail can leave AP unchanged (reproduced at AP=1) |
| Fusion notes | `score * weight / sigma²` is the minimum-variance estimator | Overstated — needs unbiased common-target measurements and a dependence model. Call it heuristic precision-weighted fusion |
| `iralign.py:17` | Median offsets tolerate a majority of wrong pairs | False in general; require sufficient inlier consensus and validate adversarial correspondences |
| OOD framing | Small Mahalanobis distance ⇒ healthy sensor | False as an implication — in-distribution sensor failure can be common in training data. Say "feature-space novelty relative to the fitted reference". Formal version of `project-maha-night-blindspot` |
| IR framing | "IR sees through glare"; a reflected target has no thermal signal | Too broad — thermal includes emitted **and reflected** radiation; emissivity, geometry, atmosphere and sensor processing matter |
| IR framing | Thermal crossover = equal physical temperature | Incomplete — it concerns apparent radiance. Do not derive physical temperature from per-frame normalized values |
| IR handoff | ~7.7 "effective bits" proves 8-bit loses nothing | Overstated — it is `log2(span/noise floor)`, not entropy or a task-preservation measurement. Report quantization error and target contrast/SNR against the raw source |
| Corruption notes | Synthetic lowlight is physically impossible sensor output | Too strong — it is an incomplete image-space model. Disclose absent photon/read-noise/exposure modelling |
| Gate notes | Gini/structure statistics are illumination invariant | Only under the exact transform's assumptions; gain invariance need not survive clipping, quantization, offsets or noise |
| Gate notes | Healthy/failed identified at threshold 0.5 | A midpoint of a hand-shaped score is not a calibrated operating point |
| [`handoff-2026-09-06-redo31.md`](handoff-2026-09-06-redo31.md) §3 | "A true zombie shows 0:00 CPU forever; `kill -9` released the CUDA context" | **Wrong as a general Linux account.** A zombie has terminated, retains only status/accounting, cannot execute and cannot be killed again. The observed growing CPU time points to live threads/children or a PID-namespace artefact. The GPU did recover, but the mechanism in that write-up is not established — correct the text and cite `waitpid(2)` |
| Phase 1 record | A single same-seed cross-machine pair identifies a hardware AP offset | Unsupported — hardware, library versions, data order and numerical paths differ. Report the paired difference with its confounds; do not subtract it as a universal correction |
| Smoke records | All Gaussian/MC/ensemble gates green ⇒ all components work | Later documents record failures the aggregate smokes missed. Name each tested contract, revision and result |

---

## 10. Suggested sequence

The review's §4 order, reconciled with what is actually running here.

| Order | Work | Tasks | Resource | Done when |
|---|---|---|---|---|
| 1 | Freeze source, labels, checkpoints, pair lists, caches and metric version into manifests; identify every dataset already used for selection | R-E1, R-E6 | CPU | A readable registry of historical vs current artifacts; no ambiguous "final" |
| 2 | Repair official-AP parity, bootstrap equivalence, strict GT loading, split/pair validation | R-A1, R-A2, R-B4, R-B5 | CPU | Adversarial fixtures fail correctly + a small real-data parity run |
| 3 | **Re-score cached predictions with the corrected evaluator and dependence-aware intervals** | R-A3, R-A5 | CPU | The change-impact table |
| 4 | Choose calibration and final evaluation data; refit gate constants on calibration only | R-B1, R-B2 | CPU | No test contribution to any parameter or checkpoint; out-of-sample clean false-alarm rate |
| 5 | Annotation QA + the comparable restored-label UQ experiment | R-B3, R-C1, R-C2 | GPU | Identical label/recipe manifests across arms; full-step parity or a declared non-inferiority test |
| 6 | Mechanism test with fixed predictions, constant/shuffled UQ controls, simple selector baselines | R-D1 | CPU | Independent evidence that the uncertainty changes and improves the decision |
| 7 | Full-system timing + an independent recording | R-E3, R-D4 | GPU + new data | A latency/accuracy/calibration tradeoff, not a single-pass proxy |
| 8 | Rewrite the architecture doc and the manuscript claims from those artifacts | R-F1, R-F2, R-F3, R-F5 | CPU | One accurate method description with explicit historical limitations |

**R-F1 runs in parallel from day one** — it has the only external deadline and depends on nothing.

Minimal baselines for the repaired fusion comparison: VIS-only, mapped IR-only, a clearly defined
uncertainty-blind aggregate, a simple image-statistic selector, the current support-scoring system,
and any proposed active-UQ variant — with detector predictions and class/geometry policy held fixed.
**Adding backbone families is secondary to making these comparisons trustworthy.**

### What this means for the machines

- **Laptop (idle since 2026-09-09 21:41, all 7 UQ retrain runs done).** Steps 1–3 are CPU-only, so
  the laptop is not the constraint. Do **not** start the UQ table from those 7 runs yet — R-C2 says
  that number has no single interpretation until the estimand is chosen (§11 Q3).
- **Server.** Unchanged: let redo31 finish, then resume the 93-run queue per
  [`handoff-2026-09-06-redo31.md`](handoff-2026-09-06-redo31.md) §6. R-F3 constrains how that grid is
  read — a within-campaign ranking under one budget and recipe, not poolable with the pilot rows.

---

## 11. Open decisions for Laksh

**Q1 — partition (blocks R-B1, R-B2, step 4).** Is any Pohang block genuinely untouched, or do we
accept that everything inspected is development data? If the latter: nested leave-one-run-out for
development plus new data for the frozen evaluation, or a narrowed claim limited to this recording?
My read, consistent with `project-benchmark-holdout`, is that there was never held-out day data — so
this is confirmation, not news, and the honest options are "new recording" or "narrow the claim".

**Q2 — re-score scope (blocks R-A5, step 3).** Full re-score under corrected AP, accepting that some
adoption decisions may not survive — or freeze current results as a named custom-AP variant and use
official AP only going forward? **Recommendation: full re-score.** The change-impact table is worth
more than the decisions it might cost, and §1 shows the bias is the size of our own noise floor.

**Q3 — UQ estimand (blocks R-C2, the UQ table, step 5).** Disagreement ranking, or predictive
likelihood? Ranking is a small honest table from the runs already on disk. Likelihood means
probabilistic members or a calibrated residual component per arm — a retrain.

**Q4 — parity contract (blocks R-C1).** Strict detector parity with isolated sigma optimization, or
drop bit-identity and declare a non-inferiority margin across matched seeds? The second is cheaper
and, given F06, more honest about what the current code guarantees.

---

## 12. Already known — do not re-litigate

The review independently confirms several things our own record already contains. Cite them as
convergent evidence, not new findings:

| Review finding | Our record |
|---|---|
| F02 no genuine holdout | `project-benchmark-holdout`; the `FIT_RUNS`/`TUNE_RUNS`/`TEST_RUNS` split |
| F05 night labels changed the problem | `project-night-restore-verdict`, [`prereg-night-label-restore.md`](prereg-night-label-restore.md), [`night-veto-axis-closed-2026-09-04.md`](night-veto-axis-closed-2026-09-04.md) |
| F09 soft UQ inert in `crossmodal26m` | `project-crossmodal-gate`, [`rebaseline-proposal-2026-09-04.md`](rebaseline-proposal-2026-09-04.md) §58 |
| F11 fusion is concatenation, not consensus | `project-fusion-mechanism` — 0.05% partner rate at IoU 0.85 |
| F12 abstain is advisory | `project-ir-night-switch-safety` |
| F16 gates on unmeasured margins | `project-gate-draw-noise`, `project-gate-magnitude-floor`, [`runs/eval/metric_noise_floor.md`](../runs/eval/metric_noise_floor.md) |
| F18 queue ownership race | [`handoff-2026-09-06-redo31.md`](handoff-2026-09-06-redo31.md) §4 — we wrote the lesson; R-E2 implements it |
| F07 degenerate sigma | [`runs/eval/nll_floor.md`](../runs/eval/nll_floor.md) — the NaN repair landed; R-C2 is the estimator question underneath it |

The genuinely **new** engineering defects are: **F03** (AP convention), **F04** (bootstrap class
drift), **F06** (clipping coupling), **F10** (covariance transform + signed homography), **F13**
(audit holes — both halves confirmed here), **F14** (cache identity), and **F17** (the novelty claim).
