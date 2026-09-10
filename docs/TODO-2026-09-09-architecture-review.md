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

### R-A1 — official-AP parity harness (F03) · P1 · M · **DONE 2026-09-09 (`a3182ba`) + 2026-09-10 (`99a6bc9`)**
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

**Closed 2026-09-10 (`99a6bc9`) as NEITHER (a) nor (b).** Checking the ground before executing my
own (b) recommendation turned up two facts that disqualified it:

* `map50_95` has **322 call sites across 70 files** plus 12 documents, so the rename half of (b) is
  a project-wide edit for a 0.00029 delta effect.
* `cocoparity.TASK_CONFIG` pins `max_dets: None`, not COCO's 100 — deliberately, so a truncating cap
  cannot masquerade as an interpolation gap. **A column labelled "COCO" would therefore be a
  mislabel of exactly the kind this review item is about** (cf. `preset="crossmodal"` naming three
  different systems, `docs/exposure-ledger-2026-09-09.md` §6). Fixing a naming defect by adding a
  second naming defect is not a fix.

Resolved instead by **declaring the convention and pinning the measurement**:

* **(c)** `apmetrics.AP_CONVENTION = "local-linear-interp"`, a third declared policy constant beside
  `MISSING_CLASS_POLICY` and `SORT_KIND`, plus `declared_policies()` for stamping all three into a
  result's config block.
* **(d)** `matching.local_ap50_95` is the honest name; `map50_95` remains a working alias and the
  returned dict keys are unchanged, so **no call site and no recorded number moves**. Its docstring
  had claimed "COCO-style" since the function was written — it is linear interpolation, not a
  first-attained-recall lookup. No deprecation warning: 70 files emitting one per eval is noise, and
  the old name is imprecise rather than wrong.
* **(e)** `cocoparity.PARITY_BOUND = 0.00028501` / `PARITY_TOLERANCE = 1.5`.
  `ap_convention_parity.py` now **fails** above 0.00042751 instead of only printing the number, and
  `smoke_cocoparity.py` case 9 asserts on every run that the trip point stays under the 0.0014 floor,
  that `max_dets` is still `None`, and that the alias stays wired.
* **(f)** `docs/ap-convention-rule-2026-09-10.md` — the standing rule: paired deltas are
  convention-safe, absolute APs are not, and Phase 1 ultralytics numbers (~0.034 between *versions*)
  never share a table with custom fusion AP.

All three smoke suites pass and `preset="crossmodal"` still reproduces
`cap_ir = 0.0023539426113108287` bit-for-bit. **Carried into R-E1 by the same commit:**
`FusionContext` now records `ir_nms` and `cap_ir_scale` — previously local variables applied and
discarded, which is precisely why no artifact could record them.

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

### R-A4 — metric contract audit (F12) · P1 · M · **DONE 2026-09-10 (`9c0f921`)**
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

**Resolved — `docs/metric-contracts-2026-09-10.md`.** All six claims reproduced before any code
moved; nothing published changes. The suite is now DECLARED rather than renamed: `DECE_CONDITIONING`,
`UQ_SUBSET`, `AURC_GRID`, `AURC_INTEGRATION` and `RANKING_METRICS` in `eval/metrics.py` each carry the
number measured for them, `sparsification` returns `aurc_trapz` and the realised coverage grid beside
the historical grid-mean `aurc` (0.4397 vs 0.4182, gap 0.0215 on a 0.0014-0.0031 floor), and
`summarize_cache` publishes `n_gt`, `n_tp_detections`, `tp_share` 0.4429 and `recall_tp_over_gt`
0.7202. `scripts/smoke_metric_contracts.py` pins all of it, always on.

**Claim 6 reproduces but the stated mechanism is wrong, and this matters for the repair.**
`support_gamma` resolves to **0.0 in both shipped presets**, and cross-modal agreement cannot
overflow at all — weights summing to 1 bound a two-stream cluster by `max(s)`. What overflows is
WBF's `k` counting cluster MEMBERS, so two overlapping boxes from the SAME stream are priced as a
confirmation: `w_stream * (s1 + s2)`, with `w_vis = 0.9930` on the clean cell. Ablation:
`consensus_distinct=True` takes the clean cell from max 1.753204 / 149 overflows to 0.970650 / zero.
Recalibrating "support scores" would therefore have fixed nothing. Measured extent: 271/422,598
(0.0641%) under `crossmodal`, 226/495,497 (0.0456%) under `adopted`, clean and glare only.

**Open (deliberately):** claim 6 is disclosed, not repaired — `consensus_distinct=True` changes fused
scores on every frame and needs a prereg and a re-score. `d_ece` never sees the overflow today (all
nine UQ caches top out at 0.979101, `conf_out_of_unit_range = 0`), so this is a latent trap rather
than a live corruption. No table has been switched from `aurc` to `aurc_trapz`, since that moves
recorded values by ~0.02 with no decision depending on it.

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

**REBUILT 2026-09-09 (`6e6bbac`), and the correction matters.** The first pass was built against
`final_system_crossmodal.json` and labelled it the shipped preset. That file is **two revisions
behind** — see the R-B2 entry and ledger §6. Rebuilt against `final_system_crossmodal_v2.json`, the
artifact the current code reproduces (`runs/eval/change_impact_v4.md`, 20 of 74 indeterminate).

**Under the current system all seven `gated vs visible_only` cells SURVIVE**, including glare
(+0.0054 macro, +0.0086 ship) and clean/ship (+0.0052). Both were reported as casualties on the
strength of superseded artifacts and are **not** casualties of the system that ships. The
`matching.map50_95` docstring instance likewise stands under the current system.

The overall shape is unchanged — the smallest cells break first, and `no_veto` still holds on 18 of
26 cells.

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

### R-B2 — the development set has become the test set (F02) · P1 · M · **LABEL + GUARD DONE 2026-09-09 (`38a128c`); nested LORO open**
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

**Done 2026-09-09** — [`docs/exposure-ledger-2026-09-09.md`](exposure-ledger-2026-09-09.md) states,
per run, what it has been used for. `DEVELOPMENT_RUNS` names all three paired day runs as
development data. **Confirmed in code, not inferred: `sel("fit") == sel("day")` is `True`.**

`load_context()` gains `role`. `role="develop"` is the default so all 63 call sites are unaffected
and recorded numbers reproduce bit-for-bit (verified by stashing `ctx.py` — `cap_vis` and `cap_ir`
identical to the last digit). `role="final"` makes `capability_sel` **required** and refuses any
selector spanning the scoring frames; `assert_final_scorable()` then intersects the prior's fit
frames with the frames about to be scored and refuses on overlap. That meets the acceptance
criterion — the final scoring path structurally cannot fit on what it reports. `sel()` also gains
`tune`/`test`/`dev`, which had been constants `sel()` could not resolve.

Making `capability_sel` required *unconditionally* would have broken 56 of 63 call sites and
silently rewritten every recorded number, so it is required exactly where it can do harm.

**Still open, and it is the expensive half:** nested leave-one-run-out with every fitted component
inside the fold. The guard can now refuse a contaminated final score; **there is still no
uncontaminated data to run one on.** The ledger §5 prices the three options.

**Found while doing this, then chased to the end (`6e6bbac`) — see the exposure ledger §6.**
**There is no drift**; every value is deterministic and bit-reproducible, and the "1.83% residual"
in the first write-up was an arithmetic error. What is broken is the **manifest**: three artifacts
all record `"preset": "crossmodal"` with different `cap_ir` (0.0092465 / 0.0094158 / 0.0023539),
because the preset gained `ir_nms=0.70` and `cap_ir_scale=4.0` in `b3d8371` and the config block
records neither. **This is R-E1's case, made concrete.** It also invalidated R-A5's input — see
that entry.

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

### R-B5 — strict GT loading (F14) · P1 · S · **DONE 2026-09-10 (`d10f07a`)**
`matching.py:29` silently treats a missing label file as an empty (background-only) frame and skips
malformed short lines. A wholly missing label path can therefore evaluate as a valid dataset.
Explicit empty labels are legitimate; missing files are not the same thing.

*Fix:* strict parsing plus a declared allow-empty manifest.

**Closed 2026-09-10.** Three declared policies in `eval/matching.py`, same style as R-A2's
`MISSING_CLASS_POLICY` and R-A4's contracts: `MISSING_LABEL_POLICY = "refuse"`,
`EMPTY_LABEL_POLICY = "allow"`, `MALFORMED_LINE_POLICY = "refuse"`. A missing file now raises;
`allow_missing=True` is the declared escape, a parameter rather than a silent default.

**An allow-empty MANIFEST was rejected in favour of a policy constant, on measurement.** The corpus
was scanned before anything changed: **0 missing label files** in 133,140 train+val images across
both modalities, **3,880 legitimately EMPTY** ones (1,550 VIS train, 2,330 IR train, 0 in either val
split), and **0 malformed lines** in 1,008,459 non-blank lines. A manifest naming 3,880 files would
be a list of ordinary data, not an exception register; the empty file is the normal case and the
missing file is the error, which is exactly the distinction the item asks for. Refusing therefore
costs nothing today and exists to catch a path or annotation-release mistake tomorrow.

Malformed lines also refuse: the old `if len(vals) < 5: continue` silently deleted ground-truth
boxes. Reproduced on a fixture -- one good line, one truncated line and one garbage token returned
**one** box and raised nothing. `smoke_apmetrics` section G is the always-on gate; `smoke_phase3`
passes end to end.

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

### R-D1 — the shipped preset does not test the UQ mechanism (F09) · P1 · M · **DONE 2026-09-10 (`76c80bc`) — NULL**
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

**Answered 2026-09-10 — [`uq-mechanism-2026-09-10.md`](uq-mechanism-2026-09-10.md).**
Pre-registered in [`prereg-uq-mechanism-ablation.md`](prereg-uq-mechanism-ablation.md) (`a8f087c`,
amendment 1 `f713961`) **before the ablation existed**; run twice, every delta identical.

**The premise is confirmed, not merely alleged.** Under `crossmodal`, `mu_d` is `1e9` and `lam` is
`0.0` on both modalities, so `r_frame_vis`/`r_frame_ir` are exactly 1.0000 on every frame and
**`w_vis` is a single constant 0.9930** across all 2,232 frames x 4 conditions.

**Verdict NULL on both paths, at every floor.** Coordinate path (real vs the same sigmas on the
wrong boxes): clean **-0.000566** [-0.000893, -0.000041], fog +0.000000, lowlight -0.000001, glare
+0.000406 — **0 of 4** conditions at 0.0014, 0.0031, 0.0060 *and* 0.0100. The most direct number is
**S1-S0 = -0.000089** on clean and ~0 elsewhere: switching on real inverse-variance weighting against
the shipped system changes nothing.

**The score path's two positives are not fusion results.** Measured: `pohang01` (all 1,032 night
frames) is **100% vetoed** and the three day runs 0%, so 46.2% is exactly the night run; and on
**fog VIS is vetoed on every frame**, so no two-stream fusion happens there at all.
`sigma_score_alpha` is applied inside `fuse_detections`' `single_passthrough` branch, so the score
path re-ranks a SINGLE stream's own detections on those frames. Fog +0.003960 and lowlight +0.001995
sit exactly where nothing is fused. **Where uncertainty could influence the fusion it does nothing;
where it shows any signal it is not fusing anything.**

**Two results pointing the wrong way, reported not suppressed:** shuffled beats real on clean with a
CI excluding zero, and real-sigma score re-ranking is **worse than constant** on clean by -0.013157
[-0.022630, -0.000678] — the largest magnitude in the table, costing 0.0167 against the shipped
system.

**A defect in the decision rule, recorded:** requiring 3 of 4 conditions when fog is structurally
incapable of showing a coordinate-path effect meant POSITIVE would have needed all three informative
conditions. It biases against POSITIVE so it cannot have manufactured the NULL, but any follow-up
prereg must count only conditions where fusion occurs.

**Consequence for the claim, per the prereg's own rule (there is no fourth run):** the thesis claim
narrows to **image-statistic sensor selection** — measured, working, and the actual mechanism — with
the Gaussian head evaluated on its own terms as a learned conditional localization-error scale
(R-C3), not as a fusion input. A separate, narrower follow-up worth its own registration: the score
path's within-stream re-ranking showed positive signs on both single-stream conditions.

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

### R-E1 — experiment manifest and cache identity (F14) · P1 · M · **SLICES 1-3 DONE 2026-09-10 (`f3364c9`, `353a521`, `6cce035`)**
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

**Slice 1 (`f3364c9`) — describe.** `eval/identity.py`, `FusionContext.inputs`, and a Provenance
footer on all 29 `write_md` reports. Audit finding: `load_context` takes 26 parameters and **15
reached no attribute at all**.

**Slice 2 — refuse.** `docs/cache-identity-2026-09-10.md`. Four defects reproduced before any code
moved; two closed.

* **D1 closed.** `load_cache` validated nothing — a payload claiming `n_frames` 99999 while holding
  10 records loaded fine, and so did records stripped of every prediction key. It now checks payload
  shape, the `n_frames` claim, per-record `image_path`, per-detection array consistency, and any
  stamped `frames_sha256` / `labels_sha256`. `build_cache` stamps all three.
* **D2 closed.** `evaluate_systems` asserted equal LENGTHS. Measured: a reversed IR cache moves gated
  fusion **-0.025630**, shift-100 **-0.027837**, and a ONE-FRAME shift **-0.000968** — *below* the
  0.0014-0.0031 paired noise floor, so no statistical check could ever have found it. Now refused in
  `evaluate_systems`, `iralign.aligned_homographies` and `LearnedGate.fit`.
* **The pairing key is the dataset's timestamp table, not the frame number.** The first version of
  the check paired on `run/ordinal` equality and refused the *legitimate* caches on 1,396 of 2,232
  frames — it had reproduced the fact `build_pairs.py` documents (16,544 of 28,388 rows carry
  different indices) and mistaken it for a defect. Measured IR-VIS offsets: `pohang00`/`pohang01`
  {0,1}, `pohang02` {0..4}, `pohang03` **-155..+1**. Not a fixed offset anywhere.
* **Disk-wide audit:** of 58 vis/ir cache pairs, **42 aligned, 0 mismatched**; the 16 that report
  mismatches are the `gauss_*_train_clean.pkl` Mahalanobis fit caches, which `load_context` hands to
  `fit_scorer` separately and nothing pairs.
* **Nothing moves:** all 8 preset x condition cells reproduce (crossmodal clean 0.271103), all 258
  caches on disk still load, and `scripts/smoke_cache_identity.py` (19 cases) is always on.

**Open after slice 2:** D3 (`split_fingerprint` is label-blind — reproduced: deleting a box and
changing a class id both leave `c3354ed2f2b1` unchanged) and D4 (the completed-run lookup omits
`epochs`/`imgsz`/`weights`/`overrides`) both need a `RESULT_FIELDS` column, and `_append_row`
deliberately refuses an older schema — so that change forces a fresh CSV and belongs in its own
slice. Also still open: nothing validated on **resume**, no checkpoint content hash, and no
comparison of a cache's `labels_sha256` against the hash recorded at TRAINING time, which is R-E1's
actual acceptance criterion.

**Slice 3 (`6cce035`) — labels and recipe.** `docs/recipe-identity-2026-09-10.md`. The two defects
slice 2 deferred, both closed, plus a third found on the way.

* **D3 closed.** `split_fingerprint` hashes frame NAMES. Reproduced: deleting a box, changing a class
  id and removing a label file all leave it at `0358ccc7de88` while the content hash moves each time.
  New column `label_fingerprint_trainval` — the ledger's algorithm, scope in the NAME because the
  same algorithm over train alone (`8ed69b5974ed`) means nothing against it. Measured now: vis
  `ae7fa57efb2b`, ir `5fd58f37c799`.
* **D4 closed.** Reproduced at the lookup: one row for (yolo26m, 0) at `epochs_cfg` 25 satisfies a
  re-request at 25, **50 and 100**. `recipe_identity` now hashes epochs/imgsz/batch/mosaic/
  close_mosaic/optimizer/patience/amp/deterministic/train_overrides plus the CONTENT hash of the
  starting checkpoint (`best.pt` is overwritten by every run that produces it, so a path is not an
  identity). `workers`/`device` excluded on purpose — host properties; including them would break
  cross-machine resume, the one thing this lookup exists for.
* **`plan_grid()`** keeps two outcomes distinct in kind: a different split/label/classes refuses the
  whole CSV; a different RECIPE on a (variant, seed) this grid will run is a **conflict, refused
  rather than re-run**, because run dir and CSV row are both keyed by `name`.
* **Third defect, found while fixing those two:** THREE implementations of the `images -> labels`
  swap (every component / rightmost SUBSTRING / last separator group) and TWO byte-identical copies
  of the content hash. The substring one rewrites the wrong component when a directory name ends in
  "images". Measured before touching anything: **0 disagreements over all 133,140 real train+val
  paths** — latent, collapsed while still latent into `src/uqfusion/data/labels.py`. The shared hash
  reproduces the ledger's `8ed69b5974ed` bit-for-bit.
* **`recover_row.py` reads the recipe from the run's own `args.yaml`** and refuses if absent —
  reconstructing it from today's config would stamp a recipe that never ran.
* Cost: ~15-24 s warm / **383 s cold** for the 107,627 VIS label files, once per grid launch. Old
  CSVs become read-only (intended; `_append_row` already refused an older schema). `run_queue.py`,
  which drives the server work, does not use `run_grid`.

**Slice 4 (2026-09-10) — the training side, and the join.** `train_gaussian` now writes
`uqfusion_labels.json` into every run directory at training **start AND end**, recording
`labels_train` and `labels_trainval` (the ledger's algorithm, scope in the key name) plus the git
revision including the dirty hash. Start *and* end because a tree that is correct before and after
is a different claim from one that was correct once: on 2026-09-03 at 21:19, 7,591 `pohang01`
train label files were rewritten while no script of this project was running, and it was caught
only because a later hash gate happened to run. Bracketing a run turns an unbounded window into a
bounded one. Failures are logged and swallowed -- a provenance record that aborts a multi-hour
training run is worse than one that is missing, and the absence is itself visible to the joiner.

`scripts/verify_label_provenance.py` is the join, and it **reports rather than refuses**: a run
trained under an older label state is not invalid, it is a run whose label state has to be named
when its numbers are quoted. Refusing there would delete history rather than describe it;
`load_cache` already refuses the case that IS an error.

**Its first verdict is that nothing is joinable yet, and that is the honest answer.** Measured:
**20 checkpoints carry no label manifest** (every `gauss_*`, `mc_*` and ensemble run on disk) and
**258 of 258 caches carry no `labels_sha256`**. Both sides of R-E1's criterion now exist in code
and neither exists in any artifact. The first Gaussian run and the first cache built from here on
are the first pair that can be compared.

**Open after slice 4:** nothing validated on **resume**; `verify_dataset_state.py` has its own `split_fingerprint` with
a different signature and computation (the F14 pattern again, noticed and not fixed); and
`train_overrides` fingerprints by `repr` for non-JSON values, which no current caller passes.

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

### R-F1 — the broad novelty claim is false (F17) · **P1 for publication** · S · **CLOSED 2026-09-10**
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

**Closed 2026-09-10.** Report: [`positioning-2026-09-10.md`](positioning-2026-09-10.md), with the seven-axis comparison table the fix asks for (domain, sensors, uncertainty target, inference-time adaptation, calibration evaluation, registration assumptions, compute). Applied to `scope.md`: §1's “Unlike most prior maritime fusion work…” and §2 gap 2's “Existing visible–infrared fusion is static” are retired, both quoted inline as superseded rather than deleted; §2 gap 1 no longer treats Gaussian YOLOv3 as the sole exception; O5 now names the test rather than promising the verdict; R24 (UA-CMDet) and R25 (DICTA 2024) added to §19.5.

**The review understated this one.** It is scoped as a citation problem, but the *system* description was false too: R-D1 measured `w_vis` as a **single constant 0.9930** across all 2,232 frames × 4 conditions, so “blends them based on live, per-frame uncertainty” did not describe the shipped code. §1 now carries a dated status note separating the aim from the measurement, and the verb in the system sentence is marked as intent.

**Deliberately not done:** the comparison table's cells for R24/R25 calibration protocol, registration assumptions and compute are left blank and labelled “not established here”. The external review verified that those works exist and what their headline mechanism is; it did not audit those axes, and neither did I. Filling them from memory is the exact failure this backlog exists to correct — they must be read from the papers before a manuscript uses the table. R-F2's bibliography corrections remain open and separate.

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

**Q2 — re-score scope (blocks R-A5, step 3). ~~Open~~ ANSWERED 2026-09-10 by measurement; the
question's premise did not survive R-A1.** As written it asked: full re-score under *corrected AP*,
or freeze current results as a named custom-AP variant and use *official AP* only going forward?
Both branches assumed a "corrected AP" exists to move to. It does not:

* The local convention is not **wrong**, it was **undeclared**. Measured delta disagreement is
  0.00028501, 5× below the 0.0014–0.0031 paired floor, so re-scoring on AP grounds buys nothing.
* "Use official AP going forward" is not available either — `TASK_CONFIG` sets `max_dets: None`, so
  `cocoparity` is not COCO in the knob that decides leaderboard comparability, and calling it that
  would repeat F14's naming defect.

We took half of branch two — `local_ap50_95` names the variant (R-A1 (d)) — and declined the other
half on the evidence above.

**What actually drove the re-score was R-A3's 1.95× interval inflation, not the AP convention**, and
R-A5 has already applied it (`f8c6b7e`, rebuilt `6e6bbac`). The live remainder of Q2 is therefore a
narrower question: **re-adjudicate vs genuinely re-score.** R-A5 widened *recorded* intervals by a
factor measured on VIS UQ-arm deltas and transported to fusion cells — an assumption. A true
re-score through `blockboot` is mechanical now. **Recommendation: re-score the rows R-A5 left
indeterminate (20 of 74), not all 74.**

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
