# Pre-registration — the Phase 3 retrain, and the first held-out evaluation this project has had

**Written 2026-09-10, before any pohang04 thermal frame is extracted, any label is drawn,
any checkpoint is trained, and any cache is built.** Committed ahead of the work so the
rules are verifiable in git history.

Supersedes nothing. Extends
[`prereg-uq-mechanism-ablation.md`](prereg-uq-mechanism-ablation.md), whose controls and
floors are inherited here and named where they are reused.

---

## 1. What this pre-registers, and what it does not

This document fixes, in advance:

* the gates that must pass **before any GPU time is spent** (§3);
* the one crossing that is still unmeasured after two negative correspondence results,
  and the declaration that it is expected to be null (§4);
* the training recipe, the arms, and how the seed count is chosen (§5);
* the mechanism test, with its controls built into the design rather than bolted on (§6);
* **the single-look policy on pohang04** (§7) — the one part of this that cannot be
  repaired if it is done wrong;
* every decision rule and threshold (§8), and all declared outcomes including the nulls (§9).

It does **not** pre-register a manuscript claim, a new fusion mechanism, or any
architecture search. Architecture is frozen (§5.1) on evidence already in the record.

**Nothing here is a prediction that results will improve.** Three of the last four
mechanism questions this project asked returned nulls, and the design below is built so
that a fourth null is cheap, early, and interpretable.

---

## 2. Why retrain at all — three findings that force it

A retrain is expensive and this project's default is not to spend GPU on questions already
answered. Three measured findings make this one necessary, and only one of them is about
model quality.

**(a) There is no held-out data, and there never was.** Measured 2026-09-10:

| run | VIS img | VIS lbl | IR img | IR lbl | paired val | role so far |
|---|---:|---:|---:|---:|---:|---|
| pohang00 | 21,768 | 21,768 | 10,918 | 10,918 | 836 (37.5%) | fit **and** report |
| pohang01 | 24,473 | 24,473 | 11,995 | 11,995 | 1,032 (46.2%) | excluded from fit; reported on; **drove the night-veto decision** |
| pohang02 | 27,795 | 27,795 | 6,175 | 6,175 | 247 (11.1%) | fit **and** report |
| pohang03 | 27,085 | 27,085 | 1,922 | 1,922 | 117 (5.2%) | fit **and** report |
| **pohang04** | 26,188 | 26,188 | **0** | **0** | **0** | **never used for anything** |

`DEVELOPMENT_RUNS` in `ctx.py` already says this for pohang00/02/03. pohang01 belongs in
that list too: excluding a run from *fitting* does not hold it out once decisions have been
made by looking at it, and the night veto was.

**pohang04's thermal stream exists upstream and was never extracted.**
`Pohang_dataset/meta/pohang04/timestamps/ir.txt` holds **22,235** thermal timestamps and
`pairs.csv` holds **22,385** stereo rows; `annotation_map.json` records the source archive.
What does not exist anywhere — not on this disk, not in the upstream annotation release —
is thermal **labels**: `labeled_frames.json` reads `tir: 0` for pohang04, against
`tir: 1922` for pohang03, which matches our disk exactly. So every IR label this project
holds came from the release, and the release's thermal coverage falls away run by run:
10,918 / 11,995 / 6,175 / 1,922 / **0**.

pohang04 is therefore one extraction and one bounded annotation pass away from being a
run-disjoint evaluation set. **It is only clean if it is held out of training, which means
it can only be created during a retrain.** That is the argument for doing this now.

**(b) The mechanism is a measured null, and the two candidate explanations were never
crossed.** R-D1 found real sigma does not beat shuffled sigma at any floor, and the shipped
`w_vis` is the single constant **0.9930** across all 2,232 frames in all four conditions.

The tempting story is that the mechanism failed because the fusion almost never fuses — at
`iou_thr` 0.85 only **0.05%** of VIS day detections have a same-class IR partner
([`experiment-log-2026-09-01.md`](experiment-log-2026-09-01.md) §3.1), with a registration
residual of **3–6 px median** (`runs/eval/x_registration_drift.md`).

**That story is already refuted as stated, and the refutation is in our own record.** Two
independent measurements say relaxing the correspondence does not help:

| what was varied | partner rate | AP effect | source |
|---|---|---|---|
| per-frame homography alignment | 0.05% → **4.02%** (80×) | **AP falls** | experiment-log-2026-09-01 §9 |
| `iou_thr` 0.95 → 0.85 → 0.55 | rises monotonically | **0.55 costs −0.0138** | crossmodal-gate-2026-09-01 §349 |

So the honest position is: **more agreement has been bought twice and paid for twice.** A
pre-registered sweep down to IoU 0.10 would be searching for the threshold that makes a null
positive, which the prior pre-registration explicitly forbids.

**What is genuinely untested is the interaction.** Every AP number above was measured with
the mechanism **inert** — `sigma_weighted=False`, `sigma_score_alpha=0.0`, `w_vis` a
constant. And `sigma_weighted` was separately measured as *bit-identical on all eight cells*
(crossmodal-gate §349) at `iou_thr` 0.85, where 99.95% of boxes have no partner to be
weighted against. Each factor was measured while the other was switched off:

* relaxed correspondence, sigma inert → merging averages a good box with a bad one and
  there is no way to down-weight the bad one. **Measured: worse.**
* live sigma, tight correspondence → nothing to arbitrate. **Measured: bit-identical.**

**Neither has been measured with the other live.** That crossing is §4, it is cheap, and
§9 declares in advance that the prior points to a null.

**(b′) A note on what this does not license.** If the crossing is null, the correspondence
question is closed for good, not re-opened at a looser threshold.

**(c) The class set and the deployed model do not match what was trained.** Production IR
is `nc=1` ship-only; the ladder trained 2-class. The deployed `yolo26m-p2feat` is not in the
ladder at all. Any retrain must train what is deployed.

---

## 3. Stage 0 — gates before any GPU time

All five are CPU or free. **If any fails, the retrain does not start.** They are listed as
gates rather than tasks because each one has already cost this project a result.

**G1 — the two machines must agree on labels.** Local label hash is `b92739202127`; server
`dgxanode01` is on `287b11c50b5a`. They do not match. Reconcile via
`scripts/label_hash_ledger.py` / `runs/label_hash_ledger.csv` and record both sides. **No training starts
while they differ**, because every artifact produced under a disagreement inherits an
asterisk that is discovered later and cannot be removed.

**G2 — the estimand is chosen (Q3).** Declared here: **disagreement ranking**, not
predictive likelihood. Ranking is answerable from checkpoints already on disk; likelihood
requires probabilistic members per arm and is out of budget. This is a scope decision, made
before the numbers, and it is what the seven existing UQ runs will be read as.

**G3 — the parity contract is chosen (Q4).** Declared here: **drop bit-identity, declare a
non-inferiority margin across matched seeds.** F06 established that detaching does not
guarantee parity, so a bit-identity contract would be a claim the code does not support.
The margin is the §8 floor.

**G4 — the detectability of every planned comparison is computed first.** For each
comparison in §5 and §6, compute the minimum detectable effect at α = 0.05, power = 0.80
using the machinery in `scripts/ir_equivalence_interval.py`, at the planned seed count.
**Any arm whose MDE exceeds the effect it is meant to resolve is either given more seeds or
cut from the design here, before it is run.** This is R-F3's finding applied forwards: an
underpowered arm does not produce a weak result, it produces an uninterpretable one.

**G5 — pohang04 is excluded from every split file, and the exclusion is verified.** Today
pohang04 supplies **9,841 of 48,136** frames in `data_vis_train_stride2.txt`, plus rows in
`fullres_vis_val.txt`, `fullres_vis_test.txt`, `phase1val_all/day.txt` and
`maha_fit_vis.txt`. A script must assert zero pohang04 frames in every training and
selection list, and it runs as a precondition on each training invocation, not once by hand.

---

## 4. Stage 1 — the correspondence × mechanism crossing (CPU, development data)

**This is a 2×2, not a sweep.** §2(b) establishes that relaxing correspondence has been
measured twice and lost twice, and that enabling sigma has been measured once and was
bit-identical — but that each was measured while the other was switched off. Stage 1
crosses them once and then stops.

Runs on **existing caches** (`runs/cache_m`). No training, no new imagery, no new threshold
search.

### 4.1 The four cells

Predictions, gate, veto, capability prior and every other constant are held fixed.

| | sigma inert (`sigma_weighted=False`) | sigma live (`sigma_weighted=True`) |
|---|---|---|
| **`iou_thr` = 0.85** (shipped) | **A** — the shipped system, already measured | **B** — already measured: *bit-identical to A* |
| **`iou_thr` = 0.55** (relaxed) | **C** — already measured: **−0.0138** | **D** — **never measured. This is the experiment.** |

A, B and C are re-run only as reproduction checks; **the single new number is D.** Their
published values are stated above so that a failure to reproduce them is itself a finding.

**Why 0.55 and not lower.** It is the loosest threshold with an existing AP measurement to
sit against, so D has a matched control. Going below it would introduce a threshold this
project has never priced, in the same step as the mechanism — confounding the two things
the crossing exists to separate. **No threshold below 0.55 is run under this
pre-registration**, whatever D shows.

### 4.2 What is measured

For all four cells: **partner rate**, **fused ship AP** (`local_ap50_95`), and — per §6's
gate — **the number of distinct `w_vis` values and its IQR**. All four conditions, day and
night reported separately.

### 4.3 Decision rule, fixed now

The question is whether the mechanism rescues the merging cost. Formally:

> **interaction = (D − C) − (B − A)**

**The crossing is judged POSITIVE only if D is non-inferior to A** — i.e. `AP(A) − AP(D) <
0.0060` with the §8 block-bootstrap CI — **on at least three of the four conditions.**

That is deliberately a high bar for a cheap experiment: D must recover essentially all of
C's −0.0138 loss purely by letting sigma arbitrate the merges that relaxing the threshold
created. A partial recovery is reported as a partial recovery and **is not** grounds for
adopting a relaxed threshold.

**Measured on `TUNE_RUNS` (pohang00), reported on `TEST_RUNS` (pohang02 + pohang03).** Both
are development data; this step is itself an exposure and is written to
[`exposure-ledger-2026-09-09.md`](exposure-ledger-2026-09-09.md) when it runs, not after.

### 4.4 Prior, declared before the run

**The record points to a null.** Correspondence relaxation has cost AP in both prior
measurements, and `sigma_weighted` has never once moved a number. Stage 1 is run because the
interaction is genuinely unmeasured and costs CPU hours, **not because it is expected to
pass.** Writing the expectation down is what stops a null from being re-litigated later as
"we never really tried".

**Stage 1 is a stop gate, and it gates only §6.** If the crossing is null, §6 does not run
as designed, the fusion is documented permanently as union aggregation rather than
consensus, and the retrain narrows to §5 plus §7 — **which is where this document's value
was always concentrated.** That outcome is **S1-NULL** in §9.

---

## 5. Stage 2 — the retrain

### 5.1 What is frozen, and why

**Architecture is frozen. There is no ladder.** Detector scale is a measured null
(+0.0003 for `yolo26m`; [`crossmodal-gate-2026-09-01.md`](crossmodal-gate-2026-09-01.md)), and R-F3 established the IR ladder
was underpowered by ~1.7× and could not have separated architectures regardless. Rebuilding
either grid spends the entire budget re-deriving results already in the record.

* **VIS:** `yolo26m`, `nc=2` (ship, buoy), `data_vis_stride2.yaml` regenerated **without
  pohang04**.
* **IR:** `yolo26m-p2feat`, `nc=1` ship-only, `data_ir_shiponly_stride2.yaml` — the deployed
  configuration, trained for the first time as the thing that is deployed.
* imgsz 640, epochs 100, optimizer `auto`, per `config.yaml`.
* **Batch is not inherited.** `config.yaml`'s 32 is sized for an H100; the server is a MIG
  `3g.40gb` slice ([`mc-server-control-2026-09-01.md`](mc-server-control-2026-09-01.md)). The largest-common-fit batch is measured
  once and recorded in the manifest, and it is identical across every arm — heterogeneous
  batch size is one of the confounds that made the Phase 1 pool exploratory.

### 5.2 Seeds

**Minimum 5 seeds for any arm that carries a claim**, subject to G4: the count is confirmed
against the computed MDE before the arm runs and raised if the MDE exceeds the effect the
arm exists to resolve. Single-seed arms may exist for plumbing checks only and may not
appear in any comparison table.

### 5.3 Label provenance is bracketed

R-E1 slice 4's `uqfusion_labels.json` is written at training **start and end** for every
run. Rationale unchanged and still live: on 2026-09-03 at 21:19, 7,591 pohang01 train label
files were rewritten while no script of this project was running. Bracketing turns an
unbounded window into a bounded one. `scripts/verify_label_provenance.py` joins checkpoints
to caches; it **reports rather than refuses**, and a run whose label state cannot be named
may not enter a results table.

---

## 6. Stage 3 — the mechanism test, controls built in

Runs only if Stage 1 adopts a rule. The arms are those of
[`prereg-uq-mechanism-ablation.md`](prereg-uq-mechanism-ablation.md) §3 — **S0 shipped, S1
real, S2 constant, S3 shuffled-within-frame, S4 shuffled-across-cache, S5–S7 the score
path** — re-run under the adopted correspondence rule and the retrained checkpoints, with
one addition and one new gate.

**The addition — an arm that must be beaten.** **S8: image-statistic selector.** Frame-level
brightness and contrast only, no predicted uncertainty of any kind. R-D1's conclusion was
that the shipped system is *image-statistic sensor selection*; S8 makes that the explicit
baseline rather than the fallback description. **If real sigma cannot beat S8, the UQ claim
does not survive, whatever S1-vs-S3 shows.**

**The new gate — the knob must actually move.** Before any arm may claim that varying a
quantity helps, that quantity must be shown to vary: report the number of distinct `w_vis`
values and its interquartile range, per condition, for every arm. **An arm whose weight takes
one distinct value is reported as inert and is excluded from the comparison**, because a
constant cannot be responsible for a difference. The shipped preset would have failed this
gate on day one at 1 distinct value across 2,232 frames — it is written here so the next
such case costs hours instead of months.

---

## 7. Stage 4 — the single look at pohang04

**This section is the reason the document exists. Everything above is repeatable; this is
not.**

### 7.1 Building it

1. Extract pohang04 thermal imagery from the archive named in `annotation_map.json`.
   Record the archive's content hash in the manifest.
2. Build pairs with `scripts/build_pairs.py` under the **same** rule adopted in Stage 1.
3. Annotate thermal frames, ship class, to the same protocol as the upstream release.

**Annotation integrity, declared in advance.** Labels must **not** be initialised from, or
corrected against, any detector trained in this project. Model-assisted labelling would make
the holdout a measurement of our own predictions. If any assistance is used it must come
from a model with no pohang04 exposure and be disclosed in the release record; the default
is unassisted. Annotation follows R-B3's release protocol, and the release is hashed and
frozen before any model sees a pohang04 frame.

**How many frames.** The current held-out side is **364 day paired frames** (pohang02 247 +
pohang03 117). A few hundred labelled thermal frames already exceeds it. **The target is set
by G4, not by ambition**: label the number of frames the MDE computation says is needed to
resolve the §8 floor, and stop there. Labelling more is not free — it delays the freeze.

*Composition note, stated precisely because it matters and is not fully measured:* pohang04
contributes **2,343** frames to `phase1val_day.txt` and **0** to `phase1val_night.txt`,
which is strong evidence the run is daylight throughout but is a measurement over the val
subset only. The full day/night composition is measured and recorded **before** the freeze,
and if any night frames exist they are reported as a separate cell.

### 7.2 The single-look policy

**pohang04 is scored exactly once**, after Stages 0–3 are complete and every constant,
threshold, checkpoint, correspondence rule and preset is frozen and committed.

* No arm is selected, tuned, ranked or dropped using a pohang04 number.
* No pohang04 number is looked at before the freeze commit exists in git.
* **If a pohang04 result is disappointing, the response is to report it.** Re-tuning after
  the look and re-scoring converts pohang04 into development data permanently and there is
  no fifth recording to replace it.
* Any second look requires a new pre-registration that states what changed, why, and
  explicitly records that the set is no longer held out.

This policy is the entire value of the stage. It is written before the data exists so that
it cannot be softened after.

---

## 8. Decision rules and thresholds, all fixed here

**Absolute floor: 0.0060.** Inherited from
[`prereg-uq-mechanism-ablation.md`](prereg-uq-mechanism-ablation.md) Amendment 1: the paired
2σ floor is 0.0014–0.0031 (`runs/eval/metric_noise_floor.md`) computed under the **IID**
bootstrap, and R-A3 measured IID intervals on this data as ~1.95× too narrow, so the floor
inflates with everything else. 0.0031 × 1.95, rounded up.

**Interval method: moving-block bootstrap, L = 20, n_boot = 1000, via `blockboot`.** L = 20
is the largest defensible block — the valid maximum is shortest-run/5 = 23 because pohang03
holds 117 frames (`runs/eval/interval_block_sensitivity_v3.md`). The sensitivity curve is
still rising at 20, so 1.95× is a lower bound on the inflation, not an estimate of it.

**Every comparison must meet both a statistical and an absolute criterion:** the block
bootstrap CI for the delta excludes zero, **and** the delta is ≥ 0.0060. A measured margin
alone degenerates to a sign test at these magnitudes
([`prereg-snms-draw-averaged-gate.md`](prereg-snms-draw-averaged-gate.md)), which is why both are required.

**Counts are reported at four floors — 0.0014, 0.0031, 0.0060, 0.0100 — with the verdict
taken at 0.0060**, so a conclusion that depends on the floor choice is visible as such.

**Day and night are reported separately, always, and never pooled into a headline.** The
paired set is 46.2% pohang01 night from a single recording; a pooled number is largely a
statement about pohang01.

**No equivalence is claimed without an interval.** Per R-F3: a non-significant test is not
an equivalence test, and "p > 0.05, therefore no effect" may not appear in any artifact
produced under this pre-registration.

---

## 9. Declared outcomes

Named in advance so that none of them is a surprise that invites a search.

* **S1-NULL** — cell D fails §4.3. Letting sigma arbitrate does not pay for the cost of
  merging, and the two prior negative results are not explained by the missing interaction.
  §6 does not run as designed. **The correspondence question is then closed permanently**,
  not re-opened at a looser threshold (§2 b′), and the fusion is documented as **union
  aggregation, not consensus** in the method description. This is the outcome §4.4 expects,
  it is a publishable structural finding about VIS+IR fusion at this registration quality,
  and it costs CPU hours to establish.
* **MECH-POSITIVE** — real sigma beats shuffled sigma **and** beats S8 under §8. The UQ
  claim is supported for the winning mechanism, on this recording, and is then subject to §7.
* **MECH-NULL** — real sigma ties shuffled sigma or fails to beat S8. The thesis claim
  narrows to image-statistic sensor selection, and the Gaussian head is evaluated on its own
  terms as a localization-error scale. **This is the outcome current evidence points to and
  it is a result, not a failure.** It is also what R-D1 already found; a second null under a
  repaired correspondence rule is substantially stronger than the first.
* **HOLDOUT-GAP** — pohang04 scores materially below the development runs. Reported as the
  headline generalization result. This is the number the project currently cannot produce
  at all, and it is informative in either direction.
* **BUDGET-CUT** — G4 shows the affordable seed count cannot resolve the floor for a given
  arm. That arm is cut and recorded as cut. **An arm that cannot be resolved is not run.**

**There is no re-run on a disappointing result.** Any follow-up requires a new
pre-registration naming what changed and why.

---

## 10. What this cannot settle

* **pohang04 is one run of one harbour, and it is daylight.** It does not test night
  generalization, weather beyond the synthetic corruption recipe, another sensor rig, or
  another port. It removes the *tuned-on-the-test-set* objection; it does not remove the
  *single-recording* objection, and no artifact here may claim otherwise.
* **The VIS labels for pohang04 come from the upstream release**, so pohang04 tests
  generalization to unseen *scenes*, not to an independent annotation process. The thermal
  labels will be ours and are therefore a different provenance from the VIS side — that
  asymmetry is disclosed wherever the number appears.
* **Corruption cells are synthetic.** Fresh seeds over the same scenes test robustness to
  the transform draw, not to new weather.
* **Stage 1 selects on development data.** The adopted correspondence rule is a tuned
  constant like any other and carries the same exposure caveat as the gate constants.
* **This does not resolve Q1.** It creates one clean evaluation run. It does not make the
  four development runs clean, and it does not retroactively repair any published number.

---

## 11. Provenance and stop rules

Every stage records the full `eval.identity.system_identity()` block, the label manifests
from both ends of training, the measured batch size, the correspondence rule and its
parameters, and the bootstrap seed. Reports go to **new filenames**; nothing published is
overwritten. This pre-registration is **never edited in place** — corrections are appended
as dated amendments, as in the document it extends.

**Hard stops:**

1. G1 fails (label hashes differ) → no training.
2. G4 cuts an arm → that arm is not run, and its absence is recorded.
3. Stage 1 finds no rule → Stage 3 does not run as designed.
4. An arm's `w_vis` takes one distinct value → inert, excluded from comparison.
5. A pohang04 number is looked at before the freeze commit → **the holdout is void**, is
   relabelled development data in `ctx.py`, and the loss is recorded in the exposure ledger.

Stop rule 5 has no remedy, which is why it is last and why §7.2 exists.

---

## Amendment 1 — 2026-09-10, Stage 0 corrections

Appended per §11. **No decision rule, threshold, arm, floor or policy changes.** Three
corrections of fact, all found while running Stage 0, all measured rather than argued.
Nothing below relaxes a gate; two of the three make a gate wider.

### A1.1 — G1 compares two hashes that are not comparable

§3's G1 sets local `b92739202127` against server `287b11c50b5a`. **That is a `tree`-scope
hash against a `train`-scope hash** — different algorithms over different file sets, which
`label_hash_ledger.py`'s docstring and `handoff-2026-09-04.md` §6 trap 1 both warn about.
The error originates at [`experiment-log-2026-09-02.md:711`](experiment-log-2026-09-02.md),
which labels the tree hash "VIS train label hash"; `prereg-night-label-restore.md:98` and
then G1 inherited it.

**The correct comparison is train-vs-train: local `8ed69b5974ed` vs server
`287b11c50b5a`.** G1's *verdict* is unchanged — the machines do differ, and no training
starts — but the difference is now measured rather than assumed:

`scripts/reconstruct_prerestore_hash.py` rebuilds the server's tree from
`runs/visfilter/visfilter_manifest.json` (`cut_dark` emptied 17,502 files wholesale) and
reproduces **`287b11c50b5a` with 616,891 boxes, bit-exact**. Since that hash folds in every
byte of all 96,275 train labels, **the two machines differ by exactly the +94,553-box night
restore and by nothing else.** G1 is therefore a transfer, not an investigation, and it
passes when the server reads `8ed69b5974ed` (train) and `b92739202127` (tree).

Full record: [`g1-label-reconciliation-2026-09-10.md`](g1-label-reconciliation-2026-09-10.md).

**Carried forward as unfinished:** the IR label tree has never been hashed across the two
machines in any scope. Stage 2 trains IR for the first time as the deployed `nc=1`, so G1
is extended to require an IR scope in the ledger before Stage 2 — not before Stage 1, which
is CPU-only on existing caches.

### A1.2 — G5's list of five is a list of fifteen

§3's G5 names `data_vis_train_stride2.txt`, `fullres_vis_val.txt`, `fullres_vis_test.txt`,
`phase1val_all/day.txt` and `maha_fit_vis.txt`. Measured 2026-09-10: **pohang04 appears in
15 lists, totalling 58,144 rows**, and the largest is the one not named —
`Pohang_dataset/visible/train.txt` at **19,687** rows, the master list every derived stride
list is cut from. Also unnamed: `data_vis_train_stride5/10/19.txt`,
`fullres_vis_train_stride5.txt`, `visible/val.txt`, `visible/test.txt`,
`derived_day/day_val_vis_stride1.txt`, and `runs/bench_bundle/uqfusion_bench/train.txt`.

Enumerating by hand is what produced a list of five. **`scripts/assert_holdout_excluded.py`
therefore takes no list of files**: it walks the split roots and checks every list it finds,
so a list added later is covered without anyone remembering. It exposes `assert_clean()` for
use as a training precondition, per G5's requirement that it not be a manual step. It
currently returns 1 across 15 of 42 lists.

All IR lists carry zero pohang04 rows, and will until Stage 4 extracts thermal frames — the
gate covers them anyway, for that reason.

### A1.3 — the pohang04 label archive is VIS, and is already on disk

An archive offered as "the 04 labels" (`all.zip`, 26,191 entries) was checked against the
tree before any use. It is **`pohang04` visible-stereo labels, 2-class, 156,652 boxes**:
12,571 `left` + 13,617 `right`, matching `Pohang_dataset/visible/labels/pohang04`'s `_L_`/`_R_`
counts exactly. Across 3,000 sampled frames every on-disk label reproduces from it by one
fixed transform — letterbox to 640×640 at aspect 0.528125 (`y' = 0.528125y + 0.2359375`,
`h' = 0.528125h`), then clip to `[0,1]`; the 148 non-matches are all boxes the archive
records outside the frame (`x = 1.0217`, `x = −0.0237`) that the tree has clipped.

**It is the raw-coordinate source of labels the project already holds.** §2(a)'s finding is
unaffected: `labeled_frames.json` still reads `tir: 0` for pohang04, no thermal label exists
anywhere, and **§7.1's annotation pass is still required in full.**

**Opened, not decided:** the archive supplies pohang04 VIS boxes in *unletterboxed* image
coordinates, which makes geometric projection into the thermal frame via
`meta/pohang04/calibration` technically possible. That is not model-assisted labelling and
so is not barred by §7.1's integrity clause — but it is not annotation either, and it would
inherit the 3–6 px registration residual measured in `runs/eval/x_registration_drift.md`
into the labels of the one held-out set this project will ever have. **It is not adopted
here.** Adopting it requires its own pre-registration stating the residual's effect on label
quality; until then §7.1's default — unassisted annotation to the release protocol — stands.

---

## Amendment 2 — 2026-09-10, the pohang04 thermal imagery is already extracted

Appended per §11. **No decision rule, threshold, arm, floor or policy changes.** One
correction of fact about what is on disk, which removes a step from §7.1 and shortens
Stage 4.

### A2.1 — the claim that is wrong

§2(a) states that pohang04's thermal stream "exists upstream and **was never extracted**",
and §7.1 step 1 accordingly instructs: *"Extract pohang04 thermal imagery from the archive
named in `annotation_map.json`."*

**The imagery is already extracted.** Measured 2026-09-10:

```
D:\Datasets\Pohang\pohang04\infrared\images\   22,235 files, 640x512, 16-bit PNG
Pohang_dataset\meta\pohang04\timestamps\ir.txt 22,235 lines
```

The counts agree exactly, and the run also carries its own `infrared/timestamp.txt`. This
is not special to pohang04 — pohang03 holds 24,468 extracted thermal frames against 1,922
labels. Imagery extraction and label availability were always independent, and only the
latter falls away run by run. The false claim came from reading `tir: 0` as a statement
about the whole thermal stream when it is a statement about labels alone.

The frames are on `D:` and have never been brought into the project tree, which is why the
`A:`-side surveys that produced §2(a) did not see them.

### A2.2 — what §7.1 step 1 becomes

Step 1 is no longer an extraction. It is a transfer, and it acquires the provenance
obligations the extraction was going to carry:

1. Record the content hash of the **extracted frame set** (not the archive, which is no
   longer the immediate source), plus the count and the `timestamp.txt` digest.
2. Convert 16-bit to the project's 8-bit representation using the same path the trained
   runs use — `D:\Datasets\Pohang\_scripts\convert_ir_to_8bit.py`. **The conversion is now
   part of the holdout's provenance**: a different mapping from the one pohang00–03 were
   built under would make pohang04 a test of the conversion as much as of the model, and
   the conversion parameters are therefore recorded in the manifest alongside the hash.
3. Copy into the project tree under the run-disjoint layout, and confirm G5 still returns 0.

Steps 2 and 3 of §7.1 — pair building under Stage 1's adopted rule, and the annotation
pass — are **unchanged**. In particular **§7.1's annotation requirement stands in full**:
no thermal label for pohang04 exists in the PoLaRIS release or anywhere on either disk,
which A2.3 records as now checked at the source rather than inferred.

### A2.3 — the absence of pohang04 thermal labels, verified at the source

§2(a) inferred this from `labeled_frames.json`. It is now checked against the release
bundles themselves. The archive offered as "the 04 labels" is **sha256-identical to the
inner `all.zip` of the preserved `D:\Datasets\Pohang\04.zip`**, so it is the authoritative
PoLaRIS bundle and not a partial copy.

| bundle | left | right | tir |
|---|---:|---:|---:|
| 00 (old format, `all/tir.zip`) | 10,787 | 10,981 | 10,918 |
| 01 (old format) | 11,995 | 12,478 | 11,995 |
| 02 | 13,685 | 14,111 | 6,175 |
| 03 | 13,145 | 13,940 | 1,922 |
| **04** | 12,571 | 13,617 | **directory absent** |

For pohang04 the `tir` directory is **absent, not empty** — there is no zero-length release
to misread. Three further checks agree: `annotation_map.json` carries a `tir` key with 0
entries, so the extractor looked and found nothing; `dynamic.zip` holds image, lidar and
radar tracking only, with no thermal stream; and a sweep of both drives finds no pohang04
file under any IR, `tir` or thermal label path. `ir-handoff-2026-08.md:38` recorded the
same conclusion in August.

**Net effect on Stage 4:** the run is one annotation pass — not one extraction and one
annotation pass — away from being a run-disjoint evaluation set. §7.2's single-look policy
is untouched and remains the binding constraint.

---

## Amendment 3 — 2026-09-10, Stage 2 trains on one machine

Appended per §11, **before any Phase 3 checkpoint is trained.** This one **changes a gate**
and says so plainly. Decided by the operator; the reasoning and the new obligation it
creates are recorded here rather than in a commit message.

### A3.1 — the decision

**Every Phase 3 checkpoint is trained on the laptop (RTX 4080 Laptop, 12,282 MiB).
`dgxanode01` trains nothing in Phase 3.**

The cost is small and measured. At steady state on the real job the server does 1013 s/epoch
against the laptop's 1058–1065 — **about 4.5%**
([`handoff-2026-08-21-full-scale.md:687`](handoff-2026-08-21-full-scale.md)). That section
also retracts the two earlier speed claims: the 1.23× came from a stripped probe with no
sigma head and no mosaic, and the "server 7% slower" figure was read off epoch 1. Neither
should be quoted again, here or anywhere.

### A3.2 — what this does to G1, stated as a change and not as a pass

G1 requires the two machines to agree on labels before training starts. **They still do not
agree, and this amendment does not reconcile them.**

What it does is remove the exposure G1 exists to prevent. §3 gives the reason for the gate
exactly: *"every artifact produced under a disagreement inherits an asterisk that is
discovered later and cannot be removed."* An asterisk is inherited when artifacts are
produced across two disagreeing label states. If one machine produces all of them, no
artifact spans the disagreement, and every Phase 3 checkpoint carries a single label
provenance — the restored tree, `train` `8ed69b5974ed` / `tree` `b92739202127`, whose
divergence from the server is now measured exactly and attributed entirely to the night
restore (Amendment 1).

**G1 is therefore restated for Phase 3, not waived:**

> **G1′ — one machine trains, and the gate proves which one.** Phase 3 training runs only on
> the host whose label hashes are recorded in `runs/label_hash_ledger.csv` for the commit
> being trained. The training precondition asserts the hostname **and** re-reads the label
> hashes, and refuses on either mismatch.

The hostname assertion is not ceremony. Without it, this amendment converts a hard stop into
a convention, and a convention is exactly what the 2026-09-03 21:19 label rewrite proved this
project cannot rely on. The mechanical check is what makes the relaxation safe, so it ships
with the amendment rather than after it.

**`dgxanode01` is recorded as holding a stale VIS label tree** (`287b11c50b5a`, pre-restore).
It is out of scope for Phase 3. Any future run there must reconcile first; that obligation
does not expire with this amendment, and the IR-scope gap noted in A1.1 stays open against
the day the server is used again.

### A3.3 — what follows mechanically

* **Blockers that dissolve:** the server-side label transfer, the cross-machine IR hash, and
  the MIG `3g.40gb` batch measurement. None are on the Phase 3 path any more.
* **§5.1's batch clause is unchanged in substance and re-anchored in fact.** "The largest
  common fit, measured once, identical across every arm" now means the 4080's ceiling at
  12,282 MiB, not the MIG's at 40 GB, and `config.yaml`'s 32 remains inapplicable — it was
  sized for an H100. The measured value is recorded in the manifest exactly as §5.1 requires,
  and heterogeneous batch across arms is still the confound it always was.
* **§5.2's seed count is unchanged, and G4 still governs it.** What changes is the budget G4
  is priced against. Measured laptop throughput
  ([`TODO-improvements.md:19`](TODO-improvements.md)) is VIS 282 s/epoch at stride-5 over
  19,256 frames and IR 125 s/epoch at stride-2 over 11,640, with Phase 2 runs early-stopping
  at 54–68 epochs. Stage 2's VIS arm is roughly twice that frame count once pohang04 is
  removed, and `yolo26m-p2feat` is heavier than the IR figure was measured on, so **neither
  number is quoted as a Stage 2 estimate here.** The budget is measured on the first arm and
  recorded before the seed count is fixed — G4 cuts arms on measured cost, never on an
  extrapolated one.

### A3.4 — what this does not change

Nothing about the estimand, the floor, the interval method, the arms, the mechanism gate, or
§7. **§7.2's single-look policy is untouched.** Training on one machine changes who produces
the checkpoints; it changes nothing about what may be looked at, when, or how often.
