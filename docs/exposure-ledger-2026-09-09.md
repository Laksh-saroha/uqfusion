# Exposure ledger — what each run has actually been used for

**Written 2026-09-09 for R-B2 (review finding F02).** This is a *label*, not a
computation, and labelling is the whole fix. The review's claim is that the development
set has become the test set; this document states, per run, what that means concretely so
no future reader has to reconstruct it from logs.

**The one-line version: this repository contains no untouched test set.**

---

## 1. The runs

| run | frames (paired val) | day/night | status | used for |
|---|---:|---|---|---|
| `pohang00` | 836 | day | **development** | `TUNE_RUNS`. Constant sweeps, gate fitting, diagnostics, and reported tables. |
| `pohang01` | 1,032 | night | **development (held out of gate fits only)** | Excluded from `FIT_RUNS`, so no gate constant was fitted on it — but it has been reported on repeatedly, and every night verdict in the project is scored here. |
| `pohang02` | 247 | day | **development** | Declared `TEST_RUNS` *after* exposure. See §2. |
| `pohang03` | 117 | day | **development** | Declared `TEST_RUNS` *after* exposure. See §2. |

`DEVELOPMENT_RUNS` in `src/uqfusion/eval/ctx.py` is now the union of the three day runs,
and `sel("dev")` returns exactly those frames.

## 2. Why `TEST_RUNS` is not a test set

Three separate reasons, each sufficient on its own.

**It was declared after the fact.** `TUNE_RUNS`/`TEST_RUNS` were introduced to describe a
run-disjoint split that did not exist while the constants were being chosen. The module's
own comment has said so all along: *"A RUN-DISJOINT SPLIT OF THE DAY FRAMES, which the
project has never had."*

**It has rejected candidates.** `docs/experiment-log-2026-09-01.md` records keeping support
IoU 0.30 because 0.55 scored poorly **on TEST**. A test set that eliminates a candidate has
participated in selection — it does not have to pick the winner to bias the estimate. This
is ordinary model-selection bias (Cawley & Talbot 2010), and it is not undone by writing a
pre-registration afterwards.

**`fit` and `day` are the same frames.** Confirmed in code, not inferred:
`sel("fit") == sel("day")` is `True`, because `FIT_RUNS` excludes only `pohang01` and
`pohang01` is entirely night. Every constant tuned on "fit-run day frames" was therefore
reported on the frames it was tuned on. This is what `project-benchmark-holdout` already
recorded and what F02 independently confirms.

## 3. What corruption seeds do and do not buy

New corruption seeds over the same scenes measure robustness to the random transform draw.
They do **not** supply new scene-level test data: the underlying imagery, its objects and
its labels are the same. A grid that looks like 11 cells × several seeds is still four
recordings.

Relatedly — and separately measured in R-A3 — those four recordings are 10 Hz video, so
even *within* a run the frames are not independent observations.

## 4. What changed in code

`load_context()` gains `role`:

* **`role="develop"`** (the default, so every one of the 63 existing call sites is
  unaffected and every recorded number is reproduced bit-for-bit) — fitting on the frames
  you report is permitted, because that is what development is.
* **`role="final"`** — `capability_sel` becomes **required**, and any selector that spans
  the scoring frames (`all`, `day`, `fit`, `test`, `dev`) is refused outright.

`FusionContext.assert_final_scorable(eval_sel)` then checks the one thing `load_context`
actually fits from evaluation frames — the capability prior — against the frames about to
be scored, and refuses on any intersection. It compares frame index sets, so it is a check
rather than a label. Everything else the context uses is either fitted on TRAIN caches (the
two Mahalanobis scorers) or read from a frozen constants file.

R-B2 asks for `capability_sel` to be made required. Making it required unconditionally
would have broken 56 of 63 call sites and silently rewritten every recorded number, so it
is required exactly where it can do harm: in a final score.

`sel()` also gains `tune`, `test` and `dev`. `TUNE_RUNS`/`TEST_RUNS` had been module
constants that `sel()` could not resolve, so a run-disjoint evaluation was expressible in
comments but not in code.

## 5. What this does not fix

**There is still nothing to run a final evaluation on.** The machinery to refuse a
contaminated final score now exists; the uncontaminated data does not. Options, in
increasing order of honesty:

1. Score on `test` (`pohang02` + `pohang03`, 364 day frames) and label it *"held-out split,
   previously inspected"* — cheap, and better than nothing, but it is not a clean estimate
   and must never be called one.
2. Nested leave-one-run-out over the three day runs, with **every** fitted component inside
   the fold. This is the honest use of the data that exists. It is also real work: the
   constants, the gate, the capability prior and the veil/structure fits all have to move
   inside the loop, and today they do not.
3. Acquire or reserve a genuinely untouched release and freeze the corruption families,
   seeds, comparisons and stopping rule *before* looking at it.

**Do not describe night as held out from detector training.** `pohang01` is held out of the
*gate* fits. D6-rev explicitly permits night training frames, and the VIS retrain of
2026-09-04..09 trained on restored `pohang01` labels. "Held out from the gate" and "held out
from the detector" are different claims and only the first is true.

## 6. Related open items

* **R-B1** — the structure gate's `ir_thr` is fitted from a midpoint that includes the
  night minimum, and the IR health fit covers all clean paired IR frames including night.
  Its 0% clean outlier rate is an in-sample construction. Same family of problem, different
  constant.
* **R-E1 / F14 — the manifest is insufficient to identify a run.** Noticed while writing
  this and then chased to the end. **Correction to the first version of this section: there
  is no drift and nothing fails to regenerate.** Every number is deterministic and
  bit-reproducible. What is broken is the *manifest*.

  Three artifacts all record `"preset": "crossmodal"` and disagree on `cap_ir`:

  | file | written | `cap_ir` | what it actually is |
  |---|---|---:|---|
  | `final_system_crossmodal.json` | 2026-09-01 12:36 | 0.009246512091269591 | no IR NMS, no cap scale |
  | `final_system_crossmodal_irnms.json` | 17:09 | 0.009415770445243315 | IR NMS 0.70, no cap scale |
  | `final_system_crossmodal_v2.json` | 19:03 | 0.0023539426113108287 | IR NMS 0.70, ÷ `cap_ir_scale` 4.0 |

  All three reproduce exactly. `preset="crossmodal"` acquired two defaults during that
  single day — `ir_nms = 0.70` and `cap_ir_scale = 4.0`, both in `b3d8371` — and the
  config block records **neither**, so the artifact cannot say which system produced it.
  The 12:36 run log gives it away only by omission: it prints `IR 0.0092 (ratio 36.2x)`
  with no "IR scaled 1/…" clause, because that clause did not exist yet.

  Verified rather than argued: recomputing the prior from
  `gauss_ir_paired_clean.pkl` gives `0.009246512091269591` bit-for-bit (the 12:36 value),
  and applying NMS 0.70 then dividing by 4 gives `0.0023539426113108287` bit-for-bit (the
  current value). `cap_vis` is identical throughout, and the result is independent of
  `SORT_KIND`, so neither R-A1's stable sort nor the R-B2 changes are involved.

  **Consequence, already acted on:** R-A5's change-impact table was built against the
  12:36 file and labelled it the shipped preset. It is two revisions behind. Rebuilt
  against `final_system_crossmodal_v2.json` — see `runs/eval/change_impact_v4.md`.

  **The fix this points at:** a config block must record every value that changes the
  system, not the preset *name*, since a preset name is a moving target. That is R-E1.
