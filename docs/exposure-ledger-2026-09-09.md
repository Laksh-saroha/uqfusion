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
| `pohang04` | 12,482 | day (solar-elevation rule, A9.4) | **held out → spent 2026-09-20** | The Phase 3 §7.2 single look, scored exactly once and never again. See §7. |

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

**Option 3 was taken, and it is now spent.** `pohang04` was reserved, its inputs frozen by
hash in commit `85a07c1`, and its families, seeds, cells and stopping rule fixed in Amendments
8 and 9 before any of it was scored. The look ran on 2026-09-20 (§7). That option is no longer
available to this project: there is no second untouched release, and re-scoring `pohang04`
would need a new pre-registration stating that it is no longer held out.

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

  **Corrected again, wider than first written.** The first pass named three artifacts.
  Enumerating all fourteen `runs/eval/final_system*.json` shows **five** recording
  `"preset": "crossmodal"`, across **three** different `cap_ir` values:

  | file | written | `cap_ir` | regime | 8 cells vs 12:36 |
  |---|---|---:|---|---|
  | `final_system_crossmodal.json` | 09-01 12:36 | 0.009246512091269591 | no IR NMS, no cap scale | — |
  | `final_system_crossmodal_hardened.json` | 13:28 | 0.009246512091269591 | no IR NMS, no cap scale | **identical 8/8** |
  | `final_system_crossmodal_rsys.json` | 15:59 | 0.009246512091269591 | no IR NMS, no cap scale | **identical 8/8** |
  | `final_system_crossmodal_irnms.json` | 17:09 | 0.009415770445243315 | IR NMS 0.70, no cap scale | 0/8 |
  | `final_system_crossmodal_v2.json` | 19:03 | 0.0023539426113108287 | IR NMS 0.70, / `cap_ir_scale` 4.0 | 0/8 |

  So the name `crossmodal` meant three different systems within one day, and the artifact
  cannot distinguish them. `hardened` and `rsys` are the two the first version of this
  section missed; they are bit-identical to 12:36 on all eight cells, which is exactly why
  the ambiguity was invisible — the `crossmodal-gate` document's two "re-runs
  bit-identical on all eight" claims are *sound*, because all three compared artifacts sit
  in the same pre-NMS regime. It is the boundary at 17:09 that the name does not record.

  **What the config block records:** `bright_soft`, `cap_ir`, `cap_vis`, `iou_thr`,
  `n_boot`, `preset`, `single_passthrough`, `veto`, `veto_filter`, `veto_rule`. Neither
  `ir_nms` nor `cap_ir_scale` appears in **any** of the fourteen files. `cap_ir` is the
  only witness to either, and only by its numeric value.

  All five reproduce exactly; there is no drift. `preset="crossmodal"` acquired two
  defaults during that single day — `ir_nms = 0.70` and `cap_ir_scale = 4.0`, both in
  `b3d8371` — so **re-running `preset="crossmodal"` today does not reproduce the numbers
  the first three artifacts recorded under that name.** The 12:36 run log gives the
  regime away only by omission: it prints `IR 0.0092 (ratio 36.2x)` with no
  "IR scaled 1/..." clause, because that clause did not exist yet.

  Verified rather than argued: recomputing the prior from `gauss_ir_paired_clean.pkl`
  gives `0.009246512091269591` bit-for-bit (the 12:36 value); applying NMS 0.70 gives
  `0.009415770445243315` (the 17:09 value); dividing that by 4 gives
  `0.0023539426113108287` bit-for-bit (the current value). `cap_vis` is identical
  throughout, and the result is independent of `SORT_KIND`, so neither R-A1's stable sort
  nor the R-B2 changes are involved.

  **Size of the regime shift, for calibration:** the largest per-cell move across the
  17:09 boundary is lowlight/day, +0.0017 (0.019429 -> 0.021137); the night cells move
  +0.0019 (0.081007 -> 0.082886). Both sit inside the 0.0014-0.0031 paired noise floor.
  No verdict in the record turns on this. What it costs is **reproducibility by name**,
  not correctness of a conclusion.

  **`crossmodal26m` and `crossmodal26m_snms` are NOT ambiguous.** Both rewrite
  `preset = "crossmodal"` at `ctx.py:391`, *before* the `ir_nms` default at line 412 and
  the `cap_ir_scale` default at line 734, so they inherit both repairs; and both were
  introduced after `b3d8371`, so neither ever meant anything else. The ambiguity is
  confined to the bare name `crossmodal` on 2026-09-01.

  **Sweep for other consumers (2026-09-10).** Requested after the finding above.
  * **Code:** `scripts/change_impact_table.py` is the only Python file that reads
    `final_system_crossmodal.json` by name, and it was corrected in `6e6bbac`. Nothing
    else in `src/` or `scripts/` keys off any `final_system*.json` filename.
  * **`docs/crossmodal-gate-2026-09-01.md`:** clean. Its headline table already carries
    the post-17:09 values (0.0829 on the night cells), and §3c states the transition
    outright: *"goes 0.0810 -> 0.0829 for the system and for the bar, so the gap the
    fusion is judged on does not"* move. The pre-NMS 0.0810 values survive only in a
    diagnostic table that the same document reconciles.
  * **`docs/eval/final_system_2026-09-01.md`:** unaffected. It records the 2026-08-20
    system, written with `preset=None`, and never claims the crossmodal name.
  * **`final_system_adopted_regress{,2}.json`** carry `preset="adopted"` with
    `cap_ir = 0.009246512091269591` at 12:37 and 17:14. `adopted` takes neither default
    (both are gated on `preset == "crossmodal"`), so that value is correct at both times
    and the name is stable. The seven `preset=None` artifacts predate the preset system.

  **Consequence, already acted on:** R-A5's change-impact table was built against the
  12:36 file and labelled it the shipped preset. Rebuilt against
  `final_system_crossmodal_v2.json` — see `docs/eval/change_impact_2026-09-09_v4.md`.

  **The fix this points at:** a config block must record every value that changes the
  system, not the preset *name*, since a preset name is a moving target. That is R-E1.

## 7. Exposure events, logged when they run

* **2026-09-14 — Phase 3 Stage 1, the correspondence × mechanism crossing**
  (`prereg-phase3-retrain-2026-09-10.md` §4, `scripts/stage1_crossing.py`). Preset
  `crossmodal26m`, `runs/cache_m`, `iou_thr` ∈ {0.85, 0.55} × `sigma_weighted` ∈ {off, on}.
  Decided on `TUNE_RUNS` (pohang00); reported on `TEST_RUNS` (pohang02+03) and on night
  (pohang01). Adds one further inspection of all four runs. **No pohang04 frame is scored.**
  Logged before the run, per §4.3.

* **2026-09-20 — Phase 3 §7.2, the pohang04 single look**
  (`prereg-phase3-retrain-2026-09-10.md` §7.2 with Amendments 8 and 9,
  `scripts/holdout_p04_look.py`, freeze commit `85a07c1`). Preset `crossmodal26m`,
  `runs/cache_p3`, five Phase 3 systems (VIS seed *k* + IR seed *k*, *k* = 0..4), 11 cells,
  VIS draws 941–944 with IR = VIS + 10. **12,482 pohang04 day pairs**, fused ship AP scored
  against VIS ground truth only (A8 — there are no thermal labels). Ran 18.6 h; outputs
  `docs/eval/holdout_p04_look.md` and `.json`, marker `runs/holdout_p04/LOOK_TAKEN.json`.

  Verdict cell `clean/clean`, rule A9.3: **NO-GAP**.

  | | |
  |---|---:|
  | `AP_ref` (pohang02+03, the **lower** development group) | 0.2898 |
  | `AP_p04` (seed mean) | 0.2682 |
  | D = `AP_ref` − `AP_p04` | **+0.0216** |
  | D 95% CI (unpaired moving-block bootstrap, L = 20, n_boot = 1000, seed 1) | **[−0.0120, +0.0502]** |

  **NO-GAP here does not mean no gap.** The point estimate clears every reported floor,
  including 0.0100; the verdict is NO-GAP solely because the interval's lower bound is
  below zero, which is the second conjunct of A9.3. All four `gap_by_floor` entries are
  `false` for that one reason. The defensible statement is *"a gap of +0.0216 was observed;
  the data exclude neither zero nor a gap as large as +0.05."* Writing it as "pohang04 shows
  no generalization gap" repeats the R-F3 error (`project-ir-ladder-underpowered`), this time
  into the paper. The interval is wide because it is unpaired and therefore carries run-level
  variance: the two development groups differ from each other by **0.1057** (pohang00 0.3955
  vs pohang02+03 0.2898), 4.9× the held-out gap. pohang04 falls just below the bottom of the
  development range, not outside it. `above_development` is `false` (U = −0.1272 against
  pohang00).

  Logged **after** the run, necessarily and by design: the marker is created with `O_EXCL`
  *before* scoring, so the exposure is incurred the instant the look starts, and a
  before-the-fact entry could describe an event that then refused at one of the six gates.

  **pohang04 is now spent.** Any further scoring of it — a different architecture, a new
  cell, a re-run of this same command — is a second look and requires a new pre-registration
  recording that the set is no longer held out.

* **2026-09-27 — Phase 3 night check, development, descriptive** (`scripts/p3_night_check.py`).
  Preset `crossmodal26m`, `runs/cache_p3/seed{0..4}`, cell `clean/clean` only, the five Phase 3
  systems. Per seed: VIS-only, IR-only, fused as shipped, and fused with the veto's night arm
  removed (V1's OFF arm, `off_arm` in `scripts/eval_night_veto.py`). Ship AP; day (pohang00/02/03)
  and night (pohang01) reported separately, never pooled (Phase 3 prereg §8). Adds one further
  inspection of all four development runs, the first under the Phase 3 checkpoints at night.
  **No pohang04 frame is scored.**

  *Why:* the paper's "fused ≥ max(VIS, IR) on every cell" was measured on pre-restore checkpoints
  whose VIS scored 0.0000 at night; §7 reports Phase 3 checkpoints trained on restored labels.
  V1 (`night-veto-axis-closed-2026-09-04.md`) measured removing the night veto from a restored-label
  VIS at **+0.1785** on clean night. This checks whether that holds for the reported system.

  *Fixed before the number:* the claim **fails on a slice** if `fused − VIS_only` or
  `fused − IR_only` (seed means) is ≤ −0.0060 **and** its between-seed 95% t-interval (df 4) lies
  entirely below zero. A paired block bootstrap (L = 20, n_boot = 1000, seed 0) is reported as
  evaluation noise only. **Adopts nothing:** the veto axis is closed (`prereg-night-veto-v3.md` §6)
  and the shipped system is frozen, so a failure is reported as a scope limit on the claim, not
  answered with a rule change. Logged before the run.

* **2026-09-27 — R-D1 under `crossmodal26m`** (`docs/prereg-uq-mechanism-ablation-26m.md`,
  `scripts/ablate_uq_mechanism.py --preset crossmodal26m`). `runs/cache_m` (the old full-scale
  checkpoints R-D1 used), conditions `clean`, `fog`, `lowlight`, `glare`, arms S0–S7, paired and
  day+night as R-D1. Only the preset moves. Adds one further inspection of all four development
  runs. **No pohang04 frame is scored.** Logged before the run.

* **2026-09-27 — R-D1 fog score-path decomposition, descriptive** (`scripts/diag_rd1_fog_score_path.py`).
  Re-scores arms S5 and S7 of the `crossmodal26m` R-D1 run on `fog` only, whole system and with
  IR detections emptied, sliced day/night, to locate the +0.0062 fog score-path delta. Same
  frames, caches and seeds as the logged R-D1 run; cannot change its verdict. **No pohang04 frame
  is scored.** Logged before the run.

* **2026-09-27 — Phase 3 corrupted cells, development, descriptive** (`scripts/p3_corrupt_cells.py`).
  Preset `crossmodal26m`, the five Phase 3 systems, VIS conditions `fog`, `lowlight`, `glare`
  (severity 2) with clean IR, VIS draws 941–944, caches `runs/cache_p3dev/`, statistics
  `runs/derived_p3dev/`. Per seed and draw: VIS-only, IR-only, fused as shipped; ship AP; day and
  night separately. Adds one further inspection of all four development runs. **No pohang04 frame
  is scored.**

  *Fixed before the number:* on each cell, the claim "fused ≥ max(VIS, IR)" **fails** if the seed
  mean of the draw-averaged `fused − VIS_only` or `fused − IR_only` is ≤ −0.0060 **and** its
  between-seed 95% t-interval (df 4) lies entirely below zero. **Adopts nothing**; a failure is a
  scope limit on the paper's claim. Logged before the run.

* **2026-09-27 — Phase 3 fog severity 1, development, descriptive** (`scripts/p3_corrupt_cells.py
  --conds fog_s1`; TODO-improvements §D.4, "the interesting middle of the range is severity 1").
  Same systems, draws, arms, day/night split and **fail criterion** as the corrupted-cells entry
  above, for one new VIS condition, `fog` at severity 1, with clean IR. Caches are built by
  `scripts/build_cache_multi.py`, which runs the corruption once per draw for all seeds. It was
  verified bit-identical to `build_cache.py` (fog s2, draw 941, 5 seeds × 40 frames, max |diff| 0).
  **No pohang04 frame is scored.** Adopts nothing. Logged before the run.

* **2026-10-09 — σ re-ranker replication, Phase 3 VIS seeds 0–4, development, registered**
  (`docs/prereg-sigma-rerank-replication-2026-10-09.md`; `scripts/fit_rerank.py --ci-vs`).
  Out-of-fold VIS re-ranker, 4 features vs the same 3 without `sigma_mean_norm`, monotone,
  λ 0.30, ship AP, the 1,200 paired day frames, caches `runs/cache_p3/seed{0..4}/`. First use of
  the re-ranker on any Phase 3 cache. **No pohang04 frame is scored.** Adopts nothing.

  *Fixed before the number:* a seed passes at floor F if σ's paired increment is ≥ F and its
  block interval (L = 20) is above zero. ≥ 4/5 seeds at 0.0060 → REPLICATES; else ≥ 4/5 at
  0.0047 → REPLICATES AT SHIP FLOOR; else FAILS, and the title reverts to the 2026-10-08
  framing. Logged before the run.

* **2026-10-10 — pohang04 corrupted cells re-scored under corruption v2: a SECOND EXPOSURE of
  pohang04, disclosed** (`docs/prereg-p04-v2-rescore-2026-10-10.md`,
  `scripts/holdout_p04_v2_rescore.py`; decided by Laksh 2026-10-10). The v1 corruptions behind the
  ten descriptive cells of the 2026-09-20 look were found broken on 2026-10-09
  (`docs/eval/corruption_v2/README.md`). Re-scores eight corrupted cells (three IR-glare cells are
  not modelled by v2) with the look's systems, preset, calibration injection, draws, statistic and
  bootstrap; only the corruption changes. **The `clean/clean` verdict is not re-scored** and the
  look markers are not written. **pohang04 is no longer held out**: nothing scored on it from here
  on may be called a held-out result. Descriptive, adopts nothing. Logged before the run.

* **2026-10-10 — Phase 3 corrupted cells under corruption v2, development, descriptive**
  (`scripts/build_corruption_v2_dev.py`, `scripts/p3_corrupt_cells.py --cache-root
  runs/cache_p3dev_v2 --stats-root runs/derived_p3dev_v2`). Same systems, draws, arms, day/night
  split and **fail criterion** as the 2026-09-27 corrupted-cells and fog-s1 entries; only the
  corruption changes (v2). The parallel scorer was first shown to reproduce the recorded v1 JSON
  exactly (max |diff| 0 over all six cells). **No pohang04 frame is scored.** Adopts nothing.
  Logged before the run.

* **2026-10-10 — R-D1 under corruption v2, development, descriptive robustness check**
  (`scripts/ablate_uq_mechanism.py --cls 0 --cache-dir runs/cache_m_v2 --bright-dir
  runs/derived_m_v2/brightness --structure-dir runs/derived_m_v2/structure`, both presets).
  `runs/cache_m_v2` is `runs/cache_m` with VIS fog, lowlight and glare (s2, seed 1) rebuilt
  under v2; everything else hard-linked. Cannot change the registered R-D1 verdict, which
  stays the v1 one. **No pohang04 frame is scored.** Logged before the run.

* **2026-10-10 — corruption v2 revised to revision 3 before any v2 cell was scored.** The
  revision-2 night lamp lifted the dark floor of every night frame above the veto's `dark`
  threshold (5th-percentile grey 16 vs 7 for real lights of the same clipped size; 100% vs 0% of
  frames above 10.5), which would have switched the glare/night veto off for a reason absent from
  real night frames. Found by comparing the shipped gate's axes on v2 frames with real TRAIN night
  frames (`scripts/calibrate_lamp_tail.py`), not by any detection score: no v2 AP had been computed
  on any cell when the change was made (the R-D1 v2 runs were stopped before writing output). The
  glare spread's tail exponent was then fitted by night (lamp, real TRAIN lights) and by day (sun,
  the TRAIN sun-entry event, `scripts/calibrate_sun_tail.py`); both give 1.5, and the day refit
  moves AE_STRENGTH from 0 to 0.375. All v2 caches were rebuilt under the new code stamp; the three
  entries above apply unchanged to revision 3.

* **2026-10-10 — two v2 development runs that were not logged before they ran, disclosed here.**
  (a) The v2 sensitivity rows (`scripts/v2_sensitivity.py`, draw 941, one constant moved per row:
  glare AE 0 / AE 1 / intensity ×0.5 / ×2 / revision-2 Lorentzian, fog AE 0 / AE 1, IR_BETA_RATIO
  0.3 / 1.0; `docs/eval/corruption_v2/sensitivity.md`). (b) Table 6 under v2
  (`scripts/ir_hazards_v2.py`; gate statistics and veto rates only, no AP). Both development,
  descriptive, adopt nothing, score no pohang04 frame; neither moved a constant (every v2 constant
  was fixed by its TRAIN fit before any of them ran). The omission was found while writing the
  2026-10-10 handoff.

* **2026-10-10 — the AP cost of the weak-IR fallback under v2, development, descriptive**
  (`scripts/v2_fallback_cost.py`). Table 6 v2 found the fallback vetoing a low-light VIS stream on
  up to 96.3% of day frames when IR is fogged or noisy (§8.2 of the corruption README). This prices
  it: VIS low light s2 (draws as built, 941) against IR fog s1 / s2 / s3 and IR noise s2 / s3 at
  the IR draw 951, five Phase 3 systems, `crossmodal26m`, ship AP, day and night apart; reported
  as VIS / IR / fused, veto rate and fused − max(VIS, IR) with the between-seed t-interval. New
  caches: `runs/cache_p3dev_v2/seed{k}/draw941_951/gauss_ir_paired_{fog_s1,fog_s3,noise_s2,noise_s3}.pkl`.
  Changes no rule and no constant; a fix would be a new rule and needs its own registration.
  **No pohang04 frame is scored.** Logged before the run.

* **2026-10-10 — the IR merge veto under corruption v2, development, descriptive**
  (`scripts/v2_ir_merge_veto.py`). The fallback-cost run above found that with clean VIS and IR fog
  s3 the union falls below VIS alone by day (−0.0096, no veto firing): a near-dead IR stream's boxes
  enter the merge. `crossmodal26m` switched the IR merge veto (drop an unhealthy IR from the merge
  while VIS is healthy) off after it was measured net-harmful on the v1 IR corruptions. This
  re-measures that one switch on the same v2 cells (VIS clean / low light × IR clean, fog s1–s3,
  noise s2–s3; draw 941 / 951; five systems; ship AP, day and night), on against off, paired by
  system. Adopts nothing, changes no preset; a change would need its own registration.
  **No pohang04 frame is scored.** Logged before the run.
  *Addendum, same day, before the run:* the IR merge veto check gains a third arm, the shipped
  preset with the cross-modal support bonus off (`support_gamma` 0), on the same cells, to locate the
  union's loss under a near-dead IR stream (IR fog s3: the merge veto drops IR on 92% of day frames
  and fused is still 0.012 below VIS alone). Descriptive; changes no preset.

* **2026-10-10 — two v1-priced decisions re-checked under corruption v2, development, descriptive.**
  (a) **The veil repair** (`scripts/v2_veil_reprice.py`): `crossmodal26m` makes the veil axis
  conditional on the night arm, a repair priced on v1 fog (whose 21-px blur is what `veil` detected).
  Three arms on the five Phase 3 systems, v2 caches, VIS draws 941–944, cells clean / fog / lowlight /
  glare / fog s1 × day / night, ship AP: shipped; veil unconditional (`veil_requires_night` off, the
  pre-repair rule); veil axis removed. Deltas paired by system on the draw mean. (b) **Stage 1 under
  v2** (`scripts/stage1_crossing.py --cache-dir runs/cache_m_v2 --bright-dir runs/derived_m_v2/brightness
  --structure-dir runs/derived_m_v2/structure`): the registered A/B/C/D crossing re-run with only the
  VIS fog / lowlight / glare caches changed to v2. Cannot change the registered S1-NULL verdict, which
  stays the v1 one; reported as a robustness check. Both adopt nothing and change no preset.
  **No pohang04 frame is scored.** Logged before the runs.

* **2026-10-10 — a candidate repair of the weak-IR fallback, priced on development, descriptive**
  (`scripts/v2_fallback_fix.py`). The veil re-price found the veil axis inert with clean IR under v2,
  acting only inside the fallback (`concentrated OR (dark AND veil)`), which is where the low-light hole
  is. One arm, the veil axis removed (`gini_by_cond` emptied; with clean IR this changed no cell), against
  the shipped preset, on VIS clean / low light / fog × IR clean, fog s1–s3, noise s2–s3, draw 941 / 951,
  five systems, ship AP, day and night; paired by system. Fogged VIS is included because protecting a
  fogged night with damaged IR is what the fallback was built for. Adopts nothing; a change would need its
  own registration. **No pohang04 frame is scored.** Logged before the run.

* **2026-10-10 — two degraded-IR repairs to `crossmodal26m`, registered, development**
  (`docs/prereg-degraded-ir-repairs-2026-10-10.md`, `scripts/v2_degraded_ir_repairs.py`). R1: the
  weak-IR fallback confirms with `concentrated` alone (drops `dark AND veil`). R2: the cross-modal
  support bonus applies only where IR is merge-healthy (`healthy_ir`). R1+R2 is the new preset
  `crossmodal26m_r`; `crossmodal26m` is unchanged and bit-identical. Verdict on the fresh VIS draws
  942–944 / IR 952–954 only (draw 941 informed R1 and is reported descriptively); five Phase 3 systems,
  60 cells (VIS clean/fog/fog_s1/glare/lowlight × IR clean, fog s1–s3, noise s2–s3 × day/night), ship
  AP, paired by system. New IR caches: `runs/cache_p3dev_v2/seed*/draw94{2,3,4}_95*/gauss_ir_paired_
  {fog_s1,fog_s2,fog_s3,noise_s2,noise_s3}.pkl`. **No pohang04 frame is read or scored.** Logged
  before the run (the caches were still building).

* **2026-10-10 — learned late fusion (LLF), registered, development** (`docs/prereg-llf-2026-10-10.md`,
  `scripts/llf.py`). A separate decision layer, not a change to `crossmodal26m` and not a fourth clause
  on the closed night axis: every VIS and IR ship box is re-scored by a gradient-boosted model from
  box, cross-stream and frame-health features, trained on draw 941 / 951 caches of the five Phase 3
  systems (20% of units). It is scored leave-one-run-out on VIS draws 942–944 / IR 952–954, 60 cells,
  ship AP, against max(VIS, IR) and the shipped preset (P1: all 60 cells hold; P2: never worse than
  shipped, better somewhere). Secondary: night blocks, leave one corruption family out, two ablations.
  Model and features fixed in the registration; nothing tuned. **No pohang04 frame is read or scored.**
  Logged before any model was trained.
  *Amendment, same day, before any LLF evaluation number:* a second arm, LLF-Net (a set transformer
  over all candidates of a frame, GPU-trained, fixed hyperparameters, primary folds only), with the
  same features, target, rows, folds, draws and criteria (`prereg-llf-2026-10-10.md` Amendment 1).
  The IR caches for draws 942–944 finished at 20:4x; both scorers were started while the last two were
  still being written, and they wait for each file.

* **2026-10-10 — LLF verdict, then LLF-2 registered.** LLF (`docs/eval/llf/llf_2026-10-10.md`)
  FAILS P1 (33/60 cells; shipped 32) and P2 (17 worse, 25 better). It removes the night rule's cost
  (+0.16 to +0.23) but falls 0.0174 below VIS on clean day. LLF-2 (`docs/prereg-llf2-2026-10-10.md`)
  restricts the learner to stream trust: `is_ir`, monotone `conf`, monotone `x_sup` and `x_higher`,
  and the frame-health features. It is scored on **fresh** draws, VIS 945–947 / IR 955–957, which
  were built for it (`runs/cache_p3dev_v2/seed*/draw94{5,6,7}_95*`). Registered after the LLF verdict
  and before any LLF-Net or LLF-2 number. **No pohang04 frame is read or scored.**

* **2026-10-10 (10:50 PM) — three no-retraining changes, registered, development**
  (`docs/prereg-no-retrain-2026-10-10.md`). E1: checkpoint ensembles (WBF @0.55 of 2 or 5 frozen
  checkpoints per stream, shipped gate on top; `scripts/seed_ensemble.py`). E2: classical image-only
  restoration before detection (method per modality selected by a rule declared in the registration on
  draw 941, every fourth frame, systems 0–2; applied where the shipped gate's own degradation statistic
  fires; `scripts/restore.py`). Both are scored on the fresh draws VIS 945–947 / IR 955–957, which LLF-2
  also uses; each system is judged by its own rule. E3: sixteen single-constant moves of
  `crossmodal26m` priced on development draws 942–944, descriptive; nothing changes on it. Seen before
  writing: one E1 smoke run (draw 941, system 0, three VIS cells) and one E2 smoke unit (draw 941, VIS
  fog, 15 frames). Nothing on draws 945–947. **No pohang04 frame is read or scored.**
  *Amendment 1, 11:00 PM, before any score on draws 945–947:* E4, the same streams fused with no gate
  (`veto=None`), and E5, five-checkpoint ensembles of gated-restored streams fused with no gate. Both
  are separate systems; `crossmodal26m` stays frozen. Prompted by an E3 smoke on draw 942 (system 0,
  IR fog s3), stated in the amendment.
  *Amendment 2, 11:20 PM, before any score on draws 945–947:* E6, three new gates (`selfconf`
  output-space health veto, `calunion` health-conditioned isotonic calibration then concatenation,
  `proben` Bayesian pair fusion; calibrators fitted on draw 941 leave-one-run-out); E7, LLF-2 plus 32
  PCA components of the frozen backbones' frame embeddings, XGBoost on the GPU, per system.
  *Amendment 3, 11:30 PM, before any score on draws 945–947:* E8, horizontal-flip test-time
  augmentation per checkpoint (WBF 0.55), alone (`tta`) and on top of E5 (`full_tta`).
  *Amendment 4, 1:25 AM 2026-10-11, before any score of its arms on draws 945–947:* E9, tiered
  concatenation (`lex`: every VIS ship box above every IR ship box, fused ≥ VIS by construction) and
  demotion instead of deletion (`demote`: the shipped output first, the vetoed stream as the tail,
  fused ≥ shipped by construction), plus `ens5_lex` and `full_tta_lex`. Prompted by a smoke on draw 941
  (stated in the amendment, table in `docs/eval/no_retrain/smoke_941_lex.md`); two arms tried there
  (`lex_sc`, `udemote`) are not registered. Seen on 945–947 before writing: the first two E7 jobs (945,
  IR clean, systems 0–1). Note (1:30 AM): the VIS-clean × IR-clean cells do not depend on the draw, so
  every 941 smoke in Amendments 1–4 showed those two cells' 945–947 values; the other 58 were unseen.

* **2026-10-10 (11:59 PM) — degraded-IR repairs verdict** (`docs/eval/corruption_v2/degraded_ir_repairs.md`).
  R1 ADOPTED (0 cells harmed at 0.0014, 5 gaining ≥ 0.0060, claim 35/60 vs shipped 32); R2 rejected (8
  harmed); R1+R2 rejected (6 harmed). Per the registration `crossmodal26m_r` is redefined to R1 alone
  (`src/uqfusion/eval/ctx.py`). Draws 942–944 were already exposed to LLF; this verdict adds no new
  exposure. **No pohang04 frame was read or scored.**

* **2026-10-11 (7:25 AM) — no-retraining verdicts and the E3 audit, development**
  (`docs/eval/no_retrain/`, `docs/eval/llf/llf2_2026-10-10.md`, `docs/eval/restore/`,
  `docs/eval/corruption_v2/constants_audit.md`). Every arm of `prereg-no-retrain-2026-10-10.md`
  (Amendments 1–4) and LLF-2 has now been scored on VIS 945–947 / IR 955–957, so **those draws are
  spent**: a further registration needs draws no system here has been scored on. E3 (descriptive, draws
  942–944, no new exposure): every fusion constant stands. One move meets E3's amendment bar, the weak-IR
  fallback switched off (25 cells better at 0.0060, none worse at 0.0014, claim fails 28 → 5). It was not
  registered: it sits on the closed night axis, and the gate-free `union`, already scored on the fresh
  draws, dominates it on 942–944 (30 better, claim fails 1). End-to-end throughput was timed on 500
  development frame pairs (`docs/eval/e2e_fps_2026-10-11.md`); nothing was scored. **No pohang04 frame was
  read or scored.**

* **2026-10-11 (10:16 AM, before any build or score) — pohang04, third scoring, registered**
  (`docs/prereg-p04-heldout-gatefree-2026-10-11.md`). Laksh decided on 2026-10-11 that pohang04 is scored
  whenever a result needs held-out validation. Systems: `shipped` (reference), `union` and `full_tta`
  (primary), `ens5_union` and `lex` (descriptive), with every arm's VIS-only and IR-only streams; the
  look's clean/clean cell and the re-score's eight v2 cells, draws 941–944 / 951–954. New pohang04 caches
  (flipped views, NLM-restored IR on flagged frames) are built for `full_tta`. Nothing is fitted on
  pohang04. `LOOK_TAKEN.json`, its mirror and the look's verdict are untouched.
  *Amendment 1, 10:48 AM, before any pohang04 cache is built:* the new views are computed in fp16
  (priced on development: the `full_tta` stream moves by at most 0.0003 ship AP). Throughput benchmarks
  between 10:20 and 10:47 AM ran the detectors on pohang04 frames for timing only; nothing was kept or
  scored.

* **2026-10-11 (2:15 PM, before any TT-BN number on draws 942–944 or any TT-BN detection on pohang04) —
  N4, IR-vouched test-time BatchNorm adaptation of the VIS detector, registered**
  (`docs/prereg-ttbn-2026-10-11.md`, `scripts/ttbn.py`). The method:
  * BN-mix statistics, N = 16, over blocks of 8 consecutive frames.
  * Used only where VIS reports itself unhealthy and IR is healthy and reads day.
  * Fused by E4 `union`.

  The development verdict is on draws 942–944 / 952–954, five systems, 60 cells. pohang04 (the
  gate-free registration's cells, draws 941–944) is scored only if development passes, in fp16 only if
  a development fp16 check passes.

  Seen before writing:
  * The screen on draw 941, systems 0–1, ungated, N ∈ {16, 64}.
  * Label-free gate rates on development draws 943/944.
  * Label-free gate rates on pohang04, from its existing frame-statistics files. No pohang04 detection
    or label was read.

  The pohang04 rates are why the gate's IR health term is the merge bound, not the screen's authority
  bound (which passes 9.5% of clean pohang04 IR frames); this is disclosed in the registration.
  Development screens N1 (self-validated support), N3 (σ-weighted ensemble coordinates) and N5
  (asymmetric support / cross-tail demotion) were run on draw 941 only and are not registered.
  *Amendment 1, 2:49 PM, before any fp16 TT-BN detection:* the fp16 build computes the batch statistics
  with `torch.batch_norm_stats`; the fp16 development check prices it together with the precision.

* **2026-10-11 (5:05 PM, before any number for either arm) — two descriptive lex arms registered.**
  * Gate-free registration, Amendment 2: `full_tta_lex` on pohang04's nine cells.
  * N4 registration, Amendment 2: `ttbn_lex` on development draws 942–944 and on pohang04.

  Both use E9 `lex` ordering (every VIS ship box above every IR ship box, no merge) on that system's
  streams. **No verdict is registered for either.** Appending boxes below every other box never lowers
  AP, so each arm is at least its VIS stream. Those streams were already seen:
  * pohang04 clean/clean `full_tta` VIS 0.2946;
  * every cell's `ens5_union` VIS;
  * development gated VIS streams of N4;
  * the pohang04 lowlight/clean partial of N4.

  No corrupted-cell `full_tta` value has been displayed. Both arms are computed from records the
  registered builds already produce. No new pohang04 inference.

* **2026-10-11 (5:15 PM, before any detection, frame statistic or label read on these frames) — night
  test on pohang01's test split, registered** (`docs/prereg-p01-nighttest-2026-10-11.md`).
  * Frames: 1,109 pairs whose VIS and IR frames are both in their test splits.
  * Arms: `shipped` (reference), `union` (primary: P-N1 Δ vs `shipped`, P-N2 claim), `lex` and
    `ens5_union` (descriptive). Clean frames only, five systems, development calibration injected.
  * First exposure of these frames to any fusion system: a scan of the 263 detection-cache directories
    under `runs/` found none of them.
  * The detectors trained on the same run's train split, in interleaved blocks; this is disclosed as
    the test's limit.

* **2026-10-11 (5:23 PM, before any N4g or N4c number on draws 942–944 and before any N4c detection)
  — two N4 variants registered** (`docs/prereg-ttbn-variants-2026-10-11.md`).
  * **N4g** adds a glare term to the VIS side of N4's gate: VIS `mean > 117.92` AND `std > 62.31`, each
    the maximum over clean development day frames.
  * **N4c** normalises each block with the previous block's statistics: no look-ahead.

  Both have development verdicts only, on draws 942–944 and the 60 cells of N4.

  Seen before writing:
  * Draw 941, label-free: the frame statistics and how they separate glare.
  * The draw-941 gated screen with `p95 > 220` as the glare term (glare day +0.0170/+0.0267).
  * pohang04, label-free: that `p95 > 220` and `std > 62.31` fire on 55% and 22% of its clean frames.
    This is why the registered term is the conjunction, which fires on 0.10%.

  No pohang04 detection or label was read.

* **2026-10-11 (5:28 PM, before any N4F detection) — N4 on top of `full_tta` (N4F), registered**
  (`docs/prereg-ttbn-fulltta-2026-10-11.md`, `scripts/ttbn_fulltta.py`).
  * Method: `full_tta` with every VIS view (unflipped and flipped) BN-mix-adapted where N4's gate fires.
  * Development: draws 945–947, N4's 60 cells, moving-block intervals; second use of those draws.
  * pohang04: only if development passes, on N4's cells.

  Nothing about N4F has been seen. Already seen: `full_tta`'s development values on 945–947 and its
  pohang04 clean/clean value; N4's development verdict and pohang04 partials.
  **This departs from the 7:25 AM entry that declared 945–947 spent.** It is disclosed in the
  registration, together with why: fresh draws for `full_tta` cost hours of GPU. A development pass
  there is weaker evidence, and pohang04 decides.

* **2026-10-11 (5:38 PM) — night test on pohang01's test split, scored** (`docs/eval/night_p01test_2026-10-11.md`).
  **P-N1 REPLICATES** (`union` − `shipped` +0.0273 [+0.0154, +0.0400]); **P-N2 PASS** (`union` claim
  −0.0010 [−0.0049, +0.0031]).

  Levels are far below development night: VIS 0.0516, `union` 0.0506, `shipped` 0.0233.

  Post hoc, descriptive:
  * The test pairs are two ~52 s stretches, and development night is two others.
  * VIS AP swings 9× between stretches (development 0.14–0.35, test 0.04–0.10).
  * Night levels are therefore episode properties.
  * The L = 20 block intervals cannot see between-episode variance.

  The diagnostics read the test frames' label counts and box areas, and per-stretch APs. They also read
  draw 941's development night AP per stretch (systems 0–2).

* **2026-10-11 (6:21 PM) — night results by episode, descriptive, post hoc** (`docs/eval/night_episodes_2026-10-11.md`,
  `scratchpad/night_episodes.py`). Not part of any registration.
  * Read: every night-test arm's ship AP per test stretch (two), the `union` − `shipped` interval at block
    lengths 20 to 500, and development night per stretch (draw 941 clean caches, five systems).
  * `union` is above `shipped` in all four stretches: test +0.0270 and +0.0589, development +0.1786 and
    +0.2822.
  * The interval's lower bound stays above zero at every block length (+0.0080 at L = 500). With two test
    stretches, no episode-level interval exists.
  * VIS ship AP spans 0.0391 to 0.3540 across the four stretches.
  * `ens5_union` is below `union` on test A (0.0327 against 0.0406) and above it on test B (0.1411 against
    0.1095).

* **2026-10-11 (6:21 PM) — N4-mix screened on draw 941 and dropped; not registered** (`scratchpad/screen_n4mix.py`).
  N4-mix replaces N4's adapted VIS record, on N4's gated frames, with a plain WBF (0.55) of the plain and
  the adapted record of the same checkpoint. Systems 0 and 1; VIS clean, lowlight, fog, fog_s1 and glare;
  IR clean.
  * On the six gated day cells, N4-mix is below N4 in five (lowlight +0.0490/+0.0677 against
    +0.0635/+0.0767 over `union`) and above it in one (fog, system 0: +0.0093 against +0.0087).
  * Merging on every frame, with no gate, is safer at night than adapting on every frame (clean night
    −0.0033/−0.0056 against −0.0074/−0.0206), but still loses on clean by day (−0.0143/−0.0016).
  * Side finding: on fog_s1 by day, ungated adaptation gains +0.0340/+0.0258, while N4 gains
    +0.0097/+0.0061; its gate fires on 62% of those frames.

  Only draw 941, the screening draw, was read. No verdict draw (942 to 947) and no pohang04 frame.

* **2026-10-11 (6:21 PM, before any flipped or restored view of the night-test frames is built) — night test
  Amendment 1: `full_tta` and `full_tta_lex`, descriptive** (`docs/prereg-p01-nighttest-2026-10-11.md`).
  * Definitions are P-F's and pohang04 Amendment 2's. The flipped and restored views are built in fp16, as
    pohang04's were.
  * No verdict: every other night-test arm has been seen, including `ens5_union`.
  * Development night: `full_tta` 0.3192, `full_tta_lex` 0.3010, `union` 0.2895.

* **2026-10-11 (6:38 PM) — night test Amendment 1 scored, descriptive, no verdict**
  (`docs/eval/night_p01test_amend1_2026-10-11.md`).
  * `full_tta` 0.0520: +0.0287 [+0.0156, +0.0393] over `shipped`, +0.0014 [−0.0065, +0.0068] over `union`;
    claim +0.0013 [−0.0022, +0.0040].
  * `full_tta_lex` 0.0535: +0.0029 [−0.0053, +0.0082] over `union`; claim +0.0028 [+0.0017, +0.0042].
  * Development night had `full_tta` +0.0297 over `union`. The test-split night does not reproduce it.
  * By stretch: test A 0.0355 against `union`'s 0.0406, test B 0.1509 against 0.1095.
  * The IR health check fails on 274 of the 1,109 night frames, which were NLM-restored. All five
    system contexts reproduce the flag.

* **2026-10-11 (6:55 PM) — N4g and N4c development verdicts** (`docs/eval/ttbn/ttbn_variants_dev_2026-10-11.md`).
  * **N4g PASSES.** Glare by day: +0.0240 [+0.0178, +0.0302] with IR clean, +0.0130 with IR fog_s1. The
    worst cell is glare/fog_s3/day, −0.0008 [−0.0014, −0.0002], inside the harm floor. Non-glare cells
    equal N4's (checked). No pohang04 test is registered for N4g.
  * **N4c FAILS**: C-H on 10 cells, with low-light day at −0.1637 [−0.1900, −0.1375].

* **2026-10-11 (7:14 PM, before any N4c2 detection on 942–944) — N4c diagnosed; N4c2 registered**
  (`docs/prereg-ttbn-variants-2026-10-11.md`, Amendment 1; `scripts/ttbn_causal2.py`).
  * Diagnosis, post hoc, from N4c's own development records: the carried statistics were measured under
    the previous block's stale normalisation. The first block that uses them collapses (mean max
    confidence 0.212 against N4's 0.750 over 1,200 day frames), and by block 25 no box is returned. N4c's
    FAIL stands; it measures that construction, not causality.
  * N4c2 normalises block b with the statistics N4 measured on block b−1 (closed-loop recording pass);
    fp16 with fused statistics.
  * Mechanics checked on draw 941 (screen draw, system 0, low light, five blocks, no label read): mean max
    confidence tracks N4 within 0.05.
  * Development on 942–944 is a second use of those draws, disclosed. pohang04 would need its own
    registration after the development verdict.

* **2026-10-11 (7:50 PM) — N4F development verdict: PASSES** (`docs/eval/ttbn/ttbn_fulltta_dev_2026-10-11.md`).
  * Draws 945–947, second use, disclosed at registration.
  * `full_tta` reproduces its no-retrain values in all 60 cells (|difference| < 5e-5).
  * Δ = `full_tta_ttbn` − `full_tta`. No cell is worse. Better cells:
    * low-light day +0.0696 [+0.0620, +0.0757];
    * low light with IR fog s1 +0.0385;
    * fog +0.0130;
    * fog s1 +0.0096;
    * low light with IR fog s2 +0.0071.
  * P-FT, the registered pohang04 test, follows. Its flipped adapted views are built after N4's pohang04
    build ends, from the same blocks. Its scoring waits for the full_tta build's draw 944.

* **2026-10-11 (8:22 PM and 8:38 PM) — screens on draw 941: multi-resolution TTA and IR-vouched restoration**
  (`runs/novel/multires/score.log`; session-scratchpad `screen_restore_gated.py`, `screen_restore_sel.py`).
  Systems 0 and 1, IR clean, draw 941 only. No verdict draw (942–947) and no pohang04 frame was read.
  * **Multi-resolution: mixed, not registered.** The 640 record WBF-merged with a 1280 one gains on clean
    (day +0.0138/+0.0234, night +0.0056/+0.0131), is mixed on low light, and loses on fog (night
    −0.0084/−0.0042).
  * **Restoration on N4's gate: positive.**
    * Auto-exposure on low light: +0.0953/+0.0965, where N4 gives +0.0635/+0.0767.
    * Dark-channel dehaze: +0.0134/+0.0153 on fog and +0.0176/+0.0129 on fog_s1.
    * "Dark" (linear luminance < 0.18) splits the screened conditions exactly.

* **2026-10-11 (8:40 PM, before any restored-frame detection on 942–944 or pohang04) — N4ae registered**
  (`docs/prereg-ttbn-restore-2026-10-11.md`, `scripts/ttbn_restore.py`).
  * Method: on N4's gate, dark frames get the auto-exposed frame's detections instead of N4's adapted
    ones; everything else is N4.
  * Development: draws 942–944 with N4's rules. These draws already carry N4's, N4g's, N4c's and N4c2's
    verdicts.
  * pohang04 only after a development pass, on P-T's cells and replicates (P-A).
  * Dehaze is a descriptive development arm only: development has no VIS noise, rain or blur condition,
    and pohang04 does.
  * Seen: the 941 screens above, N4's verdicts and P-T's partials.
  * Corrected at 8:47 PM, before any restored-frame detection: this entry and the one above were first stamped
    8:58 PM and 8:52 PM by mistake (the files were written at 8:40 PM and the screen ended at 8:38 PM).
    The screen's glare rows, finished before the prereg was written, were added to its seen-list: under
    N4g's gate, dehaze gained +0.0057/+0.0163 on glare where N4g's adaptation gained +0.0170/+0.0267.

* **2026-10-11 (9:11 PM) — N4 on pohang04: P-T REPLICATES** (`docs/eval/ttbn/ttbn_p04_2026-10-11.md`; registered
  in `docs/prereg-ttbn-2026-10-11.md`). A registered held-out look at pohang04's VIS/IR frames, fp16 adapted
  records (development fp16 check passed 3:14 PM), the gate-free scoring's 1,000 moving-block replicates.
  * No cell worse. Better (Δ ≥ +0.0060, interval above zero): noise_s2/clean +0.1823 [+0.1733, +0.1916],
    fog/clean +0.0370 [+0.0328, +0.0408], lowlight/clean +0.0186 [+0.0154, +0.0219].
  * rain_s2/clean +0.0036 (gate 0.186); blur_s3/clean +0.0001 (gate 0.092). The gate fires on no frame of
    clean/clean, clean/noise_s2 or clean/fog_s2 and on almost none of clean/blur_s2 (Δ +0.0000).
  * Descriptive: the claim (`union_ttbn` − max(adapted VIS, IR)) is negative on lowlight (−0.0054) and
    clean/blur_s2 (−0.0094, inherited from `union`). `ttbn_lex` − `lex` gives the same gains, and its claim
    is positive in every scored cell.

* **2026-10-11 (9:41 PM) — N4c2 development verdict: PASSES** (`docs/eval/ttbn/ttbn_causal2_dev_2026-10-11.md`).
  Draws 942–944 (second use, disclosed in Amendment 1). No cell worse. Better: fog/clean/day +0.0094,
  fog_s1/clean/day +0.0094, lowlight/clean/day +0.0668 [+0.0559, +0.0777], lowlight/fog_s1/day +0.0327.
  Price of causality against N4: −0.0038 [−0.0045, −0.0031] on lowlight/clean/day, within ±0.0003 on fog.

* **2026-10-11 (9:45 PM, before any N4c2 detection on pohang04) — N4c2's pohang04 test registered (P-C2)**
  (`docs/prereg-ttbn-variants-2026-10-11.md`, Amendment 2). P-T's cells, statistic, replicates and
  thresholds on Δ = `union_ttbnc2` − `union`. Only gated blocks and their predecessors' recording passes run;
  equivalence to the full grid is checked on development first. Seen: N4c2's development report, P-T and
  the gate-free pohang04 numbers. N4ae's development scoring was running; none of its numbers had been read.

* **2026-10-11 (9:56 PM, before any N4c2 detection on pohang04) — P-C2's equivalence check failed as first
  written; diagnosed; changed** (`docs/prereg-ttbn-variants-2026-10-11.md`, Amendment 2 note).
  * Sparse records against build-dev's stored ones: 39 of 880 frames differ in box count. A full-grid
    re-run differs from the stored records just as much.
  * In one process, sparse and full grid are bit-identical on all 880 frames. The variation is fp16/cuDNN
    nondeterminism across processes.
  * The check now requires bit-exact agreement within one process. Development frames only (draw 942,
    low light); no label was read.

* **2026-10-11 (10:01 PM) — N4ae development verdict: PASSES** (`docs/eval/ttbn/ttbn_restore_dev_2026-10-11.md`).
  Draws 942–944 (fifth use, disclosed in the registration). No cell worse. Better: fog/clean/day +0.0091,
  fog_s1/clean/day +0.0094, lowlight/clean/day +0.0871 [+0.0734, +0.1008], lowlight/fog_s1/day +0.0451,
  lowlight/fog_s2/day +0.0096.
  * N4ae − N4: +0.0165 [+0.0053, +0.0278] on lowlight/clean/day. On every gated fog frame (none dark) N4ae
    is N4 (Δ 0).
  * Dark = all gated low-light frames and one gated fog_s1 frame (draw 944).
  * Descriptive `sel` (dehaze where not dark): above N4 on fog by up to +0.0058 (fog_s1/clean/day
    [+0.0000, +0.0116]); not registered for pohang04.
  * P-A's pohang04 build is queued for the `full_tta` build's GPU slot (`runs/holdout_p04/q6_n4ae_p04.sh`,
    logged 10:01 PM).

* **2026-10-11 (11:17 PM) — gate-free systems on pohang04 complete: P-U PASS, P-F REPLICATES**
  (`docs/eval/holdout_p04_gatefree_2026-10-11.md`; `docs/prereg-p04-heldout-gatefree-2026-10-11.md`). Draw 944's
  `full_tta` views finished at 10:51 PM and were scored by 11:12 PM.
  * `full_tta` − `shipped`: better in all 9 cells. Smallest: noise_s2/clean +0.0083 [+0.0067, +0.0108]. Largest:
    rain_s2/clean +0.0325, lowlight/clean +0.0324, clean/clean +0.0243 [+0.0199, +0.0278].
  * Descriptive, `full_tta_lex` − `full_tta`: better on 5 cells (interval above zero) and worse on
    blur_s3/clean (−0.0016 [−0.0027, −0.0006]). Its claim (fused − max(VIS, IR)) is positive in all 9 cells.
    `full_tta`'s claim is negative but above the −0.0060 failure margin on lowlight (−0.0027) and
    clean/fog_s2 (−0.0024).

* **2026-10-12 (1:48 AM) — N4ae on pohang04: P-A REPLICATES** (`docs/eval/ttbn/ttbn_restore_p04_2026-10-11.md`;
  registered in `docs/prereg-ttbn-restore-2026-10-11.md`). P-T's cells and replicates.
  * Against `union`: no cell worse. Better: noise_s2 +0.1820 [+0.1732, +0.1913], fog +0.0368 [+0.0327,
    +0.0405], lowlight +0.0259 [+0.0231, +0.0285].
  * Descriptive, against N4:
    * lowlight +0.0073 [+0.0046, +0.0099]: auto-exposure beats adaptation on held-out low light, as on
      development.
    * rain_s2 −0.0036 [−0.0048, −0.0024]: every gated rain frame (9,291 over 4 draws) is dark, so
      auto-exposure replaced N4's adaptation there and lost N4's rain gain. N4ae on rain is +0.0000 against
      `union`.
    Development had no rain condition, as the registration disclosed.

* **2026-10-12 (2:41 AM) — P-FT scoring crashed on a frame-identity check; fixed; resumed.** At 2:38 AM
  `ttbn_fulltta.py score-p04` failed. pohang04's flipped `full_tta` views carry no `image_path` field, and
  `_adapt` asserted on it.
  * `_adapt` now identifies those frames by the raw view's path at the same index. The views are
    index-aligned to pohang04's frame list. The scoring arithmetic is unchanged.
  * No P-FT number had been produced: the crash came before any presort of a gated cell was written.
  * Resumed by `runs/holdout_p04/q9_pft_resume.sh`.

* **2026-10-12 (3:13 AM) — N4F on pohang04: P-FT REPLICATES** (`docs/eval/ttbn/ttbn_fulltta_p04_2026-10-11.md`;
  registered in `docs/prereg-ttbn-fulltta-2026-10-11.md`).
  * Against `full_tta`: no cell worse. Better: noise_s2 +0.2174 [+0.2079, +0.2265] (0.0206 → 0.2380), fog
    +0.0445 [+0.0401, +0.0488] (0.0586 → 0.1031), lowlight +0.0190 [+0.0152, +0.0232].
  * Also rain_s2 +0.0033 [+0.0022, +0.0047] and blur_s3 +0.0000. The gate fires on no clean-VIS cell.
  * Adaptation and the ensemble stack on held-out data. `full_tta_ttbn` is the highest-scoring system
    measured on all five corrupted-VIS cells. On the four clean-VIS cells `full_tta_lex` is higher by
    0.0021–0.0035.

* **2026-10-12 (4:47 AM) — N4c2 on pohang04: P-C2 REPLICATES** (`docs/eval/ttbn/ttbn_causal2_p04_2026-10-11.md`;
  registered in `docs/prereg-ttbn-variants-2026-10-11.md`, Amendment 2). P-T's cells and replicates.
  * Against `union`: no cell worse. Better: noise_s2 +0.1820 [+0.1730, +0.1913], fog +0.0363 [+0.0323,
    +0.0401], lowlight +0.0180 [+0.0148, +0.0212]. Also rain_s2 +0.0035.
  * Price of causality against N4: −0.0003 (noise), −0.0007 (fog), −0.0006 (lowlight), −0.0001 (rain).
  * Built with the sparse passes checked bit-exact against the full grid at 10:03 PM. Finished streams were
    scored during the build (`q7_c2_partial_score.sh`).
  * This closes the overnight pohang04 queue: P-A, P-FT and P-C2 all REPLICATE.

* **2026-10-12 (4:48 AM) — overnight session closed.** No pohang04 scoring is running. Boot loop stopped
  (`runs/holdout_p04/boot_loop.stop`). Handoff: `docs/handoff-2026-10-12.md`. The laptop was shut down at
  Laksh's request.

* **2026-10-12 (9:36 AM) — N4g's pohang04 test registered: P-G, a new glare/clean cell**
  (`docs/prereg-ttbn-glare-p04-2026-10-12.md`). Asked by Laksh (handoff of 10-12, §4 item 4).
  * New pohang04 cell: VIS v2 glare at severity 2, draws 941–944, IR clean. None of its detections or frame
    statistics existed when the registration was written.
  * Seen first: every development and pohang04 number to date. Also, label-free, where N4g's gate differs from
    N4's in the nine registered cells: 10 (cell, draw) pairs in clean/blur_s2, clean/fog_s2 and fog/clean.
  * **Correction:** the N4g registration (10-11) said the glare term adds a gated pohang04 frame only in VIS
    fog on draw 941. That is incomplete. The registration lists all of them. N4g's development verdict is
    unaffected.
  * Boot-loop stop file `runs/holdout_p04/boot_loop.stop` deleted at 9:35 AM (handoff §2); no loop is running.
  * **9:42 AM — P-G's development fp16 check passed.** N4g, glare/clean/day, draw 942, five systems: fp16
    0.19543 against fp32 0.19564 (Δ −0.00021, tolerance 0.0010). The fp32 re-score equals the stored development
    scores exactly. So N4g's new pohang04 blocks are built in fp16 (`runs/novel/ttbn_variants/glare_fp16_check.json`).
    No pohang04 glare detection existed at this time.
  * **9:45 AM — times corrected.** The two entries above were first stamped 9:45 AM, 9:38 AM and 10:12 AM by
    mistake. They are now set from the files' modification times: registration 9:36 AM, fp16 check 9:42 AM.
    The order of events is unchanged. The plain glare build started at 9:43 AM, after both.

* **2026-10-12 (11:46 AM) — N4g on pohang04: P-G REPLICATES** (`docs/eval/ttbn/ttbn_glare_p04_2026-10-12.md`;
  registered in `docs/prereg-ttbn-glare-p04-2026-10-12.md`). Builds started at 9:43 AM; scored 11:26 AM;
  bootstrap done 11:38 AM.
  * glare/clean (new cell): N4g's gate fires on 81.0% of frames and N4's on 1.9%. Δ vs `union`
    +0.0309 [+0.0290, +0.0327] (0.1417 → 0.1726). N4 alone +0.0000. Claim +0.0004 [−0.0012, +0.0023].
  * No cell is worse. In the three existing cells where N4g's gate adds frames (2–3 frames per draw), Δ against N4
    is +0.0000. The other six cells equal P-T by construction (gates checked equal on every draw).
  * Descriptive, glare/clean: `ttbng_lex` − `lex` +0.0333 [+0.0312, +0.0354]; `ttbng_lex` 0.1762; `union` equals
    `shipped` (the veto does not fire).
  * Speed changes, none touching the arithmetic:
    * Draws 942–944 ran with the builder's CPU thread pools capped at 2. They were checked bit-identical on
      draw 941 (5 caches × 400 frames, max |diff| 0; statistics equal).
    * Scoring used the verified fast paths (`UQF_FAST_*`), as P-T did. Three existing-cell jobs had been scored
      without them before the restart; the fast paths are byte-identical.
    * A second concurrent cache build was tried for 2 minutes and stopped: it was slower (12.4 against 14.5
      frames/s). It wrote nothing.

* **2026-10-12 (11:44 PM) — N4Fg registered: N4F with N4g's glare term, development and pohang04 glare/clean
  (P-FG)** (`docs/prereg-ttbn-fulltta-glare-2026-10-12.md`). Asked by Laksh to fill the "not scored" N4F glare
  entry of the paper's Table 3 and Fig. 2.
  * New exposure of pohang04 (the ninth scoring) if development passes: `full_tta`, N4F and N4Fg on glare/clean,
    plus N4Fg in the three existing cells where N4g's gate adds frames. No ensemble detection existed on glare/clean
    when the registration was written.
  * Development draws 945–947 get a third use (N4F's, the no-retrain verdict's), disclosed.
  * Seen first: every development and pohang04 number to date, and, label-free, where N4g's gate differs from N4's
    on 945–947 (glare only) and which adapted blocks are missing (listed in the registration).
  * **Time corrected.** The registration and this entry were first stamped 11:45 PM; the registration commit
    (`483cc5b`) is 11:44:33 PM, and they now say 11:44 PM. The development build started after that commit
    (11:44 PM, `runs/novel/ttbn_fulltta_glare/dev_chain.log`). No N4Fg number existed at either time.
  * **11:47 PM — Amendment 1:** at Laksh's request pohang04 is built and scored after the development report
    whatever its verdict (integrity checks still stop the queue). Written before any N4Fg number existed.
  * **11:56 PM — N4Fg development fp16 check:** full_tta_ttbng glare/clean/day draw 945: fp32 0.21928 fp16 0.21922 diff -0.00006 -> PASS: fp16. File `runs/novel/ttbn_fulltta_glare/fp16_check.json`.
* **2026-10-13 (12:06 AM) — N4Fg development: PASSES** (`docs/eval/ttbn/ttbn_fulltta_glare_dev_2026-10-12.md`; registered in
  `docs/prereg-ttbn-fulltta-glare-2026-10-12.md`). Draws 945–947, third use. No cell worse than `full_tta`: PASS. Cells where the glare term adds gated frames at or above −0.0014 against N4F: PASS.
  * pohang04 starts now whatever the verdict (Amendment 1).
  * **12:06 AM — P-FG: pohang04 builds started** (plain flipped glare views, then N4Fg's flipped adapted blocks).
  * **1:12 AM — P-FG: pohang04 builds done; scoring started** (`UQF_FAST_*`). Scored 1:32 AM; bootstrap done 1:44 AM.
* **2026-10-13 (1:44 AM) — N4Fg on pohang04: P-FG REPLICATES** (`docs/eval/ttbn/ttbn_fulltta_glare_p04_2026-10-12.md`; registered in
  `docs/prereg-ttbn-fulltta-glare-2026-10-12.md`, Amendment 1). Ninth scoring of pohang04. Development: PASSES.
  * glare/clean: N4Fg 0.1967 against `full_tta` 0.1656, Δ +0.0311 [+0.0286, +0.0332]; N4g's gate fires on 81.0% of
    frames. N4F (N4's gate, 1.9%) equals `full_tta` (+0.0000).
  * No cell worse than N4F: in clean/blur_s2, clean/fog_s2 and fog/clean (where the glare term adds frames) N4Fg − N4F
    is +0.0000; the six other cells equal P-FT by construction (gates checked equal on every draw).
  * Descriptive: N4Fg − N4g (one pass, P-G) +0.0240 [+0.0216, +0.0261]. N4Fg is the highest-scoring system measured on
    glare/clean (`ttbng_lex` 0.1762, N4g 0.1726).
  * 1:45 AM: report, ledger and Fig. 2 committed and pushed by the queue (`19ede9d`).
