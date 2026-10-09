# Class-set audit: Table 3a, §6.3, Figure 5, Table L and §6.8 of PAPER_DRAFT2

**Measured 2026-10-08/09.** Covers every number §5.5 says was never re-audited for class set: §6.3 (Table 3a, the paragraph after it, Figure 5) and §6.8 (Table L, the two paragraphs after it). Each number is traced to its source record and the script and metric key that wrote it. Macro rows that can be re-scored from cached detections were re-scored on ship AP (class 0), after the unchanged macro path reproduced the recorded number. **No pohang04 frame was scored.** The paired val manifest holds pohang00–03 only (2,232 rows: p00 836, p01 1,032, p02 247, p03 117). The two records that already include pohang04 (`rerank_loro_day.md`, `oracle_headroom_day.md`) were read, not re-run. PAPER_DRAFT2.md is not edited and nothing is committed.

## Verdict

**Class set changes three Table L rows, the oracle numbers and the small-object share; Table 3a rows a and b change for a different reason, the comparison bar.**

* **Table 3a / Figure 5.** Stages a and b were ranked against IR alone. Against max(VIS, IR), the table's stated bar, their worst cell is lowlight/day at −0.0180 (ship), the same as stage c. "Later found within CI" is wrong: that interval excludes zero.
* **Table L, class set.**
  * The VIS re-ranker reads +0.0063 [+0.0030, +0.0094] out of fold on ship, against +0.0037 on the macro. It clears 0.0060 by 0.0003.
  * Two-checkpoint WBF merging flips to positive: +0.0069 [+0.0011, +0.0131] at IoU 0.55, the best of six arms. The macro read it as a loss because the merge costs buoys.
  * Within-stream dedup hurts only at merge IoU ≥ 0.80.
  * The oracle is 0.3686 → 0.4653 on ship.
* **Unchanged on ship:**
  * σ-weighted WBF, TTA merging, temporal support and top-k.
  * Every row that was already ship.
  * Soft-NMS, whose failing cell is at night, where macro ≡ ship.
* **Registered on the macro, kept on the macro:** the constants re-price. Its pre-registration excludes per-class splits, so "all stand" and veil-off −0.0416 are verdicts on their registered metric. A ship re-score would be descriptive only. Its two runs (macro reproduction and ship) died overnight at draw 1/7 and were not re-run.
* **Not re-scorable:** day-only re-ranking. Its substrate includes pohang04.
* **Incidental errors, not class set** (§4):
  * The ¶ after Table 3a quoted yolo26s numbers as the shipped detector's.
  * The support multiplier's interval is frame-level and spans zero when widened.
  * The two-sided veto's −0.0810 is on clean-IR night.
  * σ in the fusion score costs 0.0101 on TUNE.
* **Bears on the headline:** the re-ranker's four inputs are `conf, log_conf, conf_over_frame_median, sigma_mean_norm`. Without σ the same arm gives +0.0014 [−0.0020, +0.0044]; σ's paired increment is +0.0050 [+0.0038, +0.0064] (§8).

**Verification (main session, 2026-10-09).** Every macro reproduction in §2 was re-compared line by line, with no line missing. Every changed number above was read back from its raw output: `x_fusion_ci.md` §4, `final_system.md` §2, `rerank_loro_ship_fixedarm`, `checkpoint_ensemble_ship`, `within_modality_ship`, `oracle_headroom{,_day}.md` §2, `sigma_score_26m.md`, `final_26m_grid_v2.md`, `change_impact_2026-09-09_v4.md` and `final_system_crossmodal*.md`.

## 0. Two facts that settle most rows before any re-run

1. **Buoys exist on only two runs.** In the paired GT, buoy boxes are on pohang02 (532) and pohang03 (68), and nowhere else: pohang00 has 0 and pohang01 (night) has 0. `ap_from_parts` / `matching.map50_95` take the class set from the GT in the selection (`MISSING_CLASS_POLICY = "drop"`). So **every night cell, and every TUNE (pohang00) number, is the same on the macro and on ship AP, to the bit.** All 600 buoys are on the TEST runs and the pooled day set.
2. **From 2026-08-31, most fusion scripts score ship.** `eval_final_system.py` (ship headline table, macro as "continuity"), `gate_lab.py`, `eval_levers.py`, `sweep_align.py`, `reprice_veto_axes.py`, `probe_veil_night_exposure.py`, `eval_extended_tuning.py` and `sweep_irnms_calib.py` all read `per_class[0]`. The macro holdouts are the 08-19/20 scripts, the VIS-only "ideas queue" probes of 09-02 (`_ideas_common.ap_of` → `map50_95`), the soft-NMS gate and the constants re-price.

So class set is not the issue for Table 3a c–e′, the veto rows, capability ratio, σ-in-score, alignment, isotonic or support. It matters for the re-price, the re-ranker, the checkpoint ensemble, within-stream dedup, top-k, σ-WBF, the oracle numbers, and, through the comparison bar rather than the class set, for Table 3a rows a and b.

## 1. Per-row table

"Recorded class set": **ship** = `per_class[0]["ap50_95"]`; **macro** = `map50_95`, the mean over ship and buoy; **≡** = macro and ship are identical on that selection (night or pohang00 only); **pooled** = a box-level rate over both classes, which is not an AP and not a macro. Intervals marked iid are frame-level paired bootstraps from the original records. Every interval this audit computed is a moving-block bootstrap, L = 20, n 1000, seed 0.

### §6.3: Table 3a, Figure 5 and the paragraph after them

| row | draft value | source (script, key) | recorded class set | registered metric | ship AP (re-scored or read from record) | verdict change |
|---|---|---|---|---|---|---|
| 3a-a (08-19) | fog/night −0.0021 vs ir_only, "later found within CI" | `docs/followup-analysis-2026-08-20.md` §1 ← `runs/eval/x_fusion_ci.md` §1 (`eval_fusion_ci.py`, `r["map50_95"]`) | macro; the cell is night, so ≡ | none (predates pre-registration) | fog/night −0.0021 [−0.0029, −0.0009] (iid), unchanged. **Against max(VIS, IR)**, the same record's §4 per-class table puts the worst cell at **lowlight/day: ship −0.0180** (gated 0.0166, VIS 0.0346, IR 0.0177); macro −0.0086 | **Changes.** The row compares against ir_only on night cells only. Against the table's stated bar, the worst cell is lowlight/day −0.0180 on ship. "Later found within CI" is also wrong: this CI excludes zero. The interval that spans zero belongs to the +0.0003 clean/night "win" (11.4% sign flips). |
| 3a-b (08-20) | fog/night tie | `x_veto_hysteresis.md` (`eval_veto_hysteresis.py`, `map50_95`): 0.0809 vs 0.0810; the same system on ship in `final_system.md` §1–2 (`eval_final_system.py`, ship headline; config = photometric veto-only, dilate-15) | macro (x_veto) / ship (final_system); ≡ on fog/night | none | fog/night −0.0001 [−0.0006, +0.0004] (iid). **Against max(VIS, IR): lowlight/day −0.0180** (gated 0.0166 vs VIS 0.0346; VIS vetoed on 100% of those frames); fog/day −0.0051 vs IR; macro lowlight/day −0.0086 | **Changes**, same reason as a: −0.0180, lowlight/day. |
| 3a-c (09-01) | lowlight/day −0.0180 | `gate_lab_rules.md` rule "B photometric OR veil [adopted]" (`gate_lab.py`, ship); also `docs/eval/final_system_2026-09-01.md` §2 | ship | none | — | none |
| 3a-d (09-01) | +0.0000 | `final_system_crossmodal.md` §1–2 (`eval_final_system.py`, ship) | ship | none | — (the macro table in the same record is ≥ 0 on every cell too) | none |
| 3a-e (09-01) | fog/clean −0.0632 | `veil_veto_repair_26m.md` "adopted" fog/day 0.0192 vs bar 0.0824; also `veto_axes_26m.md` (`reprice_veto_axes.py`), `veil_night_exposure_26m.md` (`probe_veil_night_exposure.py`), `levers_26m.md` (`eval_levers.py`) | ship (the record carries the macro, 0.0096, in its own column). The writer of `veil_veto_repair_26m.md` is not in `scripts/`; the header and the separate macro column establish ship | none | — | none |
| 3a-e′ (09-01) | +0.0000 | `veil_veto_repair_26m.md` "veil AND night": night cells +0.0000 (pass-through), day +0.0017 to +0.0053 | ship | none | — | none |
| Fig. 5 | the six Table 3a points; a is hollow ("within CI") | `scripts/paper_figures.py` `STAGES`, hard-coded from Table 3a ("no machine-readable file holds the history") | inherits | — | a and b → −0.0180, lowlight/day; drop the hollow marker | **Changes** with rows a and b |
| ¶ clean/day | gated 0.371–0.374 vs VIS 0.368 | `final_system_crossmodal{,_hardened,_rsys,_irnms,_v2}.md` §1/§2 (`eval_final_system.py`, ship) | ship (the macro would read 0.3357–0.3368 vs 0.3352) | none | — | none on class set. **Provenance error** (§4): these are yolo26s caches (`runs/cache`, VIS fog/day 0.0020) under preset `crossmodal`, not the shipped `crossmodal26m` / yolo26m. |
| ¶ glare/day | 0.296–0.298 vs 0.289 | same | ship (macro 0.2671–0.2681 vs 0.2626) | none | — | as above |
| ¶ "both clear of zero" | — | iid CIs in the same records | ship | — | — | **Interval issue** (§4): `docs/eval/change_impact_2026-09-09_v4.md` lists `crossmodal` clean/ship gated-vs-VIS +0.0034 as INDETERMINATE under the 1.95× correction ([−0.0001, +0.0080]) |
| ¶ night bit-identical to IR-only | 100% VIS veto | structural | ≡ | — | — | none |
| ¶ fog/day 0.0192 → 0.0908 | after the reorder | `veil_night_exposure_26m.md` fog/clean `crossmodal` vs `crossmodal26m` (`probe_veil_night_exposure.py`, ship) | ship | none | — | none |
| ¶ veil repair off | −0.0416 on fog/clean, "7.7× the pre-registered margin" | `reprice_constants.md` (`reprice_constants_draw_avg.py`, `ap_weighted(...)["map50_95"]`), margin 0.0054 | **macro** | `prereg-reprice-inherited-constants.md`: the statistic is "AP(A) − AP(shipped)", and rule 8 lists "per-class splits" as **not** decision inputs. The registered metric is therefore the all-class AP the code computes, i.e. the macro | kept on the macro, the registered metric: −0.0416, margin 0.0054 (7.7×). The ship re-score was started and died at draw 1/7 (`runs/eval/reprice_ship_2026-10-08.log`); not re-run | none on the registered metric; the draft now names the macro |
| ¶ removing the veto | costs 0.002–0.017 on night, fog and glare | `final_system*.md` §3 `no_veto` rows (`eval_final_system.py`, ship), pooled across snapshots | ship (night ≡) | none | night 0.0001–0.0173; fog/day 0.0040–0.0095 (macro 0.0020–0.0045) | none. The range pools five snapshots of two presets on yolo26s; the v2 snapshot alone is 0.0002–0.0147 |
| ¶ Mahalanobis re-enabled | costs 0.02–0.03 on lowlight/day | `final_system_crossmodal*.md` §3 `with_maha` (`eval_final_system.py`, ship) | ship | none | ship −0.0188 to −0.0228 (macro −0.0094 to −0.0114) | **Number:** no snapshot supports 0.03. Ship is 0.019–0.023 (the macro would read 0.009–0.011) |
| ¶ capability prior, 0.85 and veil repair "all stand" | — | `reprice_constants.md` | **macro** | as above | kept on the macro, the registered metric; ship re-score not completed (as above) | none on the registered metric; the draft now names the macro |

### §6.8: Table L

| row | draft value | source (script, key) | recorded class set | registered metric | ship AP (re-scored or read from record) | verdict change |
|---|---|---|---|---|---|---|
| L1 σ-weighted WBF | ≤ 0.0005 on every cell → inert | `sigma_wbf.md` (iou 0.85), `sigma_wbf_iou055.md` (iou 0.55) (`eval_sigma_wbf.py`, 08-19 system: D-6 ladder, prior VIS 0.2580 / IR 0.0206, `matching.map50_95`) | **macro** | none (TODO A1) | Re-scored, both thresholds (the macro path reproduced exactly first). iou 0.85: ship \|Δ\| ≤ 0.0002 on every cell, run and pooled row; block CIs on the day/night rows within [−0.00016, +0.00018]. iou 0.55: ship \|Δ\| ≤ 0.0005 (pohang00 −0.0005 ≡; pohang03 +0.0005); day rows +0.0002 / −0.0001 / −0.0001 / +0.0001, block CIs within [−0.0004, +0.0007]. The shipped preset agrees: Stage 1 B − A −0.0002 and D − C −0.0003 (ship, TUNE) | none. ≤ 0.0005 holds on ship |
| L2 σ in fusion score | +0.0010, CI spans zero → inert | `sigma_score_26m.md` "clean test" (α 0.1, `crossmodal26m`, `runs/cache_m`; writer not in `scripts/`; ship columns with a separate macro column) | ship | none | — | none on class set. **Reading issue** (§4): the same record has TUNE −0.0101 [−0.0114, −0.0086] and day −0.0103, both excluding zero. Outside TEST it hurts, which agrees with R-D1's S5 < S0 |
| L3 registration alignment | TEST −0.0010, spans zero (TUNE +0.0121) | `align_nomerge_26m.md` arm "align + iou0.95 + sup i0.55 g0.5" (`eval_levers.py`, ship) | ship (TUNE ≡) | none | — | none |
| L4 isotonic calibration | TUNE/TEST −0.0044 / −0.0034 | `irnms_calib_26m.md` §2 "calibrated" (`sweep_irnms_calib.py`, ship). `PAPER_CONTEXT_COMPILED.md` cites `final_26m_grid*.md`, which does not contain the row | ship (TUNE ≡) | none | — | none (the source citation should change) |
| L5 score re-ranking, LORO | +0.0037 OOF (in-sample +0.0149), "2,232 frames" → small | `rerank_loro.md` (`fit_rerank.py --ir-cache runs/cache_m/gauss_ir_paired_clean.pkl`, `_ideas_common.ap_from_scores` = macro). **VIS-only re-ranking on the 1,200 day frames** (3 folds), not 2,232 fused frames | **macro** | none | Same arm (4 features, monotone, λ 0.30, chosen by the macro sweep): **ship +0.0063 [+0.0030, +0.0094]**, 3/3 held-out runs improve; macro +0.0037 [+0.0018, +0.0055]. Best arm on ship: 18 features, λ 0.40, +0.0086 [+0.0028, +0.0126], but it was selected on ship. "In-sample": fit runs +0.0219 on ship (macro +0.0149); held-out pohang00 +0.0080 on both (≡) | **Changes.** On ship the fixed arm clears the 0.0047 ship floor and the 0.0060 Phase 3 floor (by 0.0003), and its CI excludes zero. "Small" no longer describes it. The OOF delta is over the λ grid the macro sweep chose; the arm was not chosen on ship |
| L6 score re-ranking, day-only | 9,284 frames, best λ = 0, delta 0.0000 → null at scale | `rerank_loro_day.md` (macro; 4 folds, **including pohang04, 2,343 frames**) | macro | none | **not re-scorable**: the substrate includes pohang04, which is spent. No per-class column in the record | **Unresolved on ship.** Given L5, "null at scale" cannot be carried to ship AP without a re-run, and that re-run is forbidden |
| L7 temporal support | lift 1.00× → dead | `signal_lift_26m.md` clean (`probe_signal_lift.py`): VIS day boxes of both classes | pooled | none | ship boxes only: 1.00× (k2), 0.98× (k1), 1.09× (k5) | none |
| L8 TTA view merging | ≈1.0× lift once confidence-matched → inert | `tta_o2m.md` (`probe_tta_o2m.py`): lift pooled; merge arms macro (iid n 300) | pooled / macro | none | matched@50 ship 0.44–1.70× (macro 0.42–1.65×); WBF arms ship −0.0053 to +0.0008, every block CI spans zero (macro +0.0013 to +0.0031) | none |
| L9 two-checkpoint ensembling, coordinate merge | −0.002 to −0.095 → hurts | `checkpoint_ensemble.md` (`probe_checkpoint_ensemble.py`, macro; VIS-only, 1,200 day frames) | **macro** | none | **WBF (coordinate-merge) arms: ship +0.0009 to +0.0069.** @0.55 plain +0.0069 [+0.0011, +0.0131], positive on 3/3 held-out runs (+0.0037 / +0.0075); @0.70 plain +0.0053 [+0.0000, +0.0114]; @0.85 +0.0009/+0.0010, spans zero. Concatenation arms −0.0180 and −0.1018 (macro −0.0089, −0.0949) | **Flips.** On ship, coordinate merging across checkpoints does not hurt; the best arm is positive with a CI excluding zero (best of six WBF arms, so the CI is optimistic). The macro's −0.002 to −0.003 is a buoy loss (buoy Δ = 2·macro − ship ≈ −0.011 at @0.55 plain). The −0.095 end of the range is the concatenation arm, which merges nothing. On the same block bootstrap, the macro WBF arms span zero too (@0.55 plain −0.0021 [−0.0057, +0.0026]; `checkpoint_ensemble_macro_blockci_2026-10-08.md`), so even the macro "hurts" was an iid result |
| L10 within-stream WBF dedup | −0.003 to −0.005 mAP50-95 → hurts | `within_modality.md` (`probe_within_modality.py`, macro; VIS-only day) | **macro** | none | WBF merge arms: ship −0.0030 to −0.0068. Block CI excludes zero at IoU 0.80 (−0.0051 [−0.0085, −0.0008]) and 0.90 (−0.0068 [−0.0101, −0.0027]); spans zero at 0.50–0.70 | **Weaker.** "Hurts" holds at merge IoU ≥ 0.80. At 0.50–0.70 it is unresolved on ship. This is the class set, not the interval type: the macro path with the same block bootstrap (`within_modality_macro_blockci_2026-10-08.md`) excludes zero at every IoU (e.g. @0.50 −0.0046 [−0.0067, −0.0004]) |
| L11 VIS soft-NMS σ = 0.5 | night worst cell −0.0000, negative on 4/4 draws → rejected | `snms_gate_draw_avg.md` (`gate_snms_draw_avg.py`, `map50_95`) | macro | `prereg-snms-draw-averaged-gate.md`: draw-averaged `delta_day` / `delta_night` ≥ 0 on every cell; no class named, so the code's macro | not re-run | **None, by construction.** The failing cell is night (blur_s3/glare_s2), where macro ≡ ship. Any day-cell change on ship can only add failures |
| L12 capability ratio ×4/×16/×64 | win clean, lose ≥ 1 cell → rejected | `extended_tuning.md` (`eval_extended_tuning.py`, ship) | ship | none (rule stated in the record: no cell may lose) | — | none |
| L13 two-sided veto | worst gap −0.0810 "on IR-corrupted night" → unsafe | `gate_lab_twosided.md` rule O "IR vetoed iff VIS kept" (`gate_lab.py`, ship) | ship (night ≡) | none | — | none on class set. **Description error** (§4): `gate_lab` has no IR corruption. −0.0810 is every night cell with clean IR: VIS is kept at night (0.0000), so IR (0.0810) is vetoed |
| L14 top-k truncation, k ≥ 50 | < 0.0002 → irrelevant | `x_topk_truncation.md` (`eval_topk_truncation.py`, preset `adopted`, `map50_95`) | **macro** | none | ship sweep (`runs/eval/cls_audit_topk_ship.log`, IR and VIS truncation; the run died before the `both` rows and the intervals). Point deltas against the adopted system (= k 400, which matches `x_fusion_ci.md` §4 gated ship on every cell): k ≥ 100 ≤ 0.0001 in magnitude; k = 50 up to −0.0006 (IR truncation, lowlight/day; fog/day −0.0003). On the macro the same k = 50 cell is −0.0004, so the recorded "< 0.0002 at k ≥ 50" was already loose on its own record | none: every delta is far below the 0.0014 floor. The row now states k ≥ 100 and k = 50 separately |

### §6.8: the two paragraphs after Table L

| claim | draft value | source (script, key) | recorded class set | registered metric | ship AP | verdict change |
|---|---|---|---|---|---|---|
| support multiplier at IoU 0.30 | TEST +0.0033 [+0.0010, +0.0067] | `align_26m.md` §2/3 "support i0.3 g0.5" (`sweep_align.py`, ship, `bootstrap_delta(..., cls=SHIP)`, **iid**) | ship | none | — | none on class set. **Interval issue** (§4): under the paper's own 1.95× correction the interval is [−0.0012, +0.0099], which spans zero |
| lifts | conf 4.80×, σ 3.00×, cross-modal@0.30 2.08×, temporal 1.00× | `signal_lift_26m.md` clean | pooled | none | ship boxes: 4.77×, 3.06×, 2.14×, 1.00× | none |
| oracle re-ranking, paired substrate | 0.3233 → 0.4293 (+0.1060) | `oracle_headroom.md` §1 (`probe_oracle_headroom.py`). **VIS-only on 1,200 day frames**, not the fused paired substrate | **macro** | none | the record's own §2: **ship 0.3686 → 0.4653 (+0.0968)**; buoy 0.2780 → 0.3932 | **Number changes**; the claim (large re-ranking headroom) does not |
| oracle, day-only VIS | 0.2771 → 0.4059 | `oracle_headroom_day.md` §1 (9,284 frames, **including pohang04**) | **macro** | none | read from the record's §2, not re-scored: **ship 0.3512 → 0.4624 (+0.1112)** | number changes; the claim does not |
| small objects | 89.8% of GT mass, lowest AP | `ap_by_size.md` §2 (`probe_ap_by_size.py`): ship and buoy GT pooled | pooled | none | ship GT: 9,512 / 10,663 = **89.2%**; ship small AP50-95 0.3382, the lowest ship bin (medium 0.6229, large 0.5092) | none (89.8 → 89.2 if ship is meant) |
| cross-modal union recall | +0.01–0.02 over VIS alone | `oracle_headroom.md` §4 | **ship** (per class; buoy +0.0000) | none | +0.0115 at IoU 0.50, +0.0205 at 0.30 | none |

## 2. Reproduction checks

Each re-score ran only after the unchanged default path reproduced the record. The comparison drops blank lines and the runtime footer and normalises `\` to `/`.

| record | re-run (`runs/eval/`) | result |
|---|---|---|
| `within_modality.md` | `within_modality_repro_macro_2026-10-08.md` | identical, every line. The only addition is the provenance block that `write_md` has appended since 09-10 |
| `checkpoint_ensemble.md` | `checkpoint_ensemble_repro_macro_2026-10-08.md` | identical (+ provenance block) |
| `tta_o2m.md` | `tta_o2m_repro_macro_2026-10-08.md` | identical (+ provenance block) |
| `rerank_loro.md` | `rerank_loro_repro_macro_2026-10-08.md` | identical (+ provenance block) |
| `signal_lift_26m.md` | `signal_lift_repro_macro_2026-10-08.md` | identical, 63 of 63 lines |
| `sigma_wbf.md` (08-19) | `sigma_wbf_repro_macro_2026-10-08.md` | identical, 33 of 33 lines |
| `sigma_wbf_iou055.md` (08-19) | `sigma_wbf_iou055_repro_macro_2026-10-08.md` | identical, 33 of 33 lines |
| `x_topk_truncation.md` (08-20) | `x_topk_truncation_repro_macro_2026-10-08.md` | §2 sweep reproduced: every printed `ir`/`vis` row and `both` k 25 equal the record's §2 table; the run died before the remaining `both` rows and §3 (`runs/eval/cls_audit_topk_macro.log`) |
| `reprice_constants.md` (09-02) | `reprice_repro_macro_2026-10-08.{md,json}` | not completed: both runs died at draw 1/7, cell 8 of 11 (`runs/eval/reprice_{repro_macro,ship}_2026-10-08.log`). Not needed for the registered verdict, which is on the macro and stands as recorded |

The re-ranker's macro block CI (`rerank_loro_macro_blockci_2026-10-08.md`) repeats the recorded OOF delta exactly: +0.003699, against +0.0037 in the record.

## 3. What changed, and why the macro hid it

The macro is (ship + buoy) / 2, so buoy Δ = 2·macro Δ − ship Δ. All 600 buoys are on TEST and the pooled day set, and only VIS emits them. So for the VIS-only probes the macro is a half-weight average with a 600-box class, and the class-set effects below are the buoy class moving differently from ship.

| row | arm | ship Δ | macro Δ | implied buoy Δ | what the macro did |
|---|---|---:|---:|---:|---|
| L5 re-ranker | 4 features, monotone, λ 0.30 (OOF) | +0.0063 | +0.0037 | +0.0011 | Halved a ship gain: buoys barely re-rank, so the ship gain is diluted by a near-zero buoy gain |
| L9 two-checkpoint | WBF @0.55 plain | +0.0069 | −0.0021 | −0.0111 | Flipped the sign: merging helps ships and hurts buoys |
| L9 two-checkpoint | B (ft) alone | +0.0020 | −0.0088 | −0.0196 | The fine-tuned checkpoint is no worse on ships; it lost buoys |
| L10 within-stream | WBF @0.50 plain | −0.0030 | −0.0046 | −0.0062 | Buoys lose more, so the macro's tighter, larger loss came from buoys |
| L10 within-stream | WBF @0.90 plain | −0.0068 | −0.0030 | +0.0008 | Here ships lose more and the macro hid it |
| ¶ oracle (VIS, 1,200 day) | perfect re-ranking | +0.0968 | +0.1060 | +0.1152 | Buoy headroom inflated the macro |

The pattern matches R-D1 (§6.4). Score-only re-ranking of a detector's own boxes is where a real ship-AP signal sits: σ in R-D1, a learned IoU predictor in L5. The macro hid both, because buoys (5.3 percent of day boxes, half the macro's weight) re-rank differently. The re-ranker result also supports the draft's closing line, "resolution and ranking, not fusion, are the levers". On ship AP, ranking is no longer just headroom: an out-of-fold learned re-ranker recovers +0.0063 of the +0.0968 oracle headroom.

**Re-price.** `docs/prereg-reprice-inherited-constants.md` rule 2 fixes the statistic as `AP(A) − AP(shipped)` and rule 8 lists per-class splits among the inputs that are not decision inputs; the code computes the macro. The registered verdict is therefore the recorded one. A ship re-score would be descriptive only (the ×8 arm was the macro's one near-miss); it is optional and was not re-run.

**Top-k.** Closed from the sweep without intervals: the largest ship delta at any k ≥ 50 is 0.0006, less than half the smallest floor.

## 4. Incidental findings (not class set; for the main session to weigh)

1. **Table 3a rows a/b and Figure 5 measure a different bar.** The 08-19/20 records ranked cells against `ir_only`, and the draft prints that worst cell under a "max(VIS, IR)" header. Against max(VIS, IR), stages a, b and c share one worst cell: lowlight/day, where the photometric veto deletes VIS on every synthetic low-light day frame. The first rewrite that closes it is d (`grad_gini` + IR night vote). The claim "each rewrite was forced by a measured failure of the previous one" still holds, but the gap curve is flat from a to c, at −0.0180 on ship.
2. **The ¶ after Table 3a is not on the shipped detector.** 0.371–0.374 / 0.296–0.298, the no-veto range and the Mahalanobis cost all come from `final_system_*` snapshots on yolo26s (`runs/cache`; VIS fog/day 0.0020), under preset `crossmodal` or the 08-20 system. The sentence "Every number in Table 3a and the paragraph after it was measured on the pre-restore full-scale checkpoints" is true only for e, e′, fog/day 0.0192 → 0.0908 and the re-price. On the shipped `crossmodal26m` / `runs/cache_m`, ship AP is clean/day 0.3792 vs VIS 0.3686 (+0.0106) and glare/day 0.3144 vs 0.3024 (+0.0120) (`final_26m_grid_v2.md`, `sigma_score_26m.md`).
3. **"Both clear of zero"** is iid. `change_impact_2026-09-09_v4.md` marks `crossmodal` clean/day gated-vs-VIS (ship, +0.0034) INDETERMINATE under the 1.95× correction.
4. **Support multiplier TEST +0.0033 [+0.0010, +0.0067]** is an iid interval (`sweep_align.py`, 09-01). Widened by the paper's own 1.95 factor it is [−0.0012, +0.0099]. "The one fusion lever that survives held-out testing" then rests on an interval that spans zero. A block-bootstrap re-run of that one delta on ship is cheap and was not done here.
5. **σ in fusion score** reads "inert" from TEST alone. The same record has TUNE −0.0101 [−0.0114, −0.0086] (pohang00, ≡ ship) and day −0.0103.
6. **Two-sided veto** −0.0810 is on night cells with clean IR, not "IR-corrupted night". `PAPER_CONTEXT_COMPILED.md` §(gate_lab) carries the same error.
7. **Table L caption** "(paired deltas, TEST unless noted)": TEST is the basis only for L2, L3, L4 and the support paragraph. L5 is OOF over three day runs; L1, L9, L10 and L14 are pooled day or all frames; L11 is draw-averaged per cell; L12 is fit-run day cells; L13 is the 8-cell grid.
8. **Row L5 says "2,232 frames"**, and the oracle sentence says "on the paired substrate". Both are VIS-only on the 1,200 day frames of the paired list.
9. **`PAPER_CONTEXT_COMPILED.md`** attributes isotonic calibration to `final_26m_grid*.md`; the source is `irnms_calib_26m.md`.
10. `rerank_loro_day.md` and `oracle_headroom_day.md` scored pohang04 frames on 2026-09-02, before the look. §3.6 should already disclose this; check that it names these two.

## 5. Proposed draft edits (old → new)

Edits are marked **[class set]**, where the audit changes a number or verdict, or **[incidental]**, where a scoped sentence misdescribes its source. Line numbers refer to PAPER_DRAFT2.md at `17003e4`.

**E1 [class set]: §5.5, line 200.**
Old: "Table 3a and the lever table in §6.8 summarise records that were not re-audited for class set."
New: "Table 3a reports ship AP. The lever table in §6.8 reports ship AP for every AP row; three rows that were recorded on the macro were re-scored on ship from cached detections, after the macro path reproduced the record (`docs/eval/class_set_audit_2026-10-08.md`). The constants re-price in §6.3 was registered on the macro and is reported on both."
Applied with two exceptions named: the day-only re-ranking row (macro; pohang04) and soft-NMS (macro, registered; failing cell at night, where macro ≡ ship).

**E2 [class set]: §6.3, line 339.**
Old: "Table 3a and Figure 5 record the worst-cell gap to the better single stream at each stage; each rewrite was forced by a measured failure of the previous one. **Every number in Table 3a and the paragraph after it was measured on the pre-restore full-scale checkpoints, whose VIS detector scored 0.0000 at night.**"
New: "Table 3a and Figure 5 record the worst-cell gap, in ship AP, to the better single stream at each stage; each rewrite was forced by a measured failure of the previous one. **Every number in Table 3a and the paragraph after it was measured on pre-restore checkpoints whose VIS detector scored 0.0000 at night: the phase-2 yolo26s detectors for stages a–d, and the full-scale yolo26m detectors for stages e and e′.**"

**E3 [class set]: Table 3a, lines 345–346, rows a and b.**
Old: "| a (08-19) | p05 photometric term, hard veto | fog/night | −0.0021 vs ir_only, later found within CI |"
New: "| a (08-19) | p05 photometric term, hard veto | lowlight/day | −0.0180 |"
Old: "| b (08-20) | soft term dropped as redundant; hysteresis | fog/night | tie |"
New: "| b (08-20) | soft term dropped as redundant; hysteresis | lowlight/day | −0.0180 |"
Add below the table: "Ship AP; each gap is against max(VIS, IR) in its own cell. The 08-19 and 08-20 records ranked cells against IR alone, and on that comparison their worst cell was fog/night: −0.0021 [−0.0029, −0.0009] at a, and −0.0001, a tie, at b. Against max(VIS, IR), the photometric veto deletes VIS on every synthetic low-light day frame, and lowlight/day is the worst cell from a through c (`runs/eval/x_fusion_ci.md` §4, `runs/eval/final_system.md` §2, `runs/eval/gate_lab_rules.md`)."

**E4 [class set]: Figure 5 caption, line 358, and `scripts/paper_figures.py` `STAGES`.**
Old: "pre-restore full-scale checkpoints, development paired frames, local AP.** Each rewrite was forced by the failure of the one before it; the hollow point was later found to lie within its interval."
New: "pre-restore checkpoints (yolo26s for a–d, yolo26m for e and e′), development paired frames, ship AP (local AP).** Each rewrite was forced by the failure of the one before it; stages a–c share their worst cell, lowlight/day."
Code: `("a", "08-19", "fog/night", -0.0021, True)` → `("a", "08-19", "lowlight/day", -0.0180, False)`; `("b", "08-20", "fog/night", 0.0, False)` → `("b", "08-20", "lowlight/day", -0.0180, False)`. Then regenerate `docs/figures/fig_gate_history.png`.

**E5 [incidental]: §6.3 paragraph, line 352, first sentence.**
Old: "Under the final preset on the pre-restore checkpoints, clean/day gated AP is 0.371–0.374 against 0.368 for VIS alone, and glare/day is 0.296–0.298 against 0.289, both clear of zero."
New: "Under the shipped preset on the pre-restore full-scale checkpoints, clean/day gated ship AP is 0.3792 against 0.3686 for VIS alone, and glare/day is 0.3144 against 0.3024. On the earlier yolo26s checkpoints under `crossmodal` the same comparison read 0.371–0.374 against 0.368 and 0.296–0.298 against 0.289; the clean/day margin there does not survive the interval correction of §5.4 (`docs/eval/change_impact_2026-09-09_v4.md`)."

**E6 [class set]: §6.3 paragraph, line 352, the veil-repair sentence.**
Old: "turning the veil repair off costs −0.0416 on fog/clean, 7.7 times the pre-registered margin."
New: "turning the veil repair off costs −0.0416 on fog/clean, 7.7 times the pre-registered margin (on the macro over ship and buoy, the re-price's registered metric)."

**E7 [class set]: §6.3 paragraph, line 352.**
Old: "Removing the veto entirely costs 0.002–0.017 on night, fog and glare. Re-enabling Mahalanobis weighting costs 0.02–0.03 specifically on lowlight/day."
New: "On the yolo26s checkpoints, removing the veto entirely costs 0.002–0.017 on night, fog and glare, and re-enabling Mahalanobis weighting costs 0.019–0.023 specifically on lowlight/day."

**E8 [class set]: §6.3 paragraph, line 352, last sentence.**
Old: "Capability-prior weights, the merge threshold of 0.85 and the veil repair were all re-priced against the measured noise floor and all stand."
New: "Capability-prior weights, the merge threshold of 0.85 and the veil repair were all re-priced against the measured noise floor, on that registered macro, and all stand."

**E9 [class set]: Table L caption, line 510.**
Old: "**Table L. Fusion and post-processing levers tested and not adopted (paired deltas, TEST unless noted).**"
New: "**Table L. Fusion and post-processing levers tested and not adopted. Paired deltas in ship AP (local AP), on the frames named in each row; TEST = pohang02 + pohang03 day.** Rows marked † were recorded on the macro over ship and buoy and are re-scored here on ship (`docs/eval/class_set_audit_2026-10-08.md`); their intervals are moving-block, L = 20."

**E10: Table L rows (old → new).**
* [incidental] "| σ in fusion score | +0.0010, CI spans zero | inert |" → "| σ in fusion score (α 0.1) | TEST +0.0010, CI spans zero; TUNE −0.0101 [−0.0114, −0.0086] | hurts on TUNE, unresolved on TEST |"
* [class set] "| Score re-ranking, leave-one-run-out, 2,232 frames | +0.0037 OOF (in-sample +0.0149) | small |" → "| VIS score re-ranking, leave-one-run-out, 1,200 day frames † | +0.0063 [+0.0030, +0.0094] OOF, 3/3 held-out runs (fit-run gain +0.0219) | clears the 0.0060 floor; not adopted |"
* [class set] "| Score re-ranking, day-only 9,284 frames | best λ = 0, delta 0.0000 | null at scale |" → "| VIS score re-ranking, day-only 9,284 frames (macro; includes pohang04) | best λ = 0, delta 0.0000 | null on the macro; not re-scorable on ship |"
* [class set] "| Two-checkpoint ensembling, coordinate merge | −0.002 to −0.095 | hurts |" → "| Two-checkpoint VIS ensembling † | WBF merge +0.0009 to +0.0069 (IoU 0.55: +0.0069 [+0.0011, +0.0131]); concatenation −0.018 to −0.102 | merging does not hurt; concatenation does |"
* [class set] "| Within-stream WBF dedup | −0.003 to −0.005 mAP50-95 | hurts |" → "| Within-stream WBF dedup † | −0.0030 to −0.0068 | hurts at merge IoU ≥ 0.80; unresolved below |"
* [incidental] "| Two-sided veto (also veto IR) | worst gap −0.0810 on IR-corrupted night | unsafe |" → "| Two-sided veto (also veto IR) | worst gap −0.0810 on every night cell, IR uncorrupted | unsafe |"
* [class set] "| Top-k truncation, k ≥ 50 | < 0.0002 | irrelevant |" → "| Per-stream top-k truncation † | k ≥ 100: ≤ 0.0001; k = 50: up to −0.0006 (IR, lowlight/day); point deltas | irrelevant |"
* Unchanged on ship (re-scored; no edit needed): σ-weighted WBF (≤ 0.0002 at IoU 0.85, ≤ 0.0005 at 0.55), TTA view merging, temporal support (1.00× on ship boxes). Already ship: alignment, isotonic, capability ratio, soft-NMS (its failing cell is night, where macro ≡ ship).

**E11 [class set]: line 529, the paragraph after Table L.**
Old: "The pattern across levers is that merging as a family, whether cross-modal, within-modal, across checkpoints or across augmentation views, is exhausted; the test-time-augmentation case is the cleanest, because its views are pixel-exact registered and merging still barely moves."
New: "Merging does not pay across modalities, within a stream or across augmentation views; the test-time-augmentation case is the cleanest, because its views are pixel-exact registered and merging still barely moves. The one exception on ship AP is merging two VIS checkpoints at IoU 0.55, +0.0069 [+0.0011, +0.0131], the best of six arms; the macro had read it as a loss, because the merge costs buoys."
[incidental, same line] Old: "(TEST +0.0033 [+0.0010, +0.0067])" → New: "(TEST +0.0033 [+0.0010, +0.0067], a frame-level interval; widened by the factor of §5.4 it spans zero)". A block re-run of this one delta is cheap and would replace the approximation.

**E12 [class set]: line 531.**
Old: "Oracle re-ranking on the paired substrate would raise mAP50-95 from 0.3233 to 0.4293 (+0.1060), and on day-only VIS from 0.2771 to 0.4059; small objects carry 89.8 percent of ground-truth mass with the lowest AP;"
New: "Oracle re-ranking of the VIS detections on the 1,200 paired day frames would raise ship AP from 0.3686 to 0.4653 (+0.0968), and on 9,284 day-only VIS frames from 0.3512 to 0.4624 (+0.1112; that substrate includes pohang04 and is read from its existing record), of which a learned out-of-fold re-ranker recovers +0.0063 on the first (Table L); small objects carry 89.2 percent of ship ground truth with the lowest ship AP;"

**E13 [incidental]: §6.4, line 453.**
Old: "Adding σ to the fusion score moves TEST by +0.0010 with a CI spanning zero."
New: "Adding σ to the fusion score moves TEST by +0.0010 with a CI spanning zero and costs 0.0101 on TUNE."

## 6. Script changes (uncommitted; defaults unchanged)

Every flag below is opt-in. Without it, each script's output is unchanged; for the six probes re-run here that was checked line by line (§2).

* `scripts/reprice_constants_draw_avg.py`: `--cls K` scores `per_class[K]["ap50_95"]` wherever the default reads `map50_95`, including the paired-bootstrap margin; `--json PATH` dumps every per-draw delta and bootstrap sd.
* `scripts/_ideas_common.py`: new helpers `metric()`, `paired_ci()` (iid by default, block with `block_len`) and `cls_note()`; `ap_from_scores(..., cls_only=None)`.
* `scripts/probe_within_modality.py`, `probe_checkpoint_ensemble.py`, `probe_tta_o2m.py`: `--cls K` (the lift screens also restrict to class K boxes) and `--block-len L`.
* `scripts/fit_rerank.py`: `--cls K`; `--block-len L` adds a block CI on the best OOF arm; `--ci-arm N MONO LAM` also CIs a fixed arm.
* `scripts/probe_signal_lift.py`: `--cls K` screens only class K boxes.
* `scripts/eval_sigma_wbf.py`: `--cls K` (the capability prior stays the macro, as configured) and `--block-len L`.
* `scripts/eval_topk_truncation.py`: `--cls K` and `--block-len L`.

Raw output (git-ignored): `runs/eval/*_repro_macro_2026-10-08.*`, `runs/eval/*_ship_2026-10-08.*`, `runs/eval/rerank_loro_{ship_fixedarm,macro_blockci}_2026-10-08.md`, `runs/eval/cls_audit_*.log`.

## 7. Applied to the draft (main session, 2026-10-09; uncommitted)

E1–E13 are applied to `PAPER_DRAFT2.md`, with these differences from §5:

* **E1** names the two macro exceptions in Table L, marked ‡ in the table.
* **E10** marks every re-scored row † and the two macro rows ‡. The re-ranker row names its inputs, including σ, and says the arm was chosen on the macro and is unregistered.
* **E12** keeps "resolution and ranking, not fusion, are the levers", but no longer calls ranking out of scope. The re-ranker is a first, unregistered measurement of it.
* **Added: §6.4 and §8.** One sentence each says that a learned VIS re-ranker whose inputs include σ gains out of fold, that its σ share was not separated, and that it is not fusion.
* **Added: §3.6** (incidental item 10). It names the two VIS-only probes that scored the pohang04 frames of the earlier validation list on 2026-09-02.
* **Code:** `scripts/paper_figures.py` `STAGES` rows a and b are changed as in E4, a run of equal stages is labelled once, and `docs/figures/fig_gate_history.{png,pdf}` is regenerated.
* **Instructions:** `PAPER_WRITING_INSTRUCTIONS.md` is updated to match: header note, rule 10, items 8–9, discussion bullet, number hygiene.

## 8. σ-free re-ranker (2026-10-09, descriptive, unregistered)

`py -3.13 scripts/fit_rerank.py --ir-cache runs/cache_m/gauss_ir_paired_clean.pkl --cls 0 --n-features 3 4 --block-len 20 --ci-arm 4 yes 0.30 --ci-vs 3 yes 0.30`. `--n-features 3` is the 4-feature prefix without `sigma_mean_norm`. `--ci-vs` is new and opt-in: it gives a paired block CI on (ci-arm − vs-arm). Raw output: `runs/eval/rerank_loro_ship_{nosigma,sigma_increment}_2026-10-09.md`. Every 4-feature row reproduces `rerank_loro_ship_fixedarm_2026-10-08.md` exactly (fixed arm +0.006342 [+0.003042, +0.009411]).

| arm (monotone, λ 0.30, ship AP, OOF, 1,200 day frames) | delta vs no re-ranking | runs won |
|---|---:|---:|
| 4 features (with σ) | +0.0063 [+0.0030, +0.0094] | 3/3 |
| 3 features (no σ) | +0.0014 [−0.0020, +0.0044] | 1/3 |
| **σ's increment (4 − 3, paired)** | **+0.0050 [+0.0038, +0.0064]** | |

* The best σ-free arm anywhere in the sweep is +0.0018 (λ 0.20, chosen after the fact). No σ-free arm wins all three held-out runs, and every 4-feature monotone arm with λ 0.1–0.5 does.
* λ 0.30 was fixed by the macro sweep for the 4-feature arm, and the σ-free comparator uses the same λ. The comparison is descriptive and was not pre-registered.
* **Bearing on R-D1.** R-D1's fixed multiplicative σ re-ranking (α = 1) costs AP against no σ on 6/8 cells. Learned jointly with confidence, σ adds +0.0050 to within-detector ranking. Both are within-stream. Neither is fusion.
