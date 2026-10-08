# Instructions for Writing the uqfusion Research Paper

> **Partly superseded 2026-09-27 — read `PAPER_DRAFT2.md` and `docs/handoff-2026-09-27.md` first.**
> * §0 rule 6 (pohang04 embargo): lifted. The look was taken on 2026-09-20 and §7 of Draft 2 reports it.
> * §2 contribution 2: the claim "≥ max(VIS, IR) on all 8 cells" belongs to the pre-restore checkpoints. On the Phase 3 systems it holds on 6 of 8 and fails on clean/night and glare/night (`docs/eval/p3_corrupt_cells_2026-09-27.md`, `docs/eval/p3_night_check_2026-09-27.md`). Draft 2's three contributions replace §2's.
> * §3.4 "pohang04 was never used for anything": false. See Draft 2 §3.6 and prereg A8.5.
> * §9 Q3 and Q4 are decided: Q3 is disagreement ranking (G2), and Draft 2 Table 2 reports the three-arm table day-primary. Q4 is a non-inferiority margin (G3), declared but never run. The Mahalanobis rebuild is done (A6.4).
> * R-D1 now also has a `crossmodal26m` result (NULL): `docs/eval/uq_mechanism_ablation_26m_2026-09-27.md`.
> * **2026-10-08: Table 1 is the patience-20 backbone benchmark** (Draft 2 §6.1; `docs/eval/bench_patience20_2026-10-08/`), on restored labels. The Phase 1 grid is now Table 1b, the record of the yolo26m selection. §0 rule 4, §2, §3 (the AP-convention line and Results item 1), §5 and §6 below are updated to match.

> Derived solely from `PAPER_CONTEXT_COMPILED.md` (compiled 2026-09-17). Every number, verdict and file name below traces to a Part/section of that document, cited in brackets as [Pn.m]. When drafting, quote from the compiled document, never from memory. Where the compiled document records a correction, use the corrected value and never the original.

---

## 0. Ground rules that override everything else

1. **The paper reports a null result as its headline.** Predicted uncertainty does not improve VIS+IR fusion at any tested floor (R-D1) [P7]. The positive, reproducible contribution is *image-statistic sensor selection with union aggregation of detections* [P0, P1.8]. Do not write the paper the design document imagined (live uncertainty-gated fusion). Write the paper the measurements support.
2. **Never describe the shipped system as "uncertainty-gated fusion."** The required phrase is "image-statistic sensor selection with union aggregation of detections" [P0]. The shipped fusion weight `w_vis` is a single constant, 0.9930, on every frame and condition [P7.1].
3. **Corrected numbers only.** The compiled document keeps both the original and corrected value of several quantities. Use only the corrected one (list in §8 below).
4. **State the AP convention on every absolute number.** Local linear-interp AP, not COCO; deltas are convention-safe (worst disagreement 0.000285, 5× below the noise floor) but absolutes are not [P9.1 item 5]. The one exception is Table 1, which reports Ultralytics 8.4.90 validation mAP (ship and buoy) from the training logs; its caption says so and none of its values sits beside a local-AP number.
5. **Every delta must be accompanied by a paired, dependence-aware interval** (block bootstrap, L=20, 1.95× inflation applied) and compared against the measured noise floor 0.0014–0.0031 (2σ paired) or the Phase 3 floor 0.0060 [P9.1 items 2 and 6]. A delta below the floor is "not resolved," never "no effect."
6. **Nothing from pohang04 appears in the paper unless the single pre-registered look has been taken.** As of the compiled document it has not [P11.6]. See §10 for how to draft around it.
7. **Report the things that point the wrong way.** The project norm is that unfavorable results are reported, not discarded (e.g., S5−S6 clean = −0.013157) [P7.3].
8. **No claims about MIT, MassMIND or SMD data.** Pohang Canal + PoLaRIS is the only dataset. The earlier manual-annotation claim about MIT was measured false and is retired [P3.13, P10.2].
9. **No claim of novelty for the σ head or for the idea of conditioning fusion on uncertainty.** Both are prior art (Gaussian YOLOv3; UA-CMDet 2022; DICTA 2024) [P10.1]. Novelty is confined to the three defensible items in §2.
10. **Name the class set on every AP.** Ship is primary (pre-registered for Phase 3; the only class IR emits). Tables 1b, 3b, 5, 7 and Figures 6–7 are ship AP; Tables 1 and 4 are the ship+buoy macro; Table 2 gives both. Never put a macro beside a ship AP, and never report a single-class stream on the macro: it reads exactly half its ship AP. Declared in Draft 2 §5.5 (TODO-improvements §D.3).

---

## 1. Working title, authorship, venue

- **Working title (current):** Uncertainty-Aware Fusion of Visible and Infrared Imagery for Reliable Maritime Object Detection [P0].
- **Recommended retitle** to match what was measured, e.g. *"Does Predicted Uncertainty Help Visible–Infrared Maritime Detection? A Pre-Registered Null Result and an Image-Statistic Sensor-Selection Baseline."* The word "uncertainty-aware fusion" in the title would describe a system that does not exist [P10.1(b)].
- Author: Laksh Saroha, ECED, Thapar Institute of Engineering and Technology, Patiala. Mentor: Dr. Sandeep Mandia. UG Research Fellowship [P0].
- No venue pinned (A3-9). Licensing/release plan not settled; code builds on Ultralytics AGPL-3.0 [P0]. Do not assert a release plan in the paper; say the repository is public at github.com/Laksh-saroha/uqfusion [P13.4, OQ-14].

---

## 2. The claims the paper is allowed to make

Exactly three defensible contributions, in this order [P10.1]:

1. **A controlled maritime uncertainty study that publishes negative results**, with a measured (not assumed) noise floor, dependence-aware intervals, declared metric contracts, and pre-registered decision rules.
2. **A lightweight, interpretable sensor-selection baseline** (`grad_gini` + `ir_p05` veto, `night AND (dark OR veil)`) that sits at or above max(VIS, IR) on all 8 benchmark cells, with its failure cases documented (the −0.0632 veil-veto regression on detector swap; 100% VIS veto on fog and at night) [P6.4, P6.5].
3. **An honest account of when uncertainty-driven fusion does not pay off** (R-D1 null; S1-NULL on correspondence relaxation) [P7, P11.3].

Secondary findings worth a subsection each (choose by page budget):

- Night VIS blindness was a **label artifact, not a sensor limit**: 0.0000 → 0.2520 mAP50-95 after restoring 94,553 boxes [P3.5].
- **Checkpoint-selection instability**: best-epoch vs epoch-mean inverts the MC-Dropout ranking; between-seed sd of best-fitness (0.00212) is 2–5× smaller than within-run epoch noise (0.0035–0.0106) [P5.7].
- **Redundancy = independence**: temporal agreement lifts 1.00×, cross-modal IoU-0.30 agreement 2.08×, confidence 4.80×, σ-below-median 3.00× [P6.5, P8.11].
- **Backbone benchmark is a negative result** (Table 1, patience 20, restored labels): yolo26x 0.2666 ± 0.0040, yolo26l 0.2664 ± 0.0070, yolo26m 0.2626 ± 0.0109 are not separable; the other five n/t models sit at 0.1777–0.1943 and yolo26n escapes the floor at 0.2384; patience barely reorders the field (ρ = 0.948 on the 24 fully logged families) except yolo26n (+0.0276). Phase 1 (Table 1b, selection record): top-8 span 0.0055 vs seed sd 0.0010–0.0066, not separable; n/t floor 0.2486–0.2567 [P4.4; Draft 2 §6.1].
- **The Mahalanobis OOD scorer's night blind spot** was reference-set contamination (782/4,000 night frames in fit set); day-only refit separates correctly (D_night 89.0 vs D_day 30.9) [P6.1, P8.9].

---

## 3. Paper flow (IMRaD), section by section

### 3.1 Abstract (≤250 words)

Order of sentences:
1. Problem: EO sensors fail silently; the risk is confident error with no reliability signal [P1.1].
2. What was built: two per-modality YOLO26 detectors with a single-pass Gaussian σ² head, a frame-level Mahalanobis OOD score, and a decision layer that selects streams by image statistics and merges the survivors by WBF [P0].
3. Pre-registered test: does real predicted uncertainty beat shuffled uncertainty in the fusion? Result: NULL, 0 of 4 conditions at the 0.0060 floor [P7.3].
4. What does work: the sensor-selection veto reaches ≥ max(VIS, IR) on all 8 cells (worst-cell gap +0.0000) [P6.4].
5. Methodological findings: noise floor 0.0014–0.0031, iid bootstrap 1.9–1.99× too narrow, night VIS "blindness" was a label artifact [P9.1, P3.5].
6. One sentence on scope: Pohang Canal + PoLaRIS only; adverse conditions simulated; no untouched test set except the single pohang04 look [P1.16, P9.2].

### 3.2 Introduction

- Motivation from [P1.1]: visible fails in fog/haze/glare/rain/darkness; infrared fails at thermal crossover.
- Literature gap, **revised wording only** [P1.1 items 1–2]: most maritime detectors output no uncertainty; single-pass localization variance and uncertainty-aware cross-modal fusion exist outside maritime; what is thin is *calibrated maritime evidence measured against a stated noise floor*.
- Explicitly **do not** write the retired claim "existing VIS-IR fusion is static" [P1.1, P10.1].
- Frame the paper as a pre-registered study whose primary hypothesis was rejected. State the three contributions from §2.
- Close with the paper's core epistemic stance in one sentence: null results are results; decisions were pre-committed and stop rules enforced [P0].

### 3.3 Related work

Build the comparison table from the replacement axes in [P10.1]: domain, sensors, uncertainty target, inference-time adaptation, calibration evaluated, registration assumptions, compute. Rows: Gaussian YOLOv3 (Choi et al., 2019), UA-CMDet (Sun et al., 2022), Zhao et al. 2024 (DICTA, doi 10.1109/DICTA63115.2024.00029), RDSC-YOLOv4 (Liu et al., 2021) and YOLOv7-Sea (Zhao et al., 2023) as the maritime rows with no uncertainty, and this work. SID-YOLOv5 and EG-YOLO, named in earlier drafts, could not be traced to any publication on 2026-10-08; do not name them.

- This work's row: "hard veto on image statistics; fusion weight constant 0.9926 (measured)" (0.9926 is the shipped `crossmodal26m`; 0.9930 belongs to the predecessor `crossmodal`); calibration column "D-ECE, interval-ECE, NLL, AUSE/AURC, declared metric contracts, measured noise floor."
- **Table R's cells for other works were filled from direct reads of each full text on 2026-10-08** (Gaussian YOLOv3 and UA-CMDet from arXiv, the rest from the published versions). Any new row needs the same read; never fill a cell from memory [P10.1].
- What the reads changed, keep it: in UA-CMDet and Zhao et al. 2024 the "uncertainty" is a per-label training-loss weight, removed at inference, and neither paper evaluates calibration; UA-CMDet's inference-time adaptation is an illumination-weighted NMS on the RGB scores, which is the closest prior mechanism to the shipped veto (both demote VIS on an image statistic); Gaussian YOLOv3 does use its σ at inference, to rescore boxes.
- Cite β-NLL (Seitzer 2022) for the NLL failure mode and its fix [P1.5]. The mean-first warm-up is the common practice Skafte et al. (2019) describe, not their proposal (they propose a split scheme); cite it that way. Cite Deep Evidential Regression as considered, not benchmarked [P1.9].
- Bibliography facts to get right [P10.2]: Pohang Canal Dataset = Chung et al., IJRR 42(12), 2023 (sensor release); PoLaRIS = Choi, Cho, Lee, Kim, Yang, Kim and Cho, ICRA 2025, pp. 13626–13632, a separate annotation release (arXiv 2412.06192); MassMIND is LWIR-only instance segmentation in seven categories, not ship/buoy; SMD is visible plus near-infrared video that need not show the same scene, so not a paired fusion testbed. The CC BY-NC 4.0 licence is Pohang's (AWS Open Data registry); no licence is stated for the PoLaRIS labels. RT-DETR and D-FINE are not cited by the text and are not in the References; add them only with a sentence that needs them. The References section of `PAPER_DRAFT2.md` holds the verified entries; copy from there.

### 3.4 Dataset

Source: [P3].

- Pohang Canal + PoLaRIS: 7.5 km route, stereo VIS 2048×1080 @10 Hz, LWIR 640×512 16-bit @10 Hz, two classes (ship, buoy), Pohang imagery under CC BY-NC 4.0, five runs; pohang01 is night; **pohang04 has zero IR labels** [P3.1].
- Verified counts (use these, from `verify_dataset_claims.py`, 2026-09-10) [P3.2]:
  - 158,319 images (VIS 127,309 / IR 31,010)
  - 1,183,736 boxes (VIS 962,960 / IR 220,776)
  - 28,388 paired VIS↔IR rows (pohang00 10,786 / 01 11,990 / 02 3,739 / 03 1,873 / 04 0)
- Pairing is by the dataset's timestamp CSV, not frame ordinal (16,544 of 28,388 pairs differ in index) [P3.2]. Registration residual measured, not assumed: 3–6 px median, within-run drift up to ~10 px [P3.1, P3.12].
- Preprocessing: VIS letterboxed 640×338 inside 640×640 (47% pad at value 114); IR per-frame min-max normalized to 8-bit then letterboxed with 64 px top/bottom pad; raw 16-bit not recoverable [P3.3, P3.4]. Note the percentile-clip IR re-export was tested and rejected (+1.75% mAP50-95 but −1.6 pt recall) [P3.3].
- Splits: interleaved K-block split, cycle of 10 (80/10/10), guard bands, verified by `audit_split.py` → 0 duplicates, 0 temporal violations, both modalities PASS [P3.6, P3.11]. Give the split table from [P3.11].
- Night-box filter and restore: this is a **required subsection** (see §5, Table D). State the original filter's bug (pad level 4 vs pad value 114), the 132,688 boxes dropped, the 38,135 legitimately flagged, the 94,553 restored, and the verdict ALIVE [P3.5].
- Hash to cite for the train-scope post-restore label state: **`8ed69b5974ed`** (711,444 boxes). Never cite `b92739202127` as the train hash; it is tree-scope [P3.5].
- Holdout: pohang04 was never used for anything before the Phase 3 pre-registration; Phase 3 splits drop it by filtering (VIS train 38,295 / val 9,009 / test 9,054); val night share rises 18.2% → 23.0%, so Phase 3 numbers are not comparable to earlier pooled val numbers [P3.9].
- Disclose the contamination audit: the shipped Mahalanobis fit list contained 819/4,000 pohang04 frames (20.5%); replacement list built; cache rebuild status must be checked before the paper states it is fixed [P3.10].

### 3.5 Method

Source: [P1.4–P1.8, P5.1, P6.4, P6.5].

Write the method in two layers and label them explicitly:

**(a) What was designed** (brief, one paragraph plus the formulas from [P1.7]): per-modality reliability `R_m = r_frame,m · r_box,m` feeding WBF weights `w_m = R̄_m / (R̄_vis + R̄_ir)`. Say this design is retained for audit and every divergence from it was measured.

**(b) What ships (`crossmodal26m`)**, the system the results describe [P1.8, P7.1]:
- Two independent detectors: VIS yolo26m nc=2; IR yolo26m-p2feat nc=1 (IR cannot see buoys: buoy AP 0.0002) [P11.4, P8.16].
- Gaussian σ² head: in-place `cv4` log-variance branch on the live Detect head, no fork; σ over LTRB in stride units; σ rides the one2one branch only on YOLO26's end2end head; postprocess overridden; NLL target unclamped at reg_max=1; β-NLL + warm-up (NLL weight 0 during warm-up); σ branch on detached features so the detector trains as the baseline **by construction** [P5.1]. Disclose that `sigma_detach_features=False` cannot be honored on end2end heads [P5.1].
- Mahalanobis OOD score on backbone features (LedoitWolf) [P1.6, P1.13]. Present it as **diagnostic only** in the shipped system, since `mu_d=1e9, lam=0` make it inert [P7.1].
- Decision layer:
  - Night arm: ask IR whether it is night (`ir_p05 > 41.5`, fitted once on clean IR) with a second vote from VIS darkness and an IR self-check (`lap_over_var`), plus an 11-statistic multivariate IR health score with authority bound at clean p99 [P6.4, P8.6].
  - Veil arm: `grad_gini` (scale-free Gini of gradient magnitude), single-frame, no temporal filter [P6.4].
  - Rule: **`night AND (dark OR veil)`** vetoes VIS; a vetoed stream is *removed from the WBF input list*, not down-weighted (WBF renormalizes weights) [P6.1, P6.5].
  - Single surviving stream is passed through untouched (`single_passthrough`) [P6.4].
  - Merge: WBF at iou_thr 0.85 with constant capability-prior weights (VIS 0.3233, IR 0.0024 after ÷4, `w_vis` 0.9926), fitted on the 1,200 clean day frames the day cells score. "Run-disjoint" in older records means night-run-excluded only; never call the prior run-disjoint (§9 item 20). At this threshold only 0.05% of VIS boxes have an IR partner, so the merge is concatenation. Document merge as effectively `merge_iou='off'` [P1.8, P8.4].
  - Cross-modal support: score multiplier at IoU 0.30, γ=0.5, never moves a coordinate; TEST delta +0.0033 [P6.5].
  - IR-stream dedup: NMS @ 0.70 [P8.4].
- State explicitly which mechanisms are **off** in the shipped preset and why: `sigma_weighted=False`, `sigma_score_alpha=0`, Mahalanobis weight inert, `vis_soft_nms` off (pre-registered rejection) [P7.1, P8.12].

Include a figure of the decision layer as it ships (see §5, Figure 1), not the design diagram.

### 3.6 Experimental protocol and statistics

Source: [P9, P1.11, P6.5, P11].

This section carries contribution 1 and must be complete:

- **Benchmark cells:** 8 cells = {clean, fog, lowlight, glare} × {day, night} on 2,232 paired frames; later 10–11 cell grids add IR corruptions [P7.1, P8.5]. Adverse conditions are Albumentations-simulated; disclose [P1.16].
- **Tune/test discipline:** TUNE = pohang00 (836 frames), TEST = pohang02+03 (364 frames); pohang01 (night) excluded from fitting. Report the overfitting trap it caught (support IoU 0.55 wins TUNE, loses all six held-out variants; 0.30 wins all six) [P6.5].
- **Exposure ledger / no untouched test set:** state plainly that pohang02/03 were declared TEST after the fact and fail a model-selection-bias test; `role="final"` now structurally refuses selectors spanning scored frames; pohang04 is the only untouched data and is reserved for exactly one look [P9.2].
- **Noise floor:** paired deltas (3–16× tighter than unpaired on informative cells; corrected 2026-09-27 from 11–52×); combined draw+bootstrap 2σ floor 0.0014–0.0031 (full range 0.0000–0.0031); buoy carries 74–75% of macro variance at 5.3% of GT mass; 2 of 11 cells carry ~zero information [P9.1 item 2, P8.13]. The floor was measured on the **macro**; where buoys do not vary, ship AP's sd is about twice the macro's, so the floor understates a ship delta's noise there by up to 2× (`runs/eval/metric_noise_floor.md` §2). Say so wherever the floor judges a ship-AP delta.
- **Dependence-aware intervals:** block bootstrap, inflation 1.9–1.99× vs iid, ceiling at L=20 (2 s) because pohang03 has 117 frames; 1.95× applied project-wide; night has no estimable between-run interval (single run) [P9.1 item 6, P8.13].
- **AP convention:** local linear-interp; delta disagreement vs COCO 0.000285; absolute disagreement up to −0.0050; Ultralytics 8.4.7 vs 8.4.90 gap ~0.034 must never sit in the same table [P9.1 item 5, P4.6]. Table 1 (Ultralytics 8.4.90 validation mAP) is the declared exception (Draft 2 §5.5).
- **Metric contracts:** D-ECE conditions on confidence only; AUSE/AURC ranking-only (rank-reversal control moves AUSE 0.0630→0.3569); NLL/interval-ECE are TP-only and published with `tp_share`; AURC is a grid mean (gap to integral 0.0215, published side by side); WBF fused confidence can exceed 1.0 (max 1.7532, 0.0641% of detections), disclosed not repaired [P9.1 item 7].
- **Pre-registration and decision rules:** list the pre-registrations that gate reported verdicts: night-label restore (ALIVE bands), reprice-inherited-constants (margin 2×hypot(sd_draw, sd_paired)), soft-NMS adoption (every-cell, draw-averaged), R-D1 mechanism ablation (≥3 of 4 conditions at 0.0060), Phase 3 retrain with Amendments A1–A9, Stage 1 crossing (non-inferiority on ≥3 of 4) [P3.5, P9.1 item 3, P8.12, P7.2, P11.5, P11.3].
- **Change-impact reclassification:** after applying both corrections, 20 of 74 previously significant findings became INDETERMINATE; large effects survive (no_veto on night/fog/glare; veil repair +0.0716) [P8.13].
- **Identity checks:** a 1-frame pairing shift moves gated fusion by −0.000968, below the noise floor and catchable only by an identity check; label-blind `split_fingerprint` replaced with `label_fingerprint_trainval`; `recipe_fingerprint` added [P12.4]. One paragraph in Methods or move to Reproducibility appendix.
- **MDE before spending:** 5-seed MDE VIS 0.01291 / IR 0.01010; the retrained-vs-deployed comparison is BUDGET-CUT and may not be reported [P8.17, P11.5 A4].

### 3.7 Results

Present in this order. Each subsection names its table/figure from §5.

1. **Backbone benchmark (Table 1, patience 20; Table 1b, Phase 1)** [P4; Draft 2 §6.1]. Lead with the negative: the YOLO26 m/l/x tier is not separable (0.2666 / 0.2664 / 0.2626, gaps below seed sd). Report the n/t floor with yolo26n as the exception, and that patience barely reorders the field except yolo26n. Keep the † footnote on Table 1: 14 runs have no continuation logs in the repo, score their ep25 best, and contribute Δ = 0 by construction; never drop it or describe all 93 as replayed. Disclose for Table 1: server base runs continued on the laptop, per-family batch with nbs 64, yolov8s seed 3 for the diverged seed 2, yolo12x seed 0's missing row, val-selected epoch (not a holdout claim). Then Table 1b as the selection record: state the selection rule and why yolo26m over yolo26x (−0.0033 mAP for 1.86× two-stream throughput, 28.5 vs 15.3 FPS) [P4.5]; on Table 1 the rule picks yolo26m again (−0.0040), and yolo26l (now second) was never timed, so say so. Disclose: ultralytics-version row tagged and excluded from bare comparison; pilot campaign split irrecoverable; heterogeneous batch bounded at +0.0017; `train_time_s` is a lower bound [P4.6]. Day/night slice: night AP 0.0000 on all 27 pre-restore checkpoints; day and pooled rankings agree on top-3, so selection stands [P4.9].
2. **UQ head calibration (Table 2)** [P8.1 `table2_gaussian.md`]: VIS d_ece 0.0663, NLL 3.34, AUSE 0.088, mAP50-95 0.2580; IR d_ece 0.0344, NLL 3.29, AUSE 0.056, mAP50-95 0.0676. Report day-only as primary and pooled as secondary with the 46.2% night share in the caption; both VIS and IR are SUSPECT on the night slice and this is not label-driven [P8.15]. **Do not rank UQ arms on mAP** [P5.7]. **Do not publish a three-arm (Gaussian / MC-Dropout / Ensemble) comparison** unless decision Q3 (estimand) is resolved; if it is included, state which estimand and disclose the R-C2 mismatch and the NLL σ=0 artifact [P5.9, P8.11].
3. **Fusion robustness (Table 3)** [P8.1 `final_system` family, `docs/eval/final_system_2026-09-01.md`]: crossmodal preset clean/day ≈0.371–0.374 vs VIS-alone ≈0.368; glare/day ≈0.296–0.298 vs 0.289; night cells exactly `ir_only`; `no_veto` costs −0.002 to −0.017 on night/fog/glare; `with_maha` costs −0.02 to −0.03 on lowlight/day. Worst-cell gap +0.0000 [P6.4]. Include the evolution table of worst-cell gap: −0.0180 → −0.0632 (detector swap) → +0.0000 [P6.3–P6.5].
4. **R-D1 mechanism ablation (Table 4)** [P7.3]: reproduce the coordinate-path table verbatim (clean −0.000566 [−0.000893, −0.000041]; fog 0; lowlight −0.000001; glare +0.000406). 0/4 at every floor. Score path 0/4 at 0.0060. Include the explanation why fog/lowlight score-path positives are within-stream re-ranking, not fusion (VIS vetoed on 100% of fog frames). Include the two wrong-way results and the self-identified rule flaw (3-of-3 informative would have been correct; bias is against a positive) [P7.3].
5. **Stage 1 correspondence crossing (Table 5)** [P11.3]: A−D clean +0.0151, fog +0.0160, lowlight +0.0014, glare +0.0115; 1/4 non-inferior; S1-NULL. State that σ was live in the plumbing (changed output on 753–836 of 836 frames) yet interaction terms sit in [−0.0002, +0.0004]. Report TEST numbers but say they were not used to override, as pre-declared.
6. **Night restore (Table D / Figure)** [P3.5]: 0.0000 → 0.2520 [0.2473, 0.2567]; day guard passed (+0.0162 vs floor −0.0045; day gain is buoy-driven, likely training budget). Then the consequence: the veto's original justification is gone, and the V1/V2/V3 veto pre-registrations returned INCONCLUSIVE / INCONCLUSIVE / VOID [P8.7]. Say the shipped benchmark was not re-run under the restore, deliberately.
7. **Safety of the IR night switch (Table 6)** [P8.6]: false-night rate under fog 43.8→86.7→94.8%; glare 19.2→26.8%; blur/rain 0%; after the two-vote + multivariate health fix, both-degraded false veto 24% → 8.7% → 1.3% at zero benchmark cost [P6.4]. Both-degraded release test: prevented 0 bad vetoes, lost 2,095 correct ones, so abstain is a flag only [P8.5].
8. **Negative and inert levers (compact table)** [P8]: σ-weighted WBF (≤0.0005), registration alignment (TEST −0.0010), isotonic calibration (−0.0044/−0.0034), re-ranking at scale (best λ=0), temporal support (1.00×), TTA merging (≈1.0×), checkpoint ensembling (−0.002 to −0.095), soft-NMS (fails night 4/4 draws), cap_ratio alternatives (lose ≥1 cell), two-sided vetoes (−0.0810). Oracle headroom +0.1060 shows the ceiling lies in re-ranking/resolution, not fusion [P8.10, P8.13 `ap_by_size`].
9. **Reprice of inherited constants**: cap_ir_scale 4, iou_thr 0.85, veil repair all STAND; veil-off costs −0.0416 (7.7× margin) [P9.1 item 3].

### 3.8 Discussion

Points to make, each grounded in a cited finding:

- Why the null is the finding: where uncertainty could influence fusion it does nothing; where it shows a signal it is not fusing [P7.3].
- Fusion at this registration quality is union aggregation; agreement-based claims are out of reach; S1-NULL closed the repair route permanently [P1.8, P1.16, P11.3].
- A veto encodes a claim about the detector, not the image (the −0.0632 lesson) [P6.5].
- Value of a redundancy axis is its independence, not its abundance [P6.5].
- Synthetic corruption ladders cannot surface reference-set contamination (real night vs synthetic lowlight, 21× different response) [P6.1].
- Checkpoint selection noise exceeds seed spread; rank UQ methods on calibration, not mAP [P5.7, P5.8].
- Small deltas on video need paired, block-bootstrapped intervals and an absolute floor; a sign test on 1e-5 is not evidence [P9.1 items 1–2].

### 3.9 Limitations (mandatory list)

From [P1.16, P9.2, P3, P5, P13]:

- No untouched test set existed before pohang04; pohang02/03 were declared TEST after the fact [P9.2].
- Pohang only; adverse conditions simulated; night = one run (pohang01), no between-run night interval estimable [P1.16, P8.13].
- pohang04 has no IR labels; pohang03 IR is sparse (1,922 frames) [P3.1, P3.2].
- Registration residual 3–6 px with within-run drift; time-varying homography never built [P3.12].
- Parity gap open: σ-attached training differs from baseline by max|Δ| 3.9e-2 in early losses; gradient-norm clipping couples the branches (R-C1); `smoke_parity.py` kept RED [P5.5].
- MC/ensemble estimand mismatch open (R-C2) [P5.9].
- Mosaic-off construction never done correctly (Option B not built) [P5.6].
- DFL-derived σ (§7.2 option a) undefined on YOLO26; never reached a headline row (OQ-10) [P5.2].
- IR architecture ladder underpowered (MDE 0.02067 vs spread 0.01193); conclusion is "cannot resolve," not "equivalent" [P4.8].
- Phase 1 reproducibility: 3 of 5 acceptance criteria fail (version mix, manifest column, wiped pilot machine) [P10.3].
- The 2026-09-03 label rewrite incident (OQ-13) is unexplained; an append-only hash ledger now bounds recurrence [P3.5].
- `preset="crossmodal"` named three different systems on one day; effect inside noise (+0.0019) but disclosed [P9.3].
- Cross-machine best-epoch numbers are not comparable (laptop vs server, opposite winners by estimator) [P5.8].
- No embedded deployment; real-time claim = single-pass design + reported FPS only [P1.16].

### 3.10 Reproducibility / implementation appendix

Use [P2, P12]. Include: two machines (RTX 4080 12 GB laptop for all Phase 3; A100 MIG 3g.40gb for Phase 1 only), batch 12 for Stage 2 (and the probe-vs-production sign flip, kept anyway for pre-registration consistency) [P5.12], smoke-suite order [P2.5], dataset gate with 13 checks [P2.5], the file pointers list [P12.5], identity checks [P12.4], and the archive verification facts [P12.6]. Queue incidents [P12.2] belong in a short "what breaks in unattended GPU queues" paragraph or are omitted; do not let them crowd the main text.

---

## 4. Narrative arc to preserve

The paper should read as a chain of measured corrections, in chronological logic [P6, P9.1]:

1. Design: live uncertainty-weighted WBF.
2. First gate (2026-08-19): night blindness traced to Mahalanobis reference contamination; hard veto adopted because soft weights renormalize [P6.1].
3. Bootstrap kills the first "win" (CI [−0.0002, +0.0008]); better IR detector inverts the night sign [P6.2].
4. Fog needs a second axis (`lap_var`); lowlight/day is unfixable by any VIS statistic [P6.3].
5. Crossmodal rewrite: ask IR whether it is night; `grad_gini`; all 8 cells ≥ max(VIS, IR) [P6.4].
6. Detector swap re-prices everything; `night AND (dark OR veil)`; tune/TEST split introduced; lift screen [P6.5].
7. Statistical audit: magnitude floor, paired noise floor, AP convention, 1.95× interval inflation, metric contracts [P9.1].
8. R-D1: the mechanism the project is named for is null [P7].
9. Night restore: the blindness was a label artifact [P3.5].
10. Phase 3: S1-NULL closes correspondence; Stage 2 retrains 5+5 seeds; Stage 4 single look pending [P11].

Each step must name the measurement that forced it. Never present the final system as if it were designed that way from the start.

---

## 5. Required tables and figures

| ID | Content | Source |
|---|---|---|
| Table 1 | Patience-20 backbone benchmark, 31 variants × 3 seeds: mAP50-95 and mAP50 (Ultralytics 8.4.90 val), ep25 value, Δ, best epoch; † on the 14 runs without continuation logs | `docs/eval/bench_patience20_2026-10-08/`; Draft 2 §6.1 |
| Table 1b | Phase 1 seed-means, `main` campaign, with n, sd, FPS (pinned clock only); the selection record | P4.4, P4.5, P4.7 |
| Table 2 | Per-modality UQ calibration (D-ECE, NLL, AUSE, AURC grid-mean + integral, tp_share, ship AP and buoy AP — never the macro, which halves single-class IR; `docs/eval/table2_per_class_2026-10-08.md`), day-only primary | P8.1, P9.1 item 7, P8.15 |
| Table 3 | 8-cell fusion table: VIS-only, IR-only, naive fusion, crossmodal26m, no_veto, with_maha; CIs from block bootstrap | P8.1, P6.4 |
| Table 4 | R-D1 arms S0–S7, coordinate and score path deltas with CIs, pass counts at 4 floors | P7.2, P7.3 |
| Table 5 | Stage 1 2×2 crossing, A−D per condition, TUNE verdict and TEST (descriptive) | P11.3 |
| Table 6 | IR night-switch false-night rates by hazard × severity, before/after hardening | P8.6, P6.4 |
| Table D | Night-box filter accounting: 749,579 → 616,891 → 749,579 → 711,444 with hashes | P3.5 |
| Table N | Noise-floor summary per cell: sd_draw, sd_paired, 2σ floor, buoy variance share | P8.13, P9.1 |
| Table L | Negative/inert levers with delta and CI | P8 (see §3.7 item 8) |
| Figure 1 | Shipped decision layer: IR-night vote, VIS dark/veil, veto → WBF concat → support multiplier; annotate `w_vis = 0.9926` (`crossmodal26m`) | P1.8, P6.4; `fig_decision_layer` |
| Figure 2 | Reliability, interval coverage and sparsification error per modality, three arms, day slice | P1.14, P8.1; `fig_uq_calibration` |
| Figure 3 | Lift screen: conf 4.80×, σ 3.00×, cross-modal 2.08×, temporal 1.00× | P6.5; `fig_lift_screen` |
| Figure 4 | Best-epoch vs epoch-mean inversion for the UQ arms | P5.7; `fig_checkpoint_selection` |
| Figure 5 | Worst-cell gap evolution across the six gate rewrites | P6; `fig_gate_history` |
| Figure 6 | Table 3b per cell: VIS, IR, fused on the five Phase 3 systems | Draft 2 §6.3; `fig_phase3_cells` |
| Figure 7 | Night restore: night mAP before/after with CI and ALIVE bands | P3.5; `fig_night_restore` |

Figures are numbered in reading order and drawn by `scripts/paper_figures.py` into `docs/figures/` (PDF for the manuscript, PNG for preview); each caption names its source file.

Caption rules: every AP caption states convention (local linear-interp), substrate (2,232 paired frames or the day-only 9,284), corruption draw seeds, and whether the CI is block-bootstrapped with the 1.95× factor.

---

## 6. Claims register: forbidden phrasings

Never write:

- "uncertainty-gated fusion" / "reliability-weighted fusion" for the shipped system.
- "the fused system exploits sensor agreement" (0.05% partner rate; concatenation) [P1.8].
- "night VIS is blind" (it was untrained) [P3.5].
- "architectures are indistinguishable" for the IR ladder (underpowered) [P4.8].
- "yolo26x is the best backbone" or "yolo26l is the best backbone" (the top three are not separable in Table 1; not separable in Table 1b either) [P4.4; Draft 2 §6.1].
- "all 93 runs were trained to patience 20 and replayed" (14 are † rows without continuation logs) [Draft 2 §6.1].
- Any comparison of a Table 1 value with a Table 1b value (different labels, split, classes, AP convention, epoch budget).
- "we annotated MIT data" or any MIT/MassMIND/SMD usage [P3.13].
- "first per-frame adaptive VIS-IR fusion" (UA-CMDet, DICTA 2024) [P10.1].
- "the server is 1.23× faster" or "7% slower" (both retracted; ~4.5%) [P2.3, P12.1].
- "bit-identical baseline with σ attached" (open parity gap) [P5.5].
- "MC-Dropout leads on mAP" (best-epoch artifact) [P5.7].
- "no effect" for any delta below the floor (say "not resolved at floor X").
- Any pohang04 result unless the look has been taken and logged.

---

## 7. Number hygiene: values that were corrected, use only the right-hand side

| Superseded | Use instead | Source |
|---|---|---|
| ~1.22M boxes | 1,183,736 | P3.2 |
| pohang03 ~13k VIS | 27,085 | P3.2 |
| post-restore hash `b92739202127` (train) | `8ed69b5974ed` (train-scope); `b92739202127` is tree-scope | P3.5 |
| R-D1 floor 0.0031 | 0.0060 | P7.2 |
| IR ladder "p=0.447 ⇒ equivalent" | MDE 0.02067 vs spread 0.01193; Tukey CI [−0.00314, +0.02700] | P4.8 |
| `gauss_vis_seed0_ft` 0.22642 | 0.24750 | P5.11 |
| `ens_vis_seed0` 0.24402 | 0.25086 | P5.11 |
| "mosaic-off costs 0.029" | 0.008 | P5.11 |
| yolo12x "0.0072 seed noise" | withdrawn (batch 16 vs 8 confound) | P4.6 |
| yolo26m seed0 0.3452 | 0.3061 (8.4.90), row tagged, never compared bare | P4.6 |
| fitness = 0.1/0.9 blend | mAP50-95 alone | P4.3 |
| 16/53 INDETERMINATE | 20/74 (v4) | P8.13, P13.1 |
| 5 pohang04 lists | 15 lists / 58,144 rows | P11.5 A1 |
| "pohang04 IR needs extraction" | 22,235 frames already on disk, no labels | P11.5 A2 |
| both-degraded false veto 24% | 1.3% after hardening (24 → 8.7 → 1.3) | P6.4 |
| MIG "H100 / 80 GB" | A100 MIG 3g.40gb, 40,320 MiB, 60 SMs | P2.3 |
| unpinned-clock FPS | pinned 1500 MHz only | P4.7 |
| Pohang and PoLaRIS as one release | two releases (IJRR 2023 sensors; ICRA 2025 labels) | P10.2 |
| D-FINE arXiv 2024/2025 | ICLR 2025 | P10.2 |

Also: `final_system_veil_INERT-BUG.md` and every underscore-prefixed `runs/eval/_*.md` except `_ir_night_robustness_s8.md` are superseded; do not cite them [P8.18].

---

## 8. The pohang04 single look (Stage 4): instructions for both outcomes

As of the compiled document the look is prepared, not taken [P11.6]. The paper must be drafted so that the held-out subsection is written **before** the look, with two pre-written result paragraphs, and the author pastes in the one that the pre-registered rule selects. Nothing else in the paper changes between the two outcomes except what this section says.

### 8.1 Fixed text that appears under either outcome

Write these before the look and do not edit them afterward:

- **The rule, verbatim:** HOLDOUT-GAP is declared iff `AP_ref − AP_p04 ≥ 0.0060` **and** the 95% CI on that delta lies entirely above zero. `AP_ref = 0.2898 [0.2580, 0.3175]`, the pohang02+03 pooled development group, chosen deliberately as the **weaker** of the two development references (pohang00 group is 0.3955 [0.3352, 0.4827]) so the test asks "worse than run-to-run variation already observed in development," not "worse than the best development number" [P11.5 A9.3, P8.17].
- **The endpoint:** fused score of the shipped `crossmodal26m` system, scored against the existing upstream VIS labels only (hash `c06611a684f4`, 26,188 label files, 156,652 boxes, 286 empty). No thermal labels exist for pohang04 and none were created [P11.5 A8].
- **Aggregation:** every one of the 5 seed pairs scored (VIS seed k with IR seed k); headline = mean over 5 seeds; no constant re-tuned after seeing results [P11.5 A9.1].
- **Scope of the verdict:** all 11 benchmark cells are scored, but **only clean/clean carries the verdict**; the other 10 (fresh corruption-draw seeds 941–944 VIS / 951–954 IR) are descriptive [P11.5 A9.2].
- **Day/night:** classified by solar elevation from GPS timestamp, not by the IR night flag [P11.5 A9.4]. pohang04 contributes only to day cells (it is the all-day run whose removal raised val night share 18.2% → 23.0%) [P3.9].
- **Irreversibility:** `LOOK_TAKEN.json` is written before scoring begins; the script refuses to run unless HEAD is a clean FREEZE commit, all 316 manifest hashes match, labels verify, and the development reference reproduces exactly (0.3894193201201913) [P11.6]. State this in one sentence so the reader knows the number was produced once.
- **Development-group spread:** 0.1057 between pohang00 and pohang02+03, larger than the pre-Phase-3 expectation of 0.033 [P8.17]. This number must be in the text under both outcomes because it is the yardstick either way.
- **Exposure ledger entry:** after the look, log it in the ledger and cite the ledger in the paper [P9.2, P11.6].

Prohibited under both outcomes [P11.5 A4, A5]:
- No "retrained vs deployed" comparison (BUDGET-CUT).
- No Gaussian / MC-Dropout / Ensemble three-arm comparison on pohang04 (VIS MC/ensemble arms trained on 9,841 pohang04 frames).
- No second look, no re-scoring with a changed constant, no per-seed cherry-pick. If any of these is ever done, it needs a new pre-registration and the paper must say so.

### 8.2 Outcome A: HOLDOUT-GAP is NOT declared

Trigger: delta < 0.0060, or the CI touches or crosses zero, or both. Three sub-cases exist and the text must say which one occurred:

- **A1, clean pass:** delta < 0.0060 and CI spans zero (or pohang04 is above `AP_ref`).
- **A2, magnitude only:** delta ≥ 0.0060 but CI spans zero. Not declared, but the point estimate exceeded the floor; report both numbers and say the rule was not met because the interval did not exclude zero.
- **A3, sign only:** CI entirely above zero but delta < 0.0060. Not declared; report that the gap is statistically resolved but below the pre-registered magnitude floor, and say "not resolved at 0.0060" in the same defined-term vocabulary as the rest of the paper [§3.6, P9.1].

What the paper may claim under A:
- "Under the pre-registered rule, the shipped sensor-selection system's clean-condition AP on an untouched run is within the run-to-run variation already observed between development groups." Use this wording; do not upgrade it to "generalizes" without qualification.
- Contribution 2 (the sensor-selection baseline) gains a held-out sentence in the abstract and conclusion. Contribution 1 and 3 are unchanged.
- The exposure-ledger limitation [P9.2] stays in Limitations word for word. One clean run does not retroactively create a test set for the 74 development-set decisions.

What the paper must still say under A:
- The verdict covers **clean/day only**. No generalization claim for night, fog, lowlight or glare on held-out data; those 10 cells are reported in a descriptive table with CIs but no pass/fail language.
- pohang04 has no IR labels, so the held-out number is VIS-GT-only. The union-GT caveat (VIS-only-GT overstates coverage: 0.258 → 0.139 under union GT) applies and must be cited [P8.14].
- The comparison is against the weaker reference by design. If pohang04 lands between 0.2898 and 0.3955, say it sits inside the development spread; if above 0.3955, say so plainly and note that it exceeds both development groups, without inferring why.
- The 1.95× interval inflation and block bootstrap apply to the pohang04 CI exactly as elsewhere [P9.1 item 6].
- Do not describe the Mahalanobis OOD scorer as validated on pohang04. It is inert in the shipped preset (`mu_d=1e9, lam=0`) so the look does not test it, and the shipped fit list was 20.5% pohang04 until replaced [P7.1, P3.10]. If the cache rebuild from `maha_fit_vis_p04out.txt` was completed before the FREEZE commit, say so and cite the manifest; if not, say the OOD reference cache is unchanged and disclose the contamination.

### 8.3 Outcome B: HOLDOUT-GAP IS declared

Trigger: delta ≥ 0.0060 **and** CI entirely above zero.

What the paper may claim under B:
- The finding is reported as a **result**, not as a failure to be repaired. State the delta, its CI, `AP_ref`, and the per-seed values in a table. Then state that the development numbers in Table 3 overstate performance on an untouched run by at least the reported delta, and that this is the direct empirical consequence of the "no untouched test set" finding [P9.2].
- The exposure-ledger analysis moves from Limitations into Results as its own subsection: pohang02/03 were declared TEST after the fact, `sel("fit")==sel("day")` was literally true in code, and 74 fusion decisions were adjudicated on scored-and-tuned frames [P9.2]. Under B this is the paper's most important methodological contribution and belongs in the abstract.
- Contribution 2 is **re-scoped**: the sensor-selection baseline is "at or above max(VIS, IR) on all 8 development cells" and is reported as **not confirmed on held-out data**. The abstract sentence about the veto must carry that qualifier. Do not drop the baseline from the paper; the 8-cell development result and its measured failure cases remain reportable as development results.
- Contribution 1 (the null result on uncertainty) is **unaffected**: R-D1 and S1-NULL are within-development comparisons whose validity does not depend on pohang04, and a held-out gap makes them, if anything, more conservative (the mechanism did nothing even on frames it was tuned on).

What the paper must NOT do under B:
- **No diagnosis from the look.** The pre-registration bought one number. Any attribution of the gap to a cause (veto thresholds fit on three runs, `mu_b` fragility to leaving out pohang03 [P8.9 `x_loro_mu_b`], the 0.1057 development-group spread being larger than expected [P8.17], registration drift [P3.12], run composition) is written as a **hypothesis list for a future pre-registration**, each item with its citation, none asserted.
- No re-tuning, no re-look, no "if we relax the veto pohang04 recovers" sentence. Any such experiment would be a new pre-registration and cannot appear in this paper.
- No claim that the retrained Stage 2 seeds would have closed the gap; that comparison is BUDGET-CUT [P11.5 A4].
- Do not use the 10 descriptive cells to argue the gap is condition-specific. They are descriptive; they may be tabulated with CIs, and the text may note which cells show the largest and smallest deltas, but no pass/fail language and no mechanism inference.
- Do not soften the verdict by re-comparing against pohang00 (0.3955) or by quoting a single best seed. The rule fixed the reference and the aggregation in advance [P11.5 A9.1, A9.3].

Required wording changes under B, by section:
- **Title/abstract:** add "and a held-out gap" or equivalent; the abstract's last sentence states the delta and that the development evaluation overstates.
- **Introduction:** the third contribution becomes "an honest account of when uncertainty-driven fusion does not pay off *and* of a development-to-holdout gap measured under a pre-registered rule."
- **Results §3.7 item 3 (Table 3):** caption gains "development cells; see §Held-out for the pohang04 result."
- **Discussion:** add the point that development-set decisions on 10 Hz video with three fit runs are exposed to selection bias that the noise floor and block bootstrap do not correct for; cite the 20/74 INDETERMINATE reclassification as the within-development analogue [P8.13].
- **Limitations:** the "no untouched test set" item is promoted to the first item and references the measured gap.
- **Future work:** a pre-registered leave-one-run-out or new-recording study (decision Q1) [P13.1 B, P13.2].

### 8.4 Outcome C: look taken but no verdict produced

If the script crashed after writing `LOOK_TAKEN.json`, the look counts as taken [P11.6]. The paper then states: the single pre-registered look was attempted on date X, the marker was written, scoring did not complete, and under the pre-registration no second attempt is permitted without a new registration. Report the held-out subsection as "no result" and keep every pohang04 prohibition in §8.1. Do not quietly re-run.

### 8.5 One-page insert to prepare before the look

Prepare a single file with both result paragraphs (A and B) fully written, the descriptive 11-cell table skeleton, and the exposure-ledger line, dated before the FREEZE commit. After the look, delete the paragraph that does not apply and commit. This keeps the writing itself out of the post-look decision path.

---

## 9. Open decisions the author must settle before the final draft

From [P13.2]. Each changes what the paper can contain:

- **Q3 (UQ estimand):** decides whether a Gaussian vs MC-Dropout vs Ensemble table can appear at all. Without it, report the Gaussian head alone in Table 2 and say the comparison is deferred.
- **Q4 (parity contract):** decides whether the paper says "trains as baseline by construction" (not supportable today) or "non-inferior within a declared margin across matched seeds" (Stage 0 G3 chose this for Phase 3).
- **Q1 (partition):** decides the wording of the held-out limitation.
- **Mahalanobis cache rebuild** from `maha_fit_vis_p04out.txt`: check whether done; the paper must not say the OOD scorer is holdout-clean until it is [P3.10].

---

## 10. Writing style rules for the manuscript

- Lead every results paragraph with the verdict, then the number with its interval, then the source table.
- Report pass counts against the pre-registered rule ("0 of 4 at 0.0060"), not p-values alone.
- Use "not resolved," "inert," "null," "non-inferior," "indeterminate" as defined terms; define each once in §3.6.
- Keep design-vs-shipped distinction visible with explicit labels ("as designed" / "as shipped").
- Prefer the project's own compact formulations where exact: "a veto encodes a claim about the detector, not about the image"; "the value of a redundancy axis is its independence, not its abundance"; "where uncertainty could influence the fusion it does nothing; where it shows a signal it is not fusing anything" [P6.5, P7.3].
- Every table row that comes from a file in `runs/eval/` or `docs/eval/` names that file in a footnote so a reader can locate the full table [P8 preamble].
- Do not mix Ultralytics-version AP with custom-fusion AP in any table [P9.1 item 5].

---

## 11. Pre-submission checklist

- [ ] Title and abstract do not promise uncertainty-gated fusion.
- [ ] `w_vis = 0.9930 (constant)` is stated in Method.
- [ ] Every absolute AP states convention and substrate.
- [ ] Every delta has a block-bootstrap CI and is compared to a named floor.
- [ ] Table 4 (R-D1) and Table 5 (S1-NULL) reproduce the compiled numbers exactly.
- [ ] Night restore accounting sums: 749,579 − 132,688 = 616,891; + 94,553 = 711,444.
- [ ] Dataset counts match §3.4 (158,319 / 1,183,736 / 28,388).
- [ ] No MIT/MassMIND/SMD data claims; PoLaRIS and Pohang cited as separate releases.
- [ ] Related-work table cells for other papers are blank where not read.
- [ ] Limitations list contains every item in §3.9.
- [ ] pohang04 section either has the single logged look or an explicit "not taken" statement.
- [ ] No superseded number from §7 appears.
- [ ] Q3 and Q4 outcomes reflected in Table 2 and Method wording.
- [ ] Results CSVs were never opened in a spreadsheet application when extracting numbers [P12.2].
