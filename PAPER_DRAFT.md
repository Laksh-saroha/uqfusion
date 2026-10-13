# Does Predicted Uncertainty Help Visible–Infrared Maritime Detection? A Pre-Registered Null Result and an Image-Statistic Sensor-Selection Baseline

**Laksh Saroha**, Department of Electronics and Communication Engineering, Thapar Institute of Engineering and Technology, Patiala.
Mentor: Dr. Sandeep Mandia. UG Research Fellowship project.

> Draft 1, 2026-09-17. Text only; figures are marked as placeholders. Every number is taken from the compiled project record (`PAPER_CONTEXT_COMPILED.md`) and carries its AP convention (local linear-interpolation AP unless stated). The held-out subsection (§7) contains both pre-written outcome paragraphs because the single pohang04 look has not been taken.

---

## Abstract

Electro-optical sensors fail silently at sea: visible cameras degrade in fog, glare and darkness, and long-wave infrared loses vessels at thermal crossover. The safety risk is a confident error with no signal that the sensor has become unreliable. We built a two-stream maritime detector on the Pohang Canal dataset with PoLaRIS annotations: independent visible (VIS) and infrared (IR) YOLO26 detectors, each with a single-pass Gaussian variance head, a frame-level Mahalanobis out-of-distribution score, and a decision layer that selects streams by image statistics and merges the survivors with Weighted Boxes Fusion. We pre-registered the question the system was designed to answer: does real predicted uncertainty, attached to the boxes it was predicted for, beat the same uncertainty values shuffled onto the wrong boxes? The answer is null. Across clean, fog, low-light and glare conditions, zero of four pass a 0.0060 AP floor on either the coordinate path or the score path; the shipped fusion weight is a single constant (0.9930) on every frame. What does work is a hard, interpretable sensor-selection veto driven by an IR darkness vote and a gradient-Gini veil statistic: it sits at or above the better single stream on all eight development cells, with a worst-cell gap of +0.0000. Along the way we measured rather than assumed the evaluation machinery. The paired two-sigma noise floor is 0.0014–0.0031 AP, frame-level iid bootstrap intervals on 10 Hz video are 1.9–1.99 times too narrow, and the visible detector's apparent night blindness (0.0000 AP) was a label-filtering artifact that vanished (0.2520 AP) once 94,553 boxes were restored. Adverse conditions are simulated, night is a single run, and no untouched test set existed before the single pre-registered look at the pohang04 run reported here.

---

## 1. Introduction

Maritime detection systems increasingly pair a visible camera with a thermal camera on the assumption that the two fail in different conditions. Visible imagery degrades in fog, haze, glare, rain and darkness. Infrared sees through glare and darkness but fails during thermal crossover, when vessel and water reach the same temperature. Neither failure announces itself. A detector that has stopped seeing simply emits fewer boxes, which a naive fusion rule reads as low uncertainty rather than as a blind sensor. The problem this project set out to address is therefore not accuracy but reliability: whether the system can tell, frame by frame, which stream to believe.

Most maritime detectors in the literature (SID-YOLOv5, EG-YOLO, RDSC-YOLOv4, YOLOv7-sea, and feature-fusion networks) output no uncertainty at all. Outside the maritime domain, single-pass localization variance is established (Gaussian YOLOv3) and uncertainty-aware cross-modal fusion is established too (UA-CMDet, 2022; DICTA 2024). We claim neither the variance head nor the idea of conditioning fusion on uncertainty as new. What is thin is the maritime evidence: whether such uncertainties are calibrated on paired VIS and LWIR maritime video, and whether the uncertainty doing the conditioning is itself trustworthy when measured with dependence-aware intervals against a stated noise floor, with null results reported.

This paper is a pre-registered study whose primary hypothesis was rejected. We designed a system in which a per-modality reliability score, combining box-level aleatoric variance and frame-level distributional distance, would weight the two streams instant by instant. We built it, measured every component, and found that the mechanism does not move the fused result. We report that null with the decision rule fixed in advance, and we report the mechanism that did work in its place.

Three contributions are defensible:

1. A controlled maritime uncertainty study that publishes negative results, with a measured noise floor, dependence-aware confidence intervals, declared metric contracts, and pre-registered decision rules.
2. A lightweight, interpretable sensor-selection baseline, a hard veto of the form `night AND (dark OR veil)` computed from image statistics, that reaches or exceeds max(VIS, IR) on all eight development cells, with its failure cases documented.
3. An honest account of when uncertainty-driven fusion does not pay off, at the level of the fusion weight (the R-D1 ablation) and at the level of cross-modal correspondence (the Stage 1 crossing).

The project's stance, stated in its scope document and enforced throughout, is that null results are results: decisions were pre-committed, stop rules were honoured, and every time a held-out number was looked at it was logged.

---

## 2. Related work

### 2.1 Uncertainty in single-stage detectors

Gaussian YOLOv3 (Choi et al., ICCV 2019) attaches a per-coordinate Gaussian to the box regressor and trains it with a negative log-likelihood. We follow that pattern. Heteroscedastic NLL has a known failure mode in which the network explains away hard examples by inflating variance instead of improving the mean (Seitzer et al., 2022); the beta-NLL weighting and a warm-up in which the mean trains under the plain box loss first (Skafte et al., 2019) are the standard mitigations, and we use both. Deep Evidential Regression was considered and not benchmarked. MC-Dropout and deep ensembles are the reference alternatives for producing the uncertainty signal on the same backbone; §3.5 and §6.2 explain why a three-arm comparison is not reported here.

### 2.2 Visible–infrared fusion with uncertainty

UA-CMDet (Sun et al., 2022) performs drone-based RGB–IR vehicle detection with uncertainty-aware learning and illumination-aware NMS at inference, which is genuinely per-frame adaptive. A DICTA 2024 paper (doi 10.1109/DICTA63115.2024.00029) proposes uncertainty-aware cross-modality fusion for visible–infrared detection. An earlier draft of this project claimed that existing VIS–IR fusion is static; that claim was wrong and is withdrawn. Learned attention is not static merely because its parameters are frozen at inference: the attention values still depend on the input.

### 2.3 Maritime detectors and datasets

The maritime YOLO variants listed in §1 report accuracy without uncertainty. The Pohang Canal dataset (Chung et al., IJRR 2023) is the sensor release; PoLaRIS (arXiv 2412.06192, ICRA 2025) is a separate, later annotation release providing YOLO-format boxes for two classes. The Singapore Maritime Dataset presents visible and infrared material separately and is not a pre-paired fusion testbed. MassMIND's seven categories are segmentation classes, not a ship/buoy taxonomy, and no instance-to-class mapping exists in our repository. The MIT Marine Perception dataset was planned and never onboarded; no data, code or annotation from it exists in this work.

### 2.4 Comparison axes

Table R positions this work against the nearest prior systems. Cells for other papers' calibration protocols, registration assumptions and compute are left blank rather than filled from memory; they will be completed from a direct read of each paper.

**Table R. Positioning.**

| Work | Domain | Sensors | Uncertainty target | Inference-time adaptation | Calibration evaluated | Registration assumption | Compute |
|---|---|---|---|---|---|---|---|
| Gaussian YOLOv3 | road | RGB | box coordinates | none | — | n/a | single pass |
| UA-CMDet (2022) | drone | RGB + IR | — | illumination-aware NMS | — | — | — |
| DICTA 2024 | generic | VIS + IR | — | — | — | — | — |
| Maritime YOLO variants | maritime | RGB (some IR) | none | none | no | n/a | single pass |
| This work | maritime | VIS + LWIR | box coordinates (σ²), frame OOD (Mahalanobis) | hard veto on image statistics; fusion weight constant 0.9930 (measured) | yes: D-ECE, interval-ECE, NLL, AUSE/AURC, declared metric contracts, measured noise floor | nearest-timestamp pairing, 3–6 px median residual | two single-pass detectors, 28.5 FPS two-stream |

---

## 3. Dataset

### 3.1 Pohang Canal and PoLaRIS

The Pohang Canal dataset covers a 7.5 km route through canal, inner and outer port, and near-coastal water, recorded by the KAIST MORIN lab. It provides stereo visible video at 2048×1080 and 10 Hz and a thermal camera at 640×512, 16-bit, 10 Hz. PoLaRIS supplies YOLO-format boxes for two classes, ship and buoy, under CC BY-NC 4.0. There are five runs: pohang00 (day, dense in both modalities), pohang01 (night), and pohang02 to pohang04 with varying IR coverage. pohang04 has no IR labels at all. The two cameras are not spatially co-registered; frames are paired by nearest timestamp, and the residual misalignment was measured rather than assumed at 3–6 px median per run, with within-run swings of up to about 10 px.

### 3.2 Verified counts

All counts below were re-derived from disk on 2026-09-10 and supersede earlier estimates in the project record.

| Quantity | VIS | IR | Total |
|---|---:|---:|---:|
| Images | 127,309 | 31,010 | 158,319 |
| Boxes | 962,960 | 220,776 | 1,183,736 |
| Paired VIS–IR frames | | | 28,388 |

Per-run image counts (VIS / IR): pohang00 21,768 / 10,918; pohang01 24,473 / 11,995; pohang02 27,795 / 6,175; pohang03 27,085 / 1,922; pohang04 26,188 / 0. Paired rows per run: pohang00 10,786; pohang01 11,990; pohang02 3,739; pohang03 1,873; pohang04 0. Pairing is defined by the dataset's own timestamp table, not by frame ordinal: 16,544 of the 28,388 pairs have different VIS and IR indices, with per-run offsets ranging from −155 to +1.

### 3.3 Preprocessing

Visible frames are letterboxed, not stretched, from 2048×1080 to 640×338 inside a 640×640 canvas (scale 0.3125, 151 px of pad value 114 top and bottom, area interpolation). Stretching would distort VIS (1.9:1 native) and IR (1.25:1) by different factors and hurt cross-modal overlap. Forty-seven percent of every stored VIS frame is therefore pad, a fact that matters in §3.5. Full-resolution originals are archived and hard-linked in a native-resolution twin used for main-backbone training.

Infrared frames are delivered as 8-bit grayscale by per-frame min–max normalization (each frame's own minimum to 0 and maximum to 255) and letterboxed from 640×512 with 64 px of pad top and bottom. This choice has consequences we disclose: thermal crossover can be visually masked because a near-isothermal vessel is stretched to full local contrast, there is no cross-frame radiometric comparability, and the raw 16-bit data cannot be recovered from the product. The 16-bit dynamic range is modest (median span 702 counts, about 7.72 effective bits), but 51 percent of frames are more than 1.5× range-inflated by outlier hot pixels. A percentile-clip re-export was built, tested and rejected: it gained 1.75 percent mAP50-95 but lost 1.6 points of recall.

### 3.4 Splits

The first split delivered with the tooling interleaved frames (train N, validation N+1), which leaks near-duplicate frames at 10 Hz. It was replaced by contiguous per-run blocks, and then, when the server-side validation set came out 77 percent buoy against a 5 percent global share, by an interleaved K-block split: each run's shared timeline is cut into K equal blocks in a cycle of ten (block index 4 to validation, 9 to test, others to train, giving 80/10/10 by construction), with guard bands at every boundary and K raised until leakage, balance and coverage gates all pass. The same ordinals are used for both modalities so that VIS–IR pairs and stereo pairs never straddle a split.

**Table S. Split sizes (images).**

| | Train | Val | Test | Duplicates across splits | Temporal-proximity violations |
|---|---:|---:|---:|---:|---:|
| VIS | 96,275 | 11,352 | 11,445 | 0 | 0 |
| IR | 23,279 | 2,234 | 2,518 | 0 | 0 |

Per-run VIS (train/val/test): pohang00 16,376/1,672/1,912; pohang01 18,826/2,068/1,629; pohang02 20,424/2,690/2,864; pohang03 20,962/2,579/2,649; pohang04 19,687/2,343/2,391. Per-run IR: pohang00 8,229/836/950; pohang01 8,800/1,034/1,111; pohang02 4,844/247/330; pohang03 1,406/117/127. Both audits pass. Production VIS training uses a stride-2 subset of 48,136 train frames; IR trains on all 23,279.

### 3.5 The night-box filter and its reversal

This is the dataset event that most shaped the project. On 2026-07-15 a train-only filter removed every box in any pohang01 frame whose content-median luminance was below 100 on a 0–255 scale, on the reasoning that a box in a black frame is a modality-copied annotation rather than something the visible camera can see. The filter emptied 17,502 of 96,275 train label files and dropped 132,688 boxes (126,948 ship, 5,740 buoy). Validation and test were never touched.

The filter had a bug. Its pad-detection constant treated only near-black pixels as padding, but VIS letterboxing pads with the value 114, so the "content median" was dominated by the 47 percent of gray pad pixels. The statistic that actually ran was closer to "frames whose content 95th-percentile luminance is below 100." A per-box audit later scored 120,829 of the deleted boxes on intensity, gradient and contrast tests. Only 38,135 failed all three; 82,694 would not individually have failed but were deleted because their whole frame was cut, and the flagged and kept populations overlap on two of the three axes.

A pre-registered restore (2026-09-02) put all 132,688 boxes back and re-dropped only the 38,135 individually flagged, a net gain of 94,553 boxes. The endpoint was night-only VIS mAP50-95 with bands DEAD below 0.005, WEAK 0.005–0.02, ALIVE at or above 0.02, and a day guard floor of −max(2·sd_paired, 0.002). The label accounting reconciles exactly.

**Table D. Train-scope label accounting.**

| State | Boxes | Train-label hash |
|---|---:|---|
| Pre-filter | 749,579 | `fd60c0834fdd` |
| After frame-level cut | 616,891 | `287b11c50b5a` |
| After full restore | 749,579 | — |
| After re-dropping 38,135 flagged | 711,444 | `8ed69b5974ed` |

The result is in §6.6. Two further provenance facts belong here. First, a widely quoted post-restore hash `b92739202127` is a tree-scope hash over all 127,309 VIS label files, not the train-scope hash; the train-scope value is `8ed69b5974ed`. Second, on 2026-09-03 at 21:19, 7,591 pohang01 train label files were rewritten back to pre-filter content with no project script running; the cause is unexplained. An append-only hash ledger now records label state so that any recurrence has a bounded window, and it separately found 8,237 orphan label files belonging to no split list.

### 3.6 Holdout and contamination

Before the Phase 3 pre-registration, pohang04 (26,188 VIS images, no IR) had been used for nothing. Phase 3 lists were built by filtering the existing lists to remove pohang04 rows, preserving every surviving frame's stride identity: VIS train 48,136 to 38,295, val 11,352 to 9,009, test 11,445 to 9,054. IR lists needed no change. Removing an all-day run shifts composition: validation night share rises from 18.2 percent to 23.0 percent, so Phase 3 numbers are not comparable to earlier pooled validation numbers.

A contamination audit found that the Mahalanobis reference list used to fit the shipped OOD scorer contained 819 of 4,000 frames (20.5 percent) from pohang04. A clean replacement list of 3,181 frames exists. Whether the reference cache was rebuilt from it before the freeze is stated in §7. The paired evaluation lists that feed every benchmark cell (2,232 frames) contain no pohang04 frames.

---

## 4. Method

We describe the system in two layers and label them. The first is what was designed. The second is what ships and what every result in §6 measures. Every divergence between the two was measured, not assumed.

### 4.1 As designed

Each modality runs its own YOLO backbone with separate weights. From each backbone a Gaussian σ² head produces per-box aleatoric uncertainty and a Mahalanobis distance on backbone features produces a frame-level out-of-distribution score. Per frame and per modality m:

- size-normalized per-box uncertainty `u_i = ¼ Σ_t σ_{i,t} / s_{i,t}` over t in {x, y, w, h}, with s the box width or height;
- confidence-weighted frame aggregate `U_box,m = Σ_i c_i u_i / Σ_i c_i`;
- calibrated OOD score `O_m = sigmoid((d_m − μ_d) / τ)`;
- reliability `R_m = r_frame,m · r_box,m` with `r_box,m = exp(−λ U_box,m)` and `r_frame,m = 1 − O_m`, falling back to `r_frame,m` on empty frames;
- temporal smoothing `R̄_m(t) = α R_m(t) + (1 − α) R̄_m(t−1)`;
- fusion weight `w_m = R̄_m / (R̄_vis + R̄_ir)` fed to Weighted Boxes Fusion (WBF).

The design document itself marked every constant and the multiplicative form as choices to be validated empirically. They were, and the validation is the subject of this paper.

### 4.2 As shipped (preset `crossmodal26m`)

**Detectors.** VIS uses yolo26m with two classes. IR uses yolo26m with a P2 feature neck and a single class, because IR cannot see buoys: its buoy AP50-95 is 0.0002. Backbone selection is in §6.1.

**Gaussian head.** A fresh log-variance branch (`cv4`) is bolted in place onto the live detection head of a loaded model; there is no fork of the training library. Variance is parameterized as log σ² over left-top-right-bottom distances in stride units and converted to pixels at inference, riding through post-processing as extra channels. Training adds a fourth loss term with beta-NLL weighting and a warm-up during which the NLL weight is zero. The σ branch reads detached features and the NLL sees a detached mean, so the deterministic detector is intended to train identically to the baseline by construction; §9 reports that this parity is not yet demonstrated. Porting to YOLO26's end-to-end head forced three changes: σ rides the one-to-one branch only, because inference decodes from it; post-processing is overridden to gather σ with the boxes' top-k index; and the NLL target is left unclamped because at reg_max = 1 the stock clamp collapses every target to a constant. One ablation, training σ on undetached features, cannot be run on end-to-end heads because the library detaches the branch upstream.

**Mahalanobis OOD score.** Backbone features are captured by a forward hook and scored against a Ledoit–Wolf covariance fit on a reference set of clean training frames. In the shipped preset the parameters `mu_d = 1e9` and `lam = 0` make this score mathematically inert in the fusion weight. It is retained as a diagnostic and discussed in §6.3 and §8.

**Decision layer.** The shipped decision layer is a hard sensor-selection veto followed by union aggregation.

1. *Night vote.* The IR stream is asked whether it is night by a 5th-percentile luminance test (`ir_p05 > 41.5`, fitted once on clean IR and frozen). A second vote requires that VIS also reads dark, an IR self-check on the ratio of Laplacian to variance (`lap_over_var`) guards against a degraded IR frame voting, and an 11-statistic multivariate health score with an authority bound at the 99th percentile of clean statistics limits how far a suspect IR frame can influence the decision.
2. *Veil statistic.* A scale-free Gini coefficient of gradient magnitude (`grad_gini`) detects fog and veiling on a single frame with no temporal filter; it was 100 percent correct on fog and 0 percent false-positive elsewhere during development.
3. *Rule.* VIS is vetoed when `night AND (dark OR veil)`. A vetoed stream is removed from the WBF input list rather than down-weighted, because WBF renormalizes whatever weights it is handed; down-weighting alone left a genuinely blind stream with weight 0.432.
4. *Single survivor.* A frame with one surviving stream returns that stream's detections untouched.
5. *Merge.* Surviving streams are passed to WBF at IoU threshold 0.85 with constant capability-prior weights fitted run-disjointly. At this threshold only 0.05 percent of VIS boxes have an IR partner, so the merge is concatenation; we document the merge as effectively off.
6. *Cross-modal support.* A score multiplier is applied to boxes with a partner at IoU 0.30 (γ = 0.5). It confirms but never moves a coordinate.
7. *IR dedup.* The IR stream is de-duplicated with NMS at 0.70 before merging.

**What is off.** `sigma_weighted = False` (σ never moves a fused coordinate), `sigma_score_alpha = 0` (σ never moves a fused score), the Mahalanobis weight is inert as above, and VIS soft-NMS is off following a pre-registered rejection (§6.8). The measured consequence is that the fusion weight `w_vis` takes exactly one value, 0.9930, across all 2,232 paired evaluation frames and all four corruption conditions.

*(Figure 1 placeholder: the shipped decision layer, IR night vote and VIS dark/veil statistics feeding the veto, survivors concatenated by WBF, support multiplier applied, annotated with `w_vis = 0.9930`.)*

---

## 5. Experimental protocol and statistics

### 5.1 Benchmark cells and substrates

The development benchmark uses 2,232 paired VIS–IR frames from the validation split. Eight cells cross four conditions (clean, fog, low-light, glare) with day and night. Later grids add IR-side corruptions for ten or eleven cells. Adverse conditions are simulated with Albumentations (fog, sun flare, rain, motion blur, Gaussian and ISO noise); they are not field-collected. Night frames come from a single run, pohang01. Day-only slices on 9,284 frames are used where noted.

### 5.2 Tune and test discipline

From 2026-09-01 onward every lever was tuned on TUNE = pohang00 (836 paired frames) and reported on TEST = pohang02 + pohang03 (364 frames), with pohang01 excluded from fitting. The split caught a real overfitting trap immediately: a support IoU of 0.55 won on TUNE but lost on all six held-out variants, while 0.30 won on all six. Under the earlier single-set practice both would have read as wins and the wrong one would have been adopted.

We state plainly that this repository contained no untouched test set before the pohang04 look. pohang00 and pohang01 were held out of gate fitting but scored repeatedly; pohang02 and pohang03 were declared TEST after the fact and already fail a model-selection-bias test, because a candidate was rejected specifically for losing on them; and the code's "fit" selector was literally identical to its "day" selector, so every fit-run day constant was reported on the frames it was tuned on. A `role="final"` evaluation context now structurally refuses any frame selector that spans scoring frames, and it is the mechanism that protects the single look in §7.

### 5.3 Noise floor

Deltas between systems are always paired on the same frames and the same corruption draw. Pairing tightens the standard deviation by 11–52× relative to unpaired resampling. The combined draw-plus-bootstrap two-sigma floor on a paired delta is 0.0014–0.0031 AP on most cells (full range 0.0000–0.0031). The buoy class carries 74–75 percent of macro-metric variance while making up 5.3 percent of day ground-truth boxes, and two of eleven cells carry essentially no buoy-variance information. A delta below the floor is reported as "not resolved," never as "no effect."

### 5.4 Dependence-aware intervals

Frames at 10 Hz are autocorrelated, and a frame-level iid bootstrap underestimates the standard error. We measured the inflation with block bootstraps up to block length L = 20 (2 s), the longest the shortest run (pohang03, 117 paired frames) permits: intervals widen by at least 1.9× and up to 1.99×. A factor of 1.95 is applied to every interval from 2026-09-10 onward, and it is the reason the Phase 3 magnitude floor is 0.0060 rather than 0.0031. Night, being a single run, has no block length at which a between-night-run interval is estimable.

### 5.5 AP convention

All absolute AP values use local linear-interpolation AP with the project's maximum detection count, not COCO AP. The two conventions disagree on deltas by at most 0.000285, five times below the noise floor, so no decision can flip on convention; they disagree on absolutes by up to −0.0050, so every absolute value states its convention. Two training-library versions (8.4.7 and 8.4.90) disagree on mAP50-95 by about 0.034 for identical weights and data; no table in this paper places numbers from different library versions side by side.

### 5.6 Metric contracts

Six claims about the uncertainty metrics were tested and hold: D-ECE conditions on confidence only; AUSE and AURC are ranking-only (a rank-reversing control moves AUSE from 0.0630 to 0.3569); NLL and interval-ECE are computed on true positives only and are published with their true-positive share and recall denominators; AURC is a grid mean, whose gap to the trapezoidal integral (0.0215) is published alongside. One defect was found and is disclosed rather than repaired: WBF can emit fused confidences above 1.0 (maximum 1.7532, on 0.0641 percent of detections) because two overlapping same-stream boxes count as confirmation. This traces to WBF mechanics, not to cross-modal support.

### 5.7 Pre-registrations and decision rules

Every verdict in §6 and §7 was fixed in advance. The registrations and their rules are: the night-label restore (ALIVE bands and a day guard); the re-pricing of inherited constants (margin 2·hypot(sd_draw, sd_paired)); soft-NMS adoption (every cell non-negative, draw-averaged); the R-D1 mechanism ablation (at least three of four conditions above a 0.0060 floor with block-bootstrap CI excluding zero); the Phase 3 retrain with nine append-only amendments; the Stage 1 correspondence crossing (non-inferiority within 0.0060 on at least three of four conditions); and the pohang04 single look (§7). After the interval and convention corrections were applied retroactively, 20 of 74 previously significant fusion findings became indeterminate; the large effects (removing the veto on night, fog and glare; the veil repair at +0.0716) survived.

### 5.8 Identity checks and power

A one-frame shift in cache pairing moves gated fusion by −0.000968, below the noise floor and therefore undetectable by any statistical test; it is catchable only by a content identity check, which the evaluation code now performs on every cache load. The split fingerprint was found to be label-blind (deleting a box leaves it unchanged) and was supplemented with a label fingerprint. Before spending Phase 3 compute we computed minimum detectable effects: with five seeds, VIS resolves 0.01291 and IR 0.01010, neither reaching the 0.0060 floor, so the pre-registration cut the retrained-versus-deployed comparison in advance and it is not reported.

---

## 6. Results

### 6.1 Backbone benchmark is a negative result

Ninety-three training runs across 31 YOLO variants were consolidated into one record. Table 1 gives the main campaign, ship-only, on the stride-2 VIS split, 100 epochs with patience 20 (no run reached 100).

**Table 1. Phase 1 backbone benchmark, mAP50-95 seed mean ± sd (main campaign, local AP, ship class).**

| Variant | n | mAP50-95 | FPS single / two-stream (fp16, clock pinned 1500 MHz) |
|---|---:|---:|---|
| yolo26x | 3 | 0.3049 ± 0.0020 | 30.7 / 15.3 |
| yolo26m | 3 | 0.3016 ± 0.0050 | 57.0 / 28.5 |
| yolo12x | 2 | 0.3007 ± 0.0049 | — |
| yolo26l | 3 | 0.2998 ± 0.0026 | — |
| yolo12m | 3 | 0.2906 ± 0.0046 | — |
| yolo12l | 3 | 0.2870 ± 0.0096 | — |
| yolo26s | 3 | 0.2813 ± 0.0045 | — |
| yolo12s | 3 | 0.2783 ± 0.0109 | — |
| yolo26n | 3 | 0.2540 ± 0.0058 | — |

The top eight variants span 0.0055 mAP50-95 against seed standard deviations of 0.0010–0.0066. They are not separable. The nominal leader's margin over the next variant (0.0016–0.0033) is below its own seed sd, and the ordering is not even stable across metric-reading conventions. The one robust finding is a capacity floor: every nano- and tiny-scale model across five families lands between 0.2486 and 0.2567.

We selected yolo26m under the rule fixed in advance (top mAP50-95, then DFL-present, then simplest fork, then FPS). No YOLO26 variant has a DFL head, so the tie inside the leading group's pooled sd falls to throughput, where 26m wins decisively for a two-detector system: 28.5 versus 15.3 FPS two-stream, a factor of 1.86, at a cost of −0.0033 mAP that exceeds 26x's own sd and is disclosed as the soft spot of the decision.

Disclosures. One row (yolo26m seed 0) was trained under library 8.4.7 and re-scored under 8.4.90 (0.3452 to 0.3061); it is tagged and never compared bare. A second, pilot campaign of 66 rows used a split whose machine was later wiped, so cross-campaign contamination cannot be quantified; only yolo12s ran in both (0.2783 versus 0.2810, inside seed sd). Batch size varied 8–32 across machines; the bounded effect is +0.0017, below the smallest seed sd. One row is inadmissible because a resume bug let it train only 14 epochs past its own peak against a required 20. Reported training time (266.8 h) is a lower bound because resumed segments were not summed. Slicing the 27 archived checkpoints by day and night without retraining shows night AP of exactly 0.0000 on all 27 (these predate the restore in §3.5) and day-only and pooled rankings agreeing on all top-three positions, so the selection stands.

A 44-of-93-run IR architecture ladder was stopped early on the basis of an ANOVA (F(12,26) = 1.037, p = 0.447) that was misread as evidence of equivalence. The minimum detectable spread at that design was 0.02067 against an observed spread of 0.01193, and the Tukey HSD interval on the largest gap is [−0.00314, +0.02700]. The correct statement is that the ladder could not resolve architecture differences, not that the architectures are equivalent. The stop stands on other grounds: the architecture was frozen before the queue was created, and the ladder trained two classes while the deployed IR configuration is single-class with a P2 neck that was never in the ladder.

### 6.2 Per-modality uncertainty calibration

**Table 2. Gaussian head calibration on 2,232 paired validation frames, no fusion (local AP).**

| Stream | D-ECE | NLL (TP-only) | AUSE | mAP50-95 |
|---|---:|---:|---:|---:|
| VIS σ head | 0.0663 | 3.34 | 0.088 | 0.2580 |
| IR σ head | 0.0344 | 3.29 | 0.056 | 0.0676 |

Day-only slices are the primary reporting basis; the pooled substrate is 46.2 percent night and is reported as secondary. On the night slice both VIS and IR fall in the SUSPECT band for calibration, and the same band on both streams indicates the night miscalibration is not label-driven: retraining on restored labels would not repair it.

The σ head is evaluated here as a localization-error-scale estimate on its own terms, not as a fusion input. Signal screens support that reading: boxes with σ below the per-frame median are true positives 3.00× more often than chance (3.39× at IoU 0.75), the second-strongest signal after confidence (4.80×). But σ is informative about error magnitude, not direction: leave-one-run-out ridge regression finds live out-of-fold R² on only two of four box edges (0.017–0.098), and applying a σ-driven coordinate correction end to end costs 0.02–0.11 mAP.

We do not rank uncertainty methods on mAP. On the best-epoch checkpoint convention an MC-Dropout arm led the ensemble by +4.53 sd; on the epoch-mean convention the same run was the worst arm at −3.00 sd. The best checkpoint is a maximum over roughly ten noisy validation epochs and rewards the noisiest run; the between-seed sd of best fitness (0.00212) is two to five times smaller than the within-run epoch-to-epoch sd (0.0035–0.0106). The same recipe run on two machines gave opposite winners depending on the estimator. A three-arm Gaussian, MC-Dropout and ensemble table is not reported in this draft because the MC and ensemble estimand is unresolved: the disagreement statistic computed among surviving detections omits the within-member variance term, produces σ = 0 and undefined NLL when members agree, and treats M = 5 as one replicate rather than five. A decision on the estimand is pending (§9).

### 6.3 Fusion robustness of the sensor-selection baseline

The decision layer in §4.2 is the sixth rewrite. Table 3a records the worst-cell gap to the better single stream at each stage; each rewrite was forced by a measured failure of the previous one.

**Table 3a. Worst-cell gap of gated fusion versus max(VIS, IR) across gate rewrites.**

| Stage (date) | Change | Worst cell | Gap |
|---|---|---|---:|
| a (08-19) | p05 photometric term, hard veto | fog/night | −0.0021 vs ir_only, later found within CI |
| b (08-20) | soft term dropped as redundant; hysteresis | fog/night | tie |
| c (09-01) | Laplacian-variance veil axis | lowlight/day | −0.0180 |
| d (09-01) | crossmodal: IR night vote, grad_gini | — | +0.0000 |
| e (09-01) | detector swap to 26m breaks veil veto | fog/clean | −0.0632 |
| e′ (09-01) | rule reordered to `night AND (dark OR veil)` | — | +0.0000 |

Under the final preset, clean/day gated AP is 0.371–0.374 against 0.368 for VIS alone, and glare/day is 0.296–0.298 against 0.289, both clear of zero. Night cells are bit-identical to IR-only by construction, because VIS is vetoed on 100 percent of night frames and the single survivor is passed through. Fog/day, where an earlier veto deleted the better stream, recovers from 0.0192 to 0.0908 after the rule reorder; turning the veil repair off costs −0.0416 on fog/clean, 7.7 times the pre-registered margin. Removing the veto entirely costs 0.002–0.017 on night, fog and glare. Re-enabling Mahalanobis weighting costs 0.02–0.03 specifically on lowlight/day. Capability-prior weights, the merge threshold of 0.85 and the veil repair were all re-priced against the measured noise floor and all stand.

Three findings from the rewrite history carry general lessons. First, the original night blindness of the fusion traced to the Mahalanobis reference set, which contained 782 of 4,000 frames from the night run, so night was in-distribution by construction and scored cleaner than day (D_night 28.4 versus D_day 30.0). The same raw darkness produced a 21× different fusion response depending on whether it was synthetic low-light (w_vis = 0.037) or real night (w_vis = 0.792). No corruption ladder could have surfaced this; a day-only refit separates correctly (D_night 89.0 versus D_day 30.9). Second, VIS brightness alone cannot distinguish a dark world from a dark sensor: synthetic low-light day has p05 = 0, darker than real night (p05 = 2.5–3.5), yet VIS still works on it. Asking the IR stream whether it is night is the right axis. Third, the veil veto's −0.0632 regression on detector swap shows that a veto encodes a claim about the detector, not about the image: the new VIS detector's own fog AP rose 41× (0.0020 to 0.0824), removing the justification for vetoing it there.

*(Figure 2 placeholder: worst-cell gap across the six rewrites.)*

### 6.4 The uncertainty mechanism is null (R-D1)

The system was designed so that predicted uncertainty would weight the fusion. Under the shipped preset it does not, and we asked whether it could. The pre-registered comparison is not real σ against a constant, since inverse-variance weighting with equal σ reduces analytically to plain WBF, but real σ against the same σ values shuffled onto the wrong boxes. Eight arms were scored: S0 shipped; S1–S4 on the coordinate path (real, constant, shuffled within frame, shuffled across cache); S5–S7 on the score path (real, constant, shuffled within frame), with α fixed at 1.0. The rule was at least three of four conditions with delta above 0.0060 and a block-bootstrap CI excluding zero.

**Table 4. R-D1, real minus shuffled σ (S1 − S3), coordinate path, 2,232 paired frames.**

| Condition | Delta | 95% CI (block, ×1.95) | Excludes zero |
|---|---:|---|---|
| clean | −0.000566 | [−0.000893, −0.000041] | yes, negative |
| fog | +0.000000 | [0, 0] | no |
| lowlight | −0.000001 | [−0.000004, +0.000008] | no |
| glare | +0.000406 | [−0.000123, +0.000764] | no |

Zero of four conditions pass at any floor tested (0.0014, 0.0031, 0.0060, 0.0100). On the score path (S5 − S7), fog (+0.003960) and low-light (+0.001995) exclude zero but clean (−0.004420) and glare (+0.000514) do not; zero of four at the 0.0060 floor. Real σ against the shipped system with no σ at all (S1 − S0) differs by at most 0.000089 in magnitude on any condition. Turning on the mechanism the project is named for changes essentially nothing.

The score-path positives on fog and low-light are not evidence of fusion working. VIS is vetoed on 100 percent of fog frames, so no cross-modal merge occurs there; the score path acts on single-stream frames and what it measures is within-stream re-ranking. Where uncertainty could influence the fusion it does nothing; where it shows a signal it is not fusing anything.

Two results point the wrong way and are reported. On clean, shuffled σ beats real σ on the coordinate path, statistically but not practically (ten times below the floor). On clean, real-σ score re-ranking is worse than a constant: S5 − S6 = −0.013157 [−0.022630, −0.000678], a cost of 0.0167 absolute AP. One flaw in the rule is self-identified: the bar should have been three of three informative conditions, since fog is structurally incapable of a coordinate-path effect. The flaw biases against a positive verdict and therefore makes the null conservative.

Independent probes agree. Inverse-variance WBF differs from stock WBF by at most 0.0005 on any cell. Adding σ to the fusion score moves TEST by +0.0010 with a CI spanning zero. The Mahalanobis weight was already inert in the shipped preset.

### 6.5 Relaxing correspondence does not rescue the mechanism (Stage 1)

The R-D1 null was measured at a merge threshold (0.85) where almost nothing merges. Two prior negative results on correspondence (a per-frame registration refinement that raised the partner rate 80× to 4.02 percent but lowered held-out AP by 0.0025; relaxing the threshold to 0.55, which cost −0.0138) were each measured with σ inert. Phase 3 Stage 1 crossed the two: threshold {0.85, 0.55} × σ-weighting {off, live}. Cells A (shipped), B (shipped threshold, σ on; bit-identical to A) and C (relaxed, σ off; C − A = −0.0138) were known. Cell D (relaxed and σ live) was the experiment. D had to be non-inferior to A within 0.0060 on at least three of four conditions on TUNE.

**Table 5. Stage 1 crossing, A − D on TUNE (pohang00), block bootstrap.**

| Condition | A − D | 95% CI | Non-inferior |
|---|---:|---|---|
| clean | +0.0151 | [+0.0095, +0.0210] | no |
| fog | +0.0160 | [+0.0104, +0.0221] | no |
| lowlight | +0.0014 | [−0.0001, +0.0040] | yes |
| glare | +0.0115 | [+0.0070, +0.0157] | no |

One of four: verdict S1-NULL, robust at every floor from 0.0014 to 0.0100. The lone pass is not a rescue; low-light passes because relaxing correspondence barely costs anything there (C − A = −0.0013), not because live σ recovered anything. σ was confirmed live in the plumbing (it changed the fused output on 753–836 of 836 clean frames), yet the interaction terms B − A and D − C sit inside [−0.0002, +0.0004] everywhere with every CI spanning zero. TEST numbers were computed and are reported for completeness: two of four non-inferior, with fog favouring the rejected relaxed setting by +0.0028. As pre-declared, they were not used to override the TUNE verdict. The correspondence question is closed; no threshold below 0.55 will be tried under this project, and the fusion is documented as union aggregation, not consensus.

### 6.6 Night visible blindness was a label artifact

Every night table before 2026-09-02 showed VIS mAP50-95 of exactly 0.0000. After the restore of §3.5, a 25-epoch fine-tune from the existing VIS checkpoint (early-stopped at epoch 20, best at epoch 10, 5.82 h) gave night-only VIS mAP50-95 of 0.2520 [0.2473, 0.2567] (se 0.0024), 12.6 times the ALIVE threshold; mAP50 rose from 0.0000 to 0.4957. The day guard passed (+0.0162 against a floor of −0.0045), with the caveat that the day gain is buoy-driven and likely reflects extra training budget rather than the restore.

The visible detector was not blind at night. It was untrained at night. The 0.0000 was manufactured by the filter. Two consequences follow. The shipped checkpoint is still genuinely 0.0000 at night, so the shipped veto still fires on 100 percent of night frames and the benchmark in §6.3 was deliberately not re-run. And the veto's original justification is gone: three subsequent pre-registered attempts to re-price the night veto with a night-capable VIS detector returned INCONCLUSIVE, INCONCLUSIVE and VOID, and that axis is closed. A related repair, adding a VIS-independent confirmation to the veil mechanism, halves the worst night gap from −0.0296 to −0.0141 without touching day.

### 6.7 Is the IR night switch safe when IR is corrupted?

The night vote trusts the IR stream. IR was uncorrupted in every benchmark cell, so we tested the frozen rule `ir_p05 > 41.5` against six IR hazards at three severities. A false night on a clear day vetoes a working VIS stream and is the dangerous direction.

**Table 6. False-night rate of the raw IR night rule under IR corruption.**

| Hazard | Severity 1 | Severity 2 | Severity 3 | Separable |
|---|---:|---:|---:|---|
| blur | 0% | 0% | 0% | yes |
| fog | 43.8% | 86.7% | 94.8% | no at s2, s3 |
| glare | 19.2% | 20.3% | 26.8% | no at any severity |
| lowlight | 0% | 0% | 0% (but 100% missed night) | no |
| noise | 0% | 0% | 0% (missed night 16.1% at s3) | yes |
| rain | 0% | 0% | 0% | yes |

The raw rule is unsafe under IR fog and glare. The two-vote requirement, the IR self-check, the multivariate health score (which detects IR glare at 64–75 percent against 12–34 percent for a single axis) and the authority bound took the false-night rate on 19 IR-corruption arms from 94.8 percent to 0 percent at zero benchmark cost, all eight cells bit-identical before and after, and the both-degraded worst-case false-veto rate from 24 percent to 8.7 percent to 1.3 percent. The remaining 1.3 percent is a missed-detection problem, not a switch-logic problem. An abstain signal derived from system-level reliability was implemented and demoted to an advisory flag: releasing the veto when both sensors are flagged prevented zero bad vetoes and lost 2,095 correct ones over 76 both-flagged pairs, and under a corrected risk–coverage metric the abstain ordering does not beat random on zero of four conditions.

### 6.8 Levers that are inert or negative

**Table L. Fusion and post-processing levers tested and not adopted (paired deltas, TEST unless noted).**

| Lever | Result | Verdict |
|---|---|---|
| σ-weighted WBF | ≤ 0.0005 on every cell | inert |
| σ in fusion score | +0.0010, CI spans zero | inert |
| Per-frame registration alignment, then merge | TEST −0.0010, spans zero (TUNE +0.0121) | does not generalize |
| Isotonic score calibration | TUNE/TEST −0.0044 / −0.0034 | hurts |
| Score re-ranking, leave-one-run-out, 2,232 frames | +0.0037 OOF (in-sample +0.0149) | small |
| Score re-ranking, day-only 9,284 frames | best λ = 0, delta 0.0000 | null at scale |
| Temporal support | lift 1.00× | dead |
| Test-time-augmentation view merging | ≈1.0× lift once confidence-matched | inert |
| Two-checkpoint ensembling, coordinate merge | −0.002 to −0.095 | hurts |
| Within-stream WBF dedup | −0.003 to −0.005 mAP50-95 | hurts |
| VIS soft-NMS σ = 0.5 | night worst cell −0.0000, negative on 4/4 draws | rejected by pre-registered every-cell rule |
| Capability-ratio alternatives (×4, ×16, ×64) | win clean, lose ≥ 1 cell | rejected by no-cell-may-lose rule |
| Two-sided veto (also veto IR) | worst gap −0.0810 on IR-corrupted night | unsafe |
| Top-k truncation, k ≥ 50 | < 0.0002 | irrelevant |

The one fusion lever that survives held-out testing is the cross-modal support multiplier at IoU 0.30 (TEST +0.0033 [+0.0010, +0.0067]). The pattern across levers is that merging as a family, whether cross-modal, within-modal, across checkpoints or across augmentation views, is exhausted; the test-time-augmentation case is the cleanest, because its views are pixel-exact registered and merging still barely moves. The value of a redundancy axis is its independence, not its abundance: confidence lifts true-positive rate 4.80×, σ 3.00×, cross-modal agreement at IoU 0.30 2.08×, and temporal persistence 1.00×, because persistent false positives are the most stable objects in a fixed scene.

The headroom lies elsewhere. Oracle re-ranking on the paired substrate would raise mAP50-95 from 0.3233 to 0.4293 (+0.1060), and on day-only VIS from 0.2771 to 0.4059; small objects carry 89.8 percent of ground-truth mass with the lowest AP; cross-modal union adds only +0.01–0.02 recall over VIS alone. Resolution and ranking, not fusion, are the levers, and both are out of scope for a detector held fixed.

The soft-NMS rejection deserves one more sentence. The first pass had no magnitude floor, so a night regression of −1.03e-5 failed the bar exactly as a regression of 1e-2 would have. The project declined to invent an equivalence margin after seeing that it would flip the verdict, and instead measured the noise floor of §5.3. The draw-averaged re-test still failed on four of four draws, and the rejection stands.

---

## 7. Held-out evaluation: the single pohang04 look

### 7.1 Protocol (fixed before the look)

pohang04 (26,188 VIS images, no IR labels) was used for nothing before the Phase 3 pre-registration and is scored exactly once. HOLDOUT-GAP is declared if and only if `AP_ref − AP_p04 ≥ 0.0060` and the 95 percent block-bootstrap interval on that delta lies entirely above zero. `AP_ref = 0.2898 [0.2580, 0.3175]` is the pooled pohang02 + pohang03 development group, deliberately the weaker of the two development references (the pohang00 group is 0.3955 [0.3352, 0.4827]) so that the test asks whether pohang04 is worse than run-to-run variation already observed in development, not worse than the best development number. The spread between the two development groups is 0.1057, larger than the 0.033 expected before Phase 3.

The endpoint is the fused score of the shipped `crossmodal26m` system against the existing VIS labels (26,188 files, 156,652 boxes, 286 empty; hash `c06611a684f4`). No thermal labels exist for pohang04 and none were drawn. All five seed pairs from the Phase 3 retrain are scored, VIS seed k with IR seed k, and the headline is the mean over seeds; no constant is re-tuned after seeing results. All eleven benchmark cells are scored with fresh corruption-draw seeds, but only the clean/clean cell carries the verdict; the other ten are descriptive. Day and night are classified by solar elevation from the GPS timestamp; pohang04 contributes only to day cells.

The look is mechanically single-shot. The scoring script refuses to run unless the repository is at a clean FREEZE commit, all 316 files in the hash manifest (10 checkpoints, 10 reference caches, 190 pohang04 caches, 76 frame-statistic files, and substrate and calibration files) verify, the labels hash correctly, and the development reference reproduces exactly (0.3894193201201913). It writes a `LOOK_TAKEN` marker before scoring begins, so a crash mid-look still counts as the look having been taken. The exposure is logged in the project's ledger.

Under this pre-registration no retrained-versus-deployed comparison is reported (the five-seed design cannot power it), and no Gaussian versus MC-Dropout versus ensemble comparison is reported on pohang04 (the VIS MC and ensemble arms were trained on 9,841 pohang04 frames).

### 7.2 Result

*As of this draft the look has not been taken. Both pre-written outcome paragraphs follow; the one selected by the rule is retained and the other deleted, with no other edits.*

**[Outcome A — HOLDOUT-GAP not declared.]** The clean/clean fused AP on pohang04, averaged over five seed pairs, is ____ [____, ____]; the delta against `AP_ref` is ____ [____, ____]. The pre-registered rule is not met because [the delta is below 0.0060 / the interval does not exclude zero / both]. Under the pre-registered rule, the shipped sensor-selection system's clean-condition AP on an untouched run is within the run-to-run variation already observed between development groups. The verdict covers clean day only; the ten descriptive cells are tabulated with intervals and carry no pass or fail language. The number is VIS-ground-truth-only, and the union-ground-truth caveat of §6.8 applies. The comparison is against the weaker development reference by design; [pohang04 sits inside the development spread / exceeds both development groups]. The Mahalanobis OOD scorer is not validated by this look: it is inert in the shipped preset, and [its reference cache was rebuilt from the pohang04-free list before the freeze / its reference cache is unchanged and was fit on a list that was 20.5 percent pohang04]. One clean run does not retroactively create a test set for the development decisions of §6, and the limitation in §9 stands.

**[Outcome B — HOLDOUT-GAP declared.]** The clean/clean fused AP on pohang04, averaged over five seed pairs, is ____ [____, ____]; the delta against `AP_ref` is ____ [____, ____], meeting both conditions of the pre-registered rule. The development results of §6.3 overstate performance on an untouched run by at least this delta. This is the direct empirical consequence of the finding in §5.2 that the repository contained no untouched test set: pohang02 and pohang03 were declared TEST after the fact, the fit and day selectors were identical in code, and 74 fusion decisions were adjudicated on scored-and-tuned frames. The sensor-selection baseline is therefore reported as at or above max(VIS, IR) on all eight development cells and not confirmed on held-out data. The null results of §6.4 and §6.5 are unaffected: they are within-development comparisons, and a held-out gap makes them, if anything, more conservative, since the mechanism did nothing even on frames it was tuned on. The pre-registration bought one number and no diagnosis. Candidate explanations are listed as hypotheses for a future registration, none asserted: the veto thresholds were fit on three runs and the brightness constant moves by 5.5 points when one fit run is left out; the development-group spread of 0.1057 already exceeded expectation; registration drift varies within and between runs; and run composition differs. No re-tuning, second look or best-seed selection is permitted or reported.

---

## 8. Discussion

**The null is the finding.** The system was built so that uncertainty would decide how to fuse. Measured against shuffled uncertainty with a floor set by the data's own noise and intervals corrected for autocorrelation, it decides nothing: zero of four conditions on both paths, and a fusion weight that is one constant on every frame. The result narrows the thesis from uncertainty-gated fusion to image-statistic sensor selection, and it repositions the Gaussian head as an error-scale estimate to be judged on its calibration, not as a fusion input.

**Fusion at this registration quality is aggregation.** With a 3–6 px median residual and within-run drift, 0.05 percent of VIS boxes have an IR partner at the adopted threshold. The system does not measure sensor agreement; it concatenates. Stage 1 closed the obvious repair: relaxing correspondence and letting σ arbitrate the merges is non-inferior on one of four conditions. Any claim about cross-modal agreement is out of reach of this system, and the design assumption that decision-level fusion tolerates misalignment was wrong as stated; decision-level fusion avoids the residual by almost never merging.

**A veto is a claim about the detector.** The veil veto that helped a small VIS detector on fog became a −0.0632 regression on a larger one whose fog AP was 41 times higher. Sensor-selection rules must be re-priced whenever the detector changes, and the rule that survives, `night AND (dark OR veil)`, is the one that asks the other sensor first.

**Redundancy is worth what it is independent of.** Temporal persistence and view agreement carry no signal beyond confidence; cross-modal agreement carries some; σ carries more. Any future fusion term should be screened for lift against confidence before it is priced in a benchmark.

**Synthetic ladders cannot find reference contamination.** The Mahalanobis scorer trusted real night because the night run was in its fit set, while distrusting synthetic low-light that looked the same. No severity sweep exposes a statistic that is wrong about the reference population; only a falsification test with the population changed does.

**Checkpoint selection is noisier than seeds.** A best-epoch maximum over a noisy curve rewards noise, and it inverted the ranking of uncertainty methods and the ranking of machines. Comparisons of uncertainty methods should be carried by calibration metrics, where separations are large relative to this noise, and every headline number should report the epoch-mean alongside.

**Small deltas on video need a floor and a block.** Without a magnitude floor a gate degenerates into a sign test on 1e-5; without block bootstrapping intervals are half their true width. Applying both corrections retroactively moved 20 of 74 findings to indeterminate. The findings that survived are the large ones, which is the pattern one should expect and want.

---

## 9. Limitations

1. No untouched test set existed before pohang04. pohang02 and pohang03 were declared TEST after the fact and fail a selection-bias test. Nested leave-one-run-out validation cannot be run because no uncontaminated held-out data existed until pohang04, which is reserved for one look.
2. Pohang only. Adverse conditions are simulated with Albumentations. Night is a single run, and no between-run night interval is estimable.
3. pohang04 has no IR labels, so the held-out number is VIS-ground-truth-only; pohang03 IR is sparse (1,922 frames). Only 8.7 percent of IR ground-truth boxes are unambiguously novel objects, and VIS-only ground truth overstates coverage (clean VIS AP 0.258 falls to 0.139 under union ground truth).
4. Registration residual is 3–6 px median with within-run drift up to 10 px; a time-varying homography was never built.
5. The bit-identical parity of σ-attached training with the baseline is not demonstrated: early losses differ by up to 3.9e-2 even with byte-identical data streams, and shared gradient-norm clipping couples the branches even though σ's loss contribution is detached. The parity smoke test is kept failing until resolved.
6. The MC-Dropout and ensemble estimand is unresolved, which is why no three-arm uncertainty comparison appears.
7. Mosaic augmentation was on throughout training and off at validation; the correct single-run mosaic-off construction was never built. A continuation-based check was neutral (+0.00004) but resets EMA state.
8. DFL-derived variance is undefined on the selected backbone (reg_max = 1) and never reached a headline row.
9. The IR architecture ladder was underpowered (minimum detectable spread 0.02067 against 0.01193 observed); its conclusion is "cannot resolve," not "equivalent."
10. Phase 1 reproducibility: one row spans two library versions, the data-manifest column does not pin the computation that ran, and the pilot campaign's machine was wiped.
11. A 2026-09-03 rewrite of 7,591 label files is unexplained; an append-only hash ledger bounds any recurrence.
12. The preset name `crossmodal` referred to three configurations on one day, differing in constants absent from saved config blocks; the largest per-cell effect (+0.0019) is inside the noise floor. Later preset names are unambiguous.
13. Best-epoch numbers are not comparable across machines; all Phase 3 training was therefore done on one machine.
14. The shipped Mahalanobis reference list contained 20.5 percent pohang04 frames; the scorer is inert in the fusion weight, and the status of the cache rebuild is stated in §7.
15. Real-time is claimed only as single-pass design plus measured throughput on a desktop GPU; no embedded deployment was performed.

---

## 10. Reproducibility and implementation notes

**Machines.** All Phase 3 training ran on one laptop (RTX 4080, 12 GB, Windows) with a CUDA-probed system interpreter; the repository's own virtual environment carries CPU-only torch, and frame statistics must run under it because its numeric libraries reproduce the development statistic files bit-exactly (the GPU interpreter's float32 sums differ by up to 4.4e-7). Phase 1 ran on an A100 MIG 3g.40gb slice (40,320 MiB, 60 SMs) with Jupyter-only access; it is about 4.5 percent faster than the laptop at steady state, and two earlier contrary measurements were retracted as artifacts. Windows pages rather than raising out-of-memory on an over-large batch (a silent 17× slowdown), so batch ceilings were probed by measured throughput: IR 14, VIS 12 by the probe, and 12 was kept for both in Stage 2 for pre-registration consistency even after a production run measured batch 16 as 16 percent faster.

**Gates.** A CPU smoke suite runs before any GPU time: benchmark, Gaussian head on the DFL family, Gaussian head on the end-to-end family (both must pass, since σ rides different branches), the UQ pipeline, and Phase 3. A dataset gate re-implements split-fingerprint, leakage, night-filter-hash and balance checks independently of the package (13 checks) and compares train-label content hashes across machines. A contamination gate asserts that pohang04 is absent from every training-consumed list; its first version silently loaded zero images while reporting a pass, and the lesson that a gate verifying the wrong property is worse than no gate is recorded.

**Identity.** Cache loads validate payload shape, frame count, image paths and detection arrays; pairing is checked by content, not length; a label fingerprint supplements the filename-only split fingerprint (VIS `ae7fa57efb2b`, IR `5fd58f37c799`); a recipe fingerprint over ten training knobs including a content hash of the weights file prevents a completed-run lookup from matching a different recipe. Hashing 107,627 VIS label files costs 15–24 s warm per grid launch.

**Queue.** Long unattended GPU queues failed in ways worth recording: a paused queue that did not exit its process, so a second runner woke both; zombie CUDA contexts holding memory while the device reported no processes (31 runs lost); a divergence watcher whose window filled exactly at the learning-rate warm-up peak and killed a healthy run (fixed, and verified to change exactly one verdict over 20 historical runs); and a genuine Stage 2 divergence at epoch 6 correctly caught by requiring two consecutive epochs of rising validation loss while training loss still fell. Results CSVs must never be opened in a spreadsheet application, which truncated floats and mangled a version string into a date.

**Code.** The σ head, fusion and veto, temporal filters, preset assembly, AP and bootstrap, COCO-parity harness, identity checks, label-path resolution, grid driver, and the freeze and look scripts for §7 are in the public repository (github.com/Laksh-saroha/uqfusion). Every benchmark row carries a split fingerprint and class tag so that a re-split or filter change cannot be silently averaged against older rows. Phase 1 archives were verified file-by-file (696/696 and 2,281/2,281 with CRC32 and SHA-256 spot checks) before the original tarballs were deleted.

---

## Acknowledgements

Placeholder.

## References

To be completed from direct reads. Entries required by the text: Chung et al., IJRR 2023 (Pohang Canal, arXiv 2303.05555); PoLaRIS, arXiv 2412.06192, ICRA 2025; Choi et al., ICCV 2019 (Gaussian YOLOv3); Sun et al., 2022 (UA-CMDet); DICTA 2024, doi 10.1109/DICTA63115.2024.00029; Seitzer et al., 2022 (beta-NLL); Skafte et al., 2019 (variance warm-up); RT-DETR, CVPR 2024; D-FINE, ICLR 2025; Weighted Boxes Fusion; Ledoit–Wolf covariance; Albumentations; SID-YOLOv5, EG-YOLO, RDSC-YOLOv4, YOLOv7-sea.
