# A Sensor-Selection Rule That Outlived Its Detector: A Pre-Registered Study of Uncertainty and Visible–Infrared Fusion for Maritime Detection

**Anonymous authors** (CVF workshops normally review double-blind; restore the author block for the camera-ready).

> Venue draft for the Maritime Computer Vision workshop (MaCVi), derived from `PAPER_DRAFT2.md` at commit `d9260ef` (2026-10-09). The target is 8 pages in CVF format, excluding references. Every number is taken from Draft 2, which cites the source file for each; none is new. Venue facts and open items are in `venues/README.md`.

---

## Abstract

Visible cameras at sea fail in fog, glare and darkness, and thermal cameras lose vessels at thermal crossover. Neither failure announces itself. We built a two-stream maritime detector on the Pohang Canal dataset with PoLaRIS boxes: a visible (VIS) and a long-wave infrared (IR) YOLO26 detector, each with a single-pass Gaussian variance head, and a decision layer meant to let predicted uncertainty decide which stream to believe. We pre-registered the test of that idea and report what it found, together with what shipped instead.

On the registered metric, ship AP, real uncertainty beats the same values shuffled onto the wrong boxes when it re-ranks scores (three of four conditions above a 0.0060 floor with the corruptions it was registered on, one of four with physically modelled ones). It does nothing as a fusion weight (zero of four). It carries information, but only about a detector's own boxes. Spending it costs AP against using no uncertainty on six of eight cells, and a pre-registered replication of a learned variant on five retrained detectors failed its floor.

What shipped is a hard veto computed from image statistics, followed by union aggregation. By day it beats the visible stream alone on every cell (+0.0059 to +0.0107 AP, five retrained systems). Its night arm was justified by a label-filtering artifact that made the visible detector score 0.0000 at night. Once the labels were restored and the detector retrained, visible night AP was 0.2535, and the unchanged rule discarded it, costing 0.1847 AP on clean night. A sensor-selection rule is a claim about the detector, not the scene, and it must be re-priced whenever the detector changes.

---

## 1. Introduction

Maritime perception stacks increasingly pair a visible camera with a thermal camera, on the assumption that the two fail in different conditions. That assumption is only useful if the system can tell, frame by frame, which sensor has failed. A blind detector emits fewer boxes, and a naive fusion rule reads fewer boxes as confidence. This work set out to answer the selection question with predicted uncertainty. It ended by answering it with image statistics, and then by measuring how that answer broke.

We make three contributions, each tied to a pre-registered decision rule.

1. **A pre-registered test of uncertainty-driven fusion, with a split answer** (§5.1). Predicted box variance (σ) is informative but not useful. As a fusion weight it is null. As a score re-ranker it beats a shuffled control, yet it is below the system without σ, and its signal is within each detector, not between the sensors.
2. **A sensor-selection baseline and a measured account of how it broke** (§5.2). An image-statistic veto improves on the visible stream by day. At night it encoded a property of one detector, and it failed when that detector was retrained on corrected labels.
3. **An evaluation protocol for small deltas on maritime video** (§4). It combines a paired noise floor, block-bootstrap intervals for 10 Hz autocorrelation, magnitude floors, a declared class set, and one logged look at an untouched run.

## 2. Related work

Gaussian YOLOv3 (Choi et al., 2019) attaches a per-coordinate Gaussian to the box regressor; we use the same head and a beta-NLL loss (Seitzer et al., 2022). UA-CMDet (Sun et al., 2022) and Zhao et al. (2024) condition visible–infrared detection on uncertainty, but their uncertainty weights the training loss and is removed at inference. Neither evaluates calibration. UA-CMDet's illumination-weighted NMS, which demotes the RGB branch on an image statistic, is the closest prior to our shipped veto. Maritime YOLO variants (Liu et al., 2021; Zhao et al., 2023) report accuracy without uncertainty. The Pohang Canal dataset (Chung et al., 2023) supplies synchronized visible and thermal video, and PoLaRIS (Choi et al., 2025) supplies ship and buoy boxes on both. MassMIND (Nirgudkar et al., 2023) is infrared-only, and the Singapore Maritime Dataset (Prasad et al., 2017) does not pair its visible and near-infrared views, so neither is a paired fusion testbed.

## 3. Data and system

**Data.** Pohang Canal follows a 7.5 km route through canal, port and near-coastal water, with stereo visible video at 2048×1080 and a thermal camera at 640×512, both at 10 Hz. There are five runs:

* pohang00, day;
* pohang01, night;
* pohang02 and pohang03, day with sparser IR;
* pohang04, which has no IR labels and is held out.

The cameras are not co-registered. Frames are paired by timestamp (28,388 pairs), with a measured residual misalignment of 3–6 px median per run. The development benchmark uses 2,232 paired validation frames (1,200 day, 1,032 night). It crosses four VIS conditions (clean, fog, low-light, glare) with day and night. The conditions are simulated. Off-the-shelf filters (Albumentations; Buslaev et al., 2020) turned out to be broken for this purpose (their fog is a fixed 21-px blur, their low light an additive clip), so the results here use a camera-chain model fitted to training frames: fog as attenuation with per-frame metric depth, low light as exposure scaling to the real night level with the measured night noise, glare with a spread and exposure response fitted to a real sun event and real night lights. Night comes from one run.

**Detectors.** Both streams use yolo26m. VIS is two-class (ship, buoy). IR is single-class with a P2 neck, because the thermal detector cannot see buoys (buoy AP 0.0002). Ship is therefore the primary class, and every AP below is local ship AP50-95 unless marked otherwise. A log-variance branch on each detection head predicts σ for the four box edges.

**Decision layer (shipped, Figure 1).** IR votes whether it is night with a 5th-percentile luminance test, and may vote only while an 11-statistic health score says the IR frame still looks like IR. VIS is vetoed when `night AND (dark OR veil)`, where `dark` is VIS 5th-percentile luminance and `veil` is a low Gini coefficient of VIS gradient magnitude. Surviving streams are merged by Weighted Boxes Fusion (Solovyev et al., 2021) at IoU 0.85 with constant capability weights. The fusion weight is therefore one value, 0.9926, on every frame. The cameras are misregistered and the threshold is strict, so only 0.05 percent of VIS boxes find an IR partner, and the merge is in effect concatenation. A box with a cross-modal partner at IoU 0.30 gets a score boost. σ and the Mahalanobis out-of-distribution score are computed but move nothing.

![decision_layer](../docs/figures/fig_decision_layer.png)

**Figure 1. The shipped decision layer.** Frame statistics from both streams drive a hard veto on VIS; the survivors are merged with constant weights, so the merge is concatenation in practice. σ and the Mahalanobis score are computed but inert. The night threshold and the IR health model are fitted in-sample (§6).

## 4. Evaluation protocol

**Paired deltas and a noise floor.** Every comparison is paired on the same frames and the same corruption draw, which tightens a delta's standard deviation 3–16× over unpaired resampling. On ship AP the two-sigma floor of a paired delta is 0.0008–0.0024. A delta below it is reported as "not resolved", never as "no effect".

**Autocorrelation.** At 10 Hz, frame-level iid bootstraps understate intervals. Moving-block bootstraps (Künsch, 1989) with blocks of up to 2 s widen them 1.9–1.99×. That factor sets the magnitude floor of every Phase 3 verdict at 0.0060; built from the ship floor it would be 0.0047. Applied retroactively, these corrections moved 20 of 74 earlier fusion findings to indeterminate.

**Class set.** IR emits no buoys, so a VIS veto deletes every buoy, and a ship-and-buoy macro moves by half a class for reasons unrelated to ship detection. On our data buoys also carry 74–75 percent of the macro's variance. The registered metric is ship AP.

**Pre-registration and the held-out run.** Each verdict below was registered before its run, with a decision rule and a floor. pohang04's verdict was scored exactly once, by a script that refuses to run unless every checkpoint and cache hash verifies. Its corrupted cells were re-scored once more after the corruptions were replaced (§3), a second exposure we disclose; the verdict was not re-scored.

## 5. Results

### 5.1 Uncertainty is informative but does not improve the fusion

**The test.** The registered comparison is real σ against the same σ shuffled onto the wrong boxes. It is not real σ against a constant, because inverse-variance fusion with equal σ reduces to plain WBF. σ enters on one of two paths:

* the coordinate path, where it weights box coordinates in the merge;
* the score path, where it multiplies each box's score by (frame-median σ / σ)^α, with α = 1 fixed before the run.

A path passes if at least three of four conditions show a delta of at least 0.0060 with a block-bootstrap interval excluding zero.

**Table 1. Real minus shuffled σ, and the score path against no σ (S5 − S0). Shipped preset, gated-fusion ship AP, 2,232 paired frames, block-bootstrap 95% CI, the registered (off-the-shelf) corruptions.**

| Condition | Coordinate path | Score path | Score path vs no σ |
|---|---|---|---|
| clean | −0.0001 [−0.0003, +0.0001] | **+0.0073** [+0.0033, +0.0124] | −0.0329 [−0.0411, −0.0234] |
| fog | −0.0000 [−0.0001, +0.0002] | **+0.0120** [+0.0094, +0.0142] | −0.0029 [−0.0056, −0.0010] |
| lowlight | −0.0000 [−0.0000, +0.0000] | +0.0034 [+0.0012, +0.0048] | −0.0083 [−0.0116, −0.0050] |
| glare | −0.0000 [−0.0001, +0.0001] | **+0.0068** [+0.0031, +0.0114] | −0.0266 [−0.0339, −0.0190] |
| passing at 0.0060 | 0 of 4 | **3 of 4** | — |

The coordinate path is null: used as the fusion weight it was designed to be, σ changes nothing. The score path passes, with the corruptions it was registered on; with the camera-chain corruptions it passes on clean alone, so that pass depends on the corruption model. Real σ carries information about which boxes are right, and the predecessor preset gives the same verdicts.

Against the system with no σ, however, the score path is below on every cell here, and on six of eight across both presets. Emptying the IR stream shows that the signal is re-ranking within each detector. By day, re-ranking VIS boxes alone accounts for the gain (+0.0153 to +0.0167), and adding IR moves it by −0.0040 to +0.0035. At night VIS is vetoed, so the night part is IR re-ranking IR boxes. σ does not tell the system which sensor to believe.

**A learned variant does not rescue it.** A re-ranker learned out of fold from confidence and σ gains +0.0063 over no re-ranking on the visible detector. σ's increment over the same re-ranker without σ is +0.0050 [+0.0038, +0.0064], but that test was not pre-registered. Its registered replication on five retrained detectors required four of five seeds to clear the floor. σ's increment was positive on all five (mean +0.0053), but it cleared 0.0060 on two and 0.0047 on three, so the replication fails.

**The metric is part of the registration.** Both recorded runs first scored the ship-and-buoy macro. The macro read NULL on both paths, because σ re-ranks buoys worse than chance on clean and glare, and that loss cancels the ship gain. Re-scoring on the registered metric reversed the score-path verdict. Scoring the macro was a deviation from the registration, and we report it as one.

### 5.2 A sensor-selection rule is a claim about the detector

**The rule was built through measured failures.** The veto in Figure 1 is the sixth rewrite of the decision layer, and each rewrite answered a measured failure of the one before it. The clearest example came at a detector change. Swapping in a larger VIS detector turned a fog veto that had helped into a −0.0632 regression, without any change to the images: the new detector's own fog AP was 41 times higher, so vetoing it on fog threw away the better stream. Reordering the rule to `night AND (dark OR veil)` repaired it.

**The night arm rested on a label artifact.** Every night table before the restore showed VIS at exactly 0.0000. A train-only filter had removed 132,688 night boxes on a luminance statistic. A bug in its padding constant meant it cut whole frames rather than individually invisible boxes. A pre-registered restore put back all but 38,135 boxes. A 25-epoch fine-tune then lifted night VIS from 0.0000 to 0.2520 [0.2473, 0.2567], 12.6 times the pre-registered ALIVE threshold (Figure 2). The visible detector had not been blind at night. It had been untrained at night.

![night_restore](../docs/figures/fig_night_restore.png)

**Figure 2. Night VIS AP before and after the label restore** (2,068 night validation frames, all ship). The dashed line is the pre-registered ALIVE threshold.

**The retrained system inherits the rule and pays for it.** We retrained both streams on the restored labels, five seeds each, with the held-out run removed. We then re-measured the frozen rule on those five systems (Table 2, Figure 3).

**Table 2. The shipped rule on the five retrained systems. Seed-mean ship AP, clean IR, development frames, camera-chain corruptions; deltas with between-seed 95% t-intervals.**

| Cell | VIS only | IR only | Fused | Fused − VIS |
|---|---:|---:|---:|---|
| clean / day | 0.3485 | 0.0218 | 0.3566 | +0.0081 [+0.0047, +0.0115] |
| fog / day | 0.0106 | 0.0218 | 0.0239 | +0.0133 [+0.0065, +0.0200] |
| lowlight / day | 0.1719 | 0.0218 | 0.1790 | +0.0070 [+0.0047, +0.0094] |
| glare / day | 0.1581 | 0.0218 | 0.1701 | +0.0120 [+0.0084, +0.0156] |
| clean / night | 0.2535 | 0.0687 | 0.0687 | **−0.1847** [−0.2039, −0.1655] |
| fog / night | 0.0439 | 0.0687 | 0.0687 | +0.0248 [+0.0172, +0.0324] |
| lowlight / night | 0.2390 | 0.0687 | 0.0687 | **−0.1703** [−0.1862, −0.1543] |
| glare / night | 0.0833 | 0.0687 | 0.0687 | **−0.0146** [−0.0268, −0.0024] |

By day the veto never fires, and union aggregation beats VIS alone on all four cells; under fog, where VIS falls below the clean IR stream, it also sits at or above IR. At night the rule is right only under fog, the one night cell where the retrained VIS detector falls below IR, and wrong on the other three, where VIS still works. Low light barely changes a night frame that is already at the night exposure level. With the off-the-shelf corruptions the rule had looked right under low light too, because their additive clip blacked out 99.9 percent of each night frame. The rule gates on darkness, but what should decide is VIS health (Figure 4 shows both on single frames). Under the old detector, which scored zero at night, the two coincided.

Three pre-registered attempts to re-price the night arm, made before the retrain, failed or were voided, and no held-out night run exists to validate a replacement. The rule is reported as frozen and wrong for the detector it ships with.

![phase3_cells](../docs/figures/fig_phase3_cells.png)

**Figure 3. The shipped rule on the five retrained systems (Table 2).** By day the fused output sits just above VIS. At night it equals IR alone: right under fog, where VIS falls below IR, wrong on the three night cells where VIS still works.

![detections](../docs/figures/fig_detections.png)

**Figure 4. The shipped rule on one day frame and one night frame, clean and fogged** (retrained system seed 0; fog on VIS only, IR clean in every row; ship class, boxes at confidence ≥ 0.25; IR warped into the VIS view by the per-frame homography; night VIS brightened for display only; frames chosen for legibility from renders showing ground truth only, never a detection). By day both streams are kept, but IR boxes enter the merge with their scores scaled by IR's small capability weight, below the display threshold. Under fog by day VIS finds nothing at the display threshold. At night the IR vote drops VIS on both frames, regardless of the ferry VIS finds on each (best IoU 0.94 clean, 0.71 fogged). One frame shows the mechanism; Table 2 gives the cost.

**Is trusting IR's night vote safe?** IR was uncorrupted in every benchmark cell, so we attacked the vote with IR blur, fog and noise at three severities. A false night on a clear day vetoes a working VIS stream. The raw rule misreads 61.4 percent of clear days as night under dense IR fog and 96.3 percent under heavy IR noise, because the dataset's per-frame stretch lifts a damaged day frame's dark percentile toward the night level.

Two votes, the multivariate health score and an authority bound take the false-night rate to 0 percent on all nine IR arms, at zero benchmark cost. That 0 percent is in-sample (§6). The weak-IR fallback is the hole: when IR is fogged or noisy and VIS is in low light, it vetoes VIS on up to 96 percent of day frames, because an underexposed day frame looks like a fogged night to the two VIS statistics the fallback reads. With the off-the-shelf corruptions this both-degraded worst case had read 1.3 percent. Priced on development frames, the fallback costs 0.0067 to 0.0125 ship AP at moderate IR damage and 0.11 to 0.16 at severe damage. A fogged IR also hurts without any veto: its scattered boxes earn VIS false positives the cross-modal support bonus, which costs up to 0.0113 where it is worth +0.0068 with clean IR.

### 5.3 What one held-out run can say

pohang04's verdict was scored once, under a pre-registered rule: a held-out gap is declared if the development reference minus pohang04 is at least 0.0060 with an interval above zero. Fused ship AP on its 12,482 day pairs, averaged over the five systems, is 0.2682 [0.2576, 0.2793]. The gap is +0.0216 [−0.0120, +0.0502]. The point estimate clears the floor, but the interval excludes neither zero nor a gap of 0.05. The two development reference groups themselves differ by 0.1057. One untouched run buys one number, not a generalization claim.

## 6. Limitations

* **One held-out run**, and it is now spent. pohang02 and pohang03 were declared test runs after the fact.
* **Simulated adverse weather and a single night run.** The camera-chain corruptions rest on one real sun event, the real night lights and stated assumptions (IR fog extinction, fog patchiness); the data contain no real fog. No between-run night interval is estimable.
* **In-sample night constants.** The IR night threshold and health model were fitted on data that includes the evaluated night run.
* **Frozen night rule.** The shipped night rule is wrong for the shipped detector, and it is reported, not repaired.
* **Two checkpoint generations, not pooled.** The R-D1 results are on pre-restore checkpoints.
* **Unvalidated fixed strength.** α was fixed and never tuned, so whether another strength beats the system without σ is untested.
* **Detector-only throughput.** 57.0 FPS is one detector's fp32 prediction step on a laptop GPU, with the clock pinned at 1500 MHz because an unpinned sweep ranked some larger models faster than smaller ones; the two-stream figure is derived, not measured.

## 7. Conclusion

For maritime VIS–IR detection on Pohang, predicted box uncertainty is a within-detector ranking signal, not a sensor-selection signal. A pre-registered test found it informative, and the same evidence shows it is not useful for fusion. What works by day is simpler: veto a stream on image statistics and take the union of the survivors. That rule must be re-measured whenever the detector changes, including a retrain on corrected labels. Ours was not, and it now discards a working visible stream at night.

## References

Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., and Kalinin, A. A. (2020). Albumentations: fast and flexible image augmentations. *Information*, 11(2), 125. doi:10.3390/info11020125.

Choi, Jiwon, Cho, D., Lee, G., Kim, H., Yang, G., Kim, J., and Cho, Y. (2025). PoLaRIS dataset: a maritime object detection and tracking dataset in Pohang Canal. In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)*, pp. 13626–13632. doi:10.1109/ICRA55743.2025.11128583. Preprint arXiv:2412.06192.

Choi, Jiwoong, Chun, D., Kim, H., and Lee, H.-J. (2019). Gaussian YOLOv3: an accurate and fast object detector using localization uncertainty for autonomous driving. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 502–511. doi:10.1109/ICCV.2019.00059.

Chung, D., Kim, J., Lee, C., and Kim, J. (2023). Pohang Canal dataset: a multimodal maritime dataset for autonomous navigation in restricted waters. *The International Journal of Robotics Research*, 42(12), 1104–1114. doi:10.1177/02783649231191145.

Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241. doi:10.1214/aos/1176347265.

Liu, T., Pang, B., Zhang, L., Yang, W., and Sun, X. (2021). Sea surface object detection algorithm based on YOLO v4 fused with reverse depthwise separable convolution (RDSC) for USV. *Journal of Marine Science and Engineering*, 9(7), 753. doi:10.3390/jmse9070753.

Nirgudkar, S., DeFilippo, M., Sacarny, M., Benjamin, M., and Robinette, P. (2023). MassMIND: Massachusetts Maritime INfrared Dataset. *The International Journal of Robotics Research*, 42(1–2), 21–32. doi:10.1177/02783649231153020.

Prasad, D. K., Rajan, D., Rachmawati, L., Rajabally, E., and Quek, C. (2017). Video processing from electro-optical sensors for object detection and tracking in a maritime environment: a survey. *IEEE Transactions on Intelligent Transportation Systems*, 18(8), 1993–2016. doi:10.1109/TITS.2016.2634580.

Seitzer, M., Tavakoli, A., Antic, D., and Martius, G. (2022). On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks. In *International Conference on Learning Representations (ICLR)*. arXiv:2203.09168.

Solovyev, R., Wang, W., and Gabruseva, T. (2021). Weighted boxes fusion: ensembling boxes from different object detection models. *Image and Vision Computing*, 107, 104117. doi:10.1016/j.imavis.2021.104117.

Sun, Y., Cao, B., Zhu, P., and Hu, Q. (2022). Drone-based RGB-infrared cross-modality vehicle detection via uncertainty-aware learning. *IEEE Transactions on Circuits and Systems for Video Technology*, 32(10), 6700–6713. doi:10.1109/TCSVT.2022.3168279. Preprint arXiv:2003.02437.

Zhao, H., Zhang, H., and Zhao, Y. (2023). YOLOv7-sea: object detection of maritime UAV images based on improved YOLOv7. In *Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision Workshops (WACVW)*, pp. 233–238. doi:10.1109/WACVW58289.2023.00029.

Zhao, J., Wang, Y., Zhang, Y., Wang, H., and Guo, Y. (2024). Uncertainty-aware cross-modality fusion for visible-infrared object detection. In *Proceedings of the International Conference on Digital Image Computing: Techniques and Applications (DICTA)*, pp. 117–125. doi:10.1109/DICTA63115.2024.00029.
