# Uncertainty Is Informative but Not a Fusion Signal: A Pre-Registered Visible–Infrared Study on Misregistered Maritime Video

**Anonymous authors** (CVF workshops normally review double-blind; restore the author block for the camera-ready).

> Venue draft for the Perception Beyond the Visible Spectrum workshop (PBVS), derived from `PAPER_DRAFT2.md` at commit `d9260ef` (2026-10-09). The target is 8 pages in CVF format, excluding references. Every number is taken from Draft 2, which cites the source file for each; none is new. Venue facts and open items are in `venues/README.md`.

---

## Abstract

Visible–infrared (VIS–IR) detectors are often proposed with uncertainty deciding how much to trust each band. We tested that idea under a pre-registered decision rule on paired maritime video from the Pohang Canal dataset. The two cameras are not co-registered and are paired by timestamp, with a measured residual of 3–6 px. Each band has its own YOLO26 detector with a single-pass Gaussian variance head.

Three findings follow.

1. **Misregistration turns decision-level fusion into concatenation.** At the adopted merge threshold, only 0.05 percent of visible boxes have an infrared partner. Relaxing correspondence and letting predicted variance arbitrate the extra merges is non-inferior on one of four conditions.
2. **Uncertainty is informative but not a fusion signal.** As a fusion weight, real variance does not beat the same values shuffled onto the wrong boxes on any of four conditions. As a score re-ranker it does, on three of four, yet the signal lies within each detector's own boxes. Using it is worse than using no variance on six of eight cells.
3. **The cross-modal signal that does work is coarse.** Infrared's vote on whether it is night selects the visible stream better than any visible statistic can. Hardened against six infrared corruptions, its false-night rate falls from 94.8 percent to 0 percent in-sample. But the veto built on it encodes a claim about one visible detector, and it broke when that detector was retrained.

---

## 1. Introduction

The case for pairing a visible and a thermal camera is that they fail differently: VIS in fog, glare and darkness, IR at thermal crossover. Turning that complementarity into a detector requires deciding, per frame or per box, which band to believe. A natural candidate is each detector's own predicted uncertainty. Gaussian heads give a per-box variance in a single pass, and uncertainty-aware VIS–IR detectors exist. Most such systems, however, assume pixel-aligned pairs. Many real platforms, including the maritime one studied here, have unsynchronized optics and only timestamp pairing.

We built the uncertainty-gated design and measured every component before deciding what to ship. This paper reports the parts that bear on multispectral fusion:

* what registration quality does to decision-level fusion (§5.1);
* whether predicted variance can arbitrate between bands (§5.2);
* which cross-modal cue actually selects a band, and how robust it is to corruption of the band it trusts (§5.3).

Each verdict was pre-registered with a decision rule and a magnitude floor.

## 2. Related work and positioning

Gaussian YOLOv3 (Choi et al., 2019) attaches a per-coordinate Gaussian to the box regressor and multiplies each box's score by one minus its mean predicted uncertainty. We use the same head, with beta-NLL training (Seitzer et al., 2022), but no σ moves a shipped score.

UA-CMDet (Sun et al., 2022) detects vehicles in paired drone RGB–IR imagery. Its uncertainty is assigned to annotations, not predicted: a rule built from cross-modal ground-truth overlap and illumination weights each object's training loss, and the module is removed after training. At inference an illumination-aware NMS scales the RGB branch's scores by an image-level illumination weight. Zhao et al. (2024) fuse VIS and IR backbone features with attention and weight branch losses by a label uncertainty estimated in training. Neither evaluates calibration.

Table 1 places this work beside them. The closest prior mechanism to our shipped layer is UA-CMDet's illumination-weighted NMS. Both demote the visible band on an image statistic rather than on a predicted uncertainty. Ours removes the band outright, and only when IR votes night.

**Table 1. Positioning against the nearest VIS–IR systems** (from direct reads of each paper's full text).

| Work | Uncertainty target | Inference-time adaptation | Calibration evaluated | Registration assumption |
|---|---|---|---|---|
| UA-CMDet (Sun et al., 2022) | per-object training-loss weights from a rule; not predicted | RGB scores × illumination weight, then NMS over branches | no | per-pair affine alignment; residual down-weighted in training |
| Zhao et al. (2024) | per-label training-loss weights; not predicted at inference | input-dependent attention over fused features | no | datasets distributed as aligned pairs |
| This work | box coordinates (σ²); frame OOD (Mahalanobis) | hard veto on image statistics; fusion weight constant 0.9926 (measured) | yes: D-ECE, interval-ECE, NLL, AUSE/AURC | nearest-timestamp pairing, 3–6 px median residual |

## 3. Data, system and protocol

**Data.** Pohang Canal (Chung et al., 2023) records stereo visible video at 2048×1080 and a thermal camera at 640×512, both at 10 Hz, along a 7.5 km canal-to-coast route. PoLaRIS (Choi et al., 2025) supplies ship and buoy boxes on both bands.

**Pairing.** Frames are paired by the dataset's timestamp table, not by frame index. 16,544 of the 28,388 pairs have different VIS and IR indices, and the measured residual misalignment is 3–6 px median per run, with within-run swings of about 10 px.

**IR export.** IR is exported as 8-bit by per-frame min–max normalization. This can mask thermal crossover, because a near-isothermal vessel is stretched to full local contrast.

**Benchmark.** The development set is 2,232 paired validation frames (1,200 day, 1,032 night from one run). It crosses four simulated VIS conditions (clean, fog, low light, glare; Buslaev et al., 2020) with day and night.

**Detectors.** Both bands use yolo26m. VIS is two-class. IR is single-class with a P2 neck, because the thermal detector cannot see buoys (buoy AP 0.0002), so ship AP is the registered metric throughout. A log-variance branch predicts σ for the four box edges.

**Decision layer (Figure 1).**

* IR votes whether it is night with a 5th-percentile luminance test. It may vote only while an 11-statistic health score says the IR frame still looks like IR.
* VIS is vetoed when `night AND (dark OR veil)`.
* Survivors are merged by Weighted Boxes Fusion (Solovyev et al., 2021) at IoU 0.85 with constant capability weights.
* A box with a cross-modal partner at IoU 0.30 gets a score boost.
* σ and a Mahalanobis out-of-distribution score are computed but inert.

![decision_layer](../docs/figures/fig_decision_layer.png)

**Figure 1. The shipped decision layer.** A hard veto driven by image statistics from both bands, then a constant-weight merge that is concatenation in practice. σ and the Mahalanobis score are computed but inert.

**Protocol.** Deltas are paired on the same frames and corruption draw, which tightens them 3–16×. On ship AP the paired two-sigma floor is 0.0008–0.0024. Intervals are moving-block bootstraps (Künsch, 1989), because frame-level ones at 10 Hz are 1.9–1.99× too narrow, and every pre-registered verdict uses a 0.0060 magnitude floor. Fusion levers were tuned on one day run (TUNE) and reported on two others (TEST).

## 4. Why decision-level fusion here is concatenation

At the merge threshold the system uses, 0.85, only 0.05 percent of VIS boxes find an IR partner. The fusion weight is the same value, 0.9926, on every one of 8,928 frame-condition pairs. The system does not measure agreement between the bands; it concatenates their detections. Two obvious repairs were measured with σ inert:

* a per-frame registration refinement raised the partner rate 80× to 4.02 percent but lowered held-out AP by 0.0025;
* relaxing the threshold to 0.55 cost 0.0138.

## 5. Results

### 5.1 Relaxing correspondence does not let uncertainty arbitrate

If misregistration starves the merge, perhaps σ could arbitrate a looser one. We crossed the merge threshold {0.85, 0.55} with σ-weighting {off, live}, and registered that the relaxed, σ-live cell D must be non-inferior to the shipped cell A within 0.0060 on at least three of four conditions on TUNE.

**Table 2. A − D on TUNE (one day run, 836 frames), ship AP, block bootstrap.**

| Condition | A − D | 95% CI | Non-inferior |
|---|---:|---|---|
| clean | +0.0151 | [+0.0095, +0.0210] | no |
| fog | +0.0160 | [+0.0104, +0.0221] | no |
| lowlight | +0.0014 | [−0.0001, +0.0040] | yes |
| glare | +0.0115 | [+0.0070, +0.0157] | no |

The relaxed, σ-live cell is non-inferior on one condition of four, at every floor from 0.0014 to 0.0100. The lone pass reflects how little relaxing costs on low light, not anything σ recovered.

σ was confirmed live: it changed the fused output on 753–836 of 836 clean frames. Yet its interaction terms sit within [−0.0002, +0.0004] everywhere. Correspondence is closed, and the fusion is documented as union aggregation.

### 5.2 Predicted uncertainty is informative but not a fusion signal

**The test.** The registered comparison is real σ against the same σ shuffled onto the wrong boxes. A comparison against a constant would not work, because inverse-variance fusion with equal σ is plain WBF. σ enters on one of two paths:

* the coordinate path, where it weights box coordinates in the merge;
* the score path, where it multiplies scores by (frame-median σ / σ)^α, with α = 1 fixed before the run.

A path passes on three of four conditions with a delta of at least 0.0060 and an interval excluding zero.

**Table 3. Real minus shuffled σ, and the score path against no σ (S5 − S0). Shipped preset, gated-fusion ship AP, 2,232 paired frames, block-bootstrap 95% CI.**

| Condition | Coordinate path | Score path | Score path vs no σ |
|---|---|---|---|
| clean | −0.0001 [−0.0003, +0.0001] | **+0.0073** [+0.0033, +0.0124] | −0.0329 [−0.0411, −0.0234] |
| fog | −0.0000 [−0.0001, +0.0002] | **+0.0120** [+0.0094, +0.0142] | −0.0029 [−0.0056, −0.0010] |
| lowlight | −0.0000 [−0.0000, +0.0000] | +0.0034 [+0.0012, +0.0048] | −0.0083 [−0.0116, −0.0050] |
| glare | −0.0000 [−0.0001, +0.0001] | **+0.0068** [+0.0031, +0.0114] | −0.0266 [−0.0339, −0.0190] |
| passing at 0.0060 | 0 of 4 | **3 of 4** | — |

**Coordinate path: NULL.** Real σ also differs from no σ by at most 0.0001 on this path. That is the expected result when almost nothing merges, but Stage 1 (§5.1) shows that letting more boxes merge does not change it.

**Score path: POSITIVE, but not fusion.** Removing the IR detections locates the gain inside each band. By day, re-ranking VIS boxes alone gives +0.0153 to +0.0167, and adding IR moves the delta by −0.0040 to +0.0035, with no consistent sign. At night VIS is vetoed, so the night part is IR re-ranking IR boxes.

**Informative, not useful.** σ orders a detector's own boxes better than chance, and it says nothing about which band to believe. Spent at the registered strength it costs AP: the system without σ is better on six of eight cells across both presets.

A re-ranker learned from confidence and σ fared no better under replication. Its σ increment was registered for replication on five retrained detectors. It was positive on all five (mean +0.0053) but cleared the floor on too few (2/5 at 0.0060, 3/5 at 0.0047), and the replication fails.

**The metric matters.** Both runs as first recorded scored a ship-and-buoy macro, and the macro read NULL on both paths. IR cannot see buoys, and σ re-ranks buoys worse than chance on clean and glare, so the macro averaged a halved ship gain with a buoy loss. The registered metric reverses that verdict, and we report the deviation.

### 5.3 The cross-modal cue that works, and how it fails

**IR selects VIS better than any VIS statistic.** VIS brightness cannot distinguish a dark world from a dark sensor. Synthetic low-light day frames are darker than real night (5th-percentile luminance 0 against 2.5–3.5), yet VIS still detects on them. Asking IR whether it is night is the axis that separates the two.

**Signal screens agree on what carries information.** Among VIS boxes on clean day frames, the true-positive rate rises:

* 4.80× with high confidence;
* 3.00× with σ below the frame median;
* 2.08× with a cross-modal partner at IoU 0.30;
* 1.00× with temporal persistence (Figure 2).

Redundancy is worth what it is independent of. A persistent false positive is the most stable object in a fixed scene.

![lift_screen](../docs/figures/fig_lift_screen.png)

**Figure 2. Which per-box signals predict a true positive.** Lift = P(TP | signal fires) / P(TP | it does not), VIS boxes on clean day paired frames. A signal at lift 1.0 cannot change an AP ranking whatever weight it gets.

**Trusting IR's vote has a failure mode.** IR was uncorrupted in every benchmark cell, so we corrupted it with six hazards at three severities. A false night on a clear day vetoes a working VIS stream. The raw vote misread 94.8 percent of clear days as night under severe IR fog, and 19–27 percent under IR glare. Two votes, an IR self-check, the multivariate health score and an authority bound took the false-night rate on 19 corruption arms to 0 percent at zero benchmark cost. That figure is in-sample, because the threshold and health model were fitted on data that includes the evaluated night run. The worst case, both bands degraded, fell from a 24 percent false-veto rate to 1.3 percent.

**The veto encodes a claim about one VIS detector.** Its night arm was written while VIS scored 0.0000 at night. That zero was a label-filtering artifact: after a pre-registered label restore, night VIS reached 0.2520. We retrained both bands, five seeds each, and kept the rule frozen (Figure 3).

* By day, union aggregation beats VIS alone on all four cells (+0.0059 to +0.0107).
* At night the rule is right where VIS fails: fog and low light, +0.068.
* It is wrong where VIS still works: clean night −0.1847, glare night −0.0872.

The rule gates on darkness, but what should decide is VIS health.

![phase3_cells](../docs/figures/fig_phase3_cells.png)

**Figure 3. The frozen rule on five retrained VIS+IR systems**, seed-mean ship AP, clean IR. At night the fused output equals IR alone.

A single held-out run, scored once, gives fused ship AP 0.2682 [0.2576, 0.2793]. Its gap to development, +0.0216 [−0.0120, +0.0502], is unresolved.

## 6. Limitations

* **Timestamp pairing only.** No time-varying homography was built.
* **Simulated corruptions; one night run.** No between-run night interval is estimable.
* **In-sample IR safety.** The IR-safety figures are in-sample.
* **Untuned σ strength.** α was fixed and never tuned.
* **Development data, pre-restore checkpoints.** The R-D1 and Stage 1 verdicts are development-data results.
* **Spent held-out run.** The held-out run is single and is now spent.

## 7. Conclusion

On misregistered VIS–IR maritime video, decision-level fusion reduces to concatenation. Predicted box variance cannot arbitrate between the bands: it is informative within each detector and inert or harmful between them. The cross-modal cue that does select a band is coarse, namely IR's vote on whether it is night, and it must be hardened against IR's own failures. Any rule built on it is a claim about the VIS detector it was written against.

## References

Buslaev, A., Iglovikov, V. I., Khvedchenya, E., Parinov, A., Druzhinin, M., and Kalinin, A. A. (2020). Albumentations: fast and flexible image augmentations. *Information*, 11(2), 125. doi:10.3390/info11020125.

Choi, Jiwon, Cho, D., Lee, G., Kim, H., Yang, G., Kim, J., and Cho, Y. (2025). PoLaRIS dataset: a maritime object detection and tracking dataset in Pohang Canal. In *Proceedings of the IEEE International Conference on Robotics and Automation (ICRA)*, pp. 13626–13632. doi:10.1109/ICRA55743.2025.11128583. Preprint arXiv:2412.06192.

Choi, Jiwoong, Chun, D., Kim, H., and Lee, H.-J. (2019). Gaussian YOLOv3: an accurate and fast object detector using localization uncertainty for autonomous driving. In *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*, pp. 502–511. doi:10.1109/ICCV.2019.00059.

Chung, D., Kim, J., Lee, C., and Kim, J. (2023). Pohang Canal dataset: a multimodal maritime dataset for autonomous navigation in restricted waters. *The International Journal of Robotics Research*, 42(12), 1104–1114. doi:10.1177/02783649231191145.

Künsch, H. R. (1989). The jackknife and the bootstrap for general stationary observations. *The Annals of Statistics*, 17(3), 1217–1241. doi:10.1214/aos/1176347265.

Seitzer, M., Tavakoli, A., Antic, D., and Martius, G. (2022). On the pitfalls of heteroscedastic uncertainty estimation with probabilistic neural networks. In *International Conference on Learning Representations (ICLR)*. arXiv:2203.09168.

Solovyev, R., Wang, W., and Gabruseva, T. (2021). Weighted boxes fusion: ensembling boxes from different object detection models. *Image and Vision Computing*, 107, 104117. doi:10.1016/j.imavis.2021.104117.

Sun, Y., Cao, B., Zhu, P., and Hu, Q. (2022). Drone-based RGB-infrared cross-modality vehicle detection via uncertainty-aware learning. *IEEE Transactions on Circuits and Systems for Video Technology*, 32(10), 6700–6713. doi:10.1109/TCSVT.2022.3168279. Preprint arXiv:2003.02437.

Zhao, J., Wang, Y., Zhang, Y., Wang, H., and Guo, Y. (2024). Uncertainty-aware cross-modality fusion for visible-infrared object detection. In *Proceedings of the International Conference on Digital Image Computing: Techniques and Applications (DICTA)*, pp. 117–125. doi:10.1109/DICTA63115.2024.00029.
