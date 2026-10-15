# Gate-free BatchNorm adaptation: development screen (unregistered), 2026-10-15

**Question.** Can IG-BN's adaptation (BN-mix, N = 16, non-overlapping 8-frame blocks) run on *every* frame, with no
gate, without hurting clean or night frames? Development data only; no pohang04 frame was read and no paper text changed.
Code: `scripts/ttbn_gatefree.py`. Draw 941 / 951 only, the draw the original TT-BN screen already used.

**Answer.** Mostly yes. **Shallow + hinge** keeps 82% of IG-BN's low-light-day gain without a gate. It applies BN-mix only
on the first 31 of 130 BatchNorm layers, and in each layer only when the block's divergence from the source statistics
exceeds the median of clean training blocks (weight scaled by (1 − D0/D)₊). Over all 60 cells it is **26 better / 0 worse
than `union`**, against ungated BN-mix's 23 / 18 and IG-BN's 5 / 0. It beats IG-BN in 13 degraded day cells and loses to
it in 2. The open risk is night. No night cell is "worse" by the rule, but 12 of 30 night cells sit at −0.0014 to −0.0037,
with intervals that cross zero. Averaging three draws (942–944) may push some of them into "worse".

## Method (all candidates)

* **Source statistics, label-free, training split only.** Detector frozen, forward passes only. Population BN
  statistics per layer from 1,600 random training frames per domain: day = pohang00/02/03, night = pohang01, both
  cameras, letterboxed like test frames. Normalisation uses the statistics of a 16-frame random batch.
* **D calibration.** 120 day and 120 night blocks of 8 consecutive raw training-run frames, none of them in the
  val, test or development lists. Per layer, D = mean over channels of the symmetric KL between the block's channel
  Gaussians and the source's.
  * Thresholds per layer and model: D0 = median or 90th percentile over all blocks against the running statistics;
    per-domain 90th percentiles against the domain statistics.
  * The nearest-domain rule picks the right domain on 100% of day blocks and 95.8% of night blocks.
* **Arms.** Every block is adapted (no gate). IG-BN on this draw is ungated BN-mix's records where IG-BN's gate fires;
  a block's statistics depend only on its own frames, as the gate ablation showed. `mix16` reproduces the original
  screen's draw-941 records exactly (16/16 frames checked, max diff 0).
* **Statistic.** Fused ship AP of E4 `union` (VIS arm stream + plain IR). Δ vs `union` per system, mean over five
  systems, 95% t-interval (df 4); one draw, so no draw averaging. The registered thresholds are applied descriptively:
  worse = Δ ≤ −0.0014 with interval below zero; better = Δ ≥ +0.0060 with interval above zero.

## Stage 1: all candidates, subset (VIS clean + low light, every 2nd block, IR clean)

| arm | clean day | clean night | low-light night | low-light day |
|---|---:|---:|---:|---:|
| IG-BN (reference) | 0 | 0 | 0 | +0.0696 |
| ungated BN-mix N=16 (reference) | −0.0095 | −0.0186 | −0.0198 | +0.0696 |
| C1 BN-mix toward nearest domain source | −0.0076 | −0.0258 | −0.0305 | +0.0721 |
| C1 oracle domain | −0.0076 | −0.0258 | −0.0307 | +0.0576 |
| C1′ domain statistics alone (N = ∞), nearest | +0.0002 | −0.0075 | −0.0098 | +0.0542 |
| C1′ oracle | +0.0002 | −0.0075 | −0.0103 | −0.0713 |
| C2 N_eff = 16·exp(D0/D), D0 = median | −0.0018 | −0.0110 | −0.0114 | +0.0646 |
| C2 hinge, D0 = p90 | −0.0000 | −0.0004 | −0.0007 | +0.0253 |
| C1+C2 nearest domain + hinge | +0.0002 | −0.0080 | −0.0102 | +0.0648 |
| C3 first 2 BN | −0.0067 | −0.0093 | −0.0173 | +0.0068 |
| C3 first 11 BN | −0.0079 | −0.0014 | −0.0057 | +0.0499 |
| C3 first 21 BN | −0.0091 | −0.0011 | −0.0021 | +0.0725 |
| C3 first 31 BN | −0.0101 | −0.0047 | −0.0048 | +0.0787 |

Readings:
* **Domain-matched sources (C1) do not help.** Every low-light *day* block picks the night source, so the gain
  survives. But BN-mix toward night statistics hurts real night frames more than ungated BN-mix does. The day source
  on dark day frames (C1′ oracle) is a disaster. Dark day and real night cannot be told apart from activations alone,
  which is the confound IG-BN's thermal term exists for.
* **The two harms have different causes.** The clean-day loss comes from the 8-frame block's own statistics: it goes
  only when the block's weight goes to ~0 (the hinge). The night loss comes from deep layers: shallow-only adaptation
  removes most of it and keeps the gain.

## Stage 2: shallow + hinge, subset (every 4th block, otherwise as Stage 1)

| arm | clean day | clean night | low-light night | low-light day |
|---|---:|---:|---:|---:|
| ungated BN-mix (reference) | −0.0110 | −0.0179 | −0.0200 | +0.0696 |
| hinge D0 = median, all layers | −0.0017 | −0.0039 | −0.0041 | +0.0574 |
| first 21 BN, hinge p90 | −0.0000 | +0.0005 | −0.0009 | +0.0219 |
| first 21 BN, hinge median | −0.0000 | −0.0011 | −0.0021 | +0.0510 |
| first 31 BN, hinge p90 | −0.0000 | +0.0003 | −0.0005 | +0.0239 |
| first 31 BN, hinge median | −0.0001 | −0.0017 | −0.0019 | +0.0561 |
| nearest domain + hinge, first 21 BN | −0.0026 | −0.0074 | −0.0083 | +0.0645 |
| nearest domain + hinge, first 31 BN | −0.0019 | −0.0064 | −0.0064 | +0.0699 |

The threshold sets the trade-off: p90 is safe but keeps a third of the gain; the median keeps ~80%. The two
median-hinge shallow arms went to the full grid.

## Stage 3: full 60 cells, all frames (draw 941, five systems, six IR conditions)

| arm | better | worse | worst cell | better / worse than IG-BN | mean Δ vs IG-BN, 24 degraded day cells |
|---|---:|---:|---|---:|---:|
| IG-BN (reference) | 5 | 0 | — | — | 0 |
| ungated BN-mix (reference) | 23 | 18 | lowlight/noise_s2/night −0.0219 [−0.0300, −0.0137] | 20 / 18 | +0.0287 |
| shallow 21 + hinge median | 20 | **0** | lowlight/fog_s3/night −0.0037 [−0.0084, +0.0010] | 13 / 2 | +0.0140 |
| **shallow 31 + hinge median** | **26** | **0** | lowlight/fog_s3/night −0.0037 [−0.0086, +0.0013] | **13 / 2** | **+0.0159** |

Shallow 31 + hinge median, selected cells (Δ [95%]):
* **Where IG-BN's gate never or barely fires: big wins over IG-BN.**
  * fog_s1/fog_s1–s3 + noise: +0.0167 to +0.0209 vs `union` (IG-BN ≤ +0.0007).
  * low light under IR fog_s2/s3/noise: +0.0556 to +0.0586 (IG-BN ≤ +0.0074).
  * fog/noise_s3: +0.0083 (IG-BN 0).
  * IR corruption switches IG-BN off; a gate-free method does not depend on the IR camera.
* **Where IG-BN fires on every frame: losses to IG-BN.**
  * lowlight/clean/day: +0.0579 vs IG-BN's +0.0708, so −0.0129 [−0.0196, −0.0062].
  * fog/clean/day: −0.0044 [−0.0068, −0.0020].
* **Glare.** About +0.005 by day (not "better": under the +0.0060 threshold), and +0.009 at night (glare/clean/night
  +0.0090 [+0.0030, +0.0149]).
* **Clean VIS.** Every clean-VIS day cell within ±0.0002 of `union`.
* **Night.**
  * Clean night −0.0021 [−0.0071, +0.0030]; low-light night −0.0022 to −0.0037.
  * All intervals cross zero; 12 of 30 night cells have a point estimate ≤ −0.0014.
  * Mean over night cells +0.0025, lifted by glare and fog nights.
* **Shallow 21 vs 31.** Shallow 21 is no safer at night (same worst cell); same 0 worse, but 20 better and a smaller gain.

Full per-cell table (generated by `report --set 3`):

#### Per cell, Δ vs `union` (mean over five systems)

| cell | union | igbn | mix16 | sh21_hinge50 | sh31_hinge50 |
|---|---:|---:|---:|---:|---:|
| clean/clean/day | 0.3566 | +0.0000 | -0.0101 | +0.0000 | -0.0001 |
| clean/clean/night | 0.2895 | +0.0000 | -0.0195 | -0.0017 | -0.0021 |
| clean/fog_s1/day | 0.3553 | +0.0000 | -0.0114 | +0.0000 | -0.0001 |
| clean/fog_s1/night | 0.2849 | +0.0000 | -0.0195 | -0.0018 | -0.0022 |
| clean/fog_s2/day | 0.3460 | +0.0000 | -0.0100 | +0.0000 | -0.0001 |
| clean/fog_s2/night | 0.2691 | +0.0000 | -0.0186 | -0.0014 | -0.0024 |
| clean/fog_s3/day | 0.3389 | +0.0000 | -0.0096 | +0.0000 | -0.0001 |
| clean/fog_s3/night | 0.2556 | +0.0000 | -0.0190 | -0.0022 | -0.0032 |
| clean/noise_s2/day | 0.3654 | +0.0000 | -0.0097 | +0.0000 | -0.0001 |
| clean/noise_s2/night | 0.2552 | +0.0000 | -0.0192 | -0.0019 | -0.0028 |
| clean/noise_s3/day | 0.3500 | +0.0000 | -0.0100 | +0.0001 | -0.0002 |
| clean/noise_s3/night | 0.2552 | +0.0000 | -0.0192 | -0.0019 | -0.0028 |
| fog/clean/day | 0.0240 | +0.0100 | +0.0101 | +0.0052 | +0.0056 |
| fog/clean/night | 0.0837 | +0.0000 | -0.0022 | +0.0038 | +0.0051 |
| fog/fog_s1/day | 0.0298 | +0.0041 | +0.0093 | +0.0051 | +0.0054 |
| fog/fog_s1/night | 0.0760 | +0.0000 | -0.0009 | +0.0050 | +0.0063 |
| fog/fog_s2/day | 0.0202 | +0.0033 | +0.0108 | +0.0060 | +0.0066 |
| fog/fog_s2/night | 0.0610 | +0.0000 | -0.0002 | +0.0057 | +0.0073 |
| fog/fog_s3/day | 0.0126 | +0.0008 | +0.0136 | +0.0079 | +0.0085 |
| fog/fog_s3/night | 0.0448 | +0.0000 | +0.0000 | +0.0056 | +0.0068 |
| fog/noise_s2/day | 0.0279 | +0.0000 | +0.0058 | +0.0029 | +0.0030 |
| fog/noise_s2/night | 0.0451 | +0.0000 | -0.0004 | +0.0050 | +0.0063 |
| fog/noise_s3/day | 0.0113 | +0.0000 | +0.0136 | +0.0080 | +0.0083 |
| fog/noise_s3/night | 0.0451 | +0.0000 | -0.0004 | +0.0049 | +0.0063 |
| fog_s1/clean/day | 0.0360 | +0.0093 | +0.0370 | +0.0174 | +0.0190 |
| fog_s1/clean/night | 0.1413 | +0.0000 | -0.0079 | -0.0004 | +0.0010 |
| fog_s1/fog_s1/day | 0.0428 | +0.0006 | +0.0345 | +0.0162 | +0.0176 |
| fog_s1/fog_s1/night | 0.1371 | +0.0000 | -0.0076 | +0.0001 | +0.0014 |
| fog_s1/fog_s2/day | 0.0344 | +0.0007 | +0.0366 | +0.0175 | +0.0189 |
| fog_s1/fog_s2/night | 0.1231 | +0.0000 | -0.0071 | +0.0005 | +0.0020 |
| fog_s1/fog_s3/day | 0.0286 | +0.0003 | +0.0386 | +0.0186 | +0.0202 |
| fog_s1/fog_s3/night | 0.1064 | +0.0000 | -0.0059 | +0.0001 | +0.0015 |
| fog_s1/noise_s2/day | 0.0393 | +0.0000 | +0.0332 | +0.0155 | +0.0167 |
| fog_s1/noise_s2/night | 0.1065 | +0.0000 | -0.0066 | -0.0002 | +0.0015 |
| fog_s1/noise_s3/day | 0.0276 | +0.0000 | +0.0390 | +0.0189 | +0.0209 |
| fog_s1/noise_s3/night | 0.1065 | +0.0000 | -0.0067 | -0.0002 | +0.0015 |
| lowlight/clean/day | 0.1783 | +0.0708 | +0.0708 | +0.0521 | +0.0579 |
| lowlight/clean/night | 0.2705 | +0.0000 | -0.0198 | -0.0026 | -0.0022 |
| lowlight/fog_s1/day | 0.1811 | +0.0362 | +0.0689 | +0.0513 | +0.0569 |
| lowlight/fog_s1/night | 0.2660 | +0.0000 | -0.0201 | -0.0027 | -0.0023 |
| lowlight/fog_s2/day | 0.1738 | +0.0074 | +0.0673 | +0.0503 | +0.0556 |
| lowlight/fog_s2/night | 0.2512 | +0.0000 | -0.0199 | -0.0028 | -0.0027 |
| lowlight/fog_s3/day | 0.1684 | +0.0008 | +0.0681 | +0.0514 | +0.0565 |
| lowlight/fog_s3/night | 0.2397 | +0.0000 | -0.0215 | -0.0037 | -0.0037 |
| lowlight/noise_s2/day | 0.1791 | +0.0000 | +0.0705 | +0.0525 | +0.0580 |
| lowlight/noise_s2/night | 0.2397 | +0.0000 | -0.0219 | -0.0037 | -0.0036 |
| lowlight/noise_s3/day | 0.1742 | +0.0000 | +0.0705 | +0.0534 | +0.0586 |
| lowlight/noise_s3/night | 0.2397 | +0.0000 | -0.0218 | -0.0037 | -0.0036 |
| glare/clean/day | 0.1703 | +0.0000 | +0.0231 | +0.0054 | +0.0054 |
| glare/clean/night | 0.1274 | +0.0000 | +0.0024 | +0.0083 | +0.0090 |
| glare/fog_s1/day | 0.1746 | +0.0000 | +0.0217 | +0.0053 | +0.0053 |
| glare/fog_s1/night | 0.1201 | +0.0000 | +0.0033 | +0.0093 | +0.0099 |
| glare/fog_s2/day | 0.1658 | +0.0000 | +0.0218 | +0.0049 | +0.0049 |
| glare/fog_s2/night | 0.1034 | +0.0000 | +0.0056 | +0.0098 | +0.0105 |
| glare/fog_s3/day | 0.1591 | +0.0000 | +0.0215 | +0.0044 | +0.0044 |
| glare/fog_s3/night | 0.0829 | +0.0000 | +0.0097 | +0.0104 | +0.0112 |
| glare/noise_s2/day | 0.1725 | +0.0000 | +0.0235 | +0.0057 | +0.0058 |
| glare/noise_s2/night | 0.0827 | +0.0000 | +0.0100 | +0.0106 | +0.0111 |
| glare/noise_s3/day | 0.1617 | +0.0000 | +0.0231 | +0.0052 | +0.0053 |
| glare/noise_s3/night | 0.0827 | +0.0000 | +0.0100 | +0.0105 | +0.0111 |

## Recommendation

**Pre-register shallow 31 + hinge median as an ungated alternative to IG-BN on draws 942–944, with the night risk
named in advance.**
* It is the first variant with 0 worse cells on all 60 cells of a development draw.
* It is better than IG-BN in 13 cells, exactly the ones where IR corruption switches IG-BN off. It loses to IG-BN only
  on the two cells where IG-BN's gate fires on every frame.
* It makes adaptation independent of the thermal camera and of any clock. That answers the reviewer's point directly:
  it needs no gate at all.

Before registering:
1. **Name the night risk.** The registered rule averages three draws, which narrows intervals. Night cells at −0.002
   to −0.004 could then read "worse" and fail D-H. Predict that, or register a non-inferiority margin for night
   (e.g. −0.005) and say so before scoring.
2. **Fix every constant from training data or this screen.**
   * Constants: k = 31 (through stride 16), D0 = per-layer median of the 240 training blocks, N = 16, n = 8.
   * Already in `runs/novel/ttbn_gatefree/calib.pt`, from training frames only.
   * k and the median were chosen on draw 941; disclose that.
3. **Optional hybrid, as a separate arm.** IG-BN where its gate fires, shallow-hinge elsewhere. It would keep IG-BN's
   lowlight/clean and fog/clean gains and add the 13 wins. Not screened here, and it reintroduces a gate.

Not recommended: domain-matched sources (C1/C1′), the exp-scaled prior, and the p90 hinge (too little gain).

**Caveats.** One draw, so draw-to-draw noise is not in the intervals. The arms were chosen on this draw
(Stages 1–2 on subsets of it). Night is the 46%-pohang01 slice and swings between episodes. Compute is 1 extra pass
per block (≈ 1.8 s per block for five detectors, fp32), with the same 0.7 s look-ahead as IG-BN.
