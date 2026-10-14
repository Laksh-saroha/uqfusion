# Novelty check: external-context gating of gradient-free normalisation TTA, 2026-10-14

Question: is "an external day-night context signal (clock-derived solar elevation, or a second sensor) gating
gradient-free normalisation TTA for detection" novel? Motivated by the registered gate ablation
(`docs/eval/ttbn/gate_ablation_dev_2026-10-14.md`, development draws only): `q_vis` alone is worse than `union` on 12
development cells, all night; `q_vis ∧ solar` is worse on none and beats IG-BN on 13 cells.

Method: web search (three rounds, about 20 queries) followed by full-text reads of the PDFs (pdftotext) for every paper in
the first table. Every venue below was checked against a proceedings page, an arXiv page or the paper's own header, and
the exceptions are listed under "Unverified". Companion note: `docs/novelty-ttbn-2026-10-12.md` (multimodal TTA).

## Closest prior work

"Grad-free" means no gradient step at test time. "Det" means the paper evaluates object detection.

| Paper | Venue, year | Signal that decides adaptation or statistics | Grad-free | Det | Read |
|---|---|---|---|---|---|
| **TTA-DAME**, Jeon et al. (arXiv 2508.12690) | Technical report for the ICCV 2023 VCL challenge; arXiv v2 31 Mar 2026; no venue named | **Image day/night classifier** (EfficientNet-B7, 94.5% val accuracy, trained on source images with brightness/colour-temperature augmentation; "simply darkening the image suffices"). Night: route to a night expert and **do not adapt**. Otherwise mean-teacher self-training | No | Yes (SHIFT) | Full text |
| **DISC**, Mirza et al. (arXiv 2204.08817) | CVPR Workshops 2022 (DOI 10.1109/CVPRW56347.2022.00339) | **Weather task-ID, given at inference.** Sec. 4 says it can come from a classifier "or simply querying additional sensors available in modern cars, such as (typically infrared-based) rain sensors". Selects stored per-weather BN first/second-order statistics in a frozen detector | Yes at test time (statistics collected beforehand) | Yes (KITTI with simulated fog/rain/snow) | Method, setting, task-ID passage |
| **DILAM**, Leitner et al. (arXiv 2305.18953) | arXiv 30 May 2023; IEEE Intelligent Vehicles Symposium 2023 per the core.ac.uk record (not seen on IEEE) | Weather label **inferred by a majority-vote linear classifier on shallow image features**; selects per-weather BN affine parameters | Plug-in is; the stored parameters were learned by activation matching | Yes | Abstract, method |
| **Yoo et al.** (arXiv 2312.08875) | CVPR 2024 | **Feature-distribution gap** (Gaussian KL) between EMA of test features and training features, as a ratio to an in-domain gap (Eq. 5, Sec. 3.4); update only above a threshold. Adaptor modules by gradient | No | Yes (COCO-C, SHIFT incl. night) | Method, Sec. 3.4 |
| **DUA**, Mirza et al. (arXiv 2112.00463) | CVPR 2022 | None: continual BN-statistics update from incoming batches. No gate | Yes (authors' repository README, via search; the paper text was only skimmed) | Yes (KITTI-Fog/Rain) | Skimmed |
| **AR-TTA**, Sójka et al. (arXiv 2309.10109) and the **challenge report** (arXiv 2310.13533) | ICCV Workshops 2023 (CVF page); a later BMVC 2024 copy exists on bmva-archive.org.uk | Dynamic BN: mix source and batch statistics with a weight set by **symmetric KL between the stored source statistics and the batch**. The report notes that night shifts the distribution far more abruptly than weather | Mixing is; the method also self-trains | No (segmentation) | Full text of the report, key sections of AR-TTA |
| **Yi et al.** (arXiv 2310.05341) | ACM MM 2024 | None. Documents that BN-statistics adaptation **loses** to the source model on ACDC night (39.5 -> 37.3 mIoU, batch size 1, SegFormer-B5; TENT 39.5) | n/a | No | Table 1 |
| **TC-PDM**, Doan et al. (arXiv 2408.14227) | arXiv technical report, Aug 2024 | None. BN-Stats and DUA are baselines on night car detection from KAIST and M3FD, applied to the infrared **or** the visible stream, ungated. Gains of 0.3 to 2.6 AP50 over no adaptation | Yes | Yes | Sec. 5.2 |
| **Night-TTA**, Liu et al. (arXiv 2307.04470) | IEEE Trans. on Artificial Intelligence, 2023 | Every night image is adapted; ensemble-teacher losses | No | No | Per the 10-12 note |
| **Dai and Van Gool**, Dark Model Adaptation (arXiv 1810.02575) | ITSC 2018 (the reference list of 2305.18953 gives this venue) | **Solar elevation angle defines civil/nautical/astronomical twilight**; training-time day -> twilight -> night curriculum | No | No | Method section |
| **Sakaridis et al.**, Map-Guided Curriculum DA (arXiv 2005.14553) | IEEE TPAMI (arXiv comment "IEEE T-PAMI 2020"; no volume seen) | Time-of-day curriculum; **GPS-based** matching of dark frames to daytime map images; training-time | No | No | Abstract, figure text |
| **Domain-Aware Dynamic Networks**, Zhang et al. (arXiv 1911.13237) | arXiv Nov 2019 | BDD100K time-of-day x weather attributes **define training domains**; at inference the domain embedding is extracted from input images | n/a | Yes (BDD100K) | Method, Sec. 5 |
| **Lengyel et al.**, Zero-Shot Day-Night DA (arXiv 2108.05137) | ICCV 2021 | Argues the day-night shift is not fixable "in a data-driven manner" by BN variants and adds colour-invariant layers; still lists per-batch AdaBN as a strong zero-shot baseline | n/a | No | Intro, tables |
| **Solar Altitude Guided Scene Illumination**, Dogan et al. (arXiv 2507.05812) | arXiv, v2 Aug 2025 | **Solar altitude computed from lat-long and local time** as a conditioning variable for diffusion image generation | n/a | No | Abstract, contributions |

Foundations, cited as in the 10-12 note: AdaBN (Li et al., arXiv 1603.04779, "Revisiting Batch Normalization For Practical
Domain Adaptation", parameter-free statistics adaptation) and Schneider et al., NeurIPS 2020 (arXiv 2006.16971,
replace training BN statistics with statistics from the corrupted images).

Model-side "when to adapt" signals, all internal and none external: EmbodiTTA (Ma et al., arXiv 2505.00986v4, formatted
for IEEE Internet of Things Journal, no acceptance stated; EMA prediction entropy as the shift trigger; classification and
SHIFT segmentation), "When to Adapt" (Antonazzi et al., arXiv 2609.37602, Sept 2026; visual change, adapter mismatch and
semantic drift; training-free; open-vocabulary segmentation), SuMi and the others in the 10-12 note.

Adjacent, read only in part, not relied on: LearnableBN (Liao et al., AAAI-26, Vol. 40 No. 9, pp. 6925-6933; entropy
minimisation by gradient, BEV 3D detection, no day/night signal); Wu et al., domain-aware BN (AAAI 2024, arXiv
2312.10165; WILDS classification, gradient); Lens (Baek et al., ICLR 2025, arXiv 2503.02170; adapts camera ISO and
shutter at capture from model confidence, not normalisation); Patch-aware BN (Qi et al., arXiv 2304.02848;
training-time, Diverse-Weather night targets; venue not determined).

## What the search did not find

* No paper uses a **clock, solar elevation, GPS timestamp or ambient-light reading to gate or condition test-time
  adaptation**. The nearest uses are training-time (Dai and Van Gool, Sakaridis et al.) or generative (Dogan et al.).
* No paper documents a **selective** TTA gate that fails on dark-but-in-distribution input. Documented night failures are
  of **ungated** BN adaptation on out-of-distribution night (Yi et al.; the AR-TTA and challenge-report observation that
  night is the abrupt shift). In our setting night is in the training set, which none of them tests.
* No paper uses a **second sensor** as the judge that gates adaptation (confirms the 10-12 note).
* Correction to the 10-12 note: "no paper applies TTA to visible-infrared detection" needs a qualifier. TC-PDM
  applies BN-Stats and DUA to night detection on KAIST and M3FD, but one stream at a time, ungated, as baselines in a
  translation paper.

## Verdict

The broad claim is **not novel**. Three pieces are prior art:
1. **Day/night gating of whether to adapt, for detection**: TTA-DAME skips adaptation for frames its day/night
   classifier calls night.
2. **An external label selecting normalisation statistics for a detector**: DISC, including the idea of getting the
   label from a vehicle sensor. DILAM does the same with an inferred label.
3. **Skipping adaptation when there is no shift**: Yoo et al. (feature gap), EmbodiTTA (entropy), SuMi (per-sample
   zero weights).

The **narrow claim is defensible** with "to our knowledge": a **non-image context signal (solar elevation from GPS time)
or a second sensor gates a gradient-free BN-mix, evaluated against the visible novelty score and other gates in a
registered ablation**. What is new is where the signal comes from and the evidence that it matters, not gating itself.
DISC takes the label as an input that selects stored statistics, and TTA-DAME derives its day/night signal from the image.

Two clauses of the candidate claim go beyond the evidence:
* **"Model-internal signals cannot separate illumination from degradation."** We tested one internal signal, `q_vis`.
  Yoo et al.'s feature gap against training statistics, EMA entropy and TTA-DAME's image day/night classifier were not
  run. A gap computed against training features could plausibly stay quiet at in-distribution night, which would make it
  a reviewer's first baseline. TTA-DAME's own note that darkening a day image is enough to make its classifier say night
  supports the confound, but it is their remark about their training data, not a test of our gates.
* **"Selective normalisation TTA must separate ..."** is a design principle supported by one detector, one dataset and
  development draws. State it as a finding, not a law.

## Safest wording

> To our knowledge, prior selective test-time adaptation decides whether to adapt from the image stream or the model
> (feature-gap, entropy, a learned day/night classifier), and prior context or sensor labels have selected stored
> normalisation statistics (DISC) or defined training domains (twilight by solar elevation) rather than gating
> adaptation. We gate a gradient-free BN-mix on a non-image signal, solar elevation computed from the frame's GPS
> time, or on a thermal day reading. In our registered development ablation, the visible novelty score alone adapts at
> night, where frames are dark but in distribution, and is worse than no adaptation on 12 development cells, all night;
> the clock gate is worse on none.

Tighten the last sentence to the held-out numbers once pohang04 scores exist. Name TTA-DAME and DISC in related work.

## Never claim

* First to gate TTA on day/night (TTA-DAME), or first to skip adaptation at night.
* First to use a known or sensor-derived label for normalisation statistics in a detector (DISC, DILAM).
* First to use time of day or solar elevation as a domain definition (Dai and Van Gool, Sakaridis et al., SHIFT and BDD
  attributes). Solar altitude as a conditioning variable exists in image generation (Dogan et al.).
* First BN-statistics TTA at night, or on infrared or visible night detection (Yi et al., TC-PDM, Lengyel et al.).
* That model-internal signals **cannot** do this. Say that the signals we tested did not, and name the ones we did not test.
* That the clock gate beats IG-BN in general. It beat IG-BN on 13 development cells, 0 worse, on development draws only.
* That no prior work applies TTA to visible-infrared detection without the TC-PDM qualifier.
* That the clock gate works at any latitude or in any scene. Polar summer and winter, tunnels and artificial lighting break
  the clock, and the second sensor is the fallback for those cases.
* "Training-free" for the whole system if any gate component was trained.

## Limits

* One search tool (web search, which returned CVF, AAAI, arXiv and mlanthology pages). IEEE Xplore, ACM DL and Google
  Scholar were not queried directly, so journal-only work could be missing.
* Unverified: DILAM's IV 2023 venue (core.ac.uk comment only); the BMVC 2024 copy of AR-TTA (first page only, not
  compared with the ICCVW paper); TTA-DAME, EmbodiTTA, TC-PDM and Dogan et al. have no venue in what I saw; DUA's
  gradient-free status rests on its README; Sakaridis et al. has no volume or page numbers.
* A continual-TTA survey (arXiv 2607.08164) reported night as the hardest ACDC condition, but only search snippets were
  seen, so it is not cited above.
* arXiv 2609.37602 (Sept 2026) and 2505.00986v4 (Oct 2026) are recent; both were read from their PDFs, not abstracts only.
* The 2026 literature moves fast. Repeat the search before submission.
