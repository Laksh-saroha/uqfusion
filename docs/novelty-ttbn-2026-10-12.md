# Novelty check: IR-gated test-time BN adaptation (N4 family), 2026-10-12

Question (Laksh, 10-12): is the N4 result novel? Five closest papers read in full text (PDFs or HTML),
plus searches for visible–infrared detection with test-time adaptation. Search date 2026-10-12.

## What N4 is, in the terms the literature uses

* **Training-free BN-statistics re-estimation:** BN-mix (Schneider et al., 2020), N = 16, blocks of 8 frames. No
  gradient, no loss, no pseudo-label, no learned parameter.
* **One modality only:** the VIS detector of a late-fusion two-detector system. IR is never adapted.
* **A per-frame, label-free gate decides whether to adapt at all.** VIS reports itself unhealthy (or glare), AND the
  IR camera says the scene is in daylight, AND IR is itself healthy.
* **IR is used only as a physical judge of scene illumination.** It answers whether VIS darkness or veiling comes
  from the camera or from the world. It is not used as a feature source, a teacher or a fusion weight.
* **Task:** maritime visible–infrared object detection, scored on ship AP.
* **Evidence:** preregistered tests on a held-out run (P-T, P-A, P-FT, P-C2, P-G).

## Closest prior work (full text read)

| Paper | Task / modalities | What adapts, how | Role of the other modality | Gate on *whether* to adapt? |
|---|---|---|---|---|
| Night-TTA (Liu et al., IEEE TAI; arXiv 2307.04470) | Night colour–thermal **segmentation** (MF, KP) | BN **affine** parameters by gradient; ensemble-teacher KL plus entropy losses | Thermal is a parallel branch and part of the teacher ensemble | No: every night test image is adapted |
| READ (Yang et al., ICLR 2024) | Audio–video **classification** (Kinetics50-C, VGGSound-C) | Fusion **attention** projections by gradient; confidence-aware entropy loss | Attention reweights toward the reliable modality | No: every sample is adapted; the corrupted encoder is not adapted |
| TSA (Chen et al., ICML 2025) | Audio–video classification, one modality shifted | Modality-specific **adapters** plus a learnable **router**, end to end, self-training | The router chooses which modality's adapter to train | Per modality, learned by gradient. It states the key motivation: adapting an unshifted modality causes **negative transfer** |
| AdaPGC (Xu et al., CVPR 2026; arXiv 2604.19093) | Audio–video classification | Fusion attention and the flagged encoder's **LayerNorm** by gradient; streaming Gaussian class statistics | Symmetric KL of each unimodal posterior against the **fused prediction** flags the corrupted modality | Routes every sample to one modality; no skip |
| CARAT (arXiv 2610.07499) | Wearable time-series classification | LayerNorm affine terms plus per-modality attenuation; weighted entropy | Each stream is scored **only against its own** clean statistics. It argues cross-modal agreement misleads when sensors measure different physics | Per modality and per sample, by one-class detectors |

Also known, not re-read here:
* MM-TTA (Shin et al., CVPR 2022): camera–LiDAR segmentation, BN updated with cross-modal pseudo-labels.
* EATA and SAR: sample filtering for entropy TTA.
* Yoo et al., CVPR 2024: "What, how and when should object detectors update".
* Illumination-aware RGB-IR fusion detectors (e.g. EMOD; UA-CMDet, already in Table R): illumination steers
  *fusion weights*, not adaptation.

No paper found applies test-time adaptation to **visible–infrared object detection** (searches for LLVIP, FLIR,
KAIST and maritime).

## Verdict

**Not novel:**
* "Adapt only the corrupted modality, using the other modality": TSA, AdaPGC, CARAT, READ.
* "Adapting where there is no shift hurts": TSA calls it negative transfer.
* BN-statistics adaptation itself: AdaBN, Schneider et al. 2020.
* Test-time adaptation on colour–thermal data at night: Night-TTA.

**Not found in prior work, so defensible as novel with that caveat:**
1. **A second sensor as a physical judge that gates adaptation.** Prior selective methods decide from the model's own
   predictions or features: a learned router (TSA), divergence from the fused posterior (AdaPGC), per-stream anomaly
   scores (CARAT). N4 decides from what the thermal camera says about the *world* (daylight), which separates a
   degraded camera from a dark scene. CARAT argues against cross-modal signals in general; N4 uses one that is not
   agreement between predictions but an independent physical reading. The night result is the evidence that this
   matters: ungated adaptation costs up to −0.031 at night, and the gate never fires there.
2. **Fully training-free and gradient-free.** Every prior multimodal method above updates parameters by gradient.
   N4 changes no weight.
3. **Test-time adaptation for visible–infrared object detection, maritime.** No prior found.
4. **Preregistered held-out replication** of the method and four variants.

**Suggested claim wording:** "To our knowledge, the first label- and gradient-free test-time adaptation for
visible–infrared detection in which the second sensor gates adaptation by judging scene illumination, rather than by
model-side agreement. Selective multimodal TTA (TSA, AdaPGC, CARAT) and colour–thermal TTA (Night-TTA) are
the closest prior work."

**Do not claim:** first multimodal or cross-modal TTA; first selective TTA; first colour–thermal TTA; that
gating is new.

## Limits of this check

* Five papers read in full; the rest come from searches. One search engine. The field moves fast: 2026 preprints
  appeared in every search.
* Not read in full: DASP (He et al., 2026, selective adapter updates from feature redundancy), SuMi (ICLR 2025),
  ABPEM (AAAI 2025). Each is gradient-based on audio–video benchmarks, judging from abstracts and citing papers.
  Read them before submission.
* Detection-specific TTA in driving (e.g. LearnableBN, arXiv 2411.18860) is single-modality.

Sources: arXiv 2307.04470; ICLR 2024 READ; PMLR v267 chen25ch (TSA); arXiv 2604.19093v1 (AdaPGC);
arXiv 2610.07499 (CARAT).
