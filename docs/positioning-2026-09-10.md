# Positioning — what this project can actually claim

**R-F1, review finding F17.** Written 2026-09-10. The backlog notes this item has the
only external deadline and depends on no repair above it.

The scope document's novelty claim is false **in two independent ways**, and only one of
them is about the literature.

---

## 1. The two failures, separately

### (a) The literature claim is contradicted by prior art

`scope.md` §2 asserts:

> **Existing visible–infrared fusion is static** — fixed rules or learned-but-static
> attention, never conditioned on a live per-frame reliability estimate.

and §1:

> Unlike most prior maritime fusion work, which blends sensors on a fixed schedule,
> this system blends them based on live, per-frame uncertainty.

Both overstate. Externally verified in
[`architecture-review-2026-09-09.md`](architecture-review-2026-09-09.md) §F17:

* **[UA-CMDet](https://github.com/SunYM2020/UA-CMDet) (2022)** combines uncertainty-aware
  cross-modal learning with **illumination-aware NMS at inference** — inference-time
  adaptation conditioned on the current frame.
* **[Uncertainty-Aware Cross-Modality Fusion for Visible-Infrared Object Detection](https://doi.org/10.1109/DICTA63115.2024.00029)**,
  DICTA 2024.
* **[Gaussian YOLOv3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.pdf)**,
  ICCV 2019 — a single-pass Gaussian localization head is established prior art. This is
  already cited as R1 in `scope.md` §19.1 as "the pattern"; it cannot simultaneously be
  the pattern and the thing that makes the head novel.

And the sharper logical point from the review: **learned attention is not "static" merely
because its parameters are frozen at inference.** The attention *values* depend on the
input, which is exactly the property the claim reserves for itself.

### (b) The system claim is contradicted by our own measurement — and this is the worse one

**R-D1, 2026-09-10** ([`uq-mechanism-2026-09-10.md`](uq-mechanism-2026-09-10.md), prereg
`a8f087c`): under the shipped preset, `mu_d = 1e9` and `lam = 0.0`, so `r_frame_vis` and
`r_frame_ir` are exactly 1.0000 on every frame, and **`w_vis` is a single constant
0.9930 across all 2,232 frames × 4 conditions.**

The shipped system does not blend "based on live, per-frame uncertainty." **It blends
with a constant weight** and selects sensors with a hard veto driven by image statistics.
Supplied with real uncertainty in a pre-registered ablation, the fusion did not beat the
same uncertainties attached to the wrong boxes — 0 of 4 conditions at every floor tested.

So §1 as written is not merely a strong claim about the field. It is an inaccurate
description of this repository. A reviewer who ran the code would find that in an
afternoon.

## 2. The comparison table that replaces the categorical language

The review asks for a comparison rather than a superlative. Axes as specified: domain,
sensors, uncertainty target, inference-time adaptation, calibration evaluation,
registration assumptions, compute.

| | **This project (as measured)** | UA-CMDet (2022) | DICTA 2024 | Gaussian YOLOv3 (2019) | Maritime YOLO variants (SID-YOLOv5, EG-YOLO, RDSC-YOLOv4, YOLOv7-sea) |
|---|---|---|---|---|---|
| **Domain** | Maritime, harbour/canal, one recording (Pohang) | Drone-based vehicle detection | VIS–IR object detection | Autonomous driving | Maritime |
| **Sensors** | VIS + LWIR, timestamp-paired | RGB + IR | VIS + IR | RGB only | RGB (mostly) |
| **Uncertainty target** | Per-box localization scale (Gaussian head) + frame-level Mahalanobis OOD | Cross-modality uncertainty for learning + NMS | Cross-modality fusion uncertainty | Per-box localization variance | **None** |
| **Inference-time adaptation** | **Hard veto on image statistics; fusion weight is a CONSTANT 0.9930 (measured)** | Illumination-aware NMS at inference | Fusion conditioned on modality uncertainty | None (single stream) | None |
| **Calibration evaluated?** | **Yes — D-ECE, interval-ECE, NLL, AUSE/AURC, with declared contracts and a measured noise floor** | Not established here — verify before submission | Not established here — verify before submission | Reports detection gains; calibration not the focus | No |
| **Registration assumptions** | **Measured, not assumed**: 3–6 px median residual; at `iou_thr` 0.85 only ~0.05% of VIS boxes have an IR partner | Not established here | Not established here | n/a | n/a |
| **Compute** | Single pass per detector; "one pass" means one **per stream** (R-E3 open) | Not established here | Not established here | Single pass | Single pass |

**Cells marked "not established here" are deliberate.** The external review verified that
these works *exist* and what their headline mechanism is; it did not audit their
calibration protocols or registration assumptions, and neither have I. Filling those
cells from memory is precisely the failure mode this backlog exists to correct. **They
must be read from the papers before this table goes into a manuscript.**

## 3. What is actually defensible

Three things, and the first two are stronger than the claim they replace because they are
measured rather than asserted.

**A controlled maritime uncertainty study, including its negative results.** This is the
real contribution. Very little published work reports: a paired noise floor computed by
the right method (0.0014–0.0031, and R-A3 shows it is itself ~1.95× understated because
it was computed IID on 10 Hz video); a declared AP convention with a measured parity
bound; metric contracts stating what D-ECE, AUSE and AURC do *not* measure; and a
pre-registered ablation that returned NULL and was published anyway.

**A lightweight, interpretable sensor-selection baseline — with its failure cases.**
`grad_gini` + `ir_p05` is a two-axis, inspectable rule that beats both single streams on
all eight cells. Its failure cases are documented and quantified rather than hidden: the
veil veto is a −0.0632 regression on `yolo26m`; VIS is vetoed on 100% of fog frames and
on the whole night run; the fusion is concatenation rather than consensus because the
geometry does not permit merging.

**An honest account of when uncertainty-driven fusion does not pay.** R-D1 is publishable
as a negative result with a mechanism: the fusion weight is constant, cross-modal clusters
are rare at the adopted IoU, and the only place predicted uncertainty showed positive
signs was re-ranking a *single* stream's detections — where no fusion is happening at all.

## 4. Replacement language for `scope.md`

Proposed, and applied to `scope.md` in this commit with the original text preserved
inline so the change is visible rather than silent.

**§1, replacing "Unlike most prior maritime fusion work…":**

> Prior work has applied uncertainty to visible–infrared fusion (UA-CMDet 2022; DICTA
> 2024) and to single-pass localization variance (Gaussian YOLOv3 2019). This project's
> contribution is not the idea but the **controlled maritime study**: a pre-registered
> evaluation of whether predicted uncertainty improves fusion on paired maritime imagery,
> with declared metric contracts, a measured noise floor, and negative results reported.
> Measured on this data, the adopted system's gain comes from **image-statistic sensor
> selection**, not from uncertainty-weighted blending.

**§2 gap 2, replacing "Existing visible–infrared fusion is static":**

> Uncertainty-conditioned visible–infrared fusion exists (UA-CMDet 2022, DICTA 2024), and
> learned attention is not static merely because its parameters are frozen — attention
> values depend on the input. **What is scarce is calibrated evaluation**: whether the
> uncertainty driving the fusion is itself trustworthy, measured against a stated noise
> floor with dependence-aware intervals, and reported when the answer is no.

**§3 O5** promises to "show uncertainty-gated fusion beats single-modality and
uncertainty-blind fusion." The first half holds — all eight cells sit at or above
max(VIS, IR). **The second half is now measured and does not hold**, and O5 should say
so rather than promise a result the project has already contradicted.

## 5. What this does not do

* **It does not audit the cited papers.** The comparison table's blank cells are honest
  gaps, and R-F2 has separate bibliography corrections (D-FINE is ICLR 2025, RT-DETR is
  CVPR 2024, Pohang and PoLaRIS are distinct releases) that are not addressed here.
* **It does not claim priority for the negative result.** Other people have surely found
  uncertainty-weighted fusion inert; the contribution is the controlled measurement on
  this data with the rule fixed in advance.
* **It changes wording, not results.** No number in this repository moves.
