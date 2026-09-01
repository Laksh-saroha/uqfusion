# The 26m swap, and every lever the fusion layer had left

**Written:** 2026-09-01. **Supersedes in part:** `docs/crossmodal-gate-2026-09-01.md`
(§3c's `cap_ir_scale`, and the veil axis inherited from
`docs/veil-veto-and-rule-sweep-2026-09-01.md`). Nothing in either has been edited
or deleted, and both still reproduce bit-for-bit under `preset="crossmodal"` on
`runs/cache/`.

Read §0 and §1 first. Two things I had stated as settled turned out to be wrong,
and one of them was load-bearing for the whole gate.

---

## 0. What changed, in one table

| | before | after |
|---|---|---|
| detectors | `runs/phase2/*` — yolo26s, VIS stride5, **IR nc=2** | `runs/full_scale/*` — **yolo26m** (VIS), **yolo26m-p2feat nc=1** (IR) |
| caches | `runs/cache/` | **`runs/cache_m/`** (new dir; the old one is untouched) |
| preset | `crossmodal` | **`crossmodal26m`** |
| veil veto | `grad_gini < 0.4826` **OR** (night AND VIS dark) | night **AND** (VIS dark **OR** `grad_gini < 0.4826`) |
| cross-modal term | WBF merging at `iou_thr` 0.85 | + **support** at IoU 0.30, γ 0.5 (score only, never coordinates) |
| day selection set | "clean fit-run day" = all 1200 day frames | **`TUNE_RUNS`** pohang00 (836) / **`TEST_RUNS`** pohang02+03 (364) |
| worst cell vs `max(VIS, IR)` | **−0.0632** | **+0.0000** |

---

## 1. Two corrections to the record

### 1.1 "The detector was the biggest lever" — wrong on clean, right on fog

I told Laksh the full-scale retrain would dominate every fusion change. On the
**clean** cell it buys essentially nothing:

| detector | VIS day | IR day | gated | night VIS |
|---|---:|---:|---:|---:|
| yolo26s | 0.3683 | 0.0181 | 0.3736 | 0.0000 |
| yolo26m | 0.3686 | 0.0192 | 0.3702 | 0.0000 |

+0.0003 on VIS, +0.0011 on IR, and night VIS is **still exactly 0.0000** — so half
the benchmark stays a single-sensor regime and no VIS/IR fusion property can be
tested there at all.

On **fog** it dominates everything else in this document:

| detector | VIS under fog (day) | IR (day) |
|---|---:|---:|
| yolo26s | **0.0020** | 0.0181 |
| yolo26m | **0.0824** | 0.0192 |

**41x.** That single number invalidates the veil veto — see §2.

### 1.2 "Fusion's day gain is WBF's consensus boost" — wrong, and not fixable

I had described the day-cell gain as agreement re-ranking VIS's own boxes. It is
not, and it cannot be. Measured directly, on day frames, share of VIS detections
with a same-class IR detection above each IoU:

| homography | IoU≥0.85 | IoU≥0.55 | IoU≥0.30 | IoU≥0.10 |
|---|---:|---:|---:|---:|
| adopted | **0.05%** | 12.24% | 32.27% | 47.75% |
| + per-frame alignment (§3) | **4.02%** | 22.61% | 43.60% | 55.73% |

`iou_thr` is **0.85**. At that overlap the two streams essentially never share a WBF
cluster, so the `min(n_models, n_cluster)/sum(weights)` bonus almost never fires
cross-modally. Fusion is ~99.9% **concatenation**: VIS boxes scaled by `w_vis`
≈0.993, IR boxes by `w_ir` ≈0.007 and appended at the bottom of the pooled ranking,
where a few of them recover GT that VIS missed. (`x_score_calibration.md` already
said "99.9% concatenation" in passing; I had not connected it to the mechanism.)

Consequence: `consensus_beta` and `consensus_distinct`, added to make that bonus
explicit and tunable, are nearly inert on real data and negative where they are not
(`consensus_distinct`: −0.0068 on the clean day cell). They stay in the code as
defaults-off levers and are not adopted.

---

## 2. The veil veto is calibrated to a weakness the deployed detector does not have

The veil axis fires on **100%** of fog frames and hands them to IR. Against yolo26s
that was correct — VIS 0.0020, IR 0.0181. Against yolo26m it is a **−0.0632**
regression, the only cell anywhere below its bar:

| | bar = max(VIS,IR) | gated | gap |
|---|---:|---:|---:|
| fog/day, 26m, `crossmodal` | 0.0824 | 0.0192 | **−0.0632** |

**No image statistic can detect this**, and that is the point worth keeping. The
axis measures the fog perfectly well; what stopped being true is the *claim*
attached to it — "a fogged VIS cannot see". That fact is not in the image. It is a
property of the checkpoint downstream.

Deleting the axis is not the repair either. It fixes the day cell and costs the
night one, because fog lifts VIS `p05` above `mu_b` on 69% of night frames and the
veil axis is exactly what covers the night arm's resulting blind spot:

| rule | fog/day | fog/night |
|---|---:|---:|
| adopted (`veil OR night`) | 0.0192 | 0.0850 |
| no veil axis | 0.0848 | **0.0503** |
| **`night AND (dark OR veil)`** | **0.0848** | **0.0850** |

So the axis is kept and made **conditional on the other sensor agreeing it is
dark**. Fog at night is still caught, by both paths; fog in daylight is caught by
neither, which is now the right answer. With the support term of §3 on top:

- fog/day **0.0192 → 0.0908**, `+0.0716 [+0.0657, +0.0784]`
- fog/night **0.0850 → 0.0850**, `+0.0000 [+0.0000, +0.0000]`
- worst cell over all eight: **−0.0632 → +0.0000**

This is the general lesson and it applies beyond fog: **a veto encodes a claim
about the detector, not about the image, so every veto axis has to be re-priced
when the detector changes.** The thresholds are pixel statistics and did not need
refitting — `verify_cache_m.py` shows they are identical to 0.000e+00 across the
swap. The rule built on them did.

---

## 3. Registration: the mechanism improves 80x and AP does not

`src/uqfusion/eval/iralign.py` estimates a per-frame IR→VIS translation from the
two **detection** sets — median offset of nearest-centre same-class pairs, greedy
one-to-one, ≥3 pairs, ≤40 px — and composes it onto the homography as `T @ H`. No
GT anywhere, so it is something the deployed system could actually do.

It fires on **47%** of frames at a median **3.1 px**, which independently reproduces
the 3–6 px residual `x_registration_drift.md` measured from GT. It raises
cross-modal agreement 80x at IoU 0.85 (§1.2 table).

**Ship AP falls.** The new agreement arrives as WBF *clusters*, and merging drags the
VIS box toward the worse-localised IR box; forbidding the merge (`iou_thr` 0.95)
recovers the tune runs from −0.0043 to +0.0006, which confirms that mechanism. But
every alignment arm is negative on the held-out runs (−0.0025 to −0.0026).

So 80x more geometric agreement carries **no usable information about which VIS
boxes are right**. That is a statement about the ceiling, not about the estimator,
and it rules out a family of ideas: better registration, a tuned agreement bonus,
and a lower `iou_thr` all target the same empty mechanism.

The one exception is worth noting: on cells where the system falls back to IR alone,
alignment does help (fog/day under the *old* veto: 0.0192 → 0.0524), because there
it is improving the IR→VIS mapping rather than a merge. Once §2 stops vetoing
daylight fog, that path is no longer taken and the gain disappears.

---

## 4. What survives: cross-modal support

Agreement is not absent — it is at the wrong *scale*, 32% at IoU 0.30 rather than
0.05% at 0.85, which is where a 3–6 px residual puts it. And `iou_thr` cannot be
lowered to reach it (−0.0138 on the fit set at 0.55) because merging across that
residual moves the box.

`support_iou` / `support_gamma` split the two: a cluster's score is multiplied by
`1 + γ` when a box from the *other* stream overlaps it at `support_iou`, and no
coordinate is touched. IR says **whether**, not **where**.

| arm | tune | **TEST** (held out) | day | fog/day |
|---|---:|---:|---:|---:|
| `crossmodal` | 0.3816 | 0.3533 | 0.3702 | 0.0192 |
| + support IoU 0.30 γ 0.5 | +0.0078 | **+0.0033** | +0.0090 | — |
| **`crossmodal26m`** (§2 + support) | +0.0078 | **+0.0033** | +0.0090 | **+0.0716** |

Confirmed on the full 10-cell extended grid (`runs/eval/final_26m_grid.md`, paired
bootstrap n=500, day frames, against `max(VIS, IR)` on the same streams):

| cell | gated | bar | delta | 95% CI |
|---|---:|---:|---:|---|
| clean/clean | 0.3792 | 0.3686 | **+0.0106** | [+0.0086, +0.0124] |
| clean × IR glare_s2 | 0.3724 | 0.3686 | +0.0038 | [+0.0019, +0.0065] |
| clean × IR blur_s2 | 0.3757 | 0.3686 | +0.0071 | [+0.0051, +0.0095] |
| blur_s3/clean | 0.0448 | 0.0425 | +0.0023 | [+0.0011, +0.0038] |
| rain_s2/clean | 0.1137 | 0.1023 | **+0.0114** | [+0.0101, +0.0125] |
| lowlight × IR glare_s2 | 0.0472 | 0.0435 | +0.0038 | [+0.0034, +0.0055] |
| blur_s3 × IR glare_s2 | 0.0432 | 0.0425 | +0.0006 | [−0.0019, +0.0019] |
| clean × IR fog_s2 / noise_s2, noise_s2/clean | — | — | +0.0000 | fully vetoed or IR-better; ties the bar |

**Zero cells below their bar**, worst +0.0000. Under `crossmodal` on the same
caches the worst was −0.0632.

IoU 0.30 rather than 0.55 is the whole reason §5 exists — see there.

---

## 5. The benchmark had no held-out day data, and it mattered immediately

`FIT_RUNS` excludes only `pohang01`, and `pohang01` is **entirely night**. So "clean
fit-run day" and "clean day" are the same 1200 frames: every day constant in this
project — `cap_ir_scale` x4 included — was selected and reported on one set, and the
only genuinely held-out run is one where VIS scores 0.0000.

`TUNE_RUNS = (pohang00,)` (836 day frames) and `TEST_RUNS = (pohang02, pohang03)`
(247 + 117 = 364) split it **by run**, because consecutive frames of one canal
transit are near-duplicates and a frame split would leak.

It earned its keep on the first arm it touched:

| support_iou | tune | TEST |
|---|---:|---:|
| 0.55 | **+0.0121** | **−0.0010** |
| 0.30 | +0.0078 | **+0.0033** |

`0.55` wins the tune runs and is negative on the held-out runs in **all six**
variants it appears in; `0.30` is positive in all six. Under the old single-set
discipline both would have read as wins, and the wrong one would have been adopted.

Caveat stated plainly: the held-out side is 364 frames and its CIs are wide
(`+0.0033` is not separated from zero at n_boot 500). It is enough to *reject* an
arm, not yet enough to certify one.

---

## 6. What was measured and rejected

| lever | result |
|---|---|
| `cap_ir_scale` re-price | x8/x16 marginally better on clean (+0.0006/+0.0010 TEST) and worse on `clean × IR glare_s2`/`blur_s2`. **Kept at x4**; the ratio moved only 142x → 135x across the swap. |
| `iou_thr` re-sweep | 0.75 and 0.95 both below 0.85 on TEST. **Unchanged.** |
| `consensus_beta`, `consensus_distinct` | inert or negative; §1.2 explains why. **Rejected.** |
| per-frame registration | §3. **Rejected** — mechanism +80x, AP negative held-out. |
| `class_veto` (IR is nc=1, so a VIS veto deletes every buoy) | **exactly zero on every cell.** VIS is only ever vetoed on night frames, and the night GT contains no buoys at all. Correctly implemented as carry-through (the first version cost −0.0065 of *ship* AP by turning vetoed frames back into two-stream frames); kept in the code, not adopted. |
| `ir_nms` re-tune | **not a lever.** Swept off/0.5/0.6/0.7/0.8/0.9 — 15.7 to 30.0 IR boxes per frame, a 2x range — and the day column is flat to ±0.0002 throughout. Night moves at most +0.0005 (0.6 over the adopted 0.7), which is the only place it can act since night is 100% IR. **Kept at 0.7.** |
| cross-modal score calibration | **worse.** Isotonic `conf -> P(TP@0.5)` per modality, fitted on pohang00 only: tune −0.0044, TEST −0.0035. It also turns out to make `cap_ir_scale` completely inert (x1, x4 and x16 give identical AP to four decimals, where without calibration they give 0.3676 / 0.3702 / 0.3713 — `replace(ctx, cap_ir=…)` was checked directly and does work). The likely reason, stated as inference rather than measurement: calibrated IR P(TP) occupies a narrow band (0.02–0.13 across IR's whole confidence range, against VIS's 0.06–0.97), so re-weighting slides that band as a block, and once the block sits below the VIS boxes near the precision–recall knee, sliding it further changes no ranking that AP can see. **Rejected.** |

---

## 7. Files

New: `runs/cache_m/` (14 caches, canonical), `runs/cache_m_stageprobe/` (4 stage
variants), `src/uqfusion/eval/iralign.py`, `scripts/{verify_cache_m,
probe_detector_swap,sweep_support,sweep_align,eval_levers,sweep_irnms_calib,
smoke_new_levers,build_paired_grid_m}.py|sh`.

Results: `runs/eval/{cache_m_verify,detector_swap_clean,support_26m,align_26m,
align_nomerge_26m,levers_26m,fog_veto_26m,veil_veto_repair_26m,final_26m_grid}.{md,json}`.

Nothing under `runs/cache/`, `runs/derived/`, or `runs/eval/` was overwritten.

## 8. What is still open

1. **Night is a dataset decision, not a fusion result.** VIS is 0.0000 on all 1032
   night frames with both detectors, and `filter_night_boxes.py --cut-dark` removed
   132k night boxes from the training labels on purpose. Night is single-sensor by
   construction. Any claim about fusion at night is a claim about IR alone.
2. **The held-out day set is 364 frames.** It can reject; it cannot yet certify.
   A larger or differently-cut day holdout is the cheapest way to raise the
   evidential standard on everything in §4.
3. **Every other veto axis is now suspect for the same reason as §2.** The
   photometric axis, the IR health bound, and the authority bound were all fitted
   or validated against yolo26s behaviour. Only the fog one has been re-priced.
4. **The learned gate** is still in no headline table.
5. **`cap_ir_scale` x4** was selected under the pre-§5 discipline and has not been
   re-selected under it.
