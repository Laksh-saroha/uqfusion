# TODO — score improvements not yet run

> **SUPERSEDED 2026-08-20.** The current list is
> [`TODO-2026-08-20-full-scale.md`](TODO-2026-08-20-full-scale.md); open items
> here were carried over with their old IDs in brackets. This file stays as the
> 2026-08-19 record.

**Created:** 2026-08-19, after the end-to-end architecture test
([`handoff-2026-08-19-fusion.md`](handoff-2026-08-19-fusion.md)).

**For the full record of what was tried and rejected — and the prioritised future
testing plan, including the falsification tests — see
[`fusion-gate-experiment-record.md`](fusion-gate-experiment-record.md).**

**Updated 2026-08-19** after the cheap-fixes pass completed. The six CPU-only fixes
are **done** — results in [`runs/eval/cheap_fixes.md`](../runs/eval/cheap_fixes.md),
analysis in [`handoff-2026-08-19-fusion.md`](handoff-2026-08-19-fusion.md) §14.
Outcome in one line: two knobs gave +2.7% clean / +2.3% glare, one hypothesis was
refuted, and the per-run breakdown found that **pohang01 is a real night run where
VIS scores exactly 0.0000 and IR carries the system** (§14.1). That finding
re-prioritised this list — see the new section **0** below, which now comes first.

Everything below section 0 is what remains.

Time estimates are grounded in this laptop's **measured** throughput
(RTX 4080 Laptop, 12 GB): VIS **282 s/epoch** (stride-5, 19,256 frames, batch 24,
imgsz 640), IR **125 s/epoch** (stride-2, 11,640 frames). Phase 2 runs early-stopped
around 54–68 epochs, so one VIS run ≈ **4.3 h**, one IR run ≈ **3.5 h**.

---

## 0. Promoted to the front by the cheap-fixes pass

These come before everything in A/B. They are all cheap, and items 0.1–0.3 are
blocking for the paper's central claim.

**Status 2026-08-19: 0.1, 0.2, 0.2b, 0.3 and 0.5 are DONE, and A1 is done with a
diagnosed null result.** The central claim holds on real data — gated fusion scores
**0.0813** on the real night run against `ir_only`'s **0.0810**, having been behind
in all four conditions before this pass, with every daylight per-run number
unchanged or better. Adopted configuration: D-6 ladder constants, capability prior
VIS 0.2580 / IR 0.0206, photometric gate on `p05` with `mu_b`=10.500 /
`tau_b`=2.625 (margin rule) combined by `min`, hard veto at `b < mu_b`, WBF
`iou_thr` 0.85.

**Read [handoff §18](handoff-2026-08-19-fusion.md) before quoting any number from
§16 or §17.** The IR capability prior was 3.3x too high in those two scripts
(0.0676 instead of 0.0206); every conclusion survived re-measurement but the
absolute values moved, and §18.2 carries the corrected table.

**The new blocker is A2, not anything in section 0.** A1 revealed that at
`iou_thr` 0.85 only **31 of 29,042** VIS detections have an IR partner above
threshold — fusion is almost entirely concatenation, so no fusion-rule improvement
can do anything until registration improves. See 0.4 and A1/A2 below.

### 0.1. Split Table 3 day/night — **DONE 2026-08-19**
**Actual: ~1 h.** `run_fusion_eval.py` now emits a "Day vs night" section (and a
`split_rows` block in the JSON), so the headline Table 3 carries the split.
Measured day-pooled visible-only is **0.3352** against the pooled 0.2580.

The split re-scores the SAME fused outputs on a frame subset — fusion is never
re-run — so the split can never disagree with the pooled row about what the system
did. Regenerate with:

    python scripts/run_fusion_eval.py --capability-weighted --iou-thr 0.85 \
        --constants runs/eval/reliability_constants.json \
        --brightness-constants runs/eval/brightness_constants.json --veto 0.5

**That `--constants` flag is not optional.** Without it the script falls back to
the D5/B5 rule (`mu_d`=35.30) while every analysis in §14-§17 uses the D-6 ladder
fit (`mu_d`=71.07), and gated glare collapses from 0.2058 to 0.0641. See §18.4.

One extra reason to split, found the hard way: pooled mAP **fell** 0.2695 ->
0.2580 under §0.2 while every per-run number rose or held. Pooled mAP ranks all
detections in one global list, so a change confined to night frames moves the
daylight precision too. Pooled numbers here move for reasons unrelated to either
population.
 No new code beyond a grouping argument to `run_fusion_eval.py`.

pohang00/02/03 give visible-only 0.4004 / 0.3653 / 0.1840. pohang01 gives **0.0000**.
Pooling them and reporting 0.2580 hides both the real daylight capability and the
real night result. The night split is the headline; the day split is the honest
"what does fusion cost on clean data" number.

Also record the sampling caveat wherever a paired-subset number appears: pohang01 is
**46%** of the 2,232 paired frames but only **18.2%** of the full 11,352-frame VIS
val, because IR val coverage concentrates in pohang00/01. Paired-subset VIS numbers
are **not** comparable to the Phase 1/2 benchmark.

### 0.2. Make the gate see darkness — ~~cheap probe before A3~~ **DONE 2026-08-19**
**Actual: ~2 h, CPU only.** Result in [handoff §16](handoff-2026-08-19-fusion.md).

Fitted on pohang00/02/03 synthetic lowlight, pohang01 held out entirely. Held-out
separation is total: every one of the 1,032 real night frames scores `r_bright`
below every one of the 1,200 day frames. Daylight mAP unchanged to four decimals;
night w_vis falls 0.792 -> 0.199 and night mAP rises 0.0769 -> 0.0791.
**A3's learned head is not needed for this failure.**

Final constants after §0.2b: statistic `p05`, margin rule, `mu_b`=**10.500**,
`tau_b`=**2.625** (`runs/eval/brightness_constants_p05_margin.json`). The first
fit — `mean`, `mu_b`=58.238, `tau_b`=17.869 — is superseded on both counts.

Root cause found and worth recording: the Mahalanobis reference set contains 782
pohang01 night frames out of 4,000, so night was never out-of-distribution and no
re-tuning of `mu_d`/`tau` could have fixed it.

The four-condition regression check makes that airtight. Synthetic lowlight on DAY
frames (brightness **5.4**) and real night (brightness **8.2**) are the same darkness,
yet D alone gives w_vis **0.037** vs **0.792** — a 21x gap. D catches the synthetic
transform because it is genuinely unusual; it is blind to real night because real
night defines its notion of normal. Consequence: fog and lowlight are **bit-identical
before/after**, so the photometric term changes exactly one thing and is inert
everywhere else *by measurement*. It also means no severity ladder, however many
corruption families it covered, could ever have surfaced this failure.

**`p05` replaced `mean` — done, handoff §16.4c.** The luminance-spoofing weakness was
real and is fixed. Selected on fit runs only via `probe_stat_robustness.py` (darken a
fit-run image, add glare/fog, measure how far the statistic is dragged back): `p05` is
**completely immune to glare** (spoof 0.000 vs `mean`'s 0.354) and 3x more fog-resistant,
with equivalent fit RMSE. The glare/night regression inverted from **-1.1% to +8.4%**,
every day split held or improved, and night `w_vis` collapses to 0.199 (0.002 under
the p05 fit that §0.2b superseded).

### 0.2b. Fix `p05`'s threshold placement — **DONE 2026-08-19**
**Actual: ~1 h, CPU.** Adopted `runs/eval/brightness_constants_p05_margin.json`.

`p05` introduced one regression: **pohang03 0.1779 -> 0.1700 (-4.4%)**. Night max p05
is 4.0 and day min is 21.0 — a wide clean gap — but `mu_b` landed at **25.4, inside the
daylight distribution**, with `tau_b`=2.52 sharp enough that pohang03 (p05 mean 25.9)
sits on the cliff and gets damped to r_bright 0.53 on a run VIS handles fine.

**Cause (measured — the first write-up here was wrong).** It is *not* that lowlight s1
produces intermediate p05 values: on fit runs lowlight s1, s2 and s3 all read p05 = **0.0
exactly**, and 0.0% of s1 frames fall inside the clean range [21, 44]. The real cause is
that `p05` is **bimodal by construction** — corrupted frames read 0, clean frames read
>= 21, nothing in between — so *every* threshold in (0, 21) fits the retention curve
equally well and the optimizer's pick came from a **within-daylight confound**: the
`b = 25.5` bin is pohang03's clean daylight at retention 0.51, because pohang03 scores
0.1840 vs the fit-run clean mean 0.3437. That is scene difficulty, not dimness, and the
fit read it as causal.

**Fix adopted: `--rule margin`** — state the placement instead of fitting it, using only
fit-run endpoints:

```
mu_b  = 0.5 * (max p05 over corrupted fit frames + min p05 over clean fit frames)
tau_b = (min_clean - max_corrupt) / 8
```

Margin [0.00, 21.00] -> `mu_b` = **10.500**, `tau_b` = **2.625**. Day r_bright floor
rises 0.150 -> **0.982**, night max stays at **0.078**.

| | before | s2/s3 refit | **margin (adopted)** | p05 original |
|---|---|---|---|---|
| pohang03 | 0.1779 | 0.1777 | **0.1779** | 0.1700 |
| pohang01 (held out) | 0.0769 | 0.0790 | **0.0791** | 0.0788 |
| glare / night | 0.0728 | 0.0784 | **0.0789** | 0.0788 |
| clean / day | 0.3303 | 0.3304 | 0.3303 | 0.3323 |
| sum, 8 day/night cells | 0.9083 | 0.9161 | **0.9168** | 0.9188 |

The s2/s3 refit also helps, but by accident: dropping s1 shifts the quantile-bin
boundaries so the confounded `b = 25.5` bin dissolves, leaving the fit unconstrained
across the gap. The original `p05` fit's higher 8-cell sum is an artifact — its pooled
clean/day gain comes from *suppressing* pohang03 in the global ranking, which per-run
costs that run 4.4%. Detail in handoff §16.4d/§16.4e.

<details><summary>original plan</summary>

**Est. ~2 h, CPU only.**

On pohang01 the VIS Mahalanobis distance is **28.4** — *lower* than pohang00's 30.0 —
on frames with mean content intensity **8.2** (max 14.9) where the detector's max confidence
over 1,032 frames is **0.0043**. Pooled-neck-feature distance does not see darkness
at all. This is A3's monotonicity ceiling, but now failing on real data instead of on
the ladder, which promotes A3 from "nice to have" to **blocking**.

Before committing to A3's 1–2 day learned head, spend 2 h on the cheap probe: add raw
frame statistics (mean intensity, low percentile, RMS contrast) as a second gate input
and check whether they separate pohang01 from pohang00. If they do, A3 may be
unnecessary.

</details>

### 0.3. Stop the gate diluting a working stream — **DONE 2026-08-19**
**Actual: ~1.5 h, CPU.** Result in [`runs/eval/veto_rule.md`](../runs/eval/veto_rule.md),
analysis in [handoff §17](handoff-2026-08-19-fusion.md).

Gated fusion on the night run now scores **0.0813 against ir_only's 0.0810** — ahead
for the first time, in three of four conditions, having trailed in all four. Every
daylight per-run number is unchanged. 8-cell day/night sum 0.9168 -> **0.9238**.

**None of the three candidate rules listed here was the answer, and the reason is
worth keeping.** The weights *did* collapse (w_vis 0.792 -> 0.199); it still was not
enough, and no floor, ceiling or conditional prior could have been, because the
problem is not in the weight space at all:

- WBF **rescales scores, it does not drop boxes** — and the rescale hits the good
  stream too. Measured: an unmatched IR box entering at conf 0.70 leaves fusion at
  **0.21**, because every single-modality cluster is scaled by that modality's
  normalised weight. A blind VIS stream does not merely add false positives, it
  attenuates IR's true positives by `w_ir`.
- mAP is **rank-based**, so scaling a false positive down does not remove the rank
  it occupies.

Neither effect shrinks as w -> 0. A failed modality has to leave the input list.

**Adopted rule — no new constant:**

    exclude modality m from fusion when   r_bright_m < 0.5   <=>   b_m < mu_b_m

`mu_b` is the §0.2b margin midpoint, already pre-registered; 0.5 is the sigmoid
midpoint, not a swept threshold. Vetoing both modalities is refused — that is the
plan-B3 abstain, signalled by `R_sys`. IR has no photometric term by design, so IR
is never vetoed.

**`D` deliberately does NOT get veto authority.** Vetoing on `r_frame < 0.5`
(`D > mu_d` OR `b < mu_b`) was run and costs **glare/day 0.2532 -> 0.1435 (-43%)**:
glare pushes `D` past `mu_d` on ~47% of *daylight fit-run* frames where VIS still
scores 0.2626 against IR's 0.0092. §9.4 already showed `D` is not monotone in
capability. A non-monotone signal may down-weight — a soft weight degrades
gracefully when the signal is wrong — but it must not hold a switch. Kept for the
record as `runs/eval/veto_rule_maha.md`.

Protocol note: that rejected variant is **better** on the held-out fog/night cell
(0.0813 vs 0.0792) and was rejected anyway, on daylight fit-run evidence. The
adopted rule is the one that scores worse on pohang01.

**Two costs, not hidden:**
1. **lowlight/day 0.0092 -> 0.0087.** Synthetic lowlight reads p05 = 0, so VIS is
   vetoed on 100% of daylight frames — but VIS still scores 0.0173 there vs IR's
   0.0092, so the veto discards the better stream. Real night has no recoverable
   signal; a synthetic multiplicative darkening does, and p05 cannot tell them apart.
2. **fog/night reaches only 0.0792**, still under ir_only. Fog lifts p05 above `mu_b`
   on 71% of night frames so the veto fires on 29% of them — the residue of p05's
   0.243 fog spoof fraction. The one cell of eight not fully closed; it needs a
   fog-robust statistic, not a different fusion rule, and none of the six in
   `frame_brightness.py` does better.

<details><summary>original note</summary>
**Est. ~1 h, after 0.2.**

On pohang01, gated fusion scores **0.0765** against ir_only's **0.0810** — the gate is
5.6% *worse* than just using IR. Synthetic fog never showed this because there the
gate correctly drove `w_vis` to 0.037. Once 0.2 makes `r_frame` respond to darkness,
verify the weights actually collapse toward IR on pohang01; if they still do not, the
fusion rule needs a floor/ceiling term, not merely a better input signal.

</details>

### 0.4. Dedup-threshold sweep — before any union-label number is quoted
**Est. 1–2 h, CPU. — the answer is probably already visible.** §19.4 measured that
only 12.5% of VIS boxes reach IoU 0.55 with any IR box, and 26.8% reach 0.40. An
IoU≥0.5 merge therefore fails on most true pairs, which is exactly the mechanism
that would re-add the same objects as new ones. Expect the +76.8% to collapse.

Union labels make fusion look far better: the advantage over visible-only goes from
**+5.0%** (VIS-only GT) to **+19.1%** (union GT). But the union adds **14,688 boxes
to 19,133 (+76.8%)**, which is implausible for two annotators labelling the same
scenes. With 5.4 px median registration error on 12–17 px boxes, the IoU≥0.5 merge
very likely fails and re-adds the *same* objects as new ones.

Sweep the merge criterion (IoU 0.3 / 0.4 / 0.5, plus a centre-distance criterion) and
report how +76.8% moves. If it collapses toward +20–30%, the +19.1% is inflated.

### 0.5. Re-sweep WBF `iou_thr` above 0.85 — **DONE 2026-08-19, keeping 0.85**
**Actual: ~30 min.** `scripts/sweep_iou_thr.py`, grid extended to 0.90 / 0.95 / 0.99
under the adopted gate configuration. **The curve turns over — 0.85 is a real
interior optimum, not a boundary artifact.**

| iou_thr | 0.40 | 0.55 | 0.70 | **0.85** | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| tuning mean (pooled) | .1311 | .1318 | .1326 | **.1328** | .1327 | .1320 | .1316 |
| tuning mean (fit runs) | .1526 | .1540 | .1560 | .1579 | **.1584** | .1579 | .1572 |

The two criteria disagree by 0.0001 and 0.0005 in opposite directions — a tie, not
a signal. **Kept 0.85**, which is the argmax under the criterion the original
protocol actually used (pooled ladder mean). The fit-run column is a leakage check
added because the seed-2 ladder contains pohang01 night frames; it returns "no
material effect", not a mandate to switch. The reporting-split comparison also
favours 0.85 but was deliberately NOT used as a reason — that would be selecting
on the report split, which B5-5 forbids.

**§19.4 later explained the whole curve**: higher `iou_thr` means less cross-modal
merging, so the sweep was tuning toward "fuse less", not "fuse better".

### 0.6. Fixed-GT risk–coverage for `R_sys`
**Est. 1–2 h.**

`R_sys` abstain is non-monotone — dropping the 10% least-reliable frames makes clean
mAP *fall* 0.2710 → 0.2209, and only 50% coverage beats full coverage. But that table
compares mAP over different frame subsets, where the GT population changes with
coverage. Redo it against a common denominator (or AURC) before recording abstain as
a negative result. The effect is large enough that it probably survives, but the
measurement is not clean.

### 0.7. Settled — do not re-run

- **IR detection floor (`skip_box_thr`) is a dead end.** Every nonzero value hurt,
  monotonically (0.1374 → 0.1258 at 0.20). The IR over-detection hypothesis is
  **refuted at the fusion level**: those low-confidence boxes cost WBF nothing. Over-
  detection remains a *training* problem — see B5.
- **Temporal smoothing `alpha` does nothing.** 1.0 → 0.1 moved the tuning mean from
  0.1371 to 0.1374. Scope decision D14 (smoothing off) is unrefuted; leave it.

---

## A. Method fixes — no retraining

### A1. σ-weighted WBF — **DONE 2026-08-19; implemented, correct, and a measured no-op**
**Actual: ~3 h.** Full analysis in [handoff §19](handoff-2026-08-19-fusion.md).

`sigma_weighted_fusion()` averages cluster coordinates by `score × w_model / σ²`
**per coordinate** — the minimum-variance estimator, not a heuristic. A
confident-but-blurry box keeps its full vote on *whether* an object is there and
loses its vote on *where* the edges are.

`scripts/smoke_sigma_wbf.py` asserts the σ path reproduces stock
`ensemble_boxes.weighted_boxes_fusion` to **6.3e-08** under constant σ across 300
randomised cases, so the A/B isolates σ rather than two fusion implementations.

**Result: ±0.0001 on every condition.** And σ is *not* uninformative — `u_box`
spans **9.1×** p05→p95 on VIS.

**Diagnosed cause, and it is the important finding.** Inverse-variance averaging
only acts inside a cluster of 2+ boxes. `scripts/diag_cross_modal_iou.py` measures
how often that happens:

| WBF `iou_thr` | VIS boxes with an IR partner | share |
|---|---|---|
| **0.85 (tuned)** | **31 of 29,042** | **0.11%** |
| 0.70 | 917 | 3.16% |
| 0.55 | 3,634 | 12.51% |
| 0.25 | 11,098 | 38.21% |

**At the tuned threshold fusion is almost entirely concatenation.** σ never gets
invoked because there is almost never a cluster to weight. Mechanism confirmed by
re-running at `iou_thr` 0.55 (a probe, not a re-tune): the glare delta grows
+0.0001 → +0.0005, tracking cluster opportunity from a base too small to matter.

**Keep it in.** It is the claim the method makes, it is correct, it costs nothing,
and on a properly registered pair it is the right estimator. Report it as
implemented with a measured null effect and a diagnosed cause — do not quietly
drop it.

### A2. Per-run registration-bias refinement, fit on TRAIN frames — **NOW THE BLOCKER**
**Est. 2 h.** Promoted by A1: with only 0.11% of VIS boxes finding an IR partner at
`iou_thr` 0.85, registration is the precondition for *any* fusion-rule improvement
to be able to do anything at all. 5.4 px median error on 12–17 px boxes makes an
IoU of 0.85 between a VIS box and its true IR counterpart nearly unreachable.

A global bias correction failed run-disjoint validation (clean −0.0% / −0.7%; the
offset estimate itself swung from dx +0.17 to −4.01 px between run pairs), so it is a
**per-run** residual, not a constant. The homography is already per-run — add a
per-run translation term fit on *training* frames, never on val, and it becomes
legitimate. Fog gains of +6.8% and +53.6% did survive held-out, so there is something
real here.

### A3. A better frame-quality signal than Mahalanobis-on-neck-features
**Est. 1–2 days. — DEMOTED back to normal priority.** §0.2's 2 h probe solved the
night failure outright, so this is no longer blocking. It remains the only
candidate fix for the blur/glare counterexample, which brightness cannot touch
(measured: `vis_clean` 62.65 vs `vis_blur_s3` 63.33 mean intensity — blur is
photometrically invisible).

Spearman(D, retention) is only **−0.715** on VIS (−0.991 on IR). Counterexample from
the ladder: `blur_s1` at D=49.5 destroys 60% of mAP while `glare_s3` at D=83.2 costs
23% — lower distance, twice the damage. **No monotone function of D can fit both**, so
a sigmoid in D has a hard ceiling on VIS. Candidates: a small learned frame-quality
head, feature-space entropy, or per-level (P3/P4/P5) distances instead of one pooled
vector.

---

## B. Training — GPU

### B1. Rect training — drop the letterbox padding — **best compute-per-gain**
**Est. 1–2 h re-prep + 4.5 h train.**

The prepared VIS tree is 640×640 with 151 padding rows top and bottom: **only 53% of
the canvas carries image** (640×338 of 409,600 px). At the *same pixel budget*, a
2048×1080-aspect rectangle is 881×465 — **1.38× linear resolution for identical
compute**. This is free resolution, not bought resolution.

Source: `D:/Datasets/Pohang_dataset_full/` (native 2048×1080, same split, same night
filter). IR is 80% efficient already, so its gain is only ~1.12×.

### B2. imgsz 1280 on the full-resolution tree — **biggest single lever**
**Est. 17–22 h.**

VIS median ship is **14 px** at model input, from ~45 native px; 86.9% of ships are
under 32² and 58.3% under 16². At imgsz 1280 the median goes to ~28 px. ~4× the
pixels, and batch must drop from 24 to roughly 6–8 on a 12 GB card, so the per-epoch
282 s becomes ~19 min → ~17 h at 54 epochs, more if it does not early-stop.

Null lever for IR — it is already at native 640×512 (handoff §4.5). Do **not** price
the two modalities together.

### B3. CLAHE IR re-export + retrain
**Est. 1–2 h export + 3.5 h train.**

The percentile experiment was rejected because min–max and percentile are **both
per-frame affine maps**, and the handoff's own conclusion was that *no global
remapping can fix local contrast*. **CLAHE is local** — it is the direct answer to
the failure that was diagnosed, and it has not been tried. Reuse
`export_ir_percentile.py`'s structure, including its min–max round-trip verification
gate (max abs diff 0 over 60 frames) so the only variable is the mapping.

### B4. `yolo26s` → `yolo26m`
**Est. ~7 h.** Phase 1 measured 0.2864 → 0.3061 on VIS (+7%). Known-good, no new code.

### B5. IR ship-only retrain
**Est. 3.5 h.** IR buoy AP is **0.00022** while emitting 29,131 buoy detections
against 596 GT. The model spends capacity on a class it cannot see. Also settles
whether dropping buoys lifts IR ship AP above its current 0.1351.

### B6. Small-object augmentation tuning
**Est. 4.5 h per config.** `scale`, `mosaic`, `copy_paste`. 86.9% of ships are under
32², so the mosaic downscaling range is plausibly hurting.

### B7. Longer patience
**Est. 4.5 h.** `gauss_vis_seed0` peaked at epoch 48 of 68 with patience 20. Some runs
may simply be undertrained.

---

## C. Open decisions carried forward

| # | Decision | Where |
|---|---|---|
| D-1 | Option C mosaic stage — keep or drop | calibration numbers now exist (§6) |
| D-2 | §12.1 parity — re-spec as multi-seed + tolerance | previous handoff §6 |
| D-4 | Thermal crossover fraction | reframed: explain **ship** 0.135, not macro 0.065 |
| D-5 | MC-dropout + deep-ensemble baselines | deferred by Laksh 2026-08-19 |
| D-7 | `r_box` is inert — re-specify or drop | see A1 |
| D-8 | Per-modality scorers/constants — ratify | never pre-registered |
| D-9 | Capability prior in fusion weights — ratify | robust across 4×–150×, but new |

---

## D. Evaluation integrity — before publication

1. **Held-out corruption family.** The gate's τ was fit on a ladder covering 6
   corruption families and is evaluated on 3 of them. Ladder 5, test on the 6th.
2. **Run-disjoint refit of the capability prior** (D-9) — currently fit on clean val
   and evaluated on clean val.
3. **Class-set choice must precede the results.** Dropping buoys moves VIS 0.2580 →
   0.2141 and IR 0.0676 → 0.1351. Picking the framing after seeing that is not
   defensible; report per-class AP always.
4. **Fog severity 2 is "sensor destroyed", not "degraded"** (VIS → 0.0012). The
   interesting middle of the range is severity 1.
5. **The paired subset is not a random sample of val.** pohang01 is 46% of the 2,232
   paired frames vs 18.2% of the full 11,352-frame VIS val, because IR val coverage
   concentrates in pohang00/01. Every paired-subset number inherits a 2.5× over-weight
   on the hardest run. State this wherever a paired number sits next to a Phase 1/2
   number, and never compare the two directly. See §0.1.
6. **Night frames are in val by design, not by accident.** `filter_night_boxes.py` cut
   132k night boxes from *train* only: *"Val/test labels are NEVER touched: they stay
   honest hard cases and the fusion showcase."* Any reviewer asking why VIS scores
   0.0000 on a whole run gets that answer, and it is the right one — but it must be
   stated up front, not discovered by them.
