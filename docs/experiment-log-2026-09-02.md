# Experiment log — 2026-09-02, the ideas queue and what survived it

Sixteen jobs generated from `docs/architecture-ideas-2026-09-01.md`, run as a
parallel queue, plus the adoption gate for the one arm that looked live. What each
asked, what it found, and where the numbers live — including the seven nulls, the
two ideas that came back unmeasured rather than refuted, and the two bugs I found
(one in a probe, one in my own gate script).

**Ground rules held throughout.** Nothing under `runs/cache/`, `runs/derived/`,
`runs/eval/` or `archive/` was overwritten. `Test_1/` untouched. Every new result
went to a new filename. `preset="crossmodal"` and `preset="crossmodal26m"` still
reproduce every published number bit-for-bit (`smoke_crossmodal_gate.py` 14/14;
`smoke_vis_soft_nms.py` check E compares the VIS stream against `load_cache` output
element by element).

**Headline: nothing shipped.** The one arm that passed a screen — VIS soft-NMS —
failed its own pre-registered adoption bar by −0.0004 on one cell. Re-drawing that
cell's corruptions at five more seeds showed the −0.0004 was the *seed*, not the
system (§4.5) — which indicts the gate itself, not just this arm. A second,
draw-averaged bar was pre-registered and run; the day arm passed every cell and the
**night arm failed at −1.03e-5 on 4 of 4 draws**, so it still does not adopt
(§4.6). §4.7 records that the bar I wrote had no magnitude floor, and why I am not
fixing that after the fact.

---

## 0. Summary — every experiment at a glance

| # | experiment | asked | found | verdict |
|---|---|---|---|---|
| I0 | day substrate | can the levers be measured on more than 1,200 frames? | **9,284 day frames, 4 runs**, incl. `pohang04` never used before | instrument built |
| I1a | re-ranker, paired | does IoU-aware re-scoring survive LORO? | best oof **+0.0037** (4 feat, λ 0.30), 2/3 runs won | looked live |
| I1b | re-ranker, day | does it survive 7.7× the frames and a 4th run? | **best arm is λ = 0.00 — do nothing.** Every non-zero λ negative | **dead** |
| I2 | σ as a coordinate correction | can σ predict the *signed* edge error? | screen passes 2/4 edges; end-to-end **−0.0126 at ×0.25** | reject |
| I3a | TTA (4 views) | is view agreement an independent signal? | WBF@0.85σ **+0.0031**, CI spans zero, worst run **−0.0125** | reject (4× cost) |
| I3b | one2many branch | does the discarded head carry anything? | cache is **dead on arrival** — 3.5 bx/fr, 0.000 agreement, AP 0.0000 | **unmeasured** |
| I4 | AP by object scale | is the loss pixels or ordering? | **89.8% of GT is `small`**, median side **12.0 px**, small rec@50 0.778 | resolution is real |
| I5 | night label audit | was `--cut-dark` reversible, and was it right? | cut is frame-level; **82,694 of 120,829 boxes were never flagged** yet went with their frame | reframed |
| I6 | per-class levers | where does the macro metric spend? | buoy headroom **> ship's** on both substrates; `support_gamma` is ship-only | reframed |
| I7 | gate-conditional `cap_ir_scale` | can the gate pick the scale? | **`IR vetoed` = 0.000 on all 11 cells** — arm ≡ constant by construction | **unmeasured** |
| I8 | two-checkpoint ensemble | does a second checkpoint disagree usefully? | matched lift 0.93–1.97×; every merge arm negative | reject |
| I9a | within-modality suppression | merge or suppress the duplicates? | **soft-NMS σ0.5 +0.0025 [+0.0021, +0.0029]**, 3/3 runs | screen passed |
| I9b | merge vs support threshold | is `iou_thr` 0.85 doing anything? | **0.95 (merging off) scores −0.0001** | architecture named |
| G1 | soft-NMS gate, run 1 | does the VIS-stream gain survive fusion? | every cell ≥ 0 — **but σ was selected on TEST** | **invalid** |
| G2 | soft-NMS gate, run 2 | same, at the pre-registered σ = 0.5 | **worst cell −0.0004** (`blur_s3/glare_s2`) | **bar not met — held** |
| G3 | corruption redraw | is that −0.0004 the system, or one draw? | same cell over 6 seeds: **mean +0.0010, sd 0.0011, negative on 2/6**; baseline swings 0.0260–0.0289 | **the bar was noise-limited** |
| G4 | draw-averaged gate | does soft-NMS pass a bar with the draw noise averaged out? | day passes every cell; **night is −1.03e-5 on `blur_s3/glare_s2`, negative on 4/4 draws** | **do not adopt** |
| G5 | metric noise floor | is the macro metric its own noise source? | buoy is 5.3% of boxes and **75% of macro variance**; buoy AP is **exactly 0.0000** on 2 of 11 cells | **gate per class** |
| G6 | paired vs unpaired delta | what is a gate actually able to resolve? | pairing buys **3–16×**; real 2σ floor **0.0014–0.0031**; shipped +0.0106 clears it 6.6× | **margin measured** |

---

## 1. Setup — two substrates, and why the second one exists

Every number before today came from `runs/cache_m/gauss_vis_paired_clean.pkl`:
1,200 day frames drawn from three runs, and drawn *unevenly* — pohang00:836,
pohang02:247, pohang03:117. A leave-one-run-out fold on that substrate holds out
117 frames in the worst case.

`scripts/build_day_substrate.py` (I0, GPU, 347 s) produced
`runs/cache_day/gauss_vis_day_clean.pkl`: **9,284 day frames over four runs** —
pohang00:1672, pohang02:2690, pohang03:2579, **pohang04:2343**. `pohang04` had
never appeared in any measurement in this project.

It is VIS-only. There is no matching IR cache, which matters in §9.1.

The between-run spread on that substrate is the honest scale of a "held-out"
difference, and it is enormous:

| run | frames | actual mAP50-95 | oracle | headroom |
|---|---:|---:|---:|---:|
| pohang00 | 1672 | 0.3758 | 0.4713 | +0.0955 |
| pohang02 | 2690 | 0.3759 | 0.4881 | +0.1122 |
| pohang03 | 2579 | 0.1970 | 0.2856 | +0.0886 |
| pohang04 | 2343 | 0.2030 | 0.3277 | +0.1248 |

A run-to-run gap of **0.179 mAP** on the same detector and the same conditions.
Any arm worth less than that has not been shown to generalise by a fold that
happens to land well. This table is the single most useful thing the queue produced.

---

## 2. I1 — the re-ranker died when it got more data

The 2026-09-01 pilot reported **+0.0419** for a learned IoU-aware re-ranker. §1.3
of that log had already established the pilot reported a fit-set number. This
session re-fitted it properly, twice.

`score = conf^(1−λ) · predIoU^λ`; coordinates are never touched. Folds are
leave-one-run-out, so every frame is scored by a model that never saw its run.

### 2.1 On the paired substrate — it looked alive

`runs/eval/rerank_loro.md`. Baseline 0.3233, oracle ceiling 0.4293 (headroom
+0.1060). Best out-of-fold arm: 4 features, monotone, λ 0.30, **+0.0037**, winning
on 2 of 3 held-out runs.

Small, but positive out of fold, and the fit/held-out split behaved as expected:

| n_feat | λ | Δ on FIT runs | Δ on HELD-OUT run |
|---|---:|---:|---:|
| 4 | 0.50 | +0.0149 | +0.0080 |
| 8 | 0.50 | +0.0403 | +0.0003 |
| 18 | 0.50 | +0.0450 | +0.0030 |

The 18-feature model gains 15× more on data it saw than on data it didn't. That is
the pilot's +0.0419 in one line.

### 2.2 On the day substrate — every non-zero λ is negative

`runs/eval/rerank_loro_day.md`. Baseline 0.2771, oracle 0.4059 (headroom
**+0.1288** — larger than on the paired substrate, so the ceiling did not shrink).

**Best out-of-fold arm: 4 features, λ = 0.00, Δ +0.0000.** Do nothing wins.

| n_feat | monotone | λ 0.10 | λ 0.30 | λ 0.50 | λ 1.00 |
|---|---|---:|---:|---:|---:|
| 4 | yes | −0.0004 | −0.0007 | −0.0023 | −0.0415 |
| 8 | yes | −0.0077 | −0.0107 | −0.0127 | −0.0581 |
| 18 | yes | −0.0101 | −0.0144 | −0.0172 | −0.0737 |

The fit/held-out gap is still there (18 feat, λ 0.50: **+0.0428** fit vs **+0.0070**
held out), so the signal inside a run is real. It just does not transfer across
runs, and with four runs instead of three there is nowhere for a lucky fold to hide.

**This is the important negative result of the session.** More data did not rescue
the arm, it killed it — which means both the pilot's +0.0419 *and* my +0.0037 were
run-selection luck. The +0.1288 oracle headroom is real and is *not reachable by
re-scoring these boxes with these features*.

The feature screen (§3 of that doc) is consistent with this and was never evidence
otherwise: `sigma_mean_norm` has Spearman **−0.630** against IoU-with-GT and `conf`
+0.549, pooled and in-sample. Strong in-sample correlation, zero out-of-fold
transfer. Recorded so the next person does not read the correlation table as a
result.

---

## 3. I9/I3 — the duplicate census, and the one arm that passed a screen

### 3.1 The census

`runs/eval/within_modality.md`. Of 19,894 VIS detections on 1,200 day frames:

| sibling IoU ≥ | boxes with a same-class sibling | share |
|---|---:|---:|
| 0.50 | 10,434 | 0.524 |
| 0.80 | 6,010 | 0.302 |
| 0.90 | 3,858 | 0.194 |

Nearly a fifth of the VIS stream is a near-exact duplicate of another VIS box.
`iou_thr` is 0.85 and `single_passthrough` skips WBF entirely on one-stream frames,
so nothing in the shipped system was removing them.

### 3.2 Merging loses; suppression wins

Every arm below is against the shipped VIS baseline 0.3233, paired bootstrap n=500.

| family | best arm | mAP50 | mAP50-95 | Δ | 95% CI | worst/best run |
|---|---|---:|---:|---:|---:|---:|
| WBF merge | @0.90 plain | 0.7871 | 0.3203 | −0.0030 | [−0.0055, −0.0014] | −0.0115 / +0.0010 |
| hard NMS | @0.90 by conf | 0.7895 | 0.3248 | +0.0016 | [+0.0008, +0.0018] | +0.0011 / +0.0037 |
| **soft-NMS** | **σ 0.5** | **0.7912** | **0.3258** | **+0.0025** | **[+0.0021, +0.0029]** | **+0.0010 / +0.0045** |

Every single WBF merge arm is negative, at every threshold, with and without
σ-weighting — and σ-weighted differs from plain by less than the CI, so the
inverse-variance argument does not apply here either. That is the fourth
independent confirmation that **moving coordinates in this system loses**
(C4 registration, C5 align-and-forbid, I8 §3.4, and now this).

Suppression, which only decays scores, is the only family that gains. Soft-NMS at
σ 0.5 is positive on **all three** held-out runs, and it is the only arm all
session with a CI that clears zero on both ends.

### 3.3 Sorting by `conf/(1+σ)` changes nothing

NMS @0.60/0.70/0.80 scored **identically** to four decimal places whether ranked by
`conf` or by `conf/(1+σ)`. σ does not reorder the duplicates. Consistent with
2026-09-01 F3 and with [[project-redundancy-independence]].

### 3.4 I9b — `iou_thr` 0.85 is not doing anything

`runs/eval/merge_support_split.md`. Merge threshold and support threshold varied
independently for the first time:

| merge iou | support iou / γ | day | Δ day | TUNE | TEST |
|---|---|---:|---:|---:|---:|
| 0.55 | 0.30 / 0.5 | 0.3196 | −0.0089 | 0.3747 | 0.3132 |
| 0.70 | 0.30 / 0.5 | 0.3252 | −0.0033 | 0.3854 | 0.3156 |
| **0.85** | **0.30 / 0.5 (shipped)** | **0.3286** | — | 0.3894 | 0.3173 |
| 0.95 | 0.30 / 0.5 | 0.3284 | **−0.0001** | 0.3888 | 0.3176 |

`merge iou` 0.95 means WBF effectively never clusters anything. It costs
**−0.0001**, and it is *better* than the shipped value on TEST. Merging is worth
nothing.

So the honest name for this architecture is **score-modulated concatenation**: two
detection lists pooled, the cross-modal `support` term re-weighting one of them,
and no coordinate ever combined. That confirms [[project-fusion-mechanism]] from
the parameter side rather than the geometry side — C1 found the streams meet on
0.05% of boxes at IoU 0.85; this finds that removing the merge step costs nothing.

`merge_iou` should become an explicit preset parameter set to "off" rather than
staying an implicit consequence of the value 0.85, so a future detector swap
re-prices it deliberately. Not done in this session.

### 3.5 I3 — TTA is a 4× bill for a CI that spans zero

`runs/eval/tta_o2m.md`. Four views (`id`, `flip`, `s0.8`, `s1.25`), 34.6 bx/fr.

The matched-lift screen (bin by confidence quantile first, per the F1 instrument)
gives `flip` @IoU 0.55 **1.65×** and `s1.25` @0.30 **0.42×** — a signal that is
*below* 1.00× in some bins, i.e. actively anti-correlated once confidence is
controlled for. Raw lift reads 2.4–6.1×; that is confidence's own 4.8× leaking in.

Best arm, WBF@0.85 σ-weighted: **+0.0031, CI [−0.0001, +0.0071], worst run
−0.0125.** Reject: the CI touches zero, one held-out run loses four times what the
mean gains, and it costs 4× inference.

The score-only `support` arms — the form that has worked before — are **exactly
0.0000** at every IoU and γ. View agreement carries nothing the score does not
already have.

---

## 4. The adoption gate, and the selection rule I got wrong

Soft-NMS was measured on the **bare VIS stream** (mAP 0.3233). The system ships
`fused_gated` under `crossmodal26m` (clean day 0.3286), and five of eight benchmark
cells run `single_passthrough`. `scripts/sweep_vis_soft_nms.py` re-measures it where
it would actually live, against the project's standing bar: **at or above the
shipped constant on every cell, day and night.**

### 4.1 Run 1 — passed the bar, failed the methodology

`runs/eval/vis_soft_nms_adoption.md` (811.5 s). Section 2 ran at **σ 0.7**, and
every one of the 11 cells came out at or above shipped (worst cell +0.0000 day and
night).

σ 0.7 was chosen by `max(by_test)` — **the best number on the held-out TEST half.**
That is my own bug, and it is exactly the C8 failure with a new label. C7 used TEST
to *reject* `iou_thr` 0.75 and 0.95; it never used it to pick among survivors. The
364 TEST frames are the only held-out day data this project has, and selecting on
them spends them.

Fixed in `scripts/sweep_vis_soft_nms.py`: `--ship-sigma` defaults to **0.5**, the
value `probe_within_modality.py` measured before this split was ever looked at, with
a guard that raises if that width is negative on TEST. TEST rejects; it does not select.

### 4.2 Run 1's own table shows why the width is not resolvable

| arm | day | Δ day | 95% CI | TUNE | TEST | night |
|---|---:|---:|---:|---:|---:|---:|
| off (shipped) | 0.3286 | — | — | 0.3894 | 0.3173 | 0.0850 |
| σ 0.3 | 0.3290 | +0.0004 | [−0.0012, +0.0021] | **0.3919** | 0.3187 | 0.0850 |
| σ 0.5 | 0.3298 | +0.0012 | [−0.0001, +0.0029] | 0.3915 | 0.3193 | 0.0850 |
| σ 0.7 | 0.3300 | +0.0015 | [+0.0002, +0.0027] | 0.3909 | **0.3196** | 0.0850 |

TUNE prefers 0.3. TEST prefers 0.7. They disagree monotonically across the whole
range, and the total spread is 0.0011. That disagreement is evidence the width is
**not resolvable on 364 frames** — not a licence to trust whichever half is larger.

Night is 0.0850 on every row including "off", to four decimals. Soft-NMS is inert at
night, as it must be: VIS scores 0.0000 there, so there are no duplicate VIS boxes
to decay. Consistent with [[project-maha-night-blindspot]].

### 4.3 Run 2 — at the pre-registered width, the bar is not met

`runs/eval/vis_soft_nms_adoption_v2.md` (778.1 s), σ 0.5:

| cell (vis/ir) | shipped day | soft-NMS day | Δ day | 95% CI | Δ night |
|---|---:|---:|---:|---:|---:|
| clean/clean | 0.3286 | 0.3298 | +0.0012 | [−0.0001, +0.0029] | +0.0000 |
| clean/glare_s2 | 0.3280 | 0.3292 | +0.0012 | [−0.0002, +0.0027] | +0.0000 |
| clean/blur_s2 | 0.3279 | 0.3291 | +0.0012 | [−0.0001, +0.0028] | +0.0000 |
| clean/noise_s2 | 0.3252 | 0.3278 | **+0.0026** | [+0.0012, +0.0040] | +0.0000 |
| clean/fog_s2 | 0.3231 | 0.3255 | **+0.0024** | [+0.0010, +0.0038] | +0.0000 |
| blur_s3/clean | 0.0269 | 0.0270 | +0.0002 | [−0.0004, +0.0027] | +0.0000 |
| noise_s2/clean | 0.0096 | 0.0096 | +0.0000 | [+0.0000, +0.0000] | +0.0000 |
| rain_s2/clean | 0.0858 | 0.0873 | +0.0015 | [+0.0011, +0.0021] | +0.0000 |
| fog/clean | 0.0482 | 0.0490 | +0.0008 | [+0.0003, +0.0018] | +0.0000 |
| lowlight/glare_s2 | 0.0235 | 0.0235 | +0.0000 | [−0.0001, +0.0000] | +0.0000 |
| **blur_s3/glare_s2** | 0.0268 | 0.0264 | **−0.0004** | [−0.0009, +0.0023] | −0.0000 |

**Worst cell: day −0.0004.** The pre-registered bar is "at or above on every cell".
It is not met.

### 4.4 The call: held, not adopted

The tempting move is to declare −0.0004 noise. Its CI is [−0.0009, +0.0023] —
*centred positive* while the point estimate is negative, which is itself a sign the
cell cannot resolve ±0.001 — and the same cell scored **+0.0004** at σ 0.7, so the
sign flips with the width. On a cell where the system scores 0.0268 in the first
place, this is very likely nothing.

That argument is exactly the one A2 refused for `cap_ir_scale` and the one D2 wishes
had been applied to the veil veto. The bar was written into the script before the
run. Loosening it *after* seeing which side the number fell on is how the veil veto
shipped a −0.0632 regression. **So: not adopted.**

What is in the tree, and what is not:

- `soft_nms_record` / `soft_nms_records` in `src/uqfusion/eval/irdedup.py` — placed
  next to `nms_record`, whose docstring already frames that module as *"a
  DETECTOR-side post-process on one stream, not a fusion operation."* Soft-NMS is
  the same kind of thing and belongs in the same place. Unlike the probe's version
  it re-indexes `sigma_ltrb`, which is required because `compute_reliability`
  indexes σ against `boxes_xyxy`.
- `vis_soft_nms` parameter in `load_context`, applied at the same point and for the
  same reason as `ir_nms` on the IR side.
- `preset="crossmodal26m_snms"` — a **separate** preset defaulting σ 0.5. It exists,
  it is measured, and it is not the default.
- **`preset="crossmodal26m"` is unchanged and remains the shipped system.** The
  +0.0106 headline and all of `final_26m_grid_v2.md` still reproduce.

Two ways forward, both legitimate, neither taken unilaterally:

1. **Buy a decision.** The bar cannot be adjudicated on 1,200 paired frames. Re-run
   the gate on a corrupted-condition day substrate at `pohang04` scale; a −0.0004
   either becomes a real cost or vanishes.
   **This option does not exist — see §10 item 6.** IR val is 2,234 frames total and
   `pohang04` has no IR at all; the 1,200 paired day frames are every paired day
   frame there is. §4.5 takes the axis that *was* available instead.
2. **Ship the narrow form.** Adopt soft-NMS only where it is unambiguous — the
   `single_passthrough` frames, and the clean-VIS cells where it gains +0.0012 to
   +0.0026 — and leave both-degraded cells alone. That is a new pre-registration,
   not a loosened one, and needs its own gate.

### 4.5 The redraw — the bar was measuring the seed

`runs/eval/snms_cell_redraw.md`. §4.4 named two ways forward. Before either, one
question had to be answered: is −0.0004 a property of *soft-NMS*, or of *one
corruption draw*? Every gate this project has run rests on a single realisation of
each corruption — blur(s3, seed 1) over VIS, glare(s2, seed 7) over IR. The
bootstrap resamples **frames**, so it is structurally blind to this.

Re-drawing that cell's two corruptions at five further seeds, kind and severity
fixed:

| vis/ir seed | shipped day | soft-NMS day | delta |
|---|---:|---:|---:|
| 1/7 (shipped) | 0.0268 | 0.0264 | −0.0004 |
| 901/911 | 0.0277 | 0.0298 | +0.0021 |
| 902/912 | 0.0260 | 0.0256 | −0.0005 |
| 903/913 | 0.0278 | 0.0291 | +0.0012 |
| 904/914 | 0.0289 | 0.0301 | +0.0012 |
| 905/915 | 0.0264 | 0.0285 | +0.0020 |

Mean +0.0010, sd 0.0011, negative on 2 of 6. The `clean/clean` control is **+0.0012
on all six rows, identical to four decimals** — no corruption, no movement, so the
spread above is the seed and nothing else.

The decisive number is not the mean. It is that the **shipped baseline itself swings
0.0260–0.0289** across draws — roughly seven times the −0.0004 that tripped the bar.

**This is a finding about the instrument, not about soft-NMS.** A single-draw
every-cell test applied to cells scoring ~0.027 cannot resolve ±0.001, and will
accept and reject arms by coin flip *in both directions*. That indicts every
adoption decision this project has made on the low-scoring corrupted cells — not
only this one.

### 4.6 The draw-averaged gate — and it still does not adopt

So the bar was re-specified, pre-registered at `docs/prereg-snms-draw-averaged-gate.md`
and committed at `1fbf735` **before the run existed**: 4 draws (shipped + seeds
901/902/903), σ held at 0.5, adopt iff the draw-averaged delta is ≥ 0 on every cell
day *and* night, TEST rejects but never selects, and sd / neg-counts / per-draw
tables / bootstrap CIs named in advance as **diagnostics, not decision inputs**.

`runs/eval/snms_gate_draw_avg.md`, 6,858 s, 21 caches rebuilt.

| cell (vis/ir) | mean delta day | sd | neg | mean delta night |
|---|---:|---:|---:|---:|
| clean/clean | +0.0012 | 0.0000 | 0/4 | +0.0000 |
| clean/glare_s2 | +0.0013 | 0.0001 | 0/4 | +0.0000 |
| clean/blur_s2 | +0.0012 | 0.0000 | 0/4 | +0.0000 |
| clean/noise_s2 | +0.0026 | 0.0001 | 0/4 | +0.0000 |
| clean/fog_s2 | +0.0026 | 0.0002 | 0/4 | +0.0000 |
| blur_s3/clean | +0.0007 | 0.0011 | 1/4 | +0.0000 |
| noise_s2/clean | +0.0000 | 0.0000 | 0/4 | +0.0000 |
| rain_s2/clean | +0.0014 | 0.0003 | 0/4 | +0.0000 |
| fog/clean | +0.0006 | 0.0008 | 1/4 | +0.0000 |
| lowlight/glare_s2 | +0.0000 | 0.0000 | 0/4 | +0.0000 |
| blur_s3/glare_s2 | +0.0006 | 0.0013 | 2/4 | **−0.0000** |

The redraw was right about the day arm: `blur_s3/glare_s2` averages **+0.0006**, and
the −0.0004 was one draw of a cell whose sd is 0.0013. TEST is +0.0020 on clean, so
rule 5 does not reject. **Every day cell passes.**

The arm fails on **night**, and it fails somewhere I was not looking.

| draw | night shipped | night soft-NMS | delta |
|---|---:|---:|---:|
| 1/7 | 0.0479743640 | 0.0479675248 | −6.84e-06 |
| 901/911 | 0.0459421531 | 0.0459334799 | −8.67e-06 |
| 902/912 | 0.0453137984 | 0.0453010358 | −1.28e-05 |
| 903/913 | 0.0464279472 | 0.0464149457 | −1.30e-05 |

Mean **−1.03e-5**, negative on **4 of 4**, over 1,032 night frames.

This is a different animal from the day failure. The day −0.0004 flipped sign across
draws; this does not. It is 40× smaller in magnitude and *perfectly sign-stable* —
the corruption seed moves the night baseline by 2.6e-3 between draws, yet the delta
stays negative every time. That is a mechanical effect, not noise: VIS is near-dead
at night, and soft-NMS decays the few VIS scores that survive into the fused list.
Averaging cannot rescue it, because there is nothing random to average.

**DO NOT ADOPT.** `vis_soft_nms` stays off; `crossmodal26m` remains the shipped
preset, `crossmodal26m_snms` remains available and measured.

### 4.7 What I got wrong writing the bar, and what I am not doing about it

Two things to put on the record, in the right order.

**First, the honest verdict stands.** The rule was fixed and committed before the
number existed. It says night ≥ 0 on every cell. Night is −1.03e-5 on one cell, on
every draw. Arguing now that 1e-5 is *too small to count* is the identical move
§4.4 refused and the veil veto made — loosening a bar after seeing which side the
number fell on. The answer is no.

**Second, the bar was badly written, and I wrote it.** It has no magnitude floor. A
1e-5 sign-stable difference and a 1e-2 regression fail it identically, which is not
a bar that expresses anything anyone believes. I wrote it with *day* noise in mind
— the whole document argues about ±0.001 on cells scoring 0.027 — and never asked
what the night arm would do, where VIS contributes almost nothing and the delta is
consequently deterministic and tiny.

The correct fix is an **equivalence margin**: a band around zero inside which a cell
counts as unchanged, fixed in advance from the measured draw-to-draw sd of that
cell. Under any margin wider than 1e-5 — and the day sd on the same cell is 1.3e-3,
*two orders of magnitude* larger — this arm passes.

I am not applying that retroactively. A margin invented after seeing that it flips
this verdict is not a pre-registration, it is a rationalisation with a formula
attached. If soft-NMS is worth re-gating, it is worth a third pre-registration
written before the fourth run, and this section is the evidence for what that
document should contain.

### 4.8 Where the gain actually lands

The two largest cells are `clean/noise_s2` (+0.0026) and `clean/fog_s2` (+0.0024) —
**the cells where IR is destroyed and VIS carries the whole load**. That is the
mechanism, and it is consistent: soft-NMS cleans up the VIS stream, and it matters
most exactly when the VIS stream is all there is. It is a detector post-process that
shows up in the fusion metric, not a fusion improvement.

### 4.9 What the bar should have been — the noise floor, measured

§4.7 said the missing piece was an equivalence margin sized from real variance,
and refused to invent one after the fact. This measures it, so a future
pre-registration has a number to quote instead of a judgement call.
`runs/eval/metric_noise_floor.md` and `runs/eval/delta_noise_floor.md`, shipped
system, day frames only (buoy has no night GT).

**The metric is macro over two very unequal classes.** Day GT is ship **10,663**
boxes and buoy **600** — buoy is 5.3% of the boxes carrying **50%** of the number.

| clean cell | sd ship | sd buoy | buoy/ship | buoy share of macro variance |
|---|---:|---:|---:|---:|
| clean/clean | 0.0058 | 0.0100 | 1.7× | 75% |

Three structural facts follow, and none of them are statistical:

* **Buoy AP is 0.2780 with sd 0.0000 on all five `clean/*` cells.** Those cells
  corrupt IR only, and IR is `nc=1`, so corrupting IR cannot move buoy at all.
  Half the metric is inert on five of eleven cells.
* **Buoy AP is exactly 0.0000 on `noise_s2/clean` and `lowlight/glare_s2`.** Macro
  there is precisely `ship/2`. Any gain on those cells is a ship-only effect
  wearing a macro disguise.
* Both cells also have a *measured delta floor of 0.0000* — they carry no
  information for any gate and should be stated as such rather than counted as
  two of eleven passing cells.

**And a correction to my own first pass.** `metric_noise_floor.md` §3 reported
`2 × sd(AP)` as a "smallest resolvable effect" and landed on **0.0120** for the
clean cell. That is the uncertainty in the metric's *level* and it is the wrong
yardstick: a gate scores both arms **on the same frames**, so the resample is
common and almost all of it cancels. Taken literally it would have put the shipped
**+0.0106 crossmodal gate inside the noise**, which is false. Measured directly:

| cell | sd unpaired | sd paired | pairing buys |
|---|---:|---:|---:|
| clean/clean | 0.0087 | 0.0008 | 11× |
| rain_s2/clean | 0.0043 | 0.0003 | 16× |
| blur_s3/glare_s2 | 0.0023 | 0.0009 | 3× |

The real per-cell floor, corruption draw and paired bootstrap in quadrature:

| cell (vis/ir) | sd draw | sd paired boot | 2×total | binding |
|---|---:|---:|---:|---|
| clean/clean | 0.0000 | 0.0008 | 0.0016 | frames |
| clean/glare_s2 | 0.0001 | 0.0007 | 0.0014 | frames |
| clean/blur_s2 | 0.0000 | 0.0007 | 0.0014 | frames |
| clean/noise_s2 | 0.0001 | 0.0008 | 0.0016 | frames |
| clean/fog_s2 | 0.0002 | 0.0008 | 0.0016 | frames |
| blur_s3/clean | 0.0011 | 0.0008 | 0.0027 | draw |
| noise_s2/clean | 0.0000 | 0.0000 | 0.0000 | — |
| rain_s2/clean | 0.0003 | 0.0003 | 0.0008 | frames |
| fog/clean | 0.0008 | 0.0003 | 0.0017 | draw |
| lowlight/glare_s2 | 0.0000 | 0.0000 | 0.0000 | — |
| blur_s3/glare_s2 | 0.0013 | 0.0009 | 0.0031 | draw |

What this settles:

1. **The shipped +0.0106 clears its cell's floor by 6.6×.** The crossmodal gate is
   not in question and never was.
2. **The −0.0004 that failed run 2 sat 8× inside its own cell's floor** (0.0031).
   §4.5 argued that from sign instability; this is the same conclusion with a
   number attached.
3. **Where to spend is now cell-dependent.** On the five clean cells the *frames*
   bind, not the draw — and §10 item 6 shows there are no more paired day frames
   in existence, so those cells are permanently at their floor. On the
   both-degraded cells the *draw* binds, and draws are cheap. More draws is the
   right purchase on exactly the cells §4.5 was about.
4. **The margin a third pre-registration should quote** is the `2×total` column,
   fixed per cell in advance. It is between 0.0014 and 0.0031 on the cells that
   carry information — two to three orders of magnitude above the −1.03e-5 night
   delta that rejected soft-NMS.

That last point is stated as a fact about the instrument, **not** as grounds to
re-open §4.7. The rejection stands; what changes is that the next gate has a
defensible margin to pre-register instead of an implicit zero.

---

---

## 5. I2 — σ knows how far, not which way

`runs/eval/sigma_residual_day.md`, 108,119 detections matched to GT at IoU ≥ 0.5
over the 9,284-frame substrate.

The head is roughly calibrated in magnitude, not just in ordering — mean σ vs mean
|err| ratios of 0.78 / 0.58 / 0.58 / 1.06 across the four edges.

The screen asks the only question that matters for a correction: is the **signed**
residual predictable out of fold?

| edge | oof R² (signed) | oof R² (\|resid\|) | verdict |
|---|---:|---:|---|
| x1 | −0.0268 | −0.1281 | dead |
| y1 | **+0.0376** | +0.3286 | live |
| x2 | −0.2770 | +0.0129 | dead |
| y2 | **+0.0986** | −0.0612 | live |

Screen verdict PASS (2/4). Applied end to end with leave-one-run-out:

| arm | mAP50-95 | Δ |
|---|---:|---:|
| baseline | 0.2771 | — |
| ridge ×0.25 | 0.2646 | −0.0126 |
| ridge ×0.5 | 0.2444 | −0.0327 |
| ridge ×1 | 0.1976 | −0.0795 |

Monotone in the wrong direction: the more of the correction you apply, the worse it
gets. Two edges out of four is not enough — moving y1 and y2 while x1 and x2 stay
put deforms the box. Reject, on both substrates, and this is the **fifth**
confirmation that coordinate-moving loses here.

---

## 6. I6 and I4 — the metric spends where no lever reaches

### 6.1 Buoy is 5% of the data and 50% of the number

`runs/eval/per_class_levers.md` and §2 of `runs/eval/oracle_headroom_day.md`:

| substrate | class | n_gt | share of GT | AP50-95 | oracle | headroom |
|---|---|---:|---:|---:|---:|---:|
| paired | ship | 10,663 | 0.947 | 0.3686 | 0.4653 | +0.0968 |
| paired | buoy | 600 | **0.053** | 0.2780 | 0.3932 | **+0.1152** |
| day | ship | 88,919 | 0.919 | 0.3512 | 0.4624 | +0.1112 |
| day | buoy | 7,871 | **0.081** | 0.2031 | 0.3495 | **+0.1464** |

Buoy headroom exceeds ship's on **both** substrates. Oracle-re-ranking one class
and leaving the other alone: ship-only **+0.0484**, buoy-only **+0.0576**. Every
lever in this project has been fitted pooled, which means fitted on ship's 10,663
boxes and charged to the class that carries half the metric.

### 6.2 `support_gamma` is a ship-only lever wearing a global name

| arm | macro | Δ | AP ship | AP buoy |
|---|---:|---:|---:|---:|
| γ 0 | 0.3241 | −0.0045 | 0.3702 | **0.2780** |
| γ 0.5 (adopted) | 0.3286 | — | 0.3792 | **0.2780** |
| γ 1 | 0.3288 | +0.0002 | 0.3796 | **0.2780** |
| γ 2 | 0.3285 | −0.0001 | 0.3790 | **0.2780** |

Buoy AP is **exactly 0.2780** at every γ. IR is `nc=1`, so a buoy box can never have
cross-modal support — the term is fitted on one class and named as if it were
global. Extends [[project-detector-scale-null]]. The right form is per class.

### 6.3 I4 — the loss is pixels, and re-ranking cannot reach it

`runs/eval/ap_by_size.md`. GT side length **median 12.0 px** on the 640 canvas; area
p5 = 34 px², p50 = 144 px².

| class | size | n_gt | share | AP50-95 | headroom | **rec@50** |
|---|---|---:|---:|---:|---:|---:|
| ship | small | 9,512 | 0.898 | **0.3382** | +0.1014 | **0.778** |
| ship | medium | 1,021 | 0.091 | 0.6229 | +0.0682 | 0.897 |
| ship | large | 130 | 0.012 | 0.5092 | +0.1057 | 0.923 |
| buoy | small | 600 | — | 0.2780 | +0.1152 | 0.878 |

**89.8% of the GT is `small`**, and small ship AP is 0.3382 against medium's 0.6229.
`rec@50` 0.778 is a **ceiling**: 22% of small GT is not covered by any detection at
any confidence. No re-ranker can retrieve a box that was never proposed — which is
the mechanical reason §2 found the +0.1288 oracle headroom unreachable.

This is the screen `i4_fullres_prep` was gated on, and it says go. Not launched:
re-prepping 127k images at full resolution is hours of GPU and tens of GB, and it
should be a deliberate decision rather than a queue side-effect.

---

## 7. I5 — the night filter deleted more than it flagged

`runs/eval/night_restore_audit.md`. Writes no labels, launches no training.

[[project-visfilter-night-cut]] records `filter_night_boxes.py --cut-dark
pohang01:100` as removing 132,688 boxes from 17,502 frames, with 1,311 brighter
frames keeping their labels. The audit adds what happened *inside* the frames it
touched:

| run | files | boxes before | after | dropped | drop rate | emptied frames |
|---|---:|---:|---:|---:|---:|---:|
| pohang01 | 17,502 | 132,688 | 0 | 132,688 | **1.000** | **17,502** |

**Read that 1.000 carefully.** `audit_night_restore.py` enumerates
`*.pre_visfilter` backups, i.e. only the files the filter modified, so the row
cannot say anything about the 1,311 frames it left alone. What it does say is that
`--cut-dark` is a **frame-level** decision: once a frame's content-median luminance
fell below 100, *every* box in it went, whatever that individual box looked like.

That is the gap. The filter's own **per-box** scores exist in
`runs/visfilter/box_scores.csv`, and flagging required *all three* of intensity,
gradient and local contrast to fail. Only **38,135** of the 120,829 scored boxes
were flagged. **82,694 boxes were scored, would not have been flagged on their own
merits, and were deleted anyway** because of the frame they sat in.

And the two populations overlap on every axis:

| axis | flagged median | flagged p90 | kept median | kept p10 |
|---|---:|---:|---:|---:|
| box_mean | 2.27 | **4.86** | 5.89 | **3.18** |
| grad | 5.60 | 7.55 | 16.81 | 9.02 |
| contrast | 1.09 | **2.95** | 2.35 | **0.37** |

Flagged p90 (4.86) sits well above kept p10 (3.18) on `box_mean`, and the contrast
distributions cross entirely. The decision rule's "cleanly separated → the filter
was right" branch does not fire.

The stake: **2,068 night val frames carrying 16,179 GT boxes**, which VIS currently
recovers **0.0000** of. That is the number the night half of every benchmark cell is
scored against, and no cross-modal lever can touch it — the training labels for it
were deleted. All 17,502 `.pre_visfilter` backups exist, so `--restore` is available.

This does not say the filter was wrong — it says the deletion was **broader than
the evidence supported**, and the 82,694 unflagged boxes are the recoverable
population. The sound next move is to restore above a threshold **as ignore-regions,
not as positives**, and fine-tune. Held: it is a training decision, not a queue job.

---

## 8. I7 and I8 — one null, one non-measurement

### 8.1 I8 — the second checkpoint is closed

`runs/eval/checkpoint_ensemble.md`. A = `gauss_vis_seed0` (0.3233), B = its
fine-tune (0.3144).

Matched lift of "B also fires": **0.93× / 1.00× / 1.55× / 1.97×** at IoU
0.10/0.30/0.55/0.75. Raw lift reads 4.2–6.8×, which is confidence's own 4.8×
(47% of VIS boxes sit below conf 0.05). Against the F1 reference points —
cross-modal 2.08×, temporal 1.00× — this is temporal-support territory.

Every arm confirms it: concat A+B **−0.0949**, WBF@0.55 −0.0021, WBF@0.85σ −0.0033,
and the score-only `support` arm — the only cross-modal form that has ever worked —
lands at **+0.0001**. Two checkpoints of the same model on the same data are not
independent. **Closed.**

### 8.2 I7 — the gated `cap_ir_scale` was never actually measured

`runs/eval/cap_ir_gated.md`, 1,787 s. Section 1 reproduces A2's tie cleanly: ×16
beats ×4 on every clean-VIS cell (+0.0006 on clean/clean) and loses on every
degraded-VIS cell (`blur_s3/clean` 0.0269 → 0.0266).

Section 2 gates the scale on the gate's IR-health flag. **`IR vetoed` = 0.000 on all
eleven cells.** The gated arm is therefore identical to a ×16 constant by
construction, and every delta in that table is the ×16 vs ×4 comparison relabelled.

The doc's own decision rule anticipated this: the conditioning signal is wrong, not
the idea. The IR-health veto does not fire on this benchmark — [[project-ir-night-switch-safety]]
records that IR is uncorrupted in every cell — so the right conditioner is `ir_d2`,
the multivariate health term, not the binary veto. **Unmeasured, not refuted.**

### 8.3 I3b — the one2many cache is broken

`runs/cache_o2m/gauss_vis_paired_clean.pkl`: **3.5 boxes/frame** (the o2o cache has
16.6), `sigma_valid` False, agreement with base **0.000 at every IoU**, and every
arm that uses its coordinates scores **AP 0.0000**.

A one2many head produces *more* boxes than one2one, not five times fewer. This is a
coordinate-handling bug in `scripts/build_tta_o2m.py`, not a property of the branch.
The o2m rows in `tta_o2m.md` measure my bug. **Unmeasured, not refuted.**

---

## 9. Two bugs

### 9.1 `probe_oracle_headroom.py` — silent cross-substrate IR misalignment

`i1_oracle_day` crashed with `IndexError: list index out of range` at line 63. The
job passed `--vis-cache runs/cache_day/...` but no `--ir-cache`, so `ir_path` fell
back to the `--cache-dir` default `runs/cache_m/gauss_ir_paired_clean.pkl` — a real,
loadable, **completely unrelated** 2,232-frame cache. `have_ir` was True and the
union-recall loop indexed a 9,284-frame VIS list against it.

**The crash was the good outcome.** Had the day substrate been ≤ 2,232 frames, it
would have computed cross-modal union recall between IR boxes and VIS frames *from
different images* and printed a plausible number. Fixed with a length-alignment
guard that disables the section and states the reason in the output:

> Skipped: `runs\cache_m\gauss_ir_paired_clean.pkl` holds 2232 frames against this
> substrate's 9284 — not index-aligned, so no IR box can be attributed to a VIS
> frame.

Two failed patch attempts on the way (a multi-line `assert old2 in s` mismatch, then
a `split`/`join` that wrote a literal backslash-n as a real newline and produced
`SyntaxError: unterminated string literal`). Third attempt clean; probe reran in 6.4 s.

### 9.2 `sweep_vis_soft_nms.py` — I selected on the held-out half

Covered in §4.1. The line was
`best = max(sorted(by_test, reverse=True), key=lambda s: by_test[s])`. It converts
the project's only held-out day data into a selection set. Caught by reading my own
output rather than by any test — which is the argument for the `--ship-sigma`
guard now in the script, and for pre-registering widths in general.

Both bugs share a shape: **a default that silently does something plausible.** A
`--ir-cache` default that points at whatever is in `--cache-dir`; a selection rule
that reads whichever column is available. Neither would have raised.

---

## 10. Corrections to things I had stated

1. **"The learned re-ranker is worth +0.0037 out of fold."** Withdrawn. That was
   three runs; on four it is **+0.0000 at best**, and every non-zero λ is negative.
   §2.2.
2. **"The gate passed on every cell."** True of run 1 only, and run 1's σ was
   selected on TEST. At the pre-registered width one cell is −0.0004. §4.
3. **"The night filter's drop rate was 1.000 across pohang01."** Mine, written
   earlier today and corrected in §7 before this log shipped. The audit only
   enumerates files that *have* a backup, so 1.000 means "every frame the filter
   touched was fully emptied", not "every pohang01 frame". 1,311 brighter frames
   kept their labels, exactly as the manifest always said. The real finding is the
   frame-level/box-level gap: **82,694 deleted boxes would not have been flagged
   on their own scores.**
4. **The o2m rows in `tta_o2m.md` are not a measurement of the one2many branch.**
   They measure a build bug. §8.3.
5. **The gated `cap_ir_scale` table is not a measurement of gating.** `IR vetoed` is
   0.000 everywhere. §8.2.
6. **"Re-run the gate on a day substrate at `pohang04` scale."** Mine, §4.4, and
   impossible. I wrote it without checking IR availability. Measured: VIS val is
   11,352 frames (pohang00 1672, 01 2068, 02 2690, 03 2579, 04 2343) but **IR val is
   2,234** (pohang00 836, 01 1034, 02 247, 03 117) and **`pohang04` has no IR at
   all**. The day half of IR is 836+247+117 = **exactly the 1,200 paired frames the
   gate already uses.** There is no wider fused substrate to move to. §4.5 takes the
   corruption-draw axis instead, which was the un-measured one all along.
7. **"The bar was tripped by draw noise, so the arm is fine."** Half right, and I
   should not have implied the rest. §4.5 is correct about the *day* cell. The
   draw-averaged gate then failed on **night**, at −1.03e-5 on 4 of 4 draws — a
   sign-stable effect that averaging cannot touch. §4.6.

---

## 11. Open

1. **The soft-NMS decision** (§4.6) — **closed: do not adopt.** Failed a
   draw-averaged, pre-registered bar on the night arm at −1.03e-5, 4/4 draws. The
   code stays in the tree behind `crossmodal26m_snms`; the shipped preset has not
   moved. Re-opening it requires a *third* pre-registration with an explicit
   equivalence margin, written before the run — §4.7 says what it should contain.
2. **Re-price the inherited constants** (§4.5, §4.9). `cap_ir_scale` ×4, the
   veil-veto repair and `iou_thr` 0.85 were decided on single-draw corrupted cells.
   §4.9 now supplies the per-cell margin to judge them against, and
   `scripts/gate_snms_draw_avg.py` already does the draw loop. **The gate itself is
   the finding** (§4.5). A single corruption draw cannot
   resolve ±0.001 on cells scoring ~0.027, and every adoption decision this project
   made on those cells was taken with that instrument. `cap_ir_scale` ×4, the veil
   veto repair and `iou_thr` 0.85 were all decided under it. Some turned on margins
   far larger than the noise; **which ones did not is unmeasured.** This is now the
   highest-value open item and it is cheap: `scripts/gate_snms_draw_avg.py` already
   does the draw loop.
2. **I4 full-resolution retrain** — the screen says go (§6.3). Not launched; hours of
   GPU and tens of GB.
3. **I5 night restore** — the audit says the deletion overreached (§7). Restoring
   82,694 boxes as ignore-regions and fine-tuning is the next move, and it is a
   training decision.
4. **Per-class levers** — `support_gamma` is provably ship-only (§6.2). Splitting it
   per class is cheap and untried.
5. **`merge_iou` as an explicit "off"** (§3.4) — naming what the architecture already
   does. Cheap, untried.
6. **I7 re-conditioned on `ir_d2`** rather than the binary veto (§8.2).
7. **I3b — fix `build_tta_o2m.py`** before the one2many branch can be called anything.
8. **The +0.1288 oracle headroom is still open and still unreached.** §2.2 closes the
   *feature-based re-ranking* route to it. §6.3 says the binding constraint is recall
   on 12-px objects, which is a detector/resolution problem.
9. Still open from 2026-09-01: the authority bound remains unpriced; `cap_ir_scale` ×4
   has still not been re-selected under the run-disjoint discipline; the learned gate
   is still in no headline table.

---

## 12. File index

**New code.** `scripts/_ideas_common.py`, `scripts/run_ideas_queue.py`,
`scripts/build_day_substrate.py`, `scripts/build_tta_o2m.py`,
`scripts/fit_rerank.py`, `scripts/probe_oracle_headroom.py`,
`scripts/probe_sigma_residual.py`, `scripts/probe_tta_o2m.py`,
`scripts/probe_within_modality.py`, `scripts/probe_checkpoint_ensemble.py`,
`scripts/probe_ap_by_size.py`, `scripts/sweep_per_class.py`,
`scripts/sweep_cap_ir_gated.py`, `scripts/sweep_merge_support_split.py`,
`scripts/sweep_vis_soft_nms.py`, `scripts/smoke_vis_soft_nms.py`,
`scripts/audit_night_restore.py`, `scripts/redraw_snms_cell.py`,
`scripts/gate_snms_draw_avg.py`, `scripts/probe_metric_noise_floor.py`,
`scripts/probe_delta_noise_floor.py`.

**Pre-registration.** `docs/prereg-snms-draw-averaged-gate.md`, committed at
`1fbf735` before `snms_gate_draw_avg.md` existed.

**Modified.** `src/uqfusion/eval/irdedup.py` (+`soft_nms_record`,
`soft_nms_records`); `src/uqfusion/eval/ctx.py` (+`vis_soft_nms`,
+`preset="crossmodal26m_snms"`). 82 insertions, 1 deletion.

**New results** (all under `runs/eval/`): `rerank_loro`, `rerank_loro_day`,
`oracle_headroom`, `oracle_headroom_day`, `sigma_residual`, `sigma_residual_day`,
`tta_o2m`, `within_modality`, `merge_support_split`, `checkpoint_ensemble`,
`cap_ir_gated`, `per_class_levers`, `ap_by_size`, `night_restore_audit`,
`vis_soft_nms_adoption`, `vis_soft_nms_adoption_v2`, `snms_cell_redraw`,
`snms_gate_draw_avg`, `metric_noise_floor`, `delta_noise_floor`.

**New caches.** `runs/cache_day/` (I0 substrate, 9,284 day frames),
`runs/cache_tta/` (4 views), `runs/cache_o2m/` (broken — see §8.3),
`runs/cache_m_draw{901..905}/` (corruption redraws; uncorrupted members hard-linked
from `runs/cache_m`, which is never written).

**Queue.** `runs/queue_ideas/` — `run_console.log`, `state.json`, `logs/` (16 jobs,
15 succeeded, `i1_oracle_day` failed then fixed and rerun).

**Companion.** `docs/architecture-ideas-2026-09-01.md` — the argued list these jobs
were generated from.

**Commits.** `44274a3` the ideas queue and the two source edits; `1fbf735` the
pre-registration and `redraw_snms_cell.py`, deliberately committed *before* the
draw-averaged gate ran; then this log and `gate_snms_draw_avg.py`.
