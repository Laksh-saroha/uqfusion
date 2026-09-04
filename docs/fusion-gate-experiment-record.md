# Fusion gate — complete experiment record

**Scope.** Everything attempted on the uncertainty gate and decision-level fusion
during the 2026-08-19 working session: what was built, what was measured, what was
tried and rejected, what turned out to be wrong, and what should be tested next.

This is deliberately a *record*, not a summary. Rejected alternatives are kept with
their numbers, because the main risk to this project is re-running a dead end or
re-adopting a variant that was already measured and beaten. Where a claim was made
and later refuted, both are here.

Companion documents:

- [`handoff-2026-08-19-fusion.md`](handoff-2026-08-19-fusion.md) — the primary
  technical account, §14–§19. Authoritative numbers live in **§18.2**.
- [`TODO-improvements.md`](TODO-improvements.md) — what remains, with measured
  time estimates.

---

## 0. The system as it stands

| Component | Setting | Where it came from |
|---|---|---|
| Detector | `yolo26s` Gaussian σ head, imgsz 640, conf 0.001 | Phase 2 |
| Frame OOD score | Mahalanobis on 896-d pooled neck features, Ledoit–Wolf | §6.4 |
| Reliability constants | **D-6 ladder fit**: VIS `mu_d`=71.07 `tau`=37.21 `lam`=1.0239; IR `mu_d`=72.52 `tau`=25.74 `lam`=0.7111 | `runs/eval/reliability_constants.json` |
| Photometric term | `p05` of content rows, **`mu_b`=10.500 `tau_b`=2.625** (margin rule), combined by `min` | §16.4e |
| Hard veto | modality excluded when `r_bright < 0.5` (⟺ `b < mu_b`) | §17.2 |
| Capability prior | VIS **0.2580** / IR **0.0206**, computed on this evaluation | §18.1 |
| WBF | `iou_thr` **0.85**, `skip_box_thr` 0.0 | §0.5 |
| Temporal smoothing | `alpha` 0.5 in analysis, 1.0 in config — inert either way | §0.7 |
| σ-weighted WBF | implemented, available, measured null on this data | §19 |

Reproduce the headline table:

```bash
.venv/Scripts/python.exe scripts/run_fusion_eval.py --capability-weighted --iou-thr 0.85 --constants runs/eval/reliability_constants.json --brightness-constants runs/eval/brightness_constants.json --veto 0.5
```

**`--constants` is not optional.** Without it the script silently falls back to the
D5/B5 rule (`mu_d`=35.30, `tau`=3.37) while every analysis here uses the D-6 ladder
fit, and gated glare reads 0.0641 instead of 0.2058.

---

## 1. Headline result

All numbers below use the corrected capability prior (§6.1). Day = pohang00/02/03,
night = pohang01, which is a real night run where the VIS detector scores **exactly
0.0000**.

| condition | split | no photometric gate | + gate | **+ gate + veto** | `ir_only` |
|---|---|---|---|---|---|
| clean | day | 0.3342 | 0.3342 | **0.3342** | 0.0092 |
| clean | night | 0.0765 | 0.0787 | **0.0813** | 0.0810 |
| fog | day | 0.0077 | 0.0077 | **0.0077** | 0.0092 |
| fog | night | 0.0786 | 0.0786 | **0.0789** | 0.0810 |
| lowlight | day | 0.0086 | 0.0089 | **0.0087** | 0.0092 |
| lowlight | night | 0.0789 | 0.0789 | **0.0813** | 0.0810 |
| glare | day | 0.2616 | 0.2616 | **0.2616** | 0.0092 |
| glare | night | 0.0707 | 0.0779 | **0.0813** | 0.0810 |
| **sum, 8 cells** | | 0.9168 | 0.9265 | **0.9350** | |

Per run, clean condition:

| run | frames | visible only | ir only | no gate | + gate | **+ veto** |
|---|---|---|---|---|---|---|
| pohang00 | 836 | 0.4004 | 0.0245 | 0.4024 | 0.4024 | **0.4024** |
| pohang01 (held out, night) | 1032 | **0.0000** | 0.0810 | 0.0765 | 0.0787 | **0.0813** |
| pohang02 | 247 | 0.3653 | 0.0031 | 0.3583 | 0.3585 | **0.3585** |
| pohang03 | 117 | 0.1840 | 0.0024 | 0.1805 | 0.1805 | **0.1805** |

**The claim that now holds:** gated fusion beats using IR alone on the night run
(0.0813 vs 0.0810) in three of four conditions, having trailed it in all four. The
margin is necessarily small — `ir_only` is the ceiling on a run where VIS
contributes exactly nothing — but the sign is right, and that is the claim the
paper makes. It is not free: fog/night remains behind at 0.0789.

---

## 2. Part I — the night-blindness investigation

### 2.1 How it surfaced

The cheap-fixes pass broke Table 3 down per recording run for the first time and
found that **pohang01 is a genuine night run**: VIS mAP exactly 0.0000, 94 boxes
emitted across 1,032 frames, max detector confidence 0.0043. IR carries the system
there entirely.

It is also **46% of the 2,232 paired val frames** while being only 18.2% of the
full 11,352-frame VIS val, because IR val coverage concentrates in pohang00/01.
Paired-subset numbers are therefore **not comparable** to the Phase 1/2 benchmark
and never should be quoted against it.

### 2.2 The gate was reading blind frames as the cleanest ones

| run | Mahalanobis D | VIS mAP |
|---|---|---|
| pohang00 (daylight) | 30.0 | 0.4004 |
| **pohang01 (night)** | **28.4** | **0.0000** |

Lower distance means *more* reliable. The gate ranked the blind run above a working
one.

### 2.3 Root cause — found, and it is not "the features are too coarse"

`runs/cache/gauss_vis_train_clean.pkl`, the Mahalanobis reference set, is 4,000
frames composed of:

| pohang00 | pohang01 | pohang02 | pohang03 | pohang04 |
|---|---|---|---|---|
| 679 | **782** | 848 | 872 | 819 |

**782 night frames are inside the reference distribution.** Night therefore
*defines* normal and can never be out-of-distribution. No re-tuning of `mu_d` or
`tau` could have fixed this — the ordering itself is wrong, not the threshold.

This is downstream of `filter_night_boxes.py` removing night **labels** from train
while night **images** stayed. Correct for training; it silently calibrated the OOD
scorer to treat darkness as ordinary.

Checked and cleared: overlap between that fit set and the 2,232 paired val frames
is **0 frames**. This is a calibration problem, not a leak.

### 2.4 Why no severity ladder could ever have found this

The decisive control. Apply synthetic lowlight to *daylight* frames and compare
against real night at matched darkness:

| | content brightness | `w_vis` from D alone |
|---|---|---|
| synthetic lowlight on day frames | 5.42 | **0.037** |
| real night (pohang01) | 8.16 | **0.792** |

Same darkness, **21× different response**. D catches the synthetic transform
because an unnatural multiplicative darkening is genuinely unusual; it is blind to
real night because real night is in its reference set.

**Consequence: no corruption ladder, however many families or severities it
covered, could have surfaced this failure.** It only appears on real data from a
run the reference set contains. This is the strongest methodological finding of the
session and belongs in the paper's limitations section regardless of what else
survives.

---

## 3. Part II — the photometric gate

### 3.1 Design

`r_frame = min(r_maha, r_bright)` where `r_bright = sigmoid((b − mu_b)/tau_b)`.

Three choices, each with a reason:

- **`min`, never a product.** The two signals answer different questions ("is this
  frame strange?" and "is this frame lit?"). Either firing is sufficient grounds to
  distrust the modality. A product lets a confident-looking D dilute a brightness
  alarm — the exact failure being fixed.
- **Content rows only.** The prepared tree is 640×640 with grey (114) padding: VIS
  content is 640×338 at y+151, **53% of the canvas**. Averaging over the whole
  canvas mixes a constant 114 into every statistic and crushes the contrast being
  measured.
- **VIS only.** On a thermal sensor "brightness" is scene temperature, not
  illumination; a dark IR frame is cold water, which is the condition IR exists
  for. No `mu_b` is fitted for IR and none should be. This asymmetry also means IR
  can never be vetoed (§4).

Corruption replay: ladder caches were corrupted in memory at build time, so the
file on disk is clean and its brightness is *not* what the detector saw.
`frame_brightness.py` replays `make_corruption(name, severity, seed)` at the same
index, which is deterministic by construction.

### 3.2 Statistic selection — six tried, `p05` adopted

The first fit used **mean intensity** and produced a regression: glare/night got
*worse*. Diagnosis: **fog and glare ADD light.** Fog lifts the night run's mean
from 8.2 to 87.8; glare to 40.7. Any statistic that can be raised by adding light
can be fooled into calling a blind frame usable.

`probe_stat_robustness.py` measures this without touching the held-out run: darken
fit-run images, add glare or fog on top, and report how far the statistic climbs
back toward its clean value.

```
spoof_frac = (stat_after_added_light − stat_dark) / (stat_clean − stat_dark)
0.0 = robust,  1.0 = fully spoofed
```

| statistic | glare | fog | worst case |
|---|---|---|---|
| **`p05`** | **0.000** | **0.243** | **0.243** ✅ adopted |
| `p50` | 0.282 | 0.885 | 0.885 |
| `mean` | 0.354 | 0.747 | 0.747 |
| `std` (RMS contrast) | 0.874 | 0.735 | 0.874 |

`range` and `frac_dark` were computed by `frame_brightness.py` and fitted but lost
on either fit quality or spoofability. `p05` is completely immune to glare and 3×
more fog-resistant than `mean`, at equivalent fit RMSE.

**This selection used fit runs only, and involves no detector at all** — it is pure
image arithmetic. It is therefore untouched by the capability-prior bug in §6.1.

### 3.3 Threshold placement — three candidates, the margin rule adopted

Switching to `p05` inverted the glare/night regression but introduced a new one:
**pohang03 fell 4.4%**. Its `r_bright` was being damped to 0.528 on a daylight run
VIS handles fine.

Measured per-run `p05`:

| run | mean | min | max |
|---|---|---|---|
| pohang00 | 34.6 | 33.0 | — |
| pohang01 (night) | 2.5 | 1.0 | **4.0** |
| pohang02 | 37.8 | 32.0 | — |
| **pohang03** | 25.9 | **21.0** | — |

**The diagnosis that turned out to be wrong.** The first write-up claimed synthetic
lowlight s1 produces *intermediate* `p05` values that drag the threshold up. That
is false, and one measurement kills it:

| condition (fit runs) | `p05` mean | min | max | % inside the clean range |
|---|---|---|---|---|
| clean | 34.40 | 21.0 | 44.0 | — |
| lowlight s1 | **0.00** | 0.0 | 0.0 | **0.0%** |
| lowlight s2 | **0.00** | 0.0 | 0.0 | 0.0% |
| lowlight s3 | **0.00** | 0.0 | 0.0 | 0.0% |

All three severities read exactly 0.0.

**The actual cause.** `p05` is **bimodal by construction** on the fit pool:
corrupted frames read 0, clean frames read ≥ 21, and there is not a single sample
in between. *Every* threshold inside (0, 21) fits the retention curve identically,
so the data does not choose one — the optimizer does, from whatever weak signal
remains. That signal was a **within-daylight confound**, visible in the fitted bins:

```
bin b=0.0    n=1200  retention 0.067     <- all corrupted frames
bin b=25.5   n=38    retention 0.513     <- pohang03's clean daylight
bin b=34.0   n=157   retention 0.948
```

pohang03's clean retention really is ~0.51 (0.1840 against the fit-run clean mean
0.3437) — but that is **scene difficulty** (smaller, more distant vessels), not
dimness. `curve_fit` read a two-point cross-run correlation as causal.

Three candidates were then measured:

| | `mu_b` | `tau_b` | day `r_bright` floor | night `r_bright` max | pohang03 |
|---|---|---|---|---|---|
| retention fit, all severities | 25.373 | 2.520 | **0.150** | 0.0002 | **0.1767** ❌ |
| retention fit, s2/s3 only | 17.341 | 5.581 | 0.658 | 0.0839 | 0.1805 |
| **margin rule** ✅ | **10.500** | **2.625** | **0.982** | 0.0775 | **0.1805** |

**Adopted: the margin rule.** State the placement instead of fitting it, using
only fit-run endpoints:

```
mu_b  = 0.5 * (max p05 over corrupted fit frames + min p05 over clean fit frames)
tau_b = (min_clean − max_corrupt) / 8      # ±4 tau spans the empty margin
```

Margin [0.00, 21.00] → `mu_b` = 10.500, `tau_b` = 2.625. No optimizer in the loop,
pre-registrable in one sentence, and it saturates on both observed populations
rather than transitioning through either.

The s2/s3 refit also works, but *by accident*: dropping s1 shifts the quantile-bin
boundaries so the confounded `b=25.5` bin dissolves, leaving the fit unconstrained
across the whole gap so it lands near the middle. Right answer, no argument behind
it.

**A trap worth recording.** The rejected retention fit shows a *higher* 8-cell sum
than the margin rule, purely because its pooled clean/day rises — and that rise
comes from *suppressing pohang03's detections in the global ranking*. The pooled
day number improves precisely because a working run was damped. Per-run, the same
setting costs that run 4.4%. **The aggregate metric rewarded the bug.**

### 3.4 The prediction that was refuted

Before running the four-condition regression check I predicted that `lowlight`
would improve under the brightness term, since `r_bright` reads 0.050 there — near
maximum alarm.

**It changed nothing.** fog and lowlight came back **bit-identical** before and
after. The reason is §2.4: D already catches synthetic corruptions, so the
photometric term is redundant there and `min` is a no-op. That refutation is what
produced the strongest finding of the session, and it is the reason the term can be
described as "changes exactly one thing, and is inert everywhere else **by
measurement**".

### 3.5 What the photometric gate does not fix

- **Blur is photometrically invisible.** `vis_clean` 62.65 vs `vis_blur_s3` 63.33
  mean intensity. §9.4's blur/glare counterexample is untouched — A3 remains the
  only candidate for it.
- **Fog raises brightness**, so `r_bright` → 1.0 under fog and contributes nothing
  there. Harmless because `min` only ever lowers reliability, but it confirms the
  term is a darkness detector, not a general quality signal.
- **It closed only ~54% of the gap to `ir_only`** on its own. Part III closes the
  rest.

---

## 4. Part III — the hard veto

### 4.1 Why down-weighting cannot work, and this is the key mechanism

After the photometric gate, `r_frame_vis` on the night run is 0.052 and yet gated
fusion still trailed `ir_only`. Two mechanisms, **neither of which shrinks as
`w → 0`**:

**1. Weights normalise.** `w_vis = R_vis·cap_vis / (R_vis·cap_vis + R_ir·cap_ir)`.
The capability prior hands VIS a 12.5× advantage (0.2580 vs 0.0206), which leaves
**`w_vis` = 0.432** on the night run — a blind stream keeping 43% of the vote on
frames where it scores exactly 0.0000.

**2. WBF rescales scores, it does not drop boxes — and the rescale hits the good
stream.** Measured directly on a two-box case:

```
both streams present:  fused conf [0.87, 0.28, 0.21]
VIS vetoed:            fused conf [0.80, 0.70]     <- IR's originals, untouched
```

An unmatched IR detection enters at 0.70 and leaves at **0.21**, because WBF scales
every single-modality cluster by that modality's normalised weight. So a blind VIS
stream does not merely inject false positives — it **attenuates IR's true
positives by `w_ir`**. And mAP is rank-based, so scaling a false positive down does
not remove the rank it occupies.

A failed modality has to leave the input list. `fuse_detections(..., veto_vis=,
veto_ir=)` does exactly that; vetoing both is refused (that case is the plan-B3
abstain, signalled by `R_sys`, not an empty frame).

### 4.2 The rule — no new constant

> Exclude modality *m* when `r_bright_m < 0.5`, i.e. `b_m < mu_b`.

`mu_b` is the §3.3 margin midpoint, already registered. 0.5 is the sigmoid
midpoint, not a swept threshold. Nothing was tuned to produce this.

Veto rates under the adopted rule:

| condition | pohang00 | pohang01 | pohang02 | pohang03 |
|---|---|---|---|---|
| clean | 0% | **100%** | 0% | 0% |
| fog | 0% | 29% | 0% | 0% |
| lowlight | 100% | 100% | 100% | 100% |
| glare | 0% | **100%** | 0% | 0% |

On clean data the veto fires on exactly the night run and nowhere else.

### 4.3 Tried and rejected — vetoing on the Mahalanobis axis

The obvious generalisation is `r_frame < 0.5`, i.e. `D > mu_d` **OR** `b < mu_b`.
It was implemented, run, and is **wrong**:

| | soft weights | veto on `r_bright` | veto on `r_frame` |
|---|---|---|---|
| glare / day | 0.2532 | **0.2532** | **0.1435** |
| VIS veto rate, glare, fit runs | — | 0% | **46 / 45 / 49%** |

Glare pushes `D` past `mu_d` on ~47% of **daylight fit-run** frames, where VIS
still scores 0.2626 against IR's 0.0092. The veto discards a stream 28× stronger
than the alternative and costs that cell **−43%**.

The principle was already in the handoff: §9.4's blur/glare counterexample shows
`D` is **not monotone in capability**. It answers *"is this frame unusual?"*; a veto
needs *"did this sensor fail?"*. **A non-monotone signal may down-weight — a soft
weight degrades gracefully when the signal is wrong — but it must not hold a
switch.** Brightness is monotone in the one direction that matters: below `mu_b` no
photons reached the sensor, and emptiness is not recoverable by any detector.

*(Those two figures are from the pre-correction run; the mechanism, not the
magnitude, is the point. Kept as `runs/eval/veto_rule_maha.md`.)*

### 4.4 The protocol point worth keeping

The rejected `r_frame` variant is **better on the held-out fog/night cell** (0.0813
vs 0.0792). It was rejected anyway, on daylight fit-run evidence. **The adopted
rule is the one that scores worse on pohang01** — about as clean a demonstration as
one can give that the selection was not driven by the held-out set.

### 4.5 Two costs, not hidden

1. **lowlight/day loses 0.0002** (0.0089 → 0.0087). Synthetic lowlight reads
   `p05` = 0 so VIS is vetoed on 100% of daylight frames — but VIS still scores
   0.0173 there against IR's 0.0092, so the veto discards the better stream. Real
   night has no recoverable signal; a synthetic multiplicative darkening does, and
   `p05` cannot distinguish them.
2. **fog/night reaches only 0.0789**, still under `ir_only`. Fog lifts `p05` above
   `mu_b` on 71% of night frames, so the veto fires on 29% of them — the residue of
   `p05`'s 0.243 fog spoof fraction. **The one cell of eight where this is not
   fully closed.** It needs a fog-robust statistic, not a different fusion rule, and
   none of the six computed does better.

---

## 5. Part IV — `iou_thr` re-sweep, and Part V — σ-weighted WBF

### 5.1 `iou_thr`: 0.85 is a real optimum, not a boundary artifact

The original sweep covered 0.40–0.85 and rose monotonically to the top of the
range, so 0.85 looked like a boundary hit. Extended to 0.99 under the adopted gate
configuration, tuned on the corrupt-seed-2 ladder and reported on seed-1 (plan
B5-5):

| `iou_thr` | 0.40 | 0.55 | 0.70 | **0.85** | 0.90 | 0.95 | 0.99 |
|---|---|---|---|---|---|---|---|
| tuning mean (pooled) | .1311 | .1318 | .1326 | **.1328** | .1327 | .1320 | .1316 |
| tuning mean (fit runs only) | .1526 | .1540 | .1560 | .1579 | **.1584** | .1579 | .1572 |

**The curve turns over.** The two criteria disagree by 0.0001 and 0.0005 in
opposite directions — a tie, not a signal.

**Kept 0.85**, on the criterion the original protocol actually used (pooled ladder
mean), where it is the argmax. The fit-run column is a leakage check added because
the seed-2 ladder *contains* pohang01 night frames (the seed split separates
corruption draws, not recording runs); it returns "no material effect", not a
mandate to switch. The reporting-split comparison also favours 0.85 but was
**deliberately not used as a reason** — that would be selecting on the report split.

### 5.2 σ-weighted WBF: implemented, verified, and a measured no-op

**The gap.** Until this, *nothing the Gaussian head produced influenced fusion*.
`r_box` is inert (0.86–0.95 in every condition, including the fog case where mAP is
0.0012), and stock WBF averages cluster coordinates weighted by
`score × model_weight` — a confidence, never a precision. A paper about
uncertainty-aware fusion rested on a Mahalanobis frame score and a brightness
statistic. Neither is the σ head.

**What was built.** `sigma_weighted_fusion()` changes exactly one thing:

```
coordinate weight:  score_i × model_weight_i / sigma_i²      (per coordinate)
```

the minimum-variance estimator for combining independent measurements — not a
heuristic. A confident-but-blurry box keeps its full vote on *whether* an object is
there and loses its vote on *where* the edges are. `sigma_ltrb`'s (l,t,r,b) maps
onto (x1,y1,x2,y2), so a box can be trusted on one edge and distrusted on another.

**Verified to BE stock WBF when σ is uninformative.** `smoke_sigma_wbf.py` runs 300
randomised two-model cases (2,260 fused boxes) and asserts that both
`use_sigma=False` *and* `use_sigma=True with constant σ` reproduce
`ensemble_boxes.weighted_boxes_fusion` to **6.3e-08** — boxes, scores and labels, in
order. Without that equality the A/B would be measuring two fusion implementations
rather than σ.

**Result: ±0.0001 on every condition.**

| condition | stock WBF | σ-weighted | Δ |
|---|---|---|---|
| clean | 0.2629 | 0.2628 | −0.0001 |
| fog | 0.0204 | 0.0204 | +0.0000 |
| lowlight | 0.0206 | 0.0206 | +0.0000 |
| glare | 0.2063 | 0.2064 | +0.0001 |

σ is **not** uninformative — `u_box` spans **9.1×** p05→p95 on VIS and 4.1× on IR.
So the null result needed an explanation, and it has a decisive one.

### 5.3 Why — fusion at `iou_thr` 0.85 is almost entirely concatenation

Inverse-variance averaging can only act inside a cluster of two or more boxes.
`diag_cross_modal_iou.py` measures how often that happens: for each of 29,042 VIS
detections, the best IoU against any of 120,877 IR detections in the same frame,
after the per-run homography.

| WBF `iou_thr` | VIS boxes with an IR partner above it | share |
|---|---|---|
| **0.85 (tuned)** | **31** | **0.11%** |
| 0.70 | 917 | 3.16% |
| 0.55 | 3,634 | 12.51% |
| 0.40 | 7,796 | 26.84% |
| 0.25 | 11,098 | 38.21% |
| 0.10 | 15,996 | 55.08% |

**31 boxes out of 29,042 have a partner.** WBF is merging two lists and rescaling
scores; it is barely fusing anything. σ never gets invoked because there is almost
never a cluster for it to weight.

Mechanism check: re-run at `iou_thr` 0.55, where 12.5% of VIS boxes have a partner,
the glare delta grows from +0.0001 to **+0.0005** — about 5×, tracking cluster
opportunity from a base so small it stays negligible. *(That run is a mechanism
probe, not a re-tune; nothing was selected on it.)*

### 5.4 What this reframes

1. **The `iou_thr` curve now has a cause.** Higher `iou_thr` means less cross-modal
   merging, which means less damage from a badly-registered IR stream. The sweep
   was tuning toward *"fuse less"*, not *"fuse better"*.
2. **A2 (per-run registration refinement) is promoted to the blocker.** It is the
   precondition for any fusion-rule improvement to be able to do anything.
3. **The union-label question is probably already answered.** If VIS and IR boxes
   rarely reach IoU 0.5, the union GT's +76.8% box count is largely the same
   objects counted twice — exactly what §0.4 suspected.

**Keep σ-weighting in.** It is the claim the method makes, it is correct, it costs
nothing, and on a properly registered pair it is the right estimator. Report it as
implemented with a measured null effect and a diagnosed cause; do not quietly drop
it.

---

## 6. Part VI — bugs and errors found in our own work

Recorded because each would otherwise be invisible to anyone reading the numbers
later.

### 6.1 The IR capability prior was 3.3× too high

```python
cap_ir = float(rc["ir"].get("map_clean", 0.0206))   # WRONG
```

The key **exists** in `reliability_constants.json` at **0.067552**, so the `0.0206`
fallback never fired. But 0.0676 is IR's clean mAP on the **IR ladder against IR
GT** — a different task on different frames. The fusion prior needs IR's capability
on *this* evaluation: boxes mapped through H into the VIS frame, scored against VIS
GT. That is **0.0206**, printed as the `ir_only` row in the same tables.

`run_fusion_eval.py` and `run_cheap_fixes.py` were always correct; only the two
scripts written that day deviated. Found while re-sweeping `iou_thr`, because that
script used 0.0206 and its numbers would not reconcile.

**Fixed** by computing `cap_ir` from the data in both scripts, so the two can no
longer diverge. Every conclusion survived re-measurement:

- Night 0.0813 vs `ir_only` 0.0810 — **unchanged**, since VIS is vetoed there and
  the weights never enter.
- The veto argument got **stronger**: `w_vis` on the night run is 0.432, not 0.199.
- The margin rule still beats the retention fit (pohang03 0.1805 vs 0.1767).
- The `p05`-over-`mean` choice is untouched — it rests on image arithmetic.
- Daylight was **understated**: clean/day 0.3303 → 0.3342, glare/day 0.2532 →
  0.2616.
- fog/day **fell** 0.0085 → 0.0077 — the honest consequence of correctly
  down-weighting IR when VIS scores 0.0012.

### 6.2 The headline table used different constants from the analysis

`run_fusion_eval.py` defaults to the D5/B5 rule (VIS `mu_d`=35.30, `tau`=3.37)
while §14–§19 all use the D-6 ladder fit (`mu_d`=71.07, `tau`=37.21). With the
sharp D5/B5 constants, glare drives `r_frame` off a cliff and gated glare collapses
to **0.0641** against 0.2058. Table 3 is now generated with `--constants`.

### 6.3 Three factual errors corrected

| Claim made | Correction |
|---|---|
| "pohang01 mean content intensity 14.8" | 14.85 is the **maximum**; the mean is **8.16** |
| "lowlight s1's `p05` overlaps daylight, dragging the threshold up" | s1/s2/s3 all read `p05` = **0.0 exactly**; 0.0% inside the clean range. Real cause in §3.3 |
| "`w_vis` stays at 0.199 on the night run" | **0.432** at the corrected prior |

### 6.4 A refuted prediction

Predicted `lowlight` would improve under the photometric term. It changed nothing —
fog and lowlight came back bit-identical. See §3.4; the refutation produced the
best finding of the session.

---

## 7. Protocol and data integrity

**Anti-leakage rules held throughout:**

- **pohang01 is held out** of every fit: the photometric constants, the statistic
  choice, the threshold rule, and the veto rule were all decided on
  pohang00/02/03 or on pure image arithmetic.
- **B5-5:** tuning on the corrupt-seed-2 ladder, reporting on seed-1. Never shared.
- The one place the two touch is `iou_thr`, because the seed-2 ladder contains
  pohang01 frames. Flagged explicitly and checked with a fit-run-only argmax
  (§5.1); it made no material difference.

**Source dataset never written.** `verify_dataset_integrity.py` turns the
convention into something checkable — image count, total bytes, sha256 over the
sorted `(path|size)` listing, and a newest-mtime bound that catches same-size
overwrites (label `.cache` files excluded, since Ultralytics rewrites them as a
normal side effect of reading a split).

Baseline: **189,505 images, 48,874,893,142 bytes**, newest image mtime 2026-08-18
17:43:40. Verified **untouched** five separate times across the session, including
after every stage of this work.

---

## 8. Consolidated table — everything tried that did not work

| Attempt | Result | Why it failed |
|---|---|---|
| Re-tune `mu_d`/`tau` for night | Impossible | Night is *inside* the reference set; the ordering is wrong, not the threshold |
| `mean` intensity statistic | glare/night −1.1% | Fog/glare ADD light; mean is spoofable (0.354 / 0.747) |
| `p50`, `std`, `range`, `frac_dark` | Rejected | Worse spoofability and/or fit quality than `p05` |
| `p05` + retention `curve_fit` | pohang03 −2.1% | Threshold unconstrained in a bimodal gap; placed by a within-daylight confound |
| `p05` + refit on s2/s3 only | Works, but by accident | Bin-boundary artifact dissolves the confounded bin; no argument behind it |
| Predicting the gate would help `lowlight` | Bit-identical | D already catches synthetic corruptions; `min` is a no-op there |
| Weight floor / ceiling / conditional prior for §0.3 | Never attempted after diagnosis | No weight-space change can work — WBF rescales, does not drop, and mAP is rank-based |
| Veto on `r_frame` (`D > mu_d`) | glare/day −43% | `D` is not monotone in capability; must not hold a switch |
| `iou_thr` 0.90 / 0.95 / 0.99 | Worse or tied | Curve turns over at 0.85 |
| σ-weighted WBF | ±0.0001 | Only 0.11% of VIS boxes have an IR partner at `iou_thr` 0.85 |
| `skip_box_thr` > 0 *(earlier)* | Monotonically worse | IR over-detection costs WBF nothing; it is a training problem |
| Temporal smoothing `alpha` *(earlier)* | 0.1371 → 0.1374 | Inert; decision D14 unrefuted |
| Global registration bias correction *(earlier)* | Failed run-disjoint | It is a per-run residual, not a constant |

---

## 9. Future testing

### 9.1 Immediate, CPU-only

| # | Test | Est. | Why now |
|---|---|---|---|
| 1 | **Dedup-threshold sweep (§0.4)** — sweep the union-GT merge criterion (IoU 0.3/0.4/0.5 + centre-distance) and report how the +76.8% box count moves | 1–2 h | §5.3 all but answers it: if VIS/IR boxes rarely reach IoU 0.5, the union GT is the same objects twice and §14.3's +19.1% is inflated. **No union-label number should be quoted until this runs.** |
| 2 | **Fixed-GT risk–coverage for `R_sys` (§0.6)** — redo against a common denominator, or use AURC | 1–2 h | The current non-monotonicity is measured over shifting frame subsets, so the GT population changes with coverage. The effect probably survives, but the measurement is not clean. |
| 3 | **Day-only Mahalanobis refit** — rebuild the reference set from daylight frames only and re-measure D on pohang01 | ~20 min | The alternative root-cause fix to §2.3. If D then separates night correctly, the photometric term becomes a redundancy check rather than the load-bearing signal — a much stronger story. |
| 4 | **Fog-robust photometric statistic** — the one open cell (§4.5). Candidates beyond the current six: local contrast in the darkest quantile, gradient-energy density, or `p05` computed after a fog-removal prior | 2–4 h | fog/night 0.0789 vs `ir_only` 0.0810 is the only cell where §0.3 is not closed |
| 5 | **σ calibration check** — is `sigma_ltrb` actually calibrated? Plot predicted σ against realised localisation error on matched detections | 2 h | §5.2 shows σ *varies*, not that it is *right*. If σ is miscalibrated, inverse-variance weighting is the wrong estimator even once registration improves. **This is a precondition for A1 mattering.** |

### 9.2 After registration (A2) — the unblocking sequence

A2 is now the blocker: 5.4 px median registration error on 12–17 px boxes makes
IoU 0.85 between a VIS box and its true IR counterpart nearly unreachable.

| # | Test | Depends on |
|---|---|---|
| 6 | **Per-run translation term fit on TRAIN frames** (A2). The homography is already per-run; a global bias correction failed run-disjoint validation with the offset swinging dx +0.17 → −4.01 px between run pairs, so it is a per-run residual | — |
| 7 | **Re-run σ-weighted WBF after A2** and re-measure the cross-modal IoU distribution. **Pre-registered prediction: if A2 lifts the >0.55 share above ~30%, the σ delta should grow by roughly the same factor it did between `iou_thr` 0.85 and 0.55 (≈5×).** If it does not, σ is miscalibrated (test 5) rather than merely unused | 6 |
| 8 | **Re-sweep `iou_thr` after A2.** The current optimum at 0.85 is a "fuse less" artifact; with better registration the optimum should *fall*, and that shift is itself evidence A2 worked | 6 |
| 9 | **Re-run the dedup sweep after A2** and compare against test 1 | 1, 6 |

### 9.3 Training-side (GPU) — the night ceiling

`ir_only` = 0.0810 is the ceiling on night, and gated fusion is at 0.0813. **No
further gate work is worth anything there.** The only way night moves is raising
`ir_only`.

| # | Test | Est. | Rationale |
|---|---|---|---|
| 10 | **IR ship-only retrain (B5)** | 3.5 h | IR buoy AP is **0.00022** against 29,131 buoy detections for 596 GT boxes. The model spends real capacity on a class it cannot see. Most certain of the training levers. |
| 11 | **CLAHE IR re-export + retrain (B3)** | 1–2 h + 3.5 h | The percentile experiment was rejected because min–max and percentile are *both per-frame affine maps* and no global remapping fixes local contrast. **CLAHE is local** — the direct answer to the diagnosed failure, and untried. Reuse `export_ir_percentile.py`'s min–max round-trip verification gate so the mapping is the only variable. |
| 12 | **Rect training (B1)** | 1–2 h + 4.5 h | Only 53% of the VIS canvas carries image. At the same pixel budget a 2048×1080-aspect rectangle is 881×465 — **1.38× linear resolution for identical compute**. Free resolution, not bought resolution. IR is already 80% efficient (~1.12×). |
| 13 | **`yolo26s` → `yolo26m` (B4)** | ~7 h | Known-good: 0.2864 → 0.3061 on VIS in Phase 1. No new code. |
| 14 | **imgsz 1280 on the full-resolution tree (B2)** | 17–22 h | VIS median ship is **14 px** at model input; 86.9% under 32², 58.3% under 16². Null lever for IR, which is already native. Do not price the two together. |
| 15 | **Address the 18.5% empty-label rate** — pohang01 is **93.1% empty** (17,523 of 18,826 frames) after the night-box filter | — | Worth checking whether those frames help or hurt; they are 19.5% of VIS train |

### 9.4 Methodological rigour — what a reviewer will ask for

| # | Test | Why |
|---|---|---|
| 16 | **MC-dropout / deep-ensemble baselines (D-5, deferred)** | The paper claims a single-pass σ head is competitive with sampling-based UQ. Currently unmeasured. |
| 17 | **A third recording run held out.** Every generalisation claim rests on pohang01 alone | One held-out run is an anecdote with an n of 1. pohang04 exists in the train split and is unused in fusion evaluation. |
| 18 | **Bootstrap CIs on the 8-cell table** | Several adopted deltas are 0.0002–0.0026. Without an error bar it is not knowable which of them are real. **This is the single most important item in this section** — the veto's night gain is +0.0026 and the lowlight/day cost is −0.0002, and both are currently unquantified. |
| 19 | **Ablate the veto and the photometric term independently on a common config** | They are currently reported as a chain (no gate → gate → gate+veto). The interaction term is unmeasured. |
| 20 | **Sensitivity of the margin rule to the fit-run set** — leave-one-fit-run-out on `mu_b` | `mu_b` = 10.500 comes from a margin whose upper endpoint is pohang03's `p05` min of 21.0, i.e. **one run defines half the rule**. If dropping pohang03 moves `mu_b` a lot, the rule is fragile. |

### 9.5 Falsification tests — what would show this work is wrong

Stating these in advance is worth more than any additional confirmation.

1. **If the day-only Mahalanobis refit (test 3) fixes night on its own**, then the
   photometric term is not load-bearing and the story simplifies to "the reference
   set was mis-composed". The gate work would still be correct but much less
   interesting.
2. **If bootstrap CIs (test 18) span zero for the night gain**, the headline claim
   — gated 0.0813 vs `ir_only` 0.0810 — does not survive, and the honest report
   becomes "gated fusion matches IR-only on night, having previously trailed it".
3. **If a third held-out run (test 17) shows `mu_b` = 10.500 vetoing daylight
   frames**, the margin rule is overfitted to three runs and needs a floor tied to
   the clean distribution.
4. **If σ turns out to be miscalibrated (test 5)**, A1 should be reported as
   implemented-and-inapplicable rather than implemented-and-blocked, which is a
   materially different claim.

---

## 10. File index

### Source

| Path | What |
|---|---|
| `src/uqfusion/uq/reliability.py` | `mu_b`/`tau_b` on the constants, optional `frame_brightness`, `r_bright` returned. Defaults reproduce §6.4 exactly |
| `src/uqfusion/uq/fusion.py` | `veto_vis`/`veto_ir` (a vetoed modality leaves the WBF input list); `sigma_weighted_fusion()`; `fuse_detections(..., sigma_weighted=)` |
| `src/uqfusion/eval/fusion_eval.py` | `brightness_*`, `veto_below`, `veto_on`, `sigma_weighted` passthrough; returns `r_bright_vis`, `r_frame_*`, `veto_*`, `ir_in_vis` |

### Scripts

| Path | What |
|---|---|
| `frame_brightness.py` | Content-region photometric stats with deterministic corruption replay |
| `fit_brightness_gate.py` | `--rule retention` or `--rule margin` (adopted); `--stat`, `--fit-conditions`; enforces the run-disjoint protocol |
| `probe_stat_robustness.py` | Luminance-spoofing probe; ranks statistics by added-light resistance, fit runs only |
| `eval_brightness_gate.py` | Photometric gate before/after, per-run and day/night |
| `eval_veto_rule.py` | Veto rates, `r_frame` distributions, before/after; `--veto-on` |
| `eval_sigma_wbf.py` | σ-weighted WBF before/after, with the `u_box` spread that makes the null interpretable |
| `smoke_sigma_wbf.py` | Asserts equality with stock WBF at constant σ (6.3e-08) and per-coordinate inverse-variance behaviour |
| `diag_cross_modal_iou.py` | Cross-modal box agreement — the explanation for the σ null result |
| `sweep_iou_thr.py` | `iou_thr` grid past the boundary, with a fit-run-only leakage check |
| `verify_dataset_integrity.py` | Proves `Pohang_dataset/` is never written |
| `run_fusion_eval.py` | Headline Table 3, now with the day/night split |

### Results *(under `runs/`, which is gitignored)*

`veto_rule.md` (adopted) · `veto_rule_maha.md` (rejected variant) ·
`brightness_gate_p05_margin.md` (adopted) · `brightness_gate_p05.md`,
`brightness_gate_p05_s23.md`, `brightness_gate_clean.md` (candidates) ·
`sigma_wbf.md`, `sigma_wbf_iou055.md` · `cross_modal_iou.md` ·
`iou_thr_sweep.md` · `table3_fusion.md` ·
`brightness_constants*.json` · `dataset_integrity_baseline.json`

**Note:** `runs/` is gitignored, so none of these result files are under version
control. The convention predates this work, but it does mean this document and the
handoff cite files that a fresh clone will not contain. A `!runs/eval/*.md`
negation in `.gitignore` would fix it cheaply — they are small text files.

---

## 11. Follow-up round (2026-08-20)

Seven further tests, the p2feat IR upgrade pushed into fusion, and the stage-3/4
training arms are recorded in `docs/followup-analysis-2026-08-20.md`. It revises
this document in three places: the night claim in §4 spans zero under a paired
bootstrap and inverts under a better IR detector; the photometric gate in §3 is
redundant once the veto is present; and the A2 per-run registration plan in §9.2
does not hold, because the within-run drift exceeds the between-run swing that
killed the global correction.
