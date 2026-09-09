# Experiment log — 2026-09-01, the 26m swap and the lever sweep

Every experiment run this session, what it asked, what it found, where the numbers live. Companion
to `docs/levers-and-the-26m-swap-2026-09-01.md` (the argued record); this is the complete list,
including the arms that failed and the two bugs I introduced and caught.

**Ground rules held throughout.** Nothing under `runs/cache/`, `runs/derived/`, `runs/eval/` or
`archive/` was overwritten. `Test_1/` untouched. Every new result went to a new filename.
`preset="crossmodal"` on `runs/cache/` still reproduces every published number bit-for-bit
(`smoke_crossmodal_gate.py` 14/14).

---

## 0. Summary — every experiment at a glance

| # | experiment | asked | found | verdict |
|---|---|---|---|---|
| A1 | corrupted-IR extended grid | can a damaged IR wrongly veto VIS end-to-end? | 10 day cells, all ≥ bar under 26s | — |
| A2 | `cap_ratio` pricing | is the ratio priceable now? | yes; ×4 is the smallest reaching the best worst-cell | adopt ×4 |
| B1 | detector swap, clean cell | what does yolo26m buy? | **+0.0003 VIS, +0.0011 IR. Night still 0.0000** | detector ≠ the lever |
| B2 | cache_m verification | are the derived stats detector-independent? | **identical to 0.000e+00** | assumption proven |
| B3 | stage report (D31) | base or fine-tuned? | VIS base 0.3686 / ft 0.3706; IR base 0.1190 / ft 0.1414 | VIS base, IR ft |
| C1 | agreement geometry | do the streams ever meet? | **0.05% at IoU 0.85**; 32% at 0.30 | fusion ≈ concatenation |
| C2 | cross-modal `support` | score-only confirmation at loose IoU | **+0.0090 day, +0.0033 held-out** | **adopt (0.30, 0.5)** |
| C3 | `consensus_beta` / `_distinct` | tune WBF's agreement bonus | inert or negative (−0.0068) | reject |
| C4 | per-frame registration | correct the 3–6 px residual | agreement **+80×**, AP **falls** | reject |
| C5 | align + forbid merge | is merging the damage? | confirmed; still negative held-out | reject |
| C6 | `class_veto` (buoy carry-through) | IR is nc=1, so a veto zeroes buoy AP | **exactly zero** — night GT has no buoys | reject |
| C7 | `iou_thr` re-sweep | 0.85 was tuned on 26s | 0.75 and 0.95 both worse on TEST | keep 0.85 |
| C8 | day holdout (`TUNE`/`TEST`) | is there any held-out day data? | **there was none** | methodology fix |
| D1 | 13-cell lever grid | everything at once on 26m | **fog/day at −0.0632** | regression found |
| D2 | fog diagnosis | why? | VIS fog **0.0020 → 0.0824**, 41× | veto mispriced |
| D3 | veil repair | `night AND (dark OR veil)` | fog/day **+0.0716**, fog/night **+0.0000** | **adopt** |
| D4 | veil/night exposure | does the repair inherit the night arm's failure? | day fine; **night −0.0296** on fog/blur_s2 | new cost found |
| E1 | axis claim test | does each axis delete the better stream? | veil −0.0632, `ir_p05` −0.3769, photometric −0.0243 | 3 flagged |
| E2 | axis ablation | what is each axis worth? | **`veto_ir` is net harmful** | **drop it** |
| E3 | authority bound sweep | loosen / tighten ×0.5 | **completely inert** on 12 cells | unpriceable here |
| E4 | night weak fallback | recover D4's cost | `lap_over_var` version: **+0.0093 ×2** | **adopt** |
| F1 | lift screen | which signals predict a TP? | sigma **3.00×**, cross-modal 2.08×, temporal **1.00×** | new instrument |
| F2 | temporal support | is the redundancy temporal? | **null** — 74% firing, zero lift | reject |
| F3 | sigma-in-score | rank on the sigma head | rejected; and the control says it's a **detector** post-process | reject (for fusion) |
| G1 | `ir_nms` re-tune | fitted against a model no longer used | flat to ±0.0002 across a 2× range | keep 0.7 |
| G2 | score calibration | 99.9% concatenation ⇒ ordering is everything | tune −0.0044, TEST −0.0035 | reject |
| H1 | final grid v2 | the whole repaired system | 7 cells better, 2 worse by ≤0.0004 | **shipped** |

---

## 1. Setup — what changed under the system

### 1.1 The detectors

| | before (`runs/phase2`) | after (`runs/full_scale`) |
|---|---|---|
| VIS | `yolo26s`, stride5, nc=2 | **`yolo26m`**, stride2, nc=2, 22.63M |
| IR | `yolo26s`, stride2, **nc=2** | **`yolo26m-p2feat`**, ship-only, **nc=1**, 23.30M |
| boxes/frame | VIS 13.0, IR 54.2 | VIS 8.9, IR 30.5 |

Both checkpoints were already local — no server pull needed. Only the server *retrain* of
`gauss_vis_seed0` (0.25589/0.24759) was never pulled; the local pair is self-consistent and fusion
needs no machine homogeneity.

**Stage chosen by D31's own rule** (highest validation mAP@50-95), applied per modality to numbers
already in `runs/queue_full_scale/state.json` — VIS base (0.24961 > ft 0.24111), IR ft (0.14214 >
base 0.11981). The paired val was **not** consulted for that choice. Reported afterwards
(`cache_m_verify.md`): on the paired val VIS base 0.3686 against ft **0.3706**, so the rule cost
−0.0020 on VIS and was right on IR. Both facts are on the record; the rule stands.

### 1.2 The caches

14 new caches in `runs/cache_m/`, stems deliberately identical to `runs/cache/`. 4 stage variants
preserved in `runs/cache_m_stageprobe/`.

**B2 — the assumption that made stem-sharing legal, proven rather than asserted.**
`verify_cache_m.py` re-derives `grad_gini`, `lap_over_var` and `p05` from the 26m caches and compares
to the files written from the 26s ones:

| cache | stat | max abs diff |
|---|---|---|
| `gauss_vis_paired_fog` | grad_gini / lap_over_var / p05 | **0.000e+00** |
| `gauss_ir_paired_glare_s2` | grad_gini / lap_over_var / p05 | **0.000e+00** |

All 14 metas identical field-for-field, all frame counts equal. So the image statistics are a
function of (image list, corruption kind, severity, seed) alone and **every fitted threshold in the
gate is detector-independent**. That is what makes the fog finding in §4 about the *rule* rather
than the threshold.

### 1.3 C8 — the benchmark had no held-out day data

`FIT_RUNS` excludes only `pohang01`, and `pohang01` is **entirely night**. So "clean fit-run day"
and "clean day" were the same 1200 frames: every day constant in this project — `cap_ir_scale` ×4
included — was selected and reported on one set, and the only genuinely held-out run is one where
VIS scores exactly 0.0000.

Frame counts: pohang00 **836** day, pohang02 **247**, pohang03 **117**, pohang01 **1032** night.

Added `TUNE_RUNS = (pohang00,)` and `TEST_RUNS = (pohang02, pohang03)`, split **by run** —
consecutive frames of one canal transit are near-duplicates and a frame split would leak. It earned
its keep immediately (C2).

---

## 2. B1 — the detector swap alone

`runs/eval/detector_swap_clean.md`. Clean cell, everything else identical.

| detector | VIS day | IR day | gated | gap | **VIS night** | macro |
|---|---:|---:|---:|---:|---:|---:|
| yolo26s | 0.3683 | 0.0181 | 0.3736 | +0.0052 | **0.0000** | 0.3368 |
| yolo26m | 0.3686 | 0.0192 | 0.3702 | +0.0017 | **0.0000** | 0.3241 |

Capability prior: VIS 0.3352→0.3233, ratio 142.4× → 134.9×. Classes IR cannot supply: none (26s,
nc=2) → **{buoy}** (26m, nc=1).

**Finding.** On the clean cell the swap buys nothing, and night VIS is *still* exactly 0.0000 with
both detectors — so half the benchmark stays single-sensor and no VIS/IR fusion property can be
tested there. (Night VIS being 0.0000 is a *dataset* decision: `filter_night_boxes.py --cut-dark`
removed 132k night boxes from the training labels on purpose.)

This is the number that made me tell you the detector wasn't the lever. §4 shows that conclusion was
drawn on the wrong cell.

---

## 3. The lever sweep

### 3.1 C1 — is cross-modal agreement even geometrically available?

Share of VIS day detections with a **same-class** IR detection above each IoU:

| homography | IoU≥0.85 | IoU≥0.55 | IoU≥0.30 | IoU≥0.10 |
|---|---:|---:|---:|---:|
| adopted | **0.05%** | 12.24% | 32.27% | 47.75% |
| + per-frame align | **4.02%** | 22.61% | 43.60% | 55.73% |

`iou_thr` is **0.85**. At that overlap the streams essentially never share a WBF cluster, so the
`min(n_models, n_cluster)/sum(weights)` agreement bonus almost never fires cross-modally. **Fusion
is ~99.9% concatenation**: VIS scaled by `w_vis`≈0.993, IR by `w_ir`≈0.007 and appended at the
bottom of the pooled ranking, where a few of them recover GT that VIS missed. This corrects a claim
I had made: the day-cell gain is *not* the consensus boost.

### 3.2 C2 — cross-modal `support` (adopted)

Score multiplied by `1+γ` when the other stream overlaps at `support_iou`; coordinates never
touched. IR says **whether**, not **where**. `runs/eval/support_26m.md`.

| arm | tune | **TEST** | day |
|---|---:|---:|---:|
| adopted | 0.3816 | 0.3533 | 0.3702 |
| support IoU 0.10 γ1.0 | **+0.0095** | +0.0009 (spans 0) | +0.0074 |
| **support IoU 0.30 γ0.5** | +0.0078 | **+0.0033** | **+0.0090** |
| support IoU 0.55 γ0.5 | +0.0121 | **−0.0010** | +0.0103 |

**The holdout's first job.** `support_iou` 0.55 wins the tune runs and is negative on the held-out
runs in **all six** variants it appears in; 0.30 is positive in all six. Under the old single-set
discipline the wrong one would have been adopted.

Night unmoved by 15 of 15 arms, as it must be — VIS is 0.0000 there, so there is no second stream to
confirm anything.

### 3.3 C3 — `consensus_beta` / `consensus_distinct` (rejected)

WBF pays a cluster both sensors saw 2× one only a single sensor saw, and nobody chose that factor.
Made explicit and tunable. Predicted inert by C1 and measured so: `consensus_distinct` −0.0068 on
the clean day cell, `consensus_beta` 2.0 −0.0084 on the held-out runs. Kept in the code,
defaults-off.

### 3.4 C4/C5 — per-frame registration (rejected)

`src/uqfusion/eval/iralign.py` estimates a per-frame IR→VIS translation from the two **detection**
sets (median offset of nearest-centre same-class pairs, greedy 1:1, ≥3 pairs, ≤40 px), composed as
`T @ H`. No GT — deployable.

Fires on **47%** of frames at a median **3.1 px**, independently reproducing the 3–6 px residual
`x_registration_drift.md` measured from GT. Raises agreement **80×** at IoU 0.85.

**Ship AP falls** (tune 0.3816 → 0.3773). Diagnosis and confirmation (`align_nomerge_26m.md`): the
new agreement arrives as WBF **clusters**, and merging drags the VIS box toward the worse-localised
IR box. Forbidding the merge (`iou_thr` 0.95) recovers tune −0.0043 → +0.0006 — the mechanism
confirmed. But every alignment arm is negative on the held-out runs (−0.0025 to −0.0026).

**80× more geometric agreement carries no usable information about which VIS boxes are right.** A
ceiling statement, not an estimator failure, and it rules out better registration, a tuned agreement
bonus, and a lower `iou_thr` together.

### 3.5 C6 — `class_veto` (rejected, and measured wrong first)

IR is nc=1, so a VIS veto deletes the only source of buoy boxes; ship AP — which every table reports
— is blind to it.

**My first implementation was wrong.** Keeping the vetoed stream *alive* so WBF could see its buoys
made the frame two-stream again, which disabled `single_passthrough` and rescaled the **survivor's
ship** scores — on vetoed frames only, so part of the pooled ordering moved and the rest did not,
for **−0.0065 of ship AP** to add buoys the ship metric cannot see. Re-implemented as a
**carry-through**: the veto still deletes the stream, and its boxes of the classes nothing else
supplies are appended unchanged. Smoke check K asserts the ship half is bit-identical.

Correctly implemented it is worth **exactly zero on every cell**: VIS is only ever vetoed on night
frames, and the night GT contains no buoys at all.

---

## 4. D1–D4 — the regression, and the repair

### 4.1 D1 — the 13-cell grid found it

`runs/eval/levers_26m.md`. Every arm, every cell, worst gap **−0.0632** — on `fog/clean`, and
identical for every arm, so not caused by any lever.

### 4.2 D2 — the diagnosis

| detector | VIS fog day | IR fog day |
|---|---:|---:|
| yolo26s | **0.0020** | 0.0181 |
| yolo26m | **0.0824** | 0.0192 |

**41×.** The veil axis fires on 100% of fog frames and hands them to IR. Against yolo26s that was
correct. Against yolo26m it destroys four times the AP it saves.

**No image statistic can detect this.** The threshold is a pixel statistic and is identical across
the swap to 0.000e+00 (§1.2). What stopped being true is the *claim attached to it* — "a fogged VIS
cannot see" — a property of the checkpoint downstream, not in the image.

### 4.3 D3 — the repair (adopted)

Deleting the axis is not the fix: fog lifts VIS `p05` above `mu_b` on 69% of night frames, so the
veil axis is exactly what covers the night arm's blind spot.

| rule | fog/day | fog/night |
|---|---:|---:|
| `veil OR (night AND dark)` (adopted) | 0.0192 | 0.0850 |
| no veil axis | 0.0848 | **0.0503** |
| **`night AND (dark OR veil)`** | **0.0848** | **0.0850** |

With `support` on top: fog/day **0.0192 → 0.0908**, `+0.0716 [+0.0657, +0.0784]`; fog/night
**+0.0000 [+0.0000, +0.0000]**; worst of eight cells −0.0632 → +0.0000.

Shipped as `preset="crossmodal26m"` — a **new** preset, so `crossmodal` still reproduces every
published number.

### 4.4 D4 — what the repair cost, on cells no grid contained

VIS fog × each IR condition (`veil_night_exposure_26m.md`). Every cache already existed; the
extended grid simply never paired corrupted IR with fogged VIS.

| cell | `crossmodal` day | `26m` day | `crossmodal` night | `26m` night |
|---|---:|---:|---:|---:|
| fog/clean | −0.0632 | **+0.0084** | +0.0000 | +0.0000 |
| fog/fog_s2 | −0.0816 | **+0.0004** | +0.0000 | −0.0000 |
| fog/glare_s2 | −0.0664 | **+0.0069** | +0.0000 | **−0.0140** |
| fog/blur_s2 | −0.0619 | **+0.0078** | +0.0000 | **−0.0296** |
| fog/noise_s2 | −0.0767 | **+0.0025** | −0.0000 | +0.0000 |

Five cells `crossmodal` fails badly and `crossmodal26m` passes. But putting the veil term *behind*
the night arm means the **authority bound can now disarm it**: on fogged night frames with a damaged
IR the arm never fires and blind VIS stays in. Worst-case false veto of a healthy day VIS: **1%**
(fog/glare_s2) — the authority bound holds in the dangerous direction.

---

## 5. E1–E4 — re-pricing every veto axis

`scripts/reprice_veto_axes.py`, `runs/eval/veto_axes_26m.md`. Two instruments; it took **both**.

### 5.1 E1 — the claim test

On exactly the frames an axis fires, `margin = AP(kept) − AP(removed)`. No fusion run. Negative =
the axis deletes the better sensor.

| axis | deletes | fires on | margin | status |
|---|---|---|---:|---|
| `veil (grad_gini)` | VIS | fog/day | **−0.0632** | repaired (§4). Also **+0.0555 on blur_s3**, which the repair disables at no measured cell cost |
| `ir_p05` raw | VIS | corrupted-IR day | **−0.3769** | already neutralised by `ir_ok` + two-of-two |
| `photometric` | VIS | lowlight/day | **−0.0243** | false, but can only fire behind the night arm |
| `ir_health` merge | IR | corrupted IR | +0.32…+0.37 | claim **holds** — see E2 |
| `ir_health` authority | vote | 2–100% | n/a | gate, priced by ablation |
| `vis_health` | gate | 1–100% | n/a | gate, priced by ablation |

### 5.2 E2 — the ablation, and the axis the claim test cannot catch

| arm | clean/glare_s2 | clean/blur_s2 | clean/noise_s2 | blur_s3/glare_s2 | clean/fog_s2 |
|---|---:|---:|---:|---:|---:|
| adopted | +0.0038 | +0.0071 | +0.0000 | +0.0006 | +0.0000 |
| **no `veto_ir`** | **+0.0094** | **+0.0093** | **+0.0038** | **+0.0022** | −0.0004 |

`veto_ir`'s claim is **true** and the veto is still a net loss: **+0.0128 summed day, +0.0047
held-out** (`veto_ir_decision_26m.md`, n_boot 1000).

> A subset AP asks *"is this stream worse?"* A veto needs *"is this stream a net **negative**?"* A
> stream 30× worse alone is still **additive** at the tail of a pooled ranking, because AP orders
> every frame's detections together and a few true positives at the bottom still raise it.

Under capability-only weights IR enters at `w_ir` ≈ 0.007, so a damaged IR was already nearly
harmless and the bound was deleting free recall. **Now off.**

Other ablations: `no veil axis` day-identical but night **−0.0347**; `no photometric` costs +0.0019
on one cell; `no vis_health gate` costs +0.0012 on one. **Cell-level oracle veil == adopted on every
day cell**, so the repair leaves no headroom there.

### 5.3 E3 — the authority bound is unpriceable on this grid

Loosening it to the merge bound and tightening it ×0.5 both give **identical numbers on all twelve
cells**. Its whole value lives in fog × corrupted-IR night cells, which only D4 contains.

### 5.4 E4 — a pre-existing bug, and what "confirmation" must mean

`clean/glare_s2` and `lowlight/glare_s2` are **−0.0093 at night under `crossmodal` too**. Nothing to
do with the fog repair; never looked at, because the original eight cells never damage IR and the
extended grid's headline was the day column.

`night_weak_fallback` lets a disarmed IR be **confirmed** rather than believed. What counts as
confirmation decides everything:

| fallback keyed on | lowlight/glare_s2 **day** | clean/glare_s2 night | fog/blur_s2 night |
|---|---:|---:|---:|
| nothing | +0.0038 | −0.0093 | −0.0296 |
| `VIS dark` | **−0.0072** | +0.0000 | −0.0141 |
| **`lap_over_var`** | **+0.0035** | **+0.0000** | **−0.0141** |

Keying on darkness re-opens §7.2 through the back door: lowlight/day `p05` is 0, *darker* than the
real night run, so a dark **sensor** and a dark **world** look identical to a brightness statistic.
`lap_over_var` — concentrated highlights on an empty field, an axis already fitted and used by **no**
preset — tells them apart (clean night 100%, clean day 0%, lowlight day 3%). Where fog destroys it
(0% on fogged night), `dark AND veil` stands in.

**A proof, not a tuning limit:** fogged **night** `p05` reaches 34 while clean **day** `p05` starts
at 21, so no single global photometric threshold can catch the first without vetoing clear daylight.
Under fog alone the separation is perfect (day min 36.0, night max 34.0) — but only under fog.

---

## 6. F1–F3 — the new instruments

### 6.1 F1 — the lift screen

`lift = P(TP | signal) / P(TP | no signal)`, per box, day frames. Seconds; no fusion run, no
bootstrap, no split. **A signal at lift 1.0 cannot help whatever weight it is given**, because
re-scoring by something uninformative preserves the ranking AP is computed from.
`runs/eval/signal_lift_26m.md`.

| signal | fires | lift@50 | lift@75 | corr(conf) |
|---|---:|---:|---:|---:|
| conf above its frame median | 48.5% | 4.80× | 4.92× | 0.72 |
| **sigma below its frame median** | 48.4% | **3.00×** | **3.39×** | 0.55 |
| cross-modal support IoU 0.30 | 32.3% | 2.08× | 2.63× | 0.31 |
| cross-modal support IoU 0.10 | 47.8% | 2.08× | 2.72× | 0.30 |
| cross-modal AND temporal | 22.3% | 1.98× | 2.66× | 0.28 |
| box larger than frame median | 48.5% | 1.39× | 1.50× | 0.24 |
| temporal support k5 | 84.7% | 1.06× | 1.37× | 0.03 |
| **temporal support k2** | 74.2% | **1.00×** | 1.29× | 0.02 |

Per-stream sigma lift: **VIS day 3.00× / 3.39×**, IR day 2.90× / **0.90×** (IR predicts presence but
not localisation — the registration story again), **IR night 10.86× / 3.55×**.

### 6.2 F2 — temporal support is a null

The exact cross-modal term with the modality axis swapped for time, on an axis that is *abundant*
here (frames are genuinely consecutive: `pohang00_L_006767`, `_006768`, …). Causal, gap-respecting.

| arm | VIS alone | gated | gap | TEST |
|---|---:|---:|---:|---:|
| no support at all | 0.3686 | 0.3702 | +0.0017 | 0.3533 |
| **cross-modal** support | 0.3686 | 0.3792 | **+0.0106** | 0.3565 |
| **temporal** support alone | 0.3681 | 0.3700 | +0.0019 | 0.3518 |
| both | 0.3681 | 0.3793 | +0.0112 | 0.3569 |

Fires on 74%, lift **1.00×**, worth ~0.0000.

> A persistent false positive — a dock edge, a reflection, a wake — is exactly what survives from
> frame to frame. Temporal consistency selects for **stable** detections, and in a fixed scene the
> false positives are the most stable things there are. A thermal signature is different physics; a
> reflection has no heat.
>
> **The value of a redundancy axis is its independence, not its abundance.**

This kills tracking, temporal smoothing and any spatio-temporal evidence field before they are
built, and it reframes the headline: fusion's small gain is not a failure to exploit an abundant
signal — it is the ceiling of the only *independent* signal available.

### 6.3 F3 — sigma in the score (rejected, after two of my own bugs)

**Two wiring bugs, the same species: a lever that is silently inert looks exactly like a lever that
does not help.**

1. The non-`sigma_weighted` path fed the local WBF a **dummy sigma column of ones**, so every day
   cell was byte-identical across α and only the passthrough branch moved.
2. It then used a **canvas-normalised** sigma where the 3.00× was measured on the
   **size-normalised** `per_box_uncertainty` — which merely penalises large boxes, and is what sent
   the first run's night cells the wrong way.

Smoke check L now asserts the term is exactly inert at α=0 and demonstrably live above it.

With that fixed, both streams together: **tune −0.0101 [−0.0114, −0.0086], held-out +0.0010 spanning
zero.** Split per stream:

| arm | day | night |
|---|---:|---:|
| sigma **VIS** only α0.1 | **−0.0102**…+0.0005 | +0.0000 |
| sigma **IR** only α0.25 | −0.0003…+0.0001 | **+0.0021** |

VIS's sigma is the entire cost, IR's the entire benefit — consistent with the lift table (sigma is
largely redundant with confidence on VIS, nearly orthogonal on IR).

**The control reverses it.** Applying the same scoring to `ir_only` takes it **0.0850 → 0.0871** at
α0.25 — *exactly* the fused system's night number:

| α | `ir_only` night | `ir_only` day |
|---:|---:|---:|
| 0.0 | 0.0850 | 0.0192 |
| 0.25 | **0.0871** | 0.0207 |
| 1.0 | 0.0877 | 0.0212 |

The bar moves with it and the gap stays +0.0000. **Sigma-in-score is a detector post-process, not a
fusion improvement** — it belongs beside `ir_nms`, where it is worth +0.0021 night / +0.0015 day on
the IR stream. Not adopted into the preset. This is the control `sweep_temporal_support.py` carried
from the start ("the bar is recomputed on the boosted streams"); here it had to be run separately,
and it changed the answer.

---

## 7. G1/G2 — the two inherited constants

`runs/eval/irnms_calib_26m.md`.

**G1 — `ir_nms` is not a lever.** Swept off / 0.5 / 0.6 / 0.7 / 0.8 / 0.9 — 15.7 to 30.0 IR boxes per
frame, a 2× range — and the day column is flat to **±0.0002** throughout. Night moves at most +0.0005
(0.6 over the adopted 0.7), the only place it can act since night is 100% IR. **Kept at 0.7 — now
because it was measured.**

**G2 — cross-modal score calibration is worse.** Isotonic `conf → P(TP@0.5)` per modality, fitted on
pohang00 only. Control passes exactly (`visible_only` moved by 0.0, as a monotone transform must).
Calibrated VIS/IR score ratio at raw conf 0.05/0.1/0.25/0.5: **4.4× / 7.0× / 10.4× / 4.1×**, against
the capability weights' 135×. Result: **tune −0.0044, TEST −0.0035.**

It also makes `cap_ir_scale` **completely inert** — ×1, ×4, ×16 give identical AP to four decimals,
where without calibration they give 0.3676 / 0.3702 / 0.3713. `replace(ctx, cap_ir=…)` was checked
directly and does work, so this is a property of the calibrated scores. *Stated as inference, not
measurement:* calibrated IR P(TP) occupies a narrow band (0.02–0.13 against VIS's 0.06–0.97), so
re-weighting slides it as a block, and once the block sits below the VIS boxes near the
precision–recall knee, sliding it further changes no ranking AP can see.

**A2 — `cap_ratio`** was also re-swept on 26m: ×8/×16 marginally better on clean and worse on
`clean × IR glare_s2`/`blur_s2`. The ratio moved only 142.4× → 134.9× across the swap. **Kept at ×4.**

---

## 8. H1 — the shipped system

`preset="crossmodal26m"` on `runs/cache_m` = capability-only weights + `night AND (dark OR veil)` +
`lap_over_var` night fallback + cross-modal support (0.30, 0.5) + `single_passthrough` + IR NMS 0.7 +
`cap_ir_scale` ×4 − `veto_ir`.

`runs/eval/final_26m_grid_v2.md`, ten extended cells, paired bootstrap n=500:

| cell | gated | bar | delta | 95% CI |
|---|---:|---:|---:|---|
| clean/clean | 0.3792 | 0.3686 | **+0.0106** | [+0.0086, +0.0124] |
| clean × IR glare_s2 | — | — | **+0.0094** | — |
| clean × IR blur_s2 | — | — | **+0.0093** | — |
| clean × IR noise_s2 | — | — | +0.0038 | — |
| clean × IR fog_s2 | — | — | **−0.0004** | — |
| blur_s3/clean | 0.0448 | 0.0425 | +0.0023 | [+0.0011, +0.0038] |
| noise_s2/clean | 0.0192 | 0.0192 | +0.0000 | IR better; ties |
| rain_s2/clean | 0.1137 | 0.1023 | **+0.0114** | [+0.0101, +0.0125] |
| lowlight × IR glare_s2 | — | — | +0.0035 | — |
| blur_s3 × IR glare_s2 | — | — | +0.0022 | — |

Against `crossmodal` on the same caches: better on **7** cells, worse on **2**. Worst day **+0.0000 →
−0.0004**, worst night **−0.0093 → −0.0081**.

**"Every cell at or above the bar" no longer holds exactly.** `clean × IR fog_s2` sits at −0.0004 —
the price of dropping `veto_ir` against +0.0126 summed elsewhere. Saying so is the point of keeping
the column; the alternative is a headline true only because one cell was tuned back.

---

## 9. Corrections to things I had stated

1. **"The full-scale retrain is the biggest lever."** Wrong on the clean cell (+0.0003) — and right
   on fog (41×), which is what made it a −0.0632 regression. The correct statement is that the
   detector matters enormously *where the gate has a claim about it*, and nowhere else.
2. **"Fusion's day gain is WBF's consensus boost."** Wrong. At `iou_thr` 0.85 only 0.05% of VIS boxes
   have an IR partner; fusion is ~99.9% concatenation.
3. **"Better registration should help."** Wrong. 80× more agreement, AP falls.
4. **`class_veto` measured before it was implemented correctly** (−0.0065 of ship AP from an
   implementation artefact). Correct implementation: exactly zero.
5. **Sigma-in-score measured twice while silently inert**, then measured as a fusion gain until the
   control showed it was a detector post-process.

---

## 10. Open

1. **Night is a dataset decision, not a fusion result.** VIS 0.0000 on all 1032 night frames with
   both detectors, by construction (`--cut-dark`). Any night claim is a claim about IR alone.
2. **The held-out day set is 364 frames.** It can reject an arm; it cannot yet certify one. A larger
   or differently-cut day holdout is the cheapest way to raise the standard on everything in §3.2.
3. **The authority bound is still unpriced** — inert on all 12 grid cells, and only exercised by the
   fog × corrupted-IR night cells of §4.4. Its ×0.5 and p90 variants remain untested end-to-end.
4. **`cap_ir_scale` ×4** was selected under the pre-§1.3 discipline and has not been re-selected
   under the run-disjoint one.
5. **The learned gate** is still in no headline table.
6. **Sigma as an IR post-process** (+0.0021 night, +0.0015 day) is measured but not integrated
   anywhere.
7. **One corruption draw per condition, one bootstrap seed.**

---

## 11. File index

**New code.** `src/uqfusion/eval/{iralign,tsupport}.py`; `scripts/{build_paired_grid_m.sh,
verify_cache_m, probe_detector_swap, sweep_support, sweep_align, eval_levers, sweep_irnms_calib,
reprice_veto_axes, probe_veil_night_exposure, sweep_temporal_support, probe_signal_lift,
smoke_new_levers}.py`.

**New results** (all under `runs/eval/`): `cache_m_verify`, `detector_swap_clean`, `support_26m`,
`align_26m`, `align_nomerge_26m`, `levers_26m`, `fog_veto_26m`, `veil_veto_repair_26m`,
`veil_night_exposure_26m`, `veto_axes_26m`, `veto_ir_decision_26m`, `night_weak_fallback_26m`,
`night_weak_fallback2_26m`, `signal_lift_26m`, `sigma_score_26m`, `sigma_score_perstream_26m`,
`irnms_calib_26m`, `final_26m_grid`, `final_26m_grid_v2`.

**New caches.** `runs/cache_m/` (14, canonical), `runs/cache_m_stageprobe/` (4).

**Commits.** `39fbf9b` → `6b1e5fb`, nine on `fusion-uq-phase3`.
