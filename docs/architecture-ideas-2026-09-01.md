# Architecture ideas — 2026-09-01, after the 26m swap

Where the remaining performance is, measured rather than guessed, and eleven
ideas ranked by what the measurements say they are worth. Companion to
`docs/experiment-log-2026-09-01.md` (the complete experiment record) and
`docs/levers-and-the-26m-swap-2026-09-01.md` (the argued record).

**Provenance warning — read before citing any number below.** Every measurement
in §1 was produced in one session with ad-hoc heredoc scripts that were *not*
committed. They are screens, not verdicts: one condition (clean), one split
(day), one seed, no bootstrap. Nothing here has the standing of a
`runs/eval/*.md` row. Before any of it is quoted in a headline table it must be
reproduced by a committed script under `scripts/`, with the bootstrap and the
run-disjoint discipline of §1.3 of the experiment log. They are recorded here
because they change which ideas are worth building, not because they settle
anything.

Nothing under `runs/` was written. All measurements read `runs/cache_m/` and
`Pohang_dataset/` only.

---

## 0. Summary — the reframe in one table

| | value |
|---|---:|
| shipped fusion gain, clean day (`final_26m_grid_v2`) | **+0.0106** |
| headroom from perfectly re-ranking the boxes VIS **already emits** | **+0.1060** |
| extra ship recall IR can supply in daylight, IoU 0.5 | **+0.0115** (123 boxes of 10663) |
| VIS day mAP50 vs mAP50-95 | 0.7844 vs **0.3233** |
| VIS day ship recall ceiling, IoU 0.50 → 0.75 | 0.791 → **0.460** |
| labelled VIS frames on disk vs frames in the day fit set | **127,309** vs **836** |

Read together: the cross-modal channel is near its ceiling and the
**localisation** channel is not. Everything the gate does — vetoes, weights,
support, calibration — decides *which stream*, and is worth ~+0.01. The
measurable headroom is in *which box, and how tight*, and is worth ~+0.11. That
is the axis nothing in the system currently acts on.

---

## 1. The measurements

All figures: `runs/cache_m/gauss_{vis,ir}_paired_clean.pkl`, day frames only
(pohang00/02/03, n=1200), macro over both classes unless stated. The VIS-alone
macro reproduces `detector_swap_clean.md`'s 0.3241 to 0.3233 and its ship column
to 0.3686 exactly, which is the check that this is the same quantity.

### 1.1 The oracle re-ranking ceiling

Keep exactly the boxes VIS emits; reorder them perfectly (sort by IoU with GT,
per class, per IoU level). This is the upper bound on **every** score-based
intervention — the support term, the capability weights, calibration, sigma in
the score, and any future re-scorer, all at once.

| | mAP50 | mAP50-95 |
|---|---:|---:|
| actual | 0.7844 | **0.3233** |
| oracle re-rank | 0.8317 | **0.4293** |
| headroom | +0.0473 | **+0.1060** |

Per IoU level — the gain is not at the ends, it is in the middle:

| IoU | actual | oracle | gain |
|---:|---:|---:|---:|
| 0.50 | 0.7844 | 0.8317 | +0.0473 |
| 0.55 | 0.6878 | 0.7723 | +0.0845 |
| 0.60 | 0.5322 | 0.6733 | +0.1411 |
| 0.65 | 0.4065 | 0.5743 | +0.1678 |
| **0.70** | 0.3171 | 0.4950 | **+0.1779** |
| 0.75 | 0.2257 | 0.3919 | +0.1661 |
| 0.80 | 0.1397 | 0.2772 | +0.1376 |
| 0.85 | 0.0890 | 0.1683 | +0.0793 |
| 0.90 | 0.0481 | 0.0891 | +0.0410 |
| 0.95 | 0.0023 | 0.0198 | +0.0175 |

Per class, and the recall ceiling that explains the shape:

| class | n_gt | n_pred | AP50 | AP50-95 | recall@50 | recall@75 | recall@95 |
|---|---:|---:|---:|---:|---:|---:|---:|
| 0 ship | 10663 | 18648 | 0.7160 | 0.3686 | 0.791 | **0.460** | 0.026 |
| 1 buoy | 600 | 1246 | 0.8528 | 0.2780 | 0.878 | **0.330** | 0.003 |

**42% of the ships VIS finds are localised too loosely to score at IoU 0.75.**
mAP50 0.784 against mAP50-95 0.323 is that fact restated. The loss is not
detection; it is precision of the boxes.

### 1.2 The cross-modal ceiling

Union recall for ship, day frames, per-run `H_ir_canvas_to_vis_canvas` from
`runs/derived/homography_ir_to_vis.json`:

| | VIS | IR | union | IR adds |
|---|---:|---:|---:|---:|
| IoU 0.50 | 0.792 | 0.222 | 0.803 | **+0.0115** (123 of 10663) |
| IoU 0.30 | 0.887 | 0.432 | 0.908 | +0.0205 |

**IR can supply at most ~1% more ships in daylight.** C1's "fusion is 99.9%
concatenation" was never a fusion bug — the recall overlap is that total, and
E2's cell-level oracle (+0.0007) was measuring the same ceiling from a different
side. The shipped +0.0106 is a large fraction of what this channel physically
contains at this IoU.

### 1.3 σ is calibrated, and has never been allowed to act

Per-edge σ against true |edge error|, on 12,517 detections matched to GT at
IoU ≥ 0.5:

| edge | pearson | spearman | mean σ (px) | mean err (px) |
|---|---:|---:|---:|---:|
| x1 | 0.378 | 0.399 | 1.13 | 1.47 |
| y1 | 0.598 | 0.380 | 1.09 | 1.73 |
| x2 | 0.640 | 0.348 | 1.24 | 2.13 |
| y2 | 0.629 | 0.371 | 0.58 | 0.57 |

The head works: right correlation *and* right magnitude. But as a **box-level**
quality signal it is weak —

| signal | spearman vs IoU-with-GT |
|---|---:|
| mean σ | **0.162** |
| conf | 0.366 |

**σ predicts edges, not boxes.** That is the missing sentence under F3: the
score-multiplier framing asked σ a question it cannot answer, so the rejection
was correct and the conclusion drawn from it ("σ is a detector post-process")
was too broad.

### 1.4 The duplicate population, and why σ has never had a cluster

**9,273 of 19,894 VIS day boxes (46.6%) have a same-class sibling at IoU ≥ 0.6.**
Two things keep them apart:

1. `iou_thr` is 0.85, so nothing between 0.6 and 0.85 ever forms a WBF cluster;
2. `single_passthrough` returns a one-stream frame *before* WBF is called at all
   (`fusion.py`, the `if single_passthrough:` branch).

So the inverse-variance coordinate averaging that `sigma_weighted_fusion` exists
to perform — the mechanism this project is named for — has **never run on a
multi-member cluster in any published number.**

### 1.5 So I ran it. Within-modality WBF on VIS day

Baseline (no within-modality merge — the shipped behaviour): all **0.3233**,
mAP50 0.7844, tune 0.3810, TEST 0.3151, 16.6 boxes/frame.

| iou_thr | σ-weighted | mAP50 | mAP50-95 | Δ | tune | TEST | Δ TEST | bx/fr |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0.50 | no | 0.7898 | 0.3187 | −0.0046 | 0.3762 | 0.3126 | −0.0024 | 11.4 |
| 0.50 | yes | 0.7863 | 0.3183 | −0.0050 | 0.3758 | 0.3126 | −0.0024 | 11.4 |
| 0.60 | no | 0.7892 | 0.3192 | −0.0041 | 0.3750 | 0.3143 | −0.0008 | 12.1 |
| **0.60** | **yes** | 0.7890 | 0.3191 | −0.0042 | 0.3747 | 0.3145 | −0.0006 | 12.2 |
| 0.70 | yes | 0.7871 | 0.3183 | −0.0050 | 0.3726 | 0.3140 | −0.0011 | 13.0 |
| 0.80 | yes | 0.7878 | 0.3184 | −0.0049 | 0.3715 | 0.3140 | −0.0011 | 13.9 |

**C4/C5's failure mode is not cross-modal — it is universal.** mAP50 rises
(duplicates removed, precision up) while mAP50-95 falls (averaging drags the
well-localised box toward the badly-localised sibling). σ-weighting moves the
result by ±0.0002 against plain averaging.

> C4 concluded that merging fails because the IR box is worse-localised and the
> registration residual makes it worse still. That explanation was too specific.
> Merging fails here because the **best member is already the best estimate**,
> and any average is a move away from it. Two boxes from the same detector on
> the same frame, in perfect registration, lose just as much.

Suppression instead of merging (greedy NMS, keep the best member, drop the rest):

| rule | mAP50 | mAP50-95 | Δ | tune | TEST | Δ TEST | bx/fr |
|---|---:|---:|---:|---:|---:|---:|---:|
| NMS 0.50 by conf | 0.7859 | 0.3220 | −0.0013 | 0.3845 | 0.3114 | −0.0036 | 11.4 |
| NMS 0.60 by conf | 0.7887 | 0.3227 | −0.0006 | 0.3843 | 0.3127 | −0.0023 | 12.2 |
| NMS 0.70 by conf | 0.7890 | 0.3233 | +0.0000 | 0.3843 | 0.3143 | −0.0007 | 13.0 |
| NMS 0.80 by conf | 0.7910 | 0.3243 | +0.0010 | 0.3849 | 0.3150 | −0.0000 | 13.9 |
| NMS 0.60 by conf/(1+σ) | 0.7890 | 0.3227 | −0.0006 | 0.3841 | 0.3128 | −0.0023 | 12.2 |
| NMS 0.70 by conf/(1+σ) | 0.7890 | 0.3233 | +0.0000 | 0.3842 | 0.3143 | −0.0007 | 13.0 |

Flat. Choosing the survivor by σ instead of conf changes nothing (±0.0001).
**Neither merging nor suppressing the duplicates is where the +0.1060 lives** —
the duplicates are a precision problem at IoU 0.5, and the headroom is a
localisation problem at IoU 0.65–0.80.

### 1.6 The learned re-ranker — the largest movement in the project, and pure overfit

`HistGradientBoostingRegressor` (300 iters, depth 6, lr 0.06), 18 features all
already in the cache — conf, log conf, σ_ltrb (4), mean σ, size-normalised σ,
w, h, area, aspect, x/y position, frame box count, conf ÷ frame-median conf, max
same-class sibling IoU, sibling count ≥0.5. Target: max IoU with same-class GT.
Fit on pohang00. Re-score `conf^(1−λ) · predIoU^λ`; coordinates untouched.

| λ | mAP50 | mAP50-95 | Δ | **tune** | Δ tune | **TEST** | Δ TEST |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.00 | 0.7844 | 0.3233 | — | 0.3810 | — | 0.3151 | — |
| 0.25 | 0.7869 | 0.3345 | +0.0112 | 0.4096 | +0.0286 | 0.3127 | −0.0023 |
| **0.50** | 0.7822 | 0.3370 | +0.0138 | **0.4229** | **+0.0419** | 0.3105 | −0.0045 |
| 0.75 | 0.7593 | 0.3319 | +0.0086 | 0.4316 | +0.0506 | 0.3010 | −0.0140 |
| 1.00 | 0.6343 | 0.2809 | −0.0424 | 0.3730 | −0.0080 | 0.2580 | −0.0570 |

**+0.0419 on tune, −0.0045 on TEST.** The largest single movement anything has
produced in this project, and it does not transfer. The holdout added in C8 has
now earned its keep three times.

This is a **data** result, not a ceiling result. The tune set is 836 frames of
one consecutive canal transit — near-duplicate neighbours, so the effective
sample count is a small fraction of that. λ=1.00 losing on tune as well confirms
the model is not merely memorising noise: it has real signal (it beats conf
in-sample by a wide margin) that is scene-specific.

### 1.7 The data that exists

VIS frames with labels on disk:

| run | VIS frames | IR counterpart | in paired day set |
|---|---:|---|---|
| pohang00 | 21,768 | yes | 836 (TUNE) |
| pohang01 | 24,473 | yes | 1032, **all night, VIS ≡ 0.0000** |
| pohang02 | 27,795 | yes | 247 (TEST) |
| pohang03 | 27,085 | yes | 117 (TEST) |
| **pohang04** | **26,188** | **none** | **never used in any experiment** |
| total | **127,309** | | **1200 day** |

The day fit set is **0.7%** of the labelled day frames available. §10.2 of the
experiment log ("the held-out day set is 364 frames") is not a property of the
dataset — it is a property of the paired stride subset. `pohang04` in particular
is a complete, independent, labelled day run that has never entered anything,
and its lack of an IR counterpart is irrelevant to every VIS-side idea below.

---

## 2. What the measurements rule out

Stated plainly so these are not re-proposed:

1. **A better cross-modal fusion operator.** IR adds 1.15% recall in daylight.
   No operator recovers more than the union contains.
2. **Averaging box coordinates, in any configuration.** Cross-modal (C4/C5),
   within-modal (§1.5), σ-weighted or not. The best member is the best estimate.
3. **σ as a box-level score multiplier.** Spearman 0.162 against conf's 0.366
   (§1.3) — F3's rejection, now with the reason.
4. **Duplicate suppression as a lever.** Flat to ±0.001 across a 0.5–0.8 sweep.

---

## 3. The ideas

### Tier 0 — the blocker

**I0. Rebuild the day fit/test substrate.** Everything in Tier 1 is a supervised
estimator, and §1.6 shows the current substrate cannot support one. Concretely:

- resample the day runs at a wider stride to get 5–10k day frames, not 1200;
- bring `pohang04` in as a fourth day run — VIS-only arms do not need its IR;
- **leave-one-run-out CV, refitting inside each fold**, so all four day runs
  serve as both fit and test rather than 836/364;
- more than one corruption draw and bootstrap seed (experiment log §10.7).

This is also the honest fix for adopting knobs at ±0.003 against CIs of ±0.002.
It gates I1 and I2, and it is the cheapest credibility upgrade available.

### Tier 1 — attack localisation (the +0.1060)

**I1. Learned IoU-aware re-ranker.** §1.6, refit on the I0 substrate. Guardrails
the pilot did not have: constrain monotone-increasing in `conf` so it can only
reorder *within* confidence bands and cannot destroy the base ranking; cut to
≤8 features; shallow trees; select λ by leave-one-run-out, never on a single
tune set. Capturing 20% of the oracle is +0.021 — twice the whole shipped fusion
system. It also **subsumes the adopted cross-modal support term as one feature**,
so it generalises what already works instead of competing with it.
*Kill criterion:* if it is still negative on held-out runs at 5–10k day frames
under LORO, the +0.1060 is not reachable from cache features and Tier 1 closes.

**I2. Use σ to move edges, not to score boxes.** §1.3 says σ predicts per-edge
error at pearson 0.38–0.64 with the right magnitude, and nothing in the system
has ever adjusted a coordinate with it. Fit a per-edge correction
`x1' = x1 + f(σ_l, w, h, conf)` on the fit runs — linear or isotonic is enough
to start. This is the one intervention that acts on localisation *directly*
rather than reordering around it, and §1.5 shows reordering is not where the
gain is.
*Screen first, ~30 seconds:* regress the **signed** residual, not its magnitude.
If only |error| is predictable and its sign is not, the correction has no
direction to move in and the idea is dead before it is built.

**I3. Give σ independent estimates to average — the two ways that exist.**
§1.4 explains why the averaging mechanism has never had a cluster. Two sources
of genuinely independent boxes:

- **(a) the one2many branch**, discarded at inference by design
  (`gaussian.py:30`, for a defensible σ-calibration reason). YOLO26's o2o head
  emits 8.9 boxes/frame; o2m + NMS is the higher-mAP path and gives multiple
  estimates per object.
- **(b) test-time augmentation** — hflip and two scales on VIS. Pixel-exact
  co-registration, so C4's registration-residual explanation is absent by
  construction.

*Both carry the same warning:* §1.5 measured averaging as negative even under
perfect registration. Screen both with `probe_signal_lift.py` before building —
does "the other pass also fires here" carry lift ≥ 2×? If the answer is the
same 1.00× that killed temporal support (F2), these are dead too, and the
redundancy-is-independence rule will have predicted it a second time.

### Tier 2 — attack recall (VIS misses 21% of ships; IR can supply 1%)

**I4. Input resolution.** `image_hw` is (640, 640) — the dataset was pre-resized
on disk. Maritime targets are small and long-range, and
`scripts/prepare_pohang_fullres.py` + `scripts/prep_fullres_lists.py` exist and
have never been used. *Screen first:* report AP@small/medium/large on the
existing caches. If the loss concentrates in small boxes, re-prepping at 960 or
1280 is plausibly worth more than the entire fusion layer. Expensive — a full
re-prep and retrain — but it is the largest single number on this list.

**I5. Night is architecturally dead, and half the benchmark with it.** VIS is
exactly 0.0000 on all 1032 night frames because `filter_night_boxes.py
--cut-dark` removed 132k boxes from the **training labels**. The consequence is
not only that night is single-sensor: the gate's entire night apparatus — the
photometric axis, the `night AND (dark OR veil)` repair, the `lap_over_var`
fallback, the authority bound — exists to defend a stream that is identically
zero. A night fine-tune, or restoring those boxes as ignore-regions rather than
background, would make 46% of the benchmark testable for fusion for the first
time and would finally let the authority bound be priced (experiment log §10.3).

**I6. Buoy is half the macro metric and nothing has ever touched it.** AP50-95
0.2780 against ship's 0.3686, on 600 GT against 10,663 — so per GT box it is
18× more leveraged on the macro number. IR is nc=1 and can never help it. Every
gate constant, veto, weight and the support term are class-agnostic. Run the
§1.1 oracle decomposition per class; if buoy's headroom is disproportionate, a
buoy-specific path is uncontested territory worth as much per AP point as
anything on the ship side.

### Tier 3 — fusion layer, still worth doing

**I7. Make `cap_ir_scale` gate-conditional.** A2 found ×8/×16 better on clean
and worse on `clean × IR glare_s2`/`blur_s2`. A global constant is *forced* to
compromise between those cells; the gate exists precisely to resolve that
conflict, and it is not consulted. Condition the scale on `ir_ok` / IR health
rather than pinning it at ×4. Follows directly from A2's own table.

**I8. Two-checkpoint VIS ensemble.** Both `gauss_vis_seed0` base (paired val
0.3686) and ft (0.3706) are on disk; D31's rule picked base and the log already
records that this cost −0.0020. F2's temporal null does **not** apply: two
checkpoints disagree on *false positives* in a way one checkpoint across
consecutive frames cannot, which is exactly the independence F2 identified as
the thing that matters. Screen with `probe_signal_lift.py`. Nearly free — both
weights already exist.

**I9. Separate the merge graph from the support graph.** One `iou_thr` currently
serves two incompatible jobs: cross-modal merging (which must not happen) and
within-modality clustering (which §1.5 says should also not happen). Make them
independent parameters. The likely outcome is that merging is off everywhere —
but right now that is an accident of the value 0.85, not a decision, and it is
not stated as one anywhere.

---

## 4. Dependency order

```
I0  substrate (more day frames, pohang04, LORO)
 |-> I1  learned re-ranker          <- largest measured headroom
 \-> I2  sigma -> edge correction   <- screen: signed residual?
I3  o2m / TTA          <- screen: probe_signal_lift, lift >= 2x?
I4  resolution         <- screen: AP@small/medium/large
I5  night labels       <- unblocks the authority bound (log 10.3)
I6  buoy               <- screen: per-class oracle
I7  cap_ir_scale gate-conditional   } independent, cheap,
I8  two-checkpoint VIS ensemble     } can run in parallel
I9  split merge/support iou_thr     } with anything above
```

Four of these open with a screen costing seconds to minutes (I2, I3, I4, I6),
and each screen can kill its idea before any training run. That is the F1 lift
screen's lesson applied forward.

---

## 5. Caveats

1. **§1.5 and §1.6 are single-condition, single-seed, unbootstrapped.** Clean,
   day, one corruption draw. They are screens.
2. **§1.6's reported spearman (pred 0.835 vs conf 0.583) is contaminated** — it
   was computed over all day frames including the fit run. The AP columns are
   clean; that one correlation pair is not, and should not be quoted.
3. **I4 and I5 both cost real training time**, unlike everything else here.
4. **The oracle of §1.1 is an upper bound on re-scoring only.** It says nothing
   about what a *feasible* estimator reaches — §1.6 is the first evidence on
   that, and it is currently negative out of sample.
5. **Every number here is day-only.** Night is excluded throughout because VIS
   is 0.0000 there, which is I5's whole point.
6. **No committed script reproduces §1.** That is the first debt to pay if any
   of these ideas is pursued.
