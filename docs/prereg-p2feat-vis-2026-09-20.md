# Pre-registration — does `p2feat` improve the VIS detector?

Written 2026-09-20, **before any `yolo26m-p2feat` VIS weight exists**, while the pohang04
single look is scoring in another process. Nothing in this document reads a Phase 3 score,
and nothing it produces may be merged into a Phase 3 artifact (§7).

Registers a **two-arm trained-model comparison** — the one class of question Amendment 4
of [`prereg-phase3-retrain-2026-09-10.md`](prereg-phase3-retrain-2026-09-10.md) cut and
forbade reporting under that pre-registration. A4.4 requires "a new pre-registration and
the seed budget above" before such a comparison may exist. This is that new
pre-registration. The seed budget is **not** met, and §5 states in advance exactly what
that costs.

---

## 1. The question

**Does injecting backbone P2/4 detail into the P3 neck feature improve the VIS detector,
relative to the stock `yolo26m` VIS detector already trained at five seeds in Phase 3
Stage 2?**

Detector-level only. Fusion is out of scope (§7).

## 2. Why it is worth asking, stated before the answer

`p2feat` is a validated lever **on IR and only on IR**. It was adopted as IR's neck
decision (D28), and [`architecture-final-2026-08-20.md`](architecture-final-2026-08-20.md)
records it as "the single biggest lever found": `ir_only` night 0.0810 → 0.1087, **+0.0277**.
[`configs/models/yolo26m-p2feat.yaml`](../configs/models/yolo26m-p2feat.yaml) says so in its
own header — "p2feat is the separate, orthogonal **IR** neck decision". VIS has run stock
`yolo26m` throughout, including all five Phase 3 seeds. It has never been tried on VIS.

The prior for VIS is arguably stronger than the one that justified it for IR.
[`TODO-improvements.md`](TODO-improvements.md) §B2 measured VIS median ship at **14 px at
model input**, with **86.9%** of ships under 32² and **58.3%** under 16². A stride-4 branch
exists for exactly that regime.

**Counter-prior, recorded so the result is not read as a surprise either way.** Detector
capacity is a measured null on this data — `yolo26s` → `yolo26m` bought **+0.0003**
(`project-detector-scale-null`). If `p2feat` acts as capacity rather than as detail
injection, the expected effect is zero.

## 3. Why `p2feat` and not `p2`

[`architecture-review-2026-09-09.md`](architecture-review-2026-09-09.md) line 350 flags that
the project's P2 record confounds head width with transferred weights, and that "a causal
P2-head conclusion needs matched initialization/capacity controls."

`p2feat` is the variant that survives that objection. Per `VARIANT_SPECS["yolo26m-p2feat"]`
in [`src/uqfusion/uq/variants.py`](../src/uqfusion/uq/variants.py): `det_level_shift: 0`,
Detect stays 3-level, "anchor count and every head width are unchanged", `min_transfer: 97.0`.
`yolo26s-p2` by contrast drops `ch[0]` 128 → 64 and would halve the variance branch.

Two consequences fixed now:

* **`sigma_width` is unchanged** between arms. Head widths are identical, so the Gaussian
  branch is not a hidden variable. This is the main reason `p2feat` is testable and `p2` is not.
* **`p2feat` is not free of capacity**: it adds a Conv + Concat + C3k2 block. The parameter
  count and GFLOPs of both arms are **reported in the results table**. A positive result
  proportionate to a capacity increase is reported as such, not as a detail-injection win.

## 4. Design

### 4.1 Arms

| arm | variant | seeds | status |
|---|---|---:|---|
| control | `yolo26m` | 0, 1, 2, 3, 4 | **already trained**, `runs/phase3_stage2/p3_vis_seed{k}` |
| treatment | `yolo26m-p2feat` | 0, 1, 2 | to train |

The control arm is reused, not retrained. It is recipe-matched by construction: identical
queue defaults, identical data yaml, same machine, same code path.

### 4.2 Recipe — cloned from the Stage 2 queue, one field changed

Queue spec, verbatim from `runs/queue_phase3_stage2/queue.json` with `variant` substituted:

```json
{"id": "p2f_vis_seed0", "kind": "gaussian", "sigma": true,
 "variant": "yolo26m-p2feat",
 "data": "runs/derived/data_vis_stride2_p04out.yaml", "seed": 0}
```

Defaults, unchanged: `imgsz 640, epochs 100, patience 20, batch 12, workers 8`,
`assert_holdout: true`, `out_subdir` → `p2feat_vis` (a new tree; Stage 2's is frozen in the
pohang04 manifest and must not be written into).

`kind: gaussian, sigma: true` is kept so the arms are the same *kind* of object. **Batch 12
is inherited, not re-measured.** `p2feat` adds a stride-4 branch; if measured reserved memory
leaves less than the 1.5 GiB driver margin of
[`batch-ceiling-2026-09-10.md`](batch-ceiling-2026-09-10.md), the batch is dropped and **the
change is recorded here before the grid runs**, because batch is a recipe field and dropping
it un-matches the arms. A batch change makes this a recipe comparison and the verdict labels
in §6 no longer apply.

### 4.3 What is measured

Scored on the development substrate only, per the shipped convention:

* **TUNE_RUNS** = pohang00. **TEST_RUNS** = pohang02 + pohang03 pooled.
* **Primary:** ship AP (class 0) on TEST_RUNS, day, clean.
* **Declared now, before any number exists** (TODO §D.3 — "Class-set choice must precede
  the results"): **ship AP, buoy AP and macro AP are all reported, always, in every table.**
  Buoy is the pre-declared mechanism probe — `p2feat`'s mechanism is small objects, so a
  ship-only gain with no buoy movement is evidence against the stated mechanism even if the
  primary passes. Buoy carries ~75% of macro variance (`project-metric-noise-floor`), so its
  interval will be wide; that is expected, not a defect.
* **AP convention:** local linear-interpolation AP, not COCO. Absolutes are convention-bound;
  deltas are convention-safe.
* Per-seed values and their sd are reported. The sd is descriptive, not a decision input.

### 4.4 Intervals — two of them, only one decides

1. **The decision interval is between-seed.** A4.1 measured VIS between-seed
   sigma = **0.00638** (df 4), and states the reason plainly: evaluation noise over one fixed
   checkpoint "would understate the MDE and pass arms that resolve nothing — R-F3's error,
   run forwards." Two independently trained architectures do not share training noise, so it
   does not cancel here.
2. **A paired moving-block bootstrap over frames** (L = 20, n_boot = 1000, seed 0, seed-mean
   statistic, both arms scored on the same frames) is **also** reported, labelled
   *evaluation noise only*. It is not the verdict and may not be quoted as one.

---

## 5. What three seeds can and cannot resolve — computed before the run

Using A4.1's sigma and its implied multiplier (3.1995, recovered from its own
MDE @ 5 seeds = 0.01291):

| allocation | MDE |
|---|---:|
| 5 control vs **3** treatment (**this design**) | **0.01491** |
| 3 vs 3 (if the control were retrained) | 0.01667 |
| 5 vs 5 | 0.01291 |
| seeds *per arm* to reach the §8 floor 0.0060 | **24** |

Reusing all five control seeds instead of three buys 0.0167 → 0.0149. That is why the
control arm is five and not three.

**Read this honestly:**

* The design **cannot** resolve a floor-sized effect. 0.0060 needs ~24 seeds per arm
  (~5 days of VIS training per arm on this laptop). That budget is not being spent, by decision.
* The design **can** resolve an IR-sized effect. The observed IR `p2feat` gain was +0.0277,
  which is **1.86×** this design's MDE. If `p2feat` does for VIS what it did for IR, three
  seeds will see it.

That asymmetry is the whole justification for running three. This is a screen for a large
effect, pre-registered as a screen, and it is **not** a test of whether `p2feat` helps a little.

---

## 6. Decision rule, fixed here

Let `δ = AP(p2feat) − AP(yolo26m)`, seed means, primary metric of §4.3, and let
**M = 0.01491** (§5), recomputed and reported to 5 dp against the realised seed counts if any
run is lost.

* **P2FEAT-VIS-POSITIVE** — `δ ≥ M` **and** the between-seed 95% interval of δ lies entirely
  above zero.
* **P2FEAT-VIS-NEGATIVE** — `δ ≤ −M` **and** that interval lies entirely below zero.
* **P2FEAT-VIS-UNRESOLVED** — anything else. **This is the expected outcome under a
  floor-sized true effect, and it is not a null.**

**UNRESOLVED may not be written, spoken or plotted as "no effect", "no difference", "p2feat
does not help VIS", or "equivalent".** `project-ir-ladder-underpowered` records precisely this
error: p = 0.45 on an underpowered ladder is not evidence of absence, and the Tukey upper bound
there ruled out the zero delta the p-value had been read as supporting. An UNRESOLVED result is
reported as *"a VIS effect as large as IR's +0.0277 is ruled out at this power; anything
smaller than 0.0149 is not measured here."*

Counts at all four §8 floors (0.0014 / 0.0031 / 0.0060 / 0.0100) are reported **descriptively**,
as is project convention, and carry no verdict at this power.

**No promotion after the fact.** If the primary is UNRESOLVED and buoy AP passes, that is
reported as a descriptive observation and a candidate for its own pre-registration. It does not
become the verdict.

---

## 7. Scope fences

* **pohang04 is untouchable.** The data yaml is `data_vis_stride2_p04out.yaml` and the queue
  runs `assert_holdout: true` before every invocation. No arm here is ever scored on pohang04.
  The Phase 3 single look covers five stock-`yolo26m` VIS seeds; a `p2feat` VIS detector is a
  **different system**, and scoring it on pohang04 would be a second look requiring its own
  pre-registration recording that the set is no longer held out (§7.2).
* **Nothing here re-opens Phase 3.** Stage 2 is closed at 5 + 5 seeds (Amendment 6). The
  `runs/phase3_stage2/` tree is hash-frozen in `holdout_p04_freeze_manifest.json` and is
  **read-only** for this work; the treatment arm writes to `runs/p2feat_vis/`.
* **No fusion number.** Fusion would add the IR arm's noise to a comparison that cannot afford
  it, and would change the shipped system. If this screen returns POSITIVE, the fusion question
  is a separate pre-registration.
* **The paper's shipped system does not change on this result.** The shipped configuration is
  frozen; a POSITIVE here is a forward-looking finding, reported as such.

## 8. Stop rules

1. Any arm that diverges is rerun once at the same seed, and the rerun's provenance is recorded
   (the Amendment 6 precedent). A second divergence at the same seed stops the grid.
2. If batch must drop below 12 (§4.2), the grid stops until this document is amended.
3. If `min_transfer` (97.0%) fails at weight load, the grid stops. A low-transfer arm is a
   different initialization and §3's matched-init argument no longer holds.
4. Partial grids are not scored. Three treatment seeds, or an amendment.

## 9. Provenance

* The exposure is logged in [`exposure-ledger-2026-09-09.md`](exposure-ledger-2026-09-09.md)
  §7 **before** the first scoring invocation, per project convention. Training is not an
  exposure; scoring the development substrate is.
* This document is committed **before** the first `p2feat` VIS weight is written.
* Amendments are appended, never edited in place.

## 10. Cost

`p3_vis_seed0` ran 37 epochs in 11,915 s ≈ **3.3 h**. `p2feat` adds a stride-4 branch, so
budget **~4–5 h per seed**, **~12–15 h** for three, assuming batch 12 holds and no divergence.
