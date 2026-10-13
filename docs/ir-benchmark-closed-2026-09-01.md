# IR architecture benchmark — closed unfinished (2026-09-01)

**Decision.** Stop `runs/queue_ir_benchmark_stride4` at 44/93 runs. The 44 completed
runs are kept and citable; the remaining 49 will not be run. Nothing deleted, queue
stays resumable — reversible if a reviewer asks.

## Why

**1. Cannot change any open decision.** IR architecture was frozen 2026-08-20
(`yolo26m-p2feat`); the queue was created 2026-08-26, six days *after* the choice it
was meant to inform. Every downstream artifact (σ-head, ensemble, MC arm, all of
Table 3) is already trained on the frozen architecture.

**2. It measures nothing.** Across the 13 variants reaching all three seeds, one-way
ANOVA: **F(12, 26) = 1.037, p = 0.45**. Between-variant spread of means 0.0119
(yolo26l 0.0732 → yolov10n 0.0612) vs mean within-variant seed sd **0.0047**. Thirteen
architectures spanning 2.6–68 M params and 8.7–258 GFLOPs are statistically
indistinguishable. The remaining 49 runs would only tighten error bars around a null.

**3. It doesn't measure the production config anyway.**
  - Ladder trains 2-class `data_ir_stride4.yaml`; production IR is **nc=1 ship-only**
    (`data_ir_shiponly.yaml`, D28/A-1 — IR buoy AP 0.00019 against 29,131 buoy
    detections for 596 GT boxes).
  - **`yolo26m-p2feat`, the deployed architecture, is not in the ladder.** All 31
    variants are stock backbones; the P2 feature-level change IR depends on appears
    in none of them.

Even a decisive result would be about a different model on a different label set.

## What the 44 runs are good for

A negative result worth one paragraph: **on Pohang IR, detector architecture is not a
lever.** Absolute mAP@50-95 is 0.061–0.073 for every variant — an order of magnitude
below VIS — locating the difficulty in the modality and small-object regime, not the
backbone. More useful than a ranking.

Compiled: `phase1_benchmark/compiled/ir_benchmark_stride4_variants.csv` (per-variant),
`ir_benchmark_stride4_runs.csv` (per-run).

## State on disk

`runs/queue_ir_benchmark_stride4` — 44 done, 2 paused, 2 diverged, 45 pending;
`control.json` left `paused: true` with a note pointing here. `run_queue.py resume`
restarts exactly where it stopped.

Supersedes nothing. Related: `docs/ir-benchmark-divergence-falsepos-2026-08-26.md`
(divergence-alarm false positives in this queue), `docs/vis-benchmark-stride4-2026-08-27.md`
(VIS counterpart, also paused).

---

## Amendment 1 — 2026-09-10 (R-F3 / F15): reason 2's statistics are wrong

**The decision stands. Reason 2's justification does not.** The text above is left exactly as recorded on 2026-09-01; this amendment is appended rather than edited in, so the reasoning that was actually used on the day stays readable.

Reason 2 concluded from **F(12, 26) = 1.037, p = 0.45** that thirteen architectures are *“statistically indistinguishable”* and that the ladder *“measures nothing”*. **A non-significant test is not an equivalence test.** Re-analysed with `scripts/ir_equivalence_interval.py` (artifact `docs/eval/ir_equivalence_2026-09-10.md`), on the same compiled per-seed table:

| quantity | value |
|---|---:|
| ANOVA, reproduced | F(12, 26) = 1.037, p = 0.447 |
| pooled within-variant sd | 0.00508 |
| observed spread of means | 0.01193 (`yolo26l` − `yolov10n`) |
| **minimum detectable spread** (α = 0.05, power = 0.80) | **0.02067** |
| observed spread / detectable | **0.58×** |
| **Tukey HSD 95% simultaneous CI, largest gap** | **[−0.00314, +0.02700]** |
| same pair, no multiplicity correction | [+0.00340, +0.02045] — **excludes zero** |

The design was powered to detect only a spread about **1.7× larger than anything the ladder produced**, so p = 0.45 was close to the expected outcome whether or not architecture matters. Equivalence is supported at **no** delta tested — 0.0014, 0.0031, 0.0050, 0.0100, 0.0150, 0.0200 all sit below the CI upper bound of 0.0270, which is itself **wider than the entire observed range of variant means**.

And the uncorrected interval for the extreme pair excludes zero. The multiplicity correction is the right call — that pair is the maximum of 78 comparisons — but the data is **not** the featureless null this document describes. It is an underpowered design that saw a suggestive gap and could not adjudicate it.

**Corrected wording for reason 2:** *this ladder could not resolve architecture differences at the scale IR operates on* — not *the architectures are indistinguishable*, and not *it measures nothing*.

**Reasons 1 and 3 are untouched and are each sufficient on their own**: the architecture was frozen six days before the queue was created, and the ladder trains the wrong class set on a recipe that does not contain the deployed `yolo26m-p2feat`. Stopping was right for those reasons. Likewise unchanged: the 44 runs' value as the negative result in “What the 44 runs are good for” rests on the **absolute** IR scale (0.061–0.073, an order of magnitude below VIS), which this amendment does not touch.

Full analysis: [`ranking-hygiene-2026-09-10.md`](ranking-hygiene-2026-09-10.md) §3.
