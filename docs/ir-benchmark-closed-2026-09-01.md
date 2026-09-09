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
