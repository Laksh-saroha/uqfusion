# The IR architecture benchmark is closed, unfinished (2026-09-01)

**Decision.** Stop `runs/queue_ir_benchmark_stride4` at 44/93 runs. The 44 that
completed are kept and remain citable; the remaining 49 will not be run. Nothing is
deleted and the queue stays resumable, so this is reversible if a reviewer asks.

## Why

**1. It cannot change any decision that is still open.** The IR architecture was
frozen on 2026-08-20 (`yolo26m-p2feat`), and the benchmark queue was created
2026-08-26 — six days *after* the choice it was meant to inform. Every downstream
artifact (the σ-head, the ensemble, the MC arm, all of Table 3) is already trained
on the frozen architecture.

**2. It measures nothing.** Across the 13 variants that reached all three seeds,
one-way ANOVA gives **F(12, 26) = 1.037, p = 0.45** — no detectable variant effect.
The between-variant spread of means is 0.0119 (yolo26l 0.0732 down to yolov10n
0.0612) against a mean within-variant seed sd of **0.0047**. Thirteen architectures
spanning 2.6–68 M parameters and 8.7–258 GFLOPs are statistically indistinguishable
on this task. Finishing the remaining 49 runs would tighten error bars around a null.

**3. It does not measure the production configuration anyway.** Two mismatches:

  - The ladder trains 2-class `data_ir_stride4.yaml`; production IR is **nc=1
    ship-only** (`data_ir_shiponly.yaml`, decision D28/A-1 — IR buoy AP measured
    0.00019 against 29,131 buoy detections for 596 GT boxes).
  - **`yolo26m-p2feat` — the architecture actually deployed — is not in the ladder.**
    The 31 variants are all stock backbones; the P2 feature-level change that IR
    depends on appears in none of them.

So even a decisive result would be about a different model on a different label set.

## What the 44 completed runs are good for

They stand as a negative result worth one paragraph in the paper: **on Pohang IR,
detector architecture choice is not a lever.** Absolute mAP@50-95 sits at 0.061–0.073
for every variant tried, an order of magnitude below the VIS side, which locates the
difficulty in the modality and the small-object regime rather than in the backbone.
That is a more useful sentence than a ranking would have been.

Compiled numbers: `phase1_benchmark/compiled/ir_benchmark_stride4_variants.csv`
(per-variant) and `ir_benchmark_stride4_runs.csv` (per-run).

## State on disk

`runs/queue_ir_benchmark_stride4` — 44 done, 2 paused, 2 diverged, 45 pending;
`control.json` left `paused: true` with a note pointing here. `run_queue.py resume`
restarts it exactly where it stopped if that is ever wanted.

Superseded by: nothing. Related: `docs/ir-benchmark-divergence-falsepos-2026-08-26.md`
(the divergence alarm false positives seen in this queue),
`docs/vis-benchmark-stride4-2026-08-27.md` (the VIS counterpart, also paused).
