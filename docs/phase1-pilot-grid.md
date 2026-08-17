# The pilot grid (2026-07-31) — provenance for the `pilot_*` rows

`phase1_benchmark/results.csv` holds 93 rows from **two training campaigns**, told apart
by the `grid` column and the `run_id` prefix. This document is the provenance record for
the 66 `pilot_*` rows. It also carries the split fingerprints, which are deliberately no
longer a CSV column.

| `grid` | run ids | rows | campaign | split fingerprint |
|---|---|---|---|---|
| `main` | `main_*` | 27 | 2026-08-03 → 08-17 | **`682dbe9f0f05`** |
| `pilot` | `pilot_*` | 66 | 2026-07-31 | **`f0220e716277`** |

Every variant is measured by exactly one campaign. `yolo12s` was run by both; the `main`
seeds are the ones kept, and the pilot's three `yolo12s` runs are retired to
`phase1_benchmark/extra/pilot_yolo12s_seed*_superseded_by_main/` — artifacts preserved,
excluded from every table. Their measurements are still the evidence discussed below.

## What the pilot is

23 detector variants × 3 seeds = **69 runs**, ship class only (`classes=[0]`), same
harness and same task as the main grid.

| | |
|---|---|
| Variants | YOLOv8 n/s/m/l/x · YOLOv9 t/s/m/c/e · YOLOv10 n/s/m/b/l/x · YOLO11 n/s/m/l/x · YOLO12 n/s |
| Config | batch 31, epochs 100, patience 20, imgsz 640, workers 8 — **uniform on all 69** |
| Library | ultralytics 8.4.90 on every row |
| Stop epochs | 25–63; none reached the 100-epoch budget |
| Size | 6.15 GB |

## It is the better-run campaign

The grid with the split problem is, in every other respect, the cleaner experiment:

| | pilot | main |
|---|---|---|
| batch | **31, uniform across all 69 runs** | 8 / 16 / 24 / 32 |
| machines | one | two, with an ultralytics version change between them |
| shared run directories | none | 2 (`main_yolo12x` seeds 0, 1) |
| resumed runs | none | 4 |
| clean `peak + patience` stops | 68 of 69 | 23 of 25 uncollided |
| split lists | **lost** | archived and verified |

All 68 banked pilot rows are `stop_reason = early_stop`, `resume_segments = 1`,
`dup_epoch_rows = 0`, `epoch_regressions = 0`, and admissible.

## The split difference — read this before comparing across `grid`

The two campaigns ran on **different train/val partitions of Pohang**, and the pilot's
partition cannot be reconstructed.

The `data_vis_stride2.yaml` behind `f0220e716277` existed only on a server that was
wiped. It matches no split generation reproducible from this repo — the current full
split, the current stride-2, the pre-resplit full split and its ordinal stride, naive
strides, cross-validation combinations and L/R camera subsets were all tested against
it, and none matched.

What the surviving artifacts do show, from the `val_batch0_labels.jpg` plots and the
archived split lists:

```
pilot val begins at   pohang00_L_006750
main  val begins at   pohang00_L_006767          (17 frames later)

current val.txt      pohang00_L blocks:  6767–7184,  12446–12863
pre-resplit train    pohang00_L blocks:  4767–11183, 11186–13439   <- contains both
frames 6750–6766     in the pre-resplit train list; in NO current list
```

The 2026-07-14 resplit dropped 3,673 frames outright (122,745 → 119,072), including the
block where the pilot's val begins. Both campaigns cut val as contiguous frame blocks
from the same sequence, offset by 17 frames — so the pilot's partition is a *third* cut,
neither the current one nor the pre-resplit one.

Two consequences follow, and they are the reason this file exists:

1. **Possible train/val contamination, unquantifiable.** If the pilot's val block was
   also 418 frames starting at 6750, it ran 6750–7167, and the main grid's val frames
   7168–7184 would have been in the pilot's *training* set. The block length is
   unknowable — the pilot's train list is gone and its artifacts plot only the first ~96
   val images — so the overlap cannot be measured, only bounded by the structure above.
2. **The training data differs regardless of contamination.** Pilot models saw a
   different pool of images with a different train/val boundary. Comparing a `pilot_*`
   row against a `main_*` row therefore varies architecture *and* training data at once.

**The project owner has reviewed this and elected to treat the two grids as
comparable.** The rows are merged accordingly. This section is the record of what that
assumption costs, and it should be stated wherever cross-grid comparisons are reported.

### The one place it is directly visible

`yolo12s` is the only variant both campaigns ran, which makes it the only available — if
n-of-2 — probe of the split difference. The pilot rows below are **no longer in
`results.csv`** (retired to `extra/`, since `main` measures this variant), but they are
the measurement that this probe rests on and are reproducible from the retained weights:

| run | grid | batch | mAP50-95 | stop epoch | in `results.csv` |
|---|---|---|---|---|---|
| `main_yolo12s_seed0` | main | 32 | 0.27198 | 31 | yes |
| `main_yolo12s_seed1` | main | 32 | 0.29089 | 38 | yes |
| `main_yolo12s_seed2` | main | 32 | 0.27189 | 32 | yes |
| `pilot_yolo12s_seed0` | pilot | 31 | 0.28183 | 39 | retired to `extra/` |
| `pilot_yolo12s_seed1` | pilot | 31 | 0.28013 | 35 | retired to `extra/` |

Seed-means 0.2783 (main, n=3) vs 0.2810 (pilot, n=2), a gap of **+0.0027** — inside the
main grid's own `yolo12s` seed sd of 0.0109. That is reassuring but not evidence of
equivalence: two seeds cannot separate a split effect from seed noise, and the batch
differs by one as well.

## `pilot_yolo12s_seed2` — never had a row

Now at `extra/pilot_yolo12s_seed2_superseded_by_main/`, it holds complete weights, args
and a curve but never had a CSV row: it peaked at epoch 13 and was killed at epoch 16,
only 3 epochs past its peak against `patience = 20`, so it never reached the
post-training validation that writes one.

It was inadmissible under the same rule the main campaign uses, so it would have been
excluded from any table regardless — and unlike the main campaign's unbanked runs it
cannot be recovered with `scripts/recover_row.py`, because re-validating needs the val
split and the split lists are gone. It has no `val/` directory for the same reason. It is
the only run in either campaign that never got a row.

## Provenance

Extracted from `runs_31_07.tar.gz` (6.0 GB), kept untouched at
`archive/phase1/pilot_2026-07-31/`, along with the source CSV
(`benchmark_results_ship_visfilter.csv`, 68 rows) and `bench_yolov8n_seed0/`, a v8n
smoke test that was never part of the grid.

Metric cells were verified identical after the move; consolidation added the
training-dynamics columns and changed nothing else.
