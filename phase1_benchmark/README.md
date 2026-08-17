# Phase 1 benchmark — consolidated record

Every Phase 1 backbone-benchmark run (Pohang maritime VIS, ship class only), its curve,
its weights, and one CSV describing all of them. **This folder is the record.** The
three trees it replaces — `runs/benchmark/runs/`, `archive/phase1/main_2026-08-10/`, and
`archive/phase1/pilot_2026-07-31/runs_31_07/` — no longer hold Phase 1 runs.

Built by `scripts/consolidate_phase1.py`, which is re-runnable and idempotent.

```
phase1_benchmark/
  results.csv          93 rows, one per (grid, variant, seed)   <- cite this
  superseded.csv       rows that were in the record and no longer are, with the reason
  runs/<grid>_<variant>_seed<n>/
      args.yaml        the training args ultralytics actually used
      results.csv      the per-epoch curve
      weights/         best.pt, last.pt
      val/             the validation output that produced this row's metrics
      plots/           training/val figures
  extra/               artifacts that are NOT result rows; see extra/CONTENTS.json
```

93 run directories. `runs/` and `extra/` are gitignored; the CSVs and this file are
tracked.

## Two training campaigns in one table

| `grid` | run ids | rows | when | variants |
|---|---|---|---|---|
| `main` | `main_*` | 27 | 2026-08-03 → 08-17 | YOLO12 s/m/l/x · YOLO26 n/s/m/l/x |
| `pilot` | `pilot_*` | 66 | 2026-07-31 | YOLOv8 · YOLOv9 · YOLOv10 · YOLO11 · YOLO12n |

**31 variants, 93 rows, 93 run directories** — one row per directory, and
`(variant, seed)` is unique: every variant is measured by exactly one campaign.

`yolo12s` was the one variant both campaigns ran. The `main` seeds 0/1/2 are kept and
the pilot's three runs are retired to `extra/pilot_yolo12s_seed*_superseded_by_main/`
(artifacts preserved, not deleted, and not in any table).

**The two campaigns ran on different train/val partitions of Pohang, and the pilot's is
unrecoverable.** They are merged here as a deliberate call by the project owner. What
that assumption costs — the possible train/val contamination, the differing training
pool, and the `yolo12s` overlap that is the only direct probe of it — is set out in
**[`docs/phase1-pilot-grid.md`](../docs/phase1-pilot-grid.md)**. Read it before reporting
any cross-grid comparison.

`split_fingerprint` is **not** a column. The fingerprints (`682dbe9f0f05` for `main`,
`f0220e716277` for `pilot`) are recorded in that document and in
`docs/phase1-experimental-record.md`.

### `run_id` carries the grid

Every run id is `<grid>_<variant>_seed<n>` and `grid` is a column, so a row always says
which campaign produced it without needing a lookup. This was originally forced —
`yolo12s` ran in both campaigns and collided — and is kept now that the collision is
resolved, because the split difference still makes the campaign a fact you need per row.
Group by `variant` alone; there is no longer any variant spanning both.

## The design

| | |
|---|---|
| Task | single-class detection (`classes=[0]`, ship) |
| Design | 31 variants × 3 seeds |
| Budget | 100 epochs, `patience = 20` — **no run reached 100** (range 24–63) |
| Image size | 640, every run |
| Metric scale | ultralytics 8.4.90, every row |
| Batch | **not constant**: 31 (66 runs), 8 (11), 32 (9), 24 (7) |
| Total compute | sum of `train_time_s`, a lower bound — see limits 1 |

## Reading `results.csv`

39 columns in five groups.

**Identity and configuration** — `run_id`, `grid`, `variant`, `seed`, `trained_on`,
`scored_on`, `batch`, `imgsz`, `epochs_cfg`, `patience_cfg`.

`batch` must be reported: ultralytics derives `accumulate = max(round(64/batch), 1)` and
scales weight decay by `batch × accumulate / 64`, so batch 24 trains at a 12.5% higher
effective weight decay than batch 8/16/32.

`epochs_cfg` is 100 on every row and no run got near it. Quote `epochs_trained`.

**Metrics** — `precision`, `recall`, `map50`, `map50_95`, from the explicit `best.pt`
validation. Byte-identical to the values in the CSVs this record replaces; consolidation
moved rows, it did not recompute them.

**Cost** — `params_m`, `gflops` (pristine nc=80 checkpoint, one pair per variant),
`train_time_s`.

**Training dynamics** — when each `best.pt` was reached and when patience fired.

**Status and provenance** — `admissible`, `exclude_reason`, `run_path`,
`train_dir_orig`, `val_dir_orig`, library/torch/commit, `classes`, `note`.

### The training-dynamics columns

| column | meaning |
|---|---|
| `best_epoch` | the fitness peak — the epoch whose weights `best.pt` holds |
| `best_map50_95_curve` | that epoch's in-training mAP50-95 (an independent check on `map50_95`) |
| `last_epoch` | the final epoch trained |
| `epochs_trained` | distinct epochs in the curve |
| `patience_gap` | `last_epoch − best_epoch` |
| `patience_fires_epoch` | `best_epoch + patience`: where an *uninterrupted* stopper fires |
| `stopper_ref_epoch` | where ultralytics' stopper was actually counting from at the end |
| `stop_reason` | how the run ended |
| `resume_segments` | training invocations (1 = never paused) |
| `dup_epoch_rows`, `epoch_regressions` | evidence of a shared run directory — a genuine resume continues the epoch count, so a *decrease* means a second process wrote into the same file |

**Fitness is `mAP50-95` alone** in ultralytics 8.4.90, not the `0.1*mAP50 +
0.9*mAP50-95` blend of older releases. Verified against `best_fitness` stored in
un-stripped checkpoints. The blend picks the wrong peak on any run whose mAP50 and
mAP50-95 crest at different epochs.

### Why `best_epoch` and `stopper_ref_epoch` differ on resumed runs

`Trainer.resume_training()` restores the optimizer, scaler, EMA and `best_fitness` — but
the `EarlyStopping` object is rebuilt one line earlier (`trainer.py:381`) with
`best_fitness=0.0` and `best_epoch=0`, and nothing restores it. Since
`EarlyStopping.__call__` treats `best_fitness == 0` as "always an improvement", the first
validated epoch after a resume becomes the stopper's new reference, and the run then
counts its 20 epochs from a *local* peak inside the final segment.

Two consequences, pulling in opposite directions:

- `best.pt` is **not** damaged. The trainer's own `best_fitness` *is* restored, so a
  weaker later epoch cannot overwrite it. `best_epoch` is the true global peak.
- Training **length** is. A resumed run can keep going well past the point an
  uninterrupted run would have stopped.

Only the `main` grid has resumed runs; every pilot run is single-segment.

### `stop_reason` values

| value | n | meaning |
|---|---|---|
| `early_stop` | 87 | never paused; stopped at exactly `best_epoch + 20` |
| `early_stop_after_resume` | 2 | resumed, but the stopper's reference was the global peak, so it still stopped at `best_epoch + 20` |
| `killed_past_patience` | 2 | the stopper had not fired, but the run had already gone ≥20 epochs past its peak when it was killed |
| `shared_dir_past_patience` | 1 | `main_yolo12x_seed0` — collided directory, ≥20 past peak |
| `shared_dir_before_patience` | 1 | `main_yolo12x_seed1` — collided directory, only 14 past peak |

## Admissibility

> A run is admissible if it trained **≥ `patience` epochs past its own peak**. Then
> `best.pt` holds the weights an uninterrupted run would have stopped on, however many
> times it was actually interrupted.

This is the **only** exclusion criterion in the record. No run is dropped by hand. It is
a statement about the counterfactual, not about whether ultralytics' own resume-damaged
stopper happened to fire — which is why `killed_past_patience` runs are admissible: the
weights are the same ones a clean run would have kept.

**92 of 93 rows are admissible.** The exception is `main_yolo12x_seed1`
(`patience_gap` 14), whose value is a lower bound on that seed. It stays in
`results.csv` as raw record; filter on `admissible == "yes"` for any table.

## Seed-means (mAP50-95, admissible rows)

| variant | grid | n | mean ± sd | params (M) | GFLOPs |
|---|---|---|---|---|---|
| yolo26x | main | 3 | **0.3049 ± 0.0020** | 58.99 | 209.5 |
| yolov9e | pilot | 3 | 0.3033 ± 0.0066 | 58.21 | 193.0 |
| yolov9c | pilot | 3 | 0.3019 ± 0.0057 | 25.59 | 104.0 |
| yolo26m | main | 3 | 0.3016 ± 0.0050 | 21.90 | 75.4 |
| yolo12x | main | 2 | 0.3007 ± 0.0049 | 59.22 | 200.3 |
| yolov8x | pilot | 3 | 0.2999 ± 0.0010 | 68.23 | 258.5 |
| yolo26l | main | 3 | 0.2998 ± 0.0026 | 26.30 | 93.8 |
| yolo11l | pilot | 3 | 0.2994 ± 0.0041 | 25.37 | 87.6 |
| yolov10x | pilot | 3 | 0.2982 ± 0.0037 | 31.81 | 171.8 |
| yolov10l | pilot | 3 | 0.2979 ± 0.0068 | 25.89 | 127.9 |
| yolo11m | pilot | 3 | 0.2977 ± 0.0046 | 20.11 | 68.5 |
| yolo11x | pilot | 3 | 0.2971 ± 0.0023 | 56.97 | 196.0 |
| yolov8m | pilot | 3 | 0.2968 ± 0.0075 | 25.90 | 79.3 |
| yolov10m | pilot | 3 | 0.2958 ± 0.0022 | 16.58 | 64.5 |
| yolov10b | pilot | 3 | 0.2948 ± 0.0059 | 20.57 | 99.4 |
| yolov9m | pilot | 3 | 0.2912 ± 0.0069 | 20.22 | 77.9 |
| yolov8l | pilot | 3 | 0.2909 ± 0.0029 | 43.69 | 165.7 |
| yolo11s | pilot | 3 | 0.2909 ± 0.0086 | 9.46 | 21.7 |
| yolo12m | main | 3 | 0.2906 ± 0.0046 | 20.20 | 68.1 |
| yolo12l | main | 3 | 0.2870 ± 0.0096 | 26.45 | 89.7 |
| yolov8s | pilot | 3 | 0.2847 ± 0.0022 | 11.17 | 28.8 |
| yolo26s | main | 3 | 0.2813 ± 0.0045 | 10.01 | 22.8 |
| yolov10s | pilot | 3 | 0.2809 ± 0.0014 | 8.13 | 25.1 |
| yolo12s | main | 3 | 0.2783 ± 0.0109 | 9.29 | 21.7 |
| yolov9s | pilot | 3 | 0.2715 ± 0.0052 | 7.32 | 27.6 |
| yolov10n | pilot | 3 | 0.2567 ± 0.0038 | 2.78 | 8.7 |
| yolo12n | pilot | 3 | 0.2560 ± 0.0067 | 2.60 | 6.7 |
| yolov9t | pilot | 3 | 0.2553 ± 0.0074 | 2.13 | 8.5 |
| yolov8n | pilot | 3 | 0.2552 ± 0.0087 | 3.16 | 8.9 |
| yolo26n | main | 3 | 0.2540 ± 0.0058 | 2.57 | 6.1 |
| yolo11n | pilot | 3 | 0.2486 ± 0.0028 | 2.62 | 6.6 |

Every variant sits in exactly one campaign, n=3 except `yolo12x` (n=2 — one
inadmissible seed).

**The top eight span 0.0055, against seed sds of 0.0010–0.0066. They are not
separable.** Pooling the campaigns widened the leading group rather than resolving it:
`yolo26x`'s margin over the second-place variant fell from 0.0033 to **0.0016**, below
its own seed sd. Nothing here supports ranking any two of the top eight against each
other. What the field does show cleanly is the capacity floor: every `n`-scale variant
lands at 0.2486–0.2567, roughly 0.05 below the leaders.

## `fps.csv` — batch-1 latency

184 measurements: every admissible run (92 checkpoints) × fp32/fp16, 500 timed frames
after 50 warmup, `imgsz=640`, end-to-end `predict()`. Regenerate with
`PYTHONPATH=src python scripts/measure_fps.py --all-seeds`.

**The GPU clock was pinned at 1500 MHz for this sweep, and that is load-bearing.** An
earlier unpinned run is kept as `fps_seed0_only.csv` — **do not quote it.** Its clock
swung 1200–2400 MHz and the drift aliased onto whichever variant ran late, producing
impossible orderings: `yolo12l` (89.7 GFLOPs) measured slower than `yolo12x` (200.3),
`yolo11l` slower than `yolo11x`, and fp16 "gains" scattered from −10% to +47% with no
relation to model size. Pinning removed every one of those inversions. `fps.csv` records
`gpu_clock_mhz` and `gpu_temp_c` per measurement so the lock is verifiable, not assumed;
the sweep held 1500–1500 MHz, 0.0% spread.

| column | meaning |
|---|---|
| `wall_ms_per_img`, `fps` | end-to-end, including per-call source setup |
| `pre_ms`, `inf_ms`, `post_ms` | the model's own reported breakdown |
| `seed`, `run_id` | which checkpoint was timed |
| `timestamp`, `gpu_temp_c`, `gpu_clock_mhz` | thermal state, to separate drift from real variation |

Two things to decide before quoting a number:

1. **Which figure.** `fps` runs ~4 ms/img slower than `pre + inf + post` because the
   protocol calls `predict()` once per image and each call rebuilds its source. That
   overhead is the harness, not the model. `inf_ms` is the fairer architecture
   comparison; `fps` is closer to a naive per-frame pipeline.
2. **The spread is not all seed variation.** Relative sd across seeds averages 4.8% and
   reaches 12.7%, concentrated in the *small* models — exactly where fixed per-call
   overhead dominates and its jitter shows up proportionally largest. Latency cannot
   depend on weight values: seeds of one variant share layer count, parameter count and
   FLOPs. Only `post_ms` can move, via NMS box count. Treat the sd as measurement
   precision, not architectural uncertainty.

fp16 helps exactly where theory says it should — compute-bound large models
(`yolov8x` +69%, `yolo26x` +35%, `yolo12x` +36%) — and does nothing for the latency-bound
nano models, several of which measure marginally slower in fp16.

## Known limits of this record

1. **`train_time_s` under-reports resumed runs.** `grid.py:219` measures one invocation.
   Rows banked by `recover_row.py` carry cumulative time; rows written by the grid do
   not. Recompute from each run's `results.csv` before publishing this column.
2. **The two campaigns ran on different splits** and the pilot's is unrecoverable.
   [`docs/phase1-pilot-grid.md`](../docs/phase1-pilot-grid.md).
3. **`main_yolo12x` seeds 0 and 1 come from a shared run directory.** Two grid processes
   wrote one `results.csv` and one `weights/` folder. Seed 0's row was re-derived from
   the surviving inner checkpoint; seed 1's original row is not reproducible from any
   checkpoint that still exists.
4. **`main_yolo26m_seed0` trained under ultralytics 8.4.7** and was re-scored under
   8.4.90 (`ultralytics_version` reads `8.4.7+val8.4.90`). Its metrics are on the right
   scale; its *curve* is not, and its early-stopping decisions used the old scale.
5. **`git_commit` reads `unknown` for 17 of the 18 `main` server rows** and for all 68
   pilot rows.
6. **The retired pilot `yolo12s` runs are in `extra/`, not in any table.** One of the
   three (`seed2`) never had a row at all: killed 3 epochs past its peak, so it never
   reached the validation that writes one.
7. **Two `main` runs are missing end-of-training plots** (`main_yolo26l_seed2`,
   `main_yolo26m_seed2`) — killed after satisfying early stopping but before ultralytics
   wrote its figures. Curves and weights are complete.

## Related documents

- [`docs/phase1-pilot-grid.md`](../docs/phase1-pilot-grid.md) — the `pilot_*` rows: what
  they cover, the split difference, and what merging them assumes
- [`docs/phase1-experimental-record.md`](../docs/phase1-experimental-record.md) —
  provenance, defects, corrections; the source for the paper's methods and limitations
- [`docs/phase1-handoff-2026-08-17.md`](../docs/phase1-handoff-2026-08-17.md) —
  operational state and traps
- [`docs/phase1-robustness-table.md`](../docs/phase1-robustness-table.md) — per-run
  metric vs curve-peak cross-check
