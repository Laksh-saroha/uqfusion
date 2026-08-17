# Phase 1 handoff — 2026-08-17 (grid COMPLETE, record audited)

Laptop backbone-benchmark grid, Pohang maritime VIS, ship-only. Supersedes
`phase1-handoff-2026-08-11.md`, which is still the reference for the ultralytics
version finding (section 1 there) — that is not repeated here.

Repo `A:\Uncertain`, branch `docs/phase1-experimental-record`.

**Revised the same day** after a checkpoint-level audit of every run dir. The
audit corrected four things this document previously got wrong; each is called
out inline as *CORRECTION*. Read section 3 before trusting any earlier curve
reading.

**Revised again the same day** after consolidating the record. All 27 runs now
live in one tree with one CSV — see section 9 — and the CSV carries the
training-dynamics columns the old one lacked. Paths below are the new ones.

---

## 1. Status: the grid is finished, audited, and consolidated

`[wrapper] grid complete at 17-08-2026 13.45.36.42`. Nothing is training.
`phase1_benchmark\results.csv` holds **93 data rows** across two training
campaigns — `grid = main` (27 rows, this grid) and `grid = pilot` (66 rows, the
2026-07-31 grid, merged in by owner decision). 31 variants, one campaign each,
93 run directories, one row per directory. All on the ultralytics 8.4.90 metric
scale, classes `[0]`, one `params_m`/`gflops` pair per variant. No run is
excluded by hand; exclusion is by the single rule in section 3.

**The two campaigns ran on different splits** — `682dbe9f0f05` for `main`,
`f0220e716277` (unrecoverable) for `pilot`. `split_fingerprint` is no longer a
CSV column; the fingerprints and what merging assumes are in
`docs\phase1-pilot-grid.md`. Run ids carry the grid
(`main_yolo12s_seed0`) so every row states its campaign. `yolo12s` was the one
variant in both: the `main` seeds are kept and the pilot's three runs are retired
to `extra\pilot_yolo12s_seed*_superseded_by_main\`. Everything below concerns
`main`.

The runs were previously split across `runs\benchmark\runs\` (laptop) and
`archive\phase1\main_2026-08-10\` (server), with different directory layouts and
two overlapping CSVs. `scripts\consolidate_phase1.py` moved all 27 into
`phase1_benchmark\runs\<variant>_seed<n>\` under one layout and merged the CSVs;
it is idempotent and re-runnable. Metric cells were moved, never recomputed —
verified cell-by-cell against the pre-move CSV.

## 2. Final seed-means (mAP50-95)

Table 1 applies the admissibility rule (section 3), which drops one row —
`yolo12x` seed 1. Both columns are given because the choice moves the order.

| variant | n | mean ± sd (Table 1) | n | mean ± sd (all rows) |
|---|---|---|---|---|
| yolo26x | 3 | **0.3049 ± 0.0020** | 3 | 0.3049 ± 0.0020 |
| yolo26m | 3 | 0.3016 ± 0.0050 | 3 | 0.3016 ± 0.0050 |
| yolo12x | 2 | 0.3007 ± 0.0049 | 3 | 0.2989 ± 0.0046 |
| yolo26l | 3 | 0.2998 ± 0.0026 | 3 | 0.2998 ± 0.0026 |
| yolo12m | 3 | 0.2906 ± 0.0046 | 3 | 0.2906 ± 0.0046 |
| yolo12l | 3 | 0.2870 ± 0.0096 | 3 | 0.2870 ± 0.0096 |
| yolo26s | 3 | 0.2813 ± 0.0045 | 3 | 0.2813 ± 0.0045 |
| yolo12s | 3 | 0.2783 ± 0.0109 | 3 | 0.2783 ± 0.0109 |
| yolo26n | 3 | 0.2540 ± 0.0058 | 3 | 0.2540 ± 0.0058 |

`yolo26x` per-seed detail:

| seed | mAP50 | mAP50-95 | train_time_s | peak ep | stopped at |
|---|---|---|---|---|---|
| 0 | 0.6380 | 0.30479 | 58239.5 (16.2 h) | 8 | ep 28 |
| 1 | 0.6362 | 0.30690 | 70877.5 (19.7 h) | 12 | ep 32 |
| 2 | 0.6192 | 0.30294 | 84996.8 (23.6 h) | 30 | ep 50 |

**CORRECTION — the 0.0072 "noise at identical settings" figure is withdrawn.**
It came from the two `yolo12x` seed-0 rows, which were never two runs at
identical settings: checkpoint metadata shows one trained at **batch 16** and
was killed at ep 16, the other at **batch 8** and ran to a natural stop. The
figure measured a batch difference between a killed run and a completed one.
Use the per-variant cross-seed sd above instead.

**Read the headline conservatively.** Within `main`, the top four (`26x`, `26m`,
`12x`, `26l`) span 0.0051 against seed sds of 0.0020–0.0050. They are not
separable, and their order is not even stable across two legitimate readings of
the same runs (section 6). **The earlier claim that the 26-series large models
beat the whole 12-series field does not survive** — `12x` sits inside the top
group.

**Pooling the `pilot` rows widens this, it does not resolve it.** Across all 31
variants the leading group becomes eight, spanning 0.0055, and `26x`'s margin
over second place (`yolov9e`, 0.3033) falls to **0.0016 — below `26x`'s own seed
sd**. The full table is in `phase1_benchmark\README.md`. The one claim that
strengthens on pooling is the capacity floor: every `n`/`t`-scale variant in
every family lands at 0.2486–0.2567.

## 3. Early stopping, resume, and the admissibility rule

**CORRECTION — fitness is `mAP50-95` alone in ultralytics 8.4.90**, not
`0.1*mAP50 + 0.9*mAP50-95`. Verified against `best_fitness` stored in
un-stripped checkpoints (`26l` seed 2: stored 0.2991 == max mAP50-95 0.29910,
against a blend of 0.33413; `26m` seed 2: 0.30201 == 0.30201). Every peak-epoch
statement in the 08-11 handoff computed with the blend should be recomputed.

**The resume defect is real.** On resume `trainer.py` restores optimizer,
scaler, EMA and `best_fitness`, but rebuilds `EarlyStopping`
(`torch_utils.py:961`) fresh with `best_epoch = 0`, so the patience countdown
restarts at the resume epoch. `best.pt` is **not** corrupted — the restored
`best_fitness` prevents a weaker later epoch from overwriting it. What can
break is training *length*.

**CORRECTION — the damage was smaller than recorded.** Re-auditing on the
correct fitness definition, **23 of the 25 uncollided runs stopped at exactly
peak + 20**. The sole over-run is `26m` seed 2 (+3 epochs). `26x` seed 2 in
particular peaked at **ep 30, not ep 6**; its stop at ep 50 is exactly peak + 20,
a clean stop, and the epochs after its resume produced the checkpoint it kept.
The earlier claim that it "trained 23.6 h for a `best.pt` already fixed at ep 6"
was an artifact of reading the curve with the wrong fitness.

**This is now in the record, per run.** `results.csv` carries `best_epoch` (the
global peak — what `best.pt` holds), `stopper_ref_epoch` (what patience was
actually counting from at the end, which differs only on resumed runs),
`last_epoch`, `patience_gap`, `patience_fires_epoch`, `resume_segments` and
`stop_reason`. The five distinct `stop_reason` values and what each implies are
documented in `phase1_benchmark\README.md`; no curve needs re-reading by hand.

### The admissibility rule

> A run is admissible if it trained **≥ patience epochs past its own peak**
> (peak by mAP50-95). Then `best.pt` holds the weights an uninterrupted run
> would have stopped on, regardless of how many times it was interrupted.

Applied uniformly, this is the *only* exclusion criterion in the record. It
admits two runs previously discarded and rejects one previously kept:

- **`26m` seed 2 — recovered.** Peak ep 10, ran to ep 33 (+23). Banked at
  mAP50-95 **0.30244**, cumulative train 39,677 s.
- **`26l` seed 2 — recovered.** Peak ep 6, ran to ep 26 (+20), i.e. exactly a
  natural stop. Its single resume at ep 16 came after the peak was banked.
  Banked at **0.29947**, cumulative train 37,533 s.
- **`yolo12x` seed 1 — rejected.** Peak ep 22, last ep 36 (+14); the lineage
  whose checkpoint survived was killed *at* its own best epoch (+0). Its row is
  a lower bound. It stays in the CSV as raw record and is excluded from Table 1.

Recovery needs `scripts/recover_row.py`, not the grid: grid.py skips by CSV row
(`_completed()`), so re-running a variant lands in the resume branch and keeps
*training* a run that was already done.

Note `grid.py:219`: `train_time_s` measures **this invocation only**, so rows
written by grid.py for resumed runs under-report. Rows banked by
`recover_row.py` carry cumulative time. **Do not publish this column as-is** —
recompute from each run's `results.csv`.

## 4. The concurrent-writer collision (server, `yolo12x` only)

Two training processes shared one run directory — including one `weights/`
folder — for `ship_yolo12x_seed0` and `ship_yolo12x_seed1`. Their `results.csv`
files interleave row-by-row with independent clocks:

```
ep 24  t 61455.4
ep  1  t  3513.4     <- second writer, own clock
ep 25  t 63972.4
ep  2  t  6646.1
```

Detection: **the epoch column decreasing in file order**. A genuine resume
continues the epoch count, so epochs only ever increase; a decrease means a
second process started from epoch 1 into the same file. Duplicate epoch numbers
are the weaker test — seed 1 has 11 regressions and **zero** duplicates, so a
duplicate-only check misses it. Both counts are now columns
(`epoch_regressions`, `dup_epoch_rows`). **Every other run dir in the grid is
single-writer** — all 25 have zero of each.

Because `best.pt` is whichever process saved last, the surviving checkpoint need
not belong to the lineage whose curve you are reading. Re-validating the
archived checkpoints on this laptop settles what each row actually measured:

| checkpoint | re-validated | CSV row | Δ |
|---|---|---|---|
| `12x` seed 2 (clean) — **control** | 0.29717 | 0.29719 | −0.00002 |
| `12x` seed 0, nested dir, batch 8, completed | 0.30414 | 0.30413 | +0.00001 |
| `12x` seed 0, outer dir, batch 16, killed | 0.29569 | 0.29691 | −0.00122 |
| `12x` seed 1, batch 8, killed | 0.29582 | 0.29548 | +0.00034 |

The control reproduces to 2e-5, so cross-machine re-validation is exact and the
two large deltas are real: **neither the 0.29691 row nor the seed-1 row is
reproducible from any checkpoint that still exists.** Both were written before a
concurrent process overwrote `best.pt`.

Resolution: `12x` seed 0 is now the artifact-backed 0.30414 row, its run dir
consolidated to `phase1_benchmark\runs\yolo12x_seed0\`. The retired 0.29691 row
is in `phase1_benchmark\superseded.csv` with its reason; the batch-16 writer's
directory is `phase1_benchmark\extra\yolo12x_seed0_outer_writer\`.

## 5. Operational traps (unchanged, all still live)

1. **A killed run writes no CSV row.** Rows are written on completion only.
   Relaunching the wrapper therefore *resumes* the killed run rather than
   skipping it, because `grid.py::_completed()` skips by CSV row. To actually
   skip one, move its run dir aside **and** remove the variant from
   `--variants`. Seeds are shared across variants in one invocation, so a
   single variant+seed cannot be skipped selectively.
2. **Self-matching liveness probes lie.** `Get-CimInstance Win32_Process |
   Where-Object { $_.CommandLine -match 'watch_grid' }` matches the PowerShell
   process running the query itself, reporting a dead watchdog as alive. Always
   constrain by `-Filter "Name='cmd.exe' OR Name='python.exe'"` first. Same bug
   as commit `727adb4`.
3. **The watchdog died repeatedly (5+ times), always during pauses**, and once
   was *alive but silent*. Process existence is not proof of health — compare
   `health.log`'s **mtime against the clock** on every check. Its last line can
   describe a long-dead process. Restart the watchdog after every pause.
4. **A closed console window kills nothing** if the job was launched with
   `Invoke-CimMethod Win32_Process Create` — confirmed live this session. The
   watchdog, however, lost its wrapper.
5. **Windows Update rebooted the machine** (Event 1074, TrustedInstaller) and
   killed grid + wrapper + watchdog, costing ~4h40m. Pause updates before a
   multi-day run — this cannot be fixed from inside the run.
6. **Verify a resume from the log, not the process.** Check for the
   `[grid] === <name>: RESUME from ...` banner plus live iteration frames.
   Ultralytics progress bars are `\r`-delimited, so `tail -n` shows nothing
   useful — use `tail -c N | tr '\r' '\n' | sed 's/\x1b\[[0-9;]*[A-Za-z]//g'`.
   Also beware: an `until ... grep -q 'it/s'` loop matches the **dataset
   scanner**, not training. Anchor on `^ *[0-9]+/100` instead.
7. **Identify which run you are stopping.** Check `[grid] ===` banners and the
   CSV row count first.
8. **NEW — never open the results CSV in Excel.** It rewrote the file on save:
   `ultralytics_version` "8.4.90" became the date serial `0.337152778` on 25
   rows and blank on one, and every float was truncated to 9 significant
   digits. It also held a write lock that made `_append_row` fail with
   `PermissionError` mid-recovery. The damaged copy is kept as
   `runs\benchmark\_excel_mangled_20260817_1410.csv`. Use
   `python -c "import csv; ..."` or a text editor.

## 6. Robustness check (`docs/phase1-robustness-table.md`)

Every Table 1 number is cross-checked against the in-training validator's
metrics from the run's own curve — a second record that does not depend on which
`best.pt` survived. Both numbers are now columns in `results.csv` (`map50_95`
and `best_map50_95_curve`), so the script reads one file and no longer has to
locate run dirs or disambiguate rows by wall clock. Regenerate with
`python scripts/make_robustness_table.py --out docs/phase1-robustness-table.md`.

Explicit-val minus curve-peak across 26 comparable runs: **+0.00110 ± 0.00092**
(range −0.0001 … +0.0026). The offset splits cleanly by family, not by machine:

| family | offset |
|---|---|
| `yolo26*` (server **and** laptop) | +0.0003 |
| `yolo12*` | +0.0021 |

Plausibly because YOLO26 is NMS-free, so its in-training and standalone
validators take the same path. **It matters**: ~0.002 is the same size as the
gaps inside the top group, so the two metric sources rank the top four
differently (Table 1: `26x > 12x > 26l > 26m`; curve-peak:
`26x > 26l > 26m > 12x`). Below the top group the order is identical under
both. Keep Table 1 on one metric source and say which.

`26m` seed 0 is excluded from that comparison: it **trained** under 8.4.7, so
its curve is on the old scale while its Table 1 metric was re-scored under
8.4.90. Its delta of −0.0387 is independent confirmation of the version finding.

## 7. VRAM behaviour of `yolo26x` (measured, batch 8)

Per-epoch reserved peaks are **flat** — no leak, no growth. Seed 1 climbs
9.13 → 9.65 G over epochs 1–3, then holds a 9.30–9.70 G band for 24+ epochs.
Seed 0 across a full run: 328,004 frames, min 8.38 G, max 9.70 G, mean 9.46 G.
The ~0.35 G sawtooth is caching-allocator fragmentation tracking variable box
counts per batch, not accumulation.

Driver-level that is ~10.7 of 12.28 GB — **87% of the card, ~1.5 GB headroom.**
The printed column is `memory_reserved` and excludes CUDA context, cuDNN
workspaces and driver overhead. Keep other GPU apps closed. `26x` at batch 16
peaks 15.31 GB, which Windows WDDM pages into host RAM instead of raising OOM —
it "runs" ~17x slower. **Do not raise batch for `26x`.**

## 8. Still open

1. **`yolo12x` seed 1 is inadmissible and unreproducible.** Retraining it on
   the laptop (~17.5 h) is the only way back to a clean n=3 at one batch size.
   Its current value is a lower bound, so a full run would likely push `12x`
   *up*, not down — it will not restore the old headline.
2. **`26m` seed 0 is the only row trained under ultralytics 8.4.7.** Metrics
   were re-scored to 8.4.90, but its early-stopping decisions used the old
   fitness scale, and it is `26m`'s strongest seed. Awkward given the paper's
   own headline is that an unpinned ultralytics corrupted the metric. ~11 h to
   redo.
3. ~~`epochs_cfg` reads 100 on every row~~ — **addressed.** `epochs_cfg` still
   reads 100 (it is the configured budget, and no run reached it), but
   `results.csv` now also carries `epochs_trained`, `best_epoch`, `last_epoch`
   and `patience_gap`. Table 1 should quote those. Actual range 24–61.
4. **`train_time_s` is not consistently defined.** `grid.py:219` times *this
   invocation only*, so grid-written rows under-report resumed runs, while rows
   banked by `recover_row.py` carry cumulative time. The 266.8 h total is a
   lower bound. Recompute from the curves before publishing the column — and
   note the two shared-dir runs cannot be recomputed that way either, since
   summing per-segment clocks double-counts overlapping writers.
5. **opencv double-installed** — `opencv-python 5.0.0.93` and
   `opencv-python-headless 4.10.0.84` both claim `cv2`; resolves to 4.10.0.
6. **Pin `ultralytics==8.4.90`** in requirements. Load-bearing.
7. **Server `requirements.lock.txt` is unusable** — captured under base python;
   redo with the env active via `python -m pip list --format=freeze`.

### Next steps
- `scripts/measure_fps.py` (needs GPU — free now)
- `scripts/make_table1.py` — must read `phase1_benchmark/results.csv`, filter on
  `admissible`, report `epochs_trained`, and recompute timing from the curves
- Methods notes: a pinned detector version is load-bearing; ultralytics'
  resume defect is measurable but smaller than first thought; the
  admissibility rule is the whole exclusion policy and no run is dropped by
  hand. All three are cleanly measured here.

## 9. Key paths

The record — everything Table 1 rests on:

```
phase1_benchmark\README.md                     layout + column dictionary — READ FIRST
phase1_benchmark\results.csv                   93 rows, 39 columns, both campaigns, FINAL
phase1_benchmark\fps.csv                       batch-1 fp32/fp16 latency + FPS per variant
phase1_benchmark\superseded.csv                rows withdrawn from the record, with reasons
phase1_benchmark\runs\<grid>_<variant>_seed<n>\  args.yaml, results.csv, weights\, val\, plots\
phase1_benchmark\extra\                        non-result artifacts; see CONTENTS.json
docs\phase1-pilot-grid.md                      the pilot rows: split difference, what merging assumes
```

93 run directories (`main_*` 27, `pilot_*` 66), one row each.

Tooling and supporting files:

```
scripts\consolidate_phase1.py                  builds the above; idempotent
scripts\recover_row.py                         bank a row for a finished-but-unbanked run
scripts\make_robustness_table.py               curve-vs-Table-1 cross-check
docs\phase1-robustness-table.md                its output
docs\phase1-experimental-record.md             provenance, defects, limitations text
src\uqfusion\bench\grid.py                     grid driver
run_tail_laptop.cmd                            launcher
runs\derived\data_vis_stride2.yaml             the split (fp 682dbe9f0f05)
runs\grid_laptop.log                           grid stdout (~240 MB)
runs\health.log                                watchdog, 1 line/10 min
```

Superseded, kept only as evidence — **do not read numbers out of these**:

```
runs\benchmark\benchmark_results_tail.csv      the pre-consolidation record
runs\benchmark\benchmark_results_tail.csv.bak_*, .replaced_*   pre-edit snapshots
runs\benchmark\_excel_mangled_20260817_1410.csv                corrupted by Excel
archive\phase1\main_2026-08-10\runs.zip        the original server tarball — KEEP
archive\phase1\pilot_2026-07-31\runs_31_07.tar.gz   the pilot's original tarball — KEEP
```

Environment: python 3.13.0, torch 2.7.1+cu118, ultralytics 8.4.90,
RTX 4080 Laptop 12282 MiB, batch 8, workers 8, imgsz 640, patience 20,
deterministic. `workers=16` is worse than 8 (2.80 vs 4.00 it/s median).

**Fitness, for reading any curve above:** ultralytics 8.4.90 ranks epochs on
**`mAP50-95` alone**. The `0.1*mAP50 + 0.9*mAP50-95` blend quoted in earlier
handoffs is wrong for this version and picks the wrong peak on runs whose
mAP50 and mAP50-95 crest at different epochs.
