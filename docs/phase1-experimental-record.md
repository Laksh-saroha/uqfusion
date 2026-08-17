# Phase 1 experimental record — provenance, defects, and corrections

Last updated **2026-08-17, grid complete and consolidated**. Source record for the
methods and limitations sections of the paper. Everything here is measured, not
inferred, unless explicitly marked.

Supersedes the 2026-08-11 revision, which was written mid-grid at 20 rows. Four of its
findings were later corrected outright; each correction is marked *CORRECTION* below.

**The record now lives in one place:** `phase1_benchmark/results.csv`, 93 rows across
two training campaigns (`grid` = `main` / `pilot`). See `phase1_benchmark/README.md` for
the layout and the column dictionary, and **`docs/phase1-pilot-grid.md` for what merging
the two campaigns assumes** (§15). `split_fingerprint` is no longer a CSV column; the
fingerprints are recorded in §1 here and in that document.

---

## 1. What the experiment is

A multi-seed detector-backbone benchmark on the Pohang maritime dataset (visible
spectrum), **ship class only**, feeding Table 1.

| | |
|---|---|
| Task | single-class detection (`--classes 0`, ship) |
| Split | `runs/derived/data_vis_stride2.yaml`, stride-2 subset |
| Split fingerprint | `682dbe9f0f05` — SHA256 over sorted `run/filename` ids, train+val |
| Train images | 48,136 |
| Val images | 11,352 (8,797 contain ≥1 ship) |
| Val instances | 105,098 |
| Design | 9 variants × 3 seeds (0,1,2) = **27 runs — all collected** |
| Budget | 100 epochs, `patience=20` early stopping — **no run reached 100** (range 24–61) |
| Image size | 640 |
| Label preprocessing | night-box filter applied (`filter_night_boxes.py --cut-dark pohang01:100`, 132k unlearnable boxes dropped); label hash `287b11c50b5a` verified identical on both machines |

All 27 `main` rows ran on fingerprint **`682dbe9f0f05`** with `classes 0`. The split
lists were verified byte-identical between the two machines via a handback archive.

**`results.csv` also holds 66 rows from the earlier `pilot` campaign**, which ran on a
different and unrecoverable partition, fingerprint **`f0220e716277`** — 22 further
variants covering YOLOv8, v9, v10 and YOLO11. They were merged into one table by
decision of the project owner; §15 records what that assumes and
`docs/phase1-pilot-grid.md` gives the full provenance. Everything in §2–§14 below
describes the `main` campaign unless it says otherwise.

**93 rows, 31 variants, one campaign per variant.** `yolo12s` was run by both; the
`main` seeds are kept and the pilot's are retired to `phase1_benchmark/extra/`, so
`(variant, seed)` is unique and no variant mixes splits.

---

## 2. Where the runs occurred

The grid was split across **two machines**. This is the origin of the most serious
defect in the dataset (§3).

### Machine A — DGX server (18 rows)

```
path         /workspace/Saroha_Work
GPU          MIG 3g.40gb slice of an H100 80GB  (40,320 MiB, 60 SMs — NOT the full card)
torch        2.12.1+cu126
ultralytics  8.4.90
workers      2   (capped by /dev/shm size)
git_commit   unrecorded ("unknown") for 17 of the 18
```

Variants: `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x`, `yolo26n`, `yolo26s` × seeds 0,1,2.

### Machine B — laptop (9 rows)

```
path         A:\Uncertain
GPU          NVIDIA RTX 4080 Laptop, 12,282 MiB
torch        2.7.1+cu118
ultralytics  8.4.7  -> upgraded to 8.4.90 on 2026-08-11
workers      8      (16 measured worse: 2.80 vs 4.00 it/s median)
batch        8      (uniform)
```

Variants: `yolo26m`, `yolo26l`, `yolo26x` × seeds 0,1,2. Machine B also re-validated
three server checkpoints (§4).

---

## 3. Defect 1 (critical, resolved): the mAP metric changed between ultralytics versions

**Ultralytics 8.4.7 reports mAP approximately 0.034 higher than 8.4.90 for identical
weights on identical data.** This is a measurement artifact. Training is unaffected.

### How it presented

The first laptop run, `yolo26m` seed 0, scored **0.3452** mAP50-95 while the best server
row was `yolo12x` at 0.2965 (seed-mean). A +0.05 jump, against a per-variant seed sd of
0.005–0.011. The entire `yolo12` family spans only 0.278–0.297, so a step of that size
at one rung of the ladder was not credible.

### The isolating experiment

The server's own trained checkpoint — `ship_yolo26s_seed0/weights/best.pt`, produced on
Machine A — was validated on Machine B under each library version. Same weights, same
images, same `classes=[0]`, same `imgsz`, same val call as `grid.py:231`.

| ultralytics | precision | recall | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 8.4.7  | 0.803 | 0.579 | 0.6619 | **0.3201** |
| 8.4.90 | 0.803 | 0.579 | 0.6235 | **0.2864** |
| server's own recorded value | — | — | 0.62349 | **0.28635** |

8.4.90 on the laptop reproduces the server's number to four decimal places.

### Where the change lives

Precision and recall shift by ~0.01% (measured on `yolo26m` seed 0: P 0.8225→0.8226,
R 0.6021→0.6022) while mAP50 shifts 6% and mAP50-95 shifts 11%. Precision and recall are
computed directly from matched detections, so **detection, NMS, and IoU matching are
unchanged**; the difference is confined to the average-precision integration over the
precision-recall curve.

The specific commit was **not** identified. If the paper needs it, diff
`ultralytics/utils/metrics.py` between tags `v8.4.7` and `v8.4.90`, most likely
`compute_ap`.

### What was ruled out first

| Hypothesis | Evidence against |
|---|---|
| Different data / split | fingerprint `682dbe9f0f05` identical; six split lists byte-identical; label hash `287b11c50b5a` matched |
| Batch size / effective batch / weight decay | server `yolo12x` seed1 (batch 8, eff 64, wd 0.000500) = 0.2955 vs seed2 (batch 24, eff 72, wd 0.000563) = 0.2972, i.e. +0.0017 — an order of magnitude below the 0.034 offset |
| GPU / torch / CUDA / worker count | all differ between machines; none reproduces the effect |
| Training divergence | offset-correcting the laptop `yolo26s` control curve made both machines agree within ±0.003 by epoch 8 |

> **CORRECTION.** The 08-11 revision compared the batch effect against a "0.0072 spread
> between two runs of the same seed at identical settings". That figure is withdrawn —
> see §4. The batch conclusion is unaffected: +0.0017 is far below 0.034 either way.

### Correction applied

- Machine B upgraded to `ultralytics==8.4.90` on 2026-08-11 (torch untouched —
  ultralytics only requires `>=1.8.0`).
- `yolo26m` seed 0 re-scored from its existing weights: **0.3452 → 0.3061** mAP50-95
  (mAP50 0.6879 → 0.6458). Its `ultralytics_version` reads `8.4.7+val8.4.90` to record
  that these weights were *trained* by 8.4.7 and only *re-scored* by 8.4.90.
- All runs from `yolo26m` seed 1 onward both train and score on 8.4.90.

### Effect on the conclusion

The headline did not survive. With all 27 rows in, `yolo26m` (0.3016) and `yolo12x`
(0.3007) are a tie, and `yolo26x` leads the field by 0.0033 against a seed sd of 0.0020.

---

## 4. Defect 2 (RESOLVED 2026-08-17): the `yolo12x` shared-directory collision

The 08-11 revision recorded two `yolo12x` seed-0 rows with "ambiguous provenance" and
treated their 0.0072 difference as an empirical floor on same-seed run-to-run variance.
A checkpoint-level audit settled what actually happened, and both conclusions were wrong.

### What happened

**Two grid processes shared one run directory** — including one `weights/` folder — for
`ship_yolo12x_seed0` and `ship_yolo12x_seed1`. Their `results.csv` files interleave
row-by-row with independent clocks:

```
ep 24  t 61455.4
ep  1  t  3513.4     <- second writer, own clock
ep 25  t 63972.4
ep  2  t  6646.1
```

Two detectors, both now computed for every run in `results.csv`:

- `dup_epoch_rows` — the same epoch number written twice
- `epoch_regressions` — the epoch column decreasing in file order

The second is the reliable one. A genuine resume continues the epoch count, so epochs
only ever increase; a decrease means a second process started from epoch 1 into the same
file. Seed 1 has **11 regressions and zero duplicates** — a duplicate-only test misses
it entirely.

`yolo12x` seeds 0 and 1 are the only affected directories. **Every other run in the grid
is single-writer** — verified, all 25 have `dup_epoch_rows = 0` and
`epoch_regressions = 0`.

### What each row actually measured

Because `best.pt` is whichever process saved last, the surviving checkpoint need not
belong to the lineage whose curve you are reading. Re-validating the archived
checkpoints on Machine B settles it:

| checkpoint | re-validated | CSV row | Δ |
|---|---|---|---|
| `12x` seed 2 (clean, single-writer) — **control** | 0.29717 | 0.29719 | −0.00002 |
| `12x` seed 0, inner dir, batch 8, ran to a natural stop | 0.30414 | 0.30413 | +0.00001 |
| `12x` seed 0, outer dir, batch 16, killed at ep 16 | 0.29569 | 0.29691 | −0.00122 |
| `12x` seed 1, batch 8, killed | 0.29582 | 0.29548 | +0.00034 |

The control reproduces to 2e-5, so cross-machine re-validation is exact and the two
large deltas are real: **neither the 0.29691 row nor the seed-1 row is reproducible from
any checkpoint that still exists.** Both were written before a concurrent process
overwrote `best.pt`.

> **CORRECTION — the 0.0072 "noise at identical settings" figure is withdrawn.** The two
> seed-0 rows were never two runs at identical settings. `args.yaml` shows one trained at
> **batch 16** and was killed at epoch 16 (peak 0.29408 at ep 6); the other trained at
> **batch 8** and ran to a natural early stop at epoch 29. The 0.0072 measured a batch
> difference between a killed run and a completed one. Use the per-variant cross-seed sd
> (0.0020–0.0109) instead.

### Resolution

- `yolo12x` seed 0 is the artifact-backed **0.30414** row, re-derived from the inner
  directory's surviving `best.pt`. The withdrawn 0.29691 row is kept in
  `phase1_benchmark/superseded.csv` with its reason.
- The outer batch-16 writer's directory is kept at
  `phase1_benchmark/extra/yolo12x_seed0_outer_writer/`.
- The server's own `ship_yolo12x_seed0_val/` cannot be attributed to either row — both
  processes wrote that one directory too. Kept as
  `extra/yolo12x_seed0_server_val_ambiguous/`, not cited.
- `yolo12x` seed 1 is **inadmissible** (§6): it trained only 14 epochs past its peak.
  Its row stays in `results.csv` as raw record and is excluded from Table 1, leaving
  `yolo12x` at n=2.

---

## 5. Defect 3 (documented, reported as a limitation): heterogeneous batch size

Ultralytics computes `accumulate = max(round(64 / batch), 1)` and scales weight decay as
`base_wd × batch × accumulate / 64`. The grid did not hold batch fixed:

| batch | runs | accumulate | effective batch | effective wd |
|---|---|---|---|---|
| 32 | 9 (`12s`, `12m`, `12l` × 3) | 2 | 64 | 0.000500 |
| 24 | 7 (`26n`×3, `26s`×3, `12x` seed2) | 3 | **72** | **0.000563** |
| 16 | `12x` seed0, outer writer (not a Table 1 row) | 4 | 64 | 0.000500 |
| 8 | `12x` seed0 + seed1, and all 9 laptop runs | 8 | 64 | 0.000500 |

So 7 of 27 rows trained at a 12.5% higher effective weight decay and a different
effective batch than the rest. `batch` is now a column in `phase1_benchmark/results.csv`
— in the old CSV this confound was invisible.

The `12x` seed1-vs-seed2 contrast bounds the effect at +0.0017, comparable to the
smallest seed sd in the grid (0.0020) — so it is **reported as a limitation, not
treated as a confound**.

---

## 6. Early stopping, resume, and the admissibility rule

> **CORRECTION — fitness is `mAP50-95` alone in ultralytics 8.4.90**, not
> `0.1*mAP50 + 0.9*mAP50-95`. Verified against `best_fitness` stored in un-stripped
> checkpoints (`26l` seed 2: stored 0.2991 == max mAP50-95 0.29910, against a blend of
> 0.33413; `26m` seed 2: 0.30201 == 0.30201). **Every peak-epoch statement computed with
> the blend in earlier revisions is wrong** and has been recomputed.

### The resume defect

`Trainer.resume_training()` restores the optimizer, scaler, EMA and `best_fitness`, but
the `EarlyStopping` object is rebuilt one line earlier (`trainer.py:381`) with
`best_fitness = 0.0` and `best_epoch = 0`, and nothing restores it. Because
`EarlyStopping.__call__` treats `best_fitness == 0` as "always an improvement"
(`torch_utils.py:995`), the first validated epoch after a resume becomes the stopper's
new reference epoch. The countdown then runs from a *local* peak inside the final
segment rather than from the run's global peak.

`best.pt` is **not** corrupted — the restored `best_fitness` prevents a weaker later
epoch from overwriting it. What breaks is training *length*: a resumed run can train
past the point an uninterrupted one would have stopped.

`results.csv` records both epochs so the distinction is legible per run:
`best_epoch` (the global peak, i.e. what `best.pt` holds) and `stopper_ref_epoch` (what
patience was actually counting from at the end).

> **CORRECTION — the damage was much smaller than the 08-11 revision recorded.**
> Recomputed on the correct fitness definition, **23 of the 25 uncollided runs stopped
> at exactly `best_epoch + 20`**. `26x` seed 2 in particular peaked at **ep 30, not
> ep 6**; its stop at ep 50 is a clean `peak + 20`, and the epochs after its resume
> produced the checkpoint it kept. The earlier claim that it "trained 23.6 h for a
> `best.pt` already fixed at ep 6" was an artifact of reading the curve with the wrong
> fitness.

### The admissibility rule

> A run is admissible if it trained **≥ `patience` epochs past its own peak**. Then
> `best.pt` holds the weights an uninterrupted run would have stopped on, however many
> times it was actually interrupted.

This is a claim about the counterfactual, not about whether ultralytics' own
resume-damaged stopper happened to fire. Two runs (`26m` seed 2, `26l` seed 2) were
killed while their reset stopper was still counting, yet had already gone ≥20 epochs
past their global peak — their weights are exactly what a clean run would have kept, so
they are admissible.

Applied uniformly, this is the **only** exclusion criterion in the record; no run is
dropped by hand. It admits two runs that had been set aside and rejects one that had
been kept:

- **`26m` seed 2 — recovered.** Peak ep 10, ran to ep 33 (+23). 0.30244, 39,677 s.
- **`26l` seed 2 — recovered.** Peak ep 6, ran to ep 26 (+20). 0.29947, 37,533 s.
- **`yolo12x` seed 1 — rejected.** Peak ep 22, last ep 36 (+14). Lower bound only.

26 of 27 rows are admissible.

### How the runs ended

| `stop_reason` | n |
|---|---|
| `early_stop` — never paused, stopped at `best_epoch + 20` | 21 |
| `early_stop_after_resume` — resumed, but the global peak fell inside the final segment | 2 |
| `killed_past_patience` — stopper had not fired, but ≥20 epochs past peak | 2 |
| `shared_dir_past_patience` — `12x` seed 0 | 1 |
| `shared_dir_before_patience` — `12x` seed 1 | 1 |

---

## 7. Defect 4: `git_commit` unrecorded for the server rows

`grid.py` writes a `git_commit` column. It reads **`unknown` for 17 of the 18 server
rows**; the exception is `yolo12x` seed 0, whose row was re-derived on the laptop and so
carries `70dc15f` — the commit of the *re-validation*, not of the training. The exact
code state that produced two thirds of the dataset is unrecorded. Worth a sentence in
limitations, and worth fixing in the harness before any future grid.

---

## 8. Defect 5: `train_time_s` under-reports resumed runs

`grid.py:219` computes `train_time = time.time() - t0`, i.e. **this invocation only**.
For any run that was paused and resumed, the value the grid wrote covers the last
segment alone. Rows banked by `scripts/recover_row.py` carry cumulative time instead, so
the column is not consistently defined across rows.

The sum over the record, 266.8 h, is therefore a **lower bound**. Recompute from each
run's `results.csv` before publishing any timing column. Note that the two shared-dir
runs cannot be recomputed this way either: summing per-segment clocks double-counts
overlapping writers.

---

## 9. Defect 6 (environmental): two opencv distributions

The 8.4.90 upgrade pulled in `opencv-python 5.0.0.93`, while `opencv-python-headless
4.10.0.84` remained installed. Both claim the `cv2` namespace; `cv2.__version__`
resolves to **4.10.0**.

This was incidentally *helpful* — it meant the upgrade changed only ultralytics, making
§3 a clean single-variable test — but it is fragile and should be resolved.

---

## 10. Runs kept out of the results

All under `phase1_benchmark/extra/`, catalogued in `extra/CONTENTS.json`.

| directory | what it is |
|---|---|
| `yolo12x_seed0_outer_writer` | the batch-16 half of the seed-0 collision; killed at ep 16; its row is withdrawn |
| `yolo12x_seed0_outer_writer_val` | laptop re-validation of the above |
| `yolo12x_seed0_server_val_ambiguous` | the server's own seed-0 validation output; cannot be attributed to either colliding row |
| `yolo12x_seed1_revalidation` | laptop re-validation of seed 1; did not reproduce the server row |
| `yolo12x_seed2_revalidation_control` | the control that proved re-validation is exact (−0.00002) |
| `yolo26m_seed0_aborted_server` | one epoch (mAP50-95 0.2625 at ep 1, batch 24), abandoned when the server session ended 2026-08-10 22:40; never written to the CSV. The only server-side `yolo26m` data point, but a single epoch cannot detect the §3 metric defect — the laptop/server gap at ep 1 was −0.003 for `26m` and +0.008 for `26s`, growing to +0.037 by ep 8 |
| `yolo26m_seed0_partial_batch8`, `..._batch16` | abandoned laptop attempts at batch 8 and 16 |

---

## 11. Operational failures during the campaign

Not scientific defects, but each cost real time and each is a reproducibility hazard.

1. **cmd.exe comment inside a line continuation.** A `REM` line placed between
   `^`-continued arguments terminates the command; the next flag is then parsed as its
   own command (`'--workers' is not recognized`). Crash-looped the grid for ~45 min
   across 13 wrapper attempts before detection.
2. **Console-close event killed a run and both crash-safety layers.** A run launched
   with `Start-Process` died with `forrtl: error (200): program aborting due to
   window-CLOSE event` when a console teardown propagated to its process group. Because
   `CTRL_CLOSE_EVENT` reaches the entire group, it also killed the retry wrapper, so the
   20-attempt recovery never fired. Fix: launch via `Invoke-CimMethod Win32_Process
   Create`, which creates no console relationship.
3. **Windows `multiprocessing` spawn deadlock.** A diagnostic script calling
   `model.val()` without an `if __name__ == "__main__":` guard hung indefinitely — each
   spawned child re-imports the module and re-enters `val()`. Symptom: process alive,
   ~10 s CPU over 25 min, no output. Also avoidable with `workers=0`.
4. **Stale progress frames misread as live.** A killed run's progress bars remain in the
   log tail and look like active training. Verification must anchor on the last
   `[wrapper] attempt` / `[grid] ===` banner and read only lines after it. Ultralytics
   also prefixes frames with ANSI `\x1b[K`, defeating `^`-anchored regexes.
5. **Scheduled monitoring does not run unattended.** Both a cron job and a
   self-scheduled wakeup loop produced zero reports while the operator was away. Only
   `scripts/watch_grid.py`, writing to `runs/health.log` every 10 min as an independent
   process, proved reliable — and even that died 5+ times, once staying *alive but
   silent*. Compare `health.log`'s mtime against the clock, not just process existence.
6. **Windows Update rebooted the machine** (Event 1074, TrustedInstaller) and killed
   grid + wrapper + watchdog, costing ~4h40m. Pause updates before a multi-day run.
7. **Never open the results CSV in Excel.** It rewrote the file on save:
   `ultralytics_version` "8.4.90" became the date serial `0.337152778` on 25 rows and
   blank on one, and every float was truncated to 9 significant digits. It also held a
   write lock that made `_append_row` fail with `PermissionError` mid-recovery. The
   damaged copy is kept as `runs/benchmark/_excel_mangled_20260817_1410.csv`.
8. **A killed run writes no CSV row, and the grid cannot recover it.** Rows are written
   on completion only, and `grid.py::_completed()` skips by CSV row — so relaunching
   *resumes* the killed run rather than skipping it, and keeps training a run that was
   already finished. `scripts/recover_row.py` does the missing half: validate `best.pt`,
   append the row, refuse if the run is inadmissible.

---

## 12. Results

`phase1_benchmark/results.csv`, 93 rows, all on the 8.4.90 metric scale. The full
31-variant table is in `phase1_benchmark/README.md`; the `main` campaign alone is:

### Seed-means, `main` campaign (mAP50-95, admissible rows only)

```
yolo26x   n=3   0.3049 ± 0.0020
yolo26m   n=3   0.3016 ± 0.0050
yolo12x   n=2   0.3007 ± 0.0049
yolo26l   n=3   0.2998 ± 0.0026
yolo12m   n=3   0.2906 ± 0.0046
yolo12l   n=3   0.2870 ± 0.0096
yolo26s   n=3   0.2813 ± 0.0045
yolo12s   n=3   0.2783 ± 0.0109
yolo26n   n=3   0.2540 ± 0.0058
```

### Pooled across both campaigns — the leading group, and how it changed

```
yolo26x   main   n=3   0.3049 ± 0.0020
yolov9e   pilot  n=3   0.3033 ± 0.0066
yolov9c   pilot  n=3   0.3019 ± 0.0057
yolo26m   main   n=3   0.3016 ± 0.0050
yolo12x   main   n=2   0.3007 ± 0.0049
yolov8x   pilot  n=3   0.2999 ± 0.0010
yolo26l   main   n=3   0.2998 ± 0.0026
yolo11l   pilot  n=3   0.2994 ± 0.0041
```

Pooling **widened** the inseparable group rather than resolving it: from four variants
spanning 0.0051 to **eight spanning 0.0055**, and `yolo26x`'s margin over second place
fell from 0.0033 to **0.0016 — below its own seed sd of 0.0020**. Any claim that one of
these eight beats another is unsupported. The one clean signal across all 31 variants is
the capacity floor: every `n`/`t`-scale model lands at 0.2486–0.2567, ~0.05 below the
leaders, with no exceptions in any family.

### Epochs actually run (`patience=20`; none reached 100)

```
12s  31 / 38 / 32      12m  46 / 61 / 45      12l  34 / 50 / 34
12x  29 / 36 / 24      26n  36 / 43 / 36      26s  31 / 31 / 42
26m  29 / 26 / 33      26l  28 / 28 / 26      26x  28 / 32 / 50
```

### Robustness

Every Table 1 number is cross-checked against the in-training validator's curve — a
second record that does not depend on which `best.pt` survived
(`docs/phase1-robustness-table.md`).

Explicit-val minus curve-peak across the 26 comparable runs: **+0.00110 ± 0.00092**
(range −0.0001 … +0.0026). The offset splits cleanly by **family, not by machine**:

| family | offset |
|---|---|
| `yolo26*` (server **and** laptop) | +0.0003 |
| `yolo12*` | +0.0021 |

Plausibly because YOLO26 is NMS-free, so its in-training and standalone validators take
the same path. **It matters**: ~0.002 is the size of the gaps inside the top group, and
the two metric sources rank the top four differently — Table 1 gives
`26x > 12x > 26l > 26m`, curve-peak gives `26x > 26l > 26m > 12x`. Below the top group
the order is identical under both. Keep Table 1 on one metric source and say which.

`26m` seed 0 is excluded from that comparison: it *trained* under 8.4.7, so its curve is
on the old scale while its Table 1 metric was re-scored under 8.4.90. Its delta of
−0.0387 is independent confirmation of §3.

### How to read the ranking

**The top four (`26x`, `26m`, `12x`, `26l`) span 0.0051 against seed sds of
0.0020–0.0050. They are not separable, and their order is not stable across two
legitimate readings of the same runs.** What is resolved: all four beat `12m`/`12l` by
≥0.009, and `26n`/`12s`/`26s` are clearly worse.

> **CORRECTION.** The claim that the 26-series large models beat the whole 12-series
> field **does not survive** — `12x` sits inside the top group.

---

## 13. Required before publication

1. **`yolo12x` seed 1 is inadmissible and unreproducible.** Retraining it on the laptop
   (~17.5 h) is the only route back to a clean n=3 at one batch size. Its current value
   is a lower bound, so a full run would likely push `12x` *up*, not down.
2. **Decide on `yolo26m` seed 0.** Its weights were trained under 8.4.7 and re-scored
   under 8.4.90. Evidence says training is version-equivalent (±0.003 by epoch 8), so
   the number stands — but its early-stopping decisions used the old fitness scale, and
   it is `26m`'s strongest seed. Awkward, given the paper's own headline is that an
   unpinned ultralytics corrupted the metric. ~11 h to redo.
3. **Report epochs actually trained, not `epochs_cfg`.** `epochs_cfg` reads 100 on every
   row and no run got near it. Use `epochs_trained` / `last_epoch` / `best_epoch`.
4. **Recompute `train_time_s`** from the curves before publishing any timing column
   (§8).
5. **Pin `ultralytics==8.4.90`** in requirements, and record the pin in the paper.
6. **Regenerate the server environment lock.** The existing `requirements.lock.txt` is
   unusable: 84 of 87 lines are conda `@ file:///home/task_.../croot/...` build paths
   and torch/ultralytics are absent entirely, because the capture ran under base
   `python` rather than the `saroha_work` env. Redo with the env activated using
   `python -m pip list --format=freeze`.
7. **Resolve the opencv double-install** (§9).
8. **`scripts/make_table1.py` must consume the new columns** — filter on `admissible`,
   report `epochs_trained`, and recompute timing.

---

## 14. Suggested limitations text

> Runs were distributed across two machines. All results are reported under ultralytics
> 8.4.90; an earlier laptop run collected under 8.4.7 was re-scored from its saved
> weights, after we established that 8.4.7 reports mAP50-95 approximately 0.034 higher
> than 8.4.90 for identical weights on identical data. Precision and recall are
> unaffected to within 0.01%, localizing the discrepancy to average-precision
> integration rather than to detection or matching. We recommend that detector
> benchmarks pin and report the exact evaluation library version; without it,
> cross-machine comparisons in this study would have been wrong by roughly five times
> the observed seed standard deviation.

> Early stopping used `patience=20` on mAP50-95. Several runs were paused and resumed;
> ultralytics rebuilds its `EarlyStopping` state on resume without restoring it, so a
> resumed run counts patience from a local rather than a global optimum. We verified
> that this does not affect which checkpoint is retained — the trainer's `best_fitness`
> *is* restored — and we admit a run only if it trained at least `patience` epochs past
> its own optimum, which is the sole exclusion criterion applied. One of 27 runs fails
> this test and is reported but excluded.

Batch-size heterogeneity (§5), the unrecorded `git_commit` for the server rows (§7), the
shared-directory collision affecting `yolo12x` (§4), and the inseparability of the top
four variants (§12) should each be stated explicitly.

---

## 15. The pilot campaign (`f0220e716277`) — merged, and what that assumes

A complete earlier grid exists: **23 variants × 3 seeds = 69 runs** covering YOLOv8
(n/s/m/l/x), YOLOv9 (t/s/m/c/e), YOLOv10 (n/s/m/b/l/x), YOLO11 (n/s/m/l/x) and YOLO12
(n/s), ship-only, ultralytics 8.4.90, run 2026-07-31.

**Its 66 surviving rows are merged into `results.csv` under `grid = pilot`**, by decision of the
project owner, who reviewed the analysis below and judged the two campaigns comparable.
Full provenance is in `docs/phase1-pilot-grid.md`. This section records the cost of that
decision so it can be stated in the paper.

### The split is different and unrecoverable

The `data_vis_stride2.yaml` behind fingerprint `f0220e716277` existed only on a server
that was wiped, and it matches no split generation reproducible from this repo — the
current full split, the current stride-2, the pre-resplit full split and its ordinal
stride, naive strides, cross-validation combinations and L/R camera subsets were all
tested. What the surviving artifacts show:

```
pilot val begins at   pohang00_L_006750
main  val begins at   pohang00_L_006767          (17 frames later)

current val.txt      pohang00_L blocks:  6767–7184,  12446–12863
pre-resplit train    pohang00_L blocks:  4767–11183, 11186–13439   <- contains both
frames 6750–6766     in the pre-resplit train list; in NO current list
```

The 2026-07-14 resplit dropped 3,673 frames outright (122,745 → 119,072), including the
block where the pilot's val begins. Both campaigns cut val as contiguous frame blocks
from the same sequence, offset by 17 frames, so the pilot's partition is a *third* cut.

Two consequences follow:

1. **Possible train/val contamination, unquantifiable.** If the pilot's val block was
   also 418 frames from 6750, it ran 6750–7167, and the `main` val frames 7168–7184 were
   in the pilot's *training* set. The block length is unknowable — the pilot's train list
   is gone and its artifacts plot only the first ~96 val images.
2. **The training data differs regardless of contamination.** Pilot models saw a
   different image pool with a different train/val boundary, so a `pilot` vs `main`
   comparison varies architecture *and* training data at once. Re-validating the pilot
   checkpoints on the current split would fix the measurement but not this.

### The one direct probe

`yolo12s` is the only variant both campaigns ran, giving 3 `main` seeds against 2
`pilot` seeds: **0.2783 vs 0.2810, a gap of +0.0027** — inside the `main` campaign's own
`yolo12s` seed sd of 0.0109. Consistent with the campaigns being comparable, but two
seeds cannot separate a split effect from seed noise, and the batch differs by one
(32 vs 31) as well. It is the only evidence available either way.

The pilot's `yolo12s` runs are **retired from the table** — `main` measures that variant —
so each variant now sits in exactly one campaign and no `(variant, seed)` repeats. Their
artifacts are kept in `phase1_benchmark/extra/pilot_yolo12s_seed*_superseded_by_main/`,
which is what keeps the probe above reproducible.

### It was the better-run experiment

This is the uncomfortable part, and it belongs in the reproducibility discussion:

| | `pilot` | `main` |
|---|---|---|
| batch | **31, uniform across all 69 runs** | 8 / 16 / 24 / 32 |
| machines | one | two, with an ultralytics version change between them |
| shared run directories | none | 2 |
| resumed runs | none | 4 |
| clean `peak + patience` stops | 68 of 69 | 23 of 25 uncollided |
| split lists | **lost** | archived and verified |

A 69-run, 6 GB, perfectly uniform campaign lost its provenance to a single missing file.
The lesson generalizes: **fingerprint the split *and* archive the split lists next to the
results.** A fingerprint proves two result sets share a split; it does not let you
reconstruct one. The `main` campaign survives only because the handback archive (§2)
carried the lists themselves.

### What the pilot contributes

1. **Family coverage.** It is the only measurement of YOLOv8, v9, v10 and YOLO11 on this
   task — the `main` campaign covers only YOLO12 and YOLO26. It answers "why were older
   families not considered" with per-seed data across 23 variants.
2. **The capacity floor.** Pooled, all six `n`/`t`-scale variants across five families
   land in 0.2486–0.2567. That is the most robust statement the record supports, and it
   needs the pilot to be a statement about detectors rather than about YOLO26.
3. **Epoch-budget calibration.** Stop epochs ran 25–63 at `patience = 20`, none near the
   100-epoch budget, consistently across four model families — which justified keeping
   the same budget for `main`.
4. **One unbanked run.** `pilot_yolo12s_seed2` has complete artifacts and no CSV row: it
   was killed 3 epochs past its peak, so it never reached the validation that writes a
   row. It is inadmissible under the §6 rule anyway, and cannot be recovered —
   re-validating needs the lost val split. Hence `pilot_yolo12s` is n=2.

### What it costs

The leading group went from four variants to eight, `yolo26x`'s margin fell below its
own seed sd (§12), and every cross-grid comparison now carries the split caveat above.
**Any table mixing `main` and `pilot` rows must say so and cite this section.**
