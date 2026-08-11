# Phase 1 experimental record — provenance, defects, and corrections

Last updated 2026-08-11, mid-grid. Source record for the methods and limitations sections
of the paper. Everything here is measured, not inferred, unless explicitly marked.

---

## 1. What the experiment is

A multi-seed detector-backbone benchmark on the Pohang maritime dataset (visible spectrum),
**ship class only**, feeding Table 1.

| | |
|---|---|
| Task | single-class detection (`--classes 0`, ship) |
| Split | `runs/derived/data_vis_stride2.yaml`, stride-2 subset |
| Split fingerprint | `682dbe9f0f05` — SHA256 over sorted `run/filename` ids, train+val |
| Train images | 48,136 |
| Val images | 11,352 (8,797 contain ≥1 ship) |
| Val instances | 105,098 |
| Design | 9 variants × 3 seeds (0,1,2) = **27 runs** |
| Budget | 100 epochs, `patience=20` early stopping — no run reached 100 |
| Image size | 640 |
| Label preprocessing | night-box filter applied (`filter_night_boxes.py --cut-dark pohang01:100`, 132k unlearnable boxes dropped); label hash `287b11c50b5a` verified identical on both machines |

All 20 rows collected so far carry `split_fingerprint 682dbe9f0f05` and `classes 0`.
The split lists were verified byte-identical between the two machines via a handback archive.

---

## 2. Where the runs occurred

The grid was split across **two machines**. This is the origin of the most serious defect
in the dataset (§3).

### Machine A — DGX server (18 unique rows + 1 duplicate)

```
path         /workspace/Saroha_Work
GPU          MIG 3g.40gb slice of an H100 80GB  (40,320 MiB, 60 SMs — NOT the full card)
torch        2.12.1+cu126
ultralytics  8.4.90
workers      2   (capped by /dev/shm size)
run_dir      /workspace/Saroha_Work/runs/benchmark/runs/...
git_commit   unrecorded ("unknown") for all 19 rows
```

Variants: `yolo12s`, `yolo12m`, `yolo12l`, `yolo12x`, `yolo26n`, `yolo26s` × seeds 0,1,2.

### Machine B — laptop (1 row so far, 8 more running)

```
path         A:\Uncertain
GPU          NVIDIA RTX 4080 Laptop, 12,282 MiB
torch        2.7.1+cu118
ultralytics  8.4.7  -> upgraded to 8.4.90 on 2026-08-11
workers      8      (16 measured worse: 2.80 vs 4.00 it/s median)
batch        8      (uniform)
run_dir      A:\Uncertain\runs\benchmark\runs\...
git_commit   6e24167
```

Variants: `yolo26m`, `yolo26l`, `yolo26x` × seeds 0,1,2.

---

## 3. Defect 1 (critical, resolved): the mAP metric changed between ultralytics versions

**Ultralytics 8.4.7 reports mAP approximately 0.034 higher than 8.4.90 for identical
weights on identical data.** This is a measurement artifact. Training is unaffected.

### How it presented

The first laptop run, `yolo26m` seed 0, scored **0.3452** mAP50-95 while the best server
row was `yolo12x` at 0.2965 (seed-mean). A +0.05 jump, against a per-variant seed sd of
0.005–0.011. The entire `yolo12` family spans only 0.278–0.297, so a step of that size at
one rung of the ladder was not credible.

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
`ultralytics/utils/metrics.py` between tags `v8.4.7` and `v8.4.90`, most likely `compute_ap`.

### What was ruled out first

| Hypothesis | Evidence against |
|---|---|
| Different data / split | fingerprint `682dbe9f0f05` identical; six split lists byte-identical; label hash `287b11c50b5a` matched |
| Batch size / effective batch / weight decay | server `yolo12x` seed1 (batch 8, eff 64, wd 0.000500) = 0.2955 vs seed2 (batch 24, eff 72, wd 0.000563) = 0.2972, i.e. +0.0017 — smaller than the 0.0072 spread between two runs of the *same* seed at identical settings |
| GPU / torch / CUDA / worker count | all differ between machines; none reproduces the effect |
| Training divergence | offset-correcting the laptop `yolo26s` control curve made both machines agree within ±0.003 by epoch 8 |

### Correction applied

- Machine B upgraded to `ultralytics==8.4.90` on 2026-08-11 (torch untouched — ultralytics
  only requires `>=1.8.0`).
- `yolo26m` seed 0 re-scored from its existing weights: **0.3452 → 0.3061** mAP50-95
  (mAP50 0.6879 → 0.6458). Row patched in place; pre-correction file preserved as
  `benchmark_results_tail.csv.bak_pre8490`.
- Its `ultralytics_version` field reads `8.4.7+val8.4.90` to record that these weights were
  *trained* by 8.4.7 and only *re-scored* by 8.4.90.
- All runs from `yolo26m` seed 1 onward both train and score on 8.4.90.

### Effect on the conclusion

The headline result did not survive. Corrected, `yolo26m` seed 0 (0.3061) beats `yolo12x`
seed 0 (0.3041) by **+0.002** — a tie, not a breakthrough.

---

## 4. Defect 2 (unresolved): duplicate `yolo12x` seed 0 rows with ambiguous provenance

Two rows exist for `yolo12x` seed 0, written by two concurrent grid processes on Machine A:

```
seed 0   mAP50-95 0.29691   train_time 47,422 s   run_dir .../ship_yolo12x_seed0
seed 0   mAP50-95 0.30413   train_time 85,413 s   run_dir .../ship_yolo12x_seed0
```

They differ by **0.0072** — larger than the batch-size effect measured in §3, and a useful
empirical floor on run-to-run variance at fixed seed.

The archive makes this worse rather than better: there is **no `ship_yolo12x_seed0`
training directory**. What survives is `ship_yolo12x_seed0_val/` (validation plots only)
and `_partial_ship_yolo12x_seed0/`, whose `args.yaml` records **`batch: 16`** and whose
`results.csv` holds 16 epochs. Batch 16 is used by no other run in the study.

**Neither CSV row can currently be matched to a known configuration.** This must be
resolved before Table 1 — as it stands the row double-weights seed 0 in any `yolo12x`
seed-mean (which is why the provisional table below shows n=4).

---

## 5. Defect 3 (documented, judged inconsequential): heterogeneous batch size on Machine A

Ultralytics computes `accumulate = max(round(64 / batch), 1)` and scales weight decay as
`base_wd × batch × accumulate / 64`. The server grid did not hold batch fixed:

| batch | rows | accumulate | effective batch | effective wd |
|---|---|---|---|---|
| 32 | 9 (`12s`, `12m`, `12l` × 3) | 2 | 64 | 0.000500 |
| 24 | 7 (`26n`×3, `26s`×3, `12x` seed2) | 3 | **72** | **0.000563** |
| 16 | `12x` seed0 partial | 4 | 64 | 0.000500 |
| 8 | `12x` seed1 | 8 | 64 | 0.000500 |

So 7 of 19 server rows trained at a 12.5% higher effective weight decay and a different
effective batch than the rest. Machine B uses batch 8 uniformly (effective 64, wd 0.000500).

The `12x` seed1-vs-seed2 contrast (§3) bounds this effect at +0.0017, below the 0.0072
same-seed spread — so it is **reported as a limitation, not treated as a confound**.

---

## 6. Defect 4 (unresolved): `git_commit` unrecorded for 19 of 20 rows

`grid.py` writes a `git_commit` column. It reads **`unknown` for all 19 server rows** and
`6e24167` only for the laptop row. The exact code state that produced the majority of the
dataset is therefore unrecorded. Worth a sentence in limitations, and worth fixing in the
harness before any future grid.

---

## 7. Defect 5 (documented): orphaned server `yolo26m` run

`archive/phase1/main_2026-08-10/.../ship_yolo26m_seed0/` contains **one epoch**
(mAP50-95 0.2625 at epoch 1, batch 24), abandoned when the server session ended on
2026-08-10 22:40. It was never written to the CSV.

It is diagnostically useful — it is the only server-side `yolo26m` data point — but it
must stay out of the results. Note that a single epoch cannot detect the §3 metric defect:
the laptop/server gap at epoch 1 was −0.003 for `yolo26m` and +0.008 for `yolo26s`, and only
grew with training (reaching +0.037 by epoch 8 on `yolo26s`).

---

## 8. Defect 6 (unresolved, environmental): two opencv distributions

The 8.4.90 upgrade pulled in `opencv-python 5.0.0.93`, while `opencv-python-headless
4.10.0.84` remained installed. Both claim the `cv2` namespace; `cv2.__version__` currently
resolves to **4.10.0**.

This was incidentally *helpful* — it meant the upgrade changed only ultralytics, making §3
a clean single-variable test — but it is fragile and should be resolved before the grid ends.

---

## 9. Operational failures during the campaign

Not scientific defects, but each cost real time and each is a reproducibility hazard worth
recording.

1. **cmd.exe comment inside a line continuation.** A `REM` line placed between `^`-continued
   arguments terminates the command; the next flag is then parsed as its own command
   (`'--workers' is not recognized`). Crash-looped the grid for ~45 min across 13 wrapper
   attempts before detection.
2. **Console-close event killed a run and both crash-safety layers.** A run launched with
   `Start-Process` died with `forrtl: error (200): program aborting due to window-CLOSE
   event` when a console teardown propagated to its process group. Because
   `CTRL_CLOSE_EVENT` reaches the entire group, it also killed the retry wrapper, so the
   20-attempt recovery never fired. Fix: launch via `Invoke-CimMethod Win32_Process Create`,
   which creates no console relationship. Loss was limited to ~7 min only because `grid.py`
   resumes from `weights/last.pt` at epoch level.
3. **Windows `multiprocessing` spawn deadlock.** A diagnostic script calling `model.val()`
   without an `if __name__ == "__main__":` guard hung indefinitely — each spawned child
   re-imports the module and re-enters `val()`. Symptom: process alive, ~10 s CPU over
   25 min, no output. Also avoidable with `workers=0`.
4. **Stale progress frames misread as live.** A killed run's progress bars remain in the log
   tail and look like active training. Verification must anchor on the last
   `[wrapper] attempt` / `[grid] ===` banner and read only lines after it. Ultralytics also
   prefixes frames with ANSI `\x1b[K`, defeating `^`-anchored regexes — strip ANSI first.
5. **Scheduled monitoring does not run unattended.** Both a cron job and a self-scheduled
   wakeup loop were tried; both produced zero reports while the operator was away. Only
   `scripts/watch_grid.py`, writing to `runs/health.log` every 10 min as an independent
   process, proved reliable.

---

## 10. Current results

`runs/benchmark/benchmark_results_tail.csv`, 20 rows, **all on the 8.4.90 metric scale**.

### Provisional seed-means (mAP50-95)

```
yolo26m   n=1   0.3061     <- one seed only; not yet a result
yolo12x   n=4   0.2984     <- inflated by the duplicate row (§4)
yolo12m   n=3   0.2906
yolo12l   n=3   0.2870
yolo26s   n=3   0.2813
yolo12s   n=3   0.2783
yolo26n   n=3   0.2540
```

### Epochs actually run (early stopping, `patience=20`; none reached 100)

```
Machine A   12s  31 / 38 / 32        12m  46 / 61 / 45       12l  34 / 50 / 34
            12x  ?  / 35 / 24        26n  36 / 43 / 36       26s  31 / 31 / 42
Machine B   26m  29 / running / pending
```

### Completion

```
DONE         yolo26m seed0                     0.3061   (8.4.7 train + 8.4.90 val)
RUNNING      yolo26m seed1                     8.4.90
PENDING      yolo26m seed2, yolo26l x3, yolo26x x3
```

Estimated remaining: ~116 h (~4.8 days) at ~11 h/run for `26m`, ~13 h for `26l`, ~19 h for `26x`.

---

## 11. Required before publication

1. **Resolve the duplicate `yolo12x` seed 0 row** (§4). Currently unmatched to any known
   configuration. Blocks Table 1.
2. **Finish the 8 remaining runs** and confirm `yolo26m`'s +0.002 lead over `yolo12x`
   survives three seeds. It currently rests on one seed against a seed sd of 0.005–0.011.
3. **Decide on `yolo26m` seed 0.** Its weights were trained under 8.4.7 and re-scored under
   8.4.90. Evidence says training is version-equivalent (±0.003 by epoch 8), so the number
   stands — but a ~11 h retrain would allow the cleaner claim that every row was both
   trained and evaluated under one version.
4. **Pin `ultralytics==8.4.90`** in requirements, and record the pin in the paper.
5. **Regenerate the server environment lock.** The existing `requirements.lock.txt` is
   unusable: 84 of 87 lines are conda `@ file:///home/task_.../croot/...` build paths and
   torch/ultralytics are absent entirely, because the capture ran under base `python`
   rather than the `saroha_work` env. Redo with the env activated using
   `python -m pip list --format=freeze`.
6. **Resolve the opencv double-install** (§8).

---

## 12. Suggested limitations text

> Runs were distributed across two machines. All results are reported under ultralytics
> 8.4.90; an earlier laptop run collected under 8.4.7 was re-scored from its saved weights,
> after we established that 8.4.7 reports mAP50-95 approximately 0.034 higher than 8.4.90
> for identical weights on identical data. Precision and recall are unaffected to within
> 0.01%, localizing the discrepancy to average-precision integration rather than to
> detection or matching. We recommend that detector benchmarks pin and report the exact
> evaluation library version; without it, cross-machine comparisons in this study would
> have been wrong by roughly five times the observed seed standard deviation.

Batch size heterogeneity (§5), the unrecorded `git_commit` for the server rows (§6), and
the unresolved duplicate row (§4) should each be stated explicitly.
