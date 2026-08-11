# Phase 1 handoff — 2026-08-11

Laptop backbone-benchmark grid, Pohang maritime VIS, ship-only. Written mid-run so a
new session can pick up without re-deriving anything.

Repo `A:\Uncertain`, branch `main`, HEAD `6e24167` (working tree dirty — see git status).

---

## 1. The headline finding: an unpinned ultralytics version corrupted the metric

**Ultralytics 8.4.7 reports mAP roughly 0.034 higher than 8.4.90 for identical weights
on identical data.** This is a measurement difference, not a training difference.

Proof (no training involved): the server's own trained `ship_yolo26s_seed0/weights/best.pt`
was validated on the laptop under both versions, same data, same `classes=[0]`, same `imgsz`:

| ultralytics | mAP50-95 | mAP50 |
|---|---|---|
| 8.4.7  | 0.3201 | 0.6619 |
| 8.4.90 | **0.2864** | **0.6235** |
| what the server itself recorded | 0.28635 | 0.62349 |

8.4.90 reproduces the server's number exactly. Precision and recall moved by only ~0.01%
(0.8225→0.8226, 0.6021→0.6022) while mAP50 moved 6% and mAP50-95 11% — so detection, NMS
and IoU matching are unchanged and the difference is localized to the average-precision
integration. The exact commit was never identified; if needed for the writeup, diff
`ultralytics/utils/metrics.py` between tags `v8.4.7` and `v8.4.90` (likely `compute_ap`).

### What this invalidated

A laptop `yolo26m` seed 0 result of **0.3452** looked like a breakthrough (best server row
was `yolo12x` at 0.2965 seed-mean). Re-scored under 8.4.90 it is **0.3061** — a +0.002 tie
with `yolo12x` seed 0 (0.3041), not a +0.05 jump.

### Ruled out along the way (don't re-investigate these)

- **Data / split** — `split_fingerprint` `682dbe9f0f05` identical both sides, all six split
  lists byte-identical, label hash `287b11c50b5a` matched.
- **Batch size and weight decay** — server `yolo12x` seed1 (batch 8, eff 64, wd 0.000500)
  scored 0.2955 vs seed2 (batch 24, eff 72, wd 0.000563) 0.2972. That +0.0017 is smaller
  than the 0.0072 spread between the two duplicate runs of seed 0 at identical settings.
  Note 7 server rows (`26n`×3, `26s`×3, `12x` seed2) did use batch 24; it does not matter.
- **GPU / torch / CUDA / worker count** — all differ between machines, none responsible.
- **Training itself** — offset-correcting the laptop `yolo26s` control curve made both
  machines agree within ±0.003 by epoch 8. The machines train equivalently.

---

## 2. What is running right now

```
PID 23368 + 8212   watch_grid.cmd        health watchdog, since 10:47
PID 12412          run_tail_laptop.cmd   retry wrapper (20 attempts)
PID 33960          run_benchmark.py      the grid, since 15:31
                   + 24 dataloader worker processes
```

Position at handoff: `ship_yolo26m_seed1`, epoch 6/100, 5 epochs banked, ultralytics 8.4.90.

### Checking on it

```
type A:\Uncertain\runs\health.log                  # every 10-min sample
findstr /V " OK " A:\Uncertain\runs\health.log     # only problems; empty == all clear
```

### If it dies

The wrapper retries 20 times, and `grid.py` resumes an interrupted run from its own
`weights/last.pt` (epoch-level, so a crash costs minutes not a whole run). If both layers
are gone, relaunch **detached** — see trap 2 below:

```powershell
Invoke-CimMethod -ClassName Win32_Process -MethodName Create -Arguments @{
  CommandLine = 'cmd.exe /c A:\Uncertain\run_tail_laptop.cmd'
  CurrentDirectory = 'A:\Uncertain' }
```

Verify it actually started by finding a real iteration counter in `runs\grid_laptop.log`,
**not** by checking the process exists — see trap 5.

---

## 3. Results state

`runs\benchmark\benchmark_results_tail.csv` — 20 rows, all on the 8.4.90 metric scale.
Backup of the pre-correction file: `benchmark_results_tail.csv.bak_pre8490`.

| ultralytics_version | rows |
|---|---|
| `8.4.90` | 19 (server) |
| `8.4.7+val8.4.90` | 1 (`yolo26m` seed 0 — trained on 8.4.7, re-scored on 8.4.90) |

Provisional seed-means (mAP50-95):

```
yolo26m   n=1   0.3061     <- single seed only
yolo12x   n=4   0.2984     <- inflated by the duplicate row, see below
yolo12m   n=3   0.2906
yolo12l   n=3   0.2870
yolo26s   n=3   0.2813
yolo12s   n=3   0.2783
yolo26n   n=3   0.2540
```

### Remaining work: 8 runs, ~118 h (~5 days)

`yolo26m` seeds 1,2 (~11 h each) → `yolo26l` ×3 (~13 h) → `yolo26x` ×3 (~19 h).
Target is 27 unique rows; the CSV will read 28 because of the duplicate.

---

## 4. Open decisions

1. **Duplicate `yolo12x` seed 0 row.** Two concurrent server processes wrote 0.2969
   (47422 s) and 0.3041 (85412 s). Both are still in the CSV. `grid.py::_completed()`
   dedups by dict overwrite so it doesn't block resume, but it double-weights seed 0 in
   any seed-mean. **Must be resolved before Table 1** — pick a canonical run.
2. **opencv is double-installed.** `opencv-python 5.0.0.93` (pulled in by the 8.4.90
   upgrade) and `opencv-python-headless 4.10.0.84` both claim the `cv2` namespace;
   `cv2.__version__` currently resolves to **4.10.0**. Harmless so far — it meant the
   upgrade changed only ultralytics — but fragile. Resolve before the grid ends.
3. **Pin `ultralytics==8.4.90`** in requirements. This is the whole lesson of the session.
4. **Server `requirements.lock.txt` is unusable** — 84 of 87 lines are conda
   `@ file:///home/task_.../croot/...` build paths and torch/ultralytics are absent,
   because the capture ran under base `python` not the `saroha_work` env. Redo with the
   env activated: `python -m pip list --format=freeze`.

### After the grid finishes

- `scripts/measure_fps.py` (needs a GPU)
- `scripts/make_table1.py` (dedup `yolo12x` seed 0 first)
- Methods line for the writeup: a pinned detector version is load-bearing, and this run
  has a clean measurement of what it costs without one.

---

## 5. Traps learned this session — all of these cost real time

1. **Never put a `REM` line between `^`-continued lines in a `.cmd`.** cmd.exe continues
   onto the comment, ends the command there, and parses the next flag as its own command
   (`'--workers' is not recognized`). This silently crash-looped the grid for ~45 min.
   Comments go above the block. `run_tail_laptop.cmd` carries a warning to this effect.
2. **Launch long runs with `Invoke-CimMethod Win32_Process Create`, not `Start-Process`.**
   A `Start-Process`-launched run died from `forrtl: error (200): program aborting due to
   window-CLOSE event` when a console teardown propagated to its process group. Because
   `CTRL_CLOSE_EVENT` hits the whole group, it killed the retry wrapper too — so the
   20-attempt crash-safety never fired. Both safety layers lived in the console that got
   disturbed.
3. **Never run a GPU job in the foreground alongside training.** Same incident as above.
   If a diagnostic is needed, detach it the same way and keep it short.
4. **Windows multiprocessing spawn needs `if __name__ == "__main__":`.** A validation
   script without it deadlocked instantly — `model.val()` spawns workers, each child
   re-imports the module and re-enters `val()`. Symptom: process alive, ~10 s of CPU over
   25 min, no output. Passing `workers=0` to `val()` also avoids it.
5. **Verify training started from an iteration counter, not process existence.** Killed
   runs leave progress frames in the log tail that look live. Anchor parsing on the last
   `[wrapper] attempt` / `[grid] ===` banner and read only lines after it. Ultralytics also
   prefixes frames with ANSI `\x1b[K`, which defeats `^`-anchored regexes — strip ANSI first.
6. **Scheduled wakeups and cron jobs do not fire while the user is away.** Both were tried
   this session and produced zero reports. `scripts/watch_grid.py` writing to
   `runs/health.log` every 10 min is the only reliable unattended monitor.
7. **Ultralytics `check_resume` reloads args from the checkpoint** — only `imgsz`, `batch`,
   `device`, `close_mosaic` pass through from a new invocation. `workers` cannot be changed
   on resume.
8. **`--data`, `--classes` and `--out-csv` must stay identical across grid invocations**,
   or `grid.py`'s split-fingerprint / classes mix-refusal aborts the run. A control
   experiment on a variant already in the CSV needs its own `--out-csv` *and* `--run-prefix`,
   otherwise the grid skips it as already done.

---

## 6. Environment

```
python 3.13.0
torch 2.7.1+cu118      (unchanged by the upgrade — ultralytics only needs >=1.8.0)
ultralytics 8.4.90     (upgraded from 8.4.7 on 2026-08-11)
cv2 -> 4.10.0          (see open decision 2)
GPU: RTX 4080 Laptop, 12282 MiB
grid: batch 8, workers 8, imgsz 640, patience 20, deterministic
```

Pre-upgrade snapshot of all 227 packages is in the session scratchpad as
`pipfreeze_before_upgrade.txt` if a rollback is ever needed.

**VRAM ceilings measured on this card** (batch 8 peaks): `26m` 4.3 GB, `26l` 5.3 GB,
`26x` 7.9 GB. `26x` at batch 16 peaks 15.31 GB, which Windows WDDM pages into host RAM
instead of raising OOM — it "runs" at 2.4 img/s, ~17x slower. **Do not raise batch for
`26x`.** Also: `workers=16` is worse than 8 (2.80 vs 4.00 it/s median) — CPU sits at ~14%
either way, so the loader was never the bottleneck.

---

## 7. Key paths

```
runs\benchmark\benchmark_results_tail.csv        results (20 rows)
runs\benchmark\benchmark_results_tail.csv.bak_pre8490
runs\benchmark\runs\<name>\results.csv           per-epoch curves
runs\grid_laptop.log                             grid stdout (~40 MB, progress bars)
runs\health.log                                  watchdog, one line / 10 min
runs\derived\data_vis_stride2.yaml               the split (fingerprint 682dbe9f0f05)
run_tail_laptop.cmd                              grid launcher + retry wrapper
watch_grid.cmd / scripts\watch_grid.py           watchdog
src\uqfusion\bench\grid.py                       grid driver (epoch-level resume)
archive\phase1\main_2026-08-10\                  server run dirs + weights (handback)
```

The archived server weights are what made the version diagnosis possible — keep them.
Note `archive\phase1\main_2026-08-10\...\ship_yolo26m_seed0\` is a 1-epoch orphan the
server never finished; it is not in the CSV and should stay out of it.
