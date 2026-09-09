# VIS 2-class benchmark @ stride 4 — 93 runs on dgxanode01, queued behind the ensemble

**Set up 2026-08-26 ~18:50 UTC / 2026-08-27 00:20 IST.** Laksh is away four days, so everything
here runs unattended and is legible on return from one file: `runs/status_dgxanode01.xlsx` on the
server.

This is the "second table" of
[`TODO-2026-08-26-phase1-classset.md`](TODO-2026-08-26-phase1-classset.md) §4.2 — widened from
that memo's 9 `main` variants to the whole 31-variant ladder, and moved from stride 2 to stride 4,
on Laksh's instruction.

---

## 1. What is queued

`runs/queue_vis_benchmark_stride4/queue.json` → outputs in `runs/vis_benchmark_stride4/`. Built by
`python scripts/run_queue.py --queue-dir runs/queue_vis_benchmark_stride4 init --matrix vis_benchmark`
(`vis_benchmark_queue()` in `scripts/run_queue.py`).

| | |
|---|---|
| runs | **93** = 31 variants × seeds {0,1,2} |
| data | `/workspace/derived/data_vis_stride4.yaml` — 24,070 of 96,275 train frames |
| classes | **both** — 0 ship, 1 buoy |
| start | cold from COCO weights (`pretrained: true`), **not** a warm start |
| schedule | imgsz 640, 100 epochs, patience 20, single stage (no mosaic-off `_ft`) |
| sizing | batch 16, workers 6 |
| order | parameter count **descending**, seeds inner: `yolov8x → yolo12x → yolo26x → yolov9e → yolo11x → … → yolo26n → yolov9t` |
| kind | `gaussian` with `sigma: false` — the stock `DetectionTrainer`, i.e. a plain detector, reusing the queue's pause/resume/heartbeat for free |

**Why largest first:** 93 runs is roughly two weeks of GPU time. The ordering decides what exists
if the sweep is cut short, and keeping each variant's three seeds adjacent means a variant that
finishes has error bars rather than a lone point.

### Two things that must be stated wherever a number from this reaches a table

1. **Class set.** Phase 1 passed `--classes 0` through `bench/grid.py`; all 93 rows of
   `phase1_benchmark/results.csv` are ship-only. This queue runs through `train_gaussian`, which
   passes **no** `classes` filter, so it trains and scores on both classes. Macro-averaged mAP
   from the two is not comparable — compare ship AP to ship AP, and report per-class AP.
2. **Train stride.** Phase 1 trained on stride 2 (48,136 frames); this is stride 4 (24,070).
   Verified: the stride-4 list is a strict **subset** of the stride-2 list, so the split
   fingerprint is unchanged and no val frame leaks in.

Striding is per-run over unique frame ordinals so stereo pairs stay together
(`uqfusion/data/subset.py`) — which is why the stride-4 list cannot be faked by taking every
second line of the stride-2 list.

### Sizing rationale

`batch 16` is Phase 1's own largest-common-fit on a 40 GB MIG slice — its record has `yolo12x`
completing at 16 and failing at 24/32 — and it is what all 14 `queue_vis_server` arms use.
`workers 6` is the `/dev/shm` ceiling measured on this box: 8 dies with "insufficient shared
memory" partway into epoch 1, and 2/4/6 measure 1.8/1.9/1.9 it/s, so 6 costs nothing. `yolo12x` is
the one variant near the memory edge; if it OOMs, the runner marks it `failed` and walks on, and
it can be re-run at batch 8 with that disclosed.

---

## 2. The chain — nothing starts until the ensemble is done

Three processes, each waiting on the one before it. Two trainings on one MIG slice measure ~8%
**slower** than running them sequentially, so this waits rather than overlaps.

```
26701  run_queue.py runs/queue_vis_server        <- the live ensemble trainer
  └─ 29989  supervise_queue.sh runs/queue_vis_server   --wait-pid 26701 --max-restarts 5
       └─ 30020  supervise_queue.sh runs/queue_vis_benchmark_stride4 --wait-pid 29989
29956  queue_xlsx_report.py (both queues) --interval 300
```

- **29989** exists so a crash of 26701 does not silently end the ensemble queue. When 26701 exits
  it checks `queue_vis_server`: nothing left → it exits immediately; runs left → it restarts the
  runner, which resumes from `last.pt`.
- **30020** waits on 29989 rather than on the ensemble queue's completion — deliberate: waiting on
  "queue complete" would deadlock the GPU for four days if the ensemble hit something
  unrecoverable. Waiting on the supervisor's pid means the benchmark starts either way.
- Both supervisors restart a dead runner with exponential backoff (60 s, doubling to a 30 min cap)
  whenever the runner exits in under 5 minutes, so a crash-looping run cannot spin the GPU.

The ensemble run was **not** paused to set this up — every step was a file write, a download, or a
process start. `ens_vis_seed3` kept its epoch.

---

## 3. The tracker — `runs/status_dgxanode01.xlsx`

`scripts/queue_xlsx_report.py`, refreshed every 5 minutes, covering **both** queues.
`scripts/xlsxlite.py` writes the workbook with the standard library alone — no `openpyxl`, because
a monitoring tool whose job is to keep working should not have a pip install on this box's flaky
egress in its critical path.

| sheet | what it holds |
|---|---|
| **Summary** | per queue: a **HEALTH** line, progress, GPU-hours spent, mean hours/run, a fitted ETA and finish date, the supervisor's heartbeat, the live epoch/batch/img-s/GPU-GB |
| **Runs** | all 93 in queue order: status, epochs, best epoch, best mAP50-95, and the mAP50/precision/recall **of that same epoch**, elapsed hours, h/epoch, errors |
| **Variants** | seeds folded per variant: seed0/1/2, mean, sd, spread — the draft benchmark table |
| **Epochs** | every epoch of every run, straight from Ultralytics' `results.csv` |
| **Log** | tail of each queue log (read by seeking; `queue_stdout.log` hits 76 MB/day and is never read whole) |

`runs/status_dgxanode01.csv` mirrors the Runs sheet in case the workbook is unreadable.

**Read HEALTH first.** It distinguishes the states that matter while nobody is watching:

- `WAITING (by design)` — a supervisor is holding the GPU for the queue ahead.
- `OK — <run> running, state.json N min old`.
- `STALLED?` — marked running but `state.json` has not moved for longer than three epochs' worth
  of time. The one that means "look at it".
- `IDLE` — unfinished runs, nothing running, no fresh supervisor heartbeat.
- `GAVE UP` — a supervisor hit `--max-restarts`.

The ETA is fitted on this queue's own finished runs as `h/epoch ≈ a + b × GFLOPs`, with expected
epochs from the mean of what has already early-stopped, so it is meaningless until the first few
runs land and sharpens after that.

---

## 4. Expected cost

Phase 1 spent **647 GPU-h** on the same 93 rows at stride 2. Halving the frames per epoch puts
this at roughly **320 GPU-h ≈ 13–14 days**, starting after the ensemble queue drains
(`ens_vis_seed3` was at epoch 10/100 when this was set up; 3 runs plus 2 fine-tunes remain, ~1–2
days). The x/l end of the ladder lands in the first week.

All 31 COCO checkpoints (1,395.7 MB) were pre-fetched before anything started —
`ir_bench_yolov8x_seed0` was already lost once to a TLS failure mid-download, and four unattended
days is 31 more chances at that.

---

## 4b. The laptop, same night

The laptop's IR queue (`runs/queue_ir_benchmark_stride4`, trimmed by Laksh from 93 runs to 30 —
yolov8 + yolo26 only) had **no supervisor at all**, and its runner died once during this session.
Since the same four-day absence covers it, it now has the same two protections:

- `scripts/supervise_queue_win.cmd runs\queue_ir_benchmark_stride4` — the Windows twin of the
  shell supervisor. Git Bash's `kill -0` cannot see native Windows pids, so `--wait-pid` is
  unusable there; instead `scripts/queue_runner_alive.py` decides whether a runner is already
  working the queue (recorded pid alive, or `live.json` moved in the last 10 min) and the
  supervisor starts one only when the answer is no. Verified against the live run: it did **not**
  start a second runner.
- the same tracker, writing `runs/status_laptop.xlsx` every 5 minutes.

To stop either: close the hidden `uqfusion-supervisor` console
(`taskkill /F /FI "WINDOWTITLE eq uqfusion-supervisor*"`). Pause the queue first if the intent is
to stop a run in progress — the supervisor does not kill, but it will restart a runner that exits.

---

## 5. On return — what to check, in order

1. Open `runs/status_dgxanode01.xlsx`, Summary sheet, read both HEALTH lines.
2. If HEALTH is `OK` or `WAITING`, nothing needs doing.
3. If `STALLED?` — check `runs/queue_vis_benchmark_stride4/supervisor.log` and
   `runner_stdout.log`. The supervisor restarts a *dead* runner; it cannot detect a *hung* one.
4. If `GAVE UP` or `IDLE` — the supervisor is gone. Restart everything with the one command in §6.
5. Look at the Variants sheet before anything else in the numbers: if seed spread swamps the
   model-size effect the way it did on the IR sweep (trimmed from 93 runs to 30 on 2026-08-26 for
   exactly that reason), the same decision is due here, and the largest-first ordering means the
   evidence is already in.

## 6. Recovery

Everything dies if the **container** restarts (`/opt/venv_match` lives on the ephemeral overlay).
After a container restart, rebuild the venv, then:

```bash
cd /workspace/uqfusion && bash scripts/bootstrap_vis_benchmark.sh
```

Idempotent and the documented recovery path: it re-discovers the ensemble runner and its
interpreter, rebuilds the stride-4 yaml only if missing, rewrites the queue's data paths to this
machine's absolute yaml, re-fetches only absent checkpoints, and refuses to start a second copy of
a supervisor or tracker already running.

To stop the benchmark without touching the ensemble:

```bash
python scripts/run_queue.py --queue-dir runs/queue_vis_benchmark_stride4 pause
pkill -f 'supervise_queue.sh --queue-dir runs/queue_vis_benchmark_stride4'
```

Pause is graceful — it stops at the next epoch checkpoint, and resume re-enters Ultralytics' own
resume path with optimizer, EMA and epoch counter intact. Kill the supervisor **before** clearing
the pause, or it will start the runner again.

---

## 7. Provenance notes

- The server's `scripts/run_queue.py` is an older revision than the laptop's (its `TERMINAL` set
  has no `"diverged"`, and it has no `vis_benchmark` matrix). **Deliberately not overwritten** —
  the live ensemble queue runs against it. `queue.json` is data, and its `cmd_run` already honours
  both `out_subdir` and `sigma`, which is everything this queue needs. The queue-building code
  lives in the laptop repo for reproducibility.
- The queue's `data` field was rewritten from the repo-relative `runs/derived/data_vis_stride4.yaml`
  to the absolute `/workspace/derived/data_vis_stride4.yaml`, matching what every
  `queue_vis_server` run uses. Derived lists on this box live at `/workspace/derived`, not inside
  the repo, and there is no `Pohang_dataset/data_vis.yaml` for `--data vis` to resolve — only the
  split lists under `/workspace/pohang/visible`. `/workspace/derived/data_vis.yaml` was synthesised
  from those three lists to feed the stride builder.
