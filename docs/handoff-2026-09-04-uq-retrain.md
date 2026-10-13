# Handoff — 2026-09-04, VIS UQ retrain launched

Read `docs/handoff-2026-09-04.md` first for the day's earlier context (server benchmark, the U1
amendment); this covers only what changed after: the VIS MC-Dropout + ensemble retrain that got
queued, and the server benchmark's failure gap.

---

## 1. What's running

| | status |
|---|---|
| **Laptop GPU (RTX 4080)** | **busy.** `mc_vis_nightfull` training (first of 6 queued runs). |
| **dgxanode01** | `vis_benchmark_stride4_ep25`, still alive, ~10 days out. **12 of 93 runs are dead** — see §3. |
| **Repo** | `fusion-uq-phase3`, pushed to `origin`, PR open against `main` (see the PR for the full diff — this branch had never been PR'd before today). |

## 2. The VIS retrain queue

**Why it exists.** Two VIS UQ arms are still trained on the pre-restore, night-emptied labels
and are hollow at night the same way the original `gauss_vis_seed0_ft` was — confirmed by direct
count, not inference:

| checkpoint | trained | night detections / 1,032 frames |
|---|---|---:|
| `mc_vis_seed0_ft_refit` (MC-Dropout) | 2026-09-01 | **28** |
| `ens_vis_seed0_ft_control` (local) + `ens_vis_seed{1..4}_ft` (dgxanode01) | 2026-08-25..28 | not measured, same family |
| `gauss_vis_nightfull` (already retrained, for contrast) | 2026-09-03, post-restore | 8,391 |

Full derivation: `runs/eval/uq_day_night_slice_nightfull.md` and
`docs/prereg-uq-day-night-slice-amendment-nightfull.md`.

**What's queued** — `runs/queue_vis_uq_retrain/queue.json`, 6 runs, sequential:

1. `mc_vis_nightfull` — MC-Dropout, cold start
2. `ens_vis_nightfull_seed0` .. `ens_vis_nightfull_seed4` — 5 ensemble members, cold start each

All six: `yolo26m`, `runs/derived/data_vis_stride2.yaml` (restored labels, hash `b92739202127`),
`imgsz 640`, `epochs 100`, `patience 20`, `batch 16`, `workers 8` — identical sizing to
`gauss_vis_nightfull`'s own `args.yaml`.

**Two decisions baked into the queue, not defaults — both reversible if you disagree:**

* **Cold start, not a continuation.** Same reasoning as `gauss_vis_nightfull`: continuing from
  `mc_vis_seed0_ft_refit` or `ens_vis_seed{0..4}_ft` would carry over weights shaped by
  blind-night training. New run ids (`mc_vis_nightfull`, `ens_vis_nightfull_seed*`) so the
  resume-safe runner cannot silently resume an old checkpoint under a shared name — see
  `full_scale_queue()`'s comment in `scripts/run_queue.py` for the 2026-08 incident it guards.
* **No mosaic-off `_ft` continuation stage.** `gauss_vis_nightfull` — already scored in today's
  U1 amendment — has none either (it early-stopped at epoch 31, well short of Ultralytics'
  fixed `close_mosaic` point at epoch 90, so it trained on mosaicked frames and was evaluated
  clean; that mismatch is baked into today's numbers whether or not the new runs get an `_ft`
  stage). Adding `_ft` only to the new runs would make the arm comparison *more* asymmetric.
  If you want honest per-arm numbers rather than matched-defect numbers, the fix is to retrofit
  `gauss_vis_nightfull` with its own `_ft` stage too (~2.8 h) — not to add `_ft` here alone.

**Cost, measured/derived** (full pricing in the 2026-09-04 chat log):

| run | basis | estimate |
|---|---|---:|
| `mc_vis_nightfull` | `gauss_vis_nightfull`'s measured cold start (31 epochs, patience 20) | ~9.8 h |
| each `ens_vis_nightfull_seed*` | same basis, per member | ~9.8 h |
| **total, 6 runs sequential** | | **~59 h** |

A point estimate from one measured run, not an average — early-stop epoch can differ per run.

### How to check on it

```bash
python scripts/run_queue.py --queue-dir runs/queue_vis_uq_retrain status
```

Dashboard (already running; closing the browser tab does not affect training):

```
http://127.0.0.1:8771
```

If the dashboard process died, restart it (safe anytime, read-only):

```bash
python scripts/dashboard.py --port 8771 --queue-dir runs/queue_vis_uq_retrain
```

### Pause / resume / restart

**Pause** (stops cleanly at the next epoch checkpoint, `last.pt` complete):
```bash
python scripts/run_queue.py --queue-dir runs/queue_vis_uq_retrain pause
```
or the Pause button in the dashboard.

**Resume** after a pause, or after the runner died/was killed/the machine slept — same command
either way, picking up from `last.pt` via Ultralytics' own resume path (optimizer/EMA/scaler/
epoch state intact):
```bash
python scripts/run_queue.py --queue-dir runs/queue_vis_uq_retrain run
```

**If a run failed** (`state.json` shows `"status": "failed"`), it is NOT retried automatically —
`failed` is terminal. Rerun by id with `--redo`:
```bash
python scripts/run_queue.py --queue-dir runs/queue_vis_uq_retrain run --only <run_id> --redo
```

**GPU interpreter, always** — the system Python, not `.venv` (CPU torch, ~20× slower);
`PYTHONPATH=src` required (`uqfusion` is not installed into it):
```
C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe
```
Full invocation used to launch this queue:
```bash
PYTHONPATH=src "C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe" -u scripts/run_queue.py --queue-dir runs/queue_vis_uq_retrain run
```

### What NOT to do

* Don't point `run_queue.py` at `runs/queue_vis_uq_retrain` with the plain `init` command — it
  would overwrite `queue.json` (`state.json` is spared by `init`'s own guard, the run list is
  not). This `queue.json` was hand-written, not generated by `cmd_init`, because no existing
  `--matrix` choice skips the `_ft` stage.
* Don't let this collide with `runs/queue/` (the old Phase 2 queue, still holding its own
  `queue.json`/`state.json`) — always pass `--queue-dir runs/queue_vis_uq_retrain` explicitly.
* Don't reuse run ids `mc_vis_seed0*` or `ens_vis_seed{0..4}*` while this runs — those are the
  OLD, night-blind checkpoints, kept on disk precisely to diff against.

---

## 3. Server (dgxanode01) — 12 dead runs in the benchmark grid

`runs/queue_vis_benchmark_ep25` (93-run VIS benchmark, stride 4 × 25 epochs). As of 2026-09-04
~10:57 UTC it is alive (`vis_bench_yolov8l_seed0`, epoch 11+, heartbeat every ~1 s) — **not**
stalled. But 12 runs failed back to back, 08:34–09:24 UTC, all with the identical crash:

```
RuntimeError: NVML_SUCCESS == r INTERNAL ASSERT FAILED at
"/pytorch/c10/cuda/CUDACachingAllocator.cpp":1016
```

Dead runs: `vis_bench_{yolo12x,yolo26x,yolov9e,yolo11x}_seed{0,1,2}` — 4 variants × 3 seeds. The
next run (`yolov8l_seed0`) trained cleanly, so this reads as a transient GPU/driver hiccup, not
a per-variant fault — but the queue has **no retry logic** (`failed` is terminal in
`run_queue.py`'s state machine), so these 12 grid cells are permanently empty in the final table
unless requeued by hand.

**To check current status** (browser session cookie required — see `project-server-dgxanode01`
memory for the driving-without-SSH method):
```
GET http://172.16.224.131:1002/api/contents/uqfusion/runs/queue_vis_benchmark_ep25/state.json?content=1&type=file&format=text
GET http://172.16.224.131:1002/api/contents/uqfusion/runs/queue_vis_benchmark_ep25/live.json?content=1&type=file&format=text
```
`live.json`'s `updated` timestamp moving forward on repeat reads = alive, not stalled.

**To requeue the 12 dead runs**, once the current 93-run pass finishes (or now, if you want to
fix the gap without waiting — the queue is single-process and sequential, so requeuing mid-run
needs a second process, which the "two concurrent trainings are 8% slower" trap in
`project-server-dgxanode01` advises against):
```bash
python scripts/run_queue.py --queue-dir runs/queue_vis_benchmark_ep25 run \
  --only vis_bench_yolo12x_seed0 vis_bench_yolo12x_seed1 vis_bench_yolo12x_seed2 \
         vis_bench_yolo26x_seed0 vis_bench_yolo26x_seed1 vis_bench_yolo26x_seed2 \
         vis_bench_yolov9e_seed0 vis_bench_yolov9e_seed1 vis_bench_yolov9e_seed2 \
         vis_bench_yolo11x_seed0 vis_bench_yolo11x_seed1 vis_bench_yolo11x_seed2 \
  --redo
```
(run on the server, matched-torch venv `/opt/venv_match/bin/python`, not this laptop.)

---

## 4. Today's other outcome — the U1 amendment (context, not new work)

Already committed (`1080f65`, `6d200cd`) before this retrain was queued:

* `docs/prereg-uq-day-night-slice-amendment-nightfull.md` — declared the detector swap
  (`gauss_vis_nightfull` for the VIS sigma-head arm) as its own question, not a silent re-run.
* `scripts/slice_uq_day_night.py` — fixed a real crash (`KeyError: 'NO-SIGNAL'` when a metric's
  day-subset separation is exactly 0) and added `--vis-sigma-cache` so the amendment could be
  scored without touching the registered defaults.
* Result: **VIS CONTAMINATED, IR SUSPECT** — but the VIS verdict is confounded by comparing a
  night-capable checkpoint against a still-blind one (§2's detection table). That confound is
  exactly what this retrain resolves once it finishes.

---

## 5. When the retrain finishes

Re-run the amendment scoring with the new checkpoints instead of the old blind ones — build
fresh caches (`sigma_vis_nightfull.pkl` already exists; add `mc_vis_nightfull.pkl` and the
5-member ensemble cache), then either extend `--vis-sigma-cache`-style overrides to the
MC-Dropout arm or write a second, similarly-declared amendment. **Not pre-registered yet — do
that before running it**, per the standing rule (`docs/handoff-2026-09-04.md` §3: no silent
re-runs).
