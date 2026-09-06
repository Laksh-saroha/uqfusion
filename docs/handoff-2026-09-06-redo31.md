# Handoff — 2026-09-06, server GPU-contention incident + 31-run redo pass

Written to be picked up cold. Covers: the laptop VIS UQ retrain (routine, still
running), and a longer story on dgxanode01 — a zombie-process GPU leak that caused
13 back-to-back OOM failures, its fix, a race condition introduced while requeuing
the dead runs, and a completely separate stale queue discovered and killed along the
way. All server-side actions were driven via Claude-in-Chrome MCP tools against the
JupyterLab origin — see `project-server-dgxanode01` memory for the driving method
(no SSH on this box).

---

## 1. What's running right now

| | status |
|---|---|
| **Laptop GPU (RTX 4080)** | `ens_vis_nightfull_seed0` training, epoch ~40/100, best_fitness 0.278, patience_gap 0 (still improving). 4 more queued after it: `ens_vis_nightfull_seed{1,2,3,4}`. |
| **dgxanode01** | Two queues live: `queue_vis_benchmark_ep25`'s original 93-run grid (**paused**, mid-`vis_bench_yolov10b_seed2`, epoch 7) + a new 31-run redo pass (**running**, pid `50019`, on `vis_bench_yolo12x_seed2`, epoch 2/25). |
| **Repo** | `fusion-uq-phase3`, no new commits this session yet — this handoff is the first. |

## 2. Laptop — VIS UQ retrain, on track

Queue: `runs/queue_vis_uq_retrain/queue.json`, 7 runs total (see
`docs/handoff-2026-09-04-uq-retrain.md` for why it exists and the cold-start /
no-ft-stage reasoning — still valid, nothing changed here).

Progress: `mc_vis_nightfull` done (0.2586), `gauss_vis_seed0_nightfull` done (0.2698,
after the AMENDED→REVERSED ft-stage back-and-forth recorded in `queue.json`'s own
note — settled on cold-start-only, matching the other two arms), `ens_vis_nightfull_seed0`
running (0.278 and rising). 4 ensemble seeds left after this one. Nothing needs
attention here; check with:
```bash
python scripts/run_queue.py --queue-dir runs/queue_vis_uq_retrain status
```

## 3. Server — the OOM crisis and its real cause

**Symptom.** By the time this session started, `queue_vis_benchmark_ep25` (93-run VIS
benchmark grid) had accumulated **18 dead runs** from an NVML/CUDACachingAllocator
assert (`yolo12x`, `26x`, `9e`, `11x`, `10x`, `12l`, all seeds — flagged in the prior
session's scheduled task as "transient GPU hiccup"), and while investigating those,
**13 more runs failed within 1-3 minutes each** with plain `CUDA error: out of memory`
— including a run (`vis_bench_yolov8m_seed2`) that had *already* successfully resumed
once earlier. That second point disproved "transient hiccup": something was
persistently eating GPU memory.

**Root cause, found by direct evidence, not inference.** `ps aux` on the server showed
zombie (`Z`-state) processes with **anomalous nonzero, growing CPU time** — a true
zombie should show 0:00 forever. One (pid 45808) had accumulated 738:04 CPU minutes
and was still climbing. Zombies can hold an orphaned CUDA/NVML context that never
appears in `nvidia-smi`'s live process table, so the MIG slice looked near-full
(40173/40192 MiB) with **zero** live processes shown. `kill -9` can't remove a zombie
from `ps` (its parent has to `wait()` on it) but *can* release the GPU-side resources
it was still holding.

**Fix, verified.** The classifier hard-blocks `kill` sent through the browser-driven
remote terminal — every attempt was refused, even after explicit user confirmation.
The user ran it directly:
```bash
kill -9 45808 6045 6046 7913 12891
```
`nvidia-smi` immediately dropped to 36195/40192 MiB with an empty process table.
The queue then ran **two full 26-epoch runs back-to-back with zero failures**
(`vis_bench_yolov10b_seed0` → 0.24265, `vis_bench_yolov10b_seed1` → 0.2472) before
this session moved on to requeuing the dead runs. Also cleaned up: 22 stale Jupyter
terminal sessions (zero effect on the training queue — it's `nohup`'d and `disown`'d,
terminal-independent).

**Important caveat for next time.** `ps aux` run *inside a Jupyter terminal* only
shows processes in that terminal's own PID namespace — it did NOT show the actual
live training process (`/opt/venv_match/bin/python ... run`) even while it was
running. `cat /proc/<pid>/cmdline` works regardless of namespace (shared host
`/proc` mount) and is the reliable way to check whether a specific pid is alive.
Don't trust a Jupyter terminal's `ps aux` to prove a process is *absent* — only to
prove one *is* present.

## 4. The redo pass — race condition on launch, now resolved

**Scope decision.** 31 failed runs existed by the time of the redo, not 18: the
original 18 NVML-assert failures plus 13 more zombie-caused OOM failures (including
`vis_bench_yolov8m_seed2`, `yolov10l×3`, `yolov9c×3`, `yolo11l×3`, `yolo26m×3`). User
chose to redo all 31 in one pass rather than two separate cycles. Full list is in
§5 below.

**What went wrong.** Sequence was: pause the original queue (pid 48457) → launch a
new `nohup` process (pid 50019) with `--only <31 runs> --redo` → the new process saw
`queue_status: paused` in the shared `state.json` and itself printed "paused —
waiting for resume" → sent a `resume` command → **both** processes share the same
`state.json`-based pause/resume flag, so pid 48457 (the *original*, full-93-run
process) also woke up and resumed on its own remaining queue, concurrently with the
new redo process. Confirmed via `ps aux` (in the *user's own* terminal, not mine —
see the namespace caveat above) showing both alive simultaneously, and via
`/proc/<pid>/cmdline`: 48457 had `... run` (no `--only`), 50019 had `... run --only
vis_bench_yolo12x_seed0 ...`.

**Fix.** User ran `kill -15 48457`. Confirmed dead via `/proc/48457/cmdline` (empty).

**Second surprise, same `ps aux` output.** A completely unrelated queue,
`queue_vis_benchmark_stride4` (12 runs, an older/superseded predecessor to the
`_ep25` grid — 100-epoch budget vs `_ep25`'s 25), had been sitting **paused since
2026-09-03** with its main process (pid 38065, ~166 CPU-hours) and 17 leftover
worker processes still alive on the same MIG slice, doing nothing but holding
resources for 3 days. Nobody in this session had prior knowledge of it. User killed
it:
```bash
kill -15 38065 40504 40505 40506 40507 40508 40509 40540 40541 40542 40543 40544 \
  40545 40546 40547 40548 40549 40550 40551
```
Also found and killed a third straggler that appeared mid-cleanup — pid 50404, a
second plain `... ep25 run` (no `--only`) process, likely spawned by the same
resume-flag race as 48457:
```bash
kill -15 50404
```
Confirmed dead via `/proc/50404/cmdline` (empty). After all three kills, `ps aux`
showed exactly one queue process (`50019`) plus its own dataloader workers — clean.

**Lesson for the runner script.** `run_queue.py`'s pause/resume is a single shared
flag in `state.json`, not scoped per-process. Launching a second `--only` process
against a queue-dir whose *original* process might still be alive (even paused) is
unsafe — a `resume` call wakes both. If this needs doing again, either confirm the
original process is fully dead (not just paused) before launching a second one, or
add a lockfile/pid-check to `run_queue.py` itself.

## 5. Redo pass — current status and the 31-run list

Live as of this handoff: `vis_bench_yolo12x_seed2`, epoch 2/25, best_fitness 0.15595,
~20.5 min/epoch (XL model, slow). This is the **first** of the 31 to survive past
the point where the two before it (`seed0`, `seed1`) died — both re-failed with the
identical `NVML_SUCCESS` assert on retry, suggesting those two specific failures may
be a genuine VRAM-fit problem for XL variants at batch 16 on a 40GB slice, not
(only) the zombie leak. Worth watching whether the pattern holds across the other
XL-class variants (`26x`, `9e`, `10x`) once the queue reaches them.

Full `--only` list (31 runs, in queue order):
```
vis_bench_yolo12x_seed0 vis_bench_yolo12x_seed1 vis_bench_yolo12x_seed2
vis_bench_yolo26x_seed0 vis_bench_yolo26x_seed1 vis_bench_yolo26x_seed2
vis_bench_yolov9e_seed0 vis_bench_yolov9e_seed1 vis_bench_yolov9e_seed2
vis_bench_yolo11x_seed0 vis_bench_yolo11x_seed1 vis_bench_yolo11x_seed2
vis_bench_yolov10x_seed0 vis_bench_yolov10x_seed1 vis_bench_yolov10x_seed2
vis_bench_yolo12l_seed0 vis_bench_yolo12l_seed1 vis_bench_yolo12l_seed2
vis_bench_yolov8m_seed2
vis_bench_yolov10l_seed0 vis_bench_yolov10l_seed1 vis_bench_yolov10l_seed2
vis_bench_yolov9c_seed0 vis_bench_yolov9c_seed1 vis_bench_yolov9c_seed2
vis_bench_yolo11l_seed0 vis_bench_yolo11l_seed1 vis_bench_yolo11l_seed2
vis_bench_yolo26m_seed0 vis_bench_yolo26m_seed1 vis_bench_yolo26m_seed2
```
Log: `redo31.log` in the repo root on the server (`nohup ... > redo31.log 2>&1 &
disown`, launched from Jupyter terminal "3").

**To check status** (browser session cookie required):
```
GET http://172.16.224.131:1002/api/contents/uqfusion/runs/queue_vis_benchmark_ep25/state.json?content=1&type=file&format=text
```
`pid` should read `50019` (or whatever it becomes if it's restarted) and
`queue_status: running`. If a run shows `failed` with a *today's-date* `finished`
timestamp, that's a genuine retry failure, not stale data from before the redo —
watch for that distinction, this session got confused by it once (turned out to be
real, not stale).

## 6. When the 31-run redo pass finishes

The **original** 93-run queue is still paused mid-`vis_bench_yolov10b_seed2` (epoch
7) and does **not** resume itself — `run_queue.py run --only ... --redo` is a
separate invocation from the plain `run` that drives the full grid. Once `redo31.log`
shows the process has exited (all 31 attempted), resume the original:
```bash
cd /workspace/uqfusion && /opt/venv_match/bin/python scripts/run_queue.py \
  --queue-dir runs/queue_vis_benchmark_ep25 run
```
**Do not launch this while the redo pass is still alive** — that's exactly the race
condition §4 describes. Check `ps aux` / `/proc/<pid>/cmdline` for `run_queue.py
... ep25` processes first; there should be none before starting this.
