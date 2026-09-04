"""Parallel, resumable queue for the architecture ideas of 2026-09-01.

`run_analysis_queue.py` is sequential because its jobs were all CPU analyses of
similar cost. These are not: the CPU probes are independent and can saturate the
cores, while the cache builds contend for one 12 GB GPU and must not overlap.

So there are two lanes:

    cpu   a pool of `--workers` processes; jobs are independent by construction
          (each reads caches read-only and writes one new markdown file)
    gpu   strictly serialised, one at a time, whatever the pool size

Dependencies are declared per job and are checked against the state file, so a
probe that needs a cache waits for the build that produces it and is skipped --
not failed -- if that build did not succeed.

Ordering within a lane is by value-if-interrupted: the screens that can KILL an
expensive idea run before the sweeps that would price it.

    python scripts/run_ideas_queue.py preflight    # every job, tiny args, ~minutes
    python scripts/run_ideas_queue.py run          # the real thing
    python scripts/run_ideas_queue.py run --only i1_rerank i2_sigma
    python scripts/run_ideas_queue.py status
    python scripts/run_ideas_queue.py reset --job i1_rerank

Held jobs (`hold=True`) are printed and skipped unless named in `--only`: they
cost GPU-hours or write large trees, and the screens in front of them exist to
say whether they are worth starting.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
QDIR = ROOT / "runs" / "queue_ideas"
STATE = QDIR / "state.json"
LOGDIR = QDIR / "logs"
TERMINAL = ("done", "failed", "skipped", "timeout")
_LOCK = threading.Lock()


class Job:
    def __init__(self, jid, script, args, pre_args, lane="cpu", needs=(), hold=False,
                 note="", produces=(), timeout=10800, pre_timeout=900):
        self.id, self.script, self.args, self.pre_args = jid, script, args, pre_args
        self.lane, self.needs, self.hold, self.note = lane, tuple(needs), hold, note
        self.produces = tuple(produces)
        #: Wall-clock ceiling per job. Generous, but finite: an unbounded job is
        #: indistinguishable from a crashed one and blocks everything behind it.
        self.timeout, self.pre_timeout = timeout, pre_timeout


# Ordering = value if interrupted. Screens that can close an idea come first.
JOBS = [
    # ---- lane cpu: the screens ------------------------------------------
    Job("i1_oracle", "scripts/probe_oracle_headroom.py",
        ["--cache-dir", "runs/cache_m", "--out", "runs/eval/oracle_headroom.md",
         "--json-out", "runs/eval/oracle_headroom.json"],
        ["--cache-dir", "runs/cache_m", "--limit", "60",
         "--out", "runs/queue_ideas/preflight/oracle_headroom.md"],
        note="I1/I6 -- the ceiling every other job is measured against"),
    Job("i2_sigma", "scripts/probe_sigma_residual.py",
        ["--cache-dir", "runs/cache_m", "--out", "runs/eval/sigma_residual.md"],
        ["--cache-dir", "runs/cache_m", "--limit", "200",
         "--out", "runs/queue_ideas/preflight/sigma_residual.md"],
        note="I2 -- can sigma move an edge? kills itself if the sign is unpredictable"),
    Job("i4_size", "scripts/probe_ap_by_size.py",
        ["--cache-dir", "runs/cache_m", "--out", "runs/eval/ap_by_size.md"],
        ["--cache-dir", "runs/cache_m", "--limit", "200",
         "--out", "runs/queue_ideas/preflight/ap_by_size.md"],
        note="I4 screen -- decides whether the resolution retrain is worth starting"),
    Job("i5_night", "scripts/audit_night_restore.py",
        ["--out", "runs/eval/night_restore_audit.md"],
        ["--out", "runs/queue_ideas/preflight/night_restore_audit.md"],
        note="I5 audit -- writes no labels, launches no training"),
    Job("i8_ckpt", "scripts/probe_checkpoint_ensemble.py",
        ["--out", "runs/eval/checkpoint_ensemble.md", "--n-boot", "500"],
        ["--out", "runs/queue_ideas/preflight/checkpoint_ensemble.md", "--n-boot", "5", "--limit", "200"],
        note="I8 -- both checkpoints already cached; no inference needed"),
    Job("i6_perclass", "scripts/sweep_per_class.py",
        ["--cache-dir", "runs/cache_m", "--out", "runs/eval/per_class_levers.md"],
        ["--cache-dir", "runs/cache_m", "--out", "runs/queue_ideas/preflight/per_class_levers.md",
         "--limit", "300", "--gammas", "0.0", "0.5"],
        timeout=5400,
        note="I6 -- buoy is 5% of the GT and 50% of the metric, and no lever touches it"),
    Job("i9_within", "scripts/probe_within_modality.py",
        ["--cache-dir", "runs/cache_m", "--out", "runs/eval/within_modality.md",
         "--n-boot", "500"],
        ["--cache-dir", "runs/cache_m", "--out", "runs/queue_ideas/preflight/within_modality.md",
         "--n-boot", "5", "--limit", "200"],
        note="I3/I9 -- merge vs suppress, with a bootstrap this time"),

    # ---- lane cpu: the fits and sweeps -----------------------------------
    Job("i1_rerank", "scripts/fit_rerank.py",
        ["--cache-dir", "runs/cache_m", "--ir-cache", "runs/cache_m/gauss_ir_paired_clean.pkl",
         "--out", "runs/eval/rerank_loro.md"],
        ["--cache-dir", "runs/cache_m", "--out", "runs/queue_ideas/preflight/rerank_loro.md",
         "--limit", "600", "--lambdas", "0.0", "0.5", "--n-features", "4"],
        note="I1 on the OLD substrate -- expected negative; the control for i1_rerank_day"),
    Job("i9_split", "scripts/sweep_merge_support_split.py",
        ["--cache-dir", "runs/cache_m", "--out", "runs/eval/merge_support_split.md",
         "--n-boot", "300"],
        ["--cache-dir", "runs/cache_m", "--out", "runs/queue_ideas/preflight/merge_support_split.md",
         "--n-boot", "3", "--cells", "1", "--merge-ious", "0.85",
         "--support-ious", "0.30", "--support-gammas", "0.5"],
        note="I9 -- separate the merge threshold from the support threshold"),
    Job("i7_capir", "scripts/sweep_cap_ir_gated.py",
        ["--cache-dir", "runs/cache_m", "--out", "runs/eval/cap_ir_gated.md",
         "--n-boot", "300"],
        ["--cache-dir", "runs/cache_m", "--out", "runs/queue_ideas/preflight/cap_ir_gated.md",
         "--n-boot", "3", "--cells", "2", "--scales", "4.0", "16.0", "1.0"],
        note="I7 -- gate-conditional cap_ir_scale"),

    # ---- lane gpu: the substrate and the extra views ---------------------
    Job("i0_substrate", "scripts/build_day_substrate.py",
        ["--conditions", "clean", "--out-dir", "runs/cache_day"],
        ["--conditions", "clean", "--limit", "40", "--out-dir", "runs/cache_day_smoke",
         "--list-dir", "runs/derived_day_smoke"],
        lane="gpu", produces=("runs/cache_day/gauss_vis_day_clean.pkl",),
        note="I0 -- 9,284 day VIS val frames over FOUR runs incl. pohang04; unblocks I1/I2"),
    Job("i3_build", "scripts/build_tta_o2m.py",
        ["--mode", "tta", "o2m"],
        ["--mode", "tta", "o2m", "--limit", "40", "--out-suffix", "_smoke"],
        lane="gpu",
        produces=("runs/cache_tta/gauss_vis_paired_clean.pkl",
                  "runs/cache_o2m/gauss_vis_paired_clean.pkl"),
        note="I3 -- TTA views and the discarded one2many branch"),

    # ---- lane cpu: jobs that depend on the GPU builds --------------------
    Job("i3_probe", "scripts/probe_tta_o2m.py",
        ["--out", "runs/eval/tta_o2m.md", "--n-boot", "300"],
        ["--out", "runs/queue_ideas/preflight/tta_o2m.md", "--n-boot", "3", "--limit", "40",
         "--tta-cache", "runs/cache_tta/gauss_vis_paired_clean_smoke.pkl",
         "--o2m-cache", "runs/cache_o2m/gauss_vis_paired_clean_smoke.pkl"],
        needs=("i3_build",),
        note="I3 -- the lift screen, then the arms"),
    Job("i1_oracle_day", "scripts/probe_oracle_headroom.py",
        ["--vis-cache", "runs/cache_day/gauss_vis_day_clean.pkl",
         "--out", "runs/eval/oracle_headroom_day.md"],
        ["--vis-cache", "runs/cache_day_smoke/gauss_vis_day_clean.pkl",
         "--out", "runs/queue_ideas/preflight/oracle_headroom_day.md"],
        needs=("i0_substrate",),
        note="does the ceiling hold on 7.7x the frames and a fourth run?"),
    Job("i1_rerank_day", "scripts/fit_rerank.py",
        ["--vis-cache", "runs/cache_day/gauss_vis_day_clean.pkl",
         "--out", "runs/eval/rerank_loro_day.md",
         "--model-out", "runs/eval/rerank_model_day.pkl"],
        ["--vis-cache", "runs/cache_day_smoke/gauss_vis_day_clean.pkl",
         "--out", "runs/queue_ideas/preflight/rerank_loro_day.md", "--limit", "40",
         "--lambdas", "0.0", "0.5", "--n-features", "4"],
        needs=("i0_substrate",),
        note="I1 THE headline job -- same fit, 7.7x the data, four-fold LORO"),
    Job("i2_sigma_day", "scripts/probe_sigma_residual.py",
        ["--vis-cache", "runs/cache_day/gauss_vis_day_clean.pkl",
         "--out", "runs/eval/sigma_residual_day.md"],
        ["--vis-cache", "runs/cache_day_smoke/gauss_vis_day_clean.pkl",
         "--out", "runs/queue_ideas/preflight/sigma_residual_day.md"],
        needs=("i0_substrate",),
        note="I2 on the wide substrate"),

    # ---- held: costs GPU-hours or writes a large tree --------------------
    Job("i4_fullres_prep", "scripts/prepare_pohang_fullres.py", [], [],
        lane="gpu", hold=True,
        note="HELD -- re-preps 127k images at full resolution (hours, tens of GB). "
             "Gated on i4_size saying the loss is in small objects."),
]

BY_ID = {j.id: j for j in JOBS}


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {}


def save_state(st: dict) -> None:
    QDIR.mkdir(parents=True, exist_ok=True)
    STATE.write_text(json.dumps(st, indent=2), encoding="utf-8")


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def key(mode: str, jid: str) -> str:
    """State keys carry the mode. Without this a green preflight marks every job
    terminal and `run` skips the entire queue, reporting success having measured
    nothing."""
    return f"{mode}:{jid}"


def run_job(job: Job, mode: str, st: dict, python: str) -> str:
    args = job.pre_args if mode == "preflight" else job.args
    LOGDIR.mkdir(parents=True, exist_ok=True)
    log = LOGDIR / f"{job.id}.{mode}.log"
    cmd = [python, "-u", str(ROOT / job.script), *args]
    with _LOCK:
        st[key(mode, job.id)] = {"state": "running", "started": now(), "cmd": " ".join(cmd),
                                 "log": str(log.relative_to(ROOT))}
        save_state(st)
    print(f"[{job.lane}] START {job.id}  ({job.note})")
    t0 = time.time()
    limit = job.pre_timeout if mode == "preflight" else job.timeout
    rc = None
    try:
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(" ".join(cmd) + f"\n[timeout {limit}s]\n\n")
            fh.flush()
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
                               timeout=limit)
        rc = p.returncode
        state = "done" if rc == 0 else "failed"
    except subprocess.TimeoutExpired:
        # A HUNG job is the failure mode a queue must survive. Without this the GPU
        # lane stalls forever and every dependent spins out its full wait.
        state = "timeout"
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n[queue] killed after {limit}s\n")
    except Exception as e:  # noqa: BLE001 - a crash in the RUNNER must not end the queue
        state = "failed"
        with open(log, "a", encoding="utf-8") as fh:
            fh.write(f"\n[queue] runner error: {type(e).__name__}: {e}\n")
    dt = time.time() - t0
    with _LOCK:
        st[key(mode, job.id)].update({"state": state, "rc": rc,
                                      "finished": now(), "seconds": round(dt, 1)})
        save_state(st)
    tag = {"done": "OK  ", "timeout": "TIME", "failed": "FAIL"}.get(state, "FAIL")
    print(f"[{job.lane}] {tag}  {job.id}  {dt:.0f}s  -> {log.relative_to(ROOT)}")
    if state != "done":
        try:
            tail = log.read_text(encoding="utf-8", errors="replace").strip().splitlines()[-6:]
            for ln in tail:
                print(f"        | {ln}")
        except OSError:
            pass
    return state


def ready(job: Job, st: dict, mode: str) -> tuple[bool, str]:
    for n in job.needs:
        s = st.get(key(mode, n), {}).get("state")
        if s == "done":
            continue
        if s in ("failed", "skipped", "timeout"):
            return False, f"dependency {n} is {s}"
        return False, f"waiting on {n}"
    return True, ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["run", "preflight", "status", "reset", "list"])
    ap.add_argument("--workers", type=int, default=4, help="CPU lane pool size")
    ap.add_argument("--only", nargs="+", default=None)
    ap.add_argument("--job", default=None, help="for reset")
    ap.add_argument("--python", default=sys.executable)
    ap.add_argument("--include-held", action="store_true")
    args = ap.parse_args()

    st = load_state()

    if args.mode == "list":
        for j in JOBS:
            print(f"  {j.id:18s} {j.lane:3s} {'HELD ' if j.hold else '     '}"
                  f"{'needs=' + ','.join(j.needs) if j.needs else '':<28s} {j.note}")
        return 0

    if args.mode == "status":
        if not st:
            print("no state yet")
            return 0
        for j in JOBS:
            s = st.get(key("run", j.id)) or st.get(key("preflight", j.id), {})
            print(f"  {j.id:18s} {s.get('state', '-'):9s} "
                  f"{str(s.get('seconds', '')):>8s}s  {s.get('log', '')}")
        return 0

    if args.mode == "reset":
        if not args.job:
            return print("--job required") or 1
        for m in ("run", "preflight"):
            st.pop(key(m, args.job), None)
        save_state(st)
        print(f"reset {args.job}")
        return 0

    sel = [j for j in JOBS if (args.only is None or j.id in args.only)]
    if args.only is None and not args.include_held:
        held = [j.id for j in sel if j.hold]
        sel = [j for j in sel if not j.hold]
        for h in held:
            print(f"[hold] {h} -- {BY_ID[h].note}")
    # Resume: anything already terminal in this mode is skipped.
    todo = [j for j in sel if st.get(key(args.mode, j.id), {}).get("state") not in TERMINAL]
    print(f"\n{len(todo)} job(s) to run in mode={args.mode}, "
          f"{len(sel) - len(todo)} already terminal. workers={args.workers}\n")

    gpu = [j for j in todo if j.lane == "gpu"]
    cpu = [j for j in todo if j.lane == "cpu"]

    # The GPU lane is one thread, strictly serial: one 12 GB card, no concurrency.
    def gpu_lane():
        for j in gpu:
            ok, why = ready(j, st, args.mode)
            while not ok and why.startswith("waiting"):
                time.sleep(2)
                ok, why = ready(j, st, args.mode)
            if not ok:
                with _LOCK:
                    st[key(args.mode, j.id)] = {"state": "skipped", "why": why,
                                                "finished": now()}
                    save_state(st)
                print(f"[gpu] SKIP  {j.id} -- {why}")
                continue
            run_job(j, args.mode, st, args.python)

    def gpu_lane_safe():
        try:
            gpu_lane()
        except Exception as e:  # noqa: BLE001 - the GPU lane dying must not hang the CPU pool
            print(f"[gpu] ERROR lane: {type(e).__name__}: {e}")
            for j in gpu:
                with _LOCK:
                    if st.get(key(args.mode, j.id), {}).get("state") not in TERMINAL:
                        st[key(args.mode, j.id)] = {"state": "failed", "why": repr(e),
                                                    "finished": now()}
            save_state(st)

    gt = threading.Thread(target=gpu_lane_safe, daemon=False)
    gt.start()

    def cpu_job(j: Job):
        try:
            _cpu_job(j)
        except Exception as e:  # noqa: BLE001 - isolate: one worker must not end the run
            print(f"[cpu] ERROR {j.id}: {type(e).__name__}: {e}")
            with _LOCK:
                st[key(args.mode, j.id)] = {"state": "failed", "why": repr(e),
                                            "finished": now()}
                save_state(st)

    def _cpu_job(j: Job):
        ok, why = ready(j, st, args.mode)
        waited = 0
        while not ok and why.startswith("waiting") and waited < 14400:
            time.sleep(3)
            waited += 3
            ok, why = ready(j, st, args.mode)
        if not ok:
            with _LOCK:
                st[key(args.mode, j.id)] = {"state": "skipped", "why": why,
                                            "finished": now()}
                save_state(st)
            print(f"[cpu] SKIP  {j.id} -- {why}")
            return
        run_job(j, args.mode, st, args.python)

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as ex:
        list(ex.map(cpu_job, cpu))
    gt.join()

    st = load_state()
    ok = sum(1 for j in sel if st.get(key(args.mode, j.id), {}).get("state") == "done")
    bad = [j.id for j in sel if st.get(key(args.mode, j.id), {}).get("state")
           in ("failed", "timeout")]
    skip = [j.id for j in sel if st.get(key(args.mode, j.id), {}).get("state") == "skipped"]
    print(f"\n=== {ok}/{len(sel)} done"
          + (f" | failed: {', '.join(bad)}" if bad else "")
          + (f" | skipped: {', '.join(skip)}" if skip else "") + " ===")
    print(f"state: {STATE.relative_to(ROOT)}   logs: {LOGDIR.relative_to(ROOT)}")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
