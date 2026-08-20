"""Sequential, resumable queue for the CPU follow-up analyses.

The GPU queue (`run_queue.py`) is training-specific — checkpoints, resume from
`last.pt`, pause from `on_model_save`. These are CPU analyses over cached
predictions: each is a subprocess that either writes its markdown or does not, so
the queue only needs to run them in order, log, record terminal state and skip
what already succeeded on a rerun.

Ordering is by value-if-interrupted, not by cost: the tests that change how every
existing number is read come first, the exploratory sweeps last.

    python scripts/run_analysis_queue.py preflight   # every test, cheap config
    python scripts/run_analysis_queue.py run         # the real thing
    python scripts/run_analysis_queue.py status
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QDIR = ROOT / "runs" / "queue_analysis"
STATE = QDIR / "state.json"
LOG = QDIR / "queue.log"
TERMINAL = ("done", "failed")

# (id, script, real args, preflight args)
JOBS = [
    ("x0_fusion_ci", "scripts/eval_fusion_ci.py",
     ["--n-boot", "1000"],
     ["--n-boot", "3", "--conditions", "clean"]),
    ("x5_capability_refit", "scripts/eval_capability_refit.py",
     ["--n-boot", "1000"],
     ["--n-boot", "3", "--conditions", "clean"]),
    ("x4_registration_drift", "scripts/diag_registration_drift.py",
     [],
     ["--bins", "4"]),
    ("x1_score_calibration", "scripts/eval_score_calibration.py",
     ["--n-boot", "1000"],
     ["--n-boot", "3", "--conditions", "clean"]),
    # The two sweeps carry many arms, so their intervals are screening-grade at
    # 500/400 resamples. The three tests whose numbers would actually be quoted
    # (x0, x5, x1) keep 1000.
    ("x3_veto_hysteresis", "scripts/eval_veto_hysteresis.py",
     ["--n-boot", "500"],
     ["--n-boot", "3", "--conditions", "clean", "--windows", "1", "5", "--modes", "majority"]),
    ("x7_topk_truncation", "scripts/eval_topk_truncation.py",
     ["--n-boot", "400"],
     ["--n-boot", "3", "--conditions", "clean", "--ks", "50"]),
]


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def log(msg: str) -> None:
    QDIR.mkdir(parents=True, exist_ok=True)
    line = f"{now()}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def load_state() -> dict:
    if STATE.is_file():
        return json.loads(STATE.read_text(encoding="utf-8"))
    return {"runs": {}, "queue_status": "never started"}


def save_state(s: dict) -> None:
    QDIR.mkdir(parents=True, exist_ok=True)
    s["updated"] = now()
    STATE.write_text(json.dumps(s, indent=2), encoding="utf-8")


def cmd_run(args) -> int:
    preflight = args.mode == "preflight"
    state = load_state()
    state["queue_status"] = "running"
    state["mode"] = args.mode
    if preflight:
        state["runs"] = {}          # a pre-flight never counts as done work
    save_state(state)

    py = sys.executable
    log(f"analysis queue: {len(JOBS)} job(s), mode={args.mode}, python={py}")
    failures = []
    for jid, script, real, pre in JOBS:
        if args.only and jid not in set(args.only):
            continue
        rs = state["runs"].setdefault(jid, {})
        # A pre-flight result must never satisfy a real run. Both write the same
        # state file, so "done" alone is not enough — the mode has to match, or a
        # 3-resample smoke result silently stands in for the 1000-resample one and
        # the real output is never written.
        same_mode = rs.get("mode") == args.mode
        if rs.get("status") in TERMINAL and same_mode and not args.redo and not preflight:
            log(f"skip {jid} — already {rs['status']}")
            continue
        extra = list(pre if preflight else real)
        if preflight:
            extra += ["--out", f"runs/eval/preflight/{jid}.md"]
        cmd = [py, str(ROOT / script)] + extra
        rs.update(status="running", started=now(), error=None, mode=args.mode,
                  cmd=" ".join(cmd[1:]))
        save_state(state)
        log(f"=== {jid}: start ({' '.join(extra) or 'no args'})")
        t0 = time.time()
        proc = subprocess.run(cmd, cwd=str(ROOT), capture_output=True, text=True)
        dt = time.time() - t0
        tail = (proc.stdout or "").strip().splitlines()[-6:]
        if proc.returncode == 0:
            rs.update(status="done", finished=now(), seconds=round(dt, 1))
            log(f"=== {jid}: done in {dt/60:.1f} min")
            for t in tail:
                log(f"    | {t}")
        else:
            err = (proc.stderr or "").strip().splitlines()[-12:]
            rs.update(status="failed", finished=now(), seconds=round(dt, 1),
                      error="\n".join(err))
            failures.append(jid)
            log(f"=== {jid}: FAILED after {dt/60:.1f} min (exit {proc.returncode})")
            for t in err:
                log(f"    ! {t}")
            if preflight:
                save_state(state)
                log("pre-flight failed — fix before running the real queue")
                state["queue_status"] = "failed"
                save_state(state)
                return 1
        save_state(state)

    state["queue_status"] = "finished" if not failures else "finished with failures"
    save_state(state)
    log(f"queue finished; {len(failures)} failure(s)" + (f": {', '.join(failures)}" if failures else ""))
    return 1 if failures else 0


def cmd_status(_args) -> int:
    s = load_state()
    print(f"queue  : {s.get('queue_status')}  (mode {s.get('mode', '-')})")
    print(f"updated: {s.get('updated', '-')}")
    print(f"{'job':24s} {'status':9s} {'min':>6s}  output")
    for jid, script, _r, _p in JOBS:
        rs = s.get("runs", {}).get(jid, {})
        secs = rs.get("seconds")
        print(f"{jid:24s} {rs.get('status', 'pending'):9s} "
              f"{(secs / 60 if secs else 0):6.1f}  {script}")
        if rs.get("error"):
            print(f"    ! {rs['error'].splitlines()[-1][:120]}")
    return 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="mode", required=True)
    for name in ("run", "preflight"):
        q = sub.add_parser(name)
        q.add_argument("--only", nargs="+")
        q.add_argument("--redo", action="store_true")
        q.set_defaults(func=cmd_run)
    sub.add_parser("status").set_defaults(func=cmd_status)
    a = p.parse_args()
    return a.func(a)


if __name__ == "__main__":
    raise SystemExit(main())
