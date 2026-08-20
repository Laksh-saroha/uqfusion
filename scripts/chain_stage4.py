"""Wait for the stage-3 GPU queue, then run the approved stage-4 work in order.

Two GPU jobs must not overlap on a 12 GB card, and stage 4's first step needs a
checkpoint stage 3 has not finished writing yet. So this waits rather than
assuming, and refuses to start if the thing it is waiting for failed.

Order, cheap-and-decisive first:

  1. `eval_ir_upgrade_fusion` on the stage-3 p2feat checkpoint — no training. It
     answers §9.3's premise directly: the screen raised IR ship AP by 9%, does
     that reach the fusion night result at all?
  2. the stage-4 training queue — B5 ship-only, B3 CLAHE, B1 rect (IR then VIS)

Step 1 depends on `s3_p2feat_640_b10`; if that run did not produce weights the
step is skipped with a loud message rather than silently evaluating the old
checkpoint and reporting a null result.

Usage:
    python scripts/chain_stage4.py [--poll 60]
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
S3_STATE = ROOT / "runs" / "queue_screen3" / "state.json"
LOG = ROOT / "runs" / "queue_screen4" / "chain.log"
TERMINAL = ("done", "failed", "skipped")


def log(msg: str) -> None:
    LOG.parent.mkdir(parents=True, exist_ok=True)
    line = f"{datetime.now(timezone.utc).astimezone().isoformat(timespec='seconds')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def stage3_done() -> tuple[bool, dict]:
    if not S3_STATE.is_file():
        return False, {}
    s = json.loads(S3_STATE.read_text(encoding="utf-8"))
    runs = s.get("runs", {})
    if not runs:
        return False, s
    every = all(v.get("status") in TERMINAL for v in runs.values())
    return bool(every and s.get("queue_status") != "running") or s.get("queue_status") in (
        "finished", "idle"), s


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--poll", type=int, default=60)
    ap.add_argument("--max-wait-h", type=float, default=8.0)
    args = ap.parse_args()

    log("chain: waiting for the stage-3 GPU queue to finish")
    deadline = time.time() + args.max_wait_h * 3600
    while True:
        ok, s = stage3_done()
        if ok:
            break
        if time.time() > deadline:
            log(f"chain: stage 3 still running after {args.max_wait_h} h — giving up rather "
                f"than starting a second GPU job alongside it")
            return 1
        time.sleep(args.poll)
    log("chain: stage 3 terminal — " + ", ".join(
        f"{k}={v.get('status')}" for k, v in s.get("runs", {}).items()))

    # ---- step 1: does the better IR detector reach fusion? (no training) -----
    p2 = s.get("runs", {}).get("s3_p2feat_640_b10", {})
    wts = p2.get("best_weights")
    if p2.get("status") == "done" and wts and Path(wts).is_file():
        log(f"chain: step 1 — IR upgrade into fusion from {wts}")
        r = subprocess.run([sys.executable, str(ROOT / "scripts" / "eval_ir_upgrade_fusion.py"),
                            "--weights", wts, "--tag", "p2feat", "--n-boot", "1000"],
                           cwd=str(ROOT))
        log(f"chain: step 1 exit {r.returncode}")
    else:
        log(f"chain: step 1 SKIPPED — s3_p2feat_640_b10 is {p2.get('status')} with no usable "
            f"best.pt. Evaluating the old checkpoint here would report a null result that "
            f"means nothing, so it is skipped instead.")

    # ---- step 2: the stage-4 training queue ---------------------------------
    clahe_yaml = ROOT / "runs" / "derived" / "data_ir_clahe.yaml"
    if not clahe_yaml.is_file():
        log(f"chain: WARNING {clahe_yaml.name} missing — the CLAHE export has not finished. "
            f"That run will fail and the queue will continue past it.")
    log("chain: step 2 — stage-4 training queue")
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_queue.py"),
                        "--queue-dir", "runs/queue_screen4", "run"], cwd=str(ROOT))
    log(f"chain: step 2 exit {r.returncode}")
    log("chain: done")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
