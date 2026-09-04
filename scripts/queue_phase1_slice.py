"""Wait for the night-restore fine-tune to release the GPU, then run the day/night slice.

The two jobs are independent in everything but hardware: the slice re-validates the
27 archived **Phase 1** checkpoints, not the checkpoint being trained. So it runs
once the card is free **whatever the fine-tune's outcome** -- success, early stop
or crash -- and this waiter deliberately does not check the training's verdict.

Waits on the process COMMAND LINE rather than a PID, because a PID freed over a
multi-hour wait can be reused by an unrelated process. Requires the pattern to be
absent on two consecutive polls before declaring the GPU free, so a momentary
gap in the process table cannot start the slice early.

Usage:
    python scripts/queue_phase1_slice.py
    python scripts/queue_phase1_slice.py --poll 120 --control 3
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WAIT_FOR = "train_night_restore"


def running(pattern: str) -> bool:
    """Any OTHER python process whose command line mentions `pattern`.

    Own PID is excluded: invoking this waiter as `--pattern train_night_restore`
    puts the pattern into its own command line, and without the exclusion it would
    match itself and wait forever.
    """
    me = os.getpid()
    ps = (
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        f"Where-Object {{ $_.CommandLine -like '*{pattern}*' "
        f"-and $_.ProcessId -ne {me} }} | "
        "Measure-Object | Select-Object -ExpandProperty Count"
    )
    try:
        out = subprocess.run(["powershell", "-NoProfile", "-Command", ps],
                             capture_output=True, text=True, timeout=60)
        return int((out.stdout or "0").strip() or 0) > 0
    except Exception as e:                      # a failed probe must not start the job
        print(f"[warn] process probe failed ({e}); assuming still running", flush=True)
        return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--poll", type=int, default=180, help="seconds between checks")
    ap.add_argument("--control", type=int, default=3)
    ap.add_argument("--out", default="runs/eval/phase1_day_night_slice.md")
    ap.add_argument("--pattern", default=WAIT_FOR)
    args = ap.parse_args()

    t0 = time.time()
    print(f"[queue] waiting for '{args.pattern}' to exit "
          f"(poll {args.poll}s)", flush=True)
    clear = 0
    while clear < 2:
        if running(args.pattern):
            clear = 0
        else:
            clear += 1
            print(f"[queue] clear {clear}/2", flush=True)
        if clear < 2:
            time.sleep(args.poll)
    print(f"[queue] GPU free after {(time.time() - t0) / 3600:.2f}h -- starting slice",
          flush=True)

    log = ROOT / "runs/logs_cache_draws/phase1_day_night_slice.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    cmd = [sys.executable, "-u", str(ROOT / "scripts/slice_phase1_day_night.py"),
           "--control", str(args.control), "--out", args.out]
    print(f"[queue] {' '.join(cmd)}", flush=True)
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, cwd=str(ROOT), stdout=fh,
                             stderr=subprocess.STDOUT)
    print(f"[queue] slice exited rc={rc}; log {log}", flush=True)
    return rc


if __name__ == "__main__":
    sys.exit(main())
