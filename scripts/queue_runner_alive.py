"""Is a runner currently working this queue? Exit 0 if yes, 1 if no.

The Windows supervisor (`supervise_queue_win.cmd`) needs this because Git Bash's
`kill -0` cannot see native Windows pids — it only knows its own MSYS ones — so
the shell supervisor's `--wait-pid` is unusable there, and starting a second
runner against one GPU is the exact mistake this must not make.

Two independent signals, either of which means "hands off":

  * the pid `state.json` records is a live process (precise, but a runner that
    died leaves the pid behind, so it can only ever say yes);
  * `live.json` moved within `--fresh` minutes (covers the window between a
    restart and its first state.json write).

    python scripts/queue_runner_alive.py --queue-dir runs/queue_ir_benchmark_stride4
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from queue_xlsx_report import pid_alive  # noqa: E402  - one definition, not two


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-dir", required=True)
    ap.add_argument("--fresh", type=float, default=10.0,
                    help="minutes of live.json staleness still counted as alive")
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    d = Path(args.queue_dir)
    if not d.is_absolute():
        d = ROOT / d

    pid = (read_json(d / "state.json", {}) or {}).get("pid")
    if pid_alive(pid):
        if not args.quiet:
            print(f"alive: runner pid {pid}")
        return 0

    updated = (read_json(d / "live.json", {}) or {}).get("updated")
    try:
        age = (datetime.now(timezone.utc).astimezone()
               - datetime.fromisoformat(str(updated))).total_seconds() / 60.0
    except (TypeError, ValueError):
        age = None
    if age is not None and age < args.fresh:
        if not args.quiet:
            print(f"alive: heartbeat {age:.1f} min old")
        return 0

    if not args.quiet:
        print(f"not running (pid {pid} gone, heartbeat "
              f"{'never' if age is None else f'{age:.0f} min old'})")
    return 1


if __name__ == "__main__":
    sys.exit(main())
