"""How many runs in a queue are not finished yet. Exit 0 when none are left.

The supervisor loop (`scripts/supervise_queue.sh`) needs a machine-readable
answer to "is this queue done?", and `run_queue.py status` prints a table for a
human. Counting is done against **queue.json**, not state.json: state.json only
contains runs that have started, so a pending count taken from it is silently
zero on a queue that has never run.

    python scripts/queue_remaining.py --queue-dir runs/queue_vis_benchmark_stride4
    # -> "12 of 93 remaining (done=81 failed=0 diverged=0)"; exit 1 while any remain
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TERMINAL = {"done", "failed", "skipped", "diverged"}


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--queue-dir", required=True)
    ap.add_argument("--quiet", action="store_true")
    args = ap.parse_args()

    d = Path(args.queue_dir)
    if not d.is_absolute():
        d = ROOT / d
    queue = read_json(d / "queue.json", None)
    if queue is None:
        print(f"no queue.json in {d}", file=sys.stderr)
        return 2  # not "done" — an unreadable queue must not end a supervisor loop
    state = read_json(d / "state.json", {}) or {}
    runs = state.get("runs", {}) or {}

    counts: dict[str, int] = {}
    remaining = 0
    for spec in queue.get("runs", []) or []:
        status = (runs.get(spec.get("id"), {}) or {}).get("status", "pending")
        counts[status] = counts.get(status, 0) + 1
        if status not in TERMINAL:
            remaining += 1

    if not args.quiet:
        tally = " ".join(f"{k}={v}" for k, v in sorted(counts.items()))
        print(f"{remaining} of {len(queue.get('runs', []))} remaining ({tally})")
    return 0 if remaining == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
