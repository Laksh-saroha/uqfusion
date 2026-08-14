"""Stop the meter and record why.

Called from the automation whenever something needs a human: a failed upload, a
grid that will not start, rows that would not be comparable. Pausing (not
destroying) keeps the disk — the 16.4 GB payload and every checkpoint survive, and
`jl resume <id> --spot` picks up where it stopped.

Usage:  python jarvislabs/pause_instance.py --reason "upload verification failed"
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "jarvislabs" / "l4_state.json"
HEALTH = ROOT / "runs" / "health_l4.log"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine-id", type=int, default=None)
    ap.add_argument("--reason", required=True)
    args = ap.parse_args()

    mid = args.machine_id
    if mid is None and STATE.is_file():
        mid = json.loads(STATE.read_text()).get("machine_id")
    if not mid:
        print("no machine id", file=sys.stderr)
        return 2

    stamp = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")
    line = f"{stamp}  STOP  paused instance {mid}: {args.reason}"
    try:
        from jarvislabs import Client

        with Client() as c:
            c.instances.pause(int(mid))
            inst = c.instances.get(int(mid))
        line += f" | status={inst.status} cost={inst.cost:.2f}"
        rc = 0
    except Exception as exc:  # noqa: BLE001 - report, never mask
        line = f"{stamp}  FAIL  could NOT pause instance {mid} ({exc!r}) — reason was: {args.reason}"
        rc = 1

    HEALTH.parent.mkdir(parents=True, exist_ok=True)
    with open(HEALTH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line)
    if STATE.is_file():
        d = json.loads(STATE.read_text())
        d["halted_reason"] = args.reason
        d["halted_at"] = stamp
        STATE.write_text(json.dumps(d, indent=2), encoding="utf-8")
    return rc


if __name__ == "__main__":
    sys.exit(main())
