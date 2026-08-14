"""Deliver a pause that the network refused to carry.

The hole this fills: every other safeguard here pauses the instance by calling the
JarvisLabs API, which needs the laptop's internet — the very thing that just died.
If the uplink drops mid-upload, the transfer dies, `autostart_v2.sh` tries to pause,
the call fails, and a rented GPU sits there billing for an empty box until somebody
notices. Nothing on the laptop can reach it to say stop.

So this process does one thing, forever, until it succeeds: watch for "the upload is
dead and training never started", then retry the pause every 30 s — through a
five-minute blip or an eight-hour outage — and fire the moment connectivity returns.
That bounds the waste to the length of the outage, which is the best any laptop-side
system can do.

What it deliberately does NOT do:

* pause during an outage that happens while training is running. That case is not a
  problem — the grid keeps training on the instance whether or not the laptop can
  see it, and the supervisor reconnects afterwards. This guard exits as soon as
  training is confirmed.
* resume anything. Coming back is a human decision.

The remaining exposure is the outage itself. Closing that needs a dead-man's switch
*on the instance*, which needs an API key stored there — a call for the account
owner to make, not for this script.

Usage:  python jarvislabs/netguard.py [--machine-id N] [--stall-minutes 12]
"""

from __future__ import annotations

import argparse
import json
import socket
import sys
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATE = ROOT / "jarvislabs" / "l4_state.json"
HEALTH = ROOT / "runs" / "health_l4.log"
SYNC_LOG = ROOT / "runs" / "sync_l4.log"
AUTOSTART_LOG = ROOT / "runs" / "autostart_l4.log"

API_HOST = "api.jarvislabs.ai"
DEAD_SYNC = ("giving up on group", "[verify] FAIL", "FINGERPRINT MISMATCH")


def now() -> str:
    return datetime.now().strftime("%Y-%m-%dT%H:%M:%S")


def log(line: str) -> None:
    HEALTH.parent.mkdir(parents=True, exist_ok=True)
    with open(HEALTH, "a", encoding="utf-8") as f:
        f.write(line + "\n")
    print(line, flush=True)


def online() -> bool:
    """Reachability of the thing we actually need, not of 'the internet'."""
    for host, port in ((API_HOST, 443), ("1.1.1.1", 443)):
        try:
            socket.setdefaulttimeout(6)
            with socket.create_connection((host, port), timeout=6):
                return True
        except OSError:
            continue
    return False


def training_confirmed() -> bool:
    for path, needle in ((AUTOSTART_LOG, "training confirmed"),
                         (ROOT / "runs" / "health_l4.log", "  OK    id=")):
        if path.is_file() and needle in path.read_text(encoding="utf-8", errors="replace"):
            return True
    return False


def upload_dead(stall_seconds: float) -> str | None:
    """Reason the upload is over, or None if it is still moving."""
    if not SYNC_LOG.is_file():
        return None
    text = SYNC_LOG.read_text(encoding="utf-8", errors="replace")
    if "[verify] OK" in text:
        return None  # finished; not this guard's problem any more
    for needle in DEAD_SYNC:
        if needle in text:
            return f"upload failed ({needle})"
    age = time.time() - SYNC_LOG.stat().st_mtime
    if age > stall_seconds:
        return f"upload stalled — sync log untouched for {age / 60:.0f} min"
    return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--machine-id", type=int, default=None)
    ap.add_argument("--stall-minutes", type=float, default=12.0,
                    help="no growth in the sync log for this long counts as a dead upload "
                         "(a healthy group lands every ~10 min)")
    ap.add_argument("--interval", type=int, default=60)
    args = ap.parse_args()

    mid = args.machine_id
    if mid is None and STATE.is_file():
        mid = json.loads(STATE.read_text()).get("machine_id")
    if not mid:
        print("no machine id", file=sys.stderr)
        return 2
    mid = int(mid)
    stall = args.stall_minutes * 60

    log(f"{now()}  INFO  netguard armed for {mid}: pauses if the upload dies before training "
        f"starts, and keeps retrying the pause until the network is back")

    reason = None
    while True:
        try:
            if reason is None:
                if training_confirmed():
                    log(f"{now()}  INFO  netguard standing down — training is running, an outage "
                        f"from here costs nothing")
                    return 0
                reason = upload_dead(stall)
                if reason:
                    net = "offline" if not online() else "online"
                    log(f"{now()}  WARN  netguard: {reason} (laptop is {net}) — pausing instance {mid}")
            if reason is not None:
                # Retrying is the entire point: the first attempts are expected to
                # fail when the uplink is the thing that broke.
                if not online():
                    time.sleep(args.interval)
                    continue
                from jarvislabs import Client

                with Client() as c:
                    inst = c.instances.get(mid)
                    if inst.status != "Running":
                        log(f"{now()}  STOP  netguard: instance {mid} is already {inst.status}, "
                            f"cost {inst.cost:.2f} — nothing to do")
                        return 0
                    c.instances.pause(mid)
                    inst = c.instances.get(mid)
                log(f"{now()}  STOP  netguard paused instance {mid} ({reason}) — status "
                    f"{inst.status}, cost {inst.cost:.2f}. Disk kept; resume with "
                    f"jl resume {mid} --spot --yes then rerun jarvislabs/autostart_v2.sh")
                if STATE.is_file():
                    d = json.loads(STATE.read_text())
                    d.update(halted=True, halted_reason=reason, halted_at=now())
                    STATE.write_text(json.dumps(d, indent=2), encoding="utf-8")
                return 0
        except Exception as exc:  # noqa: BLE001 - a guard must not die on its own bug
            log(f"{now()}  WARN  netguard error: {exc!r} (still trying)")
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
