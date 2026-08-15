"""Spend a JarvisLabs balance down to a floor, then destroy the instance.

This exists for one situation: a leftover balance you cannot withdraw and do not
intend to use, which you would rather convert into compute than leave sitting.
It buys nothing by itself — read the note in the runbook before running it.

The landing is driven by the instance's own `cost` counter, not by the account
balance. Balance updates in coarse steps (observed: unchanged for ~15-20 min,
then a ~Rs 13-18 jump), so a balance-triggered stop would overshoot by a fifth of
an hour. `cost` moves every poll and is the same number the provider bills.

Safety, in the order it matters:

  1. The cutoff is predictive. It stops when the cost projected one poll ahead
     would cross the budget, not when the current cost already has.
  2. `destroy` runs in a `finally`, so a crash, a Ctrl-C or a network error ends
     with the instance gone rather than billing unattended.
  3. Storage is deliberately tiny (10 GB, Rs 0.12/h). If this process is killed
     between polls, that is the whole exposure until the provider's own
     low-balance pause catches it.

  python jarvislabs/burn_balance.py --floor 5 --yes
"""
from __future__ import annotations

import argparse
import sys
import time
from datetime import datetime

# 20 GB is the provider's floor ("hdd: ensure this value is greater than or equal
# to 20"), not a choice — it is the smallest volume that can be left billing if
# this process dies between polls.
STORAGE_GB = 20
STORAGE_RATE = 0.0122 * STORAGE_GB  # INR/h, measured: 50 GB billed Rs 0.61/h


def log(msg: str) -> None:
    print(f"{datetime.now():%H:%M:%S}  {msg}", flush=True)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=5.0,
                    help="stop with at least this much balance left (INR)")
    ap.add_argument("--gpu", default="L4")
    ap.add_argument("--region", default="india-noida-01")
    ap.add_argument("--spot", action="store_true",
                    help="cheaper per hour, but a reclaim pauses the burn")
    ap.add_argument("--interval", type=float, default=30.0, help="poll seconds")
    ap.add_argument("--max-hours", type=float, default=6.0, help="hard wall-clock cap")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    sys.path.insert(0, str(__import__("pathlib").Path(__file__).parent))
    from jarvislabs import Client

    client = Client()
    start_balance = float(client.account.balance().balance)
    budget = start_balance - args.floor
    if budget <= 1:
        log(f"balance {start_balance:.2f} is already at or below the floor {args.floor:.2f}")
        return 0

    # vcpus/ram are required by the create endpoint even for a GPU box; the
    # per-GPU allocation for the chosen type is what the console would pick.
    rate = vcpus = ram = None
    for r in client.account.resources().server_meta:
        if r.gpu_type == args.gpu and r.region == args.region:
            rate = float(r.spot_price if args.spot else r.price_per_hour)
            vcpus, ram = int(r.cpus_per_gpu), int(r.ram_per_gpu)
            break
    if rate is None:
        log(f"no price for {args.gpu} in {args.region}")
        return 2
    rate += STORAGE_RATE

    log(f"balance {start_balance:.2f}, floor {args.floor:.2f} -> budget {budget:.2f}")
    log(f"{args.gpu}{' spot' if args.spot else ''} at {rate:.2f}/h "
        f"-> about {budget / rate:.2f} h")
    if not args.yes:
        if input("create the instance and burn it? [y/N] ").strip().lower() != "y":
            return 1

    inst = client.instances.create(
        gpu_type=args.gpu, num_gpus=1, template="pytorch", storage=STORAGE_GB,
        vcpus=vcpus, ram=ram, name="burn", region=args.region, is_spot=args.spot,
    )
    mid = int(inst.machine_id)
    log(f"created {mid} — it will be destroyed when cost reaches {budget:.2f}")

    try:
        deadline = time.time() + args.max_hours * 3600
        while True:
            time.sleep(args.interval)
            cur = client.instances.get(mid)
            cost = float(cur.cost or 0)

            # One poll of headroom, plus a poll's worth of slack for the destroy
            # call itself. Erring early costs a rupee; erring late costs the floor.
            ahead = rate * (args.interval * 2) / 3600
            bal = float(client.account.balance().balance)
            log(f"cost {cost:6.2f}/{budget:.2f}   balance {bal:7.2f}   status {cur.status}")

            if cost + ahead >= budget:
                log("budget reached")
                return 0
            if bal <= args.floor + ahead:
                log("balance hit the floor (the coarse counter caught up first)")
                return 0
            if time.time() > deadline:
                log(f"hit the {args.max_hours} h cap with {budget - cost:.2f} unspent")
                return 0
            if str(cur.status) not in ("Running", "Creating", "Pending"):
                # A spot reclaim pauses the box: it stops burning, so there is no
                # reason to keep the volume around waiting for capacity.
                log(f"instance is {cur.status} — not burning any more")
                return 0
    finally:
        try:
            client.instances.destroy(mid)
            left = float(client.account.balance().balance)
            log(f"destroyed {mid}; balance {left:.2f}")
            if client.instances.list():
                log("WARNING: instances still listed — check jarvislabs.ai")
        except Exception as exc:  # noqa: BLE001
            log(f"COULD NOT DESTROY {mid} ({exc!r}) — destroy it by hand NOW at jarvislabs.ai")


if __name__ == "__main__":
    sys.exit(main())
