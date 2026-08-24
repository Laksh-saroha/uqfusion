"""Divergence alarm for a live training run — D-12.

Watches a run's `results.csv` from outside the trainer and raises an alarm when
the run starts to come apart. Nothing here touches the training process, so it
can be started against a run that is already going (which is the whole point:
`run_queue.py`'s callbacks only reach a runner that is restarted, and you do not
restart a run at epoch 14 to install its own smoke alarm).

Why not the `mAP50 == 0` check D-12 originally specified. On the run that
actually diverged (`mc_vis_seed0_broken-20260823`) the numbers went:

    epoch 15   val/cls_loss  2.17   mAP50 0.644   mAP50-95 0.240
    epoch 16   val/cls_loss  5.02   mAP50 0.596   mAP50-95 0.221   <- first sign
    epoch 18   val/cls_loss 14.74   mAP50 0.507   mAP50-95 0.178
    epoch 20   val/cls_loss 71.85   mAP50 0.273   mAP50-95 0.092
    epoch 22   val/cls_loss  3.28   mAP50 0        mAP50-95 0      <- mAP50 == 0

`mAP50 == 0` first fires at epoch 22, six epochs after the run is unrecoverable
and 100 minutes of A100 time later. The validation classification loss is the
leading indicator, exactly as the 08-23 handoff §4 says to watch it, so that is
the primary rule here and mAP50 == 0 is kept only as a floor.

Both mAP rules ignore the first `warmup` epochs, matching the trainer's LR
warmup. A run that has not made a detection *yet* is not a run that is gone, and
the warmup LR peak produces a real mAP crater: healthy `ens_ir_seed2` fell to
0.44x at epoch 3 and recovered to finish at 0.128. The loss rule needs no such
guard — its `min_history` window does not fill until the warmup is over.

Thresholds are set from that same run rather than guessed. Over epochs 1-15 the
ratio of `val/cls_loss` to its own trailing-5 median never exceeded 1.10; at
epoch 16 it was 2.02. The 1.5 default sits between those with ~35% margin either
way. The mAP backstop (fall to 0.6x the best seen) fires at epoch 19 — later
than the loss rule, which is why it is a backstop.

On alarm the watcher can request a queue pause. That is deliberately a *pause*
and not a kill: `run_queue.py` reads `control.json` live on every batch and stops
at `on_model_save`, i.e. immediately after `last.pt` is written, so the run is
resumable with `run_queue.py --queue-dir <dir> resume` and nothing is lost. A
false positive costs a stalled run; a missed divergence costs the night's GPU.

Usage:
    python scripts/watch_divergence.py --run-dir runs/mc_dropout/mc_vis_seed0 \
        --queue-dir runs/queue_vis_server
    python scripts/watch_divergence.py --run-dir <dir> --no-pause   # alarm only
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

VAL_CLS = "val/cls_loss"
MAP50 = "metrics/mAP50(B)"
MAP5095 = "metrics/mAP50-95(B)"

# The single definition of the rule. `scripts/run_queue.py` imports these for its
# in-process alarm, so the sidecar watcher and the epoch callback cannot drift
# into disagreeing about what "diverged" means. Calibrated on
# mc_vis_seed0_broken-20260823 (see the module docstring), not chosen by taste.
DEFAULTS = {"ratio": 1.5, "window": 5, "min_history": 3, "map_frac": 0.6,
            "warmup": 3}


def thresholds(**overrides) -> SimpleNamespace:
    """Rule parameters in the shape `check` expects, with optional overrides."""
    unknown = set(overrides) - set(DEFAULTS)
    if unknown:
        raise ValueError(f"unknown threshold(s): {sorted(unknown)}")
    return SimpleNamespace(**{**DEFAULTS, **overrides})


def now() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


class Tee:
    """Print and append to a log, so a detached watcher leaves a record."""

    def __init__(self, path: Path | None):
        self.path = path
        if path:
            path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, msg: str) -> None:
        line = f"{now()}  {msg}"
        print(line, flush=True)
        if self.path:
            try:
                with open(self.path, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            except OSError:
                pass  # a watcher that dies on a locked log is worse than a gap


def read_rows(path: Path) -> list[dict]:
    """Parse results.csv, tolerating a torn read of a file being appended to.

    Ultralytics rewrites this file each epoch and we poll it from another
    process, so a short read is normal, not an error. Rows that do not parse are
    dropped and picked up on the next poll.
    """
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return []
    reader = csv.DictReader(text.splitlines())
    rows = []
    for raw in reader:
        try:
            row = {(k or "").strip(): (v or "").strip() for k, v in raw.items()}
            rows.append({
                "epoch": int(float(row["epoch"])),
                "time": float(row.get("time", 0.0) or 0.0),
                "val_cls": float(row[VAL_CLS]),
                "map50": float(row[MAP50]),
                "map5095": float(row[MAP5095]),
            })
        except (KeyError, TypeError, ValueError):
            continue
    return rows


def check(rows: list[dict], i: int, args) -> list[str]:
    """Rules for row `i`. Returns a list of alarm strings (empty = healthy)."""
    row, alarms = rows[i], []

    history = [r["val_cls"] for r in rows[max(0, i - args.window):i]]
    if len(history) >= args.min_history:
        med = statistics.median(history)
        if med > 0:
            ratio = row["val_cls"] / med
            if ratio > args.ratio:
                alarms.append(
                    f"{VAL_CLS} {row['val_cls']:.3f} is {ratio:.2f}x its trailing-"
                    f"{len(history)} median {med:.3f} (limit {args.ratio:.2f}x)")

    # Warmup is excluded from BOTH the baseline and the test. Ultralytics ramps
    # lr0 over `warmup_epochs` (3.0 here) and mAP reliably craters at the peak:
    # healthy ens_ir_seed2 went 0.0925 -> 0.1217 -> 0.0538 (0.44x) at epoch 3 and
    # recovered to finish at 0.128. Judged against a warmup-era best, this rule
    # would have killed it. Post-warmup the tightest healthy margin across 27
    # real runs is 0.63x, so the 0.60x limit stands on measured ground.
    post = rows[args.warmup:i]
    best = max((r["map5095"] for r in post), default=0.0)
    if best > 0 and i >= args.warmup and row["map5095"] < args.map_frac * best:
        alarms.append(
            f"mAP50-95 {row['map5095']:.4f} fell to "
            f"{row['map5095'] / best:.2f}x its post-warmup best {best:.4f} "
            f"(limit {args.map_frac:.2f}x)")

    # Warmup guard, same constant as the loss rule. Unguarded, this fired on
    # row 0 of any run whose first epoch has not yet produced a detection —
    # normal for a short/low-LR/from-scratch start, and it killed the
    # `smoke_queue_kinds` fine-tune at epoch 1 on 2026-08-24. Production runs
    # start from COCO weights at mAP50 0.10-0.32, so nothing live was hit, but
    # "already gone" has to mean gone, not "has not started yet".
    if row["map50"] == 0 and i >= args.warmup:
        alarms.append(
            f"mAP50 is exactly 0 at epoch {row['epoch']}, past the "
            f"{args.warmup}-epoch warmup — the run is already gone")

    return alarms


def request_pause(queue_dir: Path, say: Tee) -> None:
    """Ask the runner to stop at its next safe point (after last.pt is written)."""
    path = queue_dir / "control.json"
    try:
        current = json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}
    except (OSError, json.JSONDecodeError):
        current = {}
    current["paused"] = True
    current["updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
    current["paused_by"] = "watch_divergence"
    try:
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(current, indent=2), encoding="utf-8")
        tmp.replace(path)
        say(f"PAUSE REQUESTED via {path} — the runner stops after the next "
            f"last.pt. Resume with: run_queue.py --queue-dir {queue_dir} resume")
    except OSError as exc:
        say(f"!! could not write {path} ({exc}) — pause NOT requested, "
            f"alarm stands")


def request_skip_and_continue(queue_dir: Path, skip_ids: list[str], say: Tee,
                              timeout: float = 3600.0, poll: float = 10.0) -> bool:
    """Drop the condemned run(s) and let the queue advance to the next arm.

    `run_queue.py` offers no "abandon this run" signal, but it does offer two
    that compose into one. `control.json`'s `skip` list is consulted at the top
    of every run iteration (cmd_run), and `paused` stops the trainer at
    `on_model_save`, right after `last.pt`. So:

        1. write `skip` and `paused` together,
        2. wait for the runner to actually park (state.json status == paused),
        3. clear `paused`, leaving `skip` in place.

    On resume `cmd_run` re-enters the *same* index, hits the skip check, marks
    those ids `skipped` and walks forward to the next arm. Nothing is killed and
    no checkpoint is lost.

    Step 2 is not optional. Clearing `paused` before the runner has parked would
    let it sail on training the very run we are trying to abandon.

    Returns True if the queue was released, False if it never parked — in which
    case `paused` is deliberately LEFT SET, so the failure mode is a halted queue
    awaiting a human rather than a diverged run quietly continuing.
    """
    path = queue_dir / "control.json"
    state_path = queue_dir / "state.json"

    def read(p, default):
        try:
            return json.loads(p.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return default

    def write_control(**kw):
        cur = read(path, {})
        cur.update(kw)
        cur["updated"] = datetime.now().astimezone().isoformat(timespec="seconds")
        cur["paused_by"] = "watch_divergence"
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(cur, indent=2), encoding="utf-8")
        tmp.replace(path)

    merged = list(dict.fromkeys(list(read(path, {}).get("skip") or []) + skip_ids))
    try:
        write_control(skip=merged, paused=True)
    except OSError as exc:
        say(f"!! could not write {path} ({exc}) — NOTHING was changed, alarm stands")
        return False
    say(f"skip list = {merged}; queue paused, waiting for the runner to park "
        f"(this takes until the end of the current epoch)")

    deadline = time.time() + timeout
    while time.time() < deadline:
        runs = read(state_path, {}).get("runs", {})
        parked = [r for r in skip_ids if runs.get(r, {}).get("status") == "paused"]
        if parked or read(state_path, {}).get("queue_status") == "paused":
            try:
                write_control(paused=False)
            except OSError as exc:
                say(f"!! runner parked but could not clear paused ({exc}) — "
                    f"clear it by hand to let the queue advance")
                return False
            say(f"runner parked ({parked or 'between runs'}); paused cleared. "
                f"The queue will mark {skip_ids} skipped and move to the next arm.")
            return True
        time.sleep(poll)

    say(f"!! runner did not park within {timeout / 60:.0f} min. Leaving the queue "
        f"PAUSED on purpose — a halted queue is recoverable, a diverged run "
        f"burning GPU overnight is not. Clear it by hand once you have looked.")
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, help="dir holding results.csv")
    ap.add_argument("--results", type=Path, help="path to results.csv (overrides --run-dir)")
    ap.add_argument("--queue-dir", type=Path, help="queue dir to pause on alarm")
    ap.add_argument("--interval", type=float, default=60.0, help="poll seconds (default 60)")
    ap.add_argument("--ratio", type=float, default=DEFAULTS["ratio"],
                    help=f"val/cls_loss median multiple (default {DEFAULTS['ratio']})")
    ap.add_argument("--window", type=int, default=DEFAULTS["window"],
                    help=f"trailing median window (default {DEFAULTS['window']})")
    ap.add_argument("--min-history", type=int, default=DEFAULTS["min_history"],
                    help="epochs before the loss and zero-mAP rules arm "
                         f"(default {DEFAULTS['min_history']})")
    ap.add_argument("--map-frac", type=float, default=DEFAULTS["map_frac"],
                    help=f"mAP50-95 fraction of best (default {DEFAULTS['map_frac']})")
    ap.add_argument("--warmup", type=int, default=DEFAULTS["warmup"],
                    help="epochs the mAP rules ignore entirely, matching the "
                         f"trainer's LR warmup (default {DEFAULTS['warmup']})")
    ap.add_argument("--no-pause", action="store_true", help="alarm only, never touch control.json")
    ap.add_argument("--on-alarm", choices=("pause", "skip"), default="pause",
                    help="pause: halt the queue for a human (default). "
                         "skip: abandon --skip-ids and let the queue advance to the next arm")
    ap.add_argument("--skip-ids", nargs="+", metavar="ID",
                    help="run ids to abandon with --on-alarm skip (the diverging run "
                         "and any stage continuing from it)")
    ap.add_argument("--skip-timeout", type=float, default=3600.0,
                    help="seconds to wait for the runner to park before giving up "
                         "and leaving the queue paused (default 3600)")
    ap.add_argument("--log", type=Path, help="append to this log (default <run-dir>/divergence-watch.log)")
    ap.add_argument("--stall-factor", type=float, default=3.0,
                    help="warn if an epoch takes this multiple of the running average (default 3)")
    args = ap.parse_args()

    if args.results:
        results = args.results
    elif args.run_dir:
        results = args.run_dir / "results.csv"
    else:
        ap.error("need --run-dir or --results")

    log_path = args.log or (results.parent / "divergence-watch.log")
    say = Tee(log_path)
    sentinel = results.parent / "DIVERGENCE-ALARM.txt"

    pausing = bool(args.queue_dir) and not args.no_pause
    if args.on_alarm == "skip":
        if not args.queue_dir:
            ap.error("--on-alarm skip needs --queue-dir")
        if not args.skip_ids:
            ap.error("--on-alarm skip needs --skip-ids (be explicit about what is "
                     "abandoned; guessing the dependency graph is how the wrong arm "
                     "gets dropped)")
    say(f"watching {results}")
    say(f"rules: {VAL_CLS} > {args.ratio}x trailing-{args.window} median "
        f"(armed after {args.min_history} epochs) "
        f"| mAP50-95 < {args.map_frac}x post-warmup best | mAP50 == 0 "
        f"(both mAP rules ignore the first {args.warmup} epochs)")
    if not pausing:
        action = "log only"
    elif args.on_alarm == "skip":
        action = f"skip {args.skip_ids} and advance the queue ({args.queue_dir})"
    else:
        action = f"pause {args.queue_dir}"
    say(f"on alarm: {action}  |  poll {args.interval:.0f}s")

    seen = 0
    fired = False
    last_change = time.time()
    while True:
        rows = read_rows(results)
        if len(rows) > seen:
            for i in range(seen, len(rows)):
                row = rows[i]
                alarms = check(rows, i, args)
                flag = "ALARM" if alarms else "ok   "
                say(f"{flag} epoch {row['epoch']:>3}  {VAL_CLS} {row['val_cls']:7.3f}"
                    f"  mAP50 {row['map50']:.4f}  mAP50-95 {row['map5095']:.4f}")
                for a in alarms:
                    say(f"      -> {a}")
                if alarms and not fired:
                    fired = True
                    banner = (f"DIVERGENCE ALARM at epoch {row['epoch']}\n" +
                              "\n".join(f"  - {a}" for a in alarms) + "\n")
                    say("=" * 68)
                    say(banner.strip())
                    say("=" * 68)
                    try:
                        sentinel.write_text(f"{now()}\n{banner}", encoding="utf-8")
                    except OSError:
                        pass
                    if pausing and args.on_alarm == "skip":
                        request_skip_and_continue(args.queue_dir, list(args.skip_ids),
                                                  say, timeout=args.skip_timeout)
                        say("watcher done — the run it was watching has been "
                            "abandoned, so there is nothing further to judge")
                        return 0
                    if pausing:
                        request_pause(args.queue_dir, say)
            # An epoch that is slower than the rest usually means a stalled or
            # dead trainer, which no metric rule can see — the CSV just stops.
            if len(rows) > 1:
                gaps = [rows[k]["time"] - rows[k - 1]["time"] for k in range(1, len(rows))]
                mean_gap = sum(gaps) / len(gaps)
                if mean_gap > 0:
                    say(f"      ({mean_gap:.0f}s/epoch average)")
            seen = len(rows)
            last_change = time.time()
        else:
            quiet = time.time() - last_change
            if rows and len(rows) > 1:
                gaps = [rows[k]["time"] - rows[k - 1]["time"] for k in range(1, len(rows))]
                mean_gap = sum(gaps) / len(gaps)
                if mean_gap > 0 and quiet > args.stall_factor * mean_gap:
                    say(f"STALL? no new epoch for {quiet / 60:.0f} min "
                        f"({quiet / mean_gap:.1f}x the {mean_gap / 60:.0f} min average)")
                    last_change = time.time()  # warn once per interval, not every poll
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            say("stopped")
            return 0


if __name__ == "__main__":
    sys.exit(main())
