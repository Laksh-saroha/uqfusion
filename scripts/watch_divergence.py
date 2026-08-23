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
DEFAULTS = {"ratio": 1.5, "window": 5, "min_history": 3, "map_frac": 0.6}


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

    best = max((r["map5095"] for r in rows[:i]), default=0.0)
    if best > 0 and row["map5095"] < args.map_frac * best:
        alarms.append(
            f"mAP50-95 {row['map5095']:.4f} fell to "
            f"{row['map5095'] / best:.2f}x its best {best:.4f} "
            f"(limit {args.map_frac:.2f}x)")

    if row["map50"] == 0:
        alarms.append("mAP50 is exactly 0 — the run is already gone")

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
                    help=f"epochs before the loss rule arms (default {DEFAULTS['min_history']})")
    ap.add_argument("--map-frac", type=float, default=DEFAULTS["map_frac"],
                    help=f"mAP50-95 fraction of best (default {DEFAULTS['map_frac']})")
    ap.add_argument("--no-pause", action="store_true", help="alarm only, never touch control.json")
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
    say(f"watching {results}")
    say(f"rules: {VAL_CLS} > {args.ratio}x trailing-{args.window} median "
        f"(armed after {args.min_history} epochs) | mAP50-95 < {args.map_frac}x best "
        f"| mAP50 == 0")
    say(f"on alarm: {'pause ' + str(args.queue_dir) if pausing else 'log only'}"
        f"  |  poll {args.interval:.0f}s")

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
