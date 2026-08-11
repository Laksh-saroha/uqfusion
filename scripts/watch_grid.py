"""Health watchdog for the laptop Phase 1 grid.

Appends one line to runs/health.log every INTERVAL seconds. Runs independently of
any Claude session, so a crash that starts while nobody is watching still leaves a
trail: the failure mode this exists for is a wrapper crash-loop, which from outside
looks exactly like "still training" (a cmd.exe and a python.exe are both alive).

Lines are grep-friendly and start with a status token:

    2026-08-11T10:55:12  OK    run=ship_yolo26m_seed0 ep=28 epochs=27 rate=5.3it/s ...
    2026-08-11T11:05:12  FAIL  crash-loop: wrapper on attempt 7 ...

    grep -v " OK " runs/health.log     # every problem since launch

Usage:  python scripts/watch_grid.py          (Ctrl-C to stop)
"""

from __future__ import annotations

import csv
import re
import time
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GRID_LOG = ROOT / "runs" / "grid_laptop.log"
HEALTH_LOG = ROOT / "runs" / "health.log"
RESULTS_CSV = ROOT / "runs" / "benchmark" / "benchmark_results_tail.csv"
RUNS_DIR = ROOT / "runs" / "benchmark" / "runs"

INTERVAL = 600  # seconds between samples
TAIL_BYTES = 8 << 20  # read only the tail; the log grows ~27 MB/run from progress bars
STALE_S = 900  # log untouched this long => hung
TOTAL_RUNS = 27

ANSI_PAT = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")
ERROR_PAT = re.compile(r"Traceback|OutOfMemory|CUDA out of memory|not recognized|MemoryError")
ATTEMPT_PAT = re.compile(r"\[wrapper\] attempt (\d+)")
GRID_START_PAT = re.compile(r"\[grid\] === (\S+?):")
EPOCH_PAT = re.compile(r"^\s*(\d+)/(\d+)\s")
RATE_PAT = re.compile(r"([\d.]+)it/s")


def tail_text(path: Path, nbytes: int) -> str:
    """Last nbytes of the log, with \\r progress-bar frames split into real lines."""
    if not path.is_file():
        return ""
    with open(path, "rb") as f:
        size = path.stat().st_size
        f.seek(max(0, size - nbytes))
        raw = f.read()
    text = raw.decode("utf-8", "replace").replace("\r", "\n")
    # Ultralytics prefixes every progress frame with an ANSI erase-line ("\x1b[K")
    # and colours its labels; left in, they defeat every ^-anchored pattern below.
    return ANSI_PAT.sub("", text)


def count_rows(path: Path) -> int:
    if not path.is_file():
        return 0
    with open(path, "r", encoding="utf-8", newline="") as f:
        return max(0, sum(1 for _ in csv.reader(f)) - 1)


def active_run_dir() -> str:
    """Newest live run dir. A long run pushes its own '[grid] ===' banner out of the
    tail window (progress bars add ~27 MB per run), so the log alone can't say which
    run is active once training has been going a while."""
    if not RUNS_DIR.is_dir():
        return "?"
    cands = [d for d in RUNS_DIR.iterdir()
             if d.is_dir() and d.name.startswith("ship_")
             and not d.name.endswith("_val") and "partial" not in d.name]
    if not cands:
        return "?"
    return max(cands, key=lambda d: d.stat().st_mtime).name


def sample() -> str:
    now = datetime.now().strftime("%Y-%m-%dT%H:%M:%S")

    if not GRID_LOG.is_file():
        return f"{now}  FAIL  no grid log at {GRID_LOG}"

    age = time.time() - GRID_LOG.stat().st_mtime
    text = tail_text(GRID_LOG, TAIL_BYTES)
    lines = text.splitlines()

    attempts = [int(m.group(1)) for m in (ATTEMPT_PAT.search(l) for l in lines) if m]
    attempt = max(attempts) if attempts else 1
    errors = sum(1 for l in lines if ERROR_PAT.search(l))

    starts = [m.group(1) for m in (GRID_START_PAT.search(l) for l in lines) if m]
    run = starts[-1] if starts else active_run_dir()

    epoch = rate = "?"
    for l in reversed(lines):
        m = EPOCH_PAT.match(l)
        if m:
            epoch = f"{m.group(1)}/{m.group(2)}"
            r = RATE_PAT.search(l)
            rate = f"{r.group(1)}it/s" if r else "?"
            break

    epochs_done = 0
    rc = RUNS_DIR / run / "results.csv"
    if rc.is_file():
        epochs_done = count_rows(rc)

    rows = count_rows(RESULTS_CSV)
    body = (f"run={run} ep={epoch} epochs={epochs_done} rate={rate} "
            f"rows={rows}/{TOTAL_RUNS} attempts={attempt} errs={errors} stale={age:.0f}s")

    # Order matters: a crash-loop is the failure this watchdog exists to catch, and
    # it also makes the log go stale, so report the cause and not the symptom.
    if attempt > 1:
        return f"{now}  FAIL  crash-loop: wrapper on attempt {attempt} | {body}"
    if errors:
        return f"{now}  FAIL  {errors} error line(s) in tail | {body}"
    if age > STALE_S:
        return f"{now}  WARN  log stale {age / 60:.0f} min - hung or between runs | {body}"
    return f"{now}  OK    {body}"


def main() -> None:
    HEALTH_LOG.parent.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            line = sample()
        except Exception as exc:  # noqa: BLE001 - a watchdog must never die on its own bug
            line = f"{datetime.now().strftime('%Y-%m-%dT%H:%M:%S')}  WARN  watchdog error: {exc!r}"
        with open(HEALTH_LOG, "a", encoding="utf-8") as f:
            f.write(line + "\n")
        print(line, flush=True)
        time.sleep(INTERVAL)


if __name__ == "__main__":
    main()
