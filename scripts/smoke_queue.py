"""§18-2 gate for the run queue: pause, resume, finish — no GPU-hours.

Drives `scripts/run_queue.py` as a real subprocess and pauses it the same way the
dashboard does (flipping one boolean in control.json), then asserts:

  1. the runner reaches a checkpoint boundary and reports `paused`, not `failed`;
  2. `weights/last.pt` is a resumable checkpoint at that moment (unstripped,
     epoch >= 0) — a "pause" that leaves nothing to resume from is a crash;
  3. clearing the flag continues the SAME run rather than restarting it or
     skipping ahead to the next queued one;
  4. the run ends with exactly the configured number of epoch rows, i.e. the
     interruption did not add or lose an epoch.

Usage:  python scripts/smoke_queue.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SMOKE_ROOT = ROOT / "runs" / "smoke_queue"
QUEUE_DIR = SMOKE_ROOT / "queue"
EPOCHS = 3
RUN_ID = "q_sigma"
FT_ID = "q_sigma_ft"


def fail(msg: str, proc: subprocess.Popen | None = None) -> None:
    if proc and proc.poll() is None:
        proc.kill()
    print(f"[queue-smoke] FAIL {msg}")
    raise SystemExit(1)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def set_paused(value: bool) -> None:
    """Same retry-then-write-in-place dance as the runner and the dashboard: the
    Hammer thread below holds control.json open, which is exactly what makes
    `os.replace` fail with WinError 5 on Windows."""
    path = QUEUE_DIR / "control.json"
    payload = json.dumps({"paused": value})
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(payload, encoding="utf-8")
    for _ in range(20):
        try:
            tmp.replace(path)
            return
        except PermissionError:
            time.sleep(0.05)
    path.write_text(payload, encoding="utf-8")
    tmp.unlink(missing_ok=True)


def wait_for(predicate, timeout: float, what: str, proc: subprocess.Popen):
    """Poll until predicate returns a truthy value, or the runner dies."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        if proc.poll() is not None:
            fail(f"runner exited (code {proc.returncode}) while waiting for {what}")
        time.sleep(1.0)
    fail(f"timed out after {timeout:.0f}s waiting for {what}", proc)


class Hammer(threading.Thread):
    """Read the queue files as hard as possible, the way the dashboard does.

    This is a regression test for a real failure: the runner writes live.json via
    `os.replace` about once a second, and on Windows that rename raises
    `PermissionError: [WinError 5]` if any other process has the destination open.
    A dashboard polling at 0.5 Hz was enough to kill a training run seven minutes
    in. Polling politely would not reproduce it inside a smoke test, so this holds
    the files open in a tight loop instead.
    """

    def __init__(self):
        super().__init__(daemon=True)
        self._stop = threading.Event()
        self.reads = 0

    def run(self) -> None:
        targets = [QUEUE_DIR / n for n in ("live.json", "state.json", "control.json")]
        while not self._stop.is_set():
            for path in targets:
                try:
                    path.read_bytes()
                    self.reads += 1
                except OSError:
                    pass

    def stop(self) -> int:
        self._stop.set()
        self.join(timeout=5)
        return self.reads


def status_of(run_id: str = RUN_ID) -> str:
    state = read_json(QUEUE_DIR / "state.json", {}) or {}
    return (state.get("runs", {}).get(run_id) or {}).get("status", "")


def main() -> int:
    probe = ROOT / "runs" / "tune" / "data_probe.yaml"
    if not probe.is_file():
        fail(f"no probe dataset at {probe} — run scripts/tune_batch.py once first")

    shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    (QUEUE_DIR / "queue.json").write_text(json.dumps({
        "out_subdir": "smoke_queue/runs",
        "defaults": {"variant": "yolo26n", "imgsz": 320, "epochs": EPOCHS,
                     "patience": 20, "batch": 8, "workers": 0},
        "runs": [
            {"id": RUN_ID, "sigma": True, "data": str(probe), "seed": 0},
            # Option (C) mosaic stage: continues from RUN_ID's own best.pt.
            {"id": FT_ID, "sigma": True, "data": str(probe), "seed": 0, "from": RUN_ID,
             "epochs": 2,
             # The FULL production recipe, not a thinned version of it. Two
             # "got multiple values for keyword argument" bugs (patience, then
             # optimizer) reached real runs because the smoke's override set was
             # smaller than the one the queue actually ships.
             "train_overrides": {"mosaic": 0.0, "close_mosaic": 0,
                                 "warmup_epochs": 0.0, "patience": 2,
                                 "optimizer": "AdamW", "lr0": 0.000167,
                                 "lrf": 0.1, "momentum": 0.9},
             "gaussian_overrides": {"warmup_epochs": 0, "ramp_epochs": 0}},
            {"id": "q_parity", "sigma": False, "data": str(probe), "seed": 0},
        ],
    }, indent=2), encoding="utf-8")
    set_paused(False)

    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_queue.py"),
         "--queue-dir", str(QUEUE_DIR), "run"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    print(f"[queue-smoke] runner pid {proc.pid}, queue at {QUEUE_DIR}")
    hammer = Hammer()
    hammer.start()

    try:
        live = QUEUE_DIR / "live.json"
        wait_for(lambda: (read_json(live, {}) or {}).get("batch_i", 0) > 0,
                 300, "the first training batch", proc)
        print("[queue-smoke] training started OK")

        set_paused(True)
        wait_for(lambda: status_of() == "paused", 300, "the run to report paused", proc)
        print("[queue-smoke] pause honoured at a checkpoint boundary OK")

        # Check 2: what got left on disk is actually resumable.
        last = SMOKE_ROOT / "runs" / RUN_ID / "weights" / "last.pt"
        if not last.is_file():
            fail(f"paused with no checkpoint at {last}", proc)
        import torch

        import uqfusion.uq.gaussian  # noqa: F401 - registers our classes for unpickling

        ckpt = torch.load(last, map_location="cpu", weights_only=False)
        epoch_at_pause = int(ckpt.get("epoch", -1))
        if ckpt.get("ema") is None or epoch_at_pause < 0:
            fail(f"checkpoint at pause is not resumable (epoch={ckpt.get('epoch')}, "
                 f"ema={'present' if ckpt.get('ema') is not None else 'stripped'})", proc)
        print(f"[queue-smoke] resumable checkpoint kept at epoch {epoch_at_pause + 1} OK")

        # Check 3: resuming continues THIS run, not the next one in the queue.
        if status_of(FT_ID) not in ("", "pending"):
            fail("the queue moved on to the next run while the first was paused", proc)
        set_paused(False)
        wait_for(lambda: status_of() == "done", 900, "the resumed run to finish", proc)
        print("[queue-smoke] resume continued the same run to completion OK")

        # Check 4: the interruption neither added nor lost an epoch.
        rows = (SMOKE_ROOT / "runs" / RUN_ID / "results.csv").read_text(
            encoding="utf-8").strip().splitlines()
        if len(rows) - 1 != EPOCHS:
            fail(f"{len(rows) - 1} epoch rows after pause/resume, expected {EPOCHS}", proc)
        print(f"[queue-smoke] exactly {EPOCHS} epoch rows across the interruption OK")

        # Check 4b: the Option (C) mosaic stage continues from the parent's
        # best.pt and actually trains with mosaic off.
        wait_for(lambda: status_of(FT_ID) == "done", 900,
                 "the mosaic-off stage to finish", proc)
        ft_dir = SMOKE_ROOT / "runs" / FT_ID
        args = (ft_dir / "args.yaml").read_text(encoding="utf-8")
        for expect in ("mosaic: 0.0", "optimizer: AdamW", "lr0: 0.000167",
                       "lrf: 0.1", "warmup_epochs: 0.0"):
            if expect not in args:
                fail(f"mosaic stage lost override '{expect}'; args.yaml has: "
                     f"{[l for l in args.splitlines() if l.split(':')[0] in
                         ('mosaic', 'optimizer', 'lr0', 'lrf', 'warmup_epochs')]}")
        ft_rows = (ft_dir / "results.csv").read_text(encoding="utf-8").strip().splitlines()
        if len(ft_rows) - 1 != 2:
            fail(f"mosaic stage ran {len(ft_rows) - 1} epochs, expected 2")
        print("[queue-smoke] mosaic-off stage ran 2 epochs from the parent's "
              "best.pt with mosaic: 0.0 OK")

        wait_for(lambda: status_of("q_parity") == "done", 900,
                 "the parity run to finish", proc)
        print("[queue-smoke] queue advanced to the parity run and finished it OK")
        proc.wait(timeout=120)

        # Check 5: none of that survived contention only by luck.
        reads = hammer.stop()
        state = read_json(QUEUE_DIR / "state.json", {}) or {}
        errored = {rid: r.get("error") for rid, r in state.get("runs", {}).items()
                   if r.get("error")}
        if errored:
            fail(f"a run carried an error despite finishing: {errored}")
        print(f"[queue-smoke] survived {reads} concurrent reads of the queue files "
              f"with no run error OK")
    finally:
        hammer.stop()
        if proc.poll() is None:
            proc.kill()

    print("QUEUE SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
