"""Track one or more `run_queue.py` queues in a single .xlsx workbook.

Written for unattended operation: it is a reader. It opens no checkpoint, holds
no lock the runner needs, and every cycle is wrapped so that a half-written
`state.json`, a missing run directory or a torn `results.csv` costs one row, not
the report. Nothing it does can take a training run down.

    # one shot
    python scripts/queue_xlsx_report.py --queue-dir runs/queue_vis_benchmark_stride4

    # keep it fresh every 5 minutes (this is how it runs on the server)
    python scripts/queue_xlsx_report.py --queue-dir runs/queue_vis_benchmark_stride4 \
        --out runs/queue_vis_benchmark_stride4/status.xlsx --interval 300

Sheets:
    Summary   one block per queue: progress, GPU-hours, ETA, live heartbeat, and
              a HEALTH line that says STALLED when a "running" queue has stopped
              writing state.json — the thing worth noticing while nobody is
              watching.
    Runs      every queued run in queue order, with status, epochs, best epoch,
              best mAP50-95 and the precision/recall of that same epoch.
    Variants  seeds folded into one row per variant: mean, sd and spread of
              best mAP50-95 — the draft of the benchmark table itself.
    Epochs    every epoch of every run, straight from Ultralytics' results.csv.
    Log       the tail of the queue's own log.

`--csv` also mirrors the Runs sheet to a .csv next to the workbook, so the
numbers survive even if the workbook is unreadable for any reason.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import socket
import statistics
import sys
import time
import traceback
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from xlsxlite import write_xlsx  # noqa: E402

TERMINAL = {"done", "failed", "skipped", "diverged"}

# Measured on the Phase 1 grid at imgsz 640, nc=2 (phase1_benchmark/results.csv).
# Used for the size ordering and the ETA's cost model. The CSV is preferred when
# present; this table is the fallback so the report also works on a machine that
# only carries the code.
VARIANT_SIZE = {
    "yolov8n": (3.16, 8.9), "yolov8s": (11.17, 28.8), "yolov8m": (25.9, 79.3),
    "yolov8l": (43.69, 165.7), "yolov8x": (68.23, 258.5),
    "yolov9t": (2.13, 8.5), "yolov9s": (7.32, 27.6), "yolov9m": (20.22, 77.9),
    "yolov9c": (25.59, 104.0), "yolov9e": (58.21, 193.0),
    "yolov10n": (2.78, 8.7), "yolov10s": (8.13, 25.1), "yolov10m": (16.58, 64.5),
    "yolov10b": (20.57, 99.4), "yolov10l": (25.89, 127.9), "yolov10x": (31.81, 171.8),
    "yolo11n": (2.62, 6.6), "yolo11s": (9.46, 21.7), "yolo11m": (20.11, 68.5),
    "yolo11l": (25.37, 87.6), "yolo11x": (56.97, 196.0),
    "yolo12n": (2.6, 6.7), "yolo12s": (9.29, 21.7), "yolo12m": (20.2, 68.1),
    "yolo12l": (26.45, 89.7), "yolo12x": (59.22, 200.3),
    "yolo26n": (2.57, 6.1), "yolo26s": (10.01, 22.8), "yolo26m": (21.9, 75.4),
    "yolo26l": (26.3, 93.8), "yolo26x": (58.99, 209.5),
}

DEFAULT_EPOCHS_GUESS = 40.0  # Phase 1 mean epochs_trained was 35.5; rounded up


# ------------------------------------------------------------------ small io


def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def read_rows(path: Path) -> list[dict]:
    """Ultralytics results.csv, tolerant of a row torn by a concurrent write."""
    try:
        with path.open(newline="", encoding="utf-8") as fh:
            return [{k.strip(): v for k, v in row.items() if k} for row in csv.DictReader(fh)]
    except (OSError, ValueError):
        return []


def num(value, default=None):
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return f if f == f else default  # NaN out


def parse_ts(text):
    try:
        return datetime.fromisoformat(str(text))
    except (TypeError, ValueError):
        return None


def hours_between(a, b) -> float | None:
    ta, tb = parse_ts(a), parse_ts(b)
    if not ta or not tb:
        return None
    return (tb - ta).total_seconds() / 3600.0


def sizes_for(variant: str) -> tuple[float | None, float | None]:
    p, g = VARIANT_SIZE.get(variant, (None, None))
    return p, g


def load_phase1_sizes() -> None:
    """Refresh VARIANT_SIZE from the Phase 1 CSV when it is on this machine."""
    path = ROOT / "phase1_benchmark" / "results.csv"
    for row in read_rows(path):
        v, p, g = row.get("variant"), num(row.get("params_m")), num(row.get("gflops"))
        if v and p:
            VARIANT_SIZE[v] = (p, g if g else VARIANT_SIZE.get(v, (None, None))[1])


# --------------------------------------------------------------- one queue


class QueueReport:
    """Everything the sheets need about one queue directory, read once."""

    def __init__(self, queue_dir: Path):
        self.dir = Path(queue_dir)
        self.name = self.dir.name
        self.queue = read_json(self.dir / "queue.json", {}) or {}
        self.state = read_json(self.dir / "state.json", {}) or {}
        self.live = read_json(self.dir / "live.json", {}) or {}
        self.control = read_json(self.dir / "control.json", {}) or {}
        self.supervisor = read_json(self.dir / "supervisor.json", {}) or {}
        self.defaults = self.queue.get("defaults", {}) or {}
        self.out_subdir = self.queue.get("out_subdir", "phase2")
        self.rows: list[dict] = []
        self.epochs: list[list] = []
        self._build()

    # -- run directory ------------------------------------------------------

    def run_dir(self, run_id: str, rs: dict) -> Path:
        recorded = rs.get("run_dir")
        if recorded:
            p = Path(recorded)
            if p.is_dir():
                return p
        return ROOT / "runs" / self.out_subdir / run_id

    # -- assembly -----------------------------------------------------------

    def _build(self) -> None:
        states = self.state.get("runs", {}) or {}
        for order, spec in enumerate(self.queue.get("runs", []) or [], start=1):
            merged = {**self.defaults, **spec}
            run_id = merged.get("id", f"run{order}")
            rs = states.get(run_id, {}) or {}
            variant = merged.get("variant", "")
            params, gflops = sizes_for(variant)
            status = rs.get("status", "pending")

            rows = read_rows(self.run_dir(run_id, rs) / "results.csv")
            best_row, last_row = self._pick_rows(rows, rs)
            epochs_done = rs.get("epochs_done") or (num(last_row.get("epoch")) if last_row else None)
            started, finished = rs.get("started"), rs.get("finished")
            elapsed_h = hours_between(started, finished or now_iso())
            h_per_epoch = (elapsed_h / epochs_done) if (elapsed_h and epochs_done) else None
            if h_per_epoch is None and rs.get("epoch_time_s"):
                h_per_epoch = float(rs["epoch_time_s"]) / 3600.0

            self.rows.append({
                "order": order, "queue": self.name, "run_id": run_id, "variant": variant,
                "params_m": params, "gflops": gflops, "seed": merged.get("seed"),
                "kind": merged.get("kind", "gaussian"),
                "sigma": merged.get("sigma"),
                "batch": merged.get("batch"), "workers": merged.get("workers"),
                "imgsz": merged.get("imgsz"), "epochs_cfg": merged.get("epochs"),
                "patience": merged.get("patience"), "data": merged.get("data"),
                "status": status,
                "epochs_done": int(epochs_done) if epochs_done else None,
                "best_epoch": rs.get("best_epoch"),
                "best_map50_95": rs.get("best_map50_95"),
                "best_map50": num(best_row.get("metrics/mAP50(B)")) if best_row else None,
                "best_precision": num(best_row.get("metrics/precision(B)")) if best_row else None,
                "best_recall": num(best_row.get("metrics/recall(B)")) if best_row else None,
                "last_map50_95": rs.get("map50_95"),
                "best_fitness": rs.get("best_fitness"),
                "patience_gap": rs.get("patience_gap"),
                "epoch_time_s": rs.get("epoch_time_s"),
                "started": started, "finished": finished,
                "elapsed_h": round(elapsed_h, 2) if elapsed_h else None,
                "h_per_epoch": round(h_per_epoch, 3) if h_per_epoch else None,
                "diverged_epoch": (rs.get("divergence_alarm") or {}).get("epoch"),
                "note": rs.get("note"),
                "error": rs.get("error"),
                "run_dir": str(self.run_dir(run_id, rs)),
            })

            for row in rows:
                self.epochs.append([
                    self.name, run_id, variant, merged.get("seed"),
                    int(num(row.get("epoch"), 0)),
                    num(row.get("metrics/mAP50-95(B)")), num(row.get("metrics/mAP50(B)")),
                    num(row.get("metrics/precision(B)")), num(row.get("metrics/recall(B)")),
                    num(row.get("train/box_loss")), num(row.get("train/cls_loss")),
                    num(row.get("train/dfl_loss")), num(row.get("val/box_loss")),
                    num(row.get("val/cls_loss")), num(row.get("val/dfl_loss")),
                    num(row.get("lr/pg0")), num(row.get("time")),
                ])

    @staticmethod
    def _pick_rows(rows: list[dict], rs: dict) -> tuple[dict, dict]:
        """The best.pt epoch's row, and the newest row. `best_epoch` comes from
        Ultralytics' composite fitness, not raw mAP, so it is used when known
        rather than re-deriving a maximum that would name a different epoch."""
        if not rows:
            return {}, {}
        best_epoch = rs.get("best_epoch")
        best = None
        if best_epoch:
            best = next((r for r in rows if num(r.get("epoch")) == float(best_epoch)), None)
        if best is None:
            best = max(rows, key=lambda r: num(r.get("metrics/mAP50-95(B)"), -1.0))
        return best, rows[-1]

    # -- derived ------------------------------------------------------------

    def counts(self) -> dict:
        out = {}
        for r in self.rows:
            out[r["status"]] = out.get(r["status"], 0) + 1
        return out

    def cost_model(self) -> tuple[float, float, float]:
        """(a, b, expected_epochs) for h_per_epoch ~= a + b * gflops, fitted on
        this queue's own finished runs. Falls back to a flat mean, then to the
        Phase 1 shape, so an ETA exists from the first completed run."""
        done = [r for r in self.rows
                if r["status"] == "done" and r["h_per_epoch"] and r["gflops"]]
        eps = [r["epochs_done"] for r in self.rows if r["status"] == "done" and r["epochs_done"]]
        expected = statistics.mean(eps) if eps else DEFAULT_EPOCHS_GUESS
        if len(done) >= 2:
            xs = [r["gflops"] for r in done]
            ys = [r["h_per_epoch"] for r in done]
            mx, my = statistics.mean(xs), statistics.mean(ys)
            var = sum((x - mx) ** 2 for x in xs)
            if var > 1e-9:
                b = sum((x - mx) * (y - my) for x, y in zip(xs, ys)) / var
                return max(0.0, my - b * mx), max(0.0, b), expected
        if done:
            return statistics.mean([r["h_per_epoch"] for r in done]), 0.0, expected
        return 0.0, 0.0, expected

    def eta_hours(self) -> tuple[float | None, float, float]:
        """(remaining hours, hours spent, mean hours per finished run)."""
        a, b, expected = self.cost_model()
        spent = sum(r["elapsed_h"] or 0.0 for r in self.rows if r["status"] in TERMINAL)
        finished = [r["elapsed_h"] for r in self.rows
                    if r["status"] == "done" and r["elapsed_h"]]
        mean_run = statistics.mean(finished) if finished else 0.0
        if a == 0.0 and b == 0.0:
            return None, spent, mean_run
        remaining = 0.0
        for r in self.rows:
            if r["status"] in TERMINAL:
                continue
            per_epoch = a + b * (r["gflops"] or 0.0)
            left = expected - (r["epochs_done"] or 0)
            remaining += max(0.0, left) * max(per_epoch, 0.0)
        return remaining, spent, mean_run

    def supervisor_age_min(self) -> float | None:
        updated = parse_ts(self.supervisor.get("updated"))
        if not updated:
            return None
        return (datetime.now(timezone.utc).astimezone() - updated).total_seconds() / 60.0

    def health(self) -> str:
        """One line worth reading first. `state.json` is rewritten at every epoch
        end, so its age against the current run's epoch time is the cheapest
        honest liveness signal there is."""
        left = [r for r in self.rows if r["status"] not in TERMINAL]
        if not left:
            return "COMPLETE — every queued run is terminal"
        if self.control.get("paused"):
            return "PAUSED — control.json has paused=true; nothing will start until it is cleared"
        updated = parse_ts(self.state.get("updated"))
        age_min = ((datetime.now(timezone.utc).astimezone() - updated).total_seconds() / 60.0
                   if updated else None)
        running = [r for r in self.rows if r["status"] == "running"]
        if not running:
            # A queue deliberately waiting its turn on the GPU looks exactly like
            # an abandoned one from state.json alone. The supervisor's heartbeat
            # is what tells them apart -- and only while it is fresh.
            sup_age = self.supervisor_age_min()
            if sup_age is not None and sup_age < 5:
                state = self.supervisor.get("state", "?")
                detail = self.supervisor.get("detail", "")
                if state.startswith("waiting"):
                    return f"WAITING (by design) — {detail}"
                if state in ("running", "starting", "backoff"):
                    return f"STARTING — supervisor says '{state}': {detail}"
                if state == "gave_up":
                    return f"GAVE UP — {detail}"
            if self.supervisor:
                return (f"IDLE — {len(left)} run(s) unfinished, none running, and the "
                        f"supervisor heartbeat is {sup_age:.0f} min old"
                        if sup_age is not None else
                        f"IDLE — {len(left)} run(s) unfinished and no supervisor heartbeat")
            return (f"IDLE — {len(left)} run(s) not finished and none running; "
                    "the runner process is not working this queue")
        # An epoch of the biggest variants takes ~13 min; three of them is a
        # generous "it is not merely slow" threshold.
        budget = max(45.0, 3.0 * (running[0]["epoch_time_s"] or 600) / 60.0)
        if age_min is not None and age_min > budget:
            return (f"STALLED? — {running[0]['run_id']} is marked running but state.json "
                    f"has not moved for {age_min:.0f} min")
        return (f"OK — {running[0]['run_id']} running, state.json "
                f"{age_min:.0f} min old" if age_min is not None else "OK — running")


def now_iso() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def pid_alive(pid) -> bool | None:
    """True/False if it can be told, None if not. Windows has no signal-0 probe —
    `os.kill(pid, 0)` there raises WinError 87 rather than answering — so it goes
    through OpenProcess/GetExitCodeProcess instead."""
    try:
        pid = int(pid)
    except (TypeError, ValueError):
        return None
    if os.name == "nt":
        try:
            import ctypes

            k32 = ctypes.windll.kernel32
            handle = k32.OpenProcess(0x1000, False, pid)  # QUERY_LIMITED_INFORMATION
            if not handle:
                return False
            code = ctypes.c_ulong()
            ok = k32.GetExitCodeProcess(handle, ctypes.byref(code))
            k32.CloseHandle(handle)
            return bool(ok) and code.value == 259  # STILL_ACTIVE
        except Exception:  # noqa: BLE001 - liveness is a nicety, never fatal
            return None
    try:
        os.kill(pid, 0)
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    except OSError:
        return None


# ----------------------------------------------------------------- sheets


RUN_COLUMNS = [
    ("order", 7, "int"), ("queue", 26, None), ("run_id", 26, None), ("variant", 11, None),
    ("params_m", 10, "num2"), ("gflops", 9, "num1"), ("seed", 6, "int"),
    ("status", 11, None), ("epochs_done", 12, "int"), ("epochs_cfg", 11, "int"),
    ("best_epoch", 11, "int"), ("best_mAP50-95", 14, "num5"), ("best_mAP50", 12, "num5"),
    ("best_precision", 14, "num5"), ("best_recall", 13, "num5"),
    ("last_mAP50-95", 14, "num5"), ("best_fitness", 13, "num5"),
    ("patience_gap", 13, "int"), ("epoch_time_s", 13, "num1"),
    ("elapsed_h", 10, "num2"), ("h_per_epoch", 12, "num3"),
    ("started", 26, None), ("finished", 26, None),
    ("batch", 7, "int"), ("workers", 8, "int"), ("imgsz", 7, "int"),
    ("kind", 10, None), ("data", 38, None), ("diverged_epoch", 14, "int"),
    ("error", 60, None), ("run_dir", 60, None),
]
RUN_KEYS = ["order", "queue", "run_id", "variant", "params_m", "gflops", "seed", "status",
            "epochs_done", "epochs_cfg", "best_epoch", "best_map50_95", "best_map50",
            "best_precision", "best_recall", "last_map50_95", "best_fitness", "patience_gap",
            "epoch_time_s", "elapsed_h", "h_per_epoch", "started", "finished", "batch",
            "workers", "imgsz", "kind", "data", "diverged_epoch", "error", "run_dir"]

EPOCH_COLUMNS = [
    ("queue", 26, None), ("run_id", 26, None), ("variant", 11, None), ("seed", 6, "int"),
    ("epoch", 7, "int"), ("mAP50-95", 11, "num5"), ("mAP50", 11, "num5"),
    ("precision", 11, "num5"), ("recall", 11, "num5"),
    ("train/box", 10, "num3"), ("train/cls", 10, "num3"), ("train/dfl", 10, "num3"),
    ("val/box", 10, "num3"), ("val/cls", 10, "num3"), ("val/dfl", 10, "num3"),
    ("lr/pg0", 11, "num5"), ("cum_time_s", 11, "num1"),
]


def summary_sheet(reports: list[QueueReport]) -> dict:
    rows: list[list] = [
        ["generated", now_iso()],
        ["host", socket.gethostname()],
        ["repo", str(ROOT)],
        [],
    ]
    for rep in reports:
        counts = rep.counts()
        total = len(rep.rows)
        done = counts.get("done", 0)
        eta_h, spent_h, mean_run_h = rep.eta_hours()
        a, b, expected = rep.cost_model()
        live = rep.live
        pid = rep.state.get("pid")
        alive = pid_alive(pid)
        rows += [
            [f"=== {rep.name} ===", rep.queue.get("note", "")[:300]],
            ["HEALTH", rep.health()],
            ["queue_status", rep.state.get("queue_status", "never started")],
            ["runner pid", f"{pid}" + ("" if alive is None else f" ({'alive' if alive else 'GONE'})")],
            ["paused", bool(rep.control.get("paused"))],
            ["state.json updated", rep.state.get("updated", "-")],
            ["supervisor", (f"pid {rep.supervisor.get('pid')} | {rep.supervisor.get('state')} | "
                            f"{rep.supervisor.get('detail')} | heartbeat "
                            f"{rep.supervisor_age_min():.0f} min old | restarts "
                            f"{rep.supervisor.get('restarts')}")
             if rep.supervisor and rep.supervisor_age_min() is not None else "none"],
            ["runs total", total],
            ["done", done],
            ["running", counts.get("running", 0)],
            ["pending (never started)", counts.get("pending", 0)],
            ["failed", counts.get("failed", 0)],
            ["diverged", counts.get("diverged", 0)],
            ["paused runs", counts.get("paused", 0)],
            ["progress", f"{done}/{total} = {100.0 * done / total:.1f}%" if total else "-"],
            ["GPU-hours spent (this queue)", round(spent_h, 1)],
            ["mean hours per finished run", round(mean_run_h, 2)],
            ["ETA remaining (h)", round(eta_h, 1) if eta_h is not None else "no finished run yet"],
            ["ETA finish (est.)",
             (datetime.now().astimezone() + timedelta(hours=eta_h)).strftime("%Y-%m-%d %H:%M")
             if eta_h is not None else "-"],
            ["ETA model", f"h/epoch = {a:.4f} + {b:.6f} x GFLOPs, {expected:.0f} epochs/run "
                          "(fitted on this queue's finished runs)"],
        ]
        if live:
            rows += [
                ["live run", live.get("run_id")],
                ["live epoch", f"{live.get('epoch')}/{live.get('epochs')}"],
                ["live batch", f"{live.get('batch_i')}/{live.get('batch_n')}"],
                ["live throughput", f"{live.get('img_s')} img/s ({live.get('it_s')} it/s)"],
                ["live GPU reserved (GB)", live.get("gpu_reserved_gb")],
                ["live heartbeat", live.get("updated")],
            ]
        rows.append([])
    return {"name": "Summary", "freeze": "A2",
            "columns": [{"header": "field", "width": 30}, {"header": "value", "width": 105}],
            "rows": rows}


def runs_sheet(reports: list[QueueReport]) -> dict:
    rows = [[r.get(k) for k in RUN_KEYS] for rep in reports for r in rep.rows]
    return {"name": "Runs", "freeze": "D2", "autofilter": True,
            "columns": [{"header": h, "width": w, "fmt": f} for h, w, f in RUN_COLUMNS],
            "rows": rows}


def variants_sheet(reports: list[QueueReport]) -> dict:
    cols = [("order", 7, "int"), ("queue", 26, None), ("variant", 12, None),
            ("params_m", 10, "num2"), ("gflops", 9, "num1"), ("runs_done", 10, "int"),
            ("seed0", 11, "num5"), ("seed1", 11, "num5"), ("seed2", 11, "num5"),
            ("mean", 11, "num5"), ("sd", 11, "num5"), ("spread", 11, "num5"),
            ("mean_epochs", 12, "num1"), ("mean_hours", 11, "num2"), ("statuses", 34, None)]
    rows = []
    for rep in reports:
        seen: dict[str, list] = {}
        for r in rep.rows:
            seen.setdefault(r["variant"], []).append(r)
        for variant, group in seen.items():
            group = sorted(group, key=lambda r: (r["seed"] if r["seed"] is not None else 0))
            by_seed = {r["seed"]: r.get("best_map50_95") for r in group}
            vals = [v for v in by_seed.values() if isinstance(v, (int, float))]
            done = [r for r in group if r["status"] == "done"]
            eps = [r["epochs_done"] for r in done if r["epochs_done"]]
            hrs = [r["elapsed_h"] for r in done if r["elapsed_h"]]
            rows.append([
                min(r["order"] for r in group), rep.name, variant,
                group[0]["params_m"], group[0]["gflops"], len(done),
                by_seed.get(0), by_seed.get(1), by_seed.get(2),
                statistics.mean(vals) if vals else None,
                statistics.stdev(vals) if len(vals) > 1 else None,
                (max(vals) - min(vals)) if len(vals) > 1 else None,
                statistics.mean(eps) if eps else None,
                statistics.mean(hrs) if hrs else None,
                ", ".join(f"{r['seed']}:{r['status']}" for r in group),
            ])
    rows.sort(key=lambda r: r[0])
    return {"name": "Variants", "freeze": "D2", "autofilter": True,
            "columns": [{"header": h, "width": w, "fmt": f} for h, w, f in cols],
            "rows": rows}


def epochs_sheet(reports: list[QueueReport]) -> dict:
    rows = [row for rep in reports for row in rep.epochs]
    return {"name": "Epochs", "freeze": "C2", "autofilter": True,
            "columns": [{"header": h, "width": w, "fmt": f} for h, w, f in EPOCH_COLUMNS],
            "rows": rows}


def tail_lines(path: Path, want: int, max_bytes: int = 262_144) -> list[str]:
    """The last `want` lines, read by seeking rather than by loading the file.

    A runner's stdout log is Ultralytics' progress bar: `queue_stdout.log` was
    76 MB after one day and grows all week. Reading it whole every cycle would
    make this monitor the heaviest thing on the box. Carriage returns are the
    progress bar redrawing one line, so only the last segment of each is kept.
    """
    try:
        size = path.stat().st_size
        with path.open("rb") as fh:
            if size > max_bytes:
                fh.seek(size - max_bytes)
                fh.readline()  # discard the partial line the seek landed inside
            blob = fh.read()
    except OSError:
        return []
    text = blob.decode("utf-8", errors="replace")
    lines = [line.split("\r")[-1].rstrip() for line in text.splitlines()]
    return [line for line in lines if line][-want:]


def log_sheet(reports: list[QueueReport], tail: int) -> dict:
    rows = []
    for rep in reports:
        for log in sorted(rep.dir.glob("*.log")):
            lines = tail_lines(log, tail)
            if not lines:
                continue
            size_mb = log.stat().st_size / 2**20
            rows.append([rep.name, log.name,
                         f"--- last {len(lines)} lines of {size_mb:.1f} MB ---"])
            rows += [[rep.name, log.name, line[:400]] for line in lines]
            rows.append([])
    return {"name": "Log", "freeze": "A2",
            "columns": [{"header": "queue", "width": 26}, {"header": "file", "width": 24},
                        {"header": "line", "width": 150}],
            "rows": rows}


# ------------------------------------------------------------------- main


def build(queue_dirs: list[Path], out: Path, csv_mirror: bool, tail: int) -> str:
    reports = []
    for d in queue_dirs:
        try:
            reports.append(QueueReport(d))
        except Exception:  # noqa: BLE001 - one bad queue must not lose the others
            traceback.print_exc()
    if not reports:
        return "no readable queue directory"

    sheets = [summary_sheet(reports), runs_sheet(reports), variants_sheet(reports),
              epochs_sheet(reports), log_sheet(reports, tail)]

    last_err = None
    for attempt in range(3):
        try:
            write_xlsx(out, sheets)
            break
        except OSError as exc:  # workbook open in Excel on Windows
            last_err = exc
            time.sleep(2)
    else:
        return f"could not write {out}: {last_err}"

    if csv_mirror:
        try:
            with out.with_suffix(".csv").open("w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh)
                w.writerow([h for h, _, _ in RUN_COLUMNS])
                for rep in reports:
                    for r in rep.rows:
                        w.writerow([r.get(k) for k in RUN_KEYS])
        except OSError as exc:
            print(f"[report] csv mirror failed: {exc}", file=sys.stderr)

    parts = []
    for rep in reports:
        c = rep.counts()
        parts.append(f"{rep.name}: {c.get('done', 0)}/{len(rep.rows)} done | {rep.health()}")
    return " || ".join(parts)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--queue-dir", action="append", required=True,
                    help="queue directory to report on; repeat for several")
    ap.add_argument("--out", default=None,
                    help="workbook path (default <first queue dir>/status.xlsx)")
    ap.add_argument("--interval", type=float, default=0.0,
                    help="seconds between refreshes; 0 = write once and exit")
    ap.add_argument("--csv", action="store_true", default=True,
                    help="also mirror the Runs sheet to .csv (default on)")
    ap.add_argument("--no-csv", dest="csv", action="store_false")
    ap.add_argument("--log-lines", type=int, default=200, help="log tail per queue")
    args = ap.parse_args()

    dirs = [Path(d) if Path(d).is_absolute() else ROOT / d for d in args.queue_dir]
    out = Path(args.out) if args.out else dirs[0] / "status.xlsx"
    if not out.is_absolute():
        out = ROOT / out
    load_phase1_sizes()

    while True:
        try:
            status = build(dirs, out, args.csv, args.log_lines)
        except Exception as exc:  # noqa: BLE001 - a monitor may not die of a bad cycle
            status = f"cycle failed: {type(exc).__name__}: {exc}"
            traceback.print_exc()
        print(f"[{now_iso()}] {out.name}: {status}", flush=True)
        if args.interval <= 0:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
