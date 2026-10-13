"""Extend the dgxanode01 VIS benchmark (25 epochs) to patience 20, on this machine.

The server's `warm_start_grid_v2.py` finished 10 of the 94 runs before access was lost
(2026-09-19). This runs the rest with the same mechanism:

  * The base run's `last.pt` is stripped (epoch -1, no optimizer), so it is loaded as weights and
    trained with `epochs=100, patience=20` and the base recipe from its `args.yaml`.
  * EarlyStopping is seeded so patience CONTINUES from the base run's best epoch:
        best_fitness = base best mAP50-95,  best_epoch = -(25 - base best epoch)
    A run that had used 15 of its 20 stops after 5 more epochs without a new best.
    Ultralytics 8.4.90's fitness is mAP50-95 alone, the column the seed reads.
  * The output is `<run_id>_ext`, one per base run.

Added over the server script:

  * Resume. An interrupted extension resumes from its own `last.pt`, which carries optimizer, EMA
    and epoch. Ultralytics rebuilds EarlyStopping on resume and never restores it, so the stopper
    is re-seeded from base + extension history, keeping the earliest maximum. Rows the checkpoint
    never reached are cut from `results.csv` first.
  * A run whose base gap is already >= 20 met patience inside its base run and trains no epoch.
    The server script would have trained a 26th, and a late best there re-opens a closed run.
  * Batch planning. On Windows the driver may spill VRAM into system RAM rather than raise OOM:
    `yolo12x` at batch 16 reserved 24 GB on this 12 GB card and ran ~10x slower. Before a family's
    first run, the largest batch in BATCHES whose probe stays under 80% of VRAM is chosen
    (`batch_plan.json`; 90% before 2026-10-06). A batch below 16 keeps nbs=64, so the effective
    batch is still 64, and the deviation is recorded per run. Silent changes are refused:
    Ultralytics 8.4.90 halves the batch on a first-epoch OOM, and a spill past the card's memory
    mid-run fails the run. The one sanctioned change is a re-plan to a smaller batch: a resume
    then continues at the planned batch and records the switch (`batch_switches`).
  * A data gate before every run: `bench_ext_fingerprint.py --quick`, then
    `label_hash_ledger.py --expect 8ed69b5974ed`. That is the restored tree that full-pass
    `bench_ext_fingerprint.py` proved the server benchmark read.
  * Replay. At the end, the stopper is replayed over (base seed + extension rows). The run's
    length must equal the epoch the replay stops on, or the record says MISMATCH. All 10 server
    extensions replay exactly.
  * Monitoring. The queue writes the files `scripts/dashboard.py` reads (queue.json, state.json,
    live.json, control.json) into the output directory:
        python scripts/dashboard.py --queue-dir runs/vis_benchmark_stride4_ep25_ext --open
    "stops by" there is in extension epochs: a seeded best epoch of -6 shows as "stops by 14".
    Pause stops after the current epoch's checkpoint; Resume continues that run, patience intact.
  * Delegation. Runs listed in `<out>/delegated.json` train on another machine from the portable
    package (`scripts/bench_ext_package.py`); the queue skips them and the dashboard shows "skipped".
    The package runs this same file. On a moved package, a resumed checkpoint's paths are re-pointed.

Skipped: the 10 finished server extensions, and `yolov8s_seed2`, which diverged and is excluded
from the benchmark. The server pilot `yolo26m_seed0_ext` restarted patience (the old path bug)
and `yolo12x_seed1_ext` was cut off. Both are re-run here from their base weights.

Usage (from the repo root, GPU interpreter):
    python scripts/bench_ext_local.py --list
    python scripts/bench_ext_local.py --queue [--families yolov8n ...]
    python scripts/bench_ext_local.py --one vis_bench_yolov8n_seed0
    python scripts/bench_ext_local.py --probe [--families yolo12x ...]    # VRAM at batch 16
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "server_dgxanode01" / "workspace" / "uqfusion" / "runs" / "vis_benchmark_stride4_ep25"
SERVER_PROGRESS = BASE.parent / "queue_vis_benchmark_ep25" / "warm_start_progress.csv"
OUT = ROOT / "runs" / "vis_benchmark_stride4_ep25_ext"
DATA_YAML = ROOT / "runs" / "derived" / "data_vis_stride4.yaml"
VENV_PY = ROOT / ".venv" / "Scripts" / "python.exe"
QUEUE_JSON, STATE_JSON, LIVE_JSON, CONTROL_JSON = (OUT / f"{n}.json" for n in ("queue", "state", "live", "control"))
K = "metrics/mAP50-95(B)"
EXT_EPOCHS, PATIENCE = 100, 20
# The server used 6 workers (small models) / 2 (large). Workers set loading parallelism, not the recipe.
# Measured here on yolov8n: 6 workers 9.2 it/s at ~83% GPU; 10 workers ~8.0 it/s, because Ultralytics also
# opens 2x workers for the val loader and ~30 loader processes exhaust 32 GB RAM (17k pages/s). So 6, and 6
# rather than 2 for large models (18 loader processes either way).
SMALL_MIN_PER_EPOCH, SMALL_WORKERS, DEFAULT_WORKERS = 6.0, 6, 6
# Batch: the base recipe is 16 with nbs=64 (gradient accumulation to an effective 64). Where 16 does not fit
# in 12 GB (yolo12x reserved 24 GB), the author asked for a smaller batch: the largest of these that fits.
# All divide 64, so the accumulation keeps the effective batch, optimizer steps per epoch, LR and weight
# decay identical; only BatchNorm sees fewer images per forward. Recorded per run as a deviation.
BATCHES = (16, 8, 4, 2)
# fits = probe peak reserved VRAM (train, and a val-sized forward) under 80% of the card. It was 90% until
# 2026-10-06: yolov9c at batch 16 probed 10.45 of 10.79 GB, then trained at 10.4 GB reserved while Windows (dwm)
# and desktop apps held another ~1.2 GB, so the driver spilled 4.2 GB to system RAM and epochs went 14 -> 22-30
# min at ~30-55 W. The card-total guard never fired (10.4 < 12 GB). 80% leaves ~2.4 GB for everything else.
FIT_FRACTION = 0.80
PLAN_JSON = OUT / "batch_plan.json"
EXCLUDED = {"vis_bench_yolov8s_seed2": "diverged in the base grid (DIVERGENCE-ALARM.txt)"}
# Runs handed to another machine: {"to": "...", "runs": [...]}. The queue re-reads it before every run.
DELEGATED_JSON = OUT / "delegated.json"
TRAIN_LABEL_HASH = "8ed69b5974ed"
# The portable package (scripts/bench_ext_package.py) has no .venv and no label ledger, which needs the
# uqfusion package and the full train tree. Its gate is the fingerprint alone, which still covers every
# image and label a run reads.
LEDGER_PY = ROOT / "scripts" / "label_hash_ledger.py"
GATE_PY = VENV_PY if VENV_PY.is_file() else Path(sys.executable)
DATA_GATE = "fingerprint --quick" + (f" + label ledger {TRAIN_LABEL_HASH}" if LEDGER_PY.is_file() else "")
HEARTBEAT_S = 1.0
RC_DONE, RC_MISMATCH, RC_PAUSED = 0, 3, 4
FIELDS = ["run_id", "status", "host", "base_fitness", "gap_at_start", "ext_epochs", "ext_best_fitness",
          "ext_best_epoch", "improved_over_base", "stop", "replay", "batch", "workers", "min_per_epoch",
          "resumes", "started_at", "finished_at", "error"]


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def log(msg: str) -> None:
    line = f"{now()}  {msg}"
    print(line, flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "queue.log", "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ------------------------------------------------------------------------------- dashboard files
def read_json(path: Path, default):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data) -> None:
    """Never raises: the dashboard polls these files and a Windows rename can hit WinError 5
    while it has one open (the same hazard `run_queue.write_json` documents). Retry, then write in place."""
    try:
        payload = json.dumps(data, indent=1)
        tmp = path.with_suffix(".json.tmp")
        tmp.write_text(payload, encoding="utf-8")
        for _ in range(5):
            try:
                tmp.replace(path)
                return
            except PermissionError:
                time.sleep(0.05)
        path.write_text(payload, encoding="utf-8")
        tmp.unlink(missing_ok=True)
    except OSError:
        pass


def set_run_state(rid: str, **kw) -> None:
    st = read_json(STATE_JSON, {}) or {}
    st.setdefault("runs", {}).setdefault(rid, {}).update(kw)
    st["updated"] = now()
    write_json(STATE_JSON, st)


def set_queue_state(**kw) -> None:
    st = read_json(STATE_JSON, {}) or {}
    st.update(kw)
    st["updated"] = now()
    write_json(STATE_JSON, st)


def paused() -> bool:
    return bool((read_json(CONTROL_JSON, {}) or {}).get("paused"))


def delegated() -> dict[str, str]:
    """{run_id: where it runs} for the runs this machine must skip."""
    d = read_json(DELEGATED_JSON, {}) or {}
    return {r: d.get("to", "another machine") for r in d.get("runs", [])}


# ------------------------------------------------------------------------------- base-run facts
def orig_stats(run_dir: Path) -> tuple[float, int, float]:
    """(best mAP50-95, epochs since it, server min/epoch) from the base run's results.csv."""
    rows = list(csv.DictReader(open(run_dir / "results.csv", encoding="utf-8")))
    best = max(rows, key=lambda r: float(r[K]))             # max() keeps the first maximum
    gap = int(float(rows[-1]["epoch"])) - int(float(best["epoch"]))
    return float(best[K]), gap, float(rows[-1]["time"]) / 60.0 / len(rows)


def family(rid: str) -> str:
    return rid.replace("vis_bench_", "").rsplit("_seed", 1)[0]


def server_done() -> set[str]:
    return {r["run_id"] for r in csv.DictReader(open(SERVER_PROGRESS, encoding="utf-8")) if r["status"] == "done"}


def candidates() -> list[str]:
    base = sorted(p.name for p in BASE.iterdir() if p.is_dir() and not p.name.endswith("_ext"))
    skip = server_done() | set(EXCLUDED)
    return [r for r in base if r not in skip]


def ordered(runs: list[str]) -> list[str]:
    """Smallest family first (by the server's base min/epoch), seeds together."""
    mpe = {r: orig_stats(BASE / r)[2] for r in runs}
    fam = {}
    for r in runs:
        fam.setdefault(family(r), []).append(mpe[r])
    return sorted(runs, key=lambda r: (sum(fam[family(r)]) / len(fam[family(r)]), r))


def one_seed(runs: list[str]) -> tuple[list[str], list[str]]:
    """(kept, deferred): one run per family. A family with a finished run (here or on the server) needs none."""
    have = {family(r) for r in server_done()} | {family(r) for r in runs if (OUT / f"{r}_ext" / "ext_done.json").is_file()}
    kept, deferred = [], []
    for r in runs:
        f = family(r)
        if (OUT / f"{r}_ext" / "ext_done.json").is_file() or f not in have:
            kept.append(r)
            have.add(f)
        else:
            deferred.append(r)
    return kept, deferred


def ext_rows(ext: Path) -> list[tuple[int, float]]:
    p = ext / "results.csv"
    if not p.is_file():
        return []
    return [(int(float(r["epoch"])), float(r[K])) for r in csv.DictReader(open(p, encoding="utf-8"))]


def seed_state(base_fitness: float, gap: int, rows: list[tuple[int, float]]) -> tuple[float, int]:
    """EarlyStopping's (best_fitness, best_epoch) after the base run and `rows`: strict >, earliest max."""
    bf, be = base_fitness, -gap
    for e, f in rows:
        if f > bf:
            bf, be = f, e
    return bf, be


def replay(base_fitness: float, gap: int, rows: list[tuple[int, float]]) -> tuple[int | None, float, int]:
    """Epoch at which patience fires over `rows` (None if it never does), and the best seen."""
    bf, be = base_fitness, -gap
    for e, f in rows:
        if f > bf:
            bf, be = f, e
        if e - be >= PATIENCE:
            return e, bf, be
    return None, bf, be


def append_progress(row: dict) -> None:
    p = OUT / "progress.csv"
    new = not p.is_file()
    with open(p, "a", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow({k: row.get(k, "") for k in FIELDS})


# --------------------------------------------------------------------------------------------- one run
class _Paused(Exception):
    pass


def run_one(rid: str) -> int:
    import torch
    from ultralytics import YOLO

    base = BASE / rid
    ext = OUT / f"{rid}_ext"
    if (ext / "ext_done.json").is_file():
        log(f"=== {rid}: already done")
        return RC_DONE
    args = yaml.safe_load(open(base / "args.yaml", encoding="utf-8"))
    base_fitness, gap, mpe = orig_stats(base)
    batch_base = int(args["batch"])
    plan = read_json(PLAN_JSON, {}).get(family(rid), {})
    batch = plan.get("batch", batch_base) if plan else batch_base
    workers = SMALL_WORKERS if mpe < SMALL_MIN_PER_EPOCH else DEFAULT_WORKERS
    last = ext / "weights" / "last.pt"
    state_p = ext / "ext_state.json"
    state = json.loads(state_p.read_text()) if state_p.is_file() else {"resumes": 0, "started_at": now()}
    if gap >= PATIENCE:
        # The base run's own stopper fired at epoch 25 (patience 20 from its best): it is already done.
        ext.mkdir(parents=True, exist_ok=True)
        return finalize(rid, ext, base_fitness, gap, batch, workers, state)

    resume = False
    if last.is_file():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        ck_epoch, has_opt = ck.get("epoch", -1), ck.get("optimizer") is not None
        ck_args = ck.get("train_args") or {}
        sd = ck_args.get("save_dir")
        if ck_epoch >= 0 and has_opt and sd and os.path.normcase(os.path.abspath(sd)) != os.path.normcase(str(ext)):
            # The folder moved (the package on a drive with another letter). Ultralytics resumes into the
            # checkpoint's save_dir, so point its paths here. The weights and optimizer state are untouched.
            log(f"=== {rid}: checkpoint paths {sd} -> {ext}")
            ck["train_args"].update(data=str(DATA_YAML), project=str(OUT), save_dir=str(ext))
            torch.save(ck, last)
        del ck
        if ck_epoch >= 0 and has_opt:
            resume = True                                  # a resume keeps the batch/workers it started with
            planned, ck_batch = batch, int(ck_args.get("batch", batch))
            batch, workers = ck_batch, int(ck_args.get("workers", workers))
            if planned is not None and planned < ck_batch:
                # ...unless the family was re-planned smaller since (batch_plan.json): continue at the plan.
                # nbs 64 keeps the effective batch; the switch is recorded with the epoch it starts on.
                batch, at = planned, ck_epoch + 2
                sw = state.setdefault("batch_switches", [])
                if not any(s["at_ext_epoch"] == at for s in sw):
                    sw.append({"from": ck_batch, "to": planned, "at_ext_epoch": at, "at": now(),
                               "why": plan.get("note", "re-planned in batch_plan.json")})
                log(f"=== {rid}: batch {ck_batch} -> {planned} from ext epoch {at} (batch_plan.json)")
            if any(e > ck_epoch + 1 for e, _ in ext_rows(ext)):     # csv ran ahead of the checkpoint
                src = ext / "results.csv"
                bak = ext / f"results.csv.pre_resume_{int(time.time())}"
                src.rename(bak)
                lines = bak.read_text(encoding="utf-8").splitlines(keepends=True)
                keep = [ln for ln in lines[1:] if ln.strip() and int(float(ln.split(",")[0])) <= ck_epoch + 1]
                src.write_text(lines[0] + "".join(keep), encoding="utf-8")
            state["resumes"] += 1
        else:
            return finalize(rid, ext, base_fitness, gap, batch, workers, state)   # finished, marker lost
    elif ext.exists():
        ext.rename(ext.with_name(f"{ext.name}.aborted_{int(time.time())}"))
    if batch is None:
        raise RuntimeError(f"VRAM-GUARD: no batch in {BATCHES} fits {family(rid)} on this card (batch_plan.json)")
    ext.mkdir(parents=True, exist_ok=True)
    state.update(batch=batch, batch_base=batch_base, workers=workers)
    state_p.write_text(json.dumps(state), encoding="utf-8")
    if batch != batch_base:
        set_run_state(rid, note=f"batch {batch} (base {batch_base}), nbs 64 -> effective batch 64 unchanged")
    vram = torch.cuda.get_device_properties(0).total_memory
    tick = {"i": 0, "n": 0, "t": time.time(), "last": 0.0}

    def seed_cb(trainer):
        st = trainer.stopper
        rows = ext_rows(ext) if resume else []
        st.best_fitness, st.best_epoch = seed_state(base_fitness, gap, rows)
        if not trainer.best_fitness:
            trainer.best_fitness = base_fitness
        set_run_state(rid, best_epoch=int(st.best_epoch), best_map50_95=round(float(st.best_fitness), 5),
                      epochs_done=rows[-1][0] if rows else 0,
                      patience_gap=(rows[-1][0] if rows else 0) - int(st.best_epoch))
        log(f"[seed] {rid}: best_fitness={st.best_fitness:.5f} best_epoch={st.best_epoch} "
            f"(base {base_fitness:.5f}, gap {gap}, resume={resume}, ext rows {len(rows)}) patience={st.patience}")

    def epoch_start(trainer):
        if int(trainer.batch_size) != batch:
            raise RuntimeError(f"RECIPE-GUARD: Ultralytics reduced batch {batch} -> {trainer.batch_size} "
                               f"after an OOM; refusing to train a different recipe")
        tick.update(i=0, n=len(trainer.train_loader), t=time.time())

    def batch_end(trainer):
        tick["i"] += 1
        r = torch.cuda.memory_reserved()
        if r > vram:
            raise RuntimeError(f"VRAM-GUARD: {r / 2**30:.1f} GB reserved on a {vram / 2**30:.1f} GB card "
                               f"(driver spilled to system RAM); batch {batch} does not fit here")
        t = time.time()
        if t - tick["last"] < HEARTBEAT_S:
            return
        tick["last"] = t
        it_s = tick["i"] / max(t - tick["t"], 1e-6)
        write_json(LIVE_JSON, {"updated": now(), "run_id": rid, "epoch": int(trainer.epoch) + 1,
                               "epochs": int(trainer.epochs), "batch_i": tick["i"], "batch_n": tick["n"],
                               "it_s": round(it_s, 2), "img_s": round(it_s * batch, 1),
                               "epoch_eta_s": round((tick["n"] - tick["i"]) / it_s) if it_s > 0 else None,
                               "gpu_reserved_gb": round(r / 2**30, 2), "pause_pending": paused()})

    def fit_epoch_end(trainer):
        st, m = trainer.stopper, getattr(trainer, "metrics", None) or {}
        e = int(trainer.epoch) + 1
        set_run_state(rid, epochs_done=e, best_epoch=int(st.best_epoch), patience_gap=e - int(st.best_epoch),
                      map50_95=round(float(m.get(K, 0.0)), 5), best_map50_95=round(float(st.best_fitness), 5),
                      epoch_time_s=round(float(getattr(trainer, "epoch_time", 0.0) or 0.0), 1))

    def model_save(trainer):
        # After this epoch's checkpoint. Never when the stopper has just fired: a resume would then
        # train past the epoch the stopping rule ended the run on.
        if paused() and not trainer.stop:
            raise _Paused

    log(f"=== {rid}: {'resume' if resume else 'start'} (base_fitness={base_fitness:.5f} gap={gap}) "
        f"batch={batch} workers={workers}")
    model = YOLO(str(last if resume else base / "weights" / "last.pt"))
    for ev, fn in (("on_train_start", seed_cb), ("on_train_epoch_start", epoch_start),
                   ("on_train_batch_end", batch_end), ("on_fit_epoch_end", fit_epoch_end),
                   ("on_model_save", model_save)):
        model.add_callback(ev, fn)
    try:
        if resume:
            model.train(resume=True, batch=batch)          # Ultralytics honours a batch override on resume
        else:
            model.train(data=str(DATA_YAML), epochs=EXT_EPOCHS, patience=PATIENCE, batch=batch,
                        nbs=int(args.get("nbs", 64)), imgsz=args["imgsz"], workers=workers, seed=args["seed"],
                        deterministic=args["deterministic"], optimizer=args["optimizer"], amp=args["amp"],
                        cos_lr=args["cos_lr"], close_mosaic=args["close_mosaic"],
                        project=str(OUT), name=ext.name, exist_ok=True, verbose=True)
    except _Paused:
        log(f"=== {rid}: paused after epoch {ext_rows(ext)[-1][0] if ext_rows(ext) else 0} (resumable)")
        return RC_PAUSED
    finally:
        LIVE_JSON.unlink(missing_ok=True)
    if int(model.trainer.batch_size) != batch:
        raise RuntimeError(f"RECIPE-GUARD: trained at batch {model.trainer.batch_size}, not {batch}")
    return finalize(rid, ext, base_fitness, gap, batch, workers, state)


def finalize(rid, ext, base_fitness, gap, batch, workers, state) -> int:
    rows = ext_rows(ext)
    stop_at, bf, be = replay(base_fitness, gap, rows)
    n = rows[-1][0] if rows else 0
    if gap >= PATIENCE:
        stop, ok = "patience, in the base run", n == 0
    elif stop_at is not None:
        stop, ok = "patience", stop_at == n
    else:
        stop, ok = "epochs", n == EXT_EPOCHS
    rec = {"run_id": rid, "status": "done", "host": socket.gethostname(), "base_fitness": base_fitness,
           "gap_at_start": gap, "ext_epochs": n, "ext_best_fitness": bf, "ext_best_epoch": be,
           "improved_over_base": bf > base_fitness, "stop": stop, "replay": "OK" if ok else "MISMATCH",
           "batch": batch, "batch_base": state.get("batch_base", batch), "nbs": 64,
           "batch_switches": state.get("batch_switches", []),
           "workers": workers, "min_per_epoch": None, "resumes": state.get("resumes", 0),
           "started_at": state.get("started_at"), "finished_at": now(), "error": "",
           "rows_contiguous": [e for e, _ in rows] == list(range(1, n + 1)),
           "data_yaml": DATA_YAML.relative_to(ROOT).as_posix(), "label_hash_train": TRAIN_LABEL_HASH,
           "data_gate": DATA_GATE}
    try:
        import torch
        import ultralytics
        rec["versions"] = {"ultralytics": ultralytics.__version__, "torch": torch.__version__}
        if rows:
            rows_t = list(csv.DictReader(open(ext / "results.csv", encoding="utf-8")))
            rec["min_per_epoch"] = round(float(rows_t[-1]["time"]) / 60.0 / len(rows_t), 2)
    except Exception:                                                    # noqa: BLE001
        pass
    (ext / "ext_done.json").write_text(json.dumps(rec, indent=1), encoding="utf-8")
    append_progress(rec)
    log(f"=== {rid}: done, {n} ext epochs, best {bf:.5f}@{be} (base {base_fitness:.5f}), stop={stop}, "
        f"replay {rec['replay']}")
    return RC_DONE if ok else RC_MISMATCH


# --------------------------------------------------------------------------------------------- probe
class _ProbeDone(Exception):
    pass


class _NoFit(Exception):
    pass


def probe_one(f: str, batch: int) -> int:
    """Train family `f` for ~30 iterations at `batch` in a fresh process, then run one val-sized forward
    (2 x batch, half precision, as Ultralytics validates). Prints one JSON line."""
    import torch
    from ultralytics import YOLO

    rid = next(r for r in sorted(p.name for p in BASE.iterdir()) if not r.endswith("_ext") and family(r) == f)
    args = yaml.safe_load(open(BASE / rid / "args.yaml", encoding="utf-8"))
    vram = torch.cuda.get_device_properties(0).total_memory
    limit = FIT_FRACTION * vram
    out = {"family": f, "run": rid, "batch": batch, "limit_gb": round(limit / 2**30, 2)}

    def guard(trainer):
        if int(trainer.batch_size) != batch:
            raise _NoFit(f"Ultralytics reduced batch to {trainer.batch_size} after an OOM")

    def batch_end(trainer):                           # stop early: a spill makes every later iteration ~10x slower
        if torch.cuda.memory_reserved() > limit:
            raise _NoFit(f"{torch.cuda.memory_reserved() / 2**30:.1f} GB reserved during training")

    def end(trainer):                                 # skip the final full validation; measure one val batch instead
        out["peak_train_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        m = (trainer.ema.ema if trainer.ema else trainer.model).half().eval()
        x = torch.zeros(2 * batch, 3, args["imgsz"], args["imgsz"], device=trainer.device, dtype=torch.half)
        with torch.inference_mode():
            m(x)
        out["peak_val_gb"] = round(torch.cuda.max_memory_reserved() / 2**30, 2)
        raise _ProbeDone

    m = YOLO(str(BASE / rid / "weights" / "last.pt"))
    m.add_callback("on_train_epoch_start", guard)
    m.add_callback("on_train_batch_end", batch_end)
    m.add_callback("on_train_epoch_end", end)
    t0 = time.time()
    try:
        m.train(data=str(DATA_YAML), epochs=1, batch=batch, nbs=int(args.get("nbs", 64)), imgsz=args["imgsz"],
                workers=2, fraction=0.02, close_mosaic=0, plots=False, amp=args["amp"],
                project=str(OUT / "_probe"), name=f"{f}_b{batch}", exist_ok=True, verbose=False)
        out["fits"], out["error"] = False, "training ended without reaching the probe hook"
    except _ProbeDone:
        out["fits"] = max(out["peak_train_gb"], out["peak_val_gb"]) * 2**30 <= limit
    except _NoFit as e:
        out["fits"], out["error"] = False, str(e)
    except Exception as e:                                            # noqa: BLE001
        out["fits"], out["error"] = False, f"{type(e).__name__}: {str(e)[:160]}"
    out["seconds"] = round(time.time() - t0)
    print("PROBE " + json.dumps(out), flush=True)
    return 0


def plan_batch(f: str) -> int | None:
    """The largest batch in BATCHES that fits family `f` here, probed once and kept in batch_plan.json."""
    plan = read_json(PLAN_JSON, {})
    if f in plan:
        return plan[f]["batch"]
    OUT.mkdir(parents=True, exist_ok=True)
    tried = []
    for b in BATCHES:
        r = subprocess.run([sys.executable, "-u", __file__, "--probe-one", f, "--batch", str(b)], cwd=ROOT,
                           capture_output=True, text=True, errors="replace")
        line = next((ln for ln in r.stdout.splitlines() if ln.startswith("PROBE ")), None)
        res = json.loads(line[6:]) if line else {"batch": b, "fits": False, "error": (r.stdout + r.stderr)[-300:]}
        tried.append(res)
        log(f"[probe] {f} batch {b}: fits={res['fits']} train {res.get('peak_train_gb')} GB "
            f"val {res.get('peak_val_gb')} GB ({res.get('error', '')[:120]}) {res.get('seconds')}s")
        if res["fits"]:
            break
    chosen = next((t["batch"] for t in tried if t["fits"]), None)
    plan = read_json(PLAN_JSON, {})
    plan[f] = {"batch": chosen, "probes": tried, "at": now()}
    write_json(PLAN_JSON, plan)
    return chosen


def probe(families: list[str] | None) -> int:
    """Plan the batch for each family (largest family first) without training anything."""
    pick = {}
    for r in candidates():
        pick.setdefault(family(r), r)
    for f in families or sorted(pick, key=lambda f: -orig_stats(BASE / pick[f])[2]):
        plan_batch(f)
    return 0


# --------------------------------------------------------------------------------------------- queue
def gate(rid: str) -> str | None:
    fp = subprocess.run([str(GATE_PY), "-u", str(ROOT / "scripts" / "bench_ext_fingerprint.py"), "--quick"],
                        cwd=ROOT, capture_output=True, text=True)
    if fp.returncode != 0:
        return "fingerprint: " + (fp.stdout + fp.stderr).strip()[-300:]
    if not LEDGER_PY.is_file():
        return None
    lh = subprocess.run([str(GATE_PY), "-u", str(LEDGER_PY), "--scope", "train",
                         "--expect", TRAIN_LABEL_HASH, "--note", f"bench-ext before {rid}"],
                        cwd=ROOT, capture_output=True, text=True)
    if lh.returncode != 0:
        return "label ledger: " + (lh.stdout + lh.stderr).strip()[-300:]
    return None


def init_dashboard(runs: list[str], deferred: list[str] = ()) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    deleg = delegated()
    here = f"{len(runs)} runs" + (f", {sum(r in deleg for r in runs)} of them delegated" if deleg else "")
    write_json(QUEUE_JSON, {
        "created": now(),
        "note": "dgxanode01 ep25 VIS benchmark extended to patience 20 (warm start, patience carried over)",
        "defaults": {"variant": f"ep25 -> patience 20, {here} ({socket.gethostname()})", "imgsz": 640,
                     "batch": "16, lower where it does not fit (nbs 64)",
                     "workers": f"{SMALL_WORKERS} small / {DEFAULT_WORKERS} large",
                     "epochs": EXT_EPOCHS, "patience": PATIENCE},
        "runs": [{"id": r, "kind": family(r), "data": DATA_YAML.name} for r in runs]})
    st = read_json(STATE_JSON, {}) or {}
    for r in runs:
        rs = st.setdefault("runs", {}).setdefault(r, {})
        done = OUT / f"{r}_ext" / "ext_done.json"
        base_fitness, gap, _ = orig_stats(BASE / r)
        if done.is_file():
            d = json.loads(done.read_text())
            rs.update(status="done", epochs_done=d["ext_epochs"], best_epoch=d["ext_best_epoch"],
                      best_map50_95=round(d["ext_best_fitness"], 5), finished=d["finished_at"],
                      note=f"{d['stop']}, replay {d['replay']}, base {base_fitness:.5f}, host {d.get('host')}")
        elif r in deleg:
            rs.update(status="skipped", note=f"delegated to {deleg[r]}")
        elif r in deferred:
            rs.update(status="skipped", note="deferred: one seed per family for now (--one-seed)")
        else:
            rs.setdefault("status", "pending")
            rs.setdefault("best_epoch", -gap)
            rs.setdefault("best_map50_95", round(base_fitness, 5))
            rs.setdefault("note", f"base best {base_fitness:.5f}, gap {gap}: {max(PATIENCE - gap, 0)} epochs of patience left")
    st.update(queue_status="running", pid=os.getpid(), updated=now())
    write_json(STATE_JSON, st)


def queue(runs: list[str], deferred: list[str] = ()) -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "logs").mkdir(exist_ok=True)
    init_dashboard(runs + list(deferred), deferred)
    if os.name == "nt":             # keep Windows from sleeping while the queue runs; released when it exits
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)   # ES_CONTINUOUS | ES_SYSTEM_REQUIRED
    if deferred:
        log(f"=== --one-seed: {len(deferred)} runs deferred: {', '.join(deferred)}")
    log(f"=== queue: {len(runs)} runs, smallest family first (pid {os.getpid()}, host {socket.gethostname()}, "
        f"gate {DATA_GATE})")
    for rid in runs:
        ext = OUT / f"{rid}_ext"
        while True:
            if paused():
                set_queue_state(queue_status="paused")
                while paused():
                    time.sleep(10)
                set_queue_state(queue_status="running")
            if (ext / "ext_done.json").is_file():
                break
            deleg = delegated()
            if rid in deleg:
                set_run_state(rid, status="skipped", note=f"delegated to {deleg[rid]}")
                log(f"=== {rid}: skipped, delegated to {deleg[rid]}")
                break
            if not (ext / "weights" / "last.pt").is_file() and orig_stats(BASE / rid)[1] < PATIENCE:
                set_run_state(rid, status="probing", note="choosing the largest batch that fits")
                b = plan_batch(family(rid))
                base_b = int(yaml.safe_load(open(BASE / rid / "args.yaml", encoding="utf-8"))["batch"])
                if b is None:
                    set_run_state(rid, status="failed", finished=now(), error=f"no batch in {BATCHES} fits here")
                    append_progress({"run_id": rid, "status": "failed", "host": socket.gethostname(),
                                     "error": f"no batch in {BATCHES} fits", "finished_at": now()})
                    log(f"=== {rid}: FAILED (no batch in {BATCHES} fits)")
                    break
                set_run_state(rid, status="pending", note=(f"batch {b} (base {base_b}), nbs 64 -> effective 64"
                                                           if b != base_b else "batch 16 fits"))
            bad = gate(rid)
            if bad:
                set_run_state(rid, status="failed", error=f"DATA GATE: {bad[:200]}")
                set_queue_state(queue_status="stopped: data gate failed")
                log(f"=== {rid}: DATA GATE FAILED, queue stopped: {bad}")
                return 1
            set_run_state(rid, status="running", started=now(), error=None)
            with open(OUT / "logs" / f"{rid}.log", "a", encoding="utf-8") as fh:
                rc = subprocess.run([sys.executable, "-u", __file__, "--one", rid], cwd=ROOT,
                                    stdout=fh, stderr=subprocess.STDOUT).returncode
            if rc == RC_PAUSED:
                set_run_state(rid, status="paused")
                continue                                   # waits above, then resumes this run
            if rc in (RC_DONE, RC_MISMATCH):
                d = json.loads((ext / "ext_done.json").read_text())
                set_run_state(rid, status="done", finished=d["finished_at"], epochs_done=d["ext_epochs"],
                              best_epoch=d["ext_best_epoch"], best_map50_95=round(d["ext_best_fitness"], 5),
                              note=f"{d['stop']}, replay {d['replay']}, base {d['base_fitness']:.5f}"
                                   + (", IMPROVED" if d["improved_over_base"] else ""))
                break
            tail = (OUT / "logs" / f"{rid}.log").read_text(encoding="utf-8", errors="replace")[-4000:]
            err = next((ln.strip() for ln in reversed(tail.splitlines())
                        if "GUARD" in ln or "Error" in ln), f"rc {rc}")
            if ("GUARD" in tail or "OutOfMemory" in tail) and not (ext / "weights" / "last.pt").is_file():
                if ext.exists():                            # nothing resumable: it never fit at batch 16
                    ext.rename(ext.with_name(f"{ext.name}.nofit_{int(time.time())}"))
            append_progress({"run_id": rid, "status": "failed", "host": socket.gethostname(), "error": err[:200],
                             "finished_at": now()})
            set_run_state(rid, status="failed", finished=now(), error=err[:200])
            log(f"=== {rid}: FAILED ({err[:200]})")
            break
    set_queue_state(queue_status="finished")
    log("=== queue finished")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--list", action="store_true")
    g.add_argument("--queue", action="store_true")
    g.add_argument("--one")
    g.add_argument("--probe", action="store_true")
    g.add_argument("--probe-one", help=argparse.SUPPRESS)
    ap.add_argument("--batch", type=int, default=16, help=argparse.SUPPRESS)
    ap.add_argument("--families", nargs="+", default=None, help="restrict --queue/--probe to these families")
    ap.add_argument("--one-seed", action="store_true",
                    help="one run per family; families with a finished run get none (the rest show as skipped)")
    args = ap.parse_args()
    if args.probe_one:
        return probe_one(args.probe_one, args.batch)
    if args.one:
        return run_one(args.one)
    runs = ordered(candidates())
    if args.families:
        runs = [r for r in runs if family(r) in args.families]
    deferred: list[str] = []
    if args.one_seed:
        runs, deferred = one_seed(runs)
    if args.list:
        deleg = delegated()
        for r in runs:
            bf, gap, mpe = orig_stats(BASE / r)
            print(f"{r:30s} base {bf:.5f} gap {gap:2d} patience left {max(PATIENCE - gap, 0):2d}  "
                  f"server {mpe:5.1f} min/epoch" + (f"  -> delegated to {deleg[r]}" if r in deleg else ""))
        here = [r for r in runs if r not in deleg]
        print(f"{len(runs)} runs, {len(here)} here ({len(runs) - len(here)} delegated); floor (no new best) here "
              f"{sum(max(PATIENCE - orig_stats(BASE / r)[1], 0) for r in here)} epochs; data gate: {DATA_GATE}")
        return 0
    if args.probe:
        return probe(args.families)
    return queue(runs, deferred)


if __name__ == "__main__":
    sys.exit(main())
