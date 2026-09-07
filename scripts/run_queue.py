"""Sequential, resume-safe, pausable run queue for the Phase 2 laptop test.

One process trains the queued runs in order. Three files under ``runs/queue/``
carry the state, and the dashboard (``scripts/dashboard.py``) only ever reads
them or flips a boolean in one of them:

    queue.json    what to run, in order          (written by `init`, edit by hand)
    state.json    what has happened per run      (written by the runner)
    live.json     current epoch/batch heartbeat  (written by the runner, ~1 Hz)
    control.json  {"paused": bool, ...}          (written by the dashboard or `pause`)

**Pause is graceful, not a kill.** The runner raises out of the training loop from
the `on_model_save` callback — the first instant at which `weights/last.pt` is
complete on disk. Resuming then re-enters Ultralytics' own resume path with the
optimizer, EMA, scaler and epoch counter intact, i.e. exactly the mechanism the
Phase 1 grid already relied on after crashes. Killing the process mid-epoch also
works (you lose that epoch), but it can catch `last.pt` half-written.

Usage:
    python scripts/run_queue.py init          # write runs/queue/queue.json
    python scripts/run_queue.py run           # work the queue (this is the long one)
    python scripts/run_queue.py status        # one-shot text status
    python scripts/run_queue.py pause         # ask the running queue to stop cleanly
    python scripts/run_queue.py resume
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))  # repo is not pip-installed in the GPU interpreter
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))  # for watch_divergence, a sibling script

from uqfusion.config import load_config, resolve_data_yaml  # noqa: E402

# The divergence rule lives in the standalone watcher and is imported, not copied.
# Two definitions of "diverged" would eventually disagree, and the disagreement
# would surface as one of them failing to fire on a real run. watch_divergence
# deliberately depends on nothing (stdlib only) so a broken package cannot
# disarm the alarm; the dependency only points this way.
from watch_divergence import (  # noqa: E402
    DEFAULTS as DIVERGENCE_DEFAULTS,
    check as divergence_check,
    read_rows as read_results_csv,
    thresholds as divergence_thresholds,
)

QUEUE_DIR = ROOT / "runs" / "queue"
QUEUE_JSON = QUEUE_DIR / "queue.json"
STATE_JSON = QUEUE_DIR / "state.json"
LIVE_JSON = QUEUE_DIR / "live.json"
CONTROL_JSON = QUEUE_DIR / "control.json"
QUEUE_LOG = QUEUE_DIR / "queue.log"

# "diverged" is terminal so a restart does not silently re-enter a run the alarm
# already condemned, and so the `from=` parent guard (which requires "done")
# refuses to fine-tune off it.
TERMINAL = {"done", "failed", "skipped", "diverged"}
HEARTBEAT_S = 1.0


def set_queue_dir(path: str | Path) -> None:
    """Relocate all four queue files. Used by the smoke test so a dry run of the
    queue machinery cannot touch the real queue's state."""
    global QUEUE_DIR, QUEUE_JSON, STATE_JSON, LIVE_JSON, CONTROL_JSON, QUEUE_LOG
    QUEUE_DIR = Path(path)
    QUEUE_JSON = QUEUE_DIR / "queue.json"
    STATE_JSON = QUEUE_DIR / "state.json"
    LIVE_JSON = QUEUE_DIR / "live.json"
    CONTROL_JSON = QUEUE_DIR / "control.json"
    QUEUE_LOG = QUEUE_DIR / "queue.log"


class PauseRequested(Exception):
    """Raised out of a training callback to stop at a checkpoint boundary."""


# --------------------------------------------------------------------------- io


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")


def read_json(path: Path, default):
    """Tolerant read: the dashboard and the runner write these concurrently, and a
    torn read must never take the trainer down."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(path: Path, data, strict: bool = False) -> bool:
    """Write JSON without ever taking the trainer down. Returns True on success.

    `os.replace` gives readers an all-or-nothing view, but on Windows the rename
    fails with `PermissionError: [WinError 5]` whenever another process has the
    destination open — and the dashboard polls these files every 2 s. That race
    killed a training run once: a once-a-second *cosmetic* heartbeat write raised
    straight through an Ultralytics callback and out of the epoch loop.

    So: retry the rename briefly, then fall back to writing in place. A reader can
    catch a torn file that way, which is why every reader here goes through
    `read_json` and treats a parse failure as "no data yet" rather than an error.
    Only `strict=True` callers (queue creation) are told about a failure at all.
    """
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = json.dumps(data, indent=2)
        tmp = path.with_suffix(path.suffix + ".tmp")
        tmp.write_text(payload, encoding="utf-8")
        for _ in range(5):
            try:
                os.replace(tmp, path)
                return True
            except PermissionError:
                time.sleep(0.05)
        path.write_text(payload, encoding="utf-8")  # reader holds the target: write in place
        tmp.unlink(missing_ok=True)
        return True
    except Exception as exc:  # noqa: BLE001 - bookkeeping must never sink a run
        if strict:
            raise
        print(f"[queue] warning: could not write {path.name}: "
              f"{type(exc).__name__}: {exc}", flush=True)
        return False


def log(msg: str) -> None:
    line = f"{now()}  {msg}"
    print(f"[queue] {msg}", flush=True)
    QUEUE_LOG.parent.mkdir(parents=True, exist_ok=True)
    with open(QUEUE_LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


_LAST_CONTROL = {"paused": False}


def control() -> dict:
    """Current control flags, falling back to the last good read.

    A plain default would read "not paused" whenever the dashboard happens to be
    mid-write, so a Pause could be dropped for a whole epoch. Remembering the last
    successful read makes a torn read a no-op instead of a reversal.
    """
    global _LAST_CONTROL
    value = read_json(CONTROL_JSON, None)
    if isinstance(value, dict):
        _LAST_CONTROL = value
    return _LAST_CONTROL


def set_control(**kw) -> dict:
    c = control()
    c.update(kw)
    c["updated"] = now()
    write_json(CONTROL_JSON, c)
    return c


def load_state() -> dict:
    return read_json(STATE_JSON, {"runs": {}})


def save_state(state: dict) -> None:
    state["updated"] = now()
    write_json(STATE_JSON, state)


def run_state(state: dict, run_id: str) -> dict:
    return state.setdefault("runs", {}).setdefault(run_id, {"status": "pending"})


# ------------------------------------------------------------------- queue init


def measurements() -> tuple[list[dict], float]:
    """Every candidate scripts/tune_batch.py has measured here, plus card size."""
    rows, total_gb = [], 0.0
    for name in ("tune_batch.json", "tune_workers.json"):
        tune = read_json(ROOT / "runs" / "tune" / name, None) or {}
        rows += [r for r in tune.get("results", [])
                 if "error" not in r and r.get("img_s", 0) > 0]
        total_gb = max(total_gb, float(tune.get("total_gb") or 0.0))
    return rows, total_gb


def default_queue(cfg: dict, variant: str, batch: int | None, workers: int | None) -> dict:
    b = cfg["benchmark"]
    rows, total_gb = measurements()
    fastest = max(rows, key=lambda r: r["img_s"]) if rows else None

    defaults = {
        "variant": variant,
        "imgsz": b["imgsz"],
        # Epochs and patience are the FULL-SCALE values on purpose: this is an
        # architecture test at reduced model size, not a reduced-schedule test.
        "epochs": b["epochs"],
        "patience": b["patience"],
        "batch": batch or (fastest or {}).get("batch") or b["batch"],
        "workers": workers if workers is not None
        else (fastest or {}).get("workers", b["workers"]),
    }

    # Report the row that was actually chosen — and, when that is not the fastest
    # measured one, why giving up throughput was the point.
    chosen = next((r for r in rows if r["batch"] == defaults["batch"]
                   and r["workers"] == defaults["workers"]), None)
    if chosen:
        card = f" of {total_gb:.1f} GB" if total_gb else ""
        note = (f"measured on this GPU: {chosen['img_s']} img/s, "
                f"driver peak {chosen['driver_peak_gb']} GB{card}")
        if fastest and fastest["img_s"] > chosen["img_s"]:
            gain = 100 * (fastest["img_s"] / chosen["img_s"] - 1)
            note += (f"; fastest measured was batch {fastest['batch']}/workers "
                     f"{fastest['workers']} at {fastest['img_s']} img/s (+{gain:.1f}%, "
                     f"driver peak {fastest['driver_peak_gb']} GB), not chosen")
        defaults["_sizing_source"] = note
    elif rows:
        defaults["_sizing_source"] = "chosen by hand; not among the measured candidates"

    vis = "runs/derived/data_vis_stride5.yaml"
    ir = "runs/derived/data_ir_stride2.yaml"

    # Option (C) for the mosaic problem: Ultralytics closes mosaic at the fixed
    # epoch `epochs - close_mosaic` = 90, which an early-stopping run never
    # reaches, so every model would train on mosaicked frames and be scored on
    # clean ones. Each run therefore gets a second stage that continues from its
    # own best.pt with mosaic off. Two stages rather than a mid-run switch keeps
    # the effect attributable: the pair of checkpoints IS the ablation.
    # The learning rate is the whole ballgame here, and the first attempt got it
    # wrong: leaving `optimizer: auto` restarts a fresh schedule at full lr0
    # (0.001667), which knocks a converged model straight off its optimum —
    # measured, mAP50-95 0.2505 -> 0.2111 in one epoch, recovering only to 0.2451
    # in ten. That is not what `close_mosaic` does. Stock close_mosaic fires at
    # epoch `epochs - close_mosaic` = 90 of 100, where the linear schedule
    # lf(x) = (1 - x/epochs)(1 - lrf) + lrf has already decayed LR to 0.109 x lr0.
    # So the faithful reconstruction of "the tail of the schedule with mosaic off"
    # is lr0 x 0.1 decaying to lr0 x 0.01 — which is what these numbers are.
    # `auto` must be replaced by an explicit optimizer, because auto ignores lr0.
    ft = {
        "epochs": 10,
        "train_overrides": {
            "mosaic": 0.0,        # the point of the stage
            "close_mosaic": 0,    # nothing left to close
            "warmup_epochs": 0.0,  # LR warm-up: 3 of 10 epochs would be absurd
            "patience": 10,       # no early stop inside a 10-epoch stage
            "optimizer": "AdamW",  # what `auto` selects here; named so lr0 is honoured
            "lr0": 0.000167,      # 0.1 x auto's 0.001667 = the LR at epoch 90 of 100
            "lrf": 0.1,           # decays to 1.67e-5, matching the schedule's tail
            "momentum": 0.9,      # auto's choice, restated
        },
        # σ is already trained by this point; re-running its warm-up would zero
        # the NLL again and waste half the stage.
        "gaussian_overrides": {"warmup_epochs": 0, "ramp_epochs": 0},
    }

    def stage(run_id, sigma, data):
        return [
            {"id": run_id, "sigma": sigma, "data": data, "seed": 0},
            {"id": f"{run_id}_ft", "sigma": sigma, "data": data, "seed": 0,
             "from": run_id, **ft},
        ]

    return {
        "created": now(),
        "note": "Phase 2 architecture test on the 640 dataset, VIS at stride 5. "
                "Order is deliberate: each run is immediately followed by its "
                "mosaic-off stage, so stopping the queue early still leaves "
                "complete pairs rather than half-finished ones. The VIS sigma run "
                "goes first (it is the one that can fail); its parity twin is the "
                "§12.1 check and must stay at identical settings.",
        "defaults": defaults,
        "runs": [
            *stage("gauss_vis_seed0", True, vis),
            *stage("parity_vis_seed0", False, vis),
            *stage("gauss_ir_seed0", True, ir),
            *stage("parity_ir_seed0", False, ir),
        ],
    }


def full_scale_queue(cfg: dict) -> dict:
    """The C-1 full-scale training matrix (docs/TODO-2026-08-20-full-scale.md):
    Gaussian sigma (1) + MC-Dropout (1) + Ensemble seed replicates (M) per
    modality, at the frozen D28/D29/A-2 recipe — yolo26m, imgsz 640, 100 epochs,
    patience 20. IR carries the p2feat neck (yolo26m-p2feat, uq/variants.py) on
    every arm, per D29: baselines inherit every detector-side change, or Table 2
    would compare architectures instead of UQ methods. Batch sizes are the
    laptop-measured values from the TODO's A-2 sign-off (VIS 16, IR 10) — not a
    memory ceiling for IR (screened for accuracy), a throughput pick for VIS
    (unvalidated for mAP, Laksh's call 2026-08-20).

    Each arm gets the same mosaic-off fine-tune continuation as the Phase 2
    queue (see `ft` in `default_queue`): patience 20 will almost certainly stop
    every run before Ultralytics' fixed close_mosaic epoch (epochs -
    close_mosaic = 90), so without it every model trains entirely on mosaicked
    frames and is scored on clean ones.
    """
    b = cfg["benchmark"]
    ens_seeds = (cfg.get("baselines") or {}).get("ensemble", {}).get("seeds", [0, 1, 2, 3, 4])

    modalities = {
        "vis": {"variant": "yolo26m", "data": "runs/derived/data_vis_stride2.yaml", "batch": 16},
        "ir": {"variant": "yolo26m-p2feat", "data": "runs/derived/data_ir_shiponly.yaml", "batch": 10},
    }
    workers = 8

    # Same fine-tune shape as default_queue's `ft` (see its comment for the LR
    # derivation) — the tail of the standard 100-epoch schedule with mosaic off.
    ft_overrides = {
        "epochs": 10,
        "train_overrides": {
            "mosaic": 0.0, "close_mosaic": 0, "warmup_epochs": 0.0, "patience": 10,
            "optimizer": "AdamW", "lr0": 0.000167, "lrf": 0.1, "momentum": 0.9,
        },
        "gaussian_overrides": {"warmup_epochs": 0, "ramp_epochs": 0},
    }

    def stage(run_id: str, kind: str, mod: str, seed: int, sigma: bool | None = None) -> list[dict]:
        m = modalities[mod]
        base = {"id": run_id, "kind": kind, "variant": m["variant"], "data": m["data"],
                "batch": m["batch"], "workers": workers, "seed": seed}
        ft = {**ft_overrides, "id": f"{run_id}_ft", "kind": kind, "variant": m["variant"],
              "data": m["data"], "batch": m["batch"], "workers": workers, "seed": seed,
              "from": run_id}
        if sigma is not None:
            base["sigma"], ft["sigma"] = sigma, sigma
        return [base, ft]

    runs: list[dict] = []
    for mod in ("vis", "ir"):
        runs += stage(f"gauss_{mod}_seed0", "gaussian", mod, 0, sigma=True)
    for mod in ("vis", "ir"):
        runs += stage(f"mc_{mod}_seed0", "mc_dropout", mod, 0)
    for mod in ("vis", "ir"):
        for seed in ens_seeds:
            runs += stage(f"ens_{mod}_seed{seed}", "ensemble", mod, seed)

    return {
        "created": now(),
        "note": "C-1 full-scale matrix: Gaussian + MC-Dropout + Ensemble(M) per "
                "modality, yolo26m (IR: yolo26m-p2feat), 100 epochs / patience "
                "20, each arm followed by its mosaic-off fine-tune continuation. "
                "Order: both Gaussian arms first (everything downstream depends "
                "on them), then both MC-Dropout arms, then VIS ensemble seeds, "
                "then IR ensemble seeds.",
        # Distinct from the Phase 2 queue's "phase2" (train_gaussian's default
        # out_subdir): that directory already holds a COMPLETED run named
        # gauss_vis_seed0 (yolo26s/stride5, 2026-08-18) — reusing "phase2" here
        # would make the Gaussian dispatch "resume" that unrelated finished
        # checkpoint instead of training the real yolo26m/stride2 model.
        "out_subdir": "full_scale",
        "defaults": {"imgsz": b["imgsz"], "epochs": b["epochs"], "patience": b["patience"]},
        "runs": runs,
    }


def ir_benchmark_queue(cfg: dict) -> dict:
    """IR-modality benchmark: the full Phase 1 variant ladder x 3 seeds, at
    train_stride=4 (docs/... none yet -- this is a fresh ask, not a pre-registered
    matrix). Mirrors the *original* Phase 1 grid's shape (single-stage, no
    mosaic-off continuation) rather than phase2_queue/full_scale_queue's two-stage
    one, because this is a ranking sweep across architectures (like Table 1), not
    a final training recipe -- and it matches the 2026-08-25 ad hoc pilot
    (queue_ir_m_stride, since discarded) that first proved this data/kind
    combination trains cleanly.

    Reuses the plain-detector path already wired into the queue (kind="gaussian",
    sigma=False -- the D17/§12.1 "parity" trainer, a stock DetectionTrainer) so
    this benchmark gets pause/resume/heartbeat/divergence-alarm for free instead
    of a new code path. batch is hardcoded to the local RTX 4080's measured-safe
    value (5.08 GB reserved for yolo26m @ batch 8 on IR stride 4, 2026-08-25
    pilot) rather than config.yaml's H100-sized benchmark.batch (32) -- this
    queue is laptop-only by design (2026-08-26, Laksh).

    workers=16 (not the repo-wide default of 8): measured via `scripts/tune_batch.py
    --data runs/derived/data_ir_stride4.yaml --variant yolo26m --batches 8
    --workers 8 12 16 24 --plain` (2026-08-26; raw results in
    runs/tune/tune_workers_ir_stride4.json), i.e. the real IR-stride4 data, batch
    8, and the stock DetectionTrainer -- matching this queue's actual code path,
    not the earlier yolo26s/VIS/batch-24/gaussian-trainer sweep already on file
    (runs/tune/tune_workers.json). Measured throughput: 8->42.6 img/s, 12->40.0
    (noise -- single short probe, not re-measured), 16->44.7, 24->44.5. 16 is the
    best measured point and 24 shows no further gain, so 24's extra thread count
    buys nothing; the VRAM cost of going 8->16 is small (driver peak 7.22->7.41
    GB of 12 GB) and the ~5% throughput gain is worth taking for free across 93
    runs.
    """
    b = cfg["benchmark"]
    variants = b["variants"]  # full 31-variant ladder, same as the Phase 1 VIS grid
    seeds = [0, 1, 2]

    runs = [
        {"id": f"ir_bench_{variant}_seed{seed}", "kind": "gaussian", "sigma": False,
         "variant": variant, "data": "runs/derived/data_ir_stride4.yaml", "seed": seed}
        for variant in variants
        for seed in seeds
    ]

    return {
        "created": now(),
        "note": "IR benchmark: 31 variants (config.yaml benchmark.variants) x "
                "seeds {0,1,2} on data_ir_stride4.yaml (train stride 4, val/test "
                "full IR split). Single-stage, no mosaic-off ft continuation -- "
                "a ranking sweep, not a final recipe. Local RTX 4080 only, "
                "sequential. 93 runs total. workers=16 per the 2026-08-26 "
                "tune_batch.py sweep (runs/tune/tune_workers_ir_stride4.json): "
                "8->42.6, 12->40.0 (noise), 16->44.7, 24->44.5 img/s -- 16 is the "
                "best measured point, 24 gains nothing further.",
        "out_subdir": "ir_benchmark_stride4",
        "defaults": {
            "imgsz": b["imgsz"], "epochs": b["epochs"], "patience": b["patience"],
            "batch": 8, "workers": 16,
        },
        "runs": runs,
    }


# Phase 1's own measured parameter counts at imgsz 640, nc=2
# (phase1_benchmark/results.csv, params_m). The queue is ordered by this: the
# biggest models train first, so an interrupted or abandoned sweep still leaves
# the expensive end of the ladder finished rather than the cheap end.
VARIANT_PARAMS_M = {
    "yolov8x": 68.23, "yolo12x": 59.22, "yolo26x": 58.99, "yolov9e": 58.21,
    "yolo11x": 56.97, "yolov8l": 43.69, "yolov10x": 31.81, "yolo12l": 26.45,
    "yolo26l": 26.30, "yolov10l": 25.89, "yolov8m": 25.90, "yolov9c": 25.59,
    "yolo11l": 25.37, "yolo26m": 21.90, "yolov10b": 20.57, "yolov9m": 20.22,
    "yolo12m": 20.20, "yolo11m": 20.11, "yolov10m": 16.58, "yolov8s": 11.17,
    "yolo26s": 10.01, "yolo11s": 9.46, "yolo12s": 9.29, "yolov10s": 8.13,
    "yolov9s": 7.32, "yolov8n": 3.16, "yolov10n": 2.78, "yolo11n": 2.62,
    "yolo12n": 2.60, "yolo26n": 2.57, "yolov9t": 2.13,
}


def vis_benchmark_queue(cfg: dict) -> dict:
    """VIS-modality benchmark: the full 31-variant Phase 1 ladder x 3 seeds at
    train_stride=4, trained on BOTH classes (ship + buoy).

    This is the 2-class Table 1 that docs/TODO-2026-08-26-phase1-classset.md §4.2
    describes, widened from that memo's 9 `main` variants to the whole 31-variant
    ladder and moved from stride 2 to stride 4 (Laksh, 2026-08-26). It is a
    *cold restart from COCO*, not a fine-tune of the ship-only Phase 1
    checkpoints — §2 of that memo rules the warm-start route out, and this queue
    is what replaces it.

    Two things separate it from Phase 1's `results.csv` rows, and both must be
    stated wherever a number from it is published:

    - **Class set.** Phase 1 passed `--classes 0` through `bench/grid.py`, so all
      93 of its rows are ship-only. This path (`train_gaussian`, sigma=False)
      passes no `classes` filter at all, so it trains and scores on the yaml's
      full `names` map -- 0 ship, 1 buoy. Macro-averaged mAP therefore is NOT
      comparable to a Phase 1 row; per-class AP is.
    - **Train stride.** Phase 1 trained on `data_vis_stride2.yaml`; this trains on
      stride 4, i.e. half the frames per epoch.

    Order is by parameter count, descending, seeds inner (VARIANT_PARAMS_M): 93
    runs is weeks of GPU time, so the ordering decides what exists if it is ever
    cut short, and it keeps each variant's three seeds together so a variant that
    finishes has error bars rather than a lone point.

    Sizing is for the dgxanode01 A100 MIG 3g.40gb slice: `batch=16` is Phase 1's
    own largest-common-fit on a 40 GB slice (its record has yolo12x completing at
    16 and failing at 24/32), and `workers=6` is the /dev/shm ceiling measured on
    that box (workers 8 dies with "insufficient shared memory" partway into epoch
    1; 2/4/6 measure 1.8/1.9/1.9 it/s, so 6 costs nothing).
    """
    b = cfg["benchmark"]
    variants = list(b["variants"])
    unknown = set(variants) - set(VARIANT_PARAMS_M)
    if unknown:
        # The ordering is the point of this queue; silently appending variants of
        # unknown size would break it in exactly the way nobody would notice.
        raise ValueError(f"no measured params for {sorted(unknown)} — add them to "
                         "VARIANT_PARAMS_M (phase1_benchmark/results.csv) before "
                         "building this queue")
    ordered = sorted(variants, key=lambda v: (-VARIANT_PARAMS_M[v], v))
    seeds = [0, 1, 2]

    runs = [
        {"id": f"vis_bench_{variant}_seed{seed}", "kind": "gaussian", "sigma": False,
         "variant": variant, "data": "runs/derived/data_vis_stride4.yaml", "seed": seed}
        for variant in ordered
        for seed in seeds
    ]

    return {
        "created": now(),
        "note": "VIS 2-class benchmark: 31 variants (config.yaml benchmark.variants) "
                "x seeds {0,1,2} on data_vis_stride4.yaml (train stride 4, val/test "
                "the full VIS split), cold-started from COCO weights. BOTH classes "
                "(ship + buoy) -- unlike Phase 1's results.csv, which is ship-only "
                "(classes=0) at stride 2, so macro mAP from the two is not "
                "comparable; report per-class AP. Single-stage, no mosaic-off ft "
                "continuation -- a ranking sweep, not a final recipe. Ordered by "
                "parameter count, largest first, seeds inner. 93 runs, sized for "
                "the dgxanode01 A100 MIG 3g.40gb slice (batch 16 = Phase 1's "
                "largest-common-fit on a 40 GB slice; workers 6 = the /dev/shm "
                "ceiling measured on that box).",
        # Epoch count is part of the output path, not just the queue.json. Run ids are
        # f"vis_bench_{variant}_seed{seed}" and the runner resumes from an existing run
        # dir, so two campaigns at different epoch budgets sharing one out_subdir do not
        # collide loudly -- the second one RESUMES the first. The 2026-09-03 re-scope
        # from 100 to 25 epochs would have silently continued the 12 runs already on
        # dgxanode01 instead of starting fresh, and their 100-epoch results would have
        # been the thing that moved. Bake the budget into the path so that cannot happen.
        "out_subdir": f"vis_benchmark_stride4_ep{b['epochs']}",
        "defaults": {
            "imgsz": b["imgsz"], "epochs": b["epochs"], "patience": b["patience"],
            "batch": 16, "workers": 6,
        },
        "runs": runs,
    }


def cmd_init(args) -> int:
    cfg = load_config(args.config)
    if QUEUE_JSON.is_file() and not args.force:
        print(f"{QUEUE_JSON} exists — pass --force to overwrite (state.json is kept).")
        return 1
    if args.matrix == "full_scale":
        q = full_scale_queue(cfg)
    elif args.matrix == "ir_benchmark":
        q = ir_benchmark_queue(cfg)
    elif args.matrix == "vis_benchmark":
        q = vis_benchmark_queue(cfg)
    else:
        q = default_queue(cfg, args.variant, args.batch, args.workers)
    write_json(QUEUE_JSON, q)
    write_json(CONTROL_JSON, {"paused": False, "updated": now()})
    d = q["defaults"]
    print(f"wrote {QUEUE_JSON}  (matrix: {args.matrix})")
    print(f"  imgsz {d.get('imgsz')} | epochs {d.get('epochs')} | patience {d.get('patience')}")
    if "_sizing_source" in d:
        print(f"  batch/workers from {d['_sizing_source']}")
    for r in q["runs"]:
        kind = r.get("kind", "gaussian")
        tag = f"sigma={r['sigma']}" if "sigma" in r else kind
        print(f"  - {r['id']:<20} {tag:<14} variant={r.get('variant', d.get('variant')):<16} "
              f"batch={r.get('batch', d.get('batch')):<4} {r.get('data', d.get('data', ''))}")
    return 0


# ----------------------------------------------------------------- the run loop


def shielded(fn):
    """Run a callback for its side effects only — never let it reach the trainer.

    These callbacks exist to report on training, so a fault in one must degrade
    the reporting, not the run. PauseRequested is the single deliberate exception
    and is allowed through; everything else is logged once and swallowed.
    """
    def wrapper(trainer):
        try:
            return fn(trainer)
        except PauseRequested:
            raise
        except Exception as exc:  # noqa: BLE001 - see docstring
            print(f"[queue] warning: {fn.__name__} failed "
                  f"({type(exc).__name__}: {exc}) — training continues", flush=True)
    wrapper.__name__ = fn.__name__
    return wrapper


def check_divergence(trainer, run_id: str, rs: dict, alarm: dict) -> None:
    """End the run the first epoch it shows the 2026-08-23 divergence signature.

    Driven off ``results.csv``, not ``trainer.metrics``, for two reasons rooted in
    Ultralytics' epoch order (``engine/trainer.py``):

    - ``save_metrics`` writes the row at :557, *before* ``on_fit_epoch_end`` runs
      at :576, so the current epoch is already on disk when we look.
    - ``on_fit_epoch_end`` fires a **second** time from ``final_eval`` at :903,
      with ``trainer.epoch`` temporarily incremented and no new row written.
      Keying off unseen rows makes that call a no-op instead of a phantom epoch.

    ``trainer.stop`` is Ultralytics' own early-stop switch, tested at :585 — i.e.
    after ``save_model``/``on_model_save`` at :563. Setting it here ends the run at
    the end of *this* epoch with ``last.pt`` already complete, so nothing is lost.

    The queue is NOT paused. Stopping only this run used to look like it would
    let a dependent ``_ft`` stage start straight off a diverged parent, but that
    risk is already covered independently: ``cmd_run``'s ``from=`` parent guard
    (around line 690) refuses to start any stage whose parent status is not
    "done", diverged included, and fails that stage loudly rather than silently
    training from COCO weights. Pausing the whole queue on top of that guard was
    redundant, and it meant a single noisy epoch on an unrelated, independent run
    (2026-08-26: ``ir_bench_yolov8n_seed0`` of the 93-run, single-stage IR
    benchmark queue, no ``_ft`` stages at all) halted 92 other runs overnight
    waiting on a human. Now the run is marked ``diverged`` and the queue moves on
    to the next index exactly as it would after ``done`` or ``failed``.

    On the first call after a resume the whole existing history is judged, not
    just new epochs. A run being silently continued past a divergence (trap 3 in
    the 08-23 handoff §6) is exactly the case worth catching late.

    Note this runs under `shielded`, so a fault here degrades the alarm rather
    than the run. That is the right trade for a training process, and it is also
    why `scripts/watch_divergence.py` stays deployed as an independent backstop.
    """
    if alarm["fired"]:
        return
    csv_path = getattr(trainer, "csv", None) or Path(trainer.save_dir) / "results.csv"
    rows = read_results_csv(Path(csv_path))
    if len(rows) <= alarm["seen"]:
        return  # final_eval's second call, or a torn read — nothing new to judge
    start, alarm["seen"] = alarm["seen"], len(rows)

    for i in range(start, len(rows)):
        hits = divergence_check(rows, i, alarm["limits"])
        if not hits:
            continue
        alarm["fired"] = True
        epoch = rows[i]["epoch"]
        log(f"!!! {run_id}: DIVERGENCE ALARM at epoch {epoch}")
        for h in hits:
            log(f"!!!   {h}")
        log(f"!!! {run_id}: stopping this run. The queue continues to the next "
            f"run. This checkpoint cannot be resumed (Ultralytics strips "
            f"epoch/optimizer state on trainer.stop) — rerunning it by hand "
            f"(`run_queue.py --queue-dir {QUEUE_DIR} run --only {run_id} --redo`) "
            f"restarts it from epoch 0, it does not continue from epoch {epoch}.")
        rs["divergence_alarm"] = {"epoch": epoch, "at": now(), "reasons": hits}
        try:
            (Path(trainer.save_dir) / "DIVERGENCE-ALARM.txt").write_text(
                f"{now()}\nDIVERGENCE ALARM at epoch {epoch}\n"
                + "\n".join(f"  - {h}" for h in hits) + "\n", encoding="utf-8")
        except OSError:
            pass  # the state entry and the log already carry it
        trainer.stop = True
        return


def make_callbacks(run_id: str, state: dict, spec: dict) -> dict[str, list]:
    """Heartbeat + pause callbacks. Deliberately cheap: they run per batch."""
    tick = {"last": 0.0, "i": 0, "n": 0, "t_epoch": time.time()}
    alarm = {"seen": 0, "fired": False, "limits": divergence_thresholds()}

    def on_pretrain_routine_end(trainer):
        # Ultralytics builds a brand-new EarlyStopping() and only THEN calls
        # resume_training(ckpt) (engine/trainer.py _setup_train), so a resumed
        # run's stopper.best_fitness/best_epoch reset to 0 even though
        # trainer.best_fitness itself is correctly restored from the checkpoint.
        # Left alone, patience counts from the resume point instead of the
        # run's true best epoch, and state.json's best_epoch/best_fitness
        # silently jump forward to whatever epoch comes next after resume.
        stopper = getattr(trainer, "stopper", None)
        start_epoch = getattr(trainer, "start_epoch", 0)
        if stopper is None or start_epoch <= 0:
            return  # fresh run, nothing to carry forward
        rs = run_state(state, run_id)
        prev_epoch = rs.get("best_epoch")
        if prev_epoch is None:
            log(f"=== {run_id}: resumed but no prior best_epoch in state.json — "
                f"stopper left at its post-resume default, patience will count "
                f"from epoch {start_epoch}")
            return
        stopper.best_fitness = float(trainer.best_fitness)  # authoritative: from ckpt
        stopper.best_epoch = int(prev_epoch)  # not in the ckpt; carried from state.json
        log(f"=== {run_id}: resumed — restored stopper to best_epoch {prev_epoch}, "
            f"best_fitness {stopper.best_fitness:.5f}")

    def on_epoch_start(trainer):
        tick["i"], tick["n"] = 0, len(trainer.train_loader)
        tick["t_epoch"] = time.time()

    def on_batch_end(trainer):
        tick["i"] += 1
        t = time.time()
        if t - tick["last"] < HEARTBEAT_S:
            return
        tick["last"] = t
        import torch

        elapsed = t - tick["t_epoch"]
        it_s = tick["i"] / elapsed if elapsed > 0 else 0.0
        write_json(LIVE_JSON, {
            "updated": now(),
            "run_id": run_id,
            "epoch": int(trainer.epoch) + 1,
            "epochs": int(trainer.epochs),
            "batch_i": tick["i"],
            "batch_n": tick["n"],
            "it_s": round(it_s, 2),
            "img_s": round(it_s * int(spec["batch"]), 1),
            "epoch_eta_s": round((tick["n"] - tick["i"]) / it_s) if it_s > 0 else None,
            "gpu_reserved_gb": round(torch.cuda.memory_reserved() / 2**30, 2)
            if torch.cuda.is_available() else None,
            "pause_pending": bool(control().get("paused")),
        })

    def on_fit_epoch_end(trainer):
        rs = run_state(state, run_id)
        rs["epochs_done"] = int(trainer.epoch) + 1
        rs["epoch_time_s"] = round(getattr(trainer, "epoch_time", 0.0), 1)
        stopper = getattr(trainer, "stopper", None)
        is_best_epoch = False
        if stopper is not None:
            rs["best_fitness"] = round(float(stopper.best_fitness), 5)
            rs["best_epoch"] = int(stopper.best_epoch)
            rs["patience_gap"] = int(trainer.epoch) + 1 - int(stopper.best_epoch)
            # stopper.best_epoch is 1-based: trainer.py calls stopper(epoch + 1, ...).
            # trainer.epoch is 0-based, so the current epoch's 1-based index is +1.
            # Comparing the two directly matched one epoch LATE, which recorded the
            # epoch AFTER best.pt into best_map50_95 for every run up to 2026-08-24.
            is_best_epoch = int(trainer.epoch) + 1 == int(stopper.best_epoch)
        metrics = getattr(trainer, "metrics", None) or {}
        rs["map50_95"] = round(float(metrics.get("metrics/mAP50-95(B)", 0.0)), 5)
        rs["map50"] = round(float(metrics.get("metrics/mAP50(B)", 0.0)), 5)
        # The checkpoint's early-stop criterion is Ultralytics' composite fitness
        # (~0.9*mAP50-95 + 0.1*mAP50), not raw mAP50-95, so the epoch with the
        # highest mAP50-95 is not always the epoch saved as best.pt. Record the
        # mAP50-95 of whichever epoch IS best.pt right now (stopper.best_epoch),
        # not a separately-tracked max, so the dashboard number matches the
        # checkpoint that actually exists on disk.
        if is_best_epoch or "best_map50_95" not in rs:
            rs["best_map50_95"] = rs["map50_95"]
        check_divergence(trainer, run_id, rs, alarm)
        save_state(state)

    def on_model_save(trainer):
        # The one safe pause point: last.pt has just been written in full.
        if control().get("paused"):
            raise PauseRequested(f"paused after epoch {int(trainer.epoch) + 1}")

    return {
        "on_pretrain_routine_end": [shielded(on_pretrain_routine_end)],
        "on_train_epoch_start": [shielded(on_epoch_start)],
        "on_train_batch_end": [shielded(on_batch_end)],
        "on_fit_epoch_end": [shielded(on_fit_epoch_end)],
        "on_model_save": [shielded(on_model_save)],
    }


def wait_while_paused(state: dict) -> bool:
    """Block until unpaused. Returns False if the process was interrupted."""
    if not control().get("paused"):
        return True
    log("paused — waiting for resume (dashboard button, or `run_queue.py resume`)")
    state["queue_status"] = "paused"
    save_state(state)
    try:
        while control().get("paused"):
            time.sleep(2)
    except KeyboardInterrupt:
        return False
    log("resumed")
    state["queue_status"] = "running"
    save_state(state)
    return True


def cmd_run(args) -> int:
    cfg = load_config(args.config)
    queue = read_json(QUEUE_JSON, None)
    if queue is None:
        print(f"no queue at {QUEUE_JSON} — run `python scripts/run_queue.py init` first.")
        return 1

    from uqfusion.uq.ensemble import train_ensemble_member
    from uqfusion.uq.mc_dropout import train_mc_dropout
    from uqfusion.uq.train_gaussian import train_gaussian

    TRAINERS = {"gaussian": train_gaussian, "mc_dropout": train_mc_dropout,
                "ensemble": train_ensemble_member}

    defaults = queue.get("defaults", {})
    state = load_state()
    state["queue_status"] = "running"
    state["pid"] = os.getpid()
    state["queue_file"] = str(QUEUE_JSON)
    state["defaults"] = defaults
    save_state(state)

    specs = [{**defaults, **r} for r in queue["runs"]]
    if args.only:
        specs = [s for s in specs if s["id"] in set(args.only)]

    log(f"queue: {len(specs)} run(s) | epochs {defaults.get('epochs')} "
        f"patience {defaults.get('patience')}")

    interrupted = False
    i = 0
    while i < len(specs):
        # Index, not iteration: a paused run must be retried at the SAME position
        # when it resumes, not pushed behind the rest of the queue.
        spec = specs[i]
        run_id = spec["id"]
        rs = run_state(state, run_id)
        if rs.get("status") in TERMINAL and not args.redo:
            log(f"skip {run_id} — already {rs['status']}")
            i += 1
            continue
        if run_id in set(control().get("skip") or []):
            rs["status"] = "skipped"
            save_state(state)
            log(f"skip {run_id} — marked skipped in control.json")
            i += 1
            continue
        if not wait_while_paused(state):
            interrupted = True
            break

        # A stage that continues another run needs that run's best.pt. If the
        # parent failed or was skipped, fail loudly — silently training this from
        # COCO weights would produce a plausible model that is not the experiment.
        start_weights = None
        parent = spec.get("from")
        if parent:
            prs = state.get("runs", {}).get(parent, {})
            candidate = prs.get("best_weights")
            if prs.get("status") != "done" or not candidate or not Path(candidate).is_file():
                rs.update(status="failed", finished=now(),
                          error=f"parent run '{parent}' is {prs.get('status', 'missing')} "
                                f"with no usable best.pt — nothing to continue from")
                save_state(state)
                log(f"=== {run_id}: SKIPPED, parent '{parent}' did not produce weights")
                i += 1
                continue
            start_weights = candidate

        kind = spec.get("kind", "gaussian")
        trainer = TRAINERS.get(kind)
        if trainer is None:
            rs.update(status="failed", finished=now(), error=f"unknown run kind: {kind!r}")
            save_state(state)
            log(f"=== {run_id}: FAILED unknown kind {kind!r}")
            i += 1
            continue

        rs.update(status="running", started=rs.get("started") or now(), error=None)
        save_state(state)
        log(f"=== {run_id}: start (kind={kind}, sigma={spec.get('sigma')}, data={spec['data']}"
            + (f", from={parent}" if parent else "") + ")")

        common = dict(
            data_yaml=resolve_data_yaml(cfg, spec["data"]),
            variant=spec["variant"],
            seed=int(spec["seed"]),
            epochs=int(spec["epochs"]),
            imgsz=int(spec["imgsz"]),
            batch=int(spec["batch"]),
            workers=int(spec["workers"]),
            run_name=run_id,
            callbacks=make_callbacks(run_id, state, spec),
            weights=start_weights,
            train_overrides=spec.get("train_overrides"),
        )
        try:
            if kind == "gaussian":
                best, run_dir = train_gaussian(
                    cfg, sigma=bool(spec.get("sigma", True)),
                    out_subdir=queue.get("out_subdir", "phase2"), resume=True,
                    gaussian_overrides=spec.get("gaussian_overrides"), **common,
                )
            else:
                best, run_dir = trainer(cfg, **common)
        except PauseRequested as exc:
            rs.update(status="paused", note=str(exc))
            save_state(state)
            log(f"=== {run_id}: {exc} — checkpoint kept, will resume where it stopped")
            if not wait_while_paused(state):
                interrupted = True
                break
            continue  # same index: re-enter this run, which resumes from last.pt
        except KeyboardInterrupt:
            rs.update(status="interrupted")
            save_state(state)
            log(f"=== {run_id}: interrupted by Ctrl-C — rerun `run` to resume from last.pt")
            interrupted = True
            break
        except Exception as exc:  # noqa: BLE001 - one bad run must not sink the queue
            rs.update(status="failed", error=f"{type(exc).__name__}: {exc}"[:400],
                      finished=now())
            save_state(state)
            log(f"=== {run_id}: FAILED {type(exc).__name__}: {exc}")
            traceback.print_exc()
            if args.stop_on_fail:
                interrupted = True
                break
            i += 1
            continue

        # A run the alarm stopped returns normally — Ultralytics' own stop path is
        # a clean exit, not an exception — so the status has to be corrected here
        # or a diverged run would be recorded as "done" and fed to its `_ft` stage.
        if rs.get("divergence_alarm"):
            rs.update(status="diverged", finished=now(), run_dir=str(run_dir),
                      best_weights=str(best))
            save_state(state)
            log(f"=== {run_id}: DIVERGED at epoch "
                f"{rs['divergence_alarm']['epoch']} -> {run_dir} (queue continues)")
        else:
            rs.update(status="done", finished=now(), run_dir=str(run_dir),
                      best_weights=str(best))
            save_state(state)
            log(f"=== {run_id}: done -> {best}")
        i += 1

    state["queue_status"] = "interrupted" if interrupted else "idle"
    save_state(state)
    LIVE_JSON.unlink(missing_ok=True)
    remaining = [s["id"] for s in specs
                 if run_state(state, s["id"]).get("status") not in TERMINAL]
    log(f"queue finished ({'interrupted' if interrupted else 'all runs terminal'}); "
        f"remaining: {remaining or 'none'}")
    return 0


# ---------------------------------------------------------------------- status


def cmd_status(args) -> int:
    queue = read_json(QUEUE_JSON, None)
    if queue is None:
        print(f"no queue at {QUEUE_JSON}")
        return 1
    state, live, ctl = load_state(), read_json(LIVE_JSON, {}), control()
    print(f"queue  : {state.get('queue_status', 'never started')}"
          f"{'  [PAUSE REQUESTED]' if ctl.get('paused') else ''}")
    print(f"updated: {state.get('updated', '-')}")
    print(f"{'run':<22}{'status':<12}{'epochs':<9}{'best ep':<9}{'mAP50-95':<10}")
    for r in queue["runs"]:
        rs = state.get("runs", {}).get(r["id"], {})
        print(f"{r['id']:<22}{rs.get('status', 'pending'):<12}"
              f"{str(rs.get('epochs_done', '-')):<9}{str(rs.get('best_epoch', '-')):<9}"
              f"{str(rs.get('map50_95', '-')):<10}")
    if live:
        print(f"\nlive: {live.get('run_id')} epoch {live.get('epoch')}/{live.get('epochs')} "
              f"batch {live.get('batch_i')}/{live.get('batch_n')} "
              f"{live.get('img_s')} img/s  gpu {live.get('gpu_reserved_gb')} GB")
    return 0


def cmd_pause(args) -> int:
    set_control(paused=True)
    print("pause requested — the current run stops after its next epoch checkpoint.")
    return 0


def cmd_resume(args) -> int:
    set_control(paused=False)
    print("resume requested. If the runner process exited, start it again: "
          "python scripts/run_queue.py run")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--queue-dir", default=None,
                        help="relocate queue.json/state.json/control.json (default runs/queue)")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_init = sub.add_parser("init", help="write runs/queue/queue.json")
    p_init.add_argument("--matrix",
                        choices=["phase2", "full_scale", "ir_benchmark", "vis_benchmark"],
                        default="phase2",
                        help="phase2 = the laptop architecture test (default); "
                             "full_scale = the C-1 matrix (Gaussian+MC-Dropout+Ensemble "
                             "x VIS/IR, docs/TODO-2026-08-20-full-scale.md); "
                             "ir_benchmark = 31-variant x3-seed IR benchmark @ train_stride=4; "
                             "vis_benchmark = the same ladder on VIS, BOTH classes, "
                             "largest model first (docs/TODO-2026-08-26-phase1-classset.md)")
    p_init.add_argument("--variant", default="yolo26s", help="phase2 matrix only")
    p_init.add_argument("--batch", type=int, default=None, help="phase2 matrix only")
    p_init.add_argument("--workers", type=int, default=None, help="phase2 matrix only")
    p_init.add_argument("--force", action="store_true")
    p_init.set_defaults(func=cmd_init)

    p_run = sub.add_parser("run", help="work the queue")
    p_run.add_argument("--only", nargs="+", default=None, help="run only these ids")
    p_run.add_argument("--redo", action="store_true", help="rerun ids already marked done")
    p_run.add_argument("--stop-on-fail", action="store_true")
    p_run.set_defaults(func=cmd_run)

    sub.add_parser("status", help="one-shot status").set_defaults(func=cmd_status)
    sub.add_parser("pause", help="ask the queue to pause").set_defaults(func=cmd_pause)
    sub.add_parser("resume", help="clear the pause flag").set_defaults(func=cmd_resume)

    args = parser.parse_args()
    if args.queue_dir:
        set_queue_dir(args.queue_dir)
    signal.signal(signal.SIGINT, signal.default_int_handler)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
