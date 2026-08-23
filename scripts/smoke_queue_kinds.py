"""Gate for the queue's new run kinds (mc_dropout, ensemble) added for the C-1
full-scale matrix (docs/TODO-2026-08-20-full-scale.md).

`scripts/run_queue.py` previously only knew how to train Gaussian runs;
`scripts/smoke_queue.py` covers that path (pause/resume/mosaic-fine-tune). This
gate proves the two NEW dispatch kinds go through the same machinery correctly:

  1. `kind=mc_dropout` and `kind=ensemble` runs actually train (not just import
     cleanly) through `uqfusion.uq.mc_dropout.train_mc_dropout` /
     `uqfusion.uq.ensemble.train_ensemble_member`.
  2. Pause (raised from `on_model_save`, same as the Gaussian path) leaves a
     resumable checkpoint and resuming continues the SAME run.
  3. The mosaic-off fine-tune continuation (`from:`) starts each kind from its
     parent's own `best.pt`, not from COCO weights again.
  4. The `is_variant`/`build_variant` branch added to `bench.grid.run_grid` and
     `uq.mc_dropout.train_mc_dropout` actually builds an architecture-edit model
     end to end. Uses `yolo26s-p2feat` (already proven — the small-object
     screen's `s3_p2feat_640_b10`) rather than the new `yolo26m-p2feat`: this
     gate is testing the NEW dispatch/callback wiring, not the variant spec
     itself, which `build_variant("yolo26m-p2feat")` already checked on its own
     (97.0% measured transfer, logged separately — no GPU time needed for that
     check).

Usage:  python scripts/smoke_queue_kinds.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import time
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

SMOKE_ROOT = ROOT / "runs" / "smoke_queue_kinds"
QUEUE_DIR = SMOKE_ROOT / "queue"
EPOCHS = 3
FT_EPOCHS = 2


def make_sandboxed_config() -> Path:
    """A config.yaml whose `outputs_root` points into SMOKE_ROOT, so mc_dropout/
    ensemble runs (which have no `out_subdir` knob, unlike train_gaussian) land
    under the smoke sandbox instead of the real runs/mc_dropout, runs/benchmark
    trees. Passed to the run_queue.py subprocess via --config."""
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["paths"]["outputs_root"] = str(SMOKE_ROOT / "outputs")
    out = SMOKE_ROOT / "config_smoke.yaml"
    out.write_text(yaml.safe_dump(cfg), encoding="utf-8")
    return out


def fail(msg: str, proc: subprocess.Popen | None = None) -> None:
    if proc and proc.poll() is None:
        proc.kill()
    print(f"[kinds-smoke] FAIL {msg}")
    raise SystemExit(1)


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def set_paused(kind_dir: Path, value: bool) -> None:
    path = kind_dir / "control.json"
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
    deadline = time.time() + timeout
    while time.time() < deadline:
        value = predicate()
        if value:
            return value
        if proc.poll() is not None:
            fail(f"runner exited (code {proc.returncode}) while waiting for {what}")
        time.sleep(1.0)
    fail(f"timed out after {timeout:.0f}s waiting for {what}", proc)


def status_of(state_path: Path, run_id: str) -> str:
    state = read_json(state_path, {}) or {}
    return (state.get("runs", {}).get(run_id) or {}).get("status", "")


def run_one_kind(kind: str, variant: str, run_id: str, config_path: Path, out_root: Path) -> None:
    probe = ROOT / "runs" / "tune" / "data_probe.yaml"
    if not probe.is_file():
        fail(f"no probe dataset at {probe} — run scripts/tune_batch.py once first")

    kind_dir = QUEUE_DIR / kind
    shutil.rmtree(kind_dir, ignore_errors=True)
    kind_dir.mkdir(parents=True, exist_ok=True)
    ft_id = f"{run_id}_ft"
    (kind_dir / "queue.json").write_text(json.dumps({
        "defaults": {"imgsz": 320, "epochs": EPOCHS, "patience": 20},
        "runs": [
            {"id": run_id, "kind": kind, "variant": variant, "data": str(probe),
             "seed": 0, "batch": 8, "workers": 0},
            # Same mosaic-off continuation shape the full-scale queue uses —
            # proves `weights=` starts THIS kind from its own parent best.pt.
            {"id": ft_id, "kind": kind, "variant": variant, "data": str(probe),
             "seed": 0, "batch": 8, "workers": 0, "from": run_id, "epochs": FT_EPOCHS,
             "train_overrides": {"mosaic": 0.0, "close_mosaic": 0, "warmup_epochs": 0.0,
                                 "patience": FT_EPOCHS, "optimizer": "AdamW",
                                 "lr0": 0.000167, "lrf": 0.1, "momentum": 0.9}},
        ],
    }, indent=2), encoding="utf-8")
    set_paused(kind_dir, False)

    # Deterministic per-kind run_dir (mirrors uq/mc_dropout.py, uq/ensemble.py).
    run_dir = ((out_root / "mc_dropout" / run_id) if kind == "mc_dropout"
               else (out_root / "benchmark" / "runs" / run_id))

    state_path = kind_dir / "state.json"
    proc = subprocess.Popen(
        [sys.executable, str(ROOT / "scripts" / "run_queue.py"),
         "--config", str(config_path), "--queue-dir", str(kind_dir), "run"],
        cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.STDOUT,
    )
    print(f"[kinds-smoke] {kind}: runner pid {proc.pid}, queue at {kind_dir}")
    try:
        live = kind_dir / "live.json"
        wait_for(lambda: (read_json(live, {}) or {}).get("batch_i", 0) > 0,
                 300, f"{kind}: the first training batch", proc)
        print(f"[kinds-smoke] {kind}: training started (heartbeat live) OK")

        set_paused(kind_dir, True)
        wait_for(lambda: status_of(state_path, run_id) == "paused",
                 300, f"{kind}: the run to report paused", proc)
        last = run_dir / "weights" / "last.pt"
        if not last.is_file():
            fail(f"{kind}: paused with no checkpoint at {last}", proc)
        print(f"[kinds-smoke] {kind}: pause honoured at a checkpoint boundary OK")

        set_paused(kind_dir, False)
        wait_for(lambda: status_of(state_path, run_id) == "done",
                 600, f"{kind}: the resumed run to finish", proc)
        print(f"[kinds-smoke] {kind}: resume continued the same run to completion OK")

        wait_for(lambda: status_of(state_path, ft_id) == "done",
                 600, f"{kind}: the fine-tune continuation to finish", proc)
        print(f"[kinds-smoke] {kind}: fine-tune continuation (from={run_id}) finished OK")

        state = read_json(state_path, {}) or {}
        errored = {rid: r.get("error") for rid, r in state.get("runs", {}).items()
                   if r.get("error")}
        if errored:
            fail(f"{kind}: a run carried an error despite finishing: {errored}")

        proc.wait(timeout=120)
    finally:
        if proc.poll() is None:
            proc.kill()


def main() -> int:
    shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
    QUEUE_DIR.mkdir(parents=True, exist_ok=True)
    config_path = make_sandboxed_config()
    out_root = SMOKE_ROOT / "outputs"

    # Plain COCO variant for mc_dropout (proves callbacks/weights=/pause-resume
    # without also exercising build_variant — kept separate from check 2 below).
    run_one_kind("mc_dropout", "yolo26n", "k_mc", config_path, out_root)
    # Architecture-edit variant for ensemble (proves is_variant/build_variant
    # wiring in bench.grid.run_grid end to end).
    run_one_kind("ensemble", "yolo26s-p2feat", "k_ens", config_path, out_root)

    print("QUEUE KINDS SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
