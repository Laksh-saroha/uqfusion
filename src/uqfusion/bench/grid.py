"""Multi-seed benchmark grid (plan C5 -> Table 1, scope §9.1/§9.5).

One row per (variant, seed): train with the shared config, validate on the
common val split, append metrics to a CSV. Resume-safe: rows already in the
CSV are skipped, so a crashed grid restarts where it stopped — an 18-run grid
on a shared server needs that more than elegance.
"""

from __future__ import annotations

import csv
import subprocess
import time
from pathlib import Path

RESULT_FIELDS = [
    "variant", "seed", "precision", "recall", "map50", "map50_95",
    "params_m", "gflops", "epochs_cfg", "train_time_s", "run_dir",
    "ultralytics_version", "torch_version", "git_commit",
]


def resolve_device(cfg: dict):
    d = cfg.get("device", "auto")
    return None if d in (None, "auto") else d


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parents[3],
        ).strip()
    except Exception:  # noqa: BLE001 - commit id is metadata, never fatal
        return "unknown"


def _completed(csv_path: Path) -> set[tuple[str, str]]:
    if not csv_path.is_file():
        return set()
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        return {(row["variant"], row["seed"]) for row in csv.DictReader(f)}


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file()
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _box_metrics(metrics) -> dict[str, float]:
    """Extract P/R/mAP from an Ultralytics DetMetrics, tolerating API drift."""
    try:
        return {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
    except AttributeError:
        rd = metrics.results_dict
        return {
            "precision": float(rd["metrics/precision(B)"]),
            "recall": float(rd["metrics/recall(B)"]),
            "map50": float(rd["metrics/mAP50(B)"]),
            "map50_95": float(rd["metrics/mAP50-95(B)"]),
        }


def run_grid(
    cfg: dict,
    data_yaml: str | Path,
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    out_csv: str | Path | None = None,
    run_prefix: str = "bench",
) -> Path:
    """Run the grid; return the results CSV path. Every argument defaults to config.yaml."""
    import torch
    import ultralytics
    from ultralytics import YOLO

    b = cfg["benchmark"]
    variants = variants if variants is not None else b["variants"]
    seeds = seeds if seeds is not None else b["seeds"]
    epochs = epochs if epochs is not None else b["epochs"]
    imgsz = imgsz if imgsz is not None else b["imgsz"]
    batch = batch if batch is not None else b["batch"]
    workers = workers if workers is not None else b["workers"]

    out_root = Path(cfg["paths"]["outputs_root"]) / "benchmark"
    out_csv = Path(out_csv) if out_csv else out_root / "benchmark_results.csv"
    device = resolve_device(cfg)
    commit = _git_commit()
    done = _completed(out_csv)

    for variant in variants:
        for seed in seeds:
            if (variant, str(seed)) in done:
                print(f"[grid] skip {variant} seed {seed} — already in {out_csv.name}")
                continue
            name = f"{run_prefix}_{variant}_seed{seed}"
            print(f"[grid] === {name}: {epochs} epochs, imgsz {imgsz}, batch {batch}, data {data_yaml}")
            weights = f"{variant}.pt" if b.get("pretrained", True) else f"{variant}.yaml"
            model = YOLO(weights)
            params = sum(p.numel() for p in model.model.parameters())
            try:
                from ultralytics.utils.torch_utils import get_flops

                gflops = float(get_flops(model.model, imgsz))
            except Exception:  # noqa: BLE001 - profiling is metadata, never fatal
                gflops = float("nan")

            t0 = time.time()
            model.train(
                data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                seed=seed, deterministic=b["deterministic"], optimizer=b.get("optimizer", "auto"),
                patience=b["patience"], amp=b["amp"], workers=workers, device=device,
                project=str(out_root / "runs"), name=name, exist_ok=True, verbose=True,
            )
            train_time = time.time() - t0

            # Validate the best checkpoint explicitly so the reported numbers are
            # unambiguous (not "whatever epoch the trainer last printed").
            best = getattr(getattr(model, "trainer", None), "best", None)
            eval_model = YOLO(str(best)) if best and Path(str(best)).is_file() else model
            metrics = eval_model.val(
                data=str(data_yaml), imgsz=imgsz, device=device, split="val",
                project=str(out_root / "runs"), name=f"{name}_val", exist_ok=True,
            )

            row = {
                "variant": variant, "seed": seed,
                **_box_metrics(metrics),
                "params_m": round(params / 1e6, 2),
                "gflops": round(gflops, 1) if gflops == gflops else "",
                "epochs_cfg": epochs, "train_time_s": round(train_time, 1),
                "run_dir": str(out_root / "runs" / name),
                "ultralytics_version": ultralytics.__version__,
                "torch_version": torch.__version__, "git_commit": commit,
            }
            _append_row(out_csv, row)
            print(f"[grid] {name} done in {train_time / 60:.1f} min -> {out_csv}")
    return out_csv
