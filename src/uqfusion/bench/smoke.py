"""End-to-end CPU smoke test for the Phase 1 harness (no real data required).

Fabricates a tiny YOLO dataset — dark noise images with bright rectangles as
'ship' (class 0) and dimmer squares as 'buoy' (class 1) so training has actual
signal — then exercises the full chain: run_grid -> val -> results CSV -> FPS
measurement -> Table 1 markdown. Asserts artifacts exist and metrics parse.

This verifies harness *plumbing*, not model quality: 2 epochs on 32 synthetic
images proves the loop runs, not that anything detects ships.

Usage:  python scripts/smoke_benchmark.py
"""

from __future__ import annotations

import shutil
from pathlib import Path

import numpy as np
import yaml

from uqfusion.bench.fps import measure_fps, write_fps_csv
from uqfusion.bench.grid import run_grid
from uqfusion.bench.table import load_results, make_table1


def make_fake_dataset(root: Path, n_train: int, n_val: int, imgsz: int, seed: int = 0) -> Path:
    import cv2

    rng = np.random.default_rng(seed)
    root.mkdir(parents=True, exist_ok=True)
    for split, count in (("train", n_train), ("val", n_val)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            img = rng.integers(0, 60, size=(imgsz, imgsz, 3), dtype=np.uint8)
            labels = []
            for cls in rng.choice([0, 1], size=rng.integers(1, 3), replace=True):
                w = int(rng.integers(imgsz // 8, imgsz // 3))
                h = int(rng.integers(imgsz // 8, imgsz // 3))
                x0 = int(rng.integers(0, imgsz - w))
                y0 = int(rng.integers(0, imgsz - h))
                intensity = 230 if cls == 0 else 140
                cv2.rectangle(img, (x0, y0), (x0 + w, y0 + h), (intensity,) * 3, thickness=-1)
                labels.append(
                    f"{cls} {(x0 + w / 2) / imgsz:.6f} {(y0 + h / 2) / imgsz:.6f} "
                    f"{w / imgsz:.6f} {h / imgsz:.6f}"
                )
            # zero-padded numeric stems: the same contract real data must follow
            cv2.imwrite(str(img_dir / f"{i:06d}.jpg"), img)
            (lbl_dir / f"{i:06d}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")

    data_yaml = root / "fake_data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val",
             "names": {0: "ship", 1: "buoy"}},
            f, sort_keys=False,
        )
    return data_yaml


def run_smoke(cfg: dict) -> int:
    s = cfg["smoke"]
    smoke_root = Path(cfg["paths"]["outputs_root"]) / "smoke"
    if smoke_root.exists():
        shutil.rmtree(smoke_root)  # fresh every time — smoke must not "pass" on stale artifacts

    print("[smoke] fabricating tiny dataset ...")
    data_yaml = make_fake_dataset(
        smoke_root / "fake_data", n_train=s["n_images"], n_val=max(8, s["n_images"] // 4),
        imgsz=s["imgsz"],
    )

    print("[smoke] running 1-variant, 1-seed grid on CPU-scale settings ...")
    out_csv = run_grid(
        cfg, data_yaml,
        variants=[s["variant"]], seeds=[0], epochs=s["epochs"],
        imgsz=s["imgsz"], batch=s["batch"], workers=0,
        out_csv=smoke_root / "results.csv", run_prefix="smoke",
    )

    results = load_results(out_csv)
    assert s["variant"] in results and len(results[s["variant"]]) == 1, "grid row missing from CSV"
    row = results[s["variant"]][0]
    for key in ("map50", "map50_95", "precision", "recall"):
        float(row[key])  # must parse — values themselves are meaningless at this scale

    print("[smoke] measuring FPS on 5 frames ...")
    val_images = sorted((smoke_root / "fake_data" / "images" / "val").glob("*.jpg"))
    best = Path(row["run_dir"]) / "weights" / "best.pt"
    weights = best if best.is_file() else Path(row["run_dir"]) / "weights" / "last.pt"
    fps_row = measure_fps(cfg, s["variant"], weights, val_images, half=False, n_frames=5, warmup=2)
    write_fps_csv([fps_row], smoke_root / "fps.csv")

    print("[smoke] generating Table 1 ...")
    table = make_table1(out_csv, smoke_root / "fps.csv", smoke_root / "table1.md")
    assert s["variant"] in table

    print("\n[smoke] " + table)
    print("\nSMOKE OK — grid/val/CSV/FPS/table plumbing verified end-to-end")
    return 0
