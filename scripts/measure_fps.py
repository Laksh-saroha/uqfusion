"""Measure batch-1 FPS/latency per variant (plan C5) after the grid has run.

Discovers each variant's lowest-seed best.pt from the results CSV, times
predict() end-to-end on val images (fp32 and, on GPU, fp16), writes fps.csv.

Usage:
    python scripts/measure_fps.py --data vis
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uqfusion.bench.fps import measure_fps, write_fps_csv
from uqfusion.bench.grid import resolve_device
from uqfusion.bench.table import load_results
from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.data.lists import load_data_yaml, split_image_list


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="vis", help="dataset supplying the timing images (val split)")
    parser.add_argument("--results-csv", default=None, help="grid results CSV (default: runs/benchmark/benchmark_results.csv)")
    parser.add_argument("--out-csv", default=None, help="output (default: runs/benchmark/fps.csv)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    bench_dir = Path(cfg["paths"]["outputs_root"]) / "benchmark"
    results_csv = Path(args.results_csv) if args.results_csv else bench_dir / "benchmark_results.csv"
    out_csv = Path(args.out_csv) if args.out_csv else bench_dir / "fps.csv"

    images = split_image_list(load_data_yaml(resolve_data_yaml(cfg, args.data)), "val")

    import torch
    device = resolve_device(cfg)
    on_cpu = (device == "cpu") or (device is None and not torch.cuda.is_available())
    precisions = [False] if on_cpu else [False, True]

    rows = []
    for variant, variant_rows in sorted(load_results(results_csv).items()):
        row = min(variant_rows, key=lambda r: int(r["seed"]))
        weights = Path(row["run_dir"]) / "weights" / "best.pt"
        if not weights.is_file():
            print(f"[fps] WARNING: {weights} missing — skipping {variant}")
            continue
        for half in precisions:
            result = measure_fps(cfg, variant, weights, images, half=half)
            print(f"[fps] {variant} half={half}: {result['fps']} fps ({result['wall_ms_per_img']} ms/img)")
            rows.append(result)

    if not rows:
        print("[fps] no rows produced — has the grid run?")
        return 1
    write_fps_csv(rows, out_csv)
    print(f"[fps] written -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
