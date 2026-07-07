"""Generate Table 1 (scope §9.5) from the benchmark CSVs — mean ± std over seeds.

Usage:
    python scripts/make_table1.py
    python scripts/make_table1.py --results-csv runs/benchmark/ir_results.csv --out runs/benchmark/table1_ir.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uqfusion.bench.table import make_table1
from uqfusion.config import load_config


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--results-csv", default=None)
    parser.add_argument("--fps-csv", default=None)
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    bench_dir = Path(cfg["paths"]["outputs_root"]) / "benchmark"
    results_csv = Path(args.results_csv) if args.results_csv else bench_dir / "benchmark_results.csv"
    fps_csv = Path(args.fps_csv) if args.fps_csv else bench_dir / "fps.csv"
    out_md = Path(args.out) if args.out else bench_dir / "table1.md"

    print(make_table1(results_csv, fps_csv if fps_csv.is_file() else None, out_md))
    return 0


if __name__ == "__main__":
    sys.exit(main())
