"""Train a Gaussian-head model on real data (Phase 2 server run).

For the §12.1 parity check (A4-11's condition), train the SAME variant WITHOUT
the σ branch at identical settings via run_benchmark.py and compare mAP.

Usage:
    python scripts/train_gaussian_model.py --data vis --variant yolov8s --seed 0
    python scripts/train_gaussian_model.py --data runs/derived/data_vis_stride2.yaml --epochs 60
"""

from __future__ import annotations

import argparse
import sys

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.uq.train_gaussian import train_gaussian


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="vis")
    parser.add_argument("--variant", default="yolov8s", help="Phase 1 winner once Table 1 exists")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    best, run_dir = train_gaussian(
        cfg, resolve_data_yaml(cfg, args.data), variant=args.variant, seed=args.seed,
        epochs=args.epochs, batch=args.batch, run_name=args.name,
    )
    print(f"[gaussian] best weights: {best}\n[gaussian] run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
