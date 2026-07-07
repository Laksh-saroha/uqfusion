"""Train the MC-Dropout baseline (Phase 3; plan B6-3).

Its training results.csv doubles as the required deterministic row (validation
runs with dropout off). Dropout p and inference T live in config `baselines.mc_dropout`.

Usage:
    python scripts/train_mc_dropout.py --data vis --variant yolov8s --seed 0
"""

from __future__ import annotations

import argparse
import sys

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.uq.mc_dropout import train_mc_dropout


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="vis")
    parser.add_argument("--variant", default="yolov8s")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    parser.add_argument("--name", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    best, run_dir = train_mc_dropout(
        cfg, resolve_data_yaml(cfg, args.data), variant=args.variant, seed=args.seed,
        epochs=args.epochs, batch=args.batch, run_name=args.name,
    )
    print(f"[mc-dropout] best weights: {best}\n[mc-dropout] run dir: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
