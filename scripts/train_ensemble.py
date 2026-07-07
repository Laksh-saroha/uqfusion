"""Train the Deep Ensemble members (Phase 3; plan B6-7: members ARE the seeds).

Reuses the resume-safe Phase 1 grid runner — a crashed ensemble continues
where it stopped, and each member's deterministic row lands in the CSV.

Usage:
    python scripts/train_ensemble.py --data vis --variant yolov8s          # seeds from config (M=5)
    python scripts/train_ensemble.py --data ir --variant yolov8s --seeds 0 1 2
"""

from __future__ import annotations

import argparse
import sys

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.uq.ensemble import train_ensemble


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="vis")
    parser.add_argument("--variant", default="yolov8s")
    parser.add_argument("--seeds", nargs="*", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch", type=int, default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    weights, csv_path = train_ensemble(
        cfg, resolve_data_yaml(cfg, args.data), variant=args.variant, seeds=args.seeds,
        epochs=args.epochs, batch=args.batch,
    )
    print(f"[ensemble] member CSV: {csv_path}")
    for w in weights:
        print(f"[ensemble] member: {w}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
