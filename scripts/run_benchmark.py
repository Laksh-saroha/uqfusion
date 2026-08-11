"""Run the Phase 1 multi-seed benchmark grid (plan C5 -> Table 1).

Resume-safe: (variant, seed) pairs already in the results CSV are skipped, so
re-running after a crash continues the grid.

Typical server sequence:
    python scripts/audit_split.py                                   # must PASS first
    python scripts/make_stride_subset.py --data vis                 # -> derived yaml
    # dry-run to calibrate cost + pin batch (plan C5):
    python scripts/run_benchmark.py --data runs/derived/data_vis_stride2.yaml \
        --variants yolov8s --seeds 0 --epochs 1
    # full grid:
    python scripts/run_benchmark.py --data runs/derived/data_vis_stride2.yaml
"""

from __future__ import annotations

import argparse
import sys

from uqfusion.bench.grid import run_grid
from uqfusion.config import load_config, resolve_data_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="vis", help="vis, ir, or a dataset yaml path (e.g. derived stride yaml)")
    parser.add_argument("--variants", nargs="*", default=None, help="override benchmark.variants")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="override benchmark.seeds")
    parser.add_argument("--epochs", type=int, default=None, help="override benchmark.epochs (dry-runs)")
    parser.add_argument("--batch", type=int, default=None, help="override benchmark.batch (dry-runs)")
    parser.add_argument("--workers", type=int, default=None,
                        help="override benchmark.workers (config's 2 is the server /dev/shm cap; "
                             "a local box with real shared memory wants more)")
    parser.add_argument("--out-csv", default=None, help="override results CSV path (e.g. IR confirmation grid)")
    parser.add_argument("--run-prefix", default="bench", help="run-name prefix (use e.g. 'ir' for the IR grid)")
    parser.add_argument("--classes", nargs="*", type=int, default=None,
                        help="train/eval only these class ids (e.g. --classes 0 = ship-only); "
                             "default: all classes")
    parser.add_argument("--mosaic", type=float, default=None,
                        help="mosaic augmentation probability 0.0-1.0 (train arg); "
                             "default: ultralytics default")
    parser.add_argument("--close-mosaic", type=int, default=None,
                        help="disable mosaic for the final N epochs (train arg); "
                             "0 = keep mosaic on the whole run (clean ablation); "
                             "default: ultralytics default (10)")
    parser.add_argument("--no-resume", action="store_true",
                        help="restart an interrupted run at epoch 0 instead of continuing "
                             "from its weights/last.pt (default: resume)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data_yaml = resolve_data_yaml(cfg, args.data)
    out_csv = run_grid(
        cfg, data_yaml,
        variants=args.variants, seeds=args.seeds, epochs=args.epochs, batch=args.batch,
        workers=args.workers,
        out_csv=args.out_csv, run_prefix=args.run_prefix, classes=args.classes,
        mosaic=args.mosaic, close_mosaic=args.close_mosaic, resume=not args.no_resume,
    )
    print(f"[grid] results CSV: {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
