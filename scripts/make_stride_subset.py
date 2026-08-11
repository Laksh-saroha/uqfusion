"""Create a stride-subsampled TRAINING list + derived dataset yaml (approved A2-6).

Strides over unique per-run TIME STEPS (frame ordinals), not raw frame lists —
L/R stereo frames share an index and must be kept/dropped together, otherwise
the subset alternates cameras instead of thinning temporal near-duplicates.

Never touches the source data. Run after the split audit passes.

Usage:
    python scripts/make_stride_subset.py --data vis            # stride from config
    python scripts/make_stride_subset.py --data ir --stride 3

Prints the derived yaml path — pass that to run_benchmark.py --data.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.data.lists import load_data_yaml
from uqfusion.data.subset import make_stride_subset


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="vis", help="vis, ir, or a dataset yaml path")
    parser.add_argument("--stride", type=int, default=None,
                        help="override benchmark.train_stride from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    stride = args.stride or cfg["benchmark"]["train_stride"]
    data = load_data_yaml(resolve_data_yaml(cfg, args.data))
    out_dir = Path(cfg["paths"]["outputs_root"]) / "derived"
    derived_yaml = make_stride_subset(data, stride=stride, out_dir=out_dir)
    print(derived_yaml)
    return 0


if __name__ == "__main__":
    sys.exit(main())
