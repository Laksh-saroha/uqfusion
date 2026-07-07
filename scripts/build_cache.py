"""Build a prediction cache (Phase 3; plan B5-2): one inference pass over a
split, pickled records for all downstream metrics and ablations.

Uncertainty sources:
    gaussian  --weights <gauss best.pt>                (σ + DFL-derived + features)
    mc        --weights <mc best.pt>    (T from config baselines.mc_dropout.T)
    ensemble  --weights <m0> <m1> ...   (one per member)

Optional corruption (scope §5.2; anti-leakage rule B5-5: use DIFFERENT
--corrupt-seed for tuning vs final testing — the seed is stamped in the meta):
    --corrupt fog --severity 2 --corrupt-seed 1

Usage:
    python scripts/build_cache.py --source gaussian --weights runs/gaussian/gauss_yolov8s_seed0/weights/best.pt \
        --data vis --split val --out runs/cache/gauss_vis_val_clean.pkl
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.data.lists import load_data_yaml, split_image_list
from uqfusion.eval.cache import build_cache
from uqfusion.eval.corruptions import CORRUPTIONS, make_corruption


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--source", choices=["gaussian", "mc", "ensemble"], required=True)
    parser.add_argument("--weights", nargs="+", required=True)
    parser.add_argument("--data", default="vis")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out", required=True)
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--limit", type=int, default=None, help="cap frame count (debug)")
    parser.add_argument("--corrupt", choices=list(CORRUPTIONS), default=None)
    parser.add_argument("--severity", type=int, default=2)
    parser.add_argument("--corrupt-seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    imgsz = args.imgsz or cfg["benchmark"]["imgsz"]
    device = cfg.get("device", "auto")
    device = "cpu" if device in (None, "auto") and not _cuda() else (0 if device in (None, "auto") else device)

    if args.source == "gaussian":
        from uqfusion.uq.infer import UQPredictor

        predictor = UQPredictor(args.weights[0], device=device, imgsz=imgsz, conf=args.conf)
    elif args.source == "mc":
        from uqfusion.uq.mc_dropout import MCDropoutPredictor

        t_passes = int((cfg.get("baselines") or {}).get("mc_dropout", {}).get("T", 10))
        predictor = MCDropoutPredictor(args.weights[0], T=t_passes, device=device, imgsz=imgsz, conf=args.conf)
    else:
        from uqfusion.uq.ensemble import EnsemblePredictor

        predictor = EnsemblePredictor(args.weights, device=device, imgsz=imgsz, conf=args.conf)

    images = split_image_list(load_data_yaml(resolve_data_yaml(cfg, args.data)), args.split)
    if args.limit:
        images = images[: args.limit]

    transform = make_corruption(args.corrupt, args.severity, args.corrupt_seed) if args.corrupt else None
    meta = {
        "source": args.source, "weights": [str(w) for w in args.weights], "data": args.data,
        "split": args.split, "imgsz": imgsz, "conf": args.conf,
        "corrupt": args.corrupt, "severity": args.severity if args.corrupt else None,
        "corrupt_seed": args.corrupt_seed if args.corrupt else None,
    }
    build_cache(predictor, images, Path(args.out), meta=meta, transform=transform)
    return 0


def _cuda() -> bool:
    import torch

    return torch.cuda.is_available()


if __name__ == "__main__":
    sys.exit(main())
