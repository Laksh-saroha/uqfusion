"""Build several Gaussian prediction caches in one pass: corrupt each frame ONCE, run every model.

`build_cache.py` builds one cache per process, so N checkpoints over the same corrupted list
replay the corruption N times. The corruption depends on (kind, severity, seed, frame index)
only, never on the checkpoint, and at native resolution it is the dominant cost (fog ~290
ms/frame, single-threaded). This replays it once and feeds the same frame to every model.

The output is written through `uqfusion.eval.cache.build_cache` itself, so each file carries
exactly the payload and meta a `build_cache.py` run with the same arguments would: the
precomputed records are handed to it as a predictor. `--verify-against` checks a prefix of
the output record by record against caches built the old way.

Usage:
    python scripts/build_cache_multi.py --weights W0 W1 ... --outs O0 O1 ... \\
        --images-list runs/derived/paired_val_vis.txt --imgsz 640 --conf 0.001 \\
        --corrupt fog --severity 1 --corrupt-seed 941
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.config import load_config                                   # noqa: E402
from uqfusion.eval.cache import build_cache, load_cache                   # noqa: E402
from uqfusion.eval.corruptions import CORRUPTIONS, make_corruption        # noqa: E402

KEYS = ("boxes_xyxy", "conf", "cls", "sigma_ltrb", "feat")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--weights", nargs="+", required=True)
    ap.add_argument("--outs", nargs="+", required=True)
    ap.add_argument("--images-list", required=True)
    ap.add_argument("--data", default="vis")
    ap.add_argument("--split", default="val")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--corrupt", choices=list(CORRUPTIONS), required=True)
    ap.add_argument("--severity", type=int, default=2)
    ap.add_argument("--corrupt-seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--verify-against", nargs="+", default=None,
                    help="existing caches, one per weight, to compare the output prefix against")
    args = ap.parse_args()
    if len(args.weights) != len(args.outs):
        raise SystemExit("--weights and --outs must pair one to one")

    import cv2
    import torch
    from uqfusion.uq.infer import UQPredictor

    cfg = load_config(args.config)
    device = cfg.get("device", "auto")
    device = ("cpu" if device in (None, "auto") and not torch.cuda.is_available()
              else (0 if device in (None, "auto") else device))
    preds = [UQPredictor(w, device=device, imgsz=args.imgsz, conf=args.conf) for w in args.weights]
    lp = Path(args.images_list)
    images = [Path(ln.strip()) for ln in lp.read_text(encoding="utf-8").splitlines() if ln.strip()]
    if args.limit:
        images = images[: args.limit]
    tf = make_corruption(args.corrupt, args.severity, args.corrupt_seed)

    t0 = time.time()
    recs = [[] for _ in preds]
    for i, img in enumerate(images):
        im = cv2.imread(str(img))
        if im is None:
            raise FileNotFoundError(f"could not read image: {img}")
        c = tf(im, i)                                   # the corruption, once
        for j, p in enumerate(preds):
            recs[j].append(p(c.copy()))
        if (i + 1) % 100 == 0:
            print(f"[multi] {i + 1}/{len(images)} frames ({time.time() - t0:.0f}s)", flush=True)

    for j, (w, out) in enumerate(zip(args.weights, args.outs)):
        it = iter(recs[j])
        meta = {"source": "gaussian", "weights": [str(w)], "data": args.data, "split": args.split,
                "images_list": args.images_list, "imgsz": args.imgsz, "conf": args.conf,
                "corrupt": args.corrupt, "severity": args.severity, "corrupt_seed": args.corrupt_seed,
                "built_by": "build_cache_multi.py (corruption replayed once for all weights)"}
        build_cache(lambda _img: next(it), images, Path(out), meta=meta, transform=None, log_every=0)

    if args.verify_against:
        worst = 0.0
        for j, ref in enumerate(args.verify_against):
            got, _ = load_cache(args.outs[j])
            want, _ = load_cache(ref)
            for i, (a, b) in enumerate(zip(got, want)):
                if Path(a["image_path"]).name != Path(b["image_path"]).name:
                    raise SystemExit(f"[verify] {ref} frame {i}: different image")
                for k in KEYS:
                    x, y = np.asarray(a[k], dtype=float), np.asarray(b[k], dtype=float)
                    if x.shape != y.shape:
                        raise SystemExit(f"[verify] {ref} frame {i} {k}: shape {x.shape} vs {y.shape}")
                    if x.size:
                        worst = max(worst, float(np.max(np.abs(x - y))))
        print(f"[verify] {len(args.verify_against)} caches x {len(got)} frames: shapes identical, "
              f"max |diff| {worst:.3e}", flush=True)
    print(f"[multi] {len(preds)} caches from one corruption pass in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
