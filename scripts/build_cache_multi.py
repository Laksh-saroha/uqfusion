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
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.config import load_config                                   # noqa: E402
from uqfusion.eval.cache import build_cache, load_cache                   # noqa: E402
from uqfusion.eval.corruptions import CORRUPTIONS                         # noqa: E402
from uqfusion.eval.parallel_frames import corrupted_frames, default_workers  # noqa: E402

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
    ap.add_argument("--corrupt-version", choices=["v1", "v2"], default="v1",
                    help="v1 = the albumentations corruptions every cache before 2026-10-09 used; "
                         "v2 = corruptions_v2.py (physically modelled, content rows only)")
    ap.add_argument("--modality", choices=["vis", "ir"], default=None,
                    help="frame modality (default: --data); v2 needs it for geometry and content rows")
    ap.add_argument("--params", default=None,
                    help="JSON object overriding corruption v2 constants (a sensitivity row)")
    ap.add_argument("--workers", type=int, default=None,
                    help="processes that read + corrupt frames (default: all logical cores - 2; "
                         "0 = in-process, the serial reference)")
    ap.add_argument("--batch", type=int, default=1,
                    help="frames per forward pass. 1 (default) is bit-identical to build_cache.py; "
                         ">1 is ~1.6x faster at 4 but shifts confidences by up to ~3e-2 and AP by "
                         "~1e-5 (ship) to ~2e-4 (macro) -- docs/eval/corruption_v2/batched_inference.json. "
                         "Do not mix with batch-1 caches in a comparison.")
    ap.add_argument("--stats-out", nargs=2, default=None, metavar=("BRIGHTNESS_JSON", "STRUCTURE_JSON"),
                    help="also write the gate's frame statistics of the corrupted frames, computed "
                         "in the same pass (same payload as p3dev_corruption_prep.py writes)")
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
    modality = args.modality or args.data
    spec = {"corrupt": args.corrupt, "severity": args.severity, "corrupt_seed": args.corrupt_seed,
            "corrupt_version": args.corrupt_version, "modality": modality,
            "params": json.loads(args.params) if args.params else None,
            "stats": modality if args.stats_out else None}
    workers = default_workers() if args.workers is None else args.workers
    code_rev = None
    if args.corrupt_version == "v2":
        from uqfusion.eval.corruptions_v2 import CODE_REV as code_rev

    t0 = time.time()
    recs = [[] for _ in preds]
    br, st = [], []
    pending: list = []

    def flush() -> None:
        if args.batch == 1:
            for j, p in enumerate(preds):
                recs[j].append(p(pending[0].copy()))
        else:
            for j, p in enumerate(preds):
                recs[j].extend(p.predict_batch(pending))
        pending.clear()

    for i, c, stats in corrupted_frames(images, spec, workers=workers):   # the corruption, once
        pending.append(c)
        if len(pending) == args.batch:
            flush()
        if stats is not None:
            br.append(stats[0])
            st.append(stats[1])
        if (i + 1) % 100 == 0:
            print(f"[multi] {i + 1}/{len(images)} frames ({time.time() - t0:.0f}s, {workers} workers, "
                  f"batch {args.batch})", flush=True)
    if pending:
        flush()

    for j, (w, out) in enumerate(zip(args.weights, args.outs)):
        it = iter(recs[j])
        meta = {"source": "gaussian", "weights": [str(w)], "data": args.data, "split": args.split,
                "images_list": args.images_list, "imgsz": args.imgsz, "conf": args.conf,
                "corrupt": args.corrupt, "severity": args.severity, "corrupt_seed": args.corrupt_seed,
                "built_by": "build_cache_multi.py (corruption replayed once for all weights)"}
        if args.corrupt_version != "v1":             # v1 caches stay byte-for-byte what they were
            meta.update(corrupt_version=args.corrupt_version, modality=modality, corrupt_code=code_rev)
            if args.params:
                meta["corrupt_params"] = json.loads(args.params)
        if args.batch != 1:
            meta["infer_batch"] = args.batch
        build_cache(lambda _img: next(it), images, Path(out), meta=meta, transform=None, log_every=0)

    if args.stats_out:
        import os

        from uqfusion.eval.frame_geometry import content_rows

        for path, frames in zip(args.stats_out, (br, st)):
            payload = {"cache": [str(o) for o in args.outs], "modality": modality,
                       "content_rows": list(content_rows(modality)), "corrupt": args.corrupt,
                       "severity": args.severity, "corrupt_seed": args.corrupt_seed,
                       "corrupt_version": args.corrupt_version, "corrupt_code": code_rev,
                       "corrupt_params": json.loads(args.params) if args.params else None,
                       "n_frames": len(frames),
                       "numpy": np.__version__, "opencv": cv2.__version__, "frames": frames}
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, out)
            print(f"[multi] stats {len(frames)} frames -> {out}", flush=True)

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
