"""Per-frame photometric statistics — the cheap frame-quality probe (TODO §0.2).

**Why this exists.** The §6.4 gate scores a frame by Mahalanobis distance on
pooled neck features, i.e. it asks "are these features UNUSUAL?". Darkness is not
unusual, it is *empty*: a night frame carries almost no edges or texture, which
lands near the middle of the feature distribution. Measured on the paired val
set, pohang01 (a genuine night run, mean content intensity 8.2, max 14.9; detector
max-confidence 0.0043 over 1,032 frames, mAP exactly 0.0000) scores D = 28.4 —
*lower* than pohang00's 30.0, a daylight run where the detector gets 0.4004.

So the gate reads the blindest frames in the dataset as the cleanest ones.

The fix probed here is deliberately dumb: also measure how bright the picture
actually is. No model, no training — arithmetic over pixels.

**Letterbox awareness.** The prepared tree is 640x640 with grey (114) padding
bars: VIS content is 640x338 at y+151 (53% of the canvas), IR is 640x512 at y+64
(80%). Averaging over the whole canvas would mix a constant 114 into every
statistic and crush exactly the contrast we are trying to measure, so every
statistic here is computed on the CONTENT ROWS ONLY.

**Corruption replay.** Ladder caches were built by corrupting the image in memory
at cache-build time, so the file on disk is clean and its brightness is NOT the
brightness the detector saw. `make_corruption` is deterministic per (seed,
index), so this script replays the exact transform recorded in the cache meta.
Verified by construction: same seed, same index, same albumentations call.

**The dataset is never written.** Images are opened read-only via `cv2.imread`;
every output goes to `runs/derived/`.

Usage:
    python scripts/frame_brightness.py --cache runs/cache/ladder/vis_lowlight_s2.pkl \
        --modality vis --out runs/derived/brightness/
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.corruptions import make_corruption

# Native sensor sizes, from the prepared-tree build (handoff §4.5).
NATIVE = {"vis": (2048, 1080), "ir": (640, 512)}


def content_rows(modality: str, canvas: int = 640) -> tuple[int, int]:
    """Rows of the letterboxed canvas that actually carry image, not pad."""
    w0, h0 = NATIVE[modality]
    scale = min(canvas / w0, canvas / h0)
    nh = round(h0 * scale)
    top = (canvas - nh) // 2
    return top, top + nh


def frame_stats(gray: np.ndarray) -> dict:
    """Photometric summary of one content region, all on the 0-255 scale.

    `lap_var` is the odd one out: every other statistic here is a HISTOGRAM
    property, and a veil-type corruption is invisible to all of them. Measured on
    the paired val set, fog/day is the BRIGHTEST, cleanest-looking cell by every
    one of them (p05 58 vs clean/day's 35, std 52 vs 57) while VIS mAP collapses
    from 0.3683 to 0.0020 -- 180x worse on frames a histogram calls pristine. Fog
    is a low-pass veil: it destroys structure at object scale and leaves the global
    distribution alone, so the gate needs one term that reads structure directly.

    Variance of the Laplacian is the standard focus/blur measure and separates the
    two cleanly: fog/day 18 and fog/night 39, against >=1336 for every non-fog day
    or night cell. It is not a fog detector -- any veil, defocus, or heavy blur
    lands in the same place, which is the point. It measures whether edges survive,
    not what removed them.
    """
    import cv2

    g = gray.astype(np.float32)
    p05, p50, p95 = np.percentile(g, [5, 50, 95])
    return {
        "lap_var": float(cv2.Laplacian(g, cv2.CV_32F, ksize=3).var()),
        "mean": float(g.mean()),
        "p05": float(p05),
        "p50": float(p50),
        "p95": float(p95),
        "std": float(g.std()),                       # RMS contrast
        "range": float(p95 - p05),                   # robust dynamic range
        "frac_dark": float((g < 30).mean()),         # how much of the frame is near-black
    }


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache", required=True, help="cache whose frames (and corruption) to measure")
    parser.add_argument("--modality", choices=["vis", "ir"], required=True)
    parser.add_argument("--out", default="runs/derived/brightness")
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    load_config(args.config)

    recs, meta = load_cache(args.cache)
    if args.limit:
        recs = recs[: args.limit]

    lo, hi = content_rows(args.modality)
    transform = None
    if meta.get("corrupt"):
        transform = make_corruption(meta["corrupt"], meta["severity"], meta["corrupt_seed"])
        print(f"[bright] replaying corruption {meta['corrupt']} s{meta['severity']} "
              f"seed {meta['corrupt_seed']} — file on disk is clean, cache is not")

    rows = []
    for i, r in enumerate(recs):
        im = cv2.imread(r["image_path"])          # READ-ONLY; the source tree is never written
        if im is None:
            raise FileNotFoundError(f"could not read image: {r['image_path']}")
        if transform is not None:
            im = transform(im, i)                 # same (seed, index) as build_cache -> same pixels
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[lo:hi]
        st = frame_stats(gray)
        st["image_path"] = r["image_path"]
        st["run"] = Path(r["image_path"]).parent.name
        rows.append(st)
        if (i + 1) % 250 == 0:
            print(f"[bright] {i + 1}/{len(recs)}")

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(args.cache).stem + ".json")
    payload = {
        "cache": str(args.cache),
        "modality": args.modality,
        "content_rows": [lo, hi],
        "corrupt": meta.get("corrupt"),
        "severity": meta.get("severity"),
        "corrupt_seed": meta.get("corrupt_seed"),
        "n_frames": len(rows),
        "frames": rows,
    }
    out_path.write_text(json.dumps(payload), encoding="utf-8")

    m = np.array([r["mean"] for r in rows])
    print(f"[bright] {Path(args.cache).stem}: mean intensity {m.mean():7.2f} "
          f"(min {m.min():6.2f}, max {m.max():6.2f}) over {len(rows)} frames -> {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
