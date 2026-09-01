"""Extended per-frame STRUCTURE statistics — the input-side axis, done properly.

**The question this exists to answer.** `docs/gated-fusion-handoff.md` §7.2 claims
no frame statistic separates lowlight/day (VIS alive) from clean/night (VIS dead),
because on `lap_var` lowlight/day (152) sits between clean/night (1334) and
lowlight/night (23). That is true of `lap_var`, and the reason is that `lap_var`
is not scale-invariant: the lowlight corruption is
`RandomBrightnessContrast(brightness_limit=(-0.9, -0.7))`, i.e. multiplication by
roughly 0.1-0.3, and a Laplacian is linear, so its variance falls by the SQUARE of
that factor -- 0.04x to 0.09x -- purely from amplitude, with the scene's structure
completely intact. Real night is the opposite: harbour lamps on black water,
enormous local amplitude concentrated in a handful of pixels and nothing anywhere
else.

So the two cases differ in WHERE the gradient energy sits, not in how much of it
there is, and every statistic here is chosen to read that instead of amplitude:

* `lap_over_var`   -- lap_var / var(I). Both numerator and denominator scale by
                      k^2 under multiplication by k, so this is invariant to
                      dimming BY CONSTRUCTION, not by luck.
* `grad_gini`      -- Gini coefficient of |grad I| over pixels. Scale-free.
                      Concentrated energy (lamps on black) -> near 1; energy
                      spread over a textured scene -> lower.
* `tex_cover`      -- fraction of 16x16 tiles whose local std exceeds 10% of the
                      frame's own std. "How much of the picture carries texture",
                      normalized by the picture's own contrast.
* `tile_std_p50`   -- median local std / global std: the same idea as a robust
                      centre rather than a coverage count.
* `spec_slope`     -- slope of log radial power vs log spatial frequency. A veil
                      is a low-pass filter and steepens it; dimming is a gain and
                      does not move it at all.
* `edge_density`   -- fraction of pixels passing Canny at thresholds set from the
                      frame's OWN percentiles, so the operator is scale-free too.
* `log_range`      -- log((p95+1)/(p05+1)): dynamic range in a multiplicative
                      scale, where dimming is a shift and not a compression.

`lap_var`, `p05` and the rest of the photometric block are recomputed here as
well so one file carries everything and no analysis has to join two.

**Nothing is overwritten.** Output goes to a new directory (default
`runs/derived/structure/`), leaving `runs/derived/brightness/` -- which the
adopted gate reads -- untouched. Images are opened read-only.

**Corruption replay** is byte-identical to `frame_brightness.py`: same
`make_corruption(kind, severity, seed)` and the same frame index, so the pixels
measured are the pixels the detector saw.

Usage:
    python scripts/frame_structure.py --cache runs/cache/gauss_vis_paired_fog.pkl --modality vis
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uqfusion.config import load_config          # noqa: E402
from uqfusion.eval.cache import load_cache       # noqa: E402
from uqfusion.eval.corruptions import make_corruption  # noqa: E402

NATIVE = {"vis": (2048, 1080), "ir": (640, 512)}
_EPS = 1e-9


def content_rows(modality: str, canvas: int = 640) -> tuple[int, int]:
    """Rows of the letterboxed canvas that actually carry image, not pad.

    Identical to `frame_brightness.content_rows`; duplicated rather than imported
    so this script has no import-time dependency on a script directory.
    """
    w0, h0 = NATIVE[modality]
    scale = min(canvas / w0, canvas / h0)
    nh = round(h0 * scale)
    top = (canvas - nh) // 2
    return top, top + nh


def _gini(x: np.ndarray) -> float:
    """Gini coefficient of a non-negative vector — 0 = perfectly even, 1 = all in one place.

    Scale-free: multiplying every element by k leaves it unchanged, which is
    exactly the property the dimming case needs.
    """
    v = np.sort(np.asarray(x, dtype=np.float64).ravel())
    n = v.size
    s = v.sum()
    if n == 0 or s <= _EPS:
        return 0.0
    idx = np.arange(1, n + 1, dtype=np.float64)
    return float((2.0 * (idx * v).sum()) / (n * s) - (n + 1.0) / n)


def _spectral_slope(g: np.ndarray) -> float:
    """Slope of log10(radial power) against log10(frequency).

    Natural images sit near -2. A veil is a low-pass filter and drives it more
    negative; a gain change multiplies every frequency equally and leaves it
    exactly where it was, which is what makes it useful here.
    """
    a = g - g.mean()
    if a.std() < _EPS:
        return 0.0
    win = np.outer(np.hanning(a.shape[0]), np.hanning(a.shape[1]))
    f = np.fft.fftshift(np.abs(np.fft.fft2(a * win)) ** 2)
    cy, cx = np.array(f.shape) // 2
    y, x = np.indices(f.shape)
    r = np.sqrt((y - cy) ** 2 + (x - cx) ** 2).astype(int)
    rmax = min(cy, cx)
    prof = np.bincount(r.ravel(), f.ravel())[:rmax] / np.maximum(np.bincount(r.ravel())[:rmax], 1)
    k = np.arange(1, rmax)
    p = prof[1:]
    m = p > 0
    if m.sum() < 8:
        return 0.0
    return float(np.polyfit(np.log10(k[m]), np.log10(p[m]), 1)[0])


def _tiles(g: np.ndarray, t: int = 16) -> np.ndarray:
    """Per-tile std over a t x t grid, ragged edges dropped."""
    h, w = g.shape
    hh, ww = (h // t) * t, (w // t) * t
    if hh == 0 or ww == 0:
        return np.zeros(0)
    b = g[:hh, :ww].reshape(hh // t, t, ww // t, t).transpose(0, 2, 1, 3)
    return b.reshape(-1, t * t).std(axis=1)


def frame_stats(gray: np.ndarray) -> dict:
    """Photometric block (as in `frame_brightness.py`) plus the scale-free structure block."""
    import cv2

    g = gray.astype(np.float32)
    p05, p50, p95 = np.percentile(g, [5, 50, 95])
    std = float(g.std())
    lap = cv2.Laplacian(g, cv2.CV_32F, ksize=3)
    lap_var = float(lap.var())
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx * gx + gy * gy)
    ts = _tiles(g)
    # Canny thresholds from the frame's own gradient percentiles, so the operator
    # does not itself introduce an absolute-brightness dependence.
    hi = float(np.percentile(mag, 99)) or 1.0
    edges = cv2.Canny(np.clip(g, 0, 255).astype(np.uint8),
                      max(hi * 0.2, 1.0), max(hi * 0.5, 2.0))
    return {
        # photometric (recomputed so this file stands alone)
        "mean": float(g.mean()), "p05": float(p05), "p50": float(p50), "p95": float(p95),
        "std": std, "range": float(p95 - p05), "frac_dark": float((g < 30).mean()),
        "lap_var": lap_var,
        # scale-free structure
        "lap_over_var": float(lap_var / max(std * std, _EPS)),
        "grad_gini": _gini(mag),
        "grad_mean": float(mag.mean()),
        "tex_cover": float((ts > 0.10 * max(std, _EPS)).mean()) if ts.size else 0.0,
        "tile_std_p50": float(np.median(ts) / max(std, _EPS)) if ts.size else 0.0,
        "spec_slope": _spectral_slope(g),
        "edge_density": float((edges > 0).mean()),
        "log_range": float(np.log((p95 + 1.0) / (p05 + 1.0))),
    }


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=None)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--modality", choices=["vis", "ir"], default="vis")
    ap.add_argument("--out", default="runs/derived/structure")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()
    load_config(args.config)

    recs, meta = load_cache(args.cache)
    if args.limit:
        recs = recs[: args.limit]
    lo, hi = content_rows(args.modality)

    transform = None
    if meta.get("corrupt"):
        transform = make_corruption(meta["corrupt"], meta["severity"], meta["corrupt_seed"])
        print(f"[struct] replaying {meta['corrupt']} s{meta['severity']} "
              f"seed {meta['corrupt_seed']} — the file on disk is clean, the cache is not")

    rows = []
    for i, r in enumerate(recs):
        im = cv2.imread(r["image_path"])            # READ-ONLY
        if im is None:
            raise FileNotFoundError(f"could not read image: {r['image_path']}")
        if transform is not None:
            im = transform(im, i)                   # same (seed, index) as build_cache
        st = frame_stats(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[lo:hi])
        st["image_path"] = r["image_path"]
        st["run"] = Path(r["image_path"]).parent.name
        rows.append(st)
        if (i + 1) % 250 == 0:
            print(f"[struct] {i + 1}/{len(recs)}", flush=True)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / (Path(args.cache).stem + ".json")
    out_path.write_text(json.dumps({
        "cache": str(args.cache), "modality": args.modality, "content_rows": [lo, hi],
        "corrupt": meta.get("corrupt"), "severity": meta.get("severity"),
        "corrupt_seed": meta.get("corrupt_seed"), "n_frames": len(rows), "frames": rows,
    }), encoding="utf-8")
    print(f"[struct] {len(rows)} frames -> {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
