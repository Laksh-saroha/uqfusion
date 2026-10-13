"""Metric depth maps for corruption v2 fog: Depth Anything V2, scaled by the sea plane.

Why. Koschmieder fog needs each pixel's range. The flat-sea range from the frame's own
attitude (frame_geometry.py) is exact on open water but knows nothing above the horizon,
so a near vessel's superstructure and the whole far shore got the far range and vanished
even at the mildest severity (docs/eval/corruption_v2/vis_sheet_fog.png, first build).
A monocular depth network sees objects but only up to an unknown affine map of inverse
depth. The two are complementary: on the water, below the horizon, the network's
disparity is mapped to 1/range of the sea plane (robust: medians in 40 inverse-range
bins, so the vessels sitting on the water do not drag it; affine in inverse range,
fitted on water out to FIT_MAX_M where the targets are), and that one map puts the whole
frame, objects included, in metres. A per-frame piecewise-linear map through the bins was
tried and rejected: the far bins are nearly flat in disparity and it amplified their noise
(in-box range 1.97x the waterline range at the median, against 1.31x affine). This is the approach of the published
maritime fog sets (monocular depth + atmospheric scattering), with the scale taken from
the calibrated geometry instead of assumed.

Rules after the fit: ranges are clipped to [3, FAR_M]. The network's own range is kept
above the horizon. (An earlier rule sent everything above the horizon that came out farther
than the horizon sea to FAR_M; it forced 35% of the pixels of horizon-crossing target boxes
to FAR_M and bought nothing, since the sky's median range is already 1.1 km, t < 0.003 at
V = 775 m. Kept as --sky-rule.)

Residual bias, measured on the 3,760 VIS GT boxes of every third paired-val day frame
(boxes read for this measurement only): the median range inside a box is 1.24x the
flat-sea range of its waterline for boxes below the horizon (n = 3,096) and 1.56x for
boxes crossing it (n = 664), against 1.51x for the flat-sea map alone. Open water next to
a target reads 1.07x. Small targets are partly given their background's range, so fog
is somewhat stronger on them than physics says. A 1036-px network input was worse
(1.50x overall).
A frame whose fit is bad (slope <= 0 or R^2 < 0.9) falls back to the sea-plane range with
everything at or above the horizon capped at FALLBACK_CAP_M, and is flagged.

Night VIS frames are contrast-stretched before the network sees them (the network was not
trained on 8-grey-level images); the stretch touches only the network input.

Output, one file per frame so any frame list can be served:
    runs/derived/depth_v2/{vis,ir}/<run>/<stem>.npy      float16 metres, content rows only
    runs/derived/depth_v2/{vis,ir}/manifest_<list>.json  model, revision, per-frame fit

Runs in the isolated .venv_depth (it has `transformers`; the system GPU interpreter does
not, on purpose):
    .venv_depth/Scripts/python.exe scripts/build_depth_maps.py --list runs/derived/paired_val_vis.txt --modality vis
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.frame_geometry import content_rows             # noqa: E402
from uqfusion.eval.corruptions_v2 import geometry                  # noqa: E402

MODEL_ID = "depth-anything/Depth-Anything-V2-Small-hf"
FAR_M = 2000.0
NEAR_M = 3.0
FALLBACK_CAP_M = 300.0
FIT_MAX_M = 400.0            # water used for the fit: up to this sea-plane range
OUT = ROOT / "runs/derived/depth_v2"


def depth_path(image_path: str | Path, modality: str, root: Path | None = None) -> Path:
    root = root or OUT
    p = Path(image_path)
    return root / modality / p.parent.name / f"{p.stem}.npy"


def stretch(c: np.ndarray) -> np.ndarray:
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    if g.mean() > 50:
        return c
    lo, hi = np.percentile(g, [1, 99.5])
    x = np.clip((c.astype(np.float32) - lo) / max(hi - lo, 1.0), 0, 1) ** 0.6
    return (x * 255).astype(np.uint8)


def align(disp: np.ndarray, path: str, modality: str, sky_rule: bool = False) -> tuple[np.ndarray, dict]:
    geo = geometry(path, modality)
    dsea = geo.range_m(np.inf)
    hz = geo.horizon_rows()
    rows = np.arange(disp.shape[0])[:, None]
    sea = (rows > hz[None, :] + 4) & np.isfinite(dsea) & (dsea < FIT_MAX_M)
    info = {"geometry": type(geo).__name__}
    fit_ok = False
    if sea.sum() > 2000:
        inv, y = 1.0 / dsea[sea], disp[sea]
        qs = np.quantile(inv, np.linspace(0, 1, 41))
        xb, yb = [], []
        for a, b in zip(qs[:-1], qs[1:]):
            m = (inv >= a) & (inv <= b)
            if m.sum() > 50:
                xb.append(np.median(inv[m]))
                yb.append(np.median(y[m]))
        xb, yb = np.asarray(xb), np.asarray(yb)
        if len(xb) >= 10:
            s, t = np.polyfit(xb, yb, 1)
            r2 = 1 - np.sum((yb - (s * xb + t)) ** 2) / max(np.sum((yb - yb.mean()) ** 2), 1e-12)
            info.update(slope=float(s), offset=float(t), r2=float(r2))
            fit_ok = s > 0 and r2 >= 0.9
    if fit_ok:
        invm = (disp - info["offset"]) / info["slope"]
        d = np.where(invm > 1.0 / FAR_M, 1.0 / np.maximum(invm, 1e-12), FAR_M)
        band = (rows > hz[None, :] + 1) & (rows <= hz[None, :] + 4)
        d_hz = float(np.median(d[band])) if band.any() else FAR_M
        above = rows <= hz[None, :]
        if sky_rule:
            d = np.where(above & (d > d_hz), FAR_M, d)
        info["d_horizon_m"] = d_hz
    else:
        d = np.where(np.isfinite(dsea), np.minimum(dsea, FALLBACK_CAP_M), FALLBACK_CAP_M)
    info["fallback"] = not fit_ok
    return np.clip(d, NEAR_M, FAR_M).astype(np.float16), info


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", required=True)
    ap.add_argument("--modality", choices=["vis", "ir"], required=True)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--input-size", type=int, default=518,
                    help="network input short side (518 = the model's training size; larger resolves "
                         "smaller targets at ~quadratic cost)")
    ap.add_argument("--sky-rule", action="store_true",
                    help="send pixels above the horizon that are farther than the horizon sea to FAR_M "
                         "(off by default: it clipped the tops of horizon-crossing targets)")
    ap.add_argument("--out-root", default=None, help="write here instead of runs/derived/depth_v2 (trials)")
    args = ap.parse_args()
    import torch
    import transformers
    from transformers import AutoImageProcessor, AutoModelForDepthEstimation

    paths = [ln.strip() for ln in Path(args.list).read_text(encoding="utf-8").splitlines() if ln.strip()]
    root = Path(args.out_root) if args.out_root else OUT
    todo = [p for p in paths if args.overwrite or not depth_path(p, args.modality, root).is_file()]
    proc = AutoImageProcessor.from_pretrained(MODEL_ID, size={"height": args.input_size, "width": args.input_size})
    model = AutoModelForDepthEstimation.from_pretrained(MODEL_ID).to("cuda").eval()
    lo, hi = content_rows(args.modality)
    pool = ThreadPoolExecutor(args.threads)          # network input: read, stretch, resize, normalise
    fit_pool = ThreadPoolExecutor(max(2, args.threads // 2))   # sea-plane fit + write

    def prep(p):
        im = cv2.imread(p)
        if im is None:
            raise FileNotFoundError(p)
        c = im[lo:hi]
        return proc(images=cv2.cvtColor(stretch(c), cv2.COLOR_BGR2RGB), return_tensors="pt")["pixel_values"]

    def finish(p, disp):
        d, info = align(disp, p, args.modality, sky_rule=args.sky_rule)
        out = depth_path(p, args.modality, root)
        out.parent.mkdir(parents=True, exist_ok=True)
        np.save(out, d)
        return p, info

    t0 = time.time()
    fits, pend = {}, []
    batches = [todo[i:i + args.batch] for i in range(0, len(todo), args.batch)]
    futs = [pool.submit(prep, p) for p in batches[0]] if batches else []
    for bi, b in enumerate(batches):
        x = torch.cat([f.result() for f in futs])
        if bi + 1 < len(batches):                    # one batch of prefetch: bounded memory
            futs = [pool.submit(prep, p) for p in batches[bi + 1]]
        with torch.no_grad():
            y = model(pixel_values=x.to("cuda")).predicted_depth
            y = torch.nn.functional.interpolate(y[:, None], size=(hi - lo, 640), mode="bicubic",
                                                align_corners=False)[:, 0].float().cpu().numpy()
        pend += [fit_pool.submit(finish, p, y[k]) for k, p in enumerate(b)]
        if (bi + 1) % 20 == 0:
            print(f"[depth] {(bi + 1) * args.batch}/{len(todo)} frames ({time.time() - t0:.0f}s)", flush=True)
    for f in pend:
        p, info = f.result()
        fits[p] = info
    pool.shutdown()
    fit_pool.shutdown()

    man = root / args.modality / f"manifest_{Path(args.list).stem}.json"
    old = json.loads(man.read_text(encoding="utf-8"))["frames"] if man.is_file() else {}
    old.update(fits)
    fb = sum(v["fallback"] for v in old.values())
    man.parent.mkdir(parents=True, exist_ok=True)
    man.write_text(json.dumps({
        "model": MODEL_ID, "revision": model.config._commit_hash, "transformers": transformers.__version__,
        "torch": torch.__version__, "input_size": args.input_size, "sky_rule": args.sky_rule, "fit_max_m": FIT_MAX_M, "far_m": FAR_M, "near_m": NEAR_M, "fallback_cap_m": FALLBACK_CAP_M,
        "list": args.list, "n_frames": len(old), "n_fallback": fb, "frames": old}, indent=0), encoding="utf-8")
    r2 = [v["r2"] for v in old.values() if "r2" in v]
    print(f"[depth] {len(todo)} new, {len(old)} in manifest, {fb} fallbacks, "
          f"R2 median {np.median(r2) if r2 else float('nan'):.4f} min {min(r2) if r2 else float('nan'):.4f}; "
          f"{time.time() - t0:.0f}s -> {man}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
