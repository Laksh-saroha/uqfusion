"""Measure the real-image statistics that corruption v2 is calibrated against (2026-10-09).

TRAIN split only, so no evaluation frame tunes a corruption. Nothing is scored and no
detector runs; GT boxes are read only to measure how far away the targets are (which sets
what a given fog visibility does to them) and how much the flat-sea range map
over-states a vessel's range. The corruption itself never reads a label.

Writes `docs/eval/corruption_v2_calibration/calibration.json`:

* `vis_night` / `vis_day`: per-frame content statistics (mean, black fraction, p05/p50/p95),
  the noise level function (sigma of the sensor noise vs local intensity, robust MAD of an
  Immerkaer residual on flat pixels), and the sky band just above the horizon.
* `ir_day` / `ir_night`: the same for thermal, plus column fixed-pattern noise.
* `boxes`: flat-sea range at each GT box's bottom centre, and the range bias inside boxes.

    py -3.13 scripts/calibrate_corruptions_v2.py --workers 24
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.frame_geometry import FrameGeometry, content_rows   # noqa: E402

DS = ROOT / "Pohang_dataset"
NIGHT_RUN = "pohang01"
BINS = np.array([0, 2, 4, 8, 16, 32, 64, 128, 256], float)
M = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32) / 6.0   # iid noise sigma -> sigma


def train_list(modality: str) -> list[Path]:
    sub = "visible" if modality == "vis" else "infrared"
    base = DS / sub
    out = []
    for ln in (base / "train.txt").read_text(encoding="utf-8").splitlines():
        ln = ln.strip()
        if ln:
            p = Path(ln)
            out.append(p if p.is_absolute() else (base / p).resolve())
    return out


def nlf(gray: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Noise level per intensity bin: (count, sigma) from flat pixels only."""
    g = gray.astype(np.float32)
    r = cv2.filter2D(g, cv2.CV_32F, M, borderType=cv2.BORDER_REFLECT)
    mu = cv2.blur(g, (5, 5))
    gx = cv2.Sobel(mu, cv2.CV_32F, 1, 0)
    gy = cv2.Sobel(mu, cv2.CV_32F, 0, 1)
    flat = np.hypot(gx, gy) < 4.0 + 0.05 * mu
    cnt, sig = np.zeros(len(BINS) - 1), np.full(len(BINS) - 1, np.nan)
    b = np.digitize(mu, BINS) - 1
    for k in range(len(BINS) - 1):
        sel = flat & (b == k)
        n = int(sel.sum())
        if n >= 200:
            cnt[k], sig[k] = n, 1.4826 * float(np.median(np.abs(r[sel])))
    return cnt, sig


_LIN = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)


def channel_noise_factor(c: np.ndarray, g: np.ndarray) -> float | None:
    """mean per-channel noise variance / grey noise variance on flat pixels. 2.24 if the
    channels' noise is independent, 1.0 if it is fully correlated (demosaicing and the
    2048 -> 640 resize correlate it)."""
    mu = cv2.blur(g.astype(np.float32), (5, 5))
    flat = np.hypot(cv2.Sobel(mu, cv2.CV_32F, 1, 0), cv2.Sobel(mu, cv2.CV_32F, 0, 1)) < 4.0 + 0.05 * mu
    flat &= (mu > 2) & (mu < 250)
    if flat.sum() < 2000:
        return None
    sg = 1.4826 * np.median(np.abs(cv2.filter2D(g.astype(np.float32), cv2.CV_32F, M)[flat]))
    sc = [1.4826 * np.median(np.abs(cv2.filter2D(c[..., k].astype(np.float32), cv2.CV_32F, M)[flat])) for k in range(3)]
    return float(np.mean(np.square(sc)) / max(sg * sg, 1e-9))


def frame(args) -> dict:
    path, modality, want_boxes = args
    im = cv2.imread(str(path))
    lo, hi = content_rows(modality)
    c = im[lo:hi]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    out = {"mean": float(g.mean()), "frac_black": float((g <= 2).mean()),
           "p05": float(np.percentile(g, 5)), "p50": float(np.percentile(g, 50)),
           "p95": float(np.percentile(g, 95)), "rgb_mean": c.reshape(-1, 3).mean(0)[::-1].tolist()}
    cnt, sig = nlf(g)
    out["lin_mean"] = float(_LIN[c].mean())          # what an average-metering AE holds constant
    if modality == "vis":
        out["chan_noise_factor"] = channel_noise_factor(c, g)
    out["nlf_cnt"], out["nlf_sig"] = cnt.tolist(), sig.tolist()
    if modality == "ir":
        prof = g.astype(np.float32).mean(axis=0)
        out["col_fpn"] = float(np.std(prof - cv2.blur(prof[None], (31, 1))[0]))
    try:
        geo = FrameGeometry(path, modality)
    except KeyError:
        return out
    hz = geo.horizon_rows()
    h = g.shape[0]
    hz_med = float(np.median(hz))
    out["horizon_row"] = hz_med
    top = int(np.clip(hz_med - 40, 0, h)), int(np.clip(hz_med - 4, 0, h))
    if top[1] - top[0] >= 8:
        band = g[top[0]:top[1]]
        out["sky_band_p50"], out["sky_band_p90"] = float(np.median(band)), float(np.percentile(band, 90))
    out["sea_p50"] = float(np.median(g[int(np.clip(hz_med + 20, 0, h)):])) if hz_med + 20 < h else None
    if want_boxes:
        lab = DS / ("visible" if modality == "vis" else "infrared") / "labels" / path.parent.name / (path.stem + ".txt")
        if lab.is_file():
            rng = geo.range_m(far_m=np.inf)
            boxes = []
            for ln in lab.read_text().splitlines():
                p = ln.split()
                if len(p) < 5:
                    continue
                cls, xc, yc, bw, bh = int(p[0]), *map(float, p[1:5])
                x0, x1 = int(np.clip((xc - bw / 2) * 640, 0, 639)), int(np.clip((xc + bw / 2) * 640, 0, 639))
                y0, y1 = (yc - bh / 2) * 640 - lo, (yc + bh / 2) * 640 - lo
                y0i, y1i = int(np.clip(y0, 0, h - 1)), int(np.clip(y1, 0, h - 1))
                xm = int(np.clip(xc * 640, 0, 639))
                d_bottom = float(rng[y1i, xm])
                patch = rng[y0i:y1i + 1, x0:x1 + 1]
                boxes.append({"cls": cls, "h_px": float(bh * 640), "w_px": float(bw * 640),
                              "range_bottom": d_bottom if np.isfinite(d_bottom) else None,
                              "frac_above_horizon": float(np.mean(~np.isfinite(patch))),
                              "below_horizon_px": float(hz[xm] - y1) * -1.0})
            out["boxes"] = boxes
    return out


def summarise(rows: list[dict]) -> dict:
    s = {}
    for k in ("mean", "frac_black", "p05", "p50", "p95", "sky_band_p50", "sky_band_p90", "sea_p50",
              "col_fpn", "horizon_row", "lin_mean", "chan_noise_factor"):
        v = np.array([r[k] for r in rows if r.get(k) is not None], float)
        if len(v):
            s[k] = {"mean": float(v.mean()), "q05": float(np.quantile(v, 0.05)), "q25": float(np.quantile(v, 0.25)),
                    "q50": float(np.median(v)), "q75": float(np.quantile(v, 0.75)), "q95": float(np.quantile(v, 0.95)),
                    "n": int(len(v))}
    cnt = np.array([r["nlf_cnt"] for r in rows])
    sig = np.array([r["nlf_sig"] for r in rows])
    w = np.where(np.isnan(sig), 0, cnt)
    tot = w.sum(0)
    s["nlf"] = {"bins": BINS.tolist(),
                "sigma": [float(np.nansum(np.nan_to_num(sig[:, k]) * w[:, k]) / tot[k]) if tot[k] else None
                          for k in range(len(BINS) - 1)],
                "pixels": tot.tolist()}
    s["rgb_mean"] = np.mean([r["rgb_mean"] for r in rows], axis=0).tolist()
    return s


def box_summary(rows: list[dict]) -> dict:
    b = [x for r in rows for x in r.get("boxes", [])]
    d = np.array([x["range_bottom"] for x in b if x["range_bottom"] is not None])
    fa = np.array([x["frac_above_horizon"] for x in b])
    return {"n_boxes": len(b), "frac_bottom_at_or_above_horizon": float(np.mean([x["range_bottom"] is None for x in b])),
            "range_bottom_m": {q: float(np.quantile(d, float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")},
            "frac_boxes_with_pixels_above_horizon": float(np.mean(fa > 0)),
            "mean_frac_box_pixels_above_horizon": float(fa.mean()),
            "box_h_px_median": float(np.median([x["h_px"] for x in b]))}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--n", type=int, default=800, help="frames per (modality, day/night) group")
    ap.add_argument("--out", default=str(ROOT / "docs/eval/corruption_v2_calibration/calibration.json"))
    args = ap.parse_args()
    rng = np.random.default_rng(0)
    jobs, groups = [], {}
    for mod in ("vis", "ir"):
        tl = train_list(mod)
        for part, sel in (("night", [p for p in tl if p.parent.name == NIGHT_RUN]),
                          ("day", [p for p in tl if p.parent.name not in (NIGHT_RUN, "pohang04")])):
            pick = [sel[i] for i in sorted(rng.choice(len(sel), min(args.n, len(sel)), replace=False))]
            groups[f"{mod}_{part}"] = (len(jobs), len(jobs) + len(pick))
            jobs += [(p, mod, part == "day") for p in pick]
    t = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        res = list(ex.map(frame, jobs, chunksize=8))
    out = {"meta": {"split": "train", "night_run": NIGHT_RUN, "n_per_group": args.n, "seed": 0,
                    "excluded": "pohang04 (spent holdout)", "numpy": np.__version__, "opencv": cv2.__version__}}
    for g, (a, b) in groups.items():
        out[g] = summarise(res[a:b])
        if g.endswith("_day"):
            out[g]["boxes"] = box_summary(res[a:b])
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({g: {k: out[g][k] for k in ("mean", "frac_black", "nlf") if k in out[g]} for g in groups}, indent=1))
    for g in groups:
        if "boxes" in out[g]:
            print(g, json.dumps(out[g]["boxes"]))
    print(f"[calib] {len(jobs)} frames in {time.time() - t:.0f}s -> {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
