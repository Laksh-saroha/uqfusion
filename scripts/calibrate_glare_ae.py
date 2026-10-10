"""Does this camera's auto-exposure react to glare, and how bright is a real sun? (2026-10-10)

Corruption v2 glare adds a veiling source in linear radiance. A real camera meters the scene,
so a bright source in frame lowers the exposure and darkens everything else. Whether this
camera does that, and how strongly, is measurable on the TRAIN split without labels:

* **AE strength.** Find day frames with a large saturated area (sun glitter on the water, the
  sun itself) and compare the brightness of the REST of the frame (pixels far from any
  saturated pixel) with temporally adjacent frames of the same run whose saturated area is
  small. A converged average-metering AE (strength 1) holds the frame's mean linear radiance
  constant, so its prediction for the rest-of-frame ratio is computable per event from the
  two frames' own statistics; no AE predicts ratio 1. The fitted exponent is the slope of
  log(observed ratio) on log(predicted ratio).
* **Sun / glint intensity.** The saturated-pixel fraction and the largest compact saturated
  blob above the horizon of real sun-in-frame and glitter frames, to set the glare
  severities' intensity in units of what the camera actually records.

Writes docs/eval/corruption_v2_calibration/glare_ae.json. TRAIN only; pohang04 excluded.

    py -3.13 scripts/calibrate_glare_ae.py --workers 28
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
sys.path.insert(0, str(ROOT / "scripts"))

from calibrate_corruptions_v2 import train_list                       # noqa: E402
from uqfusion.eval.frame_geometry import FrameGeometry, content_rows   # noqa: E402

_LIN = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
SAT = 250            # all three channels at or above this = clipped
KEEP_OUT = 25        # px: pixels this close to a clipped pixel carry its veil, not the exposure


def frame(path: Path) -> dict | None:
    im = cv2.imread(str(path))
    lo, hi = content_rows("vis")
    c = im[lo:hi]
    sat = c.min(axis=2) >= SAT
    lin = _LIN[c].mean(axis=2)
    out = {"path": str(path), "run": path.parent.name, "idx": int(path.stem.split("_")[-1]),
           "sat_frac": float(sat.mean()), "lin_mean": float(lin.mean())}
    if sat.any():
        far = cv2.distanceTransform((~sat).astype(np.uint8), cv2.DIST_L2, 3) > KEEP_OUT
    else:
        far = np.ones_like(sat)
    out["far_frac"] = float(far.mean())
    out["far_lin_mean_map"] = None
    # rest-of-frame brightness on a fixed 8x8 grid of cells, so two frames are compared on
    # the same pixels (cells with too few far pixels are NaN)
    h, w = sat.shape
    cells = np.full((8, 8), np.nan)
    for i in range(8):
        for j in range(8):
            sl = (slice(i * h // 8, (i + 1) * h // 8), slice(j * w // 8, (j + 1) * w // 8))
            f = far[sl]
            if f.mean() > 0.8:
                cells[i, j] = float(lin[sl][f].mean())
    out["cells"] = cells.tolist()
    try:
        hz = FrameGeometry(path, "vis").horizon_rows()
        rows = np.arange(h)[:, None]
        above = rows < hz[None, :]
        out["sat_above"] = float((sat & above).mean())
        out["sat_below"] = float((sat & ~above).mean())
        n, lab, st, _ = cv2.connectedComponentsWithStats((sat & above).astype(np.uint8), connectivity=8)
        out["max_blob_above_px"] = int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0
    except KeyError:
        pass
    return out


# ---------------------------------------------------------------- fits (2026-10-10)
SUN_EVENT = {"before": [f"pohang00_L_{i:06d}" for i in range(18884, 18944, 6)],
             "after": [f"pohang00_L_{i:06d}" for i in range(18980, 19002, 2)]}
METER_KEYS = ("mean", "median", "mean_nonsat", "trim90", "center_w", "logmean")


def _vis(stem: str) -> Path:
    return ROOT / "Pohang_dataset/visible/images" / stem.split("_")[0] / f"{stem}.png"


def _obs(ims) -> tuple[float, float, float]:
    sat, mean, ns = [], [], []
    for c in ims:
        g = cv2.cvtColor(c[151:489], cv2.COLOR_BGR2GRAY)
        L, s = _LIN[g], g >= SAT
        sat.append(s.mean()); mean.append(L.mean()); ns.append(L[~s].mean())
    return float(np.mean(sat)), float(np.mean(mean)), float(np.mean(ns))


def meter_stats(path) -> dict:
    """Candidate AE metering statistics of one frame (linear grey)."""
    g = cv2.cvtColor(cv2.imread(str(path))[151:489], cv2.COLOR_BGR2GRAY)
    L = _LIN[g].astype(np.float64)
    sat = g >= SAT
    srt = np.sort(L.ravel())
    h, w = g.shape
    cw = np.exp(-(((np.arange(h)[:, None] - h / 2) / (h / 3)) ** 2 + ((np.arange(w)[None] - w / 2) / (w / 3)) ** 2))
    return {"mean": L.mean(), "median": float(np.median(L)), "mean_nonsat": L[~sat].mean() if (~sat).any() else np.nan,
            "trim90": srt[: int(0.9 * srt.size)].mean(), "center_w": float((L * cw).sum() / cw.sum()),
            "logmean": float(np.exp(np.log(L + 1e-4).mean()))}


RADIAL_BINS = (0, 40, 80, 120, 180, 250, 350, 800)


def _glow_centre() -> tuple[float, float]:
    """Centroid of the clipped pixels in the upper content rows of the event's 'after' frames."""
    cs = []
    for x in SUN_EVENT["after"]:
        g = cv2.cvtColor(cv2.imread(str(_vis(x)))[151:489], cv2.COLOR_BGR2GRAY)
        ys, xs = np.nonzero(g >= SAT)
        top = ys < 200
        cs.append((float(xs[top].mean()), float(ys[top].mean())))
    return tuple(np.mean(cs, axis=0))


def _profile(ims, cxy) -> tuple[np.ndarray, float]:
    """Mean linear radiance in rings around the glow centre, and the clipped fraction."""
    yy, xx = np.mgrid[0:338, 0:640]
    R = np.hypot(xx - cxy[0], yy - cxy[1])
    out, clip = [], []
    for c in ims:
        g = cv2.cvtColor(c[151:489], cv2.COLOR_BGR2GRAY)
        L = _LIN[g]
        out.append([L[(R >= a) & (R < b)].mean() for a, b in zip(RADIAL_BINS[:-1], RADIAL_BINS[1:])])
        clip.append((g >= SAT).mean())
    return np.mean(out, axis=0), float(np.mean(clip))


def _sun_job(a):
    from uqfusion.eval.corruptions import make_corruption
    I0, r0, halo, s, cxy = a
    before = [_vis(x) for x in SUN_EVENT["before"]]
    tf = make_corruption("glare", 2, 5, version="v2", modality="vis", images=before,
                         params={"SUN": {2: (I0, r0)}, "HALO_FRAC": halo, "AE_STRENGTH": s, "SRC_XY": cxy})
    return I0, r0, halo, s, _profile([tf(cv2.imread(str(p)), i) for i, p in enumerate(before)], cxy)


def _lamp_job(a):
    from uqfusion.eval.corruptions import make_corruption
    I0, r0, paths = a
    tf = make_corruption("glare", 2, 3, version="v2", modality="vis", images=paths, params={"LAMP": {2: (I0, r0)}})
    areas = []
    for i, p in enumerate(paths):
        im0 = cv2.imread(str(p))
        g0 = cv2.cvtColor(im0[151:489], cv2.COLOR_BGR2GRAY)
        g = cv2.cvtColor(tf(im0, i)[151:489], cv2.COLOR_BGR2GRAY)
        n, _lab, st, _c = cv2.connectedComponentsWithStats(((g >= SAT) & (g0 < SAT)).astype(np.uint8), connectivity=8)
        areas.append(int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0)
    return I0, r0, float(np.median(areas))


def _night_blob(path) -> int:
    g = cv2.cvtColor(cv2.imread(str(path))[151:489], cv2.COLOR_BGR2GRAY)
    n, _lab, st, _c = cv2.connectedComponentsWithStats((g >= SAT).astype(np.uint8), connectivity=8)
    return int(st[1:, cv2.CC_STAT_AREA].max()) if n > 1 else 0


def fits(ex, day_paths) -> dict:
    """AE metering statistic, AE strength + sun intensity (one real sun-entry event), and the
    night lamp ladder (real TRAIN night light sources)."""
    out = {}
    T = list(ex.map(meter_stats, day_paths[::5], chunksize=16))
    out["metering_sd_log"] = {k: float(np.nanstd(np.log([t[k] for t in T]))) for k in METER_KEYS}
    out["metering_n"] = len(T)
    # The event, as a radial profile around the real glow (an aggregate fit -- clipped area,
    # frame mean, unclipped mean -- was tried first and is NOT identifying: it preferred a far
    # veil plus a strong AE, i.e. a far field 40x darker, where the real far field did not darken).
    cxy = _glow_centre()
    B = [cv2.imread(str(_vis(x))) for x in SUN_EVENT["before"]]
    pb, _ = _profile(B, cxy)
    pa, ca = _profile([cv2.imread(str(_vis(x))) for x in SUN_EVENT["after"]], cxy)
    real = np.log(pa / pb)
    out["sun_event"] = {**SUN_EVENT, "glow_centre_xy": list(cxy), "radial_bins_px": list(RADIAL_BINS),
                        "ratio_by_ring": (pa / pb).tolist(), "clipped": ca}
    grid = [(I0, r0, h, s, cxy) for I0 in (2, 5, 10, 20, 40, 80, 160) for r0 in (10.0, 20.0, 40.0, 80.0)
            for h in (0.0, 0.003, 0.01, 0.03) for s in (0.0, 0.25, 0.5, 0.75, 0.875, 1.0)]
    res = []
    for I0, r0, h, s, (pm, cm) in ex.map(_sun_job, grid):
        e = float(((np.log(pm / pb) - real) ** 2).sum() + np.log(max(cm, 1e-4) / ca) ** 2)
        res.append({"I0": I0, "r0": r0, "HALO_FRAC": h, "AE_STRENGTH": s, "clipped": cm,
                    "ratio_by_ring": (pm / pb).tolist(), "err": e})
    best = min(res, key=lambda r: r["err"])
    out["sun_fit"] = {**best, "best_err_by_ae": {str(s): min(r["err"] for r in res if r["AE_STRENGTH"] == s)
                                                 for s in sorted({r["AE_STRENGTH"] for r in res})},
                      "next_best": sorted(res, key=lambda r: r["err"])[1:6]}
    night = [p for p in train_list("vis") if p.parent.name == "pohang01"][::4]
    a = np.array(list(ex.map(_night_blob, night, chunksize=16)))
    tgt = {1: float(np.quantile(a, 0.90)), 2: float(np.quantile(a, 0.99)), 3: float(a.max())}
    out["night_lights"] = {"n": len(a), "largest_clipped_blob_px": {"q50": float(np.median(a)), "q90": tgt[1],
                                                                     "q99": tgt[2], "max": tgt[3]}}
    sub = night[:: max(1, len(night) // 40)][:40]
    lamp = {}
    for sev, r0 in ((1, 6.0), (2, 9.0), (3, 12.0)):          # core radius ladder: ASSUMED
        I0s = np.geomspace(0.5, 40.0, 41)
        got = list(ex.map(_lamp_job, [(float(I0), r0, sub) for I0 in I0s]))
        areas = np.array([g[2] for g in got])
        i = int(np.argmin(np.abs(np.log(np.maximum(areas, 1) / tgt[sev]))))
        lamp[sev] = {"I0": float(I0s[i]), "r0": r0, "target_px": tgt[sev], "model_px": float(areas[i])}
    out["lamp_fit"] = lamp
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=28)
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--out", default=str(ROOT / "docs/eval/corruption_v2_calibration/glare_ae.json"))
    args = ap.parse_args()
    paths = [p for p in train_list("vis") if p.parent.name not in ("pohang01", "pohang04")][::args.stride]
    t = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        rows = [r for r in ex.map(frame, paths, chunksize=16) if r]
    print(f"[ae] {len(rows)} TRAIN day frames scanned ({time.time() - t:.0f}s)", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        fitted = fits(ex, paths)
    print(json.dumps(fitted, indent=1), flush=True)

    sf = np.array([r["sat_frac"] for r in rows])
    q = {k: float(np.quantile(sf, k)) for k in (0.5, 0.9, 0.99, 0.999)}
    by_run: dict[str, dict[int, dict]] = {}
    for r in rows:
        by_run.setdefault(r["run"], {})[r["idx"]] = r

    # events: frames in the top 1% of saturated fraction (and >= 0.5%), each paired with the
    # nearest frame within +-50 of the same run whose saturated fraction is <= the median
    thr = max(q[0.99], 0.005)
    events = []
    for r in rows:
        if r["sat_frac"] < thr:
            continue
        run = by_run[r["run"]]
        best = None
        for d in range(5, 51):
            for k in (r["idx"] - d, r["idx"] + d):
                o = run.get(k)
                if o is not None and o["sat_frac"] <= q[0.5]:
                    best = o
                    break
            if best:
                break
        if best is None:
            continue
        a, b = np.asarray(r["cells"]), np.asarray(best["cells"])
        ok = np.isfinite(a) & np.isfinite(b) & (b > 1e-3)
        if ok.sum() < 16:
            continue
        observed = float(np.median(a[ok] / b[ok]))
        # strength-1 prediction: the AE holds the whole-frame mean; the event frame carries
        # extra radiance in its clipped pixels (>= 1.0 each, a lower bound on what is there)
        # and in their veil, so the rest of the frame must darken by mean_ref / mean_event
        # where mean_event counts the event's clipped area at the ref frame's level plus
        # its excess. Use the observed whole-frame means directly: if AE held the mean,
        # lin_mean(event) == lin_mean(ref) and the rest darkens; with no AE, the rest is
        # unchanged and the mean rises.
        events.append({"run": r["run"], "idx": r["idx"], "ref_idx": best["idx"], "sat_frac": r["sat_frac"],
                       "sat_above": r.get("sat_above"), "sat_below": r.get("sat_below"),
                       "rest_ratio": observed, "mean_ratio": r["lin_mean"] / max(best["lin_mean"], 1e-6),
                       "n_cells": int(ok.sum())})
    ev = events
    rr = np.array([e["rest_ratio"] for e in ev]) if ev else np.zeros(0)
    mr = np.array([e["mean_ratio"] for e in ev]) if ev else np.zeros(0)
    sfe = np.array([e["sat_frac"] for e in ev]) if ev else np.zeros(0)
    # Under no AE, rest_ratio ~ 1 and the frame mean rises with the clipped area. Under a
    # converged mean-holding AE, mean_ratio ~ 1 and rest_ratio < 1. Report both and the
    # regression of log rest_ratio on the clipped fraction.
    fit = None
    if len(ev) >= 10:
        A = np.stack([np.ones(len(ev)), sfe], 1)
        coef, *_ = np.linalg.lstsq(A, np.log(rr), rcond=None)
        fit = {"log_rest_ratio_intercept": float(coef[0]), "slope_per_unit_sat_frac": float(coef[1])}
    blobs = np.array([r.get("max_blob_above_px", 0) for r in rows])
    out = {"fits": fitted, "n_frames": len(rows), "sat_frac_quantiles": q, "event_threshold": thr, "n_events": len(ev),
           "rest_ratio": {k: float(np.quantile(rr, k)) for k in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(ev) else None,
           "mean_ratio": {k: float(np.quantile(mr, k)) for k in (0.1, 0.25, 0.5, 0.75, 0.9)} if len(ev) else None,
           "fit": fit, "events": ev[:500],
           "sun_blob_above_horizon": {"frames_with_blob_ge_50px": int((blobs >= 50).sum()),
                                      "frames_with_blob_ge_200px": int((blobs >= 200).sum()),
                                      "blob_px_quantiles_when_ge_50": ({k: float(np.quantile(blobs[blobs >= 50], k))
                                                                        for k in (0.25, 0.5, 0.75, 0.95)}
                                                                       if (blobs >= 50).any() else None)},
           "sat_frac_top_frames": sorted(({"path": r["path"], "sat_frac": r["sat_frac"], "sat_above": r.get("sat_above"),
                                           "max_blob_above_px": r.get("max_blob_above_px")} for r in rows),
                                         key=lambda x: -x["sat_frac"])[:40]}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).with_name("glare_ae_frames.json").write_text(json.dumps(
        [{k: r.get(k) for k in ("path", "sat_frac", "sat_above", "sat_below", "max_blob_above_px", "lin_mean")}
         for r in rows]), encoding="utf-8")
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_frames", "sat_frac_quantiles", "n_events", "rest_ratio", "mean_ratio",
                                          "fit", "sun_blob_above_horizon")}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
