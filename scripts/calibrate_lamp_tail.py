"""Fit the night lamp's spread tail to real night lights (TRAIN pohang01), 2026-10-10.

The revision-2 lamp (`corruptions_v2.glare` at night) was fitted on one property, the area of
its clipped blob (TRAIN q90 / q99 / max of the largest clipped light). Its spread was a
Lorentzian, I0 r0^2 / (r^2 + r0^2), whose r^-2 tail never reaches the frame's dark floor: at
matched blob size (~930 px) it lifted the 5th-percentile grey level above 10.5 on 100% of
night frames (median 16), where real night frames with a light that size stay below 10.5 on
100% (median 7). That is the `dark` statistic the shipped veto reads, so the tail decides the
glare/night cell.

This fits the tail exponent beta of I0 (1 + r^2 / r0^2)^-beta (beta = 1: the Lorentzian) and
the core radius r0 jointly on the real lights, with I0 re-solved for the blob-area target at
every (beta, r0). Real reference: TRAIN pohang01 frames (stride 2) whose largest clipped blob
falls in the severity's bin, measured around the blob's centroid:
  * ring means of linear grey radiance, rings 15-30 / 30-60 / 60-120 / 120-240 px (the glow);
  * ring floors, the 5th percentile of linear radiance, rings 30-60 ... 400-800 px (the veil
    cannot be brighter than the darkest pixels at that distance, plus the scene).
Model: the same statistics around the model blob's centroid on 96 TRAIN night frames that
have no light (largest clipped blob < 20 px), through the real `make_corruption` call.
Objective: RMS of log(model / real) over the nine numbers, s2 bin only (600-1300 px, the
best-sampled); s1 / s3 keep beta and the r0 ladder (x2/3, x4/3) and re-solve I0 for their
blob targets, and are reported as checks against their own bins.

    py -3.13 scripts/calibrate_lamp_tail.py --workers 24
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
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from calibrate_corruptions_v2 import train_list            # noqa: E402

_LIN = ((np.arange(256, dtype=np.float32) / 255.0) ** 2.2).astype(np.float32)
SAT = 250
RINGS = (0, 15, 30, 60, 120, 240, 400, 800)
MEAN_RINGS = (1, 2, 3, 4)            # 15-30, 30-60, 60-120, 120-240
FLOOR_RINGS = (2, 3, 4, 5, 6)        # 30-60 ... 400-800
BLOB_TARGET = {1: 100.0, 2: 927.0, 3: 1748.0}          # TRAIN night q90 / q99 / max (calibrate_glare_ae.py)
BIN = {1: (60, 200), 2: (600, 1300), 3: (1300, 10 ** 9)}
R0_LADDER = {1: 2.0 / 3.0, 2: 1.0, 3: 4.0 / 3.0}
_YY, _XX = np.mgrid[0:338, 0:640]


def ring_stats(g: np.ndarray, cx: float, cy: float) -> tuple[list, list]:
    R = np.hypot(_XX - cx, _YY - cy)
    L = _LIN[g]
    means, floors = [], []
    for a, b in zip(RINGS[:-1], RINGS[1:]):
        m = (R >= a) & (R < b)
        means.append(float(L[m].mean()) if m.sum() > 20 else np.nan)
        floors.append(float(np.percentile(L[m], 5)) if m.sum() > 20 else np.nan)
    return means, floors


def blob(g: np.ndarray, mask: np.ndarray | None = None):
    s = g >= SAT if mask is None else (g >= SAT) & mask
    n, _lab, st, cen = cv2.connectedComponentsWithStats(s.astype(np.uint8), connectivity=8)
    if n <= 1:
        return 0, None
    j = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA]))
    return int(st[j, cv2.CC_STAT_AREA]), (float(cen[j][0]), float(cen[j][1]))


def real_frame(p: Path) -> dict:
    g = cv2.cvtColor(cv2.imread(str(p))[151:489], cv2.COLOR_BGR2GRAY)
    a, c = blob(g)
    out = {"path": str(p), "blob": a, "p05": float(np.percentile(g.astype(np.float32), 5))}
    if a >= 40:
        out["mean"], out["floor"] = ring_stats(g, *c)
    return out


def model_stats(sev, I0, r0, beta, paths, full=True) -> dict:
    from uqfusion.eval.corruptions import make_corruption
    tf = make_corruption("glare", sev, 3, version="v2", modality="vis", images=paths,
                         params={"LAMP": {sev: (I0, r0)}, "LAMP_BETA": beta})
    areas, means, floors, p05 = [], [], [], []
    for i, p in enumerate(paths):
        im0 = cv2.imread(str(p))
        g0 = cv2.cvtColor(im0[151:489], cv2.COLOR_BGR2GRAY)
        g = cv2.cvtColor(tf(im0, i)[151:489], cv2.COLOR_BGR2GRAY)
        a, c = blob(g, g0 < SAT)
        areas.append(a)
        p05.append(float(np.percentile(g.astype(np.float32), 5)))
        if full and c is not None:
            m, f = ring_stats(g, *c)
            means.append(m)
            floors.append(f)
    out = {"blob": float(np.median(areas))}
    if full:
        out.update(mean=np.nanmedian(means, 0).tolist(), floor=np.nanmedian(floors, 0).tolist(),
                   p05=float(np.median(p05)), frac_p05_gt_10_5=float(np.mean(np.asarray(p05) > 10.5)))
    return out


def solve_I0(sev, r0, beta, paths) -> float:
    """Geometric bisection on I0 for the blob-area target (blob area is monotone in I0)."""
    lo, hi = 0.5, 1.0e5
    for _ in range(13):
        mid = float(np.sqrt(lo * hi))
        if model_stats(sev, mid, r0, beta, paths, full=False)["blob"] < BLOB_TARGET[sev]:
            lo = mid
        else:
            hi = mid
    return float(np.sqrt(lo * hi))


def err(model: dict, real: dict) -> float:
    a = [np.log(model["mean"][k] / real["mean"][k]) for k in MEAN_RINGS]
    b = [np.log(model["floor"][k] / real["floor"][k]) for k in FLOOR_RINGS]
    return float(np.sqrt(np.mean(np.square(a + b))))


def job(a):
    sev, r0, beta, solve_paths, eval_paths = a
    I0 = solve_I0(sev, r0, beta, solve_paths)
    return sev, r0, beta, I0, model_stats(sev, I0, r0, beta, eval_paths)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--out", default=str(ROOT / "docs/eval/corruption_v2_calibration/lamp_tail.json"))
    args = ap.parse_args()
    t0 = time.time()
    paths = [p for p in train_list("vis") if p.parent.name == "pohang01"][::2]
    with ProcessPoolExecutor(args.workers) as ex:
        R = list(ex.map(real_frame, paths, chunksize=16))
    real = {}
    for s, (lo, hi) in BIN.items():
        rows = [r for r in R if lo <= r["blob"] < hi]
        p05 = np.array([r["p05"] for r in rows])
        real[s] = {"n": len(rows), "blob_median": float(np.median([r["blob"] for r in rows])),
                   "mean": np.nanmedian([r["mean"] for r in rows], 0).tolist(),
                   "floor": np.nanmedian([r["floor"] for r in rows], 0).tolist(),
                   "p05": float(np.median(p05)), "frac_p05_gt_10_5": float(np.mean(p05 > 10.5))}
    all_p05 = np.array([r["p05"] for r in R])
    dark = [Path(r["path"]) for r in R if r["blob"] < 20]
    rng = np.random.default_rng(0)
    ev = [dark[i] for i in sorted(rng.choice(len(dark), 96, replace=False))]
    sv = ev[::3]
    print(f"[lamp] {len(R)} real night frames, {len(dark)} without a light; bins "
          + ", ".join(f"s{s} n={real[s]['n']}" for s in real) + f" ({time.time() - t0:.0f}s)", flush=True)

    grid = [(2, r0, b, sv, ev) for b in (1.0, 1.25, 1.5, 2.0, 2.5, 3.0) for r0 in (4.0, 6.0, 9.0, 12.0, 16.0, 24.0)]
    with ProcessPoolExecutor(args.workers) as ex:
        G = list(ex.map(job, grid))
    table = [{"beta": b, "r0": r0, "I0": I0, **m, "err": err(m, real[2])} for _s, r0, b, I0, m in G]
    table.sort(key=lambda t: t["err"])
    best = table[0]
    print(f"[lamp] s2 best beta {best['beta']} r0 {best['r0']} I0 {best['I0']:.3f} err {best['err']:.3f} "
          f"({time.time() - t0:.0f}s)", flush=True)
    for t in table[:8]:
        print(f"   beta {t['beta']:<5} r0 {t['r0']:<5} I0 {t['I0']:9.3f} blob {t['blob']:6.0f} err {t['err']:.3f} "
              f"p05 {t['p05']:5.1f} frac>10.5 {t['frac_p05_gt_10_5']:.3f}")
    lor = min((t for t in table if t["beta"] == 1.0), key=lambda t: t["err"])
    jobs = [(s, best["r0"] * R0_LADDER[s], best["beta"], sv, ev) for s in (1, 3)]
    with ProcessPoolExecutor(2) as ex:
        S = list(ex.map(job, jobs))
    sev_fit = {2: {"I0": best["I0"], "r0": best["r0"], **{k: best[k] for k in ("blob", "mean", "floor", "p05", "frac_p05_gt_10_5")},
                   "err": best["err"]}}
    for s, r0, b, I0, m in S:
        sev_fit[s] = {"I0": I0, "r0": r0, **m, "err": err(m, real[s])}
    for s in (1, 2, 3):
        f = sev_fit[s]
        print(f"[lamp] s{s}: I0 {f['I0']:.3f} r0 {f['r0']:.2f} blob {f['blob']:.0f} (target {BLOB_TARGET[s]:.0f}) "
              f"p05 {f['p05']:.1f} vs real {real[s]['p05']:.1f}; frac>10.5 {f['frac_p05_gt_10_5']:.3f} vs real "
              f"{real[s]['frac_p05_gt_10_5']:.3f}; err {f['err']:.3f}", flush=True)
    out = {"rings_px": RINGS, "mean_rings": MEAN_RINGS, "floor_rings": FLOOR_RINGS, "blob_target": BLOB_TARGET,
           "bins": BIN, "real": real, "real_all_night_frac_p05_gt_10_5": float(np.mean(all_p05 > 10.5)),
           "n_real": len(R), "n_dark_eval": len(ev), "grid_s2": table, "best_lorentzian_s2": lor,
           "fit": {"LAMP_BETA": best["beta"], "LAMP": {s: (sev_fit[s]["I0"], sev_fit[s]["r0"]) for s in (1, 2, 3)}},
           "by_severity": sev_fit, "elapsed_s": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=1, default=float), encoding="utf-8")
    print(f"[lamp] wrote {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
