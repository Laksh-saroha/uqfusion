"""Refit the day sun with a free spread-tail exponent (SUN_BETA), 2026-10-10.

`calibrate_glare_ae.py` fitted the sun as a Lorentzian, I0 r0^2 / (r^2 + r0^2), on the one real
sun-entry event in TRAIN (pohang00 L 18884-19000; radiance after/before in rings around the
glow centre). Its best fit overshoots the 250-350 px ring (x1.28 against the real x1.00), the
heavy-tail symptom that `calibrate_lamp_tail.py` found at night. This repeats the same radial
fit with the spread I0 (1 + r^2 / r0^2)^-beta, beta in {1, 1.5, 2, 3}, AE strength in
{0, 0.5, 1}, no halo, and the glint held at its revision-2 scale (SUN_GLINT), so that a
steeper tail cannot buy its fit through a brighter reflection. Same objective as the original
fit: squared log error of the seven ring ratios plus the log clipped-fraction error.

    py -3.13 scripts/calibrate_sun_tail.py --workers 20
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

from calibrate_glare_ae import RADIAL_BINS, SUN_EVENT, _glow_centre, _profile, _vis   # noqa: E402

BETAS = (1.0, 1.5, 2.0, 3.0)
I0S = (5, 10, 20, 40, 80, 160, 320, 640, 1280, 2560, 5120, 10240)
R0S = (5.0, 10.0, 20.0, 40.0)
AES = (0.0, 0.5, 1.0)


def job(a):
    from uqfusion.eval.corruptions import make_corruption
    I0, r0, beta, s, cxy = a
    before = [_vis(x) for x in SUN_EVENT["before"]]
    tf = make_corruption("glare", 2, 5, version="v2", modality="vis", images=before,
                         params={"SUN": {2: (I0, r0)}, "SUN_BETA": beta, "HALO_FRAC": 0.0,
                                 "AE_STRENGTH": s, "SRC_XY": cxy})
    return I0, r0, beta, s, _profile([tf(cv2.imread(str(p)), i) for i, p in enumerate(before)], cxy)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workers", type=int, default=20)
    ap.add_argument("--out", default=str(ROOT / "docs/eval/corruption_v2_calibration/sun_tail.json"))
    ap.add_argument("--fine", action="store_true", help="second pass around the coarse optimum")
    args = ap.parse_args()
    global BETAS, I0S, R0S, AES
    if args.fine:
        BETAS, I0S = (1.25, 1.5, 1.75), (10, 14, 20, 28, 40, 56)
        R0S, AES = (20.0, 30.0, 40.0, 60.0), (0.0, 0.25, 0.375, 0.5, 0.625, 0.75)
    t0 = time.time()
    cxy = _glow_centre()
    pb, _ = _profile([cv2.imread(str(_vis(x))) for x in SUN_EVENT["before"]], cxy)
    pa, ca = _profile([cv2.imread(str(_vis(x))) for x in SUN_EVENT["after"]], cxy)
    real = np.log(pa / pb)
    grid = [(I0, r0, b, s, cxy) for b in BETAS for I0 in I0S for r0 in R0S for s in AES]
    res = []
    with ProcessPoolExecutor(args.workers) as ex:
        for I0, r0, b, s, (pm, cm) in ex.map(job, grid):
            e = float(((np.log(pm / pb) - real) ** 2).sum() + np.log(max(cm, 1e-4) / ca) ** 2)
            res.append({"I0": I0, "r0": r0, "SUN_BETA": b, "AE_STRENGTH": s, "clipped": cm,
                        "ratio_by_ring": (pm / pb).tolist(), "err": e})
    res.sort(key=lambda r: r["err"])
    best = res[0]
    by_beta = {str(b): min((r for r in res if r["SUN_BETA"] == b), key=lambda r: r["err"]) for b in BETAS}
    by_ae = {str(s): min(r["err"] for r in res if r["AE_STRENGTH"] == s) for s in AES}
    print(f"[sun] real ratio {np.round(pa / pb, 2).tolist()} clipped {ca:.3f}")
    for b, r in by_beta.items():
        print(f"[sun] beta {b}: I0 {r['I0']} r0 {r['r0']} AE {r['AE_STRENGTH']} err {r['err']:.3f} "
              f"ratio {np.round(r['ratio_by_ring'], 2).tolist()} clipped {r['clipped']:.3f}")
    print(f"[sun] best err by AE: {by_ae}")
    out = {"event": {**SUN_EVENT, "glow_centre_xy": list(cxy), "radial_bins_px": list(RADIAL_BINS),
                     "ratio_by_ring": (pa / pb).tolist(), "clipped": ca},
           "best": best, "best_by_beta": by_beta, "best_err_by_ae": by_ae, "top10": res[:10],
           "elapsed_s": time.time() - t0}
    Path(args.out).write_text(json.dumps(out, indent=1), encoding="utf-8")
    print(f"[sun] wrote {args.out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
