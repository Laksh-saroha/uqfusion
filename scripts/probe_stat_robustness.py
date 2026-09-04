"""Which photometric statistic resists being spoofed by added luminance? — TODO §0.2 follow-up.

**The failure this answers.** The `mean`-intensity gate term regressed glare/night
(0.0728 -> 0.0720) because a sun flare ADDS light: it lifts the night run's mean
from 8.2 to 40.7, so `r_bright` reads 0.318 instead of 0.060 on frames where VIS
still scores 0.0000. Fog is worse -- it lifts the same frames to 87.8. Any
statistic that can be raised by adding light can be fooled into calling a blind
frame usable.

**How this measures it without touching the held-out run.** Take FIT-RUN images
only (pohang00/02/03, all daylight), darken them synthetically, then add glare or
fog on top. A robust statistic stays low through the second step; a spoofable one
climbs back toward its clean value. No detector is involved -- this is pure image
arithmetic on the statistic itself -- and pohang01 is never read, so this cannot
leak the selection.

Reported per statistic:

    spoof_frac = (stat_after_added_light - stat_dark) / (stat_clean - stat_dark)

0.0 = fully robust (added light does not move it at all)
1.0 = fully spoofed (added light restores the clean-daylight reading)

Usage:
    python scripts/probe_stat_robustness.py --n 120
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.corruptions import make_corruption
sys.path.insert(0, str(Path(__file__).resolve().parent))
from frame_brightness import content_rows, frame_stats

FIT_RUNS = ("pohang00", "pohang02", "pohang03")
STATS = ("mean", "p05", "p50", "std", "range", "frac_dark")


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache", default="runs/cache/ladder/vis_clean.pkl")
    parser.add_argument("--n", type=int, default=120, help="fit-run frames to sample")
    parser.add_argument("--dark-severity", type=int, default=2)
    parser.add_argument("--seed", type=int, default=7, help="NOT 1 or 2 — avoids both eval and tuning seeds")
    args = parser.parse_args()
    load_config(args.config)

    recs, _ = load_cache(args.cache)
    idx = [i for i, r in enumerate(recs) if Path(r["image_path"]).parent.name in FIT_RUNS]
    idx = list(np.asarray(idx)[np.unique(np.linspace(0, len(idx) - 1, args.n).round().astype(int))])
    lo, hi = content_rows("vis")
    print(f"[probe] {len(idx)} fit-run frames ({'+'.join(FIT_RUNS)}), pohang01 never read")

    darken = make_corruption("lowlight", args.dark_severity, args.seed)
    adders = {"glare": make_corruption("glare", 2, args.seed),
              "fog": make_corruption("fog", 2, args.seed)}

    acc = {k: {s: [] for s in STATS} for k in ("clean", "dark", *adders)}
    for j, i in enumerate(idx):
        im = cv2.imread(recs[i]["image_path"])          # READ-ONLY
        dark = darken(im, j)
        for name, img in (("clean", im), ("dark", dark),
                          *[(k, f(dark, j)) for k, f in adders.items()]):
            st = frame_stats(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)[lo:hi])
            for s in STATS:
                acc[name][s].append(st[s])

    m = {k: {s: float(np.mean(v)) for s, v in d.items()} for k, d in acc.items()}

    print(f"\n[probe] statistic readings (mean over {len(idx)} frames)")
    print(f"{'stat':10s} {'clean':>9s} {'dark':>9s} {'dark+glare':>11s} {'dark+fog':>10s}")
    for s in STATS:
        print(f"{s:10s} {m['clean'][s]:9.3f} {m['dark'][s]:9.3f} "
              f"{m['glare'][s]:11.3f} {m['fog'][s]:10.3f}")

    print(f"\n[probe] spoof_frac — how far added light drags the statistic back to its clean value")
    print(f"{'stat':10s} {'glare':>9s} {'fog':>9s}   (0.000 = robust, 1.000 = fully spoofed)")
    rows = []
    for s in STATS:
        span = m["clean"][s] - m["dark"][s]
        if abs(span) < 1e-6:
            continue
        g = (m["glare"][s] - m["dark"][s]) / span
        f = (m["fog"][s] - m["dark"][s]) / span
        rows.append((s, g, f, max(abs(g), abs(f))))
        print(f"{s:10s} {g:9.3f} {f:9.3f}")

    rows.sort(key=lambda r: r[3])
    print(f"\n[probe] most robust first (by worst-case |spoof_frac|):")
    for s, g, f, w in rows:
        print(f"   {s:10s} worst {w:6.3f}")
    print(f"\n[probe] selection is on FIT RUNS ONLY — pohang01 was not read.")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
