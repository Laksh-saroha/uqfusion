"""Fit `tau_lap`, the veil/blur veto threshold, and write it into brightness_constants.json.

**Why a second veto term.** `fit_brightness_gate.py` fits the photometric axis:
"did photons reach the sensor?". That axis is blind to a veil. Measured on the
paired val set, fog/day carries the HIGHEST p05 of all eight cells (58, against
clean/day's 35) while VIS ship AP collapses from 0.3683 to 0.0020 -- so the
photometric veto never fires on the one day condition where VIS is dead, and
fusing the dead stream costs -0.0051 against `ir_only`. Fog raises brightness and
destroys edges in the same stroke; brightness and structure are near-orthogonal
here, and the gate needs both.

**Fit protocol, and why it is not tuned on the answer.** The threshold is a
NOVELTY bound on CLEAN FIT-RUN frames only -- pohang00/02/03, the same runs
`fit_brightness_gate.py` used, with pohang01 held out. No fog, blur, or rain frame
informs it. The rule is "this frame carries less edge structure than the clean
frames we fitted on, so the detector's clean-val capability does not transfer" --
a statement about the reference distribution, not about fog. That it then catches
fog is a result, not a fitting target.

`lap_var` (variance of the Laplacian) is the standard focus/blur measure. It is
not a fog detector: defocus, heavy blur, rain smear, and a dirty lens all land in
the same place, which is the intended generality.

**Why the minimum and not a quantile.** A 1% quantile of the clean fit frames sits
at 611.9, and that fires on 7.0% of the clean/day guard cell -- not because those
frames are degraded but because pohang02's own clean footage is less textured than
pohang00's (per-run clean minima: pohang00 2802.8, pohang02 529.9, pohang03
1406.3). A quantile pooled across runs therefore reads the least-textured CLEAN RUN
as a corruption. The hard minimum (508.7) leaves the guard cell at exactly 0% while
still sitting 13x above fog/night and 28x above fog/day, so the margin is not paid
for in false vetoes. That choice was made on fit-run clean data alone; no corrupted
frame informs it.

Usage:
    python scripts/fit_veil_gate.py --quantile 0.01
    python scripts/fit_veil_gate.py --report-only     # measure, write nothing
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
FIT_RUNS = ("pohang00", "pohang02", "pohang03")   # pohang01 held out, as in the photometric fit
NIGHT_RUN = "pohang01"


def load(path: Path) -> list[dict]:
    frames = json.loads(path.read_text(encoding="utf-8"))["frames"]
    if not all("lap_var" in f for f in frames):
        raise SystemExit(f"{path} has no lap_var — rerun scripts/frame_brightness.py")
    return frames


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bright-dir", default="runs/derived/brightness")
    ap.add_argument("--constants", default="runs/eval/brightness_constants.json")
    ap.add_argument("--quantile", type=float, default=0.0,
                    help="accepted false-veto rate on clean fit frames; 0 = hard novelty "
                         "bound at the minimum (default, see module docstring)")
    ap.add_argument("--report-only", action="store_true")
    args = ap.parse_args()

    bd = ROOT / args.bright_dir
    fit = [f for f in load(bd / "gauss_vis_train_clean.json") if f["run"] in FIT_RUNS]
    lv = np.array([f["lap_var"] for f in fit], dtype=float)
    tau = float(lv.min() if args.quantile <= 0 else np.quantile(lv, args.quantile))

    print(f"[veil] clean fit runs {FIT_RUNS}: n={len(lv)}  min {lv.min():.1f}  "
          f"p01 {np.percentile(lv, 1):.1f}  p05 {np.percentile(lv, 5):.1f}  "
          f"median {np.median(lv):.1f}  max {lv.max():.1f}")
    rule = "hard minimum" if args.quantile <= 0 else f"quantile {args.quantile:g}"
    print(f"[veil] tau_lap = {tau:.1f}  ({rule} of clean fit frames)")

    print(f"\n{'cell':16}{'n':>6}{'lap p50':>10}{'veil veto%':>12}")
    for cond in ("clean", "fog", "lowlight", "glare"):
        p = bd / f"gauss_vis_paired_{cond}.json"
        if not p.is_file():
            continue
        frames = load(p)
        v = np.array([f["lap_var"] for f in frames], dtype=float)
        night = np.array([f["run"] == NIGHT_RUN for f in frames])
        for lab, m in (("day", ~night), ("night", night)):
            print(f"{cond + '/' + lab:16}{int(m.sum()):>6}{np.median(v[m]):>10.1f}"
                  f"{100 * (v[m] < tau).mean():>12.1f}")

    if args.report_only:
        print("\n[veil] --report-only: constants not written")
        return 0

    cp = ROOT / args.constants
    obj = json.loads(cp.read_text(encoding="utf-8"))
    obj["vis"]["tau_lap"] = tau
    obj["vis"]["tau_lap_stat"] = "lap_var"
    obj["vis"]["tau_lap_rule"] = (
        f"novelty bound: {rule} of lap_var over CLEAN frames of the fit runs "
        f"{list(FIT_RUNS)}; no corrupted frame informs it"
    )
    obj["vis"]["tau_lap_fit_n"] = int(len(lv))
    cp.write_text(json.dumps(obj, indent=2), encoding="utf-8")
    print(f"\n[veil] wrote tau_lap={tau:.1f} -> {cp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
