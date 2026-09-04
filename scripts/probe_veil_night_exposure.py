"""The `veil AND night` repair inherits the night arm's failure mode. Measure it.

`crossmodal` vetoed VIS on `grad_gini < thr` OR `(night AND VIS dark)`. The veil
term stood on its own, so a wrong night call could not reach it.

`crossmodal26m` vetoes on `night AND (VIS dark OR grad_gini < thr)`. That repairs
fog in daylight -- the -0.0632 regression -- but it also puts the veil term BEHIND
the night arm, where a wrong night call now switches it on. The two-of-two vote
that protects the photometric path is "IR says night AND VIS looks dark"; the veil
path substitutes "the frame is veiled" for "VIS looks dark", and a fogged frame is
veiled by construction. So on a FOGGED DAY frame whose IR has been corrupted into
falsely reporting night, the new rule vetoes and the old one -- also -- vetoed, but
for a reason that did not depend on IR being wrong.

The cell that tests this is VIS fog x corrupted IR, and it is in no grid: the
extended grid pairs corrupted IR only with clean, lowlight and blur_s3 VIS. Every
cache it needs already exists.

Three things are reported per cell:
  * the false-veto rate on day frames, which is the safety number;
  * the AP cost, which is what that rate is worth now that a fogged VIS scores
    0.0824 instead of 0.0020;
  * the same for `crossmodal`, so the repair is charged for what it introduced as
    well as credited for what it fixed.

Usage:
    python scripts/probe_veil_night_exposure.py --out runs/eval/veil_night_exposure_26m.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts     # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402

SHIP = 0
#: VIS fog against every IR condition available. fog is the only VIS condition
#: that saturates the veil axis, so it is the only one where this can happen.
CELLS = [("fog", None), ("fog", "fog_s2"), ("fog", "glare_s2"),
         ("fog", "blur_s2"), ("fog", "noise_s2")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/veil_night_exposure_26m.md")
    args = ap.parse_args()
    t0 = time.time()

    rows = []
    for vis_cond, ir_cond in CELLS:
        cname = f"{vis_cond}/{ir_cond or 'clean'}"
        for preset in ("crossmodal", "crossmodal26m"):
            ctx = load_context(preset=preset, cache_dir=args.cache_dir,
                               conditions=(vis_cond,), ir_condition=ir_cond,
                               verbose=False)
            night = np.isin(ctx.runs, NIGHT_RUNS)
            day = np.flatnonzero(~night)
            r = run_systems(ctx, vis_cond)
            p = frame_parts(r["fused_gated"], ctx.gts)
            pv = frame_parts(ctx.vis_by_cond[vis_cond], ctx.gts)
            pi = frame_parts(r["ir_in_vis"], ctx.gts)

            def ap(parts, sel):
                if not len(sel):
                    return float("nan")
                e = ap_from_parts([parts[i] for i in sel])["per_class"].get(SHIP)
                return float(e["ap50_95"]) if e else 0.0

            vv = np.asarray(r["veto_vis"])
            v, i, g = ap(pv, day), ap(pi, day), ap(p, day)
            # A veto is BAD on a day frame when the stream it deletes is the
            # better one on this cell. That is the same test as the claim column
            # in reprice_veto_axes.py, applied to the composed rule.
            bad = float(np.mean(vv[~night])) if v > i else 0.0
            rows.append({"cell": cname, "preset": preset, "VIS": v, "IR": i,
                         "bar": max(v, i), "gated": g, "gap": g - max(v, i),
                         "veto_day": float(np.mean(vv[~night])),
                         "bad_veto_day": bad,
                         "night": ap(p, np.flatnonzero(night)),
                         "night_bar": max(ap(pv, np.flatnonzero(night)),
                                          ap(pi, np.flatnonzero(night)))})
            print(f"[expo] {cname:16} {preset:14} veto {rows[-1]['veto_day']:.0%}  "
                  f"day {g:.4f}  gap {rows[-1]['gap']:+.4f}  night {rows[-1]['night']:.4f} "
                  f"({time.time() - t0:.0f}s)", flush=True)

    L = ["# Does the `veil AND night` repair inherit the night arm's failure mode?", "",
         "`crossmodal` vetoed on `veil OR (night AND VIS dark)` — the veil term "
         "stood alone, so a wrong night call could not reach it. `crossmodal26m` "
         "vetoes on `night AND (VIS dark OR veil)`, which fixes fog in daylight and "
         "puts the veil term **behind** the night arm. The two-of-two vote "
         "substitutes *the frame is veiled* for *VIS looks dark*, and a fogged frame "
         "is veiled by construction — so a corrupted IR that falsely reports night "
         "can now switch the veil term on.", "",
         "VIS fog against every available IR condition. These cells are in no other "
         "grid; every cache already existed.", "",
         "| cell | preset | VIS day | IR day | veto (day) | **bad veto** | gated | gap | night |",
         "|---|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['cell']} | `{r['preset']}` | {r['VIS']:.4f} | {r['IR']:.4f} | "
                 f"{r['veto_day']:.0%} | **{r['bad_veto_day']:.0%}** | {r['gated']:.4f} | "
                 f"**{r['gap']:+.4f}** | {r['night']:.4f} |")

    by = {(r["cell"], r["preset"]): r for r in rows}
    L += ["", "## Verdict", "",
          "`bad veto` is the share of day frames where the rule deletes the stream "
          "that is better on that cell — the same test `reprice_veto_axes.py` "
          "applies to each axis, here applied to the composed rule.", ""]
    for c in [f"{v}/{i or 'clean'}" for v, i in CELLS]:
        a, b = by[(c, "crossmodal")], by[(c, "crossmodal26m")]
        L.append(f"- `{c}`: bad veto {a['bad_veto_day']:.0%} → **{b['bad_veto_day']:.0%}**, "
                 f"gap {a['gap']:+.4f} → **{b['gap']:+.4f}**, night {a['night']:.4f} → "
                 f"{b['night']:.4f}")
    worst = min(rows, key=lambda r: r["gap"] if r["preset"] == "crossmodal26m" else 1)
    L += ["", f"Worst `crossmodal26m` day cell here: **{worst['cell']}** at "
          f"{worst['gap']:+.4f}."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"[expo] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
