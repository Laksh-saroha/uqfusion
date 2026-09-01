"""Two questions the extended grid finally makes answerable.

**1. Is the capability ratio priceable now?** §3c refused to tune it because no cell
had VIS unvetoed *and* IR competitive: on every original cell VIS beat IR ~36x
where it was unvetoed at all, so pushing the ratio up was free. `blur_s3` changes
that — VIS 0.0215 against IR 0.0181, a 1.2x gap, with VIS vetoed on only 17% of
frames, so on the other 83% both streams are genuinely in play at comparable
strength. If a large ratio is going to cost anything anywhere, it costs it here.

**2. Should `veto_ir` use the authority bound instead of the merge bound?** The two
bounds were split because the decisions fail in opposite directions — but that
argument was made when IR was never damaged. The extended grid shows the merge
bound (222.7) flags only 20% of blur-corrupted IR, and `clean x IR blur_s2` is now
the worst day cell at -0.0015. The tighter authority bound (64.9) would drop IR
more readily; it also drops it on 1.1% of CLEAN day frames, which is the cost side.
Both directions are now measurable.

Selection stays on clean fit-run frames; the extended cells are the report. But an
arm that wins the selection set and *loses a corrupted cell* is rejected — that is
precisely the check the original grid could not perform.

Usage:
    python scripts/eval_extended_tuning.py --out runs/eval/extended_tuning.md
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

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts   # noqa: E402
from uqfusion.eval.ctx import FIT_RUNS, NIGHT_RUNS, load_context, run_systems  # noqa: E402

SHIP = 0
CELLS = [("clean", None), ("blur_s3", None), ("rain_s2", None), ("noise_s2", None),
         ("clean", "blur_s2"), ("clean", "glare_s2"), ("lowlight", "glare_s2"),
         ("blur_s3", "glare_s2")]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache",
                     help="prediction caches to read. runs/cache = yolo26s (phase2); "
                          "runs/cache_m = the full-scale yolo26m / yolo26m-p2feat "
                          "detectors the architecture actually specifies.")
    ap.add_argument("--out", default="runs/eval/extended_tuning.md")
    args = ap.parse_args()
    t0 = time.time()

    arms = ["adopted"] + [f"cap_ratio x{r}" for r in (4.0, 16.0, 64.0)] + ["veto_ir @authority"]
    res: dict[tuple[str, str], dict] = {}
    for vis_cond, ir_cond in CELLS:
        base = load_context(cache_dir=args.cache_dir, preset="crossmodal", conditions=(vis_cond,),
                            ir_condition=ir_cond, verbose=False)
        night = np.isin(base.runs, NIGHT_RUNS)
        day = np.flatnonzero(~night)
        fit_sel = np.flatnonzero(np.isin(base.runs, FIT_RUNS) & ~night)

        def ap(p, sel):
            e = ap_from_parts([p[i] for i in sel])["per_class"].get(SHIP)
            return float(e["ap50_95"]) if e else 0.0

        for arm in arms:
            ctx = base
            if arm.startswith("cap_ratio"):
                ctx = replace(base, cap_ir=base.cap_ir / float(arm.split("x")[1]))
            elif arm == "veto_ir @authority":
                # Use the TIGHTER bound for the merge decision too.
                ctx = replace(base, ir_bound=base.ir_bound_switch)
            r = run_systems(ctx, vis_cond)
            p = frame_parts(r["fused_gated"], ctx.gts)
            if arm == "adopted":
                pv, pi = frame_parts(base.vis_by_cond[vis_cond], base.gts), \
                    frame_parts(r["ir_in_vis"], base.gts)
                res[("_bar", f"{vis_cond}/{ir_cond or 'clean'}")] = {
                    "day": max(ap(pv, day), ap(pi, day)),
                    "fit": max(ap(pv, fit_sel), ap(pi, fit_sel))}
            res[(arm, f"{vis_cond}/{ir_cond or 'clean'}")] = {
                "day": ap(p, day), "fit": ap(p, fit_sel),
                "veto_ir": float(np.mean(np.asarray(r["veto_ir"])[~night]))}
        print(f"[ext] {vis_cond} x IR {ir_cond or 'clean'} done ({time.time() - t0:.0f}s)",
              flush=True)

    names = [f"{v}/{i or 'clean'}" for v, i in CELLS]
    L = ["# Does the extended grid price the capability ratio, and the veto_ir bound?", "",
         "Ship AP, day frames. `gap` is against `max(VIS, IR)` on the same streams. "
         "The `clean` column is the selection set (fit runs, clean, day); every "
         "other column is a report — but an arm that wins the selection set and "
         "**loses a corrupted cell** is rejected, which is the check the original "
         "grid could not perform.", "",
         "| arm | " + " | ".join(names) + " | worst gap |",
         "|---|" + "---:|" * (len(names) + 1)]
    L.append("| _bar_ | " + " | ".join(f"{res[('_bar', n)]['day']:.4f}" for n in names)
             + " | — |")
    rows = []
    for arm in arms:
        gaps = [res[(arm, n)]["day"] - res[("_bar", n)]["day"] for n in names]
        rows.append({"arm": arm, "cells": {n: res[(arm, n)]["day"] for n in names},
                     "gaps": dict(zip(names, gaps)), "worst": float(min(gaps)),
                     "fit": res[(arm, "clean/clean")]["fit"]})
        L.append(f"| {arm} | " + " | ".join(f"{res[(arm, n)]['day']:.4f}" for n in names)
                 + f" | {min(gaps):+.4f} |")

    L += ["", "## Gap to bar per cell", "",
          "| arm | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for r in rows:
        L.append(f"| {r['arm']} | " + " | ".join(f"{r['gaps'][n]:+.4f}" for n in names) + " |")

    base_row = rows[0]
    L += ["", "## Verdict", ""]
    for r in rows[1:]:
        lost = [n for n in names if r["gaps"][n] < base_row["gaps"][n] - 1e-9]
        L.append(f"- **{r['arm']}**: worst cell {r['worst']:+.4f} "
                 f"(adopted {base_row['worst']:+.4f}); "
                 + (f"loses on {', '.join(lost)}" if lost else "loses on no cell"))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"rows": rows, "bar": {n: res[("_bar", n)] for n in names}}, indent=2),
        encoding="utf-8")
    print(f"[ext] wrote {out} in {time.time() - t0:.0f}s")
    for r in rows:
        print(f"[ext] worst {r['worst']:+.4f}  {r['arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
