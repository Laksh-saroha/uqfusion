"""Can IR say *whether* without saying *where*?

`probe_detector_swap.py` measured the thing this sweep exists for: at the adopted
`iou_thr` of 0.85, only **0.05-0.08%** of VIS detections have a same-class IR
detection at that overlap. So the two streams essentially never land in the same
WBF cluster, WBF's cross-modal agreement bonus almost never fires, and whatever
fusion gains on a day cell it does not gain by agreement. It gains it by appending
IR's boxes at the bottom of the ranking, where a few of them recover GT that VIS
missed.

Agreement is not absent, though -- it is at the wrong scale. At IoU 0.30 a third
of VIS boxes have an IR partner, and at 0.10 about half. That is exactly where a
3-6 px median registration residual (`runs/eval/x_registration_drift.md`) would put
it. Lowering `iou_thr` to reach it has already been swept and loses (-0.0138 on the
fit set at 0.55), because merging across that residual drags the fused box off the
object.

Which suggests the split this sweeps: at a loose overlap IR carries evidence that a
target IS there and no usable evidence about WHERE its edges are. `support_gamma`
multiplies a box's score when another stream overlaps it at `support_iou`, and
never touches a coordinate.

`consensus_beta` and `consensus_distinct` are swept alongside, because the same
measurement predicts they are nearly inert on real data, and a prediction that
specific is worth falsifying rather than assuming.

Selection is on `TUNE_RUNS` (pohang00) day frames and the result is reported on
`TEST_RUNS` (pohang02 + pohang03), which is a RUN-DISJOINT held-out day set. That
distinction has not existed before in this project: `FIT_RUNS` excludes only
pohang01, and pohang01 is entirely night, so "fit-run day" and "day" have always
been the same 1200 frames and every day constant has been reported in-sample.

Night is reported too. VIS is 0.0000 there, so every night frame is IR alone and
no support term can move it -- if one does, something is wrong.

Usage:
    python scripts/sweep_support.py --cache-dir runs/cache_m --out runs/eval/support_26m.md
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

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import (NIGHT_RUNS, TEST_RUNS, TUNE_RUNS,                  # noqa: E402
                               load_context, run_systems)

SHIP = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/support_26m.md")
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()

    base = load_context(preset="crossmodal", cache_dir=args.cache_dir,
                        conditions=(args.condition,), verbose=True)
    night = np.isin(base.runs, NIGHT_RUNS)
    day = np.flatnonzero(~night)
    tune = np.flatnonzero(np.isin(base.runs, TUNE_RUNS) & ~night)
    test = np.flatnonzero(np.isin(base.runs, TEST_RUNS) & ~night)
    nightsel = np.flatnonzero(night)
    assert not set(tune) & set(test), "the day split must be disjoint"

    arms = [("adopted", base)]
    for iou in (0.10, 0.30, 0.50):
        for gam in (0.25, 0.5, 1.0, 2.0):
            arms.append((f"support iou{iou:g} gamma{gam:g}",
                         replace(base, support_iou=iou, support_gamma=gam)))
    for b in (0.0, 2.0):
        arms.append((f"consensus_beta {b:g}", replace(base, consensus_beta=b)))
    arms.append(("consensus_distinct", replace(base, consensus_distinct=True)))

    def apf(parts, sel):
        e = ap_from_parts([parts[i] for i in sel])
        s = e["per_class"].get(SHIP)
        return (float(s["ap50_95"]) if s else 0.0, float(e["map50_95"]))

    p_vis = frame_parts(base.vis_by_cond[args.condition], base.gts)
    rows, parts_by_arm = [], {}
    for name, ctx in arms:
        r = run_systems(ctx, args.condition)
        p = frame_parts(r["fused_gated"], ctx.gts)
        parts_by_arm[name] = p
        ds, dm = apf(p, day)
        us, _ = apf(p, tune)
        ts, tm = apf(p, test)
        ns, _ = apf(p, nightsel)
        rows.append({"arm": name, "tune": us, "test": ts, "test_macro": tm,
                     "day": ds, "day_macro": dm, "night": ns})
        print(f"[sup] {name:28} tune {us:.4f}  TEST {ts:.4f}  day {ds:.4f}  "
              f"night {ns:.4f} ({time.time() - t0:.0f}s)", flush=True)

    b0 = rows[0]
    # Selected on the TUNE runs only. Whether that choice survives is the
    # `test` column, and it is allowed to say no.
    best = max(rows[1:], key=lambda r: r["tune"])
    boot = {}
    if args.n_boot:
        for tag, sel in (("tune", tune), ("test", test), ("day", day)):
            boot[tag] = bootstrap_delta([parts_by_arm[best["arm"]][k] for k in sel],
                                        [parts_by_arm["adopted"][k] for k in sel],
                                        None, n_boot=args.n_boot, cls=SHIP)
        boot["vs_vis_day"] = bootstrap_delta([parts_by_arm[best["arm"]][k] for k in day],
                                             [p_vis[k] for k in day],
                                             None, n_boot=args.n_boot, cls=SHIP)

    L = [f"# Support, consensus, and what the geometry allows — `{args.cache_dir}`, "
         f"`{args.condition}`", "",
         "Ship AP. **tune** is pohang00 day (the selection set); **TEST** is "
         "pohang02+03 day, run-disjoint and held out. `day` is both together, "
         "reported only for comparison with every earlier table -- it is the "
         "in-sample number those tables have always shown. `support` boosts a box's "
         "score when the other stream overlaps it at `iou`, and never moves a "
         "coordinate.", "",
         "| arm | tune | **TEST** | test macro | day (in-sample) | night | tune delta | TEST delta |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['tune']:.4f} | **{r['test']:.4f}** | "
                 f"{r['test_macro']:.4f} | {r['day']:.4f} | {r['night']:.4f} | "
                 f"{r['tune'] - b0['tune']:+.4f} | {r['test'] - b0['test']:+.4f} |")

    L += ["", "## Verdict", "",
          f"- Best on the TUNE runs: **{best['arm']}**, "
          f"{best['tune'] - b0['tune']:+.4f} over adopted there and "
          f"**{best['test'] - b0['test']:+.4f} on the held-out runs**."]
    if boot:
        for tag in ("tune", "test", "day"):
            b = boot[tag]
            L.append(f"- {tag}: {b['delta']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"
                     + (" — **spans zero**" if b["spans_zero"] else ""))
        b = boot["vs_vis_day"]
        L.append(f"- vs VIS alone, day: {b['delta']:+.4f} [{b['ci_lo']:+.4f}, "
                 f"{b['ci_hi']:+.4f}]" + (" — **spans zero**" if b["spans_zero"] else ""))
    unmoved = [r["arm"] for r in rows[1:] if abs(r["night"] - b0["night"]) < 1e-9]
    L.append(f"- Night unmoved by {len(unmoved)} of {len(rows) - 1} arms, as it must be: "
             f"VIS is 0.0000 there, so there is no second stream to confirm anything.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"rows": rows, "bootstrap": boot},
                                                   indent=2), encoding="utf-8")
    print(f"[sup] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
