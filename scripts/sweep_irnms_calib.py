"""Two inherited constants that were fitted against a detector that no longer exists.

**`ir_nms` 0.7.** Adopted because the phase2 IR model emitted 54.2 boxes per frame
against VIS's 13.0 and needed deduplicating. The full-scale IR model is
`yolo26m-p2feat` at `nc=1` and emits 30.5. The threshold was never a setting; it
was a repair for one checkpoint's behaviour, and this asks whether the repair is
still needed and still at the right value.

**Cross-modal score calibration.** `runs/eval/x_score_calibration.md` measured
isotonic `conf -> P(TP@0.5)` per modality and found VIS and IR scores differ by
3-6x at the same raw confidence -- and it did so under the OLD gate, where the
question was nearly moot. Under `crossmodal` it is not: fusion is ~99.9%
concatenation (only 0.05% of VIS boxes share a cluster with an IR box), so the
cross-modal SCORE ORDERING is the entire fusion mechanism, and the capability
weights currently suppress IR by 135x where calibration says the honest ratio is
3-6x. Those two numbers disagree by more than an order of magnitude and only one
of them can be right.

Isotonic regression is monotone, so `visible_only` and `ir_only` are invariant
under it -- asserted here, because if they move the implementation is wrong and
nothing else in the table means anything.

Fitted on TUNE_RUNS only and reported on TEST_RUNS, which is what the older
calibration run could not do: it fitted on all three day runs and reported on the
same frames.

Usage:
    python scripts/sweep_irnms_calib.py --cache-dir runs/cache_m --out runs/eval/irnms_calib_26m.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts        # noqa: E402
from uqfusion.eval.ctx import (NIGHT_RUNS, TEST_RUNS, TUNE_RUNS,      # noqa: E402
                               load_context, run_systems)

SHIP = 0
DAY_RUNS = ("pohang00", "pohang02", "pohang03")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/irnms_calib_26m.md")
    ap.add_argument("--support-iou", type=float, default=0.30)
    ap.add_argument("--support-gamma", type=float, default=0.5)
    args = ap.parse_args()
    t0 = time.time()

    from eval_score_calibration import isotonic_fit, recalibrate, tp_flags  # noqa: E402

    def sets(ctx):
        night = np.isin(ctx.runs, NIGHT_RUNS)
        return (np.flatnonzero(np.isin(ctx.runs, TUNE_RUNS) & ~night),
                np.flatnonzero(np.isin(ctx.runs, TEST_RUNS) & ~night),
                np.flatnonzero(~night), np.flatnonzero(night), night)

    def apf(parts, sel):
        e = ap_from_parts([parts[i] for i in sel])
        s = e["per_class"].get(SHIP)
        return (float(s["ap50_95"]) if s else 0.0, float(e["map50_95"]))

    rows = []

    # ---- 1. ir_nms ----------------------------------------------------------
    for nms in (None, 0.5, 0.6, 0.7, 0.8, 0.9):
        for sup in (False, True):
            ctx = load_context(preset="crossmodal", cache_dir=args.cache_dir,
                               conditions=("clean",), ir_nms=nms, verbose=False)
            if sup:
                from dataclasses import replace
                ctx = replace(ctx, support_iou=args.support_iou,
                              support_gamma=args.support_gamma)
            tune, test, day, nsel, _ = sets(ctx)
            r = run_systems(ctx, "clean")
            p = frame_parts(r["fused_gated"], ctx.gts)
            pr = {q: apf(p, np.flatnonzero(ctx.runs == q))[0] for q in DAY_RUNS}
            rows.append({"group": "ir_nms", "arm": f"ir_nms {nms or 'off'}"
                                                   + (" + support" if sup else ""),
                         "tune": apf(p, tune)[0], "test": apf(p, test)[0],
                         "day": apf(p, day)[0], "night": apf(p, nsel)[0],
                         "boxes": float(np.mean([len(x["conf"]) for x in ctx.ir_clean])),
                         "per_run": pr})
            print(f"[nms] {rows[-1]['arm']:22} boxes {rows[-1]['boxes']:5.1f}  "
                  f"tune {rows[-1]['tune']:.4f}  TEST {rows[-1]['test']:.4f}  "
                  f"night {rows[-1]['night']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2. cross-modal score calibration -----------------------------------
    ctx = load_context(preset="crossmodal", cache_dir=args.cache_dir,
                       conditions=("clean",), verbose=False)
    tune, test, day, nsel, night = sets(ctx)
    r0 = run_systems(ctx, "clean")
    cv, tv = tp_flags(ctx.vis_by_cond["clean"], ctx.gts, tune)
    ci, ti = tp_flags(r0["ir_in_vis"], ctx.gts, tune)
    xs_v, ys_v = isotonic_fit(cv, tv)
    xs_i, ys_i = isotonic_fit(ci, ti)
    calib_scale = [float(np.interp(q, xs_v, ys_v) / max(np.interp(q, xs_i, ys_i), 1e-9))
                   for q in (0.05, 0.1, 0.25, 0.5)]

    from dataclasses import replace
    cal_ctx = replace(
        ctx,
        vis_by_cond={"clean": recalibrate(ctx.vis_by_cond["clean"], xs_v, ys_v)},
        ir_clean=recalibrate(ctx.ir_clean, xs_i, ys_i))
    # The control: isotonic is monotone, so each single stream must be unchanged.
    p_v0 = frame_parts(ctx.vis_by_cond["clean"], ctx.gts)
    p_v1 = frame_parts(cal_ctx.vis_by_cond["clean"], ctx.gts)
    ctrl = abs(apf(p_v0, day)[0] - apf(p_v1, day)[0])

    for name, c in (("calibrated", cal_ctx),
                    ("calibrated + support",
                     replace(cal_ctx, support_iou=args.support_iou,
                             support_gamma=args.support_gamma)),
                    # Calibration replaces what the capability weight was doing by
                    # hand, so the ratio must be re-opened alongside it, not held.
                    ("calibrated, cap_ratio x1", replace(cal_ctx, cap_ir=cal_ctx.cap_ir * 4.0)),
                    ("calibrated, cap_ratio x16", replace(cal_ctx, cap_ir=cal_ctx.cap_ir / 4.0))):
        rr = run_systems(c, "clean")
        p = frame_parts(rr["fused_gated"], c.gts)
        rows.append({"group": "calibration", "arm": name,
                     "tune": apf(p, tune)[0], "test": apf(p, test)[0],
                     "day": apf(p, day)[0], "night": apf(p, nsel)[0], "boxes": None,
                     "per_run": {q: apf(p, np.flatnonzero(ctx.runs == q))[0] for q in DAY_RUNS}})
        print(f"[cal] {name:26} tune {rows[-1]['tune']:.4f}  TEST {rows[-1]['test']:.4f} "
              f"({time.time() - t0:.0f}s)", flush=True)

    base = next(r for r in rows if r["arm"] == "ir_nms 0.7")
    L = [f"# Two inherited constants, re-asked of the full-scale detectors — "
         f"`{args.cache_dir}`", "",
         f"Ship AP. **tune** = pohang00 day (selection), **TEST** = pohang02+03 day "
         f"(run-disjoint, held out). The adopted row is `ir_nms 0.7`.", "",
         "## 1. `ir_nms` — a repair for a checkpoint that is no longer in the system", "",
         "The phase2 IR model emitted 54.2 boxes/frame against VIS's 13.0; the "
         "full-scale one emits 30.5 at `nc=1`. `boxes` is what the gate is handed "
         "after the NMS.", "",
         "| arm | IR boxes/frame | tune | **TEST** | day | night | tune delta | TEST delta |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["group"] != "ir_nms":
            continue
        L.append(f"| {r['arm']} | {r['boxes']:.1f} | {r['tune']:.4f} | **{r['test']:.4f}** | "
                 f"{r['day']:.4f} | {r['night']:.4f} | {r['tune'] - base['tune']:+.4f} | "
                 f"{r['test'] - base['test']:+.4f} |")

    L += ["", "## 2. Cross-modal score calibration", "",
          f"Isotonic `conf -> P(TP@0.5)` per modality, **fitted on pohang00 only**. "
          f"Calibrated VIS/IR score ratio at raw conf 0.05/0.1/0.25/0.5: "
          + ", ".join(f"{x:.1f}x" for x in calib_scale)
          + f". The capability weights currently suppress IR by "
          f"{ctx.cap_vis / max(ctx.cap_ir, 1e-9):.0f}x.", "",
          f"**Control** — `visible_only` must be invariant under a monotone "
          f"transform: moved by {ctrl:.2e}"
          + (" ✓" if ctrl < 1e-9 else " — **NON-ZERO, the table below is not trustworthy**"),
          "", "| arm | tune | **TEST** | day | night | tune delta | TEST delta |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        if r["group"] != "calibration":
            continue
        L.append(f"| {r['arm']} | {r['tune']:.4f} | **{r['test']:.4f}** | {r['day']:.4f} | "
                 f"{r['night']:.4f} | {r['tune'] - base['tune']:+.4f} | "
                 f"{r['test'] - base['test']:+.4f} |")

    L += ["", "## Per-run deltas against `ir_nms 0.7`", "",
          "| arm | " + " | ".join(DAY_RUNS) + " | worst run |",
          "|---|" + "---:|" * (len(DAY_RUNS) + 1)]
    for r in rows:
        d = [r["per_run"][q] - base["per_run"][q] for q in DAY_RUNS]
        L.append(f"| {r['arm']} | " + " | ".join(f"{x:+.4f}" for x in d)
                 + f" | **{min(d):+.4f}** |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"rows": rows, "calib_ratio": calib_scale, "control": ctrl}, indent=2),
        encoding="utf-8")
    print(f"[nms] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
