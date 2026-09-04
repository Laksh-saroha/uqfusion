"""Is the redundancy in this dataset temporal rather than cross-modal?

Every cross-modal mechanism in this system rests on VIS and IR observing the same
scene redundantly. Measured, they barely do: 0.05% of VIS detections have a
same-class IR detection at the operating `iou_thr`, and a registration correction
that raises that 80x makes ship AP worse. The one cross-modal term that survives
(`support` at IoU 0.30) is worth +0.0090 on the day cells.

Pohang Canal is a slow transit and the paired val frames are genuinely consecutive.
The same vessel appears frame after frame, at no registration cost. This runs the
exact analogue of the cross-modal support term along the time axis and asks which
redundancy is actually there.

**The control is the point.** Temporal support raises `visible_only` as well, so the
bar is recomputed on the boosted streams. Without that this measures a better
single-stream baseline and reports it as a fusion result. Three quantities are
reported per arm:

  VIS      what the single stream scores once boosted -- the honest new baseline
  bar      max(VIS, IR) on the SAME boosted streams
  gap      what fusion still adds on top

An arm whose `VIS` rises and whose `gap` falls has improved the detector, not the
architecture, and the table says so rather than hiding it in the headline.

Causal by default: a deployed system cannot see the future. The symmetric arm is
reported as the ceiling that restriction costs.

Selection on TUNE_RUNS (pohang00), headline on TEST_RUNS (pohang02+03).

Usage:
    python scripts/sweep_temporal_support.py --cache-dir runs/cache_m
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
from uqfusion.eval.ctx import (NIGHT_RUNS, TEST_RUNS, TUNE_RUNS,                 # noqa: E402
                               load_context, run_systems)
from uqfusion.eval.tsupport import temporal_support                              # noqa: E402

SHIP = 0
CONDS = ("clean", "fog", "lowlight", "glare")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/temporal_support_26m.md")
    ap.add_argument("--conditions", nargs="+", default=list(CONDS))
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()

    base = load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                        conditions=tuple(args.conditions), verbose=True)
    night = np.isin(base.runs, NIGHT_RUNS)
    day = np.flatnonzero(~night)
    nsel = np.flatnonzero(night)
    tune = np.flatnonzero(np.isin(base.runs, TUNE_RUNS) & ~night)
    test = np.flatnonzero(np.isin(base.runs, TEST_RUNS) & ~night)

    #: (name, k, iou, gamma, causal, boost_ir, drop_crossmodal)
    ARMS = [
        ("adopted (cross-modal support only)", None, None, None, True, False, False),
        ("no support at all", None, None, None, True, False, True),
        ("temporal k1 g0.5 (VIS)", 1, 0.30, 0.5, True, False, False),
        ("temporal k2 g0.5 (VIS)", 2, 0.30, 0.5, True, False, False),
        ("temporal k2 g1.0 (VIS)", 2, 0.30, 1.0, True, False, False),
        ("temporal k5 g0.5 (VIS)", 5, 0.30, 0.5, True, False, False),
        ("temporal k2 g0.5 (VIS+IR)", 2, 0.30, 0.5, True, True, False),
        ("temporal k2 g0.5, no cross-modal", 2, 0.30, 0.5, True, True, True),
        ("temporal k2 g0.5 SYMMETRIC (ceiling)", 2, 0.30, 0.5, False, True, False),
    ]

    def apf(parts, sel):
        e = ap_from_parts([parts[i] for i in sel])
        s = e["per_class"].get(SHIP)
        return (float(s["ap50_95"]) if s else 0.0, float(e["map50_95"]))

    rows, parts_by = [], {}
    for name, k, iou, gam, causal, do_ir, drop_cm in ARMS:
        ctx = base
        if drop_cm:
            ctx = replace(ctx, support_iou=0.0, support_gamma=0.0)
        if k is not None:
            vb = {c: temporal_support(base.vis_by_cond[c], k=k, iou=iou, gamma=gam,
                                      causal=causal) for c in args.conditions}
            ib = (temporal_support(base.ir_clean, k=k, iou=iou, gamma=gam, causal=causal)
                  if do_ir else base.ir_clean)
            ctx = replace(ctx, vis_by_cond=vb, ir_clean=ib)
        for cond in args.conditions:
            r = run_systems(ctx, cond)
            p = frame_parts(r["fused_gated"], ctx.gts)
            pv = frame_parts(ctx.vis_by_cond[cond], ctx.gts)
            pi = frame_parts(r["ir_in_vis"], ctx.gts)
            parts_by[(name, cond)] = p
            v_d, i_d = apf(pv, day)[0], apf(pi, day)[0]
            v_n, i_n = apf(pv, nsel)[0], apf(pi, nsel)[0]
            g_d, m_d = apf(p, day)
            rows.append({"arm": name, "cond": cond,
                         "VIS_day": v_d, "IR_day": i_d, "bar_day": max(v_d, i_d),
                         "gated_day": g_d, "gap_day": g_d - max(v_d, i_d), "macro_day": m_d,
                         "tune": apf(p, tune)[0], "test": apf(p, test)[0],
                         "VIS_test": apf(pv, test)[0],
                         "night": apf(p, nsel)[0], "bar_night": max(v_n, i_n),
                         "gap_night": apf(p, nsel)[0] - max(v_n, i_n)})
            print(f"[tsup] {name:38} {cond:9} VIS {v_d:.4f} gated {g_d:.4f} "
                  f"gap {g_d - max(v_d, i_d):+.4f}  TEST {rows[-1]['test']:.4f} "
                  f"night {rows[-1]['night']:.4f} ({time.time() - t0:.0f}s)", flush=True)

    by = {(r["arm"], r["cond"]): r for r in rows}
    names = [a[0] for a in ARMS]
    b0 = "adopted (cross-modal support only)"
    boot = {}
    if args.n_boot:
        best = max((n for n in names if n.startswith("temporal")),
                   key=lambda n: by[(n, "clean")]["tune"])
        for tag, sel in (("tune", tune), ("test", test), ("day", day)):
            boot[tag] = bootstrap_delta([parts_by[(best, "clean")][j] for j in sel],
                                        [parts_by[(b0, "clean")][j] for j in sel],
                                        None, n_boot=args.n_boot, cls=SHIP)
        boot["best"] = best

    L = [f"# Temporal support — is the redundancy here in time rather than across "
         f"sensors? (`{args.cache_dir}`)", "",
         "The same term as the cross-modal `support`, with the modality axis swapped "
         "for the time axis: a box's score is multiplied by `1 + γ` when a same-class "
         "box of the **same stream** appeared at IoU ≥ 0.30 within the previous *k* "
         "frames. Coordinates untouched. Causal unless marked.", "",
         "**`VIS` and `bar` are recomputed on the boosted streams.** Temporal support "
         "raises `visible_only` too, so without that this would measure a better "
         "single-stream baseline and report it as a fusion result. An arm whose `VIS` "
         "rises while its `gap` falls improved the **detector**, not the architecture.",
         ""]
    for cond in args.conditions:
        L += [f"## `{cond}`", "",
              "| arm | VIS day | bar | gated | gap | macro | tune | **TEST** | VIS TEST | night | night gap |",
              "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
        for n in names:
            r = by[(n, cond)]
            L.append(f"| {n} | {r['VIS_day']:.4f} | {r['bar_day']:.4f} | "
                     f"{r['gated_day']:.4f} | **{r['gap_day']:+.4f}** | {r['macro_day']:.4f} | "
                     f"{r['tune']:.4f} | **{r['test']:.4f}** | {r['VIS_test']:.4f} | "
                     f"{r['night']:.4f} | {r['gap_night']:+.4f} |")
        L.append("")

    L += ["## Verdict", ""]
    a0 = by[(b0, "clean")]
    for n in names[1:]:
        r = by[(n, "clean")]
        d_vis = r["VIS_day"] - a0["VIS_day"]
        d_sys = r["gated_day"] - a0["gated_day"]
        L.append(f"- `{n}`: VIS {d_vis:+.4f}, system {d_sys:+.4f}, gap "
                 f"{r['gap_day'] - a0['gap_day']:+.4f}, TEST {r['test'] - a0['test']:+.4f}"
                 + ("  — **the gain is in the stream, not the fusion**"
                    if d_vis > 1e-9 and d_sys <= d_vis + 1e-9 else ""))
    if boot:
        L += ["", f"Best temporal arm on the tune runs: **{boot['best']}**"]
        for tag in ("tune", "test", "day"):
            b = boot[tag]
            L.append(f"- {tag}: {b['delta']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"
                     + (" — **spans zero**" if b["spans_zero"] else ""))

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"rows": rows, "bootstrap": boot},
                                                   indent=2), encoding="utf-8")
    print(f"[tsup] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
