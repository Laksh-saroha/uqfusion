"""Every run-time constant of the shipped fusion, re-priced under corruption v2 (2026-10-10).

The constants of `crossmodal26m` were chosen on v1 (macro, one draw) and re-priced on v1 under a
draw-averaged bar (`reprice_constants_draw_avg.py`). v1 is broken (docs/eval/corruption_v2/), so
this asks the same question on v2: is any shipped value dominated by a neighbour?

Development draws 942-944 (VIS) / 952-954 (IR), the five Phase 3 systems, all 60 cells
(VIS clean, fog, fog s1, low light, glare x IR clean, fog s1-s3, noise s2-s3 x day/night), ship
AP. Each arm moves ONE constant; everything else is the shipped preset. Descriptive: an arm that
looks better here must be registered and scored on draws no system was tuned on before it can
replace a shipped value. The night axis of the veto is closed by registration ("no V4"); the
`veto off` row measures the cost of the gate, it does not propose re-tuning it.

    py -3.13 scripts/v2_constants_audit.py run --workers 6
    py -3.13 scripts/v2_constants_audit.py report
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

VIS_CONDS = ("clean", "fog", "fog_s1", "lowlight", "glare")
IR_CONDS = (None, "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")
SEEDS = (0, 1, 2, 3, 4)
DRAWS = (942, 943, 944)
SHIPPED_CAP_SCALE = 4.0
T_975_DF4 = 2.7764451051977987
CACHE, STATS = "runs/cache_p3dev_v2", "runs/derived_p3dev_v2"

#: (label, overrides). `cap_mul` multiplies the shipped cap_ir, which already carries the /4.
ARMS = [
    ("shipped", {}),
    ("cap_ir_scale 1", {"cap_mul": SHIPPED_CAP_SCALE / 1.0}),
    ("cap_ir_scale 2", {"cap_mul": SHIPPED_CAP_SCALE / 2.0}),
    ("cap_ir_scale 8", {"cap_mul": SHIPPED_CAP_SCALE / 8.0}),
    ("cap_ir_scale 16", {"cap_mul": SHIPPED_CAP_SCALE / 16.0}),
    ("iou_thr 0.55", {"iou_thr": 0.55}),
    ("iou_thr 0.70", {"iou_thr": 0.70}),
    ("iou_thr 0.95", {"iou_thr": 0.95}),
    ("support off", {"support_gamma": 0.0}),
    ("support gamma 0.25", {"support_gamma": 0.25}),
    ("support gamma 1.0", {"support_gamma": 1.0}),
    ("support iou 0.20", {"support_iou": 0.20}),
    ("support iou 0.50", {"support_iou": 0.50}),
    ("weak-IR fallback off", {"night_weak_fallback": False}),
    ("IR merge veto on", {"ir_merge_veto": True}),
    ("veil not night-conditional", {"veil_requires_night": False}),
    ("veto off (union)", {"veto": None}),
]


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def apply_arm(ctx, ov: dict):
    ov = dict(ov)
    mul = ov.pop("cap_mul", None)
    if mul is not None:
        ov["cap_ir"] = ctx.cap_ir * mul
    return dataclasses.replace(ctx, **ov) if ov else ctx


def job(spec):
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems
    from v2_sensitivity import ship_ap
    import seed_ensemble
    seed_ensemble._DIST.clear()
    seed_ensemble._memo_distances()      # every arm shares the context's record lists: score them once
    draw, ic, k = spec
    t0 = time.time()
    d = f"{STATS}/brightness/{sub(draw)}"
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{CACHE}/seed{k}/{sub(draw)}", bright_dir=d,
                       structure_dir=f"{STATS}/structure/{sub(draw)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ic or 'clean'}.json",
                       conditions=VIS_CONDS, ir_condition=ic, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    res = {}
    for c in VIS_CONDS:
        for label, ov in ARMS:
            a = apply_arm(ctx, ov)
            out = run_systems(a, c, compute_metrics=False)
            p = frame_parts(out["fused_gated"], a.gts)
            for s, idx in sl.items():
                res[f"{label}|{c}|{s}"] = ship_ap(p, idx)
        pv, pi = frame_parts(ctx.vis_by_cond[c], ctx.gts), frame_parts(out["ir_in_vis"], ctx.gts)
        for s, idx in sl.items():
            res[f"VIS|{c}|{s}"], res[f"IR|{c}|{s}"] = ship_ap(pv, idx), ship_ap(pi, idx)
    return spec, res, time.time() - t0


def cmd_run(args) -> int:
    out_p = ROOT / "runs/v2_constants_audit.json"
    done = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    specs = [(d, ic, k) for d in args.draws for ic in IR_CONDS for k in SEEDS if f"{d}|{ic}|{k}" not in done]
    print(f"[audit] {len(specs)} jobs ({len(done)} from checkpoint)", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for n, (spec, res, dt) in enumerate(ex.map(job, specs), 1):
            done["|".join(map(str, spec))] = res
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(done), encoding="utf-8")
            os.replace(tmp, out_p)
            print(f"[audit] {n}/{len(specs)} {spec} {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_report(args) -> int:
    raw = json.loads((ROOT / "runs/v2_constants_audit.json").read_text(encoding="utf-8"))
    # Only draws scored for every IR condition and system enter (the run may cover a subset; stated below).
    draws = [d for d in DRAWS if all(f"{d}|{ic}|{k}" in raw for ic in IR_CONDS for k in SEEDS)]

    def per_sys(name, c, ic, s):
        return np.array([np.mean([raw[f"{d}|{ic}|{k}"][f"{name}|{c}|{s}"] for d in draws]) for k in SEEDS])

    def ci(x):
        m, h = float(np.mean(x)), T_975_DF4 * float(np.std(x, ddof=1)) / np.sqrt(len(x))
        return m, m - h, m + h

    cells = [(c, ic, s) for ic in IR_CONDS for c in VIS_CONDS for s in ("day", "night")]
    lines = ["# Shipped constants re-priced under corruption v2 (development, 2026-10-10)", "",
             f"`scripts/v2_constants_audit.py`. Draws {', '.join(map(str, draws))}"
             f"{'' if len(draws) == len(DRAWS) else ' (of the registered ' + ', '.join(map(str, DRAWS)) + '; the rest were not run for compute)'}, five Phase 3 systems, "
             "60 cells, ship AP. Δ = arm − shipped fused, per system the draw mean, 95% t-interval over "
             "systems (df 4). better / worse = |Δ| ≥ floor with the interval excluding zero. Claim = "
             "fused − max(VIS, IR) failing at ≤ −0.0060 with the interval below zero. Descriptive: no "
             "arm replaces a shipped value without a registered test on unseen draws.", "",
             "| arm | better ≥0.0014 | worse ≥0.0014 | better ≥0.0060 | worse ≥0.0060 | worst Δ (cell) | "
             "best Δ (cell) | claim fails |", "|---|---:|---:|---:|---:|---|---|---:|"]
    summ = {}
    for label, _ in ARMS:
        ds, fails = [], 0
        for c, ic, s in cells:
            base = per_sys("shipped", c, ic, s)
            arm = per_sys(label, c, ic, s)
            dd = ci(arm - base)
            ds.append((dd, f"{c}/{ic or 'clean'}/{s}"))
            mx = np.maximum(per_sys("VIS", c, ic, s), per_sys("IR", c, ic, s))
            cl = ci(arm - mx)
            fails += cl[0] <= -0.0060 and cl[2] < 0
        cnt = lambda f, sign: sum(1 for (m, lo, hi), _ in ds   # noqa: E731
                                  if (sign > 0 and m >= f and lo > 0) or (sign < 0 and m <= -f and hi < 0))
        w = min(ds, key=lambda x: x[0][0])
        b = max(ds, key=lambda x: x[0][0])
        summ[label] = {"better_0014": cnt(0.0014, 1), "worse_0014": cnt(0.0014, -1),
                       "better_0060": cnt(0.0060, 1), "worse_0060": cnt(0.0060, -1),
                       "worst": [w[1], *w[0]], "best": [b[1], *b[0]], "claim_fails": int(fails),
                       "cells": {n: list(x) for x, n in ds}}
        lines.append(f"| {label} | {cnt(0.0014, 1)} | {cnt(0.0014, -1)} | {cnt(0.0060, 1)} | {cnt(0.0060, -1)} | "
                     f"{w[0][0]:+.4f} ({w[1]}) | {b[0][0]:+.4f} ({b[1]}) | {fails} |")
    out = ROOT / "docs/eval/corruption_v2/constants_audit.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("\n".join(lines[-len(ARMS):]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--draws", type=int, nargs="+", default=list(DRAWS))
    sp.add_parser("report")
    args = ap.parse_args()
    return {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
