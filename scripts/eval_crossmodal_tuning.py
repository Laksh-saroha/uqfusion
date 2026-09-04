"""Knobs the `crossmodal` preset inherited without re-checking, swept honestly.

Three of the fusion parameters were chosen against the OLD system and carried over
unexamined when the weights changed from `capability x r_frame x r_box` to the
capability prior alone. That is exactly the situation where a stale constant hides:
`iou_thr` 0.85 was picked when a per-frame soft weight was doing part of the work,
`sigma_weighted` has been off since before the Gaussian head reached fusion at all,
and the capability ratio has never been varied because until now it was multiplied
by something that moved.

**Selection protocol, and why it matters here more than usual.** Every arm below is
scored on all eight cells, but the arm to ADOPT is chosen only on the clean frames
of the fit runs (pohang00/02/03) -- the same frames every constant in this project
is fitted on. Choosing on the eight-cell table would be fitting on the test set,
and with 8 cells x ~10 arms it would find something spurious with near-certainty.
The eight-cell columns are therefore a REPORT, not a selection criterion, and the
`fit-clean` column is the only one allowed to pick.

Arms:
  iou_thr        0.55 .. 0.90 -- WBF cluster threshold
  sigma_weighted the Gaussian head's sigma reaching fused coordinates (open item)
  cap_ratio      the fixed VIS:IR weight, scaled around its fitted 36.2x
  vis_scale      keep a distrusted VIS in the merge at a reduced score instead of
                 vetoing it -- the `score_scale` path, driven by the VIS health
                 novelty score, aimed squarely at fog/day where the gate currently
                 abstains from VIS entirely and lands exactly on `ir_only`

Writes a new report; overwrites nothing.

Usage:
    python scripts/eval_crossmodal_tuning.py --out runs/eval/crossmodal_tuning.md
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
sys.path.insert(0, str(ROOT / "scripts"))

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts   # noqa: E402
from uqfusion.eval.ctx import FIT_RUNS, NIGHT_RUNS, load_context, run_systems  # noqa: E402

SHIP = 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/crossmodal_tuning.md")
    ap.add_argument("--arms", nargs="+", default=None, help="restrict, to run in parallel")
    args = ap.parse_args()

    t0 = time.time()
    ctx = load_context(preset="crossmodal")
    n = ctx.n()
    night = np.isin(ctx.runs, NIGHT_RUNS)
    splits = {"day": ~night, "night": night}
    cells = [(c, s) for c in ctx.conditions for s in ("day", "night")]
    # The selection set: clean frames of the fit runs. Day by construction.
    fit_sel = np.flatnonzero(np.isin(ctx.runs, FIT_RUNS) & ~night)

    # arm -> (ctx override, run_systems kwargs)
    ARMS: dict[str, tuple] = {"0 adopted (iou 0.85)": (ctx, {})}
    for t in (0.55, 0.65, 0.75, 0.90, 0.95):
        ARMS[f"iou_thr {t}"] = (replace(ctx, iou_thr=t), {})
    ARMS["sigma_weighted"] = (ctx, {"sigma_weighted": True})
    for r in (0.25, 0.5, 2.0, 4.0, 8.0, 16.0, 64.0, 256.0):
        # Scale the VIS:IR capability ratio around its fitted 36.2x. The prior is
        # fitted, so this is a sensitivity probe, not a search for a better value.
        ARMS[f"cap_ratio x{r}"] = (replace(ctx, cap_ir=ctx.cap_ir / r), {})
    for name, floor in (("vis_scale q", None), ("vis_scale q, floor 0.05", 0.05),
                        ("vis_scale q, floor 0.20", 0.20)):
        ARMS[name] = (ctx, {"_vis_scale": floor if floor is not None else 0.0})

    # IR duplicate suppression, selected at nms@0.70 on the fit runs
    # (`probe_ir_dedup.py`): +0.0003 [+0.0003, +0.0005] on the selection set and
    # +0.0019 [+0.0015, +0.0020] on the held-out night run. It lifts five of the
    # eight cells directly, because those cells ARE the IR stream.
    for t in (0.60, 0.70, 0.80):
        ARMS[f"ir_nms {t}"] = (ctx, {"_ir_nms": t})
    ARMS["ir_nms 0.70 + cap_ratio x4"] = (replace(ctx, cap_ir=ctx.cap_ir / 4.0),
                                          {"_ir_nms": 0.70})
    ARMS["ir_nms 0.70 + cap_ratio x16"] = (replace(ctx, cap_ir=ctx.cap_ir / 16.0),
                                           {"_ir_nms": 0.70})

    if args.arms:
        keep = set(args.arms)
        ARMS = {k: v for k, v in ARMS.items() if any(a in k for a in keep)
                or k.startswith("0 adopted")}

    parts = {}
    for cond in ctx.conditions:
        for name, (base, kw) in ARMS.items():
            kw = dict(kw)
            nms_t = kw.pop("_ir_nms", None)
            if nms_t is not None:
                from probe_ir_dedup import dedup
                kw["ir_records"] = [dedup(r, nms_t, "nms") for r in ctx.ir_clean]
            floor = kw.pop("_vis_scale", None)
            if floor is not None:
                # Instead of removing a distrusted VIS, keep it at a score scaled by
                # its own health, floored so a fully-distrusted stream still ranks
                # below IR rather than vanishing. IR keeps score 1: the scale is
                # relative, and `evaluate_systems` normalises by the per-frame max.
                q = ctx.q_vis_by_cond.get(cond, np.ones(n))
                tv = np.maximum(q, floor) * ctx.cap_vis
                kw["trust_override"] = (tv.tolist(), [ctx.cap_ir] * n)
                kw["veto_override"] = ([False] * n, [False] * n)
            res = run_systems(base, cond, **kw)
            parts[(name, cond)] = frame_parts(res["fused_gated"], ctx.gts)
            if name == "0 adopted (iou 0.85)":
                parts[("_vis", cond)] = frame_parts(ctx.vis_by_cond[cond], ctx.gts)
                parts[("_ir", cond)] = frame_parts(res["ir_in_vis"], ctx.gts)
        print(f"[tune] {cond}: {len(ARMS)} arms ({time.time() - t0:.0f}s)", flush=True)

    def ap(key, cond, sel):
        r = ap_from_parts([parts[(key, cond)][i] for i in sel])
        e = r["per_class"].get(SHIP)
        return float(e["ap50_95"]) if e else 0.0

    bar = {(c, s): max(ap("_vis", c, np.flatnonzero(splits[s])),
                       ap("_ir", c, np.flatnonzero(splits[s]))) for c, s in cells}

    rows = []
    for name in ARMS:
        cs = {f"{c}/{s}": ap(name, c, np.flatnonzero(splits[s])) for c, s in cells}
        rows.append({"arm": name, "cells": cs,
                     "fit_clean": ap(name, "clean", fit_sel),
                     "worst_gap": float(min(cs[f"{c}/{s}"] - bar[(c, s)] for c, s in cells)),
                     "sum_gap": float(sum(cs[f"{c}/{s}"] - bar[(c, s)] for c, s in cells))})

    base_fit = next(r["fit_clean"] for r in rows if r["arm"].startswith("0 adopted"))
    picked = max(rows, key=lambda r: r["fit_clean"])

    L = ["# `crossmodal` parameter sweep — selection on fit-run clean only", "",
         "Ship AP. **The `fit-clean` column is the only one allowed to select an "
         "arm**; the eight-cell columns are a report. Choosing on those would be "
         "fitting on the test set, and across 8 cells x "
         f"{len(ARMS)} arms it would find something spurious with near-certainty.", "",
         f"Selection set: clean frames of {list(FIT_RUNS)}, day — n={len(fit_sel)}.", "",
         "| arm | **fit-clean** | Δ fit | " + " | ".join(f"{c}/{s}" for c, s in cells)
         + " | worst gap |",
         "|---|---:|---:|" + "---:|" * (len(cells) + 1)]
    L.append("| _bar (max VIS, IR)_ | — | — | "
             + " | ".join(f"{bar[(c, s)]:.4f}" for c, s in cells) + " | — |")
    for r in sorted(rows, key=lambda r: -r["fit_clean"]):
        L.append(f"| {r['arm']} | **{r['fit_clean']:.4f}** | "
                 f"{r['fit_clean'] - base_fit:+.4f} | "
                 + " | ".join(f"{r['cells'][f'{c}/{s}']:.4f}" for c, s in cells)
                 + f" | {r['worst_gap']:+.4f} |")

    L += ["", "## Verdict", "",
          f"- Best on the selection set: **{picked['arm']}** "
          f"({picked['fit_clean']:.4f} vs the adopted {base_fit:.4f}, "
          f"{picked['fit_clean'] - base_fit:+.4f}).",
          f"- That arm's worst cell: {picked['worst_gap']:+.4f} "
          f"(adopted: {next(r['worst_gap'] for r in rows if r['arm'].startswith('0 adopted')):+.4f}).",
          "",
          "An arm is worth adopting only if it wins the selection column by more "
          "than the noise on it AND does not lose a cell elsewhere. A win of a few "
          "1e-4 on ~1,500 frames is not a win."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"fit_n": int(len(fit_sel)), "bar": {f"{c}/{s}": bar[(c, s)] for c, s in cells},
         "rows": rows}, indent=2), encoding="utf-8")
    print(f"[tune] wrote {out} in {time.time() - t0:.0f}s")
    for r in sorted(rows, key=lambda r: -r["fit_clean"])[:6]:
        print(f"[tune] fit {r['fit_clean']:.4f} ({r['fit_clean'] - base_fit:+.4f})  "
              f"worst {r['worst_gap']:+.4f}  {r['arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
