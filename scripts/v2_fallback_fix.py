"""A candidate repair of the weak-IR fallback under corruption v2, development, descriptive (2026-10-10).

`v2_fallback_cost.py` found the night switch's weak-IR fallback vetoing a low-light VIS on up to 96%
of day frames when IR is fogged or noisy (up to -0.16 ship AP). The fallback confirms a disarmed IR's
night call with VIS `concentrated OR (dark AND veil)`, and v2 low light trips `dark AND veil` on every
day frame. `v2_veil_reprice.py` found the veil axis inert with clean IR under v2, so its one remaining
role is that confirmation. This prices removing it (`gini_by_cond` emptied: no frame is veiled), one
arm against the shipped preset, on VIS clean / low light / fog x IR clean, fog s1-s3, noise s2-s3.
Fogged VIS is included because a fogged night with a damaged IR is what the fallback was built for.

Draw 941 (IR 951), five Phase 3 systems, ship AP, day and night; per cell the shipped fused AP, the
claim fused - max(VIS, IR) for both arms, the veto rates and the paired delta (between-seed 95% t).
Adopts nothing; changes no preset (exposure ledger, 2026-10-10). Scores no pohang04 frame.

    py -3.13 scripts/v2_fallback_fix.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from uqfusion.eval.apmetrics import frame_parts                    # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402
from v2_sensitivity import CACHE, DRAW, PRESET, SEEDS, STATS, T_975_DF4, ship_ap, sub  # noqa: E402

VIS_CONDS = ("clean", "lowlight", "fog")
IR_CONDS = (None, "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")
ARMS = ("shipped", "no_veil")
FLOOR = 0.0060


def score(job):
    k, ir_cond = job
    d = f"{STATS}/brightness/{sub(DRAW)}"
    ctx = load_context(preset=PRESET, cache_dir=f"{CACHE}/seed{k}/{sub(DRAW)}", bright_dir=d,
                       structure_dir=f"{STATS}/structure/{sub(DRAW)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json",
                       conditions=VIS_CONDS, ir_condition=ir_cond, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    gini = dict(ctx.gini_by_cond)
    out = {}
    for arm in ARMS:
        ctx.gini_by_cond = {} if arm == "no_veil" else dict(gini)
        for c in VIS_CONDS:
            r = run_systems(ctx, c)
            parts = {"fused": frame_parts(r["fused_gated"], ctx.gts)}
            if arm == "shipped":
                parts["vis"] = frame_parts(ctx.vis_by_cond[c], ctx.gts)
                parts["ir"] = frame_parts(r["ir_in_vis"], ctx.gts)
            vv = np.asarray(r["veto_vis"], dtype=bool)
            for s, idx in sl.items():
                for a, p in parts.items():
                    out[(arm, c, ir_cond, s, a)] = ship_ap(p, idx)
                out[(arm, c, ir_cond, s, "veto")] = float(vv[idx].mean())
    return k, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/eval/corruption_v2/fallback_fix.md")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    res: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, out in ex.map(score, [(k, ic) for ic in IR_CONDS for k in SEEDS]):
            for key, v in out.items():
                res.setdefault(key, {})[k] = v
    arr = lambda key: np.array([res[key][k] for k in SEEDS])

    def ci(d):
        m, h = float(d.mean()), float(T_975_DF4 * d.std(ddof=1) / np.sqrt(len(d)))
        return m, m - h, m + h

    f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
    L = [f"`{PRESET}` as shipped against the same preset with the veil axis removed; five Phase 3 systems, VIS draw "
         f"{DRAW}, IR draw {DRAW + 10}; ship AP. Claim = fused − max(VIS, IR) (fails at ≤ −{FLOOR} with the interval "
         "below zero); Δ = no veil − shipped, paired by system; 95% t-intervals (df 4).", "",
         "| VIS | IR | slice | VIS | IR | shipped: fused, veto, claim | no veil: veto, claim | Δ fused |",
         "|---|---|---|---:|---:|---|---|---|"]
    js, holds = [], {a: 0 for a in ARMS}
    for c in VIS_CONDS:
        for ic in IR_CONDS:
            for s in ("day", "night"):
                vis, ir = arr(("shipped", c, ic, s, "vis")), arr(("shipped", c, ic, s, "ir"))
                cells = {}
                for a in ARMS:
                    fu = arr((a, c, ic, s, "fused"))
                    cl = ci(fu - np.maximum(vis, ir))
                    fail = cl[0] <= -FLOOR and cl[2] < 0
                    holds[a] += not fail
                    cells[a] = (fu, cl, fail, arr((a, c, ic, s, "veto")).mean())
                dd = ci(cells["no_veil"][0] - cells["shipped"][0])
                sh, nv = cells["shipped"], cells["no_veil"]
                L.append(f"| {c} | {ic or 'clean'} | {s} | {vis.mean():.4f} | {ir.mean():.4f} | "
                         f"{sh[0].mean():.4f}, {sh[3]:.1%}, {f(sh[1])} {'**fails**' if sh[2] else 'holds'} | "
                         f"{nv[3]:.1%}, {f(nv[1])} {'**fails**' if nv[2] else 'holds'} | {f(dd)} |")
                js.append({"vis": c, "ir": ic or "clean", "slice": s, "vis_ap": float(vis.mean()),
                           "ir_ap": float(ir.mean()),
                           **{a: {"fused": float(cells[a][0].mean()), "claim": list(cells[a][1]),
                                  "fails": bool(cells[a][2]), "veto": float(cells[a][3])} for a in ARMS},
                           "delta": list(dd)})
    n = len(VIS_CONDS) * len(IR_CONDS) * 2
    L += ["", f"Cells holding the claim, of {n}: " + ", ".join(f"{a} {holds[a]}" for a in ARMS) + "."]
    out = ROOT / args.out
    out.write_text("# A candidate repair of the weak-IR fallback under corruption v2\n\n" + "\n".join(L) + "\n",
                   encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"cells": js, "holds": holds, "elapsed_s": time.time() - t0},
                                                   indent=1), encoding="utf-8")
    print("\n".join(L))
    print(f"[fix] wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    raise SystemExit(main())
