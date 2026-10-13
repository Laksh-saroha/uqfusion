"""The veil repair re-priced under corruption v2, development, descriptive (2026-10-10).

`crossmodal26m` makes the veil axis (grad_gini below its clean-fitted bound) conditional on the
night arm: VIS is vetoed on `night AND (dark OR veil)` instead of `veil OR (night AND dark)`. The
repair was priced on v1 fog, whose fixed 21-px blur is what `veil` detected (100% of v1 fog
frames; 9% of v2 fog days, `gate_axes.md`). This re-prices it on the five Phase 3 systems under
v2, three arms, one switch each, everything else the shipped preset:

  shipped       `night AND (dark OR veil)`, weak-IR fallback `concentrated OR (dark AND veil)`
  veil_uncond   `veil_requires_night` off: `veil OR (night AND dark)`, the pre-repair rule
  veil_off      the veil axis removed (no frame is veiled), in the night arm and the fallback

Cells: clean, fog, lowlight, glare, fog s1 x day / night, VIS draws 941-944, clean IR, ship AP.
Per arm: fused AP and the Table 3b claim (fused - max(VIS, IR), between-seed 95% t-interval on the
draw mean, fail at <= -0.0060 with the interval below zero); Δ = arm - shipped, paired by system.

Adopts nothing; changes no preset (exposure ledger, 2026-10-10). Scores no pohang04 frame.

    py -3.13 scripts/v2_veil_reprice.py
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

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts       # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402

PRESET, SHIP, FLOOR = "crossmodal26m", 0, 0.0060
SEEDS, DRAWS = (0, 1, 2, 3, 4), (941, 942, 943, 944)
CONDS = ("clean", "fog", "lowlight", "glare", "fog_s1")
ARMS = ("shipped", "veil_uncond", "veil_off")
CACHE, STATS = "runs/cache_p3dev_v2", "runs/derived_p3dev_v2"
T_975_DF4 = 2.7764451051977987


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def ship_ap(parts, sel) -> float:
    e = ap_from_parts(parts, sel=sel)["per_class"].get(SHIP)
    return float(e["ap50_95"]) if e else float("nan")


def score(job):
    k, v = job
    ctx = load_context(preset=PRESET, cache_dir=f"{CACHE}/seed{k}/{sub(v)}",
                       bright_dir=f"{STATS}/brightness/{sub(v)}", structure_dir=f"{STATS}/structure/{sub(v)}",
                       conditions=CONDS, verbose=False)
    if any("pohang04" in r for r in ctx.runs):
        raise SystemExit("a pohang04 frame reached the development context")
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    gini, vrn = dict(ctx.gini_by_cond), ctx.veil_requires_night
    out = {}
    for arm in ARMS:
        ctx.gini_by_cond = {} if arm == "veil_off" else dict(gini)
        ctx.veil_requires_night = False if arm == "veil_uncond" else vrn
        for c in CONDS:
            r = run_systems(ctx, c)
            parts = {"fused": frame_parts(r["fused_gated"], ctx.gts)}
            if arm == "shipped":
                parts["vis"] = frame_parts(ctx.vis_by_cond[c], ctx.gts)
                parts["ir"] = frame_parts(r["ir_in_vis"], ctx.gts)
            vv = np.asarray(r["veto_vis"], dtype=bool)
            for s, idx in sl.items():
                for a, p in parts.items():
                    out[(arm, c, s, a)] = ship_ap(p, idx)
                out[(arm, c, s, "veto")] = float(vv[idx].mean())
    return k, v, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/eval/corruption_v2/veil_reprice.md")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    res: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, v, out in ex.map(score, [(k, v) for k in SEEDS for v in DRAWS]):
            for key, x in out.items():
                res.setdefault(key, np.full((len(SEEDS), len(DRAWS)), np.nan))[SEEDS.index(k), DRAWS.index(v)] = x
    seed_mean = {key: a.mean(axis=1) for key, a in res.items()}           # draw mean per system

    def ci(d):
        m, h = float(d.mean()), float(T_975_DF4 * d.std(ddof=1) / np.sqrt(len(d)))
        return m, m - h, m + h

    f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
    L = [f"Five Phase 3 systems, `{PRESET}`, v2 caches, VIS draws 941–944 (clean IR), ship AP; seed mean of draw "
         "means. Claim = fused − max(VIS, IR), between-seed 95% t-interval (df 4); **fails** at ≤ −0.0060 with the "
         "interval below zero (Table 3b's criterion). Δ = arm − shipped, paired by system.", "",
         "| cell | VIS | IR | arm | fused | veto | claim | Δ vs shipped |", "|---|---:|---:|---|---:|---:|---|---|"]
    js, holds = [], {a: 0 for a in ARMS}
    for c in CONDS:
        for s in ("day", "night"):
            vis, ir = seed_mean[("shipped", c, s, "vis")], seed_mean[("shipped", c, s, "ir")]
            for arm in ARMS:
                fu = seed_mean[(arm, c, s, "fused")]
                cl = ci(fu - np.maximum(vis, ir))
                fail = cl[0] <= -FLOOR and cl[2] < 0
                holds[arm] += not fail
                dd = ci(fu - seed_mean[("shipped", c, s, "fused")]) if arm != "shipped" else None
                L.append(f"| {c}/{s} | {vis.mean():.4f} | {ir.mean():.4f} | {arm} | {fu.mean():.4f} | "
                         f"{seed_mean[(arm, c, s, 'veto')].mean():.1%} | {f(cl)} {'**fails**' if fail else 'holds'} | "
                         f"{f(dd) if dd else '—'} |")
                js.append({"cell": f"{c}/{s}", "arm": arm, "vis": float(vis.mean()), "ir": float(ir.mean()),
                           "fused": float(fu.mean()), "veto": float(seed_mean[(arm, c, s, 'veto')].mean()),
                           "claim": list(cl), "fails": bool(fail), "delta_vs_shipped": list(dd) if dd else None,
                           "per_system_fused": fu.tolist()})
    n = len(CONDS) * 2
    L += ["", "Cells holding the claim, of " + str(n) + ": " + ", ".join(f"{a} {holds[a]}" for a in ARMS) + "."]
    out = ROOT / args.out
    out.write_text("# The veil repair under corruption v2\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"cells": js, "holds": holds, "elapsed_s": time.time() - t0},
                                                   indent=1), encoding="utf-8")
    print("\n".join(L))
    print(f"[veil] wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    raise SystemExit(main())
