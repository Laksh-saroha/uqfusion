"""Does "fused ≥ max(VIS, IR)" hold on the corrupted cells for the Phase 3 checkpoints?

The eight-cell claim ({clean, fog, lowlight, glare} × {day, night}) was measured on the
pre-restore checkpoints. `scripts/p3_night_check.py` re-measured clean/clean on the five
Phase 3 systems; this does the three corrupted VIS conditions, at A9.2's four VIS draws
941–944, from the development caches and statistics built by `p3dev_corruption_prep.py`.

Logged in `docs/exposure-ledger-2026-09-09.md` §7 before it ran; the fail criterion is fixed
there. **Reads no pohang04 frame.** Descriptive: adopts nothing.

Per seed k and draw v: VIS-only, IR-only, and fused as shipped (`crossmodal26m`), ship AP,
day and night separately. Per seed, the four draws are averaged; the decision interval is the
between-seed 95% t-interval (df 4) on the draw-averaged deltas. The per-draw sd is reported
as corruption-draw noise.

Usage:
    python scripts/p3_corrupt_cells.py --out docs/eval/p3_corrupt_cells_2026-09-27.md
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

from _ideas_common import write_md                                      # noqa: E402
from uqfusion.eval.apmetrics import ap_from_parts, frame_parts          # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems     # noqa: E402
from uqfusion.eval.identity import system_identity                      # noqa: E402

PRESET = "crossmodal26m"
SEEDS = (0, 1, 2, 3, 4)
DRAWS = (941, 942, 943, 944)
CONDS = ("fog", "lowlight", "glare")
SHIP = 0
FLOOR = 0.0060
FLOORS = (0.0014, 0.0031, 0.0060, 0.0100)
T_975_DF4 = 2.7764451051977987
ARMS = ("vis", "ir", "fused")


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def ship_ap(parts, sel) -> float:
    e = ap_from_parts(parts, sel=sel)["per_class"].get(SHIP)
    return float(e["ap50_95"]) if e else float("nan")


def score_one(job):
    """One (seed, draw) context: ship AP per arm and veto rate per slice, for every condition."""
    k, v, conds, cache_root, stats_root, verbose = job
    ctx = load_context(preset=PRESET, cache_dir=f"{cache_root}/seed{k}/{sub(v)}",
                       bright_dir=f"{stats_root}/brightness/{sub(v)}",
                       structure_dir=f"{stats_root}/structure/{sub(v)}",
                       conditions=conds, verbose=verbose)
    if any("pohang04" in r for r in ctx.runs):
        raise SystemExit(f"seed {k} draw {v}: a pohang04 frame reached the development context")
    paths = [r["image_path"] for r in ctx.vis_by_cond[conds[0]]]
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    ap_, veto_ = {}, {}
    for c in conds:
        out = run_systems(ctx, c)
        parts = {"vis": frame_parts(ctx.vis_by_cond[c], ctx.gts),
                 "ir": frame_parts(out["ir_in_vis"], ctx.gts),
                 "fused": frame_parts(out["fused_gated"], ctx.gts)}
        vv = np.asarray(out["veto_vis"], dtype=bool)
        for s, idx in sl.items():
            for a in ARMS:
                ap_[(c, s, a)] = ship_ap(parts[a], idx)
            veto_[(c, s)] = float(vv[idx].mean())
    return k, v, paths, ap_, veto_


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/eval/p3_corrupt_cells_2026-09-27.md")
    ap.add_argument("--conds", nargs="+", default=list(CONDS),
                    help="VIS conditions; each needs caches and statistics under the draw directories")
    ap.add_argument("--cache-root", default="runs/cache_p3dev",
                    help="runs/cache_p3dev_v2 scores the corruption-v2 rebuild (docs/eval/corruption_v2/)")
    ap.add_argument("--stats-root", default="runs/derived_p3dev")
    ap.add_argument("--workers", type=int, default=10, help="(seed, draw) contexts scored in parallel")
    args = ap.parse_args()
    conds = tuple(args.conds)
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; this report is written once")
    t0 = time.time()

    # ap_[cond][slice][arm] -> array (seed, draw); veto_[cond][slice] -> array (seed, draw)
    ap_ = {c: {s: {a: np.full((len(SEEDS), len(DRAWS)), np.nan) for a in ARMS} for s in ("day", "night")}
           for c in conds}
    veto_ = {c: {s: np.full((len(SEEDS), len(DRAWS)), np.nan) for s in ("day", "night")} for c in conds}
    from concurrent.futures import ProcessPoolExecutor

    jobs = [(k, v, conds, args.cache_root, args.stats_root, i == 0 and j == 0)
            for i, k in enumerate(SEEDS) for j, v in enumerate(DRAWS)]
    with ProcessPoolExecutor(max(1, args.workers)) as ex:
        results = list(ex.map(score_one, jobs))
    paths0 = results[0][2]
    for k, v, p, a_, ve_ in results:
        if p != paths0:
            raise SystemExit(f"seed {k} draw {v}: frame order differs")
        i, j = SEEDS.index(k), DRAWS.index(v)
        for (c, s, a), x in a_.items():
            ap_[c][s][a][i, j] = x
        for (c, s), x in ve_.items():
            veto_[c][s][i, j] = x
    ctx0 = load_context(preset=PRESET, cache_dir=f"{args.cache_root}/seed{SEEDS[0]}/{sub(DRAWS[0])}",
                        bright_dir=f"{args.stats_root}/brightness/{sub(DRAWS[0])}",
                        structure_dir=f"{args.stats_root}/structure/{sub(DRAWS[0])}",
                        conditions=conds, verbose=False)
    print(f"[corrupt] {len(results)} (seed, draw) contexts scored ({time.time() - t0:.0f}s)", flush=True)

    res = {}
    for c in conds:
        for s in ("day", "night"):
            A = ap_[c][s]
            seed_mean = {a: A[a].mean(axis=1) for a in ARMS}             # draw-averaged, per seed
            d = {}
            for a in ("vis", "ir"):
                x = seed_mean["fused"] - seed_mean[a]
                m, sd = float(x.mean()), float(x.std(ddof=1))
                h = T_975_DF4 * sd / np.sqrt(len(x))
                d[f"fused-{a}"] = {"per_seed": x.tolist(), "mean": m, "seed_ci": [m - h, m + h],
                                   "below_by_floor": {str(f): bool(m <= -f and m + h < 0) for f in FLOORS}}
            fails = any(d[k]["mean"] <= -FLOOR and d[k]["seed_ci"][1] < 0 for k in d)
            res[f"{c}/{s}"] = {
                "ap_mean": {a: float(seed_mean[a].mean()) for a in ARMS},
                "ap_seed_sd": {a: float(seed_mean[a].std(ddof=1)) for a in ARMS},
                "ap_draw_sd_mean": {a: float(A[a].std(axis=1, ddof=1).mean()) for a in ARMS},
                "ap_per_seed_draw": {a: A[a].tolist() for a in ARMS},
                "veto_rate_mean": float(np.nanmean(veto_[c][s])),
                "deltas": d, "claim_fails": bool(fails)}
            r = res[f"{c}/{s}"]
            print(f"[corrupt] {c}/{s}: vis {r['ap_mean']['vis']:.4f} ir {r['ap_mean']['ir']:.4f} "
                  f"fused {r['ap_mean']['fused']:.4f} veto {r['veto_rate_mean']:.3f} claim_fails={fails}",
                  flush=True)

    def drow(key):
        r = res[key]
        cells = []
        for a in ("vis", "ir"):
            x = r["deltas"][f"fused-{a}"]
            cells.append(f"{x['mean']:+.4f} [{x['seed_ci'][0]:+.4f}, {x['seed_ci'][1]:+.4f}]")
        return (f"| {key} | {r['ap_mean']['vis']:.4f} | {r['ap_mean']['ir']:.4f} | **{r['ap_mean']['fused']:.4f}** "
                f"| {cells[0]} | {cells[1]} | {r['veto_rate_mean']:.3f} | "
                + ("**FAILS**" if r["claim_fails"] else "holds") + " |")

    L = [
        "Logged in `docs/exposure-ledger-2026-09-09.md` §7 (2026-09-27) before it ran. **No pohang04 frame "
        "is read.** Ship AP (class 0, AP50-95, local convention), preset `crossmodal26m`, five Phase 3 "
        "systems, VIS conditions " + ", ".join(f"`{c}`" for c in conds) + ", VIS draws 941–944, caches `" + args.cache_root + "/seed{k}/draw{v}_{v+10}/`, statistics "
        "`" + args.stats_root + "/`. Day and night never pooled. **Adopts nothing.**",
        "## Per cell\n\nAP is the mean over seeds of the draw-averaged value. Deltas carry the between-seed "
        "95% t-interval (df 4) on draw-averaged per-seed deltas, the decision interval. **Fails** = a delta "
        "≤ −0.0060 with its interval entirely below zero (fixed before the run).\n\n"
        "| cell | VIS only | IR only | fused | fused − VIS | fused − IR | VIS veto rate | fused ≥ max |\n"
        "|---|---:|---:|---:|---|---|---:|---|\n" + "\n".join(drow(k) for k in res),
        "## Noise\n\n| cell | between-seed sd (fused) | mean within-seed draw sd (fused) |\n|---|---:|---:|\n"
        + "\n".join(f"| {k} | {r['ap_seed_sd']['fused']:.4f} | {r['ap_draw_sd_mean']['fused']:.4f} |"
                    for k, r in res.items()),
        "The clean cells are in `docs/eval/p3_night_check_2026-09-27.md` (clean day holds, +0.0081; clean "
        "night fails, −0.1847).",
    ]
    ident = system_identity(ctx0, preset=PRESET, seeds=list(SEEDS), draws=list(DRAWS), conditions=list(conds),
                            caches=args.cache_root + "/seed{k}/draw{v}_{v+10}")
    write_md(Path(args.out), "Phase 3 corrupted cells — does fused ≥ max(VIS, IR) survive the retrain?",
             L, identity=ident)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"cells": res, "config": {"preset": PRESET, "seeds": list(SEEDS), "draws": list(DRAWS),
                                  "conditions": list(conds), "floor": FLOOR, "ship_class": SHIP,
                                  "cache_root": args.cache_root, "stats_root": args.stats_root}},
        indent=1), encoding="utf-8")
    print(f"[corrupt] done in {time.time() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
