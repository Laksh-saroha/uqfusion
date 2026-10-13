"""Sensitivity rows for the unmeasured or ridge-fitted corruption v2 constants (2026-10-10).

Each row moves ONE constant of `corruptions_v2.DEFAULTS` and rebuilds that condition's caches
(`build_corruption_v2_dev.py --draws 941 --conds <row>`); everything else is the shipped v2
cell. Scored here exactly as `p3_corrupt_cells.py` scores a cell: preset `crossmodal26m`, the
five Phase 3 systems, ship AP of the VIS-only, IR-only and fused arms, day and night apart.
Reported as the row minus its baseline, paired by system (the frames and the corruption draw
are the same; only the constant differs), mean and t-interval over the five systems.
Descriptive: it says how much a stated assumption moves a number, nothing is chosen by it.

    py -3.13 scripts/v2_sensitivity.py --out docs/eval/corruption_v2/sensitivity.md
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

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts   # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402

PRESET, SEEDS, DRAW, SHIP = "crossmodal26m", (0, 1, 2, 3, 4), 941, 0
T_975_DF4 = 2.7764451051977987
CACHE, STATS = "runs/cache_p3dev_v2", "runs/derived_p3dev_v2"
#: (row, baseline, (vis condition, ir condition) of the row, of the baseline, what moved)
ROWS = [
    ("glare_ae0", "glare", ("glare_ae0", None), ("glare", None), "AE_STRENGTH 0.375 -> 0 (no AE reaction)"),
    ("glare_ae1", "glare", ("glare_ae1", None), ("glare", None), "AE_STRENGTH 0.375 -> 1 (fully mean-holding AE)"),
    ("glare_i0half", "glare", ("glare_i0half", None), ("glare", None), "source intensity x0.5"),
    ("glare_i0x2", "glare", ("glare_i0x2", None), ("glare", None), "source intensity x2"),
    ("glare_rev2", "glare", ("glare_rev2", None), ("glare", None),
     "revision-2 glare: Lorentzian spread (beta 1), its fitted intensities, no AE"),
    ("fog_ae0", "fog", ("fog_ae0", None), ("fog", None), "AE_STRENGTH 0.375 -> 0 on fog"),
    ("fog_ae1", "fog", ("fog_ae1", None), ("fog", None), "AE_STRENGTH 0.375 -> 1 on fog (a converged AE in persistent fog)"),
    ("ir_fog_b03", "ir_fog_s2", ("clean", "fog_s2_b03"), ("clean", "fog_s2"), "IR_BETA_RATIO 0.5 -> 0.3"),
    ("ir_fog_b10", "ir_fog_s2", ("clean", "fog_s2_b10"), ("clean", "fog_s2"), "IR_BETA_RATIO 0.5 -> 1.0"),
]
ARMS = ("vis", "ir", "fused")


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def ship_ap(parts, sel) -> float:
    e = ap_from_parts(parts, sel=sel)["per_class"].get(SHIP)
    return float(e["ap50_95"]) if e else float("nan")


def score(job):
    """One system, one (vis conditions, ir condition) group -> {(vis, ir, slice, arm): AP}."""
    k, vis_conds, ir_cond = job
    d = f"{STATS}/brightness/{sub(DRAW)}"
    ctx = load_context(preset=PRESET, cache_dir=f"{CACHE}/seed{k}/{sub(DRAW)}", bright_dir=d,
                       structure_dir=f"{STATS}/structure/{sub(DRAW)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json",
                       conditions=tuple(vis_conds), ir_condition=ir_cond, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    out = {}
    for c in vis_conds:
        r = run_systems(ctx, c)
        parts = {"vis": frame_parts(ctx.vis_by_cond[c], ctx.gts), "ir": frame_parts(r["ir_in_vis"], ctx.gts),
                 "fused": frame_parts(r["fused_gated"], ctx.gts)}
        vv = np.asarray(r["veto_vis"], dtype=bool)
        for s, idx in sl.items():
            for a in ARMS:
                out[(c, ir_cond, s, a)] = ship_ap(parts[a], idx)
            out[(c, ir_cond, s, "veto")] = float(vv[idx].mean())
    return k, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/eval/corruption_v2/sensitivity.md")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    groups: dict = {}
    for _r, _b, row, base, _w in ROWS:
        for vc, ic in (row, base):
            groups.setdefault(ic, set()).add(vc)
    jobs = [(k, sorted(vcs), ic) for ic, vcs in groups.items() for k in SEEDS]
    res: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, out in ex.map(score, jobs):
            for key, v in out.items():
                res.setdefault(key, {})[k] = v
    print(f"[sens] {len(jobs)} contexts scored ({time.time() - t0:.0f}s)", flush=True)

    L = ["One constant moved per row; shipped v2 cell as the baseline; five Phase 3 systems; VIS draw "
         f"{DRAW} (IR {DRAW + 10}); paired val ({'day = pohang00/02/03, night = pohang01'}). Ship AP. "
         "Δ = row − baseline, mean over the five systems, 95% t-interval (df 4).", "",
         "| row | what moved | slice | baseline fused | Δ fused [95% CI] | Δ VIS-only | Δ IR-only | veto rate (row / base) |",
         "|---|---|---|---:|---|---:|---:|---|"]
    js = []
    for name, _b, (vc, ic), (bvc, bic), what in ROWS:
        for s in ("day", "night"):
            def arr(c, i, a):
                return np.array([res[(c, i, s, a)][k] for k in SEEDS])
            base_f = arr(bvc, bic, "fused")
            dd = {a: arr(vc, ic, a) - arr(bvc, bic, a) for a in ARMS}
            m, sd = float(dd["fused"].mean()), float(dd["fused"].std(ddof=1))
            h = T_975_DF4 * sd / np.sqrt(len(SEEDS))
            L.append(f"| {name} | {what} | {s} | {base_f.mean():.4f} | {m:+.4f} [{m - h:+.4f}, {m + h:+.4f}] | "
                     f"{dd['vis'].mean():+.4f} | {dd['ir'].mean():+.4f} | "
                     f"{arr(vc, ic, 'veto').mean():.3f} / {arr(bvc, bic, 'veto').mean():.3f} |")
            js.append({"row": name, "what": what, "slice": s, "baseline_fused": float(base_f.mean()),
                       "d_fused": m, "d_fused_ci": [m - h, m + h], "d_vis": float(dd["vis"].mean()),
                       "d_ir": float(dd["ir"].mean()),
                       "per_system": {a: dd[a].tolist() for a in ARMS}})
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("# Corruption v2 sensitivity rows\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"rows": js, "elapsed_s": time.time() - t0}, indent=1),
                                        encoding="utf-8")
    print(f"[sens] wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
