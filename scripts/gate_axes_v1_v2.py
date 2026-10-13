"""How the shipped gate's own statistics respond to v1 and v2 corruptions (2026-10-10).

The `crossmodal26m` veto reads three VIS statistics: `dark` (5th-percentile grey < mu_b),
`veil` (grad_gini < its novelty bound) and, in the weak-IR fallback, `concentrated`
(lap_over_var above its bound). Their thresholds were fitted on clean frames, but the claims
made about them ("veil is 100% correct on fog") were measured on v1 frames. This reports the
firing rate of each axis per condition and day/night slice, v1 against v2, from the statistics
files the scorers read, so that every difference between a v1 and a v2 cell can be traced to
the axis that moved.

Sources: development Phase 3 statistics (VIS draws 941-944, runs/derived_p3dev{,_v2}) and the
R-D1 mirror (seed 1, runs/derived and runs/derived_m_v2). Statistics only; no detection.

    py -3.13 scripts/gate_axes_v1_v2.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONDS = ("clean", "fog", "lowlight", "glare", "fog_s1")
DRAWS = (941, 942, 943, 944)
SETS = {"dev v1": ("runs/derived_p3dev", True), "dev v2": ("runs/derived_p3dev_v2", True),
        "R-D1 v1": ("runs/derived", False), "R-D1 v2": ("runs/derived_m_v2", False)}


def load(root: str, sub: str | None, cond: str):
    d = ROOT / root
    b = d / "brightness" / (sub or "") / f"gauss_vis_paired_{cond}.json"
    s = d / "structure" / (sub or "") / f"gauss_vis_paired_{cond}.json"
    if not (b.is_file() and s.is_file()):
        return None
    B = json.loads(b.read_text(encoding="utf-8"))["frames"]
    S = json.loads(s.read_text(encoding="utf-8"))["frames"]
    return (np.array([f["p05"] for f in B]), np.array([f["grad_gini"] for f in S]),
            np.array([f["lap_over_var"] for f in S]))


def main() -> int:
    sc = json.loads((ROOT / "runs/eval/structure_constants.json").read_text(encoding="utf-8"))["axes"]
    bc = json.loads((ROOT / "runs/eval/brightness_constants.json").read_text(encoding="utf-8"))["vis"]
    g_t, lov_t, mu = sc["grad_gini"]["threshold"], sc["lap_over_var"]["threshold"], bc["mu_b"]
    imgs = [ln.strip() for ln in (ROOT / "runs/derived/paired_val_vis.txt").read_text(encoding="utf-8").splitlines() if ln.strip()]
    night = np.array([Path(p).parent.name == "pohang01" for p in imgs])
    L = [f"Thresholds: dark = p05 < {mu}, veil = grad_gini < {g_t:.4f}, concentrated = lap_over_var > {lov_t:.3f} "
         "(fitted on clean frames). Firing rate per slice; dev rows average draws 941-944. "
         "`crossmodal26m` vetoes VIS at night on `dark OR veil`; by day it never vetoes with clean IR.", "",
         "| condition | slice | set | dark | veil | concentrated | dark OR veil | median p05 | median grad_gini |",
         "|---|---|---|---:|---:|---:|---:|---:|---:|"]
    js = []
    for cond in CONDS:
        for sl, m in (("day", ~night), ("night", night)):
            for name, (root, dev) in SETS.items():
                subs = [f"draw{v}_{v + 10}" for v in DRAWS] if dev else [None]
                got = [load(root, s, cond) for s in subs]
                got = [g for g in got if g is not None]
                if not got:
                    continue
                r = {k: float(np.mean([f(g)[m].mean() for g in got])) for k, f in (
                    ("dark", lambda g: g[0] < mu), ("veil", lambda g: g[1] < g_t),
                    ("concentrated", lambda g: g[2] > lov_t), ("dark_or_veil", lambda g: (g[0] < mu) | (g[1] < g_t)))}
                med_p05 = float(np.median(np.concatenate([g[0][m] for g in got])))
                med_g = float(np.median(np.concatenate([g[1][m] for g in got])))
                L.append(f"| {cond} | {sl} | {name} ({len(got)}) | {r['dark']:.3f} | {r['veil']:.3f} | "
                         f"{r['concentrated']:.3f} | {r['dark_or_veil']:.3f} | {med_p05:.1f} | {med_g:.3f} |")
                js.append({"cond": cond, "slice": sl, "set": name, "n_draws": len(got), **r,
                           "median_p05": med_p05, "median_grad_gini": med_g})
    out = ROOT / "docs/eval/corruption_v2/gate_axes.md"
    out.write_text("# The shipped gate's VIS axes under v1 and v2 corruptions\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
