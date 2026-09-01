"""Candidate architectures for the gated VIS-IR fusion system, measured head to head.

Everything here follows from three measurements made 2026-09-01, each of which
contradicts a load-bearing assumption in `docs/gated-fusion-handoff.md`:

1. **The Mahalanobis soft term is not merely inert, it is the mechanism of the
   lowlight/day loss.** lowlight pushes VIS's D to ~248 against a `mu_d` of 71, so
   `r_frame_vis` collapses to ~0.009 while clean IR keeps ~0.82. After the
   capability prior that leaves mean `w_vis` at 0.320 on a cell where VIS scores
   0.0346 and IR 0.0177 -- the gate hands the frame to the weaker sensor. With the
   term removed and no veto, the same cell scores 0.0381, ABOVE VIS alone. The
   handoff's §6 ablation saw this as "within noise, positive on day" because it
   was measured with the veto still on, and the veto had already thrown the cell
   away.

2. **The photometric axis is the reason lowlight/day cannot be kept.** Not a
   limitation of thresholds: lowlight/day has p05 = 0, which is DARKER than the
   real night run (p05 2.5-3.5) on frames where the detector still works. Any
   statistic monotone in brightness must therefore rank digitally-dimmed daylight
   below real night, which is exactly backwards. §7.2 asks for a rule that keeps
   lowlight/day; the answer is to stop asking brightness.

3. **`grad_gini` is a clean veil detector where `lap_var` needed a filter.** The
   Gini coefficient of gradient magnitude is scale-free, and at a hard novelty
   bound on clean fit frames it fires on 100% of fog/day and 100% of fog/night and
   0.0% of every other cell, with no temporal filtering at all.

The arms below are the systems those three facts suggest, plus the adopted system
as the reference. Soft factors are ratios or sigmoids of constants ALREADY fitted
in the repo -- no new constant is introduced by any arm.

Writes a new report; overwrites nothing.

Usage:
    python scripts/eval_architecture_v2.py --out runs/eval/architecture_v2.md --n-boot 1000
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
from uqfusion.eval.ctx import FIT_RUNS, NIGHT_RUNS, load_context, run_systems    # noqa: E402
from uqfusion.eval.hysteresis import filter_veto                                 # noqa: E402

SHIP, BUOY = 0, 1
STRUCT_DIR = ROOT / "runs" / "derived" / "structure"


# ---------------------------------------------------------------------------
# signals
# ---------------------------------------------------------------------------

def load_structure(conditions) -> tuple[dict, dict] | tuple[None, None]:
    need = {c: STRUCT_DIR / f"gauss_vis_paired_{c}.json" for c in conditions}
    train = STRUCT_DIR / "gauss_vis_train_clean.json"
    if not train.is_file() or not all(p.is_file() for p in need.values()):
        return None, None
    st = {}
    for c, p in need.items():
        fr = json.loads(p.read_text(encoding="utf-8"))["frames"]
        st[c] = {k: np.asarray([f[k] for f in fr], dtype=float)
                 for k, v in fr[0].items() if isinstance(v, (int, float))}
    tf = json.loads(train.read_text(encoding="utf-8"))["frames"]
    m = np.asarray([f["run"] in FIT_RUNS for f in tf])
    fit = {k: np.asarray([f[k] for f in tf], dtype=float)[m]
           for k, v in tf[0].items() if isinstance(v, (int, float))}
    return st, fit


def max_conf(records) -> np.ndarray:
    return np.asarray([float(np.asarray(r["conf"], dtype=float).max())
                       if len(r["conf"]) else 0.0 for r in records])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/architecture_v2.md")
    ap.add_argument("--n-boot", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=None)
    args = ap.parse_args()

    t0 = time.time()
    kw = {"conditions": tuple(args.conditions)} if args.conditions else {}
    ctx = load_context(**kw)
    n = ctx.n()
    night = np.isin(ctx.runs, NIGHT_RUNS)
    fitm = np.isin(ctx.runs, FIT_RUNS)
    splits = {"day": ~night, "night": night}
    cells = [(c, s) for c in ctx.conditions for s in ("day", "night")]

    st, st_fit = load_structure(ctx.conditions)
    if st is None:
        raise SystemExit("runs/derived/structure is incomplete — run scripts/frame_structure.py")
    gini_thr = float(st_fit["grad_gini"].min())      # veil fires BELOW this
    conc_thr = float(st_fit["lap_over_var"].max())   # concentrated fires ABOVE this
    # Per-frame detector-evidence bound. w=1 deliberately: a frame with no VIS
    # detections has nothing for the factor to scale, so the windowing that a
    # VETO needs (to avoid punishing a quiet clean frame) is unnecessary here and
    # would only blur the per-frame distinction the soft path exists to keep.
    ev_thr = float(max_conf(ctx.vis_by_cond["clean"])[fitm].min())
    print(f"[v2] novelty bounds: grad_gini<{gini_thr:.4f}  lap_over_var>{conc_thr:.3f}  "
          f"max_conf<{ev_thr:.4f}")

    cap_only = replace(ctx, c_vis=replace(ctx.c_vis, mu_d=1e9, lam=0.0),
                       c_ir=replace(ctx.c_ir, mu_d=1e9, lam=0.0))
    no_veto = ([False] * n, [False] * n)

    def photometric(cond):
        r = 1.0 / (1.0 + np.exp(-(ctx.bright_by_cond[cond] - ctx.c_vis.mu_b)
                                / max(ctx.c_vis.tau_b, 1e-9)))
        return np.asarray(filter_veto((r < ctx.veto).tolist(), ctx.order, 15, "dilate"), bool)

    def veil_lap(cond):
        return np.asarray(filter_veto((ctx.struct_by_cond[cond] < ctx.tau_lap).tolist(),
                                      ctx.order, 15, "majority"), bool)

    def veil_gini(cond):
        return st[cond]["grad_gini"] < gini_thr

    def concentrated(cond, k=31):
        m = st[cond]["lap_over_var"] > conc_thr
        return np.asarray(filter_veto(m.tolist(), ctx.order, k, "dilate"), bool) if k > 1 else m

    def evid_factor(cond):
        return np.clip(max_conf(ctx.vis_by_cond[cond]) / max(ev_thr, 1e-12), 0.0, 1.0)

    # --- the cross-modal night test ----------------------------------------
    # The photometric axis asks VIS "are you dark?" and cannot tell a dark WORLD
    # from a dark SENSOR -- which is the whole of §7.2, since lowlight/day has
    # p05 = 0, darker than the real night run's 2.5-3.5, on frames where the
    # detector still works. The other sensor can tell them apart: on lowlight/day
    # the IR frame is an ordinary daytime frame, because the corruption was applied
    # to VIS alone. Asking IR instead is the standard cross-modal consistency
    # check -- if one sensor reports darkness and the other reports daylight, the
    # sensor is the anomaly, not the scene.
    #
    # Fitted as a novelty bound on the SAME clean fit runs (all daylight), so no
    # night frame informs the threshold. Two honest caveats: Pohang IR is 8-bit via
    # per-frame min-max normalization (OQ-3), so `p05` here measures how much of a
    # frame sits at the low end of ITS OWN thermal range -- at night the sea/sky
    # contrast collapses and that floor rises; and IR is uncorrupted in all eight
    # cells of this benchmark, so this test is being graded on the easiest version
    # of its job. A deployment where both sensors can degrade needs the check run
    # both ways.
    ir_p05 = np.asarray(
        [f["p05"] for f in json.loads(
            (ROOT / "runs/derived/brightness/gauss_ir_paired_clean.json").read_text(
                encoding="utf-8"))["frames"]], dtype=float)
    ir_night_thr = float(ir_p05[fitm].max())

    def ir_night(cond):
        return ir_p05 > ir_night_thr

    # arm -> (base ctx, veto fn or None, soft-factor fn or None, single_passthrough)
    ARMS: dict[str, tuple] = {
        "0 adopted (photometric OR veil, full soft weights)":
            (ctx, lambda c: photometric(c) | veil_lap(c), None),
        "1 adopted veto, capability-only weights":
            (cap_only, lambda c: photometric(c) | veil_lap(c), None),
        "2 gini-veil veto, capability-only weights":
            (cap_only, veil_gini, None),
        "3 gini-veil veto + per-frame evidence scaling, capability-only":
            (cap_only, veil_gini, evid_factor),
        "4 gini-veil OR (dark AND concentrated) veto, capability-only":
            (cap_only, lambda c: veil_gini(c) | (photometric(c) & concentrated(c)), None),
        "5 gini-veil OR (dark AND concentrated) + evidence scaling, capability-only":
            (cap_only, lambda c: veil_gini(c) | (photometric(c) & concentrated(c)), evid_factor),
        "6 no veto, evidence scaling only, capability-only":
            (cap_only, None, evid_factor),
        "7 gini-veil OR IR-night veto, capability-only weights":
            (cap_only, lambda c: veil_gini(c) | ir_night(c), None),
        "8 arm 7 + single_passthrough":
            (cap_only, lambda c: veil_gini(c) | ir_night(c), None, True),
        "9 adopted veto + single_passthrough, full soft weights":
            (ctx, lambda c: photometric(c) | veil_lap(c), None, True),
    }

    parts, rates = {}, {}
    for cond in ctx.conditions:
        for arm, spec in ARMS.items():
            base, vfn, sfn = spec[0], spec[1], spec[2]
            over = {"single_passthrough": bool(len(spec) > 3 and spec[3])}
            if vfn is None:
                over["veto_override"] = no_veto
                vm = np.zeros(n, bool)
            else:
                vm = np.asarray(vfn(cond), bool)
                over["veto_override"] = (vm.tolist(), [False] * n)
            if sfn is not None:
                over["trust_override"] = ((base.cap_vis * sfn(cond)).tolist(),
                                          [base.cap_ir] * n)
            res = run_systems(base, cond, **over)
            parts[(arm, cond)] = frame_parts(res["fused_gated"], ctx.gts)
            for s, sel in splits.items():
                rates[(arm, cond, s)] = float(vm[sel].mean())
            if arm == list(ARMS)[0]:
                parts[("_vis", cond)] = frame_parts(ctx.vis_by_cond[cond], ctx.gts)
                parts[("_ir", cond)] = frame_parts(res["ir_in_vis"], ctx.gts)
        print(f"[v2] {cond}: {len(ARMS)} arms done ({time.time() - t0:.0f}s)", flush=True)

    def ap(key, cond, s, cls=SHIP):
        sel = np.flatnonzero(splits[s])
        r = ap_from_parts([parts[(key, cond)][i] for i in sel])
        if cls is None:
            return r["map50_95"]
        e = r["per_class"].get(cls)
        return float(e["ap50_95"]) if e else 0.0

    bar = {(c, s): max(ap("_vis", c, s), ap("_ir", c, s)) for c, s in cells}

    rows = []
    for arm in ARMS:
        cs = {f"{c}/{s}": ap(arm, c, s) for c, s in cells}
        gaps = [cs[f"{c}/{s}"] - bar[(c, s)] for c, s in cells]
        rows.append({"arm": arm, "cells": cs, "worst_gap": float(min(gaps)),
                     "sum_gap": float(sum(gaps)),
                     "rates": {f"{c}/{s}": rates[(arm, c, s)] for c, s in cells}})
    ranked = sorted(rows, key=lambda r: (-r["worst_gap"], -r["sum_gap"]))

    L = ["# Candidate architectures, measured (ship AP)", "",
         "Ship AP only — IR is nc=1 ship-only, so ship is the one class both "
         "streams can produce (handoff §5).", "",
         "Every arm shares the caches, the homography, WBF and `iou_thr` 0.85. "
         "What changes is the WEIGHTS (full soft gate vs capability prior alone), "
         "the VETO rule, and whether a per-frame trust multiplies VIS's scores.", "",
         f"Novelty bounds, all fitted on CLEAN frames of {list(FIT_RUNS)} with "
         f"pohang01 and every corrupted condition held out: `grad_gini` < "
         f"{gini_thr:.4f}, `lap_over_var` > {conc_thr:.3f}, `max_conf` < "
         f"{ev_thr:.4f}, IR `p05` > {ir_night_thr:.1f}.", "",
         "`bar` = max(VIS, IR) per cell. Ranked by WORST cell, because a gate is a "
         "safety mechanism and its worth is set by the condition it handles least "
         "well.", "",
         "| arm | " + " | ".join(f"{c}/{s}" for c, s in cells) + " | worst gap |",
         "|---|" + "---:|" * (len(cells) + 1)]
    for lbl, key in (("_bar (max VIS, IR)_", None), ("_VIS only_", "_vis"),
                     ("_IR only_", "_ir")):
        vals = [bar[(c, s)] if key is None else ap(key, c, s) for c, s in cells]
        L.append(f"| {lbl} | " + " | ".join(f"{v:.4f}" for v in vals) + " | — |")
    for r in ranked:
        L.append(f"| {r['arm']} | "
                 + " | ".join(f"{r['cells'][f'{c}/{s}']:.4f}" for c, s in cells)
                 + f" | {r['worst_gap']:+.4f} |")

    L += ["", "## Gap to bar", "",
          "| arm | " + " | ".join(f"{c}/{s}" for c, s in cells) + " | sum |",
          "|---|" + "---:|" * (len(cells) + 1)]
    for r in ranked:
        L.append(f"| {r['arm']} | "
                 + " | ".join(f"{r['cells'][f'{c}/{s}'] - bar[(c, s)]:+.4f}" for c, s in cells)
                 + f" | {r['sum_gap']:+.4f} |")

    L += ["", "## VIS veto rate", "",
          "| arm | " + " | ".join(f"{c}/{s}" for c, s in cells) + " |",
          "|---|" + "---:|" * len(cells)]
    for r in ranked:
        L.append(f"| {r['arm']} | "
                 + " | ".join(f"{r['rates'][f'{c}/{s}']:.1%}" for c, s in cells) + " |")

    boot = []
    if args.n_boot:
        best = ranked[0]["arm"]
        base = list(ARMS)[0]
        for cond, s in cells:
            sel = np.flatnonzero(splits[s])
            boot.append({"cell": f"{cond}/{s}", **bootstrap_delta(
                [parts[(best, cond)][i] for i in sel],
                [parts[(base, cond)][i] for i in sel],
                None, n_boot=args.n_boot, cls=SHIP)})
        L += ["", f"## `{best}` vs `{base}` — paired frame bootstrap "
              f"(n={args.n_boot}, seed 0, ship AP)", "",
              "| cell | best | adopted | delta | 95% CI | sign flips |",
              "|---|---:|---:|---:|---|---:|"]
        for b in boot:
            L.append(f"| {b['cell']} | {b['a']:.4f} | {b['b']:.4f} | {b['delta']:+.4f} | "
                     f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"
                     f"{' (spans 0)' if b['spans_zero'] else ''} | {b['p_sign_flip']:.1%} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"bounds": {"grad_gini": gini_thr, "lap_over_var": conc_thr, "max_conf": ev_thr},
         "bar": {f"{c}/{s}": bar[(c, s)] for c, s in cells},
         "vis": {f"{c}/{s}": ap("_vis", c, s) for c, s in cells},
         "ir": {f"{c}/{s}": ap("_ir", c, s) for c, s in cells},
         "arms": ranked, "bootstrap": boot}, indent=2), encoding="utf-8")
    print(f"[v2] wrote {out} in {time.time() - t0:.0f}s")
    for r in ranked:
        print(f"[v2] {r['worst_gap']:+.4f}  (sum {r['sum_gap']:+.4f})  {r['arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
