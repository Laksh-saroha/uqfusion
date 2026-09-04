"""Re-price every veto axis against the detectors that are actually deployed.

`docs/levers-and-the-26m-swap-2026-09-01.md` §2 found the veil axis to be a
-0.0632 regression on the full-scale checkpoints, and the reason generalises to
every other axis in the gate:

    A veto axis measures the IMAGE. The threshold is fitted on pixels and is
    detector-independent -- `verify_cache_m.py` shows every one of them is
    identical across the 26s -> 26m swap to 0.000e+00. But the DECISION attached
    to it is a claim about the detector: "when this fires, the stream I am about
    to delete is worse than the one I am keeping." That claim is not in the image
    and does not survive a detector change for free.

Fog was simply the axis where the claim broke loudest. The photometric axis, the
IR merge bound, the IR authority bound and the VIS health bound were all fitted or
validated against yolo26s behaviour and none has been re-asked.

Three instruments, cheapest first.

**1. THE CLAIM TEST.** For each axis, on exactly the frames where it FIRES, score
both streams and report `margin = AP(kept) - AP(removed)`. A negative margin means
the axis is deleting the better sensor -- which is what fog showed at -0.0632 and
what nothing else was ever checked for. No fusion run is needed, so this covers
every axis on every cell.

Two honest caveats. AP over a frame subset is not that subset's marginal
contribution to the pooled AP, so a margin is a diagnostic and not the arm's cost;
§2 measures the cost. And on NIGHT frames VIS scores exactly 0.0000, so every
VIS-removing axis is trivially right there and only the day column carries
information.

**2. THE ABLATION.** For the axes instrument 1 flags, and for the two safety bounds
regardless, run the system with the axis removed or its bound moved and report the
gap to `max(VIS, IR)` on every cell.

**3. THE SAFETY SIDE.** `probe_ir_night_robustness.py` already showed a fogged IR
misreads 95% of clear days as night, and that is pure image statistics -- unchanged
by the detector swap. What DID change is the price: the record justified the
authority bound as protecting a stream "scoring 0.3683 against one scoring 0.0177".
Those two numbers move. The bound is re-priced against the new ones.

Usage:
    python scripts/reprice_veto_axes.py --cache-dir runs/cache_m --out runs/eval/veto_axes_26m.md
    python scripts/reprice_veto_axes.py --cache-dir runs/cache --out runs/eval/veto_axes_26s.md
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

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts   # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402

SHIP = 0

#: (VIS condition, IR condition). Every axis needs cells where it fires: the veil
#: and photometric axes need fog/lowlight, the IR bounds need a corrupted IR.
CELLS = [("clean", None), ("fog", None), ("lowlight", None), ("glare", None),
         ("blur_s3", None), ("rain_s2", None),
         ("clean", "fog_s2"), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("lowlight", "glare_s2"), ("blur_s3", "glare_s2")]


def axis_flags(ctx, cond, cache_dir, ir_cond):
    """Per-frame boolean for each axis, and which stream firing would delete.

    Read from the same files the gate reads, not recomputed, so a divergence here
    would be a divergence in the gate.
    """
    n = len(ctx.gts)
    out = {}
    sc = ctx.struct_const["axes"]

    g = ctx.gini_by_cond.get(cond)
    if g is not None:
        out["veil (grad_gini)"] = (g < float(sc["grad_gini"]["threshold"]), "VIS")

    b = ctx.bright_by_cond.get(cond)
    if b is not None and ctx.c_vis.mu_b is not None:
        out["photometric (VIS p05 < mu_b)"] = (b < float(ctx.c_vis.mu_b), "VIS")

    tag = ir_cond or "clean"
    irb = ROOT / "runs/derived/brightness" / f"gauss_ir_paired_{tag}.json"
    ir_p05 = np.asarray([f["p05"] for f in
                         json.loads(irb.read_text(encoding="utf-8"))["frames"]], dtype=float)
    out["ir_p05 > thr (night, raw)"] = (ir_p05 > float(sc["ir_p05"]["threshold"]), "VIS")

    if ctx.ir_d2 is not None:
        # The merge decision fires at q_ir < 0.5, i.e. d2 > 2 x bound -- the
        # sigmoid midpoint, not the bound itself.
        out["ir_health merge (drop IR)"] = (ctx.ir_d2 > 2.0 * ctx.ir_bound, "IR")
        out["ir_health authority (mute IR)"] = (ctx.ir_d2 > ctx.ir_bound_switch, "vote")

    q = ctx.q_vis_by_cond.get(cond)
    if q is not None:
        out["vis_health (gates veto_ir)"] = (q < 0.5, "gate")

    for k, (v, who) in out.items():
        assert len(v) == n, f"{k}: {len(v)} flags for {n} frames"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--preset", default="crossmodal26m",
                    choices=["crossmodal", "crossmodal26m"])
    ap.add_argument("--out", default="runs/eval/veto_axes_26m.md")
    ap.add_argument("--skip-ablation", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    claims, ablate, streams = [], [], []
    for vis_cond, ir_cond in CELLS:
        cname = f"{vis_cond}/{ir_cond or 'clean'}"
        ctx = load_context(preset=args.preset, cache_dir=args.cache_dir,
                           conditions=(vis_cond,), ir_condition=ir_cond,
                           verbose=(cname == "clean/clean"))
        night = np.isin(ctx.runs, NIGHT_RUNS)
        res = run_systems(ctx, vis_cond)
        p_vis = frame_parts(ctx.vis_by_cond[vis_cond], ctx.gts)
        p_ir = frame_parts(res["ir_in_vis"], ctx.gts)

        def ap(parts, sel):
            if not len(sel):
                return float("nan")
            e = ap_from_parts([parts[i] for i in sel])["per_class"].get(SHIP)
            return float(e["ap50_95"]) if e else 0.0

        day = np.flatnonzero(~night)
        streams.append({"cell": cname, "VIS_day": ap(p_vis, day), "IR_day": ap(p_ir, day),
                        "VIS_night": ap(p_vis, np.flatnonzero(night)),
                        "IR_night": ap(p_ir, np.flatnonzero(night))})

        for name, (flag, who) in axis_flags(ctx, vis_cond, args.cache_dir, ir_cond).items():
            fired = np.flatnonzero(flag & ~night)
            v, i = ap(p_vis, fired), ap(p_ir, fired)
            if who == "VIS":
                kept, removed, margin = i, v, i - v
            elif who == "IR":
                kept, removed, margin = v, i, v - i
            else:                       # a gate, not a deletion -- report, do not judge
                kept = removed = margin = float("nan")
            claims.append({"cell": cname, "axis": name, "removes": who,
                           "rate_day": float(np.mean(flag[~night])),
                           "rate_night": float(np.mean(flag[night])),
                           "n_fired_day": int(len(fired)),
                           "VIS_fired": v, "IR_fired": i,
                           "kept": kept, "removed": removed, "margin": margin})
        print(f"[axis] {cname} claims done ({time.time() - t0:.0f}s)", flush=True)

        if args.skip_ablation:
            continue
        # ---- 2. the ablation, on the same cell ------------------------------
        arms = {
            "adopted (crossmodal26m)": ctx,
            # The pre-fix rule, in the same table, so the fog damage and whatever
            # the fix costs elsewhere are read off one set of numbers.
            "veil unconditional (crossmodal)": replace(ctx, veil_requires_night=False),
            "no veil axis": replace(ctx, gini_by_cond={}),
            "no photometric (night arm = IR alone)": replace(ctx, bright_by_cond={
                k: v for k, v in ctx.bright_by_cond.items() if k != vis_cond}),
            "no veto_ir (merge bound off)": replace(ctx, ir_bound=1e12),
            "authority = merge bound (loosen)": replace(ctx, ir_bound_switch=ctx.ir_bound),
            "authority p90-ish (tighten x0.5)": replace(
                ctx, ir_bound_switch=ctx.ir_bound_switch * 0.5),
            "no vis_health gate": replace(ctx, q_vis_by_cond={}),
        }
        # ORACLE VEIL: fire the axis only where IR really is the better stream on
        # this cell. Not deployable -- it reads the answer -- but it is the ceiling
        # a capability-conditioned rule could reach, and the gap between it and
        # `adopted` is what the `veil AND night` repair leaves on the table.
        if streams[-1]["IR_day"] > streams[-1]["VIS_day"]:
            arms["oracle veil (cell-level)"] = replace(ctx, veil_requires_night=False)
        else:
            arms["oracle veil (cell-level)"] = replace(ctx, gini_by_cond={})

        bar_day = max(streams[-1]["VIS_day"], streams[-1]["IR_day"])
        bar_night = max(streams[-1]["VIS_night"], streams[-1]["IR_night"])
        for aname, actx in arms.items():
            r = res if aname.startswith("adopted") else run_systems(actx, vis_cond)
            p = frame_parts(r["fused_gated"], actx.gts)
            gd, gn = ap(p, day), ap(p, np.flatnonzero(night))
            ablate.append({"cell": cname, "arm": aname, "day": gd, "night": gn,
                           "gap_day": gd - bar_day, "gap_night": gn - bar_night,
                           "veto_vis": float(np.mean(np.asarray(r["veto_vis"])[~night])),
                           "veto_ir": float(np.mean(np.asarray(r["veto_ir"])[~night]))})
        print(f"[axis] {cname} ablation done ({time.time() - t0:.0f}s)", flush=True)

    # ------------------------------------------------------------------ report
    axes = sorted({c["axis"] for c in claims})
    cells = [f"{v}/{i or 'clean'}" for v, i in CELLS]
    by = {(c["cell"], c["axis"]): c for c in claims}

    L = [f"# Re-pricing every veto axis — `{args.cache_dir}`, preset `{args.preset}`", "",
         "A veto axis measures the **image**; the decision attached to it is a claim "
         "about the **detector**. The thresholds are pixel statistics and are "
         "identical across the 26s→26m swap to 0.000e+00 (`cache_m_verify.md`). The "
         "claims are not, and only the fog one had ever been re-asked.", "",
         "## 0. What the two streams score, per cell", "",
         "| cell | VIS day | IR day | VIS night | IR night |", "|---|---:|---:|---:|---:|"]
    for s in streams:
        L.append(f"| {s['cell']} | {s['VIS_day']:.4f} | {s['IR_day']:.4f} | "
                 f"{s['VIS_night']:.4f} | {s['IR_night']:.4f} |")

    L += ["", "## 1. The claim test — on the frames each axis fires, is the deleted "
          "stream actually worse?", "",
          "Day frames only: VIS is exactly 0.0000 at night, so every VIS-removing "
          "axis is trivially right there. `margin = AP(kept) − AP(removed)` over "
          "**exactly the frames the axis fires on**. **Negative = the axis is "
          "deleting the better sensor.** A subset AP is a diagnostic, not the arm's "
          "cost — §2 measures the cost.", ""]
    for axis in axes:
        rows = [by[(c, axis)] for c in cells if (c, axis) in by]
        if not rows or all(r["n_fired_day"] == 0 for r in rows):
            L += [f"### `{axis}` — never fires on a day frame in this grid", ""]
            continue
        who = rows[0]["removes"]
        L += [f"### `{axis}` — deletes **{who}**", "",
              "| cell | fires (day) | n | VIS | IR | margin |",
              "|---|---:|---:|---:|---:|---:|"]
        for r in rows:
            if r["n_fired_day"] == 0:
                L.append(f"| {r['cell']} | 0% | 0 | — | — | — |")
                continue
            m = ("—" if r["margin"] != r["margin"] else f"**{r['margin']:+.4f}**")
            L.append(f"| {r['cell']} | {r['rate_day']:.0%} | {r['n_fired_day']} | "
                     f"{r['VIS_fired']:.4f} | {r['IR_fired']:.4f} | {m} |")
        bad = [r for r in rows if r["n_fired_day"] and r["margin"] == r["margin"]
               and r["margin"] < -1e-9]
        L += ["", ("**MISPRICED** on " + ", ".join(f"{r['cell']} ({r['margin']:+.4f})"
                                                   for r in bad)
                   if bad else "Claim holds on every cell where it fires."), ""]

    if ablate:
        L += ["", "## 2. Ablation — what each axis is actually worth", "",
              "`gap` is against `max(VIS, IR)` on the same streams. A negative gap "
              "means the system is below the better single sensor on that cell.", "",
              "| arm | " + " | ".join(cells) + " |", "|---|" + "---:|" * len(cells)]
        ab = {(a["cell"], a["arm"]): a for a in ablate}
        arm_names = list(dict.fromkeys(a["arm"] for a in ablate))
        for aname in arm_names:
            L.append(f"| {aname} | " + " | ".join(
                f"{ab[(c, aname)]['gap_day']:+.4f}" if (c, aname) in ab else "—"
                for c in cells) + " |")
        L += ["", "### Night gaps", "",
              "| arm | " + " | ".join(cells) + " |", "|---|" + "---:|" * len(cells)]
        for aname in arm_names:
            L.append(f"| {aname} | " + " | ".join(
                f"{ab[(c, aname)]['gap_night']:+.4f}" if (c, aname) in ab else "—"
                for c in cells) + " |")
        L += ["", "### Worst cell per arm (day, then night)", "",
              "| arm | worst day | worst night |", "|---|---:|---:|"]
        for aname in arm_names:
            rs = [a for a in ablate if a["arm"] == aname]
            L.append(f"| {aname} | **{min(r['gap_day'] for r in rs):+.4f}** | "
                     f"**{min(r['gap_night'] for r in rs):+.4f}** |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"streams": streams, "claims": claims, "ablation": ablate}, indent=2),
        encoding="utf-8")
    print(f"[axis] wrote {out} in {time.time() - t0:.0f}s")
    for axis in axes:
        rows = [by[(c, axis)] for c in cells if (c, axis) in by and by[(c, axis)]["n_fired_day"]]
        bad = [r for r in rows if r["margin"] == r["margin"] and r["margin"] < -1e-9]
        print(f"[axis] {axis:34} fires on {len(rows):2d} cells, MISPRICED on {len(bad)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
