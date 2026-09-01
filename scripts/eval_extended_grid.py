"""The extended grid: corrupted IR, and VIS conditions where IR should win.

The original eight cells corrupt VIS only, which leaves two questions unanswerable
on them:

  (a) **Does a degraded IR wrongly veto a healthy VIS, end to end?**
      `probe_ir_night_robustness.py` answered this at the SWITCH level from image
      statistics. It could not answer what a degraded IR *detector* then does to
      the fused result, because no corrupted IR cache existed.

  (b) **Is there any cell where VIS is unvetoed and IR is the better stream?**
      Without one the capability ratio is unpriceable: it only matters where VIS
      is unvetoed, and on every original such cell VIS wins, so a larger ratio is
      free. `docs/crossmodal-gate-2026-09-01.md` §3c refused to tune it for exactly
      this reason.

blur, noise and rain are also three corruption families the veil axis has never
seen -- it was fitted as a novelty bound on clean frames and validated on fog --
so (b) doubles as a generalisation test of a gate that never saw them.

Ship AP throughout. Every cell reports the system against `max(VIS, IR)` computed
on the SAME streams the system was given, including the IR NMS, so the bar is the
one the system is actually held to.

Usage:
    python scripts/eval_extended_grid.py --out runs/eval/extended_grid.md --n-boot 1000
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

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems              # noqa: E402

SHIP = 0

#: (VIS condition, IR condition, what the cell is for)
GRID = [
    ("clean", None, "reference"),
    # (a) healthy VIS, damaged IR — the safety question, end to end
    ("clean", "fog_s2", "IR damaged, VIS healthy"),
    ("clean", "glare_s2", "IR damaged, VIS healthy"),
    ("clean", "blur_s2", "IR damaged, VIS healthy"),
    ("clean", "noise_s2", "IR damaged, VIS healthy"),
    # (b) VIS degraded into IR's range, IR healthy — the ratio question
    ("blur_s3", None, "VIS degraded, IR healthy"),
    ("noise_s2", None, "VIS degraded, IR healthy"),
    ("rain_s2", None, "VIS degraded, IR healthy"),
    # (c) both degraded — the residual §3b could not close
    ("lowlight", "glare_s2", "both degraded"),
    ("blur_s3", "glare_s2", "both degraded"),
]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache",
                     help="prediction caches to read. runs/cache = yolo26s (phase2); "
                          "runs/cache_m = the full-scale yolo26m / yolo26m-p2feat "
                          "detectors the architecture actually specifies.")
    ap.add_argument("--out", default="runs/eval/extended_grid.md")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--preset", default="crossmodal",
                    choices=["crossmodal", "crossmodal26m"],
                    help="crossmodal26m = crossmodal plus the two repairs the "
                         "full-scale detectors force (conditional veil axis, "
                         "cross-modal support term).")
    args = ap.parse_args()

    t0 = time.time()
    rows, boots = [], []
    for vis_cond, ir_cond, purpose in GRID:
        ctx = load_context(cache_dir=args.cache_dir, preset=args.preset, conditions=(vis_cond,),
                           ir_condition=ir_cond, verbose=False)
        night = np.isin(ctx.runs, NIGHT_RUNS)
        res = run_systems(ctx, vis_cond)
        p_gate = frame_parts(res["fused_gated"], ctx.gts)
        p_vis = frame_parts(ctx.vis_by_cond[vis_cond], ctx.gts)
        p_ir = frame_parts(res["ir_in_vis"], ctx.gts)
        veto_v = np.asarray(res["veto_vis"])
        veto_i = np.asarray(res["veto_ir"])
        abst = np.asarray(res["abstain"])

        def ap(p, sel):
            e = ap_from_parts([p[i] for i in np.flatnonzero(sel)])["per_class"].get(SHIP)
            return float(e["ap50_95"]) if e else 0.0

        for lab, sel in (("day", ~night), ("night", night)):
            v, i, g = ap(p_vis, sel), ap(p_ir, sel), ap(p_gate, sel)
            bar = max(v, i)
            rows.append({
                "vis": vis_cond, "ir": ir_cond or "clean", "split": lab, "purpose": purpose,
                "VIS": v, "IR": i, "bar": bar, "gated": g, "gap": g - bar,
                "ir_better": bool(i > v),
                "veto_vis": float(veto_v[sel].mean()), "veto_ir": float(veto_i[sel].mean()),
                "abstain": float(abst[sel].mean()),
            })
            if args.n_boot and lab == "day":
                base = p_vis if v >= i else p_ir
                b = bootstrap_delta([p_gate[k] for k in np.flatnonzero(sel)],
                                    [base[k] for k in np.flatnonzero(sel)],
                                    None, n_boot=args.n_boot, cls=SHIP)
                boots.append({"cell": f"{vis_cond}/{ir_cond or 'clean'}", **b})
        print(f"[grid] {vis_cond} x IR {ir_cond or 'clean'} done "
              f"({time.time() - t0:.0f}s)", flush=True)

    L = ["# Extended grid — corrupted IR, and VIS conditions where IR should win", "",
         "Ship AP. `bar` = max(VIS, IR) on the SAME streams the system was given "
         "(including the IR NMS), so it is the bar the system is actually held to. "
         "`IR>VIS` marks the cells that did not exist before.", "",
         "The IR stream is degraded, but everything FITTED on IR — the capability "
         "prior, the health model, the night threshold — stays fitted on the clean "
         "stream. Calibration does not get to see the damage it must detect.", "",
         "| VIS | IR | split | purpose | VIS | IR | IR>VIS | bar | gated | gap | veto V/IR | abstain |",
         "|---|---|---|---|---:|---:|:--:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['vis']} | {r['ir']} | {r['split']} | {r['purpose']} | "
                 f"{r['VIS']:.4f} | {r['IR']:.4f} | {'**yes**' if r['ir_better'] else '—'} | "
                 f"{r['bar']:.4f} | {r['gated']:.4f} | **{r['gap']:+.4f}** | "
                 f"{r['veto_vis']:.0%}/{r['veto_ir']:.0%} | {r['abstain']:.0%} |")

    day = [r for r in rows if r["split"] == "day"]
    worst = min(day, key=lambda r: r["gap"])
    newcells = [r for r in day if r["ir_better"]]
    L += ["", "## Verdict", "",
          f"- Worst day cell: **{worst['vis']} x IR {worst['ir']}**, gap "
          f"{worst['gap']:+.4f}.",
          f"- Cells where IR is the better stream (new): "
          f"{', '.join(f'{r[chr(39)+chr(39)] if False else r['vis']}' for r in newcells) or 'none'}"
          f" — these are the ones that price the capability ratio.",
          f"- Cells below their bar: "
          f"{sum(1 for r in day if r['gap'] < -1e-9)} of {len(day)} day cells."]

    if boots:
        L += ["", f"## Gated vs the better single stream, day frames "
              f"(paired bootstrap n={args.n_boot})", "",
              "| cell | gated | better single | delta | 95% CI |", "|---|---:|---:|---:|---|"]
        for b in boots:
            L.append(f"| {b['cell']} | {b['a']:.4f} | {b['b']:.4f} | {b['delta']:+.4f} | "
                     f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"
                     f"{' (spans 0)' if b['spans_zero'] else ''} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"rows": rows, "bootstrap": boots},
                                                   indent=2), encoding="utf-8")
    print(f"[grid] wrote {out} in {time.time() - t0:.0f}s")
    for r in day:
        print(f"[grid] {r['vis']:10} x IR {r['ir']:9} gap {r['gap']:+.4f} "
              f"veto {r['veto_vis']:.0%}/{r['veto_ir']:.0%}"
              f"{'   <-- IR better' if r['ir_better'] else ''}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
