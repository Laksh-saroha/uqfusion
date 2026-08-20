"""Is the capability prior fitted on the report split? Refit it run-disjoint and re-measure.

§7 of the experiment record holds pohang01 out of every fit: the photometric
constants, the statistic choice, the threshold rule and the veto rule are all
decided on pohang00/02/03 or on pure image arithmetic. The capability prior is
not on that list, and in the code it is

    cap_vis = map50_95(vis_by_cond["clean"], gts)      # ALL 2,232 frames
    cap_ir  = map50_95(ir_mapped,            gts)      # 46% of them pohang01

i.e. each modality's mAP over the same frames the system is then reported on,
held-out run included. `run_fusion_eval.py`'s own docstring says pooling those
runs "into one 0.258 hides both the real daylight capability and the real night
result" — and that pooled number is the prior.

The prior is not cosmetic: §4.1 uses it to explain why down-weighting cannot
work ("the capability prior hands VIS a 12.5x advantage, which leaves w_vis at
0.432 on the night run"). If the ratio moves, that argument's arithmetic moves.

Expected scope, stated before running: on **clean/night** this should change
nothing, because VIS is vetoed there and the weights never enter (§6.1 makes the
same point about the 3.3x prior bug). Everywhere else it can move. A result that
matches that prediction is a consistency check; one that does not means something
else is going on.

Usage:
    python scripts/eval_capability_refit.py [--out runs/eval/x_capability_refit.md]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="runs/eval/x_capability_refit.md")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict the condition sweep (pre-flight uses --conditions clean)")
    args = ap.parse_args()

    t0 = time.time()
    kw = {"conditions": tuple(args.conditions)} if args.conditions else {}
    ctx_all = load_context(capability_sel="all", **kw)
    ctx_fit = load_context(capability_sel="fit", verbose=False, **kw)
    print(f"[cap] adopted (all frames): VIS {ctx_all.cap_vis:.4f} IR {ctx_all.cap_ir:.4f} "
          f"ratio {ctx_all.cap_vis / ctx_all.cap_ir:.2f}x")
    print(f"[cap] run-disjoint (fit runs): VIS {ctx_fit.cap_vis:.4f} IR {ctx_fit.cap_ir:.4f} "
          f"ratio {ctx_fit.cap_vis / ctx_fit.cap_ir:.2f}x")

    splits = {"day": ctx_all.sel("day"), "night": ctx_all.sel("night")}
    rows, boots, wrows = [], {}, []
    for cond in ctx_all.conditions:
        ra = run_systems(ctx_all, cond)
        rf = run_systems(ctx_fit, cond)
        pa = frame_parts(ra["fused_gated"], ctx_all.gts)
        pf = frame_parts(rf["fused_gated"], ctx_all.gts)
        for sname, sel in splits.items():
            rows.append({"condition": cond, "split": sname,
                         "adopted": ap_from_parts(pa, sel)["map50_95"],
                         "refit": ap_from_parts(pf, sel)["map50_95"]})
            wrows.append({"condition": cond, "split": sname,
                          "w_vis_adopted": float(np.mean(np.asarray(ra["w_vis_gated"])[sel])),
                          "w_vis_refit": float(np.mean(np.asarray(rf["w_vis_gated"])[sel]))})
            boots[(cond, sname)] = bootstrap_delta(pf, pa, sel, n_boot=args.n_boot)
        print(f"[cap] {cond:9s} " + "  ".join(
            f"{r['split']}: {r['adopted']:.4f} -> {r['refit']:.4f}"
            for r in rows if r["condition"] == cond), flush=True)

    L = ["# Capability prior — refitted run-disjoint", "",
         "| prior | VIS | IR | ratio | frames |",
         "|---|---:|---:|---:|---:|",
         f"| adopted (all clean frames, pohang01 included) | {ctx_all.cap_vis:.4f} | "
         f"{ctx_all.cap_ir:.4f} | {ctx_all.cap_vis / ctx_all.cap_ir:.2f}x | {ctx_all.n()} |",
         f"| run-disjoint (pohang00/02/03 only) | {ctx_fit.cap_vis:.4f} | {ctx_fit.cap_ir:.4f} | "
         f"{ctx_fit.cap_vis / ctx_fit.cap_ir:.2f}x | {len(ctx_all.sel('fit'))} |",
         "",
         "## 1. Effect on the table", "",
         "| condition | split | adopted | refit | delta | 95% CI | sign flips |",
         "|---|---|---:|---:|---:|---|---:|"]
    for r in rows:
        b = boots[(r["condition"], r["split"])]
        L.append(f"| {r['condition']} | {r['split']} | {r['adopted']:.4f} | {r['refit']:.4f} | "
                 f"{b['delta']:+.4f} | [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {b['p_sign_flip']:.1%} |")

    L += ["", "## 2. Effect on the weight the veto argument quotes", "",
          "§4.1 rests on `w_vis` = 0.432 on the night run under the adopted prior.",
          "",
          "| condition | split | mean w_vis (adopted) | mean w_vis (refit) |",
          "|---|---|---:|---:|"]
    for r in wrows:
        L.append(f"| {r['condition']} | {r['split']} | {r['w_vis_adopted']:.3f} | {r['w_vis_refit']:.3f} |")

    night = next(r for r in rows if r["condition"] == "clean" and r["split"] == "night")
    L += ["", "## 3. Reading", "",
          f"clean/night moved {night['refit'] - night['adopted']:+.4f} "
          f"({night['adopted']:.4f} -> {night['refit']:.4f}). The prediction registered in this "
          f"script's docstring was that it would not move at all, because VIS is vetoed on that "
          f"run and the weights never enter. Cells other than clean/night are where the prior "
          f"actually does work, and they are the ones to read for whether the protocol gap "
          f"matters to any published number."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({
        "cap_all": [ctx_all.cap_vis, ctx_all.cap_ir],
        "cap_fit": [ctx_fit.cap_vis, ctx_fit.cap_ir],
        "rows": rows, "weights": wrows,
        "bootstrap": {"|".join(k): v for k, v in boots.items()}}, indent=2), encoding="utf-8")
    print(f"[cap] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
