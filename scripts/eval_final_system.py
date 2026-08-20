"""The finalized system (2026-08-20), measured — plus the two checks that gate it.

`docs/architecture-final-2026-08-20.md` freezes the fusion architecture as:
capability-weighted WBF (run-disjoint prior), hard photometric veto with
dilate-15 hysteresis, photometric term veto-only, sigma-weighted WBF available.
This script produces that system's 8-cell table with paired bootstrap CIs and
answers the two questions the follow-up round left open on the CPU side:

1. **Gated vs `visible_only` on the day cells** (followup §1, open item 1).
   The record's clean/day and glare/day gated-vs-VIS comparisons were point
   estimates; if VIS alone wins them, no cell anywhere shows fusion beating the
   better single stream and the paper's fusion table must say so.

2. **Does the uncertainty soft weight contribute anything?** (new). The veto is
   photometric and needs no UQ; the soft weight is Mahalanobis x r_box x
   capability. Two ablations that keep the veto and delete soft terms:
       no_maha   — mu_d -> 1e9, so r_frame -> 1 (r_box and capability remain)
       cap_only  — additionally lam -> 0, so R -> 1 and the weights are the
                   pure capability prior
   If neither moves any cell, the fusion layer's uncertainty machinery is
   decorative and the honest system description is "capability-weighted merge
   plus a photometric veto" — which decides what the paper may claim.

A `no_veto` arm is included as the reference showing the veto IS load-bearing.

CPU-only, cached predictions, ~30-60 min. Usage:
    python scripts/eval_final_system.py [--n-boot 1000]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems  # noqa: E402

VARIANTS = ("adopted", "no_maha", "cap_only", "no_veto")


def variant_ctx(ctx, name: str):
    """The soft-weight ablations, as context copies. The veto (and its
    hysteresis) is untouched in all but `no_veto`."""
    if name == "adopted":
        return ctx, {}
    if name == "no_maha":
        return replace(ctx, c_vis=replace(ctx.c_vis, mu_d=1e9),
                       c_ir=replace(ctx.c_ir, mu_d=1e9)), {}
    if name == "cap_only":
        return replace(ctx, c_vis=replace(ctx.c_vis, mu_d=1e9, lam=0.0),
                       c_ir=replace(ctx.c_ir, mu_d=1e9, lam=0.0)), {}
    if name == "no_veto":
        return ctx, {"veto_below": None}
    raise ValueError(name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="runs/eval/final_system.md")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict the sweep (pre-flight uses --conditions clean)")
    args = ap.parse_args()

    t0 = time.time()
    ctx = load_context(**({"conditions": tuple(args.conditions)} if args.conditions else {}))
    splits = {"day": ctx.sel("day"), "night": ctx.sel("night")}

    # ---- run every variant once per condition; keep frame parts ------------
    parts = {}          # (variant, cond) -> gated parts
    vis_parts, ir_parts = {}, {}
    veto_rates = {}
    for cond in ctx.conditions:
        for name in VARIANTS:
            vctx, kw = variant_ctx(ctx, name)
            res = run_systems(vctx, cond, **kw)
            parts[(name, cond)] = frame_parts(res["fused_gated"], ctx.gts)
            if name == "adopted":
                vis_parts[cond] = frame_parts(ctx.vis_by_cond[cond], ctx.gts)
                ir_parts[cond] = frame_parts(res["ir_in_vis"], ctx.gts)
                for sname, sel in splits.items():
                    veto_rates[(cond, sname)] = float(
                        np.mean(np.asarray(res["veto_vis"])[sel]))
        print(f"[final] {cond}: variants done ({time.time() - t0:.0f}s)", flush=True)

    def cell(p, sel):
        return ap_from_parts(p, sel)["map50_95"]

    # ---- table 1: the finalized system, with CIs vs ir_only ----------------
    rows_main = []
    for cond in ctx.conditions:
        for sname, sel in splits.items():
            b = bootstrap_delta(parts[("adopted", cond)], ir_parts[cond], sel,
                                n_boot=args.n_boot)
            rows_main.append({
                "condition": cond, "split": sname,
                "visible_only": cell(vis_parts[cond], sel),
                "ir_only": b["b"], "gated": b["a"],
                "delta_vs_ir": b["delta"], "ci": [b["ci_lo"], b["ci_hi"]],
                "spans_zero": b["spans_zero"], "flip": b["p_sign_flip"],
                "veto_rate": veto_rates[(cond, sname)],
            })
    print(f"[final] gated-vs-ir bootstrap done ({time.time() - t0:.0f}s)", flush=True)

    # ---- table 2: gated vs visible_only on the day cells (open item 1) -----
    rows_vis = []
    for cond in ctx.conditions:
        b = bootstrap_delta(parts[("adopted", cond)], vis_parts[cond], splits["day"],
                            n_boot=args.n_boot)
        rows_vis.append({"condition": cond, "gated": b["a"], "visible_only": b["b"],
                         "delta": b["delta"], "ci": [b["ci_lo"], b["ci_hi"]],
                         "spans_zero": b["spans_zero"], "flip": b["p_sign_flip"]})
    print(f"[final] gated-vs-vis bootstrap done ({time.time() - t0:.0f}s)", flush=True)

    # ---- table 3: soft-weight ablation, deltas vs adopted ------------------
    rows_abl = []
    for name in ("no_maha", "cap_only", "no_veto"):
        for cond in ctx.conditions:
            for sname, sel in splits.items():
                a = cell(parts[(name, cond)], sel)
                ref = cell(parts[("adopted", cond)], sel)
                row = {"variant": name, "condition": cond, "split": sname,
                       "map": a, "delta": a - ref}
                if abs(a - ref) > 1e-12:
                    b = bootstrap_delta(parts[(name, cond)], parts[("adopted", cond)],
                                        sel, n_boot=args.n_boot)
                    row.update({"ci": [b["ci_lo"], b["ci_hi"]],
                                "spans_zero": b["spans_zero"]})
                rows_abl.append(row)
        print(f"[final] ablation {name} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- report ------------------------------------------------------------
    L = ["# The finalized system (2026-08-20), measured",
         "",
         f"Configuration: capability prior over fit runs (VIS {ctx.cap_vis:.4f} / "
         f"IR {ctx.cap_ir:.4f}), photometric term veto-only, hard veto at "
         f"r_bright<{ctx.veto} with {ctx.veto_filter} hysteresis, WBF iou_thr "
         f"{ctx.iou_thr}. Paired frame-level bootstrap, n={args.n_boot}, seed 0. "
         f"Caches: the record's yolo26s checkpoints (the full-scale retrain "
         f"replaces them; this table freezes the architecture, not the numbers).",
         "",
         "## 1. Eight cells, gated vs `ir_only` (and `visible_only` for reference)",
         "",
         "| cell | visible_only | ir_only | gated | delta vs ir | 95% CI | flips | VIS veto |",
         "|---|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows_main:
        L.append(f"| {r['condition']}/{r['split']} | {r['visible_only']:.4f} | "
                 f"{r['ir_only']:.4f} | {r['gated']:.4f} | {r['delta_vs_ir']:+.4f} | "
                 f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
                 f"{' (spans 0)' if r['spans_zero'] else ''} | "
                 f"{r['flip']:.1%} | {r['veto_rate']:.0%} |")

    L += ["", "## 2. Gated vs `visible_only`, day frames (the record's open item 1)", "",
          "| condition | visible_only | gated | delta | 95% CI | flips |",
          "|---|---:|---:|---:|---|---:|"]
    for r in rows_vis:
        L.append(f"| {r['condition']}/day | {r['visible_only']:.4f} | {r['gated']:.4f} | "
                 f"{r['delta']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
                 f"{' (spans 0)' if r['spans_zero'] else ''} | {r['flip']:.1%} |")

    L += ["", "## 3. Soft-weight ablation (veto kept; deltas vs adopted)", "",
          "`no_maha`: r_frame forced to 1. `cap_only`: additionally r_box forced "
          "to 1, so weights are the pure capability prior. `no_veto`: the soft "
          "system alone.", "",
          "| variant | cell | mAP | delta vs adopted | 95% CI |",
          "|---|---|---:|---:|---|"]
    for r in rows_abl:
        ci = (f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
              + (" (spans 0)" if r.get("spans_zero") else "")) if "ci" in r else "identical"
        L.append(f"| {r['variant']} | {r['condition']}/{r['split']} | {r['map']:.4f} | "
                 f"{r['delta']:+.4f} | {ci} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({
        "config": {"cap_vis": ctx.cap_vis, "cap_ir": ctx.cap_ir, "iou_thr": ctx.iou_thr,
                   "veto": ctx.veto, "veto_filter": list(ctx.veto_filter or ()),
                   "bright_soft": ctx.c_vis.bright_soft, "n_boot": args.n_boot},
        "main": rows_main, "vs_visible": rows_vis, "ablation": rows_abl},
        indent=2), encoding="utf-8")
    print(f"[final] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
