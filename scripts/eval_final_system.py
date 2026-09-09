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

from uqfusion.eval.identity import system_identity
from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems  # noqa: E402

VARIANTS = ("adopted", "no_maha", "cap_only", "no_veto")

#: Under `--preset crossmodal` the soft terms are already off, so `no_maha` and
#: `cap_only` would both be the adopted arm under another name. The informative
#: ablations there run the other way: put the Mahalanobis term BACK (`with_maha`),
#: and restore the photometric axis the preset dropped (`photometric_veto`), so
#: each removed component is charged for its own absence.
VARIANTS_CROSSMODAL = ("adopted", "with_maha", "photometric_veto", "no_veto")

# Class indices in the VIS label space, which is the frame everything is scored
# in: `runs/derived/data_vis_stride2.yaml` -> {0: ship, 1: buoy}. IR is nc=1 with
# ship at 0 (`data_ir_shiponly.yaml`), so the index means the same thing on both
# sides and the D28/A-1 "remap IR into the VIS label space" is the identity.
# Asserted against the yaml at startup rather than trusted — a silent reindex
# would swap which class this report calls the headline.
SHIP, BUOY = 0, 1


def _assert_class_indices() -> None:
    import yaml

    y = ROOT / "runs" / "derived" / "data_vis_stride2.yaml"
    if not y.is_file():  # not every host carries the derived yamls; skip quietly
        return
    names = (yaml.safe_load(y.read_text(encoding="utf-8")) or {}).get("names") or {}
    got = {int(k): str(v).lower() for k, v in names.items()}
    assert got.get(SHIP) == "ship" and got.get(BUOY) == "buoy", (
        f"class indices moved: {y.name} says {got}, this report assumes "
        f"{{{SHIP}: 'ship', {BUOY}: 'buoy'}}. Fix SHIP/BUOY before reading the table."
    )


def variant_ctx(ctx, name: str):
    """The soft-weight ablations, as context copies. The veto (and its
    hysteresis) is untouched in all but `no_veto`."""
    if name == "adopted":
        return ctx, {}
    if name == "with_maha":
        # Restore the D-6 ladder constants the crossmodal preset switches off, so
        # the term is charged for what it costs rather than credited for being
        # absent.
        import json as _json
        f = _json.loads((ROOT / "runs/eval/reliability_constants.json").read_text(
            encoding="utf-8"))
        return replace(ctx,
                       c_vis=replace(ctx.c_vis, mu_d=float(f["vis"]["mu_d"]),
                                     tau=float(f["vis"]["tau"]),
                                     lam=float(f["vis"]["lam"])),
                       c_ir=replace(ctx.c_ir, mu_d=float(f["ir"]["mu_d"]),
                                    tau=float(f["ir"]["tau"]),
                                    lam=float(f["ir"]["lam"]))), {}
    if name == "photometric_veto":
        return replace(ctx, veto_rule="photometric+veil"), {}
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
    ap.add_argument("--cache-dir", default="runs/cache",
                     help="prediction caches to read. runs/cache = yolo26s (phase2); "
                          "runs/cache_m = the full-scale yolo26m / yolo26m-p2feat "
                          "detectors the architecture actually specifies.")
    ap.add_argument("--out", default="runs/eval/final_system.md")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict the sweep (pre-flight uses --conditions clean)")
    ap.add_argument("--preset", default="adopted", choices=["adopted", "crossmodal"],
                    help="'adopted' reproduces the 2026-08-20/09-01 system; "
                         "'crossmodal' is the 2026-09-01 replacement (see ctx.load_context)")
    args = ap.parse_args()
    _assert_class_indices()

    t0 = time.time()
    ctx = load_context(cache_dir=args.cache_dir, preset=args.preset,
                       **({"conditions": tuple(args.conditions)} if args.conditions else {}))
    variants = VARIANTS_CROSSMODAL if args.preset == "crossmodal" else VARIANTS
    splits = {"day": ctx.sel("day"), "night": ctx.sel("night")}

    # ---- run every variant once per condition; keep frame parts ------------
    parts = {}          # (variant, cond) -> gated parts
    vis_parts, ir_parts = {}, {}
    veto_rates = {}
    for cond in ctx.conditions:
        for name in variants:
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

    def cell(p, sel, cls=None):
        r = ap_from_parts(p, sel)
        if cls is None:
            return r["map50_95"]
        e = r["per_class"].get(cls)
        return float(e["ap50_95"]) if e else 0.0

    # ---- table 1: the finalized system, with CIs vs ir_only ----------------
    # Reported per class, ship first. `map50_95` here is macro-averaged over ship
    # and buoy, and the two classes are not symmetric: IR is nc=1 ship-only
    # (D28/A-1), so buoys can only ever come from VIS. That distorts this exact
    # comparison in two directions at once — in day cells the gated row gains a
    # buoy contribution `ir_only` structurally cannot have, inflating the delta
    # with "VIS can see buoys" rather than "fusion helps"; in vetoed cells the
    # gated row loses buoys entirely and the macro is halved. Ship AP is the only
    # class both streams can produce, so it is the honest headline; the macro row
    # is kept for continuity with the record.
    rows_main = []
    for cond in ctx.conditions:
        for sname, sel in splits.items():
            b = bootstrap_delta(parts[("adopted", cond)], ir_parts[cond], sel,
                                n_boot=args.n_boot)
            bs = bootstrap_delta(parts[("adopted", cond)], ir_parts[cond], sel,
                                 n_boot=args.n_boot, cls=SHIP)
            rows_main.append({
                "condition": cond, "split": sname,
                "visible_only": cell(vis_parts[cond], sel),
                "ir_only": b["b"], "gated": b["a"],
                "delta_vs_ir": b["delta"], "ci": [b["ci_lo"], b["ci_hi"]],
                "spans_zero": b["spans_zero"], "flip": b["p_sign_flip"],
                "veto_rate": veto_rates[(cond, sname)],
                # ship (cls 0) — the fusable class, and the headline
                "ship_visible_only": cell(vis_parts[cond], sel, SHIP),
                "ship_ir_only": bs["b"], "ship_gated": bs["a"],
                "ship_delta_vs_ir": bs["delta"], "ship_ci": [bs["ci_lo"], bs["ci_hi"]],
                "ship_spans_zero": bs["spans_zero"], "ship_flip": bs["p_sign_flip"],
                # buoy (cls 1) — VIS-only by construction; shown to explain the gap
                "buoy_visible_only": cell(vis_parts[cond], sel, BUOY),
                "buoy_gated": cell(parts[("adopted", cond)], sel, BUOY),
                "buoy_ir_only": cell(ir_parts[cond], sel, BUOY),
            })
    print(f"[final] gated-vs-ir bootstrap done ({time.time() - t0:.0f}s)", flush=True)

    # ---- table 2: gated vs visible_only on the day cells (open item 1) -----
    rows_vis = []
    for cond in ctx.conditions:
        b = bootstrap_delta(parts[("adopted", cond)], vis_parts[cond], splits["day"],
                            n_boot=args.n_boot)
        bs = bootstrap_delta(parts[("adopted", cond)], vis_parts[cond], splits["day"],
                             n_boot=args.n_boot, cls=SHIP)
        rows_vis.append({"condition": cond, "gated": b["a"], "visible_only": b["b"],
                         "delta": b["delta"], "ci": [b["ci_lo"], b["ci_hi"]],
                         "spans_zero": b["spans_zero"], "flip": b["p_sign_flip"],
                         "ship_gated": bs["a"], "ship_visible_only": bs["b"],
                         "ship_delta": bs["delta"], "ship_ci": [bs["ci_lo"], bs["ci_hi"]],
                         "ship_spans_zero": bs["spans_zero"], "ship_flip": bs["p_sign_flip"]})
    print(f"[final] gated-vs-vis bootstrap done ({time.time() - t0:.0f}s)", flush=True)

    # ---- table 3: soft-weight ablation, deltas vs adopted ------------------
    rows_abl = []
    for name in variants[1:]:
        for cond in ctx.conditions:
            for sname, sel in splits.items():
                a = cell(parts[(name, cond)], sel)
                ref = cell(parts[("adopted", cond)], sel)
                a_s = cell(parts[(name, cond)], sel, SHIP)
                ref_s = cell(parts[("adopted", cond)], sel, SHIP)
                row = {"variant": name, "condition": cond, "split": sname,
                       "map": a, "delta": a - ref,
                       "ship_map": a_s, "ship_delta": a_s - ref_s}
                if abs(a - ref) > 1e-12:
                    b = bootstrap_delta(parts[(name, cond)], parts[("adopted", cond)],
                                        sel, n_boot=args.n_boot)
                    row.update({"ci": [b["ci_lo"], b["ci_hi"]],
                                "spans_zero": b["spans_zero"]})
                if abs(a_s - ref_s) > 1e-12:
                    bs = bootstrap_delta(parts[(name, cond)], parts[("adopted", cond)],
                                         sel, n_boot=args.n_boot, cls=SHIP)
                    row.update({"ship_ci": [bs["ci_lo"], bs["ci_hi"]],
                                "ship_spans_zero": bs["spans_zero"]})
                rows_abl.append(row)
        print(f"[final] ablation {name} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- report ------------------------------------------------------------
    L = ["# The finalized system (2026-08-20), measured",
         "",
         f"Configuration: capability prior over fit runs (VIS {ctx.cap_vis:.4f} / "
         f"IR {ctx.cap_ir:.4f}), photometric term veto-only, hard veto at "
         f"r_bright<{ctx.veto} with {ctx.veto_filter} hysteresis, WBF iou_thr "
         f"{ctx.iou_thr}. PRESET: {args.preset} (veto rule {ctx.veto_rule}, "
         f"single_passthrough={ctx.single_passthrough}). "
         f"Paired frame-level bootstrap, n={args.n_boot}, seed 0. "
         f"Caches: the record's yolo26s checkpoints (the full-scale retrain "
         f"replaces them; this table freezes the architecture, not the numbers).",
         "",
         "## 1. Eight cells, gated vs `ir_only` — SHIP AP (the headline)",
         "",
         "Ship is the only class both streams can produce: IR is nc=1 ship-only "
         "(D28/A-1 — IR buoy AP measured 0.00019). Read this table, not the macro "
         "one below. The macro delta is `(delta_ship + delta_buoy) / 2`, mixing "
         "the class fusion acts on with one only VIS can supply, so it answers no "
         "single question: it dilutes a large ship gain (where delta_buoy < "
         "delta_ship), inflates a small one (where VIS buoy AP is high and the "
         "ship gain is not), and in a vetoed cell carries a buoy zero the system "
         "was never able to avoid. None of those apply here.",
         "",
         "| cell | VIS ship | IR ship | gated ship | delta vs ir | 95% CI | flips | VIS veto |",
         "|---|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows_main:
        L.append(f"| {r['condition']}/{r['split']} | {r['ship_visible_only']:.4f} | "
                 f"{r['ship_ir_only']:.4f} | {r['ship_gated']:.4f} | "
                 f"{r['ship_delta_vs_ir']:+.4f} | "
                 f"[{r['ship_ci'][0]:+.4f}, {r['ship_ci'][1]:+.4f}]"
                 f"{' (spans 0)' if r['ship_spans_zero'] else ''} | "
                 f"{r['ship_flip']:.1%} | {r['veto_rate']:.0%} |")

    L += ["", "### 1b. The same cells, macro mAP over both classes (continuity with the record)",
          "",
          "`buoy gated` is VIS-only by construction; where it drops to ~0 the veto "
          "has removed VIS from the merge, and the macro column below is carrying "
          "that zero. This table exists so earlier macro-quoted numbers stay "
          "comparable — it is not the result.",
          "",
          "| cell | visible_only | ir_only | gated | delta vs ir | 95% CI | buoy gated | buoy VIS |",
          "|---|---:|---:|---:|---:|---|---:|---:|"]
    for r in rows_main:
        L.append(f"| {r['condition']}/{r['split']} | {r['visible_only']:.4f} | "
                 f"{r['ir_only']:.4f} | {r['gated']:.4f} | {r['delta_vs_ir']:+.4f} | "
                 f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
                 f"{' (spans 0)' if r['spans_zero'] else ''} | "
                 f"{r['buoy_gated']:.4f} | {r['buoy_visible_only']:.4f} |")

    L += ["", "## 2. Gated vs `visible_only`, day frames (the record's open item 1)", "",
          "Ship AP is the headline column here too; macro follows for continuity.", "",
          "| condition | VIS ship | gated ship | delta | 95% CI | flips |",
          "|---|---:|---:|---:|---|---:|"]
    for r in rows_vis:
        L.append(f"| {r['condition']}/day | {r['ship_visible_only']:.4f} | "
                 f"{r['ship_gated']:.4f} | {r['ship_delta']:+.4f} | "
                 f"[{r['ship_ci'][0]:+.4f}, {r['ship_ci'][1]:+.4f}]"
                 f"{' (spans 0)' if r['ship_spans_zero'] else ''} | {r['ship_flip']:.1%} |")
    L += ["", "| condition | visible_only | gated | delta (macro) | 95% CI | flips |",
          "|---|---:|---:|---:|---|---:|"]
    for r in rows_vis:
        L.append(f"| {r['condition']}/day | {r['visible_only']:.4f} | {r['gated']:.4f} | "
                 f"{r['delta']:+.4f} | [{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
                 f"{' (spans 0)' if r['spans_zero'] else ''} | {r['flip']:.1%} |")

    new_abl = ("`with_maha`: the D-6 Mahalanobis ladder constants put BACK, so the "
               "term is charged for what it costs. `photometric_veto`: the veto rule "
               "swapped back to photometric OR veil, isolating the veto change from "
               "the weight change. `no_veto`: nothing vetoed at all.")
    L += ["", "## 3. Soft-weight ablation (veto kept; deltas vs adopted)", "",
          (new_abl if args.preset == "crossmodal" else
           "`no_maha`: r_frame forced to 1. `cap_only`: additionally r_box forced "
           "to 1, so weights are the pure capability prior. `no_veto`: the soft "
           "system alone."), "",
          "| variant | cell | ship AP | ship delta | ship 95% CI | mAP | delta (macro) | 95% CI |",
          "|---|---|---:|---:|---|---:|---:|---|"]
    for r in rows_abl:
        ci = (f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]"
              + (" (spans 0)" if r.get("spans_zero") else "")) if "ci" in r else "identical"
        sci = (f"[{r['ship_ci'][0]:+.4f}, {r['ship_ci'][1]:+.4f}]"
               + (" (spans 0)" if r.get("ship_spans_zero") else "")) if "ship_ci" in r else "identical"
        L.append(f"| {r['variant']} | {r['condition']}/{r['split']} | "
                 f"{r['ship_map']:.4f} | {r['ship_delta']:+.4f} | {sci} | "
                 f"{r['map']:.4f} | {r['delta']:+.4f} | {ci} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({
        "config": {"preset": args.preset, "veto_rule": ctx.veto_rule,
                   "single_passthrough": ctx.single_passthrough,
                   "cap_vis": ctx.cap_vis, "cap_ir": ctx.cap_ir, "iou_thr": ctx.iou_thr,
                   "veto": ctx.veto, "veto_filter": list(ctx.veto_filter or ()),
                   "bright_soft": ctx.c_vis.bright_soft, "n_boot": args.n_boot,
                   # R-E1: the VALUES, not just the preset NAME. `preset="crossmodal"`
                   # meant three different systems on 2026-09-01 and these two were the
                   # difference; neither was recorded, so `cap_ir` was their only
                   # witness. See docs/exposure-ledger-2026-09-09.md section 6.
                   "ir_nms": ctx.ir_nms, "cap_ir_scale": ctx.cap_ir_scale,
                   # R-A1: which AP convention produced every number in this file
                   # is carried by the `identity` block below, with the source rev.
                   },
        # R-E1/F14: source revision (HEAD *and* a dirty hash), declared AP policies,
        # and every FusionContext value that changes the numbers. A future reader can
        # identify this run without inferring it from whichever code was checked out.
        "identity": system_identity(ctx, n_boot=args.n_boot),
        "main": rows_main, "vs_visible": rows_vis, "ablation": rows_abl},
        indent=2), encoding="utf-8")
    print(f"[final] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
