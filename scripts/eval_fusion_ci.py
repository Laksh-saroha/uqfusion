"""Bootstrap CIs, per-class AP, and the real/synthetic split of the 8-cell table.

Three questions that share one expensive computation (the ablation chain), so they
share one script.

**T0 — confidence intervals.** §9.4 item 18 of the experiment record: the adopted
configuration is defended by deltas of 0.0002 to 0.0026 and none of them carry an
error bar. The paired frame-level bootstrap in `apmetrics` resamples frames and
scores both systems on the same draw, so frame-composition noise cancels in the
difference. Reported for the headline claim (gated vs `ir_only`) and for each
increment of the chain.

**T2 — per class.** mAP is macro-averaged over ship and buoy, so a class scoring
near zero still contributes half the headline. Every adopted delta should be read
per class before it is believed. (Note the asymmetry with the IR screen: that is
IR boxes against IR GT, this is fused boxes against VIS GT — a different task,
and the buoy class does not behave the same way in the two.)

**T6 — real vs synthetic.** §2.4 is the record's strongest finding: Mahalanobis
distance catches synthetic corruptions and is blind to real degradation, a 21x
different response at matched darkness. If that is true then the fog/lowlight/
glare cells are evidence about a transform the gate can see, not about real-world
failure — and six of the eight summed cells are synthetic. This reports the sum
decomposed, so the real-evidence subtotal is visible instead of buried.

**Bonus — §9.4 item 19.** The chain is reported as no-gate -> +gate -> +gate+veto,
so the photometric term and the veto have never been ablated independently. All
four corners are run here, which makes the interaction term readable.

Usage:
    python scripts/eval_fusion_ci.py [--n-boot 1000] [--out runs/eval/x_fusion_ci.md]
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

CLASS_NAMES = {0: "ship", 1: "buoy"}

# The four corners of (photometric term, veto). "gate+veto" is the adopted system.
ARMS = {
    "nogate":    dict(brightness=False, veto=None),
    "gate":      dict(brightness=True,  veto=None),
    "veto_only": dict(brightness=True,  veto=0.5, soft_off=True),
    "gate+veto": dict(brightness=True,  veto=0.5),
}
REAL_CONDITIONS = ("clean",)          # the only cells that are not a synthetic transform


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval/x_fusion_ci.md")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict the condition sweep (pre-flight uses --conditions clean)")
    args = ap.parse_args()

    t0 = time.time()
    ctx = load_context(**({'conditions': tuple(args.conditions)} if args.conditions else {}))
    splits = {"day": ctx.sel("day"), "night": ctx.sel("night")}

    # --- run every arm on every condition, keep the per-frame match parts -----
    parts: dict = {}      # (arm, cond, system) -> per-frame parts
    scalar: dict = {}     # (arm, cond, system, split) -> mAP
    perclass: dict = {}
    for arm, spec in ARMS.items():
        for cond in ctx.conditions:
            ov = {}
            if not spec["brightness"]:
                ov["brightness_vis"] = None
            ov["veto_below"] = spec["veto"]
            if spec.get("soft_off"):
                # veto ON, photometric term OFF inside the soft weight. The switch
                # needs r_bright and the weight must not have it, so the decision is
                # taken from a probe run under the real rule and handed over
                # explicitly, while the scored run sees brightness_vis=None. That
                # isolates the veto from the soft term, which the published chain
                # (no-gate -> gate -> gate+veto) never does.
                res_probe = run_systems(ctx, cond, veto_below=0.5)
                ov = {"brightness_vis": None, "veto_below": None,
                      "veto_override": (res_probe["veto_vis"], res_probe["veto_ir"])}
            res = run_systems(ctx, cond, **ov)
            for system in ("gated_fusion", "ir_only", "visible_only"):
                recs = {"gated_fusion": res["fused_gated"], "ir_only": res["ir_in_vis"],
                        "visible_only": ctx.vis_by_cond[cond]}[system]
                p = frame_parts(recs, ctx.gts)
                parts[(arm, cond, system)] = p
                for sname, sel in splits.items():
                    r = ap_from_parts(p, sel)
                    scalar[(arm, cond, system, sname)] = r["map50_95"]
                    perclass[(arm, cond, system, sname)] = r["per_class"]
            print(f"[ci] {arm:10s} {cond:9s} "
                  + "  ".join(f"{s}/{n}={scalar[(arm, cond, s, n)]:.4f}"
                              for s in ("gated_fusion", "ir_only") for n in splits),
                  flush=True)
    print(f"[ci] arms done in {time.time() - t0:.0f}s", flush=True)

    # --- bootstrap the claims that are actually load-bearing ------------------
    t1 = time.time()
    boots = {}
    for cond in ctx.conditions:
        for sname, sel in splits.items():
            boots[("headline", cond, sname)] = bootstrap_delta(
                parts[("gate+veto", cond, "gated_fusion")], parts[("gate+veto", cond, "ir_only")],
                sel, n_boot=args.n_boot, seed=args.seed)
            boots[("veto_incr", cond, sname)] = bootstrap_delta(
                parts[("gate+veto", cond, "gated_fusion")], parts[("gate", cond, "gated_fusion")],
                sel, n_boot=args.n_boot, seed=args.seed)
            boots[("gate_incr", cond, sname)] = bootstrap_delta(
                parts[("gate", cond, "gated_fusion")], parts[("nogate", cond, "gated_fusion")],
                sel, n_boot=args.n_boot, seed=args.seed)
        print(f"[ci] bootstrapped {cond} ({time.time() - t1:.0f}s elapsed)", flush=True)
    # per-class bootstrap on the headline night claim, where the paper's claim lives
    for c in CLASS_NAMES:
        boots[("headline_cls", "clean", f"night/{CLASS_NAMES[c]}")] = bootstrap_delta(
            parts[("gate+veto", "clean", "gated_fusion")], parts[("gate+veto", "clean", "ir_only")],
            splits["night"], n_boot=args.n_boot, seed=args.seed, cls=c)
    print(f"[ci] bootstrap done in {time.time() - t1:.0f}s", flush=True)

    # ---------------------------------------------------------------- report --
    L = [f"# Bootstrap CIs, per-class AP, and the real/synthetic split",
         "",
         f"Paired frame-level bootstrap, {args.n_boot} resamples, seed {args.seed}. "
         f"{ctx.n()} paired frames ({len(splits['day'])} day / {len(splits['night'])} night). "
         f"Adopted config: iou_thr {ctx.iou_thr}, veto {ctx.veto}, "
         f"capability VIS {ctx.cap_vis:.4f} / IR {ctx.cap_ir:.4f}.",
         "",
         "## 1. The headline claim, with an interval",
         "",
         "Gated fusion (photometric gate + hard veto) minus `ir_only`, per cell.",
         "",
         "| condition | split | gated | ir_only | delta | 95% CI | sign flips | verdict |",
         "|---|---|---:|---:|---:|---|---:|---|"]
    for cond in ctx.conditions:
        for sname in splits:
            b = boots[("headline", cond, sname)]
            verdict = "spans zero" if b["spans_zero"] else ("gated wins" if b["delta"] > 0 else "ir_only wins")
            L.append(f"| {cond} | {sname} | {b['a']:.4f} | {b['b']:.4f} | {b['delta']:+.4f} | "
                     f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {b['p_sign_flip']:.1%} | {verdict} |")

    L += ["", "## 2. Each increment of the chain, with an interval", "",
          "`gate_incr` = photometric gate minus no gate. `veto_incr` = adding the veto on top.",
          "",
          "| increment | condition | split | delta | 95% CI | sign flips | verdict |",
          "|---|---|---|---:|---|---:|---|"]
    for key, label in (("gate_incr", "photometric gate"), ("veto_incr", "hard veto")):
        for cond in ctx.conditions:
            for sname in splits:
                b = boots[(key, cond, sname)]
                verdict = "spans zero" if b["spans_zero"] else ("gain" if b["delta"] > 0 else "loss")
                L.append(f"| {label} | {cond} | {sname} | {b['delta']:+.4f} | "
                         f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {b['p_sign_flip']:.1%} | {verdict} |")

    L += ["", "## 3. Independent ablation of the two terms (§9.4 item 19)", "",
          "The record reports these as a chain, so the interaction has never been read off.",
          "`veto_only` keeps the switch and removes the photometric term from the soft weight.",
          "",
          "| condition | split | nogate | gate | veto_only | gate+veto | interaction |",
          "|---|---|---:|---:|---:|---:|---:|"]
    for cond in ctx.conditions:
        for sname in splits:
            g = lambda a: scalar[(a, cond, "gated_fusion", sname)]  # noqa: E731
            inter = (g("gate+veto") - g("gate")) - (g("veto_only") - g("nogate"))
            L.append(f"| {cond} | {sname} | {g('nogate'):.4f} | {g('gate'):.4f} | "
                     f"{g('veto_only'):.4f} | {g('gate+veto'):.4f} | {inter:+.4f} |")

    L += ["", "## 4. Per-class AP under the adopted system", "",
          "| condition | split | system | " + " | ".join(f"{n} AP (n_gt)" for n in CLASS_NAMES.values())
          + " | macro mAP |",
          "|---|---|---|" + "---:|" * (len(CLASS_NAMES) + 1)]
    for cond in ctx.conditions:
        for sname in splits:
            for system in ("gated_fusion", "ir_only", "visible_only"):
                pc = perclass[("gate+veto", cond, system, sname)]
                cells = []
                for c in CLASS_NAMES:
                    d = pc.get(c)
                    cells.append("—" if d is None else f"{d['ap50_95']:.4f} ({d['n_gt']})")
                L.append(f"| {cond} | {sname} | {system} | " + " | ".join(cells)
                         + f" | {scalar[('gate+veto', cond, system, sname)]:.4f} |")

    L += ["", "### 4.1 The headline night delta, per class", "",
          "| class | gated | ir_only | delta | 95% CI | sign flips |",
          "|---|---:|---:|---:|---|---:|"]
    for c, name in CLASS_NAMES.items():
        b = boots[("headline_cls", "clean", f"night/{name}")]
        L.append(f"| {name} | {b['a']:.4f} | {b['b']:.4f} | {b['delta']:+.4f} | "
                 f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {b['p_sign_flip']:.1%} |")

    real_sum = sum(scalar[("gate+veto", c, "gated_fusion", s)]
                   for c in REAL_CONDITIONS for s in splits)
    synth_sum = sum(scalar[("gate+veto", c, "gated_fusion", s)]
                    for c in ctx.conditions if c not in REAL_CONDITIONS for s in splits)
    n_real = len([c for c in ctx.conditions if c in REAL_CONDITIONS]) * len(splits)
    n_syn = len([c for c in ctx.conditions if c not in REAL_CONDITIONS]) * len(splits)
    L += ["", "## 5. Real vs synthetic evidence in the 8-cell sum", "",
          f"§2.4 established that `D` responds 21x differently to synthetic darkening than to "
          f"real night, because real night is inside its reference set. By that argument the "
          f"corruption cells test a transform the gate can see, not real-world failure.",
          "",
          f"| cells | n | sum | mean |",
          f"|---|---:|---:|---:|",
          f"| real (clean day + clean night) | {n_real} | {real_sum:.4f} | "
          f"{real_sum / n_real if n_real else float('nan'):.4f} |",
          f"| synthetic (fog/lowlight/glare) | {n_syn} | {synth_sum:.4f} | "
          f"{synth_sum / n_syn if n_syn else float('nan'):.4f} |",
          f"| **all** | {n_real + n_syn} | **{real_sum + synth_sum:.4f}** | "
          f"{(real_sum + synth_sum) / max(n_real + n_syn, 1):.4f} |",
          "",
          f"Synthetic cells are **{100 * n_syn / max(n_real + n_syn, 1):.0f}% of the cell count** and "
          f"**{100 * synth_sum / max(real_sum + synth_sum, 1e-9):.0f}% of the summed score**."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    payload = {
        "scalar": {"|".join(map(str, k)): v for k, v in scalar.items()},
        "bootstrap": {"|".join(map(str, k)): v for k, v in boots.items()},
        "per_class": {"|".join(map(str, k)): {str(c): d for c, d in v.items()}
                      for k, v in perclass.items()},
        "n_boot": args.n_boot, "seed": args.seed,
        "cap_vis": ctx.cap_vis, "cap_ir": ctx.cap_ir,
    }
    out.with_suffix(".json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[ci] wrote {out} in {time.time() - t0:.0f}s total")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
