"""Hard-veto rule for a failed modality — TODO §0.3, evaluated end to end.

**The failure this answers.** §0.2 gave the gate an axis for darkness, and it
works: on the real night run `r_frame_vis` collapses to ~0.05. Gated fusion still
scores 0.0791 against `ir_only`'s 0.0810 — it is *still* worse than simply
ignoring VIS. Down-weighting is not enough, for two reasons that neither shrink
as w -> 0:

  1. **Weights normalize.** w_vis = R_vis*cap_vis / (R_vis*cap_vis + R_ir*cap_ir).
     The capability prior hands VIS a 12.5x advantage (0.258 vs 0.0206), so even
     R_vis = 0.016 against R_ir = 0.8 leaves w_vis at 0.199.
  2. **WBF does not drop low-weight boxes.** It rescales their scores and it
     divides every single-modality cluster by the number of input lists, so a
     blind stream's mere presence halves the surviving stream's confidences.
     mAP is rank-based; a rescaled false positive still occupies a rank.

**The rule (no new constant).** A modality is excluded from fusion when

    r_bright_m < 0.5   <=>   b_m < mu_b_m

i.e. when the frame is on the dark side of a boundary the gate ALREADY
pre-registered in §0.2b (mu_b is the margin midpoint). 0.5 is the sigmoid
midpoint, not a tuned threshold. Vetoing both modalities is refused; that case is
the plan-B3 abstain, signalled by R_sys. IR has no photometric term by design
(dark IR is cold water), so IR is never vetoed.

**Why the veto is NOT extended to the Mahalanobis axis** (`--veto-on r_frame`,
still runnable): D answers "is this frame unusual?", which is a different
question from "did this sensor fail?", and it is not monotone in capability —
§9.4's blur/glare counterexample said so on the ladder. Measured here on the FIT
runs: glare pushes D past mu_d on ~47% of DAYLIGHT pohang00/02/03 frames, where
VIS still scores 0.2626 against IR's 0.0092. Vetoing there costs glare/day
0.2532 -> 0.1435. A non-monotone signal may down-weight; it must not veto.

pohang01 is the held-out run and is not used to choose anything — including this
correction, which was decided on daylight fit-run glare.

Usage:
    python scripts/eval_veto_rule.py --out runs/eval/veto_rule.md
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.fusion_eval import evaluate_systems
from uqfusion.eval.matching import load_gt, map50_95
from uqfusion.uq.fusion import apply_homography
from uqfusion.uq.mahalanobis import MahalanobisScorer
from uqfusion.uq.reliability import ReliabilityConstants

CONDITIONS = ("clean", "fog", "lowlight", "glare")
TUNED = {"iou_thr_wbf": 0.85, "skip_box_thr": 0.0}
ORDER = ["pohang00", "pohang01", "pohang02", "pohang03"]
NIGHT_RUN = "pohang01"


def brightness_of(stem: str, stat: str, bright_dir: Path) -> np.ndarray:
    p = bright_dir / f"{stem}.json"
    if not p.is_file():
        raise SystemExit(f"missing {p} — run scripts/frame_brightness.py --cache runs/cache/{stem}.pkl")
    return np.asarray([f[stat] for f in json.loads(p.read_text(encoding="utf-8"))["frames"]], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-dir", default="runs/cache")
    parser.add_argument("--bright-dir", default="runs/derived/brightness")
    parser.add_argument("--constants", default="runs/eval/reliability_constants.json")
    parser.add_argument("--brightness-constants", default="runs/eval/brightness_constants.json")
    parser.add_argument("--homography", default="runs/derived/homography_ir_to_vis.json")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--veto", type=float, default=0.5,
                        help="veto signal below this excludes the modality; 0.5 IS the pre-registered boundary")
    parser.add_argument("--veto-on", choices=["r_bright", "r_frame"], default="r_bright",
                        help="which signal holds veto authority (see module docstring)")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--out", default="runs/eval/veto_rule.md")
    args = parser.parse_args()
    load_config(args.config)

    cache_dir, bright_dir = Path(args.cache_dir), Path(args.bright_dir)
    rc = json.loads(Path(args.constants).read_text(encoding="utf-8"))
    bc = json.loads(Path(args.brightness_constants).read_text(encoding="utf-8"))["vis"]
    stat = bc["stat"]

    def consts(key):
        return ReliabilityConstants(lam=rc[key]["lam"], mu_d=rc[key]["mu_d"], tau=rc[key]["tau"], alpha=args.alpha)

    c_vis = replace(consts("vis"), mu_b=bc["mu_b"], tau_b=bc["tau_b"], bright_stat=stat)
    c_ir = consts("ir")
    print(f"[veto] VIS mu_d={c_vis.mu_d:.2f} tau={c_vis.tau:.2f} mu_b={bc['mu_b']:.3f} "
          f"tau_b={bc['tau_b']:.3f} ({stat}) | veto at {args.veto_on} < {args.veto}")

    h_by_run = {k: np.asarray(v["H_ir_canvas_to_vis_canvas"], dtype=float) for k, v in
                json.loads(Path(args.homography).read_text(encoding="utf-8"))["runs"].items()}
    ir_recs, _ = load_cache(cache_dir / "gauss_ir_paired_clean.pkl")
    scorer_vis = MahalanobisScorer().fit(
        np.stack([r["feat"] for r in load_cache(cache_dir / "gauss_vis_train_clean.pkl")[0]]))
    scorer_ir = MahalanobisScorer().fit(
        np.stack([r["feat"] for r in load_cache(cache_dir / "gauss_ir_train_clean.pkl")[0]]))

    vis_clean_recs, _ = load_cache(cache_dir / "gauss_vis_paired_clean.pkl")
    runs = [Path(r["image_path"]).parent.name for r in vis_clean_recs]
    runs_arr = np.asarray(runs)
    h_frames = [h_by_run[r] for r in runs]
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in vis_clean_recs]
    cap_vis = map50_95(vis_clean_recs, gts)["map50_95"]
    # The capability prior must be each modality's expected absolute capability
    # ON THIS EVALUATION: fused output scored in the VIS frame against VIS GT.
    # For IR that is the IR-only row (boxes mapped through H), NOT the IR clean
    # mAP recorded in reliability_constants.json — that one is measured on the IR
    # ladder against IR GT and is 3.3x larger (0.0676 vs 0.0206), which silently
    # over-weighted IR in every table built by this script before 2026-08-19.
    # Computed here rather than read from a file so the two can never diverge.
    cap_ir = map50_95(
        [{**r, "boxes_xyxy": apply_homography(np.asarray(r["boxes_xyxy"]).reshape(-1, 4), h)}
         for r, h in zip(ir_recs, h_frames)], gts)["map50_95"]
    print(f"[cap] capability prior: VIS {cap_vis:.4f}  IR {cap_ir:.4f}  "
          f"(ratio {cap_vis / cap_ir:.1f}x)")

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    results = {}
    for cond in conditions:
        vr, _ = load_cache(cache_dir / f"gauss_vis_paired_{cond}.pkl")
        b_vis = brightness_of(f"gauss_vis_paired_{cond}", stat, bright_dir)
        common = dict(vis_records=vr, ir_records=ir_recs, scorer=scorer_vis, scorer_ir=scorer_ir,
                      constants=c_vis, constants_ir=c_ir, h_ir_to_vis=h_frames, capability_vis=cap_vis,
                      capability_ir=cap_ir, gts=gts, brightness_vis=b_vis, brightness_ir=None, **TUNED)
        before = evaluate_systems(**common)
        after = evaluate_systems(veto_below=args.veto, veto_on=args.veto_on, **common)
        results[cond] = {"before": before, "after": after}
        v, vi = np.asarray(after["veto_vis"]), np.asarray(after["veto_ir"])
        print(f"[veto] {cond:9s} gated {before['gated_fusion']['map50_95']:.4f} -> "
              f"{after['gated_fusion']['map50_95']:.4f}   "
              f"vetoed VIS {v.mean():.1%}  IR {vi.mean():.1%}")

    lines = ["# Hard-veto rule for a failed modality — TODO §0.3", "",
             f"Rule: `{args.veto_on} < {args.veto}` excludes the modality from fusion — a boundary "
             f"already pre-registered, not a swept threshold. IR has no photometric term so it is "
             f"never vetoed under the default. Vetoing both is "
             f"refused (that is the R_sys abstain case). Photometric constants: `{stat}`, "
             f"`mu_b`={bc['mu_b']:.3f}, `tau_b`={bc['tau_b']:.3f}. WBF `iou_thr`={TUNED['iou_thr_wbf']}, "
             f"`alpha`={args.alpha}.", "",
             "`before` = §0.2b adopted state (photometric gate, soft weights). "
             "`after` = same, plus the veto.", ""]

    # --- who gets vetoed --------------------------------------------------------
    lines += ["## Veto rate by run and condition (VIS / IR)", "",
              "| condition | " + " | ".join(ORDER) + " |", "|---|" + "---|" * len(ORDER)]
    for cond in conditions:
        a = results[cond]["after"]
        v, vi = np.asarray(a["veto_vis"]), np.asarray(a["veto_ir"])
        cells = []
        for name in ORDER:
            sel = runs_arr == name
            cells.append(f"{v[sel].mean():.0%} / {vi[sel].mean():.0%}" if sel.any() else "—")
        lines.append(f"| {cond} | " + " | ".join(cells) + " |")

    lines += ["", "## `r_frame` — mean (min) by run, clean condition", "",
              "| run | r_frame_vis | r_bright_vis | r_frame_ir |", "|---|---|---|---|"]
    b = results["clean"]["before"]
    rfv, rfi = np.asarray(b["r_frame_vis"]), np.asarray(b["r_frame_ir"])
    rb = np.asarray([x if x is not None else np.nan for x in b["r_bright_vis"]], dtype=float)
    for name in ORDER:
        sel = runs_arr == name
        if not sel.any():
            continue
        lines.append(f"| {name} | {rfv[sel].mean():.4f} ({rfv[sel].min():.4f}) | "
                     f"{np.nanmean(rb[sel]):.4f} ({np.nanmin(rb[sel]):.4f}) | "
                     f"{rfi[sel].mean():.4f} ({rfi[sel].min():.4f}) |")

    # --- per-run mAP ------------------------------------------------------------
    lines += ["", "## Per run — clean condition (pohang01 held out)", "",
              "| run | frames | visible only | ir only | gated (before) | **gated (+veto)** | "
              "mean w_vis before | after |", "|---|---|---|---|---|---|---|---|"]
    r = results["clean"]
    for name in ORDER:
        sel = np.flatnonzero(runs_arr == name)
        if not len(sel):
            continue
        sub_gt = [gts[i] for i in sel]
        lines.append(
            f"| {name} | {len(sel)} | "
            f"{map50_95([vis_clean_recs[i] for i in sel], sub_gt)['map50_95']:.4f} | "
            f"{map50_95([r['before']['ir_in_vis'][i] for i in sel], sub_gt)['map50_95']:.4f} | "
            f"{map50_95([r['before']['fused_gated'][i] for i in sel], sub_gt)['map50_95']:.4f} | "
            f"**{map50_95([r['after']['fused_gated'][i] for i in sel], sub_gt)['map50_95']:.4f}** | "
            f"{np.mean([r['before']['w_vis_gated'][i] for i in sel]):.3f} | "
            f"{np.mean([r['after']['w_vis_gated'][i] for i in sel]):.3f} |")

    # --- day / night ------------------------------------------------------------
    night = runs_arr == NIGHT_RUN
    lines += ["", "## Day vs night, pooled", "",
              "| condition | split | frames | visible only | ir only | gated (before) | **gated (+veto)** |",
              "|---|---|---|---|---|---|---|"]
    for cond in conditions:
        rr = results[cond]
        vr_c, _ = load_cache(cache_dir / f"gauss_vis_paired_{cond}.pkl")
        for label, sel in (("day (00+02+03)", np.flatnonzero(~night)), ("night (01)", np.flatnonzero(night))):
            g = [gts[i] for i in sel]
            lines.append(
                f"| {cond} | {label} | {len(sel)} | "
                f"{map50_95([vr_c[i] for i in sel], g)['map50_95']:.4f} | "
                f"{map50_95([rr['before']['ir_in_vis'][i] for i in sel], g)['map50_95']:.4f} | "
                f"{map50_95([rr['before']['fused_gated'][i] for i in sel], g)['map50_95']:.4f} | "
                f"**{map50_95([rr['after']['fused_gated'][i] for i in sel], g)['map50_95']:.4f}** |")

    lines += ["", "## Pooled over all 2,232 paired frames", "",
              "| condition | visible only | ir only | naive 0.5/0.5 | gated (before) | **gated (+veto)** |",
              "|---|---|---|---|---|---|"]
    for cond in conditions:
        rr = results[cond]
        lines.append(f"| {cond} | {rr['before']['visible_only']['map50_95']:.4f} | "
                     f"{rr['before']['ir_only']['map50_95']:.4f} | "
                     f"{rr['before']['naive_fusion']['map50_95']:.4f} | "
                     f"{rr['before']['gated_fusion']['map50_95']:.4f} | "
                     f"**{rr['after']['gated_fusion']['map50_95']:.4f}** |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[veto] -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
