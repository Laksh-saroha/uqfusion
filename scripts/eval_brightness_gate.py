"""Evaluate the photometric gate term end to end — TODO §0.2 / §0.3.

Re-runs the Table 3 systems twice on identical caches, identical homography and
identical tuned WBF settings, changing exactly one thing: whether `r_frame` also
sees how bright the frame is.

    BEFORE:  r_frame = 1 - sigmoid((D - mu_d)/tau)
    AFTER:   r_frame = min( that , sigmoid((b - mu_b)/tau_b) )

`mu_b`/`tau_b` come from `fit_brightness_gate.py`, fitted on pohang00/02/03
synthetic lowlight ONLY. pohang01 -- the real night run this is meant to fix --
is never seen during fitting, so its rows below are a held-out result.

The per-run table is the point. Pooled condition numbers average a run where VIS
scores 0.40 with one where it scores 0.0000, which is how the failure hid for as
long as it did.

Usage:
    python scripts/eval_brightness_gate.py --out runs/eval/brightness_gate.md
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
# Chosen on the corrupt-seed-2 ladder, applied to the seed-1 caches (plan B5-5).
TUNED = {"iou_thr_wbf": 0.85, "skip_box_thr": 0.0}


def brightness_of(stem: str, stat: str, bright_dir: Path) -> np.ndarray:
    p = bright_dir / f"{stem}.json"
    if not p.is_file():
        raise SystemExit(f"missing {p} — run scripts/frame_brightness.py --cache runs/cache/{stem}.pkl")
    return np.asarray([f[stat] for f in json.loads(p.read_text(encoding="utf-8"))["frames"]], dtype=float)


def constants_from(payload: dict, alpha: float) -> ReliabilityConstants:
    return ReliabilityConstants(lam=payload["lam"], mu_d=payload["mu_d"], tau=payload["tau"], alpha=alpha)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache-dir", default="runs/cache")
    parser.add_argument("--bright-dir", default="runs/derived/brightness")
    parser.add_argument("--constants", default="runs/eval/reliability_constants.json")
    parser.add_argument("--brightness-constants", default="runs/eval/brightness_constants.json")
    parser.add_argument("--homography", default="runs/derived/homography_ir_to_vis.json")
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--out", default="runs/eval/brightness_gate.md")
    args = parser.parse_args()
    load_config(args.config)

    cache_dir, bright_dir = Path(args.cache_dir), Path(args.bright_dir)
    rc = json.loads(Path(args.constants).read_text(encoding="utf-8"))
    bc = json.loads(Path(args.brightness_constants).read_text(encoding="utf-8"))["vis"]
    stat = bc["stat"]

    c_vis = constants_from(rc["vis"], args.alpha)
    c_ir = constants_from(rc["ir"], args.alpha)
    c_vis_b = replace(c_vis, mu_b=bc["mu_b"], tau_b=bc["tau_b"], bright_stat=stat)
    print(f"[eval] VIS constants  mu_d={c_vis.mu_d:.2f} tau={c_vis.tau:.2f} | "
          f"mu_b={bc['mu_b']:.2f} tau_b={bc['tau_b']:.2f} (stat={stat})")

    # Per-run homography, keyed by the run each frame belongs to.
    h_by_run = {k: np.asarray(v["H_ir_canvas_to_vis_canvas"], dtype=float) for k, v in
                json.loads(Path(args.homography).read_text(encoding="utf-8"))["runs"].items()}

    ir_recs, _ = load_cache(cache_dir / "gauss_ir_paired_clean.pkl")
    scorer_vis = MahalanobisScorer().fit(np.stack([r["feat"] for r in load_cache(cache_dir / "gauss_vis_train_clean.pkl")[0]]))
    scorer_ir = MahalanobisScorer().fit(np.stack([r["feat"] for r in load_cache(cache_dir / "gauss_ir_train_clean.pkl")[0]]))
    # No photometric term on IR by design: on a thermal sensor "brightness" is
    # scene temperature, not illumination, so a dark IR frame is cold water --
    # exactly the condition IR is supposed to be good at. The fix is VIS-only.

    # Clean-val capability prior (§10.4), unchanged by this experiment.
    vis_clean_recs, _ = load_cache(cache_dir / "gauss_vis_paired_clean.pkl")
    runs = [Path(r["image_path"]).parent.name for r in vis_clean_recs]
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

    lines = ["# Photometric gate term — before / after", "",
             f"Fitted on {'+'.join(bc['fit_runs'])} synthetic lowlight; **{bc['held_out_run']} "
             f"(real night) held out of the fit entirely**.", "",
             f"`mu_b` = {bc['mu_b']:.3f}, `tau_b` = {bc['tau_b']:.3f}, statistic = `{stat}`, "
             f"combination = `min`. WBF `iou_thr` = {TUNED['iou_thr_wbf']}, `alpha` = {args.alpha}.", ""]

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    results = {}
    for cond in conditions:
        vr, _ = load_cache(cache_dir / f"gauss_vis_paired_{cond}.pkl")
        b_vis = brightness_of(f"gauss_vis_paired_{cond}", stat, bright_dir)
        common = dict(vis_records=vr, ir_records=ir_recs, scorer=scorer_vis, scorer_ir=scorer_ir,
                      constants_ir=c_ir, h_ir_to_vis=h_frames, capability_vis=cap_vis,
                      capability_ir=cap_ir, gts=gts, **TUNED)
        before = evaluate_systems(constants=c_vis, **common)
        after = evaluate_systems(constants=c_vis_b, brightness_vis=b_vis, brightness_ir=None, **common)
        results[cond] = {"before": before, "after": after, "runs": runs, "b_vis": b_vis}
        print(f"[eval] {cond:9s} gated {before['gated_fusion']['map50_95']:.4f} -> "
              f"{after['gated_fusion']['map50_95']:.4f}   "
              f"mean w_vis {np.mean(before['w_vis_gated']):.3f} -> {np.mean(after['w_vis_gated']):.3f}")

    # --- pooled table --------------------------------------------------------
    lines += ["## Pooled over all 2,232 paired frames", "",
              "| condition | visible only | ir only | gated (before) | **gated (after)** | mean w_vis before | after |",
              "|---|---|---|---|---|---|---|"]
    for cond in conditions:
        r = results[cond]
        lines.append(f"| {cond} | {r['before']['visible_only']['map50_95']:.4f} | "
                     f"{r['before']['ir_only']['map50_95']:.4f} | "
                     f"{r['before']['gated_fusion']['map50_95']:.4f} | "
                     f"**{r['after']['gated_fusion']['map50_95']:.4f}** | "
                     f"{np.mean(r['before']['w_vis_gated']):.3f} | "
                     f"{np.mean(r['after']['w_vis_gated']):.3f} |")

    # --- per-run table: the one that matters ---------------------------------
    lines += ["", "## Per run — clean condition (pohang01 is the held-out real night run)", "",
              "| run | frames | mean brightness | visible only | ir only | gated (before) | **gated (after)** | mean w_vis before | after |",
              "|---|---|---|---|---|---|---|---|---|"]
    r = results["clean"]
    order = ["pohang00", "pohang01", "pohang02", "pohang03"]
    runs_arr = np.asarray(runs)
    for name in order:
        sel = np.flatnonzero(runs_arr == name)
        if not len(sel):
            continue
        sub_gt = [gts[i] for i in sel]
        vis_sub = [vis_clean_recs[i] for i in sel]
        row = {
            "vis": map50_95(vis_sub, sub_gt)["map50_95"],
            "gb": map50_95([r["before"]["fused_gated"][i] for i in sel], sub_gt)["map50_95"],
            "ga": map50_95([r["after"]["fused_gated"][i] for i in sel], sub_gt)["map50_95"],
            "io": map50_95([r["before"]["ir_in_vis"][i] for i in sel], sub_gt)["map50_95"],
        }
        lines.append(f"| {name} | {len(sel)} | {r['b_vis'][sel].mean():.1f} | {row['vis']:.4f} | "
                     f"{row['io']:.4f} | {row['gb']:.4f} | **{row['ga']:.4f}** | "
                     f"{np.mean([r['before']['w_vis_gated'][i] for i in sel]):.3f} | "
                     f"{np.mean([r['after']['w_vis_gated'][i] for i in sel]):.3f} |")

    # --- day / night pooled split (TODO §0.1) --------------------------------
    # Pooled mAP is NOT the average of per-run mAPs: it ranks every detection in
    # one global list, so injecting night-frame detections changes the precision
    # of daylight frames at the same recall. That is why the pooled row can fall
    # while every per-run row rises. Split, the two populations stop contaminating
    # each other -- which is the whole argument for reporting them separately.
    night = runs_arr == "pohang01"
    lines += ["", "## Day vs night, pooled (TODO §0.1)", "",
              "| condition | split | frames | visible only | ir only | gated (before) | **gated (after)** |",
              "|---|---|---|---|---|---|---|"]
    for cond in conditions:
        rr = results[cond]
        for label, sel in (("day (00+02+03)", np.flatnonzero(~night)), ("night (01)", np.flatnonzero(night))):
            g = [gts[i] for i in sel]
            vr_c, _ = load_cache(cache_dir / f"gauss_vis_paired_{cond}.pkl")
            lines.append(
                f"| {cond} | {label} | {len(sel)} | "
                f"{map50_95([vr_c[i] for i in sel], g)['map50_95']:.4f} | "
                f"{map50_95([rr['before']['ir_in_vis'][i] for i in sel], g)['map50_95']:.4f} | "
                f"{map50_95([rr['before']['fused_gated'][i] for i in sel], g)['map50_95']:.4f} | "
                f"**{map50_95([rr['after']['fused_gated'][i] for i in sel], g)['map50_95']:.4f}** |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[eval] -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
