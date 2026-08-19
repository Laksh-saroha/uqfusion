"""sigma-weighted WBF — TODO A1, the paper's central claim, evaluated end to end.

**The gap this closes.** Before this, *nothing the Gaussian head produced
influenced fusion*. `r_box` is inert (0.86-0.95 in every condition, including the
fog case where mAP is 0.0012), and stock WBF averages cluster coordinates
weighted by ``score x model_weight`` — a confidence, never a precision. So a
paper about uncertainty-aware fusion was resting entirely on a Mahalanobis frame
score and (since §16) a brightness statistic. Neither is the sigma head.

`sigma_weighted_fusion` changes exactly one thing: cluster coordinates are
averaged by ``score x model_weight / sigma^2``, per coordinate, which is the
minimum-variance estimator for combining independent measurements. A
confident-but-blurry box keeps its full vote on WHETHER an object is there and
loses its vote on WHERE the edges are.

`scripts/smoke_sigma_wbf.py` asserts that path reduces to stock WBF to 6.3e-08
when sigma is constant, so the before/after below measures sigma VARIATION and
not two different fusion implementations.

Configuration is the adopted one throughout: D-6 ladder constants, capability
prior computed on this evaluation (VIS 0.2580 / IR 0.0206), photometric gate on
`p05` with the margin rule, hard veto at `r_bright < 0.5`, WBF `iou_thr` 0.85.

Usage:
    python scripts/eval_sigma_wbf.py --out runs/eval/sigma_wbf.md
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
from uqfusion.uq.reliability import ReliabilityConstants, per_box_uncertainty

CONDITIONS = ("clean", "fog", "lowlight", "glare")
TUNED = {"iou_thr_wbf": 0.85, "skip_box_thr": 0.0}
ORDER = ["pohang00", "pohang01", "pohang02", "pohang03"]
NIGHT_RUN = "pohang01"


def brightness_of(stem: str, stat: str, bright_dir: Path) -> np.ndarray:
    p = bright_dir / f"{stem}.json"
    if not p.is_file():
        raise SystemExit(f"missing {p} — run scripts/frame_brightness.py first")
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
    parser.add_argument("--veto", type=float, default=0.5)
    parser.add_argument("--iou-thr", type=float, default=TUNED["iou_thr_wbf"],
                        help="MECHANISM PROBE ONLY. At the tuned 0.85 just 0.11%% of VIS boxes have "
                             "an IR partner, so sigma has almost no cluster to act in. Lowering this "
                             "tests that explanation; it is NOT a re-tune and must not be selected on.")
    parser.add_argument("--conditions", default=",".join(CONDITIONS))
    parser.add_argument("--out", default="runs/eval/sigma_wbf.md")
    args = parser.parse_args()
    load_config(args.config)

    cache_dir, bright_dir = Path(args.cache_dir), Path(args.bright_dir)
    rc = json.loads(Path(args.constants).read_text(encoding="utf-8"))
    bc = json.loads(Path(args.brightness_constants).read_text(encoding="utf-8"))["vis"]
    stat = bc["stat"]

    def consts(key):
        return ReliabilityConstants(lam=rc[key]["lam"], mu_d=rc[key]["mu_d"],
                                    tau=rc[key]["tau"], alpha=args.alpha)

    c_vis = replace(consts("vis"), mu_b=bc["mu_b"], tau_b=bc["tau_b"], bright_stat=stat)
    c_ir = consts("ir")

    h_by_run = {k: np.asarray(v["H_ir_canvas_to_vis_canvas"], dtype=float) for k, v in
                json.loads(Path(args.homography).read_text(encoding="utf-8"))["runs"].items()}
    ir_recs, _ = load_cache(cache_dir / "gauss_ir_paired_clean.pkl")
    scorer_vis = MahalanobisScorer().fit(
        np.stack([r["feat"] for r in load_cache(cache_dir / "gauss_vis_train_clean.pkl")[0]]))
    scorer_ir = MahalanobisScorer().fit(
        np.stack([r["feat"] for r in load_cache(cache_dir / "gauss_ir_train_clean.pkl")[0]]))

    vis_clean_recs, _ = load_cache(cache_dir / "gauss_vis_paired_clean.pkl")
    runs_arr = np.asarray([Path(r["image_path"]).parent.name for r in vis_clean_recs])
    h_frames = [h_by_run[r] for r in runs_arr]
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in vis_clean_recs]
    cap_vis = map50_95(vis_clean_recs, gts)["map50_95"]
    cap_ir = map50_95(
        [{**r, "boxes_xyxy": apply_homography(np.asarray(r["boxes_xyxy"]).reshape(-1, 4), h)}
         for r, h in zip(ir_recs, h_frames)], gts)["map50_95"]
    print(f"[sigma] capability prior: VIS {cap_vis:.4f}  IR {cap_ir:.4f} (ratio {cap_vis / cap_ir:.1f}x)")

    # How much sigma variation is there to exploit? If sigma is nearly constant
    # the method is a no-op BY CONSTRUCTION, and that is worth knowing first.
    u_vis = np.concatenate([per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
                            for r in vis_clean_recs if len(r["conf"])])
    u_ir = np.concatenate([per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
                           for r in ir_recs if len(r["conf"])])
    spread = {}
    for name, u in (("VIS", u_vis), ("IR", u_ir)):
        q = np.percentile(u, [5, 25, 50, 75, 95])
        spread[name] = q
        print(f"[sigma] u_box {name}: p05 {q[0]:.4f} p50 {q[2]:.4f} p95 {q[4]:.4f} "
              f"(p95/p05 = {q[4] / max(q[0], 1e-9):.1f}x)")

    conditions = tuple(c.strip() for c in args.conditions.split(",") if c.strip())
    results = {}
    for cond in conditions:
        vr, _ = load_cache(cache_dir / f"gauss_vis_paired_{cond}.pkl")
        common = dict(vis_records=vr, ir_records=ir_recs, scorer=scorer_vis, scorer_ir=scorer_ir,
                      constants=c_vis, constants_ir=c_ir, h_ir_to_vis=h_frames,
                      capability_vis=cap_vis, capability_ir=cap_ir, gts=gts,
                      brightness_vis=brightness_of(f"gauss_vis_paired_{cond}", stat, bright_dir),
                      brightness_ir=None, veto_below=args.veto,
                      iou_thr_wbf=args.iou_thr, skip_box_thr=TUNED["skip_box_thr"])
        before = evaluate_systems(**common)
        after = evaluate_systems(sigma_weighted=True, **common)
        results[cond] = {"before": before, "after": after}
        d = after["gated_fusion"]["map50_95"] - before["gated_fusion"]["map50_95"]
        print(f"[sigma] {cond:9s} gated {before['gated_fusion']['map50_95']:.4f} -> "
              f"{after['gated_fusion']['map50_95']:.4f}  ({d:+.4f})")

    lines = ["# sigma-weighted WBF — TODO A1", "",
             "Cluster coordinates averaged by `score x model_weight / sigma^2` (per coordinate) "
             "instead of by `score x model_weight`. Everything else — clustering, score rule, "
             "rescale — is byte-identical to stock WBF; `scripts/smoke_sigma_wbf.py` asserts the "
             "sigma path reproduces stock WBF to 6.3e-08 under constant sigma, so this A/B "
             "isolates sigma variation.", "",
             f"Config: D-6 ladder constants, capability prior VIS {cap_vis:.4f} / IR {cap_ir:.4f}, "
             f"`{stat}` photometric gate (`mu_b`={bc['mu_b']:.3f}), veto at `r_bright < {args.veto}`, "
             f"WBF `iou_thr` {args.iou_thr}, `alpha` {args.alpha}.", "",
             "## How much sigma variation exists to exploit", "",
             "| stream | u_box p05 | p25 | p50 | p75 | p95 | p95/p05 |", "|---|---|---|---|---|---|---|"]
    for name, q in spread.items():
        lines.append(f"| {name} | {q[0]:.4f} | {q[1]:.4f} | {q[2]:.4f} | {q[3]:.4f} | {q[4]:.4f} | "
                     f"{q[4] / max(q[0], 1e-9):.1f}x |")

    lines += ["", "## Day vs night, pooled", "",
              "| condition | split | frames | stock WBF | **sigma-weighted** | delta |",
              "|---|---|---|---|---|---|"]
    night = runs_arr == NIGHT_RUN
    for cond in conditions:
        rr = results[cond]
        for label, sel in (("day (00+02+03)", np.flatnonzero(~night)), ("night (01)", np.flatnonzero(night))):
            g = [gts[i] for i in sel]
            b = map50_95([rr["before"]["fused_gated"][i] for i in sel], g)["map50_95"]
            a = map50_95([rr["after"]["fused_gated"][i] for i in sel], g)["map50_95"]
            lines.append(f"| {cond} | {label} | {len(sel)} | {b:.4f} | **{a:.4f}** | {a - b:+.4f} |")

    lines += ["", "## Per run — clean condition", "",
              "| run | frames | stock WBF | **sigma-weighted** | delta |", "|---|---|---|---|---|"]
    r = results["clean"]
    for name in ORDER:
        sel = np.flatnonzero(runs_arr == name)
        if not len(sel):
            continue
        g = [gts[i] for i in sel]
        b = map50_95([r["before"]["fused_gated"][i] for i in sel], g)["map50_95"]
        a = map50_95([r["after"]["fused_gated"][i] for i in sel], g)["map50_95"]
        lines.append(f"| {name} | {len(sel)} | {b:.4f} | **{a:.4f}** | {a - b:+.4f} |")

    lines += ["", "## Pooled over all 2,232 paired frames", "",
              "| condition | stock WBF | **sigma-weighted** | delta |", "|---|---|---|---|"]
    for cond in conditions:
        b = results[cond]["before"]["gated_fusion"]["map50_95"]
        a = results[cond]["after"]["gated_fusion"]["map50_95"]
        lines.append(f"| {cond} | {b:.4f} | **{a:.4f}** | {a - b:+.4f} |")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[sigma] -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
