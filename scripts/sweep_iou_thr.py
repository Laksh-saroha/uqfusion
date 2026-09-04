"""Re-sweep WBF `iou_thr` past the boundary it hit — TODO §0.5.

The cheap-fixes pass swept 0.40 / 0.55 / 0.70 / 0.85 and the tuning mean rose
monotonically to the top of the range. **0.85 is a boundary hit, not an optimum**,
and it is the constant sitting under every table in §14, §16 and §17. This
extends the grid to 0.90 / 0.95 / 0.99 under the CURRENT gate configuration
(photometric term + hard veto), because the optimum can move once a modality can
leave the input list entirely.

Protocol, unchanged from plan B5-5: tune on the **corrupt-seed-2 ladder**, report
on the **corrupt-seed-1** paired caches. The two never share a corruption seed.

One caveat the original sweep did not state: the ladder is every third paired
frame, so it CONTAINS pohang01 night frames — the seed split separates corruption
draws, not recording runs. So `iou_thr` is not held out w.r.t. pohang01 the way
`mu_b` is. This script therefore reports the tuning curve twice, pooled and
restricted to the fit runs (pohang00/02/03), and flags whether the two argmaxes
agree. If they do, the concern is moot; if they do not, believe the fit-run one.

Usage:
    python scripts/sweep_iou_thr.py --out runs/eval/iou_thr_sweep.md
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.fusion_eval import evaluate_systems
from uqfusion.eval.matching import load_gt, map50_95
from uqfusion.uq.mahalanobis import MahalanobisScorer
from uqfusion.uq.reliability import fit_constants, per_box_uncertainty

CAP_VIS, CAP_IR = 0.2580, 0.0206
LADDER_CONDS = {"clean": "vis_clean", "fog": "vis_fog_s2",
                "lowlight": "vis_lowlight_s2", "glare": "vis_glare_s2"}
EVAL_CONDS = ["clean", "fog", "lowlight", "glare"]
GRID = (0.40, 0.55, 0.70, 0.85, 0.90, 0.95, 0.99)
FIT_RUNS = ("pohang00", "pohang02", "pohang03")
NIGHT_RUN = "pohang01"


def scorer_from(path) -> MahalanobisScorer:
    return MahalanobisScorer().fit(np.stack([r["feat"] for r in load_cache(path)[0]]))


def constants_from(records, scorer, fitted, key, alpha):
    u = np.concatenate([per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
                        for r in records if len(r["conf"])])
    d = np.asarray([scorer.score(r["feat"]) for r in records])
    c = fit_constants(u, d, alpha=alpha)
    return replace(c, mu_d=fitted[key]["mu_d"], tau=fitted[key]["tau"], lam=fitted[key]["lam"])


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
    parser.add_argument("--alpha", type=float, default=0.5)
    parser.add_argument("--veto", type=float, default=0.5)
    parser.add_argument("--out", default="runs/eval/iou_thr_sweep.md")
    args = parser.parse_args()
    load_config(args.config)

    cache_dir, bright_dir = Path(args.cache_dir), Path(args.bright_dir)
    fitted = json.loads(Path(args.constants).read_text(encoding="utf-8"))
    bc = json.loads(Path(args.brightness_constants).read_text(encoding="utf-8"))["vis"]
    stat = bc["stat"]

    sv = scorer_from(cache_dir / "gauss_vis_train_clean.pkl")
    si = scorer_from(cache_dir / "gauss_ir_train_clean.pkl")

    Hj = json.loads(Path("runs/derived/homography_ir_to_vis.json").read_text(encoding="utf-8"))["runs"]
    manifest = list(csv.DictReader(open("runs/derived/paired_val_manifest.csv", newline="", encoding="utf-8")))
    runs_eval = [r["run"] for r in manifest]
    H_eval = [np.asarray(Hj[r]["H_ir_canvas_to_vis_canvas"], dtype=np.float64) for r in runs_eval]

    # ---- TUNING split: corrupt-seed-2 ladder, every third paired frame -------
    ir_lad, _ = load_cache(cache_dir / "ladder/ir_clean.pkl")
    vis_lad = {c: load_cache(cache_dir / f"ladder/{n}.pkl")[0] for c, n in LADDER_CONDS.items()}
    b_lad = {c: brightness_of(n, stat, bright_dir) for c, n in LADDER_CONDS.items()}
    H_lad = H_eval[::3]
    runs_lad = np.asarray(runs_eval[::3])
    assert len(H_lad) == len(ir_lad), f"ladder H {len(H_lad)} vs frames {len(ir_lad)}"
    cv_l = replace(constants_from(vis_lad["clean"], sv, fitted, "vis", args.alpha),
                   mu_b=bc["mu_b"], tau_b=bc["tau_b"], bright_stat=stat)
    ci_l = constants_from(ir_lad, si, fitted, "ir", args.alpha)
    gts_lad = [load_gt(r["image_path"], r["image_hw"]) for r in vis_lad["clean"]]
    fit_sel = np.flatnonzero(np.isin(runs_lad, FIT_RUNS))
    print(f"[sweep] tuning on {len(ir_lad)} seed-2 ladder frames "
          f"({len(fit_sel)} fit-run, {len(ir_lad) - len(fit_sel)} night)")

    def ladder_row(thr):
        pooled, fitonly = {}, {}
        for c in LADDER_CONDS:
            res = evaluate_systems(vis_lad[c], ir_lad, sv, cv_l, h_ir_to_vis=H_lad, scorer_ir=si,
                                   constants_ir=ci_l, capability_vis=CAP_VIS, capability_ir=CAP_IR,
                                   gts=gts_lad, brightness_vis=b_lad[c], brightness_ir=None,
                                   iou_thr_wbf=thr, veto_below=args.veto)
            pooled[c] = res["gated_fusion"]["map50_95"]
            fitonly[c] = map50_95([res["fused_gated"][i] for i in fit_sel],
                                  [gts_lad[i] for i in fit_sel])["map50_95"]
        pooled["mean"] = float(np.mean([pooled[c] for c in LADDER_CONDS]))
        fitonly["mean"] = float(np.mean([fitonly[c] for c in LADDER_CONDS]))
        return pooled, fitonly

    tuning = {}
    for thr in GRID:
        tuning[thr] = ladder_row(thr)
        print(f"[sweep] iou_thr {thr:.2f}  tuning mean {tuning[thr][0]['mean']:.4f}  "
              f"(fit-run only {tuning[thr][1]['mean']:.4f})")

    best_pooled = max(GRID, key=lambda t: tuning[t][0]["mean"])
    best_fit = max(GRID, key=lambda t: tuning[t][1]["mean"])
    agree = best_pooled == best_fit
    chosen = best_fit if not agree else best_pooled
    print(f"[sweep] argmax pooled {best_pooled:.2f} | fit-run {best_fit:.2f} | "
          f"{'AGREE' if agree else 'DISAGREE -> using fit-run'} -> chosen {chosen:.2f}")

    # ---- REPORTING split: corrupt-seed-1 paired caches -----------------------
    ir_eval, _ = load_cache(cache_dir / "gauss_ir_paired_clean.pkl")
    vis_eval = {c: load_cache(cache_dir / f"gauss_vis_paired_{c}.pkl")[0] for c in EVAL_CONDS}
    b_eval = {c: brightness_of(f"gauss_vis_paired_{c}", stat, bright_dir) for c in EVAL_CONDS}
    cv = replace(constants_from(vis_eval["clean"], sv, fitted, "vis", args.alpha),
                 mu_b=bc["mu_b"], tau_b=bc["tau_b"], bright_stat=stat)
    ci = constants_from(ir_eval, si, fitted, "ir", args.alpha)
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in vis_eval["clean"]]
    runs_arr = np.asarray(runs_eval)
    night = np.flatnonzero(runs_arr == NIGHT_RUN)
    day = np.flatnonzero(runs_arr != NIGHT_RUN)

    report = {}
    for thr in sorted({0.85, chosen}):
        cells = {}
        for c in EVAL_CONDS:
            res = evaluate_systems(vis_eval[c], ir_eval, sv, cv, h_ir_to_vis=H_eval, scorer_ir=si,
                                   constants_ir=ci, capability_vis=CAP_VIS, capability_ir=CAP_IR,
                                   gts=gts, brightness_vis=b_eval[c], brightness_ir=None,
                                   iou_thr_wbf=thr, veto_below=args.veto)
            cells[c] = {
                "pooled": res["gated_fusion"]["map50_95"],
                "day": map50_95([res["fused_gated"][i] for i in day], [gts[i] for i in day])["map50_95"],
                "night": map50_95([res["fused_gated"][i] for i in night], [gts[i] for i in night])["map50_95"],
            }
        report[thr] = cells
        s = sum(cells[c][k] for c in EVAL_CONDS for k in ("day", "night"))
        print(f"[sweep] REPORT iou_thr {thr:.2f}  8-cell sum {s:.4f}")

    # ---- write ---------------------------------------------------------------
    lines = ["# WBF `iou_thr` re-sweep past the 0.85 boundary — TODO §0.5", "",
             f"Tuned on the corrupt-seed-2 ladder ({len(ir_lad)} frames), reported on the "
             f"corrupt-seed-1 paired caches (2,232 frames) — plan B5-5. Gate configuration is "
             f"the adopted one: `{stat}` photometric term, `mu_b`={bc['mu_b']:.3f}, "
             f"`tau_b`={bc['tau_b']:.3f}, hard veto at `r_bright < {args.veto}`, `alpha`={args.alpha}.", "",
             "## Tuning curve (seed-2 ladder)", "",
             "The ladder contains pohang01 night frames — the seed split separates corruption "
             "draws, not recording runs — so the curve is given pooled and restricted to the "
             "fit runs (pohang00/02/03). Decision follows the fit-run column if they disagree.", "",
             "| iou_thr | " + " | ".join(LADDER_CONDS) + " | mean (pooled) | mean (fit runs only) |",
             "|---|" + "---|" * (len(LADDER_CONDS) + 2)]
    for thr in GRID:
        p, f = tuning[thr]
        mark = " **<-**" if thr == chosen else ""
        lines.append(f"| {thr:.2f} | " + " | ".join(f"{p[c]:.4f}" for c in LADDER_CONDS) +
                     f" | **{p['mean']:.4f}** | {f['mean']:.4f}{mark} |")
    lines += ["",
              f"Argmax pooled **{best_pooled:.2f}**, argmax fit-run-only **{best_fit:.2f}** — "
              f"{'they agree, so the night frames in the ladder did not drive the choice.' if agree else 'THEY DISAGREE; the fit-run argmax is used.'}",
              f"Chosen: **iou_thr = {chosen:.2f}** (previous value 0.85).", ""]

    lines += ["## Reported on seed-1, day/night split", "",
              "| iou_thr | condition | day | night | pooled |", "|---|---|---|---|---|"]
    for thr in sorted(report):
        for c in EVAL_CONDS:
            v = report[thr][c]
            lines.append(f"| {thr:.2f} | {c} | {v['day']:.4f} | {v['night']:.4f} | {v['pooled']:.4f} |")
    lines += [""]
    for thr in sorted(report):
        s = sum(report[thr][c][k] for c in EVAL_CONDS for k in ("day", "night"))
        lines.append(f"- `iou_thr` = {thr:.2f}: 8-cell day/night sum **{s:.4f}**")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[sweep] -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
