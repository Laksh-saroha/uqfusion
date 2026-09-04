"""The six CPU-only fusion improvements, run end to end over existing caches.

Each is a separate question; none needs a GPU or a retrain.

  1. WBF `iou_thr`   — re-tuned on the SEED-2 ladder split, then reported on the
                       seed-1 evaluation caches. (The +2.8% measured earlier was
                       tuned on the evaluation set itself and does not count.)
  2. temporal alpha  — scope's D14 sets alpha=1.0 (smoothing off). This is video;
                       tuned on the ladder split like (1).
  3. union labels    — GT becomes VIS labels UNION the IR labels projected into
                       the VIS frame. Today an IR detection of a target the VIS
                       annotator never marked scores as a false positive, which
                       biases every fusion row downward.
  4. R_sys abstain   — plan B3's abstain signal is computed and never used. Build
                       the risk-coverage curve it implies.
  5. skip_box_thr    — IR emits ~54 detections/frame against VIS's ~13. Does a
                       confidence floor on the fused input help?
  6. per-run split   — clean gated mAP measured 0.3590 on pohang00+02 versus
                       0.1020 on pohang01+03. A 3.5x spread nobody has explained.

Tuning/reporting discipline: (1) and (2) are hyperparameters, so they are chosen
on the ladder caches (corrupt-seed 2) and only then reported on the evaluation
caches (corrupt-seed 1) — plan B5-5. (3)-(6) are not tuned; they are measurements.

Usage:
    python scripts/run_cheap_fixes.py --out runs/eval/cheap_fixes.md
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
from uqfusion.eval.matching import iou_matrix, load_gt, map50_95
from uqfusion.uq.fusion import apply_homography
from uqfusion.uq.mahalanobis import MahalanobisScorer
from uqfusion.uq.reliability import fit_constants, per_box_uncertainty

CAP_VIS, CAP_IR = 0.2580, 0.0206          # clean-val capability prior (D-9)
EVAL_CONDS = ["clean", "fog", "lowlight", "glare"]
LADDER_CONDS = {"clean": "vis_clean", "fog": "vis_fog_s2",
                "lowlight": "vis_lowlight_s2", "glare": "vis_glare_s2"}


def scorer_from(path) -> MahalanobisScorer:
    return MahalanobisScorer().fit(np.stack([r["feat"] for r in load_cache(path)[0]]))


def constants_from(records, scorer, fitted, key):
    u = np.concatenate([per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
                        for r in records if len(r["conf"])])
    c = fit_constants(u, np.asarray([scorer.score(r["feat"]) for r in records]))
    return replace(c, mu_d=fitted[key]["mu_d"], tau=fitted[key]["tau"])


def union_gt(vis_records, ir_records, h_frames, dedup_iou=0.5):
    """VIS labels UNION IR labels mapped into the VIS frame.

    An IR box is added only when it does not already match a VIS box of the same
    class at IoU >= dedup_iou, so a target both annotators marked stays ONE
    ground-truth object rather than becoming two.
    """
    out, added, base = [], 0, 0
    for rv, ri, h in zip(vis_records, ir_records, h_frames):
        g_v = load_gt(rv["image_path"], rv["image_hw"])
        g_i = load_gt(ri["image_path"], ri["image_hw"])
        base += len(g_v["cls"])
        if len(g_i["cls"]) == 0:
            out.append(g_v)
            continue
        proj = apply_homography(g_i["boxes_xyxy"], h)
        keep = []
        for k in range(len(proj)):
            if len(g_v["boxes_xyxy"]):
                same = g_v["cls"] == g_i["cls"][k]
                if same.any():
                    m = iou_matrix(proj[k:k + 1], g_v["boxes_xyxy"][same])
                    if m.size and m.max() >= dedup_iou:
                        continue
            keep.append(k)
        added += len(keep)
        out.append({
            "boxes_xyxy": np.vstack([g_v["boxes_xyxy"], proj[keep]]) if keep else g_v["boxes_xyxy"],
            "cls": np.concatenate([g_v["cls"], g_i["cls"][keep]]) if keep else g_v["cls"],
        })
    return out, base, added


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="runs/eval/cheap_fixes.md")
    args = parser.parse_args()
    cfg = load_config(args.config)

    md: list[str] = ["# Cheap fixes — six CPU-only experiments over existing caches", ""]

    def say(line=""):
        print(line, flush=True)
        md.append(line)

    fitted = json.loads(Path("runs/eval/reliability_constants.json").read_text(encoding="utf-8"))
    sv = scorer_from("runs/cache/gauss_vis_train_clean.pkl")
    si = scorer_from("runs/cache/gauss_ir_train_clean.pkl")

    # ---- evaluation caches (corrupt-seed 1) --------------------------------
    ir_eval, _ = load_cache("runs/cache/gauss_ir_paired_clean.pkl")
    vis_eval = {c: load_cache(f"runs/cache/gauss_vis_paired_{'clean' if c == 'clean' else c}.pkl")[0]
                for c in EVAL_CONDS}
    cv = constants_from(vis_eval["clean"], sv, fitted, "vis")
    ci = constants_from(ir_eval, si, fitted, "ir")

    Hj = json.loads(Path("runs/derived/homography_ir_to_vis.json").read_text(encoding="utf-8"))["runs"]
    manifest = list(csv.DictReader(open("runs/derived/paired_val_manifest.csv", newline="", encoding="utf-8")))
    runs_eval = [r["run"] for r in manifest]
    H_eval = [np.asarray(Hj[r]["H_ir_canvas_to_vis_canvas"], dtype=np.float64) for r in runs_eval]

    # ---- ladder caches (corrupt-seed 2) = the TUNING split -----------------
    ir_lad, _ = load_cache("runs/cache/ladder/ir_clean.pkl")
    vis_lad = {c: load_cache(f"runs/cache/ladder/{n}.pkl")[0] for c, n in LADDER_CONDS.items()}
    H_lad = H_eval[::3]
    assert len(H_lad) == len(ir_lad), f"ladder H {len(H_lad)} vs frames {len(ir_lad)}"
    cv_l = constants_from(vis_lad["clean"], sv, fitted, "vis")
    ci_l = constants_from(ir_lad, si, fitted, "ir")

    def run(vis_recs, ir_recs, hs, c_v, c_i, **kw):
        return evaluate_systems(vis_recs, ir_recs, sv, c_v, h_ir_to_vis=hs, scorer_ir=si,
                                constants_ir=c_i, capability_vis=CAP_VIS, capability_ir=CAP_IR, **kw)

    def mean_over(conds, vis_map, ir_recs, hs, c_v, c_i, **kw):
        vals = {}
        for c in conds:
            r = run(vis_map[c], ir_recs, hs, c_v, c_i, **kw)
            vals[c] = r["gated_fusion"]["map50_95"]
        vals["mean"] = float(np.mean(list(vals.values())))
        return vals

    # ======================================================================
    say("## 1. WBF `iou_thr` — tuned on the seed-2 ladder, reported on seed-1")
    say()
    say("| iou_thr | " + " | ".join(LADDER_CONDS) + " | mean (TUNING) |")
    say("|---|" + "---|" * (len(LADDER_CONDS) + 1))
    best_iou, best_score = 0.55, -1.0
    for thr in (0.40, 0.55, 0.70, 0.85):
        v = mean_over(list(LADDER_CONDS), vis_lad, ir_lad, H_lad, cv_l, ci_l, iou_thr_wbf=thr)
        say(f"| {thr:.2f} | " + " | ".join(f"{v[c]:.4f}" for c in LADDER_CONDS) + f" | **{v['mean']:.4f}** |")
        if v["mean"] > best_score:
            best_iou, best_score = thr, v["mean"]
    say()
    say(f"Chosen on the tuning split: **iou_thr = {best_iou:.2f}**.")
    say()

    # ======================================================================
    say("## 2. Temporal smoothing alpha — tuned on the seed-2 ladder")
    say()
    say("Scope decision D14 sets alpha=1.0 (smoothing off). Frames are in temporal order")
    say("within a run; note the EMA is *not* reset at run boundaries, so a small amount of")
    say("cross-run bleed exists at three points in 2,232 frames.")
    say()
    say("| alpha | " + " | ".join(LADDER_CONDS) + " | mean (TUNING) |")
    say("|---|" + "---|" * (len(LADDER_CONDS) + 1))
    best_alpha, best_a_score = 1.0, -1.0
    for a in (1.0, 0.7, 0.5, 0.3, 0.1):
        v = mean_over(list(LADDER_CONDS), vis_lad, ir_lad, H_lad,
                      replace(cv_l, alpha=a), replace(ci_l, alpha=a), iou_thr_wbf=best_iou)
        say(f"| {a:.1f} | " + " | ".join(f"{v[c]:.4f}" for c in LADDER_CONDS) + f" | **{v['mean']:.4f}** |")
        if v["mean"] > best_a_score:
            best_alpha, best_a_score = a, v["mean"]
    say()
    say(f"Chosen on the tuning split: **alpha = {best_alpha:.1f}**.")
    say()

    # ======================================================================
    say("## 5. IR detection floor (`skip_box_thr`) — tuned on the seed-2 ladder")
    say()
    say("| skip_box_thr | " + " | ".join(LADDER_CONDS) + " | mean (TUNING) |")
    say("|---|" + "---|" * (len(LADDER_CONDS) + 1))
    best_skip, best_s_score = 0.0, -1.0
    for sk in (0.0, 0.02, 0.05, 0.10, 0.20):
        v = mean_over(list(LADDER_CONDS), vis_lad, ir_lad, H_lad,
                      replace(cv_l, alpha=best_alpha), replace(ci_l, alpha=best_alpha),
                      iou_thr_wbf=best_iou, skip_box_thr=sk)
        say(f"| {sk:.2f} | " + " | ".join(f"{v[c]:.4f}" for c in LADDER_CONDS) + f" | **{v['mean']:.4f}** |")
        if v["mean"] > best_s_score:
            best_skip, best_s_score = sk, v["mean"]
    say()
    say(f"Chosen on the tuning split: **skip_box_thr = {best_skip:.2f}**.")
    say()

    # ======================================================================
    say("## Held-out result — tuned settings applied to the seed-1 evaluation caches")
    say()
    cv_f, ci_f = replace(cv, alpha=best_alpha), replace(ci, alpha=best_alpha)
    say("| VIS condition | visible only | ir only | naive | gated (before) | **gated (tuned)** |")
    say("|---|---|---|---|---|---|")
    tuned_results = {}
    for c in EVAL_CONDS:
        before = run(vis_eval[c], ir_eval, H_eval, cv, ci, iou_thr_wbf=0.55)
        after = run(vis_eval[c], ir_eval, H_eval, cv_f, ci_f,
                    iou_thr_wbf=best_iou, skip_box_thr=best_skip)
        tuned_results[c] = after
        say(f"| {c} | {before['visible_only']['map50_95']:.4f} | {before['ir_only']['map50_95']:.4f} "
            f"| {before['naive_fusion']['map50_95']:.4f} | {before['gated_fusion']['map50_95']:.4f} "
            f"| **{after['gated_fusion']['map50_95']:.4f}** |")
    say()

    # ======================================================================
    say("## 3. Union-label evaluation (VIS labels UNION projected IR labels)")
    say()
    ugts, n_base, n_added = union_gt(vis_eval["clean"], ir_eval, H_eval)
    say(f"GT boxes: {n_base:,} VIS + {n_added:,} IR-only additions = {n_base + n_added:,} "
        f"(+{n_added / max(n_base, 1) * 100:.1f}%). IR labels that already matched a VIS "
        f"label of the same class at IoU>=0.5 were merged, not duplicated.")
    say()
    say("| VIS condition | system | VIS-only GT | union GT | delta |")
    say("|---|---|---|---|---|")
    for c in EVAL_CONDS:
        u = run(vis_eval[c], ir_eval, H_eval, cv_f, ci_f, iou_thr_wbf=best_iou,
                skip_box_thr=best_skip, gts=ugts)
        for sysname in ("visible_only", "ir_only", "gated_fusion"):
            a = tuned_results[c][sysname]["map50_95"]
            b = u[sysname]["map50_95"]
            say(f"| {c} | {sysname} | {a:.4f} | {b:.4f} | {b - a:+.4f} |")
    say()

    # ======================================================================
    say("## 4. `R_sys` abstain — risk/coverage")
    say()
    say("plan B3's abstain signal, finally used: drop the lowest-R_sys frames and")
    say("re-score the gated system on what remains. A useful signal makes mAP rise")
    say("as coverage falls.")
    say()
    say("| VIS condition | 100% cov | 90% | 75% | 50% |")
    say("|---|---|---|---|---|")
    for c in EVAL_CONDS:
        res = tuned_results[c]
        rsys = np.asarray(res["R_sys"])
        fused, gts_c = res["fused_gated"], res["gts"]
        order = np.argsort(-rsys)
        cells = []
        for cov in (1.0, 0.9, 0.75, 0.5):
            k = max(int(round(cov * len(order))), 1)
            idx = order[:k]
            cells.append(f"{map50_95([fused[i] for i in idx], [gts_c[i] for i in idx])['map50_95']:.4f}")
        say(f"| {c} | " + " | ".join(cells) + " |")
    say()

    # ======================================================================
    say("## 6. Per-run breakdown — explaining the 3.5x spread")
    say()
    runs_arr = np.array(runs_eval)
    say("| run | frames | GT boxes | VIS dets | IR dets | median D (VIS) | visible only | ir only | gated |")
    say("|---|---|---|---|---|---|---|---|---|")
    d_vis_all = sv.score(np.stack([r["feat"] for r in vis_eval["clean"]]))
    for rn in sorted(set(runs_eval)):
        m = np.flatnonzero(runs_arr == rn)
        vr = [vis_eval["clean"][i] for i in m]
        ir = [ir_eval[i] for i in m]
        g = [load_gt(r["image_path"], r["image_hw"]) for r in vr]
        res = run(vr, ir, [H_eval[i] for i in m], cv_f, ci_f,
                  iou_thr_wbf=best_iou, skip_box_thr=best_skip)
        say(f"| {rn} | {len(m)} | {sum(len(x['cls']) for x in g):,} "
            f"| {sum(len(r['conf']) for r in vr):,} | {sum(len(r['conf']) for r in ir):,} "
            f"| {np.median(d_vis_all[m]):.1f} "
            f"| {res['visible_only']['map50_95']:.4f} | {res['ir_only']['map50_95']:.4f} "
            f"| {res['gated_fusion']['map50_95']:.4f} |")
    say()

    say("## Chosen settings")
    say()
    say(f"- `iou_thr` = {best_iou:.2f}")
    say(f"- `alpha` = {best_alpha:.1f}")
    say(f"- `skip_box_thr` = {best_skip:.2f}")
    say()
    say("All three were selected on the corrupt-seed-2 ladder caches and only then")
    say("applied to the corrupt-seed-1 evaluation caches (plan B5-5).")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(md) + "\n", encoding="utf-8")
    print(f"\n[cheap-fixes] -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
