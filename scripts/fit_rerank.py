"""Idea I1 -- the learned IoU-aware re-ranker, the largest measured headroom.

`probe_oracle_headroom.py` puts **+0.1060** of mAP@50-95 inside the boxes VIS
already emits, reachable by ordering alone, concentrated at IoU 0.60-0.80. The
shipped fusion system is worth +0.0106. Capturing a fifth of the oracle is twice
the whole gate.

The 2026-09-01 pilot fit this on pohang00 and got **+0.0419 on tune, -0.0045 on
TEST** -- the largest single movement in the project, and pure overfit. 836
frames of one consecutive canal transit is a handful of effective samples. This
script is that pilot with the three things it lacked:

  1. **Leave-one-run-out**, refitting inside every fold, so no frame is ever
     scored by a model that saw its run. `lambda` is chosen by LORO too --
     selecting it on a single tune set is the mistake C8 was written to prevent.
  2. **A monotonicity constraint in `conf`**, so the model can only reorder
     WITHIN a confidence band and can never destroy the base ranking. Without it
     the lambda=1.0 arm collapses (-0.0424), which is the model being handed
     authority it has not earned.
  3. **A feature ablation**, because a term that only works with 18 features has
     not been shown to work.

The target is `max IoU with same-class GT` -- regression, not TP classification.
AP@50-95 pays for localisation and a TP flag at 0.5 cannot express it.

This subsumes the adopted cross-modal `support` term as one feature, so it
generalises what already works instead of competing with it. Pass `--ir-cache`
to include it.

Usage:
    python scripts/fit_rerank.py --cache-dir runs/cache_m
    python scripts/fit_rerank.py --vis-cache runs/cache_day/gauss_vis_day_clean.pkl   # I0 substrate
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import (ROOT, ap_from_scores, day_night, fmt, gts_for,  # noqa: E402
                           load_records, loro_folds, md_table, oracle_curves, sgn,
                           subsample, write_md)

sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import frame_parts   # noqa: E402
from uqfusion.eval.matching import iou_matrix     # noqa: E402
from uqfusion.uq.fusion import apply_homography   # noqa: E402

#: Ordered so `--n-features k` takes a prefix. `conf` is index 0 and is the
#: monotone-constrained one.
FEATURES = [
    "conf", "log_conf", "conf_over_frame_median", "sigma_mean_norm",
    "sigma_mean_px", "sibling_max_iou", "log_area", "aspect",
    "sigma_x1", "sigma_y1", "sigma_x2", "sigma_y2",
    "w", "h", "cx", "cy", "n_boxes_in_frame", "sibling_count",
]
XMODAL = ["xmodal_iou", "xmodal_conf"]


def build_features(rec, ir_in_vis=None):
    b = np.asarray(rec["boxes_xyxy"], float).reshape(-1, 4)
    n = len(b)
    if n == 0:
        return np.zeros((0, len(FEATURES) + (len(XMODAL) if ir_in_vis is not None else 0)))
    c = np.asarray(rec["conf"], float)
    s = np.asarray(rec["sigma_ltrb"], float).reshape(-1, 4)
    cl = np.asarray(rec["cls"]).astype(int)
    w = np.clip(b[:, 2] - b[:, 0], 1e-6, None)
    h = np.clip(b[:, 3] - b[:, 1], 1e-6, None)
    M = iou_matrix(b, b)
    np.fill_diagonal(M, 0.0)
    M = np.where(cl[:, None] == cl[None, :], M, 0.0)
    smax = M.max(1) if n > 1 else np.zeros(n)
    scount = (M >= 0.5).sum(1)
    med = np.median(c) if n else 1.0
    cols = [c, np.log(np.clip(c, 1e-6, None)), c / max(med, 1e-6),
            s.mean(1) / np.sqrt(w * h), s.mean(1), smax,
            np.log(w * h), w / h,
            s[:, 0], s[:, 1], s[:, 2], s[:, 3],
            w, h, (b[:, 0] + b[:, 2]) / 2.0, (b[:, 1] + b[:, 3]) / 2.0,
            np.full(n, float(n)), scount.astype(float)]
    if ir_in_vis is not None:
        ib = np.asarray(ir_in_vis["boxes_xyxy"], float).reshape(-1, 4)
        if len(ib):
            X = iou_matrix(b, ib)
            X = np.where(cl[:, None] == np.asarray(ir_in_vis["cls"]).astype(int)[None, :], X, 0.0)
            j = X.argmax(1)
            cols += [X[np.arange(n), j], np.asarray(ir_in_vis["conf"], float)[j]]
        else:
            cols += [np.zeros(n), np.zeros(n)]
    return np.column_stack(cols)


def build_target(rec, gt):
    b = np.asarray(rec["boxes_xyxy"], float).reshape(-1, 4)
    if not len(b):
        return np.zeros(0)
    if not len(gt["boxes_xyxy"]):
        return np.zeros(len(b))
    M = iou_matrix(b, gt["boxes_xyxy"])
    M = np.where(np.asarray(rec["cls"]).astype(int)[:, None] == gt["cls"][None, :], M, 0.0)
    return M.max(1)


def fit_one(Xtr, ytr, n_feat, monotone, seed=0):
    from sklearn.ensemble import HistGradientBoostingRegressor
    cst = None
    if monotone:
        cst = np.zeros(n_feat, dtype=int)
        cst[0] = 1  # conf: the model may sharpen the ranking, never invert it
    return HistGradientBoostingRegressor(
        max_iter=300, max_depth=4, learning_rate=0.06, l2_regularization=1.0,
        min_samples_leaf=50, monotonic_cst=cst, random_state=seed).fit(Xtr[:, :n_feat], ytr)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--vis-cache", default=None)
    ap.add_argument("--ir-cache", default=None, help="adds the two cross-modal features")
    ap.add_argument("--homography", default="runs/derived/homography_ir_to_vis.json")
    ap.add_argument("--out", default="runs/eval/rerank_loro.md")
    ap.add_argument("--model-out", default=None)
    ap.add_argument("--lambdas", type=float, nargs="+",
                    default=[0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0])
    ap.add_argument("--n-features", type=int, nargs="+", default=[4, 8, 18])
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    vp = Path(args.vis_cache) if args.vis_cache else Path(args.cache_dir) / "gauss_vis_paired_clean.pkl"
    vis, meta = load_records(ROOT / vp)
    if args.limit:
        vis = subsample(vis, args.limit)
    gts = gts_for(vis)
    runs, day, _n = day_night(vis)
    folds = loro_folds(runs, day)
    if len(folds) < 2:
        raise SystemExit(f"need >=2 day runs for leave-one-run-out, found {len(folds)}. "
                         f"Build the I0 substrate first (scripts/build_day_substrate.py).")

    ir_in_vis = None
    names = list(FEATURES)
    if args.ir_cache:
        ir = load_records(ROOT / args.ir_cache)[0]
        if args.limit:
            ir = subsample(ir, args.limit)
        H = json.loads((ROOT / args.homography).read_text(encoding="utf-8"))["runs"]
        hb = {k: np.asarray(v["H_ir_canvas_to_vis_canvas"], float) for k, v in H.items()}
        ir_in_vis = [{**r, "boxes_xyxy": apply_homography(
            np.asarray(r["boxes_xyxy"], float).reshape(-1, 4), hb.get(runs[i]))}
            for i, r in enumerate(ir)]
        names += XMODAL

    print(f"[fit] {len(vis)} frames | day {len(day)} | folds "
          f"{[(h, len(te)) for h, _tr, te in folds]}")
    X = [build_features(vis[i], ir_in_vis[i] if ir_in_vis else None) if i in set(day.tolist())
         else np.zeros((len(vis[i]["conf"]), len(names))) for i in range(len(vis))]
    Y = [build_target(vis[i], gts[i]) for i in range(len(vis))]
    parts = frame_parts(vis, gts)
    conf = [np.asarray(r["conf"], float) for r in vis]

    base = ap_from_scores(parts, day, conf)
    A, O, _pc, _rc = oracle_curves(parts, day)
    secs = [f"Source: `{vp}`" + (f" + `{args.ir_cache}`" if args.ir_cache else "")
            + f"  \nWeights: `{meta.get('weights')}`  \n"
            f"{len(day)} day frames over {len(folds)} runs "
            f"({', '.join(h + ':' + str(len(te)) for h, _t, te in folds)}).  \n"
            f"Baseline mAP50-95 **{fmt(base)}**; oracle re-rank ceiling "
            f"**{fmt(O.mean())}** (headroom {sgn(O.mean() - A.mean())})."]

    # ---- LORO out-of-fold predictions, per (n_features, monotone) ---------
    results = {}
    for n_feat in args.n_features:
        n_feat = min(n_feat, len(names))
        for monotone in (True, False):
            pred = [np.zeros(len(c)) for c in conf]
            for _held, tr, te in folds:
                Xtr = np.concatenate([X[i] for i in tr if len(X[i])]) if len(tr) else None
                ytr = np.concatenate([Y[i] for i in tr if len(Y[i])])
                if Xtr is None or len(Xtr) < 200:
                    continue
                m = fit_one(Xtr, ytr, n_feat, monotone)
                for i in te:
                    if len(X[i]):
                        pred[i] = m.predict(X[i][:, :n_feat])
            results[(n_feat, monotone)] = pred

    # ---- lambda sweep, scored out of fold --------------------------------
    rows = []
    best = (None, -1e9)
    for (n_feat, monotone), pred in sorted(results.items()):
        for lam in args.lambdas:
            sc = [np.clip(c, 1e-9, None) ** (1 - lam) * np.clip(p, 1e-6, None) ** lam
                  for c, p in zip(conf, pred)]
            oof = ap_from_scores(parts, day, sc)
            per_run = [ap_from_scores(parts, te, sc) - ap_from_scores(parts, te, conf)
                       for _h, _tr, te in folds]
            rows.append([n_feat, "yes" if monotone else "no", f"{lam:.2f}", fmt(oof),
                         sgn(oof - base), sgn(min(per_run)), sgn(max(per_run)),
                         f"{sum(1 for v in per_run if v > 0)}/{len(per_run)}"])
            if oof - base > best[1]:
                best = ((n_feat, monotone, lam), oof - base)
    secs.append("## 1. Leave-one-run-out lambda sweep\n\n"
                "Every frame is scored by a model that never saw its run. "
                "`score = conf^(1-lambda) * predIoU^lambda`; coordinates untouched. "
                "`runs won` is how many of the held-out runs improved -- an arm that "
                "wins overall on one run's strength has not generalised.\n\n"
                + md_table(["n_feat", "monotone", "lambda", "oof mAP50-95", "delta",
                            "worst run", "best run", "runs won"], rows))

    if best[0] is not None:
        n_feat, monotone, lam = best[0]
        secs.append(f"**Best out-of-fold arm:** {n_feat} features, "
                    f"monotone={'yes' if monotone else 'no'}, lambda={lam:.2f}, "
                    f"delta **{sgn(best[1])}**.")

    # ---- 2. what the in-fold number would have said (the C8 lesson) -------
    rows = []
    for n_feat in args.n_features:
        n_feat = min(n_feat, len(names))
        held, tr, te = folds[0]
        Xtr = np.concatenate([X[i] for i in tr if len(X[i])])
        ytr = np.concatenate([Y[i] for i in tr if len(Y[i])])
        m = fit_one(Xtr, ytr, n_feat, True)
        insample = [np.zeros(len(c)) for c in conf]
        for i in tr:
            if len(X[i]):
                insample[i] = m.predict(X[i][:, :n_feat])
        for i in te:
            if len(X[i]):
                insample[i] = m.predict(X[i][:, :n_feat])
        for lam in (0.25, 0.5):
            sc = [np.clip(c, 1e-9, None) ** (1 - lam) * np.clip(p, 1e-6, None) ** lam
                  for c, p in zip(conf, insample)]
            rows.append([n_feat, f"{lam:.2f}",
                         sgn(ap_from_scores(parts, tr, sc) - ap_from_scores(parts, tr, conf)),
                         sgn(ap_from_scores(parts, te, sc) - ap_from_scores(parts, te, conf))])
    secs.append("## 2. Fit-set gain vs held-out gain -- the C8 lesson, restated\n\n"
                f"One fold ({folds[0][0]} held out), reporting both columns. The 2026-09-01 "
                "pilot reported only the first and read +0.0419 as a result.\n\n"
                + md_table(["n_feat", "lambda", "delta on FIT runs", "delta on HELD-OUT run"],
                           rows))

    # ---- 3. feature importance, honestly caveated ------------------------
    Xall = np.concatenate([X[i] for i in day if len(X[i])])
    yall = np.concatenate([Y[i] for i in day if len(Y[i])])
    from scipy.stats import spearmanr
    rows = [[names[k], fmt(spearmanr(Xall[:, k], yall).statistic, 3)]
            for k in range(len(names))]
    rows.sort(key=lambda r: -abs(float(r[1])))
    secs.append("## 3. Per-feature rank correlation with IoU-with-GT\n\n"
                "In-sample and pooled across runs -- a screen for which features carry "
                "anything at all, NOT evidence that the model generalises. Section 1 is "
                "the only section that answers that.\n\n"
                + md_table(["feature", "spearman vs IoU"], rows))

    secs.append("## 4. Decision rule\n\n"
                "Adopt only if the best arm is **positive out of fold AND wins on every "
                "held-out run**. A positive mean with a losing run is the pilot's failure "
                "with a wider error bar. If nothing is positive out of fold at this "
                "substrate size, the next move is I0 (more day frames, `pohang04`), not a "
                "bigger model -- section 2 shows the signal is real and scene-specific.")
    secs.append(f"---\n\n_Generated by `scripts/fit_rerank.py` in {time.time() - t0:.1f}s._")
    write_md(args.out, "Learned IoU-aware re-ranker, leave-one-run-out (I1)", secs)

    if args.model_out and best[0] is not None:
        import pickle
        n_feat, monotone, lam = best[0]
        Xtr = np.concatenate([X[i] for i in day if len(X[i])])
        ytr = np.concatenate([Y[i] for i in day if len(Y[i])])
        p = ROOT / args.model_out
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "wb") as fh:
            pickle.dump({"model": fit_one(Xtr, ytr, n_feat, monotone),
                         "features": names[:n_feat], "lambda": lam,
                         "note": "fitted on ALL day runs; for deployment only, never for "
                                 "scoring the runs it was fitted on"}, fh)
        print(f"[out] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
