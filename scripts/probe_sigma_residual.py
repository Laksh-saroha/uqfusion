"""Idea I2 -- can sigma MOVE an edge, rather than score a box?

The Gaussian head predicts a per-edge sigma. Measured 2026-09-01, it is real:
pearson 0.38-0.64 against the true |edge error|, with mean sigma 1.1 px against
mean error 1.5-2.1 px -- right correlation AND right magnitude. But as a
box-level quality signal it is weak (spearman 0.162 against confidence's 0.366),
which is why F3's score-multiplier framing failed. Sigma predicts EDGES, not
BOXES.

Nothing in the system has ever used it to adjust a coordinate. That matters
because `probe_within_modality.py` shows re-ranking and merging are both flat,
while `probe_oracle_headroom.py` puts +0.1060 of headroom in the IoU 0.60-0.80
band -- i.e. in localisation.

**The screen is the point, and it can kill the idea in one run.** A correction
needs a DIRECTION. If only |error| is predictable and its sign is not, sigma
says how far the edge is wrong without saying which way, and there is nothing to
apply. Section 2 regresses the SIGNED residual out of fold; if its out-of-fold
R^2 is <= 0, stop here.

Section 3 applies the fitted correction end-to-end under leave-one-run-out and
reports the AP delta, so a positive screen is not mistaken for a positive result.

Usage:
    python scripts/probe_sigma_residual.py --cache-dir runs/cache_m
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import (ROOT, ap_of, day_night, fmt, gts_for, load_records,  # noqa: E402
                           loro_folds, md_table, sgn, subsample, write_md)

sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.matching import iou_matrix   # noqa: E402

EDGES = ("x1", "y1", "x2", "y2")


def matched(records, gts, sel, iou_thr=0.5):
    """Per detection matched to GT at `iou_thr`: box, sigma, conf, signed residual.

    Signed residual is `gt_edge - pred_edge`, i.e. the correction that WOULD fix
    the box. Matching is best-IoU same-class and is deliberately not greedy-1:1 --
    this measures the geometry of the error, not a detection metric.
    """
    B, S, C, R, F = [], [], [], [], []
    for i in sel:
        r = records[i]
        b = np.asarray(r["boxes_xyxy"], float).reshape(-1, 4)
        if not len(b):
            continue
        g = gts[i]
        if not len(g["boxes_xyxy"]):
            continue
        m = iou_matrix(b, g["boxes_xyxy"])
        m = np.where(np.asarray(r["cls"]).astype(int)[:, None] == g["cls"][None, :], m, 0.0)
        j = m.argmax(1)
        keep = m[np.arange(len(b)), j] >= iou_thr
        if not keep.any():
            continue
        B.append(b[keep])
        S.append(np.asarray(r["sigma_ltrb"], float).reshape(-1, 4)[keep])
        C.append(np.asarray(r["conf"], float)[keep])
        R.append(g["boxes_xyxy"][j[keep]] - b[keep])
        F.append(np.full(int(keep.sum()), i))
    if not B:
        return (np.zeros((0, 4)),) * 3 + (np.zeros(0), np.zeros(0, int))
    return (np.concatenate(B), np.concatenate(S), np.concatenate(R),
            np.concatenate(C), np.concatenate(F))


def design(B, S, C):
    """Features for the edge correction. Deliberately small: sigma, the box scale
    it must be read against, and confidence. A correction that needs more than
    this is fitting the scene, not the geometry."""
    w = np.clip(B[:, 2] - B[:, 0], 1e-6, None)
    h = np.clip(B[:, 3] - B[:, 1], 1e-6, None)
    return w, h, np.column_stack([np.log(np.clip(C, 1e-6, None)), np.log(w), np.log(h)])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--vis-cache", default=None)
    ap.add_argument("--out", default="runs/eval/sigma_residual.md")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    from scipy.stats import pearsonr, spearmanr
    from sklearn.linear_model import Ridge

    p = Path(args.vis_cache) if args.vis_cache else Path(args.cache_dir) / "gauss_vis_paired_clean.pkl"
    vis, meta = load_records(ROOT / p)
    if args.limit:
        vis = subsample(vis, args.limit)
    gts = gts_for(vis)
    runs, day, _night = day_night(vis)
    B, S, R, C, F = matched(vis, gts, day)
    print(f"[probe] {len(vis)} frames, {len(B)} matched detections")
    secs = [f"Source: `{p}`  \nWeights: `{meta.get('weights')}`  \n"
            f"{len(B)} detections matched to GT at IoU>=0.5 over {len(day)} day frames."]

    # ---- 1. is sigma calibrated at all? ----------------------------------
    rows = []
    for e in range(4):
        rows.append([EDGES[e],
                     fmt(pearsonr(S[:, e], np.abs(R[:, e])).statistic, 3),
                     fmt(spearmanr(S[:, e], np.abs(R[:, e])).statistic, 3),
                     fmt(S[:, e].mean(), 2), fmt(np.abs(R[:, e]).mean(), 2),
                     fmt(S[:, e].mean() / max(np.abs(R[:, e]).mean(), 1e-9), 2)])
    secs.append("## 1. Is sigma calibrated against the true edge error?\n\n"
                + md_table(["edge", "pearson |err|", "spearman |err|", "mean sigma px",
                            "mean |err| px", "ratio"], rows)
                + "\n\nA ratio near 1 means the head predicts the right MAGNITUDE, not "
                  "merely the right ordering.")

    # ---- 2. THE SCREEN: is the SIGNED residual predictable out of fold? ----
    w, h, X0 = design(B, S, C)
    rows, verdict = [], []
    for e in range(4):
        scale = w if e in (0, 2) else h
        X = np.column_stack([S[:, e], S[:, e] / scale, S.mean(1), X0])
        y = R[:, e]
        oof = np.full(len(y), np.nan)
        for _held, tr, te in loro_folds(runs, day):
            mtr = np.isin(F, tr)
            mte = np.isin(F, te)
            if mtr.sum() < 100 or mte.sum() < 10:
                continue
            oof[mte] = Ridge(alpha=1.0).fit(X[mtr], y[mtr]).predict(X[mte])
        ok = np.isfinite(oof)
        # Guard the degenerate case rather than reporting it as a perfect fit. With
        # too few folds nothing is predicted, `ok` is empty, and `1 - 0/1e-12` is
        # exactly 1.0 -- which the first preflight duly printed as R^2 = 1.0000 on
        # all four edges. An unmeasurable screen must read NaN, never PASS.
        if ok.sum() < 50:
            rows.append([EDGES[e], fmt(y.mean(), 3), fmt(y.std(), 3), "--", "--",
                         "unmeasured"])
            verdict.append(False)
            continue
        ss_res = float(((y[ok] - oof[ok]) ** 2).sum())
        ss_tot = float(((y[ok] - y[ok].mean()) ** 2).sum())
        if ss_tot < 1e-9:
            rows.append([EDGES[e], fmt(y.mean(), 3), fmt(y.std(), 3), "--", "--",
                         "no variance"])
            verdict.append(False)
            continue
        r2 = 1.0 - ss_res / ss_tot
        # In-fold R^2 on |err| for contrast: predicting the MAGNITUDE is easy and
        # is not what a correction needs.
        den_abs = float(((np.abs(y[ok]) - np.abs(y[ok]).mean()) ** 2).sum())
        r2_abs = (1.0 - ((np.abs(y[ok]) - np.abs(oof[ok])) ** 2).sum() / den_abs
                  if den_abs > 1e-9 else float("nan"))
        rows.append([EDGES[e], fmt(y.mean(), 3), fmt(y.std(), 3), fmt(r2, 4), fmt(r2_abs, 4),
                     "LIVE" if r2 > 0 else "dead"])
        verdict.append(r2 > 0)
    secs.append("## 2. THE SCREEN -- is the SIGNED residual predictable out of fold?\n\n"
                "Leave-one-run-out ridge on `[sigma_e, sigma_e/scale, mean sigma, log conf, "
                "log w, log h]`. Out-of-fold R^2 <= 0 means sigma knows how far the edge is "
                "wrong but not which way, and the correction has no direction to move in.\n\n"
                + md_table(["edge", "mean resid px", "sd resid px", "**oof R^2 (signed)**",
                            "oof R^2 (|resid|)", "verdict"], rows)
                + f"\n\n**Screen verdict: {'PASS' if any(verdict) else 'FAIL'}** "
                  f"({sum(verdict)}/4 edges predictable out of fold).")

    # ---- 3. apply it end to end, LORO ------------------------------------
    if any(verdict):
        base = ap_of(vis, gts, sel=day)
        rows = [["none (baseline)", fmt(base["map50_95"]), fmt(base["map50"]), "--"]]
        for shrink in (0.25, 0.5, 1.0):
            corrected = [dict(r) for r in vis]
            for _held, tr, te in loro_folds(runs, day):
                mtr = np.isin(F, tr)
                if mtr.sum() < 100:
                    continue
                models = []
                for e in range(4):
                    scale = w if e in (0, 2) else h
                    X = np.column_stack([S[:, e], S[:, e] / scale, S.mean(1), X0])
                    models.append(Ridge(alpha=1.0).fit(X[mtr], R[mtr, e]))
                for i in te:
                    r = vis[i]
                    b = np.asarray(r["boxes_xyxy"], float).reshape(-1, 4)
                    if not len(b):
                        continue
                    s = np.asarray(r["sigma_ltrb"], float).reshape(-1, 4)
                    c = np.asarray(r["conf"], float)
                    ww, hh, xx0 = design(b, s, c)
                    nb = b.copy()
                    for e in range(4):
                        sc = ww if e in (0, 2) else hh
                        Xi = np.column_stack([s[:, e], s[:, e] / sc, s.mean(1), xx0])
                        nb[:, e] = b[:, e] + shrink * models[e].predict(Xi)
                    # A correction must not invert a box.
                    nb[:, 2] = np.maximum(nb[:, 2], nb[:, 0] + 1e-3)
                    nb[:, 3] = np.maximum(nb[:, 3], nb[:, 1] + 1e-3)
                    corrected[i] = {**r, "boxes_xyxy": nb}
            a = ap_of(corrected, gts, sel=day)
            rows.append([f"ridge correction x{shrink:g}", fmt(a["map50_95"]), fmt(a["map50"]),
                         sgn(a["map50_95"] - base["map50_95"])])
        secs.append("## 3. Applied end to end, leave-one-run-out\n\n"
                    "Every frame is corrected by a model that never saw its run. "
                    "`shrink` damps the correction; 1.0 applies it in full.\n\n"
                    + md_table(["arm", "mAP50-95", "mAP50", "delta"], rows))
    else:
        secs.append("## 3. Applied end to end\n\nSkipped -- section 2 failed the screen. "
                    "Sigma predicts the magnitude of the edge error and not its sign, so "
                    "there is no correction to apply. **Idea I2 is closed.**")

    secs.append(f"---\n\n_Generated by `scripts/probe_sigma_residual.py` in "
                f"{time.time() - t0:.1f}s._")
    write_md(args.out, "Sigma as an edge correction (I2)", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
