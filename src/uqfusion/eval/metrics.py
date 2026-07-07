"""Pre-registered calibration metrics (plan B6-5 — defined BEFORE any real run,
so there is no post-hoc metric shopping). Scope §12's classification-ECE is
made detection-appropriate as follows:

1. D-ECE (confidence calibration): detections binned by confidence into
   `n_bins` equal-width bins over [0,1]; per bin |mean confidence - precision|
   (precision = TP fraction at IoU>=0.5), weighted by bin count.

2. Regression calibration (per-edge, pooled over l,t,r,b):
   - z-score coverage: for TP detections, empirical P(|err| <= k·σ) vs the
     Gaussian nominal erf(k/√2) on a fixed k-grid; interval-ECE = mean |emp-nom|.
   - Gaussian NLL per edge: mean of 0.5·ln(2πσ²) + err²/(2σ²)   (Table 2 NLL).

3. Ranking quality:
   - Sparsification (AUSE): detections removed in 5% steps by DESCENDING
     uncertainty u; risk = mean(1 - realized IoU) over the retained set (FPs
     count as risk 1). Oracle removes by descending actual risk. AUSE = mean
     gap between the two curves.
   - AURC: area under the risk-coverage curve when retaining by ASCENDING u.

4. OOD separation: AUROC of the frame-level Mahalanobis distance, clean (=0)
   vs degraded (=1), plus histogram summaries (scope §12.3).

All functions consume flat arrays pooled across a frame set; `summarize_cache`
assembles a full Table 2 row from one prediction cache + labels.
"""

from __future__ import annotations

import math

import numpy as np

from uqfusion.eval.matching import load_gt, map50_95, match_image
from uqfusion.uq.reliability import per_box_uncertainty

COVERAGE_K = (0.5, 1.0, 1.5, 2.0, 2.5, 3.0)
_EPS = 1e-9


def d_ece(conf: np.ndarray, matched: np.ndarray, n_bins: int = 10) -> float:
    if conf.size == 0:
        return float("nan")
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges) - 1, 0, n_bins - 1)
    ece = 0.0
    for b in range(n_bins):
        m = idx == b
        if not m.any():
            continue
        ece += (m.sum() / conf.size) * abs(conf[m].mean() - matched[m].mean())
    return float(ece)


def coverage_interval_ece(err_edges: np.ndarray, sigma_edges: np.ndarray, ks=COVERAGE_K) -> dict:
    """TP-only per-edge coverage vs Gaussian nominal. err rows with NaN (FPs) are dropped."""
    ok = np.isfinite(err_edges).all(axis=1)
    err = np.abs(err_edges[ok]).ravel()
    sig = np.clip(sigma_edges[ok].ravel(), _EPS, None)
    if err.size == 0:
        return {"interval_ece": float("nan"), "coverage": {}}
    cov, gaps = {}, []
    for k in ks:
        nominal = math.erf(k / math.sqrt(2))
        empirical = float((err <= k * sig).mean())
        cov[k] = {"empirical": empirical, "nominal": nominal}
        gaps.append(abs(empirical - nominal))
    return {"interval_ece": float(np.mean(gaps)), "coverage": cov}


def gaussian_nll(err_edges: np.ndarray, sigma_edges: np.ndarray) -> float:
    ok = np.isfinite(err_edges).all(axis=1)
    err = err_edges[ok].ravel()
    sig = np.clip(sigma_edges[ok].ravel(), _EPS, None)
    if err.size == 0:
        return float("nan")
    return float((0.5 * np.log(2 * np.pi * sig**2) + err**2 / (2 * sig**2)).mean())


def sparsification(u: np.ndarray, risk: np.ndarray, steps: int = 20) -> dict:
    """AUSE + AURC. u = per-detection uncertainty; risk = per-detection 1-IoU (FP -> 1)."""
    if u.size == 0:
        return {"ause": float("nan"), "aurc": float("nan")}
    fracs = np.linspace(0, 0.95, steps)
    by_u = np.argsort(-u)       # remove most-uncertain first
    by_risk = np.argsort(-risk)  # oracle
    curve_u, curve_o = [], []
    for f in fracs:
        keep = int(round((1 - f) * u.size))
        keep = max(keep, 1)
        curve_u.append(risk[by_u][-keep:].mean() if keep else 0.0)
        curve_o.append(risk[by_risk][-keep:].mean() if keep else 0.0)
    curve_u, curve_o = np.asarray(curve_u), np.asarray(curve_o)
    # AURC over coverage = 1 - frac, retention by ascending u (same ordering)
    return {"ause": float((curve_u - curve_o).mean()), "aurc": float(curve_u.mean()),
            "fractions": fracs.tolist(), "risk_by_u": curve_u.tolist(), "risk_oracle": curve_o.tolist()}


def ood_auroc(d_clean: np.ndarray, d_degraded: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.zeros(len(d_clean)), np.ones(len(d_degraded))])
    return float(roc_auc_score(y, np.concatenate([d_clean, d_degraded])))


def summarize_cache(records: list[dict], sigma_key: str = "sigma_ltrb", iou_match: float = 0.5) -> dict:
    """One Table 2 row from one cache: D-ECE, NLL, interval-ECE, AUSE, AURC, mAP.

    sigma_key selects the uncertainty source: "sigma_ltrb" (trained Gaussian or
    cluster std) or "dfl_sigma_ltrb" (§7.2 DFL-derived row, Gaussian caches only).
    """
    confs, matcheds, errs, sigmas, us, risks = [], [], [], [], [], []
    gts = []
    for rec in records:
        gt = load_gt(rec["image_path"], rec["image_hw"])
        gts.append(gt)
        m = match_image(rec, gt, iou_thr=iou_match)
        n = len(rec["conf"])
        if n == 0:
            continue
        sigma = rec[sigma_key]
        confs.append(rec["conf"])
        matcheds.append(m["matched"])
        errs.append(m["err_edges"])
        sigmas.append(sigma)
        us.append(per_box_uncertainty(sigma, rec["boxes_xyxy"]))
        risks.append(1.0 - m["iou_realized"])

    if not confs:
        return {"error": "no detections in cache"}
    conf = np.concatenate(confs)
    matched = np.concatenate(matcheds)
    err = np.concatenate(errs)
    sigma = np.concatenate(sigmas)
    u = np.concatenate(us)
    risk = np.concatenate(risks)

    out = {
        "n_detections": int(conf.size),
        "n_frames": len(records),
        "d_ece": d_ece(conf, matched),
        "nll": gaussian_nll(err, sigma),
        **{k: v for k, v in coverage_interval_ece(err, sigma).items()},
        **{k: v for k, v in sparsification(u, risk).items() if k in ("ause", "aurc")},
        **map50_95(records, gts),
    }
    return out
