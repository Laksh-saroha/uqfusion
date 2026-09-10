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

# ------------------------------------------------------------ declared contracts
# R-A4 (docs/TODO-2026-09-09-architecture-review.md, finding F12). Every constant
# below was implicit, and each names a scope the metric's SHORT NAME oversells.
# Same discipline as R-A2's MISSING_CLASS_POLICY / SORT_KIND and R-A1's
# AP_CONVENTION: record what the number actually is, because the name is not enough.

DECE_CONDITIONING = "confidence-only"
"""What `d_ece` conditions on. It is NOT a multidimensional calibration measure.

Detections are binned by CONFIDENCE alone and each bin's mean confidence is compared
against its TP fraction at IoU >= 0.5. That answers "when this detector says 0.7, is it
right 70% of the time?" and nothing else.

It does **not** measure whether calibration holds conditional on object location, box
scale, class, or scene condition -- a detector perfectly calibrated in aggregate can be
badly miscalibrated on small or peripheral objects and score a D-ECE of 0. Quoting
D-ECE as "calibration error" without the qualifier claims the stronger property.
"""

UQ_SUBSET = "true-positives-only"
"""Which detections `gaussian_nll` and `coverage_interval_ece` are computed over.

Both drop rows where `err_edges` is not finite, which is every FALSE POSITIVE -- an
unmatched detection has no ground-truth edge to have erred against. So both metrics
describe the TP subset, and **that subset differs between detectors**: a
higher-recall arm admits harder, later matches its rival never made, and is scored on
a different population.

Measured on `sigma_vis_seed0_nightfull`: **13,779 of 31,110 detections (44.3%)** enter
NLL and interval-ECE. Comparing two arms' NLL is therefore not a like-for-like
comparison unless their TP counts and recall are shown beside it. `summarize_cache`
now reports `n_tp_detections`, `tp_share`, `n_gt` and `recall_tp_over_gt` for exactly
this reason -- R-A4 asks for the denominator to be published, not assumed equal.
"""

AURC_GRID = "20 points, linspace(0, 0.95), integer-rounded keep counts"
AURC_INTEGRATION = "grid-mean"
"""How `sparsification` turns a risk-coverage curve into a scalar. Not an integral.

Three separate departures from "area under the risk-coverage curve":

1. **It is a plain mean of 20 grid points, not a trapezoidal integral.** On
   `sigma_vis_seed0_nightfull` the mean is **0.4397** and the trapezoidal integral over
   the same points is **0.4182** -- a gap of 0.0215, roughly 7-15x the 0.0014-0.0031
   paired noise floor. The two are not interchangeable.
2. **Coverage never reaches 0.** `fracs` stops at 0.95, so the curve covers retention
   1.00 down to 0.05 and the deep-rejection tail is simply absent.
3. **Keep counts are integer-rounded**, so the realised coverage grid is not exactly
   `1 - frac`; on small subsets the rounding is visible.

None of this makes the number useless -- as a RANKING statistic between arms measured
the same way it is fine, which is how this project uses it. It does mean the value must
not be compared against an AURC from any other source, and must not be described as an
integral. `sparsification` now also returns `aurc_trapz` so the difference is visible
rather than inferred; the `aurc` key keeps its exact historical value.

Note there is a SECOND, unrelated `aurc()` in `scripts/eval_risk_coverage_fixed_gt.py`
which really is trapezoidal, over a different quantity (frame-level mAP risk, not
per-detection 1-IoU). One name, two computations -- the F14 pattern again.
"""

RANKING_METRICS = ("ause", "aurc", "aurc_trapz")
"""Metrics that depend on the ORDER of the uncertainties, never their magnitude.

`sparsification` sorts by `u` and reads off risk. Any strictly monotone rescaling of
every uncertainty leaves these bit-identical while destroying every magnitude-sensitive
property of the same sigmas.

Verified, not asserted -- multiplying every sigma by 100 on
`sigma_vis_seed0_nightfull`:

* `ause` 0.0747863360 -> 0.0747863360  (**bit-identical**)
* `aurc` 0.4397123625 -> 0.4397123625  (**bit-identical**)
* `nll` 4.201611 -> 5.208694, `interval_ece` 0.172533 -> 0.188100

So a good AUSE is evidence that the uncertainty ORDERS errors well. It is **not**
evidence that the sigmas are the right size, and an arm can win on AUSE while its
intervals cover nothing. Read AUSE/AURC beside NLL and interval-ECE, never instead.
"""


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
    """TP-only per-edge coverage vs Gaussian nominal. err rows with NaN (FPs) are dropped.

    Unlike `gaussian_nll` this KEEPS the `_EPS` clip and stays finite, because at
    sigma = 0 a zero-width interval that contains nothing is the correct answer
    rather than a disguised infinity: an infinitely confident prediction that is
    wrong should read as total under-coverage, and a coverage share is bounded by
    1 either way. Measured sensitivity to the clip is in `runs/eval/nll_floor.md`
    -- interval-ECE moves by <0.002 across four decades of floor.
    """
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


def gaussian_nll(err_edges: np.ndarray, sigma_edges: np.ndarray,
                 sigma_floor: float | None = None) -> float:
    """Mean Gaussian NLL over TP edges. **NaN when any sigma is not positive.**

    This used to clip sigma at `_EPS` and return a finite number regardless. That
    is not a repair, it is a disguise. `cluster_records` sets a cluster's sigma to
    the per-coordinate std over its members (ddof=0), so two members agreeing to
    the last float give sigma **exactly 0**, where this quantity is +inf for any
    non-zero error. Clipping rendered that infinity as ~1e17 per edge, and the
    reported mean then measured the clip constant rather than the model: 13 of
    34,776 VIS MC edges (0.037%) carried 100.0000% of a 1.891e16 NLL, and moving
    `_EPS` to 1e-12 would have moved every MC/ensemble NLL in this project by six
    orders of magnitude without a single weight changing. See
    `runs/eval/nll_floor.md`.

    So the metric now declines instead. NaN propagates to NO-SIGNAL in
    `slice_uq_day_night.py`, which is the honest reading: the mean of a set
    containing +inf is undefined, and no scalar summarises it.

    `sigma_floor` (in pixels) opts back in to a finite number, and the caller then
    owns the disclosure. Note from the ladder in `runs/eval/nll_floor.md` that NO
    floor is neutral: 53% of VIS sigma-head edges sit below 0.5 px, so a 0.5 px
    floor rewrites the sigma-head's own NLL from 5.987 to 4.336. A floor is a
    change to the metric for every arm, not a patch for the broken ones -- which
    is why there is no default.
    """
    ok = np.isfinite(err_edges).all(axis=1)
    err = err_edges[ok].ravel()
    sig = np.asarray(sigma_edges, dtype=float)[ok].ravel()
    if err.size == 0:
        return float("nan")
    if sigma_floor is not None:
        sig = np.clip(sig, float(sigma_floor), None)
    if not np.all(sig > 0.0):
        return float("nan")
    return float((0.5 * np.log(2 * np.pi * sig**2) + err**2 / (2 * sig**2)).mean())


def sparsification(u: np.ndarray, risk: np.ndarray, steps: int = 20) -> dict:
    """AUSE + AURC. u = per-detection uncertainty; risk = per-detection 1-IoU (FP -> 1)."""
    if u.size == 0:
        return {"ause": float("nan"), "aurc": float("nan")}
    fracs = np.linspace(0, 0.95, steps)
    # R-A2's tie argument applies here too: an unstable sort makes a tie between two
    # uncertainties resolve arbitrarily, and the curve is read off that order. Measured
    # effect on a real cache is ZERO (u is 99.82% unique on sigma_vis_seed0_nightfull,
    # and a permuted input reproduces ause/aurc bit-for-bit), so this is pinned for
    # reproducibility rather than because it was biting.
    by_u = np.argsort(-u, kind="stable")       # remove most-uncertain first
    by_risk = np.argsort(-risk, kind="stable")  # oracle
    curve_u, curve_o, cov = [], [], []
    for f in fracs:
        keep = int(round((1 - f) * u.size))
        keep = max(keep, 1)
        # the REALISED coverage, which integer rounding makes != (1 - f)
        cov.append(keep / u.size)
        curve_u.append(risk[by_u][-keep:].mean() if keep else 0.0)
        curve_o.append(risk[by_risk][-keep:].mean() if keep else 0.0)
    curve_u, curve_o = np.asarray(curve_u), np.asarray(curve_o)
    cov = np.asarray(cov)
    # `aurc` is a plain GRID MEAN and keeps its exact historical value so no recorded
    # number moves (see AURC_INTEGRATION). `aurc_trapz` is the integral the name
    # implies, over the realised coverage grid, reported alongside so the gap is
    # visible instead of inferred. They differ by ~0.02 on real data.
    o = np.argsort(cov)
    trapz = getattr(np, "trapezoid", np.trapz)
    return {"ause": float((curve_u - curve_o).mean()), "aurc": float(curve_u.mean()),
            "aurc_grid_mean": float(curve_u.mean()),
            "aurc_trapz": float(trapz(curve_u[o], cov[o])),
            "aurc_integration": AURC_INTEGRATION, "aurc_grid": AURC_GRID,
            "coverage_min": float(cov.min()), "coverage_max": float(cov.max()),
            "fractions": fracs.tolist(), "coverage": cov.tolist(),
            "risk_by_u": curve_u.tolist(), "risk_oracle": curve_o.tolist()}


def ood_auroc(d_clean: np.ndarray, d_degraded: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    y = np.concatenate([np.zeros(len(d_clean)), np.ones(len(d_degraded))])
    return float(roc_auc_score(y, np.concatenate([d_clean, d_degraded])))


def summarize_cache(records: list[dict], sigma_key: str = "sigma_ltrb", iou_match: float = 0.5,
                    sigma_floor: float | None = None) -> dict:
    """One Table 2 row from one cache: D-ECE, NLL, interval-ECE, AUSE, AURC, mAP.

    sigma_key selects the uncertainty source: "sigma_ltrb" (trained Gaussian or
    cluster std) or "dfl_sigma_ltrb" (§7.2 DFL-derived row, Gaussian caches only).
    """
    confs, matcheds, errs, sigmas, us, risks = [], [], [], [], [], []
    gts = []
    n_gt = 0
    for rec in records:
        gt = load_gt(rec["image_path"], rec["image_hw"])
        gts.append(gt)
        n_gt += len(gt["cls"])
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

    # R-A4/F12: NLL and interval-ECE are TP-ONLY (see UQ_SUBSET), and different
    # detectors reach different TP subsets, so the two numbers are not comparable
    # without these. Published beside every UQ number rather than left to be assumed.
    tp = np.isfinite(err).all(axis=1)
    out = {
        "n_detections": int(conf.size),
        "n_frames": len(records),
        "n_gt": int(n_gt),
        "n_tp_detections": int(tp.sum()),
        "tp_share": float(tp.mean()) if tp.size else float("nan"),
        "recall_tp_over_gt": float(tp.sum() / n_gt) if n_gt else float("nan"),
        "uq_subset": UQ_SUBSET,
        "dece_conditioning": DECE_CONDITIONING,
        # scores outside [0,1] cannot be read as probabilities and silently pile into
        # d_ece's top bin (`np.digitize` then `np.clip`). Zero on every UQ cache today;
        # FUSED output reaches 1.75, so this reports rather than assumes.
        "conf_out_of_unit_range": int(((conf < 0.0) | (conf > 1.0)).sum()),
        "d_ece": d_ece(conf, matched),
        "nll": gaussian_nll(err, sigma, sigma_floor=sigma_floor),
        "nll_sigma_floor": sigma_floor,
        "sigma_nonpositive_share": float(np.mean(
            np.asarray(sigma, dtype=float)[np.isfinite(err).all(axis=1)] <= 0.0)),
        **{k: v for k, v in coverage_interval_ece(err, sigma).items()},
        **{k: v for k, v in sparsification(u, risk).items()
           if k in ("ause", "aurc", "aurc_trapz", "aurc_integration")},
        **map50_95(records, gts),
    }
    return out
