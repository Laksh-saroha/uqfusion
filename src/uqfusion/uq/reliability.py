"""Per-modality reliability score and fusion weights (O4 — scope §6.4 exactly).

Every constant is a pre-registered statistic of a clean validation set
(decision D5 / plan B5) — `fit_constants` IS that rule, executed:

    λ:    r_box = 0.9 at the clean-val median U_box  →  λ = -ln(0.9)/median(U)
    μ_d:  95th percentile of clean-val Mahalanobis distances
    τ:    clean-val IQR of the distances (floor-guarded)

so O ≈ 0 and r_box ≈ 1 on clean data BY CONSTRUCTION. The combination rule and
α are arguments, not baked in, so the plan B5-3/B5-4 ablations are one loop
over cached predictions.

σ ordering note: sigma_ltrb columns are (left, top, right, bottom) distances —
l/r normalize by box width, t/b by box height (scope §6.4's size-normalized u_i).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np

_EPS = 1e-9


@dataclass
class ReliabilityConstants:
    lam: float
    mu_d: float
    tau: float
    alpha: float = 1.0                    # temporal smoothing OFF by default (decision D14)
    combination: str = "multiplicative"   # multiplicative | min | geometric (plan B5-3)
    gamma: float = 0.5                    # geometric-mean exponent (only used when combination="geometric")
    # --- photometric term (TODO §0.2). None = DISABLED, reproducing §6.4 byte
    # for byte. Mahalanobis distance measures whether a frame is UNUSUAL; a dark
    # frame is not unusual, it is empty, and empty sits near the middle of the
    # feature distribution. Measured: pohang01 (real night, mAP 0.0000) scores
    # D=28.4 vs pohang00's 30.0 (daylight, mAP 0.4004) — the gate ranks the
    # blind frames as CLEANER. `r_bright` supplies the axis D cannot see.
    mu_b: float | None = None             # brightness at 50% clean-mAP retention
    tau_b: float | None = None            # logistic scale of that retention curve
    bright_stat: str = "mean"

    def to_dict(self) -> dict:
        return asdict(self)


def fit_constants(
    clean_u_box: np.ndarray,
    clean_distances: np.ndarray,
    alpha: float = 1.0,
    combination: str = "multiplicative",
    gamma: float = 0.5,
) -> ReliabilityConstants:
    """Execute the pre-registered fitting rules on clean-validation statistics."""
    clean_u_box = np.asarray(clean_u_box, dtype=np.float64)
    clean_distances = np.asarray(clean_distances, dtype=np.float64)
    if clean_u_box.size == 0 or clean_distances.size == 0:
        raise ValueError("fit_constants needs non-empty clean-validation U_box and distance samples")

    median_u = float(np.median(clean_u_box))
    lam = -np.log(0.9) / max(median_u, _EPS)

    mu_d = float(np.percentile(clean_distances, 95))
    q75, q25 = np.percentile(clean_distances, [75, 25])
    tau = max(float(q75 - q25), _EPS)

    return ReliabilityConstants(lam=lam, mu_d=mu_d, tau=tau, alpha=alpha, combination=combination, gamma=gamma)


def per_box_uncertainty(sigma_ltrb: np.ndarray, boxes_xyxy: np.ndarray) -> np.ndarray:
    """Size-normalized per-detection uncertainty u_i (scope §6.4)."""
    sigma_ltrb = np.asarray(sigma_ltrb, dtype=np.float64)
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    w = np.clip(boxes[:, 2] - boxes[:, 0], _EPS, None)
    h = np.clip(boxes[:, 3] - boxes[:, 1], _EPS, None)
    scale = np.stack([w, h, w, h], axis=1)  # l,t,r,b -> w,h,w,h
    return (sigma_ltrb / scale).mean(axis=1)


def compute_reliability(
    record: dict,
    frame_distance: float,
    constants: ReliabilityConstants,
    frame_brightness: float | None = None,
) -> dict:
    """Scope §6.4 for one frame of one modality, plus the optional §0.2 photometric term.

    record: UQPredictor output (boxes_xyxy, conf, sigma_ltrb, ...).
    Returns R plus every intermediate, so calibration analysis and the learned
    gate (plan B4) read from the same record.

    `frame_brightness` is the content-region statistic from `frame_brightness.py`
    (letterbox pad excluded). It is combined by MIN, not by product: the two
    signals answer different questions ("is this frame strange?" and "is this
    frame lit?") and either one firing is sufficient grounds to distrust the
    modality. A product would let a confident-looking D dilute a brightness
    alarm, which is the exact failure being fixed.
    """
    o = 1.0 / (1.0 + np.exp(-(frame_distance - constants.mu_d) / constants.tau))
    r_frame = 1.0 - float(o)

    r_bright = None
    if constants.mu_b is not None and frame_brightness is not None:
        tau_b = max(float(constants.tau_b or _EPS), _EPS)
        r_bright = float(1.0 / (1.0 + np.exp(-(float(frame_brightness) - constants.mu_b) / tau_b)))
        r_frame = min(r_frame, r_bright)

    n = len(record["boxes_xyxy"])
    if n == 0:
        # Empty-frame fallback (scope §6.4): clear-empty-sea vs fog-blind is
        # decided by the frame-level signal alone.
        return {"R": r_frame, "r_frame": r_frame, "r_box": None, "U_box": None, "O": float(o),
                "r_bright": r_bright, "n_dets": 0}

    u = per_box_uncertainty(record["sigma_ltrb"], record["boxes_xyxy"])
    c = np.asarray(record["conf"], dtype=np.float64)
    u_box = float((c * u).sum() / max(c.sum(), _EPS))
    r_box = float(np.exp(-constants.lam * u_box))

    if constants.combination == "multiplicative":
        r = r_frame * r_box
    elif constants.combination == "min":
        r = min(r_frame, r_box)
    elif constants.combination == "geometric":
        g = constants.gamma
        r = (max(r_frame, _EPS) ** g) * (max(r_box, _EPS) ** (1.0 - g))
    else:
        raise ValueError(f"unknown combination rule: {constants.combination}")

    return {"R": float(r), "r_frame": r_frame, "r_box": r_box, "U_box": u_box, "O": float(o),
            "r_bright": r_bright, "n_dets": n}


def smooth_reliability(r_now: float, r_prev: float | None, alpha: float) -> float:
    """Temporal EMA (scope §6.4); alpha=1 disables smoothing (decision D14 headline protocol)."""
    if r_prev is None or alpha >= 1.0:
        return r_now
    return alpha * r_now + (1.0 - alpha) * r_prev


def fusion_weights(r_vis: float, r_ir: float) -> dict:
    """Normalized WBF weights plus the ABSOLUTE system reliability (plan B3).

    R_sys is deliberately not normalized away: normalization is exactly what
    hides correlated both-degraded failure (scope §7.4). Callers threshold
    R_sys (validation-calibrated) to flag "no reliable modality".
    """
    total = max(r_vis + r_ir, _EPS)
    return {
        "w_vis": r_vis / total,
        "w_ir": r_ir / total,
        "R_sys": max(r_vis, r_ir),
        "R_sys_noisy_or": 1.0 - (1.0 - r_vis) * (1.0 - r_ir),
    }
