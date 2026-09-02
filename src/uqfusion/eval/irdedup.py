"""Single-stream duplicate suppression, used on the IR stream by the `crossmodal` preset.

Kept out of `uq/fusion.py` deliberately: this is a DETECTOR-side post-process on one
stream, not a fusion operation, and conflating the two is what made the earlier
single-list-WBF artifact hard to see. It lifts the `ir_only` baseline by exactly as
much as it lifts the fused system.
"""

from __future__ import annotations

import numpy as np


def _iou(boxes: np.ndarray, b: np.ndarray) -> np.ndarray:
    xa = np.maximum(boxes[:, 0], b[0]); ya = np.maximum(boxes[:, 1], b[1])
    xb = np.minimum(boxes[:, 2], b[2]); yb = np.minimum(boxes[:, 3], b[3])
    inter = np.maximum(xb - xa, 0) * np.maximum(yb - ya, 0)
    aa = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def nms_record(rec: dict, thr: float) -> dict:
    """Greedy per-class NMS. Keeps the highest-scoring box of each cluster.

    `sigma_ltrb` travels with its box: `compute_reliability` indexes sigma against
    `boxes_xyxy`, so a record whose boxes were collapsed while sigma was not is a
    shape mismatch waiting to happen the first time a caller asks for `r_box`.
    """
    b = np.asarray(rec["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
    s = np.asarray(rec["conf"], dtype=np.float64).reshape(-1)
    c = np.asarray(rec["cls"]).reshape(-1)
    if len(s) < 2:
        return rec
    sg = np.asarray(rec.get("sigma_ltrb", np.zeros((len(s), 4))),
                    dtype=np.float64).reshape(-1, 4)
    ob, os_, oc, og = [], [], [], []
    for cl in np.unique(c):
        m = np.flatnonzero(c == cl)
        keep_b, keep_i = [], []
        for i in m[np.argsort(-s[m], kind="stable")]:
            if keep_b and _iou(np.asarray(keep_b), b[i]).max() > thr:
                continue
            keep_b.append(b[i].copy()); keep_i.append(i)
        ob += keep_b
        os_ += [float(s[i]) for i in keep_i]
        oc += [int(cl)] * len(keep_i)
        og += [sg[i] for i in keep_i]
    return {**rec, "boxes_xyxy": np.asarray(ob).reshape(-1, 4),
            "conf": np.asarray(os_), "cls": np.asarray(oc, dtype=int),
            "sigma_ltrb": np.asarray(og).reshape(-1, 4)}


def nms_records(records: list[dict], thr: float) -> list[dict]:
    return [nms_record(r, thr) for r in records]


def soft_nms_record(rec: dict, sigma_nms: float = 0.5, min_conf: float = 1e-4) -> dict:
    """Gaussian soft-NMS, per class. Decays a neighbour's score by
    `exp(-iou^2 / sigma_nms)` instead of deleting it.

    Measured on the VIS stream 2026-09-02 (`probe_within_modality.py`): +0.0025
    mAP50-95 [+0.0021, +0.0029], positive on all three held-out day runs. Hard NMS
    at 0.90 is +0.0016 -- smaller, because deleting a box removes its chance of
    being the one that matches, while decaying it only moves it down the ranking.

    Why this is not the merging family that keeps losing: no coordinate is ever
    combined. The surviving boxes are the detector's own, untouched; only the
    ORDER changes, which is the one surface the oracle probe says is still open.

    `sigma_ltrb` travels with its box for the reason `nms_record` documents --
    `compute_reliability` indexes sigma against `boxes_xyxy`.
    """
    b = np.asarray(rec["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
    s = np.asarray(rec["conf"], dtype=np.float64).reshape(-1).copy()
    c = np.asarray(rec["cls"]).reshape(-1)
    if len(s) < 2:
        return rec
    sg = np.asarray(rec.get("sigma_ltrb", np.zeros((len(s), 4))),
                    dtype=np.float64).reshape(-1, 4)
    keep: list[int] = []
    for cl in np.unique(c):
        order = np.flatnonzero(c == cl)
        order = order[np.argsort(-s[order], kind="stable")].tolist()
        while order:
            i = order.pop(0)
            keep.append(i)
            if not order:
                break
            iou = _iou(b[order], b[i])
            s[order] = s[order] * np.exp(-(iou ** 2) / sigma_nms)
            order = [o for o in order if s[o] > min_conf]
    k = np.asarray(sorted(keep), dtype=int)
    return {**rec, "boxes_xyxy": b[k], "conf": s[k], "cls": np.asarray(c)[k],
            "sigma_ltrb": sg[k]}


def soft_nms_records(records: list[dict], sigma_nms: float = 0.5) -> list[dict]:
    return [soft_nms_record(r, sigma_nms) for r in records]
