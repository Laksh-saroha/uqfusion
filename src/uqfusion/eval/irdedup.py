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
