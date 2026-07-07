"""Cross-pass/member detection clustering — THE fixed matching protocol (plan B6-4).

MC-Dropout passes and ensemble members produce T (or M) detection sets per
frame; box variance requires deciding which detections are "the same object".
Calibration metrics are sensitive to this choice, so it is defined ONCE here,
shared by both methods, and documented as method, not a knob:

  1. Pool all detections with a source id; process in descending confidence.
  2. An unused detection seeds a cluster; from every OTHER source, the highest-
     IoU unused detection of the SAME class with IoU >= `iou_thr` (default 0.55,
     matching the WBF threshold used for fusion) joins — at most one per source.
  3. Cluster outputs: confidence-weighted mean box; per-coordinate std over
     members (ddof=0) as sigma_ltrb (x1,y1,x2,y2 edge stds == LTRB edge stds);
     confidence = mean(member conf) x (support / n_sources) — the standard
     "missed by most passes -> low confidence" scaling;
  4. Singleton clusters (support 1) carry no disagreement evidence: their σ is
     set to the box size (w,h,w,h) — i.e. maximal size-normalized uncertainty
     of ~1 — a pre-registered convention, disclosed rather than silently NaN.
"""

from __future__ import annotations

import numpy as np


def cluster_records(records: list[dict], iou_thr: float = 0.55) -> dict:
    """Merge per-source records (PlainPredictor schema) into one record with
    disagreement-based sigma_ltrb and n_support."""
    n_sources = len(records)
    boxes = np.concatenate([r["boxes_xyxy"].reshape(-1, 4) for r in records], axis=0)
    confs = np.concatenate([r["conf"] for r in records], axis=0)
    clss = np.concatenate([r["cls"] for r in records], axis=0)
    srcs = np.concatenate([np.full(len(r["conf"]), i) for i, r in enumerate(records)], axis=0)
    image_hw = records[0]["image_hw"]

    # frame feature: mean over the sources that captured one (MC passes all do;
    # ensemble deliberately captures member-0 only — one consistent feature space)
    feats = [r["feat"] for r in records if "feat" in r]

    empty = {
        "boxes_xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "cls": np.zeros(0, dtype=int),
        "sigma_ltrb": np.zeros((0, 4)), "n_support": np.zeros(0, dtype=int), "image_hw": image_hw,
    }
    if feats:
        empty["feat"] = np.mean(feats, axis=0)
    if boxes.shape[0] == 0:
        return empty

    order = np.argsort(-confs)
    used = np.zeros(len(confs), dtype=bool)
    out_boxes, out_conf, out_cls, out_sigma, out_support = [], [], [], [], []

    for seed in order:
        if used[seed]:
            continue
        used[seed] = True
        members = [seed]
        for src in range(n_sources):
            if src == srcs[seed]:
                continue
            cand = np.where((~used) & (srcs == src) & (clss == clss[seed]))[0]
            if cand.size == 0:
                continue
            ious = _iou_one_to_many(boxes[seed], boxes[cand])
            best = int(np.argmax(ious))
            if ious[best] >= iou_thr:
                used[cand[best]] = True
                members.append(int(cand[best]))

        m_boxes, m_confs = boxes[members], confs[members]
        w = m_confs / max(m_confs.sum(), 1e-9)
        mean_box = (m_boxes * w[:, None]).sum(axis=0)
        if len(members) >= 2:
            sigma = m_boxes.std(axis=0)
        else:
            bw = max(mean_box[2] - mean_box[0], 1e-6)
            bh = max(mean_box[3] - mean_box[1], 1e-6)
            sigma = np.array([bw, bh, bw, bh])  # pre-registered singleton convention
        out_boxes.append(mean_box)
        out_conf.append(m_confs.mean() * len(members) / n_sources)
        out_cls.append(int(clss[seed]))
        out_sigma.append(sigma)
        out_support.append(len(members))

    order2 = np.argsort(-np.asarray(out_conf))
    result = {
        "boxes_xyxy": np.asarray(out_boxes)[order2],
        "conf": np.asarray(out_conf)[order2],
        "cls": np.asarray(out_cls, dtype=int)[order2],
        "sigma_ltrb": np.asarray(out_sigma)[order2],
        "n_support": np.asarray(out_support, dtype=int)[order2],
        "image_hw": image_hw,
    }
    if feats:
        result["feat"] = np.mean(feats, axis=0)
    return result


def _iou_one_to_many(box: np.ndarray, boxes: np.ndarray) -> np.ndarray:
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    a = (box[2] - box[0]) * (box[3] - box[1])
    b = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    return inter / np.clip(a + b - inter, 1e-9, None)
