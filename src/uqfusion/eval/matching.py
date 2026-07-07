"""GT loading, prediction-GT matching, and mAP — shared by all evaluations.

Matching rule (pre-registered): per image, predictions sorted by descending
confidence greedily claim the highest-IoU unmatched GT of the same class with
IoU >= threshold. TP/FP at IoU 0.5 feeds the calibration metrics; the 10-level
matrix (0.50:0.95:0.05) feeds mAP. AP is COCO-style 101-point interpolation
per class per level; mAP@50-95 averages over classes and levels. Implemented
locally (no pycocotools dependency) and deterministic.

Label convention: Ultralytics — label file = image path with /images/ ->
/labels/ and suffix .txt; rows `cls cx cy w h` normalized.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

IOU_LEVELS = np.round(np.arange(0.5, 1.0, 0.05), 2)  # 0.50 ... 0.95


def label_path_for(image_path: str | Path) -> Path:
    p = Path(image_path)
    parts = [("labels" if part == "images" else part) for part in p.parts]
    return Path(*parts).with_suffix(".txt")


def load_gt(image_path: str | Path, image_hw: tuple[int, int]) -> dict:
    """GT boxes (xyxy px) + classes for one image; empty arrays when no label file."""
    h, w = image_hw
    lp = label_path_for(image_path)
    boxes, clss = [], []
    if lp.is_file():
        for line in lp.read_text(encoding="utf-8").splitlines():
            vals = line.split()
            if len(vals) < 5:
                continue
            c, cx, cy, bw, bh = int(vals[0]), *map(float, vals[1:5])
            boxes.append([(cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h])
            clss.append(c)
    return {"boxes_xyxy": np.asarray(boxes, dtype=np.float64).reshape(-1, 4),
            "cls": np.asarray(clss, dtype=int)}


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if a.size == 0 or b.size == 0:
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area_a = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    area_b = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / np.clip(area_a[:, None] + area_b[None, :] - inter, 1e-9, None)


def match_image(record: dict, gt: dict, iou_thr: float = 0.5) -> dict:
    """Per-detection match info at one threshold (calibration metrics input).

    Returns matched (bool), iou_realized (0 for FP), err_edges (N,4; NaN for FP —
    signed pred-gt per xyxy edge), n_gt.
    """
    pred_boxes, pred_cls, conf = record["boxes_xyxy"], record["cls"], record["conf"]
    n = len(conf)
    matched = np.zeros(n, dtype=bool)
    iou_real = np.zeros(n)
    err = np.full((n, 4), np.nan)
    if n:
        ious = iou_matrix(pred_boxes, gt["boxes_xyxy"])
        same = pred_cls[:, None] == gt["cls"][None, :]
        ious = ious * same
        gt_used = np.zeros(len(gt["cls"]), dtype=bool)
        for i in np.argsort(-conf):
            if ious.shape[1] == 0:
                break
            j = int(np.argmax(np.where(gt_used, -1.0, ious[i])))
            if not gt_used[j] and ious[i, j] >= iou_thr:
                gt_used[j] = True
                matched[i] = True
                iou_real[i] = ious[i, j]
                err[i] = pred_boxes[i] - gt["boxes_xyxy"][j]
    return {"matched": matched, "iou_realized": iou_real, "err_edges": err, "n_gt": len(gt["cls"])}


def tp_matrix(record: dict, gt: dict, levels: np.ndarray = IOU_LEVELS) -> np.ndarray:
    """(N, len(levels)) TP flags — greedy match repeated per IoU level (mAP input)."""
    n = len(record["conf"])
    tp = np.zeros((n, len(levels)), dtype=bool)
    if n == 0:
        return tp
    ious = iou_matrix(record["boxes_xyxy"], gt["boxes_xyxy"])
    same = record["cls"][:, None] == gt["cls"][None, :]
    ious = ious * same
    order = np.argsort(-record["conf"])
    for k, thr in enumerate(levels):
        gt_used = np.zeros(len(gt["cls"]), dtype=bool)
        for i in order:
            if ious.shape[1] == 0:
                break
            j = int(np.argmax(np.where(gt_used, -1.0, ious[i])))
            if not gt_used[j] and ious[i, j] >= thr:
                gt_used[j] = True
                tp[i, k] = True
    return tp


def map50_95(records: list[dict], gts: list[dict]) -> dict:
    """COCO-style mAP@50-95 and mAP@50 over a frame set (101-point interpolation)."""
    tps, confs, pcls = [], [], []
    n_gt_per_class: dict[int, int] = {}
    for rec, gt in zip(records, gts):
        tps.append(tp_matrix(rec, gt))
        confs.append(rec["conf"])
        pcls.append(rec["cls"])
        for c in gt["cls"]:
            n_gt_per_class[int(c)] = n_gt_per_class.get(int(c), 0) + 1

    tp = np.concatenate(tps) if tps else np.zeros((0, len(IOU_LEVELS)))
    conf = np.concatenate(confs) if confs else np.zeros(0)
    cls = np.concatenate(pcls) if pcls else np.zeros(0, dtype=int)

    recall_grid = np.linspace(0, 1, 101)
    ap = np.zeros((len(n_gt_per_class), len(IOU_LEVELS)))
    for ci, (c, n_gt) in enumerate(sorted(n_gt_per_class.items())):
        mask = cls == c
        if not mask.any() or n_gt == 0:
            continue
        order = np.argsort(-conf[mask])
        tpc = tp[mask][order]
        fpc = ~tpc
        cum_tp = np.cumsum(tpc, axis=0)
        cum_fp = np.cumsum(fpc, axis=0)
        recall = cum_tp / n_gt
        precision = cum_tp / np.clip(cum_tp + cum_fp, 1e-9, None)
        for k in range(len(IOU_LEVELS)):
            prec = precision[:, k].copy()
            for i in range(len(prec) - 2, -1, -1):  # precision envelope
                prec[i] = max(prec[i], prec[i + 1])
            ap[ci, k] = np.interp(recall_grid, recall[:, k], prec, left=prec[0] if len(prec) else 0, right=0).mean()

    if ap.size == 0:
        return {"map50_95": 0.0, "map50": 0.0}
    return {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean())}
