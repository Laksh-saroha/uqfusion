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

SORT_KIND = "stable"
"""Tie semantics for every descending-confidence sort in this module (R-A1/R-A2).

`np.argsort` defaults to an unstable introsort, so detections sharing a confidence
were ordered arbitrarily -- and both the greedy match and the cumulative TP/FP run
down that order. `apmetrics.SORT_KIND` documents the measured size (0.0067 mAP on a
tie-dense synthetic fixture, 2.7e-7 on a real cache, where 99.9% of confidences are
unique). Pinned here too so `cocoparity.TASK_CONFIG`'s "stable score sort" is true of
the local evaluator as well -- pycocotools sorts with mergesort, which is stable, so
the two now agree on tie handling by construction rather than by luck."""


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
        for i in np.argsort(-conf, kind=SORT_KIND):
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
    order = np.argsort(-record["conf"], kind=SORT_KIND)
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


def local_ap50_95(records: list[dict], gts: list[dict]) -> dict:
    """This project's OWN mAP@50-95 and mAP@50 over a frame set.

    Renamed from `map50_95` on 2026-09-10 (R-A1). The old name is kept as a working
    alias below, and the returned dict keys are unchanged, so no call site and no
    recorded number moves.

    **This is not COCO AP.** The docstring said "COCO-style" for as long as the
    function existed, and that was a mislabel of exactly the kind
    `docs/exposure-ledger-2026-09-09.md` section 6 documents for
    `preset="crossmodal"`: a name that does not pin a computation. The precision
    envelope here is LINEARLY INTERPOLATED onto a 101-point recall grid
    (`np.interp`); COCO looks it up at the first attained recall (`searchsorted`).
    Ultralytics is a third rule again. See `apmetrics.AP_CONVENTION` for the full
    statement and `uqfusion.eval.cocoparity` for the COCO implementation.

    The measured delta gap is 0.00028501, 5x below the paired noise floor, so paired
    comparisons are safe; ABSOLUTE numbers are not interchangeable across conventions
    (`docs/ap-convention-rule-2026-09-10.md`).

    Also returns `per_class` — `{cls: {ap50_95, ap50, n_gt, n_pred}}`, the same
    shape `apmetrics.ap_from_parts` returns, pinned equal by `smoke_apmetrics.py`.

    **Read the per-class rows before believing any macro delta.** `map50_95` is
    macro-averaged over ship and buoy, and the two classes are not symmetric here:
    IR is nc=1 ship-only (D28/A-1 — IR buoy AP was 0.00019 against 29,131 buoy
    detections for 596 GT boxes), so buoys can only ever come from VIS and fusion
    acts on the ship half alone. A macro delta is therefore
    `(delta_ship + delta_buoy) / 2` — a mixture that answers no single question. It
    dilutes a large ship gain, inflates a small one when VIS buoy AP is high, and
    on a frame where the photometric veto drops VIS from the merge the fused output
    structurally cannot contain buoys at all: `n_gt_per_class` counts them from the
    ground truth regardless, that class scores AP 0, and the macro is roughly halved
    for a class the system was never able to emit. Properties of the metric, not of
    the gate.

    Measured instance (`eval_final_system.py`, clean/day, gated vs `visible_only`):
    macro +0.0006 CI [-0.0013, +0.0027], spanning zero; ship AP +0.0031 CI
    [+0.0015, +0.0052], not spanning zero. Same frames, opposite conclusion.
    """
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
    per_class: dict[int, dict] = {}
    for ci, (c, n_gt) in enumerate(sorted(n_gt_per_class.items())):
        mask = cls == c
        if not mask.any() or n_gt == 0:
            # Still recorded, at AP 0: this is exactly the case the docstring warns
            # about (a class present in GT that the system emitted nothing for), and
            # it is invisible in the macro mean. It stays in `ap`, so the macro value
            # is unchanged from before this breakdown existed.
            per_class[int(c)] = {"ap50_95": 0.0, "ap50": 0.0,
                                 "n_gt": int(n_gt), "n_pred": int(mask.sum())}
            continue
        order = np.argsort(-conf[mask], kind=SORT_KIND)
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
        per_class[int(c)] = {"ap50_95": float(ap[ci].mean()), "ap50": float(ap[ci, 0]),
                             "n_gt": int(n_gt), "n_pred": int(mask.sum())}

    if ap.size == 0:
        return {"map50_95": 0.0, "map50": 0.0, "per_class": {}}
    return {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean()),
            "per_class": per_class}


# Backwards-compatible alias. 322 call sites across 70 files use this name, and the
# dict keys it returns (`map50_95`, `map50`) are unchanged, so nothing breaks and no
# recorded number moves. New code should prefer `local_ap50_95`, which does not claim
# to be COCO. Deliberately NOT a deprecation warning: 70 files emitting one on every
# eval would be noise, and the name is not wrong, only imprecise.
map50_95 = local_ap50_95
