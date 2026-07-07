"""Reliability-weighted decision-level fusion (O4 — scope §6.1/§6.4; WBF per R14).

Boxes from both modalities are fused in the VIS image plane (decision D7): IR
boxes are first mapped through a static IR→VIS homography derived from the
dataset calibration. Until the calibration files arrive (open question OQ-5),
callers may pass H=None (identity) — valid for the synthetic smoke tests where
both "modalities" share the image geometry.

Per-fused-box σ propagation through WBF clusters is a Phase 3 refinement
(cluster-weighted averaging of member σ) — the PoC fuses boxes/scores only.
"""

from __future__ import annotations

import numpy as np

_EPS = 1e-9


def apply_homography(boxes_xyxy: np.ndarray, h_matrix: np.ndarray | None) -> np.ndarray:
    """Map axis-aligned boxes through H (3x3); returns the axis-aligned envelope
    of the transformed corners. H=None is the identity."""
    boxes = np.asarray(boxes_xyxy, dtype=np.float64)
    if h_matrix is None or boxes.size == 0:
        return boxes
    h_matrix = np.asarray(h_matrix, dtype=np.float64)
    x1, y1, x2, y2 = boxes.T
    corners = np.stack(
        [np.stack([x1, y1], 1), np.stack([x2, y1], 1), np.stack([x2, y2], 1), np.stack([x1, y2], 1)], axis=1
    )  # (N, 4, 2)
    ones = np.ones((*corners.shape[:2], 1))
    projected = np.concatenate([corners, ones], axis=2) @ h_matrix.T  # (N, 4, 3)
    projected = projected[..., :2] / np.clip(projected[..., 2:3], _EPS, None)
    out = np.concatenate([projected.min(axis=1), projected.max(axis=1)], axis=1)
    return out


def fuse_detections(
    vis_record: dict,
    ir_record: dict,
    w_vis: float,
    w_ir: float,
    vis_hw: tuple[int, int],
    h_ir_to_vis: np.ndarray | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
) -> dict:
    """Weighted Boxes Fusion of two modality records in the VIS frame.

    Weights come from `reliability.fusion_weights` — the uncertainty gate. A
    modality with (near-)zero weight still contributes boxes, but WBF's
    weighted averaging and score scaling let the reliable stream dominate,
    which is the soft-fusion behavior scope §6.1 asks for.
    """
    from ensemble_boxes import weighted_boxes_fusion

    h, w = vis_hw
    norm = np.array([w, h, w, h], dtype=np.float64)

    vis_boxes = np.asarray(vis_record["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
    ir_boxes = apply_homography(np.asarray(ir_record["boxes_xyxy"], dtype=np.float64).reshape(-1, 4), h_ir_to_vis)

    boxes_list, scores_list, labels_list = [], [], []
    for boxes, record in ((vis_boxes, vis_record), (ir_boxes, ir_record)):
        b = np.clip(boxes / norm, 0.0, 1.0)
        boxes_list.append(b.tolist())
        scores_list.append(np.asarray(record["conf"], dtype=np.float64).tolist())
        labels_list.append(np.asarray(record["cls"], dtype=np.float64).tolist())

    fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
        boxes_list,
        scores_list,
        labels_list,
        weights=[max(w_vis, _EPS), max(w_ir, _EPS)],
        iou_thr=iou_thr,
        skip_box_thr=skip_box_thr,
    )
    return {
        "boxes_xyxy": np.asarray(fused_boxes) * norm,
        "conf": np.asarray(fused_scores),
        "cls": np.asarray(fused_labels).astype(int),
    }
