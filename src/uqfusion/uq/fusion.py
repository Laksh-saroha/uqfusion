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


def _bb_iou(boxes: np.ndarray, new_box: np.ndarray) -> np.ndarray:
    """IoU of `new_box` against each row of `boxes` — same formula as WBF's."""
    xa = np.maximum(boxes[:, 0], new_box[0])
    ya = np.maximum(boxes[:, 1], new_box[1])
    xb = np.minimum(boxes[:, 2], new_box[2])
    yb = np.minimum(boxes[:, 3], new_box[3])
    inter = np.maximum(xb - xa, 0) * np.maximum(yb - ya, 0)
    area_a = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    area_b = (new_box[2] - new_box[0]) * (new_box[3] - new_box[1])
    return inter / (area_a + area_b - inter)


def sigma_weighted_fusion(
    boxes_list: list,
    scores_list: list,
    labels_list: list,
    sigmas_list: list,
    weights: list,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    use_sigma: bool = True,
) -> tuple:
    """WBF with inverse-variance coordinate averaging — scope's actual claim (TODO A1).

    Stock WBF averages cluster coordinates weighted by ``score x model_weight``.
    That asks "how confident is this detector?" and never "how PRECISE is this
    box?" — so the Gaussian head's sigma, the thing the whole method is named
    for, has no influence on the fused output at all.

    Here each member's coordinate weight becomes

        a_i / sigma_i^2        with a_i = score_i * model_weight_i

    per coordinate (sigma_ltrb maps to x1, y1, x2, y2), which is the
    minimum-variance estimator for combining independent measurements — not a
    heuristic. A confident-but-blurry box still votes on WHETHER an object is
    there (it keeps its full weight in the score) but loses its vote on WHERE
    the edges are, in proportion to how badly it localises them.

    **Clustering and scoring are byte-identical to `ensemble_boxes`' WBF**:
    same score x weight prefilter, same descending-score greedy IoU assignment
    against the running fused box, same `conf_type='avg'` score, same
    ``min(n_models, n_cluster) / sum(weights)`` rescale. `use_sigma=False`
    therefore reproduces stock WBF exactly, which is what makes the A/B an
    ablation of sigma alone rather than of two different fusion implementations.
    `scripts/smoke_sigma_wbf.py` asserts that equality.

    sigmas are in the SAME normalized units as boxes (divide pixel sigma by
    [W, H, W, H] before calling); zero/negative sigma is floored, since an
    infinitely precise box would take the entire weight.

    Returns (boxes, scores, labels) like `weighted_boxes_fusion`.
    """
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("fusion weights must sum to > 0")

    # --- prefilter: one row per surviving box, grouped by label --------------
    by_label: dict[int, list] = {}
    for t_idx, (bx, sc, lb, sg) in enumerate(zip(boxes_list, scores_list, labels_list, sigmas_list)):
        bx = np.asarray(bx, dtype=np.float64).reshape(-1, 4)
        sc = np.asarray(sc, dtype=np.float64).reshape(-1)
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        sg = np.asarray(sg, dtype=np.float64).reshape(-1, 4)
        for j in range(len(bx)):
            if sc[j] < skip_box_thr:
                continue
            x1, y1, x2, y2 = np.clip(bx[j], 0.0, 1.0)
            if x2 < x1:
                x1, x2 = x2, x1
            if y2 < y1:
                y1, y2 = y2, y1
            by_label.setdefault(int(lb[j]), []).append(
                (float(sc[j]) * w[t_idx], float(w[t_idx]), np.array([x1, y1, x2, y2]), sg[j]))

    out_boxes, out_scores, out_labels = [], [], []
    for label, entries in by_label.items():
        # WBF sorts by score*weight descending before the greedy assignment.
        order = np.argsort([-e[0] for e in entries], kind="stable")
        entries = [entries[i] for i in order]

        clusters: list[list] = []
        fused = np.empty((0, 4))
        for e in entries:
            idx = -1
            if len(fused):
                ious = _bb_iou(fused, e[2])
                best = int(np.argmax(ious))
                if ious[best] > iou_thr:
                    idx = best
            if idx == -1:
                clusters.append([e])
                fused = np.vstack([fused, e[2][None, :]])
            else:
                clusters[idx].append(e)
                fused[idx] = _fuse_cluster(clusters[idx], use_sigma)

        for members, box in zip(clusters, fused):
            conf = float(np.mean([m[0] for m in members]))
            # conf_type='avg', allows_overflow=False — WBF's exact rescale.
            conf *= min(len(w), len(members)) / w.sum()
            out_boxes.append(box)
            out_scores.append(conf)
            out_labels.append(label)

    if not out_boxes:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    order = np.argsort(-np.asarray(out_scores), kind="stable")
    return (np.asarray(out_boxes)[order], np.asarray(out_scores)[order],
            np.asarray(out_labels, dtype=np.float64)[order])


def _fuse_cluster(members: list, use_sigma: bool) -> np.ndarray:
    """Weighted coordinate mean over one cluster.

    Stock WBF weight is a_i = score_i * model_weight_i, uniform across the four
    coordinates. With sigma the weight becomes a_i / sigma_i^2 PER COORDINATE,
    so a box may be trusted on its left edge and distrusted on its bottom one —
    which is the whole point of a per-side sigma_ltrb head.
    """
    a = np.array([m[0] for m in members], dtype=np.float64)[:, None]     # (n, 1)
    coords = np.stack([m[2] for m in members])                            # (n, 4)
    if use_sigma:
        sig = np.stack([m[3] for m in members]).astype(np.float64)
        a = a / np.maximum(sig, _EPS) ** 2
    total = a.sum(axis=0)
    total = np.where(total <= 0, _EPS, total)
    return (coords * a).sum(axis=0) / total


def fuse_detections(
    vis_record: dict,
    ir_record: dict,
    w_vis: float,
    w_ir: float,
    vis_hw: tuple[int, int],
    h_ir_to_vis: np.ndarray | None = None,
    iou_thr: float = 0.55,
    skip_box_thr: float = 0.0,
    veto_vis: bool = False,
    veto_ir: bool = False,
    sigma_weighted: bool = False,
) -> dict:
    """Weighted Boxes Fusion of two modality records in the VIS frame.

    Weights come from `reliability.fusion_weights` — the uncertainty gate. A
    modality with (near-)zero weight still contributes boxes, but WBF's
    weighted averaging and score scaling let the reliable stream dominate,
    which is the soft-fusion behavior scope §6.1 asks for.

    `veto_*` is the TODO §0.3 hard exclusion, and it is NOT the same thing as
    w -> 0. Down-weighting rescales a modality's scores; it does not remove its
    boxes. mAP is RANK-based, so a false positive that survives with a small
    score still occupies a rank, and — more damagingly — WBF's cluster-score
    rescaling divides every single-modality cluster by the number of input
    lists, so a blind stream's mere PRESENCE halves the surviving stream's
    confidences. Neither effect shrinks with w. A vetoed modality is therefore
    dropped from the input lists entirely, and WBF runs on the survivor alone.
    Vetoing both is refused (the frame would have no detections at all); that
    case is the plan-B3 abstain, signalled by R_sys, not by an empty output.

    `sigma_weighted` routes through `sigma_weighted_fusion`, which averages
    cluster coordinates by inverse variance so the Gaussian head's sigma finally
    influences the fused output (TODO A1). False keeps stock `ensemble_boxes`
    WBF, under which sigma is inert.
    """
    from ensemble_boxes import weighted_boxes_fusion

    h, w = vis_hw
    norm = np.array([w, h, w, h], dtype=np.float64)

    vis_boxes = np.asarray(vis_record["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
    ir_boxes = apply_homography(np.asarray(ir_record["boxes_xyxy"], dtype=np.float64).reshape(-1, 4), h_ir_to_vis)

    if veto_vis and veto_ir:            # every sensor vetoed -> abstain, do not blank the frame
        veto_vis = veto_ir = False
    streams = [(vis_boxes, vis_record, max(w_vis, _EPS), veto_vis),
               (ir_boxes, ir_record, max(w_ir, _EPS), veto_ir)]

    boxes_list, scores_list, labels_list, weights, sigmas_list = [], [], [], [], []
    for boxes, record, weight, vetoed in streams:
        if vetoed:
            continue
        b = np.clip(boxes / norm, 0.0, 1.0)
        boxes_list.append(b.tolist())
        scores_list.append(np.asarray(record["conf"], dtype=np.float64).tolist())
        labels_list.append(np.asarray(record["cls"], dtype=np.float64).tolist())
        weights.append(weight)
        if sigma_weighted:
            # sigma_ltrb is in VIS-frame pixels and ordered (l, t, r, b), which
            # lines up with (x1, y1, x2, y2); divide by the same norm as boxes.
            s = np.asarray(record["sigma_ltrb"], dtype=np.float64).reshape(-1, 4)
            sigmas_list.append(s / norm)

    if sigma_weighted:
        fused_boxes, fused_scores, fused_labels = sigma_weighted_fusion(
            boxes_list, scores_list, labels_list, sigmas_list, weights,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr, use_sigma=True)
    else:
        fused_boxes, fused_scores, fused_labels = weighted_boxes_fusion(
            boxes_list,
            scores_list,
            labels_list,
            weights=weights,
            iou_thr=iou_thr,
            skip_box_thr=skip_box_thr,
        )
    return {
        "boxes_xyxy": np.asarray(fused_boxes) * norm,
        "conf": np.asarray(fused_scores),
        "cls": np.asarray(fused_labels).astype(int),
    }
