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
    consensus_beta: float = 1.0,
    consensus_distinct: bool = False,
    support_iou: float = 0.0,
    support_gamma: float = 0.0,
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

    `consensus_beta` and `consensus_distinct` expose the agreement bonus, which
    until now was an unexamined property of a third-party library. WBF multiplies
    a cluster's score by ``min(n_models, n_cluster) / sum(weights)``, so a box the
    two sensors BOTH saw is scored twice as high as one only a single sensor saw.
    That factor is where this system's day-cell gain actually comes from -- on a
    clean day frame IR contributes almost no detections of its own, yet fusion
    still beats VIS alone, because agreeing with VIS re-ranks VIS's own boxes
    upward relative to its unconfirmed ones. Nobody chose the factor 2; it is what
    `ensemble_boxes` happens to do.

    ``(1 + beta * (k - 1)) / sum(weights)`` replaces it, with k the cluster size
    capped at the number of streams. **beta = 1.0 is stock WBF exactly** (k=1 -> 1,
    k=2 -> 2), so the default remains an ablation of nothing.

    `consensus_distinct` fixes a second thing WBF does not distinguish: k counts
    cluster MEMBERS, so two overlapping boxes from the SAME sensor earn the same
    bonus as one confirmation from the other. That is self-agreement priced as
    cross-modal agreement, and on a stream that emits duplicates it is a bonus for
    being noisy. True counts distinct source streams instead.

    `support_iou` / `support_gamma` let one sensor vouch for another sensor's box
    WITHOUT moving it, and they exist because the geometry says the merge-based
    version cannot work here. Measured on the clean day cell, only **0.05-0.08%**
    of VIS detections have a same-class IR detection at IoU >= 0.85, the adopted
    `iou_thr`: at that threshold the two streams essentially never land in the same
    cluster, so WBF's agreement bonus almost never fires cross-modally and the
    day-cell gain has to come from somewhere else (IR boxes entering at the bottom
    of the ranking and recovering GT that VIS missed). Agreement only becomes
    geometrically available at IoU 0.1-0.3, where 32-48% of VIS boxes do have an IR
    partner -- which is exactly where the 3-6 px median registration residual
    (`runs/eval/x_registration_drift.md`) puts it.

    Lowering `iou_thr` to reach that regime is not the fix: it was swept and loses
    (-0.0138 on the fit set at 0.55), because merging across a 5 px residual drags
    the fused coordinates off the object. The information IR carries at that
    threshold is *whether* a target is there, not *where* its edges are.

    So: a cluster gets its score multiplied by ``1 + support_gamma`` when a box
    from a DIFFERENT input stream overlaps it at >= `support_iou`. Coordinates are
    untouched, and the supporting box still competes on its own. `support_gamma=0`
    disables it exactly.

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
                (float(sc[j]) * w[t_idx], float(w[t_idx]), np.array([x1, y1, x2, y2]), sg[j],
                 t_idx))

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

        # Cross-stream support: one pass per label, on the entries already
        # prefiltered above, so the boost sees exactly the boxes fusion saw.
        sup = np.zeros(len(clusters), dtype=bool)
        if support_gamma and len(clusters):
            ent_box = np.stack([e[2] for e in entries])
            ent_src = np.asarray([e[4] for e in entries])
            for ci, (members, box) in enumerate(zip(clusters, fused)):
                mine = {m[4] for m in members}
                other = ~np.isin(ent_src, list(mine))
                if not other.any():
                    continue
                sup[ci] = bool((_bb_iou(ent_box[other], box) >= support_iou).any())

        for ci, (members, box) in enumerate(zip(clusters, fused)):
            conf = float(np.mean([m[0] for m in members]))
            # conf_type='avg', allows_overflow=False — WBF's exact rescale when
            # consensus_beta == 1.0 and consensus_distinct is False.
            k = (len({m[4] for m in members}) if consensus_distinct
                 else min(len(w), len(members)))
            k = min(k, len(w))
            conf *= (1.0 + float(consensus_beta) * (k - 1)) / w.sum()
            if support_gamma and sup[ci]:
                conf *= 1.0 + float(support_gamma)
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
    score_scale: tuple[float, float] | None = None,
    single_passthrough: bool = False,
    veto_keep_cls: tuple[int, ...] | None = None,
    consensus_beta: float = 1.0,
    consensus_distinct: bool = False,
    support_iou: float = 0.0,
    support_gamma: float = 0.0,
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

    `score_scale` is the SOFT alternative to `veto_*`, and it exists because the
    argument that forced the hard veto is narrower than it looks. WBF computes a
    cluster score as ``mean(score_i * w_i) * min(n_models, n_cluster) /
    sum(weights)``, which is invariant to rescaling the whole weight vector — so
    with `w` normalized to sum 1, as `fusion_weights` returns it, a stream's
    weight only ever expresses its share RELATIVE to the other stream. Two frames
    where VIS is 36x better than IR and where VIS is dead both come out near the
    same `w_vis`, because normalization discards exactly the absolute level the
    gate spent its effort estimating. That, not some property of down-weighting
    itself, is why "down-weighting cannot remove a failed stream".

    Passing ``(a_vis, a_ir)`` multiplies each stream's confidences by its own
    ABSOLUTE trust before fusion and then hands WBF equal weights, so the
    normalization has nothing left to cancel. Every cluster score is then
    proportional to the trust of the stream(s) that produced it, across frames as
    well as within one — which is what a rank-based metric like AP needs, since it
    pools detections from every frame into a single ordering. The hard veto is the
    ``a_m -> 0`` limit of this, up to the ranks that zero-scored boxes still
    occupy at the very bottom of that ordering.

    None keeps the normalized-weight path exactly as it was.

    `single_passthrough` returns the surviving stream's detections unchanged when
    the veto has left only one. Measured, that is not a no-op: WBF over a single
    input list still clips every box to the canvas, still merges boxes that
    overlap above `iou_thr`, and still replaces a merged cluster's scores with
    their mean. On the paired val set an IR-only day cell scores 0.0177 as raw
    mapped boxes and 0.0166 once passed through single-list WBF -- so a fully
    vetoed day cell carries a -0.0011 floor that has nothing to do with the gate
    and nothing to do with fusion, since there is nothing left to fuse. (At night
    the same post-process is worth +0.0003, because merging genuinely helps a
    stream that emits duplicates there. Both directions are the same artifact.)

    Default False, because the adopted table was measured with it off.

    `veto_keep_cls` names the classes a veto may NOT remove, and it exists
    because the veto was designed for a two-class problem and is deployed on an
    asymmetric one. IR is `nc=1` (D28/A-1): it detects ships and nothing else. So
    when the gate vetoes VIS on a fogged or dark frame, ship detection correctly
    falls to IR -- and BUOY detection falls to nobody, because the only stream
    that ever had a buoy box was just deleted. The frame is not handed to the
    better sensor for buoys; it is handed to no sensor. Ship AP cannot see this,
    which is why it went unnoticed: it lives entirely in the macro number.

    Passing the classes IR cannot supply keeps VIS's boxes for exactly those
    classes and drops the rest, so the veto still removes VIS from the decision
    it is unfit for while leaving the one where it is unopposed. A VIS box that
    survives this way faces no competing stream, so it can neither dilute nor
    outrank anything -- the objection that forced the hard veto does not apply to
    a class only one stream produces.

    If a vetoed stream has no boxes of the kept classes it is dropped outright,
    which is the pre-existing behaviour. None or `()` reproduces it everywhere.

    `consensus_beta` / `consensus_distinct` tune the cross-modal agreement bonus,
    and `support_iou` / `support_gamma` add a looser one that boosts a box's score
    without letting the other stream move its coordinates -- see
    `sigma_weighted_fusion` for why the geometry forces that split. Non-default
    values route through the local WBF implementation, which
    `scripts/smoke_sigma_wbf.py` asserts is byte-identical to `ensemble_boxes` at
    the defaults, so the arms measure the change and not a second fusion
    implementation.
    """
    from ensemble_boxes import weighted_boxes_fusion

    h, w = vis_hw
    norm = np.array([w, h, w, h], dtype=np.float64)

    vis_boxes = np.asarray(vis_record["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
    ir_boxes = apply_homography(np.asarray(ir_record["boxes_xyxy"], dtype=np.float64).reshape(-1, 4), h_ir_to_vis)

    if veto_vis and veto_ir:            # every sensor vetoed -> abstain, do not blank the frame
        veto_vis = veto_ir = False
    # Under `score_scale` the trust rides on the SCORES and the weights are equal,
    # so WBF's `1 / sum(weights)` rescale has nothing to normalize away.
    scale = (1.0, 1.0) if score_scale is None else (float(score_scale[0]), float(score_scale[1]))
    streams = [(vis_boxes, vis_record, max(w_vis, _EPS), veto_vis, scale[0]),
               (ir_boxes, ir_record, max(w_ir, _EPS), veto_ir, scale[1])]

    # A veto with `veto_keep_cls` REDUCES the stream to the classes the other one
    # cannot supply instead of deleting it. Resolved here, before every path
    # below, so `single_passthrough` and the WBF path agree on what "alive" means.
    keep = None if not veto_keep_cls else set(int(c) for c in veto_keep_cls)
    if keep is not None and (veto_vis or veto_ir):
        reduced = []
        for boxes, record, weight, vetoed, a in streams:
            if vetoed:
                m = np.isin(np.asarray(record["cls"]).astype(int), list(keep))
                if m.any():
                    sub = {**record, "conf": np.asarray(record["conf"])[m],
                           "cls": np.asarray(record["cls"])[m]}
                    if "sigma_ltrb" in record:
                        # Carried with its own box, for the same reason the IR NMS
                        # has to: a mask applied to boxes and not to sigma leaves
                        # `sigma_weighted_fusion` two arrays of different length.
                        sub["sigma_ltrb"] = np.asarray(
                            record["sigma_ltrb"], dtype=np.float64).reshape(-1, 4)[m]
                    boxes, record, vetoed = boxes[m], sub, False
            reduced.append((boxes, record, weight, vetoed, a))
        streams = reduced

    if single_passthrough:
        alive = [(b, r, a) for b, r, _, v, a in streams if not v]
        if len(alive) == 1:
            b, record, a = alive[0]
            return {"boxes_xyxy": np.asarray(b, dtype=np.float64).reshape(-1, 4),
                    "conf": np.asarray(record["conf"], dtype=np.float64) * a,
                    "cls": np.asarray(record["cls"]).astype(int)}

    boxes_list, scores_list, labels_list, weights, sigmas_list = [], [], [], [], []
    for boxes, record, weight, vetoed, a in streams:
        if vetoed:
            continue
        if score_scale is not None:
            weight = 1.0
        b = np.clip(boxes / norm, 0.0, 1.0)
        boxes_list.append(b.tolist())
        scores_list.append((np.asarray(record["conf"], dtype=np.float64) * a).tolist())
        labels_list.append(np.asarray(record["cls"], dtype=np.float64).tolist())
        weights.append(weight)
        if sigma_weighted:
            # sigma_ltrb is in VIS-frame pixels and ordered (l, t, r, b), which
            # lines up with (x1, y1, x2, y2); divide by the same norm as boxes.
            s = np.asarray(record["sigma_ltrb"], dtype=np.float64).reshape(-1, 4)
            sigmas_list.append(s / norm)
        elif consensus_beta != 1.0 or consensus_distinct or support_gamma:
            # The local implementation needs a sigma column even when it will not
            # use one; ones make `_fuse_cluster` reduce to the stock coordinate mean.
            sigmas_list.append(np.ones_like(b))

    if sigma_weighted or consensus_beta != 1.0 or consensus_distinct or support_gamma:
        fused_boxes, fused_scores, fused_labels = sigma_weighted_fusion(
            boxes_list, scores_list, labels_list, sigmas_list, weights,
            iou_thr=iou_thr, skip_box_thr=skip_box_thr, use_sigma=sigma_weighted,
            consensus_beta=consensus_beta, consensus_distinct=consensus_distinct,
            support_iou=support_iou, support_gamma=support_gamma)
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
