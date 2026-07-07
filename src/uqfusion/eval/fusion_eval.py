"""Fusion-system evaluation + gate-level ablations over cached predictions.

Systems compared (scope §9.3, Table 3 rows): visible-only, IR-only, naive
fixed-weight fusion (0.5/0.5), uncertainty-gated fusion (§6.4), and optionally
the learned gate (§7.5). All systems consume the SAME two prediction caches —
no model is re-run — which is what makes the §6.4 ablation grid (combination
rule x α x λ-rule) a pure CPU sweep (plan B5-2).

GT convention: paired caches are index-aligned (frame i of the VIS cache and
frame i of the IR cache observe the same scene); fused output is evaluated in
the VIS frame against the VIS frame's labels (decision D7).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from uqfusion.eval.matching import load_gt, map50_95
from uqfusion.uq.fusion import fuse_detections
from uqfusion.uq.reliability import ReliabilityConstants, compute_reliability, fusion_weights, smooth_reliability


def _distances(records: list[dict], scorer) -> np.ndarray:
    return np.asarray([scorer.score(r["feat"]) for r in records])


def evaluate_systems(
    vis_records: list[dict],
    ir_records: list[dict],
    scorer,
    constants: ReliabilityConstants,
    gate=None,
    h_ir_to_vis: np.ndarray | None = None,
    iou_thr_wbf: float = 0.55,
) -> dict:
    """mAP@50-95 (and mAP@50) per system over the paired frame set. Also returns
    per-frame gate weights and R_sys for the B3 abstain analysis."""
    assert len(vis_records) == len(ir_records), "paired caches must be index-aligned"
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in vis_records]
    d_vis = _distances(vis_records, scorer)
    d_ir = _distances(ir_records, scorer)

    fused_gated, fused_naive, fused_learned = [], [], []
    w_vis_gated, r_sys_all = [], []
    r_prev_vis = r_prev_ir = None
    for rv, ri, dv, di in zip(vis_records, ir_records, d_vis, d_ir):
        rel_v = compute_reliability(rv, dv, constants)
        rel_i = compute_reliability(ri, di, constants)
        r_v = smooth_reliability(rel_v["R"], r_prev_vis, constants.alpha)
        r_i = smooth_reliability(rel_i["R"], r_prev_ir, constants.alpha)
        r_prev_vis, r_prev_ir = r_v, r_i

        w = fusion_weights(r_vis=r_v, r_ir=r_i)
        w_vis_gated.append(w["w_vis"])
        r_sys_all.append(w["R_sys"])
        hw = rv["image_hw"]
        fused_gated.append(fuse_detections(rv, ri, w["w_vis"], w["w_ir"], hw, h_ir_to_vis, iou_thr_wbf))
        fused_naive.append(fuse_detections(rv, ri, 0.5, 0.5, hw, h_ir_to_vis, iou_thr_wbf))
        if gate is not None:
            wl = gate.predict_w_vis(rv, ri, dv, di)
            fused_learned.append(fuse_detections(rv, ri, wl, 1.0 - wl, hw, h_ir_to_vis, iou_thr_wbf))

    out = {
        "visible_only": map50_95(vis_records, gts),
        "ir_only": map50_95(ir_records, gts),
        "naive_fusion": map50_95(fused_naive, gts),
        "gated_fusion": map50_95(fused_gated, gts),
        "w_vis_gated": w_vis_gated,
        "R_sys": r_sys_all,
    }
    if gate is not None:
        out["learned_gate_fusion"] = map50_95(fused_learned, gts)
    return out


def ablate_gate_rules(
    vis_records, ir_records, scorer, base_constants: ReliabilityConstants,
    combinations=("multiplicative", "min", "geometric"),
    alphas=(1.0, 0.5),
    h_ir_to_vis=None,
) -> list[dict]:
    """Plan B5-3/B5-4 sweep: gated-fusion mAP per (combination, α) — CPU only."""
    rows = []
    for comb in combinations:
        for alpha in alphas:
            constants = replace(base_constants, combination=comb, alpha=alpha)
            res = evaluate_systems(vis_records, ir_records, scorer, constants, h_ir_to_vis=h_ir_to_vis)
            rows.append({
                "combination": comb, "alpha": alpha,
                "gated_map50_95": res["gated_fusion"]["map50_95"],
                "gated_map50": res["gated_fusion"]["map50"],
                "mean_w_vis": float(np.mean(res["w_vis_gated"])),
            })
    return rows


def format_ablation_table(rows: list[dict]) -> str:
    lines = ["| Combination | α | Gated mAP@50–95 | Gated mAP@50 | mean w_vis |", "|---|---|---|---|---|"]
    for r in rows:
        lines.append(
            f"| {r['combination']} | {r['alpha']} | {r['gated_map50_95']:.3f} "
            f"| {r['gated_map50']:.3f} | {r['mean_w_vis']:.3f} |"
        )
    return "\n".join(lines)
