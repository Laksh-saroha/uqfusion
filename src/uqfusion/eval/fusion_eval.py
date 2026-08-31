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
from uqfusion.uq.fusion import apply_homography, fuse_detections
from uqfusion.uq.reliability import ReliabilityConstants, compute_reliability, fusion_weights, smooth_reliability


def _distances(records: list[dict], scorer) -> np.ndarray:
    return np.asarray([scorer.score(r["feat"]) for r in records])


def _per_frame_h(h_ir_to_vis, n: int) -> list:
    """Accept one 3x3 matrix (shared), a per-frame sequence of them, or None.

    Pohang extrinsics are per recording run, so a paired set spanning several
    runs needs a different matrix per frame; a single 3x3 is still allowed for
    the synthetic/smoke case.
    """
    if h_ir_to_vis is None:
        return [None] * n
    if isinstance(h_ir_to_vis, np.ndarray) and h_ir_to_vis.shape == (3, 3):
        return [h_ir_to_vis] * n
    seq = list(h_ir_to_vis)
    if len(seq) != n:
        raise ValueError(f"h_ir_to_vis has {len(seq)} matrices for {n} frames")
    return seq


def evaluate_systems(
    vis_records: list[dict],
    ir_records: list[dict],
    scorer,
    constants: ReliabilityConstants,
    gate=None,
    h_ir_to_vis: np.ndarray | None = None,
    iou_thr_wbf: float = 0.55,
    scorer_ir=None,
    constants_ir: ReliabilityConstants | None = None,
    capability_vis: float = 1.0,
    capability_ir: float = 1.0,
    skip_box_thr: float = 0.0,
    gts: list[dict] | None = None,
    brightness_vis: np.ndarray | None = None,
    brightness_ir: np.ndarray | None = None,
    veto_below: float | None = None,
    veto_on: str = "r_bright",
    sigma_weighted: bool = False,
    veto_override: tuple[list, list] | None = None,
) -> dict:
    """mAP@50-95 (and mAP@50) per system over the paired frame set. Also returns
    per-frame gate weights and R_sys for the B3 abstain analysis.

    `scorer_ir`/`constants_ir` default to the VIS ones (the synthetic smoke has a
    single model, so one of each is right there). On real data the two streams
    come from DIFFERENT checkpoints: their pooled-feature spaces are not
    comparable and their clean-val σ scales differ, so each modality needs its
    own scorer and its own constants. Reliability is then "degraded relative to
    THIS sensor's own clean baseline", which is the quantity the gate needs.

    `capability_*` restores what that normalization throws away. R is a RETAINED
    FRACTION, so a clean IR frame and a clean VIS frame both score ~0.9 even
    though VIS detects 12x better; the weights then sit near 0.5/0.5 on clean
    data and IR dilutes a far stronger stream. Passing each modality's clean-val
    mAP makes the weight proportional to EXPECTED ABSOLUTE capability,
    R_m * mAP_clean_m, which is the quantity WBF actually wants. Defaults of 1.0
    reproduce the pure-ratio behaviour. R_sys is deliberately left on the
    unscaled R: it is the plan-B3 abstain signal ("has every modality failed?"),
    a question about retention, not about which stream to prefer.

    `brightness_*` are the per-frame content-region statistics from
    `frame_brightness.py`, consumed only when the matching constants carry a
    fitted `mu_b` (TODO §0.2); passing None reproduces §6.4 exactly. Mahalanobis
    distance cannot see darkness -- a night frame is not unusual, it is EMPTY,
    and empty sits near the middle of the training distribution. On pohang01 the
    gate reads D=28.4, *cleaner* than daylight pohang00's 30.0, on frames where
    the detector scores 0.0000. The photometric term supplies that missing axis.

    `veto_below` is the TODO §0.3 rule: a modality scoring below it is excluded
    from fusion outright rather than down-weighted (see `fuse_detections`). Use
    0.5 — not a new constant but the sigmoid midpoint, i.e. the boundary already
    pre-registered under D5/B5. Down-weighting alone cannot fix a blind stream:
    weights normalize, so w_vis stays at 0.199 on the night run purely because
    the capability prior hands VIS a 12.5x advantage that a small r cannot
    overcome. None disables the rule and reproduces §16 exactly.

    `veto_on` chooses WHICH signal holds veto authority, and this is a
    substantive decision, not a knob:

      "r_bright"  (default) — `b < mu_b`. Brightness is monotone in capability in
                  the direction that matters: below mu_b no photons reached the
                  sensor, and an empty frame is not recoverable by any detector.
      "r_frame"   — `D > mu_d` OR `b < mu_b`. MEASURED TO BE WRONG: Mahalanobis
                  distance answers "is this frame unusual?", which is not the
                  same question. Glare on DAYLIGHT fit-run frames pushes D past
                  mu_d on ~47% of them while VIS still scores 0.2626 against IR's
                  0.0092, and vetoing there costs glare/day 0.2532 -> 0.1435.
                  A signal that is not monotone in capability may down-weight;
                  it must not veto. Kept only so that result stays reproducible.

    IR carries no photometric term by design (dark IR is cold water), so under
    the default IR can never be vetoed — which is the intended asymmetry.

    `sigma_weighted` averages cluster coordinates by inverse variance so the
    Gaussian head's sigma finally reaches the fused output (TODO A1). Until this
    existed, NOTHING the sigma head produced influenced fusion: `r_box` sits at
    0.86-0.95 in every condition including the fog case where mAP is 0.0012, and
    stock WBF weights coordinates by score alone. `scripts/smoke_sigma_wbf.py`
    asserts the sigma path reduces to stock WBF exactly when sigma is constant,
    so the A/B measures sigma and not a second fusion implementation.

    `veto_override` supplies the two per-frame boolean lists directly, bypassing
    the `r_bright < veto_below` test while leaving the SOFT gate untouched. It
    exists for the temporal-hysteresis question: the adopted veto is decided
    per frame from an instantaneous brightness reading, but darkness is a
    property of a contiguous stretch of a recording, not of one frame. Fog lifts
    p05 above mu_b on 71% of night frames, so the switch flickers on a condition
    that does not (§4.5). Smoothing `brightness_vis` instead would also move
    `r_bright` inside the soft weight, which would make the arm a test of two
    changes; this overrides only the switch. The both-vetoed case is still
    collapsed to neither, exactly as in the fitted path.

    Both the veto and sigma weighting apply to GATED fusion only. Naive 0.5/0.5
    is the fixed-weight control and must stay untouched, or it stops being a
    control.
    """
    assert len(vis_records) == len(ir_records), "paired caches must be index-aligned"
    scorer_ir = scorer if scorer_ir is None else scorer_ir
    constants_ir = constants if constants_ir is None else constants_ir
    # `gts` overrides the default VIS-frame labels — used by the union-label
    # evaluation, where GT is VIS labels UNION the IR labels projected into the
    # VIS frame (an IR-only detection of a target VIS never annotated is not a
    # false positive of the fusion system).
    if gts is None:
        gts = [load_gt(r["image_path"], r["image_hw"]) for r in vis_records]
    d_vis = _distances(vis_records, scorer)
    d_ir = _distances(ir_records, scorer_ir)

    h_frames = _per_frame_h(h_ir_to_vis, len(vis_records))
    n = len(vis_records)
    b_vis = [None] * n if brightness_vis is None else [float(x) for x in brightness_vis]
    b_ir = [None] * n if brightness_ir is None else [float(x) for x in brightness_ir]
    if len(b_vis) != n or len(b_ir) != n:
        raise ValueError("brightness arrays must be index-aligned with the paired caches")
    if veto_override is not None:
        ov_v, ov_i = ([bool(x) for x in veto_override[0]], [bool(x) for x in veto_override[1]])
        if len(ov_v) != n or len(ov_i) != n:
            raise ValueError("veto_override lists must be index-aligned with the paired caches")
    else:
        ov_v = ov_i = None
    fused_gated, fused_naive, fused_learned = [], [], []
    w_vis_gated, r_sys_all, r_bright_all = [], [], []
    rf_vis_all, rf_ir_all, veto_vis_all, veto_ir_all = [], [], [], []
    r_prev_vis = r_prev_ir = None
    for fi_, (rv, ri, dv, di, h, bv, bi) in enumerate(
            zip(vis_records, ir_records, d_vis, d_ir, h_frames, b_vis, b_ir)):
        rel_v = compute_reliability(rv, dv, constants, bv)
        rel_i = compute_reliability(ri, di, constants_ir, bi)
        r_bright_all.append(rel_v["r_bright"])
        rf_vis_all.append(rel_v["r_frame"])
        rf_ir_all.append(rel_i["r_frame"])
        veto_v = veto_i = False
        if ov_v is not None:
            veto_v, veto_i = ov_v[fi_], ov_i[fi_]
            if veto_v and veto_i:          # same abstain rule as the fitted path
                veto_v = veto_i = False
        elif veto_below is not None:
            if veto_on not in ("r_bright", "r_frame"):
                raise ValueError(f"veto_on must be 'r_bright' or 'r_frame', got {veto_on!r}")
            # A modality with no photometric term (IR) has r_bright None and is
            # never vetoed under the default rule.
            veto_v = rel_v[veto_on] is not None and rel_v[veto_on] < veto_below
            veto_i = rel_i[veto_on] is not None and rel_i[veto_on] < veto_below
            if veto_v and veto_i:          # nothing left to fuse -> abstain, keep both
                veto_v = veto_i = False
        veto_vis_all.append(veto_v)
        veto_ir_all.append(veto_i)
        r_v = smooth_reliability(rel_v["R"], r_prev_vis, constants.alpha)
        r_i = smooth_reliability(rel_i["R"], r_prev_ir, constants.alpha)
        r_prev_vis, r_prev_ir = r_v, r_i

        w = fusion_weights(r_vis=r_v * capability_vis, r_ir=r_i * capability_ir)
        w_vis_gated.append(w["w_vis"])
        r_sys_all.append(fusion_weights(r_vis=r_v, r_ir=r_i)["R_sys"])
        hw = rv["image_hw"]
        fused_gated.append(fuse_detections(rv, ri, w["w_vis"], w["w_ir"], hw, h, iou_thr_wbf,
                                          skip_box_thr, veto_vis=veto_v, veto_ir=veto_i,
                                          sigma_weighted=sigma_weighted))
        fused_naive.append(fuse_detections(rv, ri, 0.5, 0.5, hw, h, iou_thr_wbf, skip_box_thr))
        if gate is not None:
            wl = gate.predict_w_vis(rv, ri, dv, di)
            fused_learned.append(fuse_detections(rv, ri, wl, 1.0 - wl, hw, h, iou_thr_wbf, skip_box_thr))

    # The IR-only row is scored against the VIS frame's GT like every other
    # system, so its boxes must be mapped into the VIS frame FIRST. Without this
    # the row compares IR-canvas coordinates to VIS-canvas labels and collapses
    # to ~0 for reasons that have nothing to do with the IR detector. (Identity
    # H makes this a no-op, which is why the synthetic smoke never caught it.)
    ir_in_vis = [{**r, "boxes_xyxy": apply_homography(np.asarray(r["boxes_xyxy"]).reshape(-1, 4), h)}
                 for r, h in zip(ir_records, h_frames)]

    out = {
        "visible_only": map50_95(vis_records, gts),
        "ir_only": map50_95(ir_in_vis, gts),
        "naive_fusion": map50_95(fused_naive, gts),
        "gated_fusion": map50_95(fused_gated, gts),
        "w_vis_gated": w_vis_gated,
        "R_sys": r_sys_all,
        "r_bright_vis": r_bright_all,
        "r_frame_vis": rf_vis_all,
        "r_frame_ir": rf_ir_all,
        "veto_vis": veto_vis_all,
        "veto_ir": veto_ir_all,
        "fused_gated": fused_gated,
        "ir_in_vis": ir_in_vis,   # IR boxes mapped to the VIS canvas, for per-run breakdowns
        "gts": gts,
    }
    if gate is not None:
        out["learned_gate_fusion"] = map50_95(fused_learned, gts)
    return out


def ablate_gate_rules(
    vis_records, ir_records, scorer, base_constants: ReliabilityConstants,
    combinations=("multiplicative", "min", "geometric"),
    alphas=(1.0, 0.5),
    h_ir_to_vis=None,
    scorer_ir=None,
    base_constants_ir: ReliabilityConstants | None = None,
) -> list[dict]:
    """Plan B5-3/B5-4 sweep: gated-fusion mAP per (combination, α) — CPU only."""
    rows = []
    for comb in combinations:
        for alpha in alphas:
            constants = replace(base_constants, combination=comb, alpha=alpha)
            c_ir = None if base_constants_ir is None else replace(base_constants_ir, combination=comb, alpha=alpha)
            res = evaluate_systems(vis_records, ir_records, scorer, constants, h_ir_to_vis=h_ir_to_vis,
                                   scorer_ir=scorer_ir, constants_ir=c_ir)
            rows.append({
                "combination": comb, "alpha": alpha,
                "gated_map50_95": res["gated_fusion"]["map50_95"],
                "gated_map50": res["gated_fusion"]["map50"],
                # Macro mAP averages ship and buoy, and only ship can be fused (IR
                # is nc=1 ship-only, D28/A-1) — so a real ship gain reaches the
                # macro column already halved, and a VIS veto zeroes the buoy row
                # outright. Rank these rules on the per-class AP, not the macro.
                "gated_per_class": res["gated_fusion"]["per_class"],
                "mean_w_vis": float(np.mean(res["w_vis_gated"])),
            })
    return rows


def format_ablation_table(rows: list[dict]) -> str:
    """Ablation table with per-class AP beside the macro mean.

    Rank on the per-class columns — see `matching.map50_95` for why the macro
    column understates every fusion effect on this class set. Rows produced
    before `gated_per_class` existed still render, with the columns blank.
    """
    classes = sorted({int(c) for r in rows for c in (r.get("gated_per_class") or {})})
    head = ["Combination", "α", "Gated mAP@50–95"]
    head += [f"AP@50–95 cls{c}" for c in classes]
    head += ["Gated mAP@50", "mean w_vis"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        pc = r.get("gated_per_class") or {}
        cells = [str(r["combination"]), str(r["alpha"]), f"{r['gated_map50_95']:.3f}"]
        for c in classes:
            e = pc.get(c, pc.get(str(c)))
            cells.append(f"{e['ap50_95']:.3f}" if e else "—")
        cells += [f"{r['gated_map50']:.3f}", f"{r['mean_w_vis']:.3f}"]
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)
