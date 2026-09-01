"""Soft absolute-trust fusion, measured against the hard veto on all eight cells.

**The claim being tested.** `docs/gated-fusion-handoff.md` §4.2 says down-weighting
cannot remove a failed stream and the veto is therefore forced. The supporting
measurement is real -- `w_vis` sits at 0.199 on the night run -- but the
conclusion is stronger than the evidence. WBF scores a cluster as
``mean(score_i * w_i) * min(n_models, n_cluster) / sum(weights)``, which is
invariant to rescaling the weight vector, and `fusion_weights` returns weights
normalized to sum 1. So a weight can only ever say which stream is better on this
frame, never how good either one IS. A frame where VIS beats IR 36x and a frame
where VIS is blind can produce similar `w_vis`, and the absolute level the gate
worked to estimate is discarded at the final step.

AP is computed by pooling every frame's detections into ONE ranking, so that
discarded quantity is exactly the one that decides whether a blind frame's boxes
outrank a working frame's. The hard veto is a way of forcing the issue per frame.
This script asks whether carrying the absolute level through instead -- multiply
each stream's scores by its own trust, hand WBF equal weights -- does the same job
without a switch, without hysteresis, and without an all-or-nothing decision on
cells like lowlight/day where the right answer is neither "keep" nor "drop".

**Trust, and why it needs no new constants.** Each axis contributes a factor in
[0, 1] built from a constant that is already fitted and already in the repo:

    photometric  sigmoid((p05 - mu_b) / tau_b)          brightness_constants.json
    veil         min(1, lap_var / tau_lap)              brightness_constants.json
    evidence     min(1, windowed_conf / thr_novelty)    novelty bound, tau_lap protocol

times the capability prior and, optionally, the two existing soft terms
(`r_frame` from Mahalanobis, `r_box` from sigma). The veil and evidence factors
are RATIOS to their thresholds rather than sigmoids, because a hard novelty bound
supplies a location and no scale, and inventing a scale would be inventing a
constant. A ratio capped at 1 is the softening that the bound itself licenses.

The arms are named for which terms are on, so the ablation is readable off the
name. Writes a new report; overwrites nothing.

Usage:
    python scripts/eval_soft_trust.py --out runs/eval/soft_trust.md --n-boot 1000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import FIT_RUNS, NIGHT_RUNS, load_context, run_systems    # noqa: E402
from uqfusion.uq.reliability import compute_reliability, per_box_uncertainty     # noqa: E402

SHIP = 0
EV_WINDOW = 31


# ---------------------------------------------------------------------------
# per-frame factors
# ---------------------------------------------------------------------------

def evidence_stat(records: list[dict]) -> np.ndarray:
    """Summed detection confidence per frame — the detector's own response."""
    return np.asarray([float(np.asarray(r["conf"], dtype=float).sum()) for r in records])


def window_max(v: np.ndarray, order: dict, k: int) -> np.ndarray:
    if k <= 1:
        return v.copy()
    out, half = v.copy(), k // 2
    for idx in order.values():
        seq = v[idx]
        pad = np.pad(seq, (half, half), mode="edge")
        out[idx] = np.lib.stride_tricks.sliding_window_view(pad, k)[:len(seq)].max(axis=1)
    return out


def soft_factors(ctx, cond: str, ev_thr: float) -> dict[str, np.ndarray]:
    """The three input/output axes as multiplicative factors in [0, 1]."""
    n = ctx.n()
    out = {"photo": np.ones(n), "veil": np.ones(n), "evid": np.ones(n)}
    b = ctx.bright_by_cond.get(cond)
    if b is not None and ctx.c_vis.mu_b is not None:
        out["photo"] = 1.0 / (1.0 + np.exp(-(b - ctx.c_vis.mu_b) / max(ctx.c_vis.tau_b, 1e-9)))
    lap = ctx.struct_by_cond.get(cond)
    if lap is not None and ctx.tau_lap:
        out["veil"] = np.clip(lap / ctx.tau_lap, 0.0, 1.0)
    ev = window_max(evidence_stat(ctx.vis_by_cond[cond]), ctx.order, EV_WINDOW)
    out["evid"] = np.clip(ev / max(ev_thr, 1e-12), 0.0, 1.0)
    return out


def base_reliability(ctx, cond: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(r_frame, r_box) per frame for each stream, from the existing gate."""
    def one(records, scorer, consts):
        rf, rb = [], []
        for r in records:
            rel = compute_reliability(r, float(scorer.score(r["feat"])), consts, None)
            rf.append(rel["r_frame"])
            rb.append(1.0 if rel["r_box"] is None else rel["r_box"])
        return np.asarray(rf), np.asarray(rb)
    rfv, rbv = one(ctx.vis_by_cond[cond], ctx.scorer_vis, ctx.c_vis)
    rfi, rbi = one(ctx.ir_clean, ctx.scorer_ir, ctx.c_ir)
    return rfv, rbv, rfi, rbi


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/soft_trust.md")
    ap.add_argument("--n-boot", type=int, default=0)
    ap.add_argument("--conditions", nargs="+", default=None)
    args = ap.parse_args()

    t0 = time.time()
    kw = {"conditions": tuple(args.conditions)} if args.conditions else {}
    ctx = load_context(**kw)
    n = ctx.n()
    night = np.isin(ctx.runs, NIGHT_RUNS)
    fit = np.isin(ctx.runs, FIT_RUNS)
    splits = {"day": ~night, "night": night}
    cells = [(c, s) for c in ctx.conditions for s in ("day", "night")]

    # Novelty bound on the evidence axis: the lowest windowed confidence any CLEAN
    # FIT-RUN frame produced. pohang01 and every corrupted condition are held out,
    # exactly as for tau_lap.
    ev_thr = float(window_max(evidence_stat(ctx.vis_by_cond["clean"]), ctx.order,
                              EV_WINDOW)[fit].min())
    print(f"[soft] evidence novelty bound (sum_conf, window {EV_WINDOW}): {ev_thr:.4f}")

    no_veto = ([False] * n, [False] * n)

    # arm -> (use_maha, use_box, axes)
    ARMS = {
        "hard veto [adopted]": None,                       # the reference, veto path
        "soft: cap x photo x veil x evid": (False, False, ("photo", "veil", "evid")),
        "soft: cap x photo x veil": (False, False, ("photo", "veil")),
        "soft: cap x evid": (False, False, ("evid",)),
        "soft: cap x rframe x rbox x photo x veil x evid": (True, True, ("photo", "veil", "evid")),
        "soft: cap x photo x veil x evid + hard veto": (False, False, ("photo", "veil", "evid")),
    }
    HYBRID = "soft: cap x photo x veil x evid + hard veto"

    parts, extra = {}, {}
    for cond in ctx.conditions:
        f = soft_factors(ctx, cond, ev_thr)
        rfv, rbv, rfi, rbi = base_reliability(ctx, cond)
        for arm, spec in ARMS.items():
            if spec is None:
                res = run_systems(ctx, cond)                      # adopted veto path
            else:
                use_maha, use_box, axes = spec
                tv = np.full(n, ctx.cap_vis)
                ti = np.full(n, ctx.cap_ir)
                if use_maha:
                    tv, ti = tv * rfv, ti * rfi
                if use_box:
                    tv, ti = tv * rbv, ti * rbi
                for a in axes:
                    tv = tv * f[a]        # IR carries no photometric/veil/evidence term
                over = {"trust_override": (tv.tolist(), ti.tolist())}
                if arm != HYBRID:
                    over["veto_override"] = no_veto
                res = run_systems(ctx, cond, **over)
                extra[(arm, cond)] = {"t_vis_med": float(np.median(tv)),
                                      "t_ir_med": float(np.median(ti))}
            parts[(arm, cond)] = frame_parts(res["fused_gated"], ctx.gts)
            if arm == "hard veto [adopted]":
                parts[("_vis", cond)] = frame_parts(ctx.vis_by_cond[cond], ctx.gts)
                parts[("_ir", cond)] = frame_parts(res["ir_in_vis"], ctx.gts)
        print(f"[soft] {cond}: {len(ARMS)} arms done ({time.time() - t0:.0f}s)", flush=True)

    def ap(key, cond, s):
        sel = np.flatnonzero(splits[s])
        r = ap_from_parts([parts[(key, cond)][i] for i in sel])
        e = r["per_class"].get(SHIP)
        return float(e["ap50_95"]) if e else 0.0

    bar = {(c, s): max(ap("_vis", c, s), ap("_ir", c, s)) for c, s in cells}

    rows = []
    for arm in ARMS:
        cs = {f"{c}/{s}": ap(arm, c, s) for c, s in cells}
        gaps = [cs[f"{c}/{s}"] - bar[(c, s)] for c, s in cells]
        rows.append({"arm": arm, "cells": cs, "worst_gap": float(min(gaps)),
                     "sum_gap": float(sum(gaps))})
    ranked = sorted(rows, key=lambda r: (-r["worst_gap"], -r["sum_gap"]))

    L = ["# Soft absolute-trust fusion vs the hard veto (ship AP)", "",
         "Every arm uses the same caches, the same homography, the same WBF and "
         "the same `iou_thr`. The only thing that changes is how the gate's "
         "per-frame estimate reaches the fused scores: as a switch that removes a "
         "stream, or as a multiplier on that stream's confidences.", "",
         f"Evidence novelty bound: windowed (w={EV_WINDOW}) `sum_conf` < "
         f"{ev_thr:.4f}, the minimum over CLEAN frames of {list(FIT_RUNS)}.", "",
         "`bar` = max(VIS, IR) per cell — what an arm must beat to have earned "
         "the fusion. Ranked by WORST cell.", "",
         "| arm | " + " | ".join(f"{c}/{s}" for c, s in cells) + " | worst gap |",
         "|---|" + "---:|" * (len(cells) + 1)]
    L.append("| _bar (max VIS, IR)_ | " + " | ".join(f"{bar[(c, s)]:.4f}" for c, s in cells)
             + " | — |")
    L.append("| _VIS only_ | " + " | ".join(f"{ap('_vis', c, s):.4f}" for c, s in cells)
             + " | — |")
    L.append("| _IR only_ | " + " | ".join(f"{ap('_ir', c, s):.4f}" for c, s in cells)
             + " | — |")
    for r in ranked:
        L.append(f"| {r['arm']} | " + " | ".join(f"{r['cells'][f'{c}/{s}']:.4f}" for c, s in cells)
                 + f" | {r['worst_gap']:+.4f} |")

    L += ["", "## Gap to bar per cell", "",
          "| arm | " + " | ".join(f"{c}/{s}" for c, s in cells) + " |",
          "|---|" + "---:|" * len(cells)]
    for r in ranked:
        L.append(f"| {r['arm']} | "
                 + " | ".join(f"{r['cells'][f'{c}/{s}'] - bar[(c, s)]:+.4f}" for c, s in cells)
                 + " |")

    boot = []
    if args.n_boot:
        best = ranked[0]["arm"]
        for cond, s in cells:
            sel = np.flatnonzero(splits[s])
            a = [parts[(best, cond)][i] for i in sel]
            b = [parts[("hard veto [adopted]", cond)][i] for i in sel]
            boot.append({"cell": f"{cond}/{s}",
                         **bootstrap_delta(a, b, None, n_boot=args.n_boot, cls=SHIP)})
        L += ["", f"## `{best}` vs `hard veto [adopted]`, paired bootstrap "
              f"(n={args.n_boot}, ship AP)", "",
              "| cell | best | adopted | delta | 95% CI |", "|---|---:|---:|---:|---|"]
        for b in boot:
            L.append(f"| {b['cell']} | {b['a']:.4f} | {b['b']:.4f} | {b['delta']:+.4f} | "
                     f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"
                     f"{' (spans 0)' if b['spans_zero'] else ''} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"ev_thr": ev_thr, "ev_window": EV_WINDOW,
         "bar": {f"{c}/{s}": bar[(c, s)] for c, s in cells},
         "vis": {f"{c}/{s}": ap("_vis", c, s) for c, s in cells},
         "ir": {f"{c}/{s}": ap("_ir", c, s) for c, s in cells},
         "arms": ranked, "trust_medians": {f"{k[0]} | {k[1]}": v for k, v in extra.items()},
         "bootstrap": boot}, indent=2), encoding="utf-8")
    print(f"[soft] wrote {out} in {time.time() - t0:.0f}s")
    for r in ranked:
        print(f"[soft] {r['worst_gap']:+.4f}  {r['arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
