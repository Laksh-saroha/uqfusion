"""Cross-modal score calibration — the test that follows from fusion being concatenation.

§5.3 of the experiment record measured that only **31 of 29,042** VIS detections
have an IR partner above `iou_thr` 0.85. Fusion is therefore ~99.9% concatenation
of two detection lists, and mAP is rank-based over the concatenated list. That
makes one property decisive and currently untested: **are VIS and IR confidences
comparable on a common scale?** They come from two independently trained
checkpoints and there is no reason they should be. The capability prior is a
single scalar per modality attempting a job that needs a monotone recalibration.

The test: fit `conf -> P(true positive)` per modality by isotonic regression
(pool-adjacent-violators, no sklearn dependency) on the FIT runs only, apply it
to both streams, and re-run the adopted system.

The control is built in. **Isotonic regression is monotone, so per-modality AP
cannot change** — `visible_only` and `ir_only` are invariant by construction.
Any movement in the fused row is therefore cross-modal interleaving and nothing
else. The script asserts that invariance rather than assuming it.

Protocol: the calibrator is fitted on pohang00/02/03 clean and pohang01 is never
seen, matching §7. Both a fit-run-only and a held-out-run number are reported.

Usage:
    python scripts/eval_score_calibration.py [--out runs/eval/x_score_calibration.md]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems  # noqa: E402


def isotonic_fit(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pool-adjacent-violators. Returns (sorted x, fitted y) defining a step map."""
    order = np.argsort(x, kind="mergesort")
    xs, ys = x[order].astype(float), y[order].astype(float)
    # PAV over blocks of equal weight
    vals, wts = [], []
    for v in ys:
        vals.append(v)
        wts.append(1.0)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2 = vals.pop(), wts.pop()
            v1, w1 = vals.pop(), wts.pop()
            vals.append((v1 * w1 + v2 * w2) / (w1 + w2))
            wts.append(w1 + w2)
    fitted = np.repeat(vals, [int(w) for w in wts])
    return xs, fitted


def apply_iso(xs: np.ndarray, ys: np.ndarray, q: np.ndarray) -> np.ndarray:
    """Evaluate the fitted step map, clamped at the fitted range.

    PAV pools adjacent violators into blocks, so the fitted map is monotone but
    NOT strictly monotone: every raw confidence inside a block lands on the same
    value. Those ties reorder detections within a modality, and since AP is
    rank-based that moves `visible_only` — which is supposed to be invariant, and
    is the control this test depends on. (The pre-flight caught exactly this:
    0.25804683 -> 0.25706506.)

    A tie-break of `eps * raw_conf` restores the within-block order. `eps` is half
    the smallest gap between distinct fitted values and raw conf is in [0, 1], so
    the correction can never reach the next block: cross-block order is the
    calibration's, within-block order is the detector's, and neither is invented.

    It also removes exact zeros. Isotonic legitimately fits P(TP) = 0 for a
    low-confidence block, and a zero-confidence box makes WBF divide by the
    cluster's summed score and emit NaN coordinates.
    """
    if len(xs) == 0:
        return np.asarray(q, dtype=float)
    base = np.interp(q, xs, ys, left=ys[0], right=ys[-1])
    distinct = np.unique(ys)
    gap = float(np.min(np.diff(distinct))) if len(distinct) > 1 else 1e-6
    eps = 0.5 * gap
    return np.maximum(base + eps * np.asarray(q, dtype=float), 1e-9)


def tp_flags(records, gts, sel) -> tuple[np.ndarray, np.ndarray]:
    """(conf, is_tp@0.5) over the selected frames — the calibration training set."""
    parts = frame_parts([records[i] for i in sel], [gts[i] for i in sel])
    conf = np.concatenate([p["conf"] for p in parts if len(p["conf"])]) if parts else np.zeros(0)
    tp = np.concatenate([p["tp"][:, 0] for p in parts if len(p["conf"])]) if parts else np.zeros(0, bool)
    return conf, tp


def recalibrate(records, xs, ys) -> list[dict]:
    return [{**r, "conf": apply_iso(xs, ys, np.asarray(r["conf"], dtype=float))} for r in records]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="runs/eval/x_score_calibration.md")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict the condition sweep (pre-flight uses --conditions clean)")
    args = ap.parse_args()

    t0 = time.time()
    # Recorded under the 2026-08-19 adopted system; pinned so re-runs keep
    # reproducing runs/eval/x_score_calibration.md.
    ctx = load_context(capability_sel="all", bright_soft=True, veto_filter=None,
                       **({'conditions': tuple(args.conditions)} if args.conditions else {}))
    fit_sel = ctx.sel("fit")
    splits = {"day": ctx.sel("day"), "night": ctx.sel("night")}

    # IR boxes must be in the VIS frame before they can be scored against VIS GT.
    base = run_systems(ctx, "clean")
    ir_in_vis = base["ir_in_vis"]

    cv, tv = tp_flags(ctx.vis_by_cond["clean"], ctx.gts, fit_sel)
    ci, ti = tp_flags(ir_in_vis, ctx.gts, fit_sel)
    xs_v, ys_v = isotonic_fit(cv, tv)
    xs_i, ys_i = isotonic_fit(ci, ti)
    print(f"[cal] fitted on {len(fit_sel)} fit-run frames: "
          f"VIS {len(cv)} dets (TP rate {tv.mean():.4f}), IR {len(ci)} dets (TP rate {ti.mean():.4f})")

    # How far apart were the two scales? Report the calibrated value of a few
    # raw confidences — this is the quantity the capability prior approximates.
    probe = np.array([0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9])
    pv, pi = apply_iso(xs_v, ys_v, probe), apply_iso(xs_i, ys_i, probe)

    ir_cal = recalibrate(ctx.ir_clean, xs_i, ys_i)
    rows, boots, parts_keep = [], {}, {}
    for cond in ctx.conditions:
        vis_cal = recalibrate(ctx.vis_by_cond[cond], xs_v, ys_v)
        res_b = run_systems(ctx, cond)
        res_c = run_systems(ctx, cond, vis_records=vis_cal, ir_records=ir_cal)
        pb = frame_parts(res_b["fused_gated"], ctx.gts)
        pc = frame_parts(res_c["fused_gated"], ctx.gts)
        parts_keep[cond] = (pb, pc)
        # invariance control: a monotone map cannot change single-modality AP
        for name, rb, rc in (("visible_only", ctx.vis_by_cond[cond], vis_cal),
                             ("ir_only", res_b["ir_in_vis"], res_c["ir_in_vis"])):
            a = ap_from_parts(frame_parts(rb, ctx.gts))["map50_95"]
            b = ap_from_parts(frame_parts(rc, ctx.gts))["map50_95"]
            assert abs(a - b) < 1e-9, (f"{name} moved under a monotone recalibration "
                                       f"({a:.8f} -> {b:.8f}) — the map is not monotone")
        for sname, sel in splits.items():
            rows.append({"condition": cond, "split": sname,
                         "baseline": ap_from_parts(pb, sel)["map50_95"],
                         "calibrated": ap_from_parts(pc, sel)["map50_95"]})
        print(f"[cal] {cond:9s} " + "  ".join(
            f"{r['split']}: {r['baseline']:.4f} -> {r['calibrated']:.4f}"
            for r in rows if r["condition"] == cond), flush=True)

    for cond in ctx.conditions:
        pb, pc = parts_keep[cond]
        for sname, sel in splits.items():
            boots[(cond, sname)] = bootstrap_delta(pc, pb, sel, n_boot=args.n_boot)
    print(f"[cal] bootstrap done ({time.time() - t0:.0f}s)", flush=True)

    L = ["# Cross-modal score calibration", "",
         f"Isotonic `conf -> P(TP@0.5)` fitted per modality on the fit runs "
         f"({len(fit_sel)} frames, pohang00/02/03 clean) and applied to both streams. "
         f"pohang01 never seen by the fit.",
         "",
         "**Control:** isotonic regression is monotone, so `visible_only` and `ir_only` are "
         "invariant under it — asserted for every condition. Any change below is cross-modal "
         "interleaving, which is the whole mechanism when fusion is 99.9% concatenation (§5.3).",
         "",
         "## 1. The two score scales",
         "",
         "Calibrated P(TP) at the same raw confidence. If these columns differ, the two "
         "detectors' scores were never comparable and concatenating them ranks by the wrong key.",
         "",
         "| raw conf | VIS -> P(TP) | IR -> P(TP) | ratio |",
         "|---:|---:|---:|---:|"]
    for q, a, b in zip(probe, pv, pi):
        L.append(f"| {q:.2f} | {a:.4f} | {b:.4f} | {a / max(b, 1e-9):.1f}x |")

    L += ["", "## 2. Effect on gated fusion", "",
          "| condition | split | baseline | calibrated | delta | 95% CI | sign flips |",
          "|---|---|---:|---:|---:|---|---:|"]
    for r in rows:
        b = boots[(r["condition"], r["split"])]
        L.append(f"| {r['condition']} | {r['split']} | {r['baseline']:.4f} | {r['calibrated']:.4f} | "
                 f"{b['delta']:+.4f} | [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {b['p_sign_flip']:.1%} |")
    L += ["", f"Fit runs are pohang00/02/03; the `night` split is pohang01 and is held out of "
              f"the calibration fit, so those rows are the generalisation test."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({
        "rows": rows,
        "bootstrap": {"|".join(k): v for k, v in boots.items()},
        "probe": probe.tolist(), "vis_calibrated": pv.tolist(), "ir_calibrated": pi.tolist(),
    }, indent=2), encoding="utf-8")
    print(f"[cal] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
