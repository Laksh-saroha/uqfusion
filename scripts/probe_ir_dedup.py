"""Can the IR stream itself be improved? It is 5 of the 8 cells under `crossmodal`.

With `single_passthrough` on, a fully vetoed cell returns the IR stream verbatim —
so all four night cells and fog/day are EXACTLY `ir_only`, and any improvement to
the IR detections is a direct, one-for-one improvement to five of the eight cells.
That makes the IR stream the highest-leverage remaining target now that the veto
control surface is measured to be within +0.0007 of its own ceiling everywhere.

The lever available without retraining is duplicate suppression. IR boxes reach
fusion straight from the detector's NMS, and the earlier single-list WBF -- which
merged them at `iou_thr` and averaged the cluster's scores -- was worth **+0.0003
at night** while costing 0.0011 on a day cell. That says merging genuinely helps
the IR stream and the day loss came from something else in the same post-process
(canvas clipping). If so, a dedup applied on its own, at a threshold chosen on the
fit runs, should collect the night gain without the day cost.

Two operators are tried, because they differ in what they do to the surviving box:

  nms    keep the highest-scoring box of a cluster, drop the rest
  wbf    replace the cluster with its score-weighted average box and mean score

**Selection on clean fit-run frames only** (pohang00/02/03), as everywhere else
here; the eight-cell columns are a report. A threshold picked on the night cells
would be fitted on the held-out run.

Measurement only; writes one report.

Usage:
    python scripts/probe_ir_dedup.py --out runs/eval/ir_dedup.md
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
from uqfusion.eval.ctx import FIT_RUNS, NIGHT_RUNS, load_context                 # noqa: E402
from uqfusion.uq.fusion import apply_homography                                  # noqa: E402

SHIP = 0


def _iou(boxes: np.ndarray, b: np.ndarray) -> np.ndarray:
    xa = np.maximum(boxes[:, 0], b[0]); ya = np.maximum(boxes[:, 1], b[1])
    xb = np.minimum(boxes[:, 2], b[2]); yb = np.minimum(boxes[:, 3], b[3])
    inter = np.maximum(xb - xa, 0) * np.maximum(yb - ya, 0)
    aa = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    ab = (b[2] - b[0]) * (b[3] - b[1])
    return inter / np.maximum(aa + ab - inter, 1e-9)


def dedup(rec: dict, thr: float, mode: str) -> dict:
    """Greedy cluster-and-collapse within one stream, per class."""
    b = np.asarray(rec["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
    s = np.asarray(rec["conf"], dtype=np.float64).reshape(-1)
    c = np.asarray(rec["cls"]).reshape(-1)
    if len(s) < 2:
        return rec
    # sigma travels with its box: `compute_reliability` indexes sigma_ltrb against
    # boxes_xyxy, so a record whose boxes were collapsed while sigma was not is a
    # shape mismatch waiting to happen the first time a caller asks for r_box.
    sg = np.asarray(rec.get("sigma_ltrb", np.zeros((len(s), 4))),
                    dtype=np.float64).reshape(-1, 4)
    ob, os_, oc, og = [], [], [], []
    for cl in np.unique(c):
        m = np.flatnonzero(c == cl)
        order = m[np.argsort(-s[m], kind="stable")]
        kept_b, kept_s, members = [], [], []
        for i in order:
            if kept_b:
                ious = _iou(np.asarray(kept_b), b[i])
                j = int(np.argmax(ious))
                if ious[j] > thr:
                    members[j].append(i)
                    if mode == "wbf":
                        idx = members[j]
                        w = s[idx][:, None]
                        kept_b[j] = (b[idx] * w).sum(0) / max(w.sum(), 1e-9)
                        kept_s[j] = float(s[idx].mean())
                    continue
            kept_b.append(b[i].copy()); kept_s.append(float(s[i])); members.append([i])
        ob += kept_b; os_ += kept_s; oc += [int(cl)] * len(kept_b)
        og += [sg[m[0]] for m in members]
    return {**rec, "boxes_xyxy": np.asarray(ob).reshape(-1, 4),
            "conf": np.asarray(os_), "cls": np.asarray(oc, dtype=int),
            "sigma_ltrb": np.asarray(og).reshape(-1, 4)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/ir_dedup.md")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--thresholds", type=float, nargs="+",
                    default=[0.50, 0.60, 0.70, 0.80, 0.85, 0.90, 0.95])
    args = ap.parse_args()

    t0 = time.time()
    ctx = load_context(preset="crossmodal", conditions=("clean",))
    night = np.isin(ctx.runs, NIGHT_RUNS)
    fit_sel = np.flatnonzero(np.isin(ctx.runs, FIT_RUNS) & ~night)
    splits = {"fit-clean(day)": fit_sel, "all day": np.flatnonzero(~night),
              "night": np.flatnonzero(night)}

    ir_vis = [{**r, "boxes_xyxy": apply_homography(
        np.asarray(r["boxes_xyxy"]).reshape(-1, 4), h)}
        for r, h in zip(ctx.ir_clean, ctx.h_frames)]

    def score(parts, sel):
        r = ap_from_parts([parts[i] for i in sel])
        e = r["per_class"].get(SHIP)
        return float(e["ap50_95"]) if e else 0.0

    base_parts = frame_parts(ir_vis, ctx.gts)
    rows = [{"mode": "none", "thr": None,
             **{k: score(base_parts, v) for k, v in splits.items()},
             "mean_boxes": float(np.mean([len(r["conf"]) for r in ir_vis]))}]
    parts_by = {("none", None): base_parts}
    for mode in ("nms", "wbf"):
        for thr in args.thresholds:
            dd = [dedup(r, thr, mode) for r in ir_vis]
            p = frame_parts(dd, ctx.gts)
            parts_by[(mode, thr)] = p
            rows.append({"mode": mode, "thr": thr,
                         **{k: score(p, v) for k, v in splits.items()},
                         "mean_boxes": float(np.mean([len(r["conf"]) for r in dd]))})
        print(f"[dedup] {mode} done ({time.time() - t0:.0f}s)", flush=True)

    base = rows[0]
    pick = max(rows, key=lambda r: r["fit-clean(day)"])

    L = ["# IR-stream duplicate suppression — selection on fit-run clean only", "",
         "IR ship AP, boxes mapped into the VIS plane (the frame everything is "
         "scored in). Under `crossmodal` with `single_passthrough`, all four night "
         "cells and fog/day are EXACTLY this stream, so a gain here is a gain on "
         "five of the eight cells.", "",
         "**Only the `fit-clean(day)` column may select.** `night` is the held-out "
         "run and is shown as a report; picking a threshold on it would fit the "
         "held-out data.", "",
         f"n: fit-clean {len(fit_sel)}, all day {int((~night).sum())}, night "
         f"{int(night.sum())}.", "",
         "| operator | IoU | **fit-clean(day)** | Δ fit | all day | night | mean boxes/frame |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        thr_s = "—" if r["thr"] is None else f"{r['thr']:.2f}"
        L.append(f"| {r['mode']} | {thr_s} | **{r['fit-clean(day)']:.4f}** | "
                 f"{r['fit-clean(day)'] - base['fit-clean(day)']:+.4f} | "
                 f"{r['all day']:.4f} | {r['night']:.4f} | {r['mean_boxes']:.1f} |")

    boot = None
    if pick["mode"] != "none" and args.n_boot:
        p = parts_by[(pick["mode"], pick["thr"])]
        boot = {k: bootstrap_delta([p[i] for i in v], [base_parts[i] for i in v],
                                   None, n_boot=args.n_boot, cls=SHIP)
                for k, v in splits.items()}
        L += ["", f"## `{pick['mode']} @ {pick['thr']:.2f}` vs raw IR — paired bootstrap "
              f"(n={args.n_boot}, seed 0)", "",
              "| split | dedup | raw | delta | 95% CI |", "|---|---:|---:|---:|---|"]
        for k, b in boot.items():
            L.append(f"| {k} | {b['a']:.4f} | {b['b']:.4f} | {b['delta']:+.4f} | "
                     f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"
                     f"{' (spans 0)' if b['spans_zero'] else ''} |")

    L += ["", "## Verdict", "",
          f"- Best on the selection set: **{pick['mode']} @ "
          f"{'—' if pick['thr'] is None else format(pick['thr'], '.2f')}**, "
          f"{pick['fit-clean(day)']:.4f} vs raw {base['fit-clean(day)']:.4f} "
          f"({pick['fit-clean(day)'] - base['fit-clean(day)']:+.4f}).",
          f"- Same arm on the held-out night run: {pick['night']:.4f} vs "
          f"{base['night']:.4f} ({pick['night'] - base['night']:+.4f}).",
          "",
          "Adopt only if the selection-set gain is real (CI excluding zero) AND the "
          "held-out night column agrees in sign. A gain on the selection set that "
          "reverses on the held-out run is the classic sign of a threshold fitted "
          "to scene texture rather than to duplicate structure."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"rows": rows, "pick": pick,
         "bootstrap": {k: {kk: vv for kk, vv in v.items()} for k, v in (boot or {}).items()}},
        indent=2), encoding="utf-8")
    print(f"[dedup] wrote {out} in {time.time() - t0:.0f}s")
    print(f"[dedup] pick {pick['mode']} @ {pick['thr']}  fit "
          f"{pick['fit-clean(day)']:.4f} ({pick['fit-clean(day)'] - base['fit-clean(day)']:+.4f})  "
          f"night {pick['night']:.4f} ({pick['night'] - base['night']:+.4f})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
