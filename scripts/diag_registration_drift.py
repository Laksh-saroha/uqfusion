"""Is the registration residual stationary WITHIN a recording run?

A2 (per-run registration refinement) is the declared blocker in §9.2: 5.4 px
median error on 12-17 px boxes makes IoU 0.85 between a VIS box and its true IR
counterpart nearly unreachable, which is why σ-weighted WBF measured null (§5.2)
and why the `iou_thr` sweep tuned toward "fuse less" (§5.4).

The planned fix is a **per-run translation term**, chosen because a single global
bias correction failed run-disjoint validation with the offset swinging
dx +0.17 -> -4.01 px between run pairs. That failure says the residual is not
constant ACROSS runs. It says nothing about whether it is constant WITHIN one —
and if it drifts with time (vessel motion, vibration, thermal expansion of the
mount), a per-run constant fails for exactly the same reason the global one did,
one level down. That would invalidate the whole §9.2 sequence before it starts.

Measured on **labels, not detections**: IR GT boxes are mapped through the per-run
homography and matched to VIS GT boxes. Detector noise, missed detections and
false positives are all irrelevant to where the two sensors' geometry disagrees,
and including them would confound a registration measurement with a detection
one.

Reported per run: the residual's median and spread, its trend against capture
time, and — the decisive number — how much of the residual a per-run constant can
actually remove versus a time-varying one.

Usage:
    python scripts/diag_registration_drift.py [--iou-min 0.1] [--bins 10]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.ctx import load_context  # noqa: E402
from uqfusion.eval.matching import iou_matrix, load_gt  # noqa: E402
from uqfusion.uq.fusion import apply_homography  # noqa: E402


def centres(b: np.ndarray) -> np.ndarray:
    b = np.asarray(b, dtype=float).reshape(-1, 4)
    return np.column_stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2])


def frame_number(path: str) -> int:
    m = re.search(r"(\d+)$", Path(path).stem)
    if not m:
        raise ValueError(f"cannot recover a frame number from {path}")
    return int(m.group(1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iou-min", type=float, default=0.10,
                    help="minimum IoU after mapping for a GT pair to count as the same object")
    ap.add_argument("--bins", type=int, default=10, help="time bins per run")
    ap.add_argument("--out", default="runs/eval/x_registration_drift.md")
    args = ap.parse_args()

    t0 = time.time()
    ctx = load_context(conditions=("clean",))

    rows = []   # one per matched GT pair
    for i, (rv, ri, h) in enumerate(zip(ctx.vis_by_cond["clean"], ctx.ir_clean, ctx.h_frames)):
        gv = ctx.gts[i]
        gi = load_gt(ri["image_path"], ri["image_hw"])
        if not len(gv["cls"]) or not len(gi["cls"]):
            continue
        ib = apply_homography(np.asarray(gi["boxes_xyxy"], dtype=float).reshape(-1, 4), h)
        vb = np.asarray(gv["boxes_xyxy"], dtype=float).reshape(-1, 4)
        M = iou_matrix(ib, vb) * (np.asarray(gi["cls"])[:, None] == np.asarray(gv["cls"])[None, :])
        cv, ci_ = centres(vb), centres(ib)
        used = np.zeros(len(vb), dtype=bool)
        for a in np.argsort(-M.max(axis=1)):
            j = int(np.argmax(np.where(used, -1.0, M[a])))
            if used[j] or M[a, j] < args.iou_min:
                continue
            used[j] = True
            rows.append({
                "run": Path(rv["image_path"]).parent.name,
                "frame": frame_number(rv["image_path"]),
                "dx": float(cv[j, 0] - ci_[a, 0]),
                "dy": float(cv[j, 1] - ci_[a, 1]),
                "iou": float(M[a, j]),
            })
    print(f"[drift] {len(rows)} matched GT pairs at IoU>={args.iou_min} ({time.time() - t0:.0f}s)")
    if not rows:
        raise SystemExit("no GT pairs matched — check the homography or lower --iou-min")

    runs = sorted({r["run"] for r in rows})
    arr = {run: {k: np.asarray([r[k] for r in rows if r["run"] == run])
                 for k in ("frame", "dx", "dy", "iou")} for run in runs}

    L = ["# Registration residual — is it stationary within a run?", "",
         f"IR **GT** boxes mapped through the per-run homography and matched to VIS **GT** "
         f"(IoU >= {args.iou_min}, same class). {len(rows)} matched pairs across {len(runs)} runs. "
         f"Labels, not detections: this measures geometry, not the detector.",
         "",
         "## 1. Residual per run", "",
         "| run | pairs | median dx | median dy | median &#124;d&#124; | IQR dx | IQR dy |",
         "|---|---:|---:|---:|---:|---:|---:|"]
    for run in runs:
        a = arr[run]
        mag = np.hypot(a["dx"], a["dy"])
        L.append(f"| {run} | {len(a['dx'])} | {np.median(a['dx']):+.2f} | {np.median(a['dy']):+.2f} | "
                 f"{np.median(mag):.2f} | {np.subtract(*np.percentile(a['dx'], [75, 25])):.2f} | "
                 f"{np.subtract(*np.percentile(a['dy'], [75, 25])):.2f} |")

    L += ["", "## 2. Drift within each run", "",
          f"Frames binned into {args.bins} equal-count bins by capture order. If a per-run "
          f"constant is the right model, these columns are flat.",
          "",
          "| run | " + " | ".join(f"b{i+1}" for i in range(args.bins)) + " | swing |",
          "|---|" + "---:|" * (args.bins + 1)]
    drift = {}
    for run in runs:
        a = arr[run]
        o = np.argsort(a["frame"], kind="mergesort")
        f, dx = a["frame"][o], a["dx"][o]
        edges = np.linspace(0, len(dx), args.bins + 1).astype(int)
        meds = [float(np.median(dx[edges[i]:edges[i + 1]])) if edges[i + 1] > edges[i] else np.nan
                for i in range(args.bins)]
        swing = float(np.nanmax(meds) - np.nanmin(meds))
        drift[run] = {"bins_dx": meds, "swing_dx": swing}
        L.append(f"| {run} | " + " | ".join(f"{m:+.2f}" if np.isfinite(m) else "—" for m in meds)
                 + f" | **{swing:.2f}** |")

    L += ["", "### 2.1 Same, for dy", "",
          "| run | " + " | ".join(f"b{i+1}" for i in range(args.bins)) + " | swing |",
          "|---|" + "---:|" * (args.bins + 1)]
    for run in runs:
        a = arr[run]
        o = np.argsort(a["frame"], kind="mergesort")
        dy = a["dy"][o]
        edges = np.linspace(0, len(dy), args.bins + 1).astype(int)
        meds = [float(np.median(dy[edges[i]:edges[i + 1]])) if edges[i + 1] > edges[i] else np.nan
                for i in range(args.bins)]
        drift[run]["bins_dy"] = meds
        drift[run]["swing_dy"] = float(np.nanmax(meds) - np.nanmin(meds))
        L.append(f"| {run} | " + " | ".join(f"{m:+.2f}" if np.isfinite(m) else "—" for m in meds)
                 + f" | **{drift[run]['swing_dy']:.2f}** |")

    L += ["", "## 3. What a correction could actually remove", "",
          "Median absolute residual after subtracting: nothing, one global constant, one "
          "constant per run (the A2 plan), and one constant per time bin within a run "
          "(a time-varying model). The gap between the last two columns is what a per-run "
          "constant leaves on the table.",
          "",
          "| run | raw | global | per-run | per-bin | per-run leaves |",
          "|---|---:|---:|---:|---:|---:|"]
    all_dx = np.concatenate([arr[r]["dx"] for r in runs])
    all_dy = np.concatenate([arr[r]["dy"] for r in runs])
    gx, gy = float(np.median(all_dx)), float(np.median(all_dy))
    summary = {}
    for run in runs:
        a = arr[run]
        o = np.argsort(a["frame"], kind="mergesort")
        dx, dy = a["dx"][o], a["dy"][o]
        raw = float(np.median(np.hypot(dx, dy)))
        glob = float(np.median(np.hypot(dx - gx, dy - gy)))
        rx, ry = float(np.median(dx)), float(np.median(dy))
        perrun = float(np.median(np.hypot(dx - rx, dy - ry)))
        edges = np.linspace(0, len(dx), args.bins + 1).astype(int)
        res = []
        for i in range(args.bins):
            s = slice(edges[i], edges[i + 1])
            if edges[i + 1] <= edges[i]:
                continue
            res.append(np.hypot(dx[s] - np.median(dx[s]), dy[s] - np.median(dy[s])))
        perbin = float(np.median(np.concatenate(res))) if res else np.nan
        summary[run] = {"raw": raw, "global": glob, "per_run": perrun, "per_bin": perbin}
        L.append(f"| {run} | {raw:.2f} | {glob:.2f} | {perrun:.2f} | {perbin:.2f} | "
                 f"{perrun - perbin:+.2f} px |")

    worst = max(runs, key=lambda r: drift[r]["swing_dx"])
    L += ["", "## 4. Reading", "",
          f"Largest within-run dx swing: **{worst}, {drift[worst]['swing_dx']:.2f} px** across "
          f"{args.bins} time bins. Compare against the -4.18 px between-run swing that killed the "
          f"global correction (§9.2 item 6). If the within-run swing is of the same order, a "
          f"per-run constant is the same mistake one level down and A2 needs a time-varying term; "
          f"if it is small, the per-run plan is sound and this run de-risks it."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"n_pairs": len(rows), "iou_min": args.iou_min, "bins": args.bins,
         "drift": drift, "residual": summary,
         "global_dx": gx, "global_dy": gy}, indent=2), encoding="utf-8")
    print(f"[drift] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
