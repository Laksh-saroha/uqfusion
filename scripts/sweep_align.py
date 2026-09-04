"""Fix the registration first, then see what the cross-modal terms are worth.

`x_registration_drift.md` measured a 3-6 px median residual after the adopted
per-run homography, swinging 5-10 px within each run, of which about 1.5 px is
recoverable by a time-varying model. `probe_detector_swap.py` measured what that
residual costs: at `iou_thr` 0.85 only **0.05%** of VIS detections have a same-class
IR partner. A few pixels on a 30 px ship box is the difference between IoU 0.3 and
IoU 0.6, so the residual is not a rounding detail here -- it is the reason every
cross-modal mechanism in the system has almost nothing to act on.

The correction is estimated per frame from the two DETECTION sets (median offset of
nearest-centre same-class pairs), never from GT, so it is a thing the deployed
system could actually do. See `uqfusion.eval.iralign`.

Three questions, in order:

  1. Does the alignment move the AGREEMENT RATE? That is the mechanism, and it is
     measurable independently of AP. If it does not move, nothing below can.
  2. Does alignment alone change AP? On its own it should do very little: at 0.85
     the streams still will not meet, and fusion is ~99.9% concatenation.
  3. Does alignment plus `support` beat `support` alone? This is the real question.
     Support acts on loose overlap; a better registration is exactly what makes more
     of that overlap real rather than coincidental.

Selection on `TUNE_RUNS` (pohang00 day), headline on `TEST_RUNS` (pohang02+03 day),
run-disjoint. Night is reported and must not move: VIS is 0.0000 there, so there is
no second stream to align to.

Usage:
    python scripts/sweep_align.py --cache-dir runs/cache_m --out runs/eval/align_26m.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import (NIGHT_RUNS, TEST_RUNS, TUNE_RUNS,                 # noqa: E402
                               load_context, run_systems)
from uqfusion.eval.iralign import aligned_homographies                           # noqa: E402
from uqfusion.uq.fusion import apply_homography                                  # noqa: E402

SHIP = 0
THRS = (0.85, 0.55, 0.30, 0.10)


def agreement(vis_records, ir_records, h_frames, sel) -> dict:
    """Share of VIS detections with a same-class IR detection above each IoU."""
    hit = {t: 0 for t in THRS}
    tot = 0
    for k in sel:
        rv, ri, h = vis_records[k], ir_records[k], h_frames[k]
        v = np.asarray(rv["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
        if not len(v):
            continue
        b = apply_homography(np.asarray(ri["boxes_xyxy"], dtype=np.float64).reshape(-1, 4), h)
        tot += len(v)
        if not len(b):
            continue
        x1 = np.maximum(v[:, None, 0], b[None, :, 0])
        y1 = np.maximum(v[:, None, 1], b[None, :, 1])
        x2 = np.minimum(v[:, None, 2], b[None, :, 2])
        y2 = np.minimum(v[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        av = np.clip(v[:, 2] - v[:, 0], 0, None) * np.clip(v[:, 3] - v[:, 1], 0, None)
        ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
        m = inter / np.maximum(av[:, None] + ab[None, :] - inter, 1e-12)
        m = np.where(np.asarray(rv["cls"]).astype(int)[:, None]
                     == np.asarray(ri["cls"]).astype(int)[None, :], m, 0.0)
        best = m.max(axis=1)
        for t in THRS:
            hit[t] += int((best >= t).sum())
    return {t: (hit[t] / tot if tot else 0.0) for t in THRS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/align_26m.md")
    ap.add_argument("--condition", default="clean")
    ap.add_argument("--support-iou", type=float, default=0.30)
    ap.add_argument("--support-gamma", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=500)
    args = ap.parse_args()
    t0 = time.time()

    base = load_context(preset="crossmodal", cache_dir=args.cache_dir,
                        conditions=(args.condition,), verbose=True)
    night = np.isin(base.runs, NIGHT_RUNS)
    day = np.flatnonzero(~night)
    tune = np.flatnonzero(np.isin(base.runs, TUNE_RUNS) & ~night)
    test = np.flatnonzero(np.isin(base.runs, TEST_RUNS) & ~night)
    nightsel = np.flatnonzero(night)
    vis = base.vis_by_cond[args.condition]

    # ---- 1. the mechanism, before any AP is computed ------------------------
    aligns, agree, diag = {}, {}, {}
    agree["none (adopted homography)"] = agreement(vis, base.ir_clean, base.h_frames, day)
    for r in (0.75, 1.0, 1.5):
        h, d = aligned_homographies(vis, base.ir_clean, base.h_frames, radius=r)
        aligns[r] = h
        diag[r] = d
        agree[f"align radius {r:g}"] = agreement(vis, base.ir_clean, h, day)
        print(f"[algn] radius {r:g}: applied on {d['applied']:.1%} of frames, "
              f"median |shift| {np.median(np.hypot(d['dx'], d['dy'])[d['pairs'] >= 3]):.2f} px "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- 2/3. AP ------------------------------------------------------------
    arms = [("adopted", base),
            (f"support i{args.support_iou:g} g{args.support_gamma:g}",
             replace(base, support_iou=args.support_iou, support_gamma=args.support_gamma))]
    for r, h in aligns.items():
        arms.append((f"align r{r:g}", replace(base, h_frames=h)))
        arms.append((f"align r{r:g} + support", replace(
            base, h_frames=h, support_iou=args.support_iou, support_gamma=args.support_gamma)))
    # With registration corrected, a TIGHTER cluster threshold becomes defensible
    # again -- the sweep that rejected 0.55 ran against the uncorrected geometry.
    r_best = 1.0
    for t in (0.55, 0.75):
        arms.append((f"align r1 + iou {t}", replace(base, h_frames=aligns[r_best], iou_thr=t)))

    def apf(parts, sel):
        e = ap_from_parts([parts[i] for i in sel])
        s = e["per_class"].get(SHIP)
        return (float(s["ap50_95"]) if s else 0.0, float(e["map50_95"]))

    rows, parts_by = [], {}
    for name, ctx in arms:
        r = run_systems(ctx, args.condition)
        p = frame_parts(r["fused_gated"], ctx.gts)
        parts_by[name] = p
        us, _ = apf(p, tune)
        ts, tm = apf(p, test)
        ds, dm = apf(p, day)
        ns, _ = apf(p, nightsel)
        rows.append({"arm": name, "tune": us, "test": ts, "test_macro": tm,
                     "day": ds, "day_macro": dm, "night": ns})
        print(f"[algn] {name:26} tune {us:.4f}  TEST {ts:.4f}  day {ds:.4f}  "
              f"night {ns:.4f} ({time.time() - t0:.0f}s)", flush=True)

    b0 = rows[0]
    best = max(rows[1:], key=lambda r: r["tune"])
    boot = {}
    if args.n_boot:
        for tag, sel in (("tune", tune), ("test", test), ("day", day)):
            boot[tag] = bootstrap_delta([parts_by[best["arm"]][k] for k in sel],
                                        [parts_by["adopted"][k] for k in sel],
                                        None, n_boot=args.n_boot, cls=SHIP)

    L = [f"# Registration first — `{args.cache_dir}`, `{args.condition}`", "",
         "The IR->VIS offset is estimated **per frame from the two detection sets** "
         "(median offset of nearest-centre same-class pairs), never from GT, and "
         "composed onto the existing homography as a pure translation in the VIS "
         "canvas. A frame with fewer than 3 pairs, or a shift over 40 px, keeps the "
         "unmodified homography.", "",
         "## 1. The mechanism — does agreement become geometrically available?", "",
         "Share of VIS detections with a same-class IR detection above each IoU, day "
         "frames. This is measured before any AP and does not depend on the fusion "
         "code at all.", "",
         "| homography | " + " | ".join(f"IoU>={t}" for t in THRS) + " |",
         "|---|" + "---:|" * len(THRS)]
    for k, d in agree.items():
        L.append(f"| {k} | " + " | ".join(f"{d[t]:.2%}" for t in THRS) + " |")
    L += ["", "| radius | frames corrected | median &#124;shift&#124; (px) | median pairs |",
          "|---|---:|---:|---:|"]
    for r, d in diag.items():
        ok = d["pairs"] >= 3
        L.append(f"| {r:g} | {d['applied']:.1%} | "
                 f"{np.median(np.hypot(d['dx'], d['dy'])[ok]) if ok.any() else 0:.2f} | "
                 f"{np.median(d['pairs']):.0f} |")

    L += ["", "## 2/3. What it is worth", "",
          "Ship AP. **tune** = pohang00 day (selection); **TEST** = pohang02+03 day, "
          "run-disjoint and held out; `day` is both, in-sample, shown only because "
          "every earlier table in this project reports that number.", "",
          "| arm | tune | **TEST** | test macro | day | night | tune delta | TEST delta |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['arm']} | {r['tune']:.4f} | **{r['test']:.4f}** | "
                 f"{r['test_macro']:.4f} | {r['day']:.4f} | {r['night']:.4f} | "
                 f"{r['tune'] - b0['tune']:+.4f} | {r['test'] - b0['test']:+.4f} |")

    L += ["", "## Verdict", "",
          f"- Best on the tune runs: **{best['arm']}** "
          f"({best['tune'] - b0['tune']:+.4f}), held-out TEST "
          f"**{best['test'] - b0['test']:+.4f}**."]
    for tag in ("tune", "test", "day"):
        if tag in boot:
            b = boot[tag]
            L.append(f"- {tag}: {b['delta']:+.4f} [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"
                     + (" — **spans zero**" if b["spans_zero"] else ""))
    moved = [r["arm"] for r in rows[1:] if abs(r["night"] - b0["night"]) > 1e-9]
    L.append("- Night moved by: " + (", ".join(moved) if moved else "no arm")
             + " — VIS is 0.0000 there, so an arm that moves night is aligning to noise.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"rows": rows, "agreement": {k: {str(t): v for t, v in d.items()}
                                     for k, d in agree.items()},
         "bootstrap": boot}, indent=2), encoding="utf-8")
    print(f"[algn] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
