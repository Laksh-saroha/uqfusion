"""Which per-box signals actually predict a true positive? Seconds, not hours.

Measured 2026-09-01: temporal support fires on 74% of VIS day boxes and its
**lift is 1.00x** -- P(TP|fired) 0.4507 against P(TP|not) 0.4501. It carries
literally no information. Cross-modal support fires on 32% and lifts **2.08x**.

The explanation generalises past this one comparison:

    The value of a redundancy axis is its INDEPENDENCE, not its abundance. A
    persistent false positive -- a dock edge, a reflection, a wake -- is exactly
    what survives from frame to frame, so temporal consistency selects for STABLE
    detections and in a fixed scene the false positives are the most stable things
    there are. A thermal signature is different physics; a reflection has no heat.

That makes `lift = P(TP | signal) / P(TP | not signal)` the right screen for any
proposed fusion term. It needs no fusion run, no bootstrap and no held-out split to
say whether a term *can* help: a signal at lift 1.0 cannot, whatever weighting is
put on it, because re-scoring by a constant preserves the ranking AP is computed
from.

This applies it to every per-box signal the system already has. Sweeps the veto
axes' own inputs too, since a frame-level signal is a per-box signal that happens
to be constant within a frame.

Reported on day frames, per condition. `lift` uses TP at IoU 0.5; `lift@75` uses
IoU 0.75, because a signal can predict PRESENCE without predicting LOCALISATION and
AP@50-95 pays for the second.

Usage:
    python scripts/probe_signal_lift.py --cache-dir runs/cache_m
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

from uqfusion.eval.apmetrics import frame_parts                      # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402
from uqfusion.eval.tsupport import _iou, temporal_support            # noqa: E402
from uqfusion.uq.reliability import per_box_uncertainty              # noqa: E402


def cross_support(vis, ir_in_vis, thr):
    out = []
    for v_, i_ in zip(vis, ir_in_vis):
        v = np.asarray(v_["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
        b = np.asarray(i_["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
        if not len(v):
            out.append(np.zeros(0, bool))
            continue
        if not len(b):
            out.append(np.zeros(len(v), bool))
            continue
        m = _iou(v, b)
        m = np.where(np.asarray(v_["cls"]).astype(int)[:, None]
                     == np.asarray(i_["cls"]).astype(int)[None, :], m, 0.0)
        out.append((m >= thr).any(axis=1))
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", default="runs/eval/signal_lift_26m.md")
    ap.add_argument("--conditions", nargs="+", default=["clean", "fog", "glare", "lowlight"])
    args = ap.parse_args()
    t0 = time.time()

    ctx = load_context(preset="crossmodal26m", cache_dir=args.cache_dir,
                       conditions=tuple(args.conditions), verbose=True)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    day = np.flatnonzero(~night)
    rows = []

    for cond in args.conditions:
        res = run_systems(ctx, cond)
        vis = ctx.vis_by_cond[cond]
        parts = frame_parts(vis, ctx.gts)
        keep = [i for i in day if len(parts[i]["conf"])]
        tp50 = np.concatenate([parts[i]["tp"][:, 0] for i in keep])
        # NL thresholds run 0.50..0.95 in ten steps, so index 5 is IoU 0.75.
        tp75 = np.concatenate([parts[i]["tp"][:, 5] for i in keep])
        conf = np.concatenate([np.asarray(vis[i]["conf"], dtype=float) for i in keep])

        sig = {}
        for k in (1, 2, 5):
            b = temporal_support(vis, k=k, iou=0.30, gamma=1.0)
            sig[f"temporal k{k} IoU.30"] = [
                np.asarray(b[i]["conf"]) > np.asarray(vis[i]["conf"]) * 1.5 - 1e-12
                for i in range(len(vis))]
        for thr in (0.10, 0.30, 0.55):
            sig[f"cross-modal IoU{thr:.2f}"] = cross_support(vis, res["ir_in_vis"], thr)

        # Per-box signals the detector already emits but fusion never ranks on.
        sig["conf above its frame median"] = [
            np.asarray(r["conf"], dtype=float) > np.median(np.asarray(r["conf"], dtype=float))
            if len(r["conf"]) else np.zeros(0, bool) for r in vis]
        sig["sigma below its frame median (tight box)"] = [
            (per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
             < np.median(per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])))
            if len(r["conf"]) else np.zeros(0, bool) for r in vis]
        area = [(np.asarray(r["boxes_xyxy"]).reshape(-1, 4)[:, 2]
                 - np.asarray(r["boxes_xyxy"]).reshape(-1, 4)[:, 0])
                * (np.asarray(r["boxes_xyxy"]).reshape(-1, 4)[:, 3]
                   - np.asarray(r["boxes_xyxy"]).reshape(-1, 4)[:, 1]) for r in vis]
        sig["box larger than its frame median"] = [
            a > np.median(a) if len(a) else np.zeros(0, bool) for a in area]
        # The combination the architecture would actually use.
        cm = cross_support(vis, res["ir_in_vis"], 0.30)
        tm = sig["temporal k2 IoU.30"]
        sig["cross-modal AND temporal"] = [c & t for c, t in zip(cm, tm)]
        sig["cross-modal OR temporal"] = [c | t for c, t in zip(cm, tm)]

        for name, flags in sig.items():
            f = np.concatenate([flags[i] for i in keep])
            if f.all() or not f.any():
                continue
            for tag, tp in (("50", tp50), ("75", tp75)):
                a, b = tp[f].mean(), tp[~f].mean()
                rows.append({"cond": cond, "signal": name, "iou": tag,
                             "fires": float(f.mean()), "p_fired": float(a),
                             "p_not": float(b), "lift": float(a / max(b, 1e-9)),
                             "n": int(len(tp))})
            # Is it just re-reading confidence? A signal that only repeats what
            # `conf` already says cannot re-rank anything.
            rows[-1]["corr_conf"] = float(np.corrcoef(f.astype(float), conf)[0, 1])
        print(f"[lift] {cond} done ({time.time() - t0:.0f}s)", flush=True)

    L = ["# Which per-box signals predict a true positive? — the cheap screen", "",
         "`lift = P(TP | signal) / P(TP | no signal)` on VIS day boxes. It needs no "
         "fusion run, no bootstrap and no held-out split to say whether a proposed "
         "term *can* help: **a signal at lift 1.0 cannot**, whatever weight is put on "
         "it, because re-scoring by something uninformative preserves the ranking AP "
         "is computed from.", "",
         "`lift@75` uses TP at IoU 0.75 — a signal can predict **presence** without "
         "predicting **localisation**, and AP@50-95 pays for the second. "
         "`corr(conf)` catches a signal that merely re-reads confidence.", ""]
    for cond in args.conditions:
        L += [f"## `{cond}`", "",
              "| signal | fires | P(TP\\|fired) | P(TP\\|not) | **lift@50** | lift@75 | corr(conf) |",
              "|---|---:|---:|---:|---:|---:|---:|"]
        sigs = list(dict.fromkeys(r["signal"] for r in rows if r["cond"] == cond))
        for s in sigs:
            r50 = next((r for r in rows if r["cond"] == cond and r["signal"] == s
                        and r["iou"] == "50"), None)
            r75 = next((r for r in rows if r["cond"] == cond and r["signal"] == s
                        and r["iou"] == "75"), None)
            if r50 is None:
                continue
            cc = r75.get("corr_conf") if r75 else None
            L.append(f"| {s} | {r50['fires']:.1%} | {r50['p_fired']:.4f} | "
                     f"{r50['p_not']:.4f} | **{r50['lift']:.2f}x** | "
                     f"{r75['lift']:.2f}x | "
                     f"{('%.2f' % cc) if cc is not None else '—'} |")
        L.append("")

    clean = [r for r in rows if r["cond"] == args.conditions[0] and r["iou"] == "50"]
    best = sorted(clean, key=lambda r: -r["lift"])[:3]
    L += ["## Verdict", "",
          "- Highest lift on `" + args.conditions[0] + "`: "
          + ", ".join(f"**{r['signal']}** ({r['lift']:.2f}x)" for r in best),
          "- Any signal at lift ≈ 1.0 is dead on arrival and needs no AP run.",
          "", "The value of a redundancy axis is its **independence**, not its "
          "abundance. Temporal support fires twice as often as cross-modal support "
          "and lifts nothing: a persistent false positive — a dock edge, a "
          "reflection, a wake — is exactly what survives from frame to frame, so "
          "temporal consistency selects for *stable* detections, and in a fixed "
          "scene the false positives are the most stable things there are. A thermal "
          "signature is different physics; a reflection has no heat."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"rows": rows}, indent=2), encoding="utf-8")
    print(f"[lift] wrote {out} in {time.time() - t0:.0f}s")
    for r in sorted(clean, key=lambda r: -r["lift"]):
        print(f"[lift] {r['lift']:5.2f}x  fires {r['fires']:5.1%}  {r['signal']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
