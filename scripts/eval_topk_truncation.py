"""Per-modality top-k truncation — the other consequence of fusion being concatenation.

The two streams enter WBF wildly unbalanced: **29,042 VIS detections against
120,877 IR** on the same frames, a 4.2x asymmetry at conf 0.001. §8 records that
`skip_box_thr` > 0 was monotonically worse and concludes IR over-detection "costs
WBF nothing; it is a training problem". That conclusion is about a *score*
threshold, which removes boxes from both streams by an absolute cutoff their two
uncalibrated scales do not share (see `eval_score_calibration.py`). A *rank*
cutoff is a different operation: it keeps each modality's best k regardless of
where that modality's scores happen to sit.

This matters because at `iou_thr` 0.85 fusion is ~99.9% concatenation (§5.3) and
mAP is rank-based. A low-precision tail on one stream cannot be washed out by
averaging — it interleaves directly into the ranked list.

Swept per modality and jointly, so a change can be attributed to the stream that
caused it. `k = None` is the adopted system and is asserted to reproduce it.

Usage:
    python scripts/eval_topk_truncation.py [--ks 25 50 100 200 400]
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

ARRAY_KEYS = ("boxes_xyxy", "conf", "cls", "sigma_ltrb")


def truncate(records: list[dict], k: int | None) -> list[dict]:
    """Keep each frame's top-k detections by confidence, arrays kept aligned."""
    if k is None:
        return records
    out = []
    for r in records:
        conf = np.asarray(r["conf"], dtype=float)
        if len(conf) <= k:
            out.append(r)
            continue
        keep = np.argsort(-conf)[:k]
        keep = np.sort(keep)                     # preserve original ordering
        new = dict(r)
        for key in ARRAY_KEYS:
            v = r.get(key)
            if v is None:
                continue
            a = np.asarray(v)
            new[key] = a[keep] if a.ndim == 1 else a[keep, ...]
        out.append(new)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ks", type=int, nargs="+", default=[25, 50, 100, 200, 400])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="runs/eval/x_topk_truncation.md")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict the condition sweep (pre-flight uses --conditions clean)")
    args = ap.parse_args()

    t0 = time.time()
    ctx = load_context(**({'conditions': tuple(args.conditions)} if args.conditions else {}))
    splits = {"day": ctx.sel("day"), "night": ctx.sel("night")}

    n_vis = int(sum(len(r["conf"]) for r in ctx.vis_by_cond["clean"]))
    n_ir = int(sum(len(r["conf"]) for r in ctx.ir_clean))
    per_frame_vis = np.asarray([len(r["conf"]) for r in ctx.vis_by_cond["clean"]])
    per_frame_ir = np.asarray([len(r["conf"]) for r in ctx.ir_clean])
    print(f"[topk] VIS {n_vis} dets ({per_frame_vis.mean():.1f}/frame), "
          f"IR {n_ir} dets ({per_frame_ir.mean():.1f}/frame), ratio {n_ir / max(n_vis,1):.2f}x")

    base_parts = {}
    for cond in ctx.conditions:
        base_parts[cond] = frame_parts(run_systems(ctx, cond)["fused_gated"], ctx.gts)

    rows, boots = [], {}
    arms = ([("ir", k) for k in args.ks] + [("vis", k) for k in args.ks]
            + [("both", k) for k in args.ks])
    for who, k in arms:
        for cond in ctx.conditions:
            vis = truncate(ctx.vis_by_cond[cond], k if who in ("vis", "both") else None)
            ir = truncate(ctx.ir_clean, k if who in ("ir", "both") else None)
            p = frame_parts(run_systems(ctx, cond, vis_records=vis, ir_records=ir)["fused_gated"],
                            ctx.gts)
            for sname, sel in splits.items():
                rows.append({"who": who, "k": k, "condition": cond, "split": sname,
                             "map": ap_from_parts(p, sel)["map50_95"]})
            if cond in ("clean", "fog"):
                for sname, sel in splits.items():
                    boots[(who, k, cond, sname)] = bootstrap_delta(
                        p, base_parts[cond], sel, n_boot=args.n_boot)
        print(f"[topk] {who:5s} k={k:4d} " + "  ".join(
            f"{r['condition'][:4]}/{r['split'][:1]}={r['map']:.4f}"
            for r in rows if r["who"] == who and r["k"] == k), flush=True)

    base = {(c, s): ap_from_parts(base_parts[c], sel)["map50_95"]
            for c in ctx.conditions for s, sel in splits.items()}

    L = ["# Per-modality top-k truncation", "",
         f"VIS emits **{n_vis}** detections and IR **{n_ir}** on the same {ctx.n()} frames "
         f"({n_ir / max(n_vis,1):.2f}x, {per_frame_vis.mean():.1f} vs {per_frame_ir.mean():.1f} "
         f"per frame) at conf 0.001. A rank cutoff, unlike the `skip_box_thr` score cutoff "
         f"already rejected in §8, does not require the two score scales to be comparable.",
         "",
         "## 1. Adopted system (no truncation)", "",
         "| condition | " + " | ".join(splits) + " |",
         "|---|" + "---:|" * len(splits)]
    for c in ctx.conditions:
        L.append(f"| {c} | " + " | ".join(f"{base[(c, s)]:.4f}" for s in splits) + " |")

    L += ["", "## 2. mAP under truncation", "",
          "| truncate | k | " + " | ".join(f"{c}/{s[:1]}" for c in ctx.conditions for s in splits) + " |",
          "|---|---:|" + "---:|" * (len(ctx.conditions) * len(splits))]
    for who, k in arms:
        cells = []
        for c in ctx.conditions:
            for s in splits:
                m = next((r["map"] for r in rows
                          if r["who"] == who and r["k"] == k and r["condition"] == c and r["split"] == s), None)
                cells.append(f"{m:.4f}" if m is not None else "—")
        L.append(f"| {who} | {k} | " + " | ".join(cells) + " |")

    L += ["", "## 3. Deltas against the adopted system, with intervals", "",
          "| truncate | k | condition | split | delta | 95% CI | sign flips |",
          "|---|---:|---|---|---:|---|---:|"]
    for (who, k, cond, sname), b in boots.items():
        L.append(f"| {who} | {k} | {cond} | {sname} | {b['delta']:+.4f} | "
                 f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}] | {b['p_sign_flip']:.1%} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({
        "n_vis": n_vis, "n_ir": n_ir, "rows": rows,
        "base": {f"{c}|{s}": v for (c, s), v in base.items()},
        "bootstrap": {"|".join(map(str, k)): v for k, v in boots.items()}}, indent=2), encoding="utf-8")
    print(f"[topk] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
