"""What does the detector swap alone change? Clean cell, both stages, both scales.

Everything in `docs/crossmodal-gate-2026-09-01.md` was measured on `runs/phase2/*`:
yolo26s, VIS trained on stride5, and an IR model still carrying nc=2 despite
D28/A-1 having made IR ship-only. The architecture the handoff describes is
yolo26m for VIS and yolo26m-p2feat / nc=1 for IR, and those checkpoints have
existed since 2026-08-21 with no fusion number ever computed on them.

This runs the clean cell only, on whatever caches exist, and reports the four
quantities every downstream decision depends on:

  * the CAPABILITY PRIOR and its ratio -- `cap_ir_scale` was adopted at x4 to
    correct a 36x gap. If the gap moves, that constant is re-opened, because what
    it encodes is a property of the two checkpoints and nothing else.
  * VIS at NIGHT. It is exactly 0.0000 on all 1032 night frames under 26s, which
    is why four of the eight headline cells measure hand-over rather than fusion.
    A VIS model that sees anything at night changes what the benchmark is.
  * the gap of gated fusion over max(VIS, IR).
  * how often the two streams actually AGREE -- the share of fused clusters with
    a member from each stream. The day-cell gain is supposed to come from WBF's
    agreement bonus, and at `iou_thr` 0.85 with a 3-6 px registration residual
    that mechanism may simply never fire. If it does not, the gain has another
    source and the record says the wrong thing about it.

Usage:
    python scripts/probe_detector_swap.py --out runs/eval/detector_swap_clean.md
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
from uqfusion.uq.fusion import apply_homography                                  # noqa: E402

SHIP = 0


def iou_matrix(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def agreement(ctx, res, sel, thrs=(0.85, 0.55, 0.30, 0.10)) -> dict:
    """Share of VIS boxes with an IR box of the same class above each IoU.

    Not the fused clusters: what matters is whether the agreement WBF is being
    credited for is geometrically available at all. A threshold at which almost no
    VIS box has an IR partner is a threshold at which the agreement bonus cannot
    fire, whatever the fusion code does with it.
    """
    out = {t: [0, 0] for t in thrs}
    for i in sel:
        v = np.asarray(ctx.vis_by_cond[ctx.conditions[0]][i]["boxes_xyxy"]).reshape(-1, 4)
        vc = np.asarray(ctx.vis_by_cond[ctx.conditions[0]][i]["cls"]).astype(int)
        r = res["ir_in_vis"][i]
        b = np.asarray(r["boxes_xyxy"]).reshape(-1, 4)
        bc = np.asarray(r["cls"]).astype(int)
        if not len(v):
            continue
        m = iou_matrix(v, b)
        if len(b):
            m = np.where(vc[:, None] == bc[None, :], m, 0.0)
            best = m.max(axis=1)
        else:
            best = np.zeros(len(v))
        for t in thrs:
            out[t][0] += int((best >= t).sum())
            out[t][1] += len(v)
    return {t: (c / n if n else 0.0) for t, (c, n) in out.items()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/detector_swap_clean.md")
    ap.add_argument("--n-boot", type=int, default=1000)
    args = ap.parse_args()
    t0 = time.time()

    rows, agree = [], {}
    for tag, cd in (("yolo26s (phase2)", "runs/cache"),
                    ("yolo26m (full_scale)", "runs/cache_m")):
        if not (ROOT / cd / "gauss_vis_paired_clean.pkl").is_file():
            print(f"[swap] {cd} not built yet, skipping")
            continue
        ctx = load_context(preset="crossmodal", cache_dir=cd, conditions=("clean",),
                           verbose=True)
        night = np.isin(ctx.runs, NIGHT_RUNS)
        day = np.flatnonzero(~night)
        res = run_systems(ctx, "clean")
        p_g = frame_parts(res["fused_gated"], ctx.gts)
        p_v = frame_parts(ctx.vis_by_cond["clean"], ctx.gts)
        p_i = frame_parts(res["ir_in_vis"], ctx.gts)

        def apf(p, sel):
            e = ap_from_parts([p[i] for i in np.flatnonzero(sel)])
            s = e["per_class"].get(SHIP)
            return (float(s["ap50_95"]) if s else 0.0, float(e["map50_95"]))

        for lab, sel in (("day", ~night), ("night", night)):
            v, vm = apf(p_v, sel)
            i, im = apf(p_i, sel)
            g, gm = apf(p_g, sel)
            b = bootstrap_delta([p_g[k] for k in np.flatnonzero(sel)],
                                [(p_v if v >= i else p_i)[k] for k in np.flatnonzero(sel)],
                                None, n_boot=args.n_boot, cls=SHIP) if args.n_boot else {}
            rows.append({"det": tag, "split": lab, "VIS": v, "IR": i, "bar": max(v, i),
                         "gated": g, "gap": g - max(v, i), "macro": gm,
                         "cap_vis": ctx.cap_vis, "cap_ir": ctx.cap_ir,
                         "ratio": ctx.cap_vis / max(ctx.cap_ir, 1e-9),
                         "keep_cls": list(load_context(
                             preset="crossmodal", cache_dir=cd, conditions=("clean",),
                             veto_keep_cls="auto", verbose=False).veto_keep_cls),
                         **({"ci": [b["ci_lo"], b["ci_hi"]], "delta": b["delta"]} if b else {})})
        agree[tag] = agreement(ctx, res, day)
        print(f"[swap] {tag} done ({time.time() - t0:.0f}s)", flush=True)

    L = ["# The detector swap, clean cell — 26s vs the full-scale 26m", "",
         "Ship AP. `bar` = max(VIS, IR) on the same streams. Everything else in the "
         "system is identical: same gate, same constants, same corruption seeds, "
         "same paired frames. The only change is which checkpoint produced the "
         "boxes.", "",
         "| detector | split | VIS | IR | bar | gated | gap | macro | 95% CI |",
         "|---|---|---:|---:|---:|---:|---:|---:|---|"]
    for r in rows:
        ci = (f"[{r['ci'][0]:+.4f}, {r['ci'][1]:+.4f}]" if "ci" in r else "—")
        L.append(f"| {r['det']} | {r['split']} | {r['VIS']:.4f} | {r['IR']:.4f} | "
                 f"{r['bar']:.4f} | {r['gated']:.4f} | **{r['gap']:+.4f}** | "
                 f"{r['macro']:.4f} | {ci} |")

    L += ["", "## The capability prior — what `cap_ir_scale` is correcting", "",
          "| detector | VIS | IR (as the gate sees it) | ratio | classes IR cannot supply |",
          "|---|---:|---:|---:|---|"]
    seen = set()
    for r in rows:
        if r["det"] in seen:
            continue
        seen.add(r["det"])
        L.append(f"| {r['det']} | {r['cap_vis']:.4f} | {r['cap_ir']:.4f} | "
                 f"{r['ratio']:.1f}x | {r['keep_cls'] or 'none'} |")

    if agree:
        L += ["", "## Is cross-modal agreement geometrically available?", "",
              "Share of VIS detections with a same-class IR detection above each IoU, "
              "day frames. `iou_thr` is adopted at **0.85**; the registration residual "
              "is 3-6 px median (`runs/eval/x_registration_drift.md`).", "",
              "| detector | " + " | ".join(f"IoU>={t}" for t in (0.85, 0.55, 0.30, 0.10)) + " |",
              "|---|" + "---:|" * 4]
        for tag, d in agree.items():
            L.append(f"| {tag} | " + " | ".join(f"{d[t]:.2%}" for t in (0.85, 0.55, 0.30, 0.10)) + " |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"rows": rows, "agreement": {k: {str(t): v for t, v in d.items()}
                                     for k, d in agree.items()}}, indent=2), encoding="utf-8")
    print("\n".join(L))
    print(f"\n[swap] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
