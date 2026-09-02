"""Idea I3 -- do the extra views carry information, and does merging them help?

Consumes the caches `build_tta_o2m.py` writes. Two questions, in the order that
lets the first refuse the second:

  1. **The F1 lift screen.** For a base (o2o, unaugmented) detection, does "another
     view also fires here" predict a true positive? Cross-modal support measured
     2.08x and temporal support 1.00x, and the rule F2 established is that the
     value of a redundancy axis is its INDEPENDENCE, not its abundance. Two views
     of the same frame through the same weights may be no more independent than
     two consecutive frames. If the lift is ~1.00x, I3 closes here.

  2. **The arms.** Score-only support (the one cross-modal form that survived, C2),
     concatenation, and sigma-weighted WBF at several thresholds -- the last being
     the first time the project's namesake mechanism operates on genuinely
     independent, pixel-exactly co-registered estimates.

`--o2m-cache` boxes carry a PLACEHOLDER sigma (sigma rides on `one2one_cv2`
only), so sigma-weighted arms are skipped for that source and the table says so.

Usage:
    python scripts/probe_tta_o2m.py --tta-cache runs/cache_tta/gauss_vis_paired_clean.pkl
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import (ROOT, ap_of, day_night, fmt, gts_for, lift, load_records,  # noqa: E402
                           loro_folds, md_table, sgn, subsample, write_md)

sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.matching import iou_matrix                     # noqa: E402
from uqfusion.uq.fusion import sigma_weighted_fusion              # noqa: E402


def split_views(rec):
    """A multi-view record -> {view_index: single-view record}."""
    v = np.asarray(rec.get("view", np.zeros(len(rec["conf"]), int))).astype(int)
    out = {}
    for k in np.unique(v) if len(v) else []:
        m = v == k
        out[int(k)] = {**rec,
                       "boxes_xyxy": np.asarray(rec["boxes_xyxy"], float).reshape(-1, 4)[m],
                       "conf": np.asarray(rec["conf"], float)[m],
                       "cls": np.asarray(rec["cls"]).astype(int)[m],
                       "sigma_ltrb": np.asarray(rec["sigma_ltrb"], float).reshape(-1, 4)[m]}
    if not out:
        out[0] = {**rec, "boxes_xyxy": np.zeros((0, 4)), "conf": np.zeros(0),
                  "cls": np.zeros(0, int), "sigma_ltrb": np.zeros((0, 4))}
    return out


def agrees(a, b, thr):
    A = np.asarray(a["boxes_xyxy"], float).reshape(-1, 4)
    if not len(A):
        return np.zeros(0, bool)
    B = np.asarray(b["boxes_xyxy"], float).reshape(-1, 4)
    if not len(B):
        return np.zeros(len(A), bool)
    m = iou_matrix(A, B)
    m = np.where(np.asarray(a["cls"]).astype(int)[:, None]
                 == np.asarray(b["cls"]).astype(int)[None, :], m, 0.0)
    return (m >= thr).any(1)


def wbf(recs, iou_thr, use_sigma):
    hw = recs[0].get("image_hw", (640, 640))
    norm = np.array([hw[1], hw[0], hw[1], hw[0]], float)
    bl, sl, ll, gl = [], [], [], []
    for r in recs:
        b = np.asarray(r["boxes_xyxy"], float).reshape(-1, 4)
        bl.append(np.clip(b / norm, 0.0, 1.0).tolist())
        sl.append(np.asarray(r["conf"], float).tolist())
        ll.append(np.asarray(r["cls"], float).tolist())
        gl.append(np.asarray(r["sigma_ltrb"], float).reshape(-1, 4) / norm)
    fb, fs, fl = sigma_weighted_fusion(bl, sl, ll, gl, [1.0] * len(recs),
                                       iou_thr=iou_thr, use_sigma=use_sigma)
    return {**recs[0], "boxes_xyxy": np.asarray(fb) * norm, "conf": np.asarray(fs),
            "cls": np.asarray(fl).astype(int)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--tta-cache", default="runs/cache_tta/gauss_vis_paired_clean.pkl")
    ap.add_argument("--o2m-cache", default="runs/cache_o2m/gauss_vis_paired_clean.pkl")
    ap.add_argument("--base-cache", default="runs/cache_m/gauss_vis_paired_clean.pkl")
    ap.add_argument("--out", default="runs/eval/tta_o2m.md")
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    base, bmeta = load_records(ROOT / args.base_cache)
    if args.limit:
        base = subsample(base, args.limit)
    gts = gts_for(base)
    runs, day, _n = day_night(base)
    pb = frame_parts(base, gts)
    b_ap = ap_of(base, gts, sel=day)
    secs = [f"Base: `{args.base_cache}` (o2o, unaugmented) -- mAP50-95 "
            f"**{fmt(b_ap['map50_95'])}** on {len(day)} day frames."]

    sources = {}
    for tag, path in (("tta", args.tta_cache), ("o2m", args.o2m_cache)):
        p = ROOT / path
        if not p.is_file():
            secs.append(f"`{path}` absent -- {tag} not built. "
                        f"Run `scripts/build_tta_o2m.py --mode {tag}`.")
            continue
        recs, meta = load_records(p)
        if args.limit:
            recs = subsample(recs, args.limit)
        if len(recs) != len(base):
            secs.append(f"`{path}` has {len(recs)} frames against the base's {len(base)} "
                        f"-- not index-aligned, skipped.")
            continue
        sources[tag] = (recs, meta)
        secs.append(f"`{path}`: views {meta.get('views')}, sigma_valid "
                    f"{meta.get('sigma_valid')}, "
                    f"{np.mean([len(r['conf']) for r in recs]):.1f} boxes/frame.")

    if not sources:
        secs.append("\n**Nothing to measure.** Build the caches first.")
        write_md(args.out, "TTA and one2many (I3)", secs)
        return 0

    # ---- 1. THE SCREEN ----------------------------------------------------
    keep = [i for i in day if len(pb[i]["conf"])]
    tp50 = np.concatenate([pb[i]["tp"][:, 0] for i in keep])
    tp75 = np.concatenate([pb[i]["tp"][:, 5] for i in keep])
    cf = np.concatenate([np.asarray(base[i]["conf"], float) for i in keep])
    rows = []
    for tag, (recs, meta) in sources.items():
        views = {i: split_views(recs[i]) for i in keep}
        nv = max(len(v) for v in views.values()) if views else 1
        vnames = meta.get("views") or ["?"]
        for vi in range(nv):
            vname = vnames[min(vi, len(vnames) - 1)]
            for thr in (0.30, 0.55, 0.75):
                f = np.concatenate([agrees(base[i], views[i].get(vi, base[i]), thr)
                                    for i in keep])
                r50, m50 = lift(f, tp50, cf)
                r75, m75 = lift(f, tp75, cf)
                rows.append([tag, vname, f"{thr:.2f}", fmt(f.mean(), 3),
                             fmt(r50, 2) + "x", fmt(m50, 2) + "x",
                             fmt(r75, 2) + "x", fmt(m75, 2) + "x"])
    secs.append("## 1. The screen -- lift of view agreement (F1)\n\n"
                "`lift = P(TP | that view also fires) / P(TP | it does not)`, over the BASE "
                "detections. Reference points: cross-modal support **2.08x**, temporal "
                "support **1.00x**. A row near 1.00x cannot help at any weight.\n\n"
                "**Read the `matched` columns.** Raw lift inherits confidence's own 4.8x "
                "for any signal correlated with it, and 'another view also fires' is "
                "strongly correlated with confidence. The matched version bins by "
                "confidence quantile first.\n\n"
                + md_table(["source", "view", "IoU", "fires", "raw@50", "matched@50",
                            "raw@75", "matched@75"], rows))

    # ---- 2. the arms ------------------------------------------------------
    rows = []
    for tag, (recs, meta) in sources.items():
        sig_ok = bool(meta.get("sigma_valid", True))
        arms = [(f"{tag}: all views concatenated", lambda i, r=recs: r[i])]
        for thr in (0.30, 0.55):
            for g in (0.5, 1.0):
                def _sup(i, r=recs, t=thr, gg=g):
                    vs = split_views(r[i])
                    others = [v for k, v in vs.items() if k != 0]
                    if not others:
                        return base[i]
                    f = np.zeros(len(base[i]["conf"]), bool)
                    for o in others:
                        f |= agrees(base[i], o, t)
                    return {**base[i], "conf": np.asarray(base[i]["conf"], float) * (1.0 + gg * f)}
                arms.append((f"{tag}: support IoU{thr:.2f} gamma{g:g} (score only)", _sup))
        for thr in (0.55, 0.7, 0.85):
            for us in ((False, True) if sig_ok else (False,)):
                def _wbf(i, r=recs, t=thr, u=us):
                    vs = split_views(r[i])
                    return wbf(list(vs.values()) if len(vs) > 1 else [r[i]], t, u)
                arms.append((f"{tag}: WBF @{thr:.2f}" + (" sigma" if us else " plain"), _wbf))
        for name, fn in arms:
            out = [fn(i) if i in set(day.tolist()) else base[i] for i in range(len(base))]
            parts = frame_parts(out, gts)
            a = ap_of(out, gts, sel=day)
            bs = bootstrap_delta(parts, pb, sel=day, n_boot=args.n_boot, seed=0) \
                if args.n_boot else None
            per_run = [ap_of(out, gts, sel=te)["map50_95"] - ap_of(base, gts, sel=te)["map50_95"]
                       for _h, _tr, te in loro_folds(runs, day)]
            rows.append([name, fmt(a["map50"]), fmt(a["map50_95"]),
                         sgn(a["map50_95"] - b_ap["map50_95"]),
                         f"[{sgn(bs['ci_lo'])}, {sgn(bs['ci_hi'])}]" if bs else "--",
                         (sgn(min(per_run)) + " / " + sgn(max(per_run))) if per_run else "--"])
    secs.append("## 2. Arms, day frames\n\n"
                f"Paired bootstrap n={args.n_boot} against the base o2o cache.\n\n"
                + md_table(["arm", "mAP50", "mAP50-95", "delta", "95% CI",
                            "worst/best run"], rows))
    secs.append("Sigma-weighted WBF arms are omitted for `o2m`: sigma rides on "
                "`one2one_cv2` only, so those boxes carry a placeholder and weighting by "
                "it would measure the placeholder.")
    secs.append("## 3. Decision rule\n\n"
                "This is the cleanest test the project can run of *why* merging fails. "
                "TTA views are pixel-exactly co-registered, so if `WBF sigma` still loses "
                "here, the C4 explanation (registration residual) is refuted and the "
                "general one stands: **the best member is already the best estimate.** "
                "That would close merging as a family -- cross-modal, within-modal, and "
                "multi-view -- and leave score-only support as the entire fusion surface.")
    secs.append(f"---\n\n_Generated by `scripts/probe_tta_o2m.py` in {time.time() - t0:.1f}s._")
    write_md(args.out, "TTA and the one2many branch (I3)", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
