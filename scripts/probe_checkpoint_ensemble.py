"""Idea I8 -- do two VIS checkpoints disagree in a way that carries information?

Both `gauss_vis_seed0` base and `_ft` are on disk, and their clean paired caches
already exist in `runs/cache_m_stageprobe/`. D31's rule picked base (val 0.24961
against ft's 0.24111) and the experiment log already records that this cost
-0.0020 on the paired val (0.3686 against 0.3706). No fusion number has ever used
both.

**Why this is not the temporal null again.** F2 found temporal support at lift
1.00x and drew the rule: *the value of a redundancy axis is its independence, not
its abundance.* A persistent false positive -- a dock edge, a wake, a reflection
-- is exactly what survives from frame to frame, so temporal agreement selects
for stable detections and in a fixed scene the false positives are the most
stable things there are. Two checkpoints are different draws from the training
process, so they can disagree about a dock edge in a way one checkpoint across
consecutive frames cannot.

That is an argument, not a result, so this script runs the F1 lift screen FIRST
and reports the arms whatever it says. Lift ~1.00x means the axis is dead and no
weighting rescues it.

Usage:
    python scripts/probe_checkpoint_ensemble.py --a runs/cache_m_stageprobe/gauss_vis_paired_clean.pkl \
        --b runs/cache_m_stageprobe/gauss_vis_paired_clean_ft.pkl
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


def agrees(ra, rb, thr):
    a = np.asarray(ra["boxes_xyxy"], float).reshape(-1, 4)
    if not len(a):
        return np.zeros(0, bool)
    b = np.asarray(rb["boxes_xyxy"], float).reshape(-1, 4)
    if not len(b):
        return np.zeros(len(a), bool)
    m = iou_matrix(a, b)
    m = np.where(np.asarray(ra["cls"]).astype(int)[:, None]
                 == np.asarray(rb["cls"]).astype(int)[None, :], m, 0.0)
    return (m >= thr).any(1)


def concat(ra, rb, wa=1.0, wb=1.0):
    return {**ra,
            "boxes_xyxy": np.concatenate([np.asarray(ra["boxes_xyxy"], float).reshape(-1, 4),
                                          np.asarray(rb["boxes_xyxy"], float).reshape(-1, 4)]),
            "conf": np.concatenate([np.asarray(ra["conf"], float) * wa,
                                    np.asarray(rb["conf"], float) * wb]),
            "cls": np.concatenate([np.asarray(ra["cls"]).astype(int),
                                   np.asarray(rb["cls"]).astype(int)])}


def wbf2(ra, rb, iou_thr, use_sigma):
    hw = ra.get("image_hw", (640, 640))
    norm = np.array([hw[1], hw[0], hw[1], hw[0]], float)
    bl, sl, ll, gl = [], [], [], []
    for r in (ra, rb):
        b = np.asarray(r["boxes_xyxy"], float).reshape(-1, 4)
        bl.append(np.clip(b / norm, 0.0, 1.0).tolist())
        sl.append(np.asarray(r["conf"], float).tolist())
        ll.append(np.asarray(r["cls"], float).tolist())
        gl.append(np.asarray(r["sigma_ltrb"], float).reshape(-1, 4) / norm)
    fb, fs, fl = sigma_weighted_fusion(bl, sl, ll, gl, [1.0, 1.0], iou_thr=iou_thr,
                                       use_sigma=use_sigma)
    return {**ra, "boxes_xyxy": np.asarray(fb) * norm, "conf": np.asarray(fs),
            "cls": np.asarray(fl).astype(int)}


def support(ra, rb, thr, gamma):
    """Score-only confirmation -- the adopted cross-modal `support` term with the
    modality axis swapped for the checkpoint axis. Coordinates never touched,
    which is the one cross-modal arm that survived (C2)."""
    f = agrees(ra, rb, thr)
    c = np.asarray(ra["conf"], float).copy()
    if len(c):
        c = c * (1.0 + gamma * f)
    return {**ra, "conf": c}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", default="runs/cache_m_stageprobe/gauss_vis_paired_clean.pkl")
    ap.add_argument("--b", default="runs/cache_m_stageprobe/gauss_vis_paired_clean_ft.pkl")
    ap.add_argument("--out", default="runs/eval/checkpoint_ensemble.md")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    t0 = time.time()

    A, ma = load_records(ROOT / args.a)
    B, mb = load_records(ROOT / args.b)
    if len(A) != len(B):
        raise ValueError(f"caches are not index-aligned: {len(A)} vs {len(B)}")
    if args.limit:
        A, B = subsample(A, args.limit), subsample(B, args.limit)
    gts = gts_for(A)
    runs, day, _n = day_night(A)
    pa, pb = frame_parts(A, gts), frame_parts(B, gts)
    a_ap, b_ap = ap_of(A, gts, sel=day), ap_of(B, gts, sel=day)
    secs = [f"A = `{args.a}`  \n  weights `{ma.get('weights')}`  \n"
            f"B = `{args.b}`  \n  weights `{mb.get('weights')}`  \n"
            f"{len(day)} day frames. A mAP50-95 **{fmt(a_ap['map50_95'])}**, "
            f"B **{fmt(b_ap['map50_95'])}**."]

    # ---- 1. THE SCREEN (F1 lift) -----------------------------------------
    keep = [i for i in day if len(pa[i]["conf"])]
    tp50 = np.concatenate([pa[i]["tp"][:, 0] for i in keep])
    tp75 = np.concatenate([pa[i]["tp"][:, 5] for i in keep])
    cf = np.concatenate([np.asarray(A[i]["conf"], float) for i in keep])
    rows = []
    for thr in (0.10, 0.30, 0.55, 0.75):
        f = np.concatenate([agrees(A[i], B[i], thr) for i in keep])
        r50, m50 = lift(f, tp50, cf)
        r75, m75 = lift(f, tp75, cf)
        rows.append([f"B also fires @IoU{thr:.2f}", fmt(f.mean(), 3),
                     fmt(r50, 2) + "x", fmt(m50, 2) + "x",
                     fmt(r75, 2) + "x", fmt(m75, 2) + "x"])
    secs.append("## 1. The screen -- lift of checkpoint agreement (F1)\n\n"
                "`lift = P(TP | B agrees) / P(TP | B does not)`. A signal at 1.00x cannot "
                "help whatever weight it is given, because re-scoring by something "
                "uninformative preserves the ranking AP is computed from. For reference, "
                "cross-modal support measured 2.08x and temporal support 1.00x.\n\n"
                "**`matched` is the column to read.** Raw lift is easy to inflate: any "
                "signal correlated with confidence inherits confidence's own 4.8x, and "
                "47% of VIS boxes sit below conf 0.05, so 'the other checkpoint also "
                "fires' partly restates 'this box is confident'. The matched column bins "
                "by confidence quantile and averages within-bin lifts, which is the "
                "question a re-scorer faces: among boxes the detector already ranks "
                "equally, does this separate them?\n\n"
                + md_table(["signal", "fires", "raw@50", "matched@50", "raw@75",
                            "matched@75"], rows)
                + "\n\n**If the matched columns are near 1.00x, idea I8 is closed and "
                  "section 2 is confirmation, not exploration.**")

    # ---- 2. the arms ------------------------------------------------------
    arms = [("A alone (shipped)", lambda i: A[i]),
            ("B alone (ft)", lambda i: B[i]),
            ("concat A+B", lambda i: concat(A[i], B[i])),
            ("concat A + 0.5*B", lambda i: concat(A[i], B[i], 1.0, 0.5))]
    for thr in (0.55, 0.7, 0.85):
        for us in (False, True):
            arms.append((f"WBF @{thr:.2f}" + (" sigma" if us else " plain"),
                         lambda i, t=thr, u=us: wbf2(A[i], B[i], t, u)))
    for thr in (0.30, 0.55):
        for g in (0.5, 1.0):
            arms.append((f"support IoU{thr:.2f} gamma{g:g} (score only)",
                         lambda i, t=thr, gg=g: support(A[i], B[i], t, gg)))

    rows = []
    for name, fn in arms:
        out = [fn(i) for i in range(len(A))]
        parts = frame_parts(out, gts)
        a = ap_of(out, gts, sel=day)
        if args.n_boot:
            bs = bootstrap_delta(parts, pa, sel=day, n_boot=args.n_boot, seed=0)
            lo, hi = bs["ci_lo"], bs["ci_hi"]
        else:
            lo = hi = float("nan")
        per_run = []
        for _h, _tr, te in loro_folds(runs, day):
            per_run.append(ap_of(out, gts, sel=te)["map50_95"]
                           - ap_of(A, gts, sel=te)["map50_95"])
        rows.append([name, fmt(a["map50"]), fmt(a["map50_95"]),
                     sgn(a["map50_95"] - a_ap["map50_95"]),
                     f"[{sgn(lo, 4)}, {sgn(hi, 4)}]" if np.isfinite(lo) else "--",
                     (sgn(min(per_run)) + " / " + sgn(max(per_run))) if per_run else "--"])
    secs.append("## 2. Arms, day frames\n\n"
                f"Paired bootstrap n={args.n_boot} against A alone; `worst/best run` is "
                "the leave-one-run-out spread of the same delta. An arm that wins overall "
                "but loses on a run has not generalised.\n\n"
                + md_table(["arm", "mAP50", "mAP50-95", "delta vs A", "95% CI",
                            "worst/best run"], rows))
    secs.append("Note: `support` here is the C2 term with the modality axis swapped for "
                "the checkpoint axis -- score-only confirmation, coordinates untouched. "
                "It is the arm most likely to work, because it is the only cross-modal "
                "arm that ever did.")
    secs.append(f"---\n\n_Generated by `scripts/probe_checkpoint_ensemble.py` in "
                f"{time.time() - t0:.1f}s._")
    write_md(args.out, "Two-checkpoint VIS ensemble (I8)", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
