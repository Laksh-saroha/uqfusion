"""How much AP is reachable by RE-RANKING alone, and how much IR can ever add.

Two ceilings, both computed without running fusion:

  1. **Oracle re-ranking.** Keep exactly the boxes VIS emits and order them
     perfectly. This bounds every score-based lever in the system at once --
     `support`, the capability weights, `cap_ir_scale`, score calibration,
     sigma-in-score, and any future re-scorer -- because each of them can only
     permute this one list. Measured 2026-09-01 at **+0.1060** on clean day,
     against a shipped fusion gain of +0.0106.

  2. **Cross-modal union recall.** The share of GT that VIS finds, that IR finds,
     and that only IR finds. This bounds every cross-modal lever: no fusion
     operator can recover more than the union contains. Measured at **+0.0115**
     of extra ship recall at IoU 0.5 -- 123 boxes of 10,663.

Read together they say the headroom is in WHICH BOX AND HOW TIGHT, not in WHICH
STREAM. Per-class output (idea I6) exists because AP macro-averages: buoy is 5%
of the GT boxes and 50% of the metric, and no gate constant has ever touched it.

Usage:
    python scripts/probe_oracle_headroom.py --cache-dir runs/cache_m
    python scripts/probe_oracle_headroom.py --vis-cache runs/cache_day/gauss_vis_day_clean.pkl
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import (IOU_LEVELS, ROOT, DAY_RUNS, ap_of, day_night, fmt,  # noqa: E402
                           gts_for, load_records, md_table, oracle_curves, sgn,
                           subsample, write_md)

sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import frame_parts          # noqa: E402
from uqfusion.eval.matching import iou_matrix            # noqa: E402
from uqfusion.uq.fusion import apply_homography          # noqa: E402

CLS_NAMES = {0: "ship", 1: "buoy"}


def union_recall(vis, ir, gts, sel, h_by_run, runs, iou_thr):
    """Per class: share of GT covered by VIS, by IR, by either. Score-blind --
    a box counts if it overlaps, however low its confidence, because this is a
    question about what the sensors CAN see, not about ranking."""
    acc = {}
    for i in sel:
        H = h_by_run.get(runs[i])
        g = gts[i]
        for c in np.unique(g["cls"]) if len(g["cls"]) else []:
            G = g["boxes_xyxy"][g["cls"] == c]
            if not len(G):
                continue
            V = np.asarray(vis[i]["boxes_xyxy"], float).reshape(-1, 4)
            V = V[np.asarray(vis[i]["cls"]).astype(int) == c]
            I = np.asarray(ir[i]["boxes_xyxy"], float).reshape(-1, 4)
            I = I[np.asarray(ir[i]["cls"]).astype(int) == c]
            I = apply_homography(I, H)
            hv = (iou_matrix(G, V) >= iou_thr).any(1) if len(V) else np.zeros(len(G), bool)
            hi = (iou_matrix(G, I) >= iou_thr).any(1) if len(I) else np.zeros(len(G), bool)
            a = acc.setdefault(int(c), [0, 0, 0, 0])
            a[0] += len(G); a[1] += int(hv.sum()); a[2] += int(hi.sum()); a[3] += int((hv | hi).sum())
    return acc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--vis-cache", default=None,
                    help="explicit VIS cache; overrides --cache-dir (VIS-only substrates)")
    ap.add_argument("--ir-cache", default=None)
    ap.add_argument("--homography", default="runs/derived/homography_ir_to_vis.json")
    ap.add_argument("--out", default="runs/eval/oracle_headroom.md")
    ap.add_argument("--json-out", default=None)
    ap.add_argument("--limit", type=int, default=None, help="cap frames (preflight)")
    args = ap.parse_args()
    t0 = time.time()

    vis_path = Path(args.vis_cache) if args.vis_cache else Path(args.cache_dir) / "gauss_vis_paired_clean.pkl"
    ir_path = Path(args.ir_cache) if args.ir_cache else Path(args.cache_dir) / "gauss_ir_paired_clean.pkl"
    vis, vmeta = load_records(ROOT / vis_path)
    have_ir = (ROOT / ir_path).is_file()
    ir = load_records(ROOT / ir_path)[0] if have_ir else None
    # A VIS-only substrate passed via --vis-cache leaves --ir-cache at the
    # --cache-dir default, which exists but indexes a DIFFERENT frame set.
    # Frame i of one is not frame i of the other, so section 4 would silently
    # compare unrelated images if the lengths happened to match. Refuse.
    ir_skip = ""
    if have_ir and len(ir) != len(vis):
        ir_skip = (f"`{ir_path}` holds {len(ir)} frames against this substrate's "
                   f"{len(vis)} -- not index-aligned, so no IR box can be attributed "
                   f"to a VIS frame. Pass a matching `--ir-cache` to measure it.")
        have_ir, ir = False, None
    if args.limit:
        keep = set(id(r) for r in subsample(vis, args.limit))
        idx = [i for i, r in enumerate(vis) if id(r) in keep]
        vis = [vis[i] for i in idx]
        ir = [ir[i] for i in idx] if ir is not None else None
    gts = gts_for(vis)
    runs, day, night = day_night(vis)
    parts = frame_parts(vis, gts)
    print(f"[probe] {len(vis)} frames | day {len(day)} | night {len(night)} | "
          f"runs {sorted(set(runs))}")

    secs = []
    secs.append(
        f"Source: `{vis_path}`" + (f" + `{ir_path}`" if have_ir else " (VIS only)")
        + f"  \nWeights: `{vmeta.get('weights')}`  conf {vmeta.get('conf')}  "
        f"imgsz {vmeta.get('imgsz')}  \nFrames {len(vis)} "
        f"(day {len(day)}, night {len(night)}); runs {', '.join(sorted(set(runs)))}")

    # ---- 1. oracle re-ranking, all day frames ----------------------------
    A, O, per_class, recall = oracle_curves(parts, day)
    rows = [[f"{IOU_LEVELS[k]:.2f}", fmt(A[k]), fmt(O[k]), sgn(O[k] - A[k])] for k in range(len(A))]
    rows.append(["**mean**", f"**{fmt(A.mean())}**", f"**{fmt(O.mean())}**",
                 f"**{sgn(O.mean() - A.mean())}**"])
    secs.append("## 1. Oracle re-ranking ceiling (day)\n\n"
                "Same boxes, perfect order. Upper bound on every score-based lever.\n\n"
                + md_table(["IoU", "actual", "oracle", "headroom"], rows))

    # ---- 2. per class (idea I6) ------------------------------------------
    rows = []
    for c, d in sorted(per_class.items()):
        rows.append([f"{c} {CLS_NAMES.get(c, '?')}", d["n_gt"], d["n_pred"],
                     fmt(d["actual"][0]), fmt(d["actual"].mean()), fmt(d["oracle"].mean()),
                     sgn(d["oracle"].mean() - d["actual"].mean()),
                     fmt(recall[c][0], 3), fmt(recall[c][5], 3), fmt(recall[c][9], 3)])
    secs.append("## 2. Per class -- where the macro metric actually spends (I6)\n\n"
                "`AP` macro-averages over classes, so a class with 5% of the GT boxes "
                "carries 50% of the number.\n\n"
                + md_table(["class", "n_gt", "n_pred", "AP50", "AP50-95", "oracle",
                            "headroom", "rec@50", "rec@75", "rec@95"], rows))

    # ---- 3. per run, and leave-one-run-out variance -----------------------
    rows = []
    for r in DAY_RUNS:
        sel = day[runs[day] == r]
        if not len(sel):
            continue
        a, o, _pc, _rc = oracle_curves(parts, sel)
        rows.append([r, len(sel), fmt(a.mean()), fmt(o.mean()), sgn(o.mean() - a.mean())])
    secs.append("## 3. Per day run\n\n"
                "Spread across runs is the honest scale of a 'held-out' difference: an "
                "arm worth less than the between-run spread has not been shown to "
                "generalise.\n\n"
                + md_table(["run", "frames", "actual", "oracle", "headroom"], rows))

    # ---- 4. union recall (the cross-modal ceiling) ------------------------
    if have_ir:
        H = json.loads((ROOT / args.homography).read_text(encoding="utf-8"))["runs"]
        h_by_run = {k: np.asarray(v["H_ir_canvas_to_vis_canvas"], float) for k, v in H.items()}
        rows = []
        for thr in (0.50, 0.30):
            acc = union_recall(vis, ir, gts, day, h_by_run, runs, thr)
            for c, (ng, nv, ni, nu) in sorted(acc.items()):
                rows.append([f"{thr:.2f}", f"{c} {CLS_NAMES.get(c, '?')}", ng,
                             fmt(nv / ng, 3), fmt(ni / ng, 3), fmt(nu / ng, 3),
                             sgn((nu - nv) / ng), nu - nv])
        secs.append("## 4. Cross-modal union recall -- the ceiling on every IR lever\n\n"
                    "Score-blind: a GT box counts as covered if ANY detection of its class "
                    "overlaps it, however low the confidence. No fusion operator can "
                    "recover more than `union - VIS`.\n\n"
                    + md_table(["IoU", "class", "n_gt", "VIS", "IR", "union", "IR adds", "boxes"],
                               rows))
    else:
        secs.append("## 4. Cross-modal union recall\n\nSkipped: "
                    + (ir_skip or f"no IR cache (`{ir_path}` absent). "
                                  "This substrate is VIS-only by design."))

    # ---- 5. night, stated rather than omitted -----------------------------
    if len(night):
        n_ap = ap_of(vis, gts, sel=night)
        secs.append("## 5. Night\n\n"
                    f"VIS mAP@50-95 on {len(night)} night frames: **{fmt(n_ap['map50_95'])}**. "
                    "Zero is the expected value and is a DATASET decision "
                    "(`filter_night_boxes.py --cut-dark` removed 132k night boxes from the "
                    "training labels), not a fusion result. Idea I5.")

    secs.append(f"---\n\n_Generated by `scripts/probe_oracle_headroom.py` in "
                f"{time.time() - t0:.1f}s._")
    write_md(args.out, "Oracle headroom -- what re-ranking and what IR can ever be worth", secs)

    if args.json_out:
        p = ROOT / args.json_out
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps({
            "actual": A.tolist(), "oracle": O.tolist(),
            "headroom_mean": float(O.mean() - A.mean()),
            "per_class": {str(c): {"actual": d["actual"].mean(), "oracle": d["oracle"].mean(),
                                   "n_gt": d["n_gt"]} for c, d in per_class.items()},
        }, indent=2), encoding="utf-8")
        print(f"[out] {p}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
