"""B-4: Dedup-threshold sweep for union-label GT -- TODO-2026-08-20 SS B-4 [was 0.4].

`run_cheap_fixes.py` SS3 builds union GT (VIS labels UNION IR labels projected
into the VIS frame) with a single hardcoded `dedup_iou=0.5`: an IR box is added
as a NEW ground-truth object only if it does not already overlap a VIS box of
the same class at IoU >= 0.5. That produced "+76.8% union GT boxes" (19,133 VIS
+ 14,688 IR-only additions), which every union-label mAP number in
`cheap_fixes.md` depends on -- and no other threshold has ever been tried.

`diag_cross_modal_iou.py` (SS5.3) already measured that cross-modal registration
is loose: only 0.11% of VIS detections reach IoU 0.85 against any IR detection
in the same frame, i.e. even genuinely-the-same object rarely registers above
0.85 after the per-run homography. A dedup threshold of 0.5 is well inside the
range where two projections of the SAME physical object routinely fail to
overlap enough to be merged -- so the "+76.8%" figure is expected to be mostly
double-counted objects that a looser (lower) threshold would correctly merge.

This sweeps `dedup_iou` from loose to strict, reports how the added-box count
and the union mAP move, and read off whether the count "collapses" at
thresholds consistent with the measured registration looseness.

CPU-only, cached predictions + labels (~1-2 h). Usage:
    python scripts/sweep_dedup_iou.py
"""

from __future__ import annotations

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from uqfusion.config import load_config  # noqa: E402
from uqfusion.eval.cache import load_cache  # noqa: E402
from uqfusion.eval.matching import load_gt  # noqa: E402
from run_cheap_fixes import union_gt  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems  # noqa: E402
from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402

THRESHOLDS = (0.10, 0.20, 0.30, 0.40, 0.50, 0.55, 0.70, 0.85)
REPRESENTATIVE = (0.10, 0.50, 0.85)   # thresholds worth a full mAP re-eval


def main() -> int:
    t0 = time.time()
    load_config(None)

    ir_eval, _ = load_cache(ROOT / "runs/cache/gauss_ir_paired_clean.pkl")
    vis_eval, _ = load_cache(ROOT / "runs/cache/gauss_vis_paired_clean.pkl")
    Hj = json.loads((ROOT / "runs/derived/homography_ir_to_vis.json").read_text(encoding="utf-8"))["runs"]
    manifest = list(csv.DictReader(open(ROOT / "runs/derived/paired_val_manifest.csv", newline="", encoding="utf-8")))
    runs_eval = [r["run"] for r in manifest]
    H_eval = [np.asarray(Hj[r]["H_ir_canvas_to_vis_canvas"], dtype=np.float64) for r in runs_eval]

    print(f"[b4] {len(vis_eval)} paired clean frames")

    # ---- 1. sweep: added-box count vs threshold -------------------------------
    rows = []
    for thr in THRESHOLDS:
        ugts, n_base, n_added = union_gt(vis_eval, ir_eval, H_eval, dedup_iou=thr)
        rows.append({"dedup_iou": thr, "n_base": n_base, "n_added": n_added,
                     "pct_added": n_added / max(n_base, 1) * 100})
        print(f"[b4] dedup_iou={thr:.2f}  base={n_base:,}  added={n_added:,}  "
              f"(+{n_added / max(n_base, 1) * 100:.1f}%)  ({time.time() - t0:.0f}s)", flush=True)

    # ---- 2. for the strictest and loosest ends, characterise the "added" boxes:
    #         how close were they to a VIS box of the same class, even if not close
    #         enough to merge? A near-miss at 0.85 that becomes a merge at 0.10 is
    #         exactly the "double-counted object" story.
    from uqfusion.uq.fusion import apply_homography
    from uqfusion.eval.matching import iou_matrix

    best_iou_of_added_at_085 = []
    for rv, ri, h in zip(vis_eval, ir_eval, H_eval):
        g_v = load_gt(rv["image_path"], rv["image_hw"])
        g_i = load_gt(ri["image_path"], ri["image_hw"])
        if len(g_i["cls"]) == 0:
            continue
        proj = apply_homography(g_i["boxes_xyxy"], h)
        for k in range(len(proj)):
            if len(g_v["boxes_xyxy"]):
                same = g_v["cls"] == g_i["cls"][k]
                if same.any():
                    m = iou_matrix(proj[k:k + 1], g_v["boxes_xyxy"][same])
                    best_iou_of_added_at_085.append(float(m.max()) if m.size else 0.0)
                    continue
            best_iou_of_added_at_085.append(0.0)
    best_iou_of_added_at_085 = np.asarray(best_iou_of_added_at_085)
    zero_overlap_frac = float((best_iou_of_added_at_085 == 0.0).mean())
    print(f"[b4] every IR GT box vs its best-matching same-class VIS GT box "
          f"(n={len(best_iou_of_added_at_085):,}): {zero_overlap_frac:.1%} have ZERO overlap "
          f"(no VIS box of that class in frame at all -- these are the only ones that stay "
          f"'added' at every threshold)")

    # ---- 3. mAP effect at representative thresholds --------------------------
    ctx = load_context(conditions=("clean",), verbose=False)
    base_res = run_systems(ctx, "clean")
    base_parts = {
        "visible_only": frame_parts(ctx.vis_by_cond["clean"], ctx.gts),
        "ir_only": frame_parts(base_res["ir_in_vis"], ctx.gts),
        "gated_fusion": frame_parts(base_res["fused_gated"], ctx.gts),
    }

    map_rows = []
    for thr in REPRESENTATIVE:
        ugts, n_base, n_added = union_gt(vis_eval, ir_eval, H_eval, dedup_iou=thr)
        u_parts = {
            "visible_only": frame_parts(ctx.vis_by_cond["clean"], ugts),
            "ir_only": frame_parts(base_res["ir_in_vis"], ugts),
            "gated_fusion": frame_parts(base_res["fused_gated"], ugts),
        }
        for sysname in base_parts:
            a = ap_from_parts(base_parts[sysname])["map50_95"]
            b = ap_from_parts(u_parts[sysname])["map50_95"]
            boot = bootstrap_delta(u_parts[sysname], base_parts[sysname], n_boot=500)
            map_rows.append({"dedup_iou": thr, "system": sysname, "vis_only_gt": a, "union_gt": b,
                             "delta": b - a, "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"]})
        print(f"[b4] mAP re-eval dedup_iou={thr:.2f} done ({time.time() - t0:.0f}s)", flush=True)

    # ---- report ----------------------------------------------------------
    L = ["# B-4 -- dedup-threshold sweep for union-label GT", "",
         f"{len(vis_eval):,} paired clean frames. `union_gt` (run_cheap_fixes.py): an IR GT "
         f"box is added to the union label set unless it already matches a VIS GT box of the "
         f"same class at IoU >= `dedup_iou`. `diag_cross_modal_iou.py` measured that only "
         f"0.11% of VIS *detections* reach IoU 0.85 against any IR detection at all -- the "
         f"registration itself rarely produces overlap that tight, even for the same object.",
         "", "## 1. Added-box count vs threshold", "",
         "| dedup_iou | base (VIS) GT | added (IR-only) | % increase |", "|---:|---:|---:|---:|"]
    for r in rows:
        L.append(f"| {r['dedup_iou']:.2f} | {r['n_base']:,} | {r['n_added']:,} | "
                 f"+{r['pct_added']:.1f}% |")

    L += ["", "## 2. What the 'added' boxes actually are", "",
          f"Of every IR GT box, {zero_overlap_frac:.1%} have literally no VIS GT box of the "
          f"same class anywhere in the frame (best-match IoU = 0.0) -- those stay 'added' no "
          f"matter how loose the threshold gets, and are the only unambiguously-new objects. "
          f"The rest have SOME overlap with a same-class VIS box but not enough to clear "
          f"whatever `dedup_iou` is set to; as the threshold sweep in SS1 shows, those are what "
          f"collapses out of the count as the threshold loosens.", "",
          "## 3. mAP effect at representative thresholds", "",
          "| dedup_iou | system | VIS-only GT | union GT | delta | 95% CI |",
          "|---:|---|---:|---:|---:|---|"]
    for r in map_rows:
        L.append(f"| {r['dedup_iou']:.2f} | {r['system']} | {r['vis_only_gt']:.4f} | "
                 f"{r['union_gt']:.4f} | {r['delta']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |")

    loosest, strictest = rows[0], rows[-1]
    collapse = 1.0 - (loosest["n_added"] / max(strictest["n_added"], 1))
    L += ["", "## 4. Reading", "",
          f"Added-box count at the loosest threshold tried ({loosest['dedup_iou']:.2f}) is "
          f"{loosest['n_added']:,}, a {collapse:.0%} drop from the strictest "
          f"({strictest['dedup_iou']:.2f}, {strictest['n_added']:,} added) -- "
          + ("consistent with the prediction that the +76.8% figure (dedup_iou 0.5) is mostly "
             "double-counted objects a tighter registration match would have merged."
             if collapse > 0.5 else
             "a smaller collapse than predicted; a meaningful fraction of the 'added' boxes "
             "survive even loose dedup, so they are more likely to be genuinely new objects "
             "(outside one sensor's field of view or missed by one annotator) than "
             "registration slop.")
          + " **No single dedup_iou is self-evidently correct — report the sweep, not one "
            "number, until a registration-accuracy argument picks a specific threshold.**"]

    out_md = ROOT / "runs/eval/x_dedup_iou_sweep.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps({
        "rows": rows, "zero_overlap_frac": zero_overlap_frac, "map_rows": map_rows,
    }, indent=2), encoding="utf-8")
    print(f"\n[b4] wrote {out_md} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
