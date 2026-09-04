"""B-1: Day-only Mahalanobis reference refit — TODO-2026-08-20 SS B-1.

The night blind spot: the VIS Mahalanobis scorer is fit on
`runs/cache/gauss_vis_train_clean.pkl`, which is 782/4000 frames of pohang01 --
the real night run. Darkness is not unusual to a reference set that already
contains darkness, so real night scores D=28.4, *lower* than daylight pohang00's
D=30.0 (`runs/eval/brightness_constants.json`), and the OOD scorer reads the
blindest run in the dataset as the cleanest one.

Falsification test 1 (TODO): rebuild the reference set from daylight frames only
(drop training frames whose content brightness falls below the adopted veto
threshold mu_b=10.5, p05 stat -- same rule the photometric gate uses) and
re-measure D on pohang01 in the paired validation set, held out of BOTH the
train cache and the day/dark split. Two possible outcomes:

  * D_night > D_day under the day-only scorer -- the mis-composed reference set
    was the whole story, and the photometric term is a redundancy check on a
    now-correctly-behaving distributional signal.
  * D_night still < D_day -- darkness is invisible to pooled-neck-feature
    Mahalanobis regardless of what the reference set contains, and the
    photometric term is load-bearing, not redundant.

CPU-only, cached features + a one-time brightness pass over the train cache
(~20 min). Usage:
    python scripts/refit_maha_dayonly.py
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
for p in (ROOT / "src", ROOT / "scripts"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from uqfusion.eval.cache import load_cache  # noqa: E402
from uqfusion.eval.matching import load_gt, map50_95  # noqa: E402
from uqfusion.uq.mahalanobis import MahalanobisScorer  # noqa: E402
import frame_brightness as fb  # noqa: E402

TRAIN_CACHE = ROOT / "runs/cache/gauss_vis_train_clean.pkl"
VAL_CACHE = ROOT / "runs/cache/gauss_vis_paired_clean.pkl"
BRIGHT_DIR = ROOT / "runs/derived/brightness"
BRIGHT_JSON = BRIGHT_DIR / "gauss_vis_train_clean.json"
MU_B = 10.5           # adopted veto threshold, p05 stat, margin rule (D27)
HELD_OUT_RUN = "pohang01"


def ensure_brightness() -> None:
    """In-process re-implementation of frame_brightness.py's main loop (not a
    subprocess: `uqfusion` is only pip-installed in `.venv`, and shelling out
    to `sys.executable` silently picks up whatever interpreter is running this
    script, which may not have it)."""
    if BRIGHT_JSON.is_file():
        print(f"[b1] reusing {BRIGHT_JSON}")
        return
    import cv2

    print("[b1] no brightness stats for the train cache yet -- computing them "
          "(one-time pass over 4,000 frames, content rows only, no corruption to replay)")
    recs, _ = load_cache(TRAIN_CACHE)
    lo, hi = fb.content_rows("vis")
    rows = []
    for i, r in enumerate(recs):
        im = cv2.imread(r["image_path"])
        if im is None:
            raise FileNotFoundError(f"could not read image: {r['image_path']}")
        gray = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[lo:hi]
        st = fb.frame_stats(gray)
        st["image_path"] = r["image_path"]
        st["run"] = Path(r["image_path"]).parent.name
        rows.append(st)
        if (i + 1) % 500 == 0:
            print(f"[b1]   brightness {i + 1}/{len(recs)}", flush=True)
    BRIGHT_DIR.mkdir(parents=True, exist_ok=True)
    BRIGHT_JSON.write_text(json.dumps({
        "cache": str(TRAIN_CACHE), "modality": "vis", "content_rows": [lo, hi],
        "corrupt": None, "severity": None, "corrupt_seed": None,
        "n_frames": len(rows), "frames": rows,
    }), encoding="utf-8")
    print(f"[b1] wrote {BRIGHT_JSON}")


def main() -> int:
    t0 = time.time()
    ensure_brightness()

    recs, _ = load_cache(TRAIN_CACHE)
    bright = json.loads(BRIGHT_JSON.read_text(encoding="utf-8"))["frames"]
    p05 = np.asarray([f["p05"] for f in bright], dtype=float)
    if len(p05) != len(recs):
        raise SystemExit(f"brightness has {len(p05)} rows for {len(recs)} train frames "
                          f"-- delete {BRIGHT_JSON} and re-run")
    runs = np.asarray([Path(r["image_path"]).parent.name for r in recs])

    day_mask = p05 >= MU_B
    print(f"[b1] train reference set: {len(recs)} frames total")
    print(f"[b1]   kept  (p05 >= {MU_B}): {int(day_mask.sum())}  -- {Counter(runs[day_mask].tolist())}")
    print(f"[b1]   drop  (p05 <  {MU_B}): {int((~day_mask).sum())}  -- {Counter(runs[~day_mask].tolist())}")

    feats_all = np.stack([r["feat"] for r in recs])
    scorer_contaminated = MahalanobisScorer().fit(feats_all)
    scorer_dayonly = MahalanobisScorer().fit(feats_all[day_mask])

    # ---- re-measure on the paired validation set: real night vs real day -----
    val_recs, _ = load_cache(VAL_CACHE)
    val_runs = np.asarray([Path(r["image_path"]).parent.name for r in val_recs])
    val_feats = np.stack([r["feat"] for r in val_recs])
    night_sel = np.flatnonzero(val_runs == HELD_OUT_RUN)
    day_sel = np.flatnonzero(val_runs != HELD_OUT_RUN)

    gts = [load_gt(r["image_path"], r["image_hw"]) for r in val_recs]
    map_night = map50_95([val_recs[i] for i in night_sel], [gts[i] for i in night_sel])["map50_95"]
    map_day = map50_95([val_recs[i] for i in day_sel], [gts[i] for i in day_sel])["map50_95"]
    print(f"\n[b1] held-out validation: {len(night_sel)} night frames ({HELD_OUT_RUN}), "
          f"{len(day_sel)} day frames")
    print(f"[b1]   visible-only mAP: night {map_night:.4f}  day {map_day:.4f}")

    rows = []
    for name, scorer in (("contaminated (adopted, night in reference)", scorer_contaminated),
                         ("day-only (refit, night excluded)", scorer_dayonly)):
        d = scorer.score(val_feats)
        d_night, d_day = d[night_sel], d[day_sel]
        inverted = bool(np.mean(d_night) < np.mean(d_day))
        rows.append({
            "scorer": name,
            "d_night_mean": float(np.mean(d_night)), "d_night_median": float(np.median(d_night)),
            "d_day_mean": float(np.mean(d_day)), "d_day_median": float(np.median(d_day)),
            "inverted": inverted,
        })
        print(f"[b1]   {name}")
        print(f"[b1]     D_night mean/median = {np.mean(d_night):7.2f} / {np.median(d_night):7.2f}")
        print(f"[b1]     D_day   mean/median = {np.mean(d_day):7.2f} / {np.median(d_day):7.2f}   "
              f"{'-> INVERTED (night reads cleaner than day)' if inverted else '-> ok (night reads stranger, as it should)'}")

    was_inverted, now_inverted = rows[0]["inverted"], rows[1]["inverted"]
    fixed = was_inverted and not now_inverted
    verdict = ("FIXED -- day-only refit alone corrects the inversion; the "
               "photometric term is a redundancy check on this signal."
               if fixed else
               "NOT FIXED -- D(night) is still <= D(day) even off a clean reference; "
               "darkness is invisible to pooled-neck-feature Mahalanobis regardless "
               "of reference composition, and the photometric term stays load-bearing."
               if was_inverted else
               "the adopted scorer was not inverted on this exact comparison; "
               "see the numbers above rather than the one-line verdict.")
    print(f"\n[b1] VERDICT: {verdict}")

    out_md = ROOT / "runs/eval/x_maha_dayonly_refit.md"
    L = ["# B-1 -- day-only Mahalanobis reference refit (falsification test 1)", "",
         f"Adopted veto threshold reused as the day/dark cut: `mu_b={MU_B}` on p05 "
         f"content brightness (`runs/eval/brightness_constants.json`).", "",
         f"Train reference set: {len(recs)} frames -> {int(day_mask.sum())} kept as "
         f"day-only, {int((~day_mask).sum())} dropped as dark.", "",
         "| run | kept (day-only ref) | dropped (dark) |", "|---|---:|---:|"]
    for run in sorted(set(runs.tolist())):
        L.append(f"| {run} | {int(day_mask[runs == run].sum())} | {int((~day_mask[runs == run]).sum())} |")
    L += ["", f"Held-out paired validation: {len(night_sel)} real night frames "
          f"({HELD_OUT_RUN}, visible-only mAP {map_night:.4f}), {len(day_sel)} real day "
          f"frames (visible-only mAP {map_day:.4f}).", "",
          "| scorer | D_night mean | D_night median | D_day mean | D_day median | inverted? |",
          "|---|---:|---:|---:|---:|---|"]
    for r in rows:
        L.append(f"| {r['scorer']} | {r['d_night_mean']:.2f} | {r['d_night_median']:.2f} | "
                 f"{r['d_day_mean']:.2f} | {r['d_day_median']:.2f} | "
                 f"{'YES (night looks cleaner)' if r['inverted'] else 'no'} |")
    L += ["", f"## Verdict", "", verdict]

    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps({
        "mu_b": MU_B, "n_train": len(recs), "n_kept_day": int(day_mask.sum()),
        "n_dropped_dark": int((~day_mask).sum()),
        "n_night_val": len(night_sel), "n_day_val": len(day_sel),
        "map_night": map_night, "map_day": map_day,
        "rows": rows, "fixed": fixed, "verdict": verdict,
    }, indent=2), encoding="utf-8")
    print(f"\n[b1] wrote {out_md} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
