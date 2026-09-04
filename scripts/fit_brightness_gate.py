"""Fit the photometric reliability term (mu_b, tau_b) — TODO §0.2.

**The problem being fixed.** `r_frame` is a sigmoid in Mahalanobis distance D,
which measures how UNUSUAL a frame's neck features are. Darkness does not make a
frame unusual — it makes it empty, and empty lands near the middle of the
training distribution. Measured on the paired val set:

    pohang01 (real night)  D = 28.4   visible-only mAP = 0.0000
    pohang00 (daylight)    D = 30.0   visible-only mAP = 0.4004

The gate scores the blind run as *cleaner* than the working one, so it keeps
weight on a camera that sees nothing, and gated fusion (0.0765) ends up WORSE
than ir_only (0.0810) on the one genuinely adverse run in the dataset.

**The rule fitted here.** Same functional form and same fitting philosophy as
D-6, on a different axis:

    r_bright(b) = sigmoid((b - mu_b) / tau_b)  fitted to  mAP(b) / mAP(clean)

mu_b is the content brightness at which the detector retains half its clean mAP;
tau_b is the scale over which that retention decays. Combined into r_frame by
MIN (see `compute_reliability`), so either alarm firing is sufficient.

**Anti-leakage — the point of this script's structure.** The ladder is 46%
pohang01, so fitting on "the lowlight condition" would fit partly on the very
night frames the fix is meant to be tested against. Instead:

    FIT   on pohang00/02/03 only, clean + synthetic lowlight s1/s2/s3
    TEST  on pohang01, real darkness, never seen during fitting

That is disjoint in BOTH run and degradation source. A rule learned from
albumentations-darkened daylight frames that then correctly flags a real night
run is a generalization result, not a curve fitted to its own answer.

Usage:
    python scripts/fit_brightness_gate.py --modality vis \
        --out runs/eval/brightness_constants.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.matching import load_gt, map50_95
from uqfusion.uq.mahalanobis import MahalanobisScorer

FIT_RUNS = ("pohang00", "pohang02", "pohang03")
HELD_OUT_RUN = "pohang01"
FIT_CONDITIONS = ("clean", "lowlight_s1", "lowlight_s2", "lowlight_s3")


def sigmoid_up(b: np.ndarray, mu: float, tau: float) -> np.ndarray:
    """Brighter = more reliable, so the sigmoid rises (D's version falls)."""
    return 1.0 / (1.0 + np.exp(-(b - mu) / max(tau, 1e-9)))


def load_brightness(stem: str, stat: str, bright_dir: Path) -> tuple[np.ndarray, list[str]]:
    payload = json.loads((bright_dir / f"{stem}.json").read_text(encoding="utf-8"))
    return (np.asarray([f[stat] for f in payload["frames"]], dtype=np.float64),
            [f["run"] for f in payload["frames"]])


def main() -> int:
    from scipy.optimize import curve_fit
    from scipy.stats import spearmanr

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--modality", default="vis", choices=["vis", "ir"])
    parser.add_argument("--ladder-dir", default="runs/cache/ladder")
    parser.add_argument("--bright-dir", default="runs/derived/brightness")
    parser.add_argument("--fit-cache", default="runs/cache/gauss_vis_train_clean.pkl")
    parser.add_argument("--stat", default="mean", help="photometric statistic to gate on")
    parser.add_argument("--fit-conditions", default=",".join(FIT_CONDITIONS))
    parser.add_argument("--rule", choices=["retention", "margin"], default="retention",
                        help="retention = curve_fit to mAP retention (original D-6-style rule); "
                             "margin = place the transition inside the empty gap between the "
                             "corrupted and clean fit-run distributions")
    parser.add_argument("--bins", type=int, default=16)
    parser.add_argument("--out", default="runs/eval/brightness_constants.json")
    args = parser.parse_args()
    load_config(args.config)

    ladder, bright_dir = Path(args.ladder_dir), Path(args.bright_dir)
    m = args.modality
    scorer = MahalanobisScorer().fit(np.stack([r["feat"] for r in load_cache(args.fit_cache)[0]]))

    # --- assemble the FIT pool: chosen runs, chosen conditions ---------------
    clean_recs, _ = load_cache(ladder / f"{m}_clean.pkl")
    runs = np.asarray([Path(r["image_path"]).parent.name for r in clean_recs])
    fit_mask = np.isin(runs, FIT_RUNS)
    held_mask = runs == HELD_OUT_RUN
    gts_all = [load_gt(r["image_path"], r["image_hw"]) for r in clean_recs]
    print(f"[fitb] ladder {len(clean_recs)} frames: {int(fit_mask.sum())} fit "
          f"({'+'.join(FIT_RUNS)}), {int(held_mask.sum())} held out ({HELD_OUT_RUN})")

    idx_fit = np.flatnonzero(fit_mask)
    gts_fit = [gts_all[i] for i in idx_fit]
    map_clean = map50_95([clean_recs[i] for i in idx_fit], gts_fit)["map50_95"]
    print(f"[fitb] clean reference mAP on fit runs = {map_clean:.5f}")

    fit_conditions = tuple(c.strip() for c in args.fit_conditions.split(",") if c.strip())
    b_all, rec_all, gt_all, rows = [], [], [], []
    for cond in fit_conditions:
        recs, _ = load_cache(ladder / f"{m}_{cond}.pkl")
        b, _ = load_brightness(f"{m}_{cond}", args.stat, bright_dir)
        if len(b) != len(recs):
            raise SystemExit(f"{cond}: {len(b)} brightness rows vs {len(recs)} records — re-run frame_brightness.py")
        sub = [recs[i] for i in idx_fit]
        mp = map50_95(sub, gts_fit)["map50_95"]
        rows.append({"condition": cond, "mean_brightness": float(b[idx_fit].mean()),
                     "map50_95": float(mp), "retention": float(mp / max(map_clean, 1e-9))})
        b_all.append(b[idx_fit])
        rec_all += sub
        gt_all += gts_fit
        print(f"[fitb]   {cond:14s} brightness {b[idx_fit].mean():7.2f}   mAP {mp:.5f}   "
              f"retention {mp / max(map_clean, 1e-9):.4f}")

    # --- bin by brightness, measure mAP inside each bin ----------------------
    b_pool = np.concatenate(b_all)
    edges = np.unique(np.quantile(b_pool, np.linspace(0, 1, args.bins + 1)))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (b_pool >= lo) & (b_pool < hi if hi < edges[-1] else b_pool <= hi)
        if sel.sum() < 30:
            continue
        ii = np.flatnonzero(sel)
        bins.append({"b": float(np.median(b_pool[sel])), "n": int(sel.sum()),
                     "retention": float(map50_95([rec_all[i] for i in ii],
                                                 [gt_all[i] for i in ii])["map50_95"] / max(map_clean, 1e-9))})
    if len(bins) < 4:
        raise SystemExit(f"only {len(bins)} usable bins — fit pool too small")

    x = np.array([b["b"] for b in bins])
    y = np.clip([b["retention"] for b in bins], 0.0, 1.0)

    if args.rule == "margin":
        # `p05` is bimodal by construction: every corrupted fit-run frame reads
        # exactly 0.0 and every clean one reads >= 21. There are NO samples in
        # between, so curve_fit's choice of threshold inside that gap is
        # unconstrained by the data -- it is decided by within-daylight noise.
        # That is how mu_b landed at 25.4, INSIDE the clean distribution, and
        # damped pohang03 (p05 25.9, a run VIS handles fine) to r_bright 0.53.
        #
        # State the placement instead of letting the optimizer guess it: put the
        # sigmoid's 2%-98% transition exactly across the empty margin. Both
        # endpoints come from FIT RUNS ONLY, so this stays leak-free.
        dark = np.concatenate([b for b, r in zip(b_all, rows) if r["condition"] != "clean"])
        clean_b = b_all[[r["condition"] for r in rows].index("clean")]
        lo, hi = float(dark.max()), float(clean_b.min())
        if not hi > lo:
            raise SystemExit(f"no empty margin: corrupted max {lo} >= clean min {hi}; "
                             f"the margin rule needs separable fit-run distributions")
        mu_b = 0.5 * (lo + hi)
        tau_b = (hi - lo) / 8.0          # +/-4 tau spans the margin -> 1.8% at lo, 98.2% at hi
        print(f"[fitb] margin rule: corrupted max {lo:.2f} | clean min {hi:.2f} -> "
              f"empty margin {hi - lo:.2f} wide")
    else:
        (mu_b, tau_b), _ = curve_fit(sigmoid_up, x, y, p0=[float(np.median(x)), float(x.std() or 1.0)],
                                     bounds=([x.min(), 1e-3], [x.max(), np.ptp(x) * 5 + 1.0]), maxfev=20000)
    rmse = float(np.sqrt(((sigmoid_up(x, mu_b, tau_b) - y) ** 2).mean()))
    rho_b = float(spearmanr(x, y).statistic)

    # Head-to-head on the SAME frames: does brightness rank damage better than D?
    d_pool = np.concatenate([scorer.score(np.stack([r["feat"] for r in
                                                    [load_cache(ladder / f'{m}_{c}.pkl')[0][i] for i in idx_fit]]))
                             for c in fit_conditions])
    d_bin = [float(np.median(d_pool[(b_pool >= lo) & (b_pool <= hi)]))
             for lo, hi in zip(edges[:-1], edges[1:]) if ((b_pool >= lo) & (b_pool <= hi)).sum() >= 30]
    rho_d = float(spearmanr(d_bin[:len(y)], y).statistic)

    print(f"\n[fitb] FIT: mu_b={mu_b:.3f}  tau_b={tau_b:.3f}  (RMSE {rmse:.4f} over {len(bins)} bins)")
    print(f"[fitb] monotonicity on the fit pool — Spearman(brightness, retention) = {rho_b:+.3f}  "
          f"(+1 is perfect)")
    print(f"[fitb]                                 Spearman(D,          retention) = {rho_d:+.3f}  "
          f"(-1 is perfect)")

    # --- HELD OUT: pohang01, real night, never used above --------------------
    idx_held = np.flatnonzero(held_mask)
    b_held, _ = load_brightness(f"{m}_clean", args.stat, bright_dir)
    b_held = b_held[idx_held]
    d_held = scorer.score(np.stack([clean_recs[i]["feat"] for i in idx_held]))
    b_day = np.concatenate(b_all[:1])                      # clean condition, fit runs
    d_day = d_pool[: len(idx_fit)]
    r_bright_held = sigmoid_up(b_held, mu_b, tau_b)
    r_bright_day = sigmoid_up(b_day, mu_b, tau_b)
    map_held = map50_95([clean_recs[i] for i in idx_held], [gts_all[i] for i in idx_held])["map50_95"]

    print(f"\n[fitb] === HELD OUT: {HELD_OUT_RUN}, real night, {len(idx_held)} frames ===")
    print(f"[fitb]   actual visible-only mAP  {map_held:.5f}   (fit runs, clean: {map_clean:.5f})")
    print(f"[fitb]   brightness   night {b_held.mean():7.2f}  vs  day {b_day.mean():7.2f}")
    print(f"[fitb]   D            night {d_held.mean():7.2f}  vs  day {d_day.mean():7.2f}"
          f"   <- {'INVERTED' if d_held.mean() < d_day.mean() else 'ok'}")
    print(f"[fitb]   r_bright     night {r_bright_held.mean():7.4f}  vs  day {r_bright_day.mean():7.4f}")
    sep = float((r_bright_held.max() < r_bright_day.min()))
    print(f"[fitb]   separation: night max {r_bright_held.max():.4f} "
          f"{'<' if sep else '>='} day min {r_bright_day.min():.4f}  -> "
          f"{'CLEAN SEPARATION' if sep else 'OVERLAP'}")

    payload = {
        "modality": m, "stat": args.stat,
        "rule": args.rule,
        "rule_desc": ("mu_b = midpoint of the empty fit-run margin; tau_b = margin/8 (TODO §0.2b)"
                      if args.rule == "margin" else
                      "mu_b = content brightness at 50% clean-mAP retention; tau_b = logistic scale (TODO §0.2)"),
        "mu_b": float(mu_b), "tau_b": float(tau_b), "fit_rmse": rmse, "n_bins": len(bins),
        "fit_runs": list(FIT_RUNS), "fit_conditions": list(fit_conditions),
        "held_out_run": HELD_OUT_RUN,
        "spearman_brightness": rho_b, "spearman_distance": rho_d,
        "map_clean_fit_runs": float(map_clean),
        "held_out": {"n": int(len(idx_held)), "map50_95": float(map_held),
                     "brightness_mean": float(b_held.mean()), "d_mean": float(d_held.mean()),
                     "r_bright_mean": float(r_bright_held.mean()),
                     "r_bright_max": float(r_bright_held.max()),
                     "day_r_bright_min": float(r_bright_day.min()),
                     "clean_separation": bool(sep)},
        "ladder": rows, "bins": bins,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    existing[m] = payload
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[fitb] -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
