"""Re-fit the frame-reliability constants (mu_d, tau) from a severity ladder — D-6.

**Why this exists.** The pre-registered rules (decision D5 / plan B5) set

    mu_d = 95th percentile of clean-val Mahalanobis distance
    tau  = clean-val IQR of those distances

and the 2026-08-19 fusion test showed that `tau` is fit to the wrong scale
entirely. The IQR measures spread *within* clean data (VIS: 3.37); corruptions
move the distance by 40-290. `r_frame = 1 - sigmoid((D - mu_d)/tau)` therefore
saturates to 0 on ANY corruption, so the gate degenerates into a binary
"is this frame clean?" flag. Glare sat at +11.7 tau -- fully saturated -- while
the detector still delivered 78% of its clean mAP, and the gate threw the frame
away.

**The replacement rule.** `r_frame` is supposed to answer "how much of this
detector's clean capability survives on this frame?", so fit it against exactly
that, measured:

    r_frame(D) = sigmoid((mu_d - D) / tau)  fitted to  mAP(D) / mAP(clean)

over a ladder of corruptions x severities. `mu_d` becomes the distance at which
the detector retains half its clean mAP, and `tau` the scale over which that
retention decays. Both keep their meaning in the existing formula, so nothing
downstream of `compute_reliability` changes.

Frames are pooled across every condition and binned by D (quantile bins); mAP is
computed *within* each bin, so the fit targets the real metric at per-frame
distance granularity without inheriting per-frame F1 noise.

Anti-leakage (plan B5-5): the ladder must use a DIFFERENT `--corrupt-seed` from
the caches any final number is reported on. The ladder here is built at seed 2;
Table 3 is reported at seed 1.

Usage:
    python scripts/fit_reliability_constants.py --modality vis \
        --ladder-dir runs/cache/ladder --fit-cache runs/cache/gauss_vis_train_clean.pkl \
        --out runs/eval/reliability_constants.json
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
from uqfusion.uq.reliability import fit_constants, per_box_uncertainty


def sigmoid_retention(d: np.ndarray, mu: float, tau: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp((d - mu) / max(tau, 1e-9)))


def main() -> int:
    from scipy.optimize import curve_fit

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--modality", choices=["vis", "ir"], required=True)
    parser.add_argument("--ladder-dir", default="runs/cache/ladder")
    parser.add_argument("--fit-cache", required=True, help="clean TRAIN cache — the Mahalanobis fit")
    parser.add_argument("--bins", type=int, default=24)
    parser.add_argument("--out", default="runs/eval/reliability_constants.json")
    args = parser.parse_args()
    load_config(args.config)

    ladder = Path(args.ladder_dir)
    scorer = MahalanobisScorer().fit(np.stack([r["feat"] for r in load_cache(args.fit_cache)[0]]))

    clean_path = ladder / f"{args.modality}_clean.pkl"
    clean, _ = load_cache(clean_path)
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in clean]
    map_clean = map50_95(clean, gts)["map50_95"]
    print(f"[fit] {args.modality}: clean reference mAP@50-95 = {map_clean:.5f} over {len(clean)} frames")

    # --- per-condition summary (the human-readable ladder) --------------------
    conditions = [("clean", clean_path)] + [(p.stem[len(args.modality) + 1:], p)
                                            for p in sorted(ladder.glob(f"{args.modality}_*.pkl"))
                                            if p != clean_path]
    all_d, all_rec, all_gt, rows = [], [], [], []
    for name, path in conditions:
        recs, _ = load_cache(path)
        d = scorer.score(np.stack([r["feat"] for r in recs]))
        m = map50_95(recs, gts)["map50_95"]
        rows.append({"condition": name, "median_D": float(np.median(d)),
                     "map50_95": float(m), "retention": float(m / max(map_clean, 1e-9))})
        all_d.append(d)
        all_rec += recs
        all_gt += gts
        print(f"[fit]   {name:16s} median D {np.median(d):8.2f}   mAP {m:.5f}   "
              f"retention {m / max(map_clean, 1e-9):.4f}")

    # --- bin by D across ALL conditions, measure mAP inside each bin ----------
    d_all = np.concatenate(all_d)
    edges = np.unique(np.quantile(d_all, np.linspace(0, 1, args.bins + 1)))
    bins = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (d_all >= lo) & (d_all < hi if hi < edges[-1] else d_all <= hi)
        if m.sum() < 30:
            continue
        rec_b = [all_rec[i] for i in np.flatnonzero(m)]
        gt_b = [all_gt[i] for i in np.flatnonzero(m)]
        bins.append({"d": float(np.median(d_all[m])), "n": int(m.sum()),
                     "retention": float(map50_95(rec_b, gt_b)["map50_95"] / max(map_clean, 1e-9))})
    if len(bins) < 4:
        raise SystemExit(f"only {len(bins)} usable bins — ladder too small to fit")

    x = np.array([b["d"] for b in bins])
    y = np.clip([b["retention"] for b in bins], 0.0, 1.0)
    (mu, tau), _ = curve_fit(sigmoid_retention, x, y, p0=[float(np.median(x)), float(x.std() or 1.0)],
                             bounds=([x.min(), 1e-3], [x.max() * 3, np.ptp(x) * 5 + 1.0]), maxfev=20000)
    pred = sigmoid_retention(x, mu, tau)
    rmse = float(np.sqrt(((pred - y) ** 2).mean()))

    # Does D even RANK degradations by how much they hurt? A sigmoid in D can
    # only work if retention is a monotone function of D. Spearman over the
    # per-condition ladder answers that directly; -1 would be perfect.
    from scipy.stats import spearmanr

    cond_d = np.array([r["median_D"] for r in rows])
    cond_r = np.array([r["retention"] for r in rows])
    rho = float(spearmanr(cond_d, cond_r).statistic)
    rho_bin = float(spearmanr(x, y).statistic)
    print(f"\n[fit] monotonicity: Spearman(D, retention) = {rho:+.3f} over {len(rows)} conditions, "
          f"{rho_bin:+.3f} over {len(bins)} D-bins  (-1.000 = D ranks damage perfectly)")

    # --- the old rule, for comparison ----------------------------------------
    u = np.concatenate([per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"]) for r in clean if len(r["conf"])])
    old = fit_constants(u, all_d[0])

    print(f"\n[fit] {args.modality} OLD rule (clean-val 95th pct / IQR): mu_d={old.mu_d:.3f}  tau={old.tau:.3f}")
    print(f"[fit] {args.modality} NEW rule (mAP-retention ladder):      mu_d={mu:.3f}  tau={tau:.3f}"
          f"   (fit RMSE {rmse:.4f} over {len(bins)} bins)")
    print(f"[fit] tau widened {tau / old.tau:.1f}x — the old sigmoid saturated "
          f"{(x.max() - old.mu_d) / old.tau:.0f} tau past mu_d at the ladder's far end")
    print(f"[fit] lambda (box term) is unchanged by this rule: {old.lam:.4f}")

    payload = {
        "modality": args.modality,
        "rule": "mu_d = D at 50% clean-mAP retention; tau = logistic scale of the retention curve (D-6)",
        "lam": float(old.lam),
        "mu_d": float(mu), "tau": float(tau),
        "fit_rmse": rmse, "n_bins": len(bins),
        "spearman_condition": rho, "spearman_bins": rho_bin,
        "map_clean": float(map_clean),
        "previous_rule": {"mu_d": float(old.mu_d), "tau": float(old.tau),
                          "rule": "95th pct / IQR of clean-val distances (D5/B5)"},
        "ladder": rows, "bins": bins,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = json.loads(out.read_text(encoding="utf-8")) if out.is_file() else {}
    existing[args.modality] = payload
    out.write_text(json.dumps(existing, indent=2), encoding="utf-8")
    print(f"[fit] -> {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
