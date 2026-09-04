"""B-2: Leave-one-fit-run-out on `mu_b` -- TODO-2026-08-20 SS B-2 [was SS9.4-20].

`mu_b` (the photometric veto threshold, p05 stat, margin rule -- D27) is the
midpoint of the empty gap between two endpoints, both measured over the three
fit runs (pohang00/02/03) only:

    lo = max p05 over every corrupted (dark) fit-run frame
    hi = min p05 over every CLEAN fit-run frame
    mu_b = (lo + hi) / 2,  tau_b = (hi - lo) / 8

`hi` is a single min() over three runs' clean frames -- one run can set it
alone. This script drops each fit run in turn, recomputes the margin on the
remaining two, and checks how far `mu_b` moves and whether the fog/night target
cell or the clean/day guard cell (`docs/followup-analysis-2026-08-20.md` SS4)
move with it. If one run's exclusion swings `mu_b` a lot, the margin rule needs
a floor tied to the clean distribution rather than a bare min() over three
samples.

CPU-only, cached predictions (~1 h). Usage:
    python scripts/loro_mu_b.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.cache import load_cache  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems  # noqa: E402

FIT_RUNS = ("pohang00", "pohang02", "pohang03")
FIT_CONDITIONS = ("clean", "lowlight_s2", "lowlight_s3")
STAT = "p05"
LADDER_DIR = ROOT / "runs/cache/ladder"
BRIGHT_DIR = ROOT / "runs/derived/brightness"
ADOPTED_MU_B, ADOPTED_TAU_B = 10.5, 2.625
EVAL_CONDITIONS = ("clean", "fog")   # clean/day = guard, fog/night = the cell that moves (followup SS4)


def load_p05(cond: str) -> tuple[np.ndarray, np.ndarray]:
    """p05 per frame + run id per frame, for one ladder condition."""
    recs, _ = load_cache(LADDER_DIR / f"vis_{cond}.pkl")
    runs = np.asarray([Path(r["image_path"]).parent.name for r in recs])
    bright = json.loads((BRIGHT_DIR / f"vis_{cond}.json").read_text(encoding="utf-8"))["frames"]
    p05 = np.asarray([f["p05"] for f in bright], dtype=float)
    if len(p05) != len(recs):
        raise SystemExit(f"{cond}: {len(p05)} brightness rows vs {len(recs)} cache records")
    return p05, runs


def margin_mu_b(fit_runs: tuple[str, ...], p05_by_cond: dict[str, tuple[np.ndarray, np.ndarray]]):
    """The margin rule (fit_brightness_gate.py --rule margin), parameterised on
    which fit runs are included. Returns (mu_b, tau_b, lo, hi, hi_run, lo_run)."""
    clean_p05, clean_runs = p05_by_cond["clean"]
    clean_mask = np.isin(clean_runs, fit_runs)
    clean_sub, clean_sub_runs = clean_p05[clean_mask], clean_runs[clean_mask]
    hi_i = int(np.argmin(clean_sub))
    hi, hi_run = float(clean_sub[hi_i]), str(clean_sub_runs[hi_i])

    dark_vals, dark_runs = [], []
    for cond in FIT_CONDITIONS:
        if cond == "clean":
            continue
        p05, runs = p05_by_cond[cond]
        mask = np.isin(runs, fit_runs)
        dark_vals.append(p05[mask])
        dark_runs.append(runs[mask])
    dark_vals = np.concatenate(dark_vals)
    dark_runs = np.concatenate(dark_runs)
    lo_i = int(np.argmax(dark_vals))
    lo, lo_run = float(dark_vals[lo_i]), str(dark_runs[lo_i])

    if not hi > lo:
        raise SystemExit(f"no empty margin with fit_runs={fit_runs}: dark max {lo} >= clean min {hi}")
    mu_b = 0.5 * (lo + hi)
    tau_b = (hi - lo) / 8.0
    return mu_b, tau_b, lo, hi, hi_run, lo_run


def main() -> int:
    t0 = time.time()
    p05_by_cond = {cond: load_p05(cond) for cond in FIT_CONDITIONS}

    variants = [("full (adopted)", FIT_RUNS)]
    for excluded in FIT_RUNS:
        remaining = tuple(r for r in FIT_RUNS if r != excluded)
        variants.append((f"drop {excluded}", remaining))

    print(f"[b2] margin rule, {len(variants)} variants (full + leave-one-out)")
    fits = []
    for name, fit_runs in variants:
        mu_b, tau_b, lo, hi, hi_run, lo_run = margin_mu_b(fit_runs, p05_by_cond)
        fits.append({"variant": name, "fit_runs": fit_runs, "mu_b": mu_b, "tau_b": tau_b,
                     "lo": lo, "hi": hi, "hi_run": hi_run, "lo_run": lo_run})
        print(f"[b2] {name:16s} fit_runs={fit_runs}  mu_b={mu_b:7.3f}  tau_b={tau_b:6.3f}  "
              f"(clean-min {hi:.2f} from {hi_run}, dark-max {lo:.2f} from {lo_run})")

    adopted = fits[0]
    if abs(adopted["mu_b"] - ADOPTED_MU_B) > 0.05:
        print(f"[b2] WARNING: full-fit mu_b={adopted['mu_b']:.3f} does not reproduce the "
              f"recorded adopted value {ADOPTED_MU_B} -- check FIT_CONDITIONS/stat match "
              f"runs/eval/brightness_constants.json")

    # ---- evaluate downstream effect on the guard + target cells --------------
    base_ctx = load_context(conditions=EVAL_CONDITIONS, verbose=False)
    splits = {"day": base_ctx.sel("day"), "night": base_ctx.sel("night")}

    base_parts = {}
    for cond in EVAL_CONDITIONS:
        res = run_systems(base_ctx, cond)
        base_parts[cond] = frame_parts(res["fused_gated"], base_ctx.gts)

    eval_rows = []
    variant_parts = {}
    for f in fits:
        vctx = replace(base_ctx, c_vis=replace(base_ctx.c_vis, mu_b=f["mu_b"], tau_b=f["tau_b"]))
        cell = {}
        for cond in EVAL_CONDITIONS:
            res = run_systems(vctx, cond)
            variant_parts[(f["variant"], cond)] = frame_parts(res["fused_gated"], base_ctx.gts)
            for sname, sel in splits.items():
                cell[(cond, sname)] = ap_from_parts(variant_parts[(f["variant"], cond)], sel)["map50_95"]
        eval_rows.append({"variant": f["variant"], "mu_b": f["mu_b"], "tau_b": f["tau_b"], "cells": cell})
        print(f"[b2]   {f['variant']:16s} clean/day={cell[('clean','day')]:.4f}  "
              f"fog/night={cell[('fog','night')]:.4f}  ({time.time() - t0:.0f}s)", flush=True)

    # bootstrap each LORO variant's fog/night and clean/day against the full-fit baseline
    boots = {}
    for f in fits[1:]:
        for cond, sname in (("fog", "night"), ("clean", "day")):
            b = bootstrap_delta(variant_parts[(f["variant"], cond)], variant_parts[("full (adopted)", cond)],
                                splits[sname], n_boot=1000)
            boots[(f["variant"], cond, sname)] = b

    # ---- report ---------------------------------------------------------------
    L = ["# B-2 -- leave-one-fit-run-out on `mu_b` (margin rule)", "",
         f"Fit runs: {FIT_RUNS}. Margin rule: `mu_b` = midpoint of "
         f"[max p05 over dark fit-run frames, min p05 over clean fit-run frames], "
         f"`tau_b` = that gap / 8. Fit conditions: {FIT_CONDITIONS}.", "",
         "## 1. How much does `mu_b` move?", "",
         "| variant | fit runs | mu_b | tau_b | delta mu_b | clean-min from | dark-max from |",
         "|---|---|---:|---:|---:|---|---|"]
    for f in fits:
        L.append(f"| {f['variant']} | {'+'.join(f['fit_runs'])} | {f['mu_b']:.3f} | "
                 f"{f['tau_b']:.3f} | {f['mu_b'] - adopted['mu_b']:+.3f} | "
                 f"{f['hi_run']} ({f['hi']:.2f}) | {f['lo_run']} ({f['lo']:.2f}) |")

    L += ["", "## 2. Downstream effect: guard (clean/day) and target (fog/night) cells", "",
          "| variant | mu_b | clean/day | delta vs full | 95% CI | fog/night | delta vs full | 95% CI |",
          "|---|---:|---:|---:|---|---:|---:|---|"]
    for r in eval_rows:
        if r["variant"] == "full (adopted)":
            L.append(f"| {r['variant']} | {r['mu_b']:.3f} | {r['cells'][('clean','day')]:.4f} | "
                     f"-- | -- | {r['cells'][('fog','night')]:.4f} | -- | -- |")
            continue
        bd = boots[(r["variant"], "clean", "day")]
        bf = boots[(r["variant"], "fog", "night")]
        L.append(f"| {r['variant']} | {r['mu_b']:.3f} | {r['cells'][('clean','day')]:.4f} | "
                 f"{bd['delta']:+.4f} | [{bd['ci_lo']:+.4f}, {bd['ci_hi']:+.4f}]"
                 f"{' (spans 0)' if bd['spans_zero'] else ''} | "
                 f"{r['cells'][('fog','night')]:.4f} | {bf['delta']:+.4f} | "
                 f"[{bf['ci_lo']:+.4f}, {bf['ci_hi']:+.4f}]{' (spans 0)' if bf['spans_zero'] else ''} |")

    binding_run = max(fits[1:], key=lambda f: abs(f["mu_b"] - adopted["mu_b"]))
    swing = abs(binding_run["mu_b"] - adopted["mu_b"])
    L += ["", "## 3. Reading", "",
          f"Dropping **{binding_run['variant'].replace('drop ', '')}** moves `mu_b` the most "
          f"({binding_run['mu_b']:.3f} vs adopted {adopted['mu_b']:.3f}, delta "
          f"{binding_run['mu_b'] - adopted['mu_b']:+.3f}). "
          + (f"That is a {swing:.2f}-point swing from removing one of three fit runs -- "
             f"the margin rule as specified (bare min/max over 3 runs) does not have enough "
             f"data to be stable on its own and needs a floor tied to the clean distribution "
             f"(e.g. a percentile instead of a min, or pooling more day-labelled frames into "
             f"the clean endpoint) before it is reported as fixed."
             if swing > 1.0 else
             f"That swing is small relative to `tau_b` ({adopted['tau_b']:.2f}) -- the rule is "
             f"not being carried by a single run's idiosyncrasy.")]

    out_md = ROOT / "runs/eval/x_loro_mu_b.md"
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(L) + "\n", encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps({
        "fits": [{k: v for k, v in f.items()} for f in fits],
        "eval_rows": [{"variant": r["variant"], "mu_b": r["mu_b"], "tau_b": r["tau_b"],
                       "cells": {f"{c}/{s}": v for (c, s), v in r["cells"].items()}} for r in eval_rows],
        "bootstrap": {f"{k[0]}|{k[1]}|{k[2]}": v for k, v in boots.items()},
        "binding_run": binding_run["variant"], "mu_b_swing": swing,
    }, indent=2), encoding="utf-8")
    print(f"\n[b2] wrote {out_md} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
