"""Phase 3 §7 step 2 — the development reference HOLDOUT-GAP is judged against.

Executes `docs/prereg-phase3-retrain-2026-09-10.md` Amendment 9 §A9.3 and §A9.5 item 2.
**Reads no pohang04 frame.** Every frame scored here is a development day frame from
`runs/derived/paired_val_*.txt`, which carries zero pohang04 rows.

What A9 fixes, transcribed:

* Five systems, VIS seed k with IR seed k, preset `crossmodal26m`, caches
  `runs/cache_p3/seed{k}/`; every detector-dependent quantity re-derived per seed by
  `load_context`, no constant re-tuned (A9.1).
* Cell `clean/clean` only (A9.2: the verdict cell).
* Two groups: pohang00 (`TUNE_RUNS`) and pohang02 + pohang03 pooled (`TEST_RUNS`).
  pohang01 excluded (A9.3).
* Statistic: mean over the five seeds of per-seed fused ship AP (A9.1).
* `AP_ref` = the lower group, chosen on observed values and then FIXED (A9.3).
* Its bootstrap: moving-block, L = 20, n_boot = 1000, **bootstrap seed 0** for the reference
  side (A9.3 assigns seeds 0 and 1 to the two sides; the pohang04 side takes 1). Each
  resample scores all five systems on the same frames and takes their mean. The 1,000
  replicates are written out so the look differences them replicate by replicate against
  its own, rather than re-deriving them.

Usage:
    python scripts/holdout_p04_devref.py --out docs/eval/holdout_p04_devref_2026-09-14.md
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
sys.path.insert(0, str(ROOT / "scripts"))

from _ideas_common import write_md                                               # noqa: E402
from uqfusion.eval.apmetrics import _score, ap_from_parts, frame_parts, presort  # noqa: E402
from uqfusion.eval.blockboot import block_resample, run_ids, run_slices          # noqa: E402
from uqfusion.eval.ctx import TEST_RUNS, TUNE_RUNS, load_context, run_systems    # noqa: E402
from uqfusion.eval.identity import system_identity                               # noqa: E402

PRESET = "crossmodal26m"
SEEDS = (0, 1, 2, 3, 4)
CELL = "clean"
BLOCK_LEN, N_BOOT, BOOT_SEED_REF = 20, 1000, 0
SHIP = 0
GROUPS = {"pohang00": TUNE_RUNS, "pohang02+pohang03": TEST_RUNS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/eval/holdout_p04_devref_2026-09-14.md")
    args = ap.parse_args()
    t0 = time.time()
    if GROUPS["pohang00"] != ("pohang00",) or set(GROUPS["pohang02+pohang03"]) != {"pohang02", "pohang03"}:
        raise SystemExit("TUNE_RUNS/TEST_RUNS no longer match A9.3's groups")

    parts, paths, ctxs, runs_ref = {}, None, {}, None
    for k in SEEDS:
        cdir = f"runs/cache_p3/seed{k}"
        ctx = load_context(preset=PRESET, cache_dir=cdir, conditions=(CELL,), verbose=(k == 0))
        if any("pohang04" in r for r in ctx.runs):
            raise SystemExit(f"seed {k}: a pohang04 frame reached the development context")
        out = run_systems(ctx, CELL)
        parts[k] = frame_parts(out["fused_gated"], out["gts"])
        p = [r["image_path"] for r in ctx.vis_by_cond[CELL]]
        if paths is None:
            paths, runs_ref = p, ctx.runs
        elif p != paths:
            raise SystemExit(f"seed {k}: frame order differs from seed 0")
        ctxs[k] = ctx
        print(f"[ref] seed {k} scored ({time.time() - t0:.0f}s)", flush=True)

    res = {}
    for g, runs in GROUPS.items():
        sel = np.flatnonzero(np.isin(runs_ref, runs))
        per_seed = []
        for k in SEEDS:
            e = ap_from_parts(parts[k], sel=sel)["per_class"].get(SHIP)
            per_seed.append(float(e["ap50_95"]) if e else float("nan"))
        per_run = run_slices(run_ids([paths[i] for i in sel]))
        pre = {k: presort(parts[k], sel) for k in SEEDS}
        n = pre[SEEDS[0]]["n_frames"]
        rng = np.random.default_rng(BOOT_SEED_REF)
        reps = np.empty(N_BOOT)
        for t in range(N_BOOT):
            w = np.bincount(block_resample(rng, per_run, BLOCK_LEN), minlength=n)
            reps[t] = np.mean([_score(pre[k], w, SHIP) for k in SEEDS])
        kept = reps[np.isfinite(reps)]
        lo, hi = np.percentile(kept, [2.5, 97.5])
        res[g] = {"n_frames": int(len(sel)), "runs": {r: int(len(i)) for r, i in per_run.items()},
                  "per_seed_ap": per_seed, "seed_mean": float(np.mean(per_seed)),
                  "seed_sd": float(np.std(per_seed, ddof=1)),
                  "ci_lo": float(lo), "ci_hi": float(hi), "n_effective": int(kept.size),
                  "replicates": [float(x) for x in reps]}
        print(f"[ref] {g}: seed-mean {res[g]['seed_mean']:.4f} [{lo:.4f}, {hi:.4f}] "
              f"sd {res[g]['seed_sd']:.4f} n={len(sel)}", flush=True)

    ref_group = min(GROUPS, key=lambda g: res[g]["seed_mean"])
    ref = {"group": ref_group, "AP_ref": res[ref_group]["seed_mean"],
           "replicates_bootstrap_seed": BOOT_SEED_REF, "replicates": res[ref_group]["replicates"]}

    L = [
        "Executes `docs/prereg-phase3-retrain-2026-09-10.md` Amendment 9 §A9.3 and §A9.5 item 2. "
        "**No pohang04 frame is read.** Fused ship AP, `clean/clean`, preset `crossmodal26m`, "
        "five Phase 3 systems (VIS seed k + IR seed k), caches `runs/cache_p3/seed{k}/`.",
        f"## Reference — **AP_ref = {ref['AP_ref']:.4f}** ({ref_group})\n\n"
        "Fixed from here. The single look scores pohang04 as `AP_p04` (seed mean, block bootstrap "
        "seed 1) and declares HOLDOUT-GAP iff `AP_ref − AP_p04 ≥ 0.0060` **and** the unpaired 95% "
        "interval of that difference, built replicate by replicate against the 1,000 replicates "
        "stored in the JSON beside this file, lies above zero.",
        "## Both development groups\n\n"
        "| group | frames | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | **seed mean** | 95% CI | seed sd |\n"
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---:|\n"
        + "\n".join(f"| {g} | {r['n_frames']} | " + " | ".join(f"{v:.4f}" for v in r["per_seed_ap"])
                    + f" | **{r['seed_mean']:.4f}** | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | {r['seed_sd']:.4f} |"
                    for g, r in res.items()),
        f"Spread between the groups: **{abs(res['pohang00']['seed_mean'] - res['pohang02+pohang03']['seed_mean']):.4f}** "
        "(A9.3 was chosen on a pre-Phase-3 figure of 0.033; this is the Phase 3 value). "
        "The seed sd describes training variance and is not a decision input.",
    ]
    ident = system_identity(ctxs[0], preset=PRESET, seeds=list(SEEDS), cell=CELL, block_len=BLOCK_LEN,
                            n_boot=N_BOOT, bootstrap_seed=BOOT_SEED_REF,
                            caches=[f"runs/cache_p3/seed{k}" for k in SEEDS])
    write_md(Path(args.out), "pohang04 holdout — development reference (A9.3)", L, identity=ident)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"reference": ref, "groups": res,
         "config": {"preset": PRESET, "seeds": list(SEEDS), "cell": CELL, "block_len": BLOCK_LEN,
                    "n_boot": N_BOOT, "bootstrap_seed": BOOT_SEED_REF, "ship_class": SHIP}},
        indent=1), encoding="utf-8")
    print(f"[ref] AP_ref {ref['AP_ref']:.4f} ({ref_group}) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
