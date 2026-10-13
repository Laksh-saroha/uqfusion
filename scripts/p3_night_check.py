"""Does "fused ≥ max(VIS, IR)" hold for the Phase 3 checkpoints? Day and night, descriptive.

Logged in `docs/exposure-ledger-2026-09-09.md` §7 (2026-09-27) before it ran; the fail
criterion is fixed there. **Reads no pohang04 frame** — `runs/derived/paired_val_*.txt`
carries none, and the context refuses if one appears.

The paper's sensor-selection claim was measured on pre-restore checkpoints whose VIS scored
0.0000 at night, where a 100% night veto costs nothing. The Phase 3 VIS seeds trained on the
restored night labels. V1 (`docs/night-veto-axis-closed-2026-09-04.md`) measured removing the
night veto from a restored-label VIS at +0.1785 on clean night; this asks the same of the five
systems the paper's §7 reports.

Arms, per seed k (VIS seed k + IR seed k, preset `crossmodal26m`, `runs/cache_p3/seed{k}`):

  vis    the VIS stream alone
  ir     the IR stream alone, mapped to the VIS canvas
  on     fused, as shipped
  off    fused with `ir_night` and `ir_night_raw` forced all-False — V1's OFF arm, which
         removes the veto's night arm and touches nothing else

Adopts nothing. The veto axis is closed (`prereg-night-veto-v3.md` §6) and the shipped system
is frozen; a failure is a scope limit on the paper's claim.

Usage:
    python scripts/p3_night_check.py --out docs/eval/p3_night_check_2026-09-27.md
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
from eval_night_veto import off_arm                                              # noqa: E402
from uqfusion.eval.apmetrics import _score, ap_from_parts, frame_parts, presort  # noqa: E402
from uqfusion.eval.blockboot import block_resample, run_ids, run_slices          # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems              # noqa: E402
from uqfusion.eval.identity import system_identity                               # noqa: E402

PRESET = "crossmodal26m"
SEEDS = (0, 1, 2, 3, 4)
CELL = "clean"
SHIP = 0
FLOOR = 0.0060
FLOORS = (0.0014, 0.0031, 0.0060, 0.0100)
T_975_DF4 = 2.7764451051977987
BLOCK_LEN, N_BOOT, BOOT_SEED = 20, 1000, 0
ARMS = ("vis", "ir", "on", "off")
DELTAS = (("on", "vis"), ("on", "ir"), ("off", "on"), ("off", "vis"))


def ship_ap(parts, sel) -> float:
    e = ap_from_parts(parts, sel=sel)["per_class"].get(SHIP)
    return float(e["ap50_95"]) if e else float("nan")


def seed_interval(x: np.ndarray) -> tuple[float, float]:
    m, s = float(np.mean(x)), float(np.std(x, ddof=1))
    h = T_975_DF4 * s / np.sqrt(len(x))
    return m - h, m + h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/eval/p3_night_check_2026-09-27.md")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    if Path(args.out).exists():
        raise SystemExit(f"{args.out} exists; this report is written once")
    t0 = time.time()

    parts, veto, ctxs, paths, runs = {}, {}, {}, None, None
    for k in SEEDS:
        ctx = load_context(preset=PRESET, cache_dir=f"runs/cache_p3/seed{k}",
                           conditions=(CELL,), verbose=(k == 0))
        if any("pohang04" in r for r in ctx.runs):
            raise SystemExit(f"seed {k}: a pohang04 frame reached the development context")
        p = [r["image_path"] for r in ctx.vis_by_cond[CELL]]
        if paths is None:
            paths, runs = p, np.asarray(ctx.runs)
        elif p != paths:
            raise SystemExit(f"seed {k}: frame order differs from seed 0")
        r_on = run_systems(ctx, CELL)
        r_off = run_systems(off_arm(ctx), CELL)
        parts[k] = {"vis": frame_parts(ctx.vis_by_cond[CELL], ctx.gts),
                    "ir": frame_parts(r_on["ir_in_vis"], ctx.gts),
                    "on": frame_parts(r_on["fused_gated"], ctx.gts),
                    "off": frame_parts(r_off["fused_gated"], ctx.gts)}
        veto[k] = (np.asarray(r_on["veto_vis"], dtype=bool), np.asarray(r_off["veto_vis"], dtype=bool))
        ctxs[k] = ctx
        print(f"[night] seed {k} scored ({time.time() - t0:.0f}s)", flush=True)

    night = np.isin(runs, NIGHT_RUNS)
    slices = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    res = {}
    for sname, sel in slices.items():
        ap_ = {a: np.array([ship_ap(parts[k][a], sel) for k in SEEDS]) for a in ARMS}
        per_run = run_slices(run_ids([paths[i] for i in sel]))
        pre = {(k, a): presort(parts[k][a], sel) for k in SEEDS for a in ARMS}
        n = pre[(SEEDS[0], "vis")]["n_frames"]
        rng = np.random.default_rng(BOOT_SEED)
        reps = {d: np.empty(args.boot) for d in DELTAS}
        for t in range(args.boot):
            w = np.bincount(block_resample(rng, per_run, BLOCK_LEN), minlength=n)
            sc = {(k, a): _score(pre[(k, a)], w, SHIP) for k in SEEDS for a in ARMS}
            for a, b in DELTAS:
                reps[(a, b)][t] = np.mean([sc[(k, a)] - sc[(k, b)] for k in SEEDS])
        d = {}
        for a, b in DELTAS:
            x = ap_[a] - ap_[b]
            lo, hi = seed_interval(x)
            kept = reps[(a, b)][np.isfinite(reps[(a, b)])]
            blo, bhi = np.percentile(kept, [2.5, 97.5])
            d[f"{a}-{b}"] = {"per_seed": x.tolist(), "mean": float(x.mean()), "seed_ci": [lo, hi],
                             "boot_ci": [float(blo), float(bhi)], "n_boot_effective": int(kept.size),
                             "below_by_floor": {str(f): bool(x.mean() <= -f and hi < 0) for f in FLOORS}}
        fails = any(d[k]["mean"] <= -FLOOR and d[k]["seed_ci"][1] < 0 for k in ("on-vis", "on-ir"))
        res[sname] = {"n_frames": int(len(sel)), "runs": {r: int(len(i)) for r, i in per_run.items()},
                      "ap": {a: v.tolist() for a, v in ap_.items()},
                      "ap_mean": {a: float(v.mean()) for a, v in ap_.items()},
                      "ap_sd": {a: float(v.std(ddof=1)) for a, v in ap_.items()},
                      "veto_rate_on": [float(veto[k][0][sel].mean()) for k in SEEDS],
                      "veto_rate_off": [float(veto[k][1][sel].mean()) for k in SEEDS],
                      "deltas": d, "claim_fails": bool(fails)}
        print(f"[night] {sname}: " + "  ".join(f"{a} {res[sname]['ap_mean'][a]:.4f}" for a in ARMS)
              + f"  claim_fails={fails}", flush=True)

    def row(s, a):
        r = res[s]
        return (f"| {s} | `{a}` | " + " | ".join(f"{v:.4f}" for v in r["ap"][a])
                + f" | **{r['ap_mean'][a]:.4f}** | {r['ap_sd'][a]:.4f} |")

    def drow(s, key):
        x = res[s]["deltas"][key]
        return (f"| {s} | `{key}` | **{x['mean']:+.4f}** | [{x['seed_ci'][0]:+.4f}, {x['seed_ci'][1]:+.4f}] "
                f"| [{x['boot_ci'][0]:+.4f}, {x['boot_ci'][1]:+.4f}] | "
                + " ".join("Y" if x["below_by_floor"][str(f)] else "·" for f in FLOORS) + " |")

    L = [
        "Logged in `docs/exposure-ledger-2026-09-09.md` §7 (2026-09-27) before it ran. **No pohang04 "
        "frame is read.** Ship AP (class 0, AP50-95, local convention), `clean/clean`, preset "
        "`crossmodal26m`, five Phase 3 systems, caches `runs/cache_p3/seed{k}/`. Day and night are "
        "never pooled. `off` = the veto's night arm removed (V1's OFF arm). **Adopts nothing.**",
        "## Verdict on the paper's claim (fixed before the run)\n\n"
        + "\n".join(f"* **{s}**: fused ≥ max(VIS, IR) "
                    + ("**FAILS**" if res[s]["claim_fails"] else "is not refuted")
                    + f" — `on−vis` {res[s]['deltas']['on-vis']['mean']:+.4f}, "
                      f"`on−ir` {res[s]['deltas']['on-ir']['mean']:+.4f}" for s in res)
        + "\n\nFails = a seed-mean delta ≤ −0.0060 with its between-seed 95% t-interval entirely below zero.",
        "## Ship AP per arm\n\n| slice | arm | seed 0 | seed 1 | seed 2 | seed 3 | seed 4 | **mean** | sd |\n"
        "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"
        + "\n".join(row(s, a) for s in res for a in ARMS),
        "## Deltas\n\nSeed CI = between-seed 95% t-interval (df 4), the decision interval. Boot CI = paired "
        "moving-block bootstrap over frames (L = 20, seed 0), **evaluation noise only**. Last column: below "
        "zero by more than each floor 0.0014 / 0.0031 / 0.0060 / 0.0100 with the seed CI clear of zero.\n\n"
        "| slice | delta | mean | seed CI | boot CI | floors |\n|---|---|---:|---|---|---|\n"
        + "\n".join(drow(s, key) for s in res for key in res[s]["deltas"]),
        "## Veto rates (VIS dropped)\n\n| slice | frames | `on` per seed | `off` per seed |\n|---|---:|---|---|\n"
        + "\n".join(f"| {s} | {res[s]['n_frames']} | " + " ".join(f"{v:.3f}" for v in res[s]["veto_rate_on"])
                    + " | " + " ".join(f"{v:.3f}" for v in res[s]["veto_rate_off"]) + " |" for s in res),
    ]
    ident = system_identity(ctxs[0], preset=PRESET, seeds=list(SEEDS), cell=CELL, block_len=BLOCK_LEN,
                            n_boot=args.boot, bootstrap_seed=BOOT_SEED,
                            caches=[f"runs/cache_p3/seed{k}" for k in SEEDS])
    write_md(Path(args.out), "Phase 3 night check — does fused ≥ max(VIS, IR) survive the retrain?",
             L, identity=ident)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"slices": res, "config": {"preset": PRESET, "seeds": list(SEEDS), "cell": CELL, "floor": FLOOR,
                                   "block_len": BLOCK_LEN, "n_boot": args.boot, "bootstrap_seed": BOOT_SEED,
                                   "ship_class": SHIP}}, indent=1), encoding="utf-8")
    print(f"[night] done in {time.time() - t0:.0f}s -> {args.out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
