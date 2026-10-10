"""The IR merge veto under corruption v2, development, descriptive (2026-10-10).

`crossmodal26m` switched the IR merge veto off (`ctx.ir_merge_veto = False`): dropping an
unhealthy IR stream from the merge while VIS is healthy was measured net-harmful on the v1 IR
corruptions (`ctx.py`, "MEASURED HARMFUL"). Under v2, `v2_fallback_cost.py` found the union
below VIS alone with clean VIS and IR fog s3 by day (-0.0096, no veto firing): the near-dead IR
stream's boxes enter the merge. This re-measures that one switch on the fallback-cost cells,
on against off, paired by system: everything else is the shipped preset. A third arm turns the
cross-modal support bonus off instead (`support_gamma` 0: a VIS box overlapping any IR box at
IoU 0.30 no longer has its score multiplied by 1.5), to see whether junk IR boxes act through it.

Descriptive; changes no preset (exposure ledger, 2026-10-10). Scores no pohang04 frame.

    py -3.13 scripts/v2_ir_merge_veto.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from uqfusion.eval.apmetrics import frame_parts                    # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402
from v2_sensitivity import CACHE, DRAW, PRESET, SEEDS, STATS, T_975_DF4, ship_ap, sub  # noqa: E402

VIS_CONDS = ("clean", "lowlight")
IR_CONDS = (None, "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")
#: arm -> (ir_merge_veto, support_gamma or None for the preset's)
ARMS = {"shipped": (False, None), "merge_veto": (True, None), "no_support": (False, 0.0)}


def score(job):
    k, ir_cond = job
    d = f"{STATS}/brightness/{sub(DRAW)}"
    ctx = load_context(preset=PRESET, cache_dir=f"{CACHE}/seed{k}/{sub(DRAW)}", bright_dir=d,
                       structure_dir=f"{STATS}/structure/{sub(DRAW)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json",
                       conditions=VIS_CONDS, ir_condition=ir_cond, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    out = {}
    gamma = ctx.support_gamma
    for arm, (flag, g) in ARMS.items():
        ctx.ir_merge_veto, ctx.support_gamma = flag, (gamma if g is None else g)
        for c in VIS_CONDS:
            r = run_systems(ctx, c)
            parts = frame_parts(r["fused_gated"], ctx.gts)
            vi = r.get("veto_ir")
            for s, idx in sl.items():
                out[(c, ir_cond, s, arm)] = ship_ap(parts, idx)
                out[(c, ir_cond, s, arm, "ir_drop")] = (float(np.asarray(vi, bool)[idx].mean())
                                                        if vi is not None else float("nan"))
    return k, out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/eval/corruption_v2/ir_merge_veto.md")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    res: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, out in ex.map(score, [(k, ic) for ic in IR_CONDS for k in SEEDS]):
            for key, v in out.items():
                res.setdefault(key, {})[k] = v
    L = [f"Shipped `{PRESET}` with the IR merge veto off (as shipped) and on; everything else unchanged. "
         f"Five Phase 3 systems, VIS draw {DRAW}, IR draw {DRAW + 10}; ship AP of the fused output, mean "
         "over systems; Δ = on − off, paired by system, 95% t-interval (df 4). `IR dropped` is the share "
         "of frames on which the veto removes IR from the merge. `no support` = shipped with the "
         "cross-modal support bonus off.", "",
         "| VIS | IR | slice | fused, shipped | merge veto on: Δ [95% CI] | IR dropped | no support: Δ [95% CI] |",
         "|---|---|---|---:|---|---:|---|"]
    js = []
    for c in VIS_CONDS:
        for ic in IR_CONDS:
            for s in ("day", "night"):
                a = {arm: np.array([res[(c, ic, s, arm)][k] for k in SEEDS]) for arm in ARMS}
                d = {}
                for arm in ("merge_veto", "no_support"):
                    dd = a[arm] - a["shipped"]
                    m, h = float(dd.mean()), float(T_975_DF4 * dd.std(ddof=1) / np.sqrt(len(SEEDS)))
                    d[arm] = (m, m - h, m + h)
                drop = float(np.mean([res[(c, ic, s, "merge_veto", "ir_drop")][k] for k in SEEDS]))
                f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
                L.append(f"| {c} | {ic or 'clean'} | {s} | {a['shipped'].mean():.4f} | {f(d['merge_veto'])} | "
                         f"{drop:.3f} | {f(d['no_support'])} |")
                js.append({"vis": c, "ir": ic or "clean", "slice": s,
                           **{arm: float(a[arm].mean()) for arm in ARMS},
                           "delta": {arm: list(t) for arm, t in d.items()}, "ir_dropped": drop,
                           "per_system": {arm: a[arm].tolist() for arm in ARMS}})
    out = ROOT / args.out
    out.write_text("# The IR merge veto under corruption v2\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"cells": js, "elapsed_s": time.time() - t0}, indent=1),
                                        encoding="utf-8")
    print("\n".join(L))
    print(f"[irv] wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    raise SystemExit(main())
