"""`wbf_numba.wbf_fast` against the Python `sigma_weighted_fusion`, on real records: exact equality required.

    py -3.13 scripts/check_wbf_numba.py

1. Checkpoint-ensemble merges (`seed_ensemble.merge`, 10 views = 5 checkpoints + 5 flips, plain WBF 0.55)
   on development frames, VIS and IR.
2. Gate-free fusion (`run_systems`, veto off: WBF 0.85 + support 0.30 / x1.5, per-frame weights) on
   development frames, clean and corrupted cells.
Every fused record must match in boxes, scores and labels with `np.array_equal` (no tolerance).
"""

from __future__ import annotations

import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def same(a: list, b: list) -> int:
    bad = 0
    for x, y in zip(a, b):
        for key in ("boxes_xyxy", "conf", "cls"):
            if not np.array_equal(np.asarray(x[key]), np.asarray(y[key])):
                bad += 1
                break
    return bad


def main() -> int:
    import seed_ensemble as se
    import uqfusion.uq.fusion as F
    from uqfusion.eval.ctx import load_context, run_systems
    total_bad = 0
    # 1. ensemble merges
    for mod, cond in (("vis", "clean"), ("ir", None), ("vis", "fog"), ("ir", "fog_s3")):
        per = [se._per_ckpt(945, mod, cond, k, None) for k in se.SEEDS] + [se._flip_ckpt(945, mod, cond, k) for k in se.SEEDS]
        frames = list(zip(*per))
        F._FAST_WBF = False
        t = time.time()
        ref = [se.merge(list(fr)) for fr in frames]
        t_ref = time.time() - t
        F._FAST_WBF = True
        se.merge(list(frames[0]))                          # compile
        t = time.time()
        got = [se.merge(list(fr)) for fr in frames]
        t_fast = time.time() - t
        bad = same(ref, got)
        total_bad += bad
        print(f"merge {mod}/{cond}: {len(frames)} frames, {bad} differ; {t_ref:.1f}s -> {t_fast:.2f}s", flush=True)
    se._CACHE.clear()
    # 2. gate-free fusion with support
    d = f"{se.STATS}/brightness/{se.sub(941)}"
    for ic, vc in ((None, "clean"), ("fog_s3", "clean"), (None, "fog"), ("noise_s2", "lowlight")):
        ctx = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed0/{se.sub(941)}", bright_dir=d,
                           structure_dir=f"{se.STATS}/structure/{se.sub(941)}",
                           ir_bright=f"{d}/gauss_ir_paired_{ic or 'clean'}.json", conditions=(vc,),
                           ir_condition=ic, verbose=False)
        u = dataclasses.replace(ctx, veto=None)
        vis = se._stream(941, "vis", vc, (0,), None, False)
        ir = se._stream(941, "ir", ic, (0,), None, False)
        F._FAST_WBF = False
        t = time.time()
        ref = run_systems(u, vc, compute_metrics=False, vis_records=vis, ir_records=ir)["fused_gated"]
        t_ref = time.time() - t
        F._FAST_WBF = True
        t = time.time()
        got = run_systems(u, vc, compute_metrics=False, vis_records=vis, ir_records=ir)["fused_gated"]
        t_fast = time.time() - t
        bad = same(ref, got)
        total_bad += bad
        print(f"fusion {vc}/{ic}: {len(ref)} frames, {bad} differ; run_systems {t_ref:.1f}s -> {t_fast:.1f}s", flush=True)
        se._CACHE.clear()
    print("IDENTICAL" if total_bad == 0 else f"MISMATCH on {total_bad} frames")
    return 0 if total_bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
