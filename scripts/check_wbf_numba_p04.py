"""`wbf_numba` on pohang04 itself: shipped (veto on) and union fused records, fast vs reference, exact.

    py -3.13 scripts/check_wbf_numba_p04.py 3 draw941_951 blur_s2 clean
"""
import dataclasses, sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    import holdout_p04_look as look, holdout_p04_v2_rescore as v2r
    import uqfusion.uq.fusion as F
    from uqfusion.eval.ctx import run_systems
    k, s, ic, vc = int(sys.argv[1]), sys.argv[2], (None if sys.argv[3] == "clean" else sys.argv[3]), sys.argv[4]
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    sub_ = look.Pohang04() if s == "clean" else v2r.Pohang04V2()
    ctx = look.system_context(sub_, k, s, sorted({vc, "clean"}), ic, dev)
    bad = 0
    for name, c in (("shipped", ctx), ("union", dataclasses.replace(ctx, veto=None))):
        F._FAST_WBF = False
        t = time.time(); ref = run_systems(c, vc, compute_metrics=False)["fused_gated"]; t1 = time.time() - t
        F._FAST_WBF = True
        t = time.time(); got = run_systems(c, vc, compute_metrics=False)["fused_gated"]; t2 = time.time() - t
        nb = sum(any(not np.array_equal(np.asarray(a[q]), np.asarray(b[q])) for q in ("boxes_xyxy", "conf", "cls"))
                 for a, b in zip(ref, got))
        bad += nb
        print(f"{name}: {len(ref)} frames, {nb} differ; {t1:.0f}s -> {t2:.0f}s", flush=True)
    print("IDENTICAL" if bad == 0 else f"MISMATCH {bad}")
    return int(bad != 0)


if __name__ == "__main__":
    raise SystemExit(main())
