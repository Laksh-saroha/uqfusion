"""`boot_cpu` against the GPU `boot_batch` replicates already on disk (all 1,000, ==) and the CPU `_score`."""
import sys, time
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]


def main():
    from holdout_p04_v2_rescore import load_ship_pre
    from uqfusion.eval.apmetrics import _score
    from boot_cpu import boot_cpu
    import p04_heldout_score as ps
    W = np.load(ps.OUT / "boot_weights.npy")
    dirs = [d for d in ps.presort_dirs() if (d / "reps.npy").is_file()]
    pick = [dirs[i] for i in np.linspace(0, len(dirs) - 1, int(sys.argv[1]) if len(sys.argv) > 1 else 4).astype(int)]
    bad = 0
    for d in pick:
        pre = load_ship_pre(d)
        rows = np.vstack([np.ones(pre["n_frames"], np.int64), W[:2]])
        got = boot_cpu(pre, rows)
        for w, g in zip(rows, got):
            a = _score(pre, w, ps.SHIP)
            bad += not (a == g or (np.isnan(a) and np.isnan(g)))
        t = time.time()
        reps = boot_cpu(pre, W)
        dt = time.time() - t
        ref = np.load(d / "reps.npy")
        nb = int((~((reps == ref) | (np.isnan(reps) & np.isnan(ref)))).sum())
        bad += nb
        print(f"{d.relative_to(ps.PRE)}: {nb} of {len(W)} replicates differ from the GPU; {dt:.1f}s", flush=True)
    print("IDENTICAL" if bad == 0 else f"MISMATCH {bad}")
    return int(bad != 0)


if __name__ == "__main__":
    raise SystemExit(main())
