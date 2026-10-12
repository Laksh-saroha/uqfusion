"""The numba greedy match against the reference Python loop, and `compute_metrics=False` (2026-10-10).

`matching.tp_matrix` now runs its greedy match through a numba kernel (`_greedy_nb`). The outputs
are boolean, so this demands exact equality: on every frame of VIS, IR-in-VIS and fused records of
several development cells (crossmodal26m, seed 0, draw 941, IR clean and fog s3), and on 20,000
synthetic frames built to stress ties (quantised IoUs, repeated confidences, empty sides). It also
checks that `evaluate_systems(compute_metrics=False)` returns the same per-frame outputs as the
default, and reports the speed of both match paths.

    py -3.13 scripts/check_tp_numba.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from uqfusion.eval import matching as M                              # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems              # noqa: E402
from v2_sensitivity import CACHE, STATS, sub                         # noqa: E402


def inputs(rec, gt):
    ious = M.iou_matrix(np.asarray(rec["boxes_xyxy"], float).reshape(-1, 4), gt["boxes_xyxy"])
    same = np.asarray(rec["cls"])[:, None] == gt["cls"][None, :]
    order = np.argsort(-np.asarray(rec["conf"], float), kind=M.SORT_KIND).astype(np.int64)
    return np.ascontiguousarray(ious * same, dtype=np.float64), order


def main() -> int:
    assert M._greedy is not M._greedy_py, "numba path not active"
    lv = np.asarray(M.IOU_LEVELS, dtype=np.float64)
    n_frames = n_true = 0
    t_py = t_nb = 0.0
    for ic in (None, "fog_s3"):
        d = f"{STATS}/brightness/{sub(941)}"
        ctx = load_context(preset="crossmodal26m", cache_dir=f"{CACHE}/seed0/{sub(941)}", bright_dir=d,
                           structure_dir=f"{STATS}/structure/{sub(941)}",
                           ir_bright=f"{d}/gauss_ir_paired_{ic or 'clean'}.json",
                           conditions=("clean", "lowlight", "fog"), ir_condition=ic, verbose=False)
        for c in ("clean", "lowlight", "fog"):
            full = run_systems(ctx, c)
            lite = run_systems(ctx, c, compute_metrics=False)
            for key in ("veto_vis", "veto_ir", "w_vis_gated"):
                assert list(full[key]) == list(lite[key]), f"compute_metrics changed {key} on {c}/{ic}"
            for a, b in zip(full["fused_gated"], lite["fused_gated"]):
                assert np.array_equal(a["boxes_xyxy"], b["boxes_xyxy"]) and np.array_equal(a["conf"], b["conf"])
            assert "gated_fusion" not in lite and "gated_fusion" in full
            for recs in (ctx.vis_by_cond[c], full["ir_in_vis"], full["fused_gated"]):
                for rec, gt in zip(recs, ctx.gts):
                    rec = {k: np.asarray(v) for k, v in rec.items() if k in ("boxes_xyxy", "conf", "cls")}
                    ious, order = inputs(rec, gt)
                    t = time.perf_counter(); a = M._greedy_py(ious, order, lv); t_py += time.perf_counter() - t
                    t = time.perf_counter(); b = M._greedy(ious, order, lv); t_nb += time.perf_counter() - t
                    if not np.array_equal(a, b):
                        raise SystemExit(f"MISMATCH on {c}/{ic}")
                    n_frames += 1
                    n_true += int(a.sum())
    rng = np.random.default_rng(0)
    for _ in range(20000):
        n, m = int(rng.integers(0, 40)), int(rng.integers(0, 12))
        ious = np.round(rng.random((n, m)) * rng.choice([1, 0.5, 0.0]), int(rng.integers(1, 3)))
        conf = np.round(rng.random(n), 1)
        order = np.argsort(-conf, kind="stable").astype(np.int64)
        if not np.array_equal(M._greedy_py(ious, order, lv), M._greedy(np.ascontiguousarray(ious), order, lv)):
            raise SystemExit("MISMATCH on a synthetic frame")
    print(f"identical on {n_frames:,} real frames ({n_true:,} TP flags) and 20,000 synthetic tie-heavy frames; "
          f"compute_metrics=False leaves vetoes, weights and fused boxes unchanged on 6 cells. "
          f"Match time: python {t_py:.1f}s, numba {t_nb:.2f}s ({t_py / max(t_nb, 1e-9):.0f}x).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
