"""Smoke test for VIS soft-NMS and the `crossmodal26m_snms` preset.

Checks the properties that make this arm different from the merging family that
keeps losing, and the isolation that keeps every published number reproducible:

  A. sigma travels with its box (the `nms_record` shape trap, restated)
  B. coordinates are NEVER modified -- only scores and membership
  C. scores are non-increasing, and only overlapping same-class boxes decay
  D. a box with no same-class neighbour is untouched
  E. `crossmodal26m` is bit-identical with the parameter absent
  F. `crossmodal26m_snms` actually turns it on

Usage:
    python scripts/smoke_vis_soft_nms.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.irdedup import soft_nms_record, soft_nms_records  # noqa: E402


def _rec(seed=0, n=40):
    rng = np.random.default_rng(seed)
    cx, cy = rng.uniform(20, 600, n), rng.uniform(20, 600, n)
    w, h = rng.uniform(6, 60, n), rng.uniform(6, 60, n)
    b = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    return {"boxes_xyxy": b, "conf": rng.uniform(0.01, 0.99, n),
            "cls": rng.integers(0, 2, n), "sigma_ltrb": rng.uniform(0.5, 4.0, (n, 4)),
            "image_hw": (640, 640)}


def main() -> int:
    fails = []

    def check(name, ok, extra=""):
        print(f"[smoke] {name}  {'OK' if ok else 'FAIL'}{(' ' + extra) if extra else ''}")
        if not ok:
            fails.append(name)

    # ---- A/B/C on synthetic records --------------------------------------
    for seed in range(5):
        r = _rec(seed)
        o = soft_nms_record(r, 0.5)
        n = len(o["conf"])
        check(f"A shapes aligned (seed {seed})",
              len(o["boxes_xyxy"]) == n == len(o["cls"]) == len(o["sigma_ltrb"]),
              f"n={n}")
        # every surviving box must be one of the originals, coordinates untouched
        src = np.asarray(r["boxes_xyxy"], float)
        out = np.asarray(o["boxes_xyxy"], float)
        exact = all(np.isclose(out[i], src).all(1).any() for i in range(len(out)))
        check(f"B coordinates never modified (seed {seed})", exact)
        # and its sigma row must be the row that came with it
        rows_ok = True
        for i in range(len(out)):
            j = int(np.flatnonzero(np.isclose(src, out[i]).all(1))[0])
            rows_ok &= np.allclose(np.asarray(o["sigma_ltrb"])[i], r["sigma_ltrb"][j])
        check(f"A sigma row follows its box (seed {seed})", rows_ok)
        # scores only ever decay
        smax = float(np.max(o["conf"])) if n else 0.0
        check(f"C scores non-increasing (seed {seed})",
              smax <= float(np.max(r["conf"])) + 1e-12)

    # ---- D an isolated box is untouched ----------------------------------
    r = {"boxes_xyxy": np.array([[0, 0, 10, 10], [300, 300, 320, 320]], float),
         "conf": np.array([0.9, 0.4]), "cls": np.array([0, 0]),
         "sigma_ltrb": np.ones((2, 4))}
    o = soft_nms_record(r, 0.5)
    check("D disjoint boxes untouched", np.allclose(np.sort(o["conf"]), [0.4, 0.9]),
          f"conf={np.round(np.asarray(o['conf']), 4).tolist()}")

    # a heavy overlap MUST decay
    r2 = {"boxes_xyxy": np.array([[0, 0, 10, 10], [0, 0, 10, 10]], float),
          "conf": np.array([0.9, 0.9]), "cls": np.array([0, 0]),
          "sigma_ltrb": np.ones((2, 4))}
    o2 = soft_nms_record(r2, 0.5)
    lo = float(np.min(o2["conf"]))
    check("C identical duplicate decays", lo < 0.9 * np.exp(-1 / 0.5) + 1e-6,
          f"0.9 -> {lo:.4f}")

    check("soft_nms_records maps over a list",
          len(soft_nms_records([_rec(9), _rec(10)], 0.5)) == 2)

    # ---- E/F the preset isolation ----------------------------------------
    try:
        from uqfusion.eval.ctx import load_context, run_systems
        from uqfusion.eval.apmetrics import ap_from_parts, frame_parts
        cache = "runs/cache_m"
        if (ROOT / cache / "gauss_vis_paired_clean.pkl").is_file():
            a = load_context(preset="crossmodal26m", cache_dir=cache,
                             conditions=("clean",), verbose=False)
            b = load_context(preset="crossmodal26m_snms", cache_dir=cache,
                             conditions=("clean",), verbose=False)
            na = np.mean([len(r["conf"]) for r in a.vis_by_cond["clean"]])
            nb = np.mean([len(r["conf"]) for r in b.vis_by_cond["clean"]])
            ca = np.concatenate([np.asarray(r["conf"], float)
                                 for r in a.vis_by_cond["clean"]])
            cb = np.concatenate([np.asarray(r["conf"], float)
                                 for r in b.vis_by_cond["clean"]])
            # The real isolation test: `crossmodal26m` must hand back the cache
            # bit-for-bit, or `final_26m_grid_v2.md` stops reproducing.
            from uqfusion.eval.cache import load_cache
            raw, _ = load_cache(ROOT / cache / "gauss_vis_paired_clean.pkl")
            same = len(raw) == len(a.vis_by_cond["clean"]) and all(
                np.array_equal(np.asarray(x["boxes_xyxy"], float),
                               np.asarray(y["boxes_xyxy"], float))
                and np.array_equal(np.asarray(x["conf"], float),
                                   np.asarray(y["conf"], float))
                for x, y in zip(raw, a.vis_by_cond["clean"]))
            check("E crossmodal26m VIS stream bit-identical to the cache", same,
                  f"{na:.1f} boxes/frame")
            check("F crossmodal26m_snms live",
                  nb <= na and float(cb.sum()) < float(ca.sum()),
                  f"{na:.1f} -> {nb:.1f} boxes/frame, "
                  f"score mass {ca.sum():.0f} -> {cb.sum():.0f}")
            pa = frame_parts(run_systems(a, "clean")["fused_gated"], a.gts)
            pb = frame_parts(run_systems(b, "clean")["fused_gated"], b.gts)
            from uqfusion.eval.ctx import NIGHT_RUNS
            day = np.flatnonzero(~np.isin(a.runs, NIGHT_RUNS))
            m0 = ap_from_parts(pa, sel=day)["map50_95"]
            m1 = ap_from_parts(pb, sel=day)["map50_95"]
            check("F preset moves the fused day number", m1 > m0,
                  f"{m0:.4f} -> {m1:.4f} ({m1 - m0:+.4f})")
        else:
            print(f"[smoke] E/F skipped: {cache} absent")
    except Exception as e:  # noqa: BLE001
        check("E/F preset wiring", False, f"{type(e).__name__}: {e}")

    print()
    if fails:
        print(f"VIS SOFT-NMS SMOKE FAILED: {fails}")
        return 1
    print("VIS SOFT-NMS SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
