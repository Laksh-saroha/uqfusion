"""Pins `apmetrics` to `matching.map50_95` — MUST pass before any CI is quoted.

Three layers have to agree or the bootstrap is measuring its own arithmetic:

  A. `ap_from_parts` == `matching.map50_95`, pooled and on frame subsets
  B. `ap_weighted` at unit weights == `ap_from_parts`
  C. `ap_weighted` at integer weights == `ap_from_parts` on the physically
     duplicated frame list (the literal resample)

C is the one worth stating: a bootstrap draw is a dataset with frames repeated,
and AP is not linear in the frames, so "weight it" and "duplicate it" are only
the same thing if the duplication happens in conf order. It does, and this
asserts it rather than assuming it.

Runs on synthetic records — no caches, no GPU, ~2 s.

Usage:  python scripts/smoke_apmetrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, ap_weighted, frame_parts, presort  # noqa: E402
from uqfusion.eval.matching import map50_95  # noqa: E402

TOL = 1e-12


def synth(n_frames=60, seed=0):
    """Frames with real overlap structure: GT boxes plus jittered predictions,
    some frames empty, both classes present, confidences overlapping."""
    rng = np.random.default_rng(seed)
    records, gts = [], []
    for _ in range(n_frames):
        n_gt = int(rng.integers(0, 6))
        gb = np.column_stack([
            rng.uniform(0, 560, n_gt), rng.uniform(0, 560, n_gt),
            np.zeros(n_gt), np.zeros(n_gt)])
        gb[:, 2] = gb[:, 0] + rng.uniform(10, 70, n_gt)
        gb[:, 3] = gb[:, 1] + rng.uniform(10, 70, n_gt)
        gcls = rng.integers(0, 2, n_gt)
        gts.append({"boxes_xyxy": gb.astype(float), "cls": gcls.astype(int)})

        n_p = int(rng.integers(0, 9))
        if n_gt and n_p:
            pick = rng.integers(0, n_gt, n_p)
            pb = gb[pick] + rng.normal(0, 9, (n_p, 4))
            pcls = np.where(rng.random(n_p) < 0.85, gcls[pick], 1 - gcls[pick])
        else:
            pb = np.column_stack([
                rng.uniform(0, 560, n_p), rng.uniform(0, 560, n_p),
                np.zeros(n_p), np.zeros(n_p)])
            pb[:, 2] = pb[:, 0] + rng.uniform(10, 70, n_p)
            pb[:, 3] = pb[:, 1] + rng.uniform(10, 70, n_p)
            pcls = rng.integers(0, 2, n_p)
        records.append({"boxes_xyxy": pb.astype(float),
                        "cls": np.asarray(pcls, dtype=int),
                        "conf": rng.random(n_p)})
    return records, gts


def main() -> int:
    records, gts = synth()
    parts = frame_parts(records, gts)

    ref = map50_95(records, gts)
    mine = ap_from_parts(parts)
    for k in ("map50_95", "map50"):
        d = abs(ref[k] - mine[k])
        assert d <= TOL, f"A pooled {k}: {mine[k]} vs {ref[k]} (d={d:.3e})"
    print(f"[smoke] A pooled: map50-95 {mine['map50_95']:.8f} == reference OK")

    rng = np.random.default_rng(7)
    for t in range(5):
        sel = np.sort(rng.choice(len(parts), size=len(parts) // 2, replace=False))
        r = map50_95([records[i] for i in sel], [gts[i] for i in sel])
        m = ap_from_parts(parts, sel)
        d = abs(r["map50_95"] - m["map50_95"])
        assert d <= TOL, f"A subset {t}: {m['map50_95']} vs {r['map50_95']} (d={d:.3e})"
    print("[smoke] A subsets: 5 random halves match the reference exactly OK")

    pre = presort(parts)
    u = ap_weighted(pre)
    d = abs(u["map50_95"] - mine["map50_95"])
    assert d <= TOL, f"B unit weights: {u['map50_95']} vs {mine['map50_95']} (d={d:.3e})"
    assert set(u["per_class"]) == set(mine["per_class"]), "B class sets differ"
    for c in u["per_class"]:
        assert abs(u["per_class"][c]["ap50_95"] - mine["per_class"][c]["ap50_95"]) <= TOL
        assert u["per_class"][c]["n_gt"] == mine["per_class"][c]["n_gt"]
    print(f"[smoke] B unit weights == reference path, per class {list(u['per_class'])} OK")

    for t in range(4):
        w = rng.integers(0, 4, len(parts))
        if w.sum() == 0:
            continue
        fast = ap_weighted(pre, w)["map50_95"]
        # the literal resample: frame i physically present w[i] times
        dup = np.concatenate([np.full(int(k), i) for i, k in enumerate(w) if k > 0])
        slow = ap_from_parts(parts, dup)["map50_95"]
        d = abs(fast - slow)
        assert d <= TOL, f"C weights {t}: fast {fast} vs duplicated {slow} (d={d:.3e})"
    print("[smoke] C integer weights == physically duplicated frame list OK")

    empty = ap_weighted(presort(parts, np.zeros(0, dtype=int)))
    assert empty["map50_95"] == 0.0, "empty selection must be 0.0, not a crash"
    print("[smoke] D empty selection handled OK")

    print("\nAPMETRICS SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
