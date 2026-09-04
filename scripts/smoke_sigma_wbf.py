"""Prove `sigma_weighted_fusion` IS WBF when sigma carries no information — TODO A1.

The sigma-weighted fusion in `uqfusion.uq.fusion` is a reimplementation of
`ensemble_boxes.weighted_boxes_fusion`, changed in exactly one place: cluster
coordinates are averaged by ``score x model_weight / sigma^2`` instead of
``score x model_weight``. If any OTHER difference crept in — clustering order,
the score rescale, the label grouping — then an A/B between them would measure
that difference and not sigma, and the paper's central claim would rest on an
implementation artifact.

So this asserts two things on randomised inputs:

  1. ``use_sigma=False`` reproduces stock WBF to floating-point tolerance
     (boxes, scores AND labels, in order).
  2. ``use_sigma=True`` with CONSTANT sigma also reproduces stock WBF — because
     a constant factor cancels in a weighted mean. This is the stronger check:
     it says the sigma path itself is a no-op when sigma is uninformative, so
     any measured difference later is attributable to sigma VARIATION.

Then it shows the intended behaviour actually happens: with two boxes in a
cluster, tightening one box's sigma pulls the fused box toward it.

Usage:
    python scripts/smoke_sigma_wbf.py
"""

from __future__ import annotations

import sys

import warnings

import numpy as np
from ensemble_boxes import weighted_boxes_fusion

from uqfusion.uq.fusion import sigma_weighted_fusion

TOL = 1e-6


def random_case(rng, n_models=2, max_boxes=8, n_labels=2):
    boxes_list, scores_list, labels_list = [], [], []
    for _ in range(n_models):
        n = int(rng.integers(0, max_boxes + 1))
        xy = np.sort(rng.random((n, 2, 2)), axis=1).reshape(n, 4)[:, [0, 2, 1, 3]]
        boxes_list.append(xy.tolist())
        scores_list.append(rng.random(n).tolist())
        labels_list.append(rng.integers(0, n_labels, n).astype(float).tolist())
    return boxes_list, scores_list, labels_list


def compare(a, b, what):
    ab, asc, al = a
    bb, bsc, bl = b
    assert ab.shape == bb.shape, f"{what}: shape {ab.shape} vs {bb.shape}"
    if ab.size:
        db = float(np.abs(ab - bb).max())
        ds = float(np.abs(asc - bsc).max())
        dl = float(np.abs(al - bl).max())
        assert db < TOL and ds < TOL and dl < TOL, \
            f"{what}: max diff boxes {db:.3e} scores {ds:.3e} labels {dl:.3e}"
        return db, ds
    return 0.0, 0.0


def main() -> int:
    warnings.filterwarnings("ignore", message=".*value in box.*")
    rng = np.random.default_rng(0)
    worst_off, worst_const = 0.0, 0.0
    n_boxes_seen = 0

    for trial in range(300):
        bl, sl, ll = random_case(rng)
        weights = [float(rng.uniform(0.05, 1.0)), float(rng.uniform(0.05, 1.0))]
        iou_thr = float(rng.choice([0.4, 0.55, 0.7, 0.85, 0.9]))
        skip = float(rng.choice([0.0, 0.1]))

        ref = weighted_boxes_fusion(bl, sl, ll, weights=weights,
                                    iou_thr=iou_thr, skip_box_thr=skip)
        n_boxes_seen += len(ref[1])

        # 1. sigma path disabled
        zeros = [np.zeros((len(b), 4)).tolist() for b in bl]
        off = sigma_weighted_fusion(bl, sl, ll, zeros, weights,
                                    iou_thr=iou_thr, skip_box_thr=skip, use_sigma=False)
        d = compare(ref, off, f"trial {trial} use_sigma=False")
        worst_off = max(worst_off, d[0])

        # 2. sigma path enabled but CONSTANT — must cancel
        const = [np.full((len(b), 4), 0.037).tolist() for b in bl]
        on = sigma_weighted_fusion(bl, sl, ll, const, weights,
                                   iou_thr=iou_thr, skip_box_thr=skip, use_sigma=True)
        d = compare(ref, on, f"trial {trial} constant sigma")
        worst_const = max(worst_const, d[0])

    print(f"[sigma-smoke] 300 randomised cases, {n_boxes_seen} fused boxes compared")
    print(f"[sigma-smoke] use_sigma=False   max |box diff| vs stock WBF: {worst_off:.2e}")
    print(f"[sigma-smoke] constant sigma    max |box diff| vs stock WBF: {worst_const:.2e}")

    # 3. sigma actually moves the box, in the right direction and by the right amount
    # Must actually overlap, or WBF never clusters them and there is nothing to average.
    # The two boxes must overlap (or WBF never clusters them) AND differ on both
    # axes (or "y1 did not move" is trivially true because both y1 are equal).
    b = [[[0.10, 0.10, 0.30, 0.30]], [[0.20, 0.20, 0.40, 0.40]]]   # IoU = 1/7
    s = [[0.9], [0.9]]
    lab = [[0.0], [0.0]]
    wts = [1.0, 1.0]
    equal = sigma_weighted_fusion(b, s, lab, [[[0.02] * 4], [[0.02] * 4]], wts, iou_thr=0.01)
    sharp = sigma_weighted_fusion(b, s, lab, [[[0.01] * 4], [[0.02] * 4]], wts, iou_thr=0.01)
    print(f"[sigma-smoke] equal sigma       fused x1 = {equal[0][0][0]:.4f}  (midpoint 0.1500)")
    print(f"[sigma-smoke] first box 2x tighter -> x1 = {sharp[0][0][0]:.4f}  (expect 0.1200)")
    assert abs(equal[0][0][0] - 0.15) < TOL, "equal sigma must give the midpoint"
    # precisions 1/0.01^2 = 10000 and 1/0.02^2 = 2500 -> weights 0.8 / 0.2
    assert abs(sharp[0][0][0] - (0.8 * 0.10 + 0.2 * 0.20)) < TOL, "inverse-variance mean is wrong"
    assert sharp[0][0][0] < equal[0][0][0], "tightening box 1 must pull the fused box toward it"

    # 4. per-coordinate independence: tighten ONLY x1, y1 must not move
    mixed = sigma_weighted_fusion(b, s, lab, [[[0.01, 0.02, 0.02, 0.02]], [[0.02] * 4]],
                                  wts, iou_thr=0.01)
    assert abs(mixed[0][0][0] - (0.8 * 0.10 + 0.2 * 0.20)) < TOL, "x1 should move"
    assert abs(mixed[0][0][1] - 0.15) < TOL, "y1 must NOT move when only x1's sigma changed"
    print(f"[sigma-smoke] per-coordinate    x1 = {mixed[0][0][0]:.4f} moved, "
          f"y1 = {mixed[0][0][1]:.4f} unchanged")

    print("\nSIGMA-WBF SMOKE OK — reduces to stock WBF exactly; sigma moves coordinates "
          "by inverse variance, per coordinate")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
