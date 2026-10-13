"""R-A1 acceptance: the local AP against official COCO AP, case by case.

The local evaluator and pycocotools compute **different metrics** -- linear
interpolation of the precision envelope versus a first-attained-recall lookup -- so
this cannot assert they are equal. What it asserts is everything that must hold for a
measured difference to *mean* the convention and nothing else:

* the COCO conversion is faithful -- identical GT and prediction counts, identical
  class sets, on every fixture;
* each named edge case is handled by both evaluators rather than crashing;
* where the two conventions provably coincide (no predictions, no GT), they agree
  exactly;
* the detection cap and image resampling behave as documented.

The cases are the ones R-A1 names: imperfect recall, duplicate recall values, tied
scores, missing classes, no predictions, wrong classes, per-image detection caps, and
image resampling.

Runs on synthetic records -- no caches, no GPU, a few seconds.

Usage:  python scripts/smoke_cocoparity.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import AP_CONVENTION, declared_policies  # noqa: E402
from uqfusion.eval.cocoparity import (  # noqa: E402
    NOISE_FLOOR, PARITY_BOUND, PARITY_TOLERANCE, TASK_CONFIG, coco_ap,
    to_coco_dicts)
from uqfusion.eval.matching import local_ap50_95, map50_95  # noqa: E402

TOL = 1e-12


def mk(gt_boxes, gt_cls, pred_boxes, pred_cls, conf, hw=(200, 200)):
    """One frame as a (record, gt) pair."""
    record = {"boxes_xyxy": np.asarray(pred_boxes, float).reshape(-1, 4),
              "cls": np.asarray(pred_cls, int),
              "conf": np.asarray(conf, float),
              "image_hw": hw}
    gt = {"boxes_xyxy": np.asarray(gt_boxes, float).reshape(-1, 4),
          "cls": np.asarray(gt_cls, int)}
    return record, gt


def box(x, y, s=40.0):
    return [x, y, x + s, y + s]


def faithful(records, gts, label):
    """The conversion must not add, drop or relabel anything."""
    gt_dict, dets, _ = to_coco_dicts(records, gts)
    n_gt_local = sum(len(g["cls"]) for g in gts)
    n_dt_local = sum(len(r["conf"]) for r in records)
    assert len(gt_dict["annotations"]) == n_gt_local, (
        f"{label}: {len(gt_dict['annotations'])} COCO annotations vs {n_gt_local} GT boxes")
    assert len(dets) == n_dt_local, (
        f"{label}: {len(dets)} COCO detections vs {n_dt_local} predictions")
    assert len(gt_dict["images"]) == len(records), f"{label}: image count differs"
    cls_local = {int(c) for g in gts for c in g["cls"]} | {
        int(c) for r in records for c in r["cls"]}
    cls_coco = {c["id"] for c in gt_dict["categories"]}
    assert cls_local == cls_coco, f"{label}: class sets differ {cls_local} vs {cls_coco}"


def compare(records, gts, label, expect_equal=False):
    faithful(records, gts, label)
    a = map50_95(records, gts)
    b = coco_ap(records, gts)
    for c in sorted(set(a["per_class"]) & set(b["per_class"])):
        assert a["per_class"][c]["n_gt"] == b["per_class"][c]["n_gt"], (
            f"{label}: cls{c} n_gt {a['per_class'][c]['n_gt']} vs {b['per_class'][c]['n_gt']}")
    d = a["map50_95"] - b["map50_95"]
    if expect_equal:
        assert abs(d) <= TOL, f"{label}: conventions must coincide here, got {d:+.3e}"
    print(f"[smoke] {label:34s} local {a['map50_95']:.8f}  coco {b['map50_95']:.8f}  "
          f"diff {d:+.8f}")
    return a, b


def main() -> int:
    # 1. imperfect recall — GT that nothing recalls, so the curve stops short of 1.0.
    recs, gts = zip(*[
        mk([box(10, 10), box(100, 100)], [0, 0], [box(10, 10)], [0], [0.9]),
        mk([box(20, 20)], [0], [box(20, 20)], [0], [0.8]),
    ])
    compare(list(recs), list(gts), "1 imperfect recall")

    # 2. duplicate recall values — FPs between TPs leave recall flat across ranks,
    #    which is exactly where a step lookup and a linear interpolation disagree.
    recs, gts = zip(*[
        mk([box(10, 10), box(120, 120)], [0, 0],
           [box(10, 10), box(60, 60), box(80, 80), box(120, 120)], [0, 0, 0, 0],
           [0.9, 0.8, 0.7, 0.6]),
    ])
    compare(list(recs), list(gts), "2 duplicate recall values")

    # 3. tied scores — every detection at the same confidence.
    recs, gts = zip(*[
        mk([box(10, 10), box(120, 120)], [0, 0],
           [box(10, 10), box(60, 60), box(120, 120)], [0, 0, 0], [0.5, 0.5, 0.5]),
    ])
    compare(list(recs), list(gts), "3 tied scores")

    # 4. missing classes — GT holds a class the predictions never emit.
    recs, gts = zip(*[
        mk([box(10, 10), box(120, 120)], [0, 1], [box(10, 10)], [0], [0.9]),
    ])
    a, b = compare(list(recs), list(gts), "4 missing class in preds")
    assert 1 in a["per_class"] and 1 in b["per_class"], "4: absent class must still be reported"

    # 5. no predictions at all — both conventions must return 0, not crash.
    recs, gts = zip(*[
        mk([box(10, 10)], [0], np.zeros((0, 4)), np.zeros(0, int), np.zeros(0)),
    ])
    compare(list(recs), list(gts), "5 no predictions", expect_equal=True)

    # 6. wrong classes — predictions land on the boxes but under the other label, so
    #    nothing may match and AP must be 0 under both.
    recs, gts = zip(*[
        mk([box(10, 10), box(120, 120)], [0, 0],
           [box(10, 10), box(120, 120)], [1, 1], [0.9, 0.8]),
    ])
    a, b = compare(list(recs), list(gts), "6 wrong classes", expect_equal=True)
    assert a["map50_95"] == 0.0, "6: class-mismatched predictions must not match"

    # 7. per-image detection cap — COCO's default maxDets=100 would silently truncate
    #    caches built at conf 0.001. Assert the cap bites when set and does not when
    #    defaulted from the data.
    #    The cap must be shown to BITE, or this case asserts only metadata and passes
    #    vacuously. So: 20 recallable GT whose matching detections all sit BELOW 100
    #    high-confidence false positives -- the shape a conf 0.001 cache actually has.
    #    Capping the ranking then keeps only the FPs and recall collapses.
    n_gt_boxes = 20
    gt_boxes = [box(float(10 * i), float(10 * (i % 7)), 8.0) for i in range(n_gt_boxes)]
    fps = [box(400.0 + 3 * i, 400.0 + 5 * i, 8.0) for i in range(100)]
    pred_boxes = fps + gt_boxes                      # FPs first, TPs after
    conf = list(np.linspace(0.99, 0.60, 100)) + list(np.linspace(0.59, 0.10, n_gt_boxes))
    recs, gts = zip(*[
        mk(gt_boxes, [0] * n_gt_boxes, pred_boxes, [0] * len(pred_boxes), conf,
           hw=(600, 600)),
    ])
    full = coco_ap(list(recs), list(gts))
    capped = coco_ap(list(recs), list(gts), max_dets=10)
    n_pred = len(pred_boxes)
    assert full["meta"]["max_dets"] == n_pred, (
        f"7: default cap should be {n_pred}, got {full['meta']['max_dets']}")
    assert full["meta"]["max_dets_source"] == "max-per-image"
    assert capped["meta"]["max_dets"] == 10 and capped["meta"]["max_dets_source"] == "explicit"
    assert full["map50_95"] > 0.0, "7 fixture is broken: uncapped AP should be positive"
    assert capped["map50_95"] == 0.0, (
        f"7 is vacuous: capping to 10 must drop every TP, got {capped['map50_95']:.8f}")
    print(f"[smoke] {'7 detection cap bites':34s} uncapped(maxDets={n_pred}) "
          f"{full['map50_95']:.8f}  capped(10) {capped['map50_95']:.8f}")

    # 8. image resampling — a frame listed twice must become two COCO images, or a
    #    bootstrap draw would silently collapse to a unique-frame set.
    recs, gts = zip(*[
        mk([box(10, 10), box(120, 120)], [0, 1],
           [box(10, 10), box(120, 120)], [0, 1], [0.9, 0.7]),
        mk([box(30, 30)], [0], [box(30, 30)], [0], [0.6]),
    ])
    dup_r = [recs[0], recs[0], recs[1]]
    dup_g = [gts[0], gts[0], gts[1]]
    gt_dict, dets, _ = to_coco_dicts(dup_r, dup_g)
    assert len(gt_dict["images"]) == 3, "8: duplicated frames must be distinct COCO images"
    assert len({d["image_id"] for d in dets}) == 3, "8: detections must not merge across copies"
    compare(dup_r, dup_g, "8 image resampling (dup frames)")
    # 9. the pinned parity bound is COHERENT and the convention is DECLARED.
    #    R-A1 option (e): the real-cache gap is asserted by
    #    `scripts/ap_convention_parity.py`, which needs the GPU caches and cannot run
    #    here. What CAN run here, always, is the check that the bound still means what
    #    the write-ups claim -- that it sits well under the noise floor, that the
    #    tolerance does not swallow that margin, and that the convention constants are
    #    exported for stamping into result files.
    trip = PARITY_BOUND * PARITY_TOLERANCE
    assert trip < NOISE_FLOOR[0], (
        f"9: parity trip point {trip:.8f} must stay below the noise-floor lower "
        f"bound {NOISE_FLOOR[0]} -- otherwise the convention gap would be allowed to "
        f"grow until it could flip a decision")
    pol = declared_policies()
    assert pol["ap_convention"] == AP_CONVENTION == "local-linear-interp", pol
    assert set(pol) == {"ap_convention", "missing_class_policy", "sort_kind"}, pol
    # and the thing that would make a "COCO column" a lie: this is NOT COCO's cap.
    assert TASK_CONFIG["max_dets"] is None, (
        "9: TASK_CONFIG uses max-per-image, not COCO's 100 -- if that ever changes, "
        "the docs saying a 'COCO' label would be a mislabel need revisiting")
    assert map50_95 is local_ap50_95, "9: the compat alias must stay wired"
    print(f"[smoke] {'9 parity bound + declared policy':34s} "
          f"trip {trip:.8f} < floor {NOISE_FLOOR[0]}  convention {AP_CONVENTION!r}")


    print("\nCOCOPARITY SMOKE OK")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
