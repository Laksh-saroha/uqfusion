"""Official COCO AP via `pycocotools`, as the authority the local AP is checked against.

R-A1 (`docs/TODO-2026-09-09-architecture-review.md`, finding F03). The local
evaluator in `matching.map50_95` / `apmetrics` interpolates the precision envelope
**linearly** with ``np.interp``; official COCO samples the envelope at the *first
attained recall at or above* each threshold, with ``searchsorted``. Those are
different metrics, not two implementations of one.

The difference is not hypothetical and it is not tiny. Reproduced exactly, to the
digit, from the review's two hand-built examples::

    ranked outcomes    project AP    COCO AP     difference
    TP, FP, TP         0.83168317    0.83498350  -0.00330033
    TP, FP, FP, TP     0.74752475    0.75247525  -0.00495050

against a measured 2-sigma noise floor of 0.0014-0.0031
(`runs/eval/metric_noise_floor.md`) and adopted margins smaller than both.

**What this module is for.** Not to replace the local AP by fiat -- to make the two
comparable so the size of the difference on real data can be measured, which is what
R-A5's change-impact table needs. Every knob that could confound that comparison is
pinned in `TASK_CONFIG` and reported back with the result, so a divergence can be
attributed to the interpolation convention rather than to an unstated area range or a
silently truncating detection cap.

**Three conventions, not two.** Ultralytics uses its own integration convention as
well. "Matches Ultralytics" is not "matches COCO", and neither is "matches our local
AP". `docs/phase1-experimental-record.md` already records an ultralytics-version
difference of ~0.034 mAP on identical weights, which is an order above everything
discussed here -- Phase 1 ultralytics numbers must not be compared directly against
custom fusion AP under any convention.
"""

from __future__ import annotations

import contextlib
import io
from typing import Any

import numpy as np

from uqfusion.eval.matching import IOU_LEVELS

PARITY_BOUND = 0.00028501
"""Worst local-vs-COCO disagreement in a DELTA, measured on this project's caches.

Measured by `scripts/ap_convention_parity.py` and recorded in
`docs/eval/ap_convention_parity_2026-09-09.md`. Pinned here (R-A1, option (e)) so the
fact is ASSERTED rather than merely written down once: the parity script now fails if
the gap exceeds `PARITY_BOUND * PARITY_TOLERANCE`, which catches the day a change to
matching, sorting or the envelope quietly widens it.

A companion "COCO" column on every results table was the alternative and was rejected
twice over: it would always agree to 5x below the noise floor, which teaches readers
to skip it, and `TASK_CONFIG` sets `max_dets: None` rather than COCO's 100, so a
column labelled "COCO" would be mislabelled in precisely the way this whole review
item is about.
"""

PARITY_TOLERANCE = 1.5
"""Headroom on `PARITY_BOUND` before the parity script fails.

Not zero: the bound is a measurement over specific caches, and a legitimate change
(a new arm, a re-run detector) can move it slightly. 1.5x still leaves the trip point
at 0.00043, well under the 0.0014 noise-floor lower bound, so an assertion failure
means the gap grew toward mattering -- not that it wobbled.
"""

NOISE_FLOOR = (0.0014, 0.0031)
"""Paired 2-sigma noise floor (`runs/eval/metric_noise_floor.md`), for context.

`PARITY_BOUND` is 5x below the lower end. That ratio is the reason the local
convention is kept rather than replaced.
"""

TASK_CONFIG: dict[str, Any] = {
    "iou_thrs": [float(x) for x in IOU_LEVELS],
    "rec_thrs": 101,
    "area_rng": [[0.0, 1e10]],
    "area_rng_lbl": ["all"],
    "max_dets": None,
    "iscrowd": 0,
    "ignore": "none",
    "score_sort": "stable",
}
"""Every knob that could make COCO and the local AP disagree for a reason other than
the interpolation convention. Pinned so a measured divergence means what it looks like.

``iou_thrs`` -- the local 0.50:0.05:0.95 sweep, not pycocotools' default (which is the
same sweep, but stated rather than inherited).

``area_rng`` -- a single all-inclusive range. COCO's defaults also report small/medium/
large, and its headline AP filters nothing; the local metric has no area concept at all,
so anything but one open range would be comparing different object populations. **This
is worth revisiting on its own merits**: buoys are small objects and a size-stratified AP
is a real question (`docs/screen-small-object-2026-08-19.md`), but it is a different
question from parity and is not smuggled in here.

``max_dets`` -- ``None`` means "computed from the data as the maximum detections on any
one image", NOT COCO's default of 100. Caches are built at ``conf 0.001`` and routinely
carry hundreds of detections per frame, so the COCO default would silently truncate the
low-confidence tail and the resulting gap would be a detection cap masquerading as an
interpolation difference. Pass an explicit integer to study truncation deliberately.

``iscrowd`` / ``ignore`` -- neither exists in this dataset; every annotation is a hard
positive. Recorded because COCO's semantics differ from "no such thing", and F05 argues
unassessable annotations *should* eventually become ignore regions rather than negatives.

``score_sort`` -- stable, matching `apmetrics.SORT_KIND` (R-A2). pycocotools uses
``kind='mergesort'``, which is stable, so the two agree on tie handling.
"""


def to_coco_dicts(records: list[dict], gts: list[dict]) -> tuple[dict, list[dict], int]:
    """Convert the project's (records, gts) into a COCO GT dict and a detection list.

    Returns ``(gt_dict, detections, max_dets_seen)``. Image ids are positional, so a
    resampled frame list with repeats becomes distinct COCO images -- which is what a
    bootstrap draw means, and what keeps this usable for R-A5.
    """
    images, annotations, dets = [], [], []
    cats: set[int] = set()
    ann_id = 1
    max_seen = 0
    for i, (rec, gt) in enumerate(zip(records, gts)):
        h, w = rec.get("image_hw", (1, 1))
        images.append({"id": i, "width": int(w), "height": int(h)})
        gb = np.asarray(gt["boxes_xyxy"], dtype=float).reshape(-1, 4)
        gc = np.asarray(gt["cls"], dtype=int)
        for b, c in zip(gb, gc):
            bw, bh = float(b[2] - b[0]), float(b[3] - b[1])
            annotations.append({
                "id": ann_id, "image_id": i, "category_id": int(c),
                "bbox": [float(b[0]), float(b[1]), bw, bh],
                "area": bw * bh, "iscrowd": 0,
            })
            cats.add(int(c))
            ann_id += 1
        pb = np.asarray(rec["boxes_xyxy"], dtype=float).reshape(-1, 4)
        pc = np.asarray(rec["cls"], dtype=int)
        ps = np.asarray(rec["conf"], dtype=float)
        max_seen = max(max_seen, len(ps))
        for b, c, s in zip(pb, pc, ps):
            dets.append({
                "image_id": i, "category_id": int(c),
                "bbox": [float(b[0]), float(b[1]),
                         float(b[2] - b[0]), float(b[3] - b[1])],
                "score": float(s),
            })
            cats.add(int(c))
    gt_dict = {
        "images": images,
        "annotations": annotations,
        "categories": [{"id": c, "name": str(c)} for c in sorted(cats)],
    }
    return gt_dict, dets, max_seen


def coco_ap(records: list[dict], gts: list[dict], max_dets: int | None = None) -> dict:
    """Official COCO mAP@50-95 / mAP@50 / per-class, shaped like `matching.map50_95`.

    `max_dets` defaults to the maximum detections on any single image, so nothing is
    truncated -- see TASK_CONFIG. The returned ``meta`` carries the effective config so
    a number from here can never be quoted without the knobs it was produced under.
    """
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval

    gt_dict, dets, max_seen = to_coco_dicts(records, gts)
    n_dets_cap = int(max_dets) if max_dets is not None else max(1, max_seen)
    meta = {**TASK_CONFIG, "max_dets": n_dets_cap,
            "max_dets_source": "explicit" if max_dets is not None else "max-per-image",
            "max_dets_seen": int(max_seen)}

    if not gt_dict["annotations"]:
        return {"map50_95": 0.0, "map50": 0.0, "per_class": {}, "dropped_classes": [],
                "meta": meta}

    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        coco_gt = COCO()
        coco_gt.dataset = gt_dict
        coco_gt.createIndex()
        # loadRes on an empty detection list raises; an evaluation with GT but no
        # predictions is a legitimate case (AP 0), so it is handled rather than crashed.
        if not dets:
            per_class = {}
            for c in sorted({a["category_id"] for a in gt_dict["annotations"]}):
                per_class[int(c)] = {"ap50_95": 0.0, "ap50": 0.0, "excluded": False,
                                     "n_gt": sum(a["category_id"] == c
                                                 for a in gt_dict["annotations"]),
                                     "n_pred": 0}
            return {"map50_95": 0.0, "map50": 0.0, "per_class": per_class,
                    "dropped_classes": [], "meta": meta}
        coco_dt = coco_gt.loadRes(list(dets))
        ev = COCOeval(coco_gt, coco_dt, iouType="bbox")
        ev.params.iouThrs = np.asarray(TASK_CONFIG["iou_thrs"], dtype=float)
        ev.params.recThrs = np.linspace(0, 1, TASK_CONFIG["rec_thrs"])
        ev.params.areaRng = [list(r) for r in TASK_CONFIG["area_rng"]]
        ev.params.areaRngLbl = list(TASK_CONFIG["area_rng_lbl"])
        ev.params.maxDets = [n_dets_cap]
        ev.evaluate()
        ev.accumulate()

    # precision has shape (T iou, R recall, K class, A area, M maxdets); -1 marks a
    # class with no GT, which pycocotools excludes from its own mean. Same policy as
    # apmetrics.MISSING_CLASS_POLICY = "drop": undefined is not zero.
    prec = ev.eval["precision"][:, :, :, 0, 0]
    cat_ids = list(ev.params.catIds)
    n_gt = {int(c): 0 for c in cat_ids}
    n_pred = {int(c): 0 for c in cat_ids}
    for a in gt_dict["annotations"]:
        n_gt[int(a["category_id"])] = n_gt.get(int(a["category_id"]), 0) + 1
    for d in dets:
        n_pred[int(d["category_id"])] = n_pred.get(int(d["category_id"]), 0) + 1

    per_class, rows50_95, rows50, dropped = {}, [], [], []
    for k, c in enumerate(cat_ids):
        p = prec[:, :, k]
        valid = p[p > -1]
        if valid.size == 0 or n_gt.get(int(c), 0) == 0:
            dropped.append(int(c))
            per_class[int(c)] = {"ap50_95": float("nan"), "ap50": float("nan"),
                                 "n_gt": int(n_gt.get(int(c), 0)),
                                 "n_pred": int(n_pred.get(int(c), 0)), "excluded": True}
            continue
        ap = float(np.mean(np.where(p > -1, p, 0.0)))
        ap50 = float(np.mean(np.where(p[0] > -1, p[0], 0.0)))
        rows50_95.append(ap)
        rows50.append(ap50)
        per_class[int(c)] = {"ap50_95": ap, "ap50": ap50,
                             "n_gt": int(n_gt.get(int(c), 0)),
                             "n_pred": int(n_pred.get(int(c), 0)), "excluded": False}
    if not rows50_95:
        return {"map50_95": 0.0, "map50": 0.0, "per_class": per_class,
                "dropped_classes": dropped, "meta": meta}
    return {"map50_95": float(np.mean(rows50_95)), "map50": float(np.mean(rows50)),
            "per_class": per_class, "dropped_classes": dropped, "meta": meta}
