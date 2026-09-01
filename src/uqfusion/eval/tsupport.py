"""Temporal support: the redundancy this dataset actually has.

The cross-modal terms in this system all rest on VIS and IR being redundant
observers of one scene. Measured, they barely are: at the operating `iou_thr` of
0.85 only **0.05%** of VIS detections have a same-class IR detection at that
overlap, and a per-frame registration correction that raises it to 4.02% makes ship
AP *worse* (`docs/levers-and-the-26m-swap-2026-09-01.md` §1.2, §3). The one
cross-modal term that survives, `support`, works at IoU 0.30 and is worth +0.0090
on the day cells.

But Pohang Canal is a slow transit, and consecutive frames of the paired val really
are consecutive (`pohang00_L_006767`, `_006768`, ...). The same vessel appears in
frame after frame. That redundancy costs nothing to exploit, needs no homography,
and carries none of the 3-6 px registration residual that makes the cross-modal
version so weak.

So this is the exact analogue of the cross-modal `support` term with the modality
axis swapped for the time axis: a box's score is multiplied by ``1 + gamma`` when a
same-class box of the SAME stream appeared at IoU >= `iou` within the previous `k`
frames. Coordinates are never touched, for the same reason as there -- a neighbour
frame is evidence that a target is present, not a better measurement of its edges.

**Causal by default.** A deployed detector cannot see the future. `causal=False`
uses a symmetric window and exists only to measure the ceiling that costs.

**Gaps are respected.** The paired val is contiguous in blocks, not globally, so a
neighbour counts only when its recovered frame NUMBER is within `k` -- adjacency in
the manifest is not adjacency in time, and treating it as such would silently
compare frames minutes apart.

The honest control lives in the caller, not here: temporal support applied to VIS
alone raises `visible_only` too, so the bar must be recomputed on the boosted
streams. Otherwise this measures a better single-stream baseline and reports it as
a fusion result.
"""

from __future__ import annotations

import numpy as np

from uqfusion.eval.hysteresis import temporal_order


def _iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    bb = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    return inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-12)


def temporal_support(records: list[dict], k: int = 2, iou: float = 0.30,
                     gamma: float = 0.5, causal: bool = True,
                     min_conf: float = 0.0) -> list[dict]:
    """Records with confidences boosted where a neighbouring frame agrees.

    `min_conf` restricts which neighbour boxes may vouch. At the cache's 0.001
    confidence floor a frame carries 9-30 boxes, most of them noise, and noise that
    happens to overlap noise would manufacture support out of nothing. Raising this
    makes the vouching evidence stronger than the box it vouches for.
    """
    from pathlib import Path
    import re

    order = temporal_order(records)
    num = np.asarray([int(re.search(r"(\d+)$", Path(r["image_path"]).stem).group(1))
                      for r in records])
    boxes = [np.asarray(r["boxes_xyxy"], dtype=np.float64).reshape(-1, 4) for r in records]
    cls = [np.asarray(r["cls"]).astype(int).reshape(-1) for r in records]
    conf = [np.asarray(r["conf"], dtype=np.float64).reshape(-1) for r in records]

    boost = [np.ones(len(c)) for c in conf]
    for _run, idx in order.items():
        for pos, i in enumerate(idx):
            if not len(boxes[i]):
                continue
            lo = max(0, pos - k)
            hi = pos if causal else min(len(idx), pos + k + 1)
            nbrs = list(idx[lo:pos]) + ([] if causal else list(idx[pos + 1:hi]))
            sup = np.zeros(len(boxes[i]), dtype=bool)
            for j in nbrs:
                if abs(int(num[j]) - int(num[i])) > k:
                    continue                       # manifest-adjacent, not time-adjacent
                m = conf[j] >= min_conf
                if not m.any():
                    continue
                iou_m = _iou(boxes[i], boxes[j][m])
                iou_m = np.where(cls[i][:, None] == cls[j][m][None, :], iou_m, 0.0)
                sup |= (iou_m >= iou).any(axis=1)
            boost[i] = np.where(sup, 1.0 + float(gamma), 1.0)

    return [{**r, "conf": c * b} for r, c, b in zip(records, conf, boost)]
