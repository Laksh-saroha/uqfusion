"""Per-frame IR->VIS translation, estimated from detections at inference time.

`runs/eval/x_registration_drift.md` measured the geometry this corrects. Mapping IR
**GT** boxes through the adopted per-run homography and matching them to VIS GT
leaves a median absolute residual of 3-6 px, and that residual is not constant
within a run: binning each run into ten equal-count bins by capture order, dx swings
by 5-10 px across the bins. Subtracting one constant per run -- the adopted model --
leaves about 1.5 px on the table that a per-bin model removes.

Why that matters more than 1.5 px sounds. Fusion clusters at `iou_thr` 0.85, and at
that overlap only 0.05% of VIS detections have a same-class IR partner: the two
streams essentially never meet. A few pixels on a 30 px ship box is the difference
between IoU 0.3 and IoU 0.6. So registration is not a small correctness detail
here; it is the reason the cross-modal terms have almost nothing to act on.

The correction must be estimable WITHOUT GT, or it is not deployable and measuring
it proves nothing. So the offset is estimated from the two DETECTION sets:

  1. map IR boxes into the VIS canvas with the existing homography,
  2. pair each IR box with the nearest same-class VIS box whose centre lies within
     `radius` x the mean box size, one VIS box per IR box,
  3. take the COMPONENT-WISE MEDIAN of the pair offsets, which is what makes this
     survive the majority of pairs being wrong -- on a frame with a handful of
     genuine correspondences and a crowd of spurious ones the median still lands on
     the genuine shift, and a mean would not,
  4. require at least `min_pairs` pairs and reject any shift beyond `max_shift` px,
     because a frame with two coincidental pairs can produce an arbitrary offset and
     the failure mode of a bad shift is worse than no shift at all.

Frames failing any guard keep the unmodified homography, so the correction degrades
to today's behaviour rather than to something new.

The result is returned as a modified homography per frame -- `T @ H` with T a pure
translation -- so nothing downstream of `h_frames` has to know this happened.
"""

from __future__ import annotations

import numpy as np


def _centres(b: np.ndarray) -> np.ndarray:
    return np.stack([(b[:, 0] + b[:, 2]) * 0.5, (b[:, 1] + b[:, 3]) * 0.5], axis=1)


def frame_offset(vis_boxes: np.ndarray, vis_cls: np.ndarray,
                 ir_boxes_in_vis: np.ndarray, ir_cls: np.ndarray,
                 radius: float = 1.0, min_pairs: int = 3,
                 max_shift: float = 40.0) -> tuple[float, float, int]:
    """Median (dx, dy) taking IR onto VIS, and the pair count it rests on.

    `radius` is in units of the mean of the two boxes' geometric sizes, so the
    gate scales with the object rather than assuming a pixel budget: a 200 px
    vessel and a 15 px buoy tolerate very different absolute displacements.

    Returns (0.0, 0.0, n) when the frame does not clear the guards.
    """
    v = np.asarray(vis_boxes, dtype=np.float64).reshape(-1, 4)
    i = np.asarray(ir_boxes_in_vis, dtype=np.float64).reshape(-1, 4)
    if len(v) < 1 or len(i) < 1:
        return 0.0, 0.0, 0
    vc, ic = _centres(v), _centres(i)
    vcl = np.asarray(vis_cls).astype(int).reshape(-1)
    icl = np.asarray(ir_cls).astype(int).reshape(-1)
    vsz = np.sqrt(np.clip(v[:, 2] - v[:, 0], 1e-6, None) * np.clip(v[:, 3] - v[:, 1], 1e-6, None))
    isz = np.sqrt(np.clip(i[:, 2] - i[:, 0], 1e-6, None) * np.clip(i[:, 3] - i[:, 1], 1e-6, None))

    d = np.linalg.norm(ic[:, None, :] - vc[None, :, :], axis=2)          # (n_ir, n_vis)
    d = np.where(icl[:, None] == vcl[None, :], d, np.inf)
    tol = radius * 0.5 * (isz[:, None] + vsz[None, :])
    d = np.where(d <= tol, d, np.inf)

    # One VIS box per IR box, greedily by increasing distance, so a single bright
    # VIS detection cannot claim every IR box and manufacture a consensus.
    offs, used = [], set()
    for ii in np.argsort(d.min(axis=1)):
        if not np.isfinite(d[ii]).any():
            continue
        order = np.argsort(d[ii])
        for jj in order:
            if not np.isfinite(d[ii, jj]):
                break
            if jj in used:
                continue
            used.add(int(jj))
            offs.append(vc[jj] - ic[ii])
            break
    if len(offs) < min_pairs:
        return 0.0, 0.0, len(offs)
    o = np.median(np.stack(offs), axis=0)
    if not np.isfinite(o).all() or float(np.hypot(*o)) > max_shift:
        return 0.0, 0.0, len(offs)
    return float(o[0]), float(o[1]), len(offs)


def aligned_homographies(vis_records, ir_records, h_frames, **kw):
    """`h_frames` with a per-frame translation composed on, plus a diagnostic.

    The translation is applied on the LEFT (``T @ H``) because it is a correction
    in the VIS canvas, which is where the residual was measured and where fusion
    clusters. Applying it on the right would translate in the IR canvas and pick up
    the homography's scale and shear on the way out.
    """
    from uqfusion.eval.identity import assert_paired
    from uqfusion.uq.fusion import apply_homography

    # R-E1 slice 2: this measures a VIS<->IR registration residual, so a mis-paired
    # cache does not produce a wrong number here -- it produces a residual measured
    # between two different instants, which is not a registration quantity at all.
    assert_paired(vis_records, ir_records, where="aligned_homographies")

    out, dxs, dys, npairs = [], [], [], []
    for rv, ri, h in zip(vis_records, ir_records, h_frames):
        ib = apply_homography(np.asarray(ri["boxes_xyxy"], dtype=np.float64).reshape(-1, 4), h)
        dx, dy, n = frame_offset(rv["boxes_xyxy"], rv["cls"], ib, ri["cls"], **kw)
        dxs.append(dx)
        dys.append(dy)
        npairs.append(n)
        if dx == 0.0 and dy == 0.0:
            out.append(h)
            continue
        t = np.eye(3, dtype=np.float64)
        t[0, 2], t[1, 2] = dx, dy
        out.append(t @ (np.eye(3) if h is None else np.asarray(h, dtype=np.float64)))
    return out, {"dx": np.asarray(dxs), "dy": np.asarray(dys),
                 "pairs": np.asarray(npairs),
                 "applied": float(np.mean((np.asarray(dxs) != 0) | (np.asarray(dys) != 0)))}
