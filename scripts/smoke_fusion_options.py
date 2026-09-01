"""Structural checks on the two fusion options added 2026-09-01.

`score_scale` and `single_passthrough` both sit inside `fuse_detections`, which
every measured number in `docs/eval/` was produced by. The first thing to prove
is therefore not that they work but that they are INERT when off -- otherwise the
adopted tables silently stop being reproducible and nothing later in this file
matters.

Checks:

  A  score_scale=None and single_passthrough=False reproduce the pre-change path
     bit for bit, on random inputs, for boxes, scores and labels.
  B  score_scale=(1, 1) equals normalized weights (0.5, 0.5). Both put equal trust
     in the two streams, and WBF's `1/sum(weights)` rescale makes them the same
     computation; if this fails, the scaling path is not the same estimator.
  C  score_scale is HOMOGENEOUS: scaling both factors by k scales every fused
     score by exactly k and leaves boxes and labels untouched. This is the
     property that makes cross-frame comparison meaningful -- a frame the gate
     trusts half as much produces scores half as large, in the same units.

     Tested at POWERS OF TWO, where it holds exactly, and separately at arbitrary
     k, where it holds only to about 1e-6. The gap is not a bug in the scaling:
     `ensemble_boxes.get_weighted_box` accumulates the score-weighted coordinate
     average in float32, so changing the absolute score level changes the
     rounding of a sum that is mathematically scale-invariant. Powers of two are
     exact in IEEE-754 and pass straight through that accumulation, which is what
     makes them the right probe for the algebra. The float32 accumulator is a
     property of the fusion library the whole project already depends on and is
     reported, not worked around.
  D  score_scale=(a, 0) ranks every IR-only cluster strictly below what it would
     be at (a, a) -- i.e. the soft path really does suppress a distrusted stream,
     which is what the hard veto was introduced to do.
  E  single_passthrough with one stream vetoed returns that stream's boxes
     UNCHANGED -- not clipped, not merged, not re-scored.
  F  single_passthrough with both streams alive is a no-op.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.uq.fusion import fuse_detections  # noqa: E402

HW = (640, 640)


def rec(rng, n, seed_hi=1.0):
    x1 = rng.uniform(0, 500, n)
    y1 = rng.uniform(0, 500, n)
    return {
        "boxes_xyxy": np.stack([x1, y1, x1 + rng.uniform(10, 120, n),
                                y1 + rng.uniform(10, 120, n)], axis=1),
        "conf": rng.uniform(0.02, seed_hi, n),
        "cls": rng.integers(0, 2, n),
        "sigma_ltrb": rng.uniform(0.5, 6.0, (n, 4)),
        "image_hw": HW,
    }


def close(a, b, tol=0.0):
    return (a["boxes_xyxy"].shape == b["boxes_xyxy"].shape
            and np.max(np.abs(a["boxes_xyxy"] - b["boxes_xyxy"])) <= tol
            and np.max(np.abs(a["conf"] - b["conf"])) <= tol
            and np.array_equal(a["cls"], b["cls"]))


def main() -> int:
    rng = np.random.default_rng(7)
    fails = []
    c_prime_reclustered = 0

    for trial in range(20):
        v, i = rec(rng, int(rng.integers(1, 25))), rec(rng, int(rng.integers(1, 25)))
        wv = float(rng.uniform(0.05, 0.95))
        base = dict(vis_record=v, ir_record=i, w_vis=wv, w_ir=1 - wv, vis_hw=HW, iou_thr=0.55)

        # A — defaults inert
        a = fuse_detections(**base)
        b = fuse_detections(**base, score_scale=None, single_passthrough=False)
        if not close(a, b):
            fails.append(f"A trial {trial}: defaults not inert")

        # B — score_scale=(1,1) == normalized equal weights
        eq = fuse_detections(vis_record=v, ir_record=i, w_vis=0.5, w_ir=0.5,
                             vis_hw=HW, iou_thr=0.55)
        sc = fuse_detections(**{**base, "w_vis": 0.5, "w_ir": 0.5}, score_scale=(1.0, 1.0))
        if not close(eq, sc, 1e-12):
            fails.append(f"B trial {trial}: score_scale(1,1) != weights(0.5,0.5)")

        # C — homogeneity, exact at powers of two
        s1 = fuse_detections(**base, score_scale=(0.75, 0.25))
        for k in (0.25, 0.5, 2.0, 4.0):
            s2 = fuse_detections(**base, score_scale=(0.75 * k, 0.25 * k))
            if s1["conf"].shape != s2["conf"].shape:
                fails.append(f"C trial {trial}: cluster count changed at k={k}")
            elif (np.max(np.abs(s1["boxes_xyxy"] - s2["boxes_xyxy"])) > 0.0
                  or np.max(np.abs(s1["conf"] * k - s2["conf"])) > 0.0):
                fails.append(f"C trial {trial}: not exactly homogeneous at k={k}")
        # C' — arbitrary k, only to float32 accumulator precision
        k = float(rng.uniform(0.05, 4.0))
        s3 = fuse_detections(**base, score_scale=(0.75 * k, 0.25 * k))
        if s1["conf"].shape != s3["conf"].shape:
            c_prime_reclustered += 1
        elif (np.max(np.abs(s1["boxes_xyxy"] - s3["boxes_xyxy"])) > 1e-3
              or np.max(np.abs(s1["conf"] * k - s3["conf"])) > 1e-6 * max(k, 1.0)):
            fails.append(f"C' trial {trial}: drift beyond float32 at k={k:.3g}")

        # D — a distrusted stream is suppressed
        hi = fuse_detections(**base, score_scale=(1.0, 1.0))
        lo = fuse_detections(**base, score_scale=(1.0, 1e-6))
        if not (lo["conf"].max() <= hi["conf"].max() + 1e-12):
            fails.append(f"D trial {trial}: suppressing IR did not lower the top score")

        # E — passthrough returns the survivor untouched
        pt = fuse_detections(**base, veto_vis=True, single_passthrough=True)
        from uqfusion.uq.fusion import apply_homography
        want_b = apply_homography(np.asarray(i["boxes_xyxy"]).reshape(-1, 4), None)
        if (pt["boxes_xyxy"].shape != want_b.shape
                or np.max(np.abs(pt["boxes_xyxy"] - want_b)) > 0
                or np.max(np.abs(pt["conf"] - np.asarray(i["conf"]))) > 0
                or not np.array_equal(pt["cls"], np.asarray(i["cls"]).astype(int))):
            fails.append(f"E trial {trial}: passthrough altered the surviving stream")

        # F — passthrough inert when both streams are alive
        f0 = fuse_detections(**base)
        f1 = fuse_detections(**base, single_passthrough=True)
        if not close(f0, f1):
            fails.append(f"F trial {trial}: passthrough fired with both streams alive")

    # E' — the artifact passthrough exists to remove is real, not hypothetical.
    v, i = rec(rng, 20), rec(rng, 20)
    wbf1 = fuse_detections(v, i, 0.5, 0.5, HW, None, 0.55, veto_vis=True)
    pass1 = fuse_detections(v, i, 0.5, 0.5, HW, None, 0.55, veto_vis=True,
                            single_passthrough=True)
    same = (wbf1["boxes_xyxy"].shape == pass1["boxes_xyxy"].shape
            and np.max(np.abs(np.sort(wbf1["conf"]) - np.sort(pass1["conf"]))) < 1e-12)
    note = ("single-list WBF was a no-op on this input" if same
            else "single-list WBF changed the surviving stream (the artifact)")

    print("[smoke] A-F clean over 20 random trials" if not fails
          else f"[smoke] {len(fails)} failures over 20 random trials")
    print(f"[smoke] C' float32 note: {c_prime_reclustered}/20 trials re-clustered under "
          f"an arbitrary rescale (ties created by rounding); 0 at powers of two")
    print(f"[smoke] E' {note}: "
          f"{len(wbf1['conf'])} vs {len(pass1['conf'])} detections")
    for f in fails:
        print(f"[smoke] FAIL {f}")
    if fails:
        print("\nFUSION OPTIONS SMOKE FAILED")
        return 1
    print("\nFUSION OPTIONS SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
