"""The two new fusion levers must be inert at their defaults, and do exactly what
they claim when switched on.

Both changes touch `fuse_detections`, which every published number in this project
runs through. A lever that is not exactly inert at its default silently rewrites
tables nobody re-ran -- and it would do so in a way that still LOOKS right, since
the numbers would move only slightly. So the first three checks are about the OFF
state, not the on one.

  A  consensus_beta=1.0, consensus_distinct=False  ==  stock `ensemble_boxes` WBF
  B  veto_keep_cls=()                              ==  today's veto, bit for bit
  C  neither lever changes an un-vetoed frame at all
  D  consensus_beta=0 removes the agreement bonus, and only that
  E  consensus_distinct counts STREAMS, not cluster members
  F  veto_keep_cls keeps exactly the named classes of a vetoed stream
  G  veto_keep_cls with nothing to keep degrades to the plain veto
  K  the carry-through survives single_passthrough without disabling it
  H  sigma survives the class filter with its own box (the irdedup bug, again)
  I  support_gamma=0 is exactly stock WBF
  J  support boosts a loosely-confirmed box and does NOT move its coordinates
  L  sigma_score_alpha=0 is exactly inert, and >0 actually reaches the scores

Usage:  python scripts/smoke_new_levers.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.uq.fusion import fuse_detections  # noqa: E402

TOL = 1e-6
HW = (480, 640)
EMPTY = {"boxes_xyxy": np.zeros((0, 4)), "conf": np.zeros(0), "cls": np.zeros(0, int),
         "sigma_ltrb": np.zeros((0, 4)), "image_hw": HW}


def rec(n, cls_choices=(0, 1), seed=0):
    r = np.random.default_rng(seed)
    x1 = r.uniform(0, 500, n)
    y1 = r.uniform(0, 350, n)
    return {"boxes_xyxy": np.stack([x1, y1, x1 + r.uniform(20, 100, n),
                                    y1 + r.uniform(20, 100, n)], 1),
            "conf": r.uniform(0.05, 0.95, n),
            "cls": r.choice(cls_choices, n),
            "sigma_ltrb": r.uniform(0.5, 8.0, (n, 4)),
            "image_hw": HW}


def one(box, conf, cls=0):
    return {"boxes_xyxy": np.asarray(box, dtype=float).reshape(-1, 4),
            "conf": np.asarray(conf, dtype=float).reshape(-1),
            "cls": np.full(len(np.asarray(conf).reshape(-1)), cls, dtype=int),
            "sigma_ltrb": np.ones((len(np.asarray(conf).reshape(-1)), 4)),
            "image_hw": HW}


def same(a, b, what, tol=TOL):
    assert a["boxes_xyxy"].shape == b["boxes_xyxy"].shape, \
        f"{what}: shape {a['boxes_xyxy'].shape} vs {b['boxes_xyxy'].shape}"
    d = max(float(np.max(np.abs(a["boxes_xyxy"] - b["boxes_xyxy"]))) if a["boxes_xyxy"].size else 0.0,
            float(np.max(np.abs(a["conf"] - b["conf"]))) if a["conf"].size else 0.0,
            float(np.max(np.abs(np.asarray(a["cls"]) - np.asarray(b["cls"])))) if len(a["cls"]) else 0.0)
    assert d <= tol, f"{what}: max abs diff {d:.3e} > {tol:.1e}"
    return d


V, I = rec(14, seed=1), rec(11, seed=2)


def a():
    # The local implementation accumulates in float64 and `ensemble_boxes` in
    # float32, so "identical" is 1e-6 here -- the same tolerance and the same
    # reason as scripts/smoke_sigma_wbf.py.
    base = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55)
    got = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55,
                          consensus_beta=1.0, consensus_distinct=False)
    return f"max diff {same(base, got, 'A'):.2e} (float32 vs float64 accumulator)"


def b():
    for vv, ii in ((True, False), (False, True), (False, False)):
        base = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55, veto_vis=vv, veto_ir=ii)
        got = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55, veto_vis=vv, veto_ir=ii,
                              veto_keep_cls=())
        same(base, got, f"B veto=({vv},{ii})", 0.0)
    return "exact, all three veto states"


def c():
    base = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55)
    got = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55, veto_keep_cls=(1,))
    same(base, got, "C", 0.0)
    return "an un-vetoed frame never reaches the class filter"


def d():
    # One box per stream, deliberately overlapping, so there is exactly one
    # two-member cluster and its score is the whole test.
    a_ = one([[100.0, 100.0, 200.0, 200.0]], [0.8])
    b_ = one([[101.0, 101.0, 201.0, 201.0]], [0.6])
    s1 = fuse_detections(a_, b_, 0.5, 0.5, HW, None, 0.55, consensus_beta=1.0)
    s0 = fuse_detections(a_, b_, 0.5, 0.5, HW, None, 0.55, consensus_beta=0.0)
    assert len(s1["conf"]) == len(s0["conf"]) == 1, "expected exactly one cluster"
    r = float(s1["conf"][0] / s0["conf"][0])
    assert abs(r - 2.0) < 1e-5, f"beta=1 must be 2x beta=0 on a 2-member cluster, got {r:.4f}"
    # ...and must not touch a cluster only one stream produced.
    lone = one([[10.0, 10.0, 40.0, 40.0]], [0.5])
    e1 = fuse_detections(lone, EMPTY, 0.5, 0.5, HW, None, 0.55, consensus_beta=1.0)
    e0 = fuse_detections(lone, EMPTY, 0.5, 0.5, HW, None, 0.55, consensus_beta=0.0)
    same(e1, e0, "D lone cluster")
    return "2x on an agreed box, unchanged on an unconfirmed one"


def e():
    # Two OVERLAPPING boxes in ONE stream, none in the other. Stock WBF pays them
    # the agreement bonus; consensus_distinct must not.
    a_ = one([[100.0, 100.0, 200.0, 200.0], [101.0, 101.0, 201.0, 201.0]], [0.8, 0.6])
    m = fuse_detections(a_, EMPTY, 0.5, 0.5, HW, None, 0.55, consensus_distinct=False)
    s = fuse_detections(a_, EMPTY, 0.5, 0.5, HW, None, 0.55, consensus_distinct=True)
    assert len(m["conf"]) == len(s["conf"]) == 1, "expected exactly one cluster"
    r = float(m["conf"][0] / s["conf"][0])
    assert abs(r - 2.0) < 1e-5, \
        f"self-agreement must lose its 2x bonus under consensus_distinct, got {r:.4f}"
    return "two boxes from one sensor stop counting as confirmation"


def f():
    vv, ii = rec(20, cls_choices=(0, 1), seed=5), rec(9, cls_choices=(0,), seed=6)
    n_buoy = int((np.asarray(vv["cls"]) == 1).sum())
    assert n_buoy > 0, "fixture must contain buoys"
    plain = fuse_detections(vv, ii, 0.7, 0.3, HW, None, 0.55, veto_vis=True)
    assert (np.asarray(plain["cls"]) == 1).sum() == 0, \
        "fixture check: the plain veto should have deleted every buoy"
    out = fuse_detections(vv, ii, 0.7, 0.3, HW, None, 0.55, veto_vis=True, veto_keep_cls=(1,))
    assert (np.asarray(out["cls"]) == 1).sum() > 0, \
        "a vetoed VIS must still supply the class IR cannot"
    assert (np.asarray(out["cls"]) == 0).sum() == (np.asarray(plain["cls"]) == 0).sum(), \
        "the ship boxes must be exactly the ones the plain veto left"
    return f"{n_buoy} VIS buoys survive; plain veto keeps 0; ship count unchanged"


def g():
    vv, ii = rec(12, cls_choices=(0,), seed=7), rec(9, cls_choices=(0,), seed=8)
    a_ = fuse_detections(vv, ii, 0.7, 0.3, HW, None, 0.55, veto_vis=True)
    b_ = fuse_detections(vv, ii, 0.7, 0.3, HW, None, 0.55, veto_vis=True, veto_keep_cls=(1,))
    same(a_, b_, "G", 0.0)
    return "nothing to keep -> the plain veto, exactly"


def h():
    vv, ii = rec(20, cls_choices=(0, 1), seed=9), rec(9, cls_choices=(0,), seed=10)
    # sigma_weighted forces the local path, which is where a sigma array left at
    # full length while the boxes were filtered raises instead of mis-weighting.
    out = fuse_detections(vv, ii, 0.7, 0.3, HW, None, 0.55, veto_vis=True,
                          veto_keep_cls=(1,), sigma_weighted=True)
    assert len(out["conf"]) > 0, "expected surviving buoys"
    return "sigma_ltrb stays aligned with its boxes under the class filter"


def i_():
    base = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55)
    got = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55, support_iou=0.3, support_gamma=0.0)
    return f"max diff {same(base, got, 'I'):.2e}"


def j_():
    # A VIS box and an IR box overlapping at IoU 0.324 -- well below iou_thr 0.85,
    # so they stay SEPARATE clusters. That is the regime the real data is in: at
    # 0.85 only 0.05% of VIS boxes have an IR partner, at 0.30 a third of them do.
    v_ = one([[100.0, 100.0, 200.0, 200.0]], [0.40])
    ir_ = one([[130.0, 130.0, 230.0, 230.0]], [0.10])
    off = fuse_detections(v_, ir_, 0.9, 0.1, HW, None, 0.85, support_iou=0.1, support_gamma=0.0)
    on = fuse_detections(v_, ir_, 0.9, 0.1, HW, None, 0.85, support_iou=0.1, support_gamma=1.0)
    assert len(off["conf"]) == len(on["conf"]) == 2,         f"expected two separate clusters, got {len(off['conf'])}/{len(on['conf'])}"
    # Coordinates untouched -- that is the entire point of a support term.
    assert np.max(np.abs(np.sort(off["boxes_xyxy"], 0) - np.sort(on["boxes_xyxy"], 0))) < TOL,         "support must not move a single coordinate"
    r = float(np.max(on["conf"]) / np.max(off["conf"]))
    assert abs(r - 2.0) < 1e-5, f"gamma=1 must double a confirmed box, got {r:.4f}"
    # An unconfirmed box must be untouched.
    lone = fuse_detections(one([[10.0, 10.0, 40.0, 40.0]], [0.5]), EMPTY, 0.9, 0.1, HW,
                           None, 0.85, support_iou=0.1, support_gamma=1.0)
    lone0 = fuse_detections(one([[10.0, 10.0, 40.0, 40.0]], [0.5]), EMPTY, 0.9, 0.1, HW,
                            None, 0.85, support_iou=0.1, support_gamma=0.0)
    same(lone, lone0, "J lone")
    return "2x on a confirmed box, coordinates unmoved, lone box unchanged"


def k_():
    # The bug this replaced: keeping the vetoed stream ALIVE so WBF could see its
    # buoys made the frame two-stream again, which disabled single_passthrough and
    # rescaled the survivor's SHIP scores by its weight -- on vetoed frames only, so
    # part of the pooled AP ordering moved and the rest did not.
    vv, ii = rec(20, cls_choices=(0, 1), seed=11), rec(9, cls_choices=(0,), seed=12)
    plain = fuse_detections(vv, ii, 0.9, 0.1, HW, None, 0.85, veto_vis=True,
                            single_passthrough=True)
    out = fuse_detections(vv, ii, 0.9, 0.1, HW, None, 0.85, veto_vis=True,
                          single_passthrough=True, veto_keep_cls=(1,))
    ms, mp = np.asarray(out["cls"]) == 0, np.asarray(plain["cls"]) == 0
    same({"boxes_xyxy": out["boxes_xyxy"][ms], "conf": out["conf"][ms],
          "cls": np.asarray(out["cls"])[ms]},
         {"boxes_xyxy": plain["boxes_xyxy"][mp], "conf": plain["conf"][mp],
          "cls": np.asarray(plain["cls"])[mp]}, "K passthrough ship half", 0.0)
    assert (np.asarray(out["cls"]) == 1).sum() > 0, "buoys must still be carried"
    return "passthrough still fires; the IR ship scores are untouched"


def l_():
    # The bug this guards: the non-`sigma_weighted` path fed the local WBF a DUMMY
    # sigma column of ones, so `sigma_score_alpha` computed a reliability of 1.0 for
    # every box and the term was silently inert -- every day cell byte-identical
    # across alpha, with only the passthrough branch moving. A lever that is inert
    # when it should be live looks exactly like a lever that does not help.
    base = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55)
    off = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55, sigma_score_alpha=0.0)
    same(base, off, "L off", TOL)
    live = fuse_detections(V, I, 0.7, 0.3, HW, None, 0.55, sigma_score_alpha=1.0)
    d = float(np.max(np.abs(np.sort(base["conf"]) - np.sort(live["conf"]))))
    assert d > 1e-3, f"sigma_score_alpha=1 must move the scores, moved {d:.2e}"
    # Coordinates must NOT move: this arm scores on sigma, it does not weight by it.
    assert np.max(np.abs(np.sort(base["boxes_xyxy"], 0)
                         - np.sort(live["boxes_xyxy"], 0))) < TOL,         "sigma_score must not move a coordinate -- that is sigma_weighted's job"
    return f"off is exact; alpha=1 moves scores by {d:.3f} and no coordinate"


def main() -> int:
    fails = []
    print("smoke: new fusion levers")
    for name, fn in (("A stock WBF unchanged at defaults", a),
                     ("B veto_keep_cls=() is the old veto", b),
                     ("C levers do not touch an un-vetoed frame", c),
                     ("D consensus_beta scales only agreement", d),
                     ("E consensus_distinct counts streams", e),
                     ("F a vetoed stream keeps its exempt classes", f),
                     ("G exempt set empty -> plain veto", g),
                     ("H sigma survives the class filter", h),
                     ("I support_gamma=0 is stock WBF", i_),
                     ("J support boosts without moving coordinates", j_),
                     ("K carry-through keeps single_passthrough intact", k_),
                     ("L sigma_score is inert at 0 and live above it", l_)):
        try:
            note = fn()
            print(f"  [ok] {name}" + (f" -- {note}" if note else ""))
        except AssertionError as exc:
            fails.append(f"{name}: {exc}")
            print(f"  [FAIL] {name}: {exc}")
    if fails:
        print(f"\n{len(fails)} FAILED")
        return 1
    print("\nall green")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
