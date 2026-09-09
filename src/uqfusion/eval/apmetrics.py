"""Per-class AP and a frame-level bootstrap, both built on precomputed match parts.

`matching.map50_95` recomputes the greedy GT match every time it is called, which
is the expensive half. Every question asked here — per-class AP, a frame subset,
a thousand bootstrap resamples — changes only WHICH frames are pooled, never how
a frame matched. So the match is computed once per frame per system
(`frame_parts`) and every later aggregation is a concatenate-and-sort over cached
arrays.

Three layers, each checked against the one above it:

* `ap_from_parts`  — the readable reference. Reproduces `matching.map50_95`
  exactly on the same frame set (verified to 0.0 by `smoke_apmetrics.py`), and
  additionally returns the per-class breakdown.
* `presort`/`ap_weighted` — the same computation with the per-class conf sort
  hoisted out of the loop, so a bootstrap resample costs a `repeat` and a
  `cumsum` instead of a re-sort. Asserted equal to `ap_from_parts` at unit
  weights.
* `bootstrap_delta` — a PAIRED resample over frames.

Why per-class matters: mAP here is macro-averaged over ship and buoy, so a class
that scores ~0 still contributes half the headline. Every reported delta should
be read per class before it is believed.

Why the bootstrap matters: the adopted gate is defended by deltas of 0.0002 to
0.0026 on an 8-cell table. Without an interval it is not knowable which of those
are real.
"""

from __future__ import annotations

import numpy as np

from uqfusion.eval.matching import IOU_LEVELS, tp_matrix

RECALL_GRID = np.linspace(0, 1, 101)
NL = len(IOU_LEVELS)

# ---------------------------------------------------------------- declared policies
# R-A2 (docs/TODO-2026-09-09-architecture-review.md). Both of these were previously
# implicit, and both were wrong in ways large enough to matter on this project's
# margins. They are named here so a reader does not have to infer them from the code.

MISSING_CLASS_POLICY = "drop"
"""What a class with ZERO ground-truth instances contributes to the macro mean.

``"drop"`` -- it is excluded, which is the COCO convention (pycocotools reports -1
for such a category and leaves it out of the mean). An AP with no GT to recall is
undefined, not zero.

This was inconsistent between the two paths and the divergence was 0.5 mAP on a
two-class fixture: ``ap_from_parts`` derives its class set from the resample, so a
resample that drops every buoy frame is scored over {ship} alone and returns 1.0;
``presort`` froze ``classes`` from the FULL selection and ``ap_weighted`` iterated
that frozen set, scoring the vanished class 0 and returning 0.5. Same resample, same
data. Buoy carries ~75% of this project's macro variance (`project-metric-noise-floor`)
and is exactly the class a resample can drop, so the fast path was biased DOWN in
precisely the draws that widen the interval.

**Measured blast radius on real data: none.** On the 2,232-frame paired val list, buoy
GT is present in 153 frames, so the chance a bootstrap draw deletes every one of them
is ``((n-k)/n)^n`` = 1.5e-69. No published interval from that list moves. The bug is
real but it bites only on SMALL subsets -- per-cell corruption tables, per-run slices,
leave-one-run-out folds -- where a rare class can genuinely vanish from a draw. Fixed
so that it cannot bite there silently, not because it corrupted the headline numbers.

Classes excluded under this policy are still reported in ``per_class`` with
``ap50_95 = nan`` and ``excluded = True``. They just do not enter the mean.
"""

SORT_KIND = "stable"
"""Tie semantics for the descending-confidence sort.

``np.argsort`` defaults to an unstable introsort, so detections sharing a confidence
were ordered arbitrarily -- and AP depends on that order, because cumulative TP/FP is
computed down the sorted list.

How big is it? Two very different numbers, and the honest answer needs both. On a
synthetic 400-detection fixture with only four distinct confidence values, quicksort
and stable differ by **0.0067 mAP** -- above the measured 2-sigma noise floor of
0.0014-0.0031 (`runs/eval/metric_noise_floor.md`). On a **real** cache
(`sigma_vis_seed0_nightfull`, 31,110 detections) the same comparison differs by
**2.7e-7**, four orders of magnitude below that floor, because real confidences are
near-continuous: 99.9% of those 31,110 values are unique, so exact ties are rare.

So the mechanism is real and unbounded in principle, and it has not been biting.
Nothing published moves. This is fixed for reproducibility -- same cache, same answer,
and the fast and reference paths provably agree -- not because it corrupted a result.

Honest limit: a stable sort makes AP reproducible for a GIVEN input order -- same
cache, same answer, and the fast and reference paths agree. It does NOT make AP
invariant to the order detections were cached in, because "stable" means "preserve
input order within a tie". Cache order is deterministic (frame order, then model
output order), so this is sufficient for reproducibility; it is not a claim of
order-invariance, and a genuinely canonical result would need an explicit secondary
sort key.
"""

AP_CONVENTION = "local-linear-interp"
"""WHICH AP integration rule this module implements. Added 2026-09-10 for R-A1.

There is no single "mAP@50-95". At least three conventions are in play on this
project and they do not agree:

* ``"local-linear-interp"`` -- what the code here does. The precision envelope is
  linearly interpolated onto ``RECALL_GRID`` (``np.interp``), then averaged.
* COCO / pycocotools -- looks up the envelope at the FIRST ATTAINED recall
  (``searchsorted``) rather than interpolating between attained points.
  `uqfusion.eval.cocoparity` implements it for comparison.
* Ultralytics -- a third rule again, and `docs/phase1-experimental-record.md`
  records a ~0.034 mAP difference between ultralytics *versions* on identical
  weights, two orders above the local-vs-COCO gap.

**Why this is declared rather than renamed.** `matching.map50_95` had called itself
"COCO-style" since it was written, which is the same defect as `preset="crossmodal"`
naming three different systems (see `docs/exposure-ledger-2026-09-09.md` section 6): a
NAME that does not pin a computation. The fix that generalises is to record the
value. Renaming 322 call sites would have moved no number and fixed no ambiguity that
this constant does not fix; the honest alias `matching.local_ap50_95` exists for new
code, and the old name still works.

**Measured gap, not asserted.** On this project's own caches the worst
local-vs-COCO disagreement in a DELTA is 0.00028501, 5x below the 0.0014-0.0031
paired 2-sigma noise floor -- see `docs/eval/ap_convention_parity_2026-09-09.md`,
and `cocoparity.PARITY_BOUND`, which pins it so a code change cannot widen it
silently. Absolute APs differ by more (-0.0033 and -0.0050 on the hand-built cases),
so the cancellation is a property of paired comparison, not of the metric.

**The rule this implies:** never compare an absolute AP across conventions. See
`docs/ap-convention-rule-2026-09-10.md`.
"""


def declared_policies() -> dict[str, str]:
    """The three policy constants above, for stamping into a result's config block.

    R-E1: a config block must record every value that changes the number, because a
    preset or function NAME is a moving target. These three all change AP and all
    used to be implicit. A result that carries them can be re-read years later
    without inferring the convention from the code that happened to be checked out.
    """
    return {
        "ap_convention": AP_CONVENTION,
        "missing_class_policy": MISSING_CLASS_POLICY,
        "sort_kind": SORT_KIND,
    }


def frame_parts(records: list[dict], gts: list[dict]) -> list[dict]:
    """Per-frame (tp, conf, cls, gt class counts). Computed once, reused forever."""
    parts = []
    for rec, gt in zip(records, gts):
        g = np.asarray(gt["cls"], dtype=int)
        gcls, gcnt = (np.unique(g, return_counts=True) if len(g)
                      else (np.zeros(0, dtype=int), np.zeros(0, dtype=int)))
        parts.append({
            "tp": tp_matrix(rec, gt),
            "conf": np.asarray(rec["conf"], dtype=float),
            "cls": np.asarray(rec["cls"], dtype=int),
            "gt_cls": gcls,
            "gt_cnt": gcnt,
        })
    return parts


def _envelope(precision: np.ndarray) -> np.ndarray:
    """Running max from the tail — the vectorised form of the reference's
    backward `prec[i] = max(prec[i], prec[i+1])` loop, and identical to it."""
    return np.maximum.accumulate(precision[::-1], axis=0)[::-1]


def _ap_from_sorted(tp: np.ndarray, n_gt: int) -> np.ndarray:
    """AP at each IoU level for one class, from TP flags already in conf order."""
    if n_gt == 0 or len(tp) == 0:
        return np.zeros(NL)
    cum_tp = np.cumsum(tp, axis=0, dtype=np.float64)
    cum_fp = np.cumsum(~tp, axis=0, dtype=np.float64)
    recall = cum_tp / n_gt
    precision = cum_tp / np.clip(cum_tp + cum_fp, 1e-9, None)
    prec = _envelope(precision)
    out = np.empty(NL)
    for k in range(NL):
        out[k] = np.interp(RECALL_GRID, recall[:, k], prec[:, k],
                           left=prec[0, k], right=0).mean()
    return out


def ap_from_parts(parts: list[dict], sel: np.ndarray | None = None,
                  gt_sel: np.ndarray | None = None) -> dict:
    """mAP@50-95, mAP@50 and per-class AP over a frame subset. The reference path.

    `gt_sel` decouples the GT denominator from the detection subset: default
    (None) recomputes it from `sel`, exactly as before. Passing a fixed set of
    frame indices (e.g. the whole eval set) makes the denominator constant while
    `sel` still restricts which frames' predictions are pooled — the "fixed-GT"
    metric a risk-coverage curve needs (B-3): abstained frames stop contributing
    predictions but their GT boxes still count as unrecovered, instead of the
    denominator itself shrinking with coverage.
    """
    idx = np.arange(len(parts)) if sel is None else np.asarray(sel)
    gt_idx = idx if gt_sel is None else np.asarray(gt_sel)
    n_gt: dict[int, int] = {}
    tps, confs, clss = [], [], []
    for i in gt_idx:
        p = parts[i]
        for c, k in zip(p["gt_cls"], p["gt_cnt"]):
            n_gt[int(c)] = n_gt.get(int(c), 0) + int(k)
    for i in idx:
        p = parts[i]
        if len(p["conf"]):
            tps.append(p["tp"])
            confs.append(p["conf"])
            clss.append(p["cls"])
    if not n_gt:
        return {"map50_95": 0.0, "map50": 0.0, "per_class": {}, "dropped_classes": []}

    tp = np.concatenate(tps) if tps else np.zeros((0, NL), dtype=bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    cls = np.concatenate(clss) if clss else np.zeros(0, dtype=int)

    per_class, rows = {}, []
    for c in sorted(n_gt):
        m = cls == c
        order = np.argsort(-conf[m], kind=SORT_KIND)
        curve = _ap_from_sorted(tp[m][order], n_gt[c])
        rows.append(curve)
        per_class[int(c)] = {"ap50_95": float(curve.mean()), "ap50": float(curve[0]),
                             "n_gt": int(n_gt[c]), "n_pred": int(m.sum()), "excluded": False}
    ap = np.stack(rows)
    # `dropped_classes` is always empty here: this path derives its class set FROM the
    # resample, so a class with no GT never enters it. The key exists so both paths
    # return the same contract and a caller cannot tell them apart by shape.
    return {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean()),
            "per_class": per_class, "dropped_classes": []}


# --------------------------------------------------------------------------
# fast path: hoist the per-class conf sort out of the resample loop
# --------------------------------------------------------------------------

def presort(parts: list[dict], sel: np.ndarray | None = None) -> dict:
    """Per-class detections sorted by descending conf, with the frame each came
    from, plus per-frame GT counts. Everything a resample needs, sorted once."""
    idx = np.arange(len(parts)) if sel is None else np.asarray(sel)
    f = len(idx)
    classes = sorted({int(c) for i in idx for c in parts[i]["gt_cls"]})

    gt = {c: np.zeros(f, dtype=np.int64) for c in classes}
    for local, i in enumerate(idx):
        p = parts[i]
        for c, k in zip(p["gt_cls"], p["gt_cnt"]):
            if int(c) in gt:
                gt[int(c)][local] = int(k)

    out = {}
    for c in classes:
        tps, fidx, confs = [], [], []
        for local, i in enumerate(idx):
            p = parts[i]
            if not len(p["conf"]):
                continue
            m = p["cls"] == c
            if not m.any():
                continue
            tps.append(p["tp"][m])
            confs.append(p["conf"][m])
            fidx.append(np.full(int(m.sum()), local, dtype=np.int64))
        if tps:
            tp = np.concatenate(tps)
            conf = np.concatenate(confs)
            fi = np.concatenate(fidx)
            order = np.argsort(-conf, kind=SORT_KIND)
            out[c] = {"tp": np.ascontiguousarray(tp[order]), "fidx": fi[order]}
        else:
            out[c] = {"tp": np.zeros((0, NL), dtype=bool), "fidx": np.zeros(0, dtype=np.int64)}
    return {"classes": out, "gt": gt, "n_frames": f}


def ap_weighted(pre: dict, w: np.ndarray | None = None) -> dict:
    """AP under integer frame multiplicities.

    A frame drawn twice by the bootstrap contributes its detections twice, in
    place — which is what resampling the dataset means. `np.repeat` on the
    presorted array reproduces the resampled conf order exactly, so this is the
    literal resample, not a weighted approximation of it.
    """
    if w is None:
        w = np.ones(pre["n_frames"], dtype=np.int64)
    per_class, rows, dropped = {}, [], []
    for c in sorted(pre["classes"]):
        d = pre["classes"][c]
        n_gt = int(pre["gt"][c] @ w)
        if len(d["fidx"]):
            mult = w[d["fidx"]]
            tp = np.repeat(d["tp"], mult, axis=0)
        else:
            tp = d["tp"]
        if n_gt == 0:
            # MISSING_CLASS_POLICY: `presort` froze `classes` from the full selection,
            # so a class can survive into here with every one of its GT frames weighted
            # to zero. Its AP is undefined, not 0 -- scoring it 0 is what made this path
            # return 0.5 where `ap_from_parts` returned 1.0 on the same resample. Report
            # it, exclude it from the mean.
            dropped.append(c)
            per_class[c] = {"ap50_95": float("nan"), "ap50": float("nan"),
                            "n_gt": 0, "n_pred": int(len(tp)), "excluded": True}
            continue
        curve = _ap_from_sorted(tp, n_gt)
        rows.append(curve)
        per_class[c] = {"ap50_95": float(curve.mean()), "ap50": float(curve[0]),
                        "n_gt": n_gt, "n_pred": int(len(tp)), "excluded": False}
    if not rows:
        return {"map50_95": 0.0, "map50": 0.0, "per_class": per_class,
                "dropped_classes": dropped}
    ap = np.stack(rows)
    return {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean()),
            "per_class": per_class, "dropped_classes": dropped}


def _score(pre: dict, w, cls: int | None) -> float:
    """Score one resample. NaN when the requested quantity is undefined.

    A per-class request whose class has no GT in this draw used to return 0.0, the
    same MISSING_CLASS_POLICY error as `ap_weighted`: it drags the delta toward zero
    on exactly the draws where the class vanished. NaN instead, and `bootstrap_delta`
    excludes those draws and counts them.
    """
    r = ap_weighted(pre, w)
    if cls is None:
        return r["map50_95"]
    pc = r["per_class"].get(cls)
    if pc is None or pc.get("excluded"):
        return float("nan")
    return pc["ap50_95"]


def bootstrap_delta(parts_a: list[dict], parts_b: list[dict], sel: np.ndarray | None = None,
                    n_boot: int = 1000, seed: int = 0, cls: int | None = None) -> dict:
    """Paired frame-level bootstrap of (A − B).

    Paired is the point: the SAME resampled frame set scores both systems, so the
    frame-composition noise that dominates a 2,232-frame mAP cancels in the
    difference. An unpaired interval on each system separately would be several
    times wider and would not answer whether A beats B on these frames.

    Returns the observed delta, a percentile CI, and the share of resamples where
    the sign flips — the number that decides whether a +0.0003 headline survives.

    **Two warnings, both from R-A3 (`docs/TODO-2026-09-09-architecture-review.md`).**

    ``p_sign_flip`` is **not a p-value** and not the probability that a hypothesis is
    true. It is a descriptive property of the resampling distribution. The name invites
    the wrong reading; `blockboot` calls the same quantity ``sign_flip_fraction`` and
    keeps ``p_sign_flip`` only as a deprecated alias.

    This resamples **individual frames**, and the frames come from 10 Hz recordings
    where consecutive frames are nearly the same picture. The interval it returns is
    therefore **too narrow — measured at ~1.9x too narrow, as a lower bound**
    (`runs/eval/interval_block_sensitivity_v3.md`). Prefer
    `blockboot.block_bootstrap_delta`, which resamples contiguous blocks within runs and
    reports what its interval covers. This function is kept for continuity with already
    published numbers, which must be read with that factor in mind.
    """
    pa = presort(parts_a, sel)
    pb = presort(parts_b, sel)
    f = pa["n_frames"]
    rng = np.random.default_rng(seed)

    obs_a, obs_b = _score(pa, None, cls), _score(pb, None, cls)
    obs_d = obs_a - obs_b
    deltas = np.empty(n_boot)
    p = np.full(f, 1.0 / f)
    for t in range(n_boot):
        w = rng.multinomial(f, p)
        deltas[t] = _score(pa, w, cls) - _score(pb, w, cls)

    # A per-class request can be undefined on a draw that resampled away every frame
    # holding that class's GT (MISSING_CLASS_POLICY). Those draws are excluded and
    # counted rather than silently scored 0, which would pull the interval toward zero
    # on exactly the draws that ought to widen it. `n_undefined` is reported so a
    # caller can see when the interval is describing fewer draws than it asked for.
    good = np.isfinite(deltas)
    n_undef = int((~good).sum())
    kept = deltas[good]
    if kept.size < 2:
        return {
            "a": obs_a, "b": obs_b, "delta": obs_d,
            "ci_lo": float("nan"), "ci_hi": float("nan"), "se": float("nan"),
            "p_sign_flip": float("nan"), "spans_zero": None,
            "n_frames": int(f), "n_boot": int(n_boot),
            "n_undefined": n_undef, "n_effective": int(kept.size),
        }
    lo, hi = np.percentile(kept, [2.5, 97.5])
    flip = (float(np.mean(np.sign(kept) != np.sign(obs_d)))
            if np.isfinite(obs_d) and obs_d != 0 else 1.0)
    return {
        "a": obs_a, "b": obs_b, "delta": obs_d,
        "ci_lo": float(lo), "ci_hi": float(hi), "se": float(kept.std(ddof=1)),
        "p_sign_flip": flip, "spans_zero": bool(lo <= 0.0 <= hi),
        "n_frames": int(f), "n_boot": int(n_boot),
        "n_undefined": n_undef, "n_effective": int(kept.size),
    }
