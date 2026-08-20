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


def ap_from_parts(parts: list[dict], sel: np.ndarray | None = None) -> dict:
    """mAP@50-95, mAP@50 and per-class AP over a frame subset. The reference path."""
    idx = np.arange(len(parts)) if sel is None else np.asarray(sel)
    n_gt: dict[int, int] = {}
    tps, confs, clss = [], [], []
    for i in idx:
        p = parts[i]
        for c, k in zip(p["gt_cls"], p["gt_cnt"]):
            n_gt[int(c)] = n_gt.get(int(c), 0) + int(k)
        if len(p["conf"]):
            tps.append(p["tp"])
            confs.append(p["conf"])
            clss.append(p["cls"])
    if not n_gt:
        return {"map50_95": 0.0, "map50": 0.0, "per_class": {}}

    tp = np.concatenate(tps) if tps else np.zeros((0, NL), dtype=bool)
    conf = np.concatenate(confs) if confs else np.zeros(0)
    cls = np.concatenate(clss) if clss else np.zeros(0, dtype=int)

    per_class, rows = {}, []
    for c in sorted(n_gt):
        m = cls == c
        order = np.argsort(-conf[m])
        curve = _ap_from_sorted(tp[m][order], n_gt[c])
        rows.append(curve)
        per_class[int(c)] = {"ap50_95": float(curve.mean()), "ap50": float(curve[0]),
                             "n_gt": int(n_gt[c]), "n_pred": int(m.sum())}
    ap = np.stack(rows)
    return {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean()), "per_class": per_class}


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
            order = np.argsort(-conf)
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
    per_class, rows = {}, []
    for c in sorted(pre["classes"]):
        d = pre["classes"][c]
        n_gt = int(pre["gt"][c] @ w)
        if len(d["fidx"]):
            mult = w[d["fidx"]]
            tp = np.repeat(d["tp"], mult, axis=0)
        else:
            tp = d["tp"]
        curve = _ap_from_sorted(tp, n_gt)
        rows.append(curve)
        per_class[c] = {"ap50_95": float(curve.mean()), "ap50": float(curve[0]),
                        "n_gt": n_gt, "n_pred": int(len(tp))}
    if not rows:
        return {"map50_95": 0.0, "map50": 0.0, "per_class": {}}
    ap = np.stack(rows)
    return {"map50_95": float(ap.mean()), "map50": float(ap[:, 0].mean()), "per_class": per_class}


def _score(pre: dict, w, cls: int | None) -> float:
    r = ap_weighted(pre, w)
    if cls is None:
        return r["map50_95"]
    pc = r["per_class"].get(cls)
    return 0.0 if pc is None else pc["ap50_95"]


def bootstrap_delta(parts_a: list[dict], parts_b: list[dict], sel: np.ndarray | None = None,
                    n_boot: int = 1000, seed: int = 0, cls: int | None = None) -> dict:
    """Paired frame-level bootstrap of (A − B).

    Paired is the point: the SAME resampled frame set scores both systems, so the
    frame-composition noise that dominates a 2,232-frame mAP cancels in the
    difference. An unpaired interval on each system separately would be several
    times wider and would not answer whether A beats B on these frames.

    Returns the observed delta, a percentile CI, and the share of resamples where
    the sign flips — the number that decides whether a +0.0003 headline survives.
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

    lo, hi = np.percentile(deltas, [2.5, 97.5])
    flip = float(np.mean(np.sign(deltas) != np.sign(obs_d))) if obs_d != 0 else 1.0
    return {
        "a": obs_a, "b": obs_b, "delta": obs_d,
        "ci_lo": float(lo), "ci_hi": float(hi), "se": float(deltas.std(ddof=1)),
        "p_sign_flip": flip, "spans_zero": bool(lo <= 0.0 <= hi),
        "n_frames": int(f), "n_boot": int(n_boot),
    }
