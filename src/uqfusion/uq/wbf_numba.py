"""Compiled `sigma_weighted_fusion` for its default path, bit-identical to the Python reference.

Covers the path every gate-free system and every checkpoint ensemble runs: `use_sigma=False`,
`sigma_score_alpha=0`, `consensus_beta=1.0`, `consensus_distinct=False`, with or without the
cross-stream support bonus. The Python implementation spends ~3 ms per frame in interpreter loops; this
one does the same arithmetic in the same order:

* prefilter, label grouping (first-appearance order) and the stable descending sort are unchanged;
* greedy assignment against the running fused box, `argmax` with numpy's NaN rule (the first NaN wins),
  join iff IoU > `iou_thr`;
* a cluster's box is recomputed from all members after each join: the weight total is numpy's pairwise
  sum of the (n, 1) column, the weighted coordinates are summed row by row (numpy reduces axis 0 of an
  (n, 4) array sequentially), total <= 0 -> 1e-9;
* score = numpy's pairwise mean of the members' weighted scores x (1 + (k - 1)) / sum(weights), x (1 + gamma)
  when a box from another input list overlaps the fused box at IoU >= `support_iou`.

`scripts/check_wbf_numba.py` compares it with the reference on real records; the switch in
`fusion.sigma_weighted_fusion` (`UQF_FAST_WBF=1`) is off unless asked for. Not disk-cached: a stale
numba cache segfaulted on 2026-10-11, and compiling takes ~3 s per process.
"""

from __future__ import annotations

import numpy as np
from numba import njit

_EPS = 1e-9


@njit(cache=False, error_model="numpy")
def _pairwise(v, lo, n):
    """numpy's pairwise_sum (8 accumulators, block 128) over v[lo:lo+n]."""
    if n < 8:
        res = 0.0
        for i in range(n):
            res += v[lo + i]
        return res
    if n <= 128:
        r0 = v[lo]
        r1 = v[lo + 1]
        r2 = v[lo + 2]
        r3 = v[lo + 3]
        r4 = v[lo + 4]
        r5 = v[lo + 5]
        r6 = v[lo + 6]
        r7 = v[lo + 7]
        i = 8
        stop = n - (n % 8)
        while i < stop:
            r0 += v[lo + i]
            r1 += v[lo + i + 1]
            r2 += v[lo + i + 2]
            r3 += v[lo + i + 3]
            r4 += v[lo + i + 4]
            r5 += v[lo + i + 5]
            r6 += v[lo + i + 6]
            r7 += v[lo + i + 7]
            i += 8
        res = ((r0 + r1) + (r2 + r3)) + ((r4 + r5) + (r6 + r7))
        while i < n:
            res += v[lo + i]
            i += 1
        return res
    n2 = n // 2
    n2 -= n2 % 8
    return _pairwise(v, lo, n2) + _pairwise(v, lo + n2, n - n2)


@njit(cache=False, error_model="numpy")
def _mx(a, b):
    # np.maximum(a, b): a if a >= b (or a is NaN) else b
    return a if (a >= b or a != a) else b


@njit(cache=False, error_model="numpy")
def _mn(a, b):
    return a if (a <= b or a != a) else b


@njit(cache=False, error_model="numpy")
def _iou(f, g):
    """`fusion._bb_iou` for one row f (boxes) against g (new_box), same operation order."""
    xa = _mx(f[0], g[0])
    ya = _mx(f[1], g[1])
    xb = _mn(f[2], g[2])
    yb = _mn(f[3], g[3])
    inter = _mx(xb - xa, 0.0) * _mx(yb - ya, 0.0)
    area_a = (f[2] - f[0]) * (f[3] - f[1])
    area_b = (g[2] - g[0]) * (g[3] - g[1])
    return inter / (area_a + area_b - inter)


@njit(cache=False, error_model="numpy")
def _group(score, box, src, n_lists, wsum, iou_thr, support_iou, gamma):
    """One label group, entries in prefilter order. Returns (fused boxes, scores) in cluster order."""
    m = score.shape[0]
    order = np.argsort(-score, kind="mergesort")            # stable, as kind="stable"
    s = score[order]
    b = box[order]
    sr = src[order]
    fused = np.empty((m, 4))
    head = np.empty(m, np.int64)                            # members as linked lists, in join order
    tail = np.empty(m, np.int64)
    nxt = np.full(m, -1, np.int64)
    nmem = np.zeros(m, np.int64)
    mask = np.zeros(m, np.int64)                            # input lists present in the cluster, one bit each
    nc = 0
    a = np.empty(m)
    for e in range(m):
        idx = -1
        if nc > 0:
            best = 0
            bv = _iou(fused[0], b[e])
            if bv == bv:
                for c in range(1, nc):
                    v = _iou(fused[c], b[e])
                    if v != v:
                        best = c
                        bv = v
                        break
                    if v > bv:
                        best = c
                        bv = v
            if bv > iou_thr:
                idx = best
        if idx == -1:
            for j in range(4):
                fused[nc, j] = b[e, j]
            head[nc] = e
            tail[nc] = e
            nmem[nc] = 1
            mask[nc] = np.int64(1) << sr[e]
            nc += 1
        else:
            nxt[tail[idx]] = e
            tail[idx] = e
            nmem[idx] += 1
            mask[idx] |= np.int64(1) << sr[e]
            n = 0
            q = head[idx]
            while q != -1:
                a[n] = s[q]
                n += 1
                q = nxt[q]
            total = _pairwise(a, 0, n)
            if total <= 0:
                total = _EPS
            for j in range(4):
                acc = 0.0
                q = head[idx]
                i = 0
                while q != -1:
                    acc += b[q, j] * a[i]
                    i += 1
                    q = nxt[q]
                fused[idx, j] = acc / total
    out_s = np.empty(nc)
    for c in range(nc):
        n = 0
        q = head[c]
        while q != -1:
            a[n] = s[q]
            n += 1
            q = nxt[q]
        conf = _pairwise(a, 0, n) / n
        k = n_lists if n_lists < n else n
        conf = conf * ((1.0 + 1.0 * (k - 1)) / wsum)
        if gamma != 0.0:
            sup = False
            for e in range(m):
                if (mask[c] >> sr[e]) & 1:
                    continue
                if _iou(b[e], fused[c]) >= support_iou:
                    sup = True
                    break
            if sup:
                conf = conf * (1.0 + gamma)
        out_s[c] = conf
    return fused[:nc].copy(), out_s


def wbf_fast(boxes_list, scores_list, labels_list, weights, iou_thr=0.55, skip_box_thr=0.0, support_iou=0.0,
             support_gamma=0.0):
    w = np.asarray(weights, dtype=np.float64)
    if w.sum() <= 0:
        raise ValueError("fusion weights must sum to > 0")
    if len(w) > 62:
        raise ValueError("at most 62 input lists")
    wsum = float(w.sum())
    S, C, T, L = [], [], [], []
    for t_idx, (bx, sc, lb) in enumerate(zip(boxes_list, scores_list, labels_list)):
        bx = np.asarray(bx, dtype=np.float64).reshape(-1, 4)
        sc = np.asarray(sc, dtype=np.float64).reshape(-1)
        lb = np.asarray(lb, dtype=np.float64).reshape(-1)
        if not len(bx):
            continue
        keep = ~(sc < skip_box_thr)
        bx, sc, lb = bx[keep], sc[keep], lb[keep]
        c = np.clip(bx, 0.0, 1.0)
        sx = c[:, 2] < c[:, 0]
        sy = c[:, 3] < c[:, 1]
        C.append(np.stack([np.where(sx, c[:, 2], c[:, 0]), np.where(sy, c[:, 3], c[:, 1]),
                           np.where(sx, c[:, 0], c[:, 2]), np.where(sy, c[:, 1], c[:, 3])], axis=1))
        S.append(sc * w[t_idx] * 1.0)
        T.append(np.full(len(sc), t_idx, np.int64))
        L.append(lb.astype(np.int64))                      # int(label), as the reference
    ob, osc, ol = [], [], []
    if S:
        S, C, T, L = np.concatenate(S), np.concatenate(C), np.concatenate(T), np.concatenate(L)
        if len(S):
            uniq, first = np.unique(L, return_index=True)
            for lab in uniq[np.argsort(first)]:             # first-appearance order, as dict insertion
                m = L == lab
                fb, fs = _group(np.ascontiguousarray(S[m]), np.ascontiguousarray(C[m]),
                                np.ascontiguousarray(T[m]), len(w), wsum, float(iou_thr), float(support_iou),
                                float(support_gamma))
                ob.append(fb)
                osc.append(fs)
                ol.append(np.full(len(fs), float(lab), np.float64))
    if not ob:
        return np.zeros((0, 4)), np.zeros((0,)), np.zeros((0,))
    fb, fs, fl = np.vstack(ob), np.concatenate(osc), np.concatenate(ol)
    order = np.argsort(-fs, kind="stable")
    return fb[order], fs[order], fl[order]
