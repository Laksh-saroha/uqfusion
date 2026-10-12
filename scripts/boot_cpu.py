"""The pohang04 block bootstrap of ship AP on the CPU, equal (==) to `p04_heldout_score.boot_batch`.

`boot_batch` expands every detection by its frame's multiplicity and runs float64 cumulative sums,
divisions and a reverse running maximum over the expanded list on the GPU, then reads recall and the
precision envelope at the positions that bracket the 101 recall-grid points. This computes the same
values at the same positions without expanding:

* the copies of one detection are adjacent in the expanded list, so each drawn detection is a run;
  cum_tp is exact integer arithmetic (as in float64), precision at expanded position p is
  cum_tp(p) / (p + 1) and recall is cum_tp(p) / max(n_gt, 1), the same IEEE divisions;
* within a TP run precision is non-decreasing, within an FP run non-increasing, so the envelope
  (max over every later position) at p is max(run end or p itself, suffix max of the later runs):
  the same maximum over the same set of float64 values;
* `searchsorted(recall, g, right=True) - 1` is the position just before the (T*+1)-th TP copy, with
  T* the largest count whose recall is <= g (division by a positive constant is monotone).

The interpolation on the bracketing points is then numpy's, exactly as in `boot_batch`.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange


@njit(cache=False, parallel=True)
def _core(tpT, fidx, W, ngt, grid):
    R = W.shape[0]
    NL, m = tpT.shape
    G = grid.shape[0]
    K = 2 + 2 * G
    pos_o = np.zeros((R, NL, K), np.int64)
    rx_o = np.zeros((R, NL, K))
    ex_o = np.zeros((R, NL, K))
    lens = np.zeros(R, np.int64)
    for r in prange(R):
        # drawn detections, their multiplicities and start positions in the expanded list
        n = 0
        for i in range(m):
            if W[r, fidx[i]] > 0:
                n += 1
        idx = np.empty(n, np.int64)
        ci = np.empty(n, np.int64)
        start = np.empty(n, np.int64)
        q = 0
        L = 0
        for i in range(m):
            c = W[r, fidx[i]]
            if c > 0:
                idx[q] = i
                ci[q] = c
                start[q] = L
                L += c
                q += 1
        lens[r] = L
        if L == 0:
            continue
        last = L - 1
        Tb = np.empty(n, np.int64)
        Ta = np.empty(n, np.int64)
        runmax = np.empty(n)
        env_after = np.empty(n)
        for lv in range(NL):
            T = 0
            for q in range(n):
                Tb[q] = T
                if tpT[lv, idx[q]]:
                    T += ci[q]
                    runmax[q] = T / (start[q] + ci[q])
                else:
                    runmax[q] = T / (start[q] + 1)
                Ta[q] = T
            T_total = T
            mx = -np.inf
            for q in range(n - 1, -1, -1):
                env_after[q] = mx
                if runmax[q] > mx:
                    mx = runmax[q]
            # positions: 0, last, clamp(j_g), min(j_g + 1, last)
            for t in range(K):
                if t == 0:
                    p = 0
                elif t == 1:
                    p = last
                else:
                    g = grid[(t - 2) % G]
                    lo = 0
                    hi = T_total
                    while lo < hi:                      # largest T with T / ngt <= g
                        mid = (lo + hi + 1) // 2
                        if mid / ngt[r] <= g:
                            lo = mid
                        else:
                            hi = mid - 1
                    if lo >= T_total:
                        j = last
                    else:
                        need = lo + 1                   # first run whose cumulative TP reaches lo + 1
                        a = 0
                        b = n - 1
                        while a < b:
                            mid = (a + b) // 2
                            if Ta[mid] >= need:
                                b = mid
                            else:
                                a = mid + 1
                        j = start[a] + (need - Tb[a]) - 1 - 1
                    if t < 2 + G:
                        p = j if j > 0 else 0
                        if p > last:
                            p = last
                    else:
                        p = j + 1
                        if p > last:
                            p = last
                # the run holding p
                a = 0
                b = n - 1
                while a < b:
                    mid = (a + b + 1) // 2
                    if start[mid] <= p:
                        a = mid
                    else:
                        b = mid - 1
                if tpT[lv, idx[a]]:
                    ct = Tb[a] + (p - start[a] + 1)
                    e = runmax[a]
                else:
                    ct = Tb[a]
                    e = ct / (p + 1)
                if env_after[a] > e:
                    e = env_after[a]
                pos_o[r, lv, t] = p
                rx_o[r, lv, t] = ct / ngt[r]
                ex_o[r, lv, t] = e
    return pos_o, rx_o, ex_o, lens


def boot_cpu(pre: dict, W: np.ndarray, ship: int = 0) -> np.ndarray:
    """Ship AP for each row of W (integer frame multiplicities); equal to `boot_batch`."""
    from uqfusion.eval.apmetrics import NL, RECALL_GRID
    d = pre["classes"][ship]
    tpT = np.ascontiguousarray(np.array(d["tp"], dtype=np.bool_).T)               # (NL, m)
    fidx = np.ascontiguousarray(d["fidx"], dtype=np.int64)
    gt = np.asarray(pre["gt"][ship])
    W = np.ascontiguousarray(W, dtype=np.int64)
    n_gt = W @ gt
    ngt = np.maximum(n_gt, 1).astype(np.float64)
    out = np.empty(len(W))
    if tpT.shape[1] == 0:
        out[:] = [float("nan") if x == 0 else 0.0 for x in n_gt]
        return out
    pos, rx, ex, lens = _core(tpT, fidx, W, ngt, np.ascontiguousarray(RECALL_GRID, dtype=np.float64))
    for r in range(len(W)):
        if n_gt[r] == 0:
            out[r] = float("nan")
            continue
        if lens[r] == 0:
            out[r] = float(np.zeros(NL).mean())
            continue
        v = np.empty(NL)
        for k in range(NL):
            _u, first = np.unique(pos[r, k], return_index=True)
            xs, fs = rx[r, k][first], ex[r, k][first]
            v[k] = np.interp(RECALL_GRID, xs, fs, left=fs[0], right=0).mean()
        out[r] = float(v.mean())
    return out
