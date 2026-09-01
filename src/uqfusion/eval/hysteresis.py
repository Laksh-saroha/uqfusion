"""Temporal hysteresis on the hard-veto switch (finalized 2026-08-20).

The per-frame veto under-fires when a corruption pushes the photometric
statistic across `mu_b` on only part of a dark run: fog lifts `p05` above the
threshold on 71% of night frames, so the switch flickered and fog/night sat at
0.0789 against `ir_only`'s 0.0810 (record §4.5). Darkness is a property of a
contiguous stretch of a recording, not of one frame, and the measured fix
(`runs/eval/x_veto_hysteresis.md`) is morphological dilation of the veto flags
in capture order: veto if ANY frame in a window of k says veto. At the adopted
`dilate 15` the fog/night veto rate goes 29% -> 89%, the cell closes to 0.0809
(+0.0020, CI [+0.0007, +0.0028]), and the clean/day guard cell does not move by
a single digit at any window tried (k up to 61).

Only the SWITCH is filtered. Smoothing the brightness signal instead would also
move `r_bright` inside the soft weight and make one change into two; §0.7 of
the record already measured that smoothing a continuous weight is inert.

`k=1` reproduces the per-frame rule exactly (asserted where it matters, in
`scripts/eval_veto_hysteresis.py`).
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np

#: The finalized filter: (mode, window). None disables filtering.
ADOPTED_VETO_FILTER: tuple[str, int] = ("dilate", 15)

#: The veil term's filter (2026-09-01). NOT dilate, and the difference is measured.
#:
#: Dilation is asymmetric -- it only ever adds vetoes -- and it is the right filter
#: for brightness because brightness UNDER-fires inside a true dark stretch: fog
#: lifts p05 above mu_b on 71% of night frames, so the switch flickers off where it
#: should be held on. The veil statistic has the opposite failure mode. It does not
#: flicker (100% inside both fog cells, 0.3% in glare/day), so dilation has nothing
#: to repair and instead multiplies the few isolated false positives: dilating the
#: OR-ed switch turns glare/day's 4 flagged frames into 58, a 4.8% veto rate on a
#: guard cell where VIS scores 0.2892 against IR's 0.0177 -- roughly -0.013 AP, more
#: than the +0.0051 the fog fix is worth. Filtering each term with the filter matched
#: to ITS failure mode -- dilate the darkness switch, denoise the veil switch --
#: leaves every vetoed cell at 100% and both guard cells at exactly 0.0%.
ADOPTED_VEIL_FILTER: tuple[str, int] = ("majority", 15)


def temporal_order(records: list[dict]) -> dict[str, np.ndarray]:
    """{run: frame indices in ascending capture order}, from the file stem.

    Frames of one recording run are consecutive in the paired manifest, but the
    capture index is recovered from the filename rather than assumed, so a
    reordered manifest cannot silently corrupt the window.
    """
    runs, nums = [], []
    for r in records:
        p = Path(r["image_path"])
        runs.append(p.parent.name)
        m = re.search(r"(\d+)$", p.stem)
        if not m:
            raise ValueError(f"cannot recover a frame number from {p.stem!r}")
        nums.append(int(m.group(1)))
    runs, nums = np.asarray(runs), np.asarray(nums)
    out = {}
    for run in sorted(set(runs.tolist())):
        idx = np.flatnonzero(runs == run)
        out[run] = idx[np.argsort(nums[idx], kind="mergesort")]
    return out


def filter_veto(veto: list[bool], order: dict[str, np.ndarray], k: int, mode: str) -> list[bool]:
    """Apply a length-k filter to the veto flags, within each run, in time order.

    modes:
      "majority" — veto if more than half the window says veto (denoises both
                   directions, cannot extend a veto far past its evidence)
      "dilate"   — veto if ANY frame in the window says veto (hysteresis proper:
                   once the sensor is shown dark, brief brightenings do not
                   restore trust). The adopted mode.
    """
    v = np.asarray(veto, dtype=bool)
    if k <= 1:
        return v.tolist()
    half = k // 2
    out = v.copy()
    for idx in order.values():
        seq = v[idx].astype(np.int32)
        n = len(seq)
        pad = np.pad(seq, (half, half), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, k)[:n]
        if mode == "majority":
            out[idx] = win.sum(axis=1) * 2 > k
        elif mode == "dilate":
            out[idx] = win.max(axis=1) > 0
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return out.tolist()


def raw_veto_flags(brightness: np.ndarray | None, mu_b: float, tau_b: float,
                   veto_below: float, n: int,
                   structure: np.ndarray | None = None,
                   tau_lap: float | None = None) -> list[bool]:
    """The instantaneous per-frame veto decision, identical to the fitted path
    inside `evaluate_systems` (r_bright < veto_below; no brightness -> never
    vetoed). Computed standalone so the filter can run BEFORE fusion, in one
    pass, via `veto_override`.

    `structure`/`tau_lap` add the SECOND veto axis (2026-09-01). Brightness
    answers "did photons reach the sensor?"; it cannot answer "did the photons
    carry any structure?". Fog is the case that separates them: on fog/day the
    p05 statistic reads 58 -- the BRIGHTEST of all eight cells, brighter than
    clean/day's 35 -- so the photometric veto never fires, while VIS ship AP
    collapses to 0.0020 against IR's 0.0177 and fusing the dead stream costs
    -0.0051. A veil raises brightness and destroys edges at the same time, so the
    two terms are not redundant: they are near-orthogonal, and the gate needs
    both. The terms are OR-ed because either failure alone is disqualifying.

    Fitted as a NOVELTY threshold on clean fit-run frames only (pohang00/02/03,
    the same runs `fit_brightness_gate.py` used), never on the corrupted test
    conditions -- so no fog frame informs the threshold that vetoes fog.
    """
    if brightness is None and structure is None:
        return [False] * n
    veto = np.zeros(n, dtype=bool)
    if brightness is not None:
        b = np.asarray(brightness, dtype=float)
        if len(b) != n:
            raise ValueError(f"brightness has {len(b)} entries for {n} frames")
        tau = max(float(tau_b), 1e-9)
        r_bright = 1.0 / (1.0 + np.exp(-(b - float(mu_b)) / tau))
        veto |= r_bright < veto_below
    if structure is not None and tau_lap is not None:
        v = np.asarray(structure, dtype=float)
        if len(v) != n:
            raise ValueError(f"structure has {len(v)} entries for {n} frames")
        veto |= v < float(tau_lap)
    return veto.tolist()
