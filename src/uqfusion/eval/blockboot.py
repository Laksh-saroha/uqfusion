"""Dependence-aware intervals: paired contiguous-block resampling within runs.

R-A3 (`docs/TODO-2026-09-09-architecture-review.md`, finding F04). Every headline
interval in this project resamples **individual frames** from 10 Hz recordings. Pairing
across systems is correct and is not the problem; the problem is that consecutive
frames of a ship at 10 Hz are nearly the same picture, so 1,032 frames are nowhere near
1,032 independent observations. An iid frame bootstrap therefore estimates a standard
error for a dataset that does not exist, and it is too NARROW -- the opposite direction
from the reassuring R-A1 result.

The fix is the standard one for serially dependent data: resample contiguous **blocks**
long enough to carry the dependence, so a block is approximately exchangeable even
though a frame is not.

What this module changes, and what it cannot
--------------------------------------------
The resampling unit becomes a contiguous block of frames drawn **within a run**, so run
composition is held fixed and the interval covers *scene sampling within these
recordings*. It does not cover, and cannot be made to cover:

* **Between-run variation.** There are four runs (pohang00/01/02/03) and night is
  entirely pohang01. With **one night run the between-run night component cannot be
  estimated at all** -- not with a longer block, not with more draws. Any night interval
  here is conditional on that single recording.
* **Training-seed variation.** A cached prediction set is one trained model. Seed
  variance is a separate component and needs separate replicates.

`describe()` returns those statements alongside every interval, because R-A3's
acceptance criterion is that an interval names its resampling unit and the randomness
it covers.

On sign-flip fractions
----------------------
The fraction of resamples whose sign differs from the observed delta is reported as
``sign_flip_fraction``. **It is not a p-value and not the probability that a hypothesis
is true.** It is a descriptive property of the resampling distribution under this
resampling scheme. The old ``p_sign_flip`` name is kept as an alias so existing readers
do not break, and is deprecated precisely because the name invited the reading R-A3
tells us to stop making.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np

from uqfusion.eval.apmetrics import _score, presort

RUN_RE = re.compile(r"(pohang\d+)")


def run_ids(image_paths: list[str | Path]) -> np.ndarray:
    """Run label per frame, parsed from the path. Frames not matching become '?'."""
    out = []
    for p in image_paths:
        m = RUN_RE.search(str(p))
        out.append(m.group(1) if m else "?")
    return np.asarray(out)


def run_slices(runs: np.ndarray) -> dict[str, np.ndarray]:
    """Frame indices per run, in file order.

    File order IS time order here: names are `pohangNN_L_<ordinal>` and the split was
    built as contiguous per-run ordinal blocks (`scripts/prepare_pohang.py`), so
    adjacency in this array is adjacency in the recording. A block drawn over these
    indices is a genuine contiguous stretch of video, which is the whole point.
    """
    return {r: np.where(runs == r)[0] for r in sorted(set(runs.tolist()))}


def block_resample(rng: np.random.Generator, per_run: dict[str, np.ndarray],
                   block_len: int) -> np.ndarray:
    """One moving-block resample, drawn independently within each run.

    Each run of length ``n`` contributes ``ceil(n / L)`` blocks drawn with replacement
    from its ``n - L + 1`` contiguous start positions, concatenated and truncated back
    to ``n``. Run sizes are therefore preserved exactly, which is what "within runs"
    means: this interval does not resample *which* runs are present.

    ``block_len = 1`` degenerates to an iid frame bootstrap **within run**. That is the
    natural baseline for the sensitivity table -- it isolates the block length as the
    only thing changing, unlike the existing global frame bootstrap, which also lets
    run composition vary.
    """
    picks = []
    for idx in per_run.values():
        n = len(idx)
        if n == 0:
            continue
        L = max(1, min(int(block_len), n))
        n_blocks = int(np.ceil(n / L))
        starts = rng.integers(0, n - L + 1, n_blocks)
        take = (starts[:, None] + np.arange(L)[None, :]).ravel()[:n]
        picks.append(idx[take])
    return np.concatenate(picks) if picks else np.zeros(0, dtype=np.int64)


def describe(block_len: int, per_run: dict[str, np.ndarray],
             night_run: str = "pohang01") -> dict[str, Any]:
    """The provenance R-A3 requires every interval to carry."""
    sizes = {r: int(len(i)) for r, i in per_run.items()}
    night_only = [r for r in sizes if r == night_run]
    return {
        "resampling_unit": (f"contiguous block of {int(block_len)} frames, drawn within run"
                            if block_len > 1 else "single frame, drawn within run"),
        "covers": "scene sampling within these recordings, paired across systems",
        "does_not_cover": [
            "between-run variation (runs are held fixed by construction)",
            f"between-night-run variation — night is entirely {night_run}, "
            "so this component is not estimable from this data at any block length",
            "training-seed variation — each cache is one trained model",
        ],
        "runs": sizes,
        "n_runs": len(sizes),
        "night_runs": len(night_only),
        "block_len": int(block_len),
    }


def block_bootstrap_delta(parts_a: list[dict], parts_b: list[dict],
                          image_paths: list[str | Path], block_len: int,
                          sel: np.ndarray | None = None, n_boot: int = 1000,
                          seed: int = 0, cls: int | None = None) -> dict:
    """Paired contiguous-block bootstrap of (A − B), the dependence-aware interval.

    Same pairing logic as `apmetrics.bootstrap_delta` -- one resample scores both
    systems, so frame-composition noise cancels in the difference -- with the frame
    replaced by a block as the unit that gets resampled.
    """
    idx = (np.arange(len(parts_a)) if sel is None else np.asarray(sel))
    paths = [image_paths[i] for i in idx]
    per_run = run_slices(run_ids(paths))          # positions WITHIN `idx`

    pa, pb = presort(parts_a, idx), presort(parts_b, idx)
    n = pa["n_frames"]
    rng = np.random.default_rng(seed)

    obs_a, obs_b = _score(pa, None, cls), _score(pb, None, cls)
    obs_d = obs_a - obs_b

    deltas = np.empty(n_boot)
    for t in range(n_boot):
        take = block_resample(rng, per_run, block_len)
        w = np.bincount(take, minlength=n)        # multiplicities, as ap_weighted wants
        deltas[t] = _score(pa, w, cls) - _score(pb, w, cls)

    good = np.isfinite(deltas)
    kept = deltas[good]
    meta = describe(block_len, per_run)
    if kept.size < 2:
        return {"a": obs_a, "b": obs_b, "delta": obs_d, "ci_lo": float("nan"),
                "ci_hi": float("nan"), "se": float("nan"),
                "sign_flip_fraction": float("nan"), "p_sign_flip": float("nan"),
                "spans_zero": None, "n_frames": int(n), "n_boot": int(n_boot),
                "n_undefined": int((~good).sum()), "n_effective": int(kept.size), **meta}

    lo, hi = np.percentile(kept, [2.5, 97.5])
    flip = (float(np.mean(np.sign(kept) != np.sign(obs_d)))
            if np.isfinite(obs_d) and obs_d != 0 else 1.0)
    return {
        "a": obs_a, "b": obs_b, "delta": obs_d,
        "ci_lo": float(lo), "ci_hi": float(hi), "se": float(kept.std(ddof=1)),
        # NOT a p-value and NOT P(hypothesis) -- see the module docstring. `p_sign_flip`
        # is the deprecated alias, kept only so existing readers do not break.
        "sign_flip_fraction": flip, "p_sign_flip": flip,
        "spans_zero": bool(lo <= 0.0 <= hi),
        "n_frames": int(n), "n_boot": int(n_boot),
        "n_undefined": int((~good).sum()), "n_effective": int(kept.size),
        **meta,
    }
