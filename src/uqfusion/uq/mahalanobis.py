"""Frame-level distributional/OOD score (O3 — scope §6.3; R12).

Mahalanobis distance of pooled neck features (from UQPredictor's hook) to the
clean-training feature distribution. Ledoit-Wolf shrinkage keeps the covariance
stable when the feature dimension rivals the sample count (scope §11.3 B.3).
Fit ONLY on clean training/validation frames — the whole point is that degraded
frames score far (scope §6.3: catches the sensor giving up entirely, which
per-box σ cannot see when there are no boxes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class MahalanobisScorer:
    """Fit on clean features (N, D); score single vectors or batches."""

    def __init__(self):
        self.mean_: np.ndarray | None = None
        self.precision_: np.ndarray | None = None

    def fit(self, features: np.ndarray) -> "MahalanobisScorer":
        from sklearn.covariance import LedoitWolf

        features = np.asarray(features, dtype=np.float64)
        if features.ndim != 2 or features.shape[0] < 2:
            raise ValueError(f"expected (N>=2, D) feature matrix, got {features.shape}")
        lw = LedoitWolf().fit(features)
        self.mean_ = lw.location_
        self.precision_ = lw.precision_
        return self

    def score(self, features: np.ndarray) -> np.ndarray | float:
        """Mahalanobis distance(s); scalar for a single (D,) vector."""
        if self.mean_ is None:
            raise RuntimeError("MahalanobisScorer.score called before fit/load")
        features = np.asarray(features, dtype=np.float64)
        single = features.ndim == 1
        x = np.atleast_2d(features) - self.mean_
        d2 = np.einsum("ij,jk,ik->i", x, self.precision_, x)
        d = np.sqrt(np.clip(d2, 0.0, None))
        return float(d[0]) if single else d

    def score_fast(self, features: np.ndarray) -> np.ndarray:
        """`score` for a batch (N, D) with one BLAS product: ~130x faster, equal to `score` to ~1e-13
        relative but NOT bit-identical (a different summation order). Used only under
        `UQF_FAST_DIST=1` (`fusion_eval._distances`, `ctx.constants_for`), where the distance enters
        the output solely through a sigmoid the shipped constants saturate (mu_d = 1e9 -> r_frame = 1.0
        exactly); checked output-identical on pohang04 jobs before use."""
        if self.mean_ is None:
            raise RuntimeError("MahalanobisScorer.score_fast called before fit/load")
        x = np.asarray(features, dtype=np.float64) - self.mean_
        d2 = np.einsum("ij,ij->i", x @ self.precision_, x)
        return np.sqrt(np.clip(d2, 0.0, None))

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(path, mean=self.mean_, precision=self.precision_)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "MahalanobisScorer":
        data = np.load(path)
        scorer = cls()
        scorer.mean_ = data["mean"]
        scorer.precision_ = data["precision"]
        return scorer
