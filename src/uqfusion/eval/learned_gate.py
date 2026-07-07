"""Learned gate — upper-bound comparison for the hand-crafted reliability score
(scope §7.5; plan B4). NOT a replacement: if the interpretable §6.4 score
matches this ceiling, that is a result FOR the method.

Pre-registered design (D4/plan B4):
  features per frame  = [U_box, O, n_dets, mean_conf] x both modalities (8-dim;
                        U_box of an empty frame encoded as the clean-val median
                        so the feature stays finite)
  model               = sklearn LogisticRegression (tiny, CPU-seconds)
  supervision         = binary y: which modality's detections are better on
                        that frame, measured as F1@0.5 vs GT; predicted
                        probability IS the fusion weight w_vis. Frames where
                        both modalities score identically (e.g. both zero)
                        are dropped from training.
"""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from uqfusion.eval.matching import load_gt, match_image
from uqfusion.uq.reliability import ReliabilityConstants, per_box_uncertainty

_EPS = 1e-9


def frame_features(record: dict, distance: float, constants: ReliabilityConstants, u_fill: float) -> np.ndarray:
    o = 1.0 / (1.0 + np.exp(-(distance - constants.mu_d) / constants.tau))
    n = len(record["conf"])
    if n == 0:
        return np.array([u_fill, o, 0.0, 0.0])
    u = per_box_uncertainty(record["sigma_ltrb"], record["boxes_xyxy"])
    c = np.asarray(record["conf"], dtype=np.float64)
    u_box = float((c * u).sum() / max(c.sum(), _EPS))
    return np.array([u_box, o, float(n), float(c.mean())])


def frame_quality_f1(record: dict, iou_thr: float = 0.5) -> float:
    """Detection F1@0.5 of one modality's record against its frame's GT."""
    gt = load_gt(record["image_path"], record["image_hw"])
    m = match_image(record, gt, iou_thr=iou_thr)
    tp = int(m["matched"].sum())
    fp = len(record["conf"]) - tp
    fn = m["n_gt"] - tp
    return 2 * tp / max(2 * tp + fp + fn, _EPS)


class LearnedGate:
    def __init__(self):
        self.model = None
        self.u_fill = 0.0

    def fit(
        self,
        vis_records: list[dict],
        ir_records: list[dict],
        vis_distances: np.ndarray,
        ir_distances: np.ndarray,
        constants: ReliabilityConstants,
    ) -> dict:
        from sklearn.linear_model import LogisticRegression

        all_u = [
            per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
            for r in list(vis_records) + list(ir_records) if len(r["conf"])
        ]
        self.u_fill = float(np.median(np.concatenate(all_u))) if all_u else 0.0

        x, y = [], []
        for rv, ri, dv, di in zip(vis_records, ir_records, vis_distances, ir_distances):
            qv, qi = frame_quality_f1(rv), frame_quality_f1(ri)
            if abs(qv - qi) < _EPS:
                continue  # no supervision signal on this frame
            x.append(np.concatenate([
                frame_features(rv, dv, constants, self.u_fill),
                frame_features(ri, di, constants, self.u_fill),
            ]))
            y.append(1 if qv > qi else 0)  # 1 = VIS better
        if len(y) < 4 or len(set(y)) < 2:
            raise ValueError(f"not enough contrasting frames to fit the gate (n={len(y)}, classes={set(y)})")

        self.model = LogisticRegression(max_iter=1000)
        self.model.fit(np.stack(x), np.asarray(y))
        self._constants = constants
        return {"n_train_frames": len(y), "vis_better_frac": float(np.mean(y))}

    def predict_w_vis(self, vis_record: dict, ir_record: dict, vis_distance: float, ir_distance: float) -> float:
        if self.model is None:
            raise RuntimeError("LearnedGate.predict called before fit/load")
        x = np.concatenate([
            frame_features(vis_record, vis_distance, self._constants, self.u_fill),
            frame_features(ir_record, ir_distance, self._constants, self.u_fill),
        ]).reshape(1, -1)
        return float(self.model.predict_proba(x)[0, 1])

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"model": self.model, "u_fill": self.u_fill, "constants": self._constants}, f)
        return path

    @classmethod
    def load(cls, path: str | Path) -> "LearnedGate":
        with open(path, "rb") as f:
            payload = pickle.load(f)
        gate = cls()
        gate.model = payload["model"]
        gate.u_fill = payload["u_fill"]
        gate._constants = payload["constants"]
        return gate
