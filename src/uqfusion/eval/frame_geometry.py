"""Per-frame camera geometry for the letterboxed Pohang frames (corruption v2).

The v1 corruptions had no notion of where the sea, the horizon or the sky is in a
frame. Fog in particular is a function of distance (Koschmieder: t = exp(-beta d)),
so a physically motivated fog needs a range map. Pohang ships what that needs:

* `calibration/intrinsics.json` and `extrinsics.json` (camera -> body rotation),
* `navigation/ahrs.txt`, the body attitude at ~300 Hz (x, y, z, w quaternion,
  body -> local level frame; verified 2026-10-09 by drawing the predicted horizon on
  12 day frames from pohang00/02/03: it sits on the far waterline in every one, the
  inverse convention does not),
* `timestamps/{stereo,ir}.txt`, frame index -> capture time.

From these the horizon of every frame is a line, and below it the flat-sea range of a
pixel is H_CAM / sin(depression angle). Above the horizon there is no sea: the content
there is far shore or sky, and gets `far_m`.

Known limits, stated where the corruption is described: the sea is taken as a plane
(no Earth curvature; at 3.7 m the dip of the horizon is 0.06 deg, under 1 px), lens
distortion is corrected with the published coefficients, and a vessel's pixels above its
waterline are given the range of the sea BEHIND them (farther than the vessel), so
vessels are attenuated a little more than physics says. That bias is measured, not
assumed, in `scripts/calibrate_corruptions_v2.py`.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

#: Camera height above the sea surface, metres. The stereo and IR cameras sit 1.77 m above
#: the body origin (extrinsics z, FRD), and baseline.txt puts the body origin at 1.9 m
#: (median over pohang00-04; it ranges 0.3-2.9 m, so treat this as +-1 m).
H_CAM_M = 3.7
CANVAS = 640
NATIVE = {"vis": (2048, 1080), "ir": (640, 512)}
TIMESTAMPS = {"vis": "stereo.txt", "ir": "ir.txt"}
DATA_ROOT = Path(__file__).resolve().parents[3] / "Pohang_dataset"


def letterbox(modality: str, canvas: int = CANVAS) -> tuple[float, int, int, int, int]:
    """(scale, new_w, new_h, left, top) of the prepared tree's letterbox."""
    w0, h0 = NATIVE[modality]
    r = min(canvas / w0, canvas / h0)
    nw, nh = round(w0 * r), round(h0 * r)
    return r, nw, nh, (canvas - nw) // 2, (canvas - nh) // 2


def content_rows(modality: str, canvas: int = CANVAS) -> tuple[int, int]:
    """[lo, hi) rows of the canvas that carry image (identical to frame_brightness.content_rows)."""
    _, _, nh, _, top = letterbox(modality, canvas)
    return top, top + nh


def _sensor(path: Path, modality: str) -> str:
    if modality == "ir":
        return "infrared"
    return "stereo_right" if "_R_" in path.stem else "stereo_left"


def parse(path: str | Path, modality: str) -> tuple[str, str, int]:
    """(run, sensor, frame index) from a prepared-tree image path."""
    p = Path(path)
    return p.parent.name, _sensor(p, modality), int(p.stem.split("_")[-1])


@lru_cache(maxsize=16)
def _calib(run: str, sensor: str, root: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    cal = Path(root) / "meta" / run / "calibration"
    k = json.loads((cal / "intrinsics.json").read_text(encoding="utf-8"))[sensor]
    e = json.loads((cal / "extrinsics.json").read_text(encoding="utf-8"))[sensor]
    K = np.array([[k["focal_length"], 0, k["cc_x"]], [0, k["focal_length"], k["cc_y"]], [0, 0, 1.0]])
    return K, np.asarray(k["distortion_coefficients"], float), np.asarray(e["quaternion"], float)


@lru_cache(maxsize=8)
def _nav(run: str, modality: str, root: str) -> tuple[dict, np.ndarray]:
    import pandas as pd

    meta = Path(root) / "meta" / run
    ts = pd.read_csv(meta / "timestamps" / TIMESTAMPS[modality], sep=r"\s+", header=None,
                     dtype={0: float, 1: int})
    ahrs = pd.read_csv(meta / "navigation" / "ahrs.txt", sep=r"\s+", header=None,
                       usecols=range(5)).to_numpy(float)
    return dict(zip(ts[1].tolist(), ts[0].tolist())), ahrs


@lru_cache(maxsize=16)
def _rays(run: str, sensor: str, modality: str, root: str) -> np.ndarray:
    """Undistorted normalised camera rays (x, y, 1) for every content pixel, (h, w, 3)."""
    import cv2

    K, dist, _ = _calib(run, sensor, root)
    r, nw, nh, left, top = letterbox(modality)
    w0, h0 = NATIVE[modality]
    u, v = np.meshgrid(np.arange(nw, dtype=np.float64), np.arange(nh, dtype=np.float64))
    # cv2.resize maps pixel centres: x_src = (x_dst + 0.5) * w0 / nw - 0.5
    xs = (u + 0.5) * w0 / nw - 0.5
    ys = (v + 0.5) * h0 / nh - 0.5
    pts = np.stack([xs.ravel(), ys.ravel()], axis=1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, K, dist).reshape(nh, nw, 2)
    return np.concatenate([und, np.ones((nh, nw, 1))], axis=2)


def _slerp_quat(q0: np.ndarray, q1: np.ndarray, a: float) -> np.ndarray:
    if np.dot(q0, q1) < 0:
        q1 = -q1
    q = (1 - a) * q0 + a * q1          # nlerp: the gaps are <= 67 ms, attitude moves < 0.1 deg
    return q / np.linalg.norm(q)


class FrameGeometry:
    """Rotation camera -> level frame and the derived horizon / sea range of one frame."""

    def __init__(self, path: str | Path, modality: str, root: str | Path = DATA_ROOT):
        from scipy.spatial.transform import Rotation

        self.path, self.modality, root = str(path), modality, str(root)
        self.run, self.sensor, idx = parse(path, modality)
        _, _, q_bc = _calib(self.run, self.sensor, root)
        ts, ahrs = _nav(self.run, modality, root)
        if idx not in ts:
            raise KeyError(f"{path}: frame {idx} not in {self.run} {TIMESTAMPS[modality]}")
        t = ts[idx]
        j = int(np.clip(np.searchsorted(ahrs[:, 0], t), 1, len(ahrs) - 1))
        t0, t1 = ahrs[j - 1, 0], ahrs[j, 0]
        a = float(np.clip((t - t0) / max(t1 - t0, 1e-9), 0, 1))
        self.nav_gap_s = float(min(abs(t - t0), abs(t1 - t)))
        q_wb = _slerp_quat(ahrs[j - 1, 1:5], ahrs[j, 1:5], a)
        self.R_wc = (Rotation.from_quat(q_wb) * Rotation.from_quat(q_bc)).as_matrix()
        self._root = root

    def down(self) -> np.ndarray:
        """Downward component (level frame) of each content pixel's unit ray, (h, w)."""
        rays = _rays(self.run, self.sensor, self.modality, self._root)
        w = rays @ self.R_wc.T
        down = w[..., 2] / np.linalg.norm(w, axis=2)   # FRD/NED: +z is down
        return down

    def range_m(self, far_m: float, h_cam: float = H_CAM_M) -> np.ndarray:
        """Flat-sea range per content pixel in metres, capped at `far_m` (horizon and above)."""
        d = self.down()
        with np.errstate(divide="ignore"):
            rng = np.where(d > 0, h_cam / np.maximum(d, 1e-12), np.inf)
        return np.minimum(rng, far_m)

    def horizon_rows(self) -> np.ndarray:
        """Content-relative row of the horizon in every column (may lie outside the content)."""
        d = self.down()
        h = d.shape[0]
        out = np.empty(d.shape[1])
        for c in range(d.shape[1]):
            col = d[:, c]
            k = int(np.searchsorted(col, 0.0))        # `down` increases with the row
            if k <= 0:
                out[c] = -1.0 if col[0] > 0 else 0.0
            elif k >= h:
                out[c] = float(h)
            else:
                out[c] = k - 1 + (-col[k - 1]) / (col[k] - col[k - 1])
        return out
