"""Per-run IR->VIS homography from the Pohang calibration (D7 / OQ-5).

Replaces the identity-H placeholder the fusion smoke uses. `fuse_detections`
maps IR boxes into the VIS frame with a single 3x3 matrix, so this script
derives that matrix from the sensor calibration and reports how much accuracy
the 3x3 form costs.

Chain (both endpoints are the 640x640 LETTERBOXED canvases the models actually
see, not the native sensors):

    IR canvas px -> IR native px -> undistort (K_ir, D_ir) -> normalized ray
      -> rotate by R = R_stereo_left^T @ R_infrared   (relative rotation only)
      -> project + distort (K_sl, D_sl) -> VIS native px -> VIS canvas px

Two approximations, both quantified in the report this writes:

1. **Rotation only, no translation.** IR and stereo_left sit 0.11-0.21 m apart,
   so the exact mapping is range-dependent and no homography can be exact. The
   induced disparity is f*B/Z px; at the VIS canvas scale that is well under a
   pixel at long range and grows as targets close. Reported as a range table.
2. **3x3 fit.** Lens distortion makes the true mapping non-projective, so the
   3x3 matrix is a least-squares fit over a grid of IR-canvas points. The
   residual is reported; it is the price of staying compatible with
   `apply_homography`.

Extrinsics are per-run (relative rotation varies up to ~0.6 deg between runs,
~6 px on the canvas), so one matrix per run is written, not one global matrix.

Usage:
    python scripts/derive_homography.py --out runs/derived/homography_ir_to_vis.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config

# Letterbox geometry of the prepared 640x640 trees, measured from the images
# themselves (pad value 114): VIS 2048x1080 * 0.3125 -> 640x338 at y+151,
# IR 640x512 * 1.0 -> 640x512 at y+64.
VIS_NATIVE = (2048, 1080)
IR_NATIVE = (640, 512)
CANVAS = 640


def letterbox_params(w0: int, h0: int, target: int = CANVAS) -> tuple[float, int, int]:
    scale = min(target / w0, target / h0)
    nw, nh = round(w0 * scale), round(h0 * scale)
    return scale, (target - nw) // 2, (target - nh) // 2


def quat_to_R(q: list[float]) -> np.ndarray:
    """Pohang extrinsics store [x, y, z, w]; returns sensor->body rotation."""
    x, y, z, w = q
    n = np.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def K_of(intr: dict) -> np.ndarray:
    f = intr["focal_length"]
    return np.array([[f, 0.0, intr["cc_x"]], [0.0, f, intr["cc_y"]], [0.0, 0.0, 1.0]])


def exact_map(pts_ir_canvas: np.ndarray, K_ir, D_ir, K_vis, D_vis, R) -> np.ndarray:
    """IR canvas px -> VIS canvas px through the full distortion-aware chain."""
    import cv2

    s_ir, padl_ir, padt_ir = letterbox_params(*IR_NATIVE)
    s_vis, padl_vis, padt_vis = letterbox_params(*VIS_NATIVE)

    native = (pts_ir_canvas - np.array([padl_ir, padt_ir])) / s_ir
    rays = cv2.undistortPoints(native.reshape(-1, 1, 2).astype(np.float64), K_ir, D_ir).reshape(-1, 2)
    rays3 = np.hstack([rays, np.ones((len(rays), 1))])
    rot = rays3 @ R.T                     # IR optical frame -> VIS optical frame
    ok = rot[:, 2] > 1e-6                 # points behind the VIS camera have no image
    proj, _ = cv2.projectPoints(rot.reshape(-1, 1, 3), np.zeros(3), np.zeros(3), K_vis, D_vis)
    proj = proj.reshape(-1, 2)
    out = proj * s_vis + np.array([padl_vis, padt_vis])
    out[~ok] = np.nan
    return out


def main() -> int:
    import cv2

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--out", default="runs/derived/homography_ir_to_vis.json")
    parser.add_argument("--grid", type=int, default=40, help="grid resolution per axis for the fit")
    args = parser.parse_args()

    cfg = load_config(args.config)
    meta_root = Path(cfg["datasets"]["pohang"]["root"]) / "meta"
    runs = sorted(p.name for p in meta_root.iterdir()
                  if p.is_dir() and (p / "calibration" / "extrinsics.json").is_file())

    s_ir, _, padt_ir = letterbox_params(*IR_NATIVE)
    # Fit only over the IR CONTENT rows -- the pad bands carry no image and
    # letting them vote would bend the fit toward meaningless geometry.
    xs = np.linspace(0, CANVAS - 1, args.grid)
    ys = np.linspace(padt_ir, padt_ir + IR_NATIVE[1] * s_ir - 1, args.grid)
    grid = np.stack(np.meshgrid(xs, ys), -1).reshape(-1, 2)

    out: dict[str, dict] = {}
    for run in runs:
        calib = meta_root / run / "calibration"
        intr = json.loads((calib / "intrinsics.json").read_text())
        extr = json.loads((calib / "extrinsics.json").read_text())
        K_ir, K_vis = K_of(intr["infrared"]), K_of(intr["stereo_left"])
        D_ir = np.array(intr["infrared"]["distortion_coefficients"], dtype=np.float64)
        D_vis = np.array(intr["stereo_left"]["distortion_coefficients"], dtype=np.float64)
        R = quat_to_R(extr["stereo_left"]["quaternion"]).T @ quat_to_R(extr["infrared"]["quaternion"])
        baseline = float(np.linalg.norm(np.array(extr["infrared"]["translation"])
                                        - np.array(extr["stereo_left"]["translation"])))
        rel_deg = float(np.degrees(np.arccos(np.clip((np.trace(R) - 1) / 2, -1, 1))))

        dst = exact_map(grid, K_ir, D_ir, K_vis, D_vis, R)
        keep = np.isfinite(dst).all(1)
        src_k, dst_k = grid[keep], dst[keep]
        H, _ = cv2.findHomography(src_k.reshape(-1, 1, 2), dst_k.reshape(-1, 1, 2), 0)
        approx = cv2.perspectiveTransform(src_k.reshape(-1, 1, 2), H).reshape(-1, 2)
        res = np.linalg.norm(approx - dst_k, axis=1)

        out[run] = {
            "H_ir_canvas_to_vis_canvas": H.tolist(),
            "relative_rotation_deg": rel_deg,
            "baseline_m": baseline,
            "fit_residual_px": {"mean": float(res.mean()), "p95": float(np.percentile(res, 95)),
                                "max": float(res.max())},
            "n_fit_points": int(keep.sum()),
            "frame": "both endpoints are 640x640 letterboxed canvases",
            "approximation": "relative rotation only; translation ignored (see parallax_px_by_range_m)",
        }
        print(f"{run:<10} rel.rot {rel_deg:6.3f} deg  baseline {baseline:6.3f} m  "
              f"{int(keep.sum()):>5} pts  residual mean {res.mean():.3f}  "
              f"p95 {np.percentile(res, 95):.3f}  max {res.max():.3f} px")

    # Parallax the rotation-only form cannot represent, at the VIS canvas scale.
    s_vis, _, _ = letterbox_params(*VIS_NATIVE)
    f_vis = json.loads((meta_root / runs[0] / "calibration" / "intrinsics.json").read_text())["stereo_left"]["focal_length"]
    b_max = max(v["baseline_m"] for v in out.values())
    ranges = [20, 50, 100, 200, 500, 1000]
    parallax = {str(z): round(f_vis * b_max / z * s_vis, 3) for z in ranges}
    print(f"\nparallax not modelled (worst-case baseline {b_max:.3f} m), px on the 640 VIS canvas:")
    print("  " + "   ".join(f"{z} m: {parallax[str(z)]}" for z in ranges))

    payload = {"runs": out,
               "parallax_px_by_range_m": parallax,
               "worst_baseline_m": b_max,
               "note": "IR->VIS mapping for 640x640 letterboxed canvases; per-run because "
                       "extrinsics differ between runs."}
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\n[homography] -> {out_path}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
