# Calibration & IR→VIS Registration Guide

> Answers "what is the calibration file, and can one be made for the MIT dataset?" (Laksh, 2026-07-08). Applies to both datasets; decision D7 (fuse + evaluate in the VIS frame) is the consumer.

## 1. What a calibration file is

Per camera: **intrinsics** — focal lengths (fx, fy), principal point (cx, cy), lens-distortion coefficients, together the matrix **K** (3D rays → that camera's pixels); **extrinsics** — the fixed rotation **R** and translation **t** between rigidly mounted cameras (or camera → common rig frame).

Robotics datasets ship these as small YAML/JSON/txt files (the Pohang IJRR release includes them — OQ-5). They are produced at collection time by imaging a known target (checkerboard; a heated board for IR↔VIS cross-calibration), so **classic calibration cannot be redone after the fact without the boat**.

## 2. What we need: one 3×3 homography per camera pair

WBF needs IR boxes mapped into the VIS image plane — not full intrinsics or 3D geometry, just the **plane-induced homography H** (3×3, 8 DoF) from IR to VIS pixels. A single static H per pair is adequate because:

1. Maritime targets are far relative to the camera baseline, so parallax is small; for a distant scene the mapping approaches the *infinite homography* `H∞ = K_vis · R · K_ir⁻¹`, exactly computable from shipped calibration.
2. WBF tolerates imperfect overlap — residual error just lowers cross-modal IoU, absorbed by the fusion IoU threshold (0.55 default, tunable down; thermal bloom already pushes the same way).

Near-field content (dock walls, very close vessels) carries the largest residual — disclosed in the manuscript, not fixed.

**Storage:** `data/<dataset>/calibration/H_<ircam>_to_<viscam>.json`

```json
{
  "H": [[...],[...],[...]],
  "ir_camera": "ir_left", "vis_camera": "vis_left",
  "image_space": "letterboxed 640x640 (delivered format)",
  "method": "box-correspondence RANSAC | from-calibration",
  "n_correspondences": 132, "median_reproj_px": 3.1,
  "frames_used": ["..."], "date": "YYYY-MM-DD"
}
```

**Coordinate-space rule:** all delivered images are letterboxed 640×640 (OQ-7), so estimate/express H directly in that space. No resize composition is ever needed — H applies verbatim to emitted boxes.

## 3. Route A — dataset ships calibration (expected for Pohang)

1. Copy release calibration files to `data/pohang/calibration/` as-is (any format — we parse).
2. Compute `H∞ = K_vis · R · K_ir⁻¹` for the chosen pair (K's adjusted for each stream's letterbox scale/pad).
3. **Validate before trusting** (Route B step 4).

## 4. Route B — estimate H from data (works for MIT)

Requires paired VIS+IR frames (timestamp-matched) where the **same objects are visible in both modalities**, spread across the field of view.

1. **Select frames.** ≥20 paired frames per (IR cam, VIS cam) pair, targets at varied image positions, mid-to-far range (very close objects pollute the fit with parallax). If the rig was remounted between dates, estimate per date-group and compare (step 5).
2. **Collect correspondences** (in 640×640 letterboxed coords), best-first:
   - **a. Annotated-box correspondences (preferred, near-automatic):** for each vessel labeled in *both* modalities, take the 4 box corners as 4 point pairs. Needs only an object-matching pass (one visible vessel = trivial; several = match by relative position/size). Thermal bloom inflates IR boxes 10–30%, adding noise — RANSAC handles it, and many boxes average it out.
   - **b. Manual point clicks (fallback/supplement):** 15–30 distinctive static points across several frames — buoys, dock corners, horizon–structure intersections, moored-boat masts. ~1 hour, once per camera pair.
   - **c. Automatic LWIR↔VIS feature matching (refinement only):** classic descriptors (SIFT/ORB) are unreliable across these modalities; edge-map or mutual-information alignment can polish an existing H but not bootstrap it.
3. **Fit:** `cv2.findHomography(pts_ir, pts_vis, cv2.RANSAC, ransacReprojThreshold=4.0)`.
4. **Validate (all three, mandatory):** median reprojection error on *held-out* correspondences ≲5–8 px at 640 scale; visual overlay (`cv2.warpPerspective` IR over VIS — vessel/horizon edges should coincide); box test — mean IoU between mapped IR boxes and same-object VIS boxes must rise substantially vs identity (the number fusion cares about — report it).
5. **Stability check:** re-estimate on a second frame set/date-group; corner-mapping disagreement should stay within a few px. Larger drift ⇒ rig moved ⇒ per-date-group homographies.
6. Save per §2 and commit the JSON (small, and results depend on it — scope §11.1 traceability).

## 5. Dataset specifics

| | Pohang | MIT Marine Perception |
|---|---|---|
| Calibration shipped? | Expected (IJRR release) — **OQ-5: drop files in `data/pohang/calibration/`** | Check release; if absent, Route B |
| Camera pairs | stereo VIS (use left) ↔ single IR → **one H** | multiple cams per modality (dataset_requirement.md §7): **one H per used (IR, VIS) pair** — recommend fixing ONE canonical pair to keep this to a single H |
| Route B feasibility | ~28k paired labeled frames → box-correspondence route essentially automatic | manual annotation subset → box route where both-modality labels exist, else Route B-b clicks |
| Timestamp pairing | nearest-neighbor ≤50 ms (dataset convention) | VIS 12 fps / IR 30 fps → nearest-neighbor, tolerance ≤42 ms (half the VIS period) |

## 6. Tooling status

`uqfusion.uq.fusion.apply_homography` already consumes H (identity-tested in the smokes). The Route-B estimation script (`scripts/estimate_homography.py`: correspondence collection from paired labels + RANSAC + the three validations + JSON writer) is **planned for Phase 4 / MIT onboarding** — small, and only worth building against real paired data. Until an H exists, fusion runs/evaluations for that dataset are blocked (single-modality and calibration work unaffected).
