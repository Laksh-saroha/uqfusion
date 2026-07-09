# Calibration & IR→VIS Registration Guide

> Answers "what is the calibration file, and can one be made for the MIT dataset?" (Laksh, 2026-07-08). Applies to both datasets; decision D7 (fuse + evaluate in the VIS frame) is the consumer.

## 1. What a calibration file is

A camera calibration file records, per camera:

- **Intrinsics** — focal lengths (fx, fy), principal point (cx, cy), lens-distortion coefficients. Together the matrix **K**: how 3D rays map to that camera's pixels.
- **Extrinsics** — the fixed rotation **R** and translation **t** between cameras rigidly mounted on the same rig (or from each camera to a common rig frame).

Robotics datasets ship these as small YAML/JSON/txt files (the Pohang IJRR release includes them — that's open question OQ-5). They are produced at collection time by imaging a known target (checkerboard; a heated board for IR↔VIS cross-calibration) — which means **classic calibration cannot be redone after the fact without the boat**.

## 2. What we actually need: one 3×3 homography per camera pair

Fusion (WBF) needs IR boxes mapped into the VIS image plane. For that we need neither full intrinsics nor 3D geometry — just the **plane-induced homography H** (a 3×3 matrix, 8 degrees of freedom) from IR pixels to VIS pixels. Two facts make a single static H per camera pair adequate here:

1. Maritime targets are far relative to the camera baseline, so parallax is small; for a distant scene the mapping approaches the *infinite homography* `H∞ = K_vis · R · K_ir⁻¹`, which is exactly computable from shipped calibration.
2. WBF tolerates *imperfect* overlap — residual error just lowers cross-modal IoU, absorbed by the fusion IoU threshold (0.55 default, tunable down; thermal bloom already pushes the same direction).

Near-field content (dock walls, very close vessels) carries the largest residual — disclosed in the manuscript, not fixed.

**Storage format (both datasets):** `data/<dataset>/calibration/H_<ircam>_to_<viscam>.json`

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

**Coordinate-space rule:** since all delivered images are letterboxed 640×640 (OQ-7), estimate/express H directly in that 640×640 space. Then no resize composition is ever needed — H applies verbatim to the boxes our predictors emit.

## 3. Route A — dataset ships calibration (expected for Pohang)

1. Locate the calibration files in the release; copy to `data/pohang/calibration/` as-is (any format — we parse).
2. Compute `H∞ = K_vis · R · K_ir⁻¹` for the chosen camera pair (K's adjusted for the letterbox scale/pad of each stream).
3. **Validate before trusting** (same checks as Route B, step 4).

## 4. Route B — no calibration shipped: estimate H from the data (works for MIT)

Yes — this can be made for the MIT dataset. Requirements: paired VIS+IR frames (timestamp-matched) where the **same objects are visible in both modalities**, spread across the field of view.

1. **Select frames.** ≥ 20 paired frames per (IR cam, VIS cam) pair, with targets at varied image positions, mid-to-far range (avoid very close objects — parallax pollutes the fit). If the rig was remounted between collection dates, estimate per date-group and compare (step 5).
2. **Collect correspondences** (in 640×640 letterboxed coordinates), best-first:
   - **a. Annotated-box correspondences (preferred, near-automatic):** for each vessel labeled in *both* modalities of a pair, take the 4 box corners as 4 point pairs. With our annotations this needs only an object-matching pass (one visible vessel per frame = trivial; several = match by relative position/size). Caveat: thermal bloom inflates IR boxes 10–30%, adding noise — RANSAC handles it, and using many boxes averages it out.
   - **b. Manual point clicks (fallback / supplement):** 15–30 distinctive static points across several frames — buoys, dock corners, horizon–structure intersections, moored-boat masts. An hour of work, once per camera pair.
   - **c. Automatic LWIR↔VIS feature matching (optional refinement only):** classic descriptors (SIFT/ORB) are unreliable across these modalities; edge-map or mutual-information alignment can polish an existing H but should not bootstrap it.
3. **Fit:** `cv2.findHomography(pts_ir, pts_vis, cv2.RANSAC, ransacReprojThreshold=4.0)`.
4. **Validate (mandatory, all three):**
   - median reprojection error on *held-out* correspondences ≲ 5–8 px at 640 scale;
   - visual overlay: warp the IR frame with H (`cv2.warpPerspective`), blend over VIS — edges of vessels/horizon should coincide;
   - box test: mean IoU between mapped IR boxes and VIS boxes of the same objects must rise substantially vs. identity (this is the number fusion actually cares about — report it).
5. **Stability check:** re-estimate on a second frame set / date-group; corner-mapping disagreement of the two H's should stay within a few px. Larger drift ⇒ the rig moved ⇒ per-date-group homographies.
6. Save per §2's JSON format; commit the JSON (it's small, and results depend on it — scope §11.1 traceability).

## 5. Dataset specifics

| | Pohang | MIT Marine Perception |
|---|---|---|
| Calibration shipped? | Expected (IJRR release) — **OQ-5: drop files in `data/pohang/calibration/`** | Check the release; if absent, Route B |
| Camera pairs | stereo VIS (use left) ↔ single IR → **one H** | multiple cams per modality (see dataset_requirement.md §7): **one H per used (IR, VIS) pair** — recommend fixing ONE canonical pair for fusion to keep this to a single H |
| Route B feasibility | ~28k paired labeled frames → box-correspondence route is essentially automatic | manual annotation subset → box route where both-modality labels exist, else Route B-b clicks |
| Timestamp pairing | nearest-neighbor ≤ 50 ms (dataset convention) | VIS 12 fps / IR 30 fps → nearest-neighbor, tolerance ≤ 42 ms (half the VIS period) |

## 6. Tooling status

`uqfusion.uq.fusion.apply_homography` already consumes H (identity-tested in the smokes). The Route-B estimation script (`scripts/estimate_homography.py`: correspondence collection from paired labels + RANSAC + the three validations + JSON writer) is **planned for Phase 4 / MIT onboarding** — small, and only worth building against the real paired data. Until an H exists, fusion runs/evaluations for that dataset are blocked (single-modality and calibration work are unaffected).
