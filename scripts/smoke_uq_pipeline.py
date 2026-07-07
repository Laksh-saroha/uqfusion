"""End-to-end PoC smoke: σ head → Mahalanobis → reliability → gated WBF fusion.

Simulates the scope §6.1 pipeline on synthetic data, CPU-only:
  - two "modalities" of the same scenes (identity homography): one clean,
    one fog-degraded (blur + washout + noise);
  - Mahalanobis scorer fit on clean TRAIN features must separate degraded
    frames (O3 works);
  - reliability constants fit on clean val by the pre-registered rules (D5);
  - the degraded stream must get lower R̄ and lower fusion weight on most
    frames (the gate works);
  - WBF fusion runs in the common frame; empty-frame fallback exercised.

Reuses the smoke_gaussian trained weights when present (run that first).

Usage:  python scripts/smoke_uq_pipeline.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.uq.fusion import fuse_detections
from uqfusion.uq.infer import UQPredictor
from uqfusion.uq.mahalanobis import MahalanobisScorer
from uqfusion.uq.reliability import compute_reliability, fit_constants, fusion_weights, per_box_uncertainty
from uqfusion.uq.synthetic import degrade_image


def main() -> int:
    import cv2

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    imgsz = cfg["smoke"]["imgsz"]
    root = Path(cfg["paths"]["outputs_root"])
    best = root / "gaussian" / "smoke_gauss" / "weights" / "best.pt"
    if not best.is_file():
        best = best.with_name("last.pt")
    data_root = root / "smoke_gaussian" / "hetero_data"
    if not best.is_file() or not data_root.is_dir():
        print("[uq-smoke] smoke_gaussian artifacts missing — training them first ...")
        from smoke_gaussian import train_smoke_gaussian  # same scripts/ directory

        best, _, data_root, _ = train_smoke_gaussian(cfg, fresh=True)

    predictor = UQPredictor(best, device="cpu", imgsz=imgsz, conf=0.15)
    train_imgs = sorted((data_root / "images" / "train").glob("*.jpg"))
    val_imgs = sorted((data_root / "images" / "val").glob("*.jpg"))
    assert len(train_imgs) >= 8 and len(val_imgs) >= 8, "synthetic dataset too small"

    # --- O3: fit Mahalanobis on clean TRAIN features, check separation --------
    train_feats = np.stack([predictor(p)["feat"] for p in train_imgs])
    scorer = MahalanobisScorer().fit(train_feats)

    clean_records, degraded_records, d_clean, d_degraded = [], [], [], []
    for i, p in enumerate(val_imgs):
        im = cv2.imread(str(p))
        rec_c = predictor(im)
        rec_d = predictor(degrade_image(im, seed=i))
        rec_c["dist"] = scorer.score(rec_c["feat"])
        rec_d["dist"] = scorer.score(rec_d["feat"])
        clean_records.append(rec_c)
        degraded_records.append(rec_d)
        d_clean.append(rec_c["dist"])
        d_degraded.append(rec_d["dist"])

    d_clean, d_degraded = np.array(d_clean), np.array(d_degraded)
    assert d_degraded.mean() > d_clean.mean(), (
        f"Mahalanobis failed to separate degradation: clean {d_clean.mean():.2f} vs degraded {d_degraded.mean():.2f}"
    )
    frac = float((d_degraded > np.percentile(d_clean, 95)).mean())
    print(f"[uq-smoke] O3 separation: clean d̄ {d_clean.mean():.2f}, degraded d̄ {d_degraded.mean():.2f}; "
          f"{frac:.0%} of degraded frames beyond clean 95th pct")

    # --- D5: constants from clean val by the pre-registered rules -------------
    clean_u = np.concatenate(
        [per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"]) for r in clean_records if len(r["boxes_xyxy"])]
    )
    constants = fit_constants(clean_u, d_clean)
    print(f"[uq-smoke] fitted constants: λ={constants.lam:.3f}, μ_d={constants.mu_d:.2f}, τ={constants.tau:.2f}")

    # --- O4: reliability gate must favor the clean stream ---------------------
    wins, r_clean_all, r_deg_all = 0, [], []
    fused_any = False
    for rec_c, rec_d in zip(clean_records, degraded_records):
        rel_c = compute_reliability(rec_c, rec_c["dist"], constants)
        rel_d = compute_reliability(rec_d, rec_d["dist"], constants)
        r_clean_all.append(rel_c["R"])
        r_deg_all.append(rel_d["R"])
        # treat clean as IR, degraded as VIS ("fog kills the visible camera")
        w = fusion_weights(r_vis=rel_d["R"], r_ir=rel_c["R"])
        wins += w["w_ir"] > w["w_vis"]
        fused = fuse_detections(
            vis_record=rec_d, ir_record=rec_c, w_vis=w["w_vis"], w_ir=w["w_ir"],
            vis_hw=rec_c["image_hw"], h_ir_to_vis=None,
        )
        fused_any |= len(fused["boxes_xyxy"]) > 0

    n = len(clean_records)
    print(f"[uq-smoke] O4 gate: mean R clean {np.mean(r_clean_all):.3f} vs degraded {np.mean(r_deg_all):.3f}; "
          f"clean stream out-weighted degraded on {wins}/{n} frames")
    assert np.mean(r_deg_all) < np.mean(r_clean_all), "reliability did not drop under degradation"
    assert wins >= int(0.7 * n), f"gate favored the clean stream on only {wins}/{n} frames"
    assert fused_any, "WBF fusion produced no boxes on any frame"

    # --- empty-frame fallback (scope §6.4) -------------------------------------
    blank = np.full((imgsz, imgsz, 3), 20, dtype=np.uint8)
    rec_blank = predictor(blank)
    rec_blank["dist"] = scorer.score(rec_blank["feat"])
    rel_blank = compute_reliability(rec_blank, rec_blank["dist"], constants)
    if rel_blank["n_dets"] == 0:
        assert rel_blank["R"] == rel_blank["r_frame"], "empty-frame fallback R != r_frame"
        print(f"[uq-smoke] empty-frame fallback OK (R = r_frame = {rel_blank['R']:.3f})")
    else:
        print(f"[uq-smoke] WARN: blank frame produced {rel_blank['n_dets']} detections — fallback branch untested")

    print("\nUQ PIPELINE SMOKE OK — σ + OOD -> reliability -> gated WBF verified end-to-end on synthetic data")
    return 0


if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).parent))  # allow `from smoke_gaussian import ...`
    sys.exit(main())
