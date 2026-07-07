"""Phase 3 smoke gate — baselines, metrics, caches, corruptions, learned gate,
fusion-system eval — end-to-end on CPU with synthetic data. MUST pass before
any of Phase 3 touches the GPU.

Covers, with hard asserts:
  1. MC-Dropout: dropout inserts + trains; T stochastic passes disagree
     (cluster σ non-degenerate);
  2. Deep Ensemble (M=2 at smoke scale): members train via the grid; member
     disagreement produces σ;
  3. every albumentations corruption produces a valid, changed image;
  4. prediction caches round-trip (build -> load);
  5. pre-registered metrics are finite for all three uncertainty sources,
     including the DFL-derived §7.2 row from the Gaussian cache;
  6. learned gate fits and favors the clean stream;
  7. fusion-system eval: gated fusion beats the degraded single stream;
     gate ablation sweep produces its table.

Usage:  python scripts/smoke_phase3.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))  # allow `from smoke_gaussian import ...`

from uqfusion.config import load_config
from uqfusion.eval.cache import build_cache, load_cache
from uqfusion.eval.corruptions import CORRUPTIONS, make_corruption
from uqfusion.eval.fusion_eval import ablate_gate_rules, evaluate_systems, format_ablation_table
from uqfusion.eval.learned_gate import LearnedGate
from uqfusion.eval.metrics import summarize_cache
from uqfusion.uq.ensemble import EnsemblePredictor, train_ensemble
from uqfusion.uq.infer import UQPredictor
from uqfusion.uq.mahalanobis import MahalanobisScorer
from uqfusion.uq.mc_dropout import MCDropoutPredictor, train_mc_dropout
from uqfusion.uq.reliability import fit_constants, per_box_uncertainty
from uqfusion.uq.synthetic import degrade_image

METRIC_KEYS = ("d_ece", "nll", "interval_ece", "ause", "aurc", "map50_95")


def ensure_gaussian(cfg):
    root = Path(cfg["paths"]["outputs_root"])
    best = root / "gaussian" / "smoke_gauss" / "weights" / "best.pt"
    data_root = root / "smoke_gaussian" / "hetero_data"
    if not best.is_file() or not data_root.is_dir():
        print("[p3-smoke] smoke_gaussian artifacts missing — training them first ...")
        from smoke_gaussian import train_smoke_gaussian

        best, _, data_root, _ = train_smoke_gaussian(cfg, fresh=True)
    return best, data_root


def main() -> int:
    import cv2

    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    s = cfg["smoke"]
    imgsz = s["imgsz"]
    out_root = Path(cfg["paths"]["outputs_root"]) / "smoke_phase3"
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True)

    gauss_best, data_root = ensure_gaussian(cfg)
    data_yaml = data_root / "hetero_data.yaml"
    train_imgs = sorted((data_root / "images" / "train").glob("*.jpg"))
    val_imgs = sorted((data_root / "images" / "val").glob("*.jpg"))

    # --- 1. baselines train --------------------------------------------------
    print("[p3-smoke] training MC-Dropout baseline (tiny) ...")
    mc_best, _ = train_mc_dropout(cfg, data_yaml, variant=s["variant"], epochs=10,
                                  imgsz=imgsz, batch=s["batch"], workers=0, run_name="smoke_mc")
    print("[p3-smoke] training 2-member ensemble (tiny) ...")
    member_weights, _ = train_ensemble(cfg, data_yaml, variant=s["variant"], seeds=[0, 1],
                                       epochs=10, imgsz=imgsz, batch=s["batch"], workers=0,
                                       out_csv=out_root / "ensemble_members.csv")

    kw = dict(device="cpu", imgsz=imgsz, conf=0.15)
    predictors = {
        "gaussian": UQPredictor(gauss_best, **kw),
        "mc": MCDropoutPredictor(mc_best, T=5, **kw),
        "ensemble": EnsemblePredictor(member_weights, **kw),
    }
    print(f"[p3-smoke] MC dropout layers armed: {predictors['mc'].n_armed}")

    # --- 2. corruption functions all produce valid changed images -------------
    probe = cv2.imread(str(val_imgs[0]))
    for name in CORRUPTIONS:
        out = make_corruption(name, severity=2, seed=0)(probe, 0)
        assert out.shape == probe.shape and out.dtype == np.uint8, f"corruption '{name}' broke the image"
        assert not np.array_equal(out, probe), f"corruption '{name}' was a no-op"
    print(f"[p3-smoke] all {len(CORRUPTIONS)} corruptions valid: {', '.join(CORRUPTIONS)}")

    # --- 3. caches: sources x conditions --------------------------------------
    def degrade(im, i):
        return degrade_image(im, seed=i)

    fog = make_corruption("fog", severity=2, seed=0)
    caches: dict[str, list[dict]] = {}
    for src, predictor in predictors.items():
        for cond, transform in (("clean", None), ("degraded", degrade)):
            path = build_cache(predictor, val_imgs, out_root / f"{src}_{cond}.pkl",
                               meta={"source": src, "cond": cond}, transform=transform, log_every=0)
            caches[f"{src}_{cond}"] = load_cache(path)[0]
    caches["gaussian_fog"] = load_cache(
        build_cache(predictors["gaussian"], val_imgs, out_root / "gaussian_fog.pkl",
                    meta={"source": "gaussian", "cond": "fog"}, transform=fog, log_every=0)
    )[0]
    train_cache = load_cache(
        build_cache(predictors["gaussian"], train_imgs, out_root / "gaussian_train_clean.pkl",
                    meta={"source": "gaussian", "cond": "train_clean"}, log_every=0)
    )[0]

    # sampling-based σ must be non-degenerate (passes/members actually disagree)
    for src in ("mc", "ensemble"):
        sig = np.concatenate([r["sigma_ltrb"].ravel() for r in caches[f"{src}_clean"] if len(r["conf"])])
        cv = sig.std() / max(sig.mean(), 1e-9)
        assert sig.size and np.isfinite(sig).all() and cv > 0.01, f"{src} σ degenerate (CV {cv:.4f})"
        print(f"[p3-smoke] {src} cluster σ: mean {sig.mean():.2f}px, CV {cv:.3f} — non-degenerate")

    # --- 4. pre-registered metrics for every source (+ DFL row, + under shift) -
    table_rows = {}
    for label, (records, key) in {
        "gaussian_clean": (caches["gaussian_clean"], "sigma_ltrb"),
        "gaussian_dfl_clean": (caches["gaussian_clean"], "dfl_sigma_ltrb"),
        "gaussian_degraded": (caches["gaussian_degraded"], "sigma_ltrb"),
        "gaussian_fog": (caches["gaussian_fog"], "sigma_ltrb"),
        "mc_clean": (caches["mc_clean"], "sigma_ltrb"),
        "mc_degraded": (caches["mc_degraded"], "sigma_ltrb"),
        "ensemble_clean": (caches["ensemble_clean"], "sigma_ltrb"),
        "ensemble_degraded": (caches["ensemble_degraded"], "sigma_ltrb"),
    }.items():
        summary = summarize_cache(records, sigma_key=key)
        table_rows[label] = summary
        shown = {k: round(summary[k], 4) for k in METRIC_KEYS if k in summary and np.isfinite(summary.get(k, np.nan))}
        print(f"[p3-smoke] {label}: {shown}")
    for label in ("gaussian_clean", "gaussian_dfl_clean", "mc_clean", "ensemble_clean"):
        for k in METRIC_KEYS:
            v = table_rows[label].get(k)
            assert v is not None and np.isfinite(v), f"{label}.{k} not finite: {v}"
    md = ["| source/condition | " + " | ".join(METRIC_KEYS) + " |", "|" + "---|" * (len(METRIC_KEYS) + 1)]
    for label, r in table_rows.items():
        md.append(f"| {label} | " + " | ".join(f"{r.get(k, float('nan')):.4f}" for k in METRIC_KEYS) + " |")
    (out_root / "table2_smoke.md").write_text("\n".join(md) + "\n", encoding="utf-8")

    # --- 5. constants + learned gate ------------------------------------------
    scorer = MahalanobisScorer().fit(np.stack([r["feat"] for r in train_cache]))
    clean_records = caches["gaussian_clean"]
    deg_records = caches["gaussian_degraded"]
    clean_u = np.concatenate([per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"])
                              for r in clean_records if len(r["conf"])])
    clean_d = np.asarray([scorer.score(r["feat"]) for r in clean_records])
    constants = fit_constants(clean_u, clean_d)

    # Gate training needs BOTH outcomes represented (at toy scale, total fog-
    # blindness means "clean always wins" -> single-class labels). Interleave
    # which stream is degraded — the shape of real mixed-condition val data.
    vis_mixed, ir_mixed = [], []
    for i, (rc, rd) in enumerate(zip(clean_records, deg_records)):
        vis_mixed.append(rd if i % 2 == 0 else rc)
        ir_mixed.append(rc if i % 2 == 0 else rd)

    gate = LearnedGate()
    info = gate.fit(
        vis_records=vis_mixed, ir_records=ir_mixed,
        vis_distances=np.asarray([scorer.score(r["feat"]) for r in vis_mixed]),
        ir_distances=np.asarray([scorer.score(r["feat"]) for r in ir_mixed]),
        constants=constants,
    )
    w_vis = [gate.predict_w_vis(rv, ri, scorer.score(rv["feat"]), scorer.score(ri["feat"]))
             for rv, ri in zip(deg_records, clean_records)]
    frac_clean = float(np.mean([w < 0.5 for w in w_vis]))
    print(f"[p3-smoke] learned gate: trained on {info['n_train_frames']} frames; "
          f"favors clean stream on {frac_clean:.0%} of frames (mean w_vis {np.mean(w_vis):.3f})")
    assert frac_clean >= 0.7, f"learned gate favored the clean stream on only {frac_clean:.0%}"
    gate.save(out_root / "learned_gate.pkl")

    # --- 6. fusion systems + ablation sweep ------------------------------------
    systems = evaluate_systems(deg_records, clean_records, scorer, constants, gate=gate)
    print("[p3-smoke] system mAP@50-95: " + ", ".join(
        f"{k}={v['map50_95']:.3f}" for k, v in systems.items()
        if isinstance(v, dict) and "map50_95" in v))
    assert systems["gated_fusion"]["map50_95"] > systems["visible_only"]["map50_95"], (
        "gated fusion failed to beat the degraded single stream")
    assert "learned_gate_fusion" in systems

    rows = ablate_gate_rules(deg_records, clean_records, scorer, constants)
    assert len(rows) == 6, f"ablation sweep produced {len(rows)} rows, expected 6"
    (out_root / "gate_ablation_smoke.md").write_text(format_ablation_table(rows) + "\n", encoding="utf-8")
    print(format_ablation_table(rows))

    print("\nPHASE3 SMOKE OK — baselines, caches, corruptions, metrics, learned gate, "
          "fusion eval + ablation all verified end-to-end")
    return 0


if __name__ == "__main__":
    try:  # Windows cp1252 consoles: degrade non-ASCII output instead of crashing
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
