"""§18-2 tiny-subset gate for the Gaussian σ² head — MUST pass before GPU time.

Trains the σ-augmented smoke variant on synthetic heteroscedastic data (CPU,
minutes), then verifies with hard asserts:
  1. training completes; all 4 loss columns finite; warm-up epochs log nll ~ 0
     (proving the σ-freeze schedule actually engages);
  2. UQ inference returns per-detection σ that is finite, positive, and
     NON-DEGENERATE (σ varies across detections, not one collapsed value);
and reports (soft, printed): mean size-normalized uncertainty on noisy-cue
val images vs clean ones — the "σ tracks injected noise" direction check.

Usage:  python scripts/smoke_gaussian.py
"""

from __future__ import annotations

import csv
import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.uq.infer import UQPredictor
from uqfusion.uq.reliability import per_box_uncertainty
from uqfusion.uq.synthetic import make_hetero_dataset
from uqfusion.uq.train_gaussian import train_gaussian

SMOKE_GAUSSIAN_OVERRIDES = {"warmup_epochs": 2, "ramp_epochs": 2}


def train_smoke_gaussian(
    cfg: dict,
    fresh: bool = True,
    variant: str | None = None,
    run_name: str = "smoke_gauss",
    root_name: str = "smoke_gaussian",
):
    """Build the hetero dataset and train the tiny Gaussian model. Shared with the
    pipeline smoke and the end2end gate. Returns (best_weights, run_dir, data_root, noise_map)."""
    s = cfg["smoke"]
    root = Path(cfg["paths"]["outputs_root"]) / root_name
    data_root = root / "hetero_data"
    if fresh and root.exists():
        shutil.rmtree(root)

    data_yaml, noise_map = make_hetero_dataset(
        data_root, n_train=s.get("hetero_images", 48), n_val=16, imgsz=s["imgsz"]
    )
    best, run_dir = train_gaussian(
        cfg,
        data_yaml,
        variant=variant or s["variant"],
        epochs=s.get("gaussian_epochs", 10),
        imgsz=s["imgsz"],
        batch=s["batch"],
        workers=0,
        run_name=run_name,
        gaussian_overrides=SMOKE_GAUSSIAN_OVERRIDES,
    )
    return best, run_dir, data_root, noise_map


def check_loss_csv(run_dir: Path, warmup_epochs: int) -> None:
    results_csv = run_dir / "results.csv"
    assert results_csv.is_file(), f"missing {results_csv}"
    with open(results_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert rows, "results.csv is empty"
    nll_col = next((k for k in rows[0] if k.strip().endswith("train/nll_loss")), None)
    assert nll_col, f"no train/nll_loss column in {list(rows[0])}"

    nll = [float(r[nll_col]) for r in rows]
    loss_cols = [k for k in rows[0] if "loss" in k]
    for k in loss_cols:
        vals = [float(r[k]) for r in rows]
        assert all(math.isfinite(v) for v in vals), f"non-finite values in {k}: {vals}"

    warm = nll[:warmup_epochs]
    assert all(abs(v) < 1e-6 for v in warm), f"warm-up epochs should log nll=0, got {warm}"
    active = nll[warmup_epochs:]
    assert any(abs(v) > 1e-9 for v in active), f"NLL never activated after warm-up: {active}"
    print(f"[gauss-smoke] loss columns OK — nll per epoch: {[round(v, 4) for v in nll]}")


def check_sigma(run_dir: Path, best: Path, data_root: Path, noise_map: dict, imgsz: int) -> None:
    predictor = UQPredictor(best, device="cpu", imgsz=imgsz, conf=0.15)
    u_clean, u_noisy, all_sigma, dfl_all = [], [], [], []
    for img in sorted((data_root / "images" / "val").glob("*.jpg")):
        rec = predictor(img)
        if rec["boxes_xyxy"].shape[0] == 0:
            continue
        u = per_box_uncertainty(rec["sigma_ltrb"], rec["boxes_xyxy"])
        all_sigma.extend(rec["sigma_ltrb"].ravel().tolist())
        if "dfl_sigma_ltrb" in rec:
            dfl_all.extend(rec["dfl_sigma_ltrb"].ravel().tolist())
        (u_noisy if noise_map[f"val/{img.stem}"] == "noisy" else u_clean).extend(u.tolist())

    sigma = np.asarray(all_sigma)
    assert sigma.size >= 8, f"too few detections on the val set ({sigma.size // 4}) to judge σ"
    assert np.isfinite(sigma).all() and (sigma > 0).all(), "σ contains non-finite or non-positive values"
    cv = float(sigma.std() / max(sigma.mean(), 1e-9))
    assert cv > 0.01, f"σ is degenerate (coefficient of variation {cv:.4f} — one collapsed value)"
    print(f"[gauss-smoke] σ over {sigma.size // 4} detections: mean {sigma.mean():.2f}px, CV {cv:.3f} — non-degenerate")

    if u_clean and u_noisy:
        mc, mn = float(np.mean(u_clean)), float(np.mean(u_noisy))
        verdict = "PASS (direction correct)" if mn > mc else "WARN (direction not yet learned at toy scale)"
        print(f"[gauss-smoke] mean u — clean-cue: {mc:.4f} vs noisy-cue: {mn:.4f} -> {verdict}")
    else:
        print(f"[gauss-smoke] WARN: only one cue population detected "
              f"(clean n={len(u_clean)}, noisy n={len(u_noisy)}) — direction check skipped")

    if dfl_all:
        d = np.asarray(dfl_all)
        assert np.isfinite(d).all() and (d >= 0).all(), "DFL-derived σ contains non-finite/negative values"
        print(f"[gauss-smoke] DFL-derived σ (§7.2 ablation row) present: mean {d.mean():.2f}px, "
              f"CV {d.std() / max(d.mean(), 1e-9):.3f}")


def main() -> int:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    best, run_dir, data_root, noise_map = train_smoke_gaussian(cfg, fresh=True)
    assert Path(best).is_file(), f"no weights produced at {best}"
    check_loss_csv(run_dir, SMOKE_GAUSSIAN_OVERRIDES["warmup_epochs"])
    check_sigma(run_dir, best, data_root, noise_map, cfg["smoke"]["imgsz"])
    (Path(run_dir) / "noise_map.json").write_text(json.dumps(noise_map, indent=1), encoding="utf-8")
    print("\nGAUSSIAN SMOKE OK — head trains, warm-up engages, σ non-degenerate (scope §18-2 gate)")
    return 0


if __name__ == "__main__":
    try:  # Windows cp1252 consoles: degrade non-ASCII output instead of crashing
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
