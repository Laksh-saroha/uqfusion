"""Synthetic heteroscedastic data for the §18-2 tiny-subset gate (no real data).

Two image populations with a VISIBLE cue tied to label quality:
  clean: sharp rectangles, exact labels          -> σ should stay low
  noisy: blurred image + jittered box labels     -> σ should grow

Aleatoric uncertainty is only learnable when the noise level is predictable
from the input — the blur is that cue. This is the strongest σ-sanity check
available on CPU: not just "σ is non-degenerate" but "σ tracks the injected
noise" (scope §6.2's harness requirement, pre-empting the R10 failure mode).

Also provides `degrade_image` (fog-ish blur+noise) for the pipeline smoke's
simulated sensor degradation.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml


def make_hetero_dataset(
    root: Path,
    n_train: int = 48,
    n_val: int = 16,
    imgsz: int = 320,
    noisy_fraction: float = 0.5,
    label_jitter: float = 0.15,
    blur_kernel: int = 9,
    seed: int = 0,
) -> tuple[Path, dict]:
    """Build the dataset; returns (data_yaml_path, noise_map {split/stem: level})."""
    import cv2

    rng = np.random.default_rng(seed)
    root = Path(root)
    noise_map: dict[str, str] = {}

    for split, count in (("train", n_train), ("val", n_val)):
        img_dir = root / "images" / split
        lbl_dir = root / "labels" / split
        img_dir.mkdir(parents=True, exist_ok=True)
        lbl_dir.mkdir(parents=True, exist_ok=True)
        for i in range(count):
            noisy = rng.random() < noisy_fraction
            img = rng.integers(0, 60, size=(imgsz, imgsz, 3), dtype=np.uint8)
            labels = []
            for cls in rng.choice([0, 1], size=rng.integers(1, 3), replace=True):
                bw = int(rng.integers(imgsz // 6, imgsz // 3))
                bh = int(rng.integers(imgsz // 6, imgsz // 3))
                x0 = int(rng.integers(0, imgsz - bw))
                y0 = int(rng.integers(0, imgsz - bh))
                intensity = 230 if cls == 0 else 140
                cv2.rectangle(img, (x0, y0), (x0 + bw, y0 + bh), (intensity,) * 3, thickness=-1)

                cx, cy, w, h = (x0 + bw / 2), (y0 + bh / 2), float(bw), float(bh)
                if noisy:  # jitter the LABEL, not the pixels — heteroscedastic annotation noise
                    cx += rng.uniform(-label_jitter, label_jitter) * w
                    cy += rng.uniform(-label_jitter, label_jitter) * h
                    w *= 1.0 + rng.uniform(-label_jitter, label_jitter)
                    h *= 1.0 + rng.uniform(-label_jitter, label_jitter)
                labels.append(
                    f"{cls} {np.clip(cx / imgsz, 0, 1):.6f} {np.clip(cy / imgsz, 0, 1):.6f} "
                    f"{np.clip(w / imgsz, 0.01, 1):.6f} {np.clip(h / imgsz, 0.01, 1):.6f}"
                )
            if noisy:  # the visible cue σ can condition on
                img = cv2.GaussianBlur(img, (blur_kernel, blur_kernel), 0)

            stem = f"{i:06d}"
            cv2.imwrite(str(img_dir / f"{stem}.jpg"), img)
            (lbl_dir / f"{stem}.txt").write_text("\n".join(labels) + "\n", encoding="utf-8")
            noise_map[f"{split}/{stem}"] = "noisy" if noisy else "clean"

    data_yaml = root / "hetero_data.yaml"
    with open(data_yaml, "w", encoding="utf-8") as f:
        yaml.safe_dump(
            {"path": str(root), "train": "images/train", "val": "images/val",
             "names": {0: "ship", 1: "buoy"}},
            f, sort_keys=False,
        )
    (root / "noise_map.json").write_text(json.dumps(noise_map, indent=1), encoding="utf-8")
    return data_yaml, noise_map


def degrade_image(image: np.ndarray, blur_kernel: int = 21, noise_std: float = 25.0, seed: int = 0) -> np.ndarray:
    """Fog-ish degradation for the pipeline smoke: heavy blur + additive noise + washout."""
    import cv2

    rng = np.random.default_rng(seed)
    out = cv2.GaussianBlur(image, (blur_kernel, blur_kernel), 0).astype(np.float32)
    out = 0.6 * out + 0.4 * 180.0  # wash toward fog gray
    out += rng.normal(0.0, noise_std, size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)
