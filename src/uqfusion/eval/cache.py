"""Prediction caches (plan B5-2): run every model over the eval frames ONCE,
then all calibration metrics and every gate-level ablation are CPU
post-processing over the cached records — no retraining, no re-inference.

Format: one pickle per (source, split, condition): {"meta": {...}, "records":
[record, ...]} where records follow the uqfusion.uq.infer schema plus
"image_path". Meta stamps weights, thresholds, corruption, and git commit so a
cache can always be traced to what produced it (scope §11.1).
"""

from __future__ import annotations

import pickle
from pathlib import Path

from uqfusion.bench.grid import _git_commit


def build_cache(
    predictor,
    images: list,
    out_path: str | Path,
    meta: dict | None = None,
    transform=None,
    log_every: int = 50,
) -> Path:
    """Run `predictor` over `images` (optionally through `transform(im_bgr, index)`,
    e.g. a corruption) and pickle the records."""
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, img in enumerate(images):
        if transform is None:
            rec = predictor(img)
        else:
            im = cv2.imread(str(img))
            if im is None:
                raise FileNotFoundError(f"could not read image: {img}")
            rec = predictor(transform(im, i))
        rec["image_path"] = str(img)
        records.append(rec)
        if log_every and (i + 1) % log_every == 0:
            print(f"[cache] {i + 1}/{len(images)} frames")

    payload = {"meta": {**(meta or {}), "n_frames": len(records), "git_commit": _git_commit()},
               "records": records}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[cache] {len(records)} records -> {out_path}")
    return out_path


def load_cache(path: str | Path) -> tuple[list[dict], dict]:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    return payload["records"], payload["meta"]
