"""FPS / latency measurement (plan C5 protocol).

Batch=1 end-to-end predict() timing — includes preprocessing and NMS (or its
absence, for the end-to-end variants), because that asymmetry is exactly what
Table 1 should surface. Measured once per variant (seed-invariant), on the
machine that will be reported (the training GPU). fp16 is skipped on CPU.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

from uqfusion.bench.grid import resolve_device

FPS_FIELDS = [
    "variant", "weights", "device", "half", "n_images",
    "wall_ms_per_img", "fps", "pre_ms", "inf_ms", "post_ms",
]


def measure_fps(
    cfg: dict,
    variant: str,
    weights: str | Path,
    images: list[Path],
    half: bool = False,
    n_frames: int | None = None,
    warmup: int | None = None,
) -> dict:
    import torch
    from ultralytics import YOLO

    b = cfg["benchmark"]
    n_frames = n_frames or b.get("fps_frames", 500)
    warmup = warmup if warmup is not None else b.get("fps_warmup", 50)
    device = resolve_device(cfg)
    on_cpu = (device == "cpu") or (device is None and not torch.cuda.is_available())
    if half and on_cpu:
        raise ValueError("fp16 timing requested on CPU — skip half on CPU")
    if not images:
        raise ValueError("no images supplied for FPS measurement")

    model = YOLO(str(weights))
    # ultralytics 8.4.x deprecates predict's `half` in favour of `quantize`; `half`
    # still works on the pinned 8.4.90 — revisit here if the pin is ever bumped.
    kwargs = dict(imgsz=b["imgsz"], device=device, half=half, verbose=False)

    for img in (images * ((warmup // len(images)) + 1))[:warmup]:
        model.predict(str(img), **kwargs)

    sample = (images * ((n_frames // len(images)) + 1))[:n_frames]
    speeds = {"preprocess": 0.0, "inference": 0.0, "postprocess": 0.0}
    t0 = time.perf_counter()
    for img in sample:
        result = model.predict(str(img), **kwargs)[0]
        for k in speeds:
            speeds[k] += float(result.speed.get(k, 0.0))
    wall = time.perf_counter() - t0

    n = len(sample)
    return {
        "variant": variant, "weights": str(weights),
        "device": "cpu" if on_cpu else str(device if device is not None else "cuda:auto"),
        "half": half, "n_images": n,
        "wall_ms_per_img": round(wall / n * 1000, 2),
        "fps": round(n / wall, 1),
        "pre_ms": round(speeds["preprocess"] / n, 2),
        "inf_ms": round(speeds["inference"] / n, 2),
        "post_ms": round(speeds["postprocess"] / n, 2),
    }


def write_fps_csv(rows: list[dict], out_csv: str | Path) -> Path:
    out_csv = Path(out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(out_csv, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FPS_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    return out_csv
