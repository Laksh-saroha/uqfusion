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
    "variant", "seed", "run_id", "weights", "device", "half", "n_images",
    "wall_ms_per_img", "fps", "pre_ms", "inf_ms", "post_ms",
    # A laptop GPU throttles over a long sweep, and drift of that kind aliases onto
    # whichever variant happened to be measured late. Stamping time and thermal state
    # per measurement makes it separable after the fact instead of invisible.
    "timestamp", "gpu_temp_c", "gpu_clock_mhz",
]


def gpu_state() -> dict:
    """Current temperature and graphics clock, or blanks if nvidia-smi is unavailable."""
    import subprocess

    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,clocks.current.graphics",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10, check=True).stdout.strip().splitlines()[0]
        temp, clock = (x.strip() for x in out.split(","))
        return {"gpu_temp_c": temp, "gpu_clock_mhz": clock}
    except Exception:  # noqa: BLE001 - telemetry is never worth failing a measurement over
        return {"gpu_temp_c": "", "gpu_clock_mhz": ""}


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
    # `half=` still works on the pinned 8.4.90 but warns on every call, which means
    # one warning per timed frame. `quantize` is the canonical form it forwards to
    # (cfg/__init__.py maps half->quantize=16, and None means fp32); verified
    # equivalent by checking AutoBackend.fp16 and the parameter dtype.
    kwargs = dict(imgsz=b["imgsz"], device=device, quantize=16 if half else None,
                  verbose=False)

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
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"), **gpu_state(),
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
        writer = csv.DictWriter(f, fieldnames=FPS_FIELDS, restval="")
        writer.writeheader()
        writer.writerows(rows)
    return out_csv
