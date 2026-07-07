"""Environment smoke test — run after every fresh install (dev machine or GPU server).

Verifies:
  1. every pinned dependency imports,
  2. CUDA visibility (informational — expected False on the dev machine),
  3. every Table 1 benchmark variant in config.yaml resolves in the installed
     ultralytics version and builds on CPU (architecture only, no weight download).

Usage:  python scripts/smoke_env.py
"""

import importlib
import sys

MODULES = [
    "torch", "torchvision", "ultralytics", "albumentations", "ensemble_boxes",
    "pytorch_ood", "netcal", "torch_uncertainty", "sklearn", "torchmetrics",
    "scipy", "yaml", "uqfusion",
]


def main() -> int:
    failures: list[str] = []

    for mod in MODULES:
        try:
            m = importlib.import_module(mod)
            print(f"  import {mod:18s} OK ({getattr(m, '__version__', '?')})")
        except Exception as e:  # noqa: BLE001 - report every failure, don't stop at the first
            failures.append(f"import {mod}: {e}")
            print(f"  import {mod:18s} FAILED: {e}")

    try:
        import torch

        if torch.cuda.is_available():
            print(f"  CUDA available: True ({torch.cuda.get_device_name(0)})")
        else:
            print("  CUDA available: False   [expected on the dev machine; must be True on the server]")
    except Exception as e:  # noqa: BLE001
        failures.append(f"cuda check: {e}")

    try:
        from ultralytics import YOLO

        from uqfusion.config import load_config

        cfg = load_config()
        for variant in cfg["benchmark"]["variants"]:
            try:
                model = YOLO(f"{variant}.yaml")  # architecture only — no weight download
                n_params = sum(p.numel() for p in model.model.parameters())
                print(f"  variant {variant:10s} builds OK ({n_params / 1e6:.1f}M params)")
            except Exception as e:  # noqa: BLE001
                failures.append(f"variant {variant}: {e}")
                print(f"  variant {variant:10s} FAILED: {e}")
    except Exception as e:  # noqa: BLE001
        failures.append(f"variant check setup: {e}")

    if failures:
        print(f"\nSMOKE FAILED ({len(failures)}):")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nSMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
