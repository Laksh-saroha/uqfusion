"""Per-class AP for IR checkpoints (handoff D-3).

mAP is macro-averaged over ship and buoy, so a buoy AP near zero means the headline
number is roughly half ship AP -- and every delta in the screen is then a ship-AP
delta wearing a disguise. One val pass per checkpoint answers it.

Usage:  python scripts/perclass_ap.py runs/screen1/s1_base_seed0 [more run dirs...]
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import uqfusion.uq.gaussian  # noqa: F401,E402  registers safe globals for the unpickle
from ultralytics import YOLO  # noqa: E402
from ultralytics.utils import LOGGER  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--data", default="runs/derived/data_ir_stride2.yaml")
    ap.add_argument("--batch", type=int, default=8)
    args = ap.parse_args()
    LOGGER.setLevel(logging.ERROR)

    for rd in args.runs:
        d = ROOT / rd
        import yaml
        imgsz = int(yaml.safe_load((d / "args.yaml").read_text())["imgsz"])
        r = YOLO(str(d / "weights" / "best.pt")).val(
            data=str(ROOT / args.data), imgsz=imgsz, batch=args.batch,
            verbose=False, plots=False, save_json=False)
        m = r.box.maps
        print(f"RESULT {d.name:18s} imgsz {imgsz:4d}  all {r.box.map:.5f} | "
              f"ship {m[0]:.5f} | buoy {m[1]:.5f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
