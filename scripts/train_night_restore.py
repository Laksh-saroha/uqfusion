"""Fine-tune VIS on the restored night labels (I5).

`docs/prereg-night-label-restore.md`, committed at `030244e` before any label was
changed. `scripts/restore_night_perbox.py` put **+94,553** boxes back into
`pohang01` VIS train labels -- every box the frame-level `--cut-dark` deleted that
the per-box test did not actually flag -- and left the 38,135 genuinely flagged
ones out.

The question this run answers is **not** whether the fused benchmark moves. It
cannot: `veto_vis` fires on 100% of night frames, so the fused night number is
`ir_only` by construction. The question is whether a VIS detector trained on those
labels can see at night at all, scored on the 2,068 night val frames (16,179 GT
boxes, never filtered) where VIS currently recovers 0.0000.

Pre-registered bands: **DEAD** < 0.005, **WEAK** 0.005-0.02, **ALIVE** >= 0.02.
Plus a guard: day VIS-only must not regress past `-max(2*sd_paired, 0.002)`.

Settings are the shipped run's, so the only difference is the labels:
`yolo26m`, `data_vis_stride2.yaml`, `imgsz` 640, `batch` 16. Fine-tunes from the
shipped `best.pt` rather than COCO, at 25 epochs with patience 10.

**Writes to a NEW directory.** `runs/full_scale/gauss_vis_seed0/` is not touched,
so every published `crossmodal26m` number keeps reproducing from it.

The caveat the pre-registration records: this starts from a checkpoint trained on
EMPTY night labels, so the initialisation already encodes "night frames contain
nothing". ALIVE would be strong evidence; DEAD is provisional and cannot separate
"night VIS is blind" from "25 epochs could not undo the initialisation".

Usage:
    python scripts/train_night_restore.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.config import load_config                      # noqa: E402
from uqfusion.uq.train_gaussian import train_gaussian        # noqa: E402

SHIPPED = ROOT / "runs/full_scale/gauss_vis_seed0/weights/best.pt"
DATA = ROOT / "runs/derived/data_vis_stride2.yaml"
RUN_NAME = "gauss_vis_nightrestore"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--resume", action="store_true",
                    help="continue this run from its own weights/last.pt")
    args = ap.parse_args()

    assert SHIPPED.is_file(), f"missing shipped weights {SHIPPED}"
    assert DATA.is_file(), f"missing data yaml {DATA}"
    out = ROOT / "runs/full_scale" / RUN_NAME
    if args.resume:
        # Resume loads the optimizer, EMA and epoch counter from last.pt, so the
        # start weights are that checkpoint -- NOT the shipped best.pt. Passing
        # both would be ambiguous about which run is being continued.
        last = out / "weights" / "last.pt"
        assert last.is_file(), f"--resume but no {last}"
        print(f"[night-restore] RESUMING from {last.relative_to(ROOT)}", flush=True)
    else:
        assert not (out / "weights" / "best.pt").is_file(), \
            f"{out} already holds weights -- refusing to overwrite a finished run"

    cfg = load_config()
    t0 = time.time()
    print(f"[night-restore] fine-tuning from {SHIPPED.relative_to(ROOT)}", flush=True)
    print(f"[night-restore] data {DATA.relative_to(ROOT)}  imgsz {args.imgsz} "
          f"batch {args.batch} epochs {args.epochs} patience {args.patience}",
          flush=True)

    best, run_dir = train_gaussian(
        cfg, str(DATA), variant="yolo26m", seed=0,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        run_name=RUN_NAME, out_subdir="full_scale",
        resume=args.resume, weights=None if args.resume else str(SHIPPED),
        train_overrides={"patience": args.patience},
    )
    print(f"[night-restore] best {best}\n[night-restore] dir {run_dir}\n"
          f"[night-restore] {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
