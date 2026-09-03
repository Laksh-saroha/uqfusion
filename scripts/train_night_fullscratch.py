"""Cold-start the VIS detector on the RESTORED night labels — the V1 primary arm.

Registered in `docs/prereg-night-veto.md`, committed before this run is launched.

Why from scratch. `gauss_vis_nightrestore` is a fine-tune from a checkpoint whose
training data asserted that night frames are empty. For the *restore* question that
only made the test harder, so an ALIVE verdict stood. For the *veto* question the
checkpoint is the treatment rather than the test, and deciding a switch against a
detector that still carries a "night is empty" initialisation would repeat the exact
error this project already made once.

The recipe is `gauss_vis_seed0`'s, unchanged and deliberately so — yolo26m cold from
COCO, `data_vis_stride2.yaml`, imgsz 640, batch 16, epochs 100, patience 20, seed 0.
The ONLY difference from the shipped detector is the 94,553 restored boxes. Measured
cost at that recipe on the local 4080: 49,131 s (13.65 h) to an early stop at epoch 45.

Writes ONLY `runs/full_scale/gauss_vis_nightfull/`. Nothing under `gauss_vis_seed0/`
or `gauss_vis_nightrestore/` is touched.

Usage:
    python scripts/train_night_fullscratch.py
    python scripts/train_night_fullscratch.py --resume     # continue from last.pt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.config import load_config                      # noqa: E402
from uqfusion.uq.train_gaussian import train_gaussian        # noqa: E402

DATA = ROOT / "runs/derived/data_vis_stride2.yaml"
RUN_NAME = "gauss_vis_nightfull"
PREREG = ROOT / "docs/prereg-night-veto.md"
#: The restored VIS train-label hash. The whole point of this run is the labels, so
#: starting it against the pre-restore tree would burn 14 h producing the shipped
#: detector a second time.
EXPECT_LABEL_HASH = "b92739202127"


def committed(path: Path) -> bool:
    """True if `path` is tracked AND has no uncommitted changes -- the prereg gate."""
    try:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", str(path)],
                                 cwd=ROOT, capture_output=True, text=True)
        if tracked.returncode != 0:
            return False
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", str(path)],
                               cwd=ROOT, capture_output=True, text=True)
        return dirty.returncode == 0
    except Exception:
        return False


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--patience", type=int, default=20)
    ap.add_argument("--resume", action="store_true",
                    help="continue this run from its own weights/last.pt")
    ap.add_argument("--skip-hash-check", action="store_true",
                    help="escape hatch; records itself in the log")
    args = ap.parse_args()

    assert DATA.is_file(), f"missing data yaml {DATA}"
    assert PREREG.is_file(), f"missing prereg {PREREG}"
    assert committed(PREREG), (
        f"{PREREG.relative_to(ROOT)} is untracked or has uncommitted edits -- a "
        "pre-registration that can still be edited is not one. Commit it first.")

    out = ROOT / "runs/full_scale" / RUN_NAME
    if args.resume:
        last = out / "weights" / "last.pt"
        assert last.is_file(), f"--resume but no {last}"
        print(f"[nightfull] RESUMING from {last.relative_to(ROOT)}", flush=True)
    else:
        assert not (out / "weights" / "best.pt").is_file(), \
            f"{out} already holds weights -- refusing to overwrite a finished run"

    if args.skip_hash_check:
        print("[nightfull] WARNING: label-hash check skipped by flag", flush=True)
    else:
        # Hash the FULL VIS train tree, not the stride subset -- b92739202112... is
        # recorded over `--data vis`, and a stride list would hash to something else.
        from uqfusion.config import resolve_data_yaml             # noqa: PLC0415
        from uqfusion.data.lists import load_data_yaml, split_image_list  # noqa: PLC0415
        sys.path.insert(0, str(ROOT / "scripts"))
        from filter_night_boxes import label_content_hash         # noqa: PLC0415
        full = load_data_yaml(resolve_data_yaml(load_config(), "vis"))
        h = label_content_hash(split_image_list(full, "train"))
        assert h.startswith(EXPECT_LABEL_HASH), (
            f"VIS train label hash is {h[:12]}, expected {EXPECT_LABEL_HASH} (the "
            "restored tree). Training now would reproduce the shipped detector.")
        print(f"[nightfull] label hash {h[:12]} OK -- restored tree", flush=True)

    cfg = load_config()
    t0 = time.time()
    print(f"[nightfull] COLD START from COCO yolo26m (no --weights)", flush=True)
    print(f"[nightfull] data {DATA.relative_to(ROOT)}  imgsz {args.imgsz} "
          f"batch {args.batch} epochs {args.epochs} patience {args.patience}",
          flush=True)

    best, run_dir = train_gaussian(
        cfg, str(DATA), variant="yolo26m", seed=0,
        epochs=args.epochs, imgsz=args.imgsz, batch=args.batch,
        run_name=RUN_NAME, out_subdir="full_scale",
        resume=args.resume, weights=None,      # None => COCO, i.e. the cold start
        train_overrides={"patience": args.patience},
    )
    print(f"[nightfull] best {best}\n[nightfull] dir {run_dir}\n"
          f"[nightfull] {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
