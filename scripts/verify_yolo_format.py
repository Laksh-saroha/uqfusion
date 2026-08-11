"""Verify a Pohang modality is in loadable Ultralytics YOLO format.

The dataset is ALREADY YOLO format: per-run images/ and labels/ subtrees, txt
split lists, normalized `class cx cy w h` labels. This script proves it on the
current machine before a run — it does NOT modify anything.

For every image in each split it computes the label path the way Ultralytics
does (last `/images/` -> `/labels/`, extension -> `.txt`) and checks it exists.
Missing label files are allowed by YOLO (negative/background frames) but are
reported so you can eyeball whether the count is sane.

Usage:
    python scripts/verify_yolo_format.py --data vis
    python scripts/verify_yolo_format.py --data ir
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.data.lists import load_data_yaml, split_image_list


def img_to_label(img: Path) -> Path:
    """Ultralytics rule: swap the last `/images/` segment for `/labels/`, ext -> .txt."""
    s = str(img)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa not in s:
        # fall back to forward slashes (paths from txt lists may be posix-style)
        sa, sb = "/images/", "/labels/"
    head, _, tail = s.rpartition(sa)
    if not head:
        return Path(s).with_suffix(".txt")  # no /images/ segment — best effort
    return Path(head + sb + tail).with_suffix(".txt")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="vis", help="vis, ir, or a dataset yaml path")
    parser.add_argument("--show", type=int, default=5, help="how many missing labels to print per split")
    args = parser.parse_args()

    cfg = load_config(args.config)
    yaml_path = resolve_data_yaml(cfg, args.data)
    data = load_data_yaml(yaml_path)
    print(f"yaml   : {yaml_path}")
    print(f"names  : {data.get('names')}")

    ok = True
    for split in ("train", "val", "test"):
        if split not in data or data[split] is None:
            print(f"[{split}] no entry — skipped")
            continue
        imgs = split_image_list(data, split)  # raises if a split list / dir is missing
        missing_imgs = [p for p in imgs if not p.is_file()]
        with_label, missing_label = 0, []
        for p in imgs:
            lbl = img_to_label(p)
            if lbl.is_file():
                with_label += 1
            else:
                missing_label.append((p, lbl))

        print(f"\n[{split}] images listed : {len(imgs)}")
        print(f"[{split}] images on disk: {len(imgs) - len(missing_imgs)}"
              + (f"   MISSING {len(missing_imgs)}" if missing_imgs else ""))
        print(f"[{split}] with label    : {with_label}")
        print(f"[{split}] without label : {len(missing_label)}  (allowed = background frames)")

        if missing_imgs:
            ok = False
            print(f"  !! {len(missing_imgs)} listed images do not exist — this WILL break training")
            for p in missing_imgs[: args.show]:
                print(f"     missing image: {p}")
        for p, lbl in missing_label[: args.show]:
            print(f"     no label for {p.name} -> expected {lbl}")

    print("\n" + ("PASS: every listed image exists and label paths resolve."
                  if ok else "FAIL: some listed images are missing (see above)."))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
