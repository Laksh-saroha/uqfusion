"""Build a ship-only view of the IR tree (plan B5) without touching the source data.

IR buoy AP is **0.0004** while the detector emits tens of thousands of buoy boxes
against 596 GT boxes in val — capacity spent on a class it cannot see. The screen
measured why: buoys are 20:1 rarer than ships in IR training, and the val buoys
are 3.7 px wide against 10 px in train, so the class is both starved and shifted.
B5 asks whether dropping it lifts ship AP.

Construction, and the reason for each choice:

* **Images are hardlinked, never copied.** Same NTFS volume, so this costs inodes
  and no bytes, and the pixels are provably identical to the source rather than
  re-encoded.
* **Output lives under `runs/derived/`, not `Pohang_dataset/`.** The source tree is
  the archive of record and `verify_dataset_integrity.py` holds a baseline of its
  image count and hash; adding a sibling tree there would move that baseline for a
  reason that has nothing to do with data integrity.
* **Ultralytics finds labels by string-replacing `/images/` with `/labels/`**, so
  the mirror has to keep that shape. It does, and the yaml lists absolute image
  paths.
* **Frames whose only boxes were buoys are KEPT, with an empty label file.** They
  are background frames, not missing data; deleting them would confound "drop the
  buoy class" with "drop the frames buoys appear in".

Usage:
    python scripts/prep_ir_shiponly.py [--dst runs/derived/ir_shiponly]
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SRC = ROOT / "Pohang_dataset" / "infrared"
SHIP = 0


def link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:          # different volume, or link limit — fall back to a copy
        import shutil
        shutil.copy2(src, dst)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dst", default="runs/derived/ir_shiponly")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    dst = ROOT / args.dst
    stats = {}
    lists = {}
    for split in args.splits:
        list_file = SRC / f"{split}.txt"
        if not list_file.is_file():
            print(f"[shiponly] no {list_file}, skipping")
            continue
        rows = [ln.strip() for ln in list_file.read_text(encoding="utf-8").splitlines() if ln.strip()]
        kept_boxes = dropped_boxes = emptied = 0
        out_paths = []
        for rel in rows:
            src_img = Path(rel) if Path(rel).is_absolute() else (SRC / rel)
            src_img = src_img.resolve()
            parts = list(src_img.parts)
            i = len(parts) - 1 - parts[::-1].index("images")
            run = parts[i + 1] if i + 2 < len(parts) else ""
            dst_img = dst / "images" / run / src_img.name
            link(src_img, dst_img)
            out_paths.append(str(dst_img))

            lbl_parts = list(parts)
            lbl_parts[i] = "labels"
            src_lbl = Path(*lbl_parts).with_suffix(".txt")
            dst_lbl = dst / "labels" / run / (src_img.stem + ".txt")
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            keep = []
            if src_lbl.is_file():
                for line in src_lbl.read_text(encoding="utf-8").splitlines():
                    f = line.split()
                    if len(f) < 5:
                        continue
                    if int(f[0]) == SHIP:
                        keep.append(line.strip())
                        kept_boxes += 1
                    else:
                        dropped_boxes += 1
                if not keep and src_lbl.stat().st_size > 0:
                    emptied += 1
            dst_lbl.write_text(("\n".join(keep) + "\n") if keep else "", encoding="utf-8")

        lp = dst / f"{split}.txt"
        lp.write_text("\n".join(out_paths) + "\n", encoding="utf-8")
        lists[split] = lp
        stats[split] = {"frames": len(rows), "ship_boxes": kept_boxes,
                        "buoy_boxes_dropped": dropped_boxes, "frames_emptied": emptied}
        print(f"[shiponly] {split}: {len(rows)} frames, kept {kept_boxes} ship boxes, "
              f"dropped {dropped_boxes} buoy boxes, {emptied} frames became empty")

    yaml_path = ROOT / "runs" / "derived" / "data_ir_shiponly.yaml"
    body = "".join(f"{s}: {lists[s]}\n" for s in ("train", "val", "test") if s in lists)
    yaml_path.write_text(
        "# IR, buoy class removed (plan B5). Images are hardlinks to Pohang_dataset/infrared;\n"
        "# labels are the same files with class-1 rows deleted. nc=1, so mAP is ship AP\n"
        "# directly and is NOT comparable to the 2-class numbers without saying so.\n"
        + body + "names:\n  0: ship\n", encoding="utf-8")

    (dst / "stats.json").write_text(json.dumps(stats, indent=2), encoding="utf-8")
    print(f"[shiponly] wrote {yaml_path}")
    print("[shiponly] NOTE: this yaml is nc=1. Its mAP is ship AP and must not be "
          "compared to a 2-class mAP without that stated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
