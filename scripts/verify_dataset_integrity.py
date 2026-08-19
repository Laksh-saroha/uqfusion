"""Prove the source dataset was never written to.

Every analysis script in this repo is supposed to treat `Pohang_dataset/` as
read-only — caches, derived lists and evaluation outputs all go to `runs/`. This
turns that convention into something checkable.

    python scripts/verify_dataset_integrity.py --record   # once, before work
    python scripts/verify_dataset_integrity.py            # after, to verify

Checked, over image files only (label `.cache` files are rewritten by Ultralytics
as a normal side effect of reading a split, so they are deliberately excluded):

  * file count
  * total bytes
  * sha256 over the sorted (path | size) listing — catches add, delete, rename
    and any size-changing edit
  * newest mtime must not have advanced past the recorded baseline — catches a
    same-size overwrite, which the hash alone would miss

Exit code is 1 on any mismatch, so this can gate a pipeline.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import sys
from pathlib import Path

IMG_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def scan(root: str) -> dict:
    newest, newest_p, n, total = 0.0, None, 0, 0
    h = hashlib.sha256()
    for dirpath, _, filenames in sorted(os.walk(root)):
        for f in sorted(filenames):
            if os.path.splitext(f)[1].lower() not in IMG_EXT:
                continue
            p = os.path.join(dirpath, f)
            st = os.stat(p)
            n += 1
            total += st.st_size
            h.update(f"{p}|{st.st_size}\n".encode())
            if st.st_mtime > newest:
                newest, newest_p = st.st_mtime, p
    return {"n_images": n, "bytes": total, "path_size_sha256": h.hexdigest(),
            "newest_mtime": newest, "newest_path": newest_p,
            "recorded_at": dt.datetime.now().isoformat()}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="Pohang_dataset")
    parser.add_argument("--baseline", default="runs/derived/dataset_integrity_baseline.json")
    parser.add_argument("--record", action="store_true", help="write a new baseline instead of checking")
    args = parser.parse_args()

    cur = scan(args.root)
    bl_path = Path(args.baseline)

    if args.record or not bl_path.is_file():
        bl_path.parent.mkdir(parents=True, exist_ok=True)
        bl_path.write_text(json.dumps(cur, indent=2), encoding="utf-8")
        print(f"[integrity] baseline recorded: {cur['n_images']:,} images, "
              f"{cur['bytes'] / 1073741824:.2f} GB -> {bl_path}")
        return 0

    base = json.loads(bl_path.read_text(encoding="utf-8"))
    checks = [
        ("image count", cur["n_images"] == base["n_images"], f"{cur['n_images']:,} vs {base['n_images']:,}"),
        ("total bytes", cur["bytes"] == base["bytes"], f"{cur['bytes']} vs {base['bytes']}"),
        ("paths+sizes sha256", cur["path_size_sha256"] == base["path_size_sha256"], ""),
        ("newest mtime", cur["newest_mtime"] <= base["newest_mtime"],
         str(dt.datetime.fromtimestamp(cur["newest_mtime"]))),
    ]
    for name, ok, detail in checks:
        print(f"[integrity] {name:22s} {'OK' if ok else 'CHANGED'}   {detail}")

    if all(ok for _, ok, _ in checks):
        print(f"\n[integrity] VERDICT: {args.root} untouched since "
              f"{base['recorded_at'].split('.')[0]}")
        return 0
    print(f"\n[integrity] VERDICT: *** {args.root} MODIFIED *** — investigate before trusting any result")
    return 1


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
