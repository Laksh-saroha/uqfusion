"""Package the restored pohang01 VIS TRAIN labels for upload to dgxanode01.

Why this exists. The server has **zero** `.pre_visfilter` backups and does not carry
`restore_night_perbox.py`, so it cannot reconstruct the restored labels locally. The
laptop tree is already verified in the restored state, so the labels themselves ship
rather than the recipe.

Scope, deliberately narrow: pohang01 **train-split only**. `val`/`test` labels were
never touched by the filter and are not shipped -- overwriting a val label from
another machine is exactly the kind of silent divergence this project keeps paying for.

Verification travels with the payload. `manifest.json` carries the pohang01-train
content hash computed the same way `filter_night_boxes.label_content_hash` does, and
`verify_night_labels.py` recomputes it on the far side WITHOUT importing uqfusion or
reading config.yaml -- both of which are broken on that box.

Usage:
    python scripts/build_night_label_upload.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.config import load_config, resolve_data_yaml     # noqa: E402
from uqfusion.data.lists import load_data_yaml, split_image_list  # noqa: E402

NIGHT_RUN = "pohang01"

VERIFIER = '''"""Verify the uploaded pohang01 train labels, standalone.

Imports nothing from uqfusion and does not read config.yaml -- both are broken on
dgxanode01. Mirrors filter_night_boxes.label_content_hash byte for byte.

    python verify_night_labels.py /workspace/pohang/visible/labels/pohang01
"""
import hashlib, json, sys
from pathlib import Path

d = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
man = json.loads((Path(__file__).parent / "manifest.json").read_text())
names = man["files"]
missing = [n for n in names if not (d / n).is_file()]
if missing:
    print(f"FAIL: {len(missing)} of {len(names)} label files missing, e.g. {missing[:3]}")
    raise SystemExit(1)
h = hashlib.sha256()
for n in names:                      # manifest order IS sorted-by-image order
    h.update(f"{man['run']}/{n}\\n".encode())
    h.update((d / n).read_bytes())
    h.update(b"\\x00")
got = h.hexdigest()[:12]
ok = got == man["pohang01_train_label_hash"]
print(f"files      : {len(names)}")
print(f"boxes      : {sum(1 for n in names for l in (d/n).read_text().splitlines() if l.strip())}")
print(f"hash       : {got}  expected {man['pohang01_train_label_hash']}")
print("PASS" if ok else "FAIL -- do not train on this tree")
raise SystemExit(0 if ok else 1)
'''


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/upload/night_labels_pohang01.zip")
    args = ap.parse_args()

    cfg = load_config()
    data = load_data_yaml(resolve_data_yaml(cfg, "vis"))
    imgs = sorted(p for p in split_image_list(data, "train") if NIGHT_RUN in str(p))
    if not imgs:
        print(f"[fail] no {NIGHT_RUN} images in the VIS train split")
        return 1

    labels, h, boxes, missing = [], hashlib.sha256(), 0, 0
    for img in imgs:
        lp = Path(str(img).replace("images", "labels")).with_suffix(".txt")
        h.update(f"{NIGHT_RUN}/{lp.name}\n".encode())
        if lp.is_file():
            b = lp.read_bytes()
            h.update(b)
            boxes += sum(1 for ln in b.decode().splitlines() if ln.strip())
            labels.append(lp)
        else:
            missing += 1
        h.update(b"\x00")
    digest = h.hexdigest()[:12]

    manifest = {
        "run": NIGHT_RUN,
        "split": "train",
        "note": "Restored night labels from the laptop, per docs/prereg-night-label-restore.md. "
                "val/test are NOT included and must not be overwritten.",
        "n_label_files": len(labels),
        "n_images_in_split": len(imgs),
        "n_labels_absent": missing,
        "n_boxes": boxes,
        "pohang01_train_label_hash": digest,
        "all_vis_label_hash": "b92739202127",
        "vis_train_split_label_hash": "8ed69b5974ed",
        "files": [lp.name for lp in labels],
    }

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        print(f"[fail] {out} exists -- pass a different --out, nothing is overwritten")
        return 1
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
        for lp in labels:
            z.write(lp, arcname=lp.name)
        z.writestr("manifest.json", json.dumps(manifest, indent=1))
        z.writestr("verify_night_labels.py", VERIFIER)

    mb = out.stat().st_size / 1e6
    print(f"[zip] {out}")
    print(f"[zip] {len(labels)} label files, {boxes} boxes, {missing} absent")
    print(f"[zip] pohang01 train hash {digest}   size {mb:.1f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
