"""Stride subsets of the full-resolution tree, for rect training (plan B1).

The prepared 640 trees are letterboxed squares, and the padding is not free:
VIS is 640x338 of a 640x640 canvas (**53% content**) and IR is 640x512
(**80%**, i.e. 128 constant rows of value 114 — measured, not assumed). At the
same pixel budget a 2048x1080-aspect rectangle is 881x465, so rect training buys
~1.38x linear resolution on VIS for identical compute. IR's gain is only ~1.12x,
which is why VIS is the one worth the GPU time and IR is the cheap confirmation.

This writes absolute-path image lists at the same strides the 640 experiments
used, so the only variable against those runs is the canvas:

    VIS stride 5  -> matches the Phase 2 VIS training set
    IR  stride 2  -> matches the screen's IR training set

Absolute paths are deliberate. The tree's own lists are relative ("./images/..."),
and a data yaml without a `path:` key resolves them against the YAML's directory
instead of the tree — which silently points the loader somewhere that does not
exist. The derived lists under `runs/derived/` already store absolute paths for
the same reason.

Usage:
    python scripts/prep_fullres_lists.py [--root D:/Datasets/Pohang_dataset_full]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "derived"

MODALITIES = {
    "vis": {"dir": "visible", "stride": 5, "expect_hw": (1080, 2048)},
    "ir": {"dir": "infrared", "stride": 2, "expect_hw": (512, 640)},
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="D:/Datasets/Pohang_dataset_full")
    ap.add_argument("--modalities", nargs="+", default=list(MODALITIES))
    args = ap.parse_args()

    root = Path(args.root)
    if not root.is_dir():
        raise SystemExit(f"full-resolution tree not found: {root}")

    summary = {}
    for mod in args.modalities:
        spec = MODALITIES[mod]
        tree = root / spec["dir"]
        stride = spec["stride"]

        # Sanity-check the native size once per modality: rect training is a claim
        # ABOUT the aspect ratio, so a tree that is not the size we think it is
        # would invalidate the whole arm silently.
        first = None
        counts = {}
        lists = {}
        for split in ("train", "val", "test"):
            lf = tree / f"{split}.txt"
            if not lf.is_file():
                continue
            rows = [ln.strip() for ln in lf.read_text(encoding="utf-8").splitlines() if ln.strip()]
            picked = rows[::stride] if split == "train" else rows
            abs_paths = []
            for rel in picked:
                p = Path(rel) if Path(rel).is_absolute() else (tree / rel)
                abs_paths.append(str(p.resolve()))
            if first is None and abs_paths:
                from PIL import Image
                with Image.open(abs_paths[0]) as im:
                    first = (im.size[1], im.size[0])
                if first != tuple(spec["expect_hw"]):
                    raise SystemExit(f"{mod}: native size {first[1]}x{first[0]} is not the "
                                     f"expected {spec['expect_hw'][1]}x{spec['expect_hw'][0]} — "
                                     f"the rect arm's aspect argument does not hold")
            name = f"fullres_{mod}_{split}_stride{stride}.txt" if split == "train" \
                else f"fullres_{mod}_{split}.txt"
            lp = OUT / name
            lp.write_text("\n".join(abs_paths) + "\n", encoding="utf-8")
            lists[split] = lp
            counts[split] = len(abs_paths)
            print(f"[fullres] {mod}/{split}: {len(rows)} -> {len(abs_paths)} "
                  f"(stride {stride if split == 'train' else 1}) -> {lp.name}")

        yaml_path = OUT / f"data_fullres_{mod}.yaml"
        body = "".join(f"{s}: {lists[s]}\n" for s in ("train", "val", "test") if s in lists)
        h, w = first if first else (0, 0)
        yaml_path.write_text(
            f"# Full-resolution {mod} ({w}x{h} native), train at stride {stride}.\n"
            f"# For rect training: set imgsz to the LONG side and rect=True, so batches\n"
            f"# letterbox to the frame's own aspect instead of a square.\n"
            + body + "names:\n  0: ship\n  1: buoy\n", encoding="utf-8")
        summary[mod] = {"native_hw": list(first) if first else None,
                        "stride": stride, "counts": counts, "yaml": str(yaml_path)}
        print(f"[fullres] wrote {yaml_path}")

    (OUT / "fullres_lists.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
