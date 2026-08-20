"""Re-export the IR tree with CLAHE — local contrast, the untried answer (plan B3).

The percentile experiment was rejected on a specific argument: min-max and
percentile are **both per-frame affine maps**, and no global remapping can fix
local contrast. CLAHE is local, which makes it the direct answer to the failure
that was actually diagnosed rather than a second attempt at the rejected one.

Pipeline, and where CLAHE sits in it:

    raw 16-bit 640x512 -> CLAHE (16-bit, tiled) -> min-max 8-bit -> letterbox 640x640

CLAHE runs on the **raw 16-bit, unpadded** frame. Both halves of that matter. On
16-bit it sees the sensor's full dynamic range instead of a range already crushed
into 256 levels. Unpadded, because the letterbox pad is a constant 114 across 20%
of the canvas and would sit inside two rows of CLAHE tiles, flattening their
histograms and bleeding a padding artefact into real pixels — the same reasoning
that made `export_ir_percentile.py` compute its percentiles pre-padding.

The 8-bit step afterwards is min-max, i.e. the ORIGINAL mapping. That is
deliberate: the percentile mapping was rejected, and keeping min-max means CLAHE
is the only difference between this tree and the one the baseline was trained on.

The verification gate is inherited from `export_ir_percentile.py` and is the point
of reusing that module rather than re-implementing it: `--verify` reproduces the
EXISTING min-max pixels through this code path at max abs diff 0. If the shared
path can reproduce the current tree exactly, then this tree differs in exactly the
one step deliberately changed.

Output goes to `runs/derived/ir_clahe/`, not `Pohang_dataset/`: the source tree is
the archive of record and `verify_dataset_integrity.py` holds a baseline over it.

Usage:
    python scripts/export_ir_clahe.py --verify        # prove the shared path first
    python scripts/export_ir_clahe.py                 # do the export
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(Path(__file__).resolve().parent))

from export_ir_percentile import (  # noqa: E402
    SRC_TREE, TARGET, letterbox, raw_path_for, to_8bit,
)

DST_TREE = ROOT / "runs" / "derived" / "ir_clahe"
CLIP_LIMIT = 2.0
TILE = (8, 8)


def clahe_16(arr: np.ndarray, clip: float, tile: tuple[int, int]) -> np.ndarray:
    import cv2

    if arr.dtype != np.uint16:
        raise ValueError(f"expected uint16, got {arr.dtype}")
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=tile).apply(arr)


def render_clahe(stem: str, clip: float, tile: tuple[int, int]) -> np.ndarray:
    with Image.open(raw_path_for(stem)) as im:
        arr = np.array(im)
    if arr.dtype != np.uint16:
        raise ValueError(f"{stem} is {arr.dtype}, expected uint16")
    return letterbox(to_8bit(clahe_16(arr, clip, tile), "minmax"))


def verify(n: int) -> int:
    """Reproduce the existing min-max tree through the SHARED code path.

    This does not test CLAHE. It tests that everything around CLAHE — the raw
    path resolution, the 16->8 bit map, the letterbox — is byte-identical to what
    produced the tree the baseline was trained on. Only then is CLAHE the single
    variable.
    """
    from export_ir_percentile import render

    stems = sorted(p.stem for p in (SRC_TREE / "images").rglob("*.png"))
    if not stems:
        raise SystemExit(f"no images under {SRC_TREE / 'images'}")
    step = max(1, len(stems) // n)
    sample = stems[::step][:n]
    worst = 0
    for stem in sample:
        run = stem.split("_", 1)[0]
        cur = np.array(Image.open(SRC_TREE / "images" / run / f"{stem}.png"))
        rep = render(stem, "minmax")
        if cur.shape != rep.shape:
            raise SystemExit(f"{stem}: shape {rep.shape} vs existing {cur.shape}")
        worst = max(worst, int(np.abs(cur.astype(int) - rep.astype(int)).max()))
    print(f"[clahe] verify: {len(sample)} frames, max abs diff {worst}")
    if worst != 0:
        raise SystemExit("VERIFY FAILED — the shared path does not reproduce the "
                         "existing tree, so CLAHE would not be the only variable")
    print("[clahe] verify OK — CLAHE is the only difference this export introduces")
    return 0


def _one(job) -> tuple[str, bool]:
    stem, clip, tile = job
    try:
        run = stem.split("_", 1)[0]
        out = DST_TREE / "images" / run / f"{stem}.png"
        if out.is_file():
            return stem, True
        out.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(render_clahe(stem, clip, tile)).save(out, optimize=False)
        return stem, True
    except Exception as exc:                      # noqa: BLE001
        return f"{stem}: {exc}", False


def link_labels_and_lists() -> dict:
    """Labels and split lists are unchanged — CLAHE moves pixels, not boxes."""
    stats = {}
    for split in ("train", "val", "test"):
        src_list = SRC_TREE / f"{split}.txt"
        if not src_list.is_file():
            continue
        rows = [ln.strip() for ln in src_list.read_text(encoding="utf-8").splitlines() if ln.strip()]
        out_paths = []
        for rel in rows:
            src_img = (Path(rel) if Path(rel).is_absolute() else (SRC_TREE / rel)).resolve()
            parts = list(src_img.parts)
            i = len(parts) - 1 - parts[::-1].index("images")
            run = parts[i + 1]
            out_paths.append(str(DST_TREE / "images" / run / src_img.name))
            lp = list(parts)
            lp[i] = "labels"
            src_lbl = Path(*lp).with_suffix(".txt")
            dst_lbl = DST_TREE / "labels" / run / (src_img.stem + ".txt")
            dst_lbl.parent.mkdir(parents=True, exist_ok=True)
            if not dst_lbl.exists():
                try:
                    os.link(src_lbl, dst_lbl)
                except OSError:
                    dst_lbl.write_text(src_lbl.read_text(encoding="utf-8") if src_lbl.is_file() else "",
                                       encoding="utf-8")
        (DST_TREE / f"{split}.txt").write_text("\n".join(out_paths) + "\n", encoding="utf-8")
        stats[split] = len(out_paths)
    return stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--verify", action="store_true", help="reproduce min-max through the shared path and stop")
    ap.add_argument("--verify-n", type=int, default=60)
    ap.add_argument("--clip", type=float, default=CLIP_LIMIT)
    ap.add_argument("--tile", type=int, nargs=2, default=list(TILE))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 4) - 2))
    args = ap.parse_args()

    if args.verify:
        return verify(args.verify_n)
    verify(args.verify_n)          # the gate always runs before an export

    stems = sorted(p.stem for p in (SRC_TREE / "images").rglob("*.png"))
    tile = (int(args.tile[0]), int(args.tile[1]))
    print(f"[clahe] exporting {len(stems)} frames, clip={args.clip} tile={tile}, "
          f"{args.workers} workers -> {DST_TREE}")
    jobs = [(s, args.clip, tile) for s in stems]
    done = fail = 0
    errs = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        for msg, ok in ex.map(_one, jobs, chunksize=64):
            if ok:
                done += 1
                if done % 5000 == 0:
                    print(f"[clahe]   {done}/{len(stems)}", flush=True)
            else:
                fail += 1
                if len(errs) < 5:
                    errs.append(msg)
    print(f"[clahe] exported {done}, failed {fail}")
    for e in errs:
        print(f"[clahe]   ! {e}")
    if fail:
        raise SystemExit(f"{fail} frames failed to export")

    counts = link_labels_and_lists()
    yaml_path = ROOT / "runs" / "derived" / "data_ir_clahe.yaml"
    body = "".join(f"{s}: {DST_TREE / f'{s}.txt'}\n" for s in counts)
    yaml_path.write_text(
        f"# IR re-exported with CLAHE (clip {args.clip}, tile {tile}) on the raw 16-bit\n"
        f"# frame, then the ORIGINAL min-max 8-bit map, then letterbox. Labels are\n"
        f"# hardlinks: CLAHE moves pixels, not boxes.\n"
        + body + "names:\n  0: ship\n  1: buoy\n", encoding="utf-8")
    (DST_TREE / "export_meta.json").write_text(json.dumps(
        {"clip": args.clip, "tile": tile, "frames": done, "counts": counts,
         "mapping": "clahe16 -> minmax8 -> letterbox"}, indent=2), encoding="utf-8")
    print(f"[clahe] wrote {yaml_path}  ({counts})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
