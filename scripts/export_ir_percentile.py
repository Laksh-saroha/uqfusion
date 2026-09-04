"""Re-export the IR tree from the raw 16-bit archive using percentile clipping.

Why this exists (see `docs/ir-handoff-2026-08.md` §3.3): the current 8-bit IR
pixels were made with per-frame **min-max**, which keys the mapping to the single
hottest and coldest pixel in the frame. One exhaust plume or sun glint off metal
compresses the whole scene: 51% of frames lose >1.5x contrast, 27% lose >2x, and
in 28% the scene bulk is squeezed into under half the available levels. Percentile
clipping discards the outlier tails before scaling -- the same pipeline with
exactly one step swapped.

    raw 16-bit 640x512  ->  percentile 8-bit 640x512  ->  letterbox 640x640

Only pixels change. Labels and split lists are **copied** from the existing tree
rather than recomputed, because the letterbox geometry is identical and copying
makes that guaranteed rather than merely intended -- a re-derivation that drifted
by one pixel would show up as a preprocessing result. Copies, not hardlinks, so a
future label edit (e.g. the §5.2 crossover filter) cannot silently hit both trees.

The percentiles are computed on the **unpadded** raw frame. That is load-bearing:
the letterbox adds 64 rows of constant 114 top and bottom, 20% of the frame, and
folding those into a percentile would drag both bounds toward the pad value. This
is the IR-specific trap called out in `dataset-changes-2026-07.md` §1.

The raw archive on D: is opened read-only and never written; `_assert_safe_out`
refuses to run if the output path is not inside this repo.

Usage:
    python scripts/export_ir_percentile.py --verify     # prove the mapping first
    python scripts/export_ir_percentile.py              # do the export
"""

from __future__ import annotations

import argparse
import shutil
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
RAW_ROOT = Path("D:/Datasets/Pohang")
SRC_TREE = ROOT / "Pohang_dataset" / "infrared"          # min-max tree (#3)
DST_TREE = ROOT / "Pohang_dataset" / "infrared_pct"      # percentile tree

TARGET = 640          # letterbox canvas, matches prepare_pohang.py
PAD = 114             # letterbox fill, matches prepare_pohang.py
LO_PCT, HI_PCT = 0.5, 99.5   # IR-D1


def _assert_safe_out(path: Path) -> None:
    """Refuse to write outside this repo. The raw archive is the archive of record."""
    resolved = str(path.resolve()).lower()
    if not resolved.startswith(str(ROOT.resolve()).lower()):
        raise SystemExit(f"REFUSING to write outside the repo: {path}")
    if resolved.startswith(str(RAW_ROOT.resolve()).lower()):
        raise SystemExit(f"REFUSING to write into the raw archive: {path}")


def raw_path_for(stem: str) -> Path:
    """pohang00_000006 -> D:/Datasets/Pohang/pohang00/infrared/images/000006.png"""
    run, frame = stem.split("_", 1)
    return RAW_ROOT / run / "infrared" / "images" / (frame + ".png")


def to_8bit(arr: np.ndarray, mode: str) -> np.ndarray:
    """16-bit -> 8-bit. `minmax` reproduces the original; `pct` is the new mapping."""
    if mode == "minmax":
        lo, hi = float(arr.min()), float(arr.max())
    else:
        lo, hi = (float(v) for v in np.percentile(arr, (LO_PCT, HI_PCT)))
        if hi <= lo:  # flat inside the clip band; fall back rather than emit black
            lo, hi = float(arr.min()), float(arr.max())
    if hi <= lo:
        return np.zeros(arr.shape, dtype=np.uint8)
    scaled = (arr.astype(np.float32) - lo) / (hi - lo)
    return (np.clip(scaled, 0.0, 1.0) * 255 + 0.5).astype(np.uint8)


def letterbox(arr8: np.ndarray) -> np.ndarray:
    """640x512 -> 640x640, pure padding (scale is 1.0; width is already 640)."""
    h, w = arr8.shape
    if (h, w) == (TARGET, TARGET):
        return arr8
    if w != TARGET or h > TARGET:
        raise ValueError(f"unexpected IR frame size {w}x{h}; expected {TARGET}x<={TARGET}")
    canvas = np.full((TARGET, TARGET), PAD, dtype=np.uint8)
    top = (TARGET - h) // 2
    canvas[top:top + h, :] = arr8
    return canvas


def render(stem: str, mode: str) -> np.ndarray:
    src = raw_path_for(stem)
    with Image.open(src) as im:
        arr = np.array(im)
    if arr.dtype != np.uint16:
        raise ValueError(f"{src} is {arr.dtype}, expected uint16")
    return letterbox(to_8bit(arr, mode))


def all_stems() -> list[str]:
    return sorted(p.stem for p in (SRC_TREE / "images").rglob("*.png"))


def verify(n: int) -> int:
    """Reproduce the EXISTING min-max pixels through this code path.

    This is the gate that matters. It proves the raw->stem mapping, the uint16
    read, the scaling arithmetic and the letterbox all match whatever actually
    produced the current tree. If min-max round-trips at max abs diff 0, then the
    percentile tree differs in exactly the one step deliberately changed.
    """
    stems = all_stems()
    step = max(1, len(stems) // n)
    sample = stems[::step][:n]
    print(f"[verify] reproducing min-max for {len(sample)} frames spread across "
          f"{len(stems):,}")
    worst, bad = 0, []
    for stem in sample:
        run = stem.split("_", 1)[0]
        existing = np.array(Image.open(SRC_TREE / "images" / run / (stem + ".png")))
        rebuilt = render(stem, "minmax")
        if existing.shape != rebuilt.shape:
            bad.append((stem, f"shape {existing.shape} vs {rebuilt.shape}"))
            continue
        d = int(np.abs(existing.astype(np.int16) - rebuilt.astype(np.int16)).max())
        worst = max(worst, d)
        if d:
            bad.append((stem, f"max abs diff {d}"))
    if bad:
        print(f"[verify] FAIL - {len(bad)} mismatched:")
        for stem, why in bad[:10]:
            print(f"   {stem}: {why}")
        return 1
    print(f"[verify] OK - max abs diff {worst} across {len(sample)} frames")

    # Second half: what the new mapping actually buys, on those same frames.
    gains = []
    for stem in sample:
        with Image.open(raw_path_for(stem)) as im:
            arr = np.array(im)
        mm, pc = to_8bit(arr, "minmax"), to_8bit(arr, "pct")
        if mm.std() > 0:
            gains.append(float(pc.std() / mm.std()))
    if gains:
        g = np.array(gains)
        print(f"[verify] contrast ratio pct/minmax: median {np.median(g):.2f}x  "
              f"p25 {np.percentile(g, 25):.2f}x  p75 {np.percentile(g, 75):.2f}x  "
              f"max {g.max():.2f}x")
        print(f"[verify] frames improved >1.5x: {(g > 1.5).mean() * 100:.0f}%  "
              f">2x: {(g > 2).mean() * 100:.0f}%")
    return 0


def _one(stem: str) -> str:
    run = stem.split("_", 1)[0]
    dst = DST_TREE / "images" / run / (stem + ".png")
    if dst.is_file():
        return stem
    arr = render(stem, "pct")
    Image.fromarray(arr, mode="L").save(dst, optimize=False, compress_level=1)
    return stem


def export(workers: int) -> int:
    _assert_safe_out(DST_TREE)
    stems = all_stems()
    print(f"[export] {len(stems):,} frames -> {DST_TREE}")

    for run_dir in sorted((SRC_TREE / "images").iterdir()):
        (DST_TREE / "images" / run_dir.name).mkdir(parents=True, exist_ok=True)
    lbl_dst = DST_TREE / "labels"
    if not lbl_dst.exists():
        print("[export] copying labels (a copy, not a hardlink - a future label "
              "edit must not hit both trees)")
        shutil.copytree(SRC_TREE / "labels", lbl_dst,
                        ignore=shutil.ignore_patterns("*.cache"))
    for name in ("train.txt", "val.txt", "test.txt"):
        shutil.copy2(SRC_TREE / name, DST_TREE / name)
    print("[export] split lists copied verbatim - identical membership by construction")

    done = fail = 0
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_one, s): s for s in stems}
        for i, fut in enumerate(as_completed(futures), 1):
            try:
                fut.result()
                done += 1
            except Exception as exc:                      # noqa: BLE001
                fail += 1
                if fail <= 5:
                    print(f"  FAIL {futures[fut]}: {exc}")
            if i % 2500 == 0:
                print(f"  {i:,}/{len(stems):,}", flush=True)
    print(f"[export] wrote/kept {done:,}, failed {fail:,}")
    return 1 if fail else 0


def audit() -> int:
    """Post-export checks: counts, split membership, geometry, padding."""
    ok = True
    src_n = len(all_stems())
    dst_n = len(list((DST_TREE / "images").rglob("*.png")))
    print(f"[audit] images: source {src_n:,}  percentile {dst_n:,}  "
          f"{'OK' if src_n == dst_n else 'MISMATCH'}")
    ok &= src_n == dst_n

    for name in ("train.txt", "val.txt", "test.txt"):
        a = (SRC_TREE / name).read_text(encoding="utf-8").split()
        b = (DST_TREE / name).read_text(encoding="utf-8").split()
        same = a == b
        print(f"[audit] {name}: {len(b):,} entries, identical to source: {same}")
        ok &= same
        missing = [p for p in b[:400] if not (DST_TREE / p.lstrip("./")).is_file()]
        if missing:
            print(f"[audit]   {len(missing)} of the first 400 listed files are MISSING")
            ok = False

    for stem in all_stems()[::4001][:8]:
        run = stem.split("_", 1)[0]
        arr = np.array(Image.open(DST_TREE / "images" / run / (stem + ".png")))
        pad_ok = bool((arr[:64] == PAD).all() and (arr[576:] == PAD).all())
        print(f"[audit] {stem}: {arr.shape} {arr.dtype} padding-is-{PAD}: {pad_ok}")
        ok &= arr.shape == (TARGET, TARGET) and arr.dtype == np.uint8 and pad_ok

    lbl_n = len(list((DST_TREE / "labels").rglob("*.txt")))
    src_lbl_n = len(list((SRC_TREE / "labels").rglob("*.txt")))
    print(f"[audit] labels: source {src_lbl_n:,}  percentile {lbl_n:,}  "
          f"{'OK' if lbl_n == src_lbl_n else 'MISMATCH'}")
    ok &= lbl_n == src_lbl_n

    print("AUDIT OK" if ok else "AUDIT FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="reproduce the existing min-max pixels and stop")
    ap.add_argument("--audit", action="store_true", help="check an existing export")
    ap.add_argument("-n", type=int, default=60, help="frames to sample for --verify")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.verify:
        sys.exit(verify(a.n))
    if a.audit:
        sys.exit(audit())
    rc = export(a.workers)
    sys.exit(rc or audit())
