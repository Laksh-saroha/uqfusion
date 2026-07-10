"""One-time Pohang data preparation: IR letterbox to 640x640 + leakage-free re-split.

Two operations the dataset contract (dataset_requirement.md) requires before training:

  1. **IR letterbox** — infrared frames ship at native 640x512; letterbox them to
     640x640 (pad 114, aspect preserved) to match the visible stream (§5), and
     transform the YOLO labels for the padding. Originals are archived to
     `infrared_orig/` (rename, non-destructive).

  2. **Re-split** — the delivered split is frame-interleaved (train frame N, val
     frame N+1): it fails the D6-rev temporal-buffer rule and `scripts/audit_split.py`.
     Re-split into contiguous per-run ordinal blocks (train | guard | val | guard |
     test) with a guard band wider than the audit's `median_delta * min_gap_frames`,
     sized PER RUN because IR density varies (pohang02/03 IR are sparse). The same
     ordinal cuts are applied to BOTH modalities so paired VIS/IR frames never split
     across sets (fusion correctness). pohang01 (night) keeps blocks in both train
     and test, preserving the real-night eval row.

Output layout (txt-list based, contract §2 preferred form):

    visible/images/<run>/<file>.png   visible/labels/<run>/<file>.txt
    infrared/images/<run>/<file>.png  infrared/labels/<run>/<file>.txt   (640x640)
    {visible,infrared}/{train,val,test}.txt   # paths relative to the modality root

The plan is validated against the real audit_split() on the PROJECTED lists before
any file is touched (the audit reads filenames only, so planned paths need not exist
yet). Run with --execute to perform the migration; default is a dry-run plan.
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path
from statistics import median

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uqfusion.data.audit import audit_split, format_report  # noqa: E402
from uqfusion.data.lists import frame_ordinal, run_key       # noqa: E402

ROOT = Path(__file__).resolve().parents[1] / "Pohang_dataset"
TARGET = 640
PAD = 114
MIN_GAP_FRAMES = 100          # must match config data_audit.min_gap_frames
TRAIN_FRAC, VAL_FRAC = 0.80, 0.90   # cut positions (test = remainder)
GUARD_MARGIN = 1.3            # guard = required_gap * margin
GUARD_FLOOR = 150


# ---- planning -------------------------------------------------------------

def collect(mod_dir: Path):
    """{run: {ordinal: [(image_path, label_path), ...]}} over current train/val/test dirs."""
    from collections import defaultdict
    out = defaultdict(lambda: defaultdict(list))
    for split in ("train", "val", "test"):
        img_dir = mod_dir / "images" / split
        if not img_dir.is_dir():
            continue
        for img in img_dir.glob("*.png"):
            o = frame_ordinal(img)
            if o is None:
                raise SystemExit(f"no ordinal for {img}")
            lbl = mod_dir / "labels" / split / (img.stem + ".txt")
            out[run_key(img)][o].append((img, lbl))
    return out


def median_delta(ordinals) -> float:
    o = sorted(ordinals)
    deltas = [b - a for a, b in zip(o, o[1:]) if b > a]
    return median(deltas) if deltas else 1.0


def plan_cuts(vis, ir):
    """Per-run (c1, c2, guard) from the VIS ordinal timeline + per-modality density."""
    cuts = {}
    for run in sorted(set(vis) | set(ir)):
        vis_ords = sorted(vis.get(run, {}))
        ir_ords = sorted(ir.get(run, {}))
        base = vis_ords or ir_ords           # pohang04 has no IR
        c1 = base[int(TRAIN_FRAC * len(base))]
        c2 = base[int(VAL_FRAC * len(base))]
        req = median_delta(vis_ords) * MIN_GAP_FRAMES if vis_ords else 0
        if ir_ords:
            req = max(req, median_delta(ir_ords) * MIN_GAP_FRAMES)
        guard = max(GUARD_FLOOR, int(req * GUARD_MARGIN) + 1)
        cuts[run] = (c1, c2, guard)
    return cuts


def assign(o: float, c1: float, c2: float, guard: float):
    if o <= c1:
        return "train"
    if o <= c1 + guard:
        return None
    if o <= c2:
        return "val"
    if o <= c2 + guard:
        return None
    return "test"


def build_assignments(byrun, cuts):
    """{run: {ordinal: split}} with guard-band ordinals dropped (None)."""
    out = {}
    for run, ords in byrun.items():
        c1, c2, guard = cuts[run]
        out[run] = {o: assign(o, c1, c2, guard) for o in ords}
    return out


def projected_lists(byrun, assigns, mod_name: str):
    """Planned split -> [relative image paths] using the FINAL per-run layout."""
    lists = {"train": [], "val": [], "test": []}
    for run, ordmap in byrun.items():
        for o, pairs in ordmap.items():
            split = assigns[run][o]
            if split is None:
                continue
            for img, _ in pairs:
                lists[split].append(f"images/{run}/{img.name}")
    return {k: sorted(v) for k, v in lists.items()}


def audit_projected(mod_dir: Path, lists) -> dict:
    """Run the real audit on projected lists (filenames only; files need not exist)."""
    tmp = mod_dir / "_plan"
    tmp.mkdir(exist_ok=True)
    for split, items in lists.items():
        (tmp / f"{split}.txt").write_text("\n".join(items), encoding="utf-8")
    data = {
        "_yaml_path": str((mod_dir / "_plan_data.yaml").resolve()),
        "path": str(mod_dir.resolve()),
        "train": "_plan/train.txt", "val": "_plan/val.txt", "test": "_plan/test.txt",
    }
    report = audit_split(data, MIN_GAP_FRAMES)
    shutil.rmtree(tmp)
    return report


# ---- execution ------------------------------------------------------------

def letterbox_image(src: Path, dst: Path):
    im = Image.open(src)
    w, h = im.size
    scale = min(TARGET / w, TARGET / h)
    nw, nh = round(w * scale), round(h * scale)
    resized = im.resize((nw, nh), Image.BILINEAR) if (nw, nh) != (w, h) else im
    pad_l, pad_t = (TARGET - nw) // 2, (TARGET - nh) // 2
    canvas = Image.new(im.mode, (TARGET, TARGET), PAD if im.mode == "L" else (PAD, PAD, PAD))
    canvas.paste(resized, (pad_l, pad_t))
    dst.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(dst)
    return w, h, scale, pad_l, pad_t


def letterbox_label(src: Path, dst: Path, w0, h0, scale, pad_l, pad_t):
    lines_out = []
    if src.is_file():
        for line in src.read_text(encoding="utf-8").splitlines():
            p = line.split()
            if len(p) != 5:
                continue
            c, cx, cy, ww, hh = p[0], *map(float, p[1:])
            px, py = cx * w0 * scale + pad_l, cy * h0 * scale + pad_t
            pw, ph = ww * w0 * scale, hh * h0 * scale
            x1, y1, x2, y2 = px - pw / 2, py - ph / 2, px + pw / 2, py + ph / 2
            x1, x2 = max(0, min(TARGET, x1)), max(0, min(TARGET, x2))
            y1, y2 = max(0, min(TARGET, y1)), max(0, min(TARGET, y2))
            if x2 - x1 <= 1e-3 or y2 - y1 <= 1e-3:
                continue
            lines_out.append(
                f"{c} {(x1+x2)/2/TARGET:.6f} {(y1+y2)/2/TARGET:.6f} "
                f"{(x2-x1)/TARGET:.6f} {(y2-y1)/TARGET:.6f}"
            )
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text("\n".join(lines_out) + ("\n" if lines_out else ""), encoding="utf-8")


def execute_ir(ir_byrun):
    """Archive infrared -> infrared_orig, letterbox into per-run tree."""
    ir_dir = ROOT / "infrared"
    orig = ROOT / "infrared_orig"
    if not orig.exists():
        ir_dir.rename(orig)
    ir_dir.mkdir(exist_ok=True)
    # re-collect from the archived tree (byrun paths pointed at the old location)
    n = 0
    for run, ordmap in collect(orig).items():
        for o, pairs in ordmap.items():
            for img, lbl in pairs:
                dst_img = ir_dir / "images" / run / img.name
                dst_lbl = ir_dir / "labels" / run / (img.stem + ".txt")
                if dst_img.exists():
                    n += 1
                    continue
                w0, h0, scale, pl, pt = letterbox_image(img, dst_img)
                letterbox_label(lbl, dst_lbl, w0, h0, scale, pl, pt)
                n += 1
                if n % 5000 == 0:
                    print(f"  IR letterboxed {n}", flush=True)
    print(f"  IR letterboxed {n} total", flush=True)


def execute_reorg_vis(vis_byrun):
    """Move visible images/labels from split dirs into per-run dirs (metadata rename)."""
    vis = ROOT / "visible"
    n = 0
    for run, ordmap in vis_byrun.items():
        for o, pairs in ordmap.items():
            for img, lbl in pairs:
                dst_img = vis / "images" / run / img.name
                dst_lbl = vis / "labels" / run / (lbl.name)
                dst_img.parent.mkdir(parents=True, exist_ok=True)
                dst_lbl.parent.mkdir(parents=True, exist_ok=True)
                if not dst_img.exists():
                    shutil.move(str(img), str(dst_img))
                if lbl.is_file() and not dst_lbl.exists():
                    shutil.move(str(lbl), str(dst_lbl))
                n += 1
    # drop now-empty split dirs
    for split in ("train", "val", "test"):
        for sub in ("images", "labels"):
            d = vis / sub / split
            if d.is_dir() and not any(d.iterdir()):
                d.rmdir()
    print(f"  VIS reorganized {n} frames", flush=True)


def write_lists_and_yaml(mod_dir: Path, lists, names):
    for split, items in lists.items():
        (mod_dir / f"{split}.txt").write_text("\n".join(items) + "\n", encoding="utf-8")
    yaml_txt = (
        f"path: {mod_dir.resolve().as_posix()}\n"
        f"train: train.txt\nval: val.txt\ntest: test.txt\n"
        f"names:\n  0: ship\n  1: buoy\n"
    )
    return yaml_txt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--execute", action="store_true", help="perform the migration (default: plan only)")
    args = ap.parse_args()

    vis_byrun = collect(ROOT / "visible")
    ir_byrun = collect(ROOT / "infrared")
    cuts = plan_cuts(vis_byrun, ir_byrun)
    vis_assign = build_assignments(vis_byrun, cuts)
    ir_assign = build_assignments(ir_byrun, cuts)
    vis_lists = projected_lists(vis_byrun, vis_assign, "vis")
    ir_lists = projected_lists(ir_byrun, ir_assign, "ir")

    print("=== Per-run cuts (c1, c2, guard) ===")
    for run, (c1, c2, g) in cuts.items():
        print(f"  {run}: c1={c1} c2={c2} guard={g}")
    print("=== Projected split sizes ===")
    for name, ll in (("VIS", vis_lists), ("IR", ir_lists)):
        tot = sum(len(v) for v in ll.values())
        print(f"  {name}: " + ", ".join(f"{k}={len(v)}" for k, v in ll.items()) + f"  (listed {tot})")

    print("=== Audit on PROJECTED lists ===")
    ok = True
    for name, mod_dir, ll in (("VIS", ROOT / "visible", vis_lists), ("IR", ROOT / "infrared", ir_lists)):
        rep = audit_projected(mod_dir, ll)
        print(f"  {name}: {'PASS' if rep['ok'] else 'FAIL'} "
              f"(dupes={len(rep['duplicates'])}, temporal={len(rep['temporal_violations'])})")
        if not rep["ok"]:
            ok = False
            print(format_report(rep, max_examples=5))
    if not ok:
        raise SystemExit("Projected split FAILS audit — aborting before any writes.")
    print("Projected split PASSES audit for both modalities.")

    if not args.execute:
        print("\n(dry run — rerun with --execute to migrate)")
        return

    print("\n=== EXECUTING ===")
    print("[1/4] IR letterbox...")
    execute_ir(ir_byrun)
    print("[2/4] VIS reorganize...")
    execute_reorg_vis(vis_byrun)
    print("[3/4] write lists + yamls...")
    (ROOT / "data_vis.yaml").write_text(write_lists_and_yaml(ROOT / "visible", vis_lists, None), encoding="utf-8")
    (ROOT / "data_ir.yaml").write_text(write_lists_and_yaml(ROOT / "infrared", ir_lists, None), encoding="utf-8")
    print("[4/4] done. Run scripts/audit_split.py to confirm on-disk.")


if __name__ == "__main__":
    main()
