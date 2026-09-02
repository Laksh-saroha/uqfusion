"""Re-drop only the boxes the PER-BOX test actually flagged, after a full restore.

`filter_night_boxes.py --cut-dark pohang01:100` cut at FRAME level: a dark frame
lost every box in it, whatever that box's own photometric score. The per-box audit
in `runs/visfilter/box_scores.csv` (thresholds `--t-int 45 --t-grad 8
--t-contrast 10`, frame gate `--dark-median 40`) says only 38,135 of the 132,688
deleted boxes actually failed all three tests. 82,694 passed and went anyway, and
11,859 were never scored at all because their frame median sat between the audit
gate (40) and the cut threshold (100).

This applies the per-box verdict the audit already computed, rather than
re-scoring: `--restore` first, then drop exactly the rows with `flagged=1`, by
`line_idx`. Reusing the committed CSV means the labels match the audit that
justified the change, byte for byte, instead of a fresh scoring pass that could
drift.

Implements `docs/prereg-night-label-restore.md` (committed `030244e`).

Safety: label files carry `.pre_visfilter` backups written once and never
clobbered (`filter_night_boxes.py:617`), so the original pre-filter state survives
this. `val`/`test` labels are never touched -- only images listed as TRAIN. IR
labels are not touched at all.

Usage:
    python scripts/restore_night_perbox.py --dry-run
    python scripts/restore_night_perbox.py --execute
"""

from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LABELS = ROOT / "Pohang_dataset/visible/labels"
SCORES = ROOT / "runs/visfilter/box_scores.csv"
TRAIN_LIST = ROOT / "Pohang_dataset/visible/train.txt"
BACKUP = ".pre_visfilter"


def label_for(image_name: str, run: str) -> Path:
    return LABELS / run / (Path(image_name).stem + ".txt")


def nlines(p: Path) -> int:
    if not p.is_file():
        return 0
    return len([x for x in p.read_text(encoding="utf-8").splitlines() if x.strip()])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--dry-run", action="store_true")
    g.add_argument("--execute", action="store_true")
    ap.add_argument("--out", default="runs/visfilter/perbox_restore_manifest.json")
    ap.add_argument("--force", action="store_true",
                    help="re-execute even though a manifest exists (see the guard below)")
    args = ap.parse_args()
    t0 = time.time()

    # `line_idx` indexes the RESTORED label file. After a successful --execute the
    # flagged lines are gone, so the same indices now point at DIFFERENT boxes and
    # a second run would delete the wrong ones. The `max(idxs) >= len(lines)` check
    # further down catches only the files that shrank past the highest index, not
    # the rest. So refuse outright unless a fresh `--restore` has been done.
    manifest = ROOT / args.out
    if args.execute and manifest.is_file() and not args.force:
        print(f"[refuse] {manifest.relative_to(ROOT)} exists -- labels are already "
              f"re-dropped.\n         Re-running would apply line_idx to shortened "
              f"files and delete the WRONG boxes.\n         Run "
              f"`filter_night_boxes.py --restore` first, or pass --force.")
        return 1

    train = {Path(l.strip()).name for l in TRAIN_LIST.read_text(encoding="utf-8").splitlines()
             if l.strip()}
    print(f"[data] TRAIN images listed: {len(train)}")

    drop: dict[tuple[str, str], set[int]] = collections.defaultdict(set)
    seen = 0
    with open(SCORES, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            seen += 1
            if r["flagged"].strip() not in ("1", "true", "True", "yes"):
                continue
            img = Path(r["image"]).name
            if img not in train:                       # never touch val/test
                continue
            drop[(r["run"], img)].add(int(r["line_idx"]))
    n_flag = sum(len(v) for v in drop.values())
    print(f"[scores] {seen} scored boxes; {n_flag} flagged in TRAIN across "
          f"{len(drop)} files")

    before = after = 0
    changed = []
    for (run, img), idxs in sorted(drop.items()):
        lp = label_for(img, run)
        if not lp.is_file():
            print(f"  [warn] missing label {lp}")
            continue
        lines = [x for x in lp.read_text(encoding="utf-8").splitlines() if x.strip()]
        before += len(lines)
        if max(idxs) >= len(lines):
            print(f"  [ABORT] {lp.name}: line_idx {max(idxs)} >= {len(lines)} lines")
            return 1
        kept = [x for i, x in enumerate(lines) if i not in idxs]
        after += len(kept)
        changed.append((lp, kept, len(lines) - len(kept)))

    print(f"[plan] {len(changed)} files: {before} boxes -> {after} "
          f"({before - after} dropped)")

    if args.dry_run:
        print(f"[dry-run] nothing written ({time.time() - t0:.1f}s)")
        return 0

    for lp, kept, _ in changed:
        bak = lp.with_name(lp.name + BACKUP)
        if not bak.is_file():                          # never clobber the original
            bak.write_bytes(lp.read_bytes())
        lp.write_text("".join(x + "\n" for x in kept), encoding="utf-8")

    # ---- verify and hash --------------------------------------------------
    tot = 0
    h = hashlib.sha256()
    for run_dir in sorted(LABELS.iterdir()):
        if not run_dir.is_dir():
            continue
        for p in sorted(run_dir.glob("*.txt")):
            b = p.read_bytes()
            h.update(b)
            if p.name.replace(".txt", ".png") in train:
                tot += len([x for x in b.decode("utf-8").splitlines() if x.strip()])
    digest = h.hexdigest()
    p01 = sum(nlines(p) for p in sorted((LABELS / "pohang01").glob("*.txt"))
              if p.name.replace(".txt", ".png") in train)
    print(f"[verify] pohang01 TRAIN boxes now {p01}")
    print(f"[verify] all-VIS label hash {digest[:12]}")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "mode": "perbox_restore",
        "source_scores": str(SCORES.relative_to(ROOT)),
        "thresholds": {"t_int": 45.0, "t_grad": 8.0, "t_contrast": 10.0,
                       "dark_median": 40.0},
        "files_rewritten": len(changed),
        "boxes_before": before, "boxes_after": after,
        "boxes_dropped": before - after,
        "pohang01_train_boxes": p01,
        "vis_label_hash": digest,
        "prereg": "docs/prereg-night-label-restore.md",
    }, indent=2), encoding="utf-8")
    print(f"[out] {out}  ({time.time() - t0:.1f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
