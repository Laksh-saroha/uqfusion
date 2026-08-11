"""Make split .txt lists loadable by Ultralytics: prefix each line with './'.

Ultralytics' dataset loader only rewrites a txt line to be relative to the txt
file's own folder when the line starts with './'. Lines like
'images/pohang04/x.png' are otherwise resolved against the current working
directory, producing 'No such file or directory' at train time.

This rewrites train/val/test.txt for a modality so every relative line starts
with './' (the same form Ultralytics' own coco.yaml/VisDrone.yaml lists use).
Absolute lines and lines already starting with './' are left untouched, so it
is safe to re-run. The original is saved once as '<name>.txt.orig'.

Usage:
    python scripts/fix_split_lists.py --data vis
    python scripts/fix_split_lists.py --data ir
    python scripts/fix_split_lists.py --data vis --check   # report only, no write
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.data.lists import dataset_root, load_data_yaml


def _is_absolute(line: str) -> bool:
    # posix '/...', windows 'C:/...' / 'C:\...'
    return line.startswith("/") or (len(line) > 1 and line[1] == ":")


def fix_file(txt: Path, check: bool) -> tuple[int, int]:
    """Return (n_lines, n_rewritten). Rewrites in place unless check=True."""
    lines = txt.read_text(encoding="utf-8").splitlines()
    out, changed = [], 0
    for raw in lines:
        line = raw.strip()
        if not line:
            out.append(raw)
            continue
        if line.startswith("./") or _is_absolute(line):
            out.append(line)
        else:
            out.append("./" + line)
            changed += 1
    if changed and not check:
        backup = txt.with_suffix(txt.suffix + ".orig")
        if not backup.exists():
            shutil.copy2(txt, backup)
        txt.write_text("\n".join(out) + "\n", encoding="utf-8")
    return len(lines), changed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="vis", help="vis, ir, or a dataset yaml path")
    parser.add_argument("--check", action="store_true", help="report only; do not modify files")
    args = parser.parse_args()

    cfg = load_config(args.config)
    data = load_data_yaml(resolve_data_yaml(cfg, args.data))
    root = dataset_root(data)

    total_changed = 0
    for split in ("train", "val", "test"):
        entry = data.get(split)
        if entry is None:
            continue
        entries = entry if isinstance(entry, list) else [entry]
        for e in entries:
            p = Path(str(e))
            p = p if p.is_absolute() else (root / p).resolve()
            if p.suffix.lower() != ".txt":
                print(f"[{split}] {p} is not a txt list — skipped")
                continue
            if not p.is_file():
                print(f"[{split}] MISSING: {p}")
                continue
            n, changed = fix_file(p, args.check)
            total_changed += changed
            verb = "would prefix" if args.check else ("prefixed" if changed else "already ok")
            print(f"[{split}] {p.name}: {n} lines, {verb} {changed} with './'")

    if args.check:
        print(f"\nCHECK: {total_changed} lines need './' (run without --check to apply).")
    else:
        print(f"\nDONE: {total_changed} lines rewritten." if total_changed
              else "\nDONE: nothing to change (all lines already ok).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
