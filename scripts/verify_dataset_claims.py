"""Check the dataset claims in `scope.md` §5.1 against the tree on this machine.

R-F2 / F17-adjacent. The external review could only say that the *sources* do
not verify our local numbers. This script says what the local numbers actually
are, so the scope document cites a measurement rather than a memory.

Deliberately dependency-light and self-contained: it walks the label trees and
the pair tables directly and imports nothing from ``uqfusion``, so it
cross-checks the package instead of trusting it (same discipline as
``verify_dataset_state.py``). Run:

    python scripts/verify_dataset_claims.py

Exit code is 0 whether or not claims match — this REPORTS, it does not gate.
Claims are printed with their scope.md wording next to the measured value so a
mismatch is visible rather than inferred.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

RUNS = ("pohang00", "pohang01", "pohang02", "pohang03", "pohang04")
MODALITIES = ("visible", "infrared")

# The claims as they stand in scope.md §5.1 on 2026-09-10, so the comparison is
# against the document rather than against what I remember it saying.
CLAIMED = {
    "images_total": "~158k",
    "boxes_total": "~1.22M",
    "pairs_total": "~28k",
    "vis_images": "127k",
    "ir_images": "31k",
    "pohang03_ir_images": "~1,922",
    "pohang03_vis_images": "~13k",
    "pohang04_ir_labels": "zero",
}


def count_labels(root: Path, run: str | None = None) -> tuple[int, int, int]:
    """(label files, boxes, empty files) under ``root``, optionally one run."""
    n_files = n_boxes = n_empty = 0
    for dirpath, _dirnames, filenames in os.walk(root):
        for name in filenames:
            if not name.endswith(".txt"):
                continue
            path = os.path.join(dirpath, name)
            if run is not None and run not in path:
                continue
            n_files += 1
            with open(path, "rb") as fh:
                lines = [ln for ln in fh.read().split(b"\n") if ln.strip()]
            n_boxes += len(lines)
            if not lines:
                n_empty += 1
    return n_files, n_boxes, n_empty


def count_pairs(paired_dir: Path) -> dict[str, int]:
    out: dict[str, int] = {}
    for run in RUNS:
        path = paired_dir / f"{run}_pairs.csv"
        if not path.exists():
            out[run] = 0
            continue
        with open(path, newline="", encoding="utf-8") as fh:
            out[run] = max(0, sum(1 for _ in csv.reader(fh)) - 1)
    return out


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="Pohang_dataset", help="dataset tree root")
    ap.add_argument("--json", default=None, help="also write the measurements here")
    args = ap.parse_args(argv)

    root = Path(args.data)
    if not root.is_dir():
        print(f"[refuse] no dataset tree at {root.resolve()}", file=sys.stderr)
        return 2

    per_run: dict[str, dict[str, int]] = {}
    totals = {"images": 0, "boxes": 0}
    print(f"{'run':<10}{'vis_img':>10}{'vis_box':>11}{'ir_img':>10}{'ir_box':>11}")
    for run in RUNS:
        row: dict[str, int] = {}
        for modality in MODALITIES:
            files, boxes, empty = count_labels(root / modality / "labels", run)
            key = "vis" if modality == "visible" else "ir"
            row[f"{key}_images"] = files
            row[f"{key}_boxes"] = boxes
            row[f"{key}_empty"] = empty
            totals["images"] += files
            totals["boxes"] += boxes
        per_run[run] = row
        print(
            f"{run:<10}{row['vis_images']:>10}{row['vis_boxes']:>11}"
            f"{row['ir_images']:>10}{row['ir_boxes']:>11}"
        )

    pairs = count_pairs(root / "paired")
    pairs_total = sum(pairs.values())
    vis_images = sum(r["vis_images"] for r in per_run.values())
    ir_images = sum(r["ir_images"] for r in per_run.values())
    vis_boxes = sum(r["vis_boxes"] for r in per_run.values())
    ir_boxes = sum(r["ir_boxes"] for r in per_run.values())

    print()
    print(f"{'claim (scope.md 5.1)':<34}{'claimed':>12}{'measured':>14}")
    checks = [
        ("total images", CLAIMED["images_total"], totals["images"]),
        ("total boxes", CLAIMED["boxes_total"], totals["boxes"]),
        ("VIS images", CLAIMED["vis_images"], vis_images),
        ("IR images", CLAIMED["ir_images"], ir_images),
        ("paired VIS<->IR rows", CLAIMED["pairs_total"], pairs_total),
        ("pohang03 IR images", CLAIMED["pohang03_ir_images"], per_run["pohang03"]["ir_images"]),
        ("pohang03 VIS images", CLAIMED["pohang03_vis_images"], per_run["pohang03"]["vis_images"]),
        ("pohang04 IR labels", CLAIMED["pohang04_ir_labels"], per_run["pohang04"]["ir_images"]),
    ]
    for label, claimed, measured in checks:
        print(f"{label:<34}{claimed:>12}{measured:>14,}")

    print()
    print(f"VIS boxes {vis_boxes:,}   IR boxes {ir_boxes:,}")
    print("pairs per run: " + "  ".join(f"{k}={v:,}" for k, v in pairs.items()))
    print(
        "\nThese are LOCAL FILTERED counts for this tree, not the upstream PoLaRIS\n"
        "release. Cross-check the VIS tree hash with scripts/label_hash_ledger.py."
    )

    if args.json:
        payload = {
            "data_root": str(root.resolve()),
            "per_run": per_run,
            "pairs": pairs,
            "totals": {
                "images": totals["images"],
                "boxes": totals["boxes"],
                "vis_images": vis_images,
                "ir_images": ir_images,
                "vis_boxes": vis_boxes,
                "ir_boxes": ir_boxes,
                "pairs": pairs_total,
            },
            "claimed": CLAIMED,
        }
        Path(args.json).write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
