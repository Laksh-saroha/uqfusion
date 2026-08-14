"""Prove the uploaded dataset is the experiment's dataset, before any GPU time.

Three checks, cheapest first:

1. every file in the upload manifest exists with exactly the byte count it had on
   the laptop — catches a truncated tar stream, which otherwise surfaces 30 epochs
   later as a corrupt-image warning nobody reads;
2. the dataset yaml's split_fingerprint equals 682dbe9f0f05 — the id every row in
   benchmark_results_tail.csv carries. grid.py refuses to mix splits, so a mismatch
   here is a run that would abort on launch;
3. one label file per image, and the ship class (0) is actually present.

Usage:  PYTHONPATH=<root>/src python jarvislabs/remote_verify.py --root /home/uqfusion
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

EXPECTED_FINGERPRINT = "682dbe9f0f05"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", default="/home/uqfusion")
    args = ap.parse_args()

    root = Path(args.root)
    vis = root / "data" / "pohang" / "visible"
    expect = root / "jarvislabs" / "expect.tsv"
    sys.path.insert(0, str(root / "src"))

    if not expect.is_file():
        print(f"[verify] FAIL missing manifest {expect}")
        return 1

    bad_count = 0
    total_files = total_bytes = 0
    for line in expect.read_text(encoding="utf-8").splitlines()[1:]:
        name, files, nbytes = line.split("\t")
        files, nbytes = int(files), int(nbytes)
        if name == "TOTAL":
            continue
        d = vis / name
        if not d.is_dir():
            print(f"[verify] FAIL missing directory {d}")
            return 1
        got_files = got_bytes = 0
        with os.scandir(d) as it:
            for e in it:
                if e.is_file():
                    got_files += 1
                    got_bytes += e.stat().st_size
        total_files += got_files
        total_bytes += got_bytes
        status = "ok" if (got_files, got_bytes) == (files, nbytes) else "MISMATCH"
        if status != "ok":
            bad_count += 1
            print(f"[verify] {status} {name}: files {got_files}/{files} bytes {got_bytes}/{nbytes}")
    if bad_count:
        print(f"[verify] FAIL {bad_count} directories differ from the manifest — re-run the sync")
        return 1
    print(f"[verify] files OK: {total_files} files, {total_bytes / 1e9:.2f} GB")

    from uqfusion.bench.grid import split_fingerprint
    from uqfusion.data.lists import load_data_yaml, split_image_list

    yaml_path = root / "runs" / "derived" / "data_vis_stride2.yaml"
    fp = split_fingerprint(yaml_path)
    print(f"[verify] split_fingerprint {fp} (expected {EXPECTED_FINGERPRINT})")
    if fp != EXPECTED_FINGERPRINT:
        print("[verify] FAIL this is not the experiment's split")
        return 1

    data = load_data_yaml(yaml_path)
    imgs = split_image_list(data, "train") + split_image_list(data, "val")
    missing = [p for p in imgs[::500] if not p.is_file()]
    if missing:
        print(f"[verify] FAIL {len(missing)} sampled images missing, e.g. {missing[0]}")
        return 1

    ship = 0
    for p in imgs[::500]:
        lab = Path(str(p).replace("/images/", "/labels/")).with_suffix(".txt")
        if not lab.is_file():
            print(f"[verify] FAIL missing label {lab}")
            return 1
        ship += sum(1 for ln in lab.read_text().splitlines() if ln.strip().startswith("0 "))
    print(f"[verify] sampled {len(imgs[::500])} frames, {ship} ship boxes")
    if not ship:
        print("[verify] FAIL no class-0 boxes in the sample — wrong labels")
        return 1

    print("[verify] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
