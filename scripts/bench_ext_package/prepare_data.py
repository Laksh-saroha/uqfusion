"""Point the package's data files at wherever the package now lives, and verify the copy.

Standard library only. Run by setup_and_run.ps1 on every launch:

    python scripts/prepare_data.py            # write the data yaml + train list for this location
    python scripts/prepare_data.py --verify   # also sha256 every packaged file against package_manifest.json

The train list ships as paths relative to Pohang_dataset/visible (`data_vis_train_stride4.rel.txt`);
Ultralytics needs absolute ones, so they are written here. val.txt uses "./" lines, which Ultralytics
resolves against the list's own folder, so it ships unchanged. Test is not used in training and is not
packaged.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VIS = ROOT / "Pohang_dataset" / "visible"
DERIVED = ROOT / "runs" / "derived"
MANIFEST = ROOT / "package_manifest.json"
VERIFIED = ROOT / ".verified.json"


def write_data() -> None:
    rel = (DERIVED / "data_vis_train_stride4.rel.txt").read_text(encoding="utf-8").split()
    train = DERIVED / "data_vis_train_stride4.txt"
    train.write_text("".join(f"{VIS.as_posix()}/{r}\n" for r in rel), encoding="utf-8")
    (DERIVED / "data_vis_stride4.yaml").write_text(
        "# Written by scripts/prepare_data.py for this location; the ep25 benchmark data (stride-4 train + val).\n"
        f"train: {train.as_posix()}\n"
        f"val: {(VIS / 'val.txt').as_posix()}\n"
        "names:\n  0: ship\n  1: buoy\n", encoding="utf-8")
    print(f"[data] {len(rel)} train images -> {train}")


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def verify() -> bool:
    files = json.loads(MANIFEST.read_text(encoding="utf-8"))["files"]
    t0, bad = time.time(), []

    def check(item):
        rel, (size, digest) = item
        p = ROOT / rel
        if not p.is_file():
            return rel, "missing"
        if p.stat().st_size != size:
            return rel, "size"
        return (rel, "sha256") if sha256(p) != digest else None

    with ThreadPoolExecutor(8) as ex:
        for i, r in enumerate(ex.map(check, files.items()), 1):
            if r:
                bad.append(r)
            if i % 5000 == 0:
                print(f"[verify] {i}/{len(files)}", flush=True)
    for rel, why in bad[:20]:
        print(f"[verify] BAD {why}: {rel}")
    ok = not bad
    print(f"[verify] {'PASS' if ok else 'FAIL'}: {len(files) - len(bad)}/{len(files)} files match the manifest "
          f"({time.time() - t0:.0f}s)")
    if ok:
        VERIFIED.write_text(json.dumps({"root": str(ROOT), "at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                                        "files": len(files)}), encoding="utf-8")
    return ok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--verify", action="store_true", help="sha256 every packaged file against the manifest")
    args = ap.parse_args()
    write_data()
    return 0 if (not args.verify or verify()) else 1


if __name__ == "__main__":
    sys.exit(main())
