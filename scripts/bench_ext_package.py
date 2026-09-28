"""Build the portable benchmark-extension package for another machine.

Packs what the delegated runs need (`runs/vis_benchmark_stride4_ep25_ext/delegated.json`) into one
folder that trains on its own. Once extracted, the layout mirrors this repo, so the package runs
`bench_ext_local.py` unchanged:

    <dest>/START.bat, setup_and_run.ps1, requirements.txt, README.txt      (scripts/bench_ext_package/)
    <dest>/scripts/bench_ext_local.py, bench_ext_fingerprint.py, dashboard.py, prepare_data.py
    <dest>/package_manifest.json                        size + sha256 of every file, plain and packed
    <dest>/payload.tar, extracted by setup_and_run.ps1 into:
        Pohang_dataset/visible/{images,labels}/...      stride-4 train + val: the files the --quick gate pins
        Pohang_dataset/visible/val.txt
        runs/derived/data_vis_train_stride4.rel.txt     train list relative to visible/ (prepare_data.py
                                                        writes the absolute list + yaml where it lands)
        server_dgxanode01/workspace/uqfusion/runs/vis_benchmark_stride4_ep25/<run>/  args.yaml,
                                                        results.csv, weights/last.pt per delegated run
        server_dgxanode01/.../queue_vis_benchmark_ep25/warm_start_progress.csv

The data travels as one uncompressed tar because the USB stick it was built for writes ~70k small
files at ~1.3 MB/s but one sequential file at ~4 MB/s. Test images are not packaged; training never
reads them.

    python scripts/bench_ext_package.py --dest E:/Training
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import io
import json
import os
import shutil
import socket
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import bench_ext_fingerprint as fp                                   # noqa: E402
import bench_ext_local as bel                                        # noqa: E402

TEMPLATE = ROOT / "scripts" / "bench_ext_package"


def plain_files() -> list[tuple[Path, str]]:
    """(source, path relative to the package root) for the small files copied as they are."""
    items = [(ROOT / "scripts" / n, f"scripts/{n}") for n in
             ("bench_ext_local.py", "bench_ext_fingerprint.py", "dashboard.py")]
    items.append((TEMPLATE / "prepare_data.py", "scripts/prepare_data.py"))
    return items + [(TEMPLATE / n, n) for n in ("START.bat", "setup_and_run.ps1", "requirements.txt", "README.txt")]


def payload_files(runs: list[str]) -> list[tuple[Path, str]]:
    """(source, path relative to the package root) for everything packed into payload.tar."""
    items = [(fp.LOC_VAL, "Pohang_dataset/visible/val.txt")]
    for p in fp.quick_paths():
        rel = fp.canon(p)                                           # visible/images/pohangXX/...
        items.append((Path(p), f"Pohang_dataset/{rel}"))
        items.append((Path(fp.label_of(p)), f"Pohang_dataset/{fp.canon(fp.label_of(p))}"))
    base_rel = bel.BASE.relative_to(ROOT).as_posix()
    for r in runs:
        for n in ("args.yaml", "results.csv", "weights/last.pt"):
            items.append((bel.BASE / r / n, f"{base_rel}/{r}/{n}"))
    items.append((bel.SERVER_PROGRESS, bel.SERVER_PROGRESS.relative_to(ROOT).as_posix()))
    return items


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dest", required=True)
    args = ap.parse_args()
    dest = Path(args.dest)
    runs = sorted(bel.delegated(), key=bel.ordered(list(bel.delegated())).index)
    if not runs:
        print("no runs in delegated.json", flush=True)
        return 1
    missing = [r for r in runs if not (bel.BASE / r / "weights" / "last.pt").is_file()]
    if missing:
        print(f"base weights missing: {missing}", flush=True)
        return 1
    dest.mkdir(parents=True, exist_ok=True)
    packed = payload_files(runs)
    need = sum(s.stat().st_size for s, _ in packed) + 512 * 3 * len(packed)
    free = shutil.disk_usage(dest).free
    print(f"[package] {len(runs)} runs, {len(packed)} files, {need / 2**30:.2f} GiB -> {dest / 'payload.tar'} "
          f"({free / 2**30:.1f} GiB free)", flush=True)
    if need > free:
        print("[package] not enough space", flush=True)
        return 1

    files, t0 = {}, time.time()
    lines = [fp.canon(p)[len("visible/"):] for p in fp.list_paths(fp.LOC_STRIDE4)]
    rel_list = "".join(f"{ln}\n" for ln in lines).encode("utf-8")
    part = dest / "payload.tar.part"
    with open(part, "wb", buffering=16 << 20) as raw, tarfile.open(fileobj=raw, mode="w",
                                                                   format=tarfile.PAX_FORMAT) as tf:
        def add(rel: str, data: bytes, mtime: float) -> None:
            ti = tarfile.TarInfo(rel)
            ti.size, ti.mtime, ti.mode = len(data), int(mtime), 0o644
            tf.addfile(ti, io.BytesIO(data))
            files[rel] = [len(data), hashlib.sha256(data).hexdigest()]

        add("runs/derived/data_vis_train_stride4.rel.txt", rel_list, time.time())
        done = 0
        for i, (src, rel) in enumerate(packed, 1):
            data = src.read_bytes()
            add(rel, data, src.stat().st_mtime)
            done += len(data)
            if i % 5000 == 0:
                dt_s = time.time() - t0
                print(f"[package] {i}/{len(packed)} {done / 2**30:.2f} GiB, {done / 2**20 / dt_s:.1f} MB/s, "
                      f"ETA {(need - done) / (done / dt_s) / 60:.0f} min", flush=True)
    os.replace(part, dest / "payload.tar")
    for src, rel in plain_files():
        (dest / rel).parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest / rel)
        files[rel] = [src.stat().st_size, hashlib.sha256(src.read_bytes()).hexdigest()]
    head = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True)
    dirty = subprocess.run(["git", "status", "--porcelain", "scripts"], cwd=ROOT, capture_output=True, text=True)
    manifest = {"built_at": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "source_host": socket.gethostname(), "git_head": head.stdout.strip(),
                "scripts_dirty": bool(dirty.stdout.strip()), "delegated": bel.read_json(bel.DELEGATED_JSON, {}),
                "runs": runs, "train_images": len(lines), "pinned_quick_gate": fp.PINNED, "files": files}
    (dest / "package_manifest.json").write_text(json.dumps(manifest, indent=1), encoding="utf-8")
    print(f"[package] done: {len(files)} files in {time.time() - t0:.0f}s; runs: {', '.join(runs)}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
