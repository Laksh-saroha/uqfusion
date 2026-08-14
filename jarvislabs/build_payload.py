"""Build the JarvisLabs L4 payload for the yolo26x tail of the Phase 1 grid.

Produces, under jarvislabs/payload/ (all tiny — no images are copied):

    repo/runs/derived/data_vis_stride2.yaml       remote dataset yaml (Linux paths)
    repo/runs/derived/data_vis_train_stride2.txt  48136 train frames, remote paths
    repo/data/pohang/visible/val.txt              11352 val frames, remote paths
    manifest/group_NN.txt                         tar -T lists for the chunked upload
    manifest/expect.tsv                           per-run file counts + bytes (upload check)

The frame sets are COPIED from the laptop's existing lists, never re-derived, so
the split cannot drift. The script then recomputes `split_fingerprint()` on the
*remote* yaml and refuses to write anything unless it equals the fingerprint every
row in the benchmark CSV carries (682dbe9f0f05) — grid.py aborts on a mismatch, and
finding that out after a 2 h upload onto a rented GPU is exactly the waste to avoid.

Usage:
    python jarvislabs/build_payload.py [--remote-root /home/uqfusion] [--group-bytes 1073741824]
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

# Every row in runs/benchmark/benchmark_results_tail.csv carries this. The whole
# point of the L4 machine is to add rows to that same experiment.
EXPECTED_FINGERPRINT = "682dbe9f0f05"

LOCAL_VIS = REPO / "Pohang_dataset" / "visible"
LOCAL_TRAIN_LIST = REPO / "runs" / "derived" / "data_vis_train_stride2.txt"
LOCAL_VAL_LIST = LOCAL_VIS / "val.txt"


def read_list(path: Path, base: Path) -> list[Path]:
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            p = Path(line)
            out.append(p if p.is_absolute() else (base / p))
    return out


def rel_to_visible(p: Path) -> str:
    """`images/pohang00/pohang00_L_000006.png` — the path shape both sides share."""
    parts = [x for x in str(p).replace("\\", "/").split("/") if x]
    for i, part in enumerate(parts):
        if part == "images":
            return "/".join(parts[i:])
    raise ValueError(f"not under an images/ directory: {p}")


def label_rel(img_rel: str) -> str:
    return "labels/" + img_rel.split("images/", 1)[1].rsplit(".", 1)[0] + ".txt"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--remote-root", default="/home/uqfusion",
                    help="repo root on the instance (dataset goes to <root>/data/pohang/visible)")
    ap.add_argument("--group-bytes", type=int, default=1024 ** 3,
                    help="target bytes per upload group (default 1 GiB)")
    args = ap.parse_args()

    remote_root = args.remote_root.rstrip("/")
    remote_vis = f"{remote_root}/data/pohang/visible"

    train = read_list(LOCAL_TRAIN_LIST, LOCAL_VIS)
    val = read_list(LOCAL_VAL_LIST, LOCAL_VIS)
    print(f"[payload] train {len(train)} frames, val {len(val)} frames")

    train_rel = [rel_to_visible(p) for p in train]
    val_rel = [rel_to_visible(p) for p in val]

    out = REPO / "jarvislabs" / "payload"
    repo_out = out / "repo"
    (repo_out / "runs" / "derived").mkdir(parents=True, exist_ok=True)
    (repo_out / "data" / "pohang" / "visible").mkdir(parents=True, exist_ok=True)
    man = out / "manifest"
    man.mkdir(parents=True, exist_ok=True)

    train_txt = repo_out / "runs" / "derived" / "data_vis_train_stride2.txt"
    val_txt = repo_out / "data" / "pohang" / "visible" / "val.txt"
    # "\n" newlines and no BOM: these are read by Linux tools and by ultralytics.
    train_txt.write_text("".join(f"{remote_vis}/{r}\n" for r in train_rel), encoding="utf-8", newline="\n")
    val_txt.write_text("".join(f"{remote_vis}/{r}\n" for r in val_rel), encoding="utf-8", newline="\n")

    # No `test:` key: Phase 1 never touches test and its images are not uploaded,
    # so a yaml that promised one would only hand a later session a broken path.
    yaml_path = repo_out / "runs" / "derived" / "data_vis_stride2.yaml"
    yaml_path.write_text(
        f"train: {remote_root}/runs/derived/data_vis_train_stride2.txt\n"
        f"val: {remote_vis}/val.txt\n"
        "names:\n  0: ship\n  1: buoy\n",
        encoding="utf-8", newline="\n",
    )

    from uqfusion.bench.grid import split_fingerprint

    # The real yaml points at /home/uqfusion/... which this machine cannot open, so
    # fingerprint a twin that points at the staged lists. Only the ids *inside* the
    # lists feed the hash (run_key + filename, both machine-independent), so the twin
    # and the yaml the instance will use fingerprint identically.
    verify_yaml = out / "_verify_local.yaml"
    verify_yaml.write_text(
        f"train: {train_txt.as_posix()}\nval: {val_txt.as_posix()}\n"
        "names:\n  0: ship\n  1: buoy\n",
        encoding="utf-8", newline="\n",
    )
    fp = split_fingerprint(verify_yaml)
    print(f"[payload] remote yaml split_fingerprint = {fp} (expected {EXPECTED_FINGERPRINT})")
    if fp != EXPECTED_FINGERPRINT:
        print("[payload] FINGERPRINT MISMATCH — the remote split is not the experiment's split. "
              "Nothing further written; fix before uploading.", file=sys.stderr)
        return 1

    # --- upload groups -----------------------------------------------------
    # One tar stream per group, images and their labels together, so an
    # interrupted upload replays one group (~1 GiB, ~8 min at 2 MB/s) not 16 GB.
    seen: set[str] = set()
    members: list[tuple[str, int]] = []
    missing_labels = []
    for rel in train_rel + val_rel:
        if rel in seen:
            continue
        seen.add(rel)
        img = LOCAL_VIS / rel
        lab_rel = label_rel(rel)
        lab = LOCAL_VIS / lab_rel
        if not lab.is_file():
            missing_labels.append(lab_rel)
            continue
        members.append((rel, os.path.getsize(img)))
        members.append((lab_rel, os.path.getsize(lab)))

    if missing_labels:
        print(f"[payload] {len(missing_labels)} images have no label file, e.g. {missing_labels[:3]}",
              file=sys.stderr)
        return 1

    groups: list[list[str]] = [[]]
    acc = 0
    for rel, size in members:
        if acc and acc + size > args.group_bytes:
            groups.append([])
            acc = 0
        groups[-1].append(rel)
        acc += size

    for i, g in enumerate(groups):
        (man / f"group_{i:02d}.txt").write_text("".join(f"{x}\n" for x in g),
                                                encoding="utf-8", newline="\n")

    total = sum(s for _, s in members)
    print(f"[payload] {len(members)} files, {total / 1e9:.2f} GB, {len(groups)} groups "
          f"of ~{args.group_bytes / 1e9:.2f} GB")

    # Per-run expectations the remote side re-counts after extraction: a silently
    # truncated tar shows up here, not 40 epochs later.
    counts: dict[str, list[int]] = {}
    for rel, size in members:
        run = rel.split("/")[1]
        kind = rel.split("/")[0]
        key = f"{kind}/{run}"
        c = counts.setdefault(key, [0, 0])
        c[0] += 1
        c[1] += size
    with open(man / "expect.tsv", "w", encoding="utf-8", newline="\n") as f:
        f.write("dir\tfiles\tbytes\n")
        for key in sorted(counts):
            f.write(f"{key}\t{counts[key][0]}\t{counts[key][1]}\n")
        f.write(f"TOTAL\t{len(members)}\t{total}\n")

    print(f"[payload] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
