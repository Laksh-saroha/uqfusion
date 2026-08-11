"""Build a FULL-RESOLUTION twin of the 640x640 Pohang dataset, on the D: drive.

The 640 dataset (`Pohang_dataset/`) is the trainable one, but its images were
letterboxed down from 2048x1080 (VIS) and 640x512 (IR). `imgsz` is the small-object
lever (progress.md OQ-7), so this builds a second tree at NATIVE resolution carrying
the SAME split and the SAME night-box filter, letting a higher-imgsz run be compared
directly against the 31-variant grid.

Three operations, mirroring `prepare_pohang.py` + `resplit_balanced.py` +
`filter_night_boxes.py`, but ported rather than recomputed:

  1. **Layout** — the D: sources are split-baked-into-folders
     (`{images,labels}/{train,val,test}/`). Rebuild as the canonical per-run layout
     (`{images,labels}/pohang{NN}/` + `{train,val,test}.txt` lists) so future
     re-splits are list-only, exactly like the 640 tree.

     Images are HARDLINKED (`os.link`) — a second directory entry pointing at the
     same blocks, so 309 GB of VIS costs 0 new bytes and the sources stay intact.
     Labels are REAL COPIES, because step 3 rewrites them and an in-place edit
     through a hardlink would corrupt the source too.

  2. **Split** — the balanced K-block lists from the 640 tree are copied VERBATIM.
     They are already in `./images/pohang{NN}/<stem>.png` form, which resolves
     unchanged here. Verbatim copy proves identical membership AND ordering, so no
     new leakage/balance/coverage gates need re-running.

  3. **Night filter** — the 17,502 emptied label files are ported from the 640 run's
     manifest, NOT recomputed.

     Recomputing would be wrong. `filter_night_boxes.py` finds the "content region"
     with PAD_LEVEL=4, but the VIS letterbox pads with 114, so padding was never
     excluded: 47% of every 640 VIS frame sits at exactly 114 and dominates the
     median. Every day-run frame therefore reports a median of exactly 114.0
     (pohang00/02/03/04: min=max=114 across 77,449 frames); only pohang01 reports
     real values. The rule that actually ran was "pohang01 frames whose content
     ~95th-percentile luminance < 100".

     Native images have no padding, so the same statistic means something else --
     measured, a bright day frame is 33 and a night frame is 8, both under 100.
     Re-running `--cut-dark pohang01:100` here would empty essentially ALL pohang01
     train labels instead of 17,515/18,826. Porting the file list is the only way to
     keep the two trees semantically identical.

Guard-band frames (dropped by the balanced re-split, in no list) are still
materialized on disk, matching the 640 tree.

Default is a dry run that verifies every precondition and writes nothing.
Run with --execute to build, --verify to re-check an existing tree, --revert to remove it.

Usage:
    python scripts/prepare_pohang_fullres.py                 # dry run + preflight
    python scripts/prepare_pohang_fullres.py --execute
    python scripts/prepare_pohang_fullres.py --verify
    python scripts/prepare_pohang_fullres.py --revert
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

# --- defaults: this is a one-time D:-drive migration, so the machine paths live here
#     as overridable CLI defaults rather than in config.yaml (which describes the
#     repo-local 640 dataset only).
DEF_SRC_VIS = r"D:\Datasets\Pohang_YOLO\visible_old"
DEF_SRC_IR = r"D:\Datasets\Pohang labels\infrared_orig"
DEF_REF_ROOT = r"A:\Uncertain\Pohang_dataset"
DEF_REF_FILTER = r"A:\Uncertain\runs\visfilter\visfilter_manifest.json"
DEF_OUT = r"D:\Datasets\Pohang_dataset_full"

SPLITS = ("train", "val", "test")
MODALITIES = ("visible", "infrared")
BACKUP_SUFFIX = ".pre_visfilter"
RUN_RE = re.compile(r"^(pohang\d{2})_")
EXPECT = {"visible": 127_309, "infrared": 31_010}
EXPECT_DIMS = {"visible": (2048, 1080), "infrared": (640, 512)}

# `path` MUST be absolute. Ultralytics resolves a relative `path:` against its own
# SETTINGS['datasets_dir'], NOT against the yaml's folder — a relative path here
# silently points at the wrong disk. This is the one line to edit per machine.
YAML_TMPL = """# Pohang full-resolution ({dims}). Edit `path` when this folder moves machines.
path: {root}
train: train.txt
val: val.txt
test: test.txt
names:
  0: ship
  1: buoy
"""


# ----------------------------------------------------------------------------- utils
def run_of(stem: str) -> str:
    m = RUN_RE.match(stem)
    if not m:
        raise ValueError(f"cannot derive run from stem {stem!r}")
    return m.group(1)


def free_bytes(path: Path) -> int:
    return shutil.disk_usage(path).free


def tree_hash(paths: list[Path]) -> str:
    """Content hash over a label set, stable under path changes (keyed by name)."""
    h = hashlib.sha256()
    for p in sorted(paths, key=lambda x: x.name):
        h.update(p.name.encode())
        h.update(p.read_bytes() if p.is_file() else b"")
        h.update(b"\x00")
    return h.hexdigest()[:12]


def n_lines(p: Path) -> int:
    if not p.is_file():
        return 0
    return len([ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()])


def same_inode(a: Path, b: Path) -> bool:
    try:
        sa, sb = a.stat(), b.stat()
    except OSError:
        return False
    return sa.st_ino == sb.st_ino and sa.st_dev == sb.st_dev


def label_for(img: Path) -> Path:
    """Ultralytics rule: last /images/ -> /labels/, extension -> .txt."""
    s = str(img)
    for sa, sb in ((f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"),
                   ("/images/", "/labels/")):
        if sa in s:
            head, _, tail = s.rpartition(sa)
            return Path(head + sb + tail).with_suffix(".txt")
    return img.with_suffix(".txt")


# ------------------------------------------------------------------------- inventory
def inventory(src: Path) -> dict[str, dict]:
    """stem -> {img, lbl, orig_split, run} for one modality's split-foldered source."""
    out: dict[str, dict] = {}
    for split in SPLITS:
        img_dir, lbl_dir = src / "images" / split, src / "labels" / split
        if not img_dir.is_dir():
            raise SystemExit(f"[fatal] missing source dir: {img_dir}")
        for name in os.listdir(img_dir):
            stem = Path(name).stem
            if stem in out:
                raise SystemExit(f"[fatal] duplicate stem across splits: {stem}")
            out[stem] = {"img": img_dir / name, "lbl": lbl_dir / f"{stem}.txt",
                         "orig_split": split, "run": run_of(stem)}
    return out


def read_lists(ref_root: Path) -> dict[str, dict[str, list[str]]]:
    """modality -> split -> verbatim lines of the 640 tree's split lists."""
    out: dict[str, dict[str, list[str]]] = {}
    for mod in MODALITIES:
        out[mod] = {}
        for split in SPLITS:
            p = ref_root / mod / f"{split}.txt"
            if not p.is_file():
                raise SystemExit(f"[fatal] missing reference list: {p}")
            out[mod][split] = [ln for ln in p.read_text(encoding="utf-8").splitlines()
                               if ln.strip()]
    return out


def read_drops(manifest: Path) -> tuple[dict[str, list[int]], dict]:
    """(stem -> dropped line indices, raw manifest) from the 640 run's visfilter run."""
    m = json.loads(manifest.read_text(encoding="utf-8"))
    if m.get("mode") != "cut_dark":
        raise SystemExit(f"[fatal] unexpected visfilter mode {m.get('mode')!r}")
    return {Path(k).stem: v for k, v in m["dropped"].items()}, m


# -------------------------------------------------------------------------- preflight
def preflight(args, inv, lists, drops, ref_manifest) -> dict:
    print("\n=== preflight ===")
    problems: list[str] = []
    report: dict = {}

    for mod in MODALITIES:
        n = len(inv[mod])
        exp = EXPECT[mod]
        flag = "OK " if n == exp else "!! "
        print(f"[{flag}] {mod}: {n} source frames (expected {exp})")
        if n != exp:
            problems.append(f"{mod}: {n} frames, expected {exp}")
        missing_lbl = [s for s, d in inv[mod].items() if not d["lbl"].is_file()]
        if missing_lbl:
            print(f"[   ] {mod}: {len(missing_lbl)} frames without a label file "
                  f"(YOLO background frames — allowed)")
        report[f"{mod}_frames"] = n
        report[f"{mod}_missing_labels"] = len(missing_lbl)

    # every referenced stem must exist in the source tree
    for mod in MODALITIES:
        for split in SPLITS:
            stems = [Path(ln).stem for ln in lists[mod][split]]
            miss = [s for s in stems if s not in inv[mod]]
            flag = "OK " if not miss else "!! "
            print(f"[{flag}] list {mod}/{split}: {len(stems)} entries, {len(miss)} unresolvable")
            if miss:
                problems.append(f"{mod}/{split}: {len(miss)} list entries missing "
                                f"(e.g. {miss[:3]})")
            report[f"list_{mod}_{split}"] = len(stems)
        # lists must be disjoint
        seen: Counter = Counter()
        for split in SPLITS:
            seen.update(Path(ln).stem for ln in lists[mod][split])
        dupes = [s for s, c in seen.items() if c > 1]
        flag = "OK " if not dupes else "!! "
        print(f"[{flag}] list {mod}: cross-split duplicates = {len(dupes)}")
        if dupes:
            problems.append(f"{mod}: {len(dupes)} stems in more than one list")

    # night filter: every drop target must exist, be VIS train, and be a whole-frame drop
    print(f"[   ] visfilter: {len(drops)} label files to empty "
          f"(manifest {ref_manifest['date']}, cuts {ref_manifest['cuts']})")
    train_stems = {Path(ln).stem for ln in lists["visible"]["train"]}
    bad_missing, bad_not_train, bad_partial, bad_run = [], [], [], []
    total_boxes = 0
    for stem, idxs in drops.items():
        d = inv["visible"].get(stem)
        if d is None:
            bad_missing.append(stem); continue
        if stem not in train_stems:
            bad_not_train.append(stem)
        if d["run"] != "pohang01":
            bad_run.append(stem)
        n = n_lines(d["lbl"])
        total_boxes += n
        if n != len(idxs):
            bad_partial.append((stem, n, len(idxs)))
    for label, bad in (("targets missing from source", bad_missing),
                       ("targets not in the VIS train list", bad_not_train),
                       ("targets outside pohang01", bad_run),
                       ("targets where source line count != dropped count", bad_partial)):
        flag = "OK " if not bad else "!! "
        print(f"[{flag}] visfilter: {len(bad)} {label}")
        if bad:
            problems.append(f"visfilter: {len(bad)} {label} (e.g. {bad[:3]})")
    print(f"[   ] visfilter: {total_boxes} boxes would be dropped "
          f"(640 run dropped {ref_manifest['boxes_dropped']})")
    report["visfilter_files"] = len(drops)
    report["visfilter_boxes"] = total_boxes
    report["visfilter_boxes_640"] = ref_manifest["boxes_dropped"]

    # hardlink capability + space
    out = Path(args.out)
    probe_root = out.parent if out.parent.exists() else Path(out.anchor)
    src_probe = next(iter(inv["visible"].values()))["img"]
    probe = probe_root / f"._hardlink_probe_{os.getpid()}"
    try:
        os.link(src_probe, probe)
        ok = same_inode(src_probe, probe)
        probe.unlink()
        print(f"[{'OK ' if ok else '!! '}] hardlink on {probe_root.anchor}: "
              f"works, same inode={ok}")
        if not ok:
            problems.append("hardlink did not share an inode")
    except OSError as e:
        print(f"[!! ] hardlink probe failed: {e}")
        problems.append(f"hardlink unsupported: {e}")

    lbl_bytes = sum(d["lbl"].stat().st_size for d in inv["visible"].values()
                    if d["lbl"].is_file())
    lbl_bytes += sum(d["lbl"].stat().st_size for d in inv["infrared"].values()
                     if d["lbl"].is_file())
    need = int(lbl_bytes * 2.2)  # labels + .pre_visfilter backups + manifests
    have = free_bytes(probe_root)
    flag = "OK " if have > need else "!! "
    print(f"[{flag}] space on {probe_root.anchor}: need ~{need/1e6:.0f} MB "
          f"(labels only), have {have/1e9:.1f} GB")
    if have <= need:
        problems.append("insufficient free space")
    report["label_bytes"] = lbl_bytes

    if out.exists() and any(out.iterdir()) and not args.force:
        print(f"[!! ] output exists and is non-empty: {out}")
        problems.append(f"output {out} exists (use --force to continue/resume)")
    else:
        print(f"[OK ] output: {out}")

    report["problems"] = problems
    print("\n=== preflight verdict ===")
    if problems:
        print(f"FAIL — {len(problems)} problem(s):")
        for p in problems:
            print(f"  - {p}")
    else:
        print("PASS — all preconditions met")
    return report


# ---------------------------------------------------------------------------- build
def materialize(inv: dict[str, dict], out: Path, mod: str, workers: int) -> dict:
    """Hardlink images, copy labels, into per-run folders. Idempotent."""
    items = sorted(inv.items())
    linked = copied = skipped_img = skipped_lbl = no_label = 0
    t0 = time.time()

    for run in sorted({d["run"] for d in inv.values()}):
        (out / mod / "images" / run).mkdir(parents=True, exist_ok=True)
        (out / mod / "labels" / run).mkdir(parents=True, exist_ok=True)

    def one(item):
        stem, d = item
        di = out / mod / "images" / d["run"] / d["img"].name
        dl = out / mod / "labels" / d["run"] / f"{stem}.txt"
        r = [0, 0, 0, 0, 0]
        if di.exists():
            r[2] = 1
        else:
            os.link(d["img"], di)
            r[0] = 1
        if not d["lbl"].is_file():
            r[4] = 1
        elif dl.exists():
            r[3] = 1
        else:
            shutil.copy2(d["lbl"], dl)
            r[1] = 1
        return r

    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(one, items), 1):
            linked += r[0]; copied += r[1]
            skipped_img += r[2]; skipped_lbl += r[3]; no_label += r[4]
            if i % 20000 == 0:
                print(f"    [{mod}] {i}/{len(items)} ({time.time() - t0:.0f}s)")

    print(f"  [{mod}] images: {linked} linked, {skipped_img} already present")
    print(f"  [{mod}] labels: {copied} copied, {skipped_lbl} already present, "
          f"{no_label} source frames have no label (background)")
    return {"linked": linked, "copied": copied, "skipped_images": skipped_img,
            "skipped_labels": skipped_lbl, "frames_without_label": no_label,
            "seconds": round(time.time() - t0, 1)}


def apply_filter(out: Path, inv_vis: dict, drops: dict[str, list[int]]) -> dict:
    """Empty the ported label files; back each up once."""
    rewritten = already = missing = 0
    boxes = 0
    per_class: Counter = Counter()
    names = {"0": "ship", "1": "buoy"}
    for stem, idxs in sorted(drops.items()):
        d = inv_vis[stem]
        p = out / "visible" / "labels" / d["run"] / f"{stem}.txt"
        if not p.is_file():
            missing += 1
            continue
        bak = p.with_suffix(p.suffix + BACKUP_SUFFIX)
        if bak.exists():
            already += 1
            continue
        lines = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]
        for ln in lines:
            per_class[names.get(ln.split()[0], ln.split()[0])] += 1
        boxes += len(lines)
        shutil.copy2(p, bak)
        p.write_text("", encoding="utf-8")
        rewritten += 1
    print(f"  [visfilter] {rewritten} emptied, {already} already filtered, "
          f"{missing} missing; {boxes} boxes dropped {dict(per_class)}")
    return {"files_rewritten": rewritten, "already_filtered": already,
            "missing": missing, "boxes_dropped": boxes,
            "boxes_dropped_per_class": dict(per_class)}


def build(args, inv, lists, drops, ref_manifest) -> dict:
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    prep = out / "_prep"
    prep.mkdir(exist_ok=True)
    started = datetime.now().isoformat(timespec="seconds")
    result: dict = {"started": started, "out": str(out),
                    "sources": {"visible": args.src_vis, "infrared": args.src_ir},
                    "reference_640": args.ref_root,
                    "reference_visfilter": args.ref_filter}

    # 0. snapshot the original hash-based split before the layout changes
    orig = {mod: {s: d["orig_split"] for s, d in inv[mod].items()} for mod in MODALITIES}
    (prep / "original_split.json").write_text(json.dumps(orig, indent=0), encoding="utf-8")
    print(f"[0/5] original hash-split membership snapshot -> _prep/original_split.json "
          f"({sum(len(v) for v in orig.values())} stems)")

    # 1. layout
    print("[1/5] materializing per-run tree (hardlink images, copy labels)")
    result["materialize"] = {m: materialize(inv[m], out, m, args.workers) for m in MODALITIES}

    # 2. split lists, verbatim
    print("[2/5] porting split lists from the 640 tree (verbatim)")
    listinfo = {}
    for mod in MODALITIES:
        for split in SPLITS:
            body = "\n".join(lists[mod][split]) + "\n"
            (out / mod / f"{split}.txt").write_text(body, encoding="utf-8")
            listinfo[f"{mod}/{split}"] = len(lists[mod][split])
        print(f"  [{mod}] train/val/test = "
              f"{listinfo[f'{mod}/train']}/{listinfo[f'{mod}/val']}/{listinfo[f'{mod}/test']}")
    result["lists"] = listinfo

    # 3. night filter, ported
    print("[3/5] applying the ported night-box filter")
    hash_before = tree_hash([out / "visible" / "labels" / inv["visible"][Path(ln).stem]["run"]
                             / f"{Path(ln).stem}.txt" for ln in lists["visible"]["train"]])
    result["visfilter"] = apply_filter(out, inv["visible"], drops)
    hash_after = tree_hash([out / "visible" / "labels" / inv["visible"][Path(ln).stem]["run"]
                            / f"{Path(ln).stem}.txt" for ln in lists["visible"]["train"]])
    result["visfilter"]["train_label_hash_before"] = hash_before
    result["visfilter"]["train_label_hash_after"] = hash_after
    result["visfilter"]["ported_from"] = args.ref_filter
    result["visfilter"]["source_640_hashes"] = {
        "before": ref_manifest["train_label_hash_before"],
        "after": ref_manifest["train_label_hash_after"]}
    print(f"  [visfilter] VIS train label hash {hash_before} -> {hash_after}")
    (prep / "visfilter_manifest.json").write_text(
        json.dumps({"ported_from": args.ref_filter, "cuts": ref_manifest["cuts"],
                    "rationale": "ported file list, not recomputed — the 640 threshold "
                                 "is not transferable (letterbox padding dominated the "
                                 "median; see script docstring)",
                    "date": datetime.now().isoformat(timespec="seconds"),
                    **result["visfilter"],
                    "dropped": {k: v for k, v in sorted(drops.items())}}, indent=1),
        encoding="utf-8")

    # 4. yamls + meta/paired
    print("[4/5] writing dataset yamls, copying meta/ and paired/")
    for mod, alias in (("visible", "vis"), ("infrared", "ir")):
        w, h = EXPECT_DIMS[mod]
        (out / f"data_{alias}.yaml").write_text(
            YAML_TMPL.format(root=(out / mod).resolve(), dims=f"{mod} {w}x{h}"),
            encoding="utf-8")
    ref_root = Path(args.ref_root)
    for sub in ("meta", "paired"):
        src, dst = ref_root / sub, out / sub
        if src.is_dir() and not dst.exists():
            shutil.copytree(src, dst)
            print(f"  copied {sub}/ from the 640 tree")
        elif dst.exists():
            print(f"  {sub}/ already present")

    # 5. manifest
    result["finished"] = datetime.now().isoformat(timespec="seconds")
    (prep / "build_manifest.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(f"[5/5] build manifest -> _prep/build_manifest.json")
    return result


# --------------------------------------------------------------------------- verify
def verify(args, inv, lists, drops) -> int:
    from PIL import Image
    out = Path(args.out)
    print(f"\n=== verify {out} ===")
    fails: list[str] = []
    lines_out: list[str] = []

    def check(ok: bool, msg: str):
        tag = "OK " if ok else "FAIL"
        print(f"[{tag}] {msg}")
        lines_out.append(f"[{tag}] {msg}")
        if not ok:
            fails.append(msg)

    # 1. counts on disk
    for mod in MODALITIES:
        n_img = sum(len(os.listdir(p)) for p in (out / mod / "images").iterdir())
        n_lbl = sum(len([f for f in os.listdir(p) if f.endswith(".txt")])
                    for p in (out / mod / "labels").iterdir())
        check(n_img == EXPECT[mod], f"{mod}: {n_img} images on disk (expect {EXPECT[mod]})")
        check(n_lbl <= EXPECT[mod], f"{mod}: {n_lbl} label files on disk")

    # 2. every list entry resolves, image AND label path
    for mod in MODALITIES:
        for split in SPLITS:
            listed = (out / mod / f"{split}.txt").read_text(encoding="utf-8").split()
            bad_img = [x for x in listed if not (out / mod / x.lstrip("./")).is_file()]
            # a label may legitimately be absent (YOLO background frame) — but only if
            # it was absent in the source too; a label the source HAS must be here.
            bad_lbl = [x for x in listed
                       if inv[mod][Path(x).stem]["lbl"].is_file()
                       and not label_for(out / mod / x.lstrip("./")).is_file()]
            check(not bad_img,
                  f"{mod}/{split}: {len(listed)} listed, {len(bad_img)} images unresolvable")
            check(not bad_lbl,
                  f"{mod}/{split}: {len(bad_lbl)} labels missing that the source has")
            ref = len(lists[mod][split])
            check(len(listed) == ref, f"{mod}/{split}: count matches 640 tree ({ref})")

    # 3. sampled native dimensions + hardlink identity
    rng = random.Random(args.seed)
    for mod in MODALITIES:
        sample = rng.sample(sorted(inv[mod]), min(args.samples, len(inv[mod])))
        bad_dim, bad_link = [], []
        for stem in sample:
            d = inv[mod][stem]
            p = out / mod / "images" / d["run"] / d["img"].name
            if Image.open(p).size != EXPECT_DIMS[mod]:
                bad_dim.append(stem)
            if not same_inode(p, d["img"]):
                bad_link.append(stem)
        check(not bad_dim, f"{mod}: {len(sample)} sampled images are "
                           f"{EXPECT_DIMS[mod][0]}x{EXPECT_DIMS[mod][1]} "
                           f"({len(bad_dim)} wrong)")
        check(not bad_link, f"{mod}: {len(sample)} sampled images share the source "
                            f"inode ({len(bad_link)} not hardlinked)")

    # 4. label coordinate bounds (sampled) — native coords must be untouched
    for mod in MODALITIES:
        sample = rng.sample(sorted(inv[mod]), min(args.samples, len(inv[mod])))
        oob = 0
        for stem in sample:
            d = inv[mod][stem]
            p = out / mod / "labels" / d["run"] / f"{stem}.txt"
            if not p.is_file():
                continue
            for ln in p.read_text(encoding="utf-8").splitlines():
                if not ln.strip():
                    continue
                v = [float(x) for x in ln.split()[1:5]]
                if any(x < 0 or x > 1 for x in v):
                    oob += 1
        check(oob == 0, f"{mod}: {oob} sampled boxes outside [0,1]")

    # 5. eval labels are byte-identical to source (the filter must never touch them)
    for mod in MODALITIES:
        for split in ("val", "test"):
            stems = [Path(x).stem for x in
                     (out / mod / f"{split}.txt").read_text(encoding="utf-8").split()]
            src_h = tree_hash([inv[mod][s]["lbl"] for s in stems])
            new_h = tree_hash([out / mod / "labels" / inv[mod][s]["run"] / f"{s}.txt"
                               for s in stems])
            check(src_h == new_h,
                  f"{mod}/{split}: labels byte-identical to source ({src_h} vs {new_h})")

    # 6. filter applied exactly, and only where intended
    empties, missing_bak, wrong_bak = 0, 0, 0
    for stem, idxs in drops.items():
        d = inv["visible"][stem]
        p = out / "visible" / "labels" / d["run"] / f"{stem}.txt"
        bak = p.with_suffix(p.suffix + BACKUP_SUFFIX)
        if p.is_file() and p.stat().st_size == 0:
            empties += 1
        if not bak.is_file():
            missing_bak += 1
        elif n_lines(bak) != len(idxs):
            wrong_bak += 1
    check(empties == len(drops), f"visfilter: {empties}/{len(drops)} targets are empty")
    check(missing_bak == 0, f"visfilter: {missing_bak} targets without a backup")
    check(wrong_bak == 0, f"visfilter: {wrong_bak} backups with an unexpected line count")

    # no train label outside the drop set may have been emptied
    non_targets_empty = 0
    for ln in (out / "visible" / "train.txt").read_text(encoding="utf-8").split():
        stem = Path(ln).stem
        if stem in drops:
            continue
        src_lbl = inv["visible"][stem]["lbl"]
        new_lbl = out / "visible" / "labels" / inv["visible"][stem]["run"] / f"{stem}.txt"
        if not src_lbl.is_file() or not new_lbl.is_file():
            continue  # background frame in the source — nothing to lose
        if src_lbl.stat().st_size > 0 and new_lbl.stat().st_size == 0:
            non_targets_empty += 1
    check(non_targets_empty == 0,
          f"visfilter: {non_targets_empty} non-target train labels wrongly emptied")

    # 7. run coverage per split
    for mod in MODALITIES:
        cov = defaultdict(Counter)
        for split in SPLITS:
            for x in (out / mod / f"{split}.txt").read_text(encoding="utf-8").split():
                cov[run_of(Path(x).stem)][split] += 1
        for run in sorted(cov):
            c = cov[run]
            print(f"       {mod}/{run}: train={c['train']} val={c['val']} test={c['test']}")
            lines_out.append(f"       {mod}/{run}: train={c['train']} val={c['val']} "
                             f"test={c['test']}")

    verdict = "PASS" if not fails else f"FAIL ({len(fails)})"
    print(f"\n=== verify verdict: {verdict} ===")
    rep = out / "_prep" / "verify_report.txt"
    rep.parent.mkdir(parents=True, exist_ok=True)
    rep.write_text(f"verify {datetime.now().isoformat(timespec='seconds')}\n"
                   + "\n".join(lines_out) + f"\n\nverdict: {verdict}\n", encoding="utf-8")
    print(f"report -> {rep}")
    return 0 if not fails else 1


# --------------------------------------------------------------------------- revert
def revert(args, inv) -> int:
    out = Path(args.out)
    if not out.exists():
        print(f"[revert] nothing to do: {out} does not exist")
        return 0
    print(f"[revert] target: {out}")

    # Safety: refuse if any image there is NOT a hardlink of a live source file.
    # (If it isn't, deleting the tree would destroy the only copy.)
    orphans, checked = [], 0
    for mod in MODALITIES:
        d_imgs = out / mod / "images"
        if not d_imgs.is_dir():
            continue
        for run_dir in d_imgs.iterdir():
            for p in run_dir.iterdir():
                stem = p.stem
                src = inv[mod].get(stem)
                checked += 1
                if src is None or not same_inode(p, src["img"]):
                    orphans.append(str(p))
                    if len(orphans) > 20:
                        break
    print(f"[revert] checked {checked} images; {len(orphans)} not backed by a live source")
    if orphans:
        print("[revert] REFUSING — these would be destroyed, not just unlinked:")
        for o in orphans[:20]:
            print(f"    {o}")
        print("[revert] the source tree must be intact before this folder can be removed.")
        return 1

    if not args.execute:
        print(f"[revert] dry run — would delete {out} "
              f"(images are hardlinks; sources at {args.src_vis} / {args.src_ir} keep the data)")
        print("[revert] re-run with --revert --execute to actually delete")
        return 0

    shutil.rmtree(out)
    print(f"[revert] deleted {out}; source trees untouched")
    return 0


# ------------------------------------------------------------------------------ main
def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src-vis", default=DEF_SRC_VIS)
    ap.add_argument("--src-ir", default=DEF_SRC_IR)
    ap.add_argument("--ref-root", default=DEF_REF_ROOT,
                    help="the 640 dataset — source of the split lists, meta/, paired/")
    ap.add_argument("--ref-filter", default=DEF_REF_FILTER,
                    help="the 640 run's visfilter manifest — source of the drop list")
    ap.add_argument("--out", default=DEF_OUT)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--samples", type=int, default=400, help="per-modality verify sample")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--force", action="store_true", help="continue into a non-empty --out")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true", help="build the tree")
    mode.add_argument("--verify", action="store_true", help="verify an existing tree")
    mode.add_argument("--revert", action="store_true",
                      help="delete the built tree (dry run unless --execute)")
    args = ap.parse_args()

    t0 = time.time()
    print("=== source inventory ===")
    inv = {"visible": inventory(Path(args.src_vis)),
           "infrared": inventory(Path(args.src_ir))}
    print(f"  visible : {len(inv['visible'])} frames  <- {args.src_vis}")
    print(f"  infrared: {len(inv['infrared'])} frames  <- {args.src_ir}")

    if args.revert:
        return revert(args, inv)

    lists = read_lists(Path(args.ref_root))
    drops, ref_manifest = read_drops(Path(args.ref_filter))

    if args.verify:
        return verify(args, inv, lists, drops)

    report = preflight(args, inv, lists, drops, ref_manifest)
    if report["problems"]:
        print("\n[abort] preflight failed — nothing written")
        return 1

    if not args.execute:
        out = Path(args.out)
        print(f"\n=== dry run — nothing written ===")
        print(f"  would create : {out}")
        print(f"  hardlink     : {sum(len(inv[m]) for m in MODALITIES)} images (0 new bytes)")
        print(f"  copy         : label files (~{report['label_bytes']/1e6:.0f} MB)")
        print(f"  lists        : verbatim from {args.ref_root}")
        print(f"  filter       : empty {report['visfilter_files']} VIS train labels "
              f"({report['visfilter_boxes']} boxes)")
        print(f"  re-run with --execute to build")
        return 0

    print(f"\n=== build ===")
    build(args, inv, lists, drops, ref_manifest)
    rc = verify(args, inv, lists, drops)
    print(f"\ntotal {time.time() - t0:.0f}s")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
