"""Verify the Pohang dataset on THIS machine carries every prep step we applied.

Independent re-implementation on purpose: it imports **nothing** from
``uqfusion`` (only pyyaml, plus PIL/numpy for the optional pixel check), so it
cross-checks the package rather than trusting it, and it runs from a bare copy
of this one file. The algorithms mirror, byte-for-byte in result:

  * ``uqfusion/data/lists.py``      — yaml `path:` resolution, split lists, run keys
  * ``uqfusion/data/audit.py``      — cross-split duplicates + temporal buffer
  * ``uqfusion/bench/grid.py``      — ``split_fingerprint`` (train+val id hash)
  * ``scripts/filter_night_boxes.py`` — ``label_content_hash``
  * ``scripts/resplit_balanced.py`` — class-share / box-density / coverage gates

What it checks (each prints PASS / FAIL / WARN / INFO):

  1. yaml resolves; `path:` is portable (no `A:/...` drive letter)
  2. class names
  3. split lists exist, are `./`-prefixed (the Ultralytics relative-line rule)
  4. every listed image exists; image -> label paths resolve
  5. label syntax: 5 fields, class id in range, coords in [0,1]
  6. split_fingerprint of the full yaml (and of a derived stride yaml)
  7. leakage: cross-split duplicates + per-run temporal guard band
  8. balance: per-split class share, boxes/frame, per-run val/test coverage
  9. night-box filter: applied? train-only? backups present?
 10. train-label content hash (opt-in: ``--hash-labels``)
 11. stale Ultralytics `*.cache` files
 12. IR letterbox geometry (640x640, 64 pad rows of 114) — IR only
 13. resume readiness against an existing benchmark CSV

Usage (from the repo root):

    python scripts/verify_dataset_state.py --data vis \
        --stride-yaml runs/derived/data_vis_stride2.yaml \
        --csv runs/benchmark/benchmark_results_ship_visfilter.csv

    python scripts/verify_dataset_state.py --data ir          # IR tree
    python scripts/verify_dataset_state.py --data vis --fast  # skip full label read

Exit 0 = no FAIL. Exit 1 = at least one FAIL.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import random
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from statistics import median

import yaml

SPLITS = ("train", "val", "test")
RUN_PATTERN = re.compile(r"(pohang\d{2})", re.IGNORECASE)
WINDOWS_ABS = re.compile(r"^[A-Za-z]:[\\/]")

# Reference values from the local prep runs (docs/dataset-changes-2026-07.md).
REF_VISFILTER_FILES = 17502
REF_VISFILTER_BOXES = 132688
REF_LABEL_HASH_AFTER = "287b11c50b5a"
REF_LABEL_HASH_BEFORE = "fd60c0834fdd"
PAD_VALUE = 114  # both modalities letterbox with gray 114

_results: list[tuple[str, str]] = []


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
def report(status: str, title: str, *lines: str) -> None:
    _results.append((status, title))
    print(f"[{status:4s}] {title}")
    for line in lines:
        print(f"         {line}")


def section(title: str) -> None:
    print()
    print(f"=== {title} " + "=" * max(0, 68 - len(title)))


# --------------------------------------------------------------------------
# list / path resolution (mirrors uqfusion/data/lists.py)
# --------------------------------------------------------------------------
def dataset_root(data: dict, yaml_path: Path) -> Path:
    root = Path(str(data.get("path", ".")))
    return root if root.is_absolute() else (yaml_path.parent / root).resolve()


def raw_lines(list_path: Path) -> list[str]:
    with open(list_path, "r", encoding="utf-8") as f:
        return [ln.strip() for ln in f if ln.strip()]


def split_entries(data: dict, split: str) -> list[Path]:
    entries = data[split] if isinstance(data[split], list) else [data[split]]
    return [Path(str(e)) for e in entries]


def split_image_list(data: dict, yaml_path: Path, split: str) -> tuple[list[Path], list[Path]]:
    """Return (image paths, txt list files used). Directory entries are expanded."""
    root = dataset_root(data, yaml_path)
    images: list[Path] = []
    lists: list[Path] = []
    for entry in split_entries(data, split):
        p = entry if entry.is_absolute() else (root / entry).resolve()
        if p.suffix.lower() == ".txt":
            if not p.is_file():
                raise FileNotFoundError(f"split list not found: {p}")
            lists.append(p)
            for line in raw_lines(p):
                q = Path(line)
                images.append(q if q.is_absolute() else (root / q).resolve())
        elif p.is_dir():
            images.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in
                                 {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}))
        else:
            raise FileNotFoundError(f"'{split}' entry is neither a txt list nor a directory: {p}")
    return images, lists


def run_key(image_path: Path) -> str:
    m = RUN_PATTERN.search(str(image_path))
    return m.group(1).lower() if m else image_path.parent.name


def frame_ordinal(image_path: Path) -> float | None:
    stem = image_path.stem
    try:
        return float(stem)
    except ValueError:
        pass
    runs = re.findall(r"\d+", stem)
    return float(runs[-1]) if runs else None


def label_path(img: Path) -> Path:
    """Ultralytics rule: swap the LAST `/images/` for `/labels/`, extension -> .txt."""
    s = str(img)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa not in s:
        sa, sb = "/images/", "/labels/"
    head, _, tail = s.rpartition(sa)
    if not head:
        return Path(s).with_suffix(".txt")
    return Path(head + sb + tail).with_suffix(".txt")


# --------------------------------------------------------------------------
# hashes (mirror grid.split_fingerprint / filter_night_boxes.label_content_hash)
# --------------------------------------------------------------------------
def split_fingerprint(images_by_split: dict[str, list[Path]]) -> str:
    h = hashlib.sha256()
    for split in ("train", "val"):
        ids = sorted(f"{run_key(p)}/{p.name}" for p in images_by_split[split])
        h.update(f"{split}:{len(ids)}\n".encode())
        h.update("\n".join(ids).encode())
    return h.hexdigest()[:12]


def label_content_hash(images: list[Path]) -> str:
    h = hashlib.sha256()
    for img in sorted(images):
        lp = label_path(img)
        h.update(f"{run_key(img)}/{lp.name}\n".encode())
        if lp.is_file():
            h.update(lp.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()[:12]


# --------------------------------------------------------------------------
# label reading
# --------------------------------------------------------------------------
class LabelStats:
    def __init__(self) -> None:
        self.frames = 0
        self.frames_with_boxes = 0
        self.frames_empty = 0        # label file missing or zero lines
        self.boxes = 0
        self.per_class: dict[int, int] = defaultdict(int)
        self.bad_lines: list[str] = []


def scan_labels(images: list[Path], n_names: int, per_run: bool = False
                ) -> tuple[LabelStats, dict[str, LabelStats]]:
    total = LabelStats()
    runs: dict[str, LabelStats] = defaultdict(LabelStats)
    t0 = time.time()
    for i, img in enumerate(images):
        lp = label_path(img)
        boxes = 0
        bad: list[str] = []
        if lp.is_file():
            try:
                text = lp.read_text(encoding="utf-8")
            except Exception as exc:  # noqa: BLE001 - unreadable label is a finding
                bad.append(f"{lp}: unreadable ({exc})")
                text = ""
            for ln, line in enumerate(text.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    bad.append(f"{lp}:{ln}: {len(parts)} fields, expected 5")
                    continue
                try:
                    cid = int(float(parts[0]))
                    vals = [float(v) for v in parts[1:]]
                except ValueError:
                    bad.append(f"{lp}:{ln}: non-numeric field")
                    continue
                if not 0 <= cid < n_names:
                    bad.append(f"{lp}:{ln}: class id {cid} outside 0..{n_names - 1}")
                if any(not (0.0 <= v <= 1.0) for v in vals):
                    bad.append(f"{lp}:{ln}: coordinate outside [0,1] -> {vals}")
                boxes += 1
                total.per_class[cid] += 1
                if per_run:
                    runs[run_key(img)].per_class[cid] += 1

        for tgt in (total, runs[run_key(img)]) if per_run else (total,):
            tgt.frames += 1
            tgt.boxes += boxes
            if boxes:
                tgt.frames_with_boxes += 1
            else:
                tgt.frames_empty += 1
            if bad and len(tgt.bad_lines) < 20:
                tgt.bad_lines.extend(bad[: 20 - len(tgt.bad_lines)])

        if i and i % 20000 == 0:
            print(f"         ... {i}/{len(images)} labels ({time.time() - t0:.0f}s)")
    return total, dict(runs)


# --------------------------------------------------------------------------
# checks
# --------------------------------------------------------------------------
def check_yaml(yaml_path: Path, data: dict, root: Path) -> int:
    section("1-2. dataset yaml")
    print(f"         yaml : {yaml_path}")
    print(f"         path : {data.get('path')!r} -> {root}")
    raw_path = str(data.get("path", "."))
    if WINDOWS_ABS.match(raw_path):
        report("FAIL", "yaml `path:` is a Windows drive-letter path",
               f"{raw_path!r} is absolute on Windows but RELATIVE on Linux — the loader will glue it",
               "onto the yaml folder. Set `path: visible` (or `path: infrared`) instead.")
    elif not root.is_dir():
        report("FAIL", "yaml `path:` does not resolve to a directory", str(root))
    else:
        report("PASS", f"yaml `path:` resolves ({raw_path!r})")

    names = data.get("names")
    if isinstance(names, dict):
        names = [names[k] for k in sorted(names)]
    if names == ["ship", "buoy"]:
        report("PASS", "class names == ['ship', 'buoy']")
    else:
        report("WARN", f"class names are {names} (expected ['ship', 'buoy'])")
    return len(names or [])


def check_lists(data: dict, yaml_path: Path) -> dict[str, list[Path]]:
    section("3-4. split lists, image existence, label resolution")
    images_by_split: dict[str, list[Path]] = {}
    for split in SPLITS:
        if data.get(split) is None:
            report("WARN", f"[{split}] no entry in the yaml")
            continue
        try:
            imgs, lists = split_image_list(data, yaml_path, split)
        except FileNotFoundError as exc:
            report("FAIL", f"[{split}] list not resolvable", str(exc))
            continue
        images_by_split[split] = imgs

        # `./` prefix rule: Ultralytics only re-anchors a line to the txt's own
        # folder when it starts with './'. Bare relative lines resolve against CWD.
        bad_prefix = 0
        sample_bad = ""
        for lp in lists:
            for line in raw_lines(lp):
                if not (line.startswith("./") or Path(line).is_absolute()):
                    bad_prefix += 1
                    sample_bad = sample_bad or f"{lp.name}: {line}"
        if lists:
            if bad_prefix:
                report("FAIL", f"[{split}] {bad_prefix} list lines lack the './' prefix",
                       f"e.g. {sample_bad}",
                       "Ultralytics resolves these against CWD -> 'No such file or directory'.",
                       "Fix: python scripts/fix_split_lists.py --data <vis|ir>")
            else:
                report("PASS", f"[{split}] all {len(imgs)} list lines are './'-prefixed or absolute")

        missing = [p for p in imgs if not p.is_file()]
        if missing:
            report("FAIL", f"[{split}] {len(missing)} of {len(imgs)} listed images do not exist",
                   *[f"missing: {p}" for p in missing[:3]])
        else:
            report("PASS", f"[{split}] {len(imgs)} listed images all exist on disk")

        no_label = sum(1 for p in imgs if not label_path(p).is_file())
        report("INFO", f"[{split}] {len(imgs) - no_label} frames have a label file, "
                       f"{no_label} have none (background frames are legal)")
    return images_by_split


def check_fingerprint(images_by_split: dict[str, list[Path]], expect: str | None,
                      label: str = "full yaml") -> str:
    if not {"train", "val"} <= images_by_split.keys():
        report("WARN", f"[{label}] fingerprint skipped — train and val both required")
        return ""
    fp = split_fingerprint(images_by_split)
    if expect is None:
        report("INFO", f"[{label}] split_fingerprint = {fp}",
               f"train={len(images_by_split['train'])} val={len(images_by_split['val'])}")
    elif fp == expect:
        report("PASS", f"[{label}] split_fingerprint {fp} matches the expected value")
    else:
        report("FAIL", f"[{label}] split_fingerprint {fp} != expected {expect}",
               "The grid will refuse to append to a CSV stamped with a different fingerprint.",
               "Do NOT regenerate split lists or the stride subset to 'fix' this — that is what",
               "changes the fingerprint. Restore the lists the finished rows were trained on.")
    return fp


def check_leakage(images_by_split: dict[str, list[Path]], min_gap_frames: int) -> None:
    section("7. leakage audit (cross-split duplicates + temporal guard band)")
    present = [s for s in SPLITS if s in images_by_split]

    dupes: list[str] = []
    stem_seen: dict[tuple[str, str], str] = {}
    for split in present:
        for img in images_by_split[split]:
            key = (run_key(img), img.stem.lower())
            if key in stem_seen and stem_seen[key] != split:
                dupes.append(f"stem '{key[1]}' of {key[0]} in both '{stem_seen[key]}' and '{split}'")
            stem_seen[key] = split
    if dupes:
        report("FAIL", f"{len(dupes)} cross-split duplicate frames", *dupes[:5])
    else:
        report("PASS", "0 cross-split duplicate frames")

    by_run: dict[str, list[tuple[float, str, Path]]] = defaultdict(list)
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in present})
    for split in present:
        for img in images_by_split[split]:
            matrix[run_key(img)][split] += 1
            o = frame_ordinal(img)
            if o is not None:
                by_run[run_key(img)].append((o, split, img))

    violations: list[str] = []
    for run, frames in sorted(by_run.items()):
        if len(frames) < 2:
            continue
        frames.sort(key=lambda t: t[0])
        deltas = [b[0] - a[0] for a, b in zip(frames, frames[1:]) if b[0] > a[0]]
        if not deltas:
            continue
        need = median(deltas) * min_gap_frames
        for a, b in zip(frames, frames[1:]):
            crosses = (a[1] == "train") != (b[1] == "train")
            if crosses and (b[0] - a[0]) < need:
                violations.append(f"{run}: gap {b[0] - a[0]:.6g} < required {need:.6g} "
                                  f"({a[1]} {a[2].name} -> {b[1]} {b[2].name})")
    if violations:
        report("FAIL", f"{len(violations)} temporal-proximity violations "
                       f"(min_gap_frames={min_gap_frames})", *violations[:5])
    else:
        report("PASS", f"0 temporal-proximity violations (min_gap_frames={min_gap_frames})")

    print("         run x split occupancy:")
    for run, counts in sorted(matrix.items()):
        print("           " + f"{run}: " + ", ".join(f"{s}={n}" for s, n in counts.items()))


def check_balance(stats: dict[str, LabelStats], per_run: dict[str, dict[str, LabelStats]],
                  names: int) -> None:
    section("8. balance + coverage (why the split was rebuilt on 2026-07-14)")
    total_boxes = sum(s.boxes for s in stats.values())
    global_share = defaultdict(int)
    for s in stats.values():
        for c, n in s.per_class.items():
            global_share[c] += n
    g_buoy = 100.0 * global_share.get(1, 0) / max(1, total_boxes)
    print(f"         global: {total_boxes} boxes, buoy share {g_buoy:.1f}%")

    ok = True
    for split, s in stats.items():
        buoy = 100.0 * s.per_class.get(1, 0) / max(1, s.boxes)
        dens = s.boxes / max(1, s.frames)
        print(f"         {split:5s}: {s.frames:7d} frames | {s.boxes:8d} boxes | "
              f"{dens:5.2f} boxes/frame | buoy {buoy:5.1f}% | {s.frames_empty} empty-label frames")
        if split in ("val", "test"):
            # the failure that forced the re-split: val was 77% buoy at 1.35 boxes/frame
            if abs(buoy - g_buoy) > 15.0:
                ok = False
                report("FAIL", f"[{split}] buoy share {buoy:.1f}% deviates >15pp from global {g_buoy:.1f}%")
            if s.boxes and dens < 0.4 * (total_boxes / max(1, sum(x.frames for x in stats.values()))):
                ok = False
                report("FAIL", f"[{split}] box density {dens:.2f}/frame is <40% of the global rate")
    if ok:
        report("PASS", "val/test class share and box density track the global distribution")

    if per_run:
        missing = []
        for run, per_split in sorted(per_run.items()):
            have = {s for s, st in per_split.items() if st.frames > 0}
            if not {"val", "test"} <= have:
                missing.append(f"{run}: only {sorted(have)}")
        if missing:
            report("WARN", "some runs do not contribute to every eval split", *missing)
        else:
            report("PASS", "every run contributes train + val + test frames")


def check_night_filter(images_by_split: dict[str, list[Path]],
                       per_run: dict[str, dict[str, LabelStats]] | None = None) -> None:
    section("9. night-box visibility filter (VIS train-only, pohang01)")
    train = [p for p in images_by_split.get("train", []) if run_key(p) == "pohang01"]
    if not train:
        report("INFO", "no pohang01 frames in train — filter check not applicable (IR tree?)")
        return

    cached = (per_run or {}).get("pohang01", {}).get("train")
    if cached is not None:                       # reuse the full label scan
        empty, boxes = cached.frames_empty, cached.boxes
    else:                                        # --fast: cheap emptiness probe only
        empty = sum(1 for p in train if not label_path(p).is_file()
                    or not label_path(p).read_text(encoding="utf-8").strip())
        boxes = None
    frac = empty / len(train)

    detail = [f"pohang01 train frames: {len(train)}",
              f"emptied (zero boxes) : {empty}  ({100 * frac:.1f}%)",
              f"reference local run  : {REF_VISFILTER_FILES} files emptied, "
              f"{REF_VISFILTER_BOXES} boxes dropped"]
    if boxes is not None:
        detail.append(f"boxes remaining      : {boxes}")

    if frac > 0.5:
        report("PASS", "night-box filter IS applied to VIS train", *detail)
    elif frac < 0.05:
        report("FAIL", "night-box filter is NOT applied to VIS train", *detail,
               "Fix: python scripts/filter_night_boxes.py --cut-dark pohang01:100 --execute",
               "then delete the stale *.cache files and use a FRESH --out-csv.")
    else:
        report("WARN", "night-box filter looks PARTIALLY applied", *detail)

    # train-only rule: eval labels must be untouched
    for split in ("val", "test"):
        ev = [p for p in images_by_split.get(split, []) if run_key(p) == "pohang01"]
        if not ev:
            continue
        ev_empty = sum(1 for p in ev if not label_path(p).is_file()
                       or not label_path(p).read_text(encoding="utf-8").strip())
        ev_frac = ev_empty / len(ev)
        if ev_frac > 0.5:
            report("FAIL", f"[{split}] {100 * ev_frac:.1f}% of pohang01 eval frames are empty",
                   "The filter is TRAIN-ONLY — emptied eval labels erase the 'VIS misses / IR",
                   "catches' result the whole benchmark measures. Restore from *.pre_visfilter.")
        else:
            report("PASS", f"[{split}] pohang01 eval labels intact "
                           f"({ev_empty}/{len(ev)} empty, natural background rate)")

    backups = 0
    for p in train[:2000]:
        if Path(str(label_path(p)) + ".pre_visfilter").is_file():
            backups += 1
    if frac > 0.5 and backups == 0:
        report("WARN", "no *.pre_visfilter backups found beside pohang01 train labels",
               "--restore will not work on this machine; the edit is one-way here.")
    elif backups:
        report("PASS", f"undo backups present ({backups} found in the first 2000 train frames)")


def check_label_hash(images_by_split: dict[str, list[Path]], expect: str | None) -> None:
    section("10. train-label content hash")
    train = images_by_split.get("train")
    if not train:
        report("WARN", "no train split — hash skipped")
        return
    h = label_content_hash(train)
    if expect and h == expect:
        report("PASS", f"train-label content hash = {h} (matches expected)")
    elif h == REF_LABEL_HASH_AFTER:
        report("PASS", f"train-label content hash = {h} — identical to the filtered local tree")
    elif h == REF_LABEL_HASH_BEFORE:
        report("FAIL", f"train-label content hash = {h} — this is the PRE-filter hash",
               "the night filter has not been applied to this copy")
    else:
        report("INFO", f"train-label content hash = {h}",
               f"local reference: {REF_LABEL_HASH_BEFORE} (before) -> {REF_LABEL_HASH_AFTER} (after)",
               "A mismatch is expected whenever the train frame LIST differs from the local one;",
               "the pohang01 empty-frame fraction above is the decisive filter check.")


def check_caches(root: Path) -> None:
    section("11. stale Ultralytics label caches")
    caches = sorted(root.rglob("*.cache"))
    if not caches:
        report("PASS", "no *.cache files — the trainer will do a fresh label scan")
        return
    newest_label = 0.0
    for lp in list(root.rglob("labels/*/*.txt"))[:5000]:
        newest_label = max(newest_label, lp.stat().st_mtime)
    stale = [c for c in caches if c.stat().st_mtime < newest_label]
    if stale:
        report("FAIL", f"{len(stale)} of {len(caches)} label caches predate the newest label file",
               *[f"stale: {c}" for c in stale[:5]],
               "Delete them or the trainer reads the OLD boxes: find <root> -name '*.cache' -delete")
    else:
        report("WARN", f"{len(caches)} label cache(s) present and newer than the labels",
               *[f"{c}" for c in caches[:5]],
               "Fine to keep, but delete them after any label edit.")


def check_letterbox(images_by_split: dict[str, list[Path]], n: int, seed: int) -> None:
    section("12. IR letterbox geometry (640x640, 64 pad rows of 114)")
    try:
        import numpy as np
        from PIL import Image
    except ImportError:
        report("WARN", "numpy/PIL unavailable — pixel check skipped")
        return
    pool = [p for imgs in images_by_split.values() for p in imgs]
    if not pool:
        report("WARN", "no images to sample")
        return
    rng = random.Random(seed)
    sample = rng.sample(pool, min(n, len(pool)))
    bad_size, bad_pad = [], []
    for p in sample:
        with Image.open(p) as im:
            if im.size != (640, 640):
                bad_size.append(f"{p.name}: {im.size}")
                continue
            a = np.asarray(im.convert("L"))
        if not (a[:64] == PAD_VALUE).all() or not (a[576:] == PAD_VALUE).all():
            bad_pad.append(p.name)
    if bad_size:
        report("FAIL", f"{len(bad_size)}/{len(sample)} sampled IR images are not 640x640", *bad_size[:5])
    elif bad_pad:
        report("FAIL", f"{len(bad_pad)}/{len(sample)} sampled images lack the 114 pad rows",
               *bad_pad[:5], "these look like unletterboxed 640x512 originals")
    else:
        report("PASS", f"{len(sample)} sampled IR images are 640x640 with 64+64 rows of {PAD_VALUE}")


def check_stride(stride_yaml: Path, full: dict[str, list[Path]], expect_fp: str | None) -> None:
    section("6b. derived stride subset")
    if not stride_yaml.is_file():
        report("FAIL", f"stride yaml not found: {stride_yaml}",
               "The grid rows were trained through this yaml. If it is gone, the run is not",
               "resumable into the same CSV — the regenerated subset gets a new fingerprint.")
        return
    with open(stride_yaml, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    data_path = stride_yaml.resolve()
    imgs: dict[str, list[Path]] = {}
    for split in ("train", "val"):
        try:
            imgs[split], _ = split_image_list(data, data_path, split)
        except (FileNotFoundError, KeyError, TypeError) as exc:
            report("FAIL", f"stride yaml [{split}] unresolvable", str(exc))
            return
    missing = [p for p in imgs["train"] if not p.is_file()]
    if missing:
        report("FAIL", f"{len(missing)} stride-train images do not exist", str(missing[0]))
    else:
        report("PASS", f"stride train list resolves ({len(imgs['train'])} frames)")

    if "train" in full:
        full_ids = {f"{run_key(p)}/{p.name}" for p in full["train"]}
        sub_ids = {f"{run_key(p)}/{p.name}" for p in imgs["train"]}
        extra = sub_ids - full_ids
        ratio = len(sub_ids) / max(1, len(full_ids))
        if extra:
            report("FAIL", f"{len(extra)} stride frames are not in the full train list",
                   "the subset was built from a DIFFERENT train split", next(iter(extra)))
        else:
            report("PASS", f"stride train is a subset of full train ({ratio:.2%} kept)")
    check_fingerprint(imgs, expect_fp, label="stride yaml")


def check_resume(csv_path: Path, stride_fp: str, variants: list[str], seeds: list[int]) -> None:
    section("13. resume readiness against the benchmark CSV")
    if not csv_path.is_file():
        report("FAIL", f"results CSV not found: {csv_path}")
        return
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        report("WARN", f"{csv_path} has no data rows")
        return
    fps = {r.get("split_fingerprint", "") for r in rows}
    classes = {r.get("classes", "") for r in rows}
    yamls = {r.get("data_yaml", "") for r in rows}
    epochs = {r.get("epochs_cfg", "") for r in rows}
    print(f"         rows              : {len(rows)}")
    print(f"         split_fingerprint : {sorted(fps)}")
    print(f"         classes           : {sorted(classes)}")
    print(f"         data_yaml         : {sorted(yamls)}")
    print(f"         epochs_cfg        : {sorted(epochs)}")

    if len(fps) != 1:
        report("FAIL", "the CSV mixes several split fingerprints — it is not one experiment")
    elif stride_fp and fps == {stride_fp}:
        report("PASS", f"CSV fingerprint {stride_fp} matches this machine — the grid will append")
    elif stride_fp:
        report("FAIL", f"CSV fingerprint {next(iter(fps))} != this machine's {stride_fp}",
               "run_grid() raises 'the CSV belongs to a different experiment' and writes nothing.",
               "Restore the exact lists that produced the CSV, or start a fresh --out-csv",
               "(which throws away the finished runs).")

    done = {(r["variant"], r["seed"]) for r in rows}
    todo = [(v, s) for v in variants for s in seeds if (v, str(s)) not in done]
    print(f"         completed         : {len(done)} (variant, seed) rows")
    print(f"         remaining         : {len(todo)}")
    for v, s in todo:
        print(f"           - {v} seed {s}")


# --------------------------------------------------------------------------
def resolve_data_arg(value: str) -> Path:
    p = Path(value)
    if p.is_file():
        return p.resolve()
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
        from uqfusion.config import load_config, resolve_data_yaml  # noqa: PLC0415
        return Path(resolve_data_yaml(load_config(None), value)).resolve()
    except Exception as exc:  # noqa: BLE001 - fall through to a clear message
        raise SystemExit(
            f"--data {value!r} is neither an existing yaml path nor resolvable via config.yaml ({exc})"
        ) from exc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="vis", help="'vis'/'ir' alias or an explicit dataset yaml path")
    ap.add_argument("--stride-yaml", default=None, help="derived stride yaml the grid trains through")
    ap.add_argument("--csv", default=None, help="benchmark results CSV to check resume compatibility against")
    ap.add_argument("--expect-fingerprint", default=None, help="fail unless split_fingerprint equals this")
    ap.add_argument("--expect-label-hash", default=None, help="fail unless the train-label hash equals this")
    ap.add_argument("--min-gap-frames", type=int, default=100, help="temporal guard band (config data_audit)")
    ap.add_argument("--sample-pixels", type=int, default=40, help="images to open for the letterbox check")
    ap.add_argument("--seed", type=int, default=0, help="sampling seed for the pixel check")
    ap.add_argument("--fast", action="store_true",
                    help="skip the full label scan (structure + fingerprint checks only)")
    ap.add_argument("--hash-labels", action="store_true",
                    help="also compute the train-label content hash (reads every train label again; slow)")
    ap.add_argument("--variants", nargs="*", default=None, help="variant list for the remaining-runs report")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2],
                    help="seed list for the remaining-runs report")
    args = ap.parse_args()

    yaml_path = resolve_data_arg(args.data)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    root = dataset_root(data, yaml_path)
    is_ir = "infrared" in str(root).lower() or "ir" == args.data.lower()

    n_names = check_yaml(yaml_path, data, root)
    images_by_split = check_lists(data, yaml_path)

    section("5-6. label syntax + split fingerprint")
    stats: dict[str, LabelStats] = {}
    per_run: dict[str, dict[str, LabelStats]] = defaultdict(dict)
    if args.fast:
        report("WARN", "--fast: label syntax and balance checks skipped")
    else:
        bad_total = 0
        for split, imgs in images_by_split.items():
            print(f"         scanning {split} labels ({len(imgs)} frames)...")
            total, runs = scan_labels(imgs, n_names or 2, per_run=True)
            stats[split] = total
            for run, st in runs.items():
                per_run[run][split] = st
            bad_total += len(total.bad_lines)
            for line in total.bad_lines[:5]:
                print(f"           {line}")
        if bad_total:
            report("FAIL", f"{bad_total}+ malformed label lines (see samples above)")
        else:
            report("PASS", "every label line is '<class> cx cy w h', class in range, coords in [0,1]")

    full_fp = check_fingerprint(images_by_split, args.expect_fingerprint)

    stride_fp = ""
    if args.stride_yaml:
        stride_fp = ""
        check_stride(Path(args.stride_yaml), images_by_split,
                     args.expect_fingerprint if args.stride_yaml else None)
        # recompute for the resume check (check_stride already validated resolvability)
        sp = Path(args.stride_yaml)
        if sp.is_file():
            with open(sp, "r", encoding="utf-8") as f:
                sdata = yaml.safe_load(f)
            try:
                simgs = {s: split_image_list(sdata, sp.resolve(), s)[0] for s in ("train", "val")}
                stride_fp = split_fingerprint(simgs)
            except Exception:  # noqa: BLE001 - already reported above
                stride_fp = ""

    check_leakage(images_by_split, args.min_gap_frames)

    if not args.fast:
        check_balance(stats, per_run, n_names)

    if not is_ir:
        check_night_filter(images_by_split, per_run)
    else:
        section("9. night-box visibility filter")
        report("INFO", "IR tree — the night filter is VIS-train-only by design, nothing to check")

    if args.hash_labels:
        check_label_hash(images_by_split, args.expect_label_hash)

    check_caches(root)

    if is_ir:
        check_letterbox(images_by_split, args.sample_pixels, args.seed)

    if args.csv:
        variants = args.variants or [
            "yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
            "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",
            "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x",
            "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
            "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",
            "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x",
        ]
        check_resume(Path(args.csv), stride_fp or full_fp, variants, args.seeds)

    section("SUMMARY")
    counts = defaultdict(int)
    for status, _ in _results:
        counts[status] += 1
    print(f"         PASS {counts['PASS']}   FAIL {counts['FAIL']}   "
          f"WARN {counts['WARN']}   INFO {counts['INFO']}")
    for status, title in _results:
        if status in ("FAIL", "WARN"):
            print(f"         {status}: {title}")
    verdict = "FAIL" if counts["FAIL"] else "PASS"
    print(f"\nVERDICT: {verdict}")
    return 1 if counts["FAIL"] else 0


if __name__ == "__main__":
    sys.exit(main())
