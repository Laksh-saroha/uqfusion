"""Filter unlearnable night boxes from VIS TRAIN labels by measurable visibility.

Why: Pohang night VIS frames carry annotations (projected from other sensors)
for ships the camera physically cannot see. For VIS-only training those boxes
are label noise — the model is punished for missing pixels that carry no
signal. This script scores every TRAIN box photometrically and (on --execute)
drops the ones below committed thresholds. Val/test labels are NEVER touched:
they stay honest hard cases and the fusion showcase.

Scoring (per box, only in dark frames):
    frame gate   median luminance of the frame's content region (letterbox
                 padding excluded) < --dark-median. Day frames are never touched.
    box_mean     mean intensity inside the box
    grad         mean Sobel gradient magnitude inside the box (structure energy)
    contrast     |box_mean - ring_mean| vs a local background ring around the box
    FLAGGED      box_mean < --t-int AND grad < --t-grad AND contrast < --t-contrast
                 (all three must fail: conservative — any visible cue keeps the box)

IR cross-reference (informational): a flagged box whose paired IR frame (same
run, same/nearest ordinal) holds a same-class box is a confirmed
real-object-invisible-in-VIS. Flagged boxes weak in BOTH modalities are logged
as possible annotation errors — review those, don't just drop.

Frames left with zero boxes stay in the train list as background images
(Ultralytics handles empty label files; backgrounds help precision).

Modes:
    (default)   audit: score everything, write box_scores.csv + histograms +
                spot-check crops (~100 flagged stratified by size, ~200
                unflagged from the same dark frames). No writes to labels.
    --execute   re-score, back up each touched label once (*.pre_visfilter),
                rewrite labels without flagged boxes, write manifest JSON with
                thresholds + per-file drops + label-content hashes before/after.
    --restore   copy every *.pre_visfilter back over its label. Undo.

Paths come from config.yaml (--data-vis/--data-ir accept aliases or explicit
yaml paths). Frame medians are cached (frame_medians.csv) so threshold
re-calibration skips the full-dataset luminance pass.

Usage:
    python scripts/filter_night_boxes.py                        # audit, config paths
    python scripts/filter_night_boxes.py --data-vis Pohang_dataset/data_vis.yaml \
        --data-ir Pohang_dataset/data_ir.yaml                   # local test
    python scripts/filter_night_boxes.py --t-int 45 --t-grad 8 --t-contrast 10 --execute
    python scripts/filter_night_boxes.py --cut pohang01:865 --execute
        # deterministic mode: drop ALL train boxes of a run past an ordinal
        # (calibration verdict: pohang01 night segment has no learnable boxes)

Exit 0 = done; 1 = error / nothing to do.

AFTER --execute: results trained on filtered labels are a NEW experiment —
use a fresh --out-csv for any benchmark grid; never mix with unfiltered rows.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uqfusion.config import load_config, resolve_data_yaml          # noqa: E402
from uqfusion.data.labels import label_content_hash, label_path   # noqa: E402
from uqfusion.data.lists import (                                   # noqa: E402
    frame_ordinal, load_data_yaml, run_key, split_image_list,
)

BACKUP_SUFFIX = ".pre_visfilter"
PAD_LEVEL = 4          # pixels <= this count as letterbox padding when finding content
IR_ORDINAL_TOL = 5.0   # nearest-IR-frame match tolerance (ordinal units)
SCORE_CSV = "box_scores.csv"
MEDIANS_CSV = "frame_medians.csv"
MANIFEST = "visfilter_manifest.json"

CSV_HEADER = ("run,image,line_idx,cls,xc,yc,w,h,area_px,frame_median,"
              "box_mean,ring_mean,contrast,grad,flagged,ir_same_class\n")


# R-E1 slice 3: this file used to carry its own `images -> labels` swap (rightmost
# SUBSTRING, so a directory merely containing the letters "images" would have matched)
# and its own copy of the content hash. Both now come from one shared definition,
# measured to agree with all three previous implementations on every one of the
# 133,140 real train+val image paths.
_label_path = label_path


def read_boxes(lp: Path) -> list[tuple[int, list[float], str]]:
    """[(line_idx, [cls, xc, yc, w, h], raw_line)] — raw kept for byte-exact rewrite."""
    if not lp.is_file():
        return []
    out = []
    with open(lp, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            parts = line.split()
            if len(parts) >= 5:
                out.append((i, [float(x) for x in parts[:5]], line.rstrip("\n")))
    return out


def content_bbox(gray: np.ndarray) -> tuple[int, int, int, int]:
    """(y0, y1, x0, x1) of the non-padding region (letterbox bars excluded)."""
    mask = gray > PAD_LEVEL
    rows = np.flatnonzero(mask.any(axis=1))
    cols = np.flatnonzero(mask.any(axis=0))
    if rows.size == 0:  # fully dark frame: content = whole frame
        return 0, gray.shape[0], 0, gray.shape[1]
    return int(rows[0]), int(rows[-1]) + 1, int(cols[0]), int(cols[-1]) + 1


def frame_median(img: Path) -> float:
    g = cv2.imread(str(img), cv2.IMREAD_REDUCED_GRAYSCALE_4)
    if g is None:
        return float("nan")
    y0, y1, x0, x1 = content_bbox(g)
    return float(np.median(g[y0:y1, x0:x1]))


def score_boxes(img: Path, boxes) -> list[dict] | None:
    """Full-res photometric scores for every box in one frame."""
    g = cv2.imread(str(img), cv2.IMREAD_GRAYSCALE)
    if g is None:
        return None
    H, W = g.shape
    cy0, cy1, cx0, cx1 = content_bbox(g)
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    out = []
    for line_idx, (cls, xc, yc, w, h), _raw in boxes:
        x1 = max(cx0, int((xc - w / 2) * W)); x2 = min(cx1, int(np.ceil((xc + w / 2) * W)))
        y1 = max(cy0, int((yc - h / 2) * H)); y2 = min(cy1, int(np.ceil((yc + h / 2) * H)))
        if x2 - x1 < 1 or y2 - y1 < 1:  # box entirely in padding: degenerate, keep
            out.append(dict(line_idx=line_idx, cls=int(cls), xc=xc, yc=yc, w=w, h=h,
                            area_px=0, box_mean=float("nan"), ring_mean=float("nan"),
                            contrast=float("nan"), grad=float("nan")))
            continue
        box = g[y1:y2, x1:x2]
        box_mean = float(box.mean())
        grad = float(mag[y1:y2, x1:x2].mean())
        # background ring: box expanded by max(3px, half its size) per side,
        # clipped to the content region so padding never fakes contrast
        ex = max(3, (x2 - x1) // 2); ey = max(3, (y2 - y1) // 2)
        rx1, rx2 = max(cx0, x1 - ex), min(cx1, x2 + ex)
        ry1, ry2 = max(cy0, y1 - ey), min(cy1, y2 + ey)
        ring_sum = float(g[ry1:ry2, rx1:rx2].sum()) - float(box.sum())
        ring_n = (ry2 - ry1) * (rx2 - rx1) - (y2 - y1) * (x2 - x1)
        ring_mean = ring_sum / ring_n if ring_n > 0 else float("nan")
        contrast = abs(box_mean - ring_mean) if ring_n > 0 else float("nan")
        out.append(dict(line_idx=line_idx, cls=int(cls), xc=xc, yc=yc, w=w, h=h,
                        area_px=(x2 - x1) * (y2 - y1), box_mean=box_mean,
                        ring_mean=ring_mean, contrast=contrast, grad=grad))
    return out


def is_flagged(s: dict, t_int: float, t_grad: float, t_contrast: float) -> bool:
    """All three must fail. NaN (degenerate box / no ring) never flags: keep."""
    return (s["box_mean"] == s["box_mean"] and s["box_mean"] < t_int
            and s["grad"] == s["grad"] and s["grad"] < t_grad
            and s["contrast"] == s["contrast"] and s["contrast"] < t_contrast)


# ---------------------------------------------------------------- IR pairing

def build_ir_index(data_ir: dict | None) -> dict[str, list[tuple[float, Path]]]:
    """run -> sorted [(ordinal, label_path)] over ALL IR splits (informational)."""
    if data_ir is None:
        return {}
    idx: dict[str, list[tuple[float, Path]]] = defaultdict(list)
    for split in ("train", "val", "test"):
        try:
            imgs = split_image_list(data_ir, split)
        except (KeyError, FileNotFoundError):
            continue
        for p in imgs:
            o = frame_ordinal(p)
            if o is not None:
                idx[run_key(p)].append((o, _label_path(p)))
    return {r: sorted(v) for r, v in idx.items()}


def ir_same_class(ir_idx, run: str, ordinal: float | None, cls: int,
                  cache: dict) -> str:
    """'yes'/'no' = nearest IR frame holds a same-class box; 'no_ir' = no pair."""
    if ordinal is None or run not in ir_idx:
        return "no_ir"
    seq = ir_idx[run]
    from bisect import bisect_left
    i = bisect_left(seq, (ordinal,))
    best = min((c for c in (seq[i - 1] if i else None, seq[i] if i < len(seq) else None)
                if c is not None), key=lambda c: abs(c[0] - ordinal), default=None)
    if best is None or abs(best[0] - ordinal) > IR_ORDINAL_TOL:
        return "no_ir"
    lp = best[1]
    if lp not in cache:
        cache[lp] = {int(b[1][0]) for b in read_boxes(lp)}
    return "yes" if cls in cache[lp] else "no"


# ---------------------------------------------------------------- reporting

def histogram(values: list[float], title: str, lo: float, hi: float, bins: int = 16) -> str:
    vals = [v for v in values if v == v]
    if not vals:
        return f"{title}: no data"
    counts, edges = np.histogram(vals, bins=bins, range=(lo, hi))
    peak = max(int(counts.max()), 1)
    lines = [f"{title}  (n={len(vals)}, median={np.median(vals):.1f})"]
    for c, e0, e1 in zip(counts, edges[:-1], edges[1:]):
        lines.append(f"  {e0:6.1f}-{e1:6.1f} |{'#' * int(round(40 * c / peak)):<40}| {c}")
    return "\n".join(lines)


def save_crop(img_path: Path, s: dict, out: Path) -> None:
    im = cv2.imread(str(img_path))
    if im is None:
        return
    H, W = im.shape[:2]
    x1 = int((s["xc"] - s["w"] / 2) * W); x2 = int(np.ceil((s["xc"] + s["w"] / 2) * W))
    y1 = int((s["yc"] - s["h"] / 2) * H); y2 = int(np.ceil((s["yc"] + s["h"] / 2) * H))
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 0, 255), 1)
    ex, ey = max(40, (x2 - x1)), max(40, (y2 - y1))  # 3x context around the box
    cx1, cx2 = max(0, x1 - ex), min(W, x2 + ex)
    cy1, cy2 = max(0, y1 - ey), min(H, y2 + ey)
    crop = im[cy1:cy2, cx1:cx2]
    if crop.size == 0:
        return
    if crop.shape[0] < 160:  # upscale tiny crops so they are judgeable by eye
        f = 160 / crop.shape[0]
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_NEAREST)
    txt = f"i{s['box_mean']:.0f} g{s['grad']:.0f} c{s['contrast']:.0f}"
    cv2.putText(crop, txt, (2, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
    name = f"{run_key(img_path)}_{img_path.stem}_l{s['line_idx']}_cls{s['cls']}.jpg"
    cv2.imwrite(str(out / name), crop)


# ------------------------------------------------------------------ cut mode

def run_cut_mode(args, data_vis: dict, train_imgs: list[Path], out_dir: Path) -> int:
    """Deterministic per-run ordinal cut: drop ALL train boxes past the cutoff.

    Used when eyeball calibration concludes nothing past a point in a run is
    learnable (pohang01 night segment: every photometric band checked by eye,
    nothing visible). Keeps the bright head of the run untouched.
    """
    if args.cut and args.cut_dark:
        raise SystemExit("--cut and --cut-dark are mutually exclusive")
    dark_mode = bool(args.cut_dark)
    cuts: dict[str, float] = {}
    for spec in (args.cut_dark if dark_mode else args.cut):
        run, _, v = spec.partition(":")
        if not v:
            raise SystemExit(f"need RUN:VALUE, got '{spec}'")
        cuts[run.lower()] = float(v)
    medians = (load_or_compute_medians(train_imgs, out_dir / MEDIANS_CSV, args.workers)
               if dark_mode else {})

    names = data_vis.get("names", {})
    affected: list[Path] = []
    kept_frames: dict[str, int] = defaultdict(int)
    unplaced = 0  # no ordinal (ordinal mode) / unreadable frame (dark mode): never cut
    for p in train_imgs:
        r = run_key(p)
        if r not in cuts:
            continue
        key = medians.get(p, float("nan")) if dark_mode else frame_ordinal(p)
        if key is None or key != key:
            unplaced += 1
        elif (key < cuts[r]) if dark_mode else (key > cuts[r]):
            affected.append(p)
        else:
            kept_frames[r] += 1
    if unplaced:
        print(f"[cut] WARNING: {unplaced} frames in cut runs could not be placed — left untouched")

    drop_by_file: dict[Path, list] = {}
    per_class: dict[int, int] = defaultdict(int)
    frames_with_boxes = 0
    for p in affected:
        boxes = read_boxes(_label_path(p))
        if not boxes:
            continue
        frames_with_boxes += 1
        drop_by_file[_label_path(p)] = boxes
        for _li, (cls, *_), _raw in boxes:
            per_class[int(cls)] += 1
    n_drop = sum(per_class.values())

    rule = "content median < " if dark_mode else "ordinal > "
    for r, cut in sorted(cuts.items()):
        n_aff = sum(1 for p in affected if run_key(p) == r)
        print(f"[cut] {r}: {rule}{cut:.0f} -> {n_aff} train frames affected, "
              f"{kept_frames[r]} kept")
    print(f"[cut] boxes to drop: {n_drop} "
          f"({', '.join(f'{names.get(c, c)}: {n}' for c, n in sorted(per_class.items()))})")
    print(f"[cut] frames becoming background-only: {frames_with_boxes} "
          f"(all boxes in affected frames drop)")

    # boundary spot-check: last kept / first dropped frames around each cutoff,
    # plus a spread over the dropped range — verifies the cutoff, not a threshold
    rng = random.Random(args.seed)
    kdir = out_dir / "spotcheck" / "cut_kept_boundary"
    ddir = out_dir / "spotcheck" / "cut_dropped"
    for d in (kdir, ddir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.jpg"):
            old.unlink()

    def crops_for(frames: list[Path], out: Path, n: int) -> int:
        saved = 0
        for p in frames:
            for li, (cls, xc, yc, w, h), _raw in read_boxes(_label_path(p)):
                save_crop(p, dict(line_idx=li, cls=int(cls), xc=xc, yc=yc, w=w, h=h,
                                  box_mean=float("nan"), contrast=float("nan"),
                                  grad=float("nan")), out)
                saved += 1
                if saved >= n:
                    return saved
        return saved

    for r, cut in sorted(cuts.items()):
        dropped = [p for p in affected if run_key(p) == r]
        if dark_mode:
            kept = [p for p in train_imgs
                    if run_key(p) == r and p not in set(affected)
                    and medians.get(p, 0) == medians.get(p, 0)]
            kept_sample = rng.sample(kept, min(12, len(kept)))
            drop_sample = rng.sample(dropped, min(30, len(dropped))) if dropped else []
        else:
            run_frames = sorted((p for p in train_imgs if run_key(p) == r
                                 and frame_ordinal(p) is not None), key=frame_ordinal)
            kept_sample = [p for p in run_frames if frame_ordinal(p) <= cut][-12:]
            drop_sample = dropped[:12] + (rng.sample(dropped, min(24, len(dropped)))
                                          if dropped else [])
        crops_for(kept_sample, kdir, 12)
        crops_for(drop_sample, ddir, 30)
    print(f"[spotcheck] boundary-kept crops -> {kdir}")
    print(f"[spotcheck] dropped crops -> {ddir}")

    if not args.execute:
        print("\nDRY RUN — no labels modified. Check boundary crops, then --execute.")
        return 0

    hash_before = label_content_hash(train_imgs)
    n_files = 0
    for lp, boxes in sorted(drop_by_file.items()):
        bak = lp.with_name(lp.name + BACKUP_SUFFIX)
        if not bak.is_file():
            bak.write_bytes(lp.read_bytes())
        lp.write_text("", encoding="utf-8")  # full cut: frame becomes background
        n_files += 1
    hash_after = label_content_hash(train_imgs)
    manifest = {
        "script": Path(__file__).name,
        "mode": "cut_dark" if dark_mode else "cut",
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_vis_yaml": data_vis["_yaml_path"],
        "cuts": cuts,
        "rationale": "eyeball calibration: no learnable boxes past cutoff "
                     "(all photometric bands spot-checked, nothing visible)",
        "train_frames": len(train_imgs),
        "frames_affected": len(affected),
        "boxes_dropped": n_drop,
        "boxes_dropped_per_class": {str(names.get(c, c)): n for c, n in sorted(per_class.items())},
        "files_rewritten": n_files,
        "frames_now_background_only": frames_with_boxes,
        "train_label_hash_before": hash_before,
        "train_label_hash_after": hash_after,
        "dropped": {str(lp): [li for li, _b, _r in boxes]
                    for lp, boxes in sorted(drop_by_file.items())},
    }
    mpath = out_dir / MANIFEST
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[execute] emptied {n_files} label files ({n_drop} boxes) — backups *{BACKUP_SUFFIX}")
    print(f"[execute] train label hash {hash_before} -> {hash_after}")
    print(f"[execute] manifest -> {mpath}")
    print("[execute] REMINDER: delete stale Ultralytics *.cache label caches; "
          "filtered labels = new experiment, fresh --out-csv, never mix rows.")
    return 0


# ---------------------------------------------------------------- main passes

def load_or_compute_medians(imgs: list[Path], cache_csv: Path, workers: int) -> dict[Path, float]:
    cached: dict[str, float] = {}
    if cache_csv.is_file():
        with open(cache_csv, "r", encoding="utf-8") as f:
            next(f, None)
            for line in f:
                p, m = line.rsplit(",", 1)
                cached[p] = float(m)
        if all(str(p) in cached for p in imgs):
            print(f"[medians] reusing cache {cache_csv} ({len(cached)} frames)")
            return {p: cached[str(p)] for p in imgs}
        print("[medians] cache stale/incomplete — recomputing")
    t0 = time.time()
    out: dict[Path, float] = {}
    with ThreadPoolExecutor(max_workers=workers) as ex:
        for i, (p, m) in enumerate(zip(imgs, ex.map(frame_median, imgs))):
            out[p] = m
            if (i + 1) % 10000 == 0:
                print(f"[medians] {i + 1}/{len(imgs)} ({time.time() - t0:.0f}s)")
    cache_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_csv, "w", encoding="utf-8") as f:
        f.write("image,median\n")
        for p, m in out.items():
            f.write(f"{p},{m}\n")
    print(f"[medians] {len(out)} frames in {time.time() - t0:.0f}s -> {cache_csv}")
    return out


# `label_content_hash` comes from uqfusion.data.labels -- one definition, so its
# numbers stay comparable to runs/label_hash_ledger.csv.


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None)
    ap.add_argument("--data-vis", default="vis", help="alias or explicit dataset yaml")
    ap.add_argument("--data-ir", default="ir",
                    help="alias/yaml for IR cross-reference; 'none' to skip")
    ap.add_argument("--dark-median", type=float, default=40.0,
                    help="frame gate: content median luminance below this = dark frame")
    ap.add_argument("--t-int", type=float, default=45.0, help="box mean intensity threshold")
    ap.add_argument("--t-grad", type=float, default=8.0, help="box Sobel-energy threshold")
    ap.add_argument("--t-contrast", type=float, default=10.0, help="box-vs-ring threshold")
    ap.add_argument("--crops-flagged", type=int, default=100)
    ap.add_argument("--crops-unflagged", type=int, default=200)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out-dir", default=None,
                    help="default: <outputs_root>/visfilter from config.yaml")
    ap.add_argument("--seed", type=int, default=0, help="spot-check sampling seed")
    ap.add_argument("--cut", action="append", default=None, metavar="RUN:ORDINAL",
                    help="deterministic mode: drop ALL train boxes of RUN with "
                         "frame ordinal > ORDINAL (e.g. pohang01:865). Repeatable. "
                         "Replaces photometric flagging entirely — calibrated by "
                         "eye against spot-checks, committed here.")
    ap.add_argument("--cut-dark", action="append", default=None, metavar="RUN:MEDIAN",
                    help="deterministic mode: drop ALL train boxes of RUN in frames "
                         "whose content-median luminance < MEDIAN (e.g. pohang01:100). "
                         "Frame-level, camera-fair: bright frames survive on any "
                         "camera regardless of ordinal. Repeatable.")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--execute", action="store_true",
                      help="rewrite train labels (backup + manifest)")
    mode.add_argument("--restore", action="store_true",
                      help=f"restore every *{BACKUP_SUFFIX} backup")
    args = ap.parse_args()

    cfg = load_config(args.config)
    vis_yaml = args.data_vis if Path(args.data_vis).suffix in (".yaml", ".yml") \
        else resolve_data_yaml(cfg, args.data_vis)
    data_vis = load_data_yaml(vis_yaml)
    out_dir = Path(args.out_dir) if args.out_dir else Path(cfg["paths"]["outputs_root"]) / "visfilter"
    out_dir.mkdir(parents=True, exist_ok=True)

    train_imgs = split_image_list(data_vis, "train")
    print(f"[data] VIS train: {len(train_imgs)} frames ({vis_yaml})")

    # ------------------------------------------------------------- restore
    if args.restore:
        n = 0
        for img in train_imgs:
            lp = _label_path(img)
            bak = lp.with_name(lp.name + BACKUP_SUFFIX)
            if bak.is_file():
                lp.write_bytes(bak.read_bytes())
                n += 1
        print(f"[restore] {n} label files restored from {BACKUP_SUFFIX} backups")
        return 0

    if args.cut or args.cut_dark:
        return run_cut_mode(args, data_vis, train_imgs, out_dir)

    data_ir = None
    if args.data_ir.lower() != "none":
        ir_yaml = args.data_ir if Path(args.data_ir).suffix in (".yaml", ".yml") \
            else resolve_data_yaml(cfg, args.data_ir)
        data_ir = load_data_yaml(ir_yaml)
    ir_idx = build_ir_index(data_ir)
    ir_cache: dict = {}

    # -------------------------------------------------- pass 1: frame gate
    medians = load_or_compute_medians(train_imgs, out_dir / MEDIANS_CSV, args.workers)
    dark = [p for p in train_imgs if medians[p] == medians[p] and medians[p] < args.dark_median]
    print(histogram(list(medians.values()), "frame content-median luminance (all train)", 0, 255))
    print(f"[gate] dark frames (< {args.dark_median}): {len(dark)}/{len(train_imgs)} "
          f"({100 * len(dark) / max(len(train_imgs), 1):.1f}%)")
    per_run_dark = defaultdict(int)
    for p in dark:
        per_run_dark[run_key(p)] += 1
    for r in sorted(per_run_dark):
        print(f"  {r}: {per_run_dark[r]} dark frames")

    # ------------------------------------------- pass 2: score dark boxes
    dark_boxed = [(p, read_boxes(_label_path(p))) for p in dark]
    dark_boxed = [(p, b) for p, b in dark_boxed if b]
    print(f"[score] {len(dark_boxed)} dark frames carry boxes — scoring full-res")
    t0 = time.time()
    scored: list[tuple[Path, dict]] = []
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        results = ex.map(lambda pb: (pb[0], score_boxes(*pb)), dark_boxed)
        for i, (p, ss) in enumerate(results):
            if ss:
                scored.extend((p, s) for s in ss)
            if (i + 1) % 2000 == 0:
                print(f"[score] {i + 1}/{len(dark_boxed)} frames ({time.time() - t0:.0f}s)")
    print(f"[score] {len(scored)} boxes scored in {time.time() - t0:.0f}s")

    flagged = [(p, s) for p, s in scored
               if is_flagged(s, args.t_int, args.t_grad, args.t_contrast)]
    kept = [(p, s) for p, s in scored if not is_flagged(s, args.t_int, args.t_grad, args.t_contrast)]

    # --------------------------------------------------------- report/CSV
    for name, vals, lo, hi in (
        ("box_mean (dark-frame boxes)", [s["box_mean"] for _, s in scored], 0, 128),
        ("grad (dark-frame boxes)", [s["grad"] for _, s in scored], 0, 64),
        ("contrast (dark-frame boxes)", [s["contrast"] for _, s in scored], 0, 64),
    ):
        print(histogram(vals, name, lo, hi))

    csv_path = out_dir / SCORE_CSV
    n_ir_yes = n_ir_no = 0
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write(CSV_HEADER)
        for p, s in scored:
            fl = is_flagged(s, args.t_int, args.t_grad, args.t_contrast)
            ir = ir_same_class(ir_idx, run_key(p), frame_ordinal(p), s["cls"], ir_cache) \
                if fl else ""
            if ir == "yes":
                n_ir_yes += 1
            elif ir == "no":
                n_ir_no += 1
            f.write(f"{run_key(p)},{p},{s['line_idx']},{s['cls']},"
                    f"{s['xc']},{s['yc']},{s['w']},{s['h']},{s['area_px']},"
                    f"{medians[p]:.1f},{s['box_mean']:.2f},{s['ring_mean']:.2f},"
                    f"{s['contrast']:.2f},{s['grad']:.2f},{int(fl)},{ir}\n")
    print(f"[csv] box scores -> {csv_path}")

    per = defaultdict(lambda: [0, 0])
    for p, s in scored:
        k = (run_key(p), s["cls"])
        per[k][0] += 1
        per[k][1] += int(is_flagged(s, args.t_int, args.t_grad, args.t_contrast))
    print(f"\n[flags] {len(flagged)}/{len(scored)} dark-frame boxes flagged "
          f"(thresholds: int<{args.t_int}, grad<{args.t_grad}, contrast<{args.t_contrast})")
    names = data_vis.get("names", {})
    for (r, c), (n, nf) in sorted(per.items()):
        print(f"  {r} {names.get(c, c)}: {nf}/{n} flagged")
    if flagged:
        print(f"[ir-crossref] flagged boxes with same-class IR box nearby: {n_ir_yes} yes / "
              f"{n_ir_no} no / {len(flagged) - n_ir_yes - n_ir_no} no_ir "
              f"— 'no' rows may be annotation errors, review in {SCORE_CSV}")

    flagged_ids = {(p, s["line_idx"]) for p, s in flagged}
    frames_emptied = sum(
        1 for p, boxes in dark_boxed
        if boxes and all((p, li) in flagged_ids for li, _, _ in boxes))
    print(f"[frames] train frames that would become background-only: {frames_emptied}")

    # ------------------------------------------------- spot-check crops
    rng = random.Random(args.seed)
    fdir = out_dir / "spotcheck" / "flagged"
    udir = out_dir / "spotcheck" / "unflagged"
    for d in (fdir, udir):
        d.mkdir(parents=True, exist_ok=True)
        for old in d.glob("*.jpg"):
            old.unlink()
    f_sample: list[tuple[Path, dict]] = []
    if flagged:
        by_area = sorted(flagged, key=lambda ps: ps[1]["area_px"])
        q = max(1, len(by_area) // 4)
        small, rest = by_area[:q], by_area[q:]
        half = args.crops_flagged // 2
        f_sample = (rng.sample(small, min(half, len(small)))
                    + rng.sample(rest, min(args.crops_flagged - half, len(rest))))
    u_sample = rng.sample(kept, min(args.crops_unflagged, len(kept))) if kept else []
    for p, s in f_sample:
        save_crop(p, s, fdir)
    for p, s in u_sample:
        save_crop(p, s, udir)
    print(f"[spotcheck] {len(f_sample)} flagged crops -> {fdir}")
    print(f"[spotcheck] {len(u_sample)} unflagged crops -> {udir}")

    if not args.execute:
        print("\nDRY RUN — no labels modified. Eyeball the spot-check crops, adjust "
              "thresholds if needed, then re-run with --execute.")
        return 0

    # ------------------------------------------------------------ execute
    if not flagged:
        print("[execute] nothing flagged — no labels to modify")
        return 0
    hash_before = label_content_hash(train_imgs)
    drop_by_file: dict[Path, set[int]] = defaultdict(set)
    for p, s in flagged:
        drop_by_file[_label_path(p)].add(s["line_idx"])
    n_files = n_dropped = 0
    for lp, drop in sorted(drop_by_file.items()):
        bak = lp.with_name(lp.name + BACKUP_SUFFIX)
        if not bak.is_file():  # back up ONCE — repeated runs must not clobber the original
            bak.write_bytes(lp.read_bytes())
        keep_lines = [raw for li, _b, raw in read_boxes(lp) if li not in drop]
        lp.write_text("".join(line + "\n" for line in keep_lines), encoding="utf-8")
        n_files += 1
        n_dropped += len(drop)
    hash_after = label_content_hash(train_imgs)
    manifest = {
        "script": Path(__file__).name,
        "date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "data_vis_yaml": str(vis_yaml),
        "thresholds": {"dark_median": args.dark_median, "t_int": args.t_int,
                       "t_grad": args.t_grad, "t_contrast": args.t_contrast},
        "train_frames": len(train_imgs),
        "dark_frames": len(dark),
        "boxes_scored": len(scored),
        "boxes_dropped": n_dropped,
        "files_rewritten": n_files,
        "frames_now_background_only": frames_emptied,
        "ir_crossref_flagged": {"yes": n_ir_yes, "no": n_ir_no,
                                "no_ir": len(flagged) - n_ir_yes - n_ir_no},
        "train_label_hash_before": hash_before,
        "train_label_hash_after": hash_after,
        "dropped": {str(lp): sorted(drop) for lp, drop in sorted(drop_by_file.items())},
    }
    mpath = out_dir / MANIFEST
    mpath.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"\n[execute] dropped {n_dropped} boxes across {n_files} label files "
          f"(backups: *{BACKUP_SUFFIX})")
    print(f"[execute] train label hash {hash_before} -> {hash_after}")
    print(f"[execute] manifest -> {mpath}")
    print("[execute] REMINDER: filtered labels = new experiment. Use a fresh "
          "--out-csv for any benchmark grid; never mix with unfiltered rows. "
          "Stride-subset lists reference the same label files, no regeneration needed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
