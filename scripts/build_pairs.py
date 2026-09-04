"""Index-aligned VIS<->IR frame pairing (Phase 3/4; HOW_TO_RUN §5, handoff §7.4).

`uqfusion.eval.fusion_eval.evaluate_systems` consumes two caches that are
aligned element-by-element: record i of the VIS cache and record i of the IR
cache must be the same instant. This script produces the two image lists that
make that true.

Source of truth is `Pohang_dataset/paired/pohang*_pairs.csv` (timestamp-matched
by the dataset authors, column `dt_ms`). Do NOT pair by frame number: 16,544 of
the 28,388 pair rows have a DIFFERENT index on the VIS and IR side, so a
filename join silently mismatches 58% of the set.

A pair is emitted only when both frames survive into the requested split of
their own modality, so the result is leakage-consistent with the split that
trained the models.

Usage:
    python scripts/build_pairs.py --split val --out-dir runs/derived
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.data.lists import load_data_yaml, split_image_list


def load_pairs(paired_dir: Path) -> tuple[dict[str, str], dict[str, float]]:
    """vis_filename -> ir_filename, plus vis_filename -> dt_ms."""
    pairs: dict[str, str] = {}
    dts: dict[str, float] = {}
    csvs = sorted(paired_dir.glob("*_pairs.csv"))
    if not csvs:
        raise FileNotFoundError(f"no *_pairs.csv under {paired_dir}")
    for f in csvs:
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                pairs[row["stereo_L_file"]] = row["ir_file"]
                dts[row["stereo_L_file"]] = float(row["dt_ms"])
    return pairs, dts


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    parser.add_argument("--out-dir", default="runs/derived")
    parser.add_argument("--max-dt-ms", type=float, default=None,
                        help="drop pairs whose capture times differ by more than this")
    parser.add_argument("--tag", default=None, help="output name stem (default paired_<split>)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    vis_yaml = load_data_yaml(resolve_data_yaml(cfg, "vis"))
    ir_yaml = load_data_yaml(resolve_data_yaml(cfg, "ir"))
    vis_imgs = split_image_list(vis_yaml, args.split)
    ir_imgs = split_image_list(ir_yaml, args.split)

    # config.load_config() has already absolutized datasets.pohang.root
    paired_dir = Path(cfg["datasets"]["pohang"]["root"]) / "paired"
    pairs, dts = load_pairs(paired_dir)

    ir_by_name = {p.name: p for p in ir_imgs}
    rows = []
    n_no_pair = n_ir_out_of_split = n_dt_reject = 0
    for v in vis_imgs:
        ir_name = pairs.get(v.name)
        if ir_name is None:
            n_no_pair += 1
            continue
        ir_path = ir_by_name.get(ir_name)
        if ir_path is None:
            n_ir_out_of_split += 1
            continue
        dt = dts[v.name]
        if args.max_dt_ms is not None and abs(dt) > args.max_dt_ms:
            n_dt_reject += 1
            continue
        rows.append((v, ir_path, dt))

    # Deterministic order: run, then VIS frame ordinal. Both caches are written
    # in THIS order, which is what makes index i mean the same instant in both.
    rows.sort(key=lambda r: (r[0].parent.name, r[0].name))

    tag = args.tag or f"paired_{args.split}"
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    vis_list, ir_list = out_dir / f"{tag}_vis.txt", out_dir / f"{tag}_ir.txt"
    manifest = out_dir / f"{tag}_manifest.csv"
    vis_list.write_text("\n".join(str(v) for v, _, _ in rows) + "\n", encoding="utf-8")
    ir_list.write_text("\n".join(str(i) for _, i, _ in rows) + "\n", encoding="utf-8")
    with open(manifest, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "run", "vis_image", "ir_image", "dt_ms"])
        for k, (v, i, dt) in enumerate(rows):
            w.writerow([k, v.parent.name, str(v), str(i), f"{dt:.3f}"])

    by_run: dict[str, int] = {}
    for v, _, _ in rows:
        by_run[v.parent.name] = by_run.get(v.parent.name, 0) + 1
    abs_dt = sorted(abs(dt) for _, _, dt in rows)
    print(f"[pairs] split={args.split}  VIS {len(vis_imgs)} frames, IR {len(ir_imgs)} frames")
    print(f"[pairs] emitted {len(rows)} aligned pairs  {by_run}")
    print(f"[pairs] dropped: {n_no_pair} VIS with no pair row, "
          f"{n_ir_out_of_split} whose IR partner is outside the {args.split} split, "
          f"{n_dt_reject} over --max-dt-ms")
    if abs_dt:
        print(f"[pairs] |dt_ms| median {abs_dt[len(abs_dt)//2]:.1f}  p95 {abs_dt[int(0.95*len(abs_dt))]:.1f}  max {abs_dt[-1]:.1f}")
    print(f"[pairs] -> {vis_list}\n[pairs] -> {ir_list}\n[pairs] -> {manifest}")
    if not rows:
        print("[pairs] FAIL: no pairs emitted", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
