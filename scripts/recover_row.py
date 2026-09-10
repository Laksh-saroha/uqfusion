"""Bank a CSV row for a run that finished training but never got one.

Why this exists: grid.py writes a row only after its own post-training
validation (grid.py:231), so a run that was killed after it had already
satisfied early stopping leaves a complete `weights/best.pt` and no row. The
grid cannot recover it — `_completed()` skips by CSV row, so re-running the
variant lands in the resume branch and *keeps training* a run that was
already done (grid.py:199-210). This script does the missing half only:
validate `best.pt` on the val split and append the row.

Admissibility is checked, not assumed. A run is bankable only if it trained
at least `patience` epochs past its own fitness peak — the same criterion
ultralytics' EarlyStopping applies — because then `best.pt` holds the same
weights an uninterrupted run would have stopped on. Runs cut short of that
are refused: a better epoch might still have been coming.

Fitness is mAP50-95 alone in ultralytics 8.4.90 — NOT the 0.1*mAP50 +
0.9*mAP50-95 blend of older releases. Verified against the `best_fitness`
stored in un-stripped checkpoints (26l seed2: 0.2991 == max mAP50-95 0.29910,
against a blend of 0.33413). Using the blend picks the wrong peak epoch on
runs whose mAP50 and mAP50-95 crest at different epochs.

    python scripts/recover_row.py --run-dir runs/benchmark/runs/ship_yolo26l_seed2 \
        --variant yolo26l --seed 2 --data runs/derived/data_vis_stride2.yaml \
        --classes 0 --out-csv runs/benchmark/benchmark_results_tail.csv --dry-run
"""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import yaml

from uqfusion.bench.grid import (
    _append_row,
    _box_metrics,
    _completed,
    _git_commit,
    resolve_device,
    label_fingerprint,
    recipe_identity,
    split_fingerprint,
)
from uqfusion.config import load_config, resolve_data_yaml


def _fitness(row: dict, k50: str, k5095: str) -> float:
    return float(row[k5095])  # 8.4.90 ranks epochs on mAP50-95 alone; see module docstring


def read_curve(results_csv: Path) -> tuple[int, int, float]:
    """(peak_epoch, last_epoch, cumulative_train_time_s) from a run's results.csv.

    `time` is seconds since the *current invocation* started, so it resets at
    every resume; the cumulative total is the sum of each segment's last value.
    Epoch numbers can repeat if two writers shared the dir — the last row wins,
    which is also what ultralytics' own plots do.
    """
    rows = list(csv.DictReader(open(results_csv, encoding="utf-8")))
    if not rows:
        raise SystemExit(f"{results_csv} has no epoch rows")
    k50 = next(c for c in rows[0] if "mAP50(B)" in c)
    k5095 = next(c for c in rows[0] if "mAP50-95" in c)

    total, prev = 0.0, 0.0
    for r in rows:
        t = float(r["time"])
        if t < prev:  # clock reset => a new invocation started here
            total += prev
        prev = t
    total += prev

    by_epoch: dict[int, dict] = {}
    for r in rows:
        by_epoch[int(float(r["epoch"]))] = r
    peak = max(by_epoch.values(), key=lambda r: _fitness(r, k50, k5095))
    return int(float(peak["epoch"])), max(by_epoch), total


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--run-dir", required=True, help="run dir holding weights/best.pt and results.csv")
    p.add_argument("--variant", required=True, help="e.g. yolo26l — also names the pristine .pt for params/GFLOPs")
    p.add_argument("--seed", required=True, type=int)
    p.add_argument("--data", default="vis")
    p.add_argument("--classes", nargs="*", type=int, default=None)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--imgsz", type=int, default=None)
    p.add_argument("--patience", type=int, default=None,
                   help="admissibility margin; default: config benchmark.patience")
    p.add_argument("--dry-run", action="store_true", help="run every check, skip val and the CSV write")
    p.add_argument("--no-write", action="store_true",
                   help="validate and print, but never touch the CSV — use to re-derive a number "
                        "from a checkpoint without committing it to the record")
    p.add_argument("--params-from", default=None,
                   help="weights to measure params/GFLOPs from; default <repo>/<variant>.pt. Must be "
                        "the same pristine checkpoint the sibling rows used, or the column disagrees "
                        "within a variant")
    p.add_argument("--train-time-s", type=float, default=None,
                   help="override cumulative training seconds. REQUIRED for a run dir written by two "
                        "concurrent processes: summing per-segment clocks assumes sequential resumes "
                        "and double-counts overlapping writers, inflating the total several-fold")
    p.add_argument("--replace", action="store_true",
                   help="drop existing rows for this (variant, seed) before appending; the dropped "
                        "rows are written to <out-csv>.replaced_<variant>_seed<seed> first")
    args = p.parse_args()

    cfg = load_config(args.config)
    b = cfg["benchmark"]
    imgsz = args.imgsz if args.imgsz is not None else b["imgsz"]
    patience = args.patience if args.patience is not None else b["patience"]
    data_yaml = resolve_data_yaml(cfg, args.data)
    out_csv = Path(args.out_csv)
    run_dir = Path(args.run_dir)
    classes_tag = " ".join(str(c) for c in args.classes) if args.classes else "all"

    best = run_dir / "weights" / "best.pt"
    if not best.is_file():
        raise SystemExit(f"no best.pt under {run_dir}")

    # 1. the run must not already be in the CSV, and the CSV must be this experiment
    fingerprint = split_fingerprint(data_yaml)
    # R-E1 slice 3: the label state joins the split in the dataset identity, and the
    # recipe is stamped so this recovered row can be told apart from one trained under
    # a different budget. `_completed` now returns a dict per row rather than a bare
    # fingerprint, so the comparison is against the (split, labels) pair.
    label_fp = label_fingerprint(str(data_yaml))
    # The recipe must come from the RUN, not from today's config. This script exists to
    # recover a row for training that already happened, possibly under different
    # defaults; reconstructing the recipe from `cfg` would stamp a fingerprint for a
    # recipe that was never run -- the exact defect R-E1 is about. Ultralytics writes
    # `args.yaml` into every run dir, so read it and refuse if it is missing.
    args_yaml = run_dir / "args.yaml"
    if not args_yaml.is_file():
        raise SystemExit(
            f"no args.yaml under {run_dir}: the recipe this run actually used cannot be "
            "recovered, and stamping today's config defaults instead would record a "
            "recipe that never ran. Recover the run directory or add the row by hand.")
    with open(args_yaml, "r", encoding="utf-8") as f:
        ra = yaml.safe_load(f) or {}
    recipe_fp, recipe_json = recipe_identity(
        epochs=ra.get("epochs"), imgsz=ra.get("imgsz"), batch=ra.get("batch"),
        mosaic=ra.get("mosaic"), close_mosaic=ra.get("close_mosaic"),
        optimizer=ra.get("optimizer"), patience=ra.get("patience"),
        amp=ra.get("amp"), deterministic=ra.get("deterministic"),
        weights=ra.get("model"), train_overrides=None)
    dataset = (fingerprint, label_fp)
    done = _completed(out_csv, classes_tag)
    if (args.variant, str(args.seed)) in done and not (args.no_write or args.replace):
        raise SystemExit(f"{args.variant} seed {args.seed} already has a row in {out_csv} — refusing to duplicate")
    stale = sorted(k for k, v in done.items() if v["dataset"] != dataset)
    if stale:
        raise SystemExit(
            f"{out_csv} holds {len(stale)} row(s) from a different split/label state or "
            f"class filter (e.g. {stale[0]}); current is split '{fingerprint}', labels "
            f"'{label_fp}', classes '{classes_tag}'"
        )

    # 2. the run must have been trained on that same split, at the same imgsz
    import yaml as _yaml

    run_args = _yaml.safe_load(open(run_dir / "args.yaml", encoding="utf-8"))
    if int(run_args.get("imgsz", imgsz)) != imgsz:
        raise SystemExit(f"{run_dir} trained at imgsz {run_args.get('imgsz')} != {imgsz}")
    if (run_args.get("classes") or None) != (args.classes or None):
        raise SystemExit(f"{run_dir} trained with classes {run_args.get('classes')} != {args.classes}")

    # 3. the run must have earned its stop: >= patience epochs past its own peak
    peak_ep, last_ep, train_time = read_curve(run_dir / "results.csv")
    if args.train_time_s is not None:
        print(f"[recover] train_time_s overridden: {train_time:.1f}s -> {args.train_time_s:.1f}s")
        train_time = args.train_time_s
    gap = last_ep - peak_ep
    print(f"[recover] {run_dir.name}: peak ep {peak_ep}, last ep {last_ep}, "
          f"gap {gap} (patience {patience}), cumulative train {train_time:.1f}s ({train_time / 3600:.1f} h)")
    if gap < patience:
        msg = (f"{run_dir.name} stopped only {gap} epochs past its peak (< patience {patience}). "
               "It was cut short, not early-stopped — best.pt is not the weights a full run would "
               "have kept, and its metrics are a lower bound on that seed.")
        if not args.no_write:
            raise SystemExit(f"REFUSED: {msg}")
        print(f"[recover] INADMISSIBLE: {msg}\n[recover] --no-write is set, measuring anyway (not for Table 1)")

    import torch
    import ultralytics
    from ultralytics import YOLO

    # params/GFLOPs off the pristine checkpoint, not best.pt: grid.py:181 reads
    # them from whatever it loaded, so resumed rows report last.pt's slightly
    # lower counts. Table 1 wants one number per variant.
    pristine = Path(args.params_from) if args.params_from else (
        Path(cfg["_config_path"]).parent / f"{args.variant}.pt")
    if not pristine.is_file():
        raise SystemExit(
            f"pristine weights {pristine} not found. Falling back to best.pt would measure the "
            f"TRAINED nc=1 model and silently disagree with sibling rows measured off the nc=80 "
            f"checkpoint. Fetch {pristine.name}, or pass --params-from <path to the same weights "
            f"the sibling rows used>."
        )
    meta_model = YOLO(str(pristine))
    params = sum(t.numel() for t in meta_model.model.parameters())
    try:
        from ultralytics.utils.torch_utils import get_flops

        gflops = float(get_flops(meta_model.model, imgsz))
    except Exception:  # noqa: BLE001 - profiling is metadata, never fatal
        gflops = float("nan")
    print(f"[recover] params {params / 1e6:.2f} M, {gflops:.1f} GFLOPs (from {pristine.name})")

    if args.dry_run:
        print("[recover] --dry-run: not validating, not writing")
        return 0

    # Same call grid.py makes, so the row is measured identically to its siblings.
    out_root = Path(cfg["paths"]["outputs_root"]) / "benchmark"
    metrics = YOLO(str(best)).val(
        data=str(data_yaml), imgsz=imgsz, device=resolve_device(cfg), split="val",
        classes=args.classes,
        project=str(out_root / "runs"), name=f"{run_dir.name}_val", exist_ok=True,
    )

    row = {
        "variant": args.variant, "seed": args.seed,
        **_box_metrics(metrics),
        "params_m": round(params / 1e6, 2),
        "gflops": round(gflops, 1) if gflops == gflops else "",
        "epochs_cfg": ra.get("epochs", b["epochs"]), "train_time_s": round(train_time, 1),
        "run_dir": str(run_dir.resolve()),
        "ultralytics_version": ultralytics.__version__,
        "torch_version": torch.__version__, "git_commit": _git_commit(),
        "data_yaml": str(data_yaml), "split_fingerprint": fingerprint,
        "classes": classes_tag,
        "label_fingerprint_trainval": label_fp,
        "recipe_fingerprint": recipe_fp, "recipe": recipe_json,
    }
    if args.no_write:
        print(f"[recover] --no-write: {args.variant} seed {args.seed} measures mAP50 {row['map50']:.4f} "
              f"mAP50-95 {row['map50_95']:.5f} (CSV untouched)")
        return 0

    if args.replace:
        with open(out_csv, "r", encoding="utf-8", newline="") as f:
            existing = list(csv.DictReader(f))
        drop = [r for r in existing if (r["variant"], r["seed"]) == (args.variant, str(args.seed))]
        if drop:
            side = out_csv.with_suffix(out_csv.suffix + f".replaced_{args.variant}_seed{args.seed}")
            with open(side, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(existing[0].keys()))
                w.writeheader()
                w.writerows(drop)
            with open(out_csv, "w", encoding="utf-8", newline="") as f:
                w = csv.DictWriter(f, fieldnames=list(existing[0].keys()))
                w.writeheader()
                w.writerows(r for r in existing if r not in drop)
            print(f"[recover] --replace: moved {len(drop)} old row(s) to {side.name}")

    _append_row(out_csv, row)
    print(f"[recover] {args.variant} seed {args.seed} -> mAP50 {row['map50']:.4f} "
          f"mAP50-95 {row['map50_95']:.5f} -> {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
