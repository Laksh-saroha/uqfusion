"""Measure batch-1 FPS/latency per variant (plan C5) after the grid has run.

Picks each variant's lowest *admissible* seed from the consolidated record,
times predict() end-to-end on val images (fp32 and, on GPU, fp16), writes
fps.csv. FPS is a property of the architecture, not of the seed, so one
checkpoint per variant is enough — but which one is recorded (`run_id`) so the
number can be reproduced.

Must run on a CUDA build of torch: a CPU timing is not the number Table 1 wants.
The repo venv is cpu-only, so invoke the interpreter that has the GPU build and
put the package on the path rather than installing into it:

    PYTHONPATH=src python scripts/measure_fps.py
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

from uqfusion.bench.fps import measure_fps, write_fps_csv
from uqfusion.bench.grid import resolve_device
from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.data.lists import load_data_yaml, split_image_list

RECORD = Path("phase1_benchmark/results.csv")


def pick_checkpoints(results_csv: Path, record_root: Path, allow_inadmissible: bool,
                     all_seeds: bool = False) -> list[dict]:
    """Checkpoints to time, in measurement order.

    By default one row per variant — the lowest admissible seed — because latency is
    a property of the architecture, not of the weight values: seeds of one variant
    share layer count, parameter count and FLOPs exactly. Only postprocess can move
    with the weights, since NMS work scales with how many boxes clear threshold.

    With `all_seeds`, every admissible run is timed and the order is **round-robin
    over seeds, not grouped by variant**: all variants at seed 0, then all at seed 1,
    and so on. Grouping a variant's three seeds back-to-back would measure them in
    one thermal state and report a flatteringly small spread, while leaving the
    between-variant comparison fully exposed to drift. Spreading them makes each
    variant's error bar span the whole sweep, so it reflects the variability a reader
    would actually see — and drift no longer aliases onto whichever variant ran last.

    An inadmissible run stopped short of its own optimum, so its `best.pt` is not the
    checkpoint the variant would ship — but its architecture is identical, so timing
    it is harmless if nothing better exists. That fallback is opt-in, never silent.
    """
    by_variant: dict[str, list[dict]] = defaultdict(list)
    for row in csv.DictReader(open(results_csv, encoding="utf-8")):
        by_variant[row["variant"]].append(row)

    usable: dict[str, list[dict]] = {}
    for variant, rows in sorted(by_variant.items()):
        ok = [r for r in rows if r.get("admissible", "yes") == "yes"]
        if not ok:
            if not allow_inadmissible:
                print(f"[fps] SKIP {variant}: no admissible run (--allow-inadmissible to time it anyway)")
                continue
            ok = rows
        usable[variant] = sorted(ok, key=lambda r: int(r["seed"]))

    def entry(variant: str, row: dict) -> dict:
        # the consolidated record locates runs relative to itself; older CSVs
        # carry an absolute run_dir from whichever machine trained them
        run_dir = record_root / row["run_path"] if row.get("run_path") else Path(row["run_dir"])
        return {"variant": variant, "row": row, "weights": run_dir / "weights" / "best.pt"}

    if not all_seeds:
        return [entry(v, rows[0]) for v, rows in usable.items()]

    picked = []
    for i in range(max(len(r) for r in usable.values())):
        for variant, rows in usable.items():
            if i < len(rows):
                picked.append(entry(variant, rows[i]))
    return picked


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="runs/derived/data_vis_stride2.yaml",
                        help="dataset supplying the timing images (val split); defaults to the "
                             "same yaml the grid trained against so the images match. 'vis'/'ir' "
                             "are config aliases; anything else is a path")
    parser.add_argument("--results-csv", default=str(RECORD))
    parser.add_argument("--out-csv", default=None, help="default: <record dir>/fps.csv")
    parser.add_argument("--variants", nargs="*", default=None, help="subset, for a quick check")
    parser.add_argument("--frames", type=int, default=None, help="override benchmark.fps_frames")
    parser.add_argument("--all-seeds", action="store_true",
                        help="time every admissible run, not one per variant, so the latency "
                             "column can carry a spread. ~3x the runtime")
    parser.add_argument("--allow-inadmissible", action="store_true")
    parser.add_argument("--allow-cpu", action="store_true",
                        help="time on CPU anyway; the result is NOT the Table 1 number")
    args = parser.parse_args()

    cfg = load_config(args.config)
    results_csv = Path(args.results_csv)
    record_root = results_csv.parent
    out_csv = Path(args.out_csv) if args.out_csv else record_root / "fps.csv"

    images = split_image_list(load_data_yaml(resolve_data_yaml(cfg, args.data)), "val")
    print(f"[fps] {len(images)} val images from '{args.data}'")

    import torch
    device = resolve_device(cfg)
    on_cpu = (device == "cpu") or (device is None and not torch.cuda.is_available())
    if on_cpu and not args.allow_cpu:
        raise SystemExit(
            f"[fps] REFUSED: torch reports no CUDA device (torch {torch.__version__}). Batch-1 GPU "
            f"throughput is the point of this measurement; a CPU number would silently misreport it. "
            f"Run under an interpreter with a CUDA build, or pass --allow-cpu to override.")
    precisions = [False] if on_cpu else [False, True]
    print(f"[fps] torch {torch.__version__}, device {device!r}, precisions "
          f"{'fp32' if on_cpu else 'fp32+fp16'}")

    picked = pick_checkpoints(results_csv, record_root, args.allow_inadmissible, args.all_seeds)
    if args.variants:
        picked = [p for p in picked if p["variant"] in set(args.variants)]
    print(f"[fps] {len(picked)} checkpoints x {len(precisions)} precisions"
          + (" (round-robin over seeds)" if args.all_seeds else " (one seed per variant)"))

    rows = []
    for i, p in enumerate(picked, 1):
        if not p["weights"].is_file():
            print(f"[fps] WARNING: {p['weights']} missing — skipping {p['variant']}")
            continue
        for half in precisions:
            result = measure_fps(cfg, p["variant"], p["weights"], images,
                                 half=half, n_frames=args.frames)
            result["run_id"] = p["row"].get("run_id", "")
            result["seed"] = p["row"]["seed"]
            print(f"[fps] {i}/{len(picked)} {p['variant']:9} s{result['seed']} "
                  f"{'fp16' if half else 'fp32'}: "
                  f"{result['fps']:>6} fps  {result['wall_ms_per_img']:>6} ms/img  "
                  f"(pre {result['pre_ms']} / inf {result['inf_ms']} / post {result['post_ms']}) "
                  f"[{result['gpu_temp_c']}C {result['gpu_clock_mhz']}MHz]")
            rows.append(result)

    if not rows:
        print("[fps] no rows produced — has the grid run?")
        return 1
    write_fps_csv(rows, out_csv)
    print(f"[fps] written -> {out_csv}")

    # Whether the clock actually held is the difference between an error bar that
    # means "seed variation" and one that means "the GPU throttled". Report it
    # rather than leaving the reader to infer it from the raw column.
    clocks = [int(r["gpu_clock_mhz"]) for r in rows if str(r.get("gpu_clock_mhz", "")).isdigit()]
    if clocks:
        lo, hi = min(clocks), max(clocks)
        spread = (hi - lo) / hi * 100
        print(f"[fps] GPU clock across the sweep: {lo}-{hi} MHz ({spread:.1f}% spread)")
        if spread > 5:
            print(f"[fps] WARNING: the clock moved {spread:.1f}% during the sweep. Latency spread "
                  f"here is throttling, not seed variation — pin the clock "
                  f"(elevated: nvidia-smi --lock-gpu-clocks=<mhz>,<mhz>) and re-run before "
                  f"quoting any error bar.")
        else:
            print("[fps] clock held within 5% — the spread across seeds is attributable to the runs.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
