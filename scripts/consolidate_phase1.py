"""Consolidate the Phase 1 backbone benchmark into one folder with one record.

The 27 runs of the `682dbe9f0f05` grid were produced on two machines and landed
in two trees: the server grid (12s/12m/12l/12x/26n/26s) under
`archive/phase1/main_2026-08-10/`, and the laptop continuation (26m/26l/26x)
under `runs/benchmark/`. Their CSVs overlap, their directory layouts differ, and
the record carries no trace of *when* each `best.pt` was reached or *when*
early stopping fired. This script fixes all three:

  1. moves every run of this grid into `phase1_benchmark/runs/<variant>_seed<n>/`
     with an identical internal layout (args.yaml, results.csv, weights/, val/,
     plots/);
  2. merges the two CSVs into one canonically ordered `results.csv`, recording
     any superseded row in `superseded.csv` rather than dropping it;
  3. derives the training-dynamics columns from each run's own curve.

The dynamics columns need care on the runs that were paused and resumed.
`Trainer.resume_training()` (ultralytics 8.4.90) restores the optimizer, EMA and
`best_fitness` but never touches `self.stopper` — the `EarlyStopping` object is
rebuilt one line earlier at `trainer.py:381` with `best_fitness=0.0` and
`best_epoch=0`. Because `EarlyStopping.__call__` treats `best_fitness == 0` as
"always improve", the first validated epoch after a resume becomes the stopper's
new reference epoch, so a resumed run counts its patience from a *local* peak
inside the final segment, not from the run's global peak. Two epochs are
therefore recorded separately:

  best_epoch        the global fitness peak — the weights best.pt actually holds,
                    since the trainer's own best_fitness IS restored on resume
  stopper_ref_epoch the epoch ultralytics' stopper was counting from when
                    training ended; equals best_epoch on an uninterrupted run

Fitness is mAP50-95 alone in 8.4.90, not the 0.1*mAP50 + 0.9*mAP50-95 blend of
older releases (verified against `best_fitness` in un-stripped checkpoints).

    python scripts/consolidate_phase1.py --plan       # print everything, move nothing
    python scripts/consolidate_phase1.py --execute
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parents[1]
LOCAL = REPO / "runs" / "benchmark" / "runs"
ARCH = REPO / "archive" / "phase1" / "main_2026-08-10" / "runs" / "benchmark" / "runs"
DEST = REPO / "phase1_benchmark"

CUR_CSV = REPO / "runs" / "benchmark" / "benchmark_results_tail.csv"
ARCH_CSV = ARCH.parent / "benchmark_results_tail.csv"

FINGERPRINT = "682dbe9f0f05"
K5095 = "metrics/mAP50-95(B)"

# Canonical order: family, then capacity. This is the order Table 1 is read in
# and the order rows are written in.
ORDER = ["yolo12s", "yolo12m", "yolo12l", "yolo12x",
         "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]

# Both grids write into one results.csv, so rows are ordered by model generation
# and then capacity across the whole field.
ORDER_ALL = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
             "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",
             "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x",
             "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
             "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",
             "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]

# `yolo12s` seeds 0-2 were run by BOTH grids, so neither `run_id` nor
# (variant, seed) is unique across the merged table. Every run id therefore
# carries its grid, and `grid` is a column. Without it the file has duplicate
# keys and any group-by silently averages two different experiments together.
GRID_MAIN, GRID_PILOT = "main", "pilot"

# ---------------------------------------------------------------------------
# The pilot grid (2026-07-31) — a SEPARATE experiment, kept separate.
#
# 23 variants x 3 seeds on split fingerprint `f0220e716277`, whose data yaml
# existed only on a server that was wiped and which no split generation
# reproducible from this repo matches. Its numbers are therefore not comparable
# with the live grid's and the two must never share a CSV — hence its own
# subdirectory and its own results.csv, never a merge.
#
# It is otherwise the cleaner of the two grids: batch 31, epochs 100,
# patience 20, imgsz 640, workers 8, ultralytics 8.4.90 on every one of the 69
# runs, and 68 of them stop at exactly peak + patience in a single segment.
PILOT_SRC = (REPO / "archive" / "phase1" / "pilot_2026-07-31" / "runs_31_07"
             / "runs" / "benchmark" / "runs")
PILOT_CSV = PILOT_SRC.parent / "benchmark_results_ship_visfilter.csv"
PILOT_DEST = DEST / "pilot_2026-07-31"          # legacy layout, migrated away
PILOT_DOC = REPO / "docs" / "phase1-pilot-grid.md"
PILOT_FINGERPRINT = "f0220e716277"
PILOT_ORDER = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
               "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",
               "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x",
               "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
               "yolo12n", "yolo12s"]

# Variants the pilot campaign shares with `main`. The main campaign's runs are
# kept and the pilot's are dropped from the table, so every variant is measured
# by exactly one campaign and no seed number appears twice. Their artifacts are
# retained under extra/ rather than deleted.
PILOT_DROP = {"yolo12s"}

# Where each Table 1 row's artifacts live today. Written out by hand rather than
# globbed: three (variant, seed) pairs have more than one candidate directory on
# disk and the choice between them is a matter of record, not of inference.
#   train  - the directory holding the curve and the weights the row was scored on
#   val    - the validation output of the call that produced the row's metrics
#   note   - carried into the manifest
SRC: dict[tuple[str, int], dict] = {}
for _v in ("yolo12s", "yolo12m", "yolo12l"):
    for _s in range(3):
        SRC[(_v, _s)] = {"train": ARCH / f"ship_{_v}_seed{_s}",
                         "val": ARCH / f"ship_{_v}_seed{_s}_val",
                         "trained_on": "server", "scored_on": "server"}
for _v in ("yolo26n", "yolo26s"):
    for _s in range(3):
        SRC[(_v, _s)] = {"train": ARCH / f"ship_{_v}_seed{_s}",
                         "val": ARCH / f"ship_{_v}_seed{_s}_val",
                         "trained_on": "server", "scored_on": "server"}
# yolo12x seed 0: two grid processes shared one run dir. The row is backed by the
# INNER directory, re-validated on the laptop 2026-08-17 (0.30414 vs the server's
# 0.30413 for the same weights). The outer dir is kept under extra/.
SRC[("yolo12x", 0)] = {"train": ARCH / "_partial_ship_yolo12x_seed0" / "ship_yolo12x_seed0",
                       "val": LOCAL / "ship_yolo12x_seed0_val",
                       "trained_on": "server", "scored_on": "laptop",
                       "note": "shared run dir; row re-validated on the laptop from the inner dir"}
# seeds 1 and 2 keep their original server rows and hence their server val dirs.
SRC[("yolo12x", 1)] = {"train": ARCH / "ship_yolo12x_seed1",
                       "val": ARCH / "ship_yolo12x_seed1_val",
                       "trained_on": "server", "scored_on": "server",
                       "note": "shared run dir; killed before patience — excluded from Table 1"}
SRC[("yolo12x", 2)] = {"train": ARCH / "ship_yolo12x_seed2",
                       "val": ARCH / "ship_yolo12x_seed2_val",
                       "trained_on": "server", "scored_on": "server"}
for _v in ("yolo26m", "yolo26l", "yolo26x"):
    for _s in range(3):
        SRC[(_v, _s)] = {"train": LOCAL / f"ship_{_v}_seed{_s}",
                         "val": LOCAL / f"ship_{_v}_seed{_s}_val",
                         "trained_on": "laptop", "scored_on": "laptop"}
SRC[("yolo26m", 0)]["note"] = "trained under ultralytics 8.4.7, re-scored under 8.4.90"

# Artifacts that are not Table 1 rows but are part of the record.
EXTRA: list[tuple[Path, str, str]] = [
    (ARCH / "_partial_ship_yolo12x_seed0", "yolo12x_seed0_outer_writer",
     "outer half of the yolo12x seed 0 collision; its own row (0.29691) was withdrawn as "
     "unreproducible. The inner dir it contains is the Table 1 run and is moved out first."),
    (LOCAL / "_partial_ship_yolo12x_seed0_val", "yolo12x_seed0_outer_writer_val",
     "laptop re-validation of the outer writer's best.pt, 2026-08-17"),
    (ARCH / "ship_yolo12x_seed0_val", "yolo12x_seed0_server_val_ambiguous",
     "the server's own validation output for yolo12x seed 0. Both colliding grid processes "
     "wrote this one directory, so which of the two rows it belongs to cannot be recovered — "
     "kept for the record, not cited."),
    (LOCAL / "ship_yolo12x_seed1_val", "yolo12x_seed1_revalidation",
     "laptop re-validation of yolo12x seed 1; did not reproduce the server row"),
    (LOCAL / "ship_yolo12x_seed2_val", "yolo12x_seed2_revalidation_control",
     "laptop re-validation of yolo12x seed 2; reproduced the server row to -0.00002 — the "
     "control proving cross-machine re-validation is exact"),
    (ARCH / "ship_yolo26m_seed0", "yolo26m_seed0_aborted_server",
     "server attempt at yolo26m seed 0, interrupted at epoch 1; superseded by the laptop run"),
    (LOCAL / "_w8_partial_ship_yolo26m_seed0", "yolo26m_seed0_partial_batch8",
     "abandoned laptop yolo26m seed 0 at batch 8"),
    (LOCAL / "_w16_partial_ship_yolo26m_seed0", "yolo26m_seed0_partial_batch16",
     "abandoned laptop yolo26m seed 0 at batch 16"),
]

PLOT_SUFFIXES = (".png", ".jpg", ".jpeg")


def read_curve(results_csv: Path, patience: int) -> dict:
    """Derive the training-dynamics fields from one run's epoch curve.

    `time` counts seconds since the *current* invocation began, so it resets to a
    small value at every resume; the resets are what segment the curve.

    A genuine resume continues the epoch count from where the checkpoint left
    off, so the epoch column only ever increases in file order. A *decrease*
    means a second process started its own run from epoch 1 into the same file.
    That is the reliable signal of the shared-directory collision: duplicate
    epoch numbers alone miss `yolo12x` seed 1, whose two writers happened to
    cover disjoint epoch ranges. A shared dir invalidates both the segmentation
    and the wall-clock sum, so those runs report only the fields that survive it.
    """
    rows = list(csv.DictReader(open(results_csv, encoding="utf-8")))
    by_epoch: dict[int, dict] = {}
    for r in rows:
        by_epoch[int(float(r["epoch"]))] = r  # a re-run epoch overwrites the earlier one
    dup = len(rows) - len(by_epoch)
    eps = [int(float(r["epoch"])) for r in rows]
    regressions = sum(1 for a, b in zip(eps, eps[1:]) if b < a)

    peak = max(by_epoch.values(), key=lambda r: float(r[K5095]))
    best_epoch = int(float(peak["epoch"]))
    last_epoch = max(by_epoch)

    segments, cur, prev = [], [], -1.0
    for r in rows:
        t = float(r["time"])
        if t < prev:
            segments.append(cur)
            cur = []
        cur.append(r)
        prev = t
    segments.append(cur)

    out = {
        "best_epoch": best_epoch,
        "best_map50_95_curve": round(float(peak[K5095]), 5),
        "last_epoch": last_epoch,
        "epochs_trained": len(by_epoch),
        "patience_gap": last_epoch - best_epoch,
        "patience_fires_epoch": best_epoch + patience,
        "dup_epoch_rows": dup,
        "epoch_regressions": regressions,
    }
    if dup or regressions:
        # segmentation is meaningless once two writers interleave their clocks;
        # the gap verdict still holds, since it needs only the epoch/metric pairs
        past = "past" if last_epoch - best_epoch >= patience else "before"
        out |= {"resume_segments": "", "stopper_ref_epoch": "",
                "stop_reason": f"shared_dir_{past}_patience"}
        return out

    final = segments[-1]
    ref = max(final, key=lambda r: float(r[K5095]))  # the stopper's reference epoch
    ref_epoch = int(float(ref["epoch"]))
    resumed = len(segments) > 1
    if last_epoch == ref_epoch + patience:
        reason = "early_stop_after_resume" if resumed else "early_stop"
    elif last_epoch - best_epoch >= patience:
        reason = "killed_past_patience"
    else:
        reason = "killed_before_patience"
    out |= {"resume_segments": len(segments), "stopper_ref_epoch": ref_epoch,
            "stop_reason": reason}
    return out


def merge_csvs() -> tuple[list[dict], list[dict]]:
    """Return (record rows, superseded rows).

    The laptop CSV is authoritative: it already carries every server row plus the
    nine laptop runs and the re-derived yolo12x seed 0. Any archive row it does
    not reproduce exactly is not discarded — it is written to superseded.csv with
    the reason, so the withdrawn numbers stay auditable.
    """
    cur = list(csv.DictReader(open(CUR_CSV, encoding="utf-8")))
    arch = list(csv.DictReader(open(ARCH_CSV, encoding="utf-8")))
    live = {(r["variant"], r["seed"], r["train_time_s"]) for r in cur}
    superseded = []
    for r in arch:
        if (r["variant"], r["seed"], r["train_time_s"]) in live:
            continue
        cur_same = [c for c in cur if (c["variant"], c["seed"]) == (r["variant"], r["seed"])]
        if cur_same:
            why = (f"replaced by the row now in results.csv (map50_95 "
                   f"{float(cur_same[0]['map50_95']):.5f} vs {float(r['map50_95']):.5f})")
        else:
            why = "no counterpart in the live record"
        superseded.append(r | {"superseded_because": why})
    bad = [r for r in cur if r["split_fingerprint"] != FINGERPRINT]
    if bad:
        raise SystemExit(f"{len(bad)} row(s) carry a foreign split fingerprint; refusing to consolidate")
    return cur, superseded


def build_rows(cur: list[dict]) -> list[dict]:
    out = []
    for r in sorted(cur, key=lambda r: (ORDER.index(r["variant"]), int(r["seed"]))):
        v, s = r["variant"], int(r["seed"])
        src = SRC[(v, s)]
        run_id = f"{GRID_MAIN}_{v}_seed{s}"
        # re-entrant: once a run has been moved, read its curve from where it now lives
        moved = DEST / "runs" / run_id
        read_from = moved if (moved / "results.csv").is_file() else src["train"]
        args = yaml.safe_load(open(read_from / "args.yaml", encoding="utf-8"))
        patience = int(args["patience"])
        dyn = read_curve(read_from / "results.csv", patience)
        admissible = dyn["patience_gap"] >= patience
        out.append({
            "run_id": run_id, "grid": GRID_MAIN, "variant": v, "seed": s,
            "trained_on": src["trained_on"], "scored_on": src["scored_on"],
            "batch": args["batch"], "imgsz": args["imgsz"],
            "epochs_cfg": args["epochs"], "patience_cfg": patience,
            "precision": r["precision"], "recall": r["recall"],
            "map50": r["map50"], "map50_95": r["map50_95"],
            "params_m": r["params_m"], "gflops": r["gflops"],
            "train_time_s": r["train_time_s"],
            **dyn,
            "admissible": "yes" if admissible else "no",
            "exclude_reason": "" if admissible else
                f"trained only {dyn['patience_gap']} epochs past its peak (< patience {patience})",
            "run_path": f"runs/{run_id}",
            "train_dir_orig": str(src["train"].relative_to(REPO)).replace("\\", "/"),
            "val_dir_orig": str(src["val"].relative_to(REPO)).replace("\\", "/") if src["val"].is_dir() else "",
            "ultralytics_version": r["ultralytics_version"], "torch_version": r["torch_version"],
            "git_commit": r["git_commit"], "data_yaml": r["data_yaml"].replace("\\", "/"),
            "classes": r["classes"],
            "note": src.get("note", ""),
        })
    return out


def build_pilot_rows() -> tuple[list[dict], list[str]]:
    """Return (rows, run_ids_with_no_row) for the pilot grid.

    Same dynamics columns as the live grid, so the two files can be read the same
    way — but deliberately a different file. One directory (`yolo12s_seed2`) has
    complete artifacts and no CSV row: it was killed 3 epochs past its peak, so
    it never reached the post-training validation that writes a row. It cannot be
    recovered the way the live grid's missing rows were, because re-validating
    needs the split lists and those are gone.
    """
    rows = list(csv.DictReader(open(PILOT_CSV, encoding="utf-8")))
    foreign = [r for r in rows if r["split_fingerprint"] != PILOT_FINGERPRINT]
    if foreign:
        raise SystemExit(f"pilot CSV holds {len(foreign)} row(s) with an unexpected fingerprint")

    out = []
    for r in sorted(rows, key=lambda r: (PILOT_ORDER.index(r["variant"]), int(r["seed"]))):
        v, s = r["variant"], int(r["seed"])
        if v in PILOT_DROP:
            continue  # `main` measured this variant; its runs are the ones kept
        run_id = f"{GRID_PILOT}_{v}_seed{s}"
        moved = DEST / "runs" / run_id
        src = moved if (moved / "results.csv").is_file() else PILOT_SRC / f"ship_{v}_seed{s}"
        args = yaml.safe_load(open(src / "args.yaml", encoding="utf-8"))
        patience = int(args["patience"])
        dyn = read_curve(src / "results.csv", patience)
        admissible = dyn["patience_gap"] >= patience
        out.append({
            "run_id": run_id, "grid": GRID_PILOT, "variant": v, "seed": s,
            "trained_on": "server_wiped", "scored_on": "server_wiped",
            "batch": args["batch"], "imgsz": args["imgsz"],
            "epochs_cfg": args["epochs"], "patience_cfg": patience,
            "precision": r["precision"], "recall": r["recall"],
            "map50": r["map50"], "map50_95": r["map50_95"],
            "params_m": r["params_m"], "gflops": r["gflops"],
            "train_time_s": r["train_time_s"],
            **dyn,
            "admissible": "yes" if admissible else "no",
            "exclude_reason": "" if admissible else
                f"trained only {dyn['patience_gap']} epochs past its peak (< patience {patience})",
            "run_path": f"runs/{run_id}",
            "train_dir_orig": str((PILOT_SRC / f'ship_{v}_seed{s}').relative_to(REPO)).replace("\\", "/"),
            "val_dir_orig": str((PILOT_SRC / f'ship_{v}_seed{s}_val').relative_to(REPO)).replace("\\", "/")
                            if (PILOT_SRC / f"ship_{v}_seed{s}_val").is_dir() else "",
            "ultralytics_version": r["ultralytics_version"], "torch_version": r["torch_version"],
            "git_commit": r["git_commit"], "data_yaml": r["data_yaml"].replace("\\", "/"),
            "classes": r["classes"],
            "note": "",
        })

    banked = {r["run_id"] for r in out}
    on_disk = {f"{GRID_PILOT}_{d.name[len('ship_'):]}" for d in PILOT_SRC.iterdir()
               if d.is_dir() and d.name.startswith("ship_") and not d.name.endswith("_val")} \
        if PILOT_SRC.is_dir() else set()
    on_disk |= {d.name for d in (DEST / "runs").iterdir()
                if d.name.startswith(f"{GRID_PILOT}_")} if (DEST / "runs").is_dir() else set()
    dropped = {n for n in on_disk if any(n.startswith(f"{GRID_PILOT}_{v}_seed") for v in PILOT_DROP)}
    return out, sorted(on_disk - banked - dropped)


def migrate_layout(execute: bool) -> list[str]:
    """Bring an earlier two-CSV layout up to the current single-table one.

    The first consolidation put the live grid in `runs/<variant>_seed<n>` and the
    pilot in `pilot_2026-07-31/runs/<variant>_seed<n>` with its own CSV. Both are
    now one `runs/` tree keyed by `<grid>_<variant>_seed<n>`, because `yolo12s`
    seeds 0-2 exist in both grids and would otherwise overwrite each other.
    """
    log = []
    runs = DEST / "runs"
    if runs.is_dir():
        for d in sorted(runs.iterdir()):
            if d.is_dir() and not d.name.startswith((f"{GRID_MAIN}_", f"{GRID_PILOT}_")):
                dst = runs / f"{GRID_MAIN}_{d.name}"
                log.append(f"  rename   runs/{d.name} -> runs/{dst.name}")
                if execute:
                    d.rename(dst)
    old_pilot = PILOT_DEST / "runs"
    if old_pilot.is_dir():
        for d in sorted(old_pilot.iterdir()):
            if not d.is_dir():
                continue
            dst = runs / f"{GRID_PILOT}_{d.name}"
            log.append(f"  move     pilot_2026-07-31/runs/{d.name} -> runs/{dst.name}")
            if execute and not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(d), str(dst))
    if execute and PILOT_DEST.is_dir():
        stale = PILOT_DEST / "results.csv"  # superseded by the merged table
        if stale.is_file():
            stale.unlink()
            log.append("  remove   pilot_2026-07-31/results.csv (merged into results.csv)")
        readme = PILOT_DEST / "README.md"
        if readme.is_file() and not PILOT_DOC.is_file():
            shutil.move(str(readme), str(PILOT_DOC))
            log.append(f"  move     pilot_2026-07-31/README.md -> {PILOT_DOC.relative_to(REPO)}")
        elif readme.is_file():
            readme.unlink()  # already carried over to docs/
        p = PILOT_DEST / "runs"
        if p.is_dir() and not any(p.iterdir()):
            p.rmdir()
        if not any(PILOT_DEST.iterdir()):
            PILOT_DEST.rmdir()
            log.append("  remove   pilot_2026-07-31/ (contents merged)")
    return log


def move(src: Path, dst: Path, execute: bool) -> str:
    if not src.exists():
        return f"  MISSING  {src}"
    if dst.exists():
        return f"  SKIP     {dst.relative_to(REPO)} already exists"
    if execute:
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(src), str(dst))
    return f"  move     {src.relative_to(REPO)}\n           -> {dst.relative_to(REPO)}"


def lay_out_run(run_id: str, src: dict, execute: bool, dest_root: Path = DEST) -> list[str]:
    """Move one run into the uniform layout: args.yaml, results.csv, weights/, val/, plots/."""
    log = []
    dest = dest_root / "runs" / run_id
    train = src["train"]
    if dest.exists():
        return [f"  SKIP     runs/{run_id} already exists"]
    if execute:
        dest.mkdir(parents=True)
        (dest / "plots").mkdir()
    for item in sorted(train.iterdir()) if train.is_dir() else []:
        if item.is_dir() and item.name == "weights":
            log.append(move(item, dest / "weights", execute))
        elif item.is_dir() and (item / "results.csv").is_file():
            continue  # a nested run dir; it is moved as its own entry
        elif item.is_dir():
            # editor/tooling noise such as .ipynb_checkpoints — carried along so the
            # source directory empties out and leaves no husk that looks like a run
            log.append(move(item, dest / item.name, execute))
        elif item.suffix.lower() in PLOT_SUFFIXES:
            log.append(move(item, dest / "plots" / item.name, execute))
        else:
            log.append(move(item, dest / item.name, execute))
    val = src["val"]
    if val.is_dir():
        log.append(move(val, dest / "val", execute))
    else:
        log.append(f"  no val   {run_id}")
    if execute and train.is_dir() and not any(train.iterdir()):
        train.rmdir()
    return log


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--plan", action="store_true", help="print the full plan, move nothing")
    g.add_argument("--execute", action="store_true")
    args = p.parse_args()
    execute = args.execute

    if execute:
        DEST.mkdir(exist_ok=True)
        (DEST / "runs").mkdir(exist_ok=True)
        (DEST / "extra").mkdir(exist_ok=True)
    print("[consolidate] migrating layout")
    for line in migrate_layout(execute):
        print(line)

    cur, superseded = merge_csvs()
    rows = build_rows(cur)
    pilot, pilot_unbanked = build_pilot_rows()
    print(f"\n[consolidate] main grid:  {len(rows)} rows, {len(superseded)} superseded, "
          f"{sum(r['admissible'] == 'yes' for r in rows)} admissible")
    print(f"[consolidate] pilot grid: {len(pilot)} rows, "
          f"{sum(r['admissible'] == 'yes' for r in pilot)} admissible, "
          f"{len(pilot_unbanked)} run(s) with artifacts but no row: {pilot_unbanked or 'none'}")

    both = sorted(rows + pilot, key=lambda r: (ORDER_ALL.index(r["variant"]),
                                               r["grid"] != GRID_MAIN, int(r["seed"])))
    dupes = {k for k in ((r["variant"], r["seed"]) for r in both)
             if sum(1 for r in both if (r["variant"], r["seed"]) == k) > 1}
    if dupes:
        print(f"[consolidate] (variant, seed) present in BOTH grids: {sorted(dupes)} "
              f"-> disambiguated by the `grid` column and the run_id prefix")
    ids = [r["run_id"] for r in both]
    if len(ids) != len(set(ids)):
        raise SystemExit("duplicate run_id in the merged table — refusing to write")

    for r in rows:
        print(f"  {r['run_id']:22} best ep {r['best_epoch']:>3}  last ep {r['last_epoch']:>3}  "
              f"gap {r['patience_gap']:>3}  ref ep {str(r['stopper_ref_epoch']):>3}  "
              f"seg {str(r['resume_segments']):>2}  {r['stop_reason']:<24} {r['admissible']}")

    print("\n[consolidate] moving Table 1 runs")
    for r in rows:
        for line in lay_out_run(r["run_id"], SRC[(r["variant"], int(r["seed"]))], execute):
            print(line)

    print("\n[consolidate] moving non-Table-1 artifacts")
    for src, name, _why in EXTRA:
        print(move(src, DEST / "extra" / name, execute))

    print("\n[consolidate] moving pilot runs")
    for run_id in [r["run_id"] for r in pilot] + pilot_unbanked:
        bare = run_id[len(GRID_PILOT) + 1:]
        src = {"train": PILOT_SRC / f"ship_{bare}", "val": PILOT_SRC / f"ship_{bare}_val"}
        for line in lay_out_run(run_id, src, execute):
            print(line)

    if PILOT_DROP:
        print(f"\n[consolidate] retiring pilot runs of {sorted(PILOT_DROP)} "
              f"(measured by the main campaign instead)")
        for v in sorted(PILOT_DROP):
            for d in sorted((DEST / "runs").glob(f"{GRID_PILOT}_{v}_seed*")):
                print(move(d, DEST / "extra" / f"{d.name}_superseded_by_main", execute))
            for d in sorted(PILOT_SRC.glob(f"ship_{v}_seed*")) if PILOT_SRC.is_dir() else []:
                print(move(d, DEST / "extra" / f"{GRID_PILOT}_{d.name[len('ship_'):]}_superseded_by_main",
                           execute))

    if not execute:
        print("\n[consolidate] --plan: nothing was moved or written")
        return 0

    fields = list(both[0].keys())
    with open(DEST / "results.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(both)
    print(f"\n[consolidate] wrote results.csv ({len(both)} rows, {len(fields)} columns)")

    if superseded:
        # `split_fingerprint` is dropped here too: the split each grid ran on is
        # documented in phase1_benchmark/README.md and docs/phase1-experimental-record.md
        sfields = [c for c in superseded[0] if c != "split_fingerprint"]
        with open(DEST / "superseded.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=sfields, extrasaction="ignore")
            w.writeheader()
            w.writerows(superseded)
        print(f"[consolidate] wrote superseded.csv ({len(superseded)} rows)")

    (DEST / "extra" / "CONTENTS.json").write_text(
        json.dumps([{"dir": n, "was": str(s.relative_to(REPO)).replace("\\", "/"), "why": w}
                    for s, n, w in EXTRA], indent=2), encoding="utf-8")
    for line in migrate_layout(execute):  # drop the now-empty pilot subtree
        print(line)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
