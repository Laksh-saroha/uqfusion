"""Multi-seed benchmark grid (plan C5 -> Table 1, scope §9.1/§9.5).

One row per (variant, seed): train with the shared config, validate on the
common val split, append metrics to a CSV. Resume-safe: rows already in the
CSV are skipped, so a crashed grid restarts where it stopped — an 18-run grid
on a shared server needs that more than elegance.
"""

from __future__ import annotations

import csv
import hashlib
import json
import subprocess
import time
from functools import lru_cache
from pathlib import Path

RESULT_FIELDS = [
    "variant", "seed", "precision", "recall", "map50", "map50_95",
    "params_m", "gflops", "epochs_cfg", "train_time_s", "run_dir",
    "ultralytics_version", "torch_version", "git_commit",
    "data_yaml", "split_fingerprint", "classes",
    # R-E1 slice 3 (F14). `split_fingerprint` hashes frame NAMES, so every label edit
    # this project has ever made is invisible to it; `epochs_cfg` was recorded but
    # never consulted, and imgsz/batch/weights/overrides were not recorded at all.
    # Both defects were reproduced before these columns existed -- see
    # docs/recipe-identity-2026-09-10.md.
    "label_fingerprint_trainval", "recipe_fingerprint", "recipe",
]


def resolve_device(cfg: dict):
    d = cfg.get("device", "auto")
    return None if d in (None, "auto") else d


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], text=True,
            cwd=Path(__file__).resolve().parents[3],
        ).strip()
    except Exception:  # noqa: BLE001 - commit id is metadata, never fatal
        return "unknown"


def split_fingerprint(data_yaml: str | Path) -> str:
    """Machine-independent identity of the train+val frame sets.

    Hashes sorted `run/filename` ids (not absolute paths, not txt bytes), so the
    same split fingerprints identically across machines and list formats, and a
    re-split — the failure mode this guards against — changes it. Test is
    excluded: Phase 1 never touches it.
    """
    from uqfusion.data.lists import load_data_yaml, run_key, split_image_list

    data = load_data_yaml(data_yaml)
    h = hashlib.sha256()
    for split in ("train", "val"):
        ids = sorted(f"{run_key(p)}/{p.name}" for p in split_image_list(data, split))
        h.update(f"{split}:{len(ids)}\n".encode())
        h.update("\n".join(ids).encode())
    return h.hexdigest()[:12]


@lru_cache(maxsize=4)
def label_fingerprint(data_yaml: str | Path) -> str:
    """Content identity of the train+val LABELS. What `split_fingerprint` cannot see.

    `split_fingerprint` hashes sorted `run/filename` ids, so it answers "which frames?"
    and nothing about what is written in them. Reproduced on the real function: deleting
    a box and changing a class id both leave it at `c3354ed2f2b1`. That is exactly the
    edit this project made twice -- the night cut removed 132,688 boxes and the
    restoration put 94,553 back -- and a results CSV could not tell the two apart.

    **The scope is in the name on purpose.** `label_content_hash` is the ledger's
    algorithm unchanged, so this number is directly comparable to
    `runs/label_hash_ledger.csv` and `verify_dataset_state.py --expect-label-hash`
    -- but only to rows of the SAME scope. The ledger's `train` row (`8ed69b5974ed`
    on the restored tree) is over train alone and its `all` row over train+val+test;
    this is train+val, matching `split_fingerprint`, and equals neither. The ledger
    doc already records what happens when two such numbers are compared as if they
    were the same quantity.

    Cost: ~15 s over the 107,627 VIS label files with a warm cache, paid once per
    grid launch against runs measured in hours. `lru_cache` keeps it to once.
    """
    from uqfusion.data.labels import label_content_hash
    from uqfusion.data.lists import load_data_yaml, split_image_list

    data = load_data_yaml(data_yaml)
    images: list[Path] = []
    for split in ("train", "val"):
        images.extend(split_image_list(data, split))
    return label_content_hash(images)


def _sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()[:16]


# Knobs that change the EXPERIMENT. Anything absent from this list is either already a
# key (variant, seed), already its own column (classes, the two fingerprints), or
# machine-local. `workers` and `device` are deliberately excluded: they can perturb
# nondeterminism but they are properties of the host, and including them would make a
# fingerprint that never matches across machines -- which would break resume, the one
# thing this lookup exists to do.
RECIPE_KEYS = ("epochs", "imgsz", "batch", "mosaic", "close_mosaic", "optimizer",
               "patience", "amp", "deterministic", "weights", "weights_sha256",
               "train_overrides")


def recipe_identity(**knobs) -> tuple[str, str]:
    """(fingerprint, canonical json) for the training recipe.

    R-E1: *"the grid's completed-run lookup omits the recipe, so changing epochs,
    imgsz, initial weights or overrides can silently skip a different experiment."*
    Reproduced: with one row for (yolo26m, 0) at `epochs_cfg` 25, a re-run at 50 and at
    100 both skip as "already done" and the CSV keeps reporting the 25-epoch number.

    `weights_sha256` is the CONTENT hash of the starting checkpoint, not its path --
    R-E1 asks for a checkpoint content hash, and a path is not one: `best.pt` is
    overwritten by every run that produces it.
    """
    w = knobs.get("weights")
    knobs = dict(knobs)
    knobs["weights"] = str(w) if w is not None else None
    knobs["weights_sha256"] = (_sha256_file(w) if w is not None and Path(w).is_file()
                               else None)
    ordered = {k: knobs.get(k) for k in RECIPE_KEYS}
    blob = json.dumps(ordered, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode()).hexdigest()[:12], blob


def _completed(csv_path: Path, classes_tag: str) -> dict[tuple[str, str], dict]:
    """{(variant, seed): {dataset, recipe}} for rows already in the CSV.

    `dataset` is the (split, labels, classes) triple that says WHICH EXPERIMENT this
    CSV belongs to; two different datasets must never share a file. `recipe` is the
    hash of how the model was trained, which may legitimately differ between rows of
    the same CSV and must therefore never be used to refuse a mix -- only to decide
    whether a given row already answers the run being requested.

    Rows whose class filter differs, or that predate a stamping, get '' in the
    corresponding slot so the mix-refusal trips on them. That is the behaviour
    `split_fingerprint` already had; `label_fingerprint_trainval` now joins it.
    """
    if not csv_path.is_file():
        return {}
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        out = {}
        for row in csv.DictReader(f):
            same_classes = (row.get("classes") or "all") == classes_tag
            out[(row["variant"], row["seed"])] = {
                "dataset": ((row.get("split_fingerprint") or "",
                             row.get("label_fingerprint_trainval") or "")
                            if same_classes else ("", "")),
                "recipe": row.get("recipe_fingerprint") or "",
            }
        return out


def _append_row(csv_path: Path, row: dict) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not csv_path.is_file()
    if not new_file:
        # A CSV written under an older RESULT_FIELDS would silently misalign
        # columns on append — refuse instead.
        with open(csv_path, "r", encoding="utf-8", newline="") as f:
            header = next(csv.reader(f), [])
        if header != RESULT_FIELDS:
            raise RuntimeError(
                f"{csv_path} header {header} != current RESULT_FIELDS — "
                "written by an older schema. Use a fresh --out-csv."
            )
    with open(csv_path, "a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=RESULT_FIELDS)
        if new_file:
            writer.writeheader()
        writer.writerow(row)


def _box_metrics(metrics) -> dict[str, float]:
    """Extract P/R/mAP from an Ultralytics DetMetrics, tolerating API drift."""
    try:
        return {
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
        }
    except AttributeError:
        rd = metrics.results_dict
        return {
            "precision": float(rd["metrics/precision(B)"]),
            "recall": float(rd["metrics/recall(B)"]),
            "map50": float(rd["metrics/mAP50(B)"]),
            "map50_95": float(rd["metrics/mAP50-95(B)"]),
        }


def _retarget_checkpoint(ckpt_path: Path, run_dir: Path) -> None:
    """Point a checkpoint's recorded save_dir at where its run actually lives.

    `model.train(resume=True)` reloads project/name/save_dir from the checkpoint's own
    `train_args`, ignoring anything passed at the call site. A run directory that has
    been MOVED therefore resumes into its old location and silently re-creates it —
    which is exactly what happened when the ensemble members were relocated out of the
    benchmark tree. Correct the checkpoint on disk first; no-op when it already agrees.
    """
    import torch

    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    args = ck.get("train_args")
    if not isinstance(args, dict) or Path(str(args.get("save_dir", ""))) == run_dir:
        return
    args["project"], args["name"], args["save_dir"] = str(run_dir.parent), run_dir.name, str(run_dir)
    torch.save(ck, ckpt_path)
    print(f"[grid] checkpoint save_dir retargeted -> {run_dir}")


def plan_grid(done: dict, dataset: tuple, recipe_fp: str,
              variants: list, seeds: list) -> tuple[list, list]:
    """(stale, conflict) for a CSV's existing rows. Pure, so it can be tested.

    **stale** — rows from a different split fingerprint, LABEL STATE or class filter.
    Two different datasets must never share a CSV, so any of these refuses the whole
    run. `label_fingerprint_trainval` is new in R-E1 slice 3, so pre-existing rows
    carry '' and trip this: those rows were produced against a label tree that has
    since changed twice and the CSV cannot say which state each one saw.

    **conflict** — rows for a (variant, seed) THIS grid is about to run, on the same
    dataset but under a different recipe. Deliberately NOT part of the stale check:
    one CSV may legitimately hold rows trained under different budgets, and refusing
    to mix those would be wrong. What it must not do is let a different recipe be
    skipped as already done, which is R-E1's case and was reproduced: a CSV holding
    (yolo26m, 0) at 25 epochs skipped a re-request at 50 and at 100 and went on
    reporting the 25-epoch number.

    Conflicts refuse rather than re-run, because the run directory and the CSV row are
    both keyed by `name`: re-running would overwrite a different experiment's outputs,
    which is the failure this item is about.
    """
    stale = sorted(k for k, v in done.items() if v["dataset"] != dataset)
    wanted = {(str(v), str(s)) for v in variants for s in seeds}
    conflict = sorted(k for k, v in done.items()
                      if (str(k[0]), str(k[1])) in wanted
                      and v["dataset"] == dataset and v["recipe"] != recipe_fp)
    return stale, conflict


def run_grid(
    cfg: dict,
    data_yaml: str | Path,
    variants: list[str] | None = None,
    seeds: list[int] | None = None,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    out_csv: str | Path | None = None,
    run_prefix: str = "bench",
    classes: list[int] | None = None,
    mosaic: float | None = None,
    close_mosaic: int | None = None,
    resume: bool = True,
    weights: str | Path | None = None,
    callbacks: dict[str, list] | None = None,
    run_name: str | None = None,
    train_overrides: dict | None = None,
    runs_root: str | Path | None = None,
) -> Path:
    """Run the grid; return the results CSV path. Every argument defaults to config.yaml.

    resume=True picks an interrupted run back up from its own weights/last.pt instead of
    restarting it at epoch 0. Grid-level resume only ever skipped whole (variant, seed)
    rows, so a machine that died 90 epochs into yolo26x threw all 90 away.

    `weights` starts the run from an explicit checkpoint instead of the variant's COCO
    weights (a fine-tune continuation stage). `run_name` overrides the computed
    `{run_prefix}_{variant}_seed{seed}` name — required when calling this more than once
    for the same (variant, seed) (e.g. a main run and its `_ft` continuation), since the
    CSV completion check and the run dir are both keyed by `name`; pass a distinct
    `out_csv` too in that case so the two stages' rows don't collide. `callbacks` is
    registered on the model before training, same shape as `uqfusion.uq.train_gaussian`'s
    (queue heartbeat/pause wiring, kept generic so any trainer can use it).

    `runs_root` overrides where the run directories go. It defaults to
    `outputs_root/benchmark/runs`, which is correct for the Phase 1 grid itself, but a
    caller that merely *reuses* the grid runner to train something else (e.g.
    `train_ensemble_member`) should pass its own root so its runs do not land in — and
    get collected with — the benchmark tree. `out_csv` is independent of this.
    """
    import torch
    import ultralytics
    from ultralytics import YOLO

    from uqfusion.uq.variants import build_variant, is_variant

    b = cfg["benchmark"]
    variants = variants if variants is not None else b["variants"]
    seeds = seeds if seeds is not None else b["seeds"]
    epochs = epochs if epochs is not None else b["epochs"]
    imgsz = imgsz if imgsz is not None else b["imgsz"]
    batch = batch if batch is not None else b["batch"]
    workers = workers if workers is not None else b["workers"]

    out_root = Path(cfg["paths"]["outputs_root"]) / "benchmark"
    out_csv = Path(out_csv) if out_csv else out_root / "benchmark_results.csv"
    runs_dir = Path(runs_root) if runs_root else out_root / "runs"
    device = resolve_device(cfg)
    commit = _git_commit()
    fingerprint = split_fingerprint(data_yaml)
    label_fp = label_fingerprint(str(data_yaml))
    classes_tag = " ".join(str(c) for c in classes) if classes else "all"
    recipe_fp, recipe_json = recipe_identity(
        epochs=epochs, imgsz=imgsz, batch=batch, mosaic=mosaic,
        close_mosaic=close_mosaic, optimizer=b.get("optimizer", "auto"),
        patience=b["patience"], amp=b["amp"], deterministic=b["deterministic"],
        weights=weights, train_overrides=train_overrides)
    done = _completed(out_csv, classes_tag)

    # Dataset identity = which frames AND which labels AND which class filter.
    # `label_fingerprint_trainval` is new in R-E1 slice 3, so every row written
    # before it carries '' and trips the stale check. That is the intended reading:
    # those rows were produced against a label tree that has since changed twice.
    dataset = (fingerprint, label_fp)
    stale, conflict = plan_grid(done, dataset, recipe_fp, variants, seeds)
    if stale:
        raise RuntimeError(
            f"{out_csv} holds {len(stale)} row(s) whose split/label fingerprint or "
            f"classes != current (split '{fingerprint}', labels '{label_fp}', classes "
            f"'{classes_tag}') (e.g. {stale[0]}): the CSV belongs to a different "
            f"experiment. Quarantine it (mv) or pass a fresh --out-csv before running "
            f"this grid."
        )
    if conflict:
        raise RuntimeError(
            f"{out_csv} already holds {len(conflict)} row(s) for this grid's "
            f"(variant, seed) under a DIFFERENT recipe (e.g. {conflict[0]}: recipe "
            f"'{done[conflict[0]]['recipe']}' vs current '{recipe_fp}'). Skipping them "
            f"would report the other recipe's numbers as this one's; re-running them "
            f"would overwrite its run directory, since both are keyed by name. "
            f"Pass a fresh --out-csv and --run-prefix/--run-name for this recipe. "
            f"Current recipe: {recipe_json}"
        )

    for variant in variants:
        for seed in seeds:
            prev = done.get((variant, str(seed)))
            if prev is not None and prev["dataset"] == dataset and prev["recipe"] == recipe_fp:
                print(f"[grid] skip {variant} seed {seed} — already in {out_csv.name} "
                      f"under the same dataset and recipe ({recipe_fp})")
                continue
            name = run_name or f"{run_prefix}_{variant}_seed{seed}"
            # An interrupted run leaves weights/last.pt behind; the row is absent from the
            # CSV (only written after val), so we land here with a checkpoint to continue.
            last_ckpt = runs_dir / name / "weights" / "last.pt"
            resuming = resume and last_ckpt.is_file()
            if resuming:
                print(f"[grid] === {name}: RESUME from {last_ckpt}")
            elif weights is not None:
                print(f"[grid] === {name}: starting from {weights}")
            else:
                print(f"[grid] === {name}: {epochs} epochs, imgsz {imgsz}, batch {batch}, data {data_yaml}")

            if resuming:
                _retarget_checkpoint(last_ckpt, runs_dir / name)
                model = YOLO(str(last_ckpt))
            elif weights is not None:
                if not Path(weights).is_file():
                    raise FileNotFoundError(f"start weights not found: {weights}")
                model = YOLO(str(weights))
            elif is_variant(variant):
                # An architecture edit with no released checkpoint (uq/variants.py):
                # built from its yaml and seeded from the nearest base model through a
                # declared index remap.
                model, transfer_pct = build_variant(variant)
                print(f"[grid] === {name}: variant {variant}, {transfer_pct:.1f}% seeded")
            else:
                w = f"{variant}.pt" if b.get("pretrained", True) else f"{variant}.yaml"
                model = YOLO(w)

            for event, fns in (callbacks or {}).items():
                for fn in fns:
                    model.add_callback(event, fn)

            params = sum(p.numel() for p in model.model.parameters())
            try:
                from ultralytics.utils.torch_utils import get_flops

                gflops = float(get_flops(model.model, imgsz))
            except Exception:  # noqa: BLE001 - profiling is metadata, never fatal
                gflops = float("nan")

            t0 = time.time()
            # mosaic/close_mosaic are train-only hyperparameters; left unset -> Ultralytics
            # default. Not stamped in the CSV schema — the exact value lands in each run
            # dir's args.yaml, and separate --out-csv/--run-prefix keep ablation arms apart.
            extra = dict(train_overrides or {})
            if mosaic is not None:
                extra["mosaic"] = mosaic
            if close_mosaic is not None:
                extra["close_mosaic"] = close_mosaic
            if resuming:
                # resume=True makes Ultralytics reload epoch/optimiser/EMA state and every
                # hyperparameter from the checkpoint's own args.yaml — passing them again
                # here would be ignored at best and conflict at worst.
                try:
                    model.train(resume=True)
                except AssertionError as exc:
                    # "...training to N epochs is finished, nothing to resume." The run
                    # completed but died before its row was written; fall through to val.
                    if "nothing to resume" not in str(exc):
                        raise
                    print(f"[grid] {name} was already trained out — validating only")
            else:
                # One dict, merged in precedence order (extra wins), not spread as
                # separate keywords: extra can legitimately override e.g. `patience`
                # (the fine-tune stage's 10-epoch tail), which would otherwise collide
                # with the explicit keyword of the same name below.
                kwargs = dict(
                    data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                    seed=seed, deterministic=b["deterministic"], optimizer=b.get("optimizer", "auto"),
                    patience=b["patience"], amp=b["amp"], workers=workers, device=device,
                    classes=classes,
                    project=str(runs_dir), name=name, exist_ok=True, verbose=True,
                )
                kwargs.update(extra)
                model.train(**kwargs)
            # Wall time of THIS invocation. A resumed run therefore under-reports; the
            # run dir's own results.csv holds the per-epoch cumulative timing if the
            # total matters (it does not for Table 1, which ranks accuracy).
            train_time = time.time() - t0

            # Validate the best checkpoint explicitly so the reported numbers are
            # unambiguous (not "whatever epoch the trainer last printed").
            best = getattr(getattr(model, "trainer", None), "best", None)
            if not (best and Path(str(best)).is_file()):
                # No live trainer (resume that had nothing left to do): take best.pt off disk.
                best = runs_dir / name / "weights" / "best.pt"
            eval_model = YOLO(str(best)) if Path(str(best)).is_file() else model
            # workers=0 is load-bearing, not a tuning knob. `model.trainer` is still
            # referenced three lines up, so its `workers` val loaders are alive when
            # this call builds its own. Omitting workers here took Ultralytics'
            # default of 8, so peak was workers+8 loader processes against a 1.0 GB
            # /dev/shm that queue.json had already measured as fatal at 8 alone.
            # It won that race for five runs and lost it on ens_vis_seed0_ft
            # (2026-08-25), which then poisoned four more. Loading in-process removes
            # the overlap entirely. Measured metric-neutral: ens_vis_seed0's best.pt
            # scored 0.25200973629816487 mAP50-95 at both workers=8 and workers=0,
            # bit-identical to the row recorded before this change -- val has no
            # augmentation and yields batches in index order at any worker count.
            # Costs ~1.6 min per run (9.8 -> 4.2 it/s) against an ~8 h train.
            # Do NOT "fix" shm pressure by lowering the TRAINING workers instead:
            # ultralytics seeds worker w with base_seed+w and sample i is handled by
            # worker i%nw (data/build.py seed_worker), so changing nw rewrites every
            # augmentation draw and breaks comparability across the family.
            metrics = eval_model.val(
                data=str(data_yaml), imgsz=imgsz, device=device, split="val",
                classes=classes, workers=0,
                project=str(runs_dir), name=f"{name}_val", exist_ok=True,
            )

            row = {
                "variant": variant, "seed": seed,
                **_box_metrics(metrics),
                "params_m": round(params / 1e6, 2),
                "gflops": round(gflops, 1) if gflops == gflops else "",
                "epochs_cfg": epochs, "train_time_s": round(train_time, 1),
                "run_dir": str(runs_dir / name),
                "ultralytics_version": ultralytics.__version__,
                "torch_version": torch.__version__, "git_commit": commit,
                "data_yaml": str(data_yaml), "split_fingerprint": fingerprint,
                "classes": classes_tag,
                "label_fingerprint_trainval": label_fp,
                "recipe_fingerprint": recipe_fp, "recipe": recipe_json,
            }
            _append_row(out_csv, row)
            print(f"[grid] {name} done in {train_time / 60:.1f} min -> {out_csv}")
    return out_csv
