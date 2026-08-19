"""Training entry for the Gaussian-head model (O2) via the stock Ultralytics loop.

The only deviations from a standard detection run:
  1. get_model() converts the freshly built+loaded model in place (cv4 + class swap);
  2. a callback keeps criterion.epoch current so the NLL warm-up/ramp works;
  3. loss_names gains a 4th entry ("nll_loss") for logging/CSV.
Everything else — optimizer, EMA, AMP, augmentation, validator — is untouched,
which is exactly the point (scope §4: detector as-is except the variance branch).
"""

from __future__ import annotations

from pathlib import Path

from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.nn.tasks import DetectionModel
from ultralytics.utils import DEFAULT_CFG, LOGGER, RANK

from uqfusion.uq.gaussian import DEFAULT_GAUSSIAN_CFG, convert_to_gaussian
from uqfusion.uq.variants import build_variant, is_variant


def _sync_criterion_epoch(trainer) -> None:
    """on_train_epoch_start: push the trainer epoch into the loss for warm-up/ramp."""
    model = getattr(trainer.model, "module", trainer.model)  # unwrap DDP if present
    criterion = getattr(model, "criterion", None)
    if criterion is not None and hasattr(criterion, "epoch"):
        criterion.epoch = trainer.epoch


class GaussianTrainer(DetectionTrainer):
    """DetectionTrainer that trains the σ²-augmented model."""

    gaussian_cfg: dict = {}  # set per-run via make_gaussian_trainer

    def __init__(self, cfg=DEFAULT_CFG, overrides=None, _callbacks=None):
        super().__init__(cfg, overrides, _callbacks)
        self.add_callback("on_train_epoch_start", _sync_criterion_epoch)

    def get_model(self, cfg=None, weights=None, verbose=True):
        """Build the detector, convert it, THEN load weights — order is load-bearing.

        `DetectionTrainer.get_model` builds and loads in one step. Calling it first
        and converting after silently drops the σ branch on **resume**:
        `BaseModel.load` intersects the checkpoint state dict against a model that
        has no `cv4` yet, so every σ key falls out and the branch restarts from its
        zero-bias init while the detector carries on from epoch N. σ that resets
        mid-run is finite, positive and non-degenerate — it passes every aggregate
        check while being wrong, the same failure class the end2end gate exists for.
        Converting first puts the σ keys in the intersection. A fresh run is
        unaffected: COCO weights carry no `cv4`, so the branch keeps its init.
        """
        model = DetectionModel(cfg, nc=self.data["nc"], ch=self.data["channels"],
                               verbose=verbose and RANK == -1)
        model = convert_to_gaussian(model, self.gaussian_cfg)
        if weights:
            model.load(weights)
        return model

    def get_validator(self):
        validator = super().get_validator()  # parent sets 3 loss names; widen to 4
        self.loss_names = ("box_loss", "cls_loss", "dfl_loss", "nll_loss")
        return validator


def make_gaussian_trainer(gaussian_cfg: dict | None = None) -> type[GaussianTrainer]:
    """A GaussianTrainer subclass carrying this run's gaussian config.

    Ultralytics instantiates the trainer class itself (model.train(trainer=cls)),
    so per-run config rides on a dynamically configured subclass.
    """
    cfg = {**DEFAULT_GAUSSIAN_CFG, **(gaussian_cfg or {})}
    return type("ConfiguredGaussianTrainer", (GaussianTrainer,), {"gaussian_cfg": cfg})


def restore_early_stopping(trainer) -> None:
    """on_train_start: re-seed the EarlyStopping counter from the run's own history.

    Ultralytics rebuilds `EarlyStopping` in `_setup_train` and `resume_training`
    never restores it, so after a resume patience counts from a *local* peak
    instead of the run's best (observed in Phase 1 — see the experimental record).
    Uninterrupted that is a curiosity; with pause/resume as a routine operation it
    silently extends every paused run past the stopping rule the un-paused runs
    obeyed, which is not a comparison you can put in a table.

    Fitness must match Ultralytics' own definition or the restore is worse than
    doing nothing. In 8.4.90 `DetMetrics.fitness` weights are [0, 0, 0, 1] — it is
    **mAP50-95 alone**, not the 0.1·mAP50 + 0.9·mAP50-95 blend of older releases
    (the same version fact `consolidate_phase1.py` and `recover_row.py` record).
    Using the blend here inflates the restored best by ~15%: the stopper then
    never sees a real improvement, so it freezes `best_epoch` at the resume point
    and stops exactly `patience` epochs later, and `trainer.best_fitness` is
    seeded high enough that `best.pt` may never be written again. A run paused
    near its peak would silently return the checkpoint it happened to hold at the
    pause. Verified against `best.pt`'s stored `train_metrics.fitness`, which
    equals that epoch's mAP50-95 exactly.
    """
    stopper = getattr(trainer, "stopper", None)
    csv_path = Path(trainer.save_dir) / "results.csv"
    if stopper is None or not csv_path.is_file():
        return

    import csv as _csv

    best_fitness, best_epoch = 0.0, 0
    with open(csv_path, "r", encoding="utf-8", newline="") as f:
        for row in _csv.DictReader(f):
            try:
                fitness = float(row["metrics/mAP50-95(B)"])
                epoch = int(float(row["epoch"]))
            except (KeyError, ValueError, TypeError):
                continue
            if fitness > best_fitness:
                best_fitness, best_epoch = fitness, epoch
    if best_fitness <= 0:
        return

    stopper.best_fitness, stopper.best_epoch = best_fitness, best_epoch
    if not trainer.best_fitness:
        trainer.best_fitness = best_fitness
    LOGGER.info(f"[gaussian] early-stopping restored: best fitness {best_fitness:.5f} "
                f"at epoch {best_epoch} (patience {stopper.patience})")


def reseed_at_train_start(seed: int, deterministic: bool):
    """on_train_start: re-seed every RNG just before the first batch is drawn.

    §12.1 needs the σ and parity arms to be the same experiment, and they were
    not: probing the Python `random` state at `on_pretrain_routine_start` showed
    the two arms already diverged during trainer construction, while numpy and
    torch matched. Ultralytics augments with Python `random`, so the two arms saw
    different augmentations from batch 1 and drifted apart for reasons unrelated
    to σ — 0.011 mAP50-95 on the first real pair, 2.4x the Phase 1 seed sd, which
    reads exactly like "the σ branch costs accuracy".

    Rather than chase which construction-time call consumed a draw (model build,
    conversion and weight load are all clean in isolation), this resets all three
    generators after the dataloaders exist and before any augmentation is drawn.
    Both arms therefore enter the epoch loop from an identical state by
    construction. It is the same seed Ultralytics already used, so nothing about
    the run changes except that it stops depending on setup-time RNG traffic.
    """
    def cb(trainer):
        from ultralytics.utils.torch_utils import init_seeds

        init_seeds(seed, deterministic=deterministic)
    return cb


def train_gaussian(
    cfg: dict,
    data_yaml: str | Path,
    variant: str,
    seed: int = 0,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    run_name: str | None = None,
    gaussian_overrides: dict | None = None,
    resume: bool = True,
    callbacks: dict[str, list] | None = None,
    sigma: bool = True,
    out_subdir: str = "gaussian",
    weights: str | Path | None = None,
    train_overrides: dict | None = None,
):
    """Train one Gaussian-head model with the shared project config. Returns (best_weights_path, run_dir).

    `sigma=False` trains the same variant, data, seed and hyperparameters through
    the stock `DetectionTrainer` — the §12.1 parity arm. It lives here rather than
    in a separate entry point on purpose: the parity row's whole claim is that it
    differs from the σ row in exactly one thing, the trainer class, and two code
    paths that "look the same" are how that claim quietly stops being true.

    `resume=True` continues an interrupted run from its own `weights/last.pt`
    instead of restarting at epoch 0, matching the grid runner's behaviour. On
    resume Ultralytics reloads every hyperparameter from the checkpoint's own
    args; only a short allow-list (imgsz, batch, device, workers, patience,
    close_mosaic, cache, val, plots, ...) can be changed, so the rest is passed
    only on a fresh start.

    `weights` starts from an existing checkpoint instead of the COCO weights —
    the §7.2 option (C) mosaic stage trains a run, lets it early-stop, then
    continues from its own `best.pt` with `train_overrides={"mosaic": 0.0}`.
    That two-stage shape exists because Ultralytics closes mosaic at the fixed
    epoch `epochs - close_mosaic` (90 here), which an early-stopping run at ~33
    epochs never reaches: without it, every model would be trained entirely on
    mosaicked frames and evaluated on clean ones — a mismatch that lands directly
    on σ, which is fitted to the spread of whatever distribution it is shown.
    """
    from ultralytics import YOLO

    from uqfusion.bench.grid import resolve_device

    b = cfg["benchmark"]
    g = {**DEFAULT_GAUSSIAN_CFG, **(cfg.get("gaussian") or {}), **(gaussian_overrides or {})}
    out_root = Path(cfg["paths"]["outputs_root"]) / out_subdir
    name = run_name or f"{'gauss' if sigma else 'parity'}_{variant}_seed{seed}"
    run_dir = out_root / name

    last_ckpt = run_dir / "weights" / "last.pt"
    resuming = bool(resume) and last_ckpt.is_file()

    extra = dict(train_overrides or {})
    # Split the overrides: on resume Ultralytics only honours a short allow-list,
    # and silently ignores the rest (they come from the checkpoint's own args).
    resume_safe = {k: extra.pop(k) for k in
                   ("patience", "close_mosaic", "cache", "val", "plots", "save_period")
                   if k in extra}

    shared = dict(
        imgsz=imgsz if imgsz is not None else b["imgsz"],
        batch=batch if batch is not None else b["batch"],
        workers=workers if workers is not None else b["workers"],
        patience=b["patience"],
        device=resolve_device(cfg),
    )
    shared.update(resume_safe)  # per-run overrides win over the config defaults

    if resuming:
        LOGGER.info(f"[gaussian] === {name}: RESUME from {last_ckpt}")
        model = YOLO(str(last_ckpt))
    elif weights is not None:
        if not Path(weights).is_file():
            raise FileNotFoundError(f"start weights not found: {weights}")
        LOGGER.info(f"[gaussian] === {name}: starting from {weights}")
        model = YOLO(str(weights))
    elif is_variant(variant):
        # An architecture edit with no released checkpoint. Built from its yaml and
        # seeded from the nearest base model through a declared index remap, because
        # inserting a neck level renumbers every layer after it and the stock
        # name-matched load would drop the whole bottom-up PAN (see uq/variants.py).
        model, transfer_pct = build_variant(variant)
        LOGGER.info(f"[gaussian] === {name}: variant {variant}, {transfer_pct:.1f}% seeded")
    else:
        model = YOLO(f"{variant}.pt" if b.get("pretrained", True) else f"{variant}.yaml")

    for event, fns in (callbacks or {}).items():
        for fn in fns:
            model.add_callback(event, fn)
    model.add_callback("on_train_start", reseed_at_train_start(seed, b["deterministic"]))
    model.add_callback("on_train_start", restore_early_stopping)

    # sigma=False -> stock DetectionTrainer; every other argument is identical.
    trainer_kw = {"trainer": make_gaussian_trainer(g)} if sigma else {}
    if resuming:
        model.train(resume=True, **trainer_kw, **shared)
    else:
        # Built as one dict and merged in precedence order rather than spread as
        # keywords: `**shared, **extra` alongside explicit keywords raises
        # "got multiple values for keyword argument" the moment an override names
        # something already set here — which is exactly what a per-run override is
        # for. Merging lets the caller override anything, silently and correctly.
        kwargs = dict(
            data=str(data_yaml),
            epochs=epochs if epochs is not None else b["epochs"],
            seed=seed,
            deterministic=b["deterministic"],
            optimizer=b.get("optimizer", "auto"),
            amp=b["amp"],
            project=str(out_root),
            name=name,
            exist_ok=True,
            verbose=True,
        )
        kwargs.update(shared)
        kwargs.update(extra)  # per-run overrides win over everything above
        model.train(**trainer_kw, **kwargs)
    best = run_dir / "weights" / "best.pt"
    return (best if best.is_file() else run_dir / "weights" / "last.pt"), run_dir
