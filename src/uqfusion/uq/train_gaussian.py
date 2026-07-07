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
from ultralytics.utils import DEFAULT_CFG

from uqfusion.uq.gaussian import DEFAULT_GAUSSIAN_CFG, convert_to_gaussian


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
        model = super().get_model(cfg, weights, verbose)
        return convert_to_gaussian(model, self.gaussian_cfg)

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
):
    """Train one Gaussian-head model with the shared project config. Returns (best_weights_path, run_dir)."""
    from ultralytics import YOLO

    from uqfusion.bench.grid import resolve_device

    b = cfg["benchmark"]
    g = {**DEFAULT_GAUSSIAN_CFG, **(cfg.get("gaussian") or {}), **(gaussian_overrides or {})}
    out_root = Path(cfg["paths"]["outputs_root"]) / "gaussian"
    name = run_name or f"gauss_{variant}_seed{seed}"

    model = YOLO(f"{variant}.pt" if b.get("pretrained", True) else f"{variant}.yaml")
    model.train(
        trainer=make_gaussian_trainer(g),
        data=str(data_yaml),
        epochs=epochs if epochs is not None else b["epochs"],
        imgsz=imgsz if imgsz is not None else b["imgsz"],
        batch=batch if batch is not None else b["batch"],
        seed=seed,
        deterministic=b["deterministic"],
        optimizer=b.get("optimizer", "auto"),
        patience=b["patience"],
        amp=b["amp"],
        workers=workers if workers is not None else b["workers"],
        device=resolve_device(cfg),
        project=str(out_root),
        name=name,
        exist_ok=True,
        verbose=True,
    )
    run_dir = out_root / name
    best = run_dir / "weights" / "best.pt"
    return (best if best.is_file() else run_dir / "weights" / "last.pt"), run_dir
