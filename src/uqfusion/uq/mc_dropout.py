"""MC-Dropout baseline (O2 comparison — scope §9.2, R5; plan B6-3).

Ultralytics detect heads have no dropout, so MC-Dropout is NOT free on YOLO:
dropout must be inserted and the model RETRAINED — a different deterministic
model whose own detection row must be reported (its standard training
results.csv provides exactly that, since validation runs with dropout off).

Pre-fixed protocol (plan B6-3, decided before any real run):
  placement = one Dropout2d before the final 1x1 conv of each head branch
  (box cv2 and cls cv3, every level); p and T from config
  (`baselines.mc_dropout`, defaults p=0.15, T=10).

Inference: T stochastic passes (dropout re-enabled in eval), merged by the
shared clustering protocol (uq/clustering.py) into the standard record schema.
"""

from __future__ import annotations

from pathlib import Path

import torch
import torch.nn as nn
from ultralytics.models.yolo.detect import DetectionTrainer
from ultralytics.utils import DEFAULT_CFG

from uqfusion.uq.clustering import cluster_records
from uqfusion.uq.infer import PlainPredictor


def insert_head_dropout(model, p: float = 0.15):
    """Insert Dropout2d before the last conv of each cv2/cv3 branch. Idempotent."""
    head = model.model[-1]
    for branch_name in ("cv2", "cv3"):
        branch = getattr(head, branch_name, None)
        if branch is None:
            continue
        for i, seq in enumerate(branch):
            children = list(seq.children())
            if any(isinstance(c, nn.Dropout2d) for c in children):
                continue  # already inserted (resume path)
            branch[i] = nn.Sequential(*children[:-1], nn.Dropout2d(p), children[-1])
    return model


def enable_mc_dropout(model) -> int:
    """Re-enable dropout layers after model.eval(); returns how many were armed."""
    n = 0
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()
            n += 1
    if n == 0:
        raise RuntimeError("no dropout layers found — was this model trained via MCDropoutTrainer?")
    return n


class MCDropoutTrainer(DetectionTrainer):
    """Standard detection training on the dropout-augmented head (3 loss items)."""

    dropout_p: float = 0.15

    def get_model(self, cfg=None, weights=None, verbose=True):
        model = super().get_model(cfg, weights, verbose)
        return insert_head_dropout(model, self.dropout_p)


def make_mc_trainer(p: float) -> type[MCDropoutTrainer]:
    return type("ConfiguredMCDropoutTrainer", (MCDropoutTrainer,), {"dropout_p": float(p)})


def train_mc_dropout(
    cfg: dict,
    data_yaml: str | Path,
    variant: str,
    seed: int = 0,
    epochs: int | None = None,
    imgsz: int | None = None,
    batch: int | None = None,
    workers: int | None = None,
    run_name: str | None = None,
):
    """Train the MC-Dropout model with the shared config. Returns (best_weights, run_dir)."""
    from ultralytics import YOLO

    from uqfusion.bench.grid import resolve_device

    b = cfg["benchmark"]
    p = float((cfg.get("baselines") or {}).get("mc_dropout", {}).get("p", 0.15))
    out_root = Path(cfg["paths"]["outputs_root"]) / "mc_dropout"
    name = run_name or f"mc_{variant}_seed{seed}"

    model = YOLO(f"{variant}.pt" if b.get("pretrained", True) else f"{variant}.yaml")
    model.train(
        trainer=make_mc_trainer(p),
        data=str(data_yaml),
        epochs=epochs if epochs is not None else b["epochs"],
        imgsz=imgsz if imgsz is not None else b["imgsz"],
        batch=batch if batch is not None else b["batch"],
        seed=seed, deterministic=b["deterministic"], optimizer=b.get("optimizer", "auto"),
        patience=b["patience"], amp=b["amp"],
        workers=workers if workers is not None else b["workers"],
        device=resolve_device(cfg), project=str(out_root), name=name, exist_ok=True, verbose=True,
    )
    run_dir = out_root / name
    best = run_dir / "weights" / "best.pt"
    return (best if best.is_file() else run_dir / "weights" / "last.pt"), run_dir


class MCDropoutPredictor:
    """T stochastic passes -> shared clustering -> standard record (+n_support)."""

    def __init__(self, weights, T: int = 10, device="cpu", imgsz=640, conf=0.25, iou=0.7, seed: int = 0):
        self.base = PlainPredictor(weights, device=device, imgsz=imgsz, conf=conf, iou=iou, want_features=True)
        self.T = int(T)
        self.seed = int(seed)
        self.n_armed = enable_mc_dropout(self.base.model)

    def __call__(self, image) -> dict:
        torch.manual_seed(self.seed)  # reproducible pass set per image
        return cluster_records([self.base(image) for _ in range(self.T)])
