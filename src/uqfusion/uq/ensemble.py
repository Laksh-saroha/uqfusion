"""Deep Ensemble baseline (O2 comparison — scope §9.2, R4; plan B6-7).

Seed accounting per plan B6-7: the M members ARE the seed replicates — one
M-member ensemble per modality (not 3 ensembles x 5 members). Members are
STANDARD detectors trained with the shared config at seeds `baselines.ensemble.seeds`;
training therefore reuses the Phase 1 grid runner (one variant, M seeds), which
also yields each member's deterministic row for free. It passes `runs_root` so the
members land in `outputs_root/ensemble/<name>/` rather than the grid's default
`outputs_root/benchmark/runs/` — mixing them into the Phase 1 benchmark tree made
"collect the matrix" and "collect the Phase 1 grid" the same glob.

Inference: one pass per member, merged by the shared clustering protocol.
"""

from __future__ import annotations

from pathlib import Path

from uqfusion.uq.clustering import cluster_records
from uqfusion.uq.infer import PlainPredictor


def train_ensemble(cfg: dict, data_yaml, variant: str, seeds=None, epochs=None, imgsz=None,
                   batch=None, workers=None, out_csv=None):
    """Train the M members via the Phase 1 grid (resume-safe). Returns member weight paths."""
    from uqfusion.bench.grid import run_grid

    e = (cfg.get("baselines") or {}).get("ensemble", {})
    seeds = seeds if seeds is not None else e.get("seeds", [0, 1, 2, 3, 4])
    out_root = Path(cfg["paths"]["outputs_root"]) / "ensemble"
    csv_path = run_grid(
        cfg, data_yaml, variants=[variant], seeds=list(seeds), epochs=epochs,
        imgsz=imgsz, batch=batch, workers=workers,
        out_csv=out_csv or out_root / "ensemble_members.csv", run_prefix="ens",
        runs_root=out_root,
    )
    weights = []
    for seed in seeds:
        run_dir = out_root / f"ens_{variant}_seed{seed}"
        best = run_dir / "weights" / "best.pt"
        weights.append(best if best.is_file() else run_dir / "weights" / "last.pt")
    return weights, csv_path


def train_ensemble_member(cfg: dict, data_yaml, variant: str, seed: int, epochs=None,
                          imgsz=None, batch=None, workers=None, run_name: str | None = None,
                          weights=None, callbacks=None, train_overrides=None, out_csv=None):
    """Train exactly one ensemble member. Returns (best_weights, run_dir) like
    `train_gaussian`/`train_mc_dropout`, so a queue runner can dispatch to all three
    training kinds uniformly (scripts/run_queue.py's full-scale matrix).

    A distinct `run_name` (and `out_csv`) is required whenever this is called more than
    once for the same (variant, seed) — e.g. a run and its fine-tune continuation — since
    `run_grid`'s CSV-completion check and run directory are both keyed by name.
    """
    from uqfusion.bench.grid import run_grid

    name = run_name or f"ens_{variant}_seed{seed}"
    out_root = Path(cfg["paths"]["outputs_root"]) / "ensemble"
    run_grid(
        cfg, data_yaml, variants=[variant], seeds=[seed], epochs=epochs, imgsz=imgsz,
        batch=batch, workers=workers, run_prefix="ens", run_name=name,
        weights=weights, callbacks=callbacks, train_overrides=train_overrides,
        out_csv=out_csv or out_root / "csv" / f"{name}.csv",
        runs_root=out_root,
    )
    run_dir = out_root / name
    best = run_dir / "weights" / "best.pt"
    return (best if best.is_file() else run_dir / "weights" / "last.pt"), run_dir


class EnsemblePredictor:
    """One pass per member -> shared clustering -> standard record (+n_support)."""

    def __init__(self, member_weights: list, device="cpu", imgsz=640, conf=0.25, iou=0.7):
        if len(member_weights) < 2:
            raise ValueError("an ensemble needs at least 2 members")
        self.members = [
            PlainPredictor(w, device=device, imgsz=imgsz, conf=conf, iou=iou, want_features=(i == 0))
            for i, w in enumerate(member_weights)
        ]

    def __call__(self, image) -> dict:
        # only member 0 captures features (frame-level OOD needs one consistent
        # feature space, not an average over differently-initialized networks);
        # cluster_records uses exactly the sources that captured one.
        return cluster_records([m(image) for m in self.members])
