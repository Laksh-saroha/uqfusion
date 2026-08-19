"""§18-2 gate for pause/resume of a Gaussian-head run.

Pause and resume are only worth having if a resumed run is the same experiment as
an uninterrupted one. Two ways that silently fails, both checked here:

  1. **The σ branch resets.** `DetectionTrainer.get_model` builds a plain detector
     and then calls `BaseModel.load`, which intersects the checkpoint state dict
     against the model *as it exists at that moment*. Convert after loading and
     every `cv4` key falls out of the intersection: the detector resumes from
     epoch N while σ restarts from its zero-bias init. The resulting σ is finite,
     positive and non-degenerate — it passes every aggregate check while being
     wrong. Check 1 reproduces the old order to show the failure is real, then
     asserts the shipped order keeps the trained weights bit-identical.

  2. **Early stopping forgets.** Ultralytics rebuilds `EarlyStopping` in
     `_setup_train` and `resume_training` never restores it, so patience counts
     from a local peak after every resume. A paused run would then train past the
     stopping rule that the un-paused runs obeyed. Check 2 asserts the restore
     callback recovers best fitness and best epoch from the run's own results.csv.

Usage:  python scripts/smoke_resume.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

import torch  # noqa: E402

from uqfusion.config import load_config  # noqa: E402
from uqfusion.uq.gaussian import convert_to_gaussian  # noqa: E402
from uqfusion.uq.train_gaussian import restore_early_stopping, train_gaussian  # noqa: E402

SMOKE_ROOT = ROOT / "runs" / "smoke_resume"
RUN_NAME = "resume_probe"


def cv4_fingerprint(model) -> torch.Tensor:
    """Flat vector of every σ-branch parameter, in a stable order."""
    head = model.model[-1]
    parts = [p.detach().float().flatten() for name, p in head.cv4.named_parameters()]
    return torch.cat(parts)


def fail(msg: str) -> None:
    print(f"[resume-smoke] FAIL {msg}")
    raise SystemExit(1)


def main() -> int:
    cfg = load_config()
    from ultralytics.nn.tasks import DetectionModel

    smoke = cfg["smoke"]
    variant = "yolo26n"  # end2end path — the selected backbone's head family
    epochs = 3

    shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
    cfg = {**cfg, "paths": {**cfg["paths"], "outputs_root": str(SMOKE_ROOT)}}
    # warm-up/ramp off so σ actually moves inside a 3-epoch probe; a run whose σ
    # never left its init could not distinguish "loaded" from "re-initialised".
    overrides = {"warmup_epochs": 0, "ramp_epochs": 0}

    data_yaml = ROOT / "runs" / "tune" / "data_probe.yaml"
    if not data_yaml.is_file():
        fail(f"no probe dataset at {data_yaml} — run scripts/tune_batch.py once first")

    # Stop the way the queue's Pause button stops: raise out of `on_model_save`,
    # the first instant at which last.pt is complete on disk. A run allowed to
    # finish instead would be useless here — `final_eval` strips the optimizer and
    # the EMA out of last.pt, so a finished run has nothing to resume from.
    stop_after = 2

    class _Paused(Exception):
        pass

    def pause_at(trainer):
        if int(trainer.epoch) + 1 >= stop_after:
            raise _Paused(f"paused after epoch {int(trainer.epoch) + 1}")

    print(f"[resume-smoke] training {variant}, pausing after epoch {stop_after} ...")
    run_dir = SMOKE_ROOT / "gaussian" / RUN_NAME
    try:
        train_gaussian(
            cfg, data_yaml, variant=variant, seed=0, epochs=epochs,
            imgsz=smoke["imgsz"], batch=smoke["batch"], workers=0,
            run_name=RUN_NAME, gaussian_overrides=overrides, resume=False,
            out_subdir="gaussian", callbacks={"on_model_save": [pause_at]},
        )
    except _Paused as exc:
        print(f"[resume-smoke] {exc} — checkpoint boundary reached OK")
    else:
        fail("the pause callback never fired; the run trained to completion")

    last = run_dir / "weights" / "last.pt"
    if not last.is_file():
        fail(f"no checkpoint at {last}")

    # --- Check 1: the σ branch survives the rebuild-and-load that resume performs
    ckpt = torch.load(last, map_location="cpu", weights_only=False)
    if ckpt.get("ema") is None:
        fail("last.pt carries no EMA — it was stripped, so it cannot be resumed")
    if int(ckpt.get("epoch", -1)) < 0:
        fail(f"last.pt reports epoch {ckpt.get('epoch')} — stripped, not resumable")
    trained = ckpt["ema"].float()
    trained_fp = cv4_fingerprint(trained)
    if float(trained_fp.abs().sum()) == 0.0:
        fail("σ branch is all zeros after training — the probe proves nothing")

    yaml_cfg = trained.yaml

    # the shipped order: build -> convert -> load
    good = DetectionModel(yaml_cfg, nc=2, ch=3, verbose=False)
    good = convert_to_gaussian(good, {})
    good.load(trained)
    good_delta = float((cv4_fingerprint(good) - trained_fp).abs().max())

    # the old order: build -> load -> convert (reproduced to show the hazard is real)
    bad = DetectionModel(yaml_cfg, nc=2, ch=3, verbose=False)
    bad.load(trained)
    bad = convert_to_gaussian(bad, {})
    bad_delta = float((cv4_fingerprint(bad) - trained_fp).abs().max())

    if good_delta != 0.0:
        fail(f"σ weights changed across a convert-then-load rebuild (max |Δ| {good_delta:.3e})")
    if bad_delta == 0.0:
        fail("load-then-convert also preserved σ — the check can no longer detect the bug")
    print(f"[resume-smoke] σ survives resume rebuild: max |Δ| {good_delta:.2e} "
          f"(load-then-convert would have reset it, max |Δ| {bad_delta:.3f}) OK")

    # --- Check 2: early stopping is restored from the run's own history
    class _Stopper:
        best_fitness, best_epoch, patience = 0.0, 0, 20

    class _Trainer:
        save_dir, stopper, best_fitness = run_dir, _Stopper(), None

    restore_early_stopping(_Trainer())
    if _Trainer.stopper.best_fitness <= 0.0:
        fail(f"early stopping not restored from {run_dir / 'results.csv'}")
    print(f"[resume-smoke] early stopping restored: fitness "
          f"{_Trainer.stopper.best_fitness:.5f} at epoch {_Trainer.stopper.best_epoch} OK")

    # --- Check 3: the real resume path picks the run back up and finishes it
    print("[resume-smoke] resuming to completion ...")
    _, run_dir2 = train_gaussian(
        cfg, data_yaml, variant=variant, seed=0, epochs=epochs,
        imgsz=smoke["imgsz"], batch=smoke["batch"], workers=0,
        run_name=RUN_NAME, gaussian_overrides=overrides, resume=True,
        out_subdir="gaussian",
    )
    rows = (run_dir2 / "results.csv").read_text(encoding="utf-8").strip().splitlines()
    if len(rows) - 1 != epochs:
        fail(f"resume ended at {len(rows) - 1} epoch rows, expected {epochs}")
    print(f"[resume-smoke] resume advanced {stop_after} -> {len(rows) - 1} epochs, "
          f"σ carried through OK")

    print("RESUME SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
