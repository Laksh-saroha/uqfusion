"""§18-2 gate for the END2END (YOLO26) Gaussian σ² head — MUST pass before GPU time.

`smoke_gaussian.py` covers the plain-Detect/DFL path. It cannot cover the three
end2end-specific hazards, because each of them produces σ that is finite, positive
and non-degenerate — i.e. that passes every aggregate check while being wrong:

  A. σ attached to the DISCARDED branch. `Detect.forward` runs the head twice;
     inference decodes from one2one. σ on one2many is trained under a different
     target assignment (topk 10 vs 7/topk2 1) and describes a box predictor that
     never reaches the output. Aggregate σ statistics look identical.
  B. σ LOST IN THE TOP-K GATHER. Stock `Detect.postprocess` splits `[4, nc]`
     exactly, so trailing σ columns are read as class logits. Checked here by
     index-tagging every anchor and asserting σ[i] still belongs to box[i].
  C. NLL SCALED BY THE BRANCH SCHEDULE. `E2ELoss` weights one2one 0.2 → 0.9 over a
     run. A short smoke never sees the decay, so a schedule bug is invisible to it;
     asserted instead as detector parity — box/cls/dfl must be unchanged by the σ
     branch, which is D17's guarantee and is violated if the NLL joins that vector.

Also asserts the §7.2 consequence in executable form: at reg_max=1 there is no DFL
distribution, so `dfl_sigma_ltrb` MUST be absent (see docs/architecture-option-a.md).

Usage:  python scripts/smoke_gaussian_e2e.py [config.yaml]
"""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import numpy as np
import torch
from ultralytics.utils.tal import bbox2dist

from uqfusion.config import load_config
from uqfusion.uq.gaussian import (
    GaussianDetect,
    GaussianE2ELoss,
    convert_to_gaussian,
)
from uqfusion.uq.infer import UQPredictor

E2E_VARIANT = "yolo26n"  # smallest end2end variant — the gate exercises plumbing, not quality


def _build(variant: str, convert: bool):
    """Load a checkpoint and give it trainer-shaped hyperparameters.

    A `.pt` carries `model.args` as a plain dict; the criterion needs the namespace the
    trainer would have installed (`hyp.box`, `hyp.cls`, `hyp.dfl`, `hyp.epochs`).
    """
    from ultralytics import YOLO
    from ultralytics.cfg import get_cfg
    from ultralytics.utils import DEFAULT_CFG

    model = YOLO(f"{variant}.pt").model
    model.args = get_cfg(DEFAULT_CFG)
    if convert:
        convert_to_gaussian(model, {})
    return model


def check_conversion(variant: str):
    """A: conversion succeeds and cv4 shadows the branch that survives to inference."""
    model = _build(variant, convert=True)
    head = model.model[-1]
    assert isinstance(head, GaussianDetect), f"head is {type(head).__name__}, not GaussianDetect"
    assert head.end2end, f"{variant} is not an end2end head — use smoke_gaussian.py instead"
    assert head.reg_max == 1, f"expected reg_max=1 on {variant}, got {head.reg_max}"
    assert getattr(head, "cv4", None) is not None, "cv4 branch missing after conversion"
    assert head._sigma_box_head() is head.one2one_cv2, "cv4 shadows the wrong (discarded) branch"
    n = sum(p.numel() for p in head.cv4.parameters()) / 1e6
    print(f"[e2e-smoke] converted {variant}: end2end, reg_max=1, cv4 {n:.2f}M params, shadows one2one ✓")
    return model


def check_sigma_on_deployed_branch(model):
    """A: `logvars` must appear on the one2one branch and NOT on the discarded one."""
    model.train()
    with torch.no_grad():
        preds = model(torch.rand(2, 3, 320, 320))
    assert isinstance(preds, dict) and "one2one" in preds, f"unexpected train-mode output: {type(preds)}"
    assert "logvars" in preds["one2one"], "σ is NOT on the one2one branch — it would never reach inference"
    assert "logvars" not in preds["one2many"], "σ leaked onto the discarded one2many branch"
    lv = preds["one2one"]["logvars"]
    assert lv.shape[1] == 4 and torch.isfinite(lv).all(), f"bad logvars tensor {tuple(lv.shape)}"
    print(f"[e2e-smoke] logvars on one2one only, shape {tuple(lv.shape)}, finite ✓")


def check_gather_alignment(head):
    """B: index-tag every anchor and prove σ[i] still describes box[i] after top-k.

    Boxes encode their anchor index; σ encodes the same index offset by 1000. If
    `postprocess` gathers σ with a different index — or lets σ fall into `scores` —
    the decoded indices disagree and this fails exactly, regardless of training.
    """
    nc, n_anchors, k = head.nc, 60, 10
    torch.manual_seed(0)
    idx = torch.arange(n_anchors, dtype=torch.float32).view(1, n_anchors, 1)
    boxes = idx.repeat(1, 1, 4)
    scores = torch.rand(1, n_anchors, nc)
    sigma = (idx + 1000.0).repeat(1, 1, 4)
    preds = torch.cat([boxes, scores, sigma], dim=-1)

    prev_max_det = head.max_det
    head.max_det = k
    try:
        out = head.postprocess(preds)
    finally:
        head.max_det = prev_max_det

    assert out.shape[-1] == 6 + 4, f"expected 10 output columns (4 box + conf + cls + 4 σ), got {out.shape[-1]}"
    box_src = out[0, :, 0]
    sig_src = out[0, :, 6] - 1000.0
    assert torch.equal(box_src, sig_src), (
        f"σ is misaligned with its box after top-k — box anchors {box_src.tolist()} "
        f"vs σ anchors {sig_src.tolist()}"
    )
    # all four σ columns must carry the same anchor (no channel shear)
    assert torch.equal(out[0, :, 6:10] - 1000.0, box_src.view(-1, 1).repeat(1, 4)), "σ channels sheared"
    print(f"[e2e-smoke] top-k gather keeps σ with its box for all {out.shape[1]} rows ✓")


def check_unclamped_target():
    """C-adjacent: the reg_max=1 NLL target must NOT go through the DFL clamp.

    Executable proof of why `gaussian_nll` branches on `use_dfl`: at reg_max=1 the DFL
    form clamps to (0, reg_max-1-0.01) = (0, -0.01). torch's clamp returns `max` when
    min > max, so every target collapses to the constant -0.01 — zero variance and a
    negative distance, i.e. no learnable signal at all.
    """
    anchors = torch.tensor([[10.0, 10.0], [20.0, 20.0]])
    boxes = torch.tensor([[4.0, 4.0, 16.0, 16.0], [12.0, 12.0, 28.0, 28.0]])
    ok = bbox2dist(anchors, boxes)                 # what the non-DFL path uses
    bug = bbox2dist(anchors, boxes, 1 - 1)         # what the DFL path would do at reg_max=1
    assert ok.max() > 1.0 and ok.std() > 1e-3, f"unclamped target is degenerate: {ok.tolist()}"
    assert bug.std() < 1e-6 and bug.max() <= 0.0, (
        f"expected the DFL clamp to collapse targets to a constant, got {bug.tolist()}"
    )
    print(f"[e2e-smoke] NLL target unclamped (max {ok.max():.1f} stride units, sd {ok.std():.1f}); "
          f"DFL clamp would have collapsed it to {bug.flatten()[0]:.2f} ✓")


def check_detector_parity(variant: str):
    """C: the σ branch must not perturb box/cls/dfl — D17's parity guarantee.

    Same weights, same batch, stock criterion vs Gaussian criterion, NLL fully active.
    If the NLL were folded into the one2one vector (and thus into the o2m/o2o
    schedule), or if cv4 leaked into the trunk, the first three items would move.
    """
    from ultralytics.utils.loss import E2ELoss

    torch.manual_seed(0)
    img = torch.rand(2, 3, 320, 320)
    batch = {
        "img": img,
        "batch_idx": torch.tensor([0.0, 1.0]),
        "cls": torch.tensor([[0.0], [1.0]]),
        "bboxes": torch.tensor([[0.5, 0.5, 0.3, 0.3], [0.4, 0.6, 0.2, 0.25]]),
    }

    stock = _build(variant, convert=False)
    gauss = copy.deepcopy(stock)
    convert_to_gaussian(gauss, {})
    stock.train()
    gauss.train()

    crit_stock = E2ELoss(stock)
    crit_gauss = GaussianE2ELoss(gauss)
    assert isinstance(crit_gauss, E2ELoss), "Gaussian criterion must remain an E2ELoss"
    crit_gauss.epoch = 999  # past warm-up + ramp: NLL at full weight

    with torch.no_grad():
        loss_s, items_s = crit_stock(stock(img), batch)
        loss_g, items_g = crit_gauss(gauss(img), batch)

    assert items_s.numel() == 3 and items_g.numel() == 4, (
        f"expected 3 stock items and 4 Gaussian items, got {items_s.numel()} and {items_g.numel()}"
    )
    diff = (loss_g[:3] - loss_s).abs().max().item()
    assert diff < 1e-5, (
        f"σ branch perturbed the detector loss by {diff:.3e} — parity broken "
        f"(stock {loss_s.tolist()} vs gaussian {loss_g[:3].tolist()})"
    )
    nll = float(items_g[3])
    assert np.isfinite(nll) and nll != 0.0, f"NLL inactive at epoch 999: {nll}"
    print(f"[e2e-smoke] detector parity holds (max |Δ| {diff:.2e}); NLL active and finite ({nll:.4f}) ✓")

    crit_gauss.epoch = 0  # warm-up: σ frozen
    with torch.no_grad():
        _, items_warm = crit_gauss(gauss(img), batch)
    assert float(items_warm[3]) == 0.0, f"warm-up should log nll=0, got {float(items_warm[3])}"
    print("[e2e-smoke] warm-up epoch logs nll=0 — σ-freeze engages ✓")


def check_inference(best: Path, data_root: Path, imgsz: int):
    """End-to-end: σ survives the no-NMS path, and the §7.2 DFL row is absent."""
    predictor = UQPredictor(best, device="cpu", imgsz=imgsz, conf=0.15)
    assert predictor.end2end, "UQPredictor did not detect the end2end head"
    sig, n_det, saw_dfl = [], 0, False
    for img in sorted((data_root / "images" / "val").glob("*.jpg")):
        rec = predictor(img)
        nb = rec["boxes_xyxy"].shape[0]
        if nb == 0:
            continue
        assert rec["sigma_ltrb"].shape == (nb, 4), (
            f"σ shape {rec['sigma_ltrb'].shape} does not match {nb} boxes — gather misaligned"
        )
        saw_dfl = saw_dfl or ("dfl_sigma_ltrb" in rec)
        sig.extend(rec["sigma_ltrb"].ravel().tolist())
        n_det += nb

    s = np.asarray(sig)
    assert s.size >= 8, f"too few detections ({n_det}) to judge σ"
    assert np.isfinite(s).all() and (s > 0).all(), "σ contains non-finite or non-positive values"
    cv = float(s.std() / max(s.mean(), 1e-9))
    assert cv > 0.01, f"σ is degenerate (CV {cv:.4f})"
    assert not saw_dfl, "dfl_sigma_ltrb present at reg_max=1 — §7.2 option (a) cannot exist here"
    print(f"[e2e-smoke] inference: {n_det} detections, σ mean {s.mean():.2f}px CV {cv:.3f}, "
          f"no DFL row (§7.2 (a) unavailable, as designed) ✓")


def main() -> int:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    sys.path.insert(0, str(Path(__file__).parent))
    from smoke_gaussian import SMOKE_GAUSSIAN_OVERRIDES, check_loss_csv, train_smoke_gaussian

    print("--- structural gates (no training) ---")
    model = check_conversion(E2E_VARIANT)
    check_sigma_on_deployed_branch(model)
    check_gather_alignment(model.model[-1])
    check_unclamped_target()
    check_detector_parity(E2E_VARIANT)

    print("\n--- training gate ---")
    best, run_dir, data_root, _ = train_smoke_gaussian(
        cfg, fresh=True, variant=E2E_VARIANT,
        run_name="smoke_gauss_e2e", root_name="smoke_gaussian_e2e",
    )
    assert Path(best).is_file(), f"no weights produced at {best}"
    check_loss_csv(run_dir, SMOKE_GAUSSIAN_OVERRIDES["warmup_epochs"])
    check_inference(Path(best), data_root, cfg["smoke"]["imgsz"])

    print("\nEND2END GAUSSIAN SMOKE OK — σ rides one2one, survives top-k, detector parity intact")
    return 0


if __name__ == "__main__":
    try:  # Windows cp1252 consoles: degrade non-ASCII output instead of crashing
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
