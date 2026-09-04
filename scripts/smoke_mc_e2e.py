"""§18-2 gate for the END2END (YOLO26) MC-Dropout head — MUST pass before GPU time.

The counterpart to `smoke_gaussian_e2e.py`, which did not exist and whose absence
cost two full VIS runs (~30 h) on 2026-08-23. See handoff §1.5.

**The bug this gate exists to catch.** `insert_head_dropout` originally inserted
`Dropout2d` into `cv2`/`cv3`. YOLO26's `Detect` is end-to-end: it runs the head
twice and inference decodes from `preds["one2one"]`, discarding the one2many
branch. So all six dropout layers trained but were never executed at inference —
T stochastic passes came out identical, epistemic variance was exactly zero, and
`enable_mc_dropout` counted 6 layers and raised nothing. Every aggregate check
passed while the arm produced no uncertainty at all.

The lesson encoded here: **a layer that exists, and even a layer that is invoked,
is not a layer that does something.** Three checks, escalating:

  A. placement  — dropout on the deployed branch and nowhere else (structural)
  B. execution  — each layer is invoked, is in training mode, AND changes its
                  input (a call counter alone proved insufficient: an eval-mode
                  Dropout2d is invoked and is the identity)
  C. stochastic — two armed passes of the whole model produce different boxes,
                  which is the only property MC-Dropout actually needs

Three separate attempts to arm dropout during the 2026-08-23 investigation failed
silently; two would have been reported as "dropout makes no difference" had their
controls been omitted.

**Do not switch this to a yaml-built model to avoid the download.** A random-init
YOLO26n has head activations of order 1e-6, so zeroing whole channels moves the
decoded boxes by less than float32 resolution and check C passes vacuously — the
gate would go green on genuinely broken code. Pretrained weights are load-bearing
(measured: activations O(1), armed passes differ by ~271px; random init: 0.0).

Runs on CPU in a few seconds — no GPU, no dataset. `nc` is COCO's 80 rather than
the VIS head's 2; this gate tests plumbing, not quality, same as its sibling.

Usage:  python scripts/smoke_mc_e2e.py
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn
from ultralytics.nn.tasks import DetectionModel

from uqfusion.uq import mc_dropout as mcd
from uqfusion.uq.mc_dropout import (
    MCDropoutConv2d,
    deployed_head_branches,
    enable_mc_dropout,
    insert_head_dropout,
)

E2E_VARIANT = "yolo26n"  # smallest end2end variant; matches smoke_gaussian_e2e.py
P = 0.15
IMGSZ = 256


def _build(p: float = P):
    """Pretrained yolo26n with dropout inserted. Pretrained is required — see module docstring."""
    from ultralytics import YOLO

    model = YOLO(f"{E2E_VARIANT}.pt").model
    insert_head_dropout(model, p)
    return model


def _forward(model, x):
    """One forward pass, reduced to a comparable tensor."""
    with torch.no_grad():
        y = model(x)
    if isinstance(y, dict):
        y = y.get("one2one", next(iter(y.values())))
    while isinstance(y, (list, tuple)):
        y = y[0]
    return y


def _count_dropout(module) -> int:
    """Armed head convs — the current placement (see MCDropoutConv2d)."""
    return sum(1 for m in module.modules() if isinstance(m, MCDropoutConv2d))


def _count_legacy_dropout(module) -> int:
    """Inserted `nn.Dropout2d` — the pre-2026-08-31 placement, kept for check E."""
    return sum(1 for m in module.modules() if isinstance(m, nn.Dropout2d))


def check_placement():
    """A: dropout must sit on the branch that survives to inference, and only there."""
    model = _build()
    head = model.model[-1]
    assert head.end2end, f"{E2E_VARIANT} is not an end2end head — this gate targets YOLO26"
    assert deployed_head_branches(head) == ("one2one_cv2", "one2one_cv3"), (
        f"deployed_head_branches returned {deployed_head_branches(head)} for an end2end head"
    )

    on_deployed = _count_dropout(head.one2one_cv2) + _count_dropout(head.one2one_cv3)
    on_discarded = _count_dropout(head.cv2) + _count_dropout(head.cv3)
    expected = 2 * head.nl  # one per level, per branch

    assert on_deployed == expected, (
        f"expected {expected} MCDropoutConv2d on one2one_cv2/one2one_cv3, found {on_deployed}. "
        "Dropout on the discarded one2many branch is the 2026-08-23 bug: it trains but "
        "never runs at inference, so MC-Dropout yields zero variance (handoff §1.5)."
    )
    assert on_discarded == 0, (
        f"{on_discarded} MCDropoutConv2d found on the DISCARDED cv2/cv3 branch. Those layers "
        "perturb the shared trunk's gradients while contributing nothing at inference — "
        "the mechanism that diverged mc_vis_seed0 and mc_vis_seed1 at epoch 18."
    )
    assert _count_legacy_dropout(model) == 0, (
        "inserted nn.Dropout2d found. Insertion renumbers the branch's final conv and "
        "breaks every training-time reload — the 2026-08-31 defect. See MCDropoutConv2d."
    )
    print(f"[mc-e2e] placement: {on_deployed} MCDropoutConv2d on one2one (deployed), "
          f"{on_discarded} on one2many (discarded), 0 inserted layers ✓")
    return model


def check_executes_at_inference(model):
    """B: every layer is invoked, armed, and actually perturbs its input.

    The invocation counter is kept because it is the historical lesson, but it is
    NOT sufficient on its own — an eval-mode Dropout2d is invoked and returns its
    input unchanged. The per-layer training flag and in/out delta are what make
    this check real.
    """
    head = model.model[-1]
    x = torch.randn(1, 3, IMGSZ, IMGSZ)
    calls = {"n": 0}
    seen: list[tuple[str, bool, float, float]] = []
    orig = mcd.F.dropout2d

    def counting(inp, p=0.5, training=True, inplace=False):
        calls["n"] += 1
        out = orig(inp, p, training, inplace)
        seen.append(("dropout2d", bool(training), float(inp.abs().mean()),
                     float((inp - out).abs().max())))
        return out

    model.eval()
    enable_mc_dropout(model)
    mcd.F.dropout2d = counting
    try:
        _forward(model, x)
    finally:
        mcd.F.dropout2d = orig

    expected = 2 * head.nl
    assert calls["n"] == expected, (
        f"dropout2d fired {calls['n']} times, expected {expected}. Zero means the "
        "layers are not on the executed path — the failure that produced 11 epochs of "
        "mAP 0.0000 with no error (handoff §1.5)."
    )
    assert len(seen) == expected, f"hooked {len(seen)} layers, expected {expected}"

    cold = [nm for nm, training, _, _ in seen if not training]
    assert not cold, (
        f"invoked but NOT in training mode, so acting as the identity: {cold}. "
        "enable_mc_dropout did not set mc_armed, or something reset it afterwards."
    )
    inert = [(nm, d) for nm, _, _, d in seen if d == 0.0]
    assert not inert, (
        f"armed and invoked but left the input unchanged: {inert}. With p={P} across these "
        "channel counts a zero delta means dropout is not masking anything."
    )
    worst = min(d for _, _, _, d in seen)
    print(f"[mc-e2e] execution: {calls['n']}/{expected} invoked, all in training mode, "
          f"all perturbing input (smallest max|Δ| {worst:.3f}) ✓")


def check_stochastic(model):
    """C: armed, repeated passes must DIFFER — that is the whole point of MC-Dropout."""
    x = torch.randn(1, 3, IMGSZ, IMGSZ)
    model.eval()
    n = enable_mc_dropout(model)
    torch.manual_seed(1)
    a = _forward(model, x)
    torch.manual_seed(2)
    b = _forward(model, x)
    delta = (a - b).abs().max().item()
    scale = a.abs().mean().item()
    assert scale > 1e-3, (
        f"model output is degenerate (mean |out| {scale:.3e}) — the stochasticity check "
        "cannot mean anything here. Are these pretrained weights? See module docstring."
    )
    assert delta > 1e-3 * scale, (
        f"two armed passes are effectively identical (max |Δ| {delta:.3e} against output "
        f"scale {scale:.3f}) — {n} layers reported armed but epistemic variance is zero"
    )
    print(f"[mc-e2e] stochasticity: {n} layers armed, two passes differ by "
          f"max |Δ| {delta:.2f} (output scale {scale:.2f}) ✓")


def check_deterministic_when_disarmed():
    """D: WITHOUT arming, eval must be bit-identical — the deterministic row must stay so."""
    model = _build()
    model.eval()
    x = torch.randn(1, 3, IMGSZ, IMGSZ)
    a = _forward(model, x)
    b = _forward(model, x)
    delta = (a - b).abs().max().item()
    assert delta == 0.0, (
        f"eval is stochastic without arming (max |Δ| {delta:.3e}) — dropout is leaking into "
        "the deterministic pass, so this model's results.csv row would not be reproducible"
    )
    print("[mc-e2e] disarmed eval is bit-identical — the deterministic row stays deterministic ✓")


def check_guard_rejects_the_historical_bug():
    """E: `enable_mc_dropout` must reject dropout that is not on the executed path.

    Its old count-based guard passed happily while all 6 layers were dead. This
    reconstructs that exact model and asserts the guard now refuses it. Built from
    yaml because no forward pass is needed — only the guard's decision.
    """
    model = DetectionModel(f"{E2E_VARIANT}.yaml", nc=2, ch=3, verbose=False)
    head = model.model[-1]
    for branch_name in ("cv2", "cv3"):  # deliberately the DISCARDED branch
        branch = getattr(head, branch_name)
        for i, seq in enumerate(branch):
            children = list(seq.children())
            branch[i] = nn.Sequential(*children[:-1], nn.Dropout2d(P), children[-1])
    assert _count_legacy_dropout(model) == 2 * head.nl, "test fixture did not insert dropout"

    try:
        n = enable_mc_dropout(model)
    except RuntimeError as exc:
        assert "one2one" in str(exc), f"guard raised, but not about branch placement: {exc}"
        print("[mc-e2e] guard rejects one2many-only placement with a branch-specific error ✓")
        return
    raise AssertionError(
        f"enable_mc_dropout accepted a model whose {n} dropout layers are all on the "
        "discarded branch. That is the 2026-08-23 bug and the guard must refuse it."
    )


def check_survives_a_training_reload():
    """F: an MC checkpoint must reload through the TRAINING path with nothing dropped.

    The gate above this line tests a model built in memory. That is exactly what let
    the 2026-08-31 defect through: inserting `nn.Dropout2d` into the branch renumbered
    each final conv (`one2one_cv2.0.2` -> `.0.3`), and `DetectionTrainer.get_model`
    loads a checkpoint into a freshly-built plain model **by key name**, intersecting
    silently. Twelve tensors — the box and class output convs at all three scales —
    were dropped and arrived randomly initialised, which cost `mc_vis_seed0_ft`
    (0.24667 -> 0.15123) and `mc_ir_seed0_ft` (0.12973 -> 0.11894).

    The lesson the module docstring encodes needs one more clause: a model that
    trains is not a model that reloads.
    """
    from ultralytics import YOLO

    mc_sd = _build().state_dict()
    plain_sd = YOLO(f"{E2E_VARIANT}.pt").model.state_dict()

    missing = [k for k in mc_sd if k not in plain_sd]
    mismatched = [k for k in mc_sd if k in plain_sd and mc_sd[k].shape != plain_sd[k].shape]
    lost = sorted(set(missing) | set(mismatched))
    assert not lost, (
        f"{len(lost)} of {len(mc_sd)} tensors would be SILENTLY DROPPED reloading this "
        f"checkpoint through the training path: {lost[:6]}{' ...' if len(lost) > 6 else ''}. "
        "MC-Dropout must not change the module tree — see MCDropoutConv2d."
    )
    extra = [k for k in plain_sd if k not in mc_sd]
    assert not extra, f"{len(extra)} plain tensors absent from the MC checkpoint: {extra[:6]}"
    print(f"[mc-e2e] training reload: {len(mc_sd)}/{len(plain_sd)} tensors transfer, "
          "0 dropped — checkpoint is interchangeable with a plain one ✓")


def main() -> int:
    print(f"--- MC-Dropout end2end gate ({E2E_VARIANT}, p={P}, CPU) ---")
    model = check_placement()
    check_executes_at_inference(model)
    check_stochastic(model)
    check_deterministic_when_disarmed()
    check_guard_rejects_the_historical_bug()
    check_survives_a_training_reload()
    print("\n[mc-e2e] ALL GATES PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
