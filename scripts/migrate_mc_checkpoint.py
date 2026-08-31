"""Migrate a pre-2026-08-31 MC-Dropout checkpoint to the MCDropoutConv2d placement.

**Why this exists.** The old `insert_head_dropout` rebuilt each deployed head branch
as `nn.Sequential(*children[:-1], nn.Dropout2d(p), children[-1])`, which renumbered
the final conv (`one2one_cv2.0.2` -> `.0.3`). Every training-time reload
(`DetectionTrainer.get_model` -> `BaseModel.load` -> `intersect_dicts`) matches by
key name, so those 12 output-conv tensors were silently dropped -- the defect that
cost `mc_vis_seed0_ft` and `mc_ir_seed0_ft`.

Fixing `insert_head_dropout` is not enough on its own: the *parent* checkpoints on
disk still carry the shifted keys, so fine-tuning them would drop the same 12
tensors again. This script rewrites the stored module tree to the plain layout
(dropout folded into the final conv's class) so the keys match a stock model.

It is a structural rewrite, not a retrain. Every parameter tensor is carried across
by identity, and the script refuses to write unless it has verified:

  1. the migrated state_dict's keys match a stock model of the same yaml exactly;
  2. every parameter is bit-identical to the one it replaced;
  3. the migrated model, dropout disarmed, produces bit-identical output to the
     original on a fixed input -- i.e. the deterministic detection row that the
     parent's own results.csv reports is unchanged;
  4. armed, it is stochastic.

Usage:
    python scripts/migrate_mc_checkpoint.py runs/mc_dropout/mc_ir_seed0/weights/best.pt
    python scripts/migrate_mc_checkpoint.py <in.pt> --out <out.pt>   # default: <in>_mcfix.pt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from uqfusion.uq.mc_dropout import (  # noqa: E402
    MCDropoutConv2d,
    _detect_head,
    deployed_head_branches,
    enable_mc_dropout,
    insert_head_dropout,
)

IMGSZ = 256


def _forward(model, x):
    with torch.no_grad():
        y = model(x)
    if isinstance(y, dict):
        y = y.get("one2one", next(iter(y.values())))
    while isinstance(y, (list, tuple)):
        y = y[0]
    return y


def strip_inserted_dropout(model) -> int:
    """Rebuild each deployed branch without the inserted Dropout2d. Returns layers removed."""
    head = _detect_head(model)
    removed = 0
    for branch_name in deployed_head_branches(head):
        branch = getattr(head, branch_name, None)
        if branch is None:
            continue
        for i, seq in enumerate(branch):
            children = list(seq.children())
            kept = [c for c in children if not isinstance(c, nn.Dropout2d)]
            removed += len(children) - len(kept)
            if len(kept) != len(children):
                # Same module objects, same order, just without the dropout entries.
                branch[i] = nn.Sequential(*kept)
    return removed


def migrate(module, p: float):
    """Old-structure module -> new placement. Returns (module, n_removed)."""
    n = strip_inserted_dropout(module)
    insert_head_dropout(module, p)
    return module, n


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", type=Path)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("-p", type=float, default=0.15, help="dropout p (must match training)")
    args = ap.parse_args()

    out = args.out or args.ckpt.with_name(args.ckpt.stem + "_mcfix.pt")
    print(f"--- migrating {args.ckpt}")

    from ultralytics import YOLO
    from ultralytics.nn.tasks import DetectionModel

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    if not isinstance(ck, dict) or "model" not in ck:
        print("!! not an Ultralytics checkpoint dict", file=sys.stderr)
        return 1

    reference = YOLO(str(args.ckpt)).model.float().eval()
    x = torch.randn(1, 3, IMGSZ, IMGSZ)
    torch.manual_seed(0)
    before = _forward(reference, x)

    total_removed = 0
    dtypes: dict[str, torch.dtype] = {}
    for key in ("model", "ema"):
        mod = ck.get(key)
        if not isinstance(mod, nn.Module):
            continue
        # Ultralytics stores the model in fp16; keep whatever it was. Verification
        # below needs fp32, so record the original dtype and restore it before saving.
        dtypes[key] = next(mod.parameters()).dtype
        pre = {k: v.clone() for k, v in mod.float().state_dict().items()}
        mod, removed = migrate(mod.float(), args.p)
        total_removed += removed
        post = mod.state_dict()

        # 2. every parameter carried across bit-identically (modulo the index shift)
        moved = 0
        for k, v in pre.items():
            cand = post.get(k)
            if cand is None:  # shifted key: .N -> .N-1
                head_, _, tail = k.rpartition(".")
                idx = head_.rpartition(".")
                cand = post.get(f"{idx[0]}.{int(idx[2]) - 1}.{tail}")
                moved += 1
            assert cand is not None, f"{key}: tensor {k} lost in migration"
            assert torch.equal(v, cand), f"{key}: tensor {k} CHANGED VALUE in migration"
        print(f"  [{key}] removed {removed} Dropout2d, re-indexed {moved} tensors, "
              f"{len(pre)} verified bit-identical")
        ck[key] = mod

    if total_removed == 0:
        print("  nothing to migrate — this checkpoint is already on the new placement")
        return 0

    # 1. keys now match a stock model built from the same yaml
    migrated = ck["model"].float()
    stock = DetectionModel(migrated.yaml, nc=migrated.yaml["nc"], ch=3, verbose=False)
    mk, sk = set(migrated.state_dict()), set(stock.state_dict())
    assert mk == sk, f"key mismatch after migration: +{sorted(mk - sk)[:4]} -{sorted(sk - mk)[:4]}"
    print(f"  keys match a stock {migrated.yaml.get('yaml_file', 'model')}: {len(mk)}/{len(sk)} ✓")

    # 3. disarmed output unchanged  4. armed output stochastic
    check = migrated.eval()
    for m in check.modules():
        if isinstance(m, MCDropoutConv2d):
            m.mc_armed = False
    torch.manual_seed(0)
    after = _forward(check, x)
    delta = (before - after).abs().max().item()
    assert delta == 0.0, f"disarmed output CHANGED (max |Δ| {delta:.3e}) — migration is not faithful"
    print(f"  disarmed output bit-identical to the original (max |Δ| {delta:.1e}) ✓")

    n = enable_mc_dropout(check)
    torch.manual_seed(1)
    a = _forward(check, x)
    torch.manual_seed(2)
    b = _forward(check, x)
    d = (a - b).abs().max().item()
    assert d > 0, f"{n} layers armed but two passes are identical — no epistemic variance"
    print(f"  armed: {n} layers, two passes differ by max |Δ| {d:.2f} ✓")

    for key, dt in dtypes.items():  # restore the storage dtype the checkpoint arrived in
        ck[key] = ck[key].to(dt)
    torch.save(ck, out)
    print(f"--- wrote {out} ({out.stat().st_size / 1048576:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
