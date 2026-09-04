"""Tiny-subset gate for the small-object architecture variants — MUST pass before GPU time.

Mirrors `smoke_gaussian_e2e.py` in intent: prove the plumbing on something cheap so a
long screen cannot fail six hours in. Four gates per variant:

  A. weight transfer clears the floor declared in `uq/variants.py`
  B. conversion yields the right number of σ levels at the PINNED width (a P2 backbone
     silently halves c4 unless `sigma_width` holds it, which would make the arm a test
     of two changes at once)
  C. two epochs on a 48-frame slice of the REAL IR tree train to a finite NLL and write
     best.pt with four loss columns
  D. the saved checkpoint reloads with its σ branch intact at the variant's level count
     — the failure `smoke_resume.py` exists for, re-checked because inserting a neck
     level changes every key name in the state dict

Usage:  python scripts/smoke_variants.py [--variants yolo26s-p2 yolo26s-p2feat] [--device 0]
"""

from __future__ import annotations

import argparse
import csv
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.config import load_config  # noqa: E402
from uqfusion.uq.gaussian import GaussianDetect, convert_to_gaussian  # noqa: E402
from uqfusion.uq.variants import VARIANT_SPECS, build_variant  # noqa: E402

SIGMA_WIDTH = 32  # yolo26s' native c4; the screen pins every arm here


def tiny_ir_yaml(out_dir: Path, n_train: int = 48, n_val: int = 24) -> Path:
    """A data yaml over a handful of real IR frames (real tiny boxes, real letterbox)."""
    src = ROOT / "Pohang_dataset" / "infrared"
    out_dir.mkdir(parents=True, exist_ok=True)
    lists = {}
    for split, n in (("train", n_train), ("val", n_val)):
        rows = [l.strip() for l in (src / f"{split}.txt").read_text().splitlines() if l.strip()]
        if len(rows) < n:
            raise RuntimeError(f"{split}.txt has only {len(rows)} rows")
        step = max(1, len(rows) // n)
        # The tree's own lists are relative to the tree ("./images/..."), and a yaml
        # with no `path:` key resolves them against the YAML's directory instead —
        # pointing the loader at runs/smoke_variants/images. Absolutise here, matching
        # what runs/derived/*.txt already store.
        picked = [str((src / r).resolve()) for r in rows[::step][:n]]
        p = out_dir / f"tiny_{split}.txt"
        p.write_text("\n".join(picked) + "\n", encoding="utf-8")
        lists[split] = p
    yaml_path = out_dir / "data_ir_tiny.yaml"
    yaml_path.write_text(
        f"train: {lists['train']}\nval: {lists['val']}\ntest: {lists['val']}\n"
        "names:\n  0: ship\n  1: buoy\n", encoding="utf-8")
    return yaml_path


def gate_a_b(variant: str) -> tuple[int, int]:
    spec = VARIANT_SPECS[variant]
    model, pct = build_variant(variant, verbose=False)
    assert pct >= spec["min_transfer"], f"{variant}: transfer {pct:.1f}% < {spec['min_transfer']}%"
    print(f"[smoke] {variant} A: {pct:.1f}% seeded from {spec['base']} "
          f"(floor {spec['min_transfer']:.0f}%) OK")

    dm = model.model
    nl_before = dm.model[-1].nl
    convert_to_gaussian(dm, {"sigma_width": SIGMA_WIDTH})
    head = dm.model[-1]
    assert isinstance(head, GaussianDetect), "class swap did not happen"
    assert len(head.cv4) == nl_before == head.nl, \
        f"{variant}: cv4 has {len(head.cv4)} levels for a {head.nl}-level head"
    c4 = head.cv4[0][0].conv.out_channels
    assert c4 == SIGMA_WIDTH, f"{variant}: sigma_width pin failed, c4={c4} not {SIGMA_WIDTH}"
    for seq in head.cv4:
        assert float(seq[-1].bias.abs().max()) == 0.0, "logvar bias must start at zero"
    print(f"[smoke] {variant} B: {head.nl} sigma levels, c4={c4} pinned, "
          f"strides {head.stride.tolist()} OK")
    return head.nl, c4


def gate_c_d(cfg: dict, variant: str, data_yaml: Path, device: str, nl: int) -> None:
    from uqfusion.uq.train_gaussian import train_gaussian

    run_name = f"smoke_{variant}"
    run_dir = ROOT / "runs" / "smoke_variants" / run_name
    shutil.rmtree(run_dir, ignore_errors=True)

    best, run_dir = train_gaussian(
        cfg, data_yaml, variant=variant, seed=0, epochs=2, imgsz=640, batch=4, workers=2,
        run_name=run_name, out_subdir="smoke_variants", resume=False,
        gaussian_overrides={"sigma_width": SIGMA_WIDTH, "warmup_epochs": 0, "ramp_epochs": 1},
        train_overrides={"patience": 100, "plots": False, "val": True},
    )

    rows = list(csv.DictReader(open(Path(run_dir) / "results.csv", encoding="utf-8")))
    assert rows, f"{variant}: no results.csv rows"
    nll_col = next((c for c in rows[0] if "nll" in c.lower()), None)
    assert nll_col, f"{variant}: no nll loss column — loss_names not widened. Got {list(rows[0])}"
    nlls = [float(r[nll_col]) for r in rows]
    assert all(math.isfinite(v) for v in nlls), f"{variant}: non-finite NLL {nlls}"
    assert Path(best).is_file(), f"{variant}: no best.pt at {best}"
    print(f"[smoke] {variant} C: {len(rows)} epochs, NLL {nlls} finite, best.pt written OK")

    import torch
    ck = torch.load(best, map_location="cpu", weights_only=False)
    head = ck["model"].model[-1]
    assert isinstance(head, GaussianDetect), f"{variant}: reloaded head is {type(head).__name__}"
    assert len(head.cv4) == nl, f"{variant}: reloaded cv4 has {len(head.cv4)} levels, want {nl}"
    live = sum(1 for seq in head.cv4 if float(seq[-1].bias.abs().max()) > 0)
    print(f"[smoke] {variant} D: checkpoint reloads with {len(head.cv4)} sigma levels, "
          f"{live}/{nl} cv4 biases moved off zero init OK")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--variants", nargs="+", default=list(VARIANT_SPECS))
    ap.add_argument("--device", default=None)
    args = ap.parse_args()

    cfg = load_config(None)
    if args.device is not None:
        cfg["device"] = args.device
    data_yaml = tiny_ir_yaml(ROOT / "runs" / "smoke_variants")
    print(f"[smoke] tiny IR subset: {data_yaml}")

    for v in args.variants:
        nl, _ = gate_a_b(v)
        gate_c_d(cfg, v, data_yaml, cfg["device"], nl)
    print("\nVARIANT SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
