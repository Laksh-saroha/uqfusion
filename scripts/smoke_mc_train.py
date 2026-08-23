"""10-epoch MC-Dropout training smoke on REAL VIS and IR data — D-13 step (a).

`smoke_mc_e2e.py` proves the plumbing on an untrained model in seconds. It cannot
prove the thing that actually matters: that a checkpoint produced by the *training
loop* yields non-zero epistemic variance through `MCDropoutPredictor`. That needs
a real trained model, which is what this produces.

What it does, per modality:
  1. carves a small subset out of the production train/val lists (real images,
     real labels, production model + batch — only the image count and epoch count
     are reduced);
  2. trains 10 epochs through `train_mc_dropout`, i.e. the exact code path the
     full runs use;
  3. loads the resulting `best.pt` through `MCDropoutPredictor` and asserts T
     passes actually disagree — the end-to-end proof the module gate cannot give.

**What this does NOT prove.** The 2026-08-23 divergence appeared at epoch 18 of a
100-epoch run on the full set. Ten epochs on a subset cannot reproduce it. This
smoke checks *plumbing and variance*, not long-run stability; the first real
evidence on stability is epoch ~20 of the full re-run.

Usage:  python scripts/smoke_mc_train.py [--modalities vis,ir] [--train N] [--val N]
        [--epochs N] [config.yaml]

Must be run with the CUDA interpreter (`gpu_python` in config.yaml), not .venv.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))  # repo is not pip-installed in the GPU interpreter

from uqfusion.config import load_config  # noqa: E402
from uqfusion.uq.mc_dropout import MCDropoutPredictor, train_mc_dropout  # noqa: E402

MODALITIES = {
    "vis": {"yaml": "runs/derived/data_vis_stride2.yaml", "variant": "yolo26m", "batch": 16},
    "ir": {"yaml": "runs/derived/data_ir_shiponly.yaml", "variant": "yolo26m-p2feat", "batch": 10},
}


def _read_list(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def make_subset(src_yaml: Path, out_dir: Path, n_train: int, n_val: int, seed: int = 0) -> Path:
    """Evenly-strided subset of the production lists. Strided, not head-sliced:
    these lists are ordered by sequence, so the first N images are one stretch of
    one voyage in one lighting condition.

    Entries are written as ABSOLUTE paths. The production lists are inconsistent —
    the VIS train list is absolute but `visible/val.txt` is `./images/...` relative
    — and Ultralytics resolves relative entries against the list file's own parent,
    so copying them verbatim into a new directory makes every image "corrupt".
    """
    spec = yaml.safe_load(src_yaml.read_text())
    out_dir.mkdir(parents=True, exist_ok=True)
    out = {k: v for k, v in spec.items() if k not in ("train", "val", "test")}

    for split, n in (("train", n_train), ("val", n_val)):
        src_list = Path(spec[split])
        items = _read_list(src_list)
        items = [str((src_list.parent / it).resolve()) if not Path(it).is_absolute() else it
                 for it in items]
        if len(items) > n:
            step = len(items) / n
            items = [items[int(i * step)] for i in range(n)]
        missing = [i for i in items[:20] if not Path(i).is_file()]
        assert not missing, f"{split}: resolved paths do not exist, e.g. {missing[0]}"
        dst = out_dir / f"{split}.txt"
        dst.write_text("\n".join(items) + "\n")
        out[split] = str(dst)
        print(f"    {split}: {len(items)} images -> {dst.name}")

    dst_yaml = out_dir / src_yaml.name
    dst_yaml.write_text(yaml.safe_dump(out, sort_keys=False))
    return dst_yaml


def read_curve(run_dir: Path) -> list[dict]:
    import csv

    rows = list(csv.DictReader(open(run_dir / "results.csv")))
    out = []
    for r in rows:
        g = {k.strip(): v for k, v in r.items()}
        out.append({
            "epoch": int(float(g["epoch"])),
            "mAP50": float(g["metrics/mAP50(B)"]),
            "mAP": float(g["metrics/mAP50-95(B)"]),
            "vcls": float(g["val/cls_loss"]),
            "tcls": float(g["train/cls_loss"]),
        })
    return out


def check_variance(best: Path, data_yaml: Path, cfg: dict, n_images: int = 8, T: int = 10) -> dict:
    """The payoff: T passes on a real trained checkpoint must disagree."""
    spec = yaml.safe_load(Path(data_yaml).read_text())
    imgs = _read_list(Path(spec["val"]))[:n_images]
    device = "cuda" if torch.cuda.is_available() else "cpu"
    pred = MCDropoutPredictor(best, T=T, device=device, imgsz=cfg["benchmark"]["imgsz"], conf=0.25)
    print(f"    armed {pred.n_armed} dropout layers, T={T}, device={device}")
    assert pred.n_armed > 0, "MCDropoutPredictor armed no dropout layers"

    n_det, spreads = 0, []
    for im in imgs:
        rec = pred(im)
        nb = rec["boxes_xyxy"].shape[0]
        if nb == 0:
            continue
        n_det += nb
        for key in ("sigma_ltrb", "score_std", "n_support"):
            if key in rec:
                v = np.asarray(rec[key], dtype=float).ravel()
                if v.size:
                    spreads.append((key, float(np.nanmean(v)), float(np.nanmax(v))))
                break

    assert n_det > 0, f"no detections on {len(imgs)} val images — cannot judge variance"
    means = [m for _, m, _ in spreads]
    maxes = [x for _, _, x in spreads]
    key = spreads[0][0] if spreads else "?"
    mean_spread = float(np.mean(means)) if means else 0.0
    assert mean_spread > 0.0, (
        f"MC spread ({key}) is exactly 0 across {n_det} detections — T passes are identical, "
        "so this checkpoint carries no epistemic uncertainty. This is the 2026-08-23 defect "
        "(handoff §1.5); do NOT start the full runs."
    )
    print(f"    {n_det} detections, MC spread via '{key}': mean {mean_spread:.4f}, "
          f"max {max(maxes):.4f} ✓ NON-ZERO")
    return {"n_det": n_det, "spread_key": key, "spread_mean": mean_spread, "spread_max": max(maxes)}


def run_modality(mod: str, cfg: dict, args) -> dict:
    m = MODALITIES[mod]
    print(f"\n=== {mod.upper()} ===")
    root = Path(cfg["paths"]["outputs_root"]) / "smoke_mc_train" / mod
    data_yaml = make_subset(Path(m["yaml"]), root / "data", args.train, args.val)

    best, run_dir = train_mc_dropout(
        cfg, data_yaml, variant=m["variant"], seed=0, epochs=args.epochs,
        imgsz=cfg["benchmark"]["imgsz"], batch=m["batch"], workers=args.workers,
        run_name=f"smoke_mc_{mod}",
    )
    print(f"    trained -> {best}")

    curve = read_curve(Path(run_dir))
    zeros = [r["epoch"] for r in curve if r["mAP50"] == 0.0]
    last = curve[-1]
    print(f"    epochs {len(curve)}, final mAP50 {last['mAP50']:.4f} mAP50-95 {last['mAP']:.4f}, "
          f"vcls {curve[0]['vcls']:.2f} -> {last['vcls']:.2f}")
    assert not zeros, f"mAP50 collapsed to exactly 0 at epochs {zeros} — the D-12 alarm condition"
    assert last["mAP50"] > 0.05, f"final mAP50 {last['mAP50']:.4f} is implausibly low for {mod}"

    var = check_variance(Path(best), data_yaml, cfg)
    return {"modality": mod, "run_dir": str(run_dir), "best": str(best),
            "epochs": len(curve), "final_mAP50": last["mAP50"], "final_mAP": last["mAP"],
            "vcls_first": curve[0]["vcls"], "vcls_last": last["vcls"], **var}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modalities", default="vis,ir")
    ap.add_argument("--train", type=int, default=2000)
    ap.add_argument("--val", type=int, default=400)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("config", nargs="?", default=None)
    args = ap.parse_args()

    random.seed(0)
    cfg = load_config(args.config)
    print(f"CUDA: {torch.cuda.is_available()} "
          f"({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    assert torch.cuda.is_available(), "no CUDA — run with config.yaml's gpu_python, not .venv"

    results = [run_modality(m.strip(), cfg, args) for m in args.modalities.split(",")]

    out = Path(cfg["paths"]["outputs_root"]) / "smoke_mc_train" / "summary.json"
    out.write_text(json.dumps(results, indent=2))
    print("\n=== SUMMARY ===")
    for r in results:
        print(f"  {r['modality']:4s} {r['epochs']:2d}ep  mAP50 {r['final_mAP50']:.4f}  "
              f"mAP50-95 {r['final_mAP']:.4f}  MC spread {r['spread_mean']:.4f} "
              f"({r['spread_key']}, {r['n_det']} dets)")
    print(f"\nwrote {out}")
    print("[mc-smoke] BOTH MODALITIES PASSED ✓")
    return 0


if __name__ == "__main__":
    sys.exit(main())
