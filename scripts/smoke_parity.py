"""§12.1 gate: the σ branch must not perturb the detector at all.

D17 promises the deterministic detector trains bit-identically to its baseline.
The end2end gate proves that at the *gradient* level on fixed inputs. This proves
it at the *training-loop* level, which is a strictly stronger claim and the one
§12.1 actually rests on: two runs, identical in every respect except `sigma`,
must produce identical per-epoch losses while the NLL weight is zero.

It exists because that claim was false for a subtle reason. Initialising `cv4`
drew from the global RNG, so every subsequent draw — dataloader seeding,
augmentation — was offset, and the two arms saw different augmentations from
batch 1. Nothing leaked through the gradients; the trajectories simply diverged.
The measured cost on the first real pair was 0.011 mAP50-95 between the arms,
2.4x the Phase 1 seed sd, which would have read as "the σ branch costs accuracy".

Usage:  python scripts/smoke_parity.py
"""

from __future__ import annotations

import csv
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.config import load_config  # noqa: E402
from uqfusion.uq.train_gaussian import train_gaussian  # noqa: E402

SMOKE_ROOT = ROOT / "runs" / "smoke_parity"
EPOCHS = 3
COLS = ("train/box_loss", "train/cls_loss", "train/dfl_loss",
        "metrics/mAP50-95(B)", "metrics/mAP50(B)")


def fail(msg: str) -> None:
    print(f"[parity-smoke] FAIL {msg}")
    raise SystemExit(1)


def main() -> int:
    data_yaml = ROOT / "runs" / "tune" / "data_probe.yaml"
    if not data_yaml.is_file():
        fail(f"no probe dataset at {data_yaml} — run scripts/tune_batch.py once first")

    cfg = load_config()
    smoke = cfg["smoke"]
    shutil.rmtree(SMOKE_ROOT, ignore_errors=True)
    cfg = {**cfg, "paths": {**cfg["paths"], "outputs_root": str(SMOKE_ROOT)}}

    # Warm-up covers every epoch: the NLL weight stays 0 throughout, so the
    # detector has no legitimate reason to differ by even one ulp.
    for sigma in (True, False):
        name = "gauss" if sigma else "parity"
        print(f"[parity-smoke] training {name} ({EPOCHS} epochs, NLL weight 0 throughout) ...")
        train_gaussian(
            cfg, data_yaml, variant="yolo26n", seed=0, epochs=EPOCHS,
            imgsz=smoke["imgsz"], batch=smoke["batch"], workers=0,
            run_name=name, sigma=sigma, resume=False, out_subdir="runs",
            gaussian_overrides={"warmup_epochs": EPOCHS, "ramp_epochs": 0},
        )

    def rows(n):
        return list(csv.DictReader(open(SMOKE_ROOT / "runs" / n / "results.csv", encoding="utf-8")))

    G, P = rows("gauss"), rows("parity")
    if len(G) != len(P):
        fail(f"different epoch counts: gauss {len(G)}, parity {len(P)}")

    worst, diffs = 0.0, []
    for i, (a, b) in enumerate(zip(G, P), start=1):
        for c in COLS:
            va, vb = (a.get(c) or "").strip(), (b.get(c) or "").strip()
            if not va or not vb:
                continue
            d = abs(float(va) - float(vb))
            worst = max(worst, d)
            if d > 0:
                diffs.append(f"ep{i} {c}: {va} vs {vb} (Δ{d:.3e})")

    nll = [(r.get("train/nll_loss") or "").strip() for r in G]
    if any(v not in ("0", "0.0", "0.00000") for v in nll):
        fail(f"NLL was not held at zero during warm-up: {nll}")

    if diffs:
        for d in diffs[:8]:
            print(f"[parity-smoke]   {d}")
        fail(f"detector differs between the σ and parity arms (max |Δ| {worst:.3e}); "
             "something other than the NLL is reaching the detector")

    print(f"[parity-smoke] {len(G)} epochs x {len(COLS)} metrics identical between "
          f"σ and parity arms, max |Δ| {worst:.3e} OK")
    print(f"[parity-smoke] NLL held at 0 across the warm-up OK")
    print("PARITY SMOKE OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
