"""Gate-level ablation sweep over cached predictions (Phase 3; plan B5-3/B5-4).

Consumes two index-aligned caches (VIS stream, IR stream), a clean cache to
fit constants + the Mahalanobis scorer, and sweeps combination rule x α —
pure CPU, no model re-run.

Usage:
    python scripts/ablate_gate.py \
        --vis-cache runs/cache/gauss_vis_val_fog.pkl \
        --ir-cache  runs/cache/gauss_ir_val_clean.pkl \
        --clean-cache runs/cache/gauss_vis_val_clean.pkl \
        --fit-cache runs/cache/gauss_vis_train_clean.pkl \
        --out runs/eval/gate_ablation.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.fusion_eval import ablate_gate_rules, format_ablation_table
from uqfusion.uq.mahalanobis import MahalanobisScorer
from uqfusion.uq.reliability import fit_constants, per_box_uncertainty


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--vis-cache", required=True)
    parser.add_argument("--ir-cache", required=True)
    parser.add_argument("--clean-cache", required=True, help="clean val cache for constants fitting")
    parser.add_argument("--fit-cache", required=True, help="clean train cache for the Mahalanobis fit")
    parser.add_argument("--out", default=None)
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_md = Path(args.out) if args.out else Path(cfg["paths"]["outputs_root"]) / "eval" / "gate_ablation.md"

    vis_records, _ = load_cache(args.vis_cache)
    ir_records, _ = load_cache(args.ir_cache)
    clean_records, _ = load_cache(args.clean_cache)
    fit_records, _ = load_cache(args.fit_cache)

    scorer = MahalanobisScorer().fit(np.stack([r["feat"] for r in fit_records]))
    clean_u = np.concatenate([
        per_box_uncertainty(r["sigma_ltrb"], r["boxes_xyxy"]) for r in clean_records if len(r["conf"])
    ])
    clean_d = np.asarray([scorer.score(r["feat"]) for r in clean_records])
    constants = fit_constants(clean_u, clean_d)
    print(f"[ablate] constants: λ={constants.lam:.3f}, μ_d={constants.mu_d:.2f}, τ={constants.tau:.2f}")

    rows = ablate_gate_rules(vis_records, ir_records, scorer, constants)
    table = format_ablation_table(rows)
    print(table)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(table + "\n", encoding="utf-8")
    print(f"[ablate] written -> {out_md}")
    return 0


if __name__ == "__main__":
    try:  # Windows cp1252 consoles: degrade non-ASCII output instead of crashing
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
