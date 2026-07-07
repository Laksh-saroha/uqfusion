"""Compute the pre-registered calibration metrics from prediction caches
(Phase 3; plan B6-5/B6-6) — Table 2 rows, one per cache.

Each --cache entry: label=path[:sigma_key]. sigma_key defaults to sigma_ltrb;
use dfl_sigma_ltrb for the §7.2 DFL-derived ablation row on a Gaussian cache.

Usage:
    python scripts/evaluate_uq.py \
        --cache gaussian_clean=runs/cache/gauss_vis_val_clean.pkl \
        --cache gaussian_dfl=runs/cache/gauss_vis_val_clean.pkl:dfl_sigma_ltrb \
        --cache mc_fog=runs/cache/mc_vis_val_fog.pkl \
        --out runs/eval/table2.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from uqfusion.config import load_config
from uqfusion.eval.cache import load_cache
from uqfusion.eval.metrics import summarize_cache

COLUMNS = ["d_ece", "nll", "interval_ece", "ause", "aurc", "map50_95", "map50", "n_detections", "n_frames"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None)
    parser.add_argument("--cache", action="append", required=True, help="label=path[:sigma_key]")
    parser.add_argument("--out", default=None, help="markdown output (default runs/eval/table2.md)")
    args = parser.parse_args()

    cfg = load_config(args.config)
    out_md = Path(args.out) if args.out else Path(cfg["paths"]["outputs_root"]) / "eval" / "table2.md"

    rows = {}
    for entry in args.cache:
        label, spec = entry.split("=", 1)
        path, _, sigma_key = spec.partition(":")
        summary = summarize_cache(load_cache(path)[0], sigma_key=sigma_key or "sigma_ltrb")
        rows[label] = summary
        print(f"[eval] {label}: " + ", ".join(
            f"{k}={summary[k]:.4f}" if isinstance(summary.get(k), float) else f"{k}={summary.get(k)}"
            for k in COLUMNS if k in summary
        ))

    lines = ["| UQ source / condition | " + " | ".join(COLUMNS) + " |",
             "|" + "---|" * (len(COLUMNS) + 1)]
    for label, s in rows.items():
        cells = [f"{s[k]:.4f}" if isinstance(s.get(k), float) else str(s.get(k, "—")) for k in COLUMNS]
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out_md.with_suffix(".json").write_text(json.dumps(rows, indent=1, default=float), encoding="utf-8")
    print(f"[eval] written -> {out_md} (+ .json)")
    return 0


if __name__ == "__main__":
    try:  # Windows cp1252 consoles: degrade non-ASCII output instead of crashing
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
