"""Audit the train/val/test split for leakage before any training (decision D6-rev).

Usage (on the machine that has the data):
    python scripts/audit_split.py                 # audits vis + ir
    python scripts/audit_split.py --data vis
    python scripts/audit_split.py --data path/to/some_data.yaml --min-gap-frames 150

Exit code 0 = PASS (split usable), 1 = FAIL (fix the split first).
Writes runs/audit/<yaml-stem>_audit.md alongside the console report.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.data.audit import audit_split, format_report
from uqfusion.data.lists import load_data_yaml


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=None, help="config.yaml path (default: repo root)")
    parser.add_argument("--data", nargs="*", default=["vis", "ir"],
                        help="dataset yamls to audit: vis, ir, or explicit paths")
    parser.add_argument("--min-gap-frames", type=int, default=None,
                        help="override data_audit.min_gap_frames from config")
    args = parser.parse_args()

    cfg = load_config(args.config)
    min_gap = args.min_gap_frames or cfg.get("data_audit", {}).get("min_gap_frames", 100)
    out_dir = Path(cfg["paths"]["outputs_root"]) / "audit"

    all_ok = True
    for data in args.data:
        yaml_path = resolve_data_yaml(cfg, data)
        report = audit_split(load_data_yaml(yaml_path), min_gap_frames=min_gap)
        text = format_report(report)
        print("\n" + text + "\n")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / f"{Path(yaml_path).stem}_audit.md"
        out_file.write_text(text + "\n", encoding="utf-8")
        print(f"[audit] report written -> {out_file}")
        all_ok &= report["ok"]
        if report["missing_splits"]:
            all_ok = False  # a two-way split is not usable (see dataset_requirement.md §3)

    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
