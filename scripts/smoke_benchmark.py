"""CPU smoke test for the whole Phase 1 harness — no real data needed.

Fabricates a tiny synthetic dataset, then runs grid -> val -> CSV -> FPS ->
Table 1 end-to-end at toy scale. Must print SMOKE OK before any GPU time is
committed (working agreement / scope §18-2 discipline).

Usage:  python scripts/smoke_benchmark.py
"""

from __future__ import annotations

import sys

from uqfusion.bench.smoke import run_smoke
from uqfusion.config import load_config


def main() -> int:
    cfg = load_config(sys.argv[1] if len(sys.argv) > 1 else None)
    return run_smoke(cfg)


if __name__ == "__main__":
    try:  # Windows cp1252 consoles: degrade non-ASCII output instead of crashing
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
