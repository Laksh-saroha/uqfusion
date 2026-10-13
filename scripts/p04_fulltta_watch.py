"""Score `full_tta` on each corrupted draw of pohang04 as soon as that draw's flipped and restored caches
exist, then bootstrap whatever presorts are new. Low CPU priority, so the build keeps the CPU it needs.

    py -3.13 scripts/p04_fulltta_watch.py
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import p04_heldout as ph                                                             # noqa: E402

LOW = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
#: arms scored per draw; `full_tta_lex` is the gate-free registration's Amendment 2 (descriptive)
ARMS = os.environ.get("P04_WATCH_ARMS", "full_tta full_tta_lex").split()


def main() -> int:
    env = dict(os.environ, UQF_FAST_WBF="1", PYTHONIOENCODING="utf-8", P04_BOOT_R="4", P04_BOOT_MEMFRAC="0.35")
    log_dir = ROOT / "runs/holdout_p04"
    for d in ph.VIS_DRAWS:
        units = [u for u in ph.units() if u[2] == d]
        while not all(ph.unit_done(*u) for u in units):
            time.sleep(30)
        print(f"[watch] draw {d}: {len(units)} units built, scoring {' '.join(ARMS)} ({time.strftime('%I:%M %p')})", flush=True)
        with open(log_dir / f"gatefree_parts_fulltta_{d}.log", "a", encoding="utf-8") as fh:
            rc = subprocess.call([sys.executable, str(ROOT / "scripts/p04_heldout_score.py"), "parts", "--cells",
                                  "corrupted", "--arms", *ARMS, "--seeds", "0", "--split", "--subs", ph.sub(d),
                                  "--workers", "2"], cwd=str(ROOT), env=env, creationflags=LOW, stdout=fh,
                                 stderr=subprocess.STDOUT)
        print(f"[watch] draw {d}: parts rc={rc} ({time.strftime('%I:%M %p')})", flush=True)
        if rc:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
