"""Re-run the three analyses whose real results were shadowed by a pre-flight entry.

`run_analysis_queue.py` wrote pre-flight and real results into one state file, and
a pre-flight "done" satisfied the real run's skip test, so `x1`, `x3` and `x7`
were skipped with only their 3-resample smoke outputs on disk. The skip test now
also compares the mode, but the queue process already running holds the old code,
so this waits for it and re-runs those three with `--redo`.

Usage:
    python scripts/chain_analysis_redo.py
"""

from __future__ import annotations

import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
STATE = ROOT / "runs" / "queue_analysis" / "state.json"
JOBS = ["x1_score_calibration", "x3_veto_hysteresis", "x7_topk_truncation"]


def main() -> int:
    deadline = time.time() + 6 * 3600
    while time.time() < deadline:
        try:
            st = json.loads(STATE.read_text(encoding="utf-8")).get("queue_status", "")
        except (OSError, json.JSONDecodeError):
            st = ""
        if st.startswith("finished") or st == "failed":
            break
        time.sleep(30)
    else:
        print("[redo] analysis queue never reported terminal — not starting a second one")
        return 1

    print(f"[redo] analysis queue is '{st}' — re-running {', '.join(JOBS)}", flush=True)
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "run_analysis_queue.py"),
                        "run", "--only", *JOBS, "--redo"], cwd=str(ROOT))
    print(f"[redo] exit {r.returncode}")
    return r.returncode


if __name__ == "__main__":
    raise SystemExit(main())
