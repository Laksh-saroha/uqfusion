"""How close do the v2 gate statistics come to the gate thresholds? (2026-10-10)

The v2 corrupted-frame statistics are computed in the cache-build pass under the GPU interpreter
(py-3.13: numpy 2.1.3, OpenCV 4.10.0), while paper §10 required frame statistics under `.venv`
(numpy 2.4.6, OpenCV 5.0.0), which reproduces the v1 development files bit-exactly. The two
interpreters were measured to differ by at most 6.3e-7 relative on float statistics and not at all
on percentile statistics (all clean paired frames). A gate decision can flip between them only if
a statistic lies within that distance of its threshold. This counts the frames that do, with a
100x margin (1e-4 relative), over every v2 statistics file on disk.

    py -3.13 scripts/gate_threshold_proximity.py
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REL = 1e-4


def main() -> int:
    sc = json.loads((ROOT / "runs/eval/structure_constants.json").read_text(encoding="utf-8"))["axes"]
    bc = json.loads((ROOT / "runs/eval/brightness_constants.json").read_text(encoding="utf-8"))["vis"]
    thr = {("structure", "grad_gini"): sc["grad_gini"]["threshold"],
           ("structure", "lap_over_var"): sc["lap_over_var"]["threshold"],
           ("brightness", "lap_var"): bc["tau_lap"],
           ("brightness", "p05"): bc["mu_b"]}
    files = sorted(p for root in ("runs/derived_p3dev_v2", "runs/derived_m_v2", "runs/holdout_p04/v2/derived")
                   for p in (ROOT / root).rglob("gauss_vis_*.json"))
    tot, near, rows = 0, 0, []
    for p in files:
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("corrupt_version") != "v2" and not d.get("corrupt_code"):
            continue
        kind = "structure" if "structure" in p.parts else "brightness"
        for (k, stat), t in thr.items():
            if k != kind:
                continue
            v = [f[stat] for f in d["frames"] if stat in f]
            if not v:
                continue
            m = sum(abs(x - t) <= REL * abs(t) for x in v)
            gap = min(abs(x - t) / abs(t) for x in v)
            tot += len(v)
            near += m
            rows.append((p.relative_to(ROOT).as_posix(), stat, len(v), m, gap))
    for r in rows:
        print(f"{r[0]:<90} {r[1]:<13} n={r[2]:>5} within {REL:g}: {r[3]:>3}  min rel gap {r[4]:.2e}")
    print(f"TOTAL {near} of {tot} frame-statistics within {REL:g} relative of a gate threshold")
    if rows:
        print(f"Smallest relative gap anywhere: {min(r[4] for r in rows):.2e} (interpreter difference measured: <= 6.3e-7)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
