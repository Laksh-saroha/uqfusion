"""The AP cost of the weak-IR fallback under corruption v2, development, descriptive (2026-10-10).

Table 6 under v2 (`ir_hazards_v2.py`) found the night switch's weak-IR fallback vetoing a
low-light VIS stream on up to 96% of DAY frames when IR is fogged or noisy: a disarmed IR's
night call is confirmed by VIS `dark AND veil` (or `concentrated`), and v2 low light trips
both on every day frame. Table 6 counts vetoes; this prices them. VIS low light s2 (draw 941)
and clean VIS beside clean IR and IR fog s1 / s2 / s3 and IR noise s2 / s3 (IR draw 951),
five Phase 3 systems, `crossmodal26m`, ship AP, day and night apart, scored exactly as
`v2_sensitivity.py` scores a row. Reported per cell: VIS-only, IR-only, fused, the veto rate,
and fused - max(VIS, IR) with the between-seed 95% t-interval (df 4), the Table 3b claim.

Changes no rule and no constant; scores no pohang04 frame (exposure ledger, 2026-10-10).

    py -3.13 scripts/build_corruption_v2_dev.py --draws 941 --conds ir_fog_s1 ir_fog_s3 ir_noise_s2 ir_noise_s3
    py -3.13 scripts/v2_fallback_cost.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from v2_sensitivity import ARMS, DRAW, SEEDS, T_975_DF4, score   # noqa: E402

VIS_CONDS = ("lowlight", "clean")
IR_CONDS = (None, "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/eval/corruption_v2/fallback_cost.md")
    ap.add_argument("--workers", type=int, default=6)
    args = ap.parse_args()
    t0 = time.time()
    jobs = [(k, list(VIS_CONDS), ic) for ic in IR_CONDS for k in SEEDS]
    res: dict = {}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, out in ex.map(score, jobs):
            for key, v in out.items():
                res.setdefault(key, {})[k] = v
    print(f"[fb] {len(jobs)} contexts scored ({time.time() - t0:.0f}s)", flush=True)

    L = [f"Five Phase 3 systems, `crossmodal26m`, VIS draw {DRAW}, IR draw {DRAW + 10}; paired val "
         "(day = pohang00/02/03, night = pohang01). Ship AP, mean over the five systems; "
         "fused − max(VIS, IR) with its 95% t-interval (df 4) over systems. The veto rate is the share "
         "of frames on which VIS is dropped.", "",
         "| VIS | IR | slice | VIS-only | IR-only | fused | veto | fused − max(VIS, IR) [95% CI] | fused − fused at clean IR |",
         "|---|---|---|---:|---:|---:|---:|---|---:|"]
    js = []
    for vc in VIS_CONDS:
        for ic in IR_CONDS:
            for s in ("day", "night"):
                a = {x: np.array([res[(vc, ic, s, x)][k] for k in SEEDS]) for x in (*ARMS, "veto")}
                d = a["fused"] - np.maximum(a["vis"], a["ir"])
                m, h = float(d.mean()), float(T_975_DF4 * d.std(ddof=1) / np.sqrt(len(SEEDS)))
                base = np.array([res[(vc, None, s, "fused")][k] for k in SEEDS])
                dc = float((a["fused"] - base).mean())
                L.append(f"| {vc} | {ic or 'clean'} | {s} | {a['vis'].mean():.4f} | {a['ir'].mean():.4f} | "
                         f"{a['fused'].mean():.4f} | {a['veto'].mean():.3f} | {m:+.4f} [{m - h:+.4f}, {m + h:+.4f}] | "
                         f"{dc:+.4f} |")
                js.append({"vis": vc, "ir": ic or "clean", "slice": s,
                           **{x: float(a[x].mean()) for x in (*ARMS, "veto")},
                           "fused_minus_max": m, "ci": [m - h, m + h], "fused_minus_clean_ir": dc,
                           "per_system": {x: a[x].tolist() for x in (*ARMS, "veto")}})
    out = ROOT / args.out
    out.write_text("# The AP cost of the weak-IR fallback under corruption v2\n\n" + "\n".join(L) + "\n",
                   encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({"cells": js, "elapsed_s": time.time() - t0}, indent=1),
                                        encoding="utf-8")
    print("\n".join(L))
    print(f"[fb] wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    raise SystemExit(main())
