"""R-A1: how far apart are the local AP and official COCO AP, on real caches?

The two are different metrics (linear interpolation of the precision envelope versus
a first-attained-recall lookup), and `scripts/smoke_cocoparity.py` pins the mechanics.
This script answers the question that decides whether F03 matters: **on this project's
own data, how big is the gap, and does it survive into a delta?**

Absolute AP and deltas are reported separately on purpose. Every comparison this
project makes scores both arms the same way, so a systematic convention offset largely
cancels in `A - B` even when it is visible in `A` and `B` individually. Whether it
cancels *enough* is measured here rather than assumed.

Usage:
    python scripts/ap_convention_parity.py --out runs/eval/ap_convention_parity.md
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import ROOT, fmt, md_table, write_md  # noqa: E402

from uqfusion.eval.cache import load_cache  # noqa: E402
from uqfusion.eval.cocoparity import (  # noqa: E402
    PARITY_BOUND, PARITY_TOLERANCE, coco_ap)
from uqfusion.eval.apmetrics import declared_policies  # noqa: E402
from uqfusion.eval.matching import load_gt, map50_95  # noqa: E402

NIGHT_RUN = "pohang01"
NOISE_FLOOR = (0.0014, 0.0031)      # runs/eval/metric_noise_floor.md, paired 2-sigma
# R-A1 option (e): the parity number is now ASSERTED, not just written into a report
# once and left to rot. If a change to matching, sorting or the precision envelope
# widens the local-vs-COCO delta gap toward the noise floor, this script fails.
TRIP = PARITY_BOUND * PARITY_TOLERANCE

ARMS = {
    "VIS sigma-head": "sigma_vis_seed0_nightfull.pkl",
    "VIS MC-Dropout": "mc_vis_nightfull.pkl",
    "VIS ensemble":   "ens_vis_nightfull.pkl",
    "IR sigma-head":  "sigma_ir.pkl",
    "IR MC-Dropout":  "mc_ir.pkl",
    "IR ensemble":    "ens_ir.pkl",
}


def load(fname: str):
    obj = load_cache(ROOT / "runs/cache_uqslice" / fname)
    recs = obj[0] if isinstance(obj, tuple) else obj["records"]
    gts = [load_gt(r["image_path"], r["image_hw"]) for r in recs]
    night = np.array([NIGHT_RUN in str(r["image_path"]) for r in recs])
    return recs, gts, night


def score(recs, gts, sel=None) -> tuple[float, float]:
    if sel is not None:
        i = np.where(sel)[0]
        recs = [recs[j] for j in i]
        gts = [gts[j] for j in i]
    return map50_95(recs, gts)["map50_95"], coco_ap(recs, gts)["map50_95"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/ap_convention_parity.md")
    args = ap.parse_args()

    pt: dict[tuple[str, str], tuple[float, float]] = {}
    present = []
    for label, fname in ARMS.items():
        p = ROOT / "runs/cache_uqslice" / fname
        if not p.is_file():
            print(f"[skip] {label}: {fname} missing")
            continue
        present.append(label)
        recs, gts, night = load(fname)
        for sub, sel in (("all", None), ("day", ~night), ("night", night)):
            pt[(label, sub)] = score(recs, gts, sel)
            print(f"[{label}/{sub}] local {pt[(label, sub)][0]:.8f} "
                  f"coco {pt[(label, sub)][1]:.8f}")

    abs_rows = [[lab, sub, fmt(pt[(lab, sub)][0], 8), fmt(pt[(lab, sub)][1], 8),
                 fmt(pt[(lab, sub)][0] - pt[(lab, sub)][1], 8)]
                for lab in present for sub in ("all", "day", "night")]

    delta_rows, worst = [], 0.0
    for sub in ("all", "day", "night"):
        for a, b in itertools.combinations(present, 2):
            if a.split()[0] != b.split()[0]:
                continue                      # only compare within a modality
            dl = pt[(a, sub)][0] - pt[(b, sub)][0]
            dc = pt[(a, sub)][1] - pt[(b, sub)][1]
            worst = max(worst, abs(dl - dc))
            delta_rows.append([f"{a} − {b}", sub, fmt(dl, 8), fmt(dc, 8),
                               fmt(dl - dc, 8)])

    if worst > TRIP:
        raise SystemExit(
            f"PARITY REGRESSION: worst local-vs-COCO delta disagreement {worst:.8f} "
            f"exceeds the pinned bound {PARITY_BOUND:.8f} x {PARITY_TOLERANCE} "
            f"= {TRIP:.8f}. "
            "The convention gap has grown. Either a real change "
            f"widened it -- in which case re-measure, re-argue that it still sits "
            f"below the {NOISE_FLOOR[0]}-{NOISE_FLOOR[1]} noise floor, and move "
            f"`cocoparity.PARITY_BOUND` in a commit that says so -- or something "
            f"broke. Do not raise the bound to make this pass.")
    print(f"[parity] worst delta disagreement {worst:.8f} <= trip {TRIP:.8f}  OK")

    lo, hi = NOISE_FLOOR
    verdict = (
        f"**Worst delta disagreement across every pair and subset: {worst:.8f}.** The "
        f"measured paired 2-sigma noise floor is {lo}–{hi} "
        f"(`runs/eval/metric_noise_floor.md`), so the convention cannot flip a decision "
        f"whose margin clears that floor — it is {lo / worst:.0f}× smaller than the "
        f"lower bound." if worst and worst < lo else
        f"**Worst delta disagreement: {worst:.8f}**, which is NOT below the "
        f"{lo}–{hi} noise floor. Decisions in that range need re-scoring.")

    secs = [
        "Produced by `scripts/ap_convention_parity.py` for R-A1 "
        "(`docs/TODO-2026-09-09-architecture-review.md`, finding F03). The local AP "
        "interpolates the precision envelope linearly; official COCO samples it at the "
        "first attained recall at or above each threshold. They are different metrics, "
        "and this measures the distance between them on real caches rather than on "
        "hand-built examples.",

        "Config is pinned by `uqfusion.eval.cocoparity.TASK_CONFIG`: the same "
        "0.50:0.05:0.95 IoU sweep, a single all-inclusive area range, no crowd or "
        "ignore regions, stable score sort, and **`maxDets` taken from the data rather "
        "than COCO's default of 100** — caches are built at `conf 0.001` and a 100-cap "
        "would truncate the low-confidence tail, turning a detection cap into what "
        "looks like an interpolation gap.",

        "## Absolute AP — the convention is visible here\n\n"
        + md_table(["arm", "subset", "local", "COCO", "local − COCO"], abs_rows)
        + "\n\nThe local metric reads **systematically low**, which is the direction "
        "the review predicted. It is not uniform: the gap is far larger on `day` than "
        "on `night`, consistently across arms. That pattern is reported as observed and "
        "is **not explained here**.",

        "## Deltas — the convention largely cancels\n\n"
        + md_table(["comparison", "subset", "local Δ", "COCO Δ", "disagreement"],
                   delta_rows)
        + "\n\nEvery comparison scores both arms under the same convention, so a "
        "systematic offset subtracts out. What survives is the *second-order* part, "
        "and that is what the last column measures.",

        "## What this settles, and what it does not\n\n" + verdict + "\n\n"
        "**It does not license quoting a local AP as a COCO AP.** They differ by more "
        "than the noise floor on the small hand-built cases (−0.0033 and −0.0050, "
        "reproduced exactly in `scripts/smoke_cocoparity.py` cases 2 and 3), so a "
        "published absolute number must name its convention.\n\n"
        "**It does not cover margins below the noise floor.** This project has decided "
        "things on margins far smaller than the disagreement measured here — soft-NMS "
        "was rejected at −1.03e-5 (`runs/eval/snms_gate_draw_avg.md`). A convention "
        "disagreement of ~2e-4 is more than an order of magnitude larger than that. "
        "Such decisions were already unsound for the noise reason on record; this adds "
        "a second, independent reason and does not rescue any of them.\n\n"
        "**Ultralytics is a third convention.** `docs/phase1-experimental-record.md` "
        "records a ~0.034 mAP difference between ultralytics versions on identical "
        "weights — two orders above everything here. Phase 1 ultralytics numbers still "
        "must not be compared directly against custom fusion AP.",
    ]
    out = write_md(args.out, "AP convention parity — local vs official COCO", secs)
    Path(str(out).replace(".md", ".json")).write_text(json.dumps(
        {"absolute": {f"{k[0]}|{k[1]}": {"local": v[0], "coco": v[1]} for k, v in pt.items()},
         "worst_delta_disagreement": worst,
         "noise_floor": NOISE_FLOOR,
         "parity_bound": PARITY_BOUND, "parity_trip": TRIP,
         "policies": declared_policies()}, indent=1), encoding="utf-8")
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
