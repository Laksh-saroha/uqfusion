"""Per-class AP behind Table 2's mAP50-95 column: ship and buoy, per stream and arm.

Table 2 (`docs/eval/uq_day_night_slice_u2_nanpolicy_2026-09-09.md`) reports `map50_95`, the
macro over the classes present in each stream's ground truth. The IR detector is single-class
(nc=1, ship) while the IR day labels still carry buoys, so IR's macro averages its ship AP with
a buoy AP of exactly zero and reads half of it. This recomputes the column per class from the
same caches and the same day slice (1,200 paired frames, every run but pohang01), and checks
that the macro reproduces Table 2 before writing anything.

Usage:
    python scripts/table2_per_class.py --out docs/eval/table2_per_class_2026-10-08.md
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.cache import load_cache  # noqa: E402
from uqfusion.eval.matching import load_gt, map50_95  # noqa: E402

CACHE = ROOT / "runs" / "cache_uqslice"
NIGHT_RUN = "pohang01"
SHIP, BUOY = 0, 1

#: (stream, arm, cache, Table 2's published mAP50-95). Same caches as `paper_figures.py` FIG2.
ARMS = [
    ("VIS", "σ head", "sigma_vis_seed0_nightfull", 0.3384),
    ("VIS", "MC-Dropout", "mc_vis_nightfull", 0.3103),
    ("VIS", "ensemble (5)", "ens_vis_nightfull", 0.3464),
    ("IR", "σ head", "sigma_ir", 0.0707),
    ("IR", "MC-Dropout", "mc_ir", 0.0691),
    ("IR", "ensemble (5)", "ens_ir", 0.0744),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    sys.stdout.reconfigure(encoding="utf-8")   # arm names carry σ; a cp1252 console cannot

    rows = []
    for stream, arm, name, published in ARMS:
        recs, _ = load_cache(CACHE / f"{name}.pkl")
        day = [r for r in recs if NIGHT_RUN not in str(r["image_path"])]
        gts = [load_gt(r["image_path"], r["image_hw"]) for r in day]
        m = map50_95(day, gts)
        if abs(m["map50_95"] - published) > 5e-5:
            raise SystemExit(f"{stream} {arm}: macro {m['map50_95']:.4f} does not reproduce "
                             f"Table 2's {published:.4f}; refusing to write")
        emitted = sorted({int(c) for r in day for c in r["cls"]})
        pc = m["per_class"]
        buoy = pc.get(BUOY, {})
        rows.append((stream, arm, len(day), m["map50_95"], pc[SHIP]["ap50_95"],
                     buoy.get("ap50_95"), buoy.get("n_gt", 0), emitted))
        print(f"{stream} {arm:12s} macro {m['map50_95']:.4f} ship {pc[SHIP]['ap50_95']:.4f} "
              f"buoy {buoy.get('ap50_95', float('nan')):.4f} emits {emitted}")

    lines = [
        "# Table 2, per class",
        "",
        "Written by `scripts/table2_per_class.py`. Day slice: 1,200 paired validation frames (every "
        "run but `pohang01`), local AP50-95, caches in `runs/cache_uqslice/` (the ones Table 2 and "
        "Figure 2 use). Each row's macro reproduces Table 2's published `map50_95` to 5e-5; the "
        "script refuses to write otherwise.",
        "",
        "| stream | arm | frames | macro (Table 2) | ship AP | buoy AP | buoy GT | classes emitted |",
        "|---|---|---:|---:|---:|---:|---:|---|",
    ]
    for stream, arm, n, macro, ship, buoy, n_buoy, emitted in rows:
        b = "—" if buoy is None else f"{buoy:.4f}"
        lines.append(f"| {stream} | {arm} | {n} | {macro:.4f} | {ship:.4f} | {b} | {n_buoy} | "
                     f"{', '.join(map(str, emitted))} |")
    lines += [
        "",
        "**Reading.** The IR detector emits class 0 only, yet the IR day labels carry 596 buoy "
        "boxes, so the macro scores buoy at exactly 0 and IR's `map50_95` is half its ship AP on "
        "every arm. The arm ordering is the same on ship AP as on the macro, for both streams.",
    ]
    args.out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
