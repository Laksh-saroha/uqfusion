"""Is the macro metric itself the noise source every gate has been reading?

`docs/experiment-log-2026-09-02.md` §4.5 found that a single corruption draw
cannot resolve ±0.001 on the low-scoring cells. This asks the question one level
down: **how much of that noise is the macro average over two very unequal
classes?**

The headline number is `mAP@50-95` macro-averaged over ship (cls 0) and buoy
(cls 1). Measured on the paired val substrate:

* day GT: ship **10,663** boxes, buoy **600** boxes
* night GT: ship **7,870**, buoy **0** -- buoy does not exist at night

So buoy is **3.1% of the boxes carrying 50% of the metric**, estimated from 600
boxes across 1,200 frames. If buoy AP is much noisier than ship AP, then macro
variance is essentially buoy variance, every adoption decision this project has
made was reading a number dominated by 600 boxes, and the right fix is to gate
per class rather than on the macro.

Two independent variance components are separated here:

1. **Corruption draw** -- the axis §4.5 opened. Four draws (shipped 1/7 plus
   seeds 901/902/903, all nine corrupted caches rebuilt per draw by
   `gate_snms_draw_avg.py`). This is variance in the *stimulus*.
2. **Frame bootstrap** -- resample the 1,200 day frames with replacement via
   `presort`/`ap_weighted`, which is the literal resample, not an approximation.
   This is variance from a finite eval set, and it does not vanish with more
   corruption draws.

Both are reported per class and for the macro, so `var(macro)` can be attributed.
Since macro = (ship + buoy)/2, each class contributes a QUARTER of its own
variance, and buoy's share is `var_buoy / (var_ship + var_buoy)` when the two are
independent -- which is stated, not assumed, because the two classes share frames.

Day only: buoy has no night GT, so a per-class night number does not exist.

Nothing is written outside the new `--out` file. `runs/cache_m` and the draw
directories are read-only here -- no cache is built.

Usage:
    python scripts/probe_metric_noise_floor.py
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import fmt, md_table, write_md            # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import (ap_from_parts, ap_weighted,   # noqa: E402
                                     frame_parts, presort)
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402

#: The 11 benchmark cells, exactly as `gate_snms_draw_avg.py` defines them.
CELLS = [("clean", None), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("clean", "fog_s2"),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None), ("fog", None),
         ("lowlight", "glare_s2"), ("blur_s3", "glare_s2")]

#: Draws whose nine corrupted caches were ALL rebuilt at the draw seed. 904/905
#: are excluded on purpose: `redraw_snms_cell.py` re-drew only the two contested
#: caches there and hard-linked the other seven at the shipped seed, so they are
#: not independent draws for the eight cells that use those seven.
DRAWS = [("1/7 (shipped)", ROOT / "runs/cache_m"),
         ("901/911", ROOT / "runs/cache_m_draw901"),
         ("902/912", ROOT / "runs/cache_m_draw902"),
         ("903/913", ROOT / "runs/cache_m_draw903")]

SHIP, BUOY = 0, 1
NAMES = {SHIP: "ship", BUOY: "buoy"}


def cell_name(vc, ic) -> str:
    return f"{vc}/{ic or 'clean'}"


def measure_draw(cache_dir: Path) -> dict:
    """Per-cell (macro, ship AP, buoy AP) on the SHIPPED system, day frames."""
    conds = sorted({c for c, _ in CELLS})
    ircs = sorted({i for _c, i in CELLS if i})
    ctxs = {ic: load_context(preset="crossmodal26m", cache_dir=str(cache_dir),
                             conditions=tuple(conds), ir_condition=ic,
                             verbose=False)
            for ic in [None] + ircs}
    c0 = ctxs[None]
    day = np.flatnonzero(~np.isin(c0.runs, NIGHT_RUNS))
    out = {}
    for vc, ic in CELLS:
        c = ctxs[ic]
        parts = frame_parts(run_systems(c, vc)["fused_gated"], c.gts)
        a = ap_from_parts(parts, sel=day)
        pc = a["per_class"]
        out[(vc, ic)] = {
            "macro": a["map50_95"],
            SHIP: pc.get(SHIP, {}).get("ap50_95", float("nan")),
            BUOY: pc.get(BUOY, {}).get("ap50_95", float("nan")),
            "n_gt_ship": pc.get(SHIP, {}).get("n_gt", 0),
            "n_gt_buoy": pc.get(BUOY, {}).get("n_gt", 0),
        }
    return out, {(vc, ic): None for vc, ic in CELLS}


def bootstrap_cell(ctxs: dict, vc, ic, n_boot: int, seed: int) -> dict:
    """Frame-bootstrap sd of ship AP, buoy AP and macro on one cell."""
    c = ctxs[ic]
    day = np.flatnonzero(~np.isin(c.runs, NIGHT_RUNS))
    parts = frame_parts(run_systems(c, vc)["fused_gated"], c.gts)
    pre = presort(parts, sel=day)
    n = pre["n_frames"]
    rng = np.random.default_rng(seed)
    macro, ship, buoy = np.empty(n_boot), np.empty(n_boot), np.empty(n_boot)
    for t in range(n_boot):
        w = rng.multinomial(n, np.full(n, 1.0 / n))
        r = ap_weighted(pre, w)
        macro[t] = r["map50_95"]
        ship[t] = r["per_class"].get(SHIP, {}).get("ap50_95", np.nan)
        buoy[t] = r["per_class"].get(BUOY, {}).get("ap50_95", np.nan)
    return {"macro": macro, SHIP: ship, BUOY: buoy}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--out", default="runs/eval/metric_noise_floor.md")
    args = ap.parse_args()
    t0 = time.time()

    # ---- 1. draw-to-draw ---------------------------------------------------
    per_draw = []
    for label, cd in DRAWS:
        if not cd.is_dir():
            print(f"[skip] {cd} missing")
            continue
        print(f"[draw] {label}", flush=True)
        vals, _ = measure_draw(cd)
        per_draw.append((label, vals))
    assert per_draw, "no draws available"

    cells = [(vc, ic) for vc, ic in CELLS]
    draw_rows, drawsd = [], {}
    for vc, ic in cells:
        m = np.array([d[1][(vc, ic)]["macro"] for d in per_draw])
        s = np.array([d[1][(vc, ic)][SHIP] for d in per_draw])
        b = np.array([d[1][(vc, ic)][BUOY] for d in per_draw])
        sd = lambda x: float(np.std(x, ddof=1)) if len(x) > 1 else float("nan")
        drawsd[(vc, ic)] = (sd(s), sd(b), sd(m))
        draw_rows.append([cell_name(vc, ic), fmt(s.mean()), fmt(sd(s)),
                          fmt(b.mean()), fmt(sd(b)), fmt(m.mean()), fmt(sd(m))])

    # ---- 2. frame bootstrap, on the shipped draw ---------------------------
    _conds = sorted({c for c, _ in CELLS})
    _ircs = sorted({i for _c, i in CELLS if i})
    boot_ctxs = {ic: load_context(preset="crossmodal26m", cache_dir=str(DRAWS[0][1]),
                                  conditions=tuple(_conds), ir_condition=ic,
                                  verbose=False)
                 for ic in [None] + _ircs}
    boot_rows, bootsd = [], {}
    for vc, ic in cells:
        print(f"[boot] {cell_name(vc, ic)}", flush=True)
        bs = bootstrap_cell(boot_ctxs, vc, ic, args.n_boot, seed=0)
        ss, bb, mm = (float(np.nanstd(bs[SHIP], ddof=1)),
                      float(np.nanstd(bs[BUOY], ddof=1)),
                      float(np.nanstd(bs["macro"], ddof=1)))
        bootsd[(vc, ic)] = (ss, bb, mm)
        share = bb ** 2 / (ss ** 2 + bb ** 2) if (ss or bb) else float("nan")
        boot_rows.append([cell_name(vc, ic), fmt(ss), fmt(bb),
                          f"{bb / ss:.1f}x" if ss > 0 else "--",
                          fmt(mm), f"{100 * share:.0f}%"])

    # ---- 3. the combined noise floor --------------------------------------
    floor_rows = []
    for vc, ic in cells:
        _, _, dm = drawsd[(vc, ic)]
        _, _, bm = bootsd[(vc, ic)]
        tot = float(np.hypot(dm, bm))
        floor_rows.append([cell_name(vc, ic), fmt(dm), fmt(bm), fmt(tot),
                           fmt(2 * tot)])

    n_ship = per_draw[0][1][cells[0]]["n_gt_ship"]
    n_buoy = per_draw[0][1][cells[0]]["n_gt_buoy"]

    secs = [
        "Preset `crossmodal26m`, shipped system, **day frames only** (buoy has no "
        "night GT). Two independent noise sources, measured separately: the "
        "corruption draw (§4.5's axis) and the finite eval set.  \n"
        f"Day GT: ship **{n_ship}** boxes, buoy **{n_buoy}** boxes — buoy is "
        f"**{100 * n_buoy / (n_ship + n_buoy):.1f}%** of the boxes and **50%** of the "
        "macro metric.",

        "## 1. Draw-to-draw, per class\n\n"
        f"{len(per_draw)} corruption draws, all nine corrupted caches rebuilt per "
        "draw. `sd` is across draws.\n\n"
        + md_table(["cell (vis/ir)", "ship AP", "sd", "buoy AP", "sd",
                    "macro", "sd"], draw_rows),

        "## 2. Frame bootstrap, per class\n\n"
        f"{args.n_boot} resamples of the 1,200 day frames on the shipped draw. "
        "`buoy/ship` is the ratio of the two class sd's; `buoy share` is "
        "`var_buoy / (var_ship + var_buoy)`, buoy's fraction of macro variance if "
        "the two classes were independent. They are not — they share frames — so "
        "read it as an attribution, not an identity.\n\n"
        + md_table(["cell (vis/ir)", "sd ship", "sd buoy", "buoy/ship",
                    "sd macro", "buoy share"], boot_rows),

        "## 3. The macro noise floor\n\n"
        "`total` adds the two sources in quadrature. `2×total` is a rough "
        "smallest-resolvable macro effect — the size a delta has to clear before "
        "an every-cell bar is reading signal rather than the instrument.\n\n"
        + md_table(["cell (vis/ir)", "sd draw", "sd bootstrap", "total",
                    "2×total"], floor_rows),

        f"---\n\n_Generated by `scripts/probe_metric_noise_floor.py` in "
        f"{time.time() - t0:.1f}s._",
    ]
    write_md(args.out, "Is the macro metric its own noise source?", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
