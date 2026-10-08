"""The paired-delta noise floor on ship AP (class 0), beside the macro.

`scripts/probe_delta_noise_floor.py` measured the floor on
`ap_weighted(...)["map50_95"]` — the ship+buoy macro. Every headline verdict
that used that floor is on **ship AP** (IR is a single-class ship detector, so a
VIS veto zeroes buoy AP and the macro is not the quantity being gated).
`runs/eval/metric_noise_floor.md` §2 already showed ship AP's *level* sd is ~2x
the macro's on cells where buoy AP does not vary. This script measures the
*paired delta* floor on ship AP directly.

Same preset (`crossmodal26m`), same caches (`runs/cache_m` + draws 901–903),
same 11 cells, same probe arm (VIS soft-NMS 0.5), same day frames, same
bootstrap seeds and n_boot as the macro probe. The macro is recomputed in the
same run, from the same resamples' code path, so the two columns are
comparable and the macro column doubles as a reproduction check of
`runs/eval/delta_noise_floor.md`.

Ship AP is read as `ap_weighted(...)["per_class"][0]["ap50_95"]`, which on the
unweighted call is identical to `ap_from_parts(...)["per_class"][0]` (the path
`scripts/p3_night_check.py` uses); the paired bootstrap is `bootstrap_delta(...,
cls=0)`, the same function and the same per-class branch the gates call.

Night is not measured: the shipped veto drops VIS on night frames, so a VIS-only
probe arm has a delta of exactly zero there and carries no information.

No pohang04 frame is read: `load_context` serves the paired val manifest.

Usage:
    py -3.13 scripts/probe_delta_noise_floor_ship.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import fmt, md_table, sgn, write_md          # noqa: E402
from probe_delta_noise_floor import (CELLS, DRAWS, SIGMA, SHIP,  # noqa: E402
                                     cell_name, contexts)

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import (ap_weighted, bootstrap_delta,   # noqa: E402
                                     frame_parts, presort)
from uqfusion.eval.ctx import NIGHT_RUNS, run_systems                # noqa: E402


def scores(pre: dict, w=None) -> tuple[float, float]:
    """(macro, ship) from one weighted scoring. Ship is NaN if undefined."""
    r = ap_weighted(pre, w)
    pc = r["per_class"].get(SHIP)
    ship = (float("nan") if pc is None or pc.get("excluded")
            else pc["ap50_95"])
    return r["map50_95"], ship


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/eval/delta_noise_floor_ship.md")
    args = ap.parse_args()
    t0 = time.time()

    # ---- paired vs unpaired, on the shipped draw --------------------------
    base, arm = contexts(DRAWS[0][1])
    c0 = base[None]
    day = np.flatnonzero(~np.isin(c0.runs, NIGHT_RUNS))

    paired = {}
    for vc, ic in CELLS:
        print(f"[paired] {cell_name(vc, ic)}", flush=True)
        pb = frame_parts(run_systems(base[ic], vc)["fused_gated"], base[ic].gts)
        pa = frame_parts(run_systems(arm[ic], vc)["fused_gated"], arm[ic].gts)

        # same seed for both -> identical resample sequence for macro and ship
        bm = bootstrap_delta(pa, pb, sel=day, n_boot=args.n_boot, seed=0)
        bs = bootstrap_delta(pa, pb, sel=day, n_boot=args.n_boot, seed=0, cls=SHIP)

        # unpaired: independent resamples of each arm, differenced
        pre_a, pre_b = presort(pa, sel=day), presort(pb, sel=day)
        n = pre_a["n_frames"]
        ra = np.random.default_rng(1)
        rb = np.random.default_rng(2)
        p = np.full(n, 1.0 / n)
        dm, ds = np.empty(args.n_boot), np.empty(args.n_boot)
        for t in range(args.n_boot):
            ma, sa = scores(pre_a, ra.multinomial(n, p))
            mb, sb = scores(pre_b, rb.multinomial(n, p))
            dm[t], ds[t] = ma - mb, sa - sb

        paired[(vc, ic)] = {
            "macro": {"delta": bm["delta"],
                      "sd_paired_ci": (bm["ci_hi"] - bm["ci_lo"]) / (2 * 1.96),
                      "sd_paired_se": bm["se"],
                      "sd_unpaired": float(np.std(dm, ddof=1)),
                      "n_undefined": bm["n_undefined"]},
            "ship": {"delta": bs["delta"],
                     "sd_paired_ci": (bs["ci_hi"] - bs["ci_lo"]) / (2 * 1.96),
                     "sd_paired_se": bs["se"],
                     "sd_unpaired": float(np.nanstd(ds, ddof=1)),
                     "n_undefined": bs["n_undefined"]},
        }

    # ---- draw-to-draw sd of the same delta --------------------------------
    per_draw: dict = {}
    draws_used = []
    for label, cd in DRAWS:
        if not cd.is_dir():
            print(f"[draw] {label} MISSING {cd}", flush=True)
            continue
        draws_used.append(label)
        print(f"[draw] {label}", flush=True)
        b, a = contexts(cd)
        d = np.flatnonzero(~np.isin(b[None].runs, NIGHT_RUNS))
        for vc, ic in CELLS:
            qb = frame_parts(run_systems(b[ic], vc)["fused_gated"], b[ic].gts)
            qa = frame_parts(run_systems(a[ic], vc)["fused_gated"], a[ic].gts)
            ma, sa = scores(presort(qa, sel=d))
            mb, sb = scores(presort(qb, sel=d))
            per_draw.setdefault((vc, ic), {"macro": [], "ship": []})
            per_draw[(vc, ic)]["macro"].append(ma - mb)
            per_draw[(vc, ic)]["ship"].append(sa - sb)

    # ---- assemble ------------------------------------------------------------
    pair_rows, floor_rows, cmp_rows, js = [], [], [], {}
    for vc, ic in CELLS:
        name = cell_name(vc, ic)
        rec = {}
        for k in ("macro", "ship"):
            arr = np.asarray(per_draw[(vc, ic)][k])
            dsd = float(np.std(arr, ddof=1)) if len(arr) > 1 else float("nan")
            psd = paired[(vc, ic)][k]["sd_paired_ci"]
            tot = float(np.hypot(dsd, psd))
            rec[k] = dict(paired[(vc, ic)][k], sd_draw=dsd, total=tot,
                          floor_2x=2 * tot, per_draw=arr.tolist())
        js[name] = rec
        m, s = rec["macro"], rec["ship"]
        pair_rows.append([name, sgn(s["delta"]), fmt(s["sd_paired_ci"]),
                          fmt(s["sd_unpaired"]),
                          (f"{s['sd_unpaired'] / s['sd_paired_ci']:.0f}x"
                           if s["sd_paired_ci"] > 0 else "--")])
        floor_rows.append([name, fmt(s["sd_draw"]), fmt(s["sd_paired_ci"]),
                           fmt(s["total"]), fmt(s["floor_2x"]),
                           "draw" if s["sd_draw"] > s["sd_paired_ci"] else "frames"])
        ratio = (f"{s['floor_2x'] / m['floor_2x']:.2f}x" if m["floor_2x"] > 0 else "--")
        cmp_rows.append([name, sgn(m["delta"]), sgn(s["delta"]),
                         fmt(m["floor_2x"]), fmt(s["floor_2x"]), ratio])

    mx_m = max(r["macro"]["floor_2x"] for r in js.values())
    mx_s = max(r["ship"]["floor_2x"] for r in js.values())

    secs = [
        "Ship AP (class 0) variant of `runs/eval/delta_noise_floor.md`. Same preset "
        "(`crossmodal26m`), caches, 11 cells, probe arm (`vis_soft_nms` "
        f"{SIGMA}), day frames, bootstrap seeds and n_boot ({args.n_boot}). The macro "
        "is recomputed in the same run. Draws used: "
        + ", ".join(draws_used) + ". Night is not measured: the veto drops VIS at "
        "night, so a VIS-only probe arm has a delta of exactly zero there.",

        "## 1. Ship floor beside the macro floor\n\n"
        "`2×total` = 2·hypot(sd draw, sd paired bootstrap), the same construction as "
        "the macro file's §2.\n\n"
        + md_table(["cell (vis/ir)", "delta macro", "delta ship", "2×total macro",
                    "2×total ship", "ship/macro"], cmp_rows)
        + f"\n\nMax over cells: macro **{fmt(mx_m)}**, ship **{fmt(mx_s)}**. "
        f"Max × 1.95: macro {fmt(mx_m * 1.95)}, ship {fmt(mx_s * 1.95)}.",

        "## 2. Ship: paired vs unpaired\n\n"
        + md_table(["cell (vis/ir)", "delta", "sd paired", "sd unpaired", "ratio"],
                   pair_rows),

        "## 3. Ship: the noise floor for a gated delta\n\n"
        + md_table(["cell (vis/ir)", "sd draw", "sd paired boot", "total",
                    "2×total", "binding"], floor_rows),

        f"---\n\n_Generated by `scripts/probe_delta_noise_floor_ship.py` in "
        f"{time.time() - t0:.1f}s._",
    ]
    write_md(args.out, "The noise floor for a paired delta — ship AP", secs)
    Path(args.out).with_suffix(".json").write_text(
        json.dumps({"n_boot": args.n_boot, "sigma": SIGMA, "draws": draws_used,
                    "cells": js}, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
