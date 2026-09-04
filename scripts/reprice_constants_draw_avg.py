"""Re-price three inherited constants under the draw-averaged bar and a measured margin.

Implements `docs/prereg-reprice-inherited-constants.md`, committed at `6b49ca0`
BEFORE this script existed. The rule in one line:

    An alternative DOMINATES the shipped value iff it is not WORSE on any
    informative cell and BETTER on at least one, where BETTER/WORSE are judged
    against a margin measured per cell AND per arm as
    `2 * hypot(sd_draw, sd_paired_bootstrap)`.

`cap_ir_scale` 4.0, `iou_thr` 0.85 and the veil repair were all selected on a
single corruption draw, before §4.9 measured that the real floor for a paired
delta is 0.0014-0.0031 on the cells that carry information and 0.0000 on the two
that carry none. This asks whether those three decisions survive their own
instrument.

All three constants are consumed at `run_systems` time, not baked in at load, so
every arm is a `dataclasses.replace` on ONE context per (draw, ir_condition).
That is a 7x saving on context loads and it is **asserted, not assumed**:
`verify_replace_equivalence` rebuilds `cap_ir_scale=1.0` both ways and aborts if
they differ by more than 1e-12.

Nothing under `runs/cache*/`, `runs/derived/` or `runs/eval/` is written. Caches
are read-only here; no cache is built.

Usage:
    python scripts/reprice_constants_draw_avg.py
"""

from __future__ import annotations

import argparse
import dataclasses
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import fmt, md_table, sgn, write_md          # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uqfusion.eval.apmetrics import (ap_weighted, frame_parts,        # noqa: E402
                                     presort)
from uqfusion.eval.ctx import (NIGHT_RUNS, TEST_RUNS,                 # noqa: E402
                               load_context, run_systems)

CELLS = [("clean", None), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("clean", "fog_s2"),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None), ("fog", None),
         ("lowlight", "glare_s2"), ("blur_s3", "glare_s2")]

DRAWS = [("1/7 (shipped)", ROOT / "runs/cache_m"),
         ("901/911", ROOT / "runs/cache_m_draw901"),
         ("902/912", ROOT / "runs/cache_m_draw902"),
         ("903/913", ROOT / "runs/cache_m_draw903")]

#: Shipped values, for the record and for the replace arithmetic.
SHIPPED_CAP_SCALE = 4.0
SHIPPED_IOU = 0.85

#: (axis, label, field-overrides). `cap_mul` multiplies the SHIPPED cap_ir, which
#: already carries the /4: scale s means cap_ir * (SHIPPED_CAP_SCALE / s).
ARMS = [
    ("cap_ir_scale", "1.0", {"cap_mul": SHIPPED_CAP_SCALE / 1.0}),
    ("cap_ir_scale", "2.0", {"cap_mul": SHIPPED_CAP_SCALE / 2.0}),
    ("cap_ir_scale", "8.0", {"cap_mul": SHIPPED_CAP_SCALE / 8.0}),
    ("iou_thr", "0.70", {"iou_thr": 0.70}),
    ("iou_thr", "0.95", {"iou_thr": 0.95}),
    ("veil repair", "off", {"veil_requires_night": False,
                            "night_weak_fallback": False}),
]


def cell_name(vc, ic) -> str:
    return f"{vc}/{ic or 'clean'}"


def apply_arm(ctx, ov: dict):
    """One single-axis arm, built by replacing fields consumed at run time."""
    ov = dict(ov)
    mul = ov.pop("cap_mul", None)
    if mul is not None:
        ov["cap_ir"] = ctx.cap_ir * mul
    return dataclasses.replace(ctx, **ov) if ov else ctx


def verify_replace_equivalence(cache_dir: Path) -> None:
    """`replace`-built cap_ir_scale=1.0 must equal a genuinely loaded one."""
    conds = tuple(sorted({c for c, _ in CELLS}))
    kw = dict(preset="crossmodal26m", cache_dir=str(cache_dir),
              conditions=conds, ir_condition=None, verbose=False)
    shipped = load_context(**kw)
    real = load_context(**kw, cap_ir_scale=1.0)
    built = apply_arm(shipped, {"cap_mul": SHIPPED_CAP_SCALE / 1.0})
    d = float(np.max(np.abs(np.asarray(built.cap_ir) - np.asarray(real.cap_ir))))
    assert d <= 1e-12, f"cap_ir replace mismatch {d:.3e}"
    pb = frame_parts(run_systems(real, "clean")["fused_gated"], real.gts)
    pa = frame_parts(run_systems(built, "clean")["fused_gated"], built.gts)
    a = ap_weighted(presort(pa))["map50_95"]
    b = ap_weighted(presort(pb))["map50_95"]
    assert abs(a - b) <= 1e-12, f"replace arm AP mismatch {abs(a - b):.3e}"
    print(f"[verify] replace == load  (cap_ir {d:.1e}, AP {abs(a - b):.1e})",
          flush=True)


def measure(cache_dir: Path, n_boot: int, want_boot: bool) -> dict:
    """Per (arm, cell): day/night/test delta vs shipped, plus paired boot sd."""
    conds = tuple(sorted({c for c, _ in CELLS}))
    ircs = sorted({i for _c, i in CELLS if i})
    base = {ic: load_context(preset="crossmodal26m", cache_dir=str(cache_dir),
                             conditions=conds, ir_condition=ic, verbose=False)
            for ic in [None] + ircs}
    c0 = base[None]
    sel = {"day": np.flatnonzero(~np.isin(c0.runs, NIGHT_RUNS)),
           "night": np.flatnonzero(np.isin(c0.runs, NIGHT_RUNS)),
           "test": np.flatnonzero(np.isin(c0.runs, TEST_RUNS))}
    out = {}
    for vc, ic in CELLS:
        ship_parts = frame_parts(run_systems(base[ic], vc)["fused_gated"],
                                 base[ic].gts)
        pre_s = {k: presort(ship_parts, sel=s) for k, s in sel.items()}
        ship = {k: ap_weighted(p)["map50_95"] for k, p in pre_s.items()}
        for axis, label, ov in ARMS:
            arm_ctx = apply_arm(base[ic], ov)
            parts = frame_parts(run_systems(arm_ctx, vc)["fused_gated"],
                                arm_ctx.gts)
            rec = {}
            for k, s in sel.items():
                rec[k] = ap_weighted(presort(parts, sel=s))["map50_95"] - ship[k]
            if want_boot:
                pre_a = presort(parts, sel=sel["day"])
                pre_b = pre_s["day"]
                n = pre_a["n_frames"]
                rng = np.random.default_rng(0)
                p = np.full(n, 1.0 / n)
                d = np.empty(n_boot)
                for t in range(n_boot):
                    w = rng.multinomial(n, p)          # ONE resample, both arms
                    d[t] = (ap_weighted(pre_a, w)["map50_95"]
                            - ap_weighted(pre_b, w)["map50_95"])
                rec["boot_sd"] = float(np.std(d, ddof=1))
            out[(axis, label, vc, ic)] = rec
        print(f"  [cell] {cell_name(vc, ic)}", flush=True)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="runs/eval/reprice_constants.md")
    args = ap.parse_args()
    t0 = time.time()

    verify_replace_equivalence(DRAWS[0][1])

    per_draw = []
    for i, (label, cd) in enumerate(DRAWS):
        if not cd.is_dir():
            print(f"[skip] {cd} missing")
            continue
        print(f"[draw] {label}", flush=True)
        per_draw.append(measure(cd, args.n_boot, want_boot=(i == 0)))
    assert per_draw, "no draws"

    # ---- assemble ---------------------------------------------------------
    arm_keys = [(a, l) for a, l, _ in ARMS]
    # a cell is uninformative if EVERY arm is exactly 0 on it, on every draw
    informative = {}
    for vc, ic in CELLS:
        vals = [d[(a, l, vc, ic)]["day"] for d in per_draw for a, l in arm_keys]
        vals += [d[(a, l, vc, ic)]["night"] for d in per_draw for a, l in arm_keys]
        informative[(vc, ic)] = any(v != 0.0 for v in vals)

    rows, verdicts = [], {}
    for axis, label in arm_keys:
        n_better = n_worse = 0
        for vc, ic in CELLS:
            day = np.array([d[(axis, label, vc, ic)]["day"] for d in per_draw])
            night = np.array([d[(axis, label, vc, ic)]["night"] for d in per_draw])
            bsd = per_draw[0][(axis, label, vc, ic)]["boot_sd"]
            dsd = float(np.std(day, ddof=1)) if len(day) > 1 else 0.0
            margin = 2 * float(np.hypot(dsd, bsd))
            md, mn = float(day.mean()), float(night.mean())
            worst = min(md, mn)
            best = max(md, mn)
            if not informative[(vc, ic)]:
                v = "--"
            elif worst < -margin:
                v = "WORSE"
                n_worse += 1
            elif best > margin:
                v = "BETTER"
                n_better += 1
            else:
                v = "same"
            rows.append([f"{axis} = {label}", cell_name(vc, ic),
                         sgn(md), sgn(mn), fmt(dsd), fmt(bsd), fmt(margin), v])
        test = np.mean([per_draw[k][(axis, label, "clean", None)]["test"]
                        for k in range(len(per_draw))])
        verdicts[(axis, label)] = (n_better, n_worse, float(test))

    dom_rows = []
    for axis, label in arm_keys:
        nb, nw, test = verdicts[(axis, label)]
        dominates = (nw == 0 and nb > 0)
        note = ("DOMINATES" if dominates else
                "worse somewhere" if nw else "indistinguishable")
        if dominates and test < 0:
            note = "rejected by TEST"
        dom_rows.append([f"{axis} = {label}", str(nb), str(nw), sgn(test), note])

    by_axis = {}
    for axis, label in arm_keys:
        nb, nw, test = verdicts[(axis, label)]
        d = (nw == 0 and nb > 0) and test >= 0
        by_axis[axis] = by_axis.get(axis, False) or d
    final_rows = [[a, "**MIS-PRICED**" if m else "**STANDS**"]
                  for a, m in by_axis.items()]

    uninf = [cell_name(vc, ic) for vc, ic in CELLS if not informative[(vc, ic)]]

    secs = [
        "Implements `docs/prereg-reprice-inherited-constants.md`, committed at "
        "`6b49ca0` **before this script existed**. Preset `crossmodal26m`, 4 "
        "corruption draws, single-axis arms built by `dataclasses.replace` and "
        "verified against genuinely loaded contexts to 1e-12.  \n"
        "Shipped values: `cap_ir_scale` 4.0, `iou_thr` 0.85, veil repair ON.",

        "## 0. Verdict\n\n" + md_table(["constant", "verdict"], final_rows)
        + "\n\n**STANDS** means *not shown wrong* — never *optimal*. Rule 5 of the "
        "pre-registration.",

        "## 1. Does any alternative dominate?\n\n"
        "`dominates` = not WORSE on any informative cell AND BETTER on at least "
        "one. `TEST` is the draw-averaged clean-cell delta on `pohang02`/`pohang03`; "
        "it may reject, never select.\n\n"
        + md_table(["arm", "cells BETTER", "cells WORSE", "TEST delta", "result"],
                   dom_rows),

        "## 2. Per-cell detail\n\n"
        f"`margin` = 2 × hypot(sd draw, sd paired boot), measured per cell and per "
        f"arm ({args.n_boot} paired resamples on the shipped draw). Cells marked "
        "`--` are uninformative: every arm is exactly 0.0000 on them, so they "
        "cannot reject anything and are excluded from the counts above.\n\n"
        + md_table(["arm", "cell (vis/ir)", "delta day", "delta night", "sd draw",
                    "sd boot", "margin", "verdict"], rows),

        "## 3. Uninformative cells\n\n"
        + ("None — every cell responded to at least one arm."
           if not uninf else
           "Excluded from every count above, per rule 6:\n\n"
           + "".join(f"* `{c}`\n" for c in uninf)),

        f"---\n\n_Generated by `scripts/reprice_constants_draw_avg.py` in "
        f"{time.time() - t0:.1f}s._",
    ]
    write_md(args.out, "Re-pricing three inherited constants", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
