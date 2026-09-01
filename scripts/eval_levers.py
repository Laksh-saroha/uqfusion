"""Every lever the fusion layer still has, measured on one detector, one table.

The crossmodal record closed the veto surface -- the oracle veto is within +0.0007
of the adopted rule on every cell -- and concluded the remaining headroom was in
the detector. This measures what is left ON TOP of a detector swap, so the two are
not confused with each other:

  cap_ratio        the one adopted constant that is selected rather than derived.
                   It has to be re-priced whenever the detectors change, because
                   what it encodes is the VIS/IR capability gap, and that gap is a
                   property of the checkpoints.

  iou_thr          0.85 was tuned on yolo26s boxes. A different detector localises
                   differently, and this threshold decides what counts as the two
                   sensors agreeing -- so it is not a detector-independent constant
                   even though it has been treated as one.

  ir_nms           adopted at 0.7 against an IR model that emitted duplicates. The
                   full-scale IR model is nc=1 and p2feat; whether it still needs
                   deduplicating is a question about that checkpoint, not a setting.

  consensus_beta   NEW. WBF scores a cluster both sensors saw at 2x one only a
                   single sensor saw, and that factor is where this system's entire
                   day-cell gain comes from: on a clean day frame IR contributes
                   almost no detections, yet fusion still beats VIS alone, because
                   agreement re-ranks VIS's own boxes. Nobody chose 2. It is what
                   `ensemble_boxes` does.

  support_iou/gamma   NEW, and the one the geometry actually points at. At
                   `iou_thr` 0.85 only 0.05% of VIS boxes have an IR partner, so
                   cross-modal agreement is not something WBF is under-using --
                   it is something WBF cannot see at that threshold. At IoU 0.30 a
                   third of VIS boxes do have one. `support` boosts a box's score
                   on that looser overlap and never moves its coordinates, which is
                   the split a 3-6 px registration residual forces: IR knows
                   WHETHER a target is there, not WHERE its edges are.

  consensus_distinct  NEW. WBF counts cluster MEMBERS, so two overlapping boxes
                   from ONE sensor collect the same bonus as a genuine cross-modal
                   confirmation. That prices self-agreement as agreement, and
                   rewards a stream for being duplicative.

  class_veto       NEW, and the only arm here that ship AP cannot see. IR is nc=1
                   (D28/A-1). A VIS veto therefore deletes the only stream that
                   ever produced a BUOY box, so buoy AP is zero on every vetoed
                   frame -- the frame is not handed to the better sensor, it is
                   handed to no sensor. Every table in this project reports ship
                   AP, which is why this has never appeared. Reported on macro.

Selection discipline is unchanged: arms are chosen on CLEAN FIT-RUN DAY frames,
and the per-cell columns are a report. An arm that wins the selection set and
loses a cell is rejected -- but the rejection is stated, not silent.

Usage:
    python scripts/eval_levers.py --cache-dir runs/cache_m --out runs/eval/levers_26m.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts            # noqa: E402
from uqfusion.eval.ctx import FIT_RUNS, NIGHT_RUNS, load_context, run_systems  # noqa: E402

SHIP = 0

#: (VIS condition, IR condition). The original eight collapse to four VIS
#: conditions x day/night; the corrupted-IR and IR-favouring cells are the ones
#: the extended grid added, and they are the only ones that can price cap_ratio.
CELLS = [("clean", None), ("fog", None), ("lowlight", None), ("glare", None),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None),
         ("clean", "fog_s2"), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("blur_s3", "glare_s2"), ("lowlight", "glare_s2")]


def arms(base):
    """(name, group, context) -- `base` is the adopted crossmodal context."""
    out = [("adopted", "—", base)]
    for r in (1.0, 2.0, 8.0, 16.0):
        out.append((f"cap_ratio x{r:g}", "cap_ratio",
                    replace(base, cap_ir=base.cap_ir * 4.0 / r)))   # base already /4
    for t in (0.55, 0.65, 0.75, 0.95):
        out.append((f"iou_thr {t}", "iou_thr", replace(base, iou_thr=t)))
    for b in (0.0, 2.0):
        out.append((f"consensus_beta {b:g}", "consensus", replace(base, consensus_beta=b)))
    out.append(("consensus_distinct", "consensus", replace(base, consensus_distinct=True)))
    for iou, gam in ((0.10, 0.5), (0.10, 1.0), (0.30, 0.5), (0.30, 1.0), (0.50, 1.0)):
        out.append((f"support iou{iou:g} g{gam:g}", "support",
                    replace(base, support_iou=iou, support_gamma=gam)))
    out.append(("class_veto", "class_veto", base))          # veto_keep_cls filled in by caller
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache_m",
                    help="runs/cache = yolo26s (phase2); runs/cache_m = the full-scale "
                         "yolo26m / yolo26m-p2feat detectors the architecture specifies.")
    ap.add_argument("--out", default="runs/eval/levers_26m.md")
    ap.add_argument("--cells", type=int, default=0, help="cap cell count (debug)")
    args = ap.parse_args()
    t0 = time.time()

    cells = CELLS[: args.cells] if args.cells else CELLS
    names = [f"{v}/{i or 'clean'}" for v, i in cells]
    res: dict[tuple[str, str], dict] = {}
    meta: dict = {}

    for vis_cond, ir_cond in cells:
        cname = f"{vis_cond}/{ir_cond or 'clean'}"
        base = load_context(preset="crossmodal", cache_dir=args.cache_dir,
                            conditions=(vis_cond,), ir_condition=ir_cond,
                            verbose=(cname == names[0]))
        # The class-selective arm needs the exempt set derived from THESE caches.
        keep = load_context(preset="crossmodal", cache_dir=args.cache_dir,
                            conditions=(vis_cond,), ir_condition=ir_cond,
                            veto_keep_cls="auto", verbose=False).veto_keep_cls
        meta.setdefault("veto_keep_cls", list(keep))
        meta.setdefault("cap_vis", base.cap_vis)
        meta.setdefault("cap_ir", base.cap_ir)

        night = np.isin(base.runs, NIGHT_RUNS)
        day = np.flatnonzero(~night)
        fit_day = np.flatnonzero(np.isin(base.runs, FIT_RUNS) & ~night)

        def score(parts, sel):
            e = ap_from_parts([parts[i] for i in sel])
            ship = e["per_class"].get(SHIP)
            return (float(ship["ap50_95"]) if ship else 0.0, float(e["map50_95"]))

        r0 = run_systems(base, vis_cond)
        p_vis = frame_parts(base.vis_by_cond[vis_cond], base.gts)
        p_ir = frame_parts(r0["ir_in_vis"], base.gts)
        res[("_bar", cname)] = {
            "day_ship": max(score(p_vis, day)[0], score(p_ir, day)[0]),
            "day_macro": max(score(p_vis, day)[1], score(p_ir, day)[1]),
            "fit_ship": max(score(p_vis, fit_day)[0], score(p_ir, fit_day)[0]),
            "night_ship": max(score(p_vis, np.flatnonzero(night))[0],
                              score(p_ir, np.flatnonzero(night))[0]),
            "vis_day": score(p_vis, day)[0], "ir_day": score(p_ir, day)[0]}

        for name, group, ctx in arms(base):
            kw = {"veto_keep_cls": keep} if name == "class_veto" else {}
            r = r0 if (name == "adopted") else run_systems(ctx, vis_cond, **kw)
            p = frame_parts(r["fused_gated"], ctx.gts)
            ds, dm = score(p, day)
            fs, _ = score(p, fit_day)
            ns, nm = score(p, np.flatnonzero(night))
            res[(name, cname)] = {"group": group, "day_ship": ds, "day_macro": dm,
                                  "fit_ship": fs, "night_ship": ns, "night_macro": nm,
                                  "veto_vis": float(np.mean(np.asarray(r["veto_vis"])[~night]))}
        print(f"[lev] {cname} done ({time.time() - t0:.0f}s)", flush=True)

    arm_names = [a[0] for a in arms(load_context(preset="crossmodal",
                                                 cache_dir=args.cache_dir,
                                                 conditions=("clean",), verbose=False))]

    def gap(a, n, key="day_ship"):
        bar = res[("_bar", n)]["day_ship" if key.endswith("ship") else "day_macro"]
        return res[(a, n)][key] - bar

    L = [f"# Every remaining fusion lever — `{args.cache_dir}`", "",
         f"VIS capability {meta['cap_vis']:.4f}, IR {meta['cap_ir']:.4f} "
         f"(ratio {meta['cap_vis'] / max(meta['cap_ir'], 1e-9):.1f}x as the gate sees it). "
         f"Classes IR cannot supply: **{meta['veto_keep_cls'] or 'none'}**.", "",
         "Ship AP on day frames. `bar` = max(VIS, IR) on the same streams the system "
         "was given. Arms are selected on `clean/clean` **fit-run day** frames only; "
         "every other column is a report.", "",
         "| arm | " + " | ".join(names) + " | worst gap | fit-set |",
         "|---|" + "---:|" * (len(names) + 2)]
    L.append("| _bar_ | " + " | ".join(f"{res[('_bar', n)]['day_ship']:.4f}" for n in names)
             + " | — | " + f"{res[('_bar', names[0])]['fit_ship']:.4f} |")
    rows = []
    for a in arm_names:
        gaps = [gap(a, n) for n in names]
        rows.append({"arm": a, "group": res[(a, names[0])]["group"],
                     "worst": float(min(gaps)), "mean": float(np.mean(gaps)),
                     "fit": res[(a, names[0])]["fit_ship"],
                     "gaps": dict(zip(names, gaps)),
                     "macro_gaps": {n: gap(a, n, "day_macro") for n in names}})
        L.append(f"| {a} | " + " | ".join(f"{res[(a, n)]['day_ship']:.4f}" for n in names)
                 + f" | **{min(gaps):+.4f}** | {res[(a, names[0])]['fit_ship']:.4f} |")

    L += ["", "## Gap to bar, per cell (ship AP, day)", "",
          "| arm | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for r in rows:
        L.append(f"| {r['arm']} | " + " | ".join(f"{r['gaps'][n]:+.4f}" for n in names) + " |")

    L += ["", "## Macro mAP, day — where the class-selective veto lives", "",
          "Ship AP is blind to this column by construction. A VIS veto with IR at "
          "`nc=1` removes every buoy box on the frame.", "",
          "| arm | " + " | ".join(names) + " |", "|---|" + "---:|" * len(names)]
    for r in rows:
        L.append(f"| {r['arm']} | "
                 + " | ".join(f"{res[(r['arm'], n)]['day_macro']:.4f}" for n in names) + " |")

    base_row = rows[0]
    L += ["", "## Verdict", "",
          f"- Selection set (`clean/clean`, fit runs, day): adopted "
          f"{base_row['fit']:.4f}.", ""]
    for r in rows[1:]:
        lost = [n for n in names if r["gaps"][n] < base_row["gaps"][n] - 1e-9]
        won = [n for n in names if r["gaps"][n] > base_row["gaps"][n] + 1e-9]
        verdict = ("**reject** — wins the selection set and loses a cell"
                   if r["fit"] > base_row["fit"] + 1e-9 and lost else
                   "**adopt candidate**" if r["worst"] > base_row["worst"] + 1e-9 else
                   "no better")
        L.append(f"- `{r['arm']}` (fit {r['fit']:+.4f} vs adopted, worst cell "
                 f"{r['worst']:+.4f}): {verdict}; +{len(won)} cells / -{len(lost)} cells.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"meta": meta, "cells": names, "rows": rows,
         "raw": {f"{a}|{n}": v for (a, n), v in res.items()}}, indent=2), encoding="utf-8")
    print(f"[lev] wrote {out} in {time.time() - t0:.0f}s")
    for r in rows:
        print(f"[lev] worst {r['worst']:+.4f}  mean {r['mean']:+.4f}  fit {r['fit']:.4f}  {r['arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
