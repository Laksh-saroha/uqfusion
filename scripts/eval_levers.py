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

Selection discipline is TIGHTENED, because the old one was weaker than it read.
`FIT_RUNS` excludes only pohang01 and pohang01 is entirely night, so "clean fit-run
day" and "clean day" are the SAME 1200 frames: every day constant in this project
has been selected and reported on one set. Arms are chosen here on `TUNE_RUNS`
(pohang00, 836 day frames) and the headline column is `TEST_RUNS` (pohang02 +
pohang03, 364 day frames) -- run-disjoint, so consecutive near-duplicate frames of
one transit cannot leak across it. The per-cell columns remain a report; an arm
that wins the tune runs and loses a cell is rejected, and the rejection is stated.

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
from uqfusion.eval.ctx import (NIGHT_RUNS, TEST_RUNS, TUNE_RUNS,               # noqa: E402
                               load_context, run_systems)
from uqfusion.eval.iralign import aligned_homographies                        # noqa: E402

#: The three day runs, kept separate so an arm can be required to help on ALL of
#: them. pohang00 is 836 day frames, pohang02 247, pohang03 117 -- so a mean over
#: frames is a mean over pohang00, and the worst-RUN delta is the statistic that
#: actually asks whether an arm generalises across scenes.
DAY_RUNS = ("pohang00", "pohang02", "pohang03")

SHIP = 0

#: (VIS condition, IR condition). The original eight collapse to four VIS
#: conditions x day/night; the corrupted-IR and IR-favouring cells are the ones
#: the extended grid added, and they are the only ones that can price cap_ratio.
CELLS = [("clean", None), ("fog", None), ("lowlight", None), ("glare", None),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None),
         ("clean", "fog_s2"), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("blur_s3", "glare_s2"), ("lowlight", "glare_s2")]


def arms(base, align_h=None):
    """(name, group, context) -- `base` is the adopted crossmodal context.

    The arm list is the survivors of two clean-cell sweeps, not a fresh grid:
    `runs/eval/support_26m.md` and `runs/eval/align_nomerge_26m.md` already
    eliminated `iou_thr` 0.55/0.65 (-0.0138 and worse), `consensus_beta` in both
    directions (-0.0084 on the held-out runs at beta=2), and `support_iou` 0.55
    (negative on the held-out runs in every one of six variants, while winning the
    tune runs -- which is exactly the failure the holdout exists to catch).
    """
    out = [("adopted", "—", base)]
    for r in (1.0, 2.0, 8.0, 16.0):
        out.append((f"cap_ratio x{r:g}", "cap_ratio",
                    replace(base, cap_ir=base.cap_ir * 4.0 / r)))   # base already /4
    for t in (0.75, 0.95):
        out.append((f"iou_thr {t}", "iou_thr", replace(base, iou_thr=t)))
    out.append(("consensus_distinct", "consensus", replace(base, consensus_distinct=True)))
    for iou, gam in ((0.10, 0.5), (0.30, 0.5), (0.30, 1.0)):
        out.append((f"support i{iou:g} g{gam:g}", "support",
                    replace(base, support_iou=iou, support_gamma=gam)))
    out.append(("class_veto", "class_veto", base))     # veto_keep_cls supplied by caller
    out.append(("class_veto + support i0.3 g0.5", "combined",
                replace(base, support_iou=0.30, support_gamma=0.5)))
    if align_h is not None:
        out.append(("align + iou0.95 + sup i0.3 g0.5", "align",
                    replace(base, h_frames=align_h, iou_thr=0.95,
                            support_iou=0.30, support_gamma=0.5)))
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
        tune = np.flatnonzero(np.isin(base.runs, TUNE_RUNS) & ~night)
        test = np.flatnonzero(np.isin(base.runs, TEST_RUNS) & ~night)

        def score(parts, sel):
            e = ap_from_parts([parts[i] for i in sel])
            ship = e["per_class"].get(SHIP)
            return (float(ship["ap50_95"]) if ship else 0.0, float(e["map50_95"]))

        align_h, _ = aligned_homographies(base.vis_by_cond[vis_cond], base.ir_clean,
                                          base.h_frames, radius=1.0)
        per_run = {r: np.flatnonzero((base.runs == r) & ~night) for r in DAY_RUNS}

        r0 = run_systems(base, vis_cond)
        p_vis = frame_parts(base.vis_by_cond[vis_cond], base.gts)
        p_ir = frame_parts(r0["ir_in_vis"], base.gts)
        res[("_bar", cname)] = {
            "day_ship": max(score(p_vis, day)[0], score(p_ir, day)[0]),
            "day_macro": max(score(p_vis, day)[1], score(p_ir, day)[1]),
            "tune_ship": max(score(p_vis, tune)[0], score(p_ir, tune)[0]),
            "test_ship": max(score(p_vis, test)[0], score(p_ir, test)[0]),
            "per_run": {r: max(score(p_vis, ix)[0], score(p_ir, ix)[0])
                        for r, ix in per_run.items()},
            "night_ship": max(score(p_vis, np.flatnonzero(night))[0],
                              score(p_ir, np.flatnonzero(night))[0]),
            "vis_day": score(p_vis, day)[0], "ir_day": score(p_ir, day)[0]}

        for name, group, ctx in arms(base, align_h):
            kw = ({"veto_keep_cls": keep}
                  if name.startswith("class_veto") else {})
            r = r0 if (name == "adopted") else run_systems(ctx, vis_cond, **kw)
            p = frame_parts(r["fused_gated"], ctx.gts)
            ds, dm = score(p, day)
            us, _ = score(p, tune)
            ts, tm = score(p, test)
            ns, nm = score(p, np.flatnonzero(night))
            res[(name, cname)] = {"group": group, "day_ship": ds, "day_macro": dm,
                                  "tune_ship": us, "test_ship": ts, "test_macro": tm,
                                  "night_ship": ns, "night_macro": nm,
                                  "per_run": {r: score(p, ix)[0] for r, ix in per_run.items()},
                                  "veto_vis": float(np.mean(np.asarray(r["veto_vis"])[~night]))}
        print(f"[lev] {cname} done ({time.time() - t0:.0f}s)", flush=True)

    # Names come from the same call the loop used, alignment arm included --
    # deriving them from a second, differently-parameterised call silently drops
    # any arm that only exists in one of them.
    arm_names = [a[0] for a in arms(load_context(
        preset="crossmodal", cache_dir=args.cache_dir, conditions=("clean",),
        verbose=False), align_h)]

    def gap(a, n, key="day_ship"):
        bar = res[("_bar", n)]["day_ship" if key.endswith("ship") else "day_macro"]
        return res[(a, n)][key] - bar

    L = [f"# Every remaining fusion lever — `{args.cache_dir}`", "",
         f"VIS capability {meta['cap_vis']:.4f}, IR {meta['cap_ir']:.4f} "
         f"(ratio {meta['cap_vis'] / max(meta['cap_ir'], 1e-9):.1f}x as the gate sees it). "
         f"Classes IR cannot supply: **{meta['veto_keep_cls'] or 'none'}**.", "",
         "Ship AP on day frames. `bar` = max(VIS, IR) on the same streams the system "
         "was given. Arms are selected on `clean/clean` **pohang00 day** frames "
         "(`tune`); `TEST` is pohang02+03 day, run-disjoint and held out. Every "
         "per-cell column is the full day set and therefore in-sample, exactly as "
         "in every earlier table -- which is why `TEST` is the column to read.", "",
         "| arm | " + " | ".join(names) + " | worst gap | tune | **TEST** |",
         "|---|" + "---:|" * (len(names) + 3)]
    L.append("| _bar_ | " + " | ".join(f"{res[('_bar', n)]['day_ship']:.4f}" for n in names)
             + " | — | " + f"{res[('_bar', names[0])]['tune_ship']:.4f} | "
             + f"{res[('_bar', names[0])]['test_ship']:.4f} |")
    rows = []
    for a in arm_names:
        gaps = [gap(a, n) for n in names]
        rows.append({"arm": a, "group": res[(a, names[0])]["group"],
                     "worst": float(min(gaps)), "mean": float(np.mean(gaps)),
                     "tune": res[(a, names[0])]["tune_ship"],
                     "test": res[(a, names[0])]["test_ship"],
                     "gaps": dict(zip(names, gaps)),
                     "macro_gaps": {n: gap(a, n, "day_macro") for n in names}})
        L.append(f"| {a} | " + " | ".join(f"{res[(a, n)]['day_ship']:.4f}" for n in names)
                 + f" | **{min(gaps):+.4f}** | {res[(a, names[0])]['tune_ship']:.4f}"
                 + f" | **{res[(a, names[0])]['test_ship']:.4f}** |")

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

    L += ["", "## Per-run consistency on `clean/clean` — leave-one-run-out", "",
          "Ship AP delta against adopted, on each day run separately. pohang00 is "
          "836 frames, pohang02 247, pohang03 117, so a frame-weighted day number "
          "is very nearly pohang00 alone. **worst run** is the statistic that asks "
          "whether an arm helps across scenes rather than fitting the biggest one.",
          "", "| arm | " + " | ".join(DAY_RUNS) + " | worst run |",
          "|---|" + "---:|" * (len(DAY_RUNS) + 1)]
    b_pr = res[("adopted", names[0])]["per_run"]
    for a in arm_names:
        pr = res[(a, names[0])]["per_run"]
        d = [pr[r] - b_pr[r] for r in DAY_RUNS]
        L.append(f"| {a} | " + " | ".join(f"{x:+.4f}" for x in d)
                 + f" | **{min(d):+.4f}** |")

    base_row = rows[0]
    L += ["", "## Verdict", "",
          f"- Adopted: tune {base_row['tune']:.4f}, held-out TEST "
          f"{base_row['test']:.4f}.", ""]
    for r in rows[1:]:
        lost = [n for n in names if r["gaps"][n] < base_row["gaps"][n] - 1e-9]
        won = [n for n in names if r["gaps"][n] > base_row["gaps"][n] + 1e-9]
        dt, dh = r["tune"] - base_row["tune"], r["test"] - base_row["test"]
        if dt > 1e-9 and dh <= 1e-9:
            verdict = "**reject — wins the tune runs and does not hold out**"
        elif dt > 1e-9 and lost:
            verdict = "**reject — wins the tune runs and loses a cell**"
        elif dt > 1e-9 and dh > 1e-9:
            verdict = "**adopt candidate**"
        else:
            verdict = "no better on the tune runs"
        L.append(f"- `{r['arm']}` (tune {dt:+.4f}, TEST {dh:+.4f}, worst cell "
                 f"{r['worst']:+.4f}): {verdict}; +{len(won)} cells / -{len(lost)} cells.")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"meta": meta, "cells": names, "rows": rows,
         "raw": {f"{a}|{n}": v for (a, n), v in res.items()}}, indent=2), encoding="utf-8")
    print(f"[lev] wrote {out} in {time.time() - t0:.0f}s")
    for r in rows:
        print(f"[lev] worst {r['worst']:+.4f}  tune {r['tune']:.4f}  "
              f"TEST {r['test']:.4f}  {r['arm']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
