"""`docs/prereg-night-veto-v3.md` — health as the sole authority, and the stop rule.

The rule under test, transcribed:

    veto_vis = ( ir_night OR (ir_night_raw AND NOT ir_ok) ) AND (q < thr)

The night signals decide only WHEN the question is asked; health alone decides
the answer. Injected as `veto_health_mode="sole_authority"`, which also folds the
weak fallback in rather than leaving it to fire beside the new clause — V2
measured that leaving it beside cost the verdict, vetoing a HEALTHY night VIS at
46-100% through `concentrated`, a VIS texture statistic IR corruption cannot move.

**The instrument and threshold are INHERITED**, not re-selected: `q_refit` at the
Youden point Stage 0 chose and Stage 1 used. There is no screening stage in V3.
Re-running selection after seeing V2's cells is the selection the two-stage
structure exists to prevent, so this script reads both from Stage 0's artefact
and refuses if they cannot be reproduced exactly.

ADOPT REQUIRES ALL THREE:
  (a) night delta >= +floor on the clean-VIS / clean-IR cell;
  (b) NO night cell below max(VIS_only, IR_only) - floor;
  (c) day false-veto stays at 0.0% on every clean-VIS cell, across every IR
      corruption arm. Hard requirement, because V3 collapses the VIS side of a
      two-vote safety property to one instrument.

REJECT if (a) fails by <= -floor. INCONCLUSIVE otherwise, and INCONCLUSIVE leaves
the veto standing. Day guard: every day cell within floor of V1, else VOID.

THE STOP RULE (registration section 6) is printed with the verdict, because it is
the part a reader is most likely to forget: **if this does not ADOPT, the axis is
closed** — no V4, no fourth clause, no re-selection, no moving the floor.

Usage:
    python scripts/v3_night_veto.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _ideas_common import fmt, md_table, sgn, write_md                     # noqa: E402
from stage0_night_veto_v2 import (BAR, CELLS as S0_CELLS, DRAWS,           # noqa: E402
                                  auroc, q_for, refit_vis_health)
from stage1_night_veto_v2 import CELLS, FLOOR_ABS, PRIMARY_CELL, youden    # noqa: E402
from uqfusion.eval.apmetrics import (ap_from_parts, bootstrap_delta,       # noqa: E402
                                     frame_parts)
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems        # noqa: E402

PREREG = "docs/prereg-night-veto-v3.md"
STAGE0 = ROOT / "runs/eval/night_veto_v2_stage0.json"
STAGE1 = ROOT / "runs/eval/night_veto_v2_stage1.json"
N_BOOT, BOOT_SEED = 2000, 0


def committed(rel: str) -> bool:
    try:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                                 cwd=ROOT, capture_output=True).returncode == 0
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel],
                               cwd=ROOT, capture_output=True).returncode != 0
        return tracked and not dirty
    except OSError:
        return False


def measure(cache_dir: Path, vh: dict, thr: float) -> dict:
    conds = sorted({c for c, _ in CELLS})
    ircs = sorted({i for _c, i in CELLS if i})
    q = {c: q_for(vh, c) for c in conds}
    ctxs = {ic: load_context(preset="crossmodal26m", cache_dir=str(cache_dir),
                             conditions=tuple(conds), ir_condition=ic, verbose=False)
            for ic in [None] + ircs}
    c0 = ctxs[None]
    night = np.flatnonzero(np.isin(c0.runs, NIGHT_RUNS))
    day = np.flatnonzero(~np.isin(c0.runs, NIGHT_RUNS))
    out = {"sel": {"day": day, "night": night}, "cells": {}}
    for vc, ic in CELLS:
        v1 = ctxs[ic]
        v3 = replace(v1, veto_health_by_cond=q, veto_health_thr=thr,
                     veto_health_mode="sole_authority")
        r1, r3 = run_systems(v1, vc), run_systems(v3, vc)
        p1 = frame_parts(r1["fused_gated"], v1.gts)
        p3 = frame_parts(r3["fused_gated"], v1.gts)
        cell = {"parts_v1": p1, "parts_v3": p3,
                "veto_v1_night": float(np.asarray(r1["veto_vis"], bool)[night].mean()),
                "veto_v3_night": float(np.asarray(r3["veto_vis"], bool)[night].mean()),
                "veto_v1_day": float(np.asarray(r1["veto_vis"], bool)[day].mean()),
                "veto_v3_day": float(np.asarray(r3["veto_vis"], bool)[day].mean())}
        for k, s in (("day", day), ("night", night)):
            cell[f"v1_{k}"] = ap_from_parts(p1, sel=s)["map50_95"]
            cell[f"v3_{k}"] = ap_from_parts(p3, sel=s)["map50_95"]
            cell[f"d_{k}"] = cell[f"v3_{k}"] - cell[f"v1_{k}"]
        cell["vis_night"] = ap_from_parts(
            frame_parts(v1.vis_by_cond[vc], v1.gts), sel=night)["map50_95"]
        cell["ir_night"] = ap_from_parts(
            frame_parts(r1["ir_in_vis"], v1.gts), sel=night)["map50_95"]
        out["cells"][(vc, ic)] = cell
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="runs/eval/night_veto_v3.md")
    args = ap.parse_args()

    if not committed(PREREG):
        print(f"[gate] {PREREG} not committed clean -- refusing to run.")
        return 1
    for f in (STAGE0, STAGE1):
        if not f.is_file():
            print(f"[gate] missing {f} -- V3 inherits from V2 and may not precede it.")
            return 1
    s0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    s1 = json.loads(STAGE1.read_text(encoding="utf-8"))
    winner = s0.get("winner")
    if s0.get("call") != "PROCEED" or winner != "q_refit":
        print(f"[gate] Stage 0 call={s0.get('call')} winner={winner!r} -- V3 "
              f"implements `q_refit` only and does not substitute an instrument.")
        return 1

    vh = refit_vis_health()
    c_probe = load_context(preset="crossmodal26m", cache_dir=DRAWS[0],
                           conditions=("clean",), verbose=False)
    night = np.flatnonzero(np.isin(c_probe.runs, NIGHT_RUNS))
    score = np.concatenate([q_for(vh, vc)[night] for vc, _i, _y in S0_CELLS])
    label = np.concatenate([np.full(len(night), y) for _v, _i, y in S0_CELLS])
    auc = auroc(score, label)
    thr, J, tpr, fpr = youden(score, label)
    if abs(auc - s0["auroc"][winner]) > 1e-6 or abs(thr - s1["threshold"]) > 1e-12:
        print(f"[gate] inherited instrument does not reproduce: AUROC {auc:.6f} vs "
              f"{s0['auroc'][winner]:.6f}, thr {thr!r} vs {s1['threshold']!r}. "
              f"Refusing -- V3 must run the SAME instrument V2 ran.")
        return 1
    print(f"[gate] prereg committed; inherited `{winner}` thr {thr:.6f} "
          f"(AUROC {auc:.4f}, J {J:.4f}, TPR {tpr:.4f}, FPR {fpr:.4f}) reproduced")

    draws = [measure(Path(d), vh, thr) for d in DRAWS]
    for d, dr in zip(DRAWS, draws):
        print(f"[measure] {Path(d).name}: clean-night "
              f"{dr['cells'][PRIMARY_CELL]['d_night']:+.4f}", flush=True)
    sel = draws[0]["sel"]

    cells = {}
    for key in CELLS:
        dn = [dr["cells"][key]["d_night"] for dr in draws]
        dd = [dr["cells"][key]["d_day"] for dr in draws]
        ses = [bootstrap_delta(dr["cells"][key]["parts_v3"],
                               dr["cells"][key]["parts_v1"], sel=sel["night"],
                               n_boot=args.boot, seed=BOOT_SEED)["se"] for dr in draws]
        se = float(np.mean(ses))
        m = lambda k: float(np.mean([dr["cells"][key][k] for dr in draws]))  # noqa: E731
        cells[key] = {
            "d_night": float(np.mean(dn)), "sd_night": float(np.std(dn, ddof=1)),
            "d_day_max": float(np.max(np.abs(dd))), "se": se,
            "floor": max(2 * se, FLOOR_ABS),
            "v1_night": m("v1_night"), "v3_night": m("v3_night"),
            "vis_night": m("vis_night"), "ir_night": m("ir_night"),
            "veto_v1_night": m("veto_v1_night"), "veto_v3_night": m("veto_v3_night"),
            "veto_v1_day": m("veto_v1_day"), "veto_v3_day": m("veto_v3_day"),
        }
        cells[key]["best_single"] = max(cells[key]["vis_night"], cells[key]["ir_night"])

    p = cells[PRIMARY_CELL]
    floor = p["floor"]
    clause_a = p["d_night"] >= floor
    breaches = [(k, v) for k, v in cells.items()
                if v["v3_night"] < v["best_single"] - v["floor"]]
    clause_b = not breaches
    # (c) clean-VIS cells only: the day frames whose VIS is healthy.
    cvis = [(k, v) for k, v in cells.items() if k[0] == "clean"]
    c_fail = [(k, v) for k, v in cvis if v["veto_v3_day"] > 0.0]
    clause_c = not c_fail
    day_break = [(k, v) for k, v in cells.items() if v["d_day_max"] > v["floor"]]

    if p["d_night"] <= -floor:
        verdict = "REJECT"
    elif clause_a and clause_b and clause_c:
        verdict = "ADOPT"
    else:
        verdict = "INCONCLUSIVE"
    final = "VOID" if day_break else verdict

    def nm(k):
        return f"`{k[0]}` / IR `{k[1] or 'clean'}`"

    stop = ("**Stop rule (registration §6): this axis is now CLOSED.** V3 did not "
            "ADOPT. The result is *no health-gated night rule clears the "
            "both-degraded bar on this detector*, and the veto stands. No V4, no "
            "fourth clause, no re-selection of the instrument, no moving the floor "
            "or clause (b). Registered before this number was known, precisely "
            "because each round so far removed exactly the obstacle the last one "
            "hit."
            if final != "ADOPT" else
            "**ADOPT.** All three clauses hold. The stop rule in registration §6 is "
            "not invoked; what follows is the re-baseline question, not another "
            "iteration of this one.")

    secs = [
        f"**Pre-registration:** [`{PREREG}`]({PREREG}). **Rule:** "
        f"`(ir_night OR (ir_night_raw AND NOT ir_ok)) AND (q < thr)` — health is "
        f"the sole authority, the night signals decide only when to ask, and the "
        f"weak fallback is folded in rather than left to fire beside it. "
        f"**Instrument inherited, not re-selected:** `{winner}` at the Youden "
        f"threshold {thr:.6f} from Stage 0 (AUROC {auc:.4f}), reproduced exactly "
        f"before this ran. Preset `crossmodal26m`, `iou_thr` 0.85, {len(DRAWS)} "
        f"paired draws, n_boot {args.boot}. Baseline is V1 as shipped.",
        "",
        f"## Verdict — **{final}**",
        "",
        f"    clause (a)  clean-night delta {sgn(p['d_night'])} vs floor {fmt(floor)}"
        f"    -> {'PASS' if clause_a else 'FAIL'}",
        f"    clause (b)  night cells below max(VIS, IR) - floor: {len(breaches)}"
        f"          -> {'PASS' if clause_b else 'FAIL'}",
        f"    clause (c)  clean-VIS cells with a non-zero day veto: {len(c_fail)}"
        f"      -> {'PASS' if clause_c else 'FAIL'}",
        f"    day guard   cells beyond floor: {len(day_break)}"
        f"                    -> {'PASS' if not day_break else 'VOID'}",
        "",
        f"V1 (shipped) {fmt(p['v1_night'])} -> V3 {fmt(p['v3_night'])} on the clean "
        f"night cell; VIS-only {fmt(p['vis_night'])}, IR-only {fmt(p['ir_night'])}.",
        "",
        stop,
        "",
        "## Every cell",
        "",
        md_table(["cell", "V1 night", "V3 night", "delta", "sd", "floor",
                  "VIS only", "IR only", "max(VIS,IR)", "clause (b)",
                  "veto night V1 / V3", "veto day V1 / V3", "day |delta| max"],
                 [[nm(k), fmt(v["v1_night"]), fmt(v["v3_night"]), sgn(v["d_night"]),
                   fmt(v["sd_night"], 5), fmt(v["floor"], 5), fmt(v["vis_night"]),
                   fmt(v["ir_night"]), fmt(v["best_single"]),
                   "**below**" if v["v3_night"] < v["best_single"] - v["floor"] else "ok",
                   f"{v['veto_v1_night']:.0%} / {v['veto_v3_night']:.0%}",
                   f"{v['veto_v1_day']:.1%} / {v['veto_v3_day']:.1%}",
                   fmt(v["d_day_max"], 6)]
                  for k, v in cells.items()]),
        "",
        "## Comparison with V2, which is the point of V3",
        "",
        md_table(["cell", "V1 night", "V2 night", "V3 night", "VIS only"],
                 [[nm(k), fmt(v["v1_night"]),
                   fmt(s1["cells"][f"{k[0]}|{k[1] or 'clean'}"]["v2_night"]),
                   fmt(v["v3_night"]), fmt(v["vis_night"])]
                  for k, v in cells.items()]),
        "",
        "## Caveats carried from the registration",
        "",
        "1. **V3 removes a vote.** The shipped safety story is two independent "
        "votes plus the authority bound; V3 collapses the VIS side to one "
        "instrument. Clause (c) is what verifies the day protection survives, and "
        "on day frames with a corrupted IR the authority bound already disarms IR "
        "on 82.1–100.0% of them — so that protection now rests on `ir_night_raw` "
        "with health as the second gate rather than the only one.",
        "2. **The instrument is saturated.** `q = clip(bound/d2, 0, 1)` pins to "
        "exactly 1.0 for 100% of clean-night frames and 1.59% of degraded-night "
        "ones. It is a binary novelty-bound test with no margin to degrade "
        "gracefully — right or wrong, per frame.",
        "3. **Stage 0 selected `q_refit` on these same 1,032 night frames.** "
        "`pohang01` is the only night run and no split can create a second one.",
        "4. **One seed, early-stopped**, and a **synthetic** negative class.",
        "5. **Fusion at 0.85 is ~99.9% concatenation** — any gain is union recall.",
        "6. **This deploys nothing.** Under the shipped `gauss_vis_seed0` VIS "
        "scores 0.0000 at night. V3 is meaningful only with the re-baseline in "
        "`docs/rebaseline-proposal-2026-09-04.md`.",
    ]
    out = write_md(ROOT / args.out, "Night veto V3 — health as sole authority", secs)
    js = out.with_suffix(".json")
    js.write_text(json.dumps({
        "instrument": winner, "threshold": thr, "verdict": final,
        "clause_a": bool(clause_a), "clause_b": bool(clause_b),
        "clause_c": bool(clause_c), "axis_closed": final != "ADOPT",
        "cells": {f"{k[0]}|{k[1] or 'clean'}": v for k, v in cells.items()},
    }, indent=1), encoding="utf-8")
    print(f"\n[v3] {final}   clean-night {sgn(p['d_night'])}  floor {fmt(floor)}")
    print(f"[v3] clauses a={clause_a} b={clause_b} c={clause_c}")
    if final != "ADOPT":
        print("[v3] STOP RULE: axis closed, the veto stands.")
    print(f"[out] {out}\n[out] {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
