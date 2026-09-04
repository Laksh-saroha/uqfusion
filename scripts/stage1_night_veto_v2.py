"""Stage 1 of `docs/prereg-night-veto-v2.md` — the gate.

Runs ONLY if Stage 0 said PROCEED, and only on the instrument Stage 0 selected.
Every band, floor and guard below is transcribed from the registration.

THE RULE UNDER TEST, literally:

    veto_vis  =  ir_night  AND  (health < thr)

replacing `crossmodal26m`'s `night & (dark | veil)`. It is injected through
`FusionContext.veto_health_by_cond`, a field that is EMPTY by default so no
existing caller and no published number moves. The weak fallback underneath is
deliberately untouched — the registration replaces the night arm and nothing
else.

THE THRESHOLD is the Youden point of the winning instrument's ROC, read off the
curve and not tuned against mAP. The ROC is recomputed here rather than
deserialised because the refit is deterministic: same input file, same fit set,
same helper, so the curve is identical to Stage 0's by construction, and a
recomputation that disagreed would itself be the finding.

ADOPT REQUIRES BOTH:
  (a) draw-averaged night delta >= +floor on the clean-VIS / clean-IR cell;
  (b) NO night cell below max(VIS_only, IR_only) - floor.
(b) is V1's guard 2 generalised, and it is the clause V1 failed. A rule that wins
on average and loses when a sensor degrades is not an improvement.

REJECT if (a) fails by <= -floor. INCONCLUSIVE otherwise, and INCONCLUSIVE
leaves the veto standing: the default is shipped behaviour.

Nothing under `runs/cache*/`, `runs/derived/` or `runs/eval/` is overwritten;
`write_md` refuses an existing --out.

Usage:
    python scripts/stage1_night_veto_v2.py
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

from _ideas_common import fmt, md_table, sgn, write_md                    # noqa: E402
from stage0_night_veto_v2 import (BAR, CELLS as S0_CELLS, DRAWS,          # noqa: E402
                                  auroc, q_for, refit_vis_health)
from uqfusion.eval.apmetrics import (ap_from_parts, bootstrap_delta,      # noqa: E402
                                     frame_parts)
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems       # noqa: E402

PREREG = "docs/prereg-night-veto-v2.md"
STAGE0 = ROOT / "runs/eval/night_veto_v2_stage0.json"

#: All 11 V1 cells. Stage 0 scored 10 (it drops the unbanded `noise_s2`); Stage 1
#: scores every one, because clause (b) is a claim about EVERY night cell.
CELLS = [("clean", None), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("clean", "fog_s2"),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None), ("fog", None),
         ("lowlight", "glare_s2"), ("blur_s3", "glare_s2")]
PRIMARY_CELL = ("clean", None)
FLOOR_ABS = 0.002
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


def youden(score: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Threshold maximising TPR - FPR. Returns (thr, J, tpr, fpr).

    Ties are resolved toward the LOWER threshold, i.e. toward vetoing less. The
    asymmetry is measured, not assumed: V1 put the cost of vetoing a healthy
    night VIS at -0.1785 and the cost of failing to veto a fogged one at -0.0737,
    so the expensive error is over-vetoing and a tie should fall away from it.
    """
    order = np.argsort(-score, kind="mergesort")
    s, yy = score[order], y[order]
    p, n = float((y == 1).sum()), float((y == 0).sum())
    tp = np.cumsum(yy == 1) / p
    fp = np.cumsum(yy == 0) / n
    j = tp - fp
    k = int(np.argmax(j))
    return float(s[k]), float(j[k]), float(tp[k]), float(fp[k])


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
        v2 = replace(v1, veto_health_by_cond=q, veto_health_thr=thr)
        r1, r2 = run_systems(v1, vc), run_systems(v2, vc)
        p1 = frame_parts(r1["fused_gated"], v1.gts)
        p2 = frame_parts(r2["fused_gated"], v1.gts)
        p_vis = frame_parts(v1.vis_by_cond[vc], v1.gts)
        p_ir = frame_parts(r1["ir_in_vis"], v1.gts)
        cell = {"parts_v1": p1, "parts_v2": p2,
                "veto_v1_night": float(np.asarray(r1["veto_vis"], bool)[night].mean()),
                "veto_v2_night": float(np.asarray(r2["veto_vis"], bool)[night].mean()),
                "veto_v2_day": float(np.asarray(r2["veto_vis"], bool)[day].mean())}
        for k, s in (("day", day), ("night", night)):
            cell[f"v1_{k}"] = ap_from_parts(p1, sel=s)["map50_95"]
            cell[f"v2_{k}"] = ap_from_parts(p2, sel=s)["map50_95"]
            cell[f"d_{k}"] = cell[f"v2_{k}"] - cell[f"v1_{k}"]
        cell["vis_night"] = ap_from_parts(p_vis, sel=night)["map50_95"]
        cell["ir_night"] = ap_from_parts(p_ir, sel=night)["map50_95"]
        out["cells"][(vc, ic)] = cell
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="runs/eval/night_veto_v2_stage1.md")
    args = ap.parse_args()

    if not committed(PREREG):
        print(f"[gate] {PREREG} not committed clean -- refusing to run.")
        return 1
    if not STAGE0.is_file():
        print(f"[gate] missing {STAGE0} -- Stage 1 may not run before Stage 0.")
        return 1
    s0 = json.loads(STAGE0.read_text(encoding="utf-8"))
    if s0.get("call") != "PROCEED" or not s0.get("winner"):
        print(f"[gate] Stage 0 said {s0.get('call')} -- Stage 1 does not run.")
        return 1
    winner = s0["winner"]
    if winner != "q_refit":
        print(f"[gate] this script implements `q_refit` only; Stage 0 chose "
              f"{winner!r}. Refusing rather than substituting an instrument.")
        return 1
    print(f"[gate] prereg committed; Stage 0 PROCEED with `{winner}` "
          f"(AUROC {s0['auroc'][winner]:.4f} >= {BAR})")

    # --- the threshold, from the ROC and nothing else -----------------------
    vh = refit_vis_health()
    qs, ys = [], []
    for vc, _ic, y in S0_CELLS:
        qq = q_for(vh, vc)
        qs.append(qq)
        ys.append(y)
    # night selection needs a context; take it from the first draw
    c_probe = load_context(preset="crossmodal26m", cache_dir=DRAWS[0],
                           conditions=("clean",), verbose=False)
    night = np.flatnonzero(np.isin(c_probe.runs, NIGHT_RUNS))
    score = np.concatenate([q[night] for q in qs])
    label = np.concatenate([np.full(len(night), y) for y in ys])
    auc_check = auroc(score, label)
    thr, J, tpr, fpr = youden(score, label)
    print(f"[roc] AUROC {auc_check:.4f} (Stage 0 recorded "
          f"{s0['auroc'][winner]:.4f})   Youden thr {thr:.6f}  J {J:.4f}  "
          f"TPR {tpr:.4f}  FPR {fpr:.4f}")
    if abs(auc_check - s0["auroc"][winner]) > 1e-6:
        print("[gate] recomputed ROC disagrees with Stage 0 -- refusing. The "
              "refit is deterministic, so a disagreement is a defect, not noise.")
        return 1

    draws = [measure(Path(d), vh, thr) for d in DRAWS]
    for d, dr in zip(DRAWS, draws):
        print(f"[measure] {Path(d).name}: clean-night delta "
              f"{dr['cells'][PRIMARY_CELL]['d_night']:+.4f}", flush=True)
    sel = draws[0]["sel"]

    cells = {}
    for key in CELLS:
        dn = [dr["cells"][key]["d_night"] for dr in draws]
        dd = [dr["cells"][key]["d_day"] for dr in draws]
        ses = [bootstrap_delta(dr["cells"][key]["parts_v2"],
                               dr["cells"][key]["parts_v1"], sel=sel["night"],
                               n_boot=args.boot, seed=BOOT_SEED)["se"] for dr in draws]
        se = float(np.mean(ses))
        m = lambda k: float(np.mean([dr["cells"][key][k] for dr in draws]))  # noqa: E731
        cells[key] = {
            "d_night": float(np.mean(dn)), "sd_night": float(np.std(dn, ddof=1)),
            "d_day_max": float(np.max(np.abs(dd))), "se": se,
            "floor": max(2 * se, FLOOR_ABS),
            "v1_night": m("v1_night"), "v2_night": m("v2_night"),
            "vis_night": m("vis_night"), "ir_night": m("ir_night"),
            "veto_v1_night": m("veto_v1_night"), "veto_v2_night": m("veto_v2_night"),
            "veto_v2_day": m("veto_v2_day"),
        }
        cells[key]["best_single"] = max(cells[key]["vis_night"], cells[key]["ir_night"])

    p = cells[PRIMARY_CELL]
    floor = p["floor"]
    clause_a = p["d_night"] >= floor
    breaches = [(k, v) for k, v in cells.items()
                if v["v2_night"] < v["best_single"] - v["floor"]]
    clause_b = not breaches
    day_break = [(k, v) for k, v in cells.items() if v["d_day_max"] > v["floor"]]

    if p["d_night"] <= -floor:
        verdict = "REJECT"
    elif clause_a and clause_b:
        verdict = "ADOPT"
    else:
        verdict = "INCONCLUSIVE"
    final = "VOID" if day_break else verdict

    def nm(k):
        return f"`{k[0]}` / IR `{k[1] or 'clean'}`"

    secs = [
        f"**Pre-registration:** [`{PREREG}`]({PREREG}). **Instrument:** `{winner}`, "
        f"selected by Stage 0 at AUROC {s0['auroc'][winner]:.4f} against a bar of "
        f"{BAR}. **Threshold:** Youden point {thr:.6f} (J {J:.4f}, TPR {tpr:.4f}, "
        f"FPR {fpr:.4f}), read off the ROC and not tuned against mAP. Preset "
        f"`crossmodal26m`, `iou_thr` 0.85, {len(DRAWS)} paired draws, n_boot "
        f"{args.boot}. Arms differ in one clause: `night & (dark | veil)` becomes "
        f"`night & (q < thr)`.",
        "",
        f"## Verdict — **{final}**",
        "",
        f"    clause (a)  clean-night delta {sgn(p['d_night'])}  vs floor "
        f"{fmt(floor)}   -> {'PASS' if clause_a else 'FAIL'}",
        f"    clause (b)  night cells below max(VIS, IR) - floor: {len(breaches)}"
        f"   -> {'PASS' if clause_b else 'FAIL'}",
        f"    day guard   cells beyond floor: {len(day_break)}"
        f"   -> {'PASS' if not day_break else 'VOID'}",
        "",
        f"V1 (shipped) {fmt(p['v1_night'])}  ->  V2 {fmt(p['v2_night'])} on the "
        f"clean night cell; VIS-only {fmt(p['vis_night'])}, IR-only "
        f"{fmt(p['ir_night'])}.",
        "",
        ("**ADOPT.** Both clauses hold. This is the outcome V1 could not reach: the "
         "gain on a healthy night VIS is kept AND no night cell falls below its own "
         "best single stream." if final == "ADOPT" else
         "**INCONCLUSIVE leaves the veto standing.** The registration is explicit "
         "that the default is shipped behaviour and that an unresolved measurement "
         "is not a licence to change it." if final == "INCONCLUSIVE" else
         "**REJECT.** The health-gated rule loses on the cell it was built to win."
         if final == "REJECT" else
         "**VOID.** A day cell moved beyond its floor, so the rule escaped the "
         "night path and the run is not interpreted."),
        "",
        "## Every cell",
        "",
        md_table(["cell", "V1 night", "V2 night", "delta", "sd", "floor",
                  "VIS only", "IR only", "max(VIS,IR)", "clause (b)",
                  "veto fires V1 / V2 (night)", "day |delta| max"],
                 [[nm(k), fmt(v["v1_night"]), fmt(v["v2_night"]), sgn(v["d_night"]),
                   fmt(v["sd_night"], 5), fmt(v["floor"], 5), fmt(v["vis_night"]),
                   fmt(v["ir_night"]), fmt(v["best_single"]),
                   "**below**" if v["v2_night"] < v["best_single"] - v["floor"] else "ok",
                   f"{v['veto_v1_night']:.0%} / {v['veto_v2_night']:.0%}",
                   fmt(v["d_day_max"], 6)]
                  for k, v in cells.items()]),
        "",
        "## What the rule actually does",
        "",
        "V1 vetoes VIS on 100% of night frames in every cell. V2 vetoes only where "
        "the health instrument says the VIS stream is degraded, so the veto rate "
        "becomes a property of the cell rather than of the clock. The column above "
        "is the whole mechanism.",
        "",
        "## Caveats carried from the registration",
        "",
        "1. **Stage 0 selected on these same 1,032 night frames.** `pohang01` is the "
        "only night run and no split can create a second one. Cell-level labels, a "
        "bar fixed in advance and a registered abandonment outcome were the only "
        "defences available, and this paragraph travels with any ADOPT.",
        "2. **The instrument's negative class was four synthetic corruptions.** It "
        "may not separate real degradation.",
        "3. **One seed, early-stopped.** `gauss_vis_nightfull` is seed 0 and stopped "
        "at epoch 11 of 100 with the LR still annealing.",
        "4. **Fusion at 0.85 is ~99.9% concatenation** — 0.143% of VIS boxes had an "
        "IR partner on the clean night cell in V1. Any gain is union recall.",
        "5. **This deploys nothing.** Under the shipped `gauss_vis_seed0`, VIS "
        "scores 0.0000 at night and V2 is a longer route to the same answer. V2 is "
        "only meaningful with the re-baseline priced in "
        "`docs/rebaseline-proposal-2026-09-04.md`, whose night-capable VIS ensemble "
        "does not exist and costs ~40 h.",
    ]
    out = write_md(ROOT / args.out, "Night veto V2 — Stage 1 verdict", secs)
    js = out.with_suffix(".json")
    js.write_text(json.dumps({
        "instrument": winner, "threshold": thr, "youden_J": J,
        "verdict": final, "verdict_before_day_guard": verdict,
        "clause_a": bool(clause_a), "clause_b": bool(clause_b),
        "cells": {f"{k[0]}|{k[1] or 'clean'}": v for k, v in cells.items()},
    }, indent=1), encoding="utf-8")
    print(f"\n[stage1] {final}   clean-night {sgn(p['d_night'])}  floor {fmt(floor)}")
    print(f"[out] {out}\n[out] {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
