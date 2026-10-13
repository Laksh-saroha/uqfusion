"""Two degraded-IR repairs to `crossmodal26m`, registered, development (2026-10-10).

Pre-registration: `docs/prereg-degraded-ir-repairs-2026-10-10.md`. R1 drops `dark AND veil` from the
weak-IR fallback's confirmation; R2 withholds the cross-modal support bonus where IR is not
merge-healthy. At registration R1+R2 was `preset="crossmodal26m_r"`; the verdict (R1 adopted, R2 and
R1+R2 rejected) made the preset R1 alone. Each arm is toggled here directly on the same loaded
`crossmodal26m` context, so the arms differ only in the two switches.

Verdict on VIS draws 942-944 (IR 952-954); draw 941 is reported beside it, descriptively, because R1
was chosen after seeing it. Five Phase 3 systems, 60 cells, ship AP; per system the draw mean, then
Δ = arm - shipped paired by system with a 95% t-interval (df 4). Scores no pohang04 frame.

    py -3.13 scripts/v2_degraded_ir_repairs.py
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

from uqfusion.eval.apmetrics import frame_parts                    # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems  # noqa: E402
from v2_sensitivity import CACHE, PRESET, SEEDS, STATS, T_975_DF4, ship_ap, sub  # noqa: E402

VIS_CONDS = ("clean", "fog", "fog_s1", "glare", "lowlight")
IR_CONDS = (None, "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")
VERDICT_DRAWS, DESC_DRAWS = (942, 943, 944), (941,)
ARMS = {"shipped": (True, False), "R1": (False, False), "R2": (True, True), "R1+R2": (False, True)}
DECIDE, FLOORS, BENEFIT = 0.0014, (0.0008, 0.0014, 0.0024, 0.0060), 0.0060
CLAIM_FLOOR = 0.0060



def wait_ready(k: int, draw: int, ic, timeout: float = 7200.0) -> None:
    """Block until the IR cache and its gate statistics for this cell exist and stopped growing
    (lets scoring start while the last caches of a build are still being written)."""
    import time as _t
    if ic is None:
        return
    tag = f"gauss_ir_paired_{ic}"
    paths = [ROOT / CACHE / f"seed{k}" / sub(draw) / f"{tag}.pkl",
             ROOT / STATS / "brightness" / sub(draw) / f"{tag}.json",
             ROOT / STATS / "structure" / sub(draw) / f"{tag}.json"]
    t0, last = _t.time(), None
    while True:
        if all(p.is_file() for p in paths):
            sizes = [p.stat().st_size for p in paths]
            if sizes == last:
                return
            last = sizes
        if _t.time() - t0 > timeout:
            raise SystemExit(f"timed out waiting for {paths}")
        _t.sleep(10)

def score(job):
    k, draw, ir_cond = job
    wait_ready(k, draw, ir_cond)
    d = f"{STATS}/brightness/{sub(draw)}"
    ctx = load_context(preset=PRESET, cache_dir=f"{CACHE}/seed{k}/{sub(draw)}", bright_dir=d,
                       structure_dir=f"{STATS}/structure/{sub(draw)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json",
                       conditions=VIS_CONDS, ir_condition=ir_cond, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    out = {}
    for arm, (fb_veil, sup_health) in ARMS.items():
        ctx.fallback_veil, ctx.support_requires_ir_health = fb_veil, sup_health
        for c in VIS_CONDS:
            r = run_systems(ctx, c, compute_metrics=False)   # scored below from the per-frame outputs
            parts = {"fused": frame_parts(r["fused_gated"], ctx.gts)}
            if arm == "shipped":
                parts["vis"] = frame_parts(ctx.vis_by_cond[c], ctx.gts)
                parts["ir"] = frame_parts(r["ir_in_vis"], ctx.gts)
            vv = np.asarray(r["veto_vis"], dtype=bool)
            for s, idx in sl.items():
                for a, p in parts.items():
                    out[(arm, c, ir_cond, s, a)] = ship_ap(p, idx)
                out[(arm, c, ir_cond, s, "veto")] = float(vv[idx].mean())
    return k, draw, out


def ci(d):
    m, h = float(d.mean()), float(T_975_DF4 * d.std(ddof=1) / np.sqrt(len(d)))
    return m, m - h, m + h


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="docs/eval/corruption_v2/degraded_ir_repairs.md")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    t0 = time.time()
    draws = VERDICT_DRAWS + DESC_DRAWS
    res: dict = {}
    jobs = [(k, dr, ic) for dr in draws for ic in IR_CONDS for k in SEEDS]
    # per-job checkpoints: a restart resumes instead of starting over (a session end killed one run)
    ck = ROOT / "runs" / "v2_repairs_ckpt"
    ck.mkdir(parents=True, exist_ok=True)

    def absorb(k, dr, out):
        for key, v in out.items():
            res.setdefault(tuple(key), {})[(k, dr)] = v

    todo = []
    for job in jobs:
        f = ck / f"{job[0]}_{job[1]}_{job[2] or 'clean'}.json"
        if f.is_file():
            absorb(job[0], job[1], {tuple(None if x == "None" else x for x in kk.split("|")): v
                                    for kk, v in json.loads(f.read_text(encoding="utf-8")).items()})
        else:
            todo.append(job)
    print(f"[repairs] {len(jobs) - len(todo)} jobs from checkpoints, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for n_done, (k, dr, out) in enumerate(ex.map(score, todo), 1):
            ir_c = next(iter(out))[2]
            (ck / f"{k}_{dr}_{ir_c or 'clean'}.json").write_text(
                json.dumps({"|".join(str(x) for x in kk): v for kk, v in out.items()}), encoding="utf-8")
            absorb(k, dr, out)
            if n_done % 10 == 0:
                print(f"[repairs] {n_done}/{len(todo)} jobs ({time.time() - t0:.0f}s)", flush=True)

    def arr(key, ds):            # per system, the mean over the given draws
        return np.array([np.mean([res[key][(k, dr)] for dr in ds]) for k in SEEDS])

    f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
    cells = [(c, ic, s) for c in VIS_CONDS for ic in IR_CONDS for s in ("day", "night")]
    js = {"cells": [], "verdict": {}, "elapsed_s": None}
    L = [f"Registered in `docs/prereg-degraded-ir-repairs-2026-10-10.md`. `{PRESET}` against R1 (fallback "
         "confirms with `concentrated` alone), R2 (support bonus only where IR is merge-healthy) and R1+R2 "
         f"(`crossmodal26m_r`). Five Phase 3 systems; verdict draws VIS {VERDICT_DRAWS} / IR "
         f"{tuple(d + 10 for d in VERDICT_DRAWS)} (per-system draw mean); ship AP; Δ = arm − shipped, "
         "paired by system, 95% t-interval (df 4). Claim = fused − max(VIS, IR), fails at ≤ −0.0060 with "
         "the interval below zero.", ""]
    hdr = "| VIS | IR | slice | VIS | IR | shipped fused, veto, claim | " + " | ".join(
        f"{a}: Δ, veto" for a in ARMS if a != "shipped") + " |"
    L += ["## 1. Verdict draws (942–944)", "", hdr, "|" + "---|" * (6 + len(ARMS) - 1)]
    stats = {a: {"harm": {fl: 0 for fl in FLOORS}, "inv_break": [], "benefit": [], "holds": 0}
             for a in ARMS}
    for c, ic, s in cells:
        vis = arr(("shipped", c, ic, s, "vis"), VERDICT_DRAWS)
        ir = arr(("shipped", c, ic, s, "ir"), VERDICT_DRAWS)
        sh = arr(("shipped", c, ic, s, "fused"), VERDICT_DRAWS)
        row = {"vis": c, "ir": ic or "clean", "slice": s, "vis_ap": float(vis.mean()),
               "ir_ap": float(ir.mean())}
        txt = []
        for a in ARMS:
            fu = arr((a, c, ic, s, "fused"), VERDICT_DRAWS)
            cl = ci(fu - np.maximum(vis, ir))
            fails = cl[0] <= -CLAIM_FLOOR and cl[2] < 0
            stats[a]["holds"] += not fails
            veto = float(np.mean([res[(a, c, ic, s, "veto")][(k, dr)] for k in SEEDS for dr in VERDICT_DRAWS]))
            entry = {"fused": float(fu.mean()), "claim": list(cl), "claim_fails": bool(fails), "veto": veto}
            if a == "shipped":
                txt.append(f"{fu.mean():.4f}, {veto:.1%}, {f(cl)} {'**fails**' if fails else 'holds'}")
            else:
                dd = ci(fu - sh)
                entry["delta"] = list(dd)
                for fl in FLOORS:
                    stats[a]["harm"][fl] += dd[0] <= -fl and dd[2] < 0
                if ic is None and abs(dd[0]) >= DECIDE:
                    stats[a]["inv_break"].append(f"{c}/{s}")
                if dd[0] >= BENEFIT and dd[1] > 0:
                    stats[a]["benefit"].append(f"{c}/{ic}/{s}")
                txt.append(f"{f(dd)}, {veto:.1%}")
            row[a] = entry
        js["cells"].append(row)
        L.append(f"| {c} | {ic or 'clean'} | {s} | {vis.mean():.4f} | {ir.mean():.4f} | " + " | ".join(txt) + " |")

    L += ["", "## 2. Decision (prereg §4)", "",
          "| arm | (a) cells harmed at 0.0008 / **0.0014** / 0.0024 / 0.0060 | (b) clean-IR cells moved ≥ 0.0014 "
          "| (c) cells gaining ≥ 0.0060, interval > 0 | verdict | claim holds, of 60 |", "|---|---|---|---|---|---|"]
    adopt = {}
    for a in ARMS:
        st = stats[a]
        if a == "shipped":
            L.append(f"| shipped | — | — | — | — | {st['holds']} |")
            continue
        ok = st["harm"][DECIDE] == 0 and not st["inv_break"] and bool(st["benefit"])
        adopt[a] = ok
        L.append(f"| {a} | " + " / ".join((f"**{st['harm'][fl]}**" if fl == DECIDE else str(st['harm'][fl]))
                                        for fl in FLOORS)
                 + f" | {len(st['inv_break'])}{(' (' + ', '.join(st['inv_break']) + ')') if st['inv_break'] else ''}"
                 + f" | {len(st['benefit'])} | {'**ADOPT**' if ok else 'reject'} | {st['holds']} |")
        js["verdict"][a] = {"harm": {str(k): v for k, v in st["harm"].items()}, "inv_break": st["inv_break"],
                            "benefit": st["benefit"], "adopt": ok, "claim_holds": st["holds"]}
    js["verdict"]["shipped"] = {"claim_holds": stats["shipped"]["holds"]}
    if adopt["R1"] and adopt["R2"]:
        both = js["verdict"]["R1+R2"]
        final = "R1+R2" if (both["harm"][str(DECIDE)] == 0 and not both["inv_break"]) else "none (R1+R2 fails a/b)"
    elif adopt["R1"] or adopt["R2"]:
        final = "R1" if adopt["R1"] else "R2"
    else:
        final = "none"
    js["verdict"]["crossmodal26m_r"] = final
    L += ["", f"**`crossmodal26m_r` = {final}** (prereg §4).", ""]

    # ---- draw 941, descriptive -------------------------------------------------
    L += ["## 3. Draw 941 / 951 alone (descriptive; informed R1)", "",
          "| VIS | IR | slice | " + " | ".join(f"{a} Δ" for a in ARMS if a != "shipped") + " |",
          "|" + "---|" * (3 + len(ARMS) - 1)]
    for c, ic, s in cells:
        sh = arr(("shipped", c, ic, s, "fused"), DESC_DRAWS)
        ds = [ci(arr((a, c, ic, s, "fused"), DESC_DRAWS) - sh) for a in ARMS if a != "shipped"]
        if any(abs(d[0]) >= 0.0005 for d in ds):
            L.append(f"| {c} | {ic or 'clean'} | {s} | " + " | ".join(f(d) for d in ds) + " |")
    L.append("\nRows where every arm moves the cell by less than 0.0005 are omitted.")
    js["elapsed_s"] = time.time() - t0
    out = ROOT / args.out
    out.write_text("# Two degraded-IR repairs to `crossmodal26m`, registered\n\n" + "\n".join(L) + "\n",
                   encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("\n".join(L))
    print(f"[repairs] wrote {out} ({time.time() - t0:.0f}s)")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    raise SystemExit(main())
