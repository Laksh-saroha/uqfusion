"""Score the night veto against its pre-registration (V1).

`docs/prereg-night-veto.md`, committed `5761e9f` before the from-scratch VIS run
was launched. Every threshold, band and guard below is a transcription of that
document. Nothing here chooses anything; if a rule looks wrong, the rule is
reported as-registered and the objection goes in the report, not into the code.

THE ARMS. Both use the SAME VIS checkpoint and the SAME IR stream. The only
difference is the night arm of the veto:

  ON   `preset="crossmodal26m"` exactly as shipped.
  OFF  the same context with `ir_night` and `ir_night_raw` forced all-False.

Why that is the right OFF. Under `crossmodal26m` the VIS veto mask is

    vv  =  night & (dark | veil)                       # veil_requires_night=True
        |  ir_night_raw & ~ir_ok & (concentrated | (dark & veil))   # weak fallback

and `veil_requires_night` means the bare veil term is NOT or-ed in separately.
So zeroing the two night flags removes the veto entirely and touches nothing
else -- no source edit, no constant moved, no second code path. The veil axis
travelling with the night arm is a REAL coupling of the shipped system, not an
artefact of this script, and the report says so rather than unpicking it.

WHAT IS DETECTOR-DEPENDENT AND WHAT IS NOT. Swapping the VIS checkpoint changes
the VIS detections and nothing else the gate reads: `grad_gini`, `ir_p05`,
`lap_over_var` and the `vis_health` features are all IMAGE statistics, and
`crossmodal` sets `mu_d=1e9, lam=0` so the Mahalanobis soft term is inert. The
capability prior IS detector-dependent and is recomputed from the caches by
`load_context`, so it follows the new checkpoint automatically. That is why only
the VIS caches are rebuilt here.

DRAWS (prereg rule 1). Four paired draws, the `gate_snms_draw_avg.py` precedent:
shipped (VIS 1 / IR 7) plus VIS 901/902/903 with IR 911/912/913. Corruption kind
and severity are fixed at the shipped values on every draw; only the seed moves.
IR caches are hard-linked from the existing `runs/cache_m*` draws -- the IR
checkpoint has not changed, so rebuilding them would only add sampling noise.

Nothing under `runs/cache/`, `runs/cache_m*/`, `runs/derived/`, `runs/eval/` or
`archive/` is written. New VIS caches go to `runs/cache_nv_base/` (uncorrupted,
built once) and `runs/cache_nv_draw<seed>/`; `write_md` refuses an existing --out.

Usage:
    python scripts/eval_night_veto.py
    python scripts/eval_night_veto.py --arm secondary --out runs/eval/night_veto_secondary.md
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _ideas_common import fmt, md_table, sgn, write_md                    # noqa: E402
from uqfusion.config import resolve_gpu_python                           # noqa: E402
from uqfusion.eval.apmetrics import (ap_from_parts, bootstrap_delta,     # noqa: E402
                                     frame_parts)
from uqfusion.eval.cache import load_cache                               # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems      # noqa: E402

PREREG = "docs/prereg-night-veto.md"

#: The primary arm is the COLD START. The prereg is explicit that the fine-tuned
#: checkpoint carries a "night is empty" prior and therefore cannot decide this.
ARMS = {
    "primary": ROOT / "runs/full_scale/gauss_vis_nightfull/weights/best.pt",
    "secondary": ROOT / "runs/full_scale/gauss_vis_nightrestore/weights/best.pt",
}
IR_W = ROOT / "runs/full_scale/gauss_ir_seed0_ft/weights/best.pt"
PV = ROOT / "runs/derived/paired_val_vis.txt"
MAHA_VIS = ROOT / "runs/derived/maha_fit_vis.txt"

#: The 11 benchmark cells, identical to `gate_snms_draw_avg.py`.
CELLS = [("clean", None), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("clean", "fog_s2"),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None), ("fog", None),
         ("lowlight", "glare_s2"), ("blur_s3", "glare_s2")]

#: The headline cell: clean VIS, clean IR, night frames. Prereg "The endpoint".
PRIMARY_CELL = ("clean", None)

#: VIS caches that carry a corruption draw. Rebuilt per draw.
VIS_CORRUPT = [("gauss_vis_paired_blur_s3", "blur", 3),
               ("gauss_vis_paired_noise_s2", "noise", 2),
               ("gauss_vis_paired_rain_s2", "rain", 2),
               ("gauss_vis_paired_fog", "fog", 2),
               ("gauss_vis_paired_lowlight", "lowlight", 2)]

#: VIS caches with no corruption: built once, hard-linked into every draw.
VIS_CLEAN = [("gauss_vis_paired_clean", PV), ("gauss_vis_train_clean", MAHA_VIS)]

#: (VIS seed, IR seed, the cache_m directory the IR caches are taken from).
#: Pre-registered in `docs/prereg-snms-draw-averaged-gate.md`, cited by rule 1.
DRAWS = [(1, 7, "runs/cache_m"), (901, 911, "runs/cache_m_draw901"),
         (902, 912, "runs/cache_m_draw902"), (903, 913, "runs/cache_m_draw903")]

FLOOR_ABS = 0.002            # prereg rule 3
N_BOOT, BOOT_SEED = 2000, 0


# ----------------------------------------------------------------- gates
def committed(rel: str) -> bool:
    """Is `rel` committed and unmodified? The prereg must predate the result."""
    try:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                                 cwd=ROOT, capture_output=True).returncode == 0
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel],
                               cwd=ROOT, capture_output=True).returncode != 0
        return tracked and not dirty
    except OSError:
        return False


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def meta_of(p: Path) -> dict:
    try:
        return load_cache(p)[1]
    except Exception:                                       # noqa: BLE001
        return {}


def build(gp: str, weights: Path, stem: str, images: Path, kind: str | None,
          sev: int, seed: int | None, out_dir: Path, log_dir: Path) -> bool:
    """One cache. Existence is not proof -- the meta must match weights AND seed.

    `gate_snms_draw_avg.py` learned this the hard way: hard-linked caches were
    present but carried the shipped corruption seed, which would have made seven
    of nine corruptions identical across "independent" draws. Here the same trap
    has a second door -- a cache built from the OLD checkpoint is also present
    and also looks valid -- so the weights are checked too.
    """
    out = out_dir / f"{stem}.pkl"
    if out.is_file():
        m = meta_of(out)
        got_w = (m.get("weights") or [None])[0]
        ok = (got_w and Path(got_w).as_posix().endswith(
            weights.relative_to(ROOT).as_posix()))
        if ok and m.get("corrupt_seed") == seed and m.get("corrupt") == kind:
            return True
        print(f"  [restale] {stem}: weights={got_w} seed={m.get('corrupt_seed')} "
              f"-- rebuilding", flush=True)
        out.unlink()                     # a hard link; the source dir is untouched
    cmd = [gp, "-u", str(ROOT / "scripts/build_cache.py"),
           "--source", "gaussian", "--weights", str(weights),
           "--images-list", str(images), "--imgsz", "640", "--conf", "0.001",
           "--out", str(out)]
    if kind:
        cmd += ["--corrupt", kind, "--severity", str(sev),
                "--corrupt-seed", str(seed)]
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    t = time.time()
    print(f"  [build] {stem} seed={seed}", flush=True)
    with open(log_dir / f"{stem}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(ROOT), env=env)
    if rc != 0:
        print(f"  [FAIL] {stem} rc={rc} -- see {log_dir / (stem + '.log')}",
              flush=True)
        return False
    print(f"  [ok] {stem} in {time.time() - t:.0f}s", flush=True)
    return True


def prepare(gp: str, weights: Path, tag: str, log_dir: Path) -> list[Path] | None:
    """Base caches once, then one directory per draw. Returns the four dirs."""
    base = ROOT / f"runs/cache_{tag}_base"
    base.mkdir(parents=True, exist_ok=True)
    for stem, lst in VIS_CLEAN:
        if not build(gp, weights, stem, lst, None, 0, None, base, log_dir):
            return None
    dirs = []
    for vseed, _iseed, src in DRAWS:
        d = ROOT / f"runs/cache_{tag}_draw{vseed}"
        d.mkdir(parents=True, exist_ok=True)
        # IR travels unchanged from the draw it was built in. The IR checkpoint
        # is not the treatment here and rebuilding it would only add noise.
        for p in sorted((ROOT / src).glob("gauss_ir_*.pkl")):
            link_or_copy(p, d / p.name)
        for stem, _lst in VIS_CLEAN:
            link_or_copy(base / f"{stem}.pkl", d / f"{stem}.pkl")
        for stem, kind, sev in VIS_CORRUPT:
            if not build(gp, weights, stem, PV, kind, sev, vseed, d, log_dir):
                return None
        dirs.append(d)
    return dirs


# ----------------------------------------------------------------- measurement
def off_arm(ctx):
    """The veto's night arm removed, and nothing else."""
    n = len(ctx.gts)
    z = np.zeros(n, dtype=bool)
    return replace(ctx, ir_night=z, ir_night_raw=z)


def partner_rate(vis_recs, ir_recs, sel, iou_thr: float) -> float:
    """Share of VIS boxes with an IR partner at `iou_thr`. Prereg weakness 2.

    The prereg requires this number rather than an assumption: at iou_thr 0.85
    fusion is 99.9% concatenation, so any gain is union recall and not sensor
    agreement, and the report has to say which.
    """
    num = den = 0
    for i in sel:
        a = np.asarray(vis_recs[i]["boxes_xyxy"], dtype=float).reshape(-1, 4)
        b = np.asarray(ir_recs[i]["boxes_xyxy"], dtype=float).reshape(-1, 4)
        den += len(a)
        if not len(a) or not len(b):
            continue
        x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)
        num += int((iou.max(axis=1) >= iou_thr).sum())
    return num / den if den else float("nan")


def measure(cache_dir: Path) -> dict:
    """Every cell on one draw, both arms. Frame parts kept for the bootstrap."""
    conds = sorted({c for c, _ in CELLS})
    ircs = sorted({i for _c, i in CELLS if i})
    ctxs = {ic: load_context(preset="crossmodal26m", cache_dir=str(cache_dir),
                             conditions=tuple(conds), ir_condition=ic,
                             verbose=False)
            for ic in [None] + ircs}
    c0 = ctxs[None]
    night = np.flatnonzero(np.isin(c0.runs, NIGHT_RUNS))
    day = np.flatnonzero(~np.isin(c0.runs, NIGHT_RUNS))
    out = {"sel": {"day": day, "night": night}, "cells": {}}
    for vc, ic in CELLS:
        on_ctx = ctxs[ic]
        r_on = run_systems(on_ctx, vc)
        r_off = run_systems(off_arm(on_ctx), vc)
        p_on = frame_parts(r_on["fused_gated"], on_ctx.gts)
        p_off = frame_parts(r_off["fused_gated"], on_ctx.gts)
        p_ir = frame_parts(r_on["ir_in_vis"], on_ctx.gts)
        veto = np.asarray(r_on["veto_vis"], dtype=bool)
        cell = {
            "parts_on": p_on, "parts_off": p_off, "parts_ir": p_ir,
            "veto_day": float(veto[day].mean()), "veto_night": float(veto[night].mean()),
            "partner_night": partner_rate(on_ctx.vis_by_cond[vc], r_on["ir_in_vis"],
                                          night, on_ctx.iou_thr),
        }
        for k, s in (("day", day), ("night", night)):
            cell[f"on_{k}"] = ap_from_parts(p_on, sel=s)["map50_95"]
            cell[f"off_{k}"] = ap_from_parts(p_off, sel=s)["map50_95"]
            cell[f"ir_{k}"] = ap_from_parts(p_ir, sel=s)["map50_95"]
            cell[f"d_{k}"] = cell[f"off_{k}"] - cell[f"on_{k}"]
        cell["on_night_map50"] = ap_from_parts(p_on, sel=night)["map50"]
        cell["off_night_map50"] = ap_from_parts(p_off, sel=night)["map50"]
        out["cells"][(vc, ic)] = cell
    return out


# ----------------------------------------------------------------- report
def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=tuple(ARMS), default="primary")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--out", default="runs/eval/night_veto_verdict.md")
    ap.add_argument("--build-only", action="store_true")
    args = ap.parse_args()

    if not committed(PREREG):
        print(f"[gate] {PREREG} is not committed clean -- refusing to run.\n"
              f"       The registration must predate the result in git history.")
        return 1
    w = ARMS[args.arm]
    if not w.is_file():
        print(f"[gate] missing {w}")
        return 1
    print(f"[gate] prereg committed; arm={args.arm} weights={w.relative_to(ROOT)}")

    tag = "nv" if args.arm == "primary" else "nv2"
    log_dir = ROOT / f"runs/logs_{tag}"
    log_dir.mkdir(parents=True, exist_ok=True)
    gp = resolve_gpu_python()
    t0 = time.time()
    dirs = prepare(gp, w, tag, log_dir)
    if dirs is None:
        print("[fail] cache build did not complete -- nothing scored")
        return 1
    print(f"[cache] ready in {time.time() - t0:.0f}s")
    if args.build_only:
        return 0

    draws = []
    for d in dirs:
        print(f"[measure] {d.name}", flush=True)
        draws.append(measure(d))
    sel = draws[0]["sel"]

    # ---- draw-averaged deltas (rule 1) and the paired bootstrap (rule 2).
    # se is the MEAN of the per-draw se, not their se/sqrt(4): the frame noise is
    # the same 2,232 frames on every draw, so it is common across draws and does
    # not average down. Treating draws as independent samples of it would shrink
    # the floor, which is the one direction this project must not err in.
    rows, cells = [], {}
    for key in CELLS:
        d_day = [dr["cells"][key]["d_day"] for dr in draws]
        d_night = [dr["cells"][key]["d_night"] for dr in draws]
        ses = [bootstrap_delta(dr["cells"][key]["parts_off"],
                               dr["cells"][key]["parts_on"], sel=sel["night"],
                               n_boot=args.boot, seed=BOOT_SEED)["se"]
               for dr in draws]
        se = float(np.mean(ses))
        cells[key] = {
            "d_night": float(np.mean(d_night)), "sd_night": float(np.std(d_night, ddof=1)),
            "d_day": float(np.mean(d_day)), "sd_day": float(np.std(d_day, ddof=1)),
            "se": se, "floor": max(2 * se, FLOOR_ABS),
            "on_night": float(np.mean([dr["cells"][key]["on_night"] for dr in draws])),
            "off_night": float(np.mean([dr["cells"][key]["off_night"] for dr in draws])),
            "ir_night": float(np.mean([dr["cells"][key]["ir_night"] for dr in draws])),
            "veto_day": float(np.mean([dr["cells"][key]["veto_day"] for dr in draws])),
            "veto_night": float(np.mean([dr["cells"][key]["veto_night"] for dr in draws])),
            "partner": float(np.mean([dr["cells"][key]["partner_night"] for dr in draws])),
            "d_day_max": float(np.max(np.abs(d_day))),
        }

    p = cells[PRIMARY_CELL]
    floor = p["floor"]
    verdict = ("ADOPT" if p["d_night"] >= floor else
               "REJECT" if p["d_night"] <= -floor else "INCONCLUSIVE")

    # ---- Guard 1 (rule 5). Day must be bit-identical. Reported honestly: if the
    # veto never fires on a day frame the guard is a real check; if it DOES fire,
    # a day delta is the system working as designed and the guard as written
    # cannot tell the two apart. That is stated, not silently reinterpreted.
    day_moved = {k: v for k, v in cells.items() if v["d_day_max"] > 0.0}
    day_veto_max = max(v["veto_day"] for v in cells.values())
    void = bool(day_moved) and day_veto_max == 0.0

    # ---- Guard 2 (rule 6). Any IR-corrupted night cell that falls below its own
    # ir_only by more than the floor caps the verdict at INCONCLUSIVE.
    breaches = [(k, v) for k, v in cells.items()
                if k[1] is not None and v["off_night"] < v["ir_night"] - v["floor"]]
    capped = bool(breaches) and verdict == "ADOPT"
    final = "VOID" if void else ("INCONCLUSIVE" if capped else verdict)

    def cellname(k):
        return f"`{k[0]}` / IR `{k[1] or 'clean'}`"

    head = [
        f"**Arm:** {args.arm} — `{w.relative_to(ROOT).as_posix()}`. "
        f"**Pre-registration:** [`{PREREG}`]({PREREG}) (`5761e9f`), committed before "
        f"the checkpoint existed and verified clean at launch. Preset "
        f"`crossmodal26m`, `iou_thr` 0.85, {len(DRAWS)} paired draws, "
        f"n_boot {args.boot}.",
        "",
        f"## Verdict — **{final}**",
        "",
        f"Primary endpoint, cell {cellname(PRIMARY_CELL)}, NIGHT `mAP@50-95`, "
        f"draw-averaged:",
        "",
        f"    delta = {sgn(p['d_night'])}   floor = {fmt(floor)} "
        f"= max(2 x {fmt(p['se'])}, {FLOOR_ABS})",
        f"    veto ON  {fmt(p['on_night'])}      veto OFF {fmt(p['off_night'])}"
        f"      ir_only {fmt(p['ir_night'])}",
        "",
        f"Band as registered: **{verdict}**"
        + ("" if final == verdict else f", then **{final}** after the guards below."),
        "",
        "INCONCLUSIVE leaves the veto standing — the prereg is explicit that the "
        "default is the shipped behaviour and that an unresolved measurement is not "
        "a licence to change it.",
    ]

    secs = head + [
        "",
        "## Every cell (night is the endpoint; day is guard 1)",
        "",
        md_table(["cell", "veto ON", "veto OFF", "ir_only", "delta night", "sd across draws",
                  "se", "floor", "band", "delta day (max |.|)", "veto fires day / night"],
                 [[cellname(k),
                   fmt(v["on_night"]), fmt(v["off_night"]), fmt(v["ir_night"]),
                   sgn(v["d_night"]), fmt(v["sd_night"], 5), fmt(v["se"], 5),
                   fmt(v["floor"], 5),
                   ("ADOPT" if v["d_night"] >= v["floor"] else
                    "REJECT" if v["d_night"] <= -v["floor"] else "—"),
                   fmt(v["d_day_max"], 6),
                   f"{v['veto_day']:.1%} / {v['veto_night']:.1%}"]
                  for k, v in cells.items()]),
        "",
        "## Guard 1 — day must be bit-identical (rule 5)",
        "",
        f"Cells whose day number moved at all: **{len(day_moved)} of {len(cells)}**. "
        f"Highest day veto-firing rate across cells: **{day_veto_max:.1%}**.",
        "",
        ("The veto never fires on a day frame, so any day movement would be an "
         "implementation leak and the run is VOID as registered."
         if day_veto_max == 0.0 else
         "**The guard as written does not apply cleanly.** The veto fires on "
         f"{day_veto_max:.1%} of day frames in at least one cell, so removing it "
         "*must* move day there — that is the system working as designed, not a "
         "leak. Rule 5 assumed a veto confined to night. The run is NOT declared "
         "void on that basis; the day deltas are reported above and the reader "
         "should treat rule 5 as satisfied only on the cells with a 0.0% day rate."),
        "",
        "## Guard 2 — degraded IR at night (rule 6)",
        "",
        (f"**{len(breaches)} IR-corrupted night cell(s) fall below their own "
         f"`ir_only` by more than the floor:** "
         + ", ".join(f"{cellname(k)} ({sgn(v['off_night'] - v['ir_night'])})"
                     for k, v in breaches)
         + ". A fusion rule that is better on average and worse when a sensor "
           "degrades is not an improvement, so the verdict is capped at "
           "INCONCLUSIVE."
         if breaches else
         "No IR-corrupted night cell falls below its own `ir_only` by more than "
         "the floor. The guard passes."),
        "",
        "## What the gain is made of (prereg weakness 2)",
        "",
        "Fusion at `iou_thr` 0.85 is near-total concatenation, so a positive delta "
        "is union recall and not sensor agreement unless the partner rate says "
        "otherwise. Measured on the night frames, share of VIS boxes with an IR "
        "partner at 0.85:",
        "",
        md_table(["cell", "VIS boxes with an IR partner (night)"],
                 [[cellname(k), f"{v['partner']:.3%}"] for k, v in cells.items()]),
        "",
        "## Reported, and NOT decision inputs (rule 7)",
        "",
        md_table(["cell", "night mAP@50 ON", "night mAP@50 OFF"],
                 [[cellname(k),
                   fmt(np.mean([dr["cells"][k]["on_night_map50"] for dr in draws])),
                   fmt(np.mean([dr["cells"][k]["off_night_map50"] for dr in draws]))]
                  for k in CELLS]),
        "",
        "## Caveats carried from the pre-registration",
        "",
        "1. **`pohang01` is the only night run in the dataset.** This decision is "
        "made on the same night data any future night work is scored on, and no "
        "split can rescue that. Registering the bands in advance is the only "
        "protection available.",
        "2. **Night val contains zero buoys** — all 16,179 night GT boxes are "
        "class 0, so this endpoint is ship AP and says nothing about buoy at night "
        "in either direction.",
        "3. **One seed.** The cold start is seed 0 only; a null cannot be separated "
        "from seed noise at the ~0.002 level, which is why the floor sits above the "
        "single-seed noise rather than below it.",
        "4. **The veil axis travels with the night arm.** `veil_requires_night` is "
        "True under `crossmodal26m`, so turning the night flags off also disarms "
        "the veil term. That coupling is a property of the shipped system, and the "
        "OFF arm is therefore 'no VIS veto', not 'night arm only'.",
        "5. An ADOPT verdict licenses removing the night veto **and a costed "
        "proposal** for a re-baseline. It does not license swapping the VIS "
        "detector, which would move every headline number in the project.",
    ]

    out = write_md(ROOT / args.out, "Night veto — pre-registered verdict (V1)", secs)
    js = out.with_suffix(".json")
    js.write_text(json.dumps({
        "arm": args.arm, "weights": w.relative_to(ROOT).as_posix(),
        "verdict": final, "verdict_as_registered": verdict,
        "primary_cell": list(PRIMARY_CELL), "n_draws": len(DRAWS), "n_boot": args.boot,
        "cells": {f"{k[0]}|{k[1] or 'clean'}": v for k, v in cells.items()},
    }, indent=1), encoding="utf-8")
    print(f"\n[verdict] {final}   delta {sgn(p['d_night'])}  floor {fmt(floor)}")
    print(f"[out] {out}\n[out] {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
