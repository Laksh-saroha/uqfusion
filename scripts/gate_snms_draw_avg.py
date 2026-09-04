"""Draw-averaged adoption gate for VIS soft-NMS.

Implements `docs/prereg-snms-draw-averaged-gate.md` **exactly**. That document was
committed (1fbf735) before this script was ever run; the decision rule below is a
transcription of it, not a fresh judgement, and must not be edited to fit results.

The rule, restated so it is readable at the point of use:

  * 4 draws -- shipped (VIS 1 / IR 7) plus VIS 901,902,903 / IR 911,912,913.
    Corruption KIND and SEVERITY are fixed at the shipped values on every draw;
    only the seed moves.
  * `vis_soft_nms` = 0.5, unchanged, not re-tuned here.
  * Per cell, average `delta_day` and `delta_night` across the 4 draws.
  * ADOPT iff draw-averaged delta >= 0 on EVERY cell, day AND night.
  * TEST rejects (negative draw-averaged delta on the clean cell -> reject) but is
    never used to select anything.
  * sd across draws, per-draw tables and bootstrap CIs are REPORTED and are NOT
    decision inputs.

Why draws rather than more frames: IR val holds 2,234 frames total and `pohang04`
has no IR at all, so the 1,200 paired day frames are every paired day frame that
exists. The corruption draw is the only un-measured axis. See SS0 of
`runs/eval/snms_cell_redraw.md`.

Nothing under `runs/cache/`, `runs/cache_m/`, `runs/derived/` or `runs/eval/` is
modified. Per-draw caches go to `runs/cache_m_draw<seed>/`, with uncorrupted
members hard-linked from `runs/cache_m`.

Usage:
    python scripts/gate_snms_draw_avg.py
"""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import fmt, md_table, sgn, write_md   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from uqfusion.config import load_config, resolve_gpu_python              # noqa: E402
from uqfusion.eval.apmetrics import ap_from_parts, frame_parts           # noqa: E402
from uqfusion.eval.ctx import (NIGHT_RUNS, TEST_RUNS, TUNE_RUNS,         # noqa: E402
                               load_context, run_systems)

VIS_W = ROOT / "runs/full_scale/gauss_vis_seed0/weights/best.pt"
IR_W = ROOT / "runs/full_scale/gauss_ir_seed0_ft/weights/best.pt"
PV = ROOT / "runs/derived/paired_val_vis.txt"
PI = ROOT / "runs/derived/paired_val_ir.txt"

#: The 11 benchmark cells, identical to `sweep_vis_soft_nms.py`.
CELLS = [("clean", None), ("clean", "glare_s2"), ("clean", "blur_s2"),
         ("clean", "noise_s2"), ("clean", "fog_s2"),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None), ("fog", None),
         ("lowlight", "glare_s2"), ("blur_s3", "glare_s2")]

#: Every corrupted cache the cells above need: (modality, stem, kind, severity).
#: `clean` VIS/IR and the Mahalanobis fit caches carry no corruption and are
#: hard-linked unchanged. VIS `glare` is deliberately absent -- no cell uses it.
CORRUPTED = [
    ("vis", "gauss_vis_paired_blur_s3", "blur", 3),
    ("vis", "gauss_vis_paired_noise_s2", "noise", 2),
    ("vis", "gauss_vis_paired_rain_s2", "rain", 2),
    ("vis", "gauss_vis_paired_fog", "fog", 2),
    ("vis", "gauss_vis_paired_lowlight", "lowlight", 2),
    ("ir", "gauss_ir_paired_glare_s2", "glare", 2),
    ("ir", "gauss_ir_paired_blur_s2", "blur", 2),
    ("ir", "gauss_ir_paired_noise_s2", "noise", 2),
    ("ir", "gauss_ir_paired_fog_s2", "fog", 2),
]
CORRUPT_STEMS = {s for _m, s, _k, _v in CORRUPTED}

SHIPPED_VIS_SEED, SHIPPED_IR_SEED = 1, 7
#: Pre-registered in docs/prereg-snms-draw-averaged-gate.md SS"The rule" item 1.
NEW_SEEDS = (901, 902, 903)
SIGMA = 0.5


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def cache_seed(p: Path) -> int | None:
    """The `corrupt_seed` a cache was actually built with, from its own meta."""
    try:
        from uqfusion.eval.cache import load_cache
        _recs, meta = load_cache(p)
        return int(meta.get("corrupt_seed"))
    except Exception:                                    # noqa: BLE001
        return None


def build(gp: str, mod: str, stem: str, kind: str, sev: int, seed: int,
          out_dir: Path, log_dir: Path) -> bool:
    out = out_dir / f"{stem}.pkl"
    # Existence is NOT proof of the right draw. `redraw_snms_cell.py` hard-linked
    # every non-contested cache from `cache_m` into these directories, so seven of
    # the nine corrupted caches were present but carried the SHIPPED seed. Skipping
    # on existence would have silently made 7/9 corruptions identical across
    # "independent" draws. Verify the seed from the cache's own meta instead.
    if out.is_file():
        got = cache_seed(out)
        if got == seed:
            return True
        print(f"  [restale] {stem}: meta seed {got} != {seed}, rebuilding",
              flush=True)
        out.unlink()                 # a hard link: `cache_m`'s copy is untouched
    w, lst = (IR_W, PI) if mod == "ir" else (VIS_W, PV)
    cmd = [gp, "-u", str(ROOT / "scripts/build_cache.py"),
           "--source", "gaussian", "--weights", str(w),
           "--images-list", str(lst), "--imgsz", "640", "--conf", "0.001",
           "--corrupt", kind, "--severity", str(sev), "--corrupt-seed", str(seed),
           "--out", str(out)]
    # The GPU interpreter is the system Python; `uqfusion` is not installed in it,
    # so PYTHONPATH has to carry src/ exactly as the shell builders do.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    t = time.time()
    print(f"  [build] {stem} seed={seed}", flush=True)
    with open(log_dir / f"{stem}.log", "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(ROOT), env=env)
    if rc != 0:
        print(f"  [FAIL] {stem} rc={rc}", flush=True)
        return False
    print(f"  [ok] {stem} in {time.time() - t:.0f}s", flush=True)
    return True


def prepare_draw(gp: str, seed: int, base: Path, log_dir: Path) -> Path | None:
    out_dir = ROOT / f"runs/cache_m_draw{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    for p in sorted(base.glob("*.pkl")):
        if p.stem not in CORRUPT_STEMS:
            link_or_copy(p, out_dir / p.name)
    ok = True
    for mod, stem, kind, sev in CORRUPTED:
        s = seed + 10 if mod == "ir" else seed
        ok &= build(gp, mod, stem, kind, sev, s, out_dir, log_dir)
    return out_dir if ok else None


def measure_draw(cache_dir: Path) -> dict:
    """All 11 cells on one draw. Returns {cell: (d_day, d_night, d_tune, d_test)}."""
    conds = sorted({c for c, _ in CELLS})
    ircs = sorted({i for _c, i in CELLS if i})

    def ctx(ir_cond, snms):
        return load_context(preset="crossmodal26m", cache_dir=str(cache_dir),
                            conditions=tuple(conds), ir_condition=ir_cond,
                            vis_soft_nms=snms, verbose=False)

    base = {ic: ctx(ic, None) for ic in [None] + ircs}
    snms = {ic: ctx(ic, SIGMA) for ic in [None] + ircs}
    c0 = base[None]
    sel = {"day": np.flatnonzero(~np.isin(c0.runs, NIGHT_RUNS)),
           "night": np.flatnonzero(np.isin(c0.runs, NIGHT_RUNS)),
           "tune": np.flatnonzero(np.isin(c0.runs, TUNE_RUNS)),
           "test": np.flatnonzero(np.isin(c0.runs, TEST_RUNS))}
    out = {}
    for vc, ic in CELLS:
        pb = frame_parts(run_systems(base[ic], vc)["fused_gated"], base[ic].gts)
        pa = frame_parts(run_systems(snms[ic], vc)["fused_gated"], snms[ic].gts)
        d = {}
        for k, s in sel.items():
            d[k] = (ap_from_parts(pa, sel=s)["map50_95"]
                    - ap_from_parts(pb, sel=s)["map50_95"])
        b = {k: ap_from_parts(pb, sel=s)["map50_95"] for k, s in sel.items()}
        a = {k: ap_from_parts(pa, sel=s)["map50_95"] for k, s in sel.items()}
        out[(vc, ic)] = (d, b, a)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/eval/snms_gate_draw_avg.md")
    args = ap.parse_args()
    t0 = time.time()

    gp = str(resolve_gpu_python(load_config()))
    base = ROOT / "runs/cache_m"
    log_dir = ROOT / "runs/logs_cache_draws"
    log_dir.mkdir(parents=True, exist_ok=True)

    draws = []          # (label, per-cell dict)
    print(f"[draw] shipped {SHIPPED_VIS_SEED}/{SHIPPED_IR_SEED}", flush=True)
    draws.append((f"{SHIPPED_VIS_SEED}/{SHIPPED_IR_SEED} (shipped)",
                  measure_draw(base)))
    for s in NEW_SEEDS:
        print(f"[draw] {s}/{s + 10}", flush=True)
        d = prepare_draw(gp, s, base, log_dir)
        if d is None:
            print(f"[abort] draw {s} failed to build -- the pre-registered N is 4 "
                  f"and a partial N is a different experiment", flush=True)
            return 1
        draws.append((f"{s}/{s + 10}", measure_draw(d)))
        print(f"[draw] {s} done at {time.time() - t0:.0f}s", flush=True)

    n = len(draws)
    # ---- the pre-registered statistic: mean over draws, per cell ----
    rows, worst_day, worst_night = [], 1.0, 1.0
    for vc, ic in CELLS:
        dd = np.array([d[1][(vc, ic)][0]["day"] for d in draws])
        dn = np.array([d[1][(vc, ic)][0]["night"] for d in draws])
        b_day = np.mean([d[1][(vc, ic)][1]["day"] for d in draws])
        a_day = np.mean([d[1][(vc, ic)][2]["day"] for d in draws])
        worst_day = min(worst_day, dd.mean())
        worst_night = min(worst_night, dn.mean())
        rows.append([f"{vc}/{ic or 'clean'}", fmt(b_day), fmt(a_day),
                     sgn(dd.mean()), f"{dd.std(ddof=1):.4f}",
                     f"{int((dd < 0).sum())}/{n}",
                     sgn(dn.mean()), f"{int((dn < 0).sum())}/{n}"])

    clean_test = np.mean([d[1][("clean", None)][0]["test"] for d in draws])
    clean_tune = np.mean([d[1][("clean", None)][0]["tune"] for d in draws])

    secs = [
        f"Implements `docs/prereg-snms-draw-averaged-gate.md`, committed at "
        f"`1fbf735` **before this run**. Preset `crossmodal26m`, `vis_soft_nms` "
        f"{SIGMA}, 1200 paired day frames, {n} corruption draws.  \n"
        "Corruption kind and severity are fixed at the shipped values on every "
        "draw; only the seed moves.",

        "## 1. The pre-registered statistic — draw-averaged delta per cell\n\n"
        "`sd` and `neg` are REPORTED, not decision inputs (pre-registration item 6). "
        "The decision reads one thing: whether the mean columns are all >= 0.\n\n"
        + md_table(["cell (vis/ir)", "shipped day", "soft-NMS day",
                    "mean delta day", "sd", "neg", "mean delta night", "neg"], rows)
        + f"\n\n**Worst cell, draw-averaged: day {sgn(worst_day)}, "
          f"night {sgn(worst_night)}.**",

        "## 2. Per-draw detail (diagnostic)\n\n"
        "Each draw's `delta day` on every cell. Spread here is corruption noise, "
        "which is the whole reason this gate exists.\n\n"
        + md_table(["cell (vis/ir)"] + [lbl for lbl, _ in draws],
                   [[f"{vc}/{ic or 'clean'}"]
                    + [sgn(d[(vc, ic)][0]["day"]) for _l, d in draws]
                    for vc, ic in CELLS]),

        "## 3. TEST — rejects, never selects\n\n"
        f"Draw-averaged delta on the clean cell: TUNE {sgn(clean_tune)}, "
        f"**TEST {sgn(clean_test)}**.\n\n"
        + ("TEST is not negative, so rule 5 does not reject."
           if clean_test >= 0 else
           "**TEST is negative — rule 5 rejects regardless of rule 4.**"),

        "## 4. Verdict against the pre-registered rule\n\n"
        f"* every-cell day >= 0: **{'YES' if worst_day >= 0 else 'NO'}** "
        f"(worst {sgn(worst_day)})\n"
        f"* every-cell night >= 0: **{'YES' if worst_night >= 0 else 'NO'}** "
        f"(worst {sgn(worst_night)})\n"
        f"* TEST not negative on clean: **{'YES' if clean_test >= 0 else 'NO'}** "
        f"({sgn(clean_test)})\n\n"
        + ("### ADOPT\n\nAll three pre-registered conditions hold. "
           "`crossmodal26m_snms` becomes the shipped preset and the headline table "
           "is re-baselined under it; `crossmodal26m` keeps reproducing the "
           "published numbers under its own name."
           if (worst_day >= 0 and worst_night >= 0 and clean_test >= 0) else
           "### DO NOT ADOPT\n\nAt least one pre-registered condition fails. This is "
           "now a reproducible cost averaged over draws, not a single-draw artefact. "
           "`vis_soft_nms` stays off and `crossmodal26m` remains the shipped preset."),

        f"---\n\n_Generated by `scripts/gate_snms_draw_avg.py` in "
        f"{time.time() - t0:.1f}s._",
    ]
    write_md(args.out, "VIS soft-NMS — draw-averaged adoption gate", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
