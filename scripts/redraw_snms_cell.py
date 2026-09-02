"""Is the soft-NMS gate's one negative cell a real cost, or one corruption draw?

`vis_soft_nms_adoption_v2.md` failed the every-cell bar on exactly one cell,
`blur_s3/glare_s2`, at **-0.0004** with a CI of [-0.0009, +0.0023]. Ten cells
gained. The obvious follow-up -- re-gate on a wider substrate -- is **impossible**:
IR val is 2,234 frames total (pohang00 836, pohang01 1034, pohang02 247,
pohang03 117) and `pohang04` has no IR at all, so the 1,200 paired day frames the
gate already uses ARE every paired day frame in existence.

What IS still un-measured is the thing the 2026-09-01 log listed as open in
SS10.7: **one corruption draw per condition, one bootstrap seed.** That -0.0004
rests on a single realisation of blur(severity 3, seed 1) over VIS and
glare(severity 2, seed 7) over IR. The bootstrap resamples FRAMES; it cannot see
draw-to-draw variance, so it cannot answer this.

This re-draws that cell's two corruptions at several seeds and reports the
distribution of the delta. The question is not "is the mean positive" -- it is
**"is the sign stable"**. If the delta flips sign across draws, -0.0004 is noise
in the corruption and the every-cell bar was tripped by a coin flip. If it is
negative on every draw, soft-NMS has a real cost on both-degraded frames and the
bar did its job.

`clean/clean` is carried alongside as a control: it has no corruption, so its
delta MUST be identical on every draw. If it moves, this script is wrong.

Nothing under `runs/cache/`, `runs/cache_m/`, `runs/derived/` or `runs/eval/` is
modified. Per-draw caches go to NEW directories `runs/cache_m_draw<seed>/`, with
the uncorrupted members hard-linked from `runs/cache_m` so a draw costs ~15 MB
rather than 202 MB.

Usage:
    python scripts/redraw_snms_cell.py --draws 5
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
from uqfusion.eval.apmetrics import (ap_from_parts, bootstrap_delta,     # noqa: E402
                                     frame_parts)
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems      # noqa: E402

VIS_W = ROOT / "runs/full_scale/gauss_vis_seed0/weights/best.pt"
IR_W = ROOT / "runs/full_scale/gauss_ir_seed0_ft/weights/best.pt"
PV = ROOT / "runs/derived/paired_val_vis.txt"
PI = ROOT / "runs/derived/paired_val_ir.txt"

#: The contested cell's two corruptions. Kind and severity are FIXED at the
#: shipped values -- only the seed moves, because the seed is the thing whose
#: influence is unmeasured.
VIS_STEM, VIS_KIND, VIS_SEV = "gauss_vis_paired_blur_s3", "blur", 3
IR_STEM, IR_KIND, IR_SEV = "gauss_ir_paired_glare_s2", "glare", 2

#: The shipped draw, already measured in `vis_soft_nms_adoption_v2.md`.
SHIPPED_VIS_SEED, SHIPPED_IR_SEED = 1, 7


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def build(gp: str, mod: str, stem: str, lst: Path, kind: str, sev: int,
          seed: int, out_dir: Path, log_dir: Path) -> bool:
    out = out_dir / f"{stem}.pkl"
    if out.is_file():
        print(f"[skip] {out}")
        return True
    w = IR_W if mod == "ir" else VIS_W
    cmd = [gp, "-u", str(ROOT / "scripts/build_cache.py"),
           "--source", "gaussian", "--weights", str(w),
           "--images-list", str(lst), "--imgsz", "640", "--conf", "0.001",
           "--corrupt", kind, "--severity", str(sev), "--corrupt-seed", str(seed),
           "--out", str(out)]
    log = log_dir / f"{stem}.log"
    print(f"[build] {out.name} seed={seed}")
    t = time.time()
    # The GPU interpreter is the system Python, not the venv, so `uqfusion` is not
    # installed in it -- the shell builders `export PYTHONPATH="$PWD/src"` for
    # exactly this reason and `build_day_substrate.py` does the same.
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    with open(log, "w", encoding="utf-8") as fh:
        rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT,
                             cwd=str(ROOT), env=env)
    if rc != 0:
        print(f"[FAIL] {out} rc={rc} -- see {log}")
        return False
    print(f"[ok] {out.name} in {time.time() - t:.0f}s")
    return True


def prepare_draw(gp: str, seed: int, base: Path, log_dir: Path) -> Path | None:
    """One draw = the shipped cache_m with the two contested caches re-drawn."""
    out_dir = ROOT / f"runs/cache_m_draw{seed}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # everything that is NOT the two contested caches is shared, unchanged
    for p in sorted(base.glob("*.pkl")):
        if p.stem in (VIS_STEM, IR_STEM):
            continue
        link_or_copy(p, out_dir / p.name)
    ok = build(gp, "vis", VIS_STEM, PV, VIS_KIND, VIS_SEV, seed, out_dir, log_dir)
    ok &= build(gp, "ir", IR_STEM, PI, IR_KIND, IR_SEV, seed + 10, out_dir, log_dir)
    return out_dir if ok else None


def measure(cache_dir: Path, n_boot: int) -> dict:
    """Both cells, shipped vs soft-NMS, on one draw."""
    out = {}
    for vc, ic, tag in [("blur_s3", "glare_s2", "contested"),
                        ("clean", None, "control")]:
        kw = dict(preset="crossmodal26m", cache_dir=str(cache_dir),
                  conditions=(vc,), ir_condition=ic, verbose=False)
        b = load_context(**kw)
        a = load_context(**kw, vis_soft_nms=0.5)
        pb = frame_parts(run_systems(b, vc)["fused_gated"], b.gts)
        pa = frame_parts(run_systems(a, vc)["fused_gated"], a.gts)
        day = np.flatnonzero(~np.isin(b.runs, NIGHT_RUNS))
        d0 = ap_from_parts(pb, sel=day)["map50_95"]
        d1 = ap_from_parts(pa, sel=day)["map50_95"]
        bs = bootstrap_delta(pa, pb, sel=day, n_boot=n_boot, seed=0)
        out[tag] = (d0, d1, d1 - d0, bs["ci_lo"], bs["ci_hi"])
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, default=5,
                    help="how many NEW corruption seeds (the shipped one is always included)")
    ap.add_argument("--first-seed", type=int, default=101)
    ap.add_argument("--n-boot", type=int, default=300)
    ap.add_argument("--out", default="runs/eval/snms_cell_redraw.md")
    args = ap.parse_args()
    t0 = time.time()

    gp = str(resolve_gpu_python(load_config()))
    base = ROOT / "runs/cache_m"
    log_dir = ROOT / "runs/logs_cache_draws"
    log_dir.mkdir(parents=True, exist_ok=True)

    seeds = [SHIPPED_VIS_SEED] + [args.first_seed + i for i in range(args.draws)]
    rows, ctrl_rows, deltas = [], [], []
    for i, s in enumerate(seeds):
        if i == 0:
            cd = base                      # the shipped draw, already on disk
            label = f"{s} / {SHIPPED_IR_SEED} (shipped)"
        else:
            got = prepare_draw(gp, s, base, log_dir)
            if got is None:
                print(f"[warn] draw seed {s} failed to build; skipping")
                continue
            cd, label = got, f"{s} / {s + 10}"
        m = measure(cd, args.n_boot)
        d0, d1, dd, lo, hi = m["contested"]
        deltas.append(dd)
        rows.append([label, fmt(d0), fmt(d1), sgn(dd), f"[{sgn(lo)}, {sgn(hi)}]"])
        c0, c1, cd_, _, _ = m["control"]
        ctrl_rows.append([label, fmt(c0), fmt(c1), sgn(cd_)])
        print(f"[draw {label}] contested {sgn(dd)}   control {sgn(cd_)}")

    arr = np.asarray(deltas, dtype=float)
    n_neg = int((arr < 0).sum())
    secs = [
        "Cell `blur_s3/glare_s2` -- the single cell that failed the adoption bar in "
        "`vis_soft_nms_adoption_v2.md` at **-0.0004**. Preset `crossmodal26m`, "
        "`vis_soft_nms` 0.5, 1200 paired day frames.  \n"
        "Only the CORRUPTION SEED changes between rows; kind and severity are fixed "
        "at the shipped `blur` s3 (VIS) and `glare` s2 (IR).",

        "## 0. Why this and not a wider substrate\n\n"
        "IR val holds **2,234 frames total** -- pohang00 836, pohang01 1034, "
        "pohang02 247, pohang03 117 -- and `pohang04` has **no IR at all**. The day "
        "half of that is 836+247+117 = **1,200 frames**, which is exactly the paired "
        "substrate the gate already uses. There is no wider fused substrate to move "
        "to; re-gating at `pohang04` scale is not a thing that can be done. The "
        "un-measured axis is the corruption draw, not the frame count.",

        "## 1. The contested cell across corruption draws\n\n"
        "The question is **sign stability**, not the mean. A delta that flips sign "
        "across draws is noise in the corruption, and the bar was tripped by a coin "
        "flip.\n\n"
        + md_table(["vis seed / ir seed", "shipped day", "soft-NMS day", "delta day",
                    "95% CI (frames)"], rows),

        "## 2. Control -- `clean/clean`, which has no corruption at all\n\n"
        "Identical on every row, or this script is measuring something other than the "
        "seed.\n\n"
        + md_table(["vis seed / ir seed", "shipped day", "soft-NMS day", "delta day"],
                   ctrl_rows),

        f"## 3. Verdict\n\n"
        f"* draws measured: **{len(arr)}**\n"
        f"* delta mean **{sgn(arr.mean())}**, sd {arr.std(ddof=1) if len(arr) > 1 else float('nan'):.4f}, "
        f"range [{sgn(arr.min())}, {sgn(arr.max())}]\n"
        f"* negative on **{n_neg} of {len(arr)}** draws\n\n"
        + ("**Sign is NOT stable** -- the -0.0004 is a property of one corruption "
           "draw, not of the system. The every-cell bar was tripped by draw noise, "
           "and re-running the gate over draws is the honest form of that bar."
           if 0 < n_neg < len(arr) else
           "**Negative on every draw** -- soft-NMS has a real, reproducible cost on "
           "both-degraded frames. The bar did its job and the parameter stays off."
           if n_neg == len(arr) else
           "**Negative on no draw but the shipped one** -- the shipped draw is the "
           "outlier. Worth one more confirmation before acting."),

        f"---\n\n_Generated by `scripts/redraw_snms_cell.py` in {time.time() - t0:.1f}s._",
    ]
    write_md(args.out, "soft-NMS -- is the one failing cell a corruption draw?", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
