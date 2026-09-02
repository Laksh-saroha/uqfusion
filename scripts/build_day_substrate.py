"""Idea I0 -- the blocker. Build a day substrate large enough to fit an estimator on.

The paired day set is **1200 frames** over three runs, and its TUNE half is 836
frames of one consecutive canal transit. `fit_rerank.py`'s pilot got +0.0419 on
that set and -0.0045 held out. That is a data result, not a ceiling result.

What is actually on disk:

    VIS val   11,352 frames   pohang00 1672 | 01 2068 | 02 2690 | 03 2579 | 04 2343
    IR  val    2,234 frames   pohang00  836 | 01 1034 | 02  247 | 03  117 | (no 04)

The paired set is small because IR val is small. **Every VIS-side idea (I1, I2,
I3, I4, I6) needs no IR at all**, so it can use all 9,284 day VIS val frames over
FOUR runs -- 7.7x the frames and a fourth independent run for leave-one-run-out.

`pohang04` matters most: 2,343 day frames that have never entered any experiment,
because it has no IR counterpart and every cache built so far was paired.

This writes VIS-only caches to `runs/cache_day/` -- a NEW directory. Nothing
under `runs/cache/`, `runs/cache_m/`, `runs/derived/` or `runs/eval/` is touched,
so every published number stays reproducible.

Usage:
    python scripts/build_day_substrate.py                    # all day val frames
    python scripts/build_day_substrate.py --stride 4         # every 4th (faster)
    python scripts/build_day_substrate.py --limit 40 --out-dir runs/cache_day_smoke
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.config import load_config, resolve_data_yaml, resolve_gpu_python  # noqa: E402
from uqfusion.data.lists import load_data_yaml, split_image_list                # noqa: E402

DAY_RUNS = ("pohang00", "pohang02", "pohang03", "pohang04")
#: VIS conditions worth having on the substrate. `clean` is what I1/I2 fit on;
#: the corruptions let the re-ranker be tested for robustness without a rebuild.
CONDITIONS = {
    "clean": None,
    "fog": ("fog", 2, 3),
    "rain_s2": ("rain", 2, 11),
    "blur_s3": ("blur", 3, 5),
    "lowlight": ("lowlight", 2, 4),
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out-dir", default="runs/cache_day")
    ap.add_argument("--list-dir", default="runs/derived_day")
    ap.add_argument("--weights", default="runs/full_scale/gauss_vis_seed0/weights/best.pt")
    ap.add_argument("--stride", type=int, default=1, help="take every Nth day frame")
    ap.add_argument("--limit", type=int, default=None, help="cap frames per condition (smoke)")
    ap.add_argument("--conditions", nargs="+", default=["clean"],
                    help=f"any of {sorted(CONDITIONS)}")
    ap.add_argument("--conf", type=float, default=0.001)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    t0 = time.time()

    cfg = load_config()
    images = split_image_list(load_data_yaml(resolve_data_yaml(cfg, "vis")), "val")
    day = [p for p in images if Path(p).name.split("_")[0] in DAY_RUNS]
    day = day[:: max(args.stride, 1)]
    if args.limit and args.limit < len(day):
        # Even spacing, not a head slice: the val list is ordered by run, so
        # `day[:limit]` would hand a smoke run a single run and leave
        # leave-one-run-out with no folds at all.
        import numpy as np
        keep = np.unique(np.linspace(0, len(day) - 1, int(args.limit)).astype(int))
        day = [day[i] for i in keep]
    by_run: dict[str, int] = {}
    for p in day:
        r = Path(p).name.split("_")[0]
        by_run[r] = by_run.get(r, 0) + 1
    print(f"[substrate] {len(day)} day VIS val frames (stride {args.stride}): {by_run}")
    if len(by_run) < 2:
        raise SystemExit("need at least two day runs for leave-one-run-out")

    list_dir = ROOT / args.list_dir
    list_dir.mkdir(parents=True, exist_ok=True)
    list_path = list_dir / f"day_val_vis_stride{args.stride}.txt"
    list_path.write_text("\n".join(str(p) for p in day) + "\n", encoding="utf-8")
    print(f"[substrate] list -> {list_path}")

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    py = resolve_gpu_python(cfg, require_cuda=not args.dry_run)
    print(f"[substrate] interpreter {py}")

    rc = 0
    for cond in args.conditions:
        if cond not in CONDITIONS:
            raise SystemExit(f"unknown condition {cond!r}; have {sorted(CONDITIONS)}")
        out = out_dir / f"gauss_vis_day_{cond}.pkl"
        if out.is_file():
            print(f"[skip] {out} exists")
            continue
        cmd = [py, "-u", str(ROOT / "scripts" / "build_cache.py"),
               "--source", "gaussian", "--weights", str(ROOT / args.weights),
               "--images-list", str(list_path), "--imgsz", str(args.imgsz),
               "--conf", str(args.conf), "--out", str(out)]
        spec = CONDITIONS[cond]
        if spec:
            kind, sev, seed = spec
            cmd += ["--corrupt", kind, "--severity", str(sev), "--corrupt-seed", str(seed)]
        print("[run] " + " ".join(cmd))
        if args.dry_run:
            continue
        log = ROOT / "runs" / "logs_cache_day"
        log.mkdir(parents=True, exist_ok=True)
        # `build_cache.py` imports `uqfusion` from the source tree, and the GPU
        # interpreter is the bare system Python with no editable install -- the
        # shell builders export PYTHONPATH for exactly this reason.
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        with open(log / f"{cond}.log", "w", encoding="utf-8") as fh:
            p = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT),
                               env=env)
        if p.returncode != 0:
            print(f"[FAIL] {out} -- see runs/logs_cache_day/{cond}.log")
            rc = 1
        else:
            print(f"[ok] {out}")

    print(f"[substrate] done in {time.time() - t0:.1f}s")
    print("\nNext:\n"
          "  python scripts/fit_rerank.py --vis-cache runs/cache_day/gauss_vis_day_clean.pkl "
          "--out runs/eval/rerank_loro_day.md\n"
          "  python scripts/probe_oracle_headroom.py "
          "--vis-cache runs/cache_day/gauss_vis_day_clean.pkl "
          "--out runs/eval/oracle_headroom_day.md")
    return rc


if __name__ == "__main__":
    sys.exit(main())
