"""Development corruption caches and image statistics for the five Phase 3 systems.

Prep only: builds inputs, scores nothing, reads no label. The Phase 3 development caches
under `runs/cache_p3/` are clean-only, so the paper's "fused ≥ max(VIS, IR)" claim on the
corrupted day cells has never been measured on the checkpoints §7 reports. This builds
what that measurement needs, for the VIS conditions of the original eight-cell claim —
`fog`, `lowlight`, `glare` (severity 2, the shipped kinds) — at A9.2's four Phase 3 VIS
draws 941–944.

Two inputs per (condition, draw), because `crossmodal26m` reads image statistics as well
as detections:

1. **Image statistics** (brightness + structure). Pixel arithmetic, no checkpoint, so one
   file per stream serves all five seeds. Computed with `holdout_p04_frame_stats.measure`,
   after its `verify` reproduces development files bit-exactly. Run under `.venv`.
2. **Detection caches**, one per seed, via `build_cache.py` on the GPU interpreter. An
   existing cache is reused only if its meta names the same weights, corruption and seed.

Layout, mirroring the look's per-draw substrate (nothing under `runs/cache_p3/` or
`runs/derived/` is written; clean files are hard-linked in so each draw directory is a
complete `load_context` input):

    runs/cache_p3dev/seed{k}/draw{v}_{v+10}/gauss_*.pkl
    runs/derived_p3dev/{brightness,structure}/draw{v}_{v+10}/gauss_*.json

Usage:
    python scripts/p3dev_corruption_prep.py            # stats, then caches
    python scripts/p3dev_corruption_prep.py --stats-only     # CPU half
    python scripts/p3dev_corruption_prep.py --caches-only    # GPU half, in parallel
    python scripts/p3dev_corruption_prep.py --caches-only --shard 3/10   # one of ten builders
    python scripts/p3dev_corruption_prep.py --stats-only --draws 942     # one draw's statistics
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2                                                        # noqa: E402

import holdout_p04_frame_stats as hfs                             # noqa: E402
from eval_night_veto import build, link_or_copy                   # noqa: E402
from uqfusion.config import resolve_gpu_python                    # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
VIS_DRAWS = (941, 942, 943, 944)
CONDS = [("gauss_vis_paired_fog", "fog", 2),
         ("gauss_vis_paired_lowlight", "lowlight", 2),
         ("gauss_vis_paired_glare", "glare", 2)]
CLEAN_PKL = ("gauss_vis_paired_clean", "gauss_ir_paired_clean",
             "gauss_vis_train_clean", "gauss_ir_train_clean")
CLEAN_JSON = ("gauss_vis_paired_clean", "gauss_ir_paired_clean")
PV = ROOT / "runs/derived/paired_val_vis.txt"
CACHE = ROOT / "runs/cache_p3dev"
STATS = ROOT / "runs/derived_p3dev"


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def stats(draws=VIS_DRAWS) -> None:
    hfs.verify(40)
    paths = hfs.read_list(PV)
    for v in draws:
        for d in ("brightness", "structure"):
            (STATS / d / sub(v)).mkdir(parents=True, exist_ok=True)
            for stem in CLEAN_JSON:
                link_or_copy(ROOT / "runs/derived" / d / f"{stem}.json", STATS / d / sub(v) / f"{stem}.json")
        for stem, kind, sev in CONDS:
            outs = {d: STATS / d / sub(v) / f"{stem}.json" for d in ("brightness", "structure")}

            def ok(p: Path) -> bool:
                if not p.is_file():
                    return False
                h = json.loads(p.read_text(encoding="utf-8"))
                return (h["corrupt"], h["severity"], h["corrupt_seed"], h["n_frames"]) == (kind, sev, v, len(paths))
            if all(ok(p) for p in outs.values()):
                print(f"[stats] {sub(v)}/{stem} skip (verified)", flush=True)
                continue
            t = time.time()
            rows, br, st = hfs.measure(paths, "vis", kind, sev, v)
            for d, frames in (("brightness", br), ("structure", st)):
                payload = {"cache": f"runs/cache_p3dev/seed*/{sub(v)}/{stem}.pkl", "modality": "vis",
                           "content_rows": list(rows), "corrupt": kind, "severity": sev, "corrupt_seed": v,
                           "n_frames": len(frames), "numpy": np.__version__, "opencv": cv2.__version__,
                           "frames": frames}
                tmp = outs[d].with_suffix(".json.tmp")
                tmp.write_text(json.dumps(payload), encoding="utf-8")
                os.replace(tmp, outs[d])
            print(f"[stats] {sub(v)}/{stem} {len(br)} frames in {time.time() - t:.0f}s", flush=True)


def caches(draws=VIS_DRAWS, shard: tuple[int, int] = (0, 1)) -> int:
    """Build the caches of one shard. The unit of work is a (seed, draw) DIRECTORY, so no two
    shards ever write, link or log into the same place.

    Draw-major order: directories are numbered draw by draw, so with enough shards the early
    draws finish first as complete five-seed sets. Throughput is bound by the fog corruption,
    ~290 ms/frame and single-threaded (one OpenCV thread is as fast as 32), so a ~11 min cache
    uses one core of 32 while the GPU idles; shards are how the build uses the machine.
    """
    i, n = shard
    gp = resolve_gpu_python(None)
    dirs = [(v, k) for v in draws for k in SEEDS]
    mine = [dk for j, dk in enumerate(dirs) if j % n == i]
    t0, fails = time.time(), 0
    print(f"[cache] shard {i}/{n}: {[f'seed{k}/{sub(v)}' for v, k in mine]}", flush=True)
    for v, k in mine:
        w = ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt"
        d = CACHE / f"seed{k}" / sub(v)
        d.mkdir(parents=True, exist_ok=True)
        log_dir = CACHE / "logs" / f"seed{k}_{sub(v)}"
        log_dir.mkdir(parents=True, exist_ok=True)
        for stem in CLEAN_PKL:
            link_or_copy(ROOT / f"runs/cache_p3/seed{k}/{stem}.pkl", d / f"{stem}.pkl")
        for stem, kind, sev in CONDS:
            if not build(gp, w, stem, PV, kind, sev, v, d, log_dir):
                fails += 1
        print(f"[cache] seed{k}/{sub(v)} done (elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    print(f"[cache] shard {i}/{n} {'ALL OK' if not fails else f'{fails} FAILED'}", flush=True)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stats-only", action="store_true")
    g.add_argument("--caches-only", action="store_true",
                   help="caches do not read the statistics, so the two halves can run in parallel")
    ap.add_argument("--draws", type=int, nargs="+", default=list(VIS_DRAWS), choices=VIS_DRAWS)
    ap.add_argument("--shard", default="0/1", help="i/n: build every n-th (seed, draw) directory from i")
    args = ap.parse_args()
    i, n = (int(x) for x in args.shard.split("/"))
    if not 0 <= i < n:
        raise SystemExit(f"bad --shard {args.shard}")
    if not args.caches_only:
        stats(tuple(args.draws))
    return 0 if args.stats_only else caches(tuple(args.draws), (i, n))


if __name__ == "__main__":
    sys.exit(main())
