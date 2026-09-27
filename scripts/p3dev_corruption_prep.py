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


def stats() -> None:
    hfs.verify(40)
    paths = hfs.read_list(PV)
    for v in VIS_DRAWS:
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


def caches() -> int:
    gp = resolve_gpu_python(None)
    log_dir = CACHE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    t0, fails = time.time(), 0
    for k in SEEDS:
        w = ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt"
        for v in VIS_DRAWS:
            d = CACHE / f"seed{k}" / sub(v)
            d.mkdir(parents=True, exist_ok=True)
            for stem in CLEAN_PKL:
                link_or_copy(ROOT / f"runs/cache_p3/seed{k}/{stem}.pkl", d / f"{stem}.pkl")
            for stem, kind, sev in CONDS:
                if not build(gp, w, stem, PV, kind, sev, v, d, log_dir):
                    fails += 1
        print(f"[cache] seed {k} done (elapsed {(time.time() - t0) / 60:.0f} min)", flush=True)
    print(f"[cache] {'ALL OK' if not fails else f'{fails} FAILED'}", flush=True)
    return 1 if fails else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--stats-only", action="store_true")
    g.add_argument("--caches-only", action="store_true",
                   help="caches do not read the statistics, so the two halves can run in parallel")
    args = ap.parse_args()
    if not args.caches_only:
        stats()
    return 0 if args.stats_only else caches()


if __name__ == "__main__":
    sys.exit(main())
