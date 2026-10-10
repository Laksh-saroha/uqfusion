"""Phase 3 development caches + gate statistics under corruption v2, on every core.

Mirrors `p3dev_corruption_prep.py --multi` (same five checkpoints, same paired list, same
VIS draws 941-944, conf 0.001, imgsz 640), but corrupts with `corruptions_v2.py` and writes
to a SEPARATE tree, so nothing registered is touched:

    runs/cache_p3dev_v2/seed{k}/draw{v}_{v+10}/gauss_*.pkl
    runs/derived_p3dev_v2/{brightness,structure}/draw{v}_{v+10}/gauss_*.json

The layout is what `load_context(cache_dir=..., bright_dir=..., structure_dir=...)` reads, so
`p3_corrupt_cells.py`-style scoring can point at it unchanged. Clean caches and clean
statistics are hard-linked in from runs/cache_p3/ and runs/derived/ (they do not depend on
the corruption).

Parallelism, two levels:
* `--jobs` (draw, condition) units run at once, each one `build_cache_multi.py` process
  that corrupts each frame once and runs all five seeds on it, so the GPU sees several
  inference streams;
* inside each, `--workers` processes read and corrupt frames and compute the gate
  statistics in the same pass (no second replay).
Defaults split the machine's logical cores between the jobs.

pohang04 is spent: this builds development cells only.

    py -3.13 scripts/build_corruption_v2_dev.py                    # fog, lowlight, glare s2 x 4 draws
    py -3.13 scripts/build_corruption_v2_dev.py --conds fog fog_s1 --draws 941 --jobs 2
    py -3.13 scripts/build_corruption_v2_dev.py --limit 64 --out-tag _smoke   # quick end-to-end check
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from eval_night_veto import link_or_copy                        # noqa: E402
from uqfusion.config import resolve_gpu_python                  # noqa: E402
from uqfusion.eval.cache import load_cache                      # noqa: E402

SEEDS = (0, 1, 2, 3, 4)
VIS_DRAWS = (941, 942, 943, 944)
#: name -> (stem, kind, severity, modality, params). Names after the first five are the
#: sensitivity rows of docs/eval/corruption_v2/README.md (one constant moved each) and the
#: degraded-IR cells of the fallback-cost check; build them at one draw (--draws 941).
ALL_CONDS = {"fog": ("gauss_vis_paired_fog", "fog", 2, "vis", None),
             "lowlight": ("gauss_vis_paired_lowlight", "lowlight", 2, "vis", None),
             "glare": ("gauss_vis_paired_glare", "glare", 2, "vis", None),
             "fog_s1": ("gauss_vis_paired_fog_s1", "fog", 1, "vis", None),
             "ir_fog_s2": ("gauss_ir_paired_fog_s2", "fog", 2, "ir", None),
             "glare_ae0": ("gauss_vis_paired_glare_ae0", "glare", 2, "vis", {"AE_STRENGTH": 0.0}),
             "glare_ae1": ("gauss_vis_paired_glare_ae1", "glare", 2, "vis", {"AE_STRENGTH": 1.0}),
             "glare_i0half": ("gauss_vis_paired_glare_i0half", "glare", 2, "vis", {"I0_SCALE": 0.5}),
             "glare_i0x2": ("gauss_vis_paired_glare_i0x2", "glare", 2, "vis", {"I0_SCALE": 2.0}),
             # revision 2's glare: Lorentzian spread (beta 1) with its own fitted intensities, no AE
             "glare_rev2": ("gauss_vis_paired_glare_rev2", "glare", 2, "vis",
                            {"SUN_BETA": 1.0, "LAMP_BETA": 1.0, "AE_STRENGTH": 0.0,
                             "SUN": {1: (8.0, 7.0), 2: (20.0, 10.0), 3: (53.0, 14.0)},
                             "LAMP": {1: (2.586, 6.0), 2: (6.212, 9.0), 3: (6.212, 12.0)}}),
             "fog_ae0": ("gauss_vis_paired_fog_ae0", "fog", 2, "vis", {"AE_STRENGTH": 0.0}),
             "fog_ae1": ("gauss_vis_paired_fog_ae1", "fog", 2, "vis", {"AE_STRENGTH": 1.0}),
             "ir_fog_s2_b03": ("gauss_ir_paired_fog_s2_b03", "fog", 2, "ir", {"IR_BETA_RATIO": 0.3}),
             "ir_fog_s2_b10": ("gauss_ir_paired_fog_s2_b10", "fog", 2, "ir", {"IR_BETA_RATIO": 1.0}),
             # the weak-IR fallback's cost (scripts/v2_fallback_cost.py): degraded IR beside low-light VIS
             "ir_fog_s1": ("gauss_ir_paired_fog_s1", "fog", 1, "ir", None),
             "ir_fog_s3": ("gauss_ir_paired_fog_s3", "fog", 3, "ir", None),
             "ir_noise_s2": ("gauss_ir_paired_noise_s2", "noise", 2, "ir", None),
             "ir_noise_s3": ("gauss_ir_paired_noise_s3", "noise", 3, "ir", None)}
CLEAN_PKL = ("gauss_vis_paired_clean", "gauss_ir_paired_clean", "gauss_vis_train_clean", "gauss_ir_train_clean")
CLEAN_JSON = ("gauss_vis_paired_clean", "gauss_ir_paired_clean")
PV = ROOT / "runs/derived/paired_val_vis.txt"
LISTS = {"vis": PV, "ir": ROOT / "runs/derived/paired_val_ir.txt"}


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def cache_ok(out: Path, weights: Path, kind: str, sev: int, seed: int, n: int, batch: int = 1,
             params: dict | None = None) -> bool:
    from uqfusion.eval.corruptions_v2 import CODE_REV
    if not out.is_file():
        return False
    try:
        m = load_cache(out)[1]
    except Exception:                                   # noqa: BLE001 -- a torn file is simply rebuilt
        return False
    got = Path((m.get("weights") or [""])[0]).as_posix()
    return (got.endswith(weights.relative_to(ROOT).as_posix()) and m.get("corrupt") == kind
            and m.get("severity") == sev and m.get("corrupt_seed") == seed
            and m.get("corrupt_version") == "v2" and m.get("n_frames") == n
            and m.get("infer_batch", 1) == batch and m.get("corrupt_code") == CODE_REV
            and _norm(m.get("corrupt_params")) == _norm(params))


def _norm(params):
    """Params as they read back from a cache (JSON: string keys, lists for tuples)."""
    import json
    return json.loads(json.dumps(params)) if params else None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--draws", type=int, nargs="+", default=list(VIS_DRAWS), choices=VIS_DRAWS)
    ap.add_argument("--conds", nargs="+", default=["fog", "lowlight", "glare"], choices=list(ALL_CONDS))
    ap.add_argument("--jobs", type=int, default=4, help="(draw, condition) units at once (GPU streams)")
    ap.add_argument("--workers", type=int, default=None, help="corruption processes per job "
                    "(default: (logical cores - jobs - 1) // jobs)")
    ap.add_argument("--limit", type=int, default=None, help="first N frames only (smoke test)")
    ap.add_argument("--batch", type=int, default=1, help="frames per forward pass (1 = comparable with "
                    "the batch-1 clean caches; see build_cache_multi.py --batch)")
    ap.add_argument("--out-tag", default="", help="suffix for the output trees, e.g. _smoke")
    args = ap.parse_args()

    cache_root = ROOT / f"runs/cache_p3dev_v2{args.out_tag}"
    stats_root = ROOT / f"runs/derived_p3dev_v2{args.out_tag}"
    n_frames = sum(1 for ln in PV.read_text(encoding="utf-8").splitlines() if ln.strip())
    if args.limit:
        n_frames = min(n_frames, args.limit)
    cpus = os.cpu_count() or 4
    workers = args.workers or max(1, (cpus - args.jobs - 1) // args.jobs)
    gp = resolve_gpu_python(None)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")

    units = []
    for v in args.draws:
        for k in SEEDS:
            d = cache_root / f"seed{k}" / sub(v)
            d.mkdir(parents=True, exist_ok=True)
            if not args.limit:                           # a smoke tree would mix frame counts
                for stem in CLEAN_PKL:
                    link_or_copy(ROOT / f"runs/cache_p3/seed{k}/{stem}.pkl", d / f"{stem}.pkl")
        for kd in ("brightness", "structure"):
            sd = stats_root / kd / sub(v)
            sd.mkdir(parents=True, exist_ok=True)
            if not args.limit:
                for stem in CLEAN_JSON:
                    link_or_copy(ROOT / "runs/derived" / kd / f"{stem}.json", sd / f"{stem}.json")
        for c in args.conds:
            units.append((v, *ALL_CONDS[c]))
    if any(ALL_CONDS[c][3] == "ir" for c in args.conds):
        n_ir = sum(1 for ln in LISTS["ir"].read_text(encoding="utf-8").splitlines() if ln.strip())
        if n_ir != sum(1 for ln in PV.read_text(encoding="utf-8").splitlines() if ln.strip()):
            raise SystemExit("paired VIS and IR lists differ in length")

    log_dir = cache_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    def run(u) -> tuple[str, bool]:
        v, stem, kind, sev, mod, params = u
        cseed = v + 10 if mod == "ir" else v                  # IR draw = VIS draw + 10, as everywhere
        todo = [(ROOT / f"runs/phase3_stage2/p3_{mod}_seed{k}/weights/best.pt",
                 cache_root / f"seed{k}" / sub(v) / f"{stem}.pkl") for k in SEEDS]
        stats = [stats_root / kd / sub(v) / f"{stem}.json" for kd in ("brightness", "structure")]
        name = f"{sub(v)}/{stem}"
        if all(cache_ok(o, w, kind, sev, cseed, n_frames, args.batch, params) for w, o in todo) and all(p.is_file() for p in stats):
            print(f"[v2] {name} skip (verified)", flush=True)
            return name, True
        cmd = [gp, "-u", str(ROOT / "scripts/build_cache_multi.py"),
               "--weights", *[str(w) for w, _ in todo], "--outs", *[str(o) for _, o in todo],
               "--images-list", str(LISTS[mod]), "--imgsz", "640", "--conf", "0.001", "--data", mod,
               "--corrupt", kind, "--severity", str(sev), "--corrupt-seed", str(cseed),
               "--corrupt-version", "v2", "--modality", mod, "--workers", str(workers),
               "--stats-out", *[str(p) for p in stats]]
        if params:
            import json
            cmd += ["--params", json.dumps(params)]
        if args.limit:
            cmd += ["--limit", str(args.limit)]
        if args.batch != 1:
            cmd += ["--batch", str(args.batch)]
        t = time.time()
        print(f"[v2] {name} start ({workers} workers)", flush=True)
        with open(log_dir / f"{sub(v)}_{stem}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        ok = rc == 0 and all(cache_ok(o, w, kind, sev, cseed, n_frames, args.batch, params) for w, o in todo)
        print(f"[v2] {name} {'ok' if ok else f'FAIL rc={rc}'} in {time.time() - t:.0f}s "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
        return name, ok

    with ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(run, units))
    bad = [n for n, ok in res if not ok]
    print(f"[v2] {len(res) - len(bad)}/{len(res)} units ok in {(time.time() - t0) / 60:.1f} min"
          + (f"; FAILED: {bad}" if bad else ""), flush=True)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
