"""Phase 3 §7 step 3 — build the pohang04 prediction caches. **Builds; never scores.**

Executes `docs/prereg-phase3-retrain-2026-09-10.md` Amendment 9 §A9.2 and §A9.5 item 3.
This file imports no AP, matching or fusion code. Reading an AP off these caches is the
single look, and happens only after the freeze commit, in a different script.

1. **Pair lists** from `Pohang_dataset/paired/pohang04_pairs.csv` (step 1), sorted by
   VIS frame name as `build_pairs.py` sorts: `runs/holdout_p04/p04_pairs_{vis,ir}.txt`
   plus a `p04_pairs_manifest.csv` in the `paired_val_manifest.csv` format. The row count
   must equal step 1's manifest.
2. **Caches, per Phase 3 system k = 0..4** (VIS seed k, IR seed k; A9.1), under
   `runs/holdout_p04/caches/seed{k}/`:
     clean/                  gauss_vis_paired_clean.pkl, gauss_ir_paired_clean.pkl
     draw{v}_{v+10}/         the 9 corrupted streams of the 11-cell grid, v in 941..944
   Kinds, severities and file stems are `gate_snms_draw_avg.CORRUPTED` verbatim; IR
   corruption seed = VIS seed + 10 (A9.2). 2 + 9 x 4 = 38 per system, 190 total.

Deliberately **not** under `runs/cache*`: `assert_holdout_excluded.py --caches` globs
that prefix, and holdout caches are pohang04-bearing by design. Filing them there would
turn the contamination gate permanently red and train everyone to ignore it.

Safety, from the draw-avg lesson that existence is not proof: a cache is built to a
`.tmp` path and renamed only after its own meta is checked against the job (weights,
images list, frame count, corruption kind / severity / seed). An existing cache is
skipped only if its meta passes the same check; otherwise it is rebuilt. One builder
per directory, via a lock file.

Usage:
    python scripts/holdout_p04_build_caches.py
"""

from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from uqfusion.config import load_config, resolve_gpu_python   # noqa: E402
from uqfusion.eval.cache import load_cache                      # noqa: E402
from gate_snms_draw_avg import CORRUPTED                        # noqa: E402

HP = ROOT / "runs" / "holdout_p04"
CACHES = HP / "caches"
PAIRS = ROOT / "Pohang_dataset" / "paired" / "pohang04_pairs.csv"
VIS_IMG = ROOT / "Pohang_dataset" / "visible" / "images" / "pohang04"
IR_IMG = ROOT / "Pohang_dataset" / "infrared" / "images" / "pohang04"
SEEDS = (0, 1, 2, 3, 4)
VIS_DRAWS = (941, 942, 943, 944)          # A9.2; IR = VIS + 10
IMGSZ, CONF = 640, 0.001


LOG = CACHES / "build.log"


def log(msg: str) -> None:
    line = f"{time.strftime('%Y-%m-%dT%H:%M:%S')}  {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")


def write_lists(write: bool) -> tuple[Path, Path, int]:
    """Build the pair lists; write them only when `write`, otherwise require that the files
    on disk already say exactly this. Parallel shards never write -- one `--lists-only`
    call does, first -- so no two processes race on the same file."""
    manifest = json.loads((HP / "step1_manifest.json").read_text(encoding="utf-8"))
    with open(PAIRS, newline="", encoding="utf-8") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["stereo_L_file"])
    if len(rows) != manifest["pairs"]["emitted"]:
        raise SystemExit(f"pair table has {len(rows)} rows, step 1 emitted {manifest['pairs']['emitted']}")
    vis = [VIS_IMG / r["stereo_L_file"] for r in rows]
    ir = [IR_IMG / r["ir_file"] for r in rows]
    missing = [p for p in vis + ir if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} paired images missing, e.g. {missing[0]}")
    lv, li = HP / "p04_pairs_vis.txt", HP / "p04_pairs_ir.txt"
    want_v = "\n".join(str(p) for p in vis) + "\n"
    want_i = "\n".join(str(p) for p in ir) + "\n"
    if not write:
        for p, want in ((lv, want_v), (li, want_i)):
            if not p.is_file() or p.read_text(encoding="utf-8") != want:
                raise SystemExit(f"{p} missing or stale -- run once with --lists-only first")
        return lv, li, len(rows)
    lv.write_text(want_v, encoding="utf-8")
    li.write_text(want_i, encoding="utf-8")
    with open(HP / "p04_pairs_manifest.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "run", "vis_image", "ir_image", "dt_ms"])
        for k, (v, i, r) in enumerate(zip(vis, ir, rows)):
            w.writerow([k, "pohang04", str(v), str(i), r["dt_ms"]])
    return lv, li, len(rows)


def jobs(lv: Path, li: Path) -> list[dict]:
    out = []
    for k in SEEDS:
        w = {"vis": ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt",
             "ir": ROOT / f"runs/phase3_stage2/p3_ir_seed{k}/weights/best.pt"}
        lst = {"vis": lv, "ir": li}
        for mod in ("vis", "ir"):
            out.append({"mod": mod, "weights": w[mod], "list": lst[mod], "kind": None, "sev": None,
                        "cseed": None, "out": CACHES / f"seed{k}/clean/gauss_{mod}_paired_clean.pkl"})
        for v in VIS_DRAWS:
            for mod, stem, kind, sev in CORRUPTED:
                cs = v + 10 if mod == "ir" else v
                out.append({"mod": mod, "weights": w[mod], "list": lst[mod], "kind": kind, "sev": sev,
                            "cseed": cs, "out": CACHES / f"seed{k}/draw{v}_{v + 10}/{stem}.pkl"})
    return out


def meta_ok(path: Path, job: dict, n: int) -> str | None:
    """None if the cache is exactly the job, else the reason it is not."""
    try:
        recs, m = load_cache(path)
    except Exception as exc:                                   # noqa: BLE001
        return f"unreadable ({type(exc).__name__})"
    checks = {
        "n_records": (len(recs), n),
        "weights": (Path(m["weights"][0]).resolve(), job["weights"].resolve()),
        "images_list": (Path(m["images_list"]).resolve(), job["list"].resolve()),
        "corrupt": (m.get("corrupt"), job["kind"]),
        "severity": (m.get("severity"), job["sev"]),
        "corrupt_seed": (m.get("corrupt_seed"), job["cseed"]),
        "conf": (m.get("conf"), CONF),
    }
    bad = [f"{k} {a!r} != {b!r}" for k, (a, b) in checks.items() if a != b]
    return "; ".join(bad) or None


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard", type=int, default=0, help="this worker's index")
    ap.add_argument("--of", type=int, default=1, help="number of parallel workers")
    ap.add_argument("--lists-only", action="store_true", help="write the pair lists and exit")
    args = ap.parse_args()
    global LOG
    LOG = CACHES / f"build.shard{args.shard}of{args.of}.log"
    if not 0 <= args.shard < args.of:
        raise SystemExit("--shard must be in [0, --of)")
    CACHES.mkdir(parents=True, exist_ok=True)
    # Parallel workers take disjoint jobs (job index mod --of), so no two ever write the
    # same cache; each holds its own lock so a second copy of the SAME shard is refused.
    lock = CACHES / f"builder.shard{args.shard}of{args.of}.lock"
    try:
        fd = os.open(lock, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode())
        os.close(fd)
    except FileExistsError:
        raise SystemExit(f"{lock} exists: another builder may be running (delete it if not)")
    try:
        lv, li, n = write_lists(write=args.lists_only)
        log(f"pair lists: {n} frames -> {lv.name}, {li.name}")
        if args.lists_only:
            return 0
        gp = str(resolve_gpu_python(load_config()))
        env = dict(os.environ)
        env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
        todo = [j for i, j in enumerate(jobs(lv, li)) if i % args.of == args.shard]
        log(f"shard {args.shard}/{args.of}: {len(todo)} caches planned")
        t0, failed = time.time(), []
        for j, job in enumerate(todo, 1):
            out: Path = job["out"]
            tag = f"[{j}/{len(todo)}] {out.relative_to(CACHES)}"
            if out.is_file():
                why = meta_ok(out, job, n)
                if why is None:
                    log(f"{tag} skip (meta verified)")
                    continue
                log(f"{tag} stale: {why} -- rebuilding")
                out.unlink()
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".pkl.tmp")
            tmp.unlink(missing_ok=True)
            cmd = [gp, "-u", str(ROOT / "scripts/build_cache.py"), "--source", "gaussian",
                   "--weights", str(job["weights"]), "--data", job["mod"],
                   "--images-list", str(job["list"]), "--imgsz", str(IMGSZ), "--conf", str(CONF),
                   "--out", str(tmp)]
            if job["kind"]:
                cmd += ["--corrupt", job["kind"], "--severity", str(job["sev"]),
                        "--corrupt-seed", str(job["cseed"])]
            t = time.time()
            with open(out.with_suffix(".log"), "w", encoding="utf-8") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
            why = "build exited %d" % rc if rc else meta_ok(tmp, job, n)
            if why:
                failed.append(str(out))
                log(f"{tag} FAILED: {why}")
                continue
            os.replace(tmp, out)
            log(f"{tag} ok in {time.time() - t:.0f}s (elapsed {(time.time() - t0) / 3600:.2f} h)")
        log(f"done: {len(todo) - len(failed)}/{len(todo)} verified" + (f"; FAILED {failed}" if failed else ""))
        return 1 if failed else 0
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    sys.exit(main())
