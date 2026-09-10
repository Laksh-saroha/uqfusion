"""§5.1 -- measure the largest batch that actually fits, per arm, on this card.

Pre-registered at `docs/prereg-phase3-retrain-2026-09-10.md` §5.1: "The largest-common-fit
batch is measured once and recorded in the manifest, and it is identical across every arm."
Amendment 3 re-anchors that on this laptop (RTX 4080 Laptop, 12,282 MiB) rather than the
MIG slice.

**Why this cannot be an OOM probe.** On Windows WDDM the driver does not raise OOM when a
run exceeds VRAM -- it pages the excess into host RAM and the run *succeeds*, slowly.
`phase1-experimental-record.md:688` measured `yolo26x` at batch 16 peaking 15.31 GB on a
12.28 GB card and running at 2.4 img/s, about **17x slower**. A probe that asks "did it
crash" would have called that batch a pass and handed Stage 2 a silent 17x slowdown across
every seed. So the measurement here is **seconds per image**, and the ceiling is the last
batch before throughput degrades -- memory is recorded alongside but does not decide.

Each batch runs in a fresh subprocess. The caching allocator's fragmentation survives within
a process, so probing several batch sizes in one would let an early large batch poison a
later small one and vice versa.

The arms are the two §5.1 trains: VIS `yolo26m` nc=2 and IR `yolo26m-p2feat` nc=1. The
common ceiling is the smaller of the two, and it is expected to be IR's -- the p2feat neck
carries a P2-level feature map, which is where the memory goes.

Usage:
    python scripts/probe_batch_ceiling.py --arm ir  --batches 8,10,12,14,16
    python scripts/probe_batch_ceiling.py --arm vis --batches 8,12,16,20,24

Writes `runs/eval/batch_ceiling_<arm>.json`. Run under the GPU interpreter
(`config.yaml: gpu_python`), never the project venv -- that one is CPU torch and would
report a ceiling for the wrong device.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

ARMS = {
    "vis": {"variant": "yolo26m",
            "data": "runs/derived/data_vis_stride2_p04out.yaml"},
    "ir":  {"variant": "yolo26m-p2feat",
            "data": "runs/derived/data_ir_shiponly_stride2.yaml"},
}

#: Fraction of the training set per probe. Small enough to be quick, large enough that
#: steady state dominates the warm-up epoch's first iterations. A small fraction also
#: makes setup a large share of wall time, which flattens the ms/img curve and can hide
#: a real throughput gradient -- the 15% IR ladder read 90-93 ms/img flat across batch
#: 8-16, and the full epoch then showed 39.7 down to 35.9. Use 1.0 for anything that
#: goes in the manifest.
#:
#: It was expected that a subset would also UNDER-read the peak, by missing the densest
#: frames the +/-0.35 GiB sawtooth tracks. Measured, it did not: IR's 15% subset read
#: 10.36 GiB at batch 16 against the full epoch's 10.12. The difference sits inside the
#: sawtooth, so subset-vs-full is a throughput concern here, not a memory one.
FRACTION = 0.03

#: This card, and the constants section 16 of phase1-experimental-record.md measured on it.
CARD_GIB = 11.99
DRIVER_OVERHEAD_GIB = 1.0    # driver-level sits ~1.0 GiB above `memory_reserved`
SAFE_HEADROOM_GIB = 1.5      # the record's stated safe operating margin


def run_one(arm: str, batch: int, fraction: float, workers: int) -> dict:
    """One batch size, in its own process. Returns the child's JSON verdict."""
    code = f"""
import json, sys, time, warnings
warnings.filterwarnings('ignore')
sys.path.insert(0, r'{ROOT / "src"}')
import torch
from uqfusion.config import load_config
from uqfusion.uq.train_gaussian import train_gaussian

cfg = load_config()
torch.cuda.reset_peak_memory_stats()
t0 = time.time()
err = None
try:
    train_gaussian(
        cfg, r'{ROOT / ARMS[arm]["data"]}', variant='{ARMS[arm]["variant"]}',
        seed=0, epochs=1, batch={batch}, workers={workers},
        run_name='probe_batch_{arm}_b{batch}', resume=False,
        out_subdir='probe_batch',
        train_overrides=dict(fraction={fraction}, val=False, plots=False,
                             save=False, cache=False, patience=0, verbose=False),
    )
except Exception as e:
    err = f'{{type(e).__name__}}: {{e}}'[:400]
el = time.time() - t0
print('@@' + json.dumps(dict(
    batch={batch}, elapsed=el, error=err,
    peak_reserved_gib=torch.cuda.max_memory_reserved()/1024**3,
    peak_alloc_gib=torch.cuda.max_memory_allocated()/1024**3)))
"""
    gp = _gpu_python()
    t0 = time.time()
    # Ultralytics emits bytes cp1252 cannot decode; without an explicit utf-8 codec and
    # `errors='replace'` the *parent* dies decoding the child's progress bar, which looks
    # exactly like a training failure and is not one.
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    p = subprocess.run([gp, "-c", code], capture_output=True, text=True,
                       encoding="utf-8", errors="replace",
                       cwd=ROOT, timeout=3600, env=env)
    out = [ln for ln in p.stdout.splitlines() if ln.startswith("@@")]
    if not out:
        return {"batch": batch, "error": (p.stderr or p.stdout)[-400:],
                "elapsed": time.time() - t0}
    return json.loads(out[-1][2:])


def _gpu_python() -> str:
    from uqfusion.config import load_config
    gp = load_config().get("gpu_python")
    if not gp:
        return sys.executable
    if not Path(gp).is_file():
        raise SystemExit(f"config gpu_python does not exist: {gp}")
    return gp


def n_frames(arm: str, fraction: float) -> int:
    lists = (ROOT / ARMS[arm]["data"]).read_text(encoding="utf-8")
    for ln in lists.splitlines():
        if ln.strip().startswith("train:"):
            p = Path(ln.split(":", 1)[1].strip())
            p = p if p.is_absolute() else ROOT / p
            n = sum(1 for x in p.read_text(encoding="utf-8").splitlines() if x.strip())
            return max(int(n * fraction), 1)
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=sorted(ARMS), required=True)
    ap.add_argument("--batches", default="8,10,12,16",
                    help="comma-separated ladder, ascending")
    ap.add_argument("--fraction", type=float, default=FRACTION)
    ap.add_argument("--workers", type=int, default=8,
                    help="8 measured better than 16 on this card "
                         "(phase1-experimental-record.md section 16)")
    args = ap.parse_args()

    batches = [int(b) for b in args.batches.split(",") if b.strip()]
    frames = n_frames(args.arm, args.fraction)
    print(f"[probe] arm={args.arm} variant={ARMS[args.arm]['variant']} "
          f"frames/probe~{frames:,} workers={args.workers}")

    rows = []
    for b in batches:
        r = run_one(args.arm, b, args.fraction, args.workers)
        r["s_per_img"] = (r["elapsed"] / frames) if frames and not r.get("error") else None
        rows.append(r)
        if r.get("error"):
            print(f"  batch {b:>3}: ERROR  {r['error'][:120]}")
        else:
            print(f"  batch {b:>3}: {r['elapsed']:7.1f}s  "
                  f"{r['s_per_img']*1000:6.1f} ms/img  "
                  f"peak {r['peak_reserved_gib']:5.2f} GiB")

    ok = [r for r in rows if r.get("s_per_img")]
    verdict = None
    if ok:
        best = min(r["s_per_img"] for r in ok)
        for r in ok:
            # Throughput criterion. Paging is not a 15% effect -- it was 17x -- so this
            # separates "past the sweet spot" from "spilling into host RAM" with room to
            # spare. On this card it often does not discriminate at all: the first IR
            # ladder ran 90-93 ms/img flat from batch 8 to 16, because these sizes are
            # not compute-bound here. A criterion that cannot fire must not be the only
            # one, or it silently endorses the largest batch tried.
            r["degraded"] = r["s_per_img"] > best * 1.15
            # Headroom criterion, which is what actually binds. `memory_reserved`
            # excludes CUDA context, cuDNN workspaces and driver overhead;
            # phase1-experimental-record.md section 16 measured a 9.70 GiB reserved band
            # sitting at ~10.7 GiB driver-level, so overhead is ~1.0 GiB, and calls
            # ~1.5 GiB of headroom the safe operating point.
            r["driver_gib"] = r["peak_reserved_gib"] + DRIVER_OVERHEAD_GIB
            r["headroom_gib"] = CARD_GIB - r["driver_gib"]
            r["tight"] = r["headroom_gib"] < SAFE_HEADROOM_GIB
        good = [r for r in ok if not r["degraded"] and not r["tight"]]
        verdict = max((r["batch"] for r in good), default=None)
    print(f"[probe] {args.arm} ceiling: batch {verdict}"
          f"   (throughput+headroom; {SAFE_HEADROOM_GIB} GiB headroom required)")
    for r in ok:
        flag = "TIGHT" if r["tight"] else ("SLOW" if r["degraded"] else "ok")
        print(f"        batch {r['batch']:>3}  driver ~{r['driver_gib']:5.2f} GiB  "
              f"headroom {r['headroom_gib']:5.2f} GiB  {flag}")

    out = ROOT / "runs" / "eval" / f"batch_ceiling_{args.arm}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"arm": args.arm, "variant": ARMS[args.arm]["variant"],
         "data": ARMS[args.arm]["data"], "fraction": args.fraction,
         "workers": args.workers, "frames_per_probe": frames,
         "rows": rows, "ceiling": verdict}, indent=2), encoding="utf-8")
    print(f"[probe] wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
