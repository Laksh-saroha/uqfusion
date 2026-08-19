"""Measure the fastest (batch, workers) for one variant on THIS GPU.

Why measured and not reasoned: on Windows/WDDM an over-large batch does not raise
OOM — the driver pages the excess into host RAM and the run "succeeds" ~17x slower
(docs/phase1-experimental-record.md §16). A sizing probe that only asks "did it
run?" therefore picks the worst setting. This one ranks by measured throughput and
reports both the torch-level and driver-level memory peaks, so the paging cliff is
visible as a throughput collapse rather than a crash.

It trains the real model on a real (tiny) subset of the real dataset for one epoch
per candidate, so the number includes augmentation, AMP and the dataloader.

Usage:
    python scripts/tune_batch.py --variant yolo26s --data runs/derived/data_vis_stride2.yaml
    python scripts/tune_batch.py --batches 16 24 32 --workers 8 12 --iters 60
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))  # repo is not pip-installed in the GPU interpreter

from uqfusion.config import load_config, resolve_data_yaml  # noqa: E402

PROBE_DIR = ROOT / "runs" / "tune"


class SmiSampler(threading.Thread):
    """Driver-level VRAM peak. torch's `memory_reserved` excludes the CUDA context,
    cuDNN workspaces and driver overhead — ~1.5 GB on this card — which is exactly
    the margin that decides whether a batch pages."""

    def __init__(self, interval: float = 0.5):
        super().__init__(daemon=True)
        self.interval, self.peak_mib, self._stop = interval, 0, threading.Event()

    def run(self) -> None:
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
                    text=True, stderr=subprocess.DEVNULL,
                )
                self.peak_mib = max(self.peak_mib, int(out.strip().splitlines()[0]))
            except Exception:  # noqa: BLE001 - sampling is metadata, never fatal
                pass
            self._stop.wait(self.interval)

    def stop(self) -> int:
        self._stop.set()
        self.join(timeout=3)
        return self.peak_mib


def build_probe_yaml(data_yaml: str, n_train: int, n_val: int) -> Path:
    """A tiny train/val pair drawn from the real lists, so the probe reads real images."""
    from uqfusion.data.lists import load_data_yaml, split_image_list

    data = load_data_yaml(data_yaml)
    train = [str(p) for p in split_image_list(data, "train")][:n_train]
    val = [str(p) for p in split_image_list(data, "val")][:n_val]
    if not train or not val:
        raise RuntimeError(f"{data_yaml} yielded no images (train={len(train)}, val={len(val)})")

    PROBE_DIR.mkdir(parents=True, exist_ok=True)
    tp, vp = PROBE_DIR / "probe_train.txt", PROBE_DIR / "probe_val.txt"
    tp.write_text("\n".join(train) + "\n", encoding="utf-8")
    vp.write_text("\n".join(val) + "\n", encoding="utf-8")
    yp = PROBE_DIR / "data_probe.yaml"
    yp.write_text(
        f"train: {tp}\nval: {vp}\ntest: {vp}\nnames:\n  0: ship\n  1: buoy\n", encoding="utf-8"
    )
    return yp


def probe(cfg: dict, variant: str, data_yaml: Path, batch: int, workers: int,
          gaussian: bool, imgsz: int, warmup_iters: int) -> dict:
    """One candidate. Returns throughput + both memory peaks, or an `error` key."""
    import torch
    from ultralytics import YOLO

    from uqfusion.bench.grid import resolve_device
    from uqfusion.uq.gaussian import DEFAULT_GAUSSIAN_CFG
    from uqfusion.uq.train_gaussian import make_gaussian_trainer

    name = f"{variant}_b{batch}_w{workers}{'_g' if gaussian else ''}"
    shutil.rmtree(PROBE_DIR / name, ignore_errors=True)

    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()

    # Timed from `warmup_iters` onward: the first batches carry cuDNN autotuning,
    # allocator growth and worker spin-up, none of which recur for 40k more.
    state = {"n": 0, "t0": None, "t1": None}

    def on_batch_end(trainer):
        state["n"] += 1
        if state["n"] == warmup_iters:
            state["t0"] = time.perf_counter()
        elif state["n"] > warmup_iters:
            state["t1"] = time.perf_counter()

    b = cfg["benchmark"]
    model = YOLO(f"{variant}.pt")
    model.add_callback("on_train_batch_end", on_batch_end)
    smi = SmiSampler()
    smi.start()
    err = None
    try:
        kwargs = dict(
            data=str(data_yaml), epochs=1, imgsz=imgsz, batch=batch, workers=workers,
            seed=0, deterministic=b["deterministic"], optimizer=b.get("optimizer", "auto"),
            amp=b["amp"], device=resolve_device(cfg), project=str(PROBE_DIR), name=name,
            exist_ok=True, verbose=False, plots=False, save=False, val=True,
        )
        if gaussian:
            g = {**DEFAULT_GAUSSIAN_CFG, **(cfg.get("gaussian") or {})}
            kwargs["trainer"] = make_gaussian_trainer(g)
        model.train(**kwargs)
    except Exception as exc:  # noqa: BLE001 - a failed candidate is a result, not a crash
        err = f"{type(exc).__name__}: {exc}"[:200]
    finally:
        smi_peak = smi.stop()

    torch_peak = torch.cuda.max_memory_reserved() / 2**30
    del model
    torch.cuda.empty_cache()

    timed = max(0, state["n"] - warmup_iters)
    secs = (state["t1"] - state["t0"]) if (state["t0"] and state["t1"]) else 0.0
    it_s = (timed / secs) if secs > 0 else 0.0
    return {
        "batch": batch, "workers": workers, "iters_timed": timed,
        "it_s": round(it_s, 3), "img_s": round(it_s * batch, 1),
        "torch_peak_gb": round(torch_peak, 2), "driver_peak_gb": round(smi_peak / 1024, 2),
        **({"error": err} if err else {}),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default=None)
    parser.add_argument("--data", default="runs/derived/data_vis_stride2.yaml")
    parser.add_argument("--variant", default="yolo26s")
    parser.add_argument("--batches", type=int, nargs="+", default=[16, 24, 32])
    parser.add_argument("--workers", type=int, nargs="+", default=[8])
    parser.add_argument("--imgsz", type=int, default=None)
    parser.add_argument("--iters", type=int, default=90, help="probe images = iters x batch")
    parser.add_argument("--warmup-iters", type=int, default=15)
    parser.add_argument("--n-val", type=int, default=64)
    parser.add_argument("--plain", action="store_true", help="probe without the sigma head")
    parser.add_argument("--out", default=str(PROBE_DIR / "tune_batch.json"))
    args = parser.parse_args()

    import torch
    if not torch.cuda.is_available():
        print("[tune] CUDA is not available — this probe is meaningless on CPU.", file=sys.stderr)
        return 2

    cfg = load_config(args.config)
    imgsz = args.imgsz if args.imgsz is not None else cfg["benchmark"]["imgsz"]
    data_yaml = resolve_data_yaml(cfg, args.data)
    total_gb = torch.cuda.get_device_properties(0).total_memory / 2**30
    print(f"[tune] {torch.cuda.get_device_name(0)} ({total_gb:.1f} GB) | "
          f"{args.variant} | imgsz {imgsz} | sigma head: {not args.plain}")

    probe_yaml = build_probe_yaml(data_yaml, n_train=args.iters * max(args.batches), n_val=args.n_val)

    results = []
    for workers in args.workers:
        for batch in args.batches:
            print(f"[tune] --- batch {batch}, workers {workers} ...", flush=True)
            r = probe(cfg, args.variant, probe_yaml, batch, workers,
                      gaussian=not args.plain, imgsz=imgsz, warmup_iters=args.warmup_iters)
            results.append(r)
            print(f"[tune] batch {r['batch']:>3} workers {r['workers']:>2} -> "
                  f"{r['img_s']:>6.1f} img/s ({r['it_s']} it/s) | "
                  f"torch peak {r['torch_peak_gb']} GB | driver peak {r['driver_peak_gb']} GB"
                  + (f" | ERROR {r['error']}" if "error" in r else ""), flush=True)

    ok = [r for r in results if "error" not in r and r["img_s"] > 0]
    print("\n[tune] ranked by throughput")
    print(f"  {'batch':>5} {'workers':>7} {'img/s':>8} {'torch GB':>9} {'driver GB':>10}")
    for r in sorted(ok, key=lambda r: -r["img_s"]):
        print(f"  {r['batch']:>5} {r['workers']:>7} {r['img_s']:>8.1f} "
              f"{r['torch_peak_gb']:>9} {r['driver_peak_gb']:>10}")

    if ok:
        best = max(ok, key=lambda r: r["img_s"])
        headroom = total_gb - best["driver_peak_gb"]
        print(f"\n[tune] fastest: batch {best['batch']}, workers {best['workers']} "
              f"({best['img_s']} img/s), driver headroom {headroom:.2f} GB")
        if headroom < 0.8:
            print("[tune] WARNING: <0.8 GB driver headroom. Any other GPU application "
                  "(browser, desktop compositor) can push this run into WDDM paging, "
                  "which does not OOM — it just gets ~17x slower. Prefer the next size down.")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"variant": args.variant, "imgsz": imgsz,
                               "gpu": torch.cuda.get_device_name(0),
                               "total_gb": round(total_gb, 2), "results": results}, indent=2),
                   encoding="utf-8")
    print(f"[tune] wrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
