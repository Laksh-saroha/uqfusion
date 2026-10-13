"""End-to-end two-stream throughput (paper §9 item 18), measured rather than derived.

The paper's 57.0 FPS is one detector's `predict()`, and the two-stream 28.5 halves it; neither covers
the σ head, the image statistics, or the decision layer. This times the whole per-frame pipeline on
development paired frames, batch 1, sequentially:

  1. read the VIS and IR frames;
  2. VIS detector with the σ head and pooled features (`UQPredictor`, conf 0.001 as in every cache);
  3. IR detector, the same;
  4. image statistics both gates read (VIS brightness + structure, IR brightness + structure);
  5. the decision layer: the shipped `crossmodal26m` veto and WBF merge for one frame
     (`fuse_detections` with the shipped constants), and, separately, LLF's features and model.

Also times `ultralytics` `predict()` of one detector on the same frames, the protocol of the 57.0
figure, so the end-to-end number can be read as a ratio to it. The GPU clock is NOT pinned here (the
57.0 run pinned it at 1500 MHz); the clock is logged per stage. Phase 3 seed-0 checkpoints.

    py -3.13 scripts/measure_e2e_fps.py --n 500 --warmup 50
"""

from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

VIS_W = ROOT / "runs/phase3_stage2/p3_vis_seed0/weights/best.pt"
IR_W = ROOT / "runs/phase3_stage2/p3_ir_seed0/weights/best.pt"


def clock() -> str:
    try:
        return subprocess.run(["nvidia-smi", "--query-gpu=clocks.current.graphics,temperature.gpu",
                               "--format=csv,noheader"], capture_output=True, text=True, timeout=10).stdout.strip()
    except Exception:                                    # noqa: BLE001
        return ""


def main() -> int:
    import cv2
    import torch
    from ultralytics import YOLO

    from frame_structure import content_rows, frame_stats
    from llf import candidates, frame_features
    from uqfusion.uq.fusion import fuse_detections
    from uqfusion.uq.infer import UQPredictor

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--n", type=int, default=500)
    ap.add_argument("--warmup", type=int, default=50)
    ap.add_argument("--out", default="docs/eval/e2e_fps_2026-10-10.md")
    args = ap.parse_args()
    dev = "cuda:0"
    vis_list = (ROOT / "runs/derived/paired_val_vis.txt").read_text(encoding="utf-8").split()
    ir_list = (ROOT / "runs/derived/paired_val_ir.txt").read_text(encoding="utf-8").split()
    step = max(1, len(vis_list) // (args.n + args.warmup))
    pairs = list(zip(vis_list[::step], ir_list[::step]))[: args.n + args.warmup]
    H = json.loads((ROOT / "runs/derived/homography_ir_to_vis.json").read_text(encoding="utf-8"))["runs"]
    hb = {k: np.asarray(v["H_ir_canvas_to_vis_canvas"], float) for k, v in H.items()}
    run_of = lambda p: Path(p).stem.split("_")[0]
    vis_lo, vis_hi = content_rows("vis")
    ir_lo, ir_hi = content_rows("ir")
    llf_model = None
    mp = ROOT / "runs/llf/models/loro_pohang00.pkl"
    if mp.is_file():
        with open(mp, "rb") as fh:
            llf_model = pickle.load(fh)

    vis_p = UQPredictor(VIS_W, device=dev, conf=0.001)
    ir_p = UQPredictor(IR_W, device=dev, conf=0.001)
    stages = ["read", "vis_det", "ir_det", "stats", "gate_wbf", "llf"]
    t = {s: [] for s in stages}
    clocks = []

    def sync():
        torch.cuda.synchronize()

    for j, (vp, ip) in enumerate(pairs):
        rec = j >= args.warmup
        sync(); a = time.perf_counter()
        vim, iim = cv2.imread(vp), cv2.imread(ip)
        b = time.perf_counter()
        vr = vis_p(vim); sync()
        c = time.perf_counter()
        ir = ir_p(iim); sync()
        d = time.perf_counter()
        vs = frame_stats(cv2.cvtColor(vim, cv2.COLOR_BGR2GRAY)[vis_lo:vis_hi])
        is_ = frame_stats(cv2.cvtColor(iim, cv2.COLOR_BGR2GRAY)[ir_lo:ir_hi])
        e = time.perf_counter()
        # the shipped decision for one frame: thresholds on the statistics just computed, then the
        # merge with the shipped constants (w_vis 0.9926, IoU 0.85, support 0.30 / 0.5, passthrough)
        veto = bool(is_["p05"] > 41.5 and (vs["p05"] < 10.5 or vs["grad_gini"] < 0.4826))
        fuse_detections(vr, ir, 0.9926, 0.0074, vr["image_hw"], hb.get(run_of(vp)), 0.85, 0.0,
                        veto_vis=veto, single_passthrough=True, support_iou=0.30, support_gamma=0.5)
        f = time.perf_counter()
        if llf_model is not None:
            fs = np.array([vs["p05"], vs["p50"], vs["std"], np.log1p(vs["lap_var"]), vs["lap_over_var"],
                           vs["grad_gini"], vs["edge_density"], 1.0, is_["p05"], is_["p50"], is_["std"], 2.0])
            X, _B, _ = frame_features(candidates(vr, ir, hb.get(run_of(vp))), fs, vr["image_hw"])
            if len(X):
                llf_model["model"].predict(X[:, llf_model["cols"]])
        g = time.perf_counter()
        if rec:
            for s, v in zip(stages, (b - a, c - b, d - c, e - d, f - e, g - f)):
                t[s].append(v * 1000)
            if j % 100 == 0:
                clocks.append(clock())

    # the 57.0 protocol on the same frames: ultralytics predict(), one detector, file path in
    y = YOLO(str(VIS_W))
    for vp, _ in pairs[: args.warmup]:
        y.predict(vp, imgsz=640, device=dev, verbose=False)
    sync(); a = time.perf_counter()
    for vp, _ in pairs[args.warmup:]:
        y.predict(vp, imgsz=640, device=dev, verbose=False)
    sync()
    det_only = (time.perf_counter() - a) * 1000 / (len(pairs) - args.warmup)

    med = {s: float(np.median(v)) for s, v in t.items()}
    mean = {s: float(np.mean(v)) for s, v in t.items()}
    shipped = sum(mean[s] for s in ("read", "vis_det", "ir_det", "stats", "gate_wbf"))
    llf_total = sum(mean[s] for s in ("read", "vis_det", "ir_det", "stats", "llf"))
    rows = [f"| {s} | {mean[s]:.2f} | {med[s]:.2f} | {np.percentile(t[s], 95):.2f} |" for s in stages]
    L = [f"{len(pairs) - args.warmup} development paired frames (every {step}th), batch 1, sequential, after "
         f"{args.warmup} warm-up pairs; Phase 3 seed-0 checkpoints; {torch.cuda.get_device_name(0)}; GPU clock "
         f"not pinned (samples: {', '.join(clocks[:5])}).", "",
         "| stage | mean ms | median ms | p95 ms |", "|---|---:|---:|---:|", *rows, "",
         f"* **Shipped two-stream pipeline, end to end: {shipped:.1f} ms per frame pair = {1000 / shipped:.1f} FPS.**",
         f"* With LLF in place of the shipped gate: {llf_total:.1f} ms = {1000 / llf_total:.1f} FPS"
         + ("" if llf_model is not None else " (no LLF model on disk; stage not timed)") + ".",
         f"* One detector's `predict()` on the same frames and clock (the protocol of the paper's 57.0): "
         f"{det_only:.1f} ms = {1000 / det_only:.1f} FPS; halved, {500 / det_only:.1f} FPS for two streams.",
         f"* End-to-end ÷ derived two-stream: {(1000 / shipped) / (500 / det_only):.2f}."]
    out = ROOT / args.out
    out.write_text("# End-to-end two-stream throughput\n\n" + "\n".join(L) + "\n", encoding="utf-8")
    (out.with_suffix(".json")).write_text(json.dumps({"mean_ms": mean, "median_ms": med, "det_only_ms": det_only,
                                                      "shipped_ms": shipped, "llf_ms": llf_total, "clocks": clocks},
                                                     indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    raise SystemExit(main())
