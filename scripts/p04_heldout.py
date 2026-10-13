"""The gate-free systems on the held-out run, pohang04 (2026-10-11).

Executes `docs/prereg-p04-heldout-gatefree-2026-10-11.md`. Third scoring of pohang04, disclosed in the
exposure ledger; `LOOK_TAKEN.json` and its mirror are never opened for writing.

Arms: `shipped` (reference), `union` and `full_tta` (primary), `ens5_union` and `lex` (descriptive),
each with its VIS-only and IR-only streams. Cells: the look's clean/clean and the v2 re-score's eight
cells, draws VIS 941-944 / IR 951-954. Assembly is the look's (`holdout_p04_look.system_context`,
development calibration injected); ensemble arms get the capability prior `seed_ensemble.py` computes on
development for the same arm.

New caches (`full_tta` only), under runs/holdout_p04/tta/ (outside runs/cache* like the look's trees):
    seed{k}/{clean|draw{v}_{v+10}}/gauss_{mod}_paired_{cond}_flip.pkl      every frame, boxes mapped back
    seed{k}/{clean|draw{v}_{v+10}}/gauss_ir_paired_{cond}_nlm.pkl          flagged frames only
The flip is of the corrupted, unrestored frame; restoration replaces a record only where the IR
merge-health statistic fires (`ir_d2 > ir_bound`, a frame statistic, seed-independent). Inference is
`predict_batch` at batch 8 with one letterbox per frame shared by the five checkpoints.

    py -3.13 scripts/p04_heldout.py flags
    py -3.13 scripts/p04_heldout.py build --jobs 2
    py -3.13 scripts/p04_heldout.py devcal
    py -3.13 scripts/p04_heldout.py score --stage clean
    py -3.13 scripts/p04_heldout.py report
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

HP = ROOT / "runs" / "holdout_p04"
TTA = HP / "tta"
LISTS = {"vis": HP / "p04_pairs_vis.txt", "ir": HP / "p04_pairs_ir.txt"}
SEEDS = (0, 1, 2, 3, 4)
VIS_DRAWS = (941, 942, 943, 944)
#: pohang04 v2 conditions (holdout_p04_v2_rescore.STREAMS), name -> (kind, severity)
VIS_CONDS = {"clean": None, "fog": ("fog", 2), "lowlight": ("lowlight", 2), "blur_s3": ("blur", 3),
             "noise_s2": ("noise", 2), "rain_s2": ("rain", 2)}
IR_CONDS = {"clean": None, "fog_s2": ("fog", 2), "blur_s2": ("blur", 2), "noise_s2": ("noise", 2)}
BATCH = int(os.environ.get("P04_BATCH", "8"))


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def where(cond: str, draw: int | None) -> str:
    return "clean" if cond == "clean" else sub(draw)


def flip_out(k: int, mod: str, cond: str, draw: int | None) -> Path:
    return TTA / f"seed{k}" / where(cond, draw) / f"gauss_{mod}_paired_{cond}_flip.pkl"


def nlm_out(k: int, cond: str, draw: int | None) -> Path:
    return TTA / f"seed{k}" / where(cond, draw) / f"gauss_ir_paired_{cond}_nlm.pkl"


def stats_dir(kind: str, cond: str, draw: int | None) -> Path:
    return HP / "derived" / kind / "clean" if cond == "clean" else HP / "v2" / "derived" / kind / sub(draw)


def images(mod: str) -> list[str]:
    return [ln.strip() for ln in LISTS[mod].read_text(encoding="utf-8").splitlines() if ln.strip()]


# ------------------------------------------------------------------ IR restoration flags
def ir_flags(cond: str, draw: int | None) -> np.ndarray:
    """`ctx.ir_d2 > ctx.ir_bound`, computed exactly as `load_context` does (ctx.py, IR self-check), from
    pohang04's own IR structure statistics and the frozen health model. Checked against a context in
    `score`."""
    sconst = json.loads((ROOT / "runs/eval/structure_constants.json").read_text(encoding="utf-8"))
    hm = sconst["axes"]["ir_health"]
    fr = json.loads((stats_dir("structure", cond, draw) / f"gauss_ir_paired_{cond}.json")
                    .read_text(encoding="utf-8"))["frames"]
    X = np.stack([[f[k] for k in hm["keys"]] for f in fr]).astype(float)
    for j, k in enumerate(hm["keys"]):
        if k in hm["log1p_keys"]:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Z = (X - np.asarray(hm["mean"])) / np.asarray(hm["std"])
    d2 = np.einsum("ij,jk,ik->i", Z, np.asarray(hm["precision"]), Z)
    return d2 > float(hm["bound"])


def cmd_flags(args) -> int:
    out = {}
    for cond in IR_CONDS:
        for d in ([None] if cond == "clean" else VIS_DRAWS):
            f = ir_flags(cond, d)
            out[f"{cond}|{d}"] = {"n": int(f.size), "flagged": int(f.sum())}
            print(f"[flags] IR {cond} {d}: {f.sum()}/{f.size} ({100 * f.mean():.1f}%)", flush=True)
    (TTA).mkdir(parents=True, exist_ok=True)
    (TTA / "flags.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


# ------------------------------------------------------------------ build
_W: dict = {}


def _init(spec: dict) -> None:
    import cv2
    cv2.setNumThreads(1)
    from uqfusion.eval.corruptions import make_corruption
    _W.update(spec)
    _W["tf"] = (make_corruption(spec["kind"], spec["sev"], spec["seed"], version="v2",
                                modality=spec["mod"], images=spec["images"]) if spec.get("kind") else None)


def _one(i: int):
    import cv2
    from restore import restore
    im = cv2.imread(_W["images"][i])
    if im is None:
        raise FileNotFoundError(_W["images"][i])
    if _W["tf"] is not None:
        im = _W["tf"](im, i)
    rest = restore(im, "nlm", "ir", _W["rows"]) if i in _W["flagged"] else None
    return i, np.ascontiguousarray(im[:, ::-1]), rest, im.shape[1]


def _chunk(idx):
    return [_one(i) for i in idx]


def frames(spec: dict, idx: list[int], workers: int, chunk: int = 8):
    from collections import deque
    chunks = [idx[a:a + chunk] for a in range(0, len(idx), chunk)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(spec,)) as ex:
        q: deque = deque()
        it = iter(chunks)
        for c in it:
            q.append(ex.submit(_chunk, c))
            if len(q) >= 3 * workers:
                break
        while q:
            yield from q.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(ex.submit(_chunk, nxt))


def _multi_batch(preds, ims: list) -> list[list[dict]]:
    """`predict_batch` for several checkpoints on the same frames: the letterbox and tensor are built
    once (the code of `PlainPredictor.predict_batch`, unchanged) and each model runs on that tensor."""
    import torch
    from ultralytics.utils import ops
    p0 = preds[0]
    ts = []
    for im0 in ims:
        im = p0._letterbox(image=im0)
        ts.append(torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float().div_(255.0))
    t = torch.stack(ts).to(p0.device)
    outs = []
    for p in preds:
        assert p.end2end and p.has_sigma
        with torch.no_grad():
            out = p.model(t)
        y = out[0] if isinstance(out, (tuple, list)) else out
        recs = []
        for im0, yi in zip(ims, y):
            det = yi[yi[:, 4] >= p.conf]
            h0, w0 = im0.shape[:2]
            boxes = det[:, :4].clone()
            if boxes.shape[0]:
                boxes = ops.scale_boxes(t.shape[2:], boxes, im0.shape)
            gain = min(t.shape[2] / h0, t.shape[3] / w0)
            det_c = det.cpu()
            recs.append({"boxes_xyxy": boxes.cpu().numpy(), "conf": det_c[:, 4].numpy(),
                         "cls": det_c[:, 5].numpy().astype(int), "image_hw": (h0, w0),
                         "sigma_ltrb": (det_c[:, 6:10] / gain).numpy()})
        outs.append(recs)
    return outs


class MultiPipe:
    """`_multi_batch`, pipelined: a batch's letterbox, pinned upload and five forwards are queued, and
    only then is the PREVIOUS batch read back, so the GPU never waits on Python. Same inputs, same
    kernels, same order of operations per model as `_multi_batch` (checked bit-identical)."""

    def __init__(self, preds):
        self.preds, self.pending = preds, None

    def _launch(self, ims):
        import torch
        p0 = self.preds[0]
        ts = []
        for im0 in ims:
            im = p0._letterbox(image=im0)
            ts.append(torch.from_numpy(np.ascontiguousarray(im[..., ::-1].transpose(2, 0, 1))).float().div_(255.0))
        t = torch.stack(ts).pin_memory().to(p0.device, non_blocking=True)
        with torch.no_grad():
            ys = [p.model(t) for p in self.preds]
        return t.shape[2:], [y[0] if isinstance(y, (tuple, list)) else y for y in ys]

    def _read(self, ims, tag, tshape, ys):
        from ultralytics.utils import ops
        outs = []
        for p, y in zip(self.preds, ys):
            recs = []
            for im0, yi in zip(ims, y):
                det = yi[yi[:, 4] >= p.conf]
                h0, w0 = im0.shape[:2]
                boxes = det[:, :4].clone()
                if boxes.shape[0]:
                    boxes = ops.scale_boxes(tshape, boxes, im0.shape)
                gain = min(tshape[0] / h0, tshape[1] / w0)
                det_c = det.cpu()
                recs.append({"boxes_xyxy": boxes.cpu().numpy(), "conf": det_c[:, 4].numpy(),
                             "cls": det_c[:, 5].numpy().astype(int), "image_hw": (h0, w0),
                             "sigma_ltrb": (det_c[:, 6:10] / gain).numpy()})
            outs.append(recs)
        return tag, outs

    def submit(self, ims, tag):
        """Queue `ims`; return the previous batch's (tag, per-model records), or None."""
        nxt = (ims, tag, *self._launch(ims))
        prev, self.pending = self.pending, nxt
        return None if prev is None else self._read(*prev)

    def flush(self):
        prev, self.pending = self.pending, None
        return None if prev is None else self._read(*prev)


def _f32(rec: dict) -> dict:
    return {k: (v.astype(np.float32) if isinstance(v, np.ndarray) and v.dtype == np.float16 else v)
            for k, v in rec.items()}


def setup_fp16(preds) -> None:
    """fp16 mode (amendment of 2026-10-11): autocast fp16, channels_last weights, cuDNN autotuning.
    Priced on development before use: the full_tta stream moves by at most 0.0003 ship AP."""
    import torch
    torch.backends.cudnn.benchmark = True
    for p in preds:
        p.model.to(memory_format=torch.channels_last)


def amp(enabled: bool):
    import contextlib
    import torch
    return torch.autocast("cuda", dtype=torch.float16) if enabled else contextlib.nullcontext()


def _unflip(rec: dict, w: int) -> dict:
    from restore import unflip
    return unflip(_f32(rec), w)


def cmd_build_unit(args) -> int:
    import torch
    from uqfusion.eval.corruptions_v2 import CODE_REV
    from uqfusion.uq.infer import UQPredictor
    from restore import _rows
    mod, cond, draw = args.mod, args.cond, (None if args.cond == "clean" else args.draw)
    kc = (VIS_CONDS if mod == "vis" else IR_CONDS)[cond]
    lst = images(mod)
    n = len(lst)
    flagged = set(np.flatnonzero(ir_flags(cond, draw)).tolist()) if mod == "ir" else set()
    seed = (draw or 0) + (10 if mod == "ir" else 0)
    spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": seed, "mod": mod,
            "images": lst, "rows": _rows(mod), "flagged": flagged}
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_{mod}_seed{k}/weights/best.pt", device=0,
                         imgsz=640, conf=0.001) for k in SEEDS]
    if args.fp16:
        setup_fp16(preds)
    flips = [[None] * n for _ in SEEDS]
    nlms = [{} for _ in SEEDS]
    buf_f, buf_r = [], []
    t0 = time.time()

    def flush_f():
        if buf_f:
            with amp(args.fp16):
                out = _multi_batch(preds, [im for _, im, _ in buf_f])
            for j in range(len(SEEDS)):
                for (i, _im, w), r in zip(buf_f, out[j]):
                    flips[j][i] = _unflip(r, w)
            buf_f.clear()

    def flush_r():
        if buf_r:
            with amp(args.fp16):
                out = _multi_batch(preds, [im for _, im in buf_r])
            for j in range(len(SEEDS)):
                for (i, _im), r in zip(buf_r, out[j]):
                    nlms[j][i] = _f32(r)
            buf_r.clear()

    for c, (i, fim, rim, w) in enumerate(frames(spec, list(range(n)), args.workers)):
        buf_f.append((i, fim, w))
        if rim is not None:
            buf_r.append((i, rim))
        if len(buf_f) >= BATCH:
            flush_f()
        if len(buf_r) >= BATCH:
            flush_r()
        if (c + 1) % 1000 == 0:
            print(f"[unit] {mod}/{cond}/{draw} {c + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
    flush_f()
    flush_r()
    meta = {"images_list": str(LISTS[mod]), "mod": mod, "cond": cond, "draw": draw, "imgsz": 640, "conf": 0.001,
            "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
            "corrupt_seed": seed if kc else None, "corrupt_version": "v2" if kc else None,
            "corrupt_code": CODE_REV if kc else None, "infer_batch": BATCH,
            "built_by": "p04_heldout.py build-unit", "torch": torch.__version__,
            "precision": "fp16 autocast, channels_last, cudnn.benchmark" if args.fp16 else "fp32"}
    for j, k in enumerate(SEEDS):
        w = str(ROOT / f"runs/phase3_stage2/p3_{mod}_seed{k}/weights/best.pt")
        o = flip_out(k, mod, cond, draw)
        o.parent.mkdir(parents=True, exist_ok=True)
        assert all(r is not None for r in flips[j])
        tmp = o.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump({"meta": {**meta, "weights": w, "view": "hflip of the unrestored frame, boxes mapped back"},
                         "records": flips[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, o)
        if mod == "ir":
            o = nlm_out(k, cond, draw)
            tmp = o.with_suffix(".tmp")
            with open(tmp, "wb") as fh:
                pickle.dump({"meta": {**meta, "weights": w, "view": "NLM-restored frame, flagged frames only",
                                      "flagged": sorted(flagged)}, "records": nlms[j]}, fh,
                            protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(tmp, o)
    print(f"[unit] {mod}/{cond}/{draw} done in {time.time() - t0:.0f}s ({n} frames, {len(flagged)} restored)",
          flush=True)
    return 0


def cmd_bench_unit(args) -> int:
    """The build-unit pipeline on `--frames` frames from `--offset`, nothing written. Loads the models,
    waits at a file barrier until `--procs` processes are ready, then times the frames (decode,
    corruption, restoration in workers; one letterbox per frame; five checkpoints at batch 8)."""
    import torch
    from uqfusion.uq.infer import UQPredictor
    from restore import _rows
    torch.backends.cudnn.benchmark = bool(args.cudnn_bench)
    mod, cond, draw = args.mod, args.cond, (None if args.cond == "clean" else args.draw)
    kc = (VIS_CONDS if mod == "vis" else IR_CONDS)[cond]
    lst = images(mod)
    idx = list(range(args.offset, args.offset + args.frames))
    flagged = set(np.flatnonzero(ir_flags(cond, draw)).tolist()) if mod == "ir" else set()
    spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": (draw or 0) + (10 if mod == "ir" else 0),
            "mod": mod, "images": lst, "rows": _rows(mod), "flagged": flagged}
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_{mod}_seed{k}/weights/best.pt", device=0,
                         imgsz=640, conf=0.001) for k in SEEDS]
    if args.fp16:
        setup_fp16(preds)
    for _ in range(3):
        with amp(args.fp16):
            _multi_batch(preds, [np.zeros((640, 640, 3), np.uint8)] * BATCH)         # warm-up
    bar = Path(args.barrier)
    (bar / f"ready_{os.getpid()}").write_text("1")
    while len(list(bar.glob("ready_*"))) < args.procs:
        time.sleep(0.05)
    t0 = time.time()
    n_pass, buf_f, buf_r = 0, [], []
    pipe = MultiPipe(preds) if args.pipe else None

    def go(buf):
        with amp(args.fp16):
            if pipe is None:
                _multi_batch(preds, list(buf))
            else:
                pipe.submit(list(buf), None)

    for i, fim, rim, w in frames(spec, idx, args.workers):
        buf_f.append(fim)
        if rim is not None:
            buf_r.append(rim)
        for buf in (buf_f, buf_r):
            if len(buf) >= BATCH:
                go(buf)
                n_pass += len(buf) * len(SEEDS)
                buf.clear()
    for buf in (buf_f, buf_r):
        if buf:
            go(buf)
            n_pass += len(buf) * len(SEEDS)
    if pipe is not None:
        pipe.flush()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(json.dumps({"frames": len(idx), "passes": n_pass, "seconds": dt,
                      "max_mem_gb": torch.cuda.max_memory_allocated() / 1e9}), flush=True)
    return 0


def cmd_bench(args) -> int:
    """`--procs` concurrent bench-unit processes per setting; GPU utilisation and memory sampled by
    nvidia-smi while they run. Throughput = all frames / slowest process."""
    import shutil
    import threading
    from uqfusion.config import resolve_gpu_python
    gp = resolve_gpu_python(None)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    rows = []
    for P in args.procs:
        bar = TTA / "bench_barrier"
        shutil.rmtree(bar, ignore_errors=True)
        bar.mkdir(parents=True)
        procs = [subprocess.Popen([gp, "-u", str(ROOT / "scripts/p04_heldout.py"), "bench-unit", "--mod", args.mod,
                                   "--cond", args.cond, "--draw", str(args.draw), "--frames", str(args.frames),
                                   "--offset", str(1000 + j * 2000), "--workers", str(args.workers),
                                   "--procs", str(P), "--barrier", str(bar)] + (["--cudnn-bench"] if args.cudnn_bench else [])
                                  + (["--fp16"] if args.fp16 else []) + (["--pipe"] if args.pipe else []),
                                  stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, cwd=str(ROOT), env=env)
                 for j in range(P)]
        samples, stop = [], threading.Event()

        def sample():
            while len(list(bar.glob("ready_*"))) < P and not stop.is_set():
                time.sleep(0.1)
            while not stop.is_set():
                q = subprocess.run(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used",
                                    "--format=csv,noheader,nounits"], capture_output=True, text=True).stdout
                u, m = (float(x) for x in q.strip().split(","))
                samples.append((u, m))
                time.sleep(0.5)
        th = threading.Thread(target=sample)
        th.start()
        outs = [json.loads(p.communicate()[0].strip().splitlines()[-1]) for p in procs]
        stop.set()
        th.join()
        dt = max(o["seconds"] for o in outs)
        fr = sum(o["frames"] for o in outs)
        ps = sum(o["passes"] for o in outs)
        su = np.asarray(samples) if samples else np.zeros((1, 2))
        row = {"procs": P, "frames_per_s": fr / dt, "passes_per_s": ps / dt, "gpu_util_mean": float(su[:, 0].mean()),
               "vram_peak_mib": float(su[:, 1].max()), "seconds": dt}
        rows.append(row)
        print(json.dumps(row), flush=True)
    out = TTA / f"bench_{args.mod}_{args.cond}{'_cudnnbench' if args.cudnn_bench else ''}{'_fp16' if args.fp16 else ''}{'_pipe' if args.pipe else ''}.json"
    out.write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return 0


def units() -> list[tuple[str, str, int | None]]:
    u = [("vis", "clean", None), ("ir", "clean", None)]
    for d in VIS_DRAWS:
        u += [("vis", c, d) for c in VIS_CONDS if c != "clean"]
        u += [("ir", c, d) for c in IR_CONDS if c != "clean"]
    return u


def unit_done(mod, cond, d) -> bool:
    outs = [flip_out(k, mod, cond, d) for k in SEEDS] + ([nlm_out(k, cond, d) for k in SEEDS] if mod == "ir" else [])
    return all(o.is_file() for o in outs)


def cmd_build(args) -> int:
    from concurrent.futures import ThreadPoolExecutor
    from uqfusion.config import resolve_gpu_python
    gp = resolve_gpu_python(None)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    (TTA / "logs").mkdir(parents=True, exist_ok=True)
    todo = [u for u in units() if not unit_done(*u)]
    print(f"[build] {len(todo)} units to build", flush=True)
    t0 = time.time()

    def run(u):
        mod, cond, d = u
        heavy = (VIS_CONDS if mod == "vis" else IR_CONDS)[cond]
        w = 6 if heavy and heavy[0] in ("blur", "rain") else (8 if not heavy else args.workers)
        cmd = [gp, "-u", str(ROOT / "scripts/p04_heldout.py"), "build-unit", "--mod", mod, "--cond", cond,
               "--draw", str(d or 0), "--workers", str(w)] + (["--fp16"] if args.fp16 else [])
        print(f"[build] {u} start", flush=True)
        with open(TTA / "logs" / f"{mod}_{cond}_{d}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        ok = rc == 0 and unit_done(*u)
        print(f"[build] {u} {'ok' if ok else f'FAIL rc={rc}'} (elapsed {(time.time() - t0) / 60:.1f} min)",
              flush=True)
        return u, ok

    with ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(run, todo))
    bad = [u for u, ok in res if not ok]
    print(f"[build] {len(res) - len(bad)}/{len(res)} ok" + (f"; FAILED {bad}" if bad else ""), flush=True)
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("flags")
    u = sp.add_parser("build-unit")
    u.add_argument("--mod", choices=["vis", "ir"], required=True)
    u.add_argument("--cond", required=True)
    u.add_argument("--draw", type=int, default=0)
    u.add_argument("--workers", type=int, default=6)
    u.add_argument("--fp16", action="store_true")
    b = sp.add_parser("build")
    b.add_argument("--jobs", type=int, default=2)
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--fp16", action="store_true")
    bu = sp.add_parser("bench-unit")
    bu.add_argument("--mod", choices=["vis", "ir"], required=True)
    bu.add_argument("--cond", required=True)
    bu.add_argument("--draw", type=int, default=0)
    bu.add_argument("--frames", type=int, default=400)
    bu.add_argument("--offset", type=int, default=0)
    bu.add_argument("--workers", type=int, default=6)
    bu.add_argument("--procs", type=int, default=1)
    bu.add_argument("--barrier", required=True)
    bu.add_argument("--cudnn-bench", action="store_true")
    bu.add_argument("--fp16", action="store_true")
    bu.add_argument("--pipe", action="store_true")
    be = sp.add_parser("bench")
    be.add_argument("--mod", choices=["vis", "ir"], required=True)
    be.add_argument("--cond", required=True)
    be.add_argument("--draw", type=int, default=941)
    be.add_argument("--frames", type=int, default=400)
    be.add_argument("--workers", type=int, default=6)
    be.add_argument("--procs", type=int, nargs="+", default=[1, 2, 3, 4])
    be.add_argument("--cudnn-bench", action="store_true")
    be.add_argument("--fp16", action="store_true")
    be.add_argument("--pipe", action="store_true")
    args = ap.parse_args()
    return {"flags": cmd_flags, "build-unit": cmd_build_unit, "build": cmd_build,
            "bench-unit": cmd_bench_unit, "bench": cmd_bench}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
