"""Restoration before detection (2026-10-10): image-only repairs ahead of the frozen detectors.

Nothing here retrains a detector. Each corrupted frame is replayed exactly as the cache builder
replays it (`make_corruption(kind, sev, seed, version="v2", ...)` at the frame's index in the
paired list), passed through a classical restoration, and detected by the same Phase 3
checkpoints at the same settings (imgsz 640, conf 0.001, batch 1). The gate statistics the
fusion rule reads stay those of the corrupted frame: restoration changes what the detectors
see, not what the gate measures.

Restorations (all image-only; none reads the corruption's parameters, depth maps or labels):

  VIS  ae        auto-exposure: linear-light gain to a mean of 0.18, brighten only, at most x8
       clahe     CLAHE on L* (clip 2, 8x8 tiles)
       dcp       dark-channel-prior dehaze (He et al. 2009; patch 15, omega 0.95, t >= 0.15,
                 guided-filter refinement r 40, eps 1e-3)
       bil_dcp   bilateral denoise (d 5, sigma 25/5), then dcp
  IR   nlm       non-local means, h = 1.2 x the Immerkaer noise sd, clipped to [3, 40]
       median3   3x3 median
       clahe     CLAHE (clip 2, 8x8 tiles)
       nlm_clahe nlm, then clahe

Subcommands:
  dev         draw 941 (the descriptive draw), every 4th paired frame, three systems: AP per
              (condition, method) against the unrestored stream. Picks nothing by itself.
  trainstats  gate statistics on clean TRAIN frames (the gate's calibration set)
  build       full restored caches for one method per modality on the registered draws
              (docs/prereg-restore-2026-10-10.md); written beside, never over, the raw caches
  score       the registered comparison

    py -3.13 scripts/restore.py dev --workers 8
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from collections import deque
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")

PAIRED = {"vis": ROOT / "runs/derived/paired_val_vis.txt", "ir": ROOT / "runs/derived/paired_val_ir.txt"}
#: name -> (kind, severity, modality); None = clean
VIS_CONDS = {"clean": None, "fog": ("fog", 2), "fog_s1": ("fog", 1), "lowlight": ("lowlight", 2),
             "glare": ("glare", 2)}
IR_CONDS = {"clean": None, "fog_s1": ("fog", 1), "fog_s2": ("fog", 2), "fog_s3": ("fog", 3),
            "noise_s2": ("noise", 2), "noise_s3": ("noise", 3)}
METHODS = {"vis": ("ae", "clahe", "dcp", "bil_dcp"), "ir": ("nlm", "median3", "clahe", "nlm_clahe")}
SHIP = 0
IR_NMS = 0.70          # the shipped IR stream's detector-side NMS (ctx.load_context, crossmodal)


# ---------------------------------------------------------------- restorations
def _rows(modality: str) -> tuple[int, int]:
    from uqfusion.eval.frame_geometry import content_rows
    return content_rows(modality)


def noise_sd(g: np.ndarray) -> float:
    """Immerkaer (1996) fast noise-variance estimate, grey levels."""
    import cv2
    k = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)
    r = cv2.filter2D(g.astype(np.float32), -1, k, borderType=cv2.BORDER_REFLECT)[1:-1, 1:-1]
    h, w = g.shape
    return float(np.sqrt(np.pi / 2) * np.abs(r).sum() / (6.0 * (w - 2) * (h - 2)))


def _guided(I: np.ndarray, p: np.ndarray, r: int, eps: float) -> np.ndarray:
    import cv2
    m = lambda x: cv2.boxFilter(x, -1, (r, r))          # noqa: E731
    mI, mp = m(I), m(p)
    a = (m(I * p) - mI * mp) / (m(I * I) - mI * mI + eps)
    b = mp - a * mI
    return m(a) * I + m(b)


def dcp(x: np.ndarray, omega=0.95, t0=0.15, patch=15, r=40, eps=1e-3) -> np.ndarray:
    """x: float32 BGR in [0, 1] -> dehazed, same range."""
    import cv2
    ker = np.ones((patch, patch), np.uint8)
    dark = cv2.erode(x.min(axis=2), ker)
    n = max(1, int(dark.size * 0.001))
    idx = np.argpartition(dark.ravel(), -n)[-n:]
    A = np.maximum(x.reshape(-1, 3)[idx].mean(axis=0), 0.05).astype(np.float32)
    t = 1.0 - omega * cv2.erode((x / A).min(axis=2), ker)
    t = _guided(cv2.cvtColor(x, cv2.COLOR_BGR2GRAY), t.astype(np.float32), r, eps)
    t = np.clip(t, t0, 1.0)[..., None]
    return np.clip((x - A) / t + A, 0.0, 1.0)


def _clahe(g: np.ndarray, clip=2.0, tiles=8) -> np.ndarray:
    import cv2
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(tiles, tiles)).apply(g)


def restore(im: np.ndarray, method: str, modality: str, rows: tuple[int, int]) -> np.ndarray:
    """Restore the content rows of a 640x640 letterboxed BGR frame; padding is left as it is."""
    import cv2
    lo, hi = rows
    c = im[lo:hi]
    if modality == "vis":
        if method == "ae":
            lin = (c.astype(np.float32) / 255.0) ** 2.2
            m = float((lin * np.array([0.114, 0.587, 0.299], np.float32)).sum(axis=2).mean())
            g = float(np.clip(0.18 / max(m, 1e-6), 1.0, 8.0))
            out = (np.clip(lin * g, 0, 1) ** (1 / 2.2)) * 255.0
        elif method == "clahe":
            lab = cv2.cvtColor(c, cv2.COLOR_BGR2LAB)
            lab[..., 0] = _clahe(lab[..., 0])
            out = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        elif method == "dcp":
            out = dcp(c.astype(np.float32) / 255.0) * 255.0
        elif method == "bil_dcp":
            out = dcp(cv2.bilateralFilter(c, 5, 25, 5).astype(np.float32) / 255.0) * 255.0
        else:
            raise ValueError(method)
    else:
        g = c[..., 0]
        if method in ("nlm", "nlm_clahe"):
            h = float(np.clip(1.2 * noise_sd(g), 3.0, 40.0))
            g = cv2.fastNlMeansDenoising(g, None, h, 7, 21)
            if method == "nlm_clahe":
                g = _clahe(g)
        elif method == "median3":
            g = cv2.medianBlur(g, 3)
        elif method == "clahe":
            g = _clahe(g)
        else:
            raise ValueError(method)
        out = np.repeat(g[..., None], 3, axis=2)
    res = im.copy()
    res[lo:hi] = np.clip(np.asarray(out, np.float32) + 0.5, 0, 255).astype(np.uint8) \
        if np.asarray(out).dtype != np.uint8 else out
    return res


def gate_feats(im: np.ndarray, modality: str, rows: tuple[int, int]) -> dict:
    """Image-only statistics a restoration gate may read (computed on the corrupted frame)."""
    import cv2
    lo, hi = rows
    c = im[lo:hi]
    g = cv2.cvtColor(c, cv2.COLOR_BGR2GRAY)
    f = {"noise_sd": noise_sd(g), "p01": float(np.percentile(g, 1)), "p99": float(np.percentile(g, 99))}
    if modality == "vis":
        lin = (c.astype(np.float32) / 255.0) ** 2.2
        f["lum_lin"] = float((lin * np.array([0.114, 0.587, 0.299], np.float32)).sum(axis=2).mean())
        dark = cv2.erode(c.min(axis=2), np.ones((15, 15), np.uint8))
        f["dark_med"] = float(np.median(dark))
    return f


# ---------------------------------------------------------------- frame pool
_W: dict = {}


def _init(spec: dict) -> None:
    import cv2
    cv2.setNumThreads(1)
    from uqfusion.eval.corruptions import make_corruption
    _W.update(spec)
    _W["tf"] = (make_corruption(spec["kind"], spec["sev"], spec["seed"], version="v2",
                                modality=spec["mod"], images=spec["images"])
                if spec.get("kind") else None)


def _one(i: int):
    import cv2
    im = cv2.imread(_W["images"][i])
    if im is None:
        raise FileNotFoundError(_W["images"][i])
    if _W["tf"] is not None:
        im = _W["tf"](im, i)
    out = {"raw": im} if _W.get("keep_raw", True) else {}
    for m in _W["methods"]:
        out[m] = restore(im, m, _W["mod"], _W["rows"])
    return i, out, gate_feats(im, _W["mod"], _W["rows"])


def _chunk(idx):
    return [_one(i) for i in idx]


def frames(spec: dict, idx: list[int], workers: int, chunk: int = 4):
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


# ---------------------------------------------------------------- scoring helpers
def _base():
    """GTs, homographies and runs of the paired frames (from the shipped context, clean, seed 0)."""
    from uqfusion.eval.ctx import load_context
    from v2_sensitivity import CACHE, STATS, sub
    d = f"{STATS}/brightness/{sub(941)}"
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{CACHE}/seed0/{sub(941)}", bright_dir=d,
                       structure_dir=f"{STATS}/structure/{sub(941)}",
                       ir_bright=f"{d}/gauss_ir_paired_clean.json", conditions=("clean",), verbose=False)
    return ctx.gts, ctx.h_frames, ctx.runs


def stream_records(recs: list[dict], modality: str, h_frames: list, idx) -> list[dict]:
    """What the fusion consumes and what ir_only is scored on: IR gets the shipped NMS and the
    per-frame homography onto the VIS canvas."""
    if modality == "vis":
        return recs
    from uqfusion.eval.irdedup import nms_records
    from uqfusion.uq.fusion import apply_homography
    recs = nms_records(recs, IR_NMS)
    return [{**r, "boxes_xyxy": apply_homography(np.asarray(r["boxes_xyxy"]).reshape(-1, 4), h_frames[i])}
            for r, i in zip(recs, idx)]


def ship_ap_sel(recs, gts, sel) -> float:
    from uqfusion.eval.apmetrics import ap_from_parts, frame_parts
    parts = frame_parts([recs[j] for j in sel], [gts[j] for j in sel])
    e = ap_from_parts(parts)["per_class"].get(SHIP)
    return float(e["ap50_95"]) if e else float("nan")


def gate_flags(draw: int, mod: str, cond: str) -> np.ndarray:
    """Per paired frame: does the shipped gate's degradation statistic for this stream fire?
    VIS: veil (grad_gini below its threshold) OR dark (brightness statistic below mu_b).
    IR: ir_d2 above the merge-health bound. Seed-independent (frame statistics only)."""
    from uqfusion.eval.ctx import load_context
    from v2_sensitivity import CACHE, STATS, sub
    d = f"{STATS}/brightness/{sub(draw)}"
    ic = None if (mod == "vis" or cond == "clean") else cond
    vc = cond if mod == "vis" else "clean"
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{CACHE}/seed0/{sub(draw)}", bright_dir=d,
                       structure_dir=f"{STATS}/structure/{sub(draw)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ic or 'clean'}.json",
                       conditions=(vc,), ir_condition=ic, verbose=False)
    if mod == "vis":
        veil = ctx.gini_by_cond[vc] < float(ctx.struct_const["axes"]["grad_gini"]["threshold"])
        dark = ctx.bright_by_cond[vc] < float(ctx.c_vis.mu_b)
        return np.asarray(veil | dark, bool)
    return np.asarray(ctx.ir_d2 > ctx.ir_bound, bool)


# ---------------------------------------------------------------- dev
def cmd_dev(args) -> int:
    import torch
    from uqfusion.uq.infer import UQPredictor
    from uqfusion.eval.ctx import NIGHT_RUNS

    gts, hf, runs = _base()
    lists = {m: [ln.strip() for ln in PAIRED[m].read_text(encoding="utf-8").splitlines() if ln.strip()]
             for m in PAIRED}
    n = len(lists["vis"])
    idx = list(range(0, n, args.stride))
    night = np.isin(runs[idx], NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    gts_s = [gts[i] for i in idx]
    out_p = ROOT / args.out
    out_p.parent.mkdir(parents=True, exist_ok=True)
    res = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    dev = 0 if torch.cuda.is_available() else "cpu"
    units = [("vis", c, VIS_CONDS[c]) for c in args.vis] + [("ir", c, IR_CONDS[c]) for c in args.ir]
    for mod, cname, kc in units:
        key = f"{mod}/{cname}"
        if key in res and not args.redo:
            print(f"[dev] {key} cached", flush=True)
            continue
        t0 = time.time()
        preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_{mod}_seed{k}/weights/best.pt", device=dev,
                             imgsz=640, conf=0.001) for k in args.seeds]
        draw = 941 + (10 if mod == "ir" else 0)
        spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": draw, "mod": mod,
                "images": lists[mod], "methods": list(METHODS[mod]), "rows": _rows(mod)}
        variants = ("raw",) + METHODS[mod]
        recs = {v: [[] for _ in preds] for v in variants}
        feats, tr = [], {v: 0.0 for v in variants}
        pend = []

        def flush():
            # dev only: raw and restored go through the same batched forward pass, so the
            # comparison is like for like; the registered build uses batch 1 like the raw caches
            for v in variants:
                batch = [ims[v] for ims in pend]
                for j, p in enumerate(preds):
                    recs[v][j].extend(p.predict_batch(batch) if args.batch > 1 else [p(b.copy()) for b in batch])
            pend.clear()

        for i, ims, f in frames(spec, idx, args.workers):
            feats.append(f)
            pend.append(ims)
            if len(pend) >= args.batch:
                flush()
        if pend:
            flush()
        flag = gate_flags(941, mod, cname)[idx]
        row = {"feats": feats, "ap": {}, "gate_rate": {s: float(flag[sel].mean()) for s, sel in sl.items()}}
        for v in variants:
            for tag, use in ((v, None),) + ((("gated:" + v), flag),) * (v != "raw"):
                row["ap"][tag] = {}
                for s, sel in sl.items():
                    row["ap"][tag][s] = [
                        ship_ap_sel(stream_records([r if (use is None or use[q]) else recs["raw"][j][q]
                                                    for q, r in enumerate(recs[v][j])], mod, hf, idx), gts_s, sel)
                        for j in range(len(preds))]
        res[key] = row
        tmp = out_p.with_suffix(".tmp")
        tmp.write_text(json.dumps(res), encoding="utf-8")
        os.replace(tmp, out_p)
        msg = " | ".join(f"{v} d{np.mean(row['ap'][v]['day']):.4f} n{np.mean(row['ap'][v]['night']):.4f}"
                         for v in row["ap"]) + f" | gate {row['gate_rate']}"
        print(f"[dev] {key} ({time.time() - t0:.0f}s): {msg}", flush=True)
    return 0


def cmd_trainstats(args) -> int:
    from uqfusion.eval.cache import load_cache
    out = {}
    for mod in ("vis", "ir"):
        recs, _ = load_cache(ROOT / f"runs/cache_p3/seed0/gauss_{mod}_train_clean.pkl")
        paths = [r["image_path"] for r in recs]
        rng = np.random.default_rng(0)
        pick = sorted(rng.choice(len(paths), size=min(args.n, len(paths)), replace=False).tolist())
        spec = {"kind": None, "sev": 0, "seed": 0, "mod": mod, "images": paths, "methods": [],
                "rows": _rows(mod), "keep_raw": False}
        feats = []
        for i, _ims, f in frames(spec, pick, args.workers, chunk=16):
            f["run"] = Path(paths[i]).parent.name
            feats.append(f)
        out[mod] = feats
        print(f"[trainstats] {mod}: {len(feats)} frames", flush=True)
    p = ROOT / "runs/restore/train_feats.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out), encoding="utf-8")
    return 0


# ---------------------------------------------------------------- registered run
RCACHE = "runs/cache_p3dev_v2_restore"
FRESH = (945, 946, 947)
T_975_DF4 = 2.7764451051977987


def rcache_path(k: int, mod: str, cond: str, draw: int | None) -> Path:
    where = "clean" if cond == "clean" else f"draw{draw}_{draw + 10}"
    return ROOT / RCACHE / f"seed{k}" / where / f"gauss_{mod}_paired_{cond}.pkl"


def flip_path(k: int, mod: str, cond: str, draw: int | None) -> Path:
    q = rcache_path(k, mod, cond, draw)
    return q.with_name(q.stem + "_flip.pkl")


def unflip(rec: dict, w: int) -> dict:
    """Detections on a horizontally flipped frame, mapped back to the original frame."""
    b = np.asarray(rec["boxes_xyxy"], float).reshape(-1, 4)
    s = np.asarray(rec["sigma_ltrb"], float).reshape(-1, 4)
    return {**rec, "boxes_xyxy": np.column_stack([w - b[:, 2], b[:, 1], w - b[:, 0], b[:, 3]]),
            "sigma_ltrb": s[:, [2, 1, 0, 3]]}


def cmd_build_unit(args) -> int:
    """One (modality, condition, draw): replay the corruption once; detect with all five checkpoints on
    the restored frame (unless --method none) and, with --flip, on the horizontally flipped raw frame."""
    import torch
    from uqfusion.eval.cache import build_cache
    from uqfusion.uq.infer import UQPredictor
    mod, cond = args.mod, args.cond
    kc = (VIS_CONDS if mod == "vis" else IR_CONDS)[cond]
    lst = [ln.strip() for ln in PAIRED[mod].read_text(encoding="utf-8").splitlines() if ln.strip()]
    seed = args.draw + (10 if mod == "ir" else 0)
    dev = 0 if torch.cuda.is_available() else "cpu"
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_{mod}_seed{k}/weights/best.pt", device=dev,
                         imgsz=640, conf=0.001) for k in range(5)]
    method = None if args.method == "none" else args.method
    spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": seed, "mod": mod,
            "images": lst, "methods": [method] if method else [], "rows": _rows(mod), "keep_raw": bool(args.flip)}
    recs = [[] for _ in preds]
    frecs = [[] for _ in preds]
    t0 = time.time()
    for i, ims, _f in frames(spec, list(range(len(lst))), args.workers):
        for j, pr in enumerate(preds):
            if method:
                recs[j].append(pr(ims[method].copy()))
            if args.flip:
                raw = ims["raw"]
                frecs[j].append(unflip(pr(np.ascontiguousarray(raw[:, ::-1])), raw.shape[1]))
        if (i + 1) % 200 == 0:
            print(f"[build] {mod}/{cond}/{args.draw} {i + 1}/{len(lst)} ({time.time() - t0:.0f}s)", flush=True)
    code = None
    if kc:
        from uqfusion.eval.corruptions_v2 import CODE_REV as code
    where = None if cond == "clean" else args.draw
    for k in range(5):
        meta = {"source": "gaussian",
                "weights": [str(ROOT / f"runs/phase3_stage2/p3_{mod}_seed{k}/weights/best.pt")],
                "data": mod, "split": "val", "images_list": str(PAIRED[mod]), "imgsz": 640, "conf": 0.001,
                "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                "corrupt_seed": seed if kc else None, "corrupt_version": "v2" if kc else None,
                "modality": mod, "corrupt_code": code,
                "built_by": "restore.py build-unit (corruption replayed once, all five weights)"}
        if method:
            it = iter(recs[k])
            build_cache(lambda _img, it=it: next(it), lst, rcache_path(k, mod, cond, where),
                        meta={**meta, "restore": method}, transform=None, log_every=0)
        if args.flip:
            it = iter(frecs[k])
            build_cache(lambda _img, it=it: next(it), lst, flip_path(k, mod, cond, where),
                        meta={**meta, "tta": "hflip of the unrestored frame, boxes mapped back"},
                        transform=None, log_every=0)
    print(f"[build] {mod}/{cond}/{args.draw} done in {time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_build(args) -> int:
    import subprocess
    from concurrent.futures import ThreadPoolExecutor
    units = [("vis", "clean", FRESH[0]), ("ir", "clean", FRESH[0])]
    units += [("vis", c, d) for d in args.draws for c in VIS_CONDS if c != "clean"]
    units += [("ir", c, d) for d in args.draws for c in IR_CONDS if c != "clean"]
    method = {"vis": args.vis_method or "none", "ir": args.ir_method or "none"}
    units = [u for u in units if method[u[0]] != "none" or args.flip]
    log_dir = ROOT / RCACHE / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    t0 = time.time()

    def run(u):
        mod, cond, d = u
        w = None if cond == "clean" else d
        outs = ([rcache_path(k, mod, cond, w) for k in range(5)] if method[mod] != "none" else []) + \
               ([flip_path(k, mod, cond, w) for k in range(5)] if args.flip else [])
        if all(o.is_file() for o in outs):
            print(f"[restore-build] {u} skip (exists)", flush=True)
            return u, True
        cmd = [sys.executable, "-u", str(ROOT / "scripts/restore.py"), "build-unit", "--mod", mod,
               "--cond", cond, "--draw", str(d), "--method", method[mod], "--workers", str(args.workers)] + \
              (["--flip"] if args.flip else [])
        print(f"[restore-build] {u} start", flush=True)
        with open(log_dir / f"{mod}_{cond}_{d}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        ok = rc == 0 and all(o.is_file() for o in outs)
        print(f"[restore-build] {u} {'ok' if ok else f'FAIL rc={rc}'} "
              f"(elapsed {(time.time() - t0) / 60:.1f} min)", flush=True)
        return u, ok

    with ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(run, units))
    bad = [u for u, ok in res if not ok]
    print(f"[restore-build] {len(res) - len(bad)}/{len(res)} units ok" + (f"; FAILED {bad}" if bad else ""),
          flush=True)
    return 1 if bad else 0


def _score_job(spec):
    """(draw, ir_cond, k) -> {arm|vis_cond|slice: {vis, ir, fused}} for arms raw / gated / always."""
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.cache import load_cache
    from uqfusion.eval.ctx import NIGHT_RUNS, capability_prior, load_context, run_systems
    from uqfusion.eval.irdedup import nms_records
    from v2_sensitivity import CACHE, STATS, ship_ap, sub
    import seed_ensemble
    seed_ensemble._DIST.clear()
    seed_ensemble._memo_distances()      # arms that share a record list score its distances once
    draw, ic, k, vm, im_ = spec
    d = f"{STATS}/brightness/{sub(draw)}"
    vconds = tuple(VIS_CONDS)
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{CACHE}/seed{k}/{sub(draw)}", bright_dir=d,
                       structure_dir=f"{STATS}/structure/{sub(draw)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ic or 'clean'}.json",
                       conditions=vconds, ir_condition=ic, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    thr = float(ctx.struct_const["axes"]["grad_gini"]["threshold"])
    vflag = {c: (ctx.gini_by_cond[c] < thr) | (ctx.bright_by_cond[c] < float(ctx.c_vis.mu_b)) for c in vconds}
    iflag = ctx.ir_d2 > ctx.ir_bound
    # the clean-IR flags, for the clean IR stream the capability prior is computed on
    ctx0 = ctx if ic is None else load_context(
        preset="crossmodal26m", cache_dir=f"{CACHE}/seed{k}/{sub(draw)}", bright_dir=d,
        structure_dir=f"{STATS}/structure/{sub(draw)}", ir_bright=f"{d}/gauss_ir_paired_clean.json",
        conditions=("clean",), ir_condition=None, verbose=False)
    iflag0 = ctx0.ir_d2 > ctx0.ir_bound

    def rest(mod, cond):
        r, _ = load_cache(rcache_path(k, mod, cond, None if cond == "clean" else draw))
        return nms_records(r, IR_NMS) if mod == "ir" else r

    def mix(raw, res, flag):
        return [b if f else a for a, b, f in zip(raw, res, flag)]

    raw_ir, raw_ir0 = ctx.ir_clean, ctx0.ir_clean
    rv = {c: rest("vis", c) for c in vconds} if vm else {}
    ri = rest("ir", ic or "clean") if im_ else None
    ri0 = rest("ir", "clean") if im_ else None
    cap0 = (ctx.cap_vis, ctx.cap_ir)
    out = {}
    for arm in ("raw", "gated", "always"):
        if arm == "raw":
            V = {c: ctx.vis_by_cond[c] for c in vconds}
            I, I0 = raw_ir, raw_ir0
        elif arm == "gated":
            V = {c: (mix(ctx.vis_by_cond[c], rv[c], vflag[c]) if vm else ctx.vis_by_cond[c]) for c in vconds}
            I = mix(raw_ir, ri, iflag) if im_ else raw_ir
            I0 = mix(raw_ir0, ri0, iflag0) if im_ else raw_ir0
        else:
            V = {c: (rv[c] if vm else ctx.vis_by_cond[c]) for c in vconds}
            I = ri if im_ else raw_ir
            I0 = ri0 if im_ else raw_ir0
        if arm == "raw":
            ctx.cap_vis, ctx.cap_ir = cap0
        else:
            cv, ci = capability_prior(V["clean"], I0, ctx.gts, ctx.h_frames, ctx.cap_fit_frames)
            ctx.cap_vis, ctx.cap_ir = cv, ci / ctx.cap_ir_scale
        for c in vconds:
            o = run_systems(ctx, c, compute_metrics=False, vis_records=V[c], ir_records=I)
            parts = {"vis": frame_parts(V[c], ctx.gts), "ir": frame_parts(o["ir_in_vis"], ctx.gts),
                     "fused": frame_parts(o["fused_gated"], ctx.gts)}
            for s, idx in sl.items():
                out[f"{arm}|{c}|{s}"] = {n: ship_ap(p, idx) for n, p in parts.items()}
    rates = {f"vis_{c}|{s}": float(vflag[c][idx].mean()) for c in vconds for s, idx in sl.items()}
    rates.update({f"ir|{s}": float(iflag[idx].mean()) for s, idx in sl.items()})
    return spec, out, rates


def cmd_score(args) -> int:
    out_p = ROOT / f"runs/restore/score_{args.tag}.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    specs = [(d, ic, k, args.vis_method, args.ir_method) for d in args.draws
             for ic in [None] + [c for c in IR_CONDS if c != "clean"] for k in range(5)
             if f"{d}|{ic}|{k}" not in done]
    print(f"[restore-score] {len(specs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for n, (spec, res, rates) in enumerate(ex.map(_score_job, specs), 1):
            done[f"{spec[0]}|{spec[1]}|{spec[2]}"] = {"ap": res, "rates": rates}
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(done), encoding="utf-8")
            os.replace(tmp, out_p)
            print(f"[restore-score] {n}/{len(specs)} {spec[:3]} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_report(args) -> int:
    raw = json.loads((ROOT / f"runs/restore/score_{args.tag}.json").read_text(encoding="utf-8"))
    irs = [None] + [c for c in IR_CONDS if c != "clean"]

    def ps(arm, c, ic, s, n):
        return np.array([np.mean([raw[f"{d}|{ic}|{k}"]["ap"][f"{arm}|{c}|{s}"][n] for d in args.draws])
                         for k in range(5)])

    def ci(x):
        m, h = float(np.mean(x)), T_975_DF4 * float(np.std(x, ddof=1)) / np.sqrt(len(x))
        return m, m - h, m + h

    f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"   # noqa: E731
    fail = lambda x: x[0] <= -0.0060 and x[2] < 0              # noqa: E731
    lines = [f"# Restoration before detection, {args.tag} (draws {', '.join(map(str, args.draws))})", "",
             "Ship AP, five Phase 3 systems, per system the draw mean; Δ = arm − raw for the same system, "
             "95% t-interval (df 4). Claim = fused − max(VIS, IR) failing at ≤ −0.0060 with the interval "
             "below zero. Pre-registered in `docs/prereg-no-retrain-2026-10-10.md` (E2).", ""]
    summ = {}
    for arm in ("gated", "always"):
        lines += [f"## {arm}", "",
                  "| VIS | IR | slice | raw VIS / IR / fused | Δ VIS | Δ IR | Δ fused | claim raw | claim arm |",
                  "|---|---|---|---|---|---|---|---|---|"]
        cnt = {"better_0060": 0, "better_0014": 0, "worse_0014": 0, "worse_0060": 0, "fail_raw": 0, "fail_arm": 0}
        for ic in irs:
            for c in VIS_CONDS:
                for s in ("day", "night"):
                    b = {n: ps("raw", c, ic, s, n) for n in ("vis", "ir", "fused")}
                    a = {n: ps(arm, c, ic, s, n) for n in ("vis", "ir", "fused")}
                    dv, di, df = (ci(a[n] - b[n]) for n in ("vis", "ir", "fused"))
                    cr = ci(b["fused"] - np.maximum(b["vis"], b["ir"]))
                    ca = ci(a["fused"] - np.maximum(a["vis"], a["ir"]))
                    cnt["better_0060"] += df[0] >= 0.0060 and df[1] > 0
                    cnt["better_0014"] += df[0] >= 0.0014 and df[1] > 0
                    cnt["worse_0014"] += df[0] <= -0.0014 and df[2] < 0
                    cnt["worse_0060"] += df[0] <= -0.0060 and df[2] < 0
                    cnt["fail_raw"] += fail(cr)
                    cnt["fail_arm"] += fail(ca)
                    summ[f"{arm}|{c}|{ic}|{s}"] = {"raw": {n: float(b[n].mean()) for n in b}, "d_vis": dv,
                                                   "d_ir": di, "d_fused": df, "claim_raw": cr, "claim_arm": ca}
                    lines.append(f"| {c} | {ic or 'clean'} | {s} | {b['vis'].mean():.4f} / {b['ir'].mean():.4f} / "
                                 f"{b['fused'].mean():.4f} | {f(dv)} | {f(di)} | {f(df)} | "
                                 f"{cr[0]:+.4f}{' FAIL' if fail(cr) else ''} | {ca[0]:+.4f}{' FAIL' if fail(ca) else ''} |")
        improves = cnt["worse_0014"] == 0 and cnt["better_0060"] >= 1
        summ[f"{arm}|summary"] = {**{k: int(v) for k, v in cnt.items()}, "IMPROVES": bool(improves)}
        lines += ["", f"**{arm}: {'IMPROVES' if improves else 'does not improve'}**: better ≥ 0.0060 on "
                  f"{cnt['better_0060']} cells (≥ 0.0014: {cnt['better_0014']}), worse ≤ −0.0014 on "
                  f"{cnt['worse_0014']} (≤ −0.0060: {cnt['worse_0060']}); claim fails raw {cnt['fail_raw']}, "
                  f"{arm} {cnt['fail_arm']}.", ""]
    keys = sorted({k for v in raw.values() for k in v["rates"]})
    rates = {k: float(np.mean([v["rates"][k] for v in raw.values() if k in v["rates"]])) for k in keys}
    lines += ["## Gate firing rates (mean over draws and IR conditions)", "",
              ", ".join(f"{k} {v:.1%}" for k, v in rates.items()), ""]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("\n".join(ln for ln in lines if ln.startswith("**")))
    return 0


def cmd_select(args) -> int:
    """The declared selection rule (prereg E2), applied to the development run. Both the objective
    and the clean constraint are read on the GATED deployment (`gated:<method>`), the arm E2 deploys;
    `always` is printed beside it."""
    dev = json.loads((ROOT / args.dev).read_text(encoding="utf-8"))
    out, lines = {}, []
    for mod, conds in (("vis", VIS_CONDS), ("ir", IR_CONDS)):
        corrupted = [c for c in conds if c != "clean"]
        if f"{mod}/clean" not in dev:
            raise SystemExit(f"development run incomplete for {mod}")
        if any(f"{mod}/{c}" not in dev for c in corrupted):
            # The clean constraint is checked first. If no method passes it, no corrupted cell can
            # change the answer, and the units that were not run are not needed (VIS, 2026-10-10:
            # stopped after clean and fog for exactly that reason).
            cl = dev[f"{mod}/clean"]["ap"]
            if any(all(float(np.mean(cl[f"gated:{m}"][s]) - np.mean(cl["raw"][s])) >= -0.0014 for s in ("day", "night"))
                   for m in METHODS[mod]):
                raise SystemExit(f"development run incomplete for {mod}")
            corrupted = [c for c in corrupted if f"{mod}/{c}" in dev]
        best, rows = None, []
        for m in METHODS[mod]:
            def d(c, s, tag):
                ap = dev[f"{mod}/{c}"]["ap"]
                return float(np.mean(ap[tag][s]) - np.mean(ap["raw"][s]))
            gain = float(np.mean([d(c, s, f"gated:{m}") for c in corrupted for s in ("day", "night")]))
            gain_always = float(np.mean([d(c, s, m) for c in corrupted for s in ("day", "night")]))
            clean = {s: d("clean", s, f"gated:{m}") for s in ("day", "night")}
            ok = all(v >= -0.0014 for v in clean.values())
            rows.append((m, gain, gain_always, clean, ok))
            if ok and (best is None or gain > best[1]):
                best = (m, gain)
        out[mod] = best[0] if best else None
        lines.append(f"{mod}: " + "; ".join(f"{m} gain(gated) {g:+.4f} (always {ga:+.4f}), clean day {c['day']:+.4f} "
                                           f"night {c['night']:+.4f}{'' if ok else ' FAILS clean'}" for m, g, ga, c, ok in rows)
                     + f" -> {out[mod] or 'none (modality not restored)'}")
    (ROOT / "runs/restore/selection.json").write_text(json.dumps({"selected": out, "detail": lines}, indent=1),
                                                       encoding="utf-8")
    print("\n".join(lines))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    d = sp.add_parser("dev")
    d.add_argument("--vis", nargs="*", default=list(VIS_CONDS))
    d.add_argument("--ir", nargs="*", default=list(IR_CONDS))
    d.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    d.add_argument("--stride", type=int, default=4)
    d.add_argument("--workers", type=int, default=8)
    d.add_argument("--redo", action="store_true")
    d.add_argument("--out", default="runs/restore/dev.json")
    d.add_argument("--batch", type=int, default=16)
    t = sp.add_parser("trainstats")
    t.add_argument("--n", type=int, default=1500)
    t.add_argument("--workers", type=int, default=8)
    u = sp.add_parser("build-unit")
    u.add_argument("--mod", choices=["vis", "ir"], required=True)
    u.add_argument("--cond", required=True)
    u.add_argument("--draw", type=int, required=True)
    u.add_argument("--method", required=True)
    u.add_argument("--workers", type=int, default=6)
    u.add_argument("--flip", action="store_true")
    b = sp.add_parser("build")
    b.add_argument("--flip", action="store_true")
    b.add_argument("--draws", type=int, nargs="+", default=list(FRESH))
    b.add_argument("--vis-method", default=None)
    b.add_argument("--ir-method", default=None)
    b.add_argument("--jobs", type=int, default=4)
    b.add_argument("--workers", type=int, default=6)
    sc = sp.add_parser("score")
    sc.add_argument("--draws", type=int, nargs="+", default=list(FRESH))
    sc.add_argument("--vis-method", default=None)
    sc.add_argument("--ir-method", default=None)
    sc.add_argument("--tag", default="fresh")
    sc.add_argument("--workers", type=int, default=8)
    se = sp.add_parser("select")
    se.add_argument("--dev", default="runs/restore/dev.json")
    rp = sp.add_parser("report")
    rp.add_argument("--draws", type=int, nargs="+", default=list(FRESH))
    rp.add_argument("--tag", default="fresh")
    rp.add_argument("--out", default="docs/eval/restore/restore_2026-10-10.md")
    args = ap.parse_args()
    return {"dev": cmd_dev, "trainstats": cmd_trainstats, "build-unit": cmd_build_unit, "build": cmd_build,
            "score": cmd_score, "report": cmd_report, "select": cmd_select}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
