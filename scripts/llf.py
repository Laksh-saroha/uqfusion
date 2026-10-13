"""Learned late fusion (LLF): one learned score for every ship box from either stream.

The shipped decision layer is a hand-written gate (a night veto, a veil axis, a weak-IR fallback, a
support bonus) rewritten six times, each rewrite forced by a measured failure of the one before
(paper §6.3), and still wrong at night for the retrained detector (−0.1847 on clean night, §6.6) and
under a degraded IR (§6.7). LLF replaces the whole layer with one learned object, in the spirit of
CLOCs (Pang et al., IROS 2020): every ship box from VIS and from IR (warped into the VIS canvas) is
re-scored by a gradient-boosted model from

  * the box itself (stream, confidence, rank, size, position, σ),
  * its relation to the other stream (best IoU, the other box's confidence, support at IoU 0.30,
    centre distance in units of its own size, whether a more confident other-stream box overlaps),
  * the frame's health signals for both sensors (VIS brightness, veil and texture statistics, VIS
    health, IR brightness and IR novelty, box counts and top confidences per stream).

Vetoing a stream, trusting the other one and cross-modal support become things the model can learn
per box; nothing about them is written by hand. Its target is the box's AP50-95 credit under a
one-to-one assignment: for each ground-truth ship, the candidate (from either stream) with the
highest IoU ≥ 0.5 earns the fraction of the ten IoU thresholds it clears; every other candidate
earns 0, so a cross-stream duplicate is taught to rank low. Confidence enters monotonically.
VIS buoy boxes pass through unchanged (the metric is ship AP; IR is nc = 1).

Registered in `docs/prereg-llf-2026-10-10.md` before any evaluation number existed.

    py -3.13 scripts/llf.py extract      # draw 941 training rows (5 systems x 30 cells, subsampled)
    py -3.13 scripts/llf.py fit          # one model per fold
    py -3.13 scripts/llf.py eval         # draws 942-944, every frame, out of fold
    py -3.13 scripts/llf.py report
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
import zlib
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

from uqfusion.eval.apmetrics import frame_parts                       # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems   # noqa: E402
from uqfusion.eval.fusion_eval import _per_frame_h                    # noqa: E402
from uqfusion.eval.matching import iou_matrix                         # noqa: E402
from uqfusion.uq.fusion import apply_homography                       # noqa: E402
from v2_sensitivity import CACHE, PRESET, SEEDS, STATS, T_975_DF4, ship_ap, sub  # noqa: E402

VIS_CONDS = ("clean", "fog", "fog_s1", "glare", "lowlight")
IR_CONDS = (None, "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")
TRAIN_DRAW, EVAL_DRAWS = 941, (942, 943, 944)
RUNS = ("pohang00", "pohang01", "pohang02", "pohang03")
NIGHT_BLOCKS = 4
KEEP_P = 0.20                 # fraction of (system, cell, frame) units kept for training
IR_CAP = 100                  # IR boxes per frame, by confidence (top-k >= 100 is irrelevant, Table L)
THR = np.round(np.arange(0.50, 0.951, 0.05), 2)
OUT = ROOT / "runs" / "llf"
FAMILIES = {
    "vis_fog": lambda c, ic: c in ("fog", "fog_s1"),
    "vis_lowlight": lambda c, ic: c == "lowlight",
    "vis_glare": lambda c, ic: c == "glare",
    "ir_fog": lambda c, ic: ic in ("fog_s1", "fog_s2", "fog_s3"),
    "ir_noise": lambda c, ic: ic in ("noise_s2", "noise_s3"),
}

BOX = ["is_ir", "conf", "log_conf", "rank", "conf_rel", "log_area", "aspect", "cx", "cy",
       "sigma_norm", "sib_max_iou", "sib_higher"]
XM = ["x_iou", "x_conf", "x_sup", "x_n30", "x_dist", "x_higher"]
FRAME = ["v_p05", "v_p50", "v_std", "v_loglapvar", "v_lov", "v_gini", "v_edge", "q_vis",
         "i_p05", "i_p50", "i_std", "i_logd2", "n_vis25", "n_ir25", "top3_vis", "top3_ir"]
FEATS = BOX + XM + FRAME
VARIANT_FEATS = {"full": FEATS, "noframe": BOX + XM, "noxm": BOX + FRAME,
                 # LLF-2 (docs/prereg-llf2-2026-10-10.md): only what a gate does -- stream, own confidence
                 # (monotone), cross-stream support (monotone up) and duplication (monotone down), frame health
                 "trust": ["is_ir", "conf", "x_sup", "x_higher"] + FRAME}
MONO = {"conf": 1, "x_sup": 1, "x_higher": -1}
HGB = dict(max_iter=500, learning_rate=0.05, max_leaf_nodes=63, min_samples_leaf=200,
           l2_regularization=1.0, random_state=0)


# ---------------------------------------------------------------- features --------------------------
def _frames(p: Path) -> list:
    return json.loads(p.read_text(encoding="utf-8"))["frames"]


def frame_stats(ctx, draw: int, c: str, ic: str | None) -> np.ndarray:
    """(n_frames, 12) frame-health columns, index-aligned with the paired caches."""
    b = _frames(ROOT / STATS / "brightness" / sub(draw) / f"gauss_vis_paired_{c}.json")
    s = _frames(ROOT / STATS / "structure" / sub(draw) / f"gauss_vis_paired_{c}.json")
    ib = _frames(ROOT / STATS / "brightness" / sub(draw) / f"gauss_ir_paired_{ic or 'clean'}.json")
    n = len(ctx.runs)
    if not (len(b) == len(s) == len(ib) == n):
        raise SystemExit(f"frame statistics misaligned for {c}/{ic}: {len(b)} {len(s)} {len(ib)} {n}")
    if [f["run"] for f in b] != list(ctx.runs) or [f["run"] for f in ib] != list(ctx.runs):
        raise SystemExit(f"frame statistics are not in cache order for {c}/{ic}")
    q = np.asarray(ctx.q_vis_by_cond.get(c, np.ones(n)), float)
    d2 = np.asarray(ctx.ir_d2 if ctx.ir_d2 is not None else np.zeros(n), float)
    return np.column_stack([
        [f["p05"] for f in b], [f["p50"] for f in b], [f["std"] for f in b],
        np.log1p([max(f["lap_var"], 0.0) for f in b]), [f["lap_over_var"] for f in s],
        [f["grad_gini"] for f in s], [f["edge_density"] for f in s], q,
        [f["p05"] for f in ib], [f["p50"] for f in ib], [f["std"] for f in ib],
        np.log1p(np.clip(d2, 0, None))]).astype(np.float64)


def candidates(vr: dict, ir: dict, H) -> dict:
    vb = np.asarray(vr["boxes_xyxy"], float).reshape(-1, 4)
    vc = np.asarray(vr["conf"], float).ravel()
    vcl = np.asarray(vr["cls"]).astype(int).ravel()
    vs = np.asarray(vr["sigma_ltrb"], float).reshape(-1, 4)
    ship = vcl == 0
    ib = np.asarray(ir["boxes_xyxy"], float).reshape(-1, 4)
    icf = np.asarray(ir["conf"], float).ravel()
    icl = np.asarray(ir["cls"]).astype(int).ravel()
    isg = np.asarray(ir["sigma_ltrb"], float).reshape(-1, 4)
    keep = np.flatnonzero(icl == 0)
    keep = keep[np.argsort(-icf[keep], kind="stable")][:IR_CAP]
    ib, icf, isg = ib[keep], icf[keep], isg[keep]
    return {"vb": vb[ship], "vc": vc[ship], "vs": vs[ship], "ib_raw": ib,
            "ib": apply_homography(ib, H) if len(ib) else ib, "ic": icf, "is": isg,
            "buoy_b": vb[~ship], "buoy_c": vc[~ship], "buoy_cls": vcl[~ship]}


def _rank(c: np.ndarray) -> np.ndarray:
    r = np.empty(len(c))
    r[np.argsort(-c, kind="stable")] = np.arange(len(c))
    return r


def frame_features(cd: dict, fstat: np.ndarray, hw) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """-> (X (n, len(FEATS)), boxes (n, 4) in the VIS canvas, is_ir (n,))."""
    nv, ni = len(cd["vc"]), len(cd["ic"])
    B = np.vstack([cd["vb"], cd["ib"]]) if nv + ni else np.zeros((0, 4))
    C = np.concatenate([cd["vc"], cd["ic"]])
    isir = np.concatenate([np.zeros(nv), np.ones(ni)])
    n = nv + ni
    if n == 0:
        return np.zeros((0, len(FEATS))), B, isir
    H_, W_ = float(hw[0]), float(hw[1])
    w = np.clip(B[:, 2] - B[:, 0], 1e-3, None)
    h = np.clip(B[:, 3] - B[:, 1], 1e-3, None)
    raw = np.vstack([cd["vb"], cd["ib_raw"]])          # sigma lives in each stream's own canvas
    rw = np.clip(raw[:, 2] - raw[:, 0], 1e-3, None)
    rh = np.clip(raw[:, 3] - raw[:, 1], 1e-3, None)
    sig = np.concatenate([cd["vs"].mean(1) if nv else np.zeros(0), cd["is"].mean(1) if ni else np.zeros(0)])
    rank = np.concatenate([_rank(cd["vc"]), _rank(cd["ic"])])
    top = np.concatenate([np.full(nv, cd["vc"].max() if nv else 1.0), np.full(ni, cd["ic"].max() if ni else 1.0)])
    M = iou_matrix(B, B)
    np.fill_diagonal(M, 0.0)
    same = isir[:, None] == isir[None, :]
    Ms, Mx = np.where(same, M, 0.0), np.where(same, 0.0, M)
    higher = C[None, :] > C[:, None]
    jx = Mx.argmax(1)
    x_iou = Mx[np.arange(n), jx]
    sup = Mx >= 0.30
    ctr = np.column_stack([(B[:, 0] + B[:, 2]) / 2, (B[:, 1] + B[:, 3]) / 2])
    D = np.sqrt(((ctr[:, None, :] - ctr[None, :, :]) ** 2).sum(-1))
    D = np.where(same, np.inf, D).min(1) / np.sqrt(w * h)
    vis25, ir25 = cd["vc"] >= 0.25, cd["ic"] >= 0.25
    top3 = lambda a: float(np.sort(a)[::-1][:3].mean()) if len(a) else 0.0
    fr = np.concatenate([fstat, [vis25.sum(), ir25.sum(), top3(cd["vc"]), top3(cd["ic"])]])
    X = np.column_stack([
        isir, C, np.log(np.clip(C, 1e-6, None)), rank, C / np.clip(top, 1e-6, None),
        np.log(w * h), w / h, (B[:, 0] + B[:, 2]) / (2 * W_), (B[:, 1] + B[:, 3]) / (2 * H_),
        sig / np.sqrt(rw * rh), Ms.max(1), ((Ms >= 0.5) & higher).any(1),
        x_iou, np.where(x_iou > 0, C[jx], 0.0), (sup * C[None, :]).max(1), sup.sum(1),
        np.clip(np.nan_to_num(D, posinf=10.0), 0, 10.0), (sup & higher).any(1),
        np.broadcast_to(fr, (n, len(fr)))])
    return X.astype(np.float32), B, isir


def frame_target(B: np.ndarray, gt: dict) -> np.ndarray:
    y = np.zeros(len(B), dtype=np.float32)
    g = np.asarray(gt["boxes_xyxy"], float).reshape(-1, 4)[np.asarray(gt["cls"]).astype(int).ravel() == 0]
    if not len(B) or not len(g):
        return y
    M = iou_matrix(B, g)
    best = M.argmax(0)
    for gi, ci in enumerate(best):
        iou = M[ci, gi]
        if iou >= 0.5:
            y[ci] = max(y[ci], float((iou >= THR - 1e-9).mean()))
    return y



def wait_ready(k: int, draw: int, ic, timeout: float = 6 * 3600.0) -> None:
    """Block until the IR cache and its gate statistics for this cell exist and stopped growing
    (lets scoring start while the last caches of a build are still being written)."""
    import time as _t
    tags = [f"gauss_vis_paired_{c}" for c in VIS_CONDS if c != "clean"] + ([f"gauss_ir_paired_{ic}"] if ic else [])
    paths = [q for tag in tags for q in (ROOT / CACHE / f"seed{k}" / sub(draw) / f"{tag}.pkl",
                                         ROOT / STATS / "brightness" / sub(draw) / f"{tag}.json",
                                         ROOT / STATS / "structure" / sub(draw) / f"{tag}.json")]
    t0, last = _t.time(), None
    while True:
        if all(p.is_file() for p in paths):
            sizes = [p.stat().st_size for p in paths]
            if sizes == last:
                return
            last = sizes
        if _t.time() - t0 > timeout:
            raise SystemExit(f"timed out waiting for {paths}")
        _t.sleep(10)

def load(k: int, draw: int, ic: str | None, verbose=False):
    d = f"{STATS}/brightness/{sub(draw)}"
    return load_context(preset=PRESET, cache_dir=f"{CACHE}/seed{k}/{sub(draw)}", bright_dir=d,
                        structure_dir=f"{STATS}/structure/{sub(draw)}",
                        ir_bright=f"{d}/gauss_ir_paired_{ic or 'clean'}.json",
                        conditions=VIS_CONDS, ir_condition=ic, verbose=verbose)


def night_block(ctx) -> np.ndarray:
    """Contiguous capture-order blocks of the night run (-1 on day frames)."""
    blk = np.full(len(ctx.runs), -1)
    idx = np.flatnonzero(np.isin(ctx.runs, NIGHT_RUNS))
    order = idx[np.argsort([int(Path(ctx.vis_by_cond["clean"][i]["image_path"]).stem.split("_")[-1]) for i in idx],
                         kind="stable")]
    for b, part in enumerate(np.array_split(order, NIGHT_BLOCKS)):
        blk[part] = b
    return blk


# ---------------------------------------------------------------- extract ---------------------------
def _extract_job(job):
    k, ic = job
    ctx = load(k, TRAIN_DRAW, ic)
    H = _per_frame_h(ctx.h_frames, len(ctx.runs))
    run_id = np.array([RUNS.index(r) for r in ctx.runs])
    blk = night_block(ctx)
    Xs, ys, meta = [], [], []
    for ci, c in enumerate(VIS_CONDS):
        fs = frame_stats(ctx, TRAIN_DRAW, c, ic)
        for i, (vr, ir) in enumerate(zip(ctx.vis_by_cond[c], ctx.ir_clean)):
            u = zlib.crc32(f"{k}|{c}|{ic}|{i}".encode()) / 2**32
            if u >= KEEP_P:
                continue
            X, B, _ = frame_features(candidates(vr, ir, H[i]), fs[i], vr["image_hw"])
            if not len(X):
                continue
            Xs.append(X)
            ys.append(frame_target(B, ctx.gts[i]))
            meta.append(np.tile([k, ci, IR_CONDS.index(ic), run_id[i], blk[i], i], (len(X), 1)))
    out = OUT / "train" / f"seed{k}_{ic or 'clean'}.npz"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out, X=np.concatenate(Xs), y=np.concatenate(ys), meta=np.concatenate(meta).astype(np.int32))
    return str(out), int(sum(len(x) for x in Xs))


def cmd_extract(args) -> None:
    t0 = time.time()
    jobs = [(k, ic) for ic in IR_CONDS for k in SEEDS]
    with ProcessPoolExecutor(args.workers) as ex:
        for p, n in ex.map(_extract_job, jobs):
            print(f"[extract] {p}: {n} rows ({time.time() - t0:.0f}s)", flush=True)


# ---------------------------------------------------------------- fit -------------------------------
def fold_specs() -> dict:
    """name -> (row filter on meta columns [seed, vis_c, ir_c, run, block, frame], feature variant)."""
    sp = {}
    for r, run in enumerate(RUNS):
        sp[f"loro_{run}"] = (lambda m, r=r: m[:, 3] != r, "full")
        sp[f"noframe_{run}"] = (lambda m, r=r: m[:, 3] != r, "noframe")
        sp[f"noxm_{run}"] = (lambda m, r=r: m[:, 3] != r, "noxm")
        for fam, pred in FAMILIES.items():
            tab = np.array([[pred(c, ic) for ic in IR_CONDS] for c in VIS_CONDS])
            sp[f"lofo_{fam}_{run}"] = (lambda m, r=r, tab=tab: (m[:, 3] != r) & ~tab[m[:, 1], m[:, 2]], "full")
    night = RUNS.index(NIGHT_RUNS[0])
    for b in range(NIGHT_BLOCKS):
        sp[f"nightblk_{b}"] = (lambda m, b=b: (m[:, 3] != night) | (m[:, 4] != b), "full")
        sp[f"trustnb_{b}"] = (lambda m, b=b: (m[:, 3] != night) | (m[:, 4] != b), "trust")
    for r, run in enumerate(RUNS):
        sp[f"trust_{run}"] = (lambda m, r=r: m[:, 3] != r, "trust")
    return sp


def cmd_fit(args) -> None:
    from sklearn.ensemble import HistGradientBoostingRegressor
    t0 = time.time()
    parts = [np.load(p) for p in sorted((OUT / "train").glob("*.npz"))]
    X = np.concatenate([p["X"] for p in parts])
    y = np.concatenate([p["y"] for p in parts])
    meta = np.concatenate([p["meta"] for p in parts])
    print(f"[fit] {len(X):,} rows, {X.shape[1]} features, positives {np.mean(y > 0):.3%} "
          f"({time.time() - t0:.0f}s)", flush=True)
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    specs = fold_specs()
    names = [n for n in specs if not args.only or any(n.startswith(o) for o in args.only)]
    for name in names:
        out = OUT / "models" / f"{name}.pkl"
        if out.is_file() and not args.force:
            continue
        rows, variant = specs[name]
        sel = rows(meta)
        cols = [FEATS.index(f) for f in VARIANT_FEATS[variant]]
        cst = [MONO.get(f, 0) if (variant == "trust" or f == "conf") else 0 for f in VARIANT_FEATS[variant]]
        t1 = time.time()
        m = HistGradientBoostingRegressor(monotonic_cst=cst, **HGB).fit(X[sel][:, cols], y[sel])
        with open(out, "wb") as fh:
            pickle.dump({"model": m, "features": VARIANT_FEATS[variant], "cols": cols,
                         "rows": int(sel.sum()), "name": name}, fh)
        print(f"[fit] {name}: {int(sel.sum()):,} rows, {time.time() - t1:.0f}s", flush=True)


# ---------------------------------------------------------------- eval ------------------------------
_MODELS: dict = {}


def _models():
    if not _MODELS:
        for p in (OUT / "models").glob("*.pkl"):
            with open(p, "rb") as fh:
                _MODELS[p.stem] = pickle.load(fh)
    return _MODELS


def _predict(name: str, X: np.ndarray) -> np.ndarray:
    m = _models()[name]
    return m["model"].predict(X[:, m["cols"]])


def _fused(cands, Bs, scores, hws) -> list[dict]:
    out = []
    for cd, B, s, hw in zip(cands, Bs, scores, hws):
        out.append({"boxes_xyxy": np.vstack([B, cd["buoy_b"]]) if len(cd["buoy_b"]) else B,
                    "conf": np.concatenate([s, cd["buoy_c"]]),
                    "cls": np.concatenate([np.zeros(len(B), int), cd["buoy_cls"]]), "image_hw": hw})
    return out


def _eval_job(job):
    k, draw, ic, want = job
    wait_ready(k, draw, ic)
    ctx = load(k, draw, ic)
    n = len(ctx.runs)
    H = _per_frame_h(ctx.h_frames, n)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    blk = night_block(ctx)
    out = {}
    # a pass that adds only `net` / `lofo` reuses the baselines already in eval.json
    base_needed = want is None or any(v not in ("net", "lofo", "trust", "trustnb") for v in want) or draw not in EVAL_DRAWS
    for c in VIS_CONDS:
        if base_needed:
            r = run_systems(ctx, c)
            base = {"shipped": frame_parts(r["fused_gated"], ctx.gts), "vis": frame_parts(ctx.vis_by_cond[c], ctx.gts),
                    "ir": frame_parts(r["ir_in_vis"], ctx.gts)}
        else:
            r, base = None, {"vis": frame_parts(ctx.vis_by_cond[c], ctx.gts)}
        fs = frame_stats(ctx, draw, c, ic)
        cands, Xs, Bs, isirs = [], [], [], []
        for i, (vr, ir) in enumerate(zip(ctx.vis_by_cond[c], ctx.ir_clean)):
            cd = candidates(vr, ir, H[i])
            X, B, isir = frame_features(cd, fs[i], vr["image_hw"])
            cands.append(cd), Xs.append(X), Bs.append(B), isirs.append(isir)
        hws = [vr["image_hw"] for vr in ctx.vis_by_cond[c]]
        lens = np.array([len(x) for x in Xs])
        Xall = np.concatenate(Xs)
        starts = np.concatenate([[0], np.cumsum(lens)])

        def per_frame(model_of_frame):
            """model name per frame -> list of per-frame score arrays."""
            s = np.zeros(len(Xall))
            names = np.array(model_of_frame)
            for name in np.unique(names):
                rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in np.flatnonzero(names == name)])
                if len(rows):
                    s[rows] = _predict(name, Xall[rows])
            return [np.clip(s[starts[i]:starts[i + 1]], 1e-6, None) + 1e-7 * Xs[i][:, 1] for i in range(n)]

        variants = {"llf": [f"loro_{ru}" for ru in ctx.runs],
                    "noframe": [f"noframe_{ru}" for ru in ctx.runs],
                    "noxm": [f"noxm_{ru}" for ru in ctx.runs],
                    "nightblk": [(f"nightblk_{blk[i]}" if night[i] else f"loro_{ctx.runs[i]}") for i in range(n)],
                    "trust": [f"trust_{ru}" for ru in ctx.runs],
                    "trustnb": [(f"trustnb_{blk[i]}" if night[i] else f"trust_{ctx.runs[i]}") for i in range(n)]}
        for fam, pred in FAMILIES.items():
            if pred(c, ic):
                variants[f"lofo_{fam}"] = [f"lofo_{fam}_{ru}" for ru in ctx.runs]
        have = set(_models())
        variants = {v: m for v, m in variants.items()
                    if (want is None or v.split("_")[0] in want or v in want) and set(m) <= have}
        parts = dict(base)
        for v, mnames in variants.items():
            parts[v] = frame_parts(_fused(cands, Bs, per_frame(mnames), hws), ctx.gts)
        if (want is None or "net" in want) and all((OUT / "net" / f"loro_{ru}.pt").is_file() for ru in RUNS):
            from llf_net import predict_frames
            pn = predict_frames(Xs, [f"loro_{ru}" for ru in ctx.runs])
            parts["net"] = frame_parts(_fused(cands, Bs, [np.clip(q, 1e-6, None) + 1e-7 * X[:, 1] if len(X) else q
                                                          for q, X in zip(pn, Xs)], hws), ctx.gts)
        # mechanics check: the VIS candidates at their own confidence, IR dropped, through the same
        # record builder and AP path, must reproduce VIS-only exactly
        idp = frame_parts(_fused(cands, [cd["vb"] for cd in cands], [cd["vc"] for cd in cands], hws), ctx.gts)
        for s, idx in sl.items():
            a, b = ship_ap(idp, idx), ship_ap(base["vis"], idx)
            if abs(a - b) > 1e-12:
                raise SystemExit(f"LLF mechanics check failed on seed{k} draw{draw} {c}/{ic} {s}: {a} vs {b}")
            for name, p in parts.items():
                if name == "vis" and not base_needed:
                    continue
                out[(c, ic, s, name)] = ship_ap(p, idx)
            if r is not None:
                out[(c, ic, s, "veto")] = float(np.asarray(r["veto_vis"], bool)[idx].mean())
    return k, draw, ic, out


def cmd_eval(args) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", str(args.threads))
    t0 = time.time()
    jobs = [(k, dr, ic, args.variants) for dr in args.draws for ic in IR_CONDS for k in SEEDS]
    ep = OUT / "eval.json"
    res = json.loads(ep.read_text(encoding="utf-8")) if ep.is_file() else {}
    # per-job checkpoints for this pass: a restart resumes instead of starting over
    ck = OUT / "ckpt" / ("-".join(sorted(args.variants)) if args.variants else "all")
    ck.mkdir(parents=True, exist_ok=True)
    todo = []
    for job in jobs:
        f = ck / f"{job[0]}_{job[1]}_{job[2] or 'clean'}.json"
        if f.is_file():
            res.update(json.loads(f.read_text(encoding="utf-8")))
        else:
            todo.append(job)
    print(f"[eval] {len(jobs) - len(todo)} jobs from checkpoints, {len(todo)} to run", flush=True)
    with ProcessPoolExecutor(args.workers) as ex:
        for n_done, (k, dr, ic, out) in enumerate(ex.map(_eval_job, todo), 1):
            got = {f"{k}|{dr}|{c}|{ic_ or 'clean'}|{s}|{name}": v for (c, ic_, s, name), v in out.items()}
            (ck / f"{k}_{dr}_{ic or 'clean'}.json").write_text(json.dumps(got), encoding="utf-8")
            res.update(got)
            print(f"[eval] {n_done}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    # merge into whatever is on disk now: another pass may have written since this one started
    latest = json.loads(ep.read_text(encoding="utf-8")) if ep.is_file() else {}
    latest.update(res)
    ep.write_text(json.dumps(latest, indent=0), encoding="utf-8")
    print(f"[eval] wrote {ep} ({time.time() - t0:.0f}s)")


# ---------------------------------------------------------------- report ----------------------------
def cmd_report(args) -> None:
    res = json.loads((OUT / "eval.json").read_text(encoding="utf-8"))
    D, P = tuple(args.draws), args.primary

    def arr(c, ic, s, name):
        return np.array([np.mean([res[f"{k}|{dr}|{c}|{ic or 'clean'}|{s}|{name}"] for dr in D])
                         for k in SEEDS])

    def ci(d):
        m, h = float(d.mean()), float(T_975_DF4 * d.std(ddof=1) / np.sqrt(len(d)))
        return m, m - h, m + h
    f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]"
    cells = [(c, ic, s) for c in VIS_CONDS for ic in IR_CONDS for s in ("day", "night")]
    floor = 0.0060
    L, js = [], {"cells": [], "summary": {}}

    def claim_fails(t):
        return t[0] <= -floor and t[2] < 0

    L += [f"## 1. Primary: `{P}` out of fold (leave-one-run-out), verdict draws {D[0]}–{D[-1]}", "",
          f"| VIS | IR | slice | VIS | IR | shipped | {P} | {P} − max(VIS, IR) | {P} − shipped |",
          "|---|---|---|---:|---:|---:|---:|---|---|"]
    summ = {"llf_holds": 0, "shipped_holds": 0, "harm": 0, "gain": 0, "harm_0014": 0, "gain_0014": 0}
    for c, ic, s in cells:
        vis, ir, sh, ll = (arr(c, ic, s, n) for n in ("vis", "ir", "shipped", P))
        mx = np.maximum(vis, ir)
        cl, cs, dd = ci(ll - mx), ci(sh - mx), ci(ll - sh)
        summ["llf_holds"] += not claim_fails(cl)
        summ["shipped_holds"] += not claim_fails(cs)
        summ["harm"] += dd[0] <= -floor and dd[2] < 0
        summ["gain"] += dd[0] >= floor and dd[1] > 0
        summ["harm_0014"] += dd[0] <= -0.0014 and dd[2] < 0
        summ["gain_0014"] += dd[0] >= 0.0014 and dd[1] > 0
        L.append(f"| {c} | {ic or 'clean'} | {s} | {vis.mean():.4f} | {ir.mean():.4f} | {sh.mean():.4f} | "
                 f"{ll.mean():.4f} | {f(cl)} {'**fails**' if claim_fails(cl) else 'holds'} | {f(dd)} |")
        js["cells"].append({"vis": c, "ir": ic or "clean", "slice": s, "vis_ap": float(vis.mean()),
                            "ir_ap": float(ir.mean()), "shipped": float(sh.mean()), "llf": float(ll.mean()),
                            "llf_minus_max": list(cl), "shipped_minus_max": list(cs), "llf_minus_shipped": list(dd)})
    js["summary"] = summ
    p1 = summ["llf_holds"] == len(cells)
    p2 = summ["harm"] == 0 and summ["gain"] >= 1
    L += ["", f"Claim fused ≥ max(VIS, IR) (fails at ≤ −{floor} with the interval below zero) holds on "
          f"**{summ['llf_holds']} of {len(cells)}** cells under `{P}`, {summ['shipped_holds']} under the shipped preset.",
          f"`{P}` against the shipped preset: {summ['harm']} cells worse and {summ['gain']} better beyond {floor} "
          f"(at 0.0014: {summ['harm_0014']} worse, {summ['gain_0014']} better).", "",
          f"**P1 (every cell at or above max(VIS, IR)): {'PASS' if p1 else 'FAIL'}. "
          f"P2 (no cell worse than shipped, at least one better, at {floor}): {'PASS' if p2 else 'FAIL'}.**"]
    js["summary"].update({"P1": p1, "P2": p2})

    # secondary analyses
    def sec(title, name_of_cell, cell_filter=lambda c, ic, s: True):
        probe = next(((c, ic, s) for c, ic, s in cells if name_of_cell(c, ic, s) and cell_filter(c, ic, s)), None)
        if probe is None or f"0|{D[0]}|{probe[0]}|{probe[1] or 'clean'}|{probe[2]}|{name_of_cell(*probe)}" not in res:
            return [f"## {title}", "", "Not scored yet.", ""]
        rows, holds, n = [], 0, 0
        for c, ic, s in cells:
            name = name_of_cell(c, ic, s)
            if name is None or not cell_filter(c, ic, s):
                continue
            vis, ir, sh, ll, v = (arr(c, ic, s, nm) for nm in ("vis", "ir", "shipped", P, name))
            cl, dd = ci(v - np.maximum(vis, ir)), ci(v - ll)
            holds += not claim_fails(cl)
            n += 1
            rows.append(f"| {c} | {ic or 'clean'} | {s} | {v.mean():.4f} | {f(cl)} {'**fails**' if claim_fails(cl) else 'holds'} | {f(dd)} |")
        return [f"## {title}", "", f"Claim holds on {holds} of {n} cells.", "",
                "| VIS | IR | slice | AP | − max(VIS, IR) | − LLF primary |", "|---|---|---|---:|---|---|"] + rows + [""]

    if P == "llf" and f"0|{D[0]}|clean|clean|day|net" in res:
        nn = {"holds": 0, "harm": 0, "gain": 0}
        rows = []
        for c, ic, s in cells:
            vis, ir, sh, nt, ll = (arr(c, ic, s, n) for n in ("vis", "ir", "shipped", "net", P))
            cl, dd, dl = ci(nt - np.maximum(vis, ir)), ci(nt - sh), ci(nt - ll)
            nn["holds"] += not claim_fails(cl)
            nn["harm"] += dd[0] <= -floor and dd[2] < 0
            nn["gain"] += dd[0] >= floor and dd[1] > 0
            rows.append(f"| {c} | {ic or 'clean'} | {s} | {nt.mean():.4f} | {f(cl)} {'**fails**' if claim_fails(cl) else 'holds'} "
                        f"| {f(dd)} | {f(dl)} |")
        np1, np2 = nn["holds"] == len(cells), nn["harm"] == 0 and nn["gain"] >= 1
        L += ["", "## 1b. LLF-Net (set transformer, Amendment 1), same folds and draws", "",
              f"Claim holds on **{nn['holds']} of {len(cells)}** cells; against shipped {nn['harm']} worse, {nn['gain']} better "
              f"beyond {floor}. **P1: {'PASS' if np1 else 'FAIL'}. P2: {'PASS' if np2 else 'FAIL'}.**", "",
              "| VIS | IR | slice | AP | − max(VIS, IR) | − shipped | − LLF |", "|---|---|---|---:|---|---|---|"] + rows
        js["summary"]["net"] = {**nn, "P1": np1, "P2": np2}
    L += [""] + sec("2. Night seen in training (pohang01 in four capture-order blocks)",
                    lambda c, ic, s: "trustnb" if P == "trust" else "nightblk", lambda c, ic, s: s == "night")
    if P != "trust":                      # LLF-2 registered the primary and the night blocks only
        for fam, pred in FAMILIES.items():
            L += sec(f"3. Leave one corruption family out: {fam} (never in training, nor the test run)",
                     lambda c, ic, s, fam=fam, pred=pred: f"lofo_{fam}" if pred(c, ic) else None)
        L += sec("4. Ablation: no frame-health features", lambda c, ic, s: "noframe")
        L += sec("5. Ablation: no cross-stream features", lambda c, ic, s: "noxm")
    out = ROOT / args.out
    if P == "trust":
        head = ("# LLF-2, learned stream trust, development, registered\n\n"
                "Registered in `docs/prereg-llf2-2026-10-10.md`. Five Phase 3 systems; every frame scored by a "
                "model that never saw its run; training rows from VIS draw 941 / IR 951, scored on the fresh draws "
                f"VIS {', '.join(map(str, D))} / IR {', '.join(str(d + 10) for d in D)} (per-system draw mean); "
                "ship AP; 95% t-intervals over systems (df 4).\n\n")
    else:
        head = ("# Learned late fusion (LLF), development, registered\n\n"
                "Registered in `docs/prereg-llf-2026-10-10.md`. Five Phase 3 systems; every frame scored by a model "
                "that never saw its run; trained on VIS draw 941 / IR 951, scored on VIS 942–944 / IR 952–954 "
                "(per-system draw mean); ship AP; 95% t-intervals over systems (df 4).\n\n")
    out.write_text(head + "\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(js, indent=1), encoding="utf-8")
    print("\n".join(L[:8] + L[-40:]))
    print(f"[report] wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--workers", type=int, default=10)
    fi = sp.add_parser("fit")
    fi.add_argument("--only", nargs="*", default=None)
    fi.add_argument("--force", action="store_true")
    ev = sp.add_parser("eval")
    ev.add_argument("--workers", type=int, default=8)
    ev.add_argument("--threads", type=int, default=4)
    ev.add_argument("--draws", type=int, nargs="+", default=list(EVAL_DRAWS))
    ev.add_argument("--variants", nargs="*", default=None,
                    help="llf noframe noxm nightblk lofo (default: every variant whose models exist)")
    rp = sp.add_parser("report")
    rp.add_argument("--out", default="docs/eval/llf/llf_2026-10-10.md")
    rp.add_argument("--draws", type=int, nargs="+", default=list(EVAL_DRAWS))
    rp.add_argument("--primary", default="llf", help="llf, or trust (LLF-2)")
    args = ap.parse_args()
    {"extract": cmd_extract, "fit": cmd_fit, "eval": cmd_eval, "report": cmd_report}[args.cmd](args)
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:                                    # noqa: BLE001
        pass
    raise SystemExit(main())
