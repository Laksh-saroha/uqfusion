"""Test-time BatchNorm adaptation of the VIS detector, vouched for by IR (N4, 2026-10-11).

Registered in `docs/prereg-ttbn-2026-10-11.md`. A corrupted camera shifts every layer's activation
statistics; the detector normalises them with the statistics of its training data. BN-mix (Schneider et
al., NeurIPS 2020) re-estimates them on the test batch and mixes them with the source statistics at prior
strength N:

    mu = (N mu_src + n mu_batch) / (N + n),   var likewise,   n = 8 frames.

No weight changes, no labels. The batch is 8 consecutive frames of the stream in capture order
(non-overlapping blocks, so a frame's statistics include up to 7 later frames: 0.7 s of look-ahead at
10 Hz). Development screen (draw 941, seeds 0-1): large gains by day (VIS lowlight +0.06..+0.08 ship AP),
losses at night and -0.007 on clean day frames. So the adapted records are used only where VIS reports
itself unhealthy (`q_vis < 1`, the shipped VIS health score) AND IR reads day and is healthy
(`~ir_night_raw & ir_d2 <= ir_bound`, the shipped night threshold and IR merge-health bound): a dark or
veiled camera that the thermal camera places in daylight is underexposed or fogged, not in the night.
Elsewhere the plain records stand. IR is not adapted. Fusion is E4 `union`.

A block none of whose frames is gated (under every IR condition scored) is not run: its adapted records
would never be used, and a block's statistics depend only on its own frames.

    py -3.13 scripts/ttbn.py build --target dev --conds clean fog fog_s1 lowlight glare
    py -3.13 scripts/ttbn.py score-dev && py -3.13 scripts/ttbn.py report-dev
    py -3.13 scripts/ttbn.py check-fp16                    # decides pohang04's precision
    py -3.13 scripts/ttbn.py build --target p04 --conds fog lowlight blur_s3 noise_s2 rain_s2 clean [--fp16]
    py -3.13 scripts/ttbn.py score-p04 && py -3.13 scripts/ttbn.py report-p04
    py -3.13 scripts/ttbn.py score-dev-lex && py -3.13 scripts/ttbn.py report-dev-lex    # Amendment 2
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

SEEDS = (0, 1, 2, 3, 4)
PRIOR_N = 16.0
BATCH = 8
DEV_LIST = ROOT / "runs/derived/paired_val_vis.txt"
DEV_DRAWS = (942, 943, 944)
DEV_CONDS = {"clean": None, "fog": ("fog", 2), "fog_s1": ("fog", 1), "lowlight": ("lowlight", 2),
             "glare": ("glare", 2)}
DEV_IR_CONDS = (None, "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")
P04_CONDS = {"clean": None, "fog": ("fog", 2), "lowlight": ("lowlight", 2), "blur_s3": ("blur", 3),
             "noise_s2": ("noise", 2), "rain_s2": ("rain", 2)}
#: the IR conditions each pohang04 VIS condition is paired with in the registered cells
P04_IR_OF = {"clean": (None, "blur_s2", "noise_s2", "fog_s2")}
DEV_OUT = ROOT / "runs/novel/ttbn_dev"
P04_OUT = ROOT / "runs/holdout_p04/ttbn"
FP16_CHECK = DEV_OUT / "fp16_check.json"
FP16_TOL = 0.0010
T_975_DF4 = 2.7764451051977987
_W: dict = {}
MODE = {"n": None}


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def out_path(target: str, k: int, cond: str, draw: int | None, fp16: bool = False) -> Path:
    base = DEV_OUT if target == "dev" else P04_OUT
    where = "clean" if cond == "clean" else sub(draw)
    return base / f"seed{k}" / where / f"gauss_vis_paired_{cond}_bn{int(PRIOR_N)}{'_fp16' if fp16 else ''}.pkl"


# ------------------------------------------------------------------ the gate from frame statistics
_SC: dict = {}


def _sconst() -> dict:
    if not _SC:
        _SC.update(json.loads((ROOT / "runs/eval/structure_constants.json").read_text(encoding="utf-8")))
    return _SC


def _d2(path: Path, m: dict) -> np.ndarray:
    fr = json.loads(Path(path).read_text(encoding="utf-8"))["frames"]
    X = np.stack([[f[k] for k in m["keys"]] for f in fr]).astype(float)
    for j, k in enumerate(m["keys"]):
        if k in m["log1p_keys"]:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Z = (X - np.asarray(m["mean"])) / np.asarray(m["std"])
    return np.einsum("ij,jk,ik->i", Z, np.asarray(m["precision"]), Z)


def vis_unhealthy(vis_struct: Path) -> np.ndarray:
    """`q_vis < 1`, as `load_context` computes it."""
    vh = _sconst()["axes"]["vis_health"]
    q = np.clip(float(vh["bound"]) / np.maximum(_d2(vis_struct, vh), 1e-12), 0, 1)
    return q < 1.0


def ir_day_ok(ir_bright: Path, ir_struct: Path) -> np.ndarray:
    """IR reads day (`~(ir_p05 > 41.5)`) and is healthy by the merge bound (`ir_d2 <= ir_bound`)."""
    s = _sconst()["axes"]
    p05 = np.asarray([f["p05"] for f in json.loads(Path(ir_bright).read_text(encoding="utf-8"))["frames"]], float)
    return (~(p05 > float(s["ir_p05"]["threshold"]))) & (_d2(ir_struct, s["ir_health"]) <= float(s["ir_health"]["bound"]))


def _dev_stats(draw: int, mod: str, cond, kind: str) -> Path:
    import seed_ensemble as se
    return ROOT / se.STATS / kind / sub(draw) / f"gauss_{mod}_paired_{cond or 'clean'}.json"


def _p04_stats(mod: str, cond, draw, kind: str) -> Path:
    import p04_heldout as ph
    c = cond or "clean"
    return ph.stats_dir(kind, c, None if c == "clean" else draw) / f"gauss_{mod}_paired_{c}.json"


def needed(target: str, cond: str, draw: int | None) -> np.ndarray:
    """Frames gated under any IR condition (and, for the draw-free clean VIS stream, any draw) scored."""
    out = None
    if target == "dev":
        for d in (DEV_DRAWS if cond == "clean" else (draw,)):
            q = vis_unhealthy(_dev_stats(d, "vis", cond, "structure"))
            for ic in DEV_IR_CONDS:
                g = q & ir_day_ok(_dev_stats(d, "ir", ic, "brightness"), _dev_stats(d, "ir", ic, "structure"))
                out = g if out is None else out | g
    else:
        import p04_heldout as ph
        for d in (ph.VIS_DRAWS if cond == "clean" else (draw,)):
            q = vis_unhealthy(_p04_stats("vis", cond, d, "structure"))
            for ic in P04_IR_OF.get(cond, (None,)):
                g = q & ir_day_ok(_p04_stats("ir", ic, d, "brightness"), _p04_stats("ir", ic, d, "structure"))
                out = g if out is None else out | g
    return out


def gate_p04(vc: str, ic, s: str) -> np.ndarray:
    """The gate of one registered pohang04 cell and draw, from its frame-statistics files."""
    draw = None if s == "clean" else int(s[4:7])
    return (vis_unhealthy(_p04_stats("vis", vc, draw, "structure"))
            & ir_day_ok(_p04_stats("ir", ic, draw, "brightness"), _p04_stats("ir", ic, draw, "structure")))


def gate_ctx(ctx, cond: str) -> np.ndarray:
    """The gate from a loaded context: `q_vis < 1` & IR reads day & IR within the merge bound."""
    assert float(ctx.ir_bound) == float(_sconst()["axes"]["ir_health"]["bound"]), "context's IR bound differs"
    return ((np.asarray(ctx.q_vis_by_cond[cond]) < 1.0) & ~np.asarray(ctx.ir_night_raw, bool)
            & (np.asarray(ctx.ir_d2) <= ctx.ir_bound))


# ------------------------------------------------------------------ inference
def patch_bn(model) -> int:
    """Every BatchNorm2d normalises with the BN-mix statistics while MODE['n'] is set."""
    import torch
    import torch.nn.functional as F
    n_bn = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            n_bn += 1

            def fwd(x, m=m):
                N = MODE["n"]
                if N is None:
                    return F.batch_norm(x, m.running_mean, m.running_var, m.weight, m.bias, False, 0.0, m.eps)
                if MODE.get("fused"):
                    # fp16 builds: one fused pass with fp32 accumulation (SyncBN's kernel), ~6x faster
                    # than an fp32 copy + two reductions; equal to them to ~5e-7 relative. The fp16
                    # development check (check-fp16) prices this together with the precision.
                    mu_t, invstd = torch.batch_norm_stats(x, 0.0)
                    var_t = invstd.pow(-2)
                else:
                    xf = x.float()
                    mu_t = xf.mean(dim=(0, 2, 3))
                    var_t = xf.var(dim=(0, 2, 3), unbiased=False)
                a = x.shape[0] / (N + x.shape[0])
                mu = (1 - a) * m.running_mean + a * mu_t
                var = (1 - a) * m.running_var + a * var_t
                return F.batch_norm(x, mu, var, m.weight, m.bias, False, 0.0, m.eps)
            m.forward = fwd
    return n_bn


def _low() -> None:
    """This process yields the CPU to the pohang04 build (BELOW_NORMAL_PRIORITY_CLASS)."""
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)


def _init(spec):
    import cv2
    cv2.setNumThreads(1)
    _low()                                  # decoding and corruption yield to the main build
    from uqfusion.eval.corruptions import make_corruption
    _W.update(spec)
    _W["tf"] = (make_corruption(spec["kind"], spec["sev"], spec["seed"], version="v2", modality="vis",
                                images=spec["images"]) if spec.get("kind") else None)


def _chunk(idx):
    import cv2
    out = []
    for i in idx:
        im = cv2.imread(_W["images"][i])
        if im is None:
            raise FileNotFoundError(_W["images"][i])
        if _W["tf"] is not None:
            im = _W["tf"](im, i)
        out.append((i, im))
    return out


def _frames(spec, chunks, workers):
    """Decoded (and corrupted) chunks in order, at most 3 x workers in flight, so memory stays bounded."""
    from collections import deque
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(spec,)) as ex:
        q: deque = deque()
        it = iter(chunks)
        for c in it:
            q.append(ex.submit(_chunk, c))
            if len(q) >= 3 * workers:
                break
        while q:
            yield q.popleft().result()
            nxt = next(it, None)
            if nxt is not None:
                q.append(ex.submit(_chunk, nxt))


def cmd_build(args) -> int:
    import torch
    import p04_heldout as ph
    from uqfusion.eval.corruptions_v2 import CODE_REV
    from uqfusion.uq.infer import UQPredictor
    if args.target == "p04" and args.fp16:
        chk = json.loads(FP16_CHECK.read_text(encoding="utf-8")) if FP16_CHECK.is_file() else {}
        if not chk.get("pass"):
            raise SystemExit("pohang04 in fp16 needs a passed development check (check-fp16)")
    conds = DEV_CONDS if args.target == "dev" else P04_CONDS
    draws = args.draws or (DEV_DRAWS if args.target == "dev" else ph.VIS_DRAWS)
    lst = ([ln.strip() for ln in open(DEV_LIST) if ln.strip()] if args.target == "dev" else ph.images("vis"))
    todo = []
    for c in args.conds:
        for d in ([None] if c == "clean" else draws):
            if not all(out_path(args.target, k, c, d, args.fp16).is_file() for k in SEEDS):
                todo.append((c, d))
    print(f"[ttbn] {args.target}: {len(lst)} frames, {len(todo)} streams to build "
          f"({'fp16' if args.fp16 else 'fp32'})", flush=True)
    if not todo:
        return 0
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=0, imgsz=640,
                         conf=0.001) for k in SEEDS]
    nb = [patch_bn(p.model) for p in preds]
    if args.fp16:
        ph.setup_fp16(preds)
        MODE["fused"] = True
    print(f"[ttbn] BN layers {nb}", flush=True)
    for c, d in todo:
        kc = conds[c]
        need = needed(args.target, c, d)
        assert len(need) == len(lst)
        idx = list(range(len(lst)))
        chunks = [idx[a:a + BATCH] for a in range(0, len(idx), BATCH) if need[a:a + BATCH].any()]
        recs = [[None] * len(lst) for _ in SEEDS]
        t0 = time.time()
        if chunks:
            spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": d or 0, "images": lst}
            MODE["n"] = PRIOR_N
            for n_done, ch in enumerate(_frames(spec, chunks, args.workers), 1):
                with torch.no_grad(), ph.amp(args.fp16):
                    out = ph._multi_batch(preds, [im for _, im in ch])
                for j in range(len(SEEDS)):
                    for (i, _), r in zip(ch, out[j]):
                        recs[j][i] = {**(ph._f32(r) if args.fp16 else r), "image_path": lst[i]}
                if n_done % 200 == 0:
                    print(f"[ttbn] {c}/{d} {n_done}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
            MODE["n"] = None
        for j, k in enumerate(SEEDS):
            o = out_path(args.target, k, c, d, args.fp16)
            o.parent.mkdir(parents=True, exist_ok=True)
            meta = {"target": args.target, "cond": c, "draw": d, "prior_N": PRIOR_N, "batch": BATCH,
                    "blocks": "non-overlapping, capture order; blocks with no gated frame not run",
                    "gated_frames": int(need.sum()), "blocks_run": len(chunks),
                    "precision": "fp16 autocast, channels_last, cudnn.benchmark" if args.fp16 else "fp32",
                    "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                    "corrupt_seed": d if kc else None, "corrupt_version": "v2" if kc else None,
                    "corrupt_code": CODE_REV if kc else None,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": meta, "records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[ttbn] {c}/{d} done: {len(chunks)} blocks, {int(need.sum())} gated frames, "
              f"{time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ scoring helpers
def gated_stream(target, k, cond, draw, plain, gate, fp16=False):
    """Adapted records where `gate` fires, the plain records elsewhere. Each adapted record keeps the
    plain record's frame identity and `feat` (the feature hook is not read: `feat` feeds only the
    Mahalanobis distance, which leaves the gate-free fusion unchanged)."""
    if not np.any(gate):
        return plain                                         # the gate never fires: the plain stream exactly
    with open(out_path(target, k, cond, draw, fp16), "rb") as fh:
        rr = pickle.load(fh)["records"]
    assert len(rr) == len(plain) == len(gate)
    out = []
    for a, b, f in zip(rr, plain, gate):
        if not f:
            out.append(b)
            continue
        if a is None:
            raise SystemExit(f"gated frame without an adapted record ({target}, {k}, {cond}, {draw})")
        assert Path(a["image_path"]).name == Path(b["image_path"]).name
        out.append({**a, "image_path": b["image_path"], "feat": b["feat"]})
    return out


# ------------------------------------------------------------------ development
def _dev_job(spec):
    import dataclasses
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems
    from v2_sensitivity import ship_ap
    draw, ir_cond, k, conds, fp16 = spec
    _low()
    se._memo_distances()
    d = f"{se.STATS}/brightness/{sub(draw)}"
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed{k}/{sub(draw)}", bright_dir=d,
                       structure_dir=f"{se.STATS}/structure/{sub(draw)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json",
                       conditions=tuple(conds), ir_condition=ir_cond, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    u = dataclasses.replace(ctx, veto=None)
    res = {}
    for c in conds:
        g = gate_ctx(ctx, c)
        vis = ctx.vis_by_cond[c]
        vis_g = gated_stream("dev", k, c, None if c == "clean" else draw, vis, g, fp16)
        o0 = run_systems(u, c, compute_metrics=False)
        o1 = run_systems(u, c, compute_metrics=False, vis_records=vis_g)
        parts = {"vis": frame_parts(vis, ctx.gts), "ir": frame_parts(o0["ir_in_vis"], ctx.gts),
                 "union": frame_parts(o0["fused_gated"], ctx.gts), "vis_ttbn": frame_parts(vis_g, ctx.gts),
                 "union_ttbn": frame_parts(o1["fused_gated"], ctx.gts)}
        if not fp16:
            parts["shipped"] = frame_parts(run_systems(ctx, c, compute_metrics=False)["fused_gated"], ctx.gts)
        for s, idx in sl.items():
            res[f"{c}|{s}"] = {n: ship_ap(p, idx) for n, p in parts.items()}
            res[f"{c}|{s}"]["gate"] = float(g[idx].mean())
    return spec, res


def _run_dev(specs, out_p: Path, workers: int, key_of, fn=None) -> dict:
    allres = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    specs = [s for s in specs if key_of(s) not in allres]
    print(f"[score] {len(specs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(workers, max_tasks_per_child=1) as ex:
        for n, (spec, res) in enumerate(ex.map(fn or _dev_job, specs), 1):
            allres[key_of(spec)] = res
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(allres), encoding="utf-8")
            os.replace(tmp, out_p)
            print(f"[score] {n}/{len(specs)} {key_of(spec)} ({time.time() - t0:.0f}s)", flush=True)
    return allres


def cmd_score_dev(args) -> int:
    specs = [(d, ic, k, tuple(DEV_CONDS), False) for d in DEV_DRAWS for ic in DEV_IR_CONDS for k in SEEDS]
    specs.sort(key=lambda s: (s[1] is not None, s[0], s[2]))      # IR clean first: the cells the gate covers most
    _run_dev(specs, DEV_OUT / "score.json", args.workers, lambda s: f"{s[0]}|{s[1]}|{s[2]}")
    return 0


def _dev_lex_job(spec):
    """Amendment 2 (descriptive): E9 `lex` on the plain VIS stream and on the gated (`union_ttbn`) VIS
    stream, each over the plain IR stream mapped to the VIS canvas."""
    import dataclasses
    import seed_ensemble as se
    from gate_zoo import build_lex_records
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems
    from v2_sensitivity import ship_ap
    draw, ir_cond, k, conds, fp16 = spec
    _low()
    se._memo_distances()
    d = f"{se.STATS}/brightness/{sub(draw)}"
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed{k}/{sub(draw)}", bright_dir=d,
                       structure_dir=f"{se.STATS}/structure/{sub(draw)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json",
                       conditions=tuple(conds), ir_condition=ir_cond, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    u = dataclasses.replace(ctx, veto=None)
    res = {}
    for c in conds:
        g = gate_ctx(ctx, c)
        vis = ctx.vis_by_cond[c]
        vis_g = gated_stream("dev", k, c, None if c == "clean" else draw, vis, g, fp16)
        irv = run_systems(u, c, compute_metrics=False)["ir_in_vis"]
        z = np.zeros(len(vis), bool)
        parts = {"lex": frame_parts(build_lex_records(vis, irv, z), ctx.gts),
                 "ttbn_lex": frame_parts(build_lex_records(vis_g, irv, z), ctx.gts)}
        for s, idx in sl.items():
            res[f"{c}|{s}"] = {n: ship_ap(p, idx) for n, p in parts.items()}
    return spec, res


def cmd_score_dev_lex(args) -> int:
    specs = [(d, ic, k, tuple(DEV_CONDS), False) for d in DEV_DRAWS for ic in DEV_IR_CONDS for k in SEEDS]
    specs.sort(key=lambda s: (s[1] is not None, s[0], s[2]))
    _run_dev(specs, DEV_OUT / "score_lex.json", args.workers, lambda s: f"{s[0]}|{s[1]}|{s[2]}", _dev_lex_job)
    return 0


def cmd_report_dev_lex(args) -> int:
    """Amendment 2, descriptive: Δ = ttbn_lex − lex per cell (mean over draws per system, t-interval over
    systems), Δ vs shipped, and the claim with VIS = the gated stream; D-H/D-G counts as description."""
    res = json.loads((DEV_OUT / "score.json").read_text(encoding="utf-8"))
    lx = json.loads((DEV_OUT / "score_lex.json").read_text(encoding="utf-8"))
    rows, out, worse, better = [], {}, [], []
    for c in DEV_CONDS:
        for ic in DEV_IR_CONDS:
            for s in ("day", "night"):
                per_k = {n: [] for n in ("d", "d_ship", "lex", "ttbn_lex", "claim", "claim_lex", "gate")}
                for k in SEEDS:
                    rs = [res[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in DEV_DRAWS]
                    ls = [lx[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in DEV_DRAWS]
                    per_k["d"].append(np.mean([b["ttbn_lex"] - b["lex"] for b in ls]))
                    per_k["d_ship"].append(np.mean([b["ttbn_lex"] - r["shipped"] for r, b in zip(rs, ls)]))
                    per_k["lex"].append(np.mean([b["lex"] for b in ls]))
                    per_k["ttbn_lex"].append(np.mean([b["ttbn_lex"] for b in ls]))
                    per_k["claim"].append(np.mean([b["ttbn_lex"] - max(r["vis_ttbn"], r["ir"]) for r, b in zip(rs, ls)]))
                    per_k["claim_lex"].append(np.mean([b["lex"] - max(r["vis"], r["ir"]) for r, b in zip(rs, ls)]))
                    per_k["gate"].append(np.mean([r["gate"] for r in rs]))
                m, lo, hi = _cell_stats(per_k["d"])
                ms, los, his = _cell_stats(per_k["d_ship"])
                cl, cllo, clhi = _cell_stats(per_k["claim"])
                cx = float(np.mean(per_k["claim_lex"]))
                cell = f"{c}/{ic or 'clean'}/{s}"
                out[cell] = {"delta": m, "ci": [lo, hi], "delta_vs_shipped": ms, "ci_vs_shipped": [los, his],
                             "gate": float(np.mean(per_k["gate"])), "lex": float(np.mean(per_k["lex"])),
                             "ttbn_lex": float(np.mean(per_k["ttbn_lex"])), "claim": [cl, cllo, clhi],
                             "claim_lex": cx, "per_system": [float(x) for x in per_k["d"]]}
                if m <= -0.0014 and hi < 0:
                    worse.append(cell)
                if m >= 0.0060 and lo > 0:
                    better.append(cell)
                rows.append(f"| {cell} | {out[cell]['gate']:.3f} | {out[cell]['lex']:.4f} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                            f"{ms:+.4f} [{los:+.4f}, {his:+.4f}] | {cx:+.4f} | {cl:+.4f} [{cllo:+.4f}, {clhi:+.4f}] |")
    L = ["# N4 under lex ordering (`ttbn_lex`), development, descriptive", "",
         "`docs/prereg-ttbn-2026-10-11.md`, Amendment 2: no verdict is registered. Draws VIS 942–944 / IR 952–954, "
         "five systems. Δ = `ttbn_lex` − `lex` (ship AP), mean over the three draws per system, then over the five "
         "systems; 95% t-interval (df 4). Claim = fused − max(VIS, IR), VIS = that arm's VIS stream.", "",
         f"Counts with D-H/D-G's thresholds (description): {len(worse)} cells with Δ ≤ −0.0014 and interval below "
         f"zero ({', '.join(worse) or 'none'}); {len(better)} cells with Δ ≥ +0.0060 and interval above zero "
         f"({', '.join(better) or 'none'}).", "",
         "| cell (VIS/IR/slice) | gate | lex | Δ vs lex [95%] | Δ vs shipped [95%] | claim lex | claim ttbn_lex [95%] |",
         "|---|---:|---:|---:|---:|---:|---:|"] + rows
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"worse": worse, "better": better, "cells": out}, indent=1),
                                      encoding="utf-8")
    print("\n".join(L[:6]))
    return 0


def _cell_stats(vals: list[float]) -> tuple[float, float, float]:
    v = np.asarray(vals, float)
    m = float(v.mean())
    h = float(T_975_DF4 * v.std(ddof=1) / np.sqrt(len(v))) if len(v) > 1 else float("nan")
    return m, m - h, m + h


def cmd_report_dev(args) -> int:
    res = json.loads((DEV_OUT / "score.json").read_text(encoding="utf-8"))
    rows, out = [], {}
    worse, better = [], []
    for c in DEV_CONDS:
        for ic in DEV_IR_CONDS:
            for s in ("day", "night"):
                per_k = {n: [] for n in ("d", "d_ship", "gate", "union", "union_ttbn", "claim", "claim_u")}
                for k in SEEDS:
                    rs = [res[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in DEV_DRAWS]
                    per_k["d"].append(np.mean([r["union_ttbn"] - r["union"] for r in rs]))
                    per_k["d_ship"].append(np.mean([r["union_ttbn"] - r["shipped"] for r in rs]))
                    per_k["gate"].append(np.mean([r["gate"] for r in rs]))
                    per_k["union"].append(np.mean([r["union"] for r in rs]))
                    per_k["union_ttbn"].append(np.mean([r["union_ttbn"] for r in rs]))
                    per_k["claim"].append(np.mean([r["union_ttbn"] - max(r["vis_ttbn"], r["ir"]) for r in rs]))
                    per_k["claim_u"].append(np.mean([r["union"] - max(r["vis"], r["ir"]) for r in rs]))
                m, lo, hi = _cell_stats(per_k["d"])
                ms, los, his = _cell_stats(per_k["d_ship"])
                cl, cllo, clhi = _cell_stats(per_k["claim"])
                cu, culo, cuhi = _cell_stats(per_k["claim_u"])
                cell = f"{c}/{ic or 'clean'}/{s}"
                out[cell] = {"delta": m, "ci": [lo, hi], "delta_vs_shipped": ms, "ci_vs_shipped": [los, his],
                             "gate": float(np.mean(per_k["gate"])), "union": float(np.mean(per_k["union"])),
                             "union_ttbn": float(np.mean(per_k["union_ttbn"])), "claim": [cl, cllo, clhi],
                             "claim_union": [cu, culo, cuhi], "per_system": [float(x) for x in per_k["d"]]}
                if m <= -0.0014 and hi < 0:
                    worse.append(cell)
                if m >= 0.0060 and lo > 0:
                    better.append(cell)
                rows.append(f"| {cell} | {out[cell]['gate']:.3f} | {out[cell]['union']:.4f} | {m:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                            f"{ms:+.4f} [{los:+.4f}, {his:+.4f}] | {cu:+.4f} | {cl:+.4f} [{cllo:+.4f}, {clhi:+.4f}] |")
    dh, dg = not worse, bool(better)
    verdict = "PASSES" if (dh and dg) else "FAILS"
    L = ["# N4: IR-vouched test-time BN adaptation, development verdict", "",
         "`docs/prereg-ttbn-2026-10-11.md`. Draws VIS 942–944 / IR 952–954, five systems. Δ = `union_ttbn` − "
         "`union` (ship AP), mean over the three draws per system, then over the five systems; 95% t-interval "
         "(df 4). Claim = fused − max(VIS, IR) of the arm's own streams.", "",
         f"**N4 {verdict} development.** D-H (no cell Δ ≤ −0.0014 with interval below zero): "
         f"{'PASS' if dh else 'FAIL (' + ', '.join(worse) + ')'}. D-G (a cell Δ ≥ +0.0060 with interval above "
         f"zero): {'PASS (' + str(len(better)) + ' cells: ' + ', '.join(better) + ')' if dg else 'FAIL'}.", "",
         "| cell (VIS/IR/slice) | gate | union | Δ vs union [95%] | Δ vs shipped [95%] | claim union | claim union_ttbn [95%] |",
         "|---|---:|---:|---:|---:|---:|---:|"] + rows
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"verdict": verdict, "worse": worse, "better": better,
                                                   "cells": out}, indent=1), encoding="utf-8")
    print("\n".join(L[:6]))
    return 0


def cmd_check_fp16(args) -> int:
    """fp16 against fp32 on development: VIS fog, fog_s1, lowlight, draw 942, IR clean, day, five systems."""
    conds = ("fog", "fog_s1", "lowlight")
    specs = [(942, None, k, conds, fp) for k in SEEDS for fp in (False, True)]
    res = _run_dev(specs, DEV_OUT / "fp16_check_scores.json", args.workers, lambda s: f"{s[2]}|{'fp16' if s[4] else 'fp32'}")
    diffs = {}
    for c in conds:
        a = np.mean([res[f"{k}|fp32"][f"{c}|day"]["union_ttbn"] for k in SEEDS])
        b = np.mean([res[f"{k}|fp16"][f"{c}|day"]["union_ttbn"] for k in SEEDS])
        diffs[c] = {"fp32": float(a), "fp16": float(b), "diff": float(b - a)}
    ok = all(abs(v["diff"]) <= FP16_TOL for v in diffs.values())
    FP16_CHECK.write_text(json.dumps({"pass": bool(ok), "tolerance": FP16_TOL, "cells": diffs}, indent=1),
                          encoding="utf-8")
    print(json.dumps(diffs, indent=1), "\nPASS" if ok else "\nFAIL: pohang04 is built in fp32", flush=True)
    return 0


# ------------------------------------------------------------------ pohang04
#: presorts each gated pohang04 job writes (`ttbn_lex/fused`: Amendment 2)
P04_PRESORTS = ("union_ttbn/fused", "ttbn/vis", "ttbn_lex/fused")


def _p04_cells():
    import p04_heldout_score as psc
    return [psc.VERDICT_CELL] + psc.CORRUPTED


def _p04_job(job):
    import dataclasses
    import holdout_p04_look as look
    import holdout_p04_v2_rescore as v2r
    import p04_heldout_score as psc
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import run_systems
    k, s, ic, vc, fp16 = job
    _low()
    se._memo_distances()
    tag, key = psc.cell_tag((vc, ic)), f"seed{k}_{s}"
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    sub_ = look.Pohang04() if s == "clean" else v2r.Pohang04V2()
    ctx = look.system_context(sub_, k, s, sorted({vc, "clean"}), ic, dev)
    g = gate_ctx(ctx, vc)
    if not np.array_equal(g, gate_p04(vc, ic, s)):      # the shortcut in cmd_score_p04 relies on this
        raise SystemExit(f"context gate differs from the frame-statistics gate ({k}, {s}, {ic}, {vc})")
    info = {"gate": float(g.mean()), "n_gated": int(g.sum())}
    if g.any():
        draw = None if s == "clean" else int(s[4:7])
        vis = ctx.vis_by_cond[vc]
        vis_g = gated_stream("p04", k, vc, draw, vis, g, fp16)
        out = run_systems(dataclasses.replace(ctx, veto=None), vc, compute_metrics=False, vis_records=vis_g)
        from gate_zoo import build_lex_records
        lex_g = build_lex_records(vis_g, out["ir_in_vis"], np.zeros(len(vis_g), bool))     # Amendment 2
        for name, recs in (("union_ttbn/fused", out["fused_gated"]), ("ttbn/vis", vis_g), ("ttbn_lex/fused", lex_g)):
            d = psc.PRE / name / tag / key
            if not (d / "obs.json").is_file():
                info[name] = psc.save_pre(frame_parts(recs, ctx.gts), d)
    return job[:4], info


def cmd_score_p04(args) -> int:
    import p04_heldout_score as psc
    fp16 = bool(json.loads(FP16_CHECK.read_text(encoding="utf-8")).get("pass")) if FP16_CHECK.is_file() else False
    gp = P04_OUT / "gates.json"
    gates = json.loads(gp.read_text(encoding="utf-8")) if gp.is_file() else {}
    jobs, waiting = [], 0
    for vc, ic in _p04_cells():
        for s in psc.subs_of((vc, ic)):
            # the gate is a frame statistic (seed-free): where it fires on no frame, union_ttbn IS union and
            # no context is loaded (each scored job asserts this gate equals its context's)
            g0 = gate_p04(vc, ic, s)
            for k in SEEDS:
                key = f"{k}|{s}|{ic}|{vc}"
                tag, pk = psc.cell_tag((vc, ic)), f"seed{k}_{s}"
                if key in gates and (not g0.any() or all((psc.PRE / n / tag / pk / "obs.json").is_file()
                                                         for n in P04_PRESORTS)):
                    continue
                if not g0.any():
                    gates[key] = {"gate": 0.0, "n_gated": 0, "from": "frame statistics; no context loaded"}
                elif out_path("p04", k, vc, None if s == "clean" or vc == "clean" else int(s[4:7]), fp16).is_file():
                    jobs.append((k, s, ic, vc, fp16))
                else:
                    waiting += 1                                 # adapted records not built yet
    print(f"[p04] {len(jobs)} jobs ({'fp16' if fp16 else 'fp32'} adapted records), {waiting} waiting for the build",
          flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (j, info) in enumerate(ex.map(_p04_job, jobs), 1):
            jk = f"{j[0]}|{j[1]}|{j[2]}|{j[3]}"
            gates[jk] = {**gates.get(jk, {}), **info}
            tmp = gp.with_suffix(".tmp")
            tmp.write_text(json.dumps(gates, indent=1), encoding="utf-8")
            os.replace(tmp, gp)
            print(f"[p04] {n}/{len(jobs)} {j} gate {info['gate']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_report_p04(args) -> int:
    import p04_heldout_score as psc
    gates = json.loads((P04_OUT / "gates.json").read_text(encoding="utf-8"))
    L = ["# N4 on the held-out run (pohang04)", "",
         "`docs/prereg-ttbn-2026-10-11.md`. Ship AP against the VIS ground truth, mean over the five systems "
         "and the four draws of a corrupted cell. Δ = `union_ttbn` − `union` (and − `shipped`), paired; 95% "
         "interval from the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1).", "",
         "| cell (VIS/IR) | gate | union | Δ vs union [95%] | Δ vs shipped [95%] | claim union_ttbn [95%] |",
         "|---|---:|---:|---:|---:|---:|"]
    res, worse, better, complete = {}, [], [], True
    for c in _p04_cells():
        tag = psc.cell_tag(c)
        gs = [gates.get(f"{k}|{s}|{c[1]}|{c[0]}") for s in psc.subs_of(c) for k in SEEDS]
        if any(x is None for x in gs):
            L.append(f"| {tag.replace('__', '/')} | pending | | | | |")
            complete = False
            continue
        gate = float(np.mean([x["gate"] for x in gs]))
        un, sh = psc.series("union/fused", c), psc.series("shipped/fused", c)
        if gate == 0.0:
            res[tag] = {"gate": 0.0, "delta": 0.0, "note": "the gate fires on no frame: union_ttbn is union"}
            L.append(f"| {tag.replace('__', '/')} | 0 | {'' if un is None else f'{un[0]:.4f}'} | 0 (identical) | | |")
            continue
        tt, tv, ir = psc.series("union_ttbn/fused", c), psc.series("ttbn/vis", c), psc.series("single/ir", c)
        if None in (tt, un, sh, tv, ir):
            L.append(f"| {tag.replace('__', '/')} | {gate:.3f} | pending | | | |")
            complete = False
            continue
        d, (lo, hi) = tt[0] - un[0], psc.ci(tt[1] - un[1])
        ds, (los, his) = tt[0] - sh[0], psc.ci(tt[1] - sh[1])
        cl, (cllo, clhi) = tt[0] - max(tv[0], ir[0]), psc.ci(tt[1] - np.maximum(tv[1], ir[1]))
        res[tag] = {"gate": gate, "union": un[0], "union_ttbn": tt[0], "delta": d, "ci": [lo, hi],
                    "delta_vs_shipped": ds, "ci_vs_shipped": [los, his], "claim": [cl, cllo, clhi]}
        if d <= -0.0014 and hi < 0:
            worse.append(tag)
        if d >= 0.0060 and lo > 0:
            better.append(tag)
        L.append(f"| {tag.replace('__', '/')} | {gate:.3f} | {un[0]:.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                 f"{ds:+.4f} [{los:+.4f}, {his:+.4f}] | {cl:+.4f} [{cllo:+.4f}, {clhi:+.4f}] |")
    L += ["", "## `ttbn_lex` − `lex` (Amendment 2, descriptive)", "",
          "E9 `lex` on the gated VIS stream against `lex` on the plain VIS stream (the gate-free scoring's arm); "
          "paired, same replicates. No verdict is registered.", "",
          "| cell (VIS/IR) | lex | ttbn_lex | Δ vs lex [95%] | Δ vs shipped [95%] | claim ttbn_lex [95%] |",
          "|---|---:|---:|---:|---:|---:|"]
    lw, lb = [], []
    for c in _p04_cells():
        tag = psc.cell_tag(c)
        lx, sh = psc.series("lex/fused", c), psc.series("shipped/fused", c)
        if tag in res and res[tag]["gate"] == 0.0:
            L.append(f"| {tag.replace('__', '/')} | {'' if lx is None else f'{lx[0]:.4f}'} | = lex | 0 (identical) | | |")
            continue
        tl, tv, ir = psc.series("ttbn_lex/fused", c), psc.series("ttbn/vis", c), psc.series("single/ir", c)
        if tag not in res or None in (tl, lx, sh, tv, ir):
            L.append(f"| {tag.replace('__', '/')} | pending | | | | |")
            continue
        d, (lo, hi) = tl[0] - lx[0], psc.ci(tl[1] - lx[1])
        ds, (los, his) = tl[0] - sh[0], psc.ci(tl[1] - sh[1])
        cl, (cllo, clhi) = tl[0] - max(tv[0], ir[0]), psc.ci(tl[1] - np.maximum(tv[1], ir[1]))
        res[tag]["ttbn_lex"] = {"lex": lx[0], "ttbn_lex": tl[0], "delta": d, "ci": [lo, hi], "delta_vs_shipped": ds,
                                "ci_vs_shipped": [los, his], "claim": [cl, cllo, clhi]}
        if d <= -0.0014 and hi < 0:
            lw.append(tag)
        if d >= 0.0060 and lo > 0:
            lb.append(tag)
        L.append(f"| {tag.replace('__', '/')} | {lx[0]:.4f} | {tl[0]:.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                 f"{ds:+.4f} [{los:+.4f}, {his:+.4f}] | {cl:+.4f} [{cllo:+.4f}, {clhi:+.4f}] |")
    L += ["", f"Counts with P-T's thresholds (description): {len(lw)} worse ({', '.join(lw) or 'none'}), "
              f"{len(lb)} better ({', '.join(lb) or 'none'})."]
    v = ("PENDING" if not complete else
         "REPLICATES" if (not worse and better) else
         f"DOES NOT REPLICATE (worse: {', '.join(worse) or 'none'}; better: {', '.join(better) or 'none'})")
    L[3:3] = ["", f"**P-T: {v}.**"]
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"P-T": v, "cells": res}, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--target", choices=["dev", "p04"], required=True)
    b.add_argument("--draws", type=int, nargs="+", default=None)
    b.add_argument("--conds", nargs="+", required=True)
    b.add_argument("--workers", type=int, default=6)
    b.add_argument("--fp16", action="store_true")
    for name in ("score-dev", "check-fp16", "score-p04", "score-dev-lex"):
        s = sp.add_parser(name)
        s.add_argument("--workers", type=int, default=4)
    r = sp.add_parser("report-dev")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_dev_2026-10-11.md")
    r = sp.add_parser("report-p04")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_p04_2026-10-11.md")
    r = sp.add_parser("report-dev-lex")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_lex_dev_2026-10-11.md")
    args = ap.parse_args()
    return {"build": cmd_build, "score-dev": cmd_score_dev, "report-dev": cmd_report_dev,
            "check-fp16": cmd_check_fp16, "score-p04": cmd_score_p04, "report-p04": cmd_report_p04,
            "score-dev-lex": cmd_score_dev_lex, "report-dev-lex": cmd_report_dev_lex}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
