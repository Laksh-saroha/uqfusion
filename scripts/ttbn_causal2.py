"""N4c2: N4, one block late (`docs/prereg-ttbn-variants-2026-10-11.md`, Amendment 1).

N4c (registered) normalised block b with the statistics of each layer's input measured during block b−1's
own causal pass. Those statistics were measured under b−1's stale normalisation, not N4's. The first block
that uses them collapses, and the recursion does not recover within a run (`docs/eval/ttbn/` N4c report and
the trace in the Amendment). N4c2 normalises block b with the statistics **N4 measured on block b−1**: block
b−1 is also run through N4's closed-loop BN-mix, whose per-layer batch statistics are stored, and block b is
normalised with BN-mix(source, those statistics). A run's first block: source statistics. No look-ahead.
Two passes per block; fp16 autocast with fused statistics, as N4's pohang04 build.

    py -3.13 scripts/ttbn_causal2.py smoke --draw 941 --blocks 6     # CPU mechanics check (screen draw)
    py -3.13 scripts/ttbn_causal2.py build-dev                       # GPU, draws 942-944
    py -3.13 scripts/ttbn_causal2.py score-dev
    py -3.13 scripts/ttbn_causal2.py report-dev
    py -3.13 scripts/ttbn_causal2.py check-sparse                    # GPU: sparse passes == full grid (dev)
    py -3.13 scripts/ttbn_causal2.py build-p04                       # P-C2 (Amendment 2)
    py -3.13 scripts/ttbn_causal2.py score-p04
    py -3.13 scripts/ttbn_causal2.py report-p04
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
import ttbn                                                    # noqa: E402
import ttbn_variants as tv                                     # noqa: E402

OUT = ROOT / "runs" / "novel" / "ttbn_variants" / "causal2"
STATE = {"mode": None}          # None: source; "apply": BN-mix(source, previous block's N4 statistics); "record": N4


def out_path(k: int, cond: str, draw: int) -> Path:
    return OUT / f"seed{k}" / ttbn.sub(draw) / f"gauss_vis_paired_{cond}_bn16_fp16.pkl"


def patch_bn_delayed(model) -> int:
    """'record': N4's closed-loop BN-mix on this block, storing its batch statistics in `m._new`.
    'apply': BN-mix of the source statistics and `m._rec` (the previous block's N4 statistics; source if None)."""
    import torch
    import torch.nn.functional as F
    n_bn = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            n_bn += 1
            m._rec, m._new = None, None

            def fwd(x, m=m):
                mode = STATE["mode"]
                if mode is None or (mode == "apply" and m._rec is None):
                    return F.batch_norm(x, m.running_mean, m.running_var, m.weight, m.bias, False, 0.0, m.eps)
                if mode == "record":
                    if x.dtype == torch.float16:
                        mu_t, invstd = torch.batch_norm_stats(x, 0.0)          # as ttbn.patch_bn's fused path
                        var_t = invstd.pow(-2)
                    else:
                        xf = x.float()
                        mu_t, var_t = xf.mean(dim=(0, 2, 3)), xf.var(dim=(0, 2, 3), unbiased=False)
                    n = x.shape[0]
                    m._new = (mu_t, var_t, n)
                else:
                    mu_t, var_t, n = m._rec
                a = n / (ttbn.PRIOR_N + n)
                mu = (1 - a) * m.running_mean + a * mu_t
                var = (1 - a) * m.running_var + a * var_t
                return F.batch_norm(x, mu, var, m.weight, m.bias, False, 0.0, m.eps)
            m.forward = fwd
    return n_bn


def _bns(preds):
    import torch
    return [m for p in preds for m in p.model.modules() if isinstance(m, torch.nn.BatchNorm2d)]


def _reset(preds) -> None:
    for m in _bns(preds):
        m._rec, m._new = None, None


def _commit(preds) -> None:
    for m in _bns(preds):
        m._rec, m._new = m._new, None


def _block(preds, ims, ph):
    """Block b: output with the previous block's N4 statistics, then N4's pass to record this block's."""
    import torch
    with torch.no_grad(), ph.amp(next(preds[0].model.parameters()).is_cuda):
        STATE["mode"] = "apply"
        out = ph._multi_batch(preds, ims)
        STATE["mode"] = "record"
        ph._multi_batch(preds, ims)
        STATE["mode"] = None
    _commit(preds)
    return out


def _chunks(need, lst):
    """N4c's per-run grid: every block of 8 listed frames of a run with a needed frame, in order."""
    runs = [Path(p).parent.name for p in lst]
    chunks, first = [], set()
    for r in dict.fromkeys(runs):
        ir_ = [i for i, x in enumerate(runs) if x == r]
        if not need[ir_].any():
            continue
        first.add(len(chunks))
        chunks += [ir_[a:a + ttbn.BATCH] for a in range(0, len(ir_), ttbn.BATCH)]
    return chunks, first


def _predictors(device):
    import p04_heldout as ph
    from uqfusion.uq.infer import UQPredictor
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=device, imgsz=640,
                         conf=0.001) for k in ttbn.SEEDS]
    if device != "cpu":
        ph.setup_fp16(preds)
    return preds


def cmd_build_dev(args) -> int:
    import p04_heldout as ph
    from uqfusion.eval.corruptions_v2 import CODE_REV
    ttbn._low()
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    todo = [(c, d, tv._needed(d, c, tv._gate_n4_stats)) for c in tv.CAUSAL_CONDS for d in ttbn.DEV_DRAWS
            if not all(out_path(k, c, d).is_file() for k in ttbn.SEEDS)]
    todo = [t for t in todo if t[2].any()]
    preds = _predictors(0)
    print(f"[causal2] BN layers {[patch_bn_delayed(p.model) for p in preds]}; {len(todo)} streams", flush=True)
    for c, d, need in todo:
        kc = ttbn.DEV_CONDS[c]
        chunks, first = _chunks(need, lst)
        recs = [[None] * len(lst) for _ in ttbn.SEEDS]
        t0 = time.time()
        spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": d, "images": lst}
        for n_done, ch in enumerate(ttbn._frames(spec, chunks, args.workers)):
            if n_done in first:
                _reset(preds)
            out = _block(preds, [im for _, im in ch], ph)
            for j in range(len(ttbn.SEEDS)):
                for (i, _), r in zip(ch, out[j]):
                    recs[j][i] = {**ph._f32(r), "image_path": lst[i]}
            if (n_done + 1) % 100 == 0:
                print(f"[causal2] {c}/{d} {n_done + 1}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
        for j, k in enumerate(ttbn.SEEDS):
            o = out_path(k, c, d)
            o.parent.mkdir(parents=True, exist_ok=True)
            meta = {"variant": "N4c2", "cond": c, "draw": d, "prior_N": ttbn.PRIOR_N, "batch": ttbn.BATCH,
                    "blocks_run": len(chunks), "needed_frames": int(need.sum()),
                    "blocks": "N4c's per-run grid; block b normalised with N4's statistics of block b-1 "
                              "(closed-loop record pass); a run's first block at source",
                    "precision": "fp16 autocast, channels_last, fused statistics",
                    "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                    "corrupt_seed": d if kc else None, "corrupt_version": "v2" if kc else None,
                    "corrupt_code": CODE_REV if kc else None,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": meta, "records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[causal2] {c}/{d} done: {len(chunks)} blocks, {int(need.sum())} needed frames, "
              f"{time.time() - t0:.0f}s", flush=True)
    return 0


def cmd_smoke(args) -> int:
    """Mechanics only, on the screen draw: N4c2's per-block mean max confidence beside N4's (closed loop)."""
    import torch
    import p04_heldout as ph
    torch.set_num_threads(3)
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    runs = [Path(p).parent.name for p in lst]
    ir_ = [i for i, x in enumerate(runs) if x == "pohang00"]
    chunks = [ir_[a:a + ttbn.BATCH] for a in range(0, len(ir_), ttbn.BATCH)][:args.blocks]
    from uqfusion.uq.infer import UQPredictor
    w = ROOT / "runs/phase3_stage2/p3_vis_seed0/weights/best.pt"
    pa = UQPredictor(w, device="cpu", imgsz=640, conf=0.001)
    ttbn.patch_bn(pa.model)
    pb = UQPredictor(w, device="cpu", imgsz=640, conf=0.001)
    patch_bn_delayed(pb.model)
    kc = ttbn.DEV_CONDS[args.cond]
    spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": args.draw, "images": lst}
    mc = lambda rs: np.mean([float(np.max(r["conf"])) if len(r["conf"]) else 0.0 for r in rs])   # noqa: E731
    for b, ch in enumerate(ttbn._frames(spec, chunks, 2)):
        ims = [im for _, im in ch]
        with torch.no_grad():
            ttbn.MODE["n"] = ttbn.PRIOR_N
            ra = ph._multi_batch([pa], ims)[0]
            ttbn.MODE["n"] = None
        rb = _block([pb], ims, ph)[0]
        print(f"[smoke] {args.cond}/{args.draw} block {b}: mean max conf N4 {mc(ra):.3f}  N4c2 {mc(rb):.3f}", flush=True)
    return 0


# ------------------------------------------------------------------ scoring
def _dev_job(spec):
    import dataclasses
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems
    from v2_sensitivity import ship_ap
    draw, ir_cond, k = spec
    ttbn._low()
    se._memo_distances()
    d = f"{se.STATS}/brightness/{ttbn.sub(draw)}"
    conds = tuple(ttbn.DEV_CONDS)
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed{k}/{ttbn.sub(draw)}", bright_dir=d,
                       structure_dir=f"{se.STATS}/structure/{ttbn.sub(draw)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json",
                       conditions=conds, ir_condition=ir_cond, verbose=False)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    u = dataclasses.replace(ctx, veto=None)
    res = {}
    for c in conds:
        g4 = ttbn.gate_ctx(ctx, c)
        vis = ctx.vis_by_cond[c]
        vis_c = vis if (c == "clean" or not g4.any()) else tv._stream(out_path(k, c, draw), vis, g4)
        parts = {"union": run_systems(u, c, compute_metrics=False)["fused_gated"],
                 "union_ttbnc2": run_systems(u, c, compute_metrics=False, vis_records=vis_c)["fused_gated"],
                 "vis_ttbnc2": vis_c}
        parts = {n: frame_parts(r, ctx.gts) for n, r in parts.items()}
        for s, idx in sl.items():
            res[f"{c}|{s}"] = {n: ship_ap(p, idx) for n, p in parts.items()}
            res[f"{c}|{s}"]["gate"] = float(g4[idx].mean())
    return spec, res


def cmd_score_dev(args) -> int:
    out_p = OUT / "score_dev.json"
    allres = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    specs = [(d, ic, k) for d in ttbn.DEV_DRAWS for ic in ttbn.DEV_IR_CONDS for k in ttbn.SEEDS
             if f"{d}|{ic}|{k}" not in allres]
    print(f"[score] {len(specs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (spec, res) in enumerate(ex.map(_dev_job, specs), 1):
            allres["|".join(map(str, spec))] = res
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(allres), encoding="utf-8")
            os.replace(tmp, out_p)
            print(f"[score] {n}/{len(specs)} {spec} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_report_dev(args) -> int:
    res = json.loads((OUT / "score_dev.json").read_text(encoding="utf-8"))
    n4 = json.loads((ttbn.DEV_OUT / "score.json").read_text(encoding="utf-8"))
    out, rows, worse, better = {}, [], [], []
    for c in ttbn.DEV_CONDS:
        for ic in ttbn.DEV_IR_CONDS:
            for s in ("day", "night"):
                pk = {n: [] for n in ("c", "c4", "gate")}
                for k in ttbn.SEEDS:
                    rs = [res[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in ttbn.DEV_DRAWS]
                    r4 = [n4[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in ttbn.DEV_DRAWS]
                    pk["c"].append(np.mean([r["union_ttbnc2"] - r["union"] for r in rs]))
                    pk["c4"].append(np.mean([r["union_ttbnc2"] - q["union_ttbn"] for r, q in zip(rs, r4)]))
                    pk["gate"].append(np.mean([r["gate"] for r in rs]))
                cc, clo, chi = ttbn._cell_stats(pk["c"])
                c4, c4lo, c4hi = ttbn._cell_stats(pk["c4"])
                cell = f"{c}/{ic or 'clean'}/{s}"
                out[cell] = {"n4c2": [cc, clo, chi], "n4c2_minus_n4": [c4, c4lo, c4hi], "gate": float(np.mean(pk["gate"]))}
                if cc <= -0.0014 and chi < 0:
                    worse.append(cell)
                if cc >= 0.0060 and clo > 0:
                    better.append(cell)
                rows.append(f"| {cell} | {out[cell]['gate']:.3f} | {cc:+.4f} [{clo:+.4f}, {chi:+.4f}] | "
                            f"{c4:+.4f} [{c4lo:+.4f}, {c4hi:+.4f}] |")
    v = "PASSES" if (not worse and better) else "FAILS"
    L = ["# N4c2 (N4, one block late) on development", "",
         "`docs/prereg-ttbn-variants-2026-10-11.md`, Amendment 1. Draws VIS 942–944 / IR 952–954, five systems. "
         "Δ = `union_ttbnc2` − `union` (ship AP), mean over the three draws per system, then over the five "
         "systems; 95% t-interval (df 4). `N4c2 − N4` = `union_ttbnc2` − `union_ttbn` (N4's development scores, "
         "fp32), descriptive: the price of causality.", "",
         f"**N4c2 {v} development.** C-H: {'PASS' if not worse else 'FAIL (' + ', '.join(worse) + ')'}. "
         f"C-G: {'PASS (' + str(len(better)) + ' cells: ' + ', '.join(better) + ')' if better else 'FAIL'}.", "",
         "| cell (VIS/IR/slice) | N4 gate | N4c2 Δ vs union [95%] | N4c2 − N4 [95%] |", "|---|---:|---:|---:|"] + rows
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"N4c2": v, "worse": worse, "better": better, "cells": out}, indent=1),
                                      encoding="utf-8")
    print("\n".join(L[:6]))
    return 0


# ------------------------------------------------------------------ pohang04 (P-C2, Amendment 2)
P04_OUT = ROOT / "runs" / "holdout_p04" / "ttbn_causal2"
P04_PRESORTS = ("union_ttbnc2/fused", "ttbnc2/vis")
SPARSE_CHECK = OUT / "sparse_check.json"


def out_path_p04(k: int, cond: str, draw) -> Path:
    where = "clean" if draw is None else ttbn.sub(draw)
    return P04_OUT / f"seed{k}" / where / f"gauss_vis_paired_{cond}_bn16_fp16.pkl"


def _plan(need, lst):
    """N4c's per-run grid reduced to the passes whose results are used, in grid order:
    (block frames, first block of its run, output pass, recording pass). Block b's output reads N4's
    statistics of block b-1 only, and those come from b-1's own closed-loop pass, so the records equal
    the full grid's."""
    runs = [Path(p).parent.name for p in lst]
    plan = []
    for r in dict.fromkeys(runs):
        ir_ = [i for i, x in enumerate(runs) if x == r]
        blocks = [ir_[a:a + ttbn.BATCH] for a in range(0, len(ir_), ttbn.BATCH)]
        out = [bool(need[b].any()) for b in blocks]
        for j, b in enumerate(blocks):
            rec = j + 1 < len(blocks) and out[j + 1]
            if out[j] or rec:
                plan.append((b, j == 0, out[j], rec))
    return plan


def _run_plan(preds, plan, spec, workers, ph, recs, lst, tag):
    import torch
    bns = _bns(preds)
    t0 = time.time()
    _reset(preds)
    chunks = [p[0] for p in plan]
    for n_done, (ch, (blk, first, o, r)) in enumerate(zip(ttbn._frames(spec, chunks, workers), plan), 1):
        assert [i for i, _ in ch] == blk
        if first:
            _reset(preds)
        elif o and any(m._rec is None for m in bns):
            raise SystemExit(f"{tag}: block {blk[0]} would be normalised without its predecessor's statistics")
        ims = [im for _, im in ch]
        with torch.no_grad(), ph.amp(next(preds[0].model.parameters()).is_cuda):
            if o:
                STATE["mode"] = "apply"
                out = ph._multi_batch(preds, ims)
                for j in range(len(preds)):
                    for (i, _), rr in zip(ch, out[j]):
                        recs[j][i] = {**ph._f32(rr), "image_path": lst[i]}
            if r:
                STATE["mode"] = "record"
                ph._multi_batch(preds, ims)
            STATE["mode"] = None
        if r:
            _commit(preds)
        else:
            _reset(preds)
        if n_done % 200 == 0:
            print(f"[causal2-p04] {tag} {n_done}/{len(plan)} blocks ({time.time() - t0:.0f}s)", flush=True)
    return time.time() - t0


def _cmp(x, y, idx) -> dict:
    exact = cnt = 0
    dconf = 0.0
    for j in range(len(x)):
        for i in idx:
            a, b = x[j][i], y[j][i]
            if len(a["conf"]) != len(b["conf"]):
                cnt += 1
                continue
            if all(np.array_equal(a[k], b[k]) for k in ("conf", "boxes_xyxy", "cls", "sigma_ltrb")):
                exact += 1
            if len(a["conf"]):
                dconf = max(dconf, float(np.max(np.abs(np.sort(a["conf"]) - np.sort(b["conf"])))))
    return {"frames": len(x) * len(idx), "bit_exact": exact, "box_count_differs": cnt, "max_sorted_conf_diff": dconf}


def cmd_check_sparse(args) -> int:
    """Development, GPU: the sparse passes reproduce the full grid (draw 942, low light). Every `--every`-th
    block with a needed frame is kept. The sparse plan and the full grid (every block of those runs, as
    build-dev) run in this process and must agree bit for bit on the kept frames, all five systems.
    Descriptive: both against build-dev's stored records (another process: fp16 inference with cuDNN
    autotuning is not bit-reproducible across processes)."""
    import p04_heldout as ph
    ttbn._low()
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    need_full = tv._needed(args.draw, args.cond, tv._gate_n4_stats)
    chunks, _first = _chunks(need_full, lst)
    keep = [ch for ch in chunks if need_full[ch].any()][::args.every]
    need = np.zeros(len(lst), bool)
    for ch in keep:
        need[[i for i in ch if need_full[i]]] = True
    idx = np.flatnonzero(need)
    preds = _predictors(0)
    for p in preds:
        patch_bn_delayed(p.model)
    kc = ttbn.DEV_CONDS[args.cond]
    spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": args.draw, "images": lst}
    R, sec = {}, {}
    for name, nd in (("sparse", need), ("full", need_full)):
        R[name] = [[None] * len(lst) for _ in ttbn.SEEDS]
        sec[name] = round(_run_plan(preds, _plan(nd, lst), spec, args.workers, ph, R[name], lst,
                                    f"check {name} {args.cond}/{args.draw}"))
    stored = []
    for k in ttbn.SEEDS:
        with open(out_path(k, args.cond, args.draw), "rb") as fh:
            stored.append(pickle.load(fh)["records"])
    firsts = {p[0][0] for p in _plan(need, lst) if p[1]}
    res = {"cond": args.cond, "draw": args.draw, "blocks_kept": len(keep),
           "first_blocks_kept": sum(1 for ch in keep if ch[0] in firsts), "passes_sparse": len(_plan(need, lst)),
           "sparse_vs_full_same_process": _cmp(R["sparse"], R["full"], idx),
           "descriptive_full_vs_stored": _cmp(R["full"], stored, idx),
           "descriptive_sparse_vs_stored": _cmp(R["sparse"], stored, idx), "seconds": sec}
    c = res["sparse_vs_full_same_process"]
    res["pass"] = bool(c["bit_exact"] == c["frames"])
    SPARSE_CHECK.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[check-sparse] {json.dumps(res)}", flush=True)
    return 0 if res["pass"] else 1


def cmd_build_p04(args) -> int:
    import p04_heldout as ph
    from uqfusion.eval.corruptions_v2 import CODE_REV
    chk = json.loads(SPARSE_CHECK.read_text(encoding="utf-8")) if SPARSE_CHECK.is_file() else {}
    if not chk.get("pass"):
        raise SystemExit("the pohang04 build needs a passed sparse check (check-sparse)")
    ttbn._low()
    lst = ph.images("vis")
    todo = [(c, d) for c in args.conds for d in ([None] if c == "clean" else ph.VIS_DRAWS)
            if not all(out_path_p04(k, c, d).is_file() for k in ttbn.SEEDS)]
    preds = _predictors(0)
    print(f"[causal2-p04] BN layers {[patch_bn_delayed(p.model) for p in preds]}; {len(lst)} frames, "
          f"{len(todo)} streams", flush=True)
    for c, d in todo:
        kc = ttbn.P04_CONDS[c]
        need = ttbn.needed("p04", c, d)
        assert len(need) == len(lst)
        plan = _plan(need, lst)
        recs = [[None] * len(lst) for _ in ttbn.SEEDS]
        spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": d or 0, "images": lst}
        sec = _run_plan(preds, plan, spec, args.workers, ph, recs, lst, f"{c}/{d}") if plan else 0.0
        for j, k in enumerate(ttbn.SEEDS):
            o = out_path_p04(k, c, d)
            o.parent.mkdir(parents=True, exist_ok=True)
            meta = {"variant": "N4c2", "target": "p04", "cond": c, "draw": d, "prior_N": ttbn.PRIOR_N,
                    "batch": ttbn.BATCH, "needed_frames": int(need.sum()), "passes": len(plan),
                    "output_blocks": sum(p[2] for p in plan), "recording_passes": sum(p[3] for p in plan),
                    "blocks": "N4c's per-run grid; block b normalised with N4's statistics of block b-1 "
                              "(closed-loop record pass); a run's first block at source; only used passes run",
                    "precision": "fp16 autocast, channels_last, fused statistics",
                    "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                    "corrupt_seed": d if kc else None, "corrupt_version": "v2" if kc else None,
                    "corrupt_code": CODE_REV if kc else None,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": meta, "records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[causal2-p04] {c}/{d} done: {sum(p[2] for p in plan)} output blocks, {len(plan)} passes, "
              f"{int(need.sum())} needed frames, {sec:.0f}s", flush=True)
    return 0


def _p04_job(job):
    import dataclasses
    import holdout_p04_look as look
    import holdout_p04_v2_rescore as v2r
    import p04_heldout_score as psc
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import run_systems
    k, s, ic, vc = job
    ttbn._low()
    se._memo_distances()
    tag, key = psc.cell_tag((vc, ic)), f"seed{k}_{s}"
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    sub_ = look.Pohang04() if s == "clean" else v2r.Pohang04V2()
    ctx = look.system_context(sub_, k, s, sorted({vc, "clean"}), ic, dev)
    g = ttbn.gate_ctx(ctx, vc)
    if not np.array_equal(g, ttbn.gate_p04(vc, ic, s)):
        raise SystemExit(f"context gate differs from the frame-statistics gate ({k}, {s}, {ic}, {vc})")
    vd = None if (s == "clean" or vc == "clean") else int(s[4:7])
    vis = ctx.vis_by_cond[vc]
    vis_c = tv._stream(out_path_p04(k, vc, vd), vis, g)
    out = run_systems(dataclasses.replace(ctx, veto=None), vc, compute_metrics=False, vis_records=vis_c)
    info = {"gate": float(g.mean())}
    for name, recs in (("union_ttbnc2/fused", out["fused_gated"]), ("ttbnc2/vis", vis_c)):
        d = psc.PRE / name / tag / key
        if not (d / "obs.json").is_file():
            info[name] = psc.save_pre(frame_parts(recs, ctx.gts), d)
    return job, info


def cmd_score_p04(args) -> int:
    import p04_heldout_score as psc
    jobs, waiting = [], 0
    for vc, ic in [psc.VERDICT_CELL] + psc.CORRUPTED:
        for s in psc.subs_of((vc, ic)):
            if not ttbn.gate_p04(vc, ic, s).any():
                continue
            vd = None if (s == "clean" or vc == "clean") else int(s[4:7])
            for k in ttbn.SEEDS:
                tag, pk = psc.cell_tag((vc, ic)), f"seed{k}_{s}"
                if all((psc.PRE / n / tag / pk / "obs.json").is_file() for n in P04_PRESORTS):
                    continue
                if out_path_p04(k, vc, vd).is_file():
                    jobs.append((k, s, ic, vc))
                else:
                    waiting += 1
    print(f"[causal2-p04] {len(jobs)} jobs, {waiting} waiting for the build", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (j, info) in enumerate(ex.map(_p04_job, jobs), 1):
            print(f"[causal2-p04] {n}/{len(jobs)} {j} {info} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_report_p04(args) -> int:
    import p04_heldout_score as psc
    L = ["# N4c2 on pohang04 (P-C2)", "",
         "`docs/prereg-ttbn-variants-2026-10-11.md`, Amendment 2. P-T's cells, statistic and replicates: ship AP "
         "against the VIS ground truth, mean over the five systems and the four draws of a corrupted cell; Δ "
         "paired; 95% interval from the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1). "
         "`− N4` is against `union_ttbn` (N4's pohang04 records), the price of causality.", "",
         "| cell (VIS/IR) | gate | union | N4c2 Δ vs union [95%] | N4c2 − N4 [95%] |", "|---|---:|---:|---:|---:|"]
    res, worse, better, complete = {}, [], [], True
    for c in [psc.VERDICT_CELL] + psc.CORRUPTED:
        tag = psc.cell_tag(c)
        gate = float(np.mean([ttbn.gate_p04(c[0], c[1], s).mean() for s in psc.subs_of(c)]))
        un = psc.series("union/fused", c)
        if gate == 0.0:
            res[tag] = {"gate": 0.0, "delta": 0.0, "note": "the gate fires on no frame: N4c2 is union"}
            L.append(f"| {tag.replace('__', '/')} | 0 | {'' if un is None else f'{un[0]:.4f}'} | 0 (identical) | |")
            continue
        a, t4 = psc.series("union_ttbnc2/fused", c), psc.series("union_ttbn/fused", c)
        if None in (a, un, t4):
            L.append(f"| {tag.replace('__', '/')} | {gate:.3f} | pending | | |")
            complete = False
            continue
        d, (lo, hi) = a[0] - un[0], psc.ci(a[1] - un[1])
        d4, (lo4, hi4) = a[0] - t4[0], psc.ci(a[1] - t4[1])
        res[tag] = {"gate": gate, "union": un[0], "union_ttbnc2": a[0], "delta": d, "ci": [lo, hi],
                    "minus_n4": d4, "ci_minus_n4": [lo4, hi4]}
        if d <= -0.0014 and hi < 0:
            worse.append(tag)
        if d >= 0.0060 and lo > 0:
            better.append(tag)
        L.append(f"| {tag.replace('__', '/')} | {gate:.3f} | {un[0]:.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                 f"{d4:+.4f} [{lo4:+.4f}, {hi4:+.4f}] |")
    v = ("PENDING" if not complete else "REPLICATES" if (not worse and better) else
         f"DOES NOT REPLICATE (worse: {', '.join(worse) or 'none'}; better: {', '.join(better) or 'none'})")
    L[3:3] = ["", f"**P-C2: {v}.**"]
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"P-C2": v, "cells": res}, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    s = sp.add_parser("smoke")
    s.add_argument("--draw", type=int, default=941)
    s.add_argument("--cond", default="lowlight")
    s.add_argument("--blocks", type=int, default=6)
    b = sp.add_parser("build-dev")
    b.add_argument("--workers", type=int, default=4)
    s = sp.add_parser("score-dev")
    s.add_argument("--workers", type=int, default=3)
    r = sp.add_parser("report-dev")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_causal2_dev_2026-10-11.md")
    c = sp.add_parser("check-sparse")
    c.add_argument("--draw", type=int, default=942)
    c.add_argument("--cond", default="lowlight")
    c.add_argument("--every", type=int, default=7)
    c.add_argument("--workers", type=int, default=3)
    b = sp.add_parser("build-p04")
    b.add_argument("--conds", nargs="+", default=["lowlight", "noise_s2", "fog", "rain_s2", "blur_s3", "clean"])
    b.add_argument("--workers", type=int, default=4)
    s = sp.add_parser("score-p04")
    s.add_argument("--workers", type=int, default=2)
    r = sp.add_parser("report-p04")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_causal2_p04_2026-10-11.md")
    args = ap.parse_args()
    return {"smoke": cmd_smoke, "build-dev": cmd_build_dev, "score-dev": cmd_score_dev,
            "report-dev": cmd_report_dev, "check-sparse": cmd_check_sparse, "build-p04": cmd_build_p04,
            "score-p04": cmd_score_p04, "report-p04": cmd_report_p04}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
