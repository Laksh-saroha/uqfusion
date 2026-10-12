"""N4 on top of `full_tta` (N4F, `docs/prereg-ttbn-fulltta-2026-10-11.md`).

`full_tta_ttbn` = `full_tta` with every VIS view BN-mix-adapted on the frames N4's gate passes: each
checkpoint's unflipped view by N4's adapted record, its flipped view by the same BN-mix on the flipped
block (mapped back). Adapted views in fp16 with fused statistics (N4's pohang04 precision).

    py -3.13 scripts/ttbn_fulltta.py build-dev        # draws 945-947, both adapted views
    py -3.13 scripts/ttbn_fulltta.py score-dev        # presorts per arm, cell, draw and slice
    py -3.13 scripts/ttbn_fulltta.py boot-dev         # moving-block replicates (CPU)
    py -3.13 scripts/ttbn_fulltta.py report-dev
"""

from __future__ import annotations

import argparse
import dataclasses
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

import ttbn  # noqa: E402

OUT = ROOT / "runs/novel/ttbn_fulltta"
PRE = OUT / "pre"
DEV_DRAWS = (945, 946, 947)
BUILD_CONDS = ("fog", "fog_s1", "lowlight", "glare")       # clean: N4's gate never fires on clean VIS
ARMS = ("full_tta", "full_tta_ttbn")


def out_path(view: str, k: int, cond: str, draw: int) -> Path:
    return OUT / view / f"seed{k}" / ttbn.sub(draw) / f"gauss_vis_paired_{cond}_bn16_fp16.pkl"


def _gate_stats(draw, cond, ic):
    return ttbn.vis_unhealthy(ttbn._dev_stats(draw, "vis", cond, "structure")) & ttbn.ir_day_ok(
        ttbn._dev_stats(draw, "ir", ic, "brightness"), ttbn._dev_stats(draw, "ir", ic, "structure"))


def _needed(draw, cond):
    out = None
    for ic in ttbn.DEV_IR_CONDS:
        g = _gate_stats(draw, cond, ic)
        out = g if out is None else out | g
    return out


# ------------------------------------------------------------------ build
def cmd_build_dev(args) -> int:
    import torch
    import p04_heldout as ph
    from uqfusion.eval.corruptions_v2 import CODE_REV
    from uqfusion.uq.infer import UQPredictor
    chk = json.loads(ttbn.FP16_CHECK.read_text(encoding="utf-8")) if ttbn.FP16_CHECK.is_file() else {}
    if not chk.get("pass"):
        raise SystemExit("fp16 adapted views need N4's passed development check")
    ttbn._low()
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    todo = [(c, d) for c in BUILD_CONDS for d in DEV_DRAWS
            if not all(out_path(v, k, c, d).is_file() for v in ("raw", "flip") for k in ttbn.SEEDS)]
    print(f"[n4f] dev: {len(lst)} frames, {len(todo)} streams to build (fp16, fused statistics)", flush=True)
    if not todo:
        return 0
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=0, imgsz=640,
                         conf=0.001) for k in ttbn.SEEDS]
    nb = [ttbn.patch_bn(p.model) for p in preds]
    ph.setup_fp16(preds)
    ttbn.MODE["fused"] = True
    print(f"[n4f] BN layers {nb}", flush=True)
    for c, d in todo:
        kc = ttbn.DEV_CONDS[c]
        need = _needed(d, c)
        idx = list(range(len(lst)))
        chunks = [idx[a:a + ttbn.BATCH] for a in range(0, len(idx), ttbn.BATCH) if need[a:a + ttbn.BATCH].any()]
        recs = {v: [[None] * len(lst) for _ in ttbn.SEEDS] for v in ("raw", "flip")}
        t0 = time.time()
        if chunks:
            spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": d, "images": lst}
            ttbn.MODE["n"] = ttbn.PRIOR_N
            for n_done, ch in enumerate(ttbn._frames(spec, chunks, args.workers), 1):
                ims = [im for _, im in ch]
                with torch.no_grad(), ph.amp(True):
                    out_r = ph._multi_batch(preds, ims)
                    out_f = ph._multi_batch(preds, [np.ascontiguousarray(im[:, ::-1]) for im in ims])
                for j in range(len(ttbn.SEEDS)):
                    for (i, im), r, f in zip(ch, out_r[j], out_f[j]):
                        recs["raw"][j][i] = {**ph._f32(r), "image_path": lst[i]}
                        recs["flip"][j][i] = {**ph._unflip(f, im.shape[1]), "image_path": lst[i]}
                if n_done % 100 == 0:
                    print(f"[n4f] {c}/{d} {n_done}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
            ttbn.MODE["n"] = None
        for v in ("raw", "flip"):
            for j, k in enumerate(ttbn.SEEDS):
                o = out_path(v, k, c, d)
                o.parent.mkdir(parents=True, exist_ok=True)
                meta = {"view": v, "cond": c, "draw": d, "prior_N": ttbn.PRIOR_N, "batch": ttbn.BATCH,
                        "blocks": "N4's grid; blocks with a gated frame (any IR condition)", "blocks_run": len(chunks),
                        "gated_frames": int(need.sum()), "precision": "fp16 autocast, channels_last, fused statistics",
                        "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None, "corrupt_seed": d,
                        "corrupt_version": "v2", "corrupt_code": CODE_REV,
                        "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
                with open(o.with_suffix(".tmp"), "wb") as fh:
                    pickle.dump({"meta": meta, "records": recs[v][j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(o.with_suffix(".tmp"), o)
        print(f"[n4f] {c}/{d} done: {len(chunks)} blocks, {int(need.sum())} gated frames, {time.time() - t0:.0f}s",
              flush=True)
    return 0


# ------------------------------------------------------------------ score
def _adapt(views: list[list[dict]], adapted: list[list[dict]], gate: np.ndarray, names=None) -> list[list[dict]]:
    """Each view list with its adapted record on gated frames (frame identity and `feat` kept). View records
    without `image_path` (pohang04's flipped `full_tta` views, index-aligned to the frame list) are identified
    by `names`, the raw view's paths."""
    out = []
    for pl, ad in zip(views, adapted):
        v = []
        for i, (a, f) in enumerate(zip(pl, gate)):
            if not f:
                v.append(a)
                continue
            b = ad[i]
            if b is None:
                raise SystemExit("gated frame without an adapted record")
            na = a.get("image_path") or (names[i] if names is not None else None)
            assert na is not None and Path(b["image_path"]).name == Path(na).name
            v.append({**b, "image_path": na, "feat": a.get("feat"), "image_hw": a["image_hw"]})
        out.append(v)
    return out


def _load_recs(p: Path):
    with open(p, "rb") as fh:
        return pickle.load(fh)["records"]


def _dev_job(spec):
    import holdout_p04_v2_rescore as v2r
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, capability_prior, load_context, run_systems
    from v2_sensitivity import ship_ap
    draw, ir_cond = spec
    ttbn._low()
    se._CACHE.clear()
    se._memo_distances()
    d = f"{se.STATS}/brightness/{se.sub(draw)}"
    kw = dict(bright_dir=d, structure_dir=f"{se.STATS}/structure/{se.sub(draw)}", verbose=False)
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed0/{se.sub(draw)}",
                       ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json", conditions=se.VIS_CONDS,
                       ir_condition=ir_cond, **kw)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    iflag = ctx.ir_d2 > ctx.ir_bound
    if ir_cond is None:
        iflag0 = iflag
    else:
        c0 = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed0/{se.sub(draw)}",
                          ir_bright=f"{d}/gauss_ir_paired_clean.json", conditions=("clean",), ir_condition=None, **kw)
        iflag0 = c0.ir_d2 > c0.ir_bound
    mem = tuple(ttbn.SEEDS)
    cv, ci = capability_prior(se._stream(draw, "vis", "clean", mem, None, True),
                              se._stream(draw, "ir", None, mem, iflag0, True), ctx.gts, ctx.h_frames, ctx.cap_fit_frames)
    ctx.cap_vis, ctx.cap_ir = cv, ci / ctx.cap_ir_scale
    u = dataclasses.replace(ctx, veto=None)
    ir = se._stream(draw, "ir", ir_cond, mem, iflag, True)
    res = {}
    for c in se.VIS_CONDS:
        g = ttbn.gate_ctx(ctx, c)
        vis = se._stream(draw, "vis", c, mem, None, True)
        o = run_systems(u, c, compute_metrics=False, vis_records=vis, ir_records=ir)
        recs = {"full_tta": o["fused_gated"]}
        if g.any():
            raw = [se._per_ckpt(draw, "vis", c, k, None) for k in mem]
            flp = [se._flip_ckpt(draw, "vis", c, k) for k in mem]
            raw_a = _adapt(raw, [_load_recs(out_path("raw", k, c, draw)) for k in mem], g)
            flp_a = _adapt(flp, [_load_recs(out_path("flip", k, c, draw)) for k in mem], g)
            vis_t = [se.merge(list(fr)) for fr in zip(*(raw_a + flp_a))]
            recs["full_tta_ttbn"] = run_systems(u, c, compute_metrics=False, vis_records=vis_t,
                                                ir_records=ir)["fused_gated"]
        else:
            recs["full_tta_ttbn"] = recs["full_tta"]
        parts = {a: frame_parts(r, ctx.gts) for a, r in recs.items()}
        for s, idx in sl.items():
            cell = f"{c}|{ir_cond}|{s}"
            res[cell] = {"gate": float(g[idx].mean())}
            for a, p in parts.items():
                dd = PRE / a / cell.replace("|", "__") / f"draw{draw}"
                if not (dd / "obs.json").is_file():
                    v2r.save_ship_pre(p, idx, dd)
                    (dd / "obs.json").write_text(json.dumps({"ap": ship_ap(p, idx)}), encoding="utf-8")
                res[cell][a] = json.loads((dd / "obs.json").read_text())["ap"]
    return spec, res


def cmd_score_dev(args) -> int:
    out_p = OUT / "score_dev.json"
    allres = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    specs = [(d, ic) for d in DEV_DRAWS for ic in ttbn.DEV_IR_CONDS if f"{d}|{ic}" not in allres]
    print(f"[score] {len(specs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (spec, res) in enumerate(ex.map(_dev_job, specs), 1):
            allres[f"{spec[0]}|{spec[1]}"] = res
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(allres), encoding="utf-8")
            os.replace(tmp, out_p)
            print(f"[score] {n}/{len(specs)} {spec} ({time.time() - t0:.0f}s)", flush=True)
    return 0


# ------------------------------------------------------------------ bootstrap
def _weights(slice_: str) -> np.ndarray:
    import holdout_p04_look as look
    from holdout_p04_v2_rescore import resample_weights
    from uqfusion.eval.ctx import NIGHT_RUNS
    wp = OUT / f"boot_weights_{slice_}.npy"
    if not wp.is_file():
        paths = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
        night = np.isin([Path(p).parent.name for p in paths], NIGHT_RUNS)
        idx = np.flatnonzero(night if slice_ == "night" else ~night)
        np.save(wp, resample_weights(paths, idx, look.N_BOOT))
    return np.load(wp)


def cmd_boot_dev(args) -> int:
    import p04_heldout_score as psc
    from boot_cpu import boot_cpu
    from holdout_p04_v2_rescore import load_ship_pre
    ttbn._low()
    W = {s: _weights(s) for s in ("day", "night")}
    todo = sorted(p.parent for p in PRE.rglob("obs.json") if not (p.parent / "reps.npy").is_file())
    print(f"[boot] {len(todo)} presorts", flush=True)
    checked = set()
    for n, d in enumerate(todo, 1):
        s = d.parent.name.rsplit("__", 1)[1]
        pre = load_ship_pre(d)
        if s not in checked:
            print(f"[boot] {s}: checked {psc._check(pre, W[s], None, None, cpu=True)} rows equal to the CPU reference",
                  flush=True)
            checked.add(s)
        np.save(d / "reps.npy", boot_cpu(pre, W[s]))
        if n % 50 == 0 or n == len(todo):
            print(f"[boot] {n}/{len(todo)}", flush=True)
    return 0


# ------------------------------------------------------------------ report
def cmd_report_dev(args) -> int:
    res = json.loads((OUT / "score_dev.json").read_text(encoding="utf-8"))
    ref = {}
    for f in ("fresh.json", "fresh2.json", "fresh3.json"):     # reruns agree where they overlap
        ref.update(json.loads((ROOT / "runs/seed_ens" / f).read_text(encoding="utf-8")))
    rows, out, worse, better, bad = [], {}, [], [], []
    for c in ("clean", "fog", "fog_s1", "lowlight", "glare"):
        for ic in ttbn.DEV_IR_CONDS:
            for s in ("day", "night"):
                cell = f"{c}|{ic}|{s}"
                vals = {a: [res[f"{d}|{ic}"][cell][a] for d in DEV_DRAWS] for a in ARMS}
                for d, v in zip(DEV_DRAWS, vals["full_tta"]):           # the registered reproduction check
                    want = ref[f"{d}|{ic}|full_tta"][f"{c}|0|{s}"]["fused"]
                    if abs(v - want) >= 5e-5:
                        bad.append(f"{cell} draw {d}: {v:.6f} vs {want:.6f}")
                base = PRE / "full_tta" / cell.replace("|", "__")
                tb = PRE / "full_tta_ttbn" / cell.replace("|", "__")
                r0 = np.mean([np.load(base / f"draw{d}" / "reps.npy") for d in DEV_DRAWS], axis=0)
                r1 = np.mean([np.load(tb / f"draw{d}" / "reps.npy") for d in DEV_DRAWS], axis=0)
                dlt = float(np.mean(vals["full_tta_ttbn"]) - np.mean(vals["full_tta"]))
                rr = (r1 - r0)[np.isfinite(r1 - r0)]
                lo, hi = (float(x) for x in np.percentile(rr, [2.5, 97.5]))
                gate = float(np.mean([res[f"{d}|{ic}"][cell]["gate"] for d in DEV_DRAWS]))
                key = f"{c}/{ic or 'clean'}/{s}"
                out[key] = {"full_tta": float(np.mean(vals["full_tta"])), "full_tta_ttbn": float(np.mean(vals["full_tta_ttbn"])),
                            "delta": dlt, "ci": [lo, hi], "gate": gate}
                if dlt <= -0.0014 and hi < 0:
                    worse.append(key)
                if dlt >= 0.0060 and lo > 0:
                    better.append(key)
                rows.append(f"| {key} | {gate:.3f} | {out[key]['full_tta']:.4f} | {out[key]['full_tta_ttbn']:.4f} | "
                            f"{dlt:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    if bad:
        L = ["# N4F development: REPRODUCTION CHECK FAILED", "", "`full_tta` recomputed here differs from the "
             "no-retrain value; nothing is reported until this is explained.", ""] + [f"* {b}" for b in bad]
        verdict = "CHECK FAILED"
    else:
        verdict = "PASSES" if (not worse and better) else "FAILS"
        L = ["# N4 on top of `full_tta` (N4F), development verdict", "",
             "`docs/prereg-ttbn-fulltta-2026-10-11.md`. Draws VIS 945–947 / IR 955–957 (second use). Δ = "
             "`full_tta_ttbn` − `full_tta` (ship AP), mean over the three draws; 95% interval from 1,000 moving-block "
             "replicates (L = 20, seed 1) over the slice's frames. `full_tta` reproduces its no-retrain values in all "
             "60 cells (|difference| < 5e-5).", "",
             f"**N4F {verdict} development.** F-H: {'PASS' if not worse else 'FAIL (' + ', '.join(worse) + ')'}. "
             f"F-G: {'PASS (' + str(len(better)) + ' cells: ' + ', '.join(better) + ')' if better else 'FAIL'}.", "",
             "| cell (VIS/IR/slice) | gate | full_tta | full_tta_ttbn | Δ [95%] |", "|---|---:|---:|---:|---:|"] + rows
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"verdict": verdict, "worse": worse, "better": better, "bad": bad,
                                                   "cells": out}, indent=1), encoding="utf-8")
    print("\n".join(L[:8]))
    return 0


# ------------------------------------------------------------------ pohang04 (only after a development pass)
P04_OUT = ROOT / "runs/holdout_p04/ttbn_fulltta"
DEV_REPORT = ROOT / "docs/eval/ttbn/ttbn_fulltta_dev_2026-10-11.json"


def p04_flip_path(k: int, cond: str, draw) -> Path:
    where = "clean" if cond == "clean" else ttbn.sub(draw)
    return P04_OUT / f"seed{k}" / where / f"gauss_vis_paired_{cond}_bn16_fp16_flip.pkl"


def _dev_passed() -> None:
    v = json.loads(DEV_REPORT.read_text(encoding="utf-8")).get("verdict") if DEV_REPORT.is_file() else None
    if v != "PASSES":
        raise SystemExit(f"N4F's pohang04 test runs only after a development pass (verdict: {v})")


def cmd_build_p04(args) -> int:
    """The flipped adapted views for exactly the blocks N4's pohang04 build ran (its unflipped adapted
    records are reused)."""
    import torch
    import p04_heldout as ph
    from uqfusion.eval.corruptions_v2 import CODE_REV
    from uqfusion.uq.infer import UQPredictor
    _dev_passed()
    ttbn._low()
    lst = ph.images("vis")
    todo = []
    for c in args.conds:
        for d in ([None] if c == "clean" else ph.VIS_DRAWS):
            if not all(p04_flip_path(k, c, d).is_file() for k in ttbn.SEEDS):
                if not all(ttbn.out_path("p04", k, c, d, True).is_file() for k in ttbn.SEEDS):
                    print(f"[n4f-p04] {c}/{d}: N4's adapted records not built yet, skipped", flush=True)
                    continue
                todo.append((c, d))
    print(f"[n4f-p04] {len(lst)} frames, {len(todo)} streams to build (fp16, fused statistics)", flush=True)
    if not todo:
        return 0
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=0, imgsz=640,
                         conf=0.001) for k in ttbn.SEEDS]
    nb = [ttbn.patch_bn(p.model) for p in preds]
    ph.setup_fp16(preds)
    ttbn.MODE["fused"] = True
    print(f"[n4f-p04] BN layers {nb}", flush=True)
    for c, d in todo:
        kc = ttbn.P04_CONDS[c]
        need = ttbn.needed("p04", c, d)
        idx = list(range(len(lst)))
        chunks = [idx[a:a + ttbn.BATCH] for a in range(0, len(idx), ttbn.BATCH) if need[a:a + ttbn.BATCH].any()]
        recs = [[None] * len(lst) for _ in ttbn.SEEDS]
        t0 = time.time()
        if chunks:
            spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": d or 0, "images": lst}
            ttbn.MODE["n"] = ttbn.PRIOR_N
            for n_done, ch in enumerate(ttbn._frames(spec, chunks, args.workers), 1):
                with torch.no_grad(), ph.amp(True):
                    out = ph._multi_batch(preds, [np.ascontiguousarray(im[:, ::-1]) for _, im in ch])
                for j in range(len(ttbn.SEEDS)):
                    for (i, im), r in zip(ch, out[j]):
                        recs[j][i] = {**ph._unflip(r, im.shape[1]), "image_path": lst[i]}
                if n_done % 200 == 0:
                    print(f"[n4f-p04] {c}/{d} {n_done}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
            ttbn.MODE["n"] = None
        for j, k in enumerate(ttbn.SEEDS):
            o = p04_flip_path(k, c, d)
            o.parent.mkdir(parents=True, exist_ok=True)
            meta = {"view": "flip", "cond": c, "draw": d, "prior_N": ttbn.PRIOR_N, "batch": ttbn.BATCH,
                    "blocks": "N4's pohang04 blocks", "blocks_run": len(chunks), "gated_frames": int(need.sum()),
                    "precision": "fp16 autocast, channels_last, fused statistics",
                    "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                    "corrupt_seed": d if kc else None, "corrupt_version": "v2" if kc else None,
                    "corrupt_code": CODE_REV if kc else None,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": meta, "records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[n4f-p04] {c}/{d} done: {len(chunks)} blocks, {int(need.sum())} gated frames, "
              f"{time.time() - t0:.0f}s", flush=True)
    return 0


def _p04_job(job):
    import holdout_p04_look as look
    import holdout_p04_v2_rescore as v2r
    import p04_heldout as ph
    import p04_heldout_score as psc
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import run_systems
    vc, ic, s = job
    ttbn._low()
    se._memo_distances()
    tag, key = psc.cell_tag((vc, ic)), f"all_{s}"
    d = psc.PRE / "full_tta_ttbn/fused" / tag / key
    if (d / "obs.json").is_file():
        return job, json.loads((d / "obs.json").read_text())["ap"]
    devcal = json.loads((psc.OUT / "devcal.json").read_text(encoding="utf-8"))
    dev = look.dev_context(0, "runs/cache_p3/seed0")
    sub_ = look.Pohang04() if s == "clean" else v2r.Pohang04V2()
    ctx = look.system_context(sub_, 0, s, sorted({vc, "clean"}), ic, dev)
    g = ttbn.gate_ctx(ctx, vc)
    if not np.array_equal(g, ttbn.gate_p04(vc, ic, s)):
        raise SystemExit(f"context gate differs from the frame-statistics gate ({vc}, {ic}, {s})")
    icond = ic or "clean"
    draw = None if s == "clean" else int(s[4:7])
    flag = ph.ir_flags(icond, draw)
    if not np.array_equal(flag, ctx.ir_d2 > ctx.ir_bound):
        raise SystemExit(f"IR flag differs from the context's ({vc}, {ic}, {s})")
    ir = psc.ens_stream("ir", icond, s, "full", flag)
    vd = None if vc == "clean" else draw
    raw = [psc._view(k, "vis", vc, s, False, None) for k in ttbn.SEEDS]
    flp = [psc._view(k, "vis", vc, s, True, None) for k in ttbn.SEEDS]
    raw_a = _adapt(raw, [_load_recs(ttbn.out_path("p04", k, vc, vd, True)) for k in ttbn.SEEDS], g)
    flp_a = _adapt(flp, [_load_recs(p04_flip_path(k, vc, vd)) for k in ttbn.SEEDS], g,
                   names=[r["image_path"] for r in raw[0]])
    vis_t = [se.merge(list(fr), psc.ENS_IOU) for fr in zip(*(raw_a + flp_a))]
    ctx.cap_vis, ctx.cap_ir = devcal["full_tta"]["cap_vis"], devcal["full_tta"]["cap_ir"]
    out = run_systems(dataclasses.replace(ctx, veto=None), vc, compute_metrics=False, vis_records=vis_t,
                      ir_records=ir)
    return job, psc.save_pre(frame_parts(out["fused_gated"], ctx.gts), d)


def cmd_score_p04(args) -> int:
    import p04_heldout_score as psc
    _dev_passed()
    jobs, zero, waiting = [], [], 0
    for vc, ic in [psc.VERDICT_CELL] + psc.CORRUPTED:
        for s in psc.subs_of((vc, ic)):
            if not ttbn.gate_p04(vc, ic, s).any():
                zero.append((vc, ic, s))
                continue
            vd = None if vc == "clean" else int(s[4:7])
            if all(p04_flip_path(k, vc, vd).is_file() for k in ttbn.SEEDS):
                jobs.append((vc, ic, s))
            else:
                waiting += 1
    print(f"[n4f-p04] {len(jobs)} jobs, {len(zero)} gate-free (Δ = 0 exactly), {waiting} waiting for the build",
          flush=True)
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for j, ap in ex.map(_p04_job, jobs):
            print(f"[n4f-p04] {j} {ap:.4f}", flush=True)
    return 0


def cmd_report_p04(args) -> int:
    import p04_heldout_score as psc
    L = ["# N4 on top of `full_tta` (N4F) on pohang04", "",
         "`docs/prereg-ttbn-fulltta-2026-10-11.md`. Ship AP against the VIS ground truth; both arms are ensembles "
         "of all five systems, mean over the four draws of a corrupted cell. Δ = `full_tta_ttbn` − `full_tta`, "
         "paired; 95% interval from the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1).", "",
         "| cell (VIS/IR) | gated frames | full_tta | full_tta_ttbn | Δ [95%] |", "|---|---:|---:|---:|---:|"]
    res, worse, better, complete = {}, [], [], True
    for c in [psc.VERDICT_CELL] + psc.CORRUPTED:
        tag = psc.cell_tag(c)
        rate = float(np.mean([ttbn.gate_p04(c[0], c[1], s).mean() for s in psc.subs_of(c)]))
        a = psc.series("full_tta/fused", c)
        if rate == 0.0:
            res[tag] = {"gate": 0.0, "delta": 0.0, "note": "the gate fires on no frame"}
            L.append(f"| {tag.replace('__', '/')} | 0 | {'' if a is None else f'{a[0]:.4f}'} | = full_tta | 0 (identical) |")
            continue
        b = psc.series("full_tta_ttbn/fused", c)
        if a is None or b is None:
            L.append(f"| {tag.replace('__', '/')} | {rate:.3f} | pending | | |")
            complete = False
            continue
        dl, (lo, hi) = b[0] - a[0], psc.ci(b[1] - a[1])
        res[tag] = {"gate": rate, "full_tta": a[0], "full_tta_ttbn": b[0], "delta": dl, "ci": [lo, hi]}
        if dl <= -0.0014 and hi < 0:
            worse.append(tag)
        if dl >= 0.0060 and lo > 0:
            better.append(tag)
        L.append(f"| {tag.replace('__', '/')} | {rate:.3f} | {a[0]:.4f} | {b[0]:.4f} | {dl:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    v = ("PENDING" if not complete else "REPLICATES" if (not worse and better) else
         f"DOES NOT REPLICATE (worse: {', '.join(worse) or 'none'}; better: {', '.join(better) or 'none'})")
    L[3:3] = ["", f"**P-FT: {v}.**"]
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"P-FT": v, "cells": res}, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build-dev")
    b.add_argument("--workers", type=int, default=4)
    s = sp.add_parser("score-dev")
    s.add_argument("--workers", type=int, default=3)
    sp.add_parser("boot-dev")
    r = sp.add_parser("report-dev")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_fulltta_dev_2026-10-11.md")
    b = sp.add_parser("build-p04")
    b.add_argument("--conds", nargs="+", default=["lowlight", "noise_s2", "fog", "rain_s2", "blur_s3", "clean"])
    b.add_argument("--workers", type=int, default=6)
    s = sp.add_parser("score-p04")
    s.add_argument("--workers", type=int, default=2)
    r = sp.add_parser("report-p04")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_fulltta_p04_2026-10-11.md")
    args = ap.parse_args()
    return {"build-dev": cmd_build_dev, "score-dev": cmd_score_dev, "boot-dev": cmd_boot_dev,
            "report-dev": cmd_report_dev, "build-p04": cmd_build_p04, "score-p04": cmd_score_p04,
            "report-p04": cmd_report_p04}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
