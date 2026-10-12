"""N4Fg: N4F with N4g's glare term in the gate (`docs/prereg-ttbn-fulltta-glare-2026-10-12.md`).

`full_tta_ttbng` = `full_tta` (five checkpoints, raw + flipped VIS views, IR NLM where flagged, plain WBF,
`union`) with every VIS view BN-mix-adapted on the frames N4g's gate passes:
(q_vis < 1 OR (VIS mean > 117.92 AND VIS std > 62.31)) AND ir_p05 <= 41.5 AND ir_d2 <= 222.7.
N4's BN-mix (N = 16, 8-frame blocks on N4's grid), fp16 adapted views with fused statistics.

Records. A block's adapted output depends only on its own frames, so every block N4F (development and
pohang04) or N4g (P-G) already ran is reused; only the missing blocks are built here. Where N4g's gate equals
N4's, N4Fg is N4F by construction.

    py -3.13 scripts/ttbn_fulltta_glare.py plan-dev
    py -3.13 scripts/ttbn_fulltta_glare.py build-dev              # missing blocks, raw + flip, fp16
    py -3.13 scripts/ttbn_fulltta_glare.py check-fp16             # one glare draw, fp32 against fp16
    py -3.13 scripts/ttbn_fulltta_glare.py score-dev
    py -3.13 scripts/ttbn_fulltta_glare.py boot-dev
    py -3.13 scripts/ttbn_fulltta_glare.py report-dev
  pohang04 (only after a development pass):
    py -3.13 scripts/ttbn_fulltta_glare.py build-p04-flip         # plain flipped glare views (full_tta)
    py -3.13 scripts/ttbn_fulltta_glare.py build-p04-adapted      # missing adapted flipped blocks
    py -3.13 scripts/ttbn_fulltta_glare.py score-p04
    py -3.13 scripts/p04_heldout_score.py boot
    py -3.13 scripts/ttbn_fulltta_glare.py report-p04
"""

from __future__ import annotations

import argparse
import dataclasses
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

import ttbn  # noqa: E402
import ttbn_fulltta as tf  # noqa: E402
import ttbn_glare_p04 as tgp  # noqa: E402
import ttbn_variants as tv  # noqa: E402

OUT = ROOT / "runs/novel/ttbn_fulltta_glare"
PRE = OUT / "pre"
P04_OUT = ROOT / "runs/holdout_p04/ttbn_fulltta_glare"
DEV_DRAWS = tf.DEV_DRAWS
CHECK_DRAW = 945
DEV_REPORT = ROOT / "docs/eval/ttbn/ttbn_fulltta_glare_dev_2026-10-12.json"
FLOOR_H, FLOOR_G = -0.0014, 0.0060
VIEWS = ("raw", "flip")


# ------------------------------------------------------------------ gates and blocks (frame statistics)
def gate4_dev(d, c, ic):
    return tf._gate_stats(d, c, ic)


def gateg_dev(d, c, ic):
    return tv.gate_g_stats(d, c, ic)


def needed_g_dev(d, c):
    out = None
    for ic in ttbn.DEV_IR_CONDS:
        g = gateg_dev(d, c, ic)
        out = g if out is None else out | g
    return out


def differs_dev(d, c) -> list:
    """IR conditions where N4g's gate differs from N4's for this development stream and draw."""
    return [ic for ic in ttbn.DEV_IR_CONDS if not np.array_equal(gate4_dev(d, c, ic), gateg_dev(d, c, ic))]


def _ran(p: Path) -> set[int]:
    if not p.is_file():
        return set()
    with open(p, "rb") as fh:
        rr = pickle.load(fh)["records"]
    return {i // ttbn.BATCH for i, r in enumerate(rr) if r is not None}


def dev_path(view: str, k: int, c: str, d: int, fp16: bool = True) -> Path:
    return OUT / view / f"seed{k}" / ttbn.sub(d) / f"gauss_vis_paired_{c}_bn16{'_fp16' if fp16 else ''}.pkl"


def plan_dev() -> list[tuple[str, int, np.ndarray, list[int]]]:
    """(stream, draw, N4g-needed frames, blocks N4F's development build did not run) where N4g's gate differs."""
    out = []
    for c in tf.BUILD_CONDS:
        for d in DEV_DRAWS:
            if not differs_dev(d, c):
                continue
            need = needed_g_dev(d, c)
            ran = _ran(tf.out_path("raw", 0, c, d))
            assert ran == _ran(tf.out_path("flip", 0, c, d))
            out.append((c, d, need, sorted(tgp.blocks_of(need) - ran)))
    return out


def cmd_plan_dev(args) -> int:
    for c in ttbn.DEV_CONDS:
        for d in DEV_DRAWS:
            dif = differs_dev(d, c)
            if dif:
                print(f"{c}/{d}: gates differ under IR {dif}", flush=True)
    for c, d, need, new in plan_dev():
        print(f"{c}/{d}: N4g needs {int(need.sum())} frames in {len(tgp.blocks_of(need))} blocks; "
              f"{len(new)} not run by N4F", flush=True)
    return 0


# ------------------------------------------------------------------ inference (N4F's loop)
def _predictors(fp16: bool):
    import p04_heldout as ph
    from uqfusion.uq.infer import UQPredictor
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=0, imgsz=640,
                         conf=0.001) for k in ttbn.SEEDS]
    for p in preds:
        ttbn.patch_bn(p.model)
    ttbn.MODE["fused"] = False
    if fp16:
        ph.setup_fp16(preds)
        ttbn.MODE["fused"] = True
    return preds


def _run(preds, lst, kc, seed, chunks, workers, fp16, views, tag):
    """Adapted raw and/or flipped records of `chunks` (N4's BN-mix on each block), as `ttbn_fulltta`."""
    import torch
    import p04_heldout as ph
    recs = {v: [[None] * len(lst) for _ in ttbn.SEEDS] for v in views}
    if not chunks:
        return recs
    spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": seed, "images": lst}
    t0 = time.time()
    ttbn.MODE["n"] = ttbn.PRIOR_N
    for n_done, ch in enumerate(ttbn._frames(spec, chunks, workers), 1):
        ims = [im for _, im in ch]
        with torch.no_grad(), ph.amp(fp16):
            out_r = ph._multi_batch(preds, ims) if "raw" in views else None
            out_f = (ph._multi_batch(preds, [np.ascontiguousarray(im[:, ::-1]) for im in ims])
                     if "flip" in views else None)
        for j in range(len(ttbn.SEEDS)):
            for n, (i, im) in enumerate(ch):
                if out_r is not None:
                    recs["raw"][j][i] = {**ph._f32(out_r[j][n]), "image_path": lst[i]}
                if out_f is not None:
                    recs["flip"][j][i] = {**ph._unflip(out_f[j][n], im.shape[1]), "image_path": lst[i]}
        if n_done % 200 == 0:
            print(f"[{tag}] {n_done}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
    ttbn.MODE["n"] = None
    return recs


def _write(path: Path, meta: dict, recs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".tmp"), "wb") as fh:
        pickle.dump({"meta": meta, "records": recs}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path.with_suffix(".tmp"), path)


def _meta(c, d, kc, blocks, fp16, k, **kw):
    from uqfusion.eval.corruptions_v2 import CODE_REV
    return {"variant": "N4Fg", "cond": c, "draw": d, "prior_N": ttbn.PRIOR_N, "batch": ttbn.BATCH,
            "blocks": blocks, "precision": "fp16 autocast, channels_last, fused statistics" if fp16 else "fp32",
            "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
            "corrupt_seed": d if kc else None, "corrupt_version": "v2" if kc else None,
            "corrupt_code": CODE_REV if kc else None,
            "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt"), **kw}


def cmd_build_dev(args) -> int:
    ttbn._low()
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    todo = [t for t in plan_dev() if not all(dev_path(v, k, t[0], t[1]).is_file() for v in VIEWS for k in ttbn.SEEDS)]
    print(f"[n4fg-dev] {len(todo)} streams to build (fp16)", flush=True)
    if not todo:
        return 0
    preds = _predictors(True)
    for c, d, need, new in todo:
        kc = ttbn.DEV_CONDS[c]
        chunks = [list(range(b * ttbn.BATCH, min((b + 1) * ttbn.BATCH, len(lst)))) for b in new]
        t0 = time.time()
        recs = _run(preds, lst, kc, d, chunks, args.workers, True, VIEWS, f"n4fg-dev {c}/{d}")
        for v in VIEWS:
            for j, k in enumerate(ttbn.SEEDS):
                _write(dev_path(v, k, c, d), _meta(c, d, kc, "N4's grid; only blocks N4F's development build did not run",
                                                   True, k, view=v, new_blocks=[int(b) for b in new],
                                                   needed_frames=int(need.sum())), recs[v][j])
        print(f"[n4fg-dev] {c}/{d} done: {len(new)} blocks, {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ development records and scoring
def _load(p: Path):
    with open(p, "rb") as fh:
        return pickle.load(fh)["records"]


def dev_adapted(view: str, k: int, c: str, d: int, fp32_full: bool = False):
    """N4Fg's adapted records of one view: N4F's where N4F ran the block, else this build's.
    `fp32_full`: the fp32 check build (every N4g block of the check draw, nothing reused)."""
    if fp32_full:
        return _load(dev_path(view, k, c, d, fp16=False))
    base = _load(tf.out_path(view, k, c, d))
    p = dev_path(view, k, c, d)
    if not p.is_file():
        return base
    new = _load(p)
    return [a if a is not None else b for a, b in zip(base, new)]


def _dev_ctx(draw, ir_cond):
    import seed_ensemble as se
    from uqfusion.eval.ctx import NIGHT_RUNS, capability_prior, load_context
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
    ir = se._stream(draw, "ir", ir_cond, mem, iflag, True)
    return ctx, sl, ir


def _gate_g_ctx(ctx, c, glare_bright: Path) -> np.ndarray:
    irok = ~np.asarray(ctx.ir_night_raw, bool) & (np.asarray(ctx.ir_d2) <= ctx.ir_bound)
    return ((np.asarray(ctx.q_vis_by_cond[c]) < 1.0) | tv.glare_term(glare_bright)) & irok


def _ttbng(u, c, draw, gate, ir, fp32_full=False):
    import seed_ensemble as se
    from uqfusion.eval.ctx import run_systems
    mem = tuple(ttbn.SEEDS)
    raw = [se._per_ckpt(draw, "vis", c, k, None) for k in mem]
    flp = [se._flip_ckpt(draw, "vis", c, k) for k in mem]
    raw_a = tf._adapt(raw, [dev_adapted("raw", k, c, draw, fp32_full) for k in mem], gate)
    flp_a = tf._adapt(flp, [dev_adapted("flip", k, c, draw, fp32_full) for k in mem], gate)
    vis_t = [se.merge(list(fr)) for fr in zip(*(raw_a + flp_a))]
    return run_systems(u, c, compute_metrics=False, vis_records=vis_t, ir_records=ir)["fused_gated"]


def _dev_job(spec):
    import holdout_p04_v2_rescore as v2r
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import run_systems
    from v2_sensitivity import ship_ap
    draw, ir_cond = spec
    ttbn._low()
    se._CACHE.clear()
    se._memo_distances()
    ctx, sl, ir = _dev_ctx(draw, ir_cond)
    u = dataclasses.replace(ctx, veto=None)
    mem = tuple(ttbn.SEEDS)
    res = {}
    for c in se.VIS_CONDS:
        g4 = ttbn.gate_ctx(ctx, c)
        gg = _gate_g_ctx(ctx, c, ttbn._dev_stats(draw, "vis", c, "brightness"))
        if not np.array_equal(g4, gate4_dev(draw, c, ir_cond)) or not np.array_equal(gg, gateg_dev(draw, c, ir_cond)):
            raise SystemExit(f"context gate differs from the frame-statistics gate ({draw}, {ir_cond}, {c})")
        vis = se._stream(draw, "vis", c, mem, None, True)
        recs = {"full_tta": run_systems(u, c, compute_metrics=False, vis_records=vis, ir_records=ir)["fused_gated"]}
        equal = bool(np.array_equal(g4, gg))
        if not equal:
            recs["full_tta_ttbn"] = _ttbng(u, c, draw, g4, ir) if g4.any() else recs["full_tta"]
            recs["full_tta_ttbng"] = _ttbng(u, c, draw, gg, ir)
        parts = {a: frame_parts(r, ctx.gts) for a, r in recs.items()}
        for s, idx in sl.items():
            cell = f"{c}|{ir_cond}|{s}"
            res[cell] = {"gate": float(g4[idx].mean()), "gate_g": float(gg[idx].mean()), "equal": equal,
                         "extra": int((gg & ~g4)[idx].sum())}
            for a, p in parts.items():
                res[cell][a] = ship_ap(p, idx)
                if a == "full_tta_ttbng":
                    dd = PRE / a / cell.replace("|", "__") / f"draw{draw}"
                    if not (dd / "obs.json").is_file():
                        v2r.save_ship_pre(p, idx, dd)
                        (dd / "obs.json").write_text(json.dumps({"ap": res[cell][a]}), encoding="utf-8")
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


# ------------------------------------------------------------------ development fp16 check
def _check_job(fp32_full):
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from v2_sensitivity import ship_ap
    ttbn._low()
    se._CACHE.clear()
    se._memo_distances()
    ctx, sl, ir = _dev_ctx(CHECK_DRAW, None)
    gg = _gate_g_ctx(ctx, "glare", ttbn._dev_stats(CHECK_DRAW, "vis", "glare", "brightness"))
    u = dataclasses.replace(ctx, veto=None)
    return fp32_full, ship_ap(frame_parts(_ttbng(u, "glare", CHECK_DRAW, gg, ir, fp32_full), ctx.gts), sl["day"])


def cmd_check_fp16(args) -> int:
    """fp16 against fp32 on one glare draw: N4Fg's glare/clean/day ship AP must move by at most 0.0010."""
    chk = OUT / "fp16_check.json"
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    need = needed_g_dev(CHECK_DRAW, "glare")
    blocks = sorted(tgp.blocks_of(need))
    if _ran(tf.out_path("raw", 0, "glare", CHECK_DRAW)):
        raise SystemExit("the check draw must be one where N4F ran no glare block (the fp16 side is then all new)")
    if not all(dev_path(v, k, "glare", CHECK_DRAW) .is_file() for v in VIEWS for k in ttbn.SEEDS):
        raise SystemExit("run build-dev first")
    if not all(dev_path(v, k, "glare", CHECK_DRAW, fp16=False).is_file() for v in VIEWS for k in ttbn.SEEDS):
        ttbn._low()
        preds = _predictors(False)
        chunks = [list(range(b * ttbn.BATCH, min((b + 1) * ttbn.BATCH, len(lst)))) for b in blocks]
        recs = _run(preds, lst, ttbn.DEV_CONDS["glare"], CHECK_DRAW, chunks, args.workers, False, VIEWS, "check")
        for v in VIEWS:
            for j, k in enumerate(ttbn.SEEDS):
                _write(dev_path(v, k, "glare", CHECK_DRAW, fp16=False),
                       _meta("glare", CHECK_DRAW, ttbn.DEV_CONDS["glare"], "every N4g block (fp32 check)", False, k,
                             view=v), recs[v][j])
        del preds
    with ProcessPoolExecutor(2, max_tasks_per_child=1) as ex:
        r = dict(ex.map(_check_job, (True, False)))
    a, b = r[True], r[False]
    ok = abs(b - a) <= ttbn.FP16_TOL
    chk.write_text(json.dumps({"pass": bool(ok), "tolerance": ttbn.FP16_TOL, "cell": "glare/clean/day",
                               "draw": CHECK_DRAW, "fp32": a, "fp16": b, "diff": b - a, "blocks": len(blocks)},
                              indent=1), encoding="utf-8")
    print(f"[check] full_tta_ttbng glare/clean/day draw {CHECK_DRAW}: fp32 {a:.5f} fp16 {b:.5f} diff {b - a:+.5f} -> "
          f"{'PASS: fp16' if ok else 'FAIL: fp32'}", flush=True)
    return 0


# ------------------------------------------------------------------ development bootstrap and report
def cmd_boot_dev(args) -> int:
    from boot_cpu import boot_cpu
    from holdout_p04_v2_rescore import load_ship_pre
    ttbn._low()
    W = {s: tf._weights(s) for s in ("day", "night")}
    todo = sorted(p.parent for p in PRE.rglob("obs.json") if not (p.parent / "reps.npy").is_file())
    print(f"[boot] {len(todo)} presorts", flush=True)
    for n, d in enumerate(todo, 1):
        s = d.parent.name.rsplit("__", 1)[1]
        np.save(d / "reps.npy", boot_cpu(load_ship_pre(d), W[s]))
        if n % 10 == 0 or n == len(todo):
            print(f"[boot] {n}/{len(todo)}", flush=True)
    return 0


def _ref_full_tta() -> tuple[dict, dict]:
    per_draw = {}
    for f in ("fresh.json", "fresh2.json", "fresh3.json"):
        per_draw.update(json.loads((ROOT / "runs/seed_ens" / f).read_text(encoding="utf-8")))
    mean = json.loads((ROOT / "docs/eval/no_retrain/ensembles_fresh.json").read_text(encoding="utf-8"))
    return per_draw, mean


def cmd_report_dev(args) -> int:
    res = json.loads((OUT / "score_dev.json").read_text(encoding="utf-8"))
    n4f = json.loads((ROOT / "docs/eval/ttbn/ttbn_fulltta_dev_2026-10-11.json").read_text(encoding="utf-8"))["cells"]
    n4f_raw = json.loads((tf.OUT / "score_dev.json").read_text(encoding="utf-8"))
    chk = json.loads((OUT / "fp16_check.json").read_text(encoding="utf-8"))
    ref_d, ref_m = _ref_full_tta()
    rows, out, bad, worse_h, worse_n4f = [], {}, [], [], []
    for c in ttbn.DEV_CONDS:
        for ic in ttbn.DEV_IR_CONDS:
            for s in ("day", "night"):
                cell = f"{c}|{ic}|{s}"
                key = f"{c}/{ic or 'clean'}/{s}"
                rr = [res[f"{d}|{ic}"][cell] for d in DEV_DRAWS]
                ft = [r["full_tta"] for r in rr]
                for d, v in zip(DEV_DRAWS, ft):                         # the registered reproduction checks
                    if abs(v - ref_d[f"{d}|{ic}|full_tta"][f"{c}|0|{s}"]["fused"]) >= 5e-5:
                        bad.append(f"{key} draw {d}: {v:.6f} vs per-draw reference")
                    if v != n4f_raw[f"{d}|{ic}"][cell]["full_tta"]:
                        bad.append(f"{key} draw {d}: {v!r} vs N4F's development full_tta")
                m = ref_m[f"full_tta|{c}|{ic}|{s}"]["arm"]["fused"]
                if round(float(np.mean(ft)), 4) != round(m, 4):
                    bad.append(f"{key}: mean {np.mean(ft):.6f} vs ensembles_fresh.json {m:.6f}")
                gate = float(np.mean([r["gate"] for r in rr]))
                gate_g = float(np.mean([r["gate_g"] for r in rr]))
                extra = int(sum(r["extra"] for r in rr))
                if all(r["equal"] for r in rr):
                    o = n4f[key]
                    out[key] = {"gate": gate, "gate_g": gate_g, "equal": True, "full_tta": o["full_tta"],
                                "full_tta_ttbng": o["full_tta_ttbn"], "delta": o["delta"], "ci": o["ci"],
                                "minus_n4f": 0.0}
                    dl, (lo, hi) = o["delta"], o["ci"]
                    rows.append(f"| {key} | {gate:.3f} | {gate_g:.3f} | {o['full_tta']:.4f} | = N4F | "
                                f"{dl:+.4f} [{lo:+.4f}, {hi:+.4f}] (N4F's) | 0 (gates equal) |")
                else:
                    for d, r in zip(DEV_DRAWS, rr):                    # N4F recomputed here must equal N4F
                        if r["full_tta_ttbn"] != n4f_raw[f"{d}|{ic}"][cell]["full_tta_ttbn"]:
                            bad.append(f"{key} draw {d}: N4F recomputed {r['full_tta_ttbn']!r} vs stored")
                    base = tf.PRE / "full_tta" / cell.replace("|", "__")
                    b4 = tf.PRE / "full_tta_ttbn" / cell.replace("|", "__")
                    bg = PRE / "full_tta_ttbng" / cell.replace("|", "__")
                    r0 = np.mean([np.load(base / f"draw{d}" / "reps.npy") for d in DEV_DRAWS], axis=0)
                    r4 = np.mean([np.load(b4 / f"draw{d}" / "reps.npy") for d in DEV_DRAWS], axis=0)
                    rg = np.mean([np.load(bg / f"draw{d}" / "reps.npy") for d in DEV_DRAWS], axis=0)
                    vg = float(np.mean([r["full_tta_ttbng"] for r in rr]))
                    v4 = float(np.mean([r["full_tta_ttbn"] for r in rr]))
                    dl = vg - float(np.mean(ft))
                    x = (rg - r0)[np.isfinite(rg - r0)]
                    lo, hi = (float(q) for q in np.percentile(x, [2.5, 97.5]))
                    d4 = vg - v4
                    y = (rg - r4)[np.isfinite(rg - r4)]
                    lo4, hi4 = (float(q) for q in np.percentile(y, [2.5, 97.5]))
                    out[key] = {"gate": gate, "gate_g": gate_g, "equal": False, "extra_frames": extra,
                                "full_tta": float(np.mean(ft)), "full_tta_ttbn": v4, "full_tta_ttbng": vg,
                                "delta": dl, "ci": [lo, hi], "minus_n4f": d4, "ci_minus_n4f": [lo4, hi4]}
                    if extra > 0 and d4 < FLOOR_H:
                        worse_n4f.append(key)
                    rows.append(f"| {key} | {gate:.3f} | {gate_g:.3f} | {np.mean(ft):.4f} | {vg:.4f} | "
                                f"{dl:+.4f} [{lo:+.4f}, {hi:+.4f}] | {d4:+.4f} [{lo4:+.4f}, {hi4:+.4f}] |")
                if out[key]["delta"] <= FLOOR_H and out[key]["ci"][1] < 0:
                    worse_h.append(key)
    if bad:
        L = ["# N4Fg development: REPRODUCTION CHECK FAILED", "", "Nothing is reported until this is explained.", ""] + \
            [f"* {b}" for b in bad]
        verdict = "CHECK FAILED"
    else:
        verdict = "PASSES" if not worse_h and not worse_n4f else "FAILS"
        L = ["# N4Fg (N4F with N4g's glare term) on development", "",
             "`docs/prereg-ttbn-fulltta-glare-2026-10-12.md`. Draws VIS 945–947 / IR 955–957 (N4F's; a third use). "
             "Ship AP; Δ = `full_tta_ttbng` − `full_tta`, mean over the three draws; 95% interval from N4F's 1,000 "
             "moving-block replicates (L = 20, seed 1) over the slice's frames. `full_tta` recomputed here equals "
             "its per-draw no-retrain values (|difference| < 5e-5), `docs/eval/no_retrain/ensembles_fresh.json` to "
             "the fourth decimal, and N4F's development `full_tta` exactly, in all 60 cells. Where the gates differ, "
             "N4F recomputed here equals N4F's stored scores exactly.", "",
             f"Precision: fp16 (check on glare/clean/day, draw {chk['draw']}: fp32 {chk['fp32']:.5f}, fp16 "
             f"{chk['fp16']:.5f}, Δ {chk['diff']:+.5f}, tolerance {chk['tolerance']}; "
             f"{'PASS' if chk['pass'] else 'FAIL'}).", "",
             f"**N4Fg {verdict} development.** No cell worse than `full_tta`: "
             f"{'PASS' if not worse_h else 'FAIL (' + ', '.join(worse_h) + ')'}. Cells where the glare term adds "
             f"gated frames at or above −0.0014 against N4F: "
             f"{'PASS' if not worse_n4f else 'FAIL (' + ', '.join(worse_n4f) + ')'}.", "",
             "| cell (VIS/IR/slice) | N4 gate | N4g gate | full_tta | full_tta_ttbng | Δ vs full_tta [95%] | "
             "N4Fg − N4F [95%] |", "|---|---:|---:|---:|---:|---:|---:|"] + rows
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"verdict": verdict, "worse": worse_h, "worse_vs_n4f": worse_n4f,
                                                   "bad": bad, "fp16_check": chk, "cells": out}, indent=1),
                                      encoding="utf-8")
    print("\n".join(L[:10]))
    return 0


# ------------------------------------------------------------------ pohang04 (only after a development pass)
GLARE_CELL = ("glare", None)
#: existing P-T cells where N4g's gate differs from N4's on some draw (P-G's registration)
OTHER_BUILT = [("clean", "blur_s2"), ("clean", "fog_s2"), ("fog", None)]


def _dev_passed() -> None:
    v = json.loads(DEV_REPORT.read_text(encoding="utf-8")).get("verdict") if DEV_REPORT.is_file() else None
    if v != "PASSES":
        raise SystemExit(f"N4Fg's pohang04 test runs only after a development pass (verdict: {v})")


def p04_new_flip(k: int, cond: str, draw) -> Path:
    where = "clean" if cond == "clean" else ttbn.sub(draw)
    return P04_OUT / f"seed{k}" / where / f"gauss_vis_paired_{cond}_bn16_fp16_flip.pkl"


def _flip_unit(args) -> int:
    """p04_heldout's build-unit for the VIS glare stream (glare is not one of its registered conditions)."""
    import p04_heldout as ph
    ph.VIS_CONDS["glare"] = tgp.GLARE
    return ph.cmd_build_unit(args)


def cmd_build_p04_flip(args) -> int:
    import p04_heldout as ph
    from uqfusion.config import resolve_gpu_python
    _dev_passed()
    gp = resolve_gpu_python(None)
    env = dict(os.environ, OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    (ph.TTA / "logs").mkdir(parents=True, exist_ok=True)
    for d in ph.VIS_DRAWS:
        if all(ph.flip_out(k, "vis", "glare", d).is_file() for k in ttbn.SEEDS):
            print(f"[flip] glare/{d} skip (built)", flush=True)
            continue
        cmd = [gp, "-u", str(Path(__file__)), "_flip-unit", "--mod", "vis", "--cond", "glare", "--draw", str(d),
               "--workers", str(args.workers), "--fp16"]
        t = time.time()
        print(f"[flip] glare/{d} start", flush=True)
        with open(ph.TTA / "logs" / f"vis_glare_{d}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        if rc != 0 or not all(ph.flip_out(k, "vis", "glare", d).is_file() for k in ttbn.SEEDS):
            raise SystemExit(f"[flip] glare/{d} FAILED rc={rc}")
        print(f"[flip] glare/{d} ok in {time.time() - t:.0f}s", flush=True)
    return 0


def plan_p04():
    """(stream, draw, N4g-needed frames, flipped blocks neither N4F nor an earlier N4Fg build ran)."""
    out = []
    for vc in ("clean", "fog", "glare"):
        for d in ([None] if vc == "clean" else tgp.p04_draws()):
            need = tgp.needed_g(vc, d)
            ran = _ran(tf.p04_flip_path(0, vc, d)) if vc in ttbn.P04_CONDS else set()
            new = sorted(tgp.blocks_of(need) - ran)
            if new:
                out.append((vc, d, need, new))
    return out


def cmd_build_p04_adapted(args) -> int:
    import p04_heldout as ph
    _dev_passed()
    ttbn._low()
    lst = ph.images("vis")
    todo = [t for t in plan_p04() if not all(p04_new_flip(k, t[0], t[1]).is_file() for k in ttbn.SEEDS)]
    print(f"[n4fg-p04] {len(todo)} streams to build (fp16, flipped views)", flush=True)
    if not todo:
        return 0
    preds = _predictors(True)
    for vc, d, need, new in todo:
        kc = tgp.GLARE if vc == "glare" else ttbn.P04_CONDS[vc]
        chunks = [list(range(b * ttbn.BATCH, min((b + 1) * ttbn.BATCH, len(lst)))) for b in new]
        t0 = time.time()
        recs = _run(preds, lst, kc, d or 0, chunks, args.workers, True, ("flip",), f"n4fg-p04 {vc}/{d}")
        for j, k in enumerate(ttbn.SEEDS):
            _write(p04_new_flip(k, vc, d), _meta(vc, d, kc, "N4's grid; flipped blocks N4F's pohang04 build did not run",
                                                 True, k, view="flip", target="p04", new_blocks=[int(b) for b in new],
                                                 needed_frames=int(need.sum())), recs["flip"][j])
        print(f"[n4fg-p04] {vc}/{d} done: {len(new)} blocks, {time.time() - t0:.0f}s", flush=True)
    return 0


def _merge_recs(*lists):
    out = None
    for rr in lists:
        if rr is None:
            continue
        out = list(rr) if out is None else [a if a is not None else b for a, b in zip(out, rr)]
    return out


def _p04_adapted(view: str, k: int, vc: str, draw):
    """Raw: N4's pohang04 records, then P-G's new N4g blocks. Flipped: N4F's, then this build's."""
    def opt(p):
        return _load(p) if p.is_file() else None
    vd = None if vc == "clean" else draw
    if view == "raw":
        return _merge_recs(opt(ttbn.out_path("p04", k, vc, vd, True)), opt(tgp.new_path(k, vc, vd, True)))
    return _merge_recs(opt(tf.p04_flip_path(k, vc, vd)) if vc in ttbn.P04_CONDS else None, opt(p04_new_flip(k, vc, vd)))


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
    draw = int(s[4:7])
    devcal = json.loads((psc.OUT / "devcal.json").read_text(encoding="utf-8"))
    dev = look.dev_context(0, "runs/cache_p3/seed0")
    sub_ = tgp._substrate() if vc == "glare" else v2r.Pohang04V2()
    ctx = look.system_context(sub_, 0, s, sorted({vc, "clean"}), ic, dev)
    g4 = ttbn.gate_ctx(ctx, vc)
    gg = _gate_g_ctx(ctx, vc, ttbn._p04_stats("vis", vc, draw, "brightness"))
    if not np.array_equal(g4, ttbn.gate_p04(vc, ic, s)) or not np.array_equal(gg, tgp.gate_g_p04(vc, ic, s)):
        raise SystemExit(f"context gate differs from the frame-statistics gate ({vc}, {ic}, {s})")
    icond = ic or "clean"
    flag = ph.ir_flags(icond, draw)
    if not np.array_equal(flag, ctx.ir_d2 > ctx.ir_bound):
        raise SystemExit(f"IR flag differs from the context's ({vc}, {ic}, {s})")
    ir = psc.ens_stream("ir", icond, s, "full", flag)
    raw = [psc._view(k, "vis", vc, s, False, None) for k in ttbn.SEEDS]
    flp = [psc._view(k, "vis", vc, s, True, None) for k in ttbn.SEEDS]
    names = [r["image_path"] for r in raw[0]]
    ad_r = [_p04_adapted("raw", k, vc, draw) for k in ttbn.SEEDS]
    ad_f = [_p04_adapted("flip", k, vc, draw) for k in ttbn.SEEDS]
    ctx.cap_vis, ctx.cap_ir = devcal["full_tta"]["cap_vis"], devcal["full_tta"]["cap_ir"]
    u = dataclasses.replace(ctx, veto=None)

    def fused(gate):
        vis_t = [se.merge(list(fr), psc.ENS_IOU) for fr in
                 zip(*(tf._adapt(raw, ad_r, gate) + tf._adapt(flp, ad_f, gate, names=names)))]
        return vis_t, run_systems(u, vc, compute_metrics=False, vis_records=vis_t, ir_records=ir)

    out = {"gate": float(g4.mean()), "gate_g": float(gg.mean()), "extra": int((gg & ~g4).sum())}

    def put(name, recs):
        d = psc.PRE / name / tag / key
        if not (d / "obs.json").is_file():
            psc.save_pre(frame_parts(recs, ctx.gts), d)

    if (vc, ic) == GLARE_CELL:                         # full_tta and N4F on the new cell
        vis0 = psc.ens_stream("vis", vc, s, "full")
        o0 = run_systems(u, vc, compute_metrics=False, vis_records=vis0, ir_records=ir)
        put("full/vis", vis0)
        put("full/ir", o0["ir_in_vis"])
        put("full_tta/fused", o0["fused_gated"])
        put("full_tta_ttbn/fused", fused(g4)[1]["fused_gated"] if g4.any() else o0["fused_gated"])
    put("full_tta_ttbng/fused", fused(gg)[1]["fused_gated"])
    return job, out


def p04_jobs():
    import p04_heldout_score as psc
    jobs = []
    for c in [GLARE_CELL] + OTHER_BUILT:
        for s in psc.subs_of(c):
            if c == GLARE_CELL or not np.array_equal(tgp.gate_g_p04(c[0], c[1], s), ttbn.gate_p04(c[0], c[1], s)):
                jobs.append((c[0], c[1], s))
    return jobs


def cmd_score_p04(args) -> int:
    _dev_passed()
    gp = P04_OUT / "gates.json"
    gates = json.loads(gp.read_text(encoding="utf-8")) if gp.is_file() else {}
    jobs = [j for j in p04_jobs() if "|".join(map(str, j)) not in gates]
    print(f"[n4fg-p04] {len(jobs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (j, info) in enumerate(ex.map(_p04_job, jobs), 1):
            gates["|".join(map(str, j))] = info
            P04_OUT.mkdir(parents=True, exist_ok=True)
            tmp = gp.with_suffix(".tmp")
            tmp.write_text(json.dumps(gates, indent=1), encoding="utf-8")
            os.replace(tmp, gp)
            print(f"[n4fg-p04] {n}/{len(jobs)} {j} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def _series_or(name, c, fallback):
    import p04_heldout_score as psc
    v = psc.series(name, c)
    return fallback if v is None else v


def cmd_report_p04(args) -> int:
    import p04_heldout_score as psc
    pft = json.loads((ROOT / "docs/eval/ttbn/ttbn_fulltta_p04_2026-10-11.json").read_text(encoding="utf-8"))["cells"]
    rows, res, worse = [], {}, []
    for c in [psc.VERDICT_CELL] + psc.CORRUPTED:
        tag, name = psc.cell_tag(c), psc.cell_tag(c).replace("__", "/")
        subs = psc.subs_of(c)
        gg = [tgp.gate_g_p04(c[0], c[1], s) for s in subs]
        g4 = [ttbn.gate_p04(c[0], c[1], s) for s in subs]
        rg, r4 = float(np.mean([x.mean() for x in gg])), float(np.mean([x.mean() for x in g4]))
        if all(np.array_equal(a, b) for a, b in zip(gg, g4)):
            res[tag] = {"gate_g": rg, "gate": r4, "equal": True, "minus_n4f": 0.0, "p_ft": pft[tag]}
            rows.append(f"| {name} | {rg:.3f} | {r4:.3f} | = N4F (gates equal on every draw) | 0 |")
            continue
        ft = psc.series("full_tta/fused", c)
        n4f = ft if not any(x.any() for x in g4) else psc.series("full_tta_ttbn/fused", c)
        # draws where the gates are equal: N4Fg's stream is N4F's, and its presort is N4F's
        vals, reps = [], []
        for s, a, b in zip(subs, gg, g4):
            d = psc.PRE / ("full_tta_ttbng/fused" if not np.array_equal(a, b) else
                           ("full_tta_ttbn/fused" if b.any() else "full_tta/fused")) / tag / f"all_{s}"
            vals.append(json.loads((d / "obs.json").read_text())["ap"])
            reps.append(np.load(d / "reps.npy"))
        tg = (float(np.mean(vals)), np.mean(reps, axis=0))
        dl, (lo, hi) = tg[0] - n4f[0], psc.ci(tg[1] - n4f[1])
        res[tag] = {"gate_g": rg, "gate": r4, "equal": False, "full_tta_ttbng": tg[0], "full_tta_ttbn": n4f[0],
                    "minus_n4f": dl, "ci_minus_n4f": [lo, hi]}
        if dl <= FLOOR_H and hi < 0:
            worse.append(tag)
        rows.append(f"| {name} | {rg:.3f} | {r4:.3f} | {tg[0]:.4f} (N4F {n4f[0]:.4f}) | {dl:+.4f} [{lo:+.4f}, {hi:+.4f}] |")
    c = GLARE_CELL
    ft, n4f, tg = psc.series("full_tta/fused", c), psc.series("full_tta_ttbn/fused", c), psc.series("full_tta_ttbng/fused", c)
    un, n4g = psc.series("union/fused", c), psc.series("union_ttbng/fused", c)
    if None in (ft, n4f, tg, un, n4g):
        raise SystemExit("glare/clean presorts or replicates missing; run score-p04 and boot first")
    dl, (lo, hi) = tg[0] - ft[0], psc.ci(tg[1] - ft[1])
    d4, (lo4, hi4) = n4f[0] - ft[0], psc.ci(n4f[1] - ft[1])
    dg, (log_, hig) = tg[0] - n4g[0], psc.ci(tg[1] - n4g[1])
    rg = float(np.mean([tgp.gate_g_p04("glare", None, s).mean() for s in psc.subs_of(c)]))
    r4 = float(np.mean([ttbn.gate_p04("glare", None, s).mean() for s in psc.subs_of(c)]))
    gain = dl >= FLOOR_G and lo > 0
    v = ("REPLICATES" if gain and not worse else
         f"DOES NOT REPLICATE (glare/clean {'gains' if gain else 'does not gain'}; "
         f"worse than N4F: {', '.join(worse) or 'none'})")
    L = ["# N4Fg (N4F with N4g's glare term) on pohang04's glare cell", "",
         "`docs/prereg-ttbn-fulltta-glare-2026-10-12.md`. Ship AP against the VIS ground truth; every arm here is an "
         "ensemble of all five systems, mean over the four draws (VIS v2 glare severity 2, draws 941–944, IR clean). "
         "Δ = `full_tta_ttbng` − `full_tta`, paired; 95% interval from the gate-free scoring's 1,000 moving-block "
         "replicates (L = 20, seed 1).", "",
         f"**P-FG: {v}.**", "",
         "| glare/clean | ship AP | Δ vs full_tta [95%] |", "|---|---:|---:|",
         f"| `full_tta` | {ft[0]:.4f} | |",
         f"| N4F (`full_tta_ttbn`, N4's gate {r4:.1%}) | {n4f[0]:.4f} | {d4:+.4f} [{lo4:+.4f}, {hi4:+.4f}] |",
         f"| **N4Fg** (`full_tta_ttbng`, N4g's gate {rg:.1%}) | **{tg[0]:.4f}** | **{dl:+.4f} [{lo:+.4f}, {hi:+.4f}]** |",
         "", "Descriptive (one pass per stream, P-G):",
         f"* `union` {un[0]:.4f}, N4g (`union_ttbng`) {n4g[0]:.4f}; N4Fg − N4g: {dg:+.4f} [{log_:+.4f}, {hig:+.4f}].", "",
         "## The other pohang04 cells (gate-equality check against P-FT)", "",
         "Where N4g's gate equals N4's on every draw, N4Fg is N4F (P-FT's values). Where it adds frames, N4Fg is built "
         "and scored; a cell fails if N4Fg − N4F ≤ −0.0014 with its interval below zero.", "",
         "| cell (VIS/IR) | N4g gate | N4 gate | N4Fg | N4Fg − N4F [95%] |", "|---|---:|---:|---:|---:|"] + rows
    res["glare__clean"] = {"gate_g": rg, "gate": r4, "full_tta": ft[0], "full_tta_ttbn": n4f[0], "full_tta_ttbng": tg[0],
                           "delta": dl, "ci": [lo, hi], "n4f_delta": d4, "n4f_ci": [lo4, hi4], "union": un[0],
                           "union_ttbng": n4g[0], "minus_n4g": dg, "ci_minus_n4g": [log_, hig]}
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"P-FG": v, "worse": worse, "cells": res}, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("plan-dev")
    b = sp.add_parser("build-dev")
    b.add_argument("--workers", type=int, default=6)
    c = sp.add_parser("check-fp16")
    c.add_argument("--workers", type=int, default=6)
    s = sp.add_parser("score-dev")
    s.add_argument("--workers", type=int, default=3)
    sp.add_parser("boot-dev")
    r = sp.add_parser("report-dev")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_fulltta_glare_dev_2026-10-12.md")
    b = sp.add_parser("build-p04-flip")
    b.add_argument("--workers", type=int, default=8)
    u = sp.add_parser("_flip-unit")
    for a, t, dflt in (("--mod", str, "vis"), ("--cond", str, "glare"), ("--draw", int, 0), ("--workers", int, 8)):
        u.add_argument(a, type=t, default=dflt)
    u.add_argument("--fp16", action="store_true")
    b = sp.add_parser("build-p04-adapted")
    b.add_argument("--workers", type=int, default=6)
    s = sp.add_parser("score-p04")
    s.add_argument("--workers", type=int, default=2)
    r = sp.add_parser("report-p04")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_fulltta_glare_p04_2026-10-12.md")
    args = ap.parse_args()
    return {"plan-dev": cmd_plan_dev, "build-dev": cmd_build_dev, "check-fp16": cmd_check_fp16,
            "score-dev": cmd_score_dev, "boot-dev": cmd_boot_dev, "report-dev": cmd_report_dev,
            "build-p04-flip": cmd_build_p04_flip, "_flip-unit": _flip_unit,
            "build-p04-adapted": cmd_build_p04_adapted, "score-p04": cmd_score_p04,
            "report-p04": cmd_report_p04}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
