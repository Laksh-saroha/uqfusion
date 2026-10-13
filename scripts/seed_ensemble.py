"""Checkpoint ensembles, restoration and the gate-free union, composed (2026-10-10).

`docs/prereg-no-retrain-2026-10-10.md` E1, E4, E5. No detector is retrained.

* **Ensembles (E1).** The five Phase 3 systems are five independent trainings of the same VIS and IR
  detectors. An ensemble stream merges, frame by frame, the records of several checkpoints of one
  modality with plain WBF (IoU 0.55, equal weights, no sigma; the arm the class-set audit found
  positive on ship AP for two checkpoints of one model). IR keeps its shipped detector-side NMS (0.70)
  per checkpoint before the merge.
* **Restoration (E2 streams).** Where the shipped gate's degradation statistic fires on a frame
  (VIS `veil OR dark`, IR `ir_d2 > ir_bound`), a checkpoint's record is replaced by its record on the
  restored frame (`scripts/restore.py build`), before NMS and merging.
* **Gate.** `shipped` = `crossmodal26m` as is; `union` (E4) = the veto switched off, so every frame
  merges both streams with the shipped weights, merge IoU and support term; `lex` (E9, Amendment 4) =
  no merge: every VIS ship box ranks above every IR ship box (`gate_zoo.build_lex_records`), so the
  fused AP cannot fall below the arm's own VIS AP.

The capability prior is recomputed on each arm's own clean streams (it is calibration of the
streams it weighs). Every arm gives five systems paired with `single` by k; under `ens5` the five
share their streams, so the interval is the single systems' spread, not the ensemble's.

    py -3.13 scripts/seed_ensemble.py run --draws 945 946 947 --tag fresh --arms single ens2 ens5 union ens5_union
    py -3.13 scripts/seed_ensemble.py run --draws 945 946 947 --tag fresh --arms full ens5_restore --vis-restore --ir-restore
    py -3.13 scripts/seed_ensemble.py report --tag fresh --draws 945 946 947
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

VIS_CONDS = ("clean", "fog", "fog_s1", "lowlight", "glare")
IR_CONDS = (None, "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")
#: arm -> (members, restored streams, gate, + horizontal-flip view per checkpoint)
ARMS = {"single": ("k", False, "shipped", False),
        "ens2": ("k2", False, "shipped", False),
        "ens5": ("all", False, "shipped", False),
        "union": ("k", False, "union", False),               # E4
        "ens5_union": ("all", False, "union", False),        # E5 decomposition
        "ens5_restore": ("all", True, "shipped", False),     # E5 decomposition
        "full": ("all", True, "union", False),               # E5
        "tta": ("k", False, "shipped", True),                # E8
        "full_tta": ("all", True, "union", True),            # E8 on top of E5
        "ens5_lex": ("all", False, "lex", False),            # E9: tiered concatenation, VIS tier first
        "full_tta_lex": ("all", True, "lex", True)}          # E9 on top of E5 + E8
SEEDS = (0, 1, 2, 3, 4)
ENS_IOU = 0.55
IR_NMS = 0.70
T_975_DF4 = 2.7764451051977987
CACHE, STATS = "runs/cache_p3dev_v2", "runs/derived_p3dev_v2"


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def members(arm: str, k: int) -> tuple[int, ...]:
    m = ARMS[arm][0]
    return {"k": (k,), "k2": (k, (k + 1) % 5), "all": SEEDS}[m]


def merge(recs: list[dict], iou: float = ENS_IOU) -> dict:
    """Plain WBF of one frame's records from several checkpoints (VIS canvas or IR canvas)."""
    from uqfusion.uq.fusion import sigma_weighted_fusion
    if len(recs) == 1:
        return recs[0]
    hw = recs[0].get("image_hw", (640, 640))
    norm = np.array([hw[1], hw[0], hw[1], hw[0]], float)
    bl, sl, ll, gl = [], [], [], []
    for r in recs:
        b = np.asarray(r["boxes_xyxy"], float).reshape(-1, 4)
        bl.append(np.clip(b / norm, 0.0, 1.0).tolist())
        sl.append(np.asarray(r["conf"], float).tolist())
        ll.append(np.asarray(r["cls"], float).tolist())
        gl.append(np.asarray(r.get("sigma_ltrb", np.zeros((len(b), 4))), float).reshape(-1, 4) / norm)
    fb, fs, fl = sigma_weighted_fusion(bl, sl, ll, gl, [1.0] * len(recs), iou_thr=iou, use_sigma=False)
    fb = np.asarray(fb, float).reshape(-1, 4) * norm
    return {**recs[0], "boxes_xyxy": fb, "conf": np.asarray(fs, float),
            "cls": np.asarray(fl).astype(int), "sigma_ltrb": np.zeros((len(fb), 4))}


_CACHE: dict = {}
_DIST: dict = {}


def _memo_distances() -> None:
    """Arms that share a record list (single and union; the ens5 arms) re-score the same frames'
    Mahalanobis distances on every call, a quarter of `run_systems`. Cache them by object identity,
    pinning the list and scorer so an id cannot be reused. Pure cache: outputs are unchanged."""
    import uqfusion.eval.fusion_eval as fe
    if getattr(fe._distances, "_memo", False):
        return
    orig = fe._distances

    def dist(records, scorer):
        hit = _DIST.get((id(records), id(scorer)))
        if hit is not None and hit[0] is records and hit[1] is scorer:
            return hit[2]
        out = orig(records, scorer)
        _DIST[(id(records), id(scorer))] = (records, scorer, out)
        return out
    dist._memo = True
    fe._distances = dist


def _per_ckpt(draw, mod, cond, k, flag):
    """One checkpoint's stream: raw, or raw with restored records where `flag` fires; IR NMS'd."""
    from uqfusion.eval.cache import load_cache
    from uqfusion.eval.irdedup import nms_records
    from restore import rcache_path
    key = ("ck", draw, mod, cond, k, flag is not None)
    if key in _CACHE:
        return _CACHE[key]
    r, _ = load_cache(ROOT / CACHE / f"seed{k}" / sub(draw) / f"gauss_{mod}_paired_{cond or 'clean'}.pkl")
    if flag is not None:
        c = cond or "clean"
        rr, meta = load_cache(rcache_path(k, mod, c, None if c == "clean" else draw))
        assert len(rr) == len(r) and all(Path(a["image_path"]).name == Path(b["image_path"]).name
                                         for a, b in zip(r[:50], rr[:50]))
        r = [b if f else a for a, b, f in zip(r, rr, flag)]
    out = nms_records(r, IR_NMS) if mod == "ir" else r
    _CACHE[key] = out
    return out


def _flip_ckpt(draw, mod, cond, k):
    """One checkpoint's detections on the horizontally flipped (unrestored) frame; IR NMS'd."""
    from uqfusion.eval.cache import load_cache
    from uqfusion.eval.irdedup import nms_records
    from restore import flip_path
    key = ("fl", draw, mod, cond, k)
    if key not in _CACHE:
        c = cond or "clean"
        r, _ = load_cache(flip_path(k, mod, c, None if c == "clean" else draw))
        _CACHE[key] = nms_records(r, IR_NMS) if mod == "ir" else r
    return _CACHE[key]


def _stream(draw, mod, cond, mem, flag, flip=False):
    key = ("st", draw, mod, cond, mem, flag is not None, flip)
    if key not in _CACHE:
        per = [_per_ckpt(draw, mod, cond, k, flag) for k in mem]
        if flip:
            per += [_flip_ckpt(draw, mod, cond, k) for k in mem]
        _CACHE[key] = per[0] if len(per) == 1 else [merge(list(fr)) for fr in zip(*per)]
    return _CACHE[key]


def job(spec):
    """(draw, ir_cond, arms, vis_restore, ir_restore) -> {arm: {vis_cond|k|slice: {vis, ir, fused}}}.
    One context per system k serves every arm."""
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, capability_prior, load_context, run_systems
    from gate_zoo import build_lex_records
    from v2_sensitivity import ship_ap
    draw, ir_cond, arms, vr, irr = spec
    _CACHE.clear()
    _DIST.clear()
    _memo_distances()
    t0 = time.time()
    res = {a: {} for a in arms}
    shared = {}
    d = f"{STATS}/brightness/{sub(draw)}"
    kw = dict(bright_dir=d, structure_dir=f"{STATS}/structure/{sub(draw)}", verbose=False)
    for k in SEEDS:
        ctx = load_context(preset="crossmodal26m", cache_dir=f"{CACHE}/seed{k}/{sub(draw)}",
                           ir_bright=f"{d}/gauss_ir_paired_{ir_cond or 'clean'}.json",
                           conditions=VIS_CONDS, ir_condition=ir_cond, **kw)
        night = np.isin(ctx.runs, NIGHT_RUNS)
        sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
        thr = float(ctx.struct_const["axes"]["grad_gini"]["threshold"])
        vflag = {c: (ctx.gini_by_cond[c] < thr) | (ctx.bright_by_cond[c] < float(ctx.c_vis.mu_b))
                 for c in VIS_CONDS}
        iflag = ctx.ir_d2 > ctx.ir_bound
        if ir_cond is None:
            iflag0 = iflag
        else:
            c0 = load_context(preset="crossmodal26m", cache_dir=f"{CACHE}/seed{k}/{sub(draw)}",
                              ir_bright=f"{d}/gauss_ir_paired_clean.json", conditions=("clean",),
                              ir_condition=None, **kw)
            iflag0 = c0.ir_d2 > c0.ir_bound
        cap0 = (ctx.cap_vis, ctx.cap_ir)
        # An arm on all five checkpoints sees the same streams for every k, and its gate reads image
        # statistics only, so its result repeats across k (checked bit-identical on 945/clean IR for
        # ens5, ens5_union and ens5_lex). Reuse it whenever everything the gate reads is identical.
        h = hashlib.sha1()
        for x in ([np.asarray(vflag[c]) for c in VIS_CONDS] + [np.asarray(iflag), np.asarray(iflag0)]
                  + [np.asarray(ctx.bright_by_cond[c]) for c in VIS_CONDS]
                  + [np.asarray(ctx.gini_by_cond[c]) for c in VIS_CONDS]
                  + [np.asarray(ctx.ir_d2), np.asarray([float(ctx.c_vis.mu_b), float(ctx.ir_bound), float(ctx.cap_ir_scale)])]
                  + ([np.asarray(ctx.ir_night)] if ctx.ir_night is not None else [])):
            h.update(np.ascontiguousarray(x).tobytes())
        sig = h.hexdigest()
        for arm in arms:
            mem = members(arm, k)
            if ARMS[arm][0] == "all" and (arm, sig) in shared:
                for c in VIS_CONDS:
                    for s in sl:
                        res[arm][f"{c}|{k}|{s}"] = dict(shared[(arm, sig)][f"{c}|{s}"])
                continue
            rest = ARMS[arm][1]
            fv = (lambda c: vflag[c]) if (rest and vr) else (lambda c: None)
            fi, fi0 = (iflag, iflag0) if (rest and irr) else (None, None)
            fl = ARMS[arm][3]
            ir = _stream(draw, "ir", ir_cond, mem, fi, fl)
            if arm == "single":
                ctx.cap_vis, ctx.cap_ir = cap0
            else:
                cv, ci = capability_prior(_stream(draw, "vis", "clean", mem, fv("clean"), fl),
                                          _stream(draw, "ir", None, mem, fi0, fl),
                                          ctx.gts, ctx.h_frames, ctx.cap_fit_frames)
                ctx.cap_vis, ctx.cap_ir = cv, ci / ctx.cap_ir_scale
            run_ctx = ctx if ARMS[arm][2] in ("shipped", "lex") else dataclasses.replace(ctx, veto=None)
            for c in VIS_CONDS:
                vis = _stream(draw, "vis", c, mem, fv(c), fl)
                out = run_systems(run_ctx, c, compute_metrics=False, vis_records=vis, ir_records=ir)
                g = ARMS[arm][2]
                fused = (build_lex_records(vis, out["ir_in_vis"], np.zeros(len(vis), bool)) if g == "lex"
                         else out["fused_gated"])
                parts = {"vis": frame_parts(vis, ctx.gts), "ir": frame_parts(out["ir_in_vis"], ctx.gts),
                         "fused": frame_parts(fused, ctx.gts)}
                for s, idx in sl.items():
                    res[arm][f"{c}|{k}|{s}"] = {n: ship_ap(p, idx) for n, p in parts.items()}
                    res[arm][f"{c}|{k}|{s}"]["veto"] = (0.0 if ARMS[arm][2] == "lex"
                                                        else float(np.asarray(out["veto_vis"], bool)[idx].mean()))
            if ARMS[arm][0] == "all":
                shared[(arm, sig)] = {f"{c}|{s}": res[arm][f"{c}|{k}|{s}"] for c in VIS_CONDS for s in sl}
        ctx.cap_vis, ctx.cap_ir = cap0
        _DIST.clear()
        # keep the five-checkpoint merges and the next system's checkpoint; drop the rest
        for key in [q for q in _CACHE if (q[0] == "st" and len(q[4]) < len(SEEDS))
                    or (q[0] in ("ck", "fl") and q[4] != (k + 1) % len(SEEDS))]:
            del _CACHE[key]
    return spec, res, time.time() - t0


def cmd_run(args) -> int:
    out_p = ROOT / f"runs/seed_ens/{args.tag}.json"
    out_p.parent.mkdir(parents=True, exist_ok=True)
    done = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    specs = []
    for d in args.draws:
        for ic in IR_CONDS:
            todo = tuple(a for a in args.arms if f"{d}|{ic}|{a}" not in done)
            if todo:
                specs.append((d, ic, todo, args.vis_restore, args.ir_restore))
    print(f"[ens] {len(specs)} jobs, arms {args.arms}", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for n, (spec, res, dt) in enumerate(ex.map(job, specs), 1):
            for a, r in res.items():
                done[f"{spec[0]}|{spec[1]}|{a}"] = r
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(done), encoding="utf-8")
            os.replace(tmp, out_p)
            print(f"[ens] {n}/{len(specs)} {spec[:2]} {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    return 0


def table(tag: str, draws) -> dict:
    """{(arm, vis_cond, ir_cond, slice): {stream: per-system draw-mean array (5,)}}"""
    raw = json.loads((ROOT / f"runs/seed_ens/{tag}.json").read_text(encoding="utf-8"))
    out = {}
    for a in ARMS:
        for ic in IR_CONDS:
            for c in VIS_CONDS:
                for s in ("day", "night"):
                    rows = [raw.get(f"{d}|{ic}|{a}") for d in draws]
                    if any(r is None for r in rows):
                        continue
                    m = np.mean(np.asarray([[[r[f"{c}|{k}|{s}"][n] for n in ("vis", "ir", "fused", "veto")]
                                             for k in SEEDS] for r in rows], float), axis=0)
                    out[(a, c, ic, s)] = {"vis": m[:, 0], "ir": m[:, 1], "fused": m[:, 2], "veto": m[:, 3]}
    return out


def ci(x) -> tuple[float, float, float]:
    x = np.asarray(x, float)
    m, h = float(x.mean()), T_975_DF4 * float(x.std(ddof=1)) / np.sqrt(len(x))
    return m, m - h, m + h


def cmd_report(args) -> int:
    T = table(args.tag, args.draws)
    f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"   # noqa: E731
    fail = lambda x: x[0] <= -0.0060 and x[2] < 0              # noqa: E731
    lines = [f"# Ensembles, union and restoration composed, `{args.tag}` (draws {', '.join(map(str, args.draws))})",
             "", "`scripts/seed_ensemble.py`, registered in `docs/prereg-no-retrain-2026-10-10.md` (E1, E4, E5). "
             "Ship AP; per system the draw mean; Δ = arm − single (shipped) for the same system k, mean and 95% "
             "t-interval over the five systems (df 4). IMPROVES = no cell worse at 0.0014 with the interval below "
             "zero, and at least one better at 0.0060 with the interval above zero. Claim = fused − max(VIS, IR) "
             "within the arm, failing at ≤ −0.0060 with the interval below zero.", ""]
    summ = {}
    head = ["| arm | IMPROVES | better ≥0.0060 | better ≥0.0014 | worse ≤−0.0014 | worse ≤−0.0060 | worst Δ fused (cell) "
            "| claim fails | mean fused, 60 cells |", "|---|---|---:|---:|---:|---:|---|---:|---|"]
    detail = []
    base_mean = None
    for a in ARMS:
        if a == "single" or not any(key[0] == a for key in T):
            continue
        cnt = dict(b60=0, b14=0, w14=0, w60=0, fails=0)
        worst, means, rows = None, [], []
        for ic in IR_CONDS:
            for c in VIS_CONDS:
                for s in ("day", "night"):
                    b, e = T.get(("single", c, ic, s)), T.get((a, c, ic, s))
                    if b is None or e is None:
                        continue
                    dv, di, dfu = (ci(e[n] - b[n]) for n in ("vis", "ir", "fused"))
                    ce = ci(e["fused"] - np.maximum(e["vis"], e["ir"]))
                    cnt["b60"] += dfu[0] >= 0.0060 and dfu[1] > 0
                    cnt["b14"] += dfu[0] >= 0.0014 and dfu[1] > 0
                    cnt["w14"] += dfu[0] <= -0.0014 and dfu[2] < 0
                    cnt["w60"] += dfu[0] <= -0.0060 and dfu[2] < 0
                    cnt["fails"] += fail(ce)
                    means.append((float(b["fused"].mean()), float(e["fused"].mean())))
                    cell = f"{c}/{ic or 'clean'}/{s}"
                    if worst is None or dfu[0] < worst[0][0]:
                        worst = (dfu, cell)
                    summ[f"{a}|{c}|{ic}|{s}"] = {"single": {n: float(b[n].mean()) for n in ("vis", "ir", "fused", "veto")},
                                                 "arm": {n: float(e[n].mean()) for n in ("vis", "ir", "fused", "veto")},
                                                 "d_vis": dv, "d_ir": di, "d_fused": dfu, "claim_arm": ce}
                    rows.append(f"| {c} | {ic or 'clean'} | {s} | {b['fused'].mean():.4f} | {e['fused'].mean():.4f} | "
                                f"{f(dfu)} | {f(dv)} | {f(di)} | {e['veto'].mean():.0%} | {ce[0]:+.4f}{' FAIL' if fail(ce) else ''} |")
        imp = cnt["w14"] == 0 and cnt["b60"] >= 1
        mb, ma = np.mean([m[0] for m in means]), np.mean([m[1] for m in means])
        base_mean = mb
        summ[f"{a}|summary"] = {**{k: int(v) for k, v in cnt.items()}, "IMPROVES": bool(imp), "cells": len(means),
                                "mean_single": float(mb), "mean_arm": float(ma)}
        head.append(f"| {a} | {'**yes**' if imp else 'no'} | {cnt['b60']} | {cnt['b14']} | {cnt['w14']} | {cnt['w60']} | "
                    f"{worst[0][0]:+.4f} ({worst[1]}) | {cnt['fails']} | {mb:.4f} → {ma:.4f} |")
        detail += [f"## {a}", "", "| VIS | IR | slice | single fused | arm fused | Δ fused | Δ VIS | Δ IR | arm veto | claim (arm) |",
                   "|---|---|---|---:|---:|---|---|---|---:|---|", *rows, ""]
    sf = sum(fail(ci(T[("single", c, ic, s)]["fused"] - np.maximum(T[("single", c, ic, s)]["vis"], T[("single", c, ic, s)]["ir"])))
             for ic in IR_CONDS for c in VIS_CONDS for s in ("day", "night") if ("single", c, ic, s) in T)
    lines += head + ["", f"Claim fails under shipped (`single`): {sf} of "
                     f"{sum(1 for key in T if key[0] == 'single')} cells.", ""] + detail
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("\n".join(head))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run")
    r.add_argument("--draws", type=int, nargs="+", required=True)
    r.add_argument("--arms", nargs="+", default=["single", "ens2", "ens5", "union", "ens5_union"], choices=list(ARMS))
    r.add_argument("--tag", required=True)
    r.add_argument("--workers", type=int, default=6)
    r.add_argument("--vis-restore", action="store_true")
    r.add_argument("--ir-restore", action="store_true")
    p = sp.add_parser("report")
    p.add_argument("--tag", required=True)
    p.add_argument("--draws", type=int, nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.cmd == "report" and args.out is None:
        args.out = f"docs/eval/no_retrain/ensembles_{args.tag}.md"
    return {"run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
