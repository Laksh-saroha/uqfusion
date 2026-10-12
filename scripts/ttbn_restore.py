"""N4ae (`docs/prereg-ttbn-restore-2026-10-11.md`): N4, except that gated frames that are dark (mean linear luminance
below 0.18, where auto-exposure acts) get the auto-exposed frame's detections instead of N4's adapted ones.
Descriptive development arm `sel`: gated frames that are not dark get the dark-channel-dehazed frame's detections.

    py -3.13 scripts/ttbn_restore.py build --target dev     # GPU: ae on dark, dcp on the rest, N4's gated frames
    py -3.13 scripts/ttbn_restore.py score-dev
    py -3.13 scripts/ttbn_restore.py report-dev
    py -3.13 scripts/ttbn_restore.py build --target p04     # only after a development pass: ae on dark only
    py -3.13 scripts/ttbn_restore.py score-p04
    py -3.13 scripts/ttbn_restore.py report-p04

Restored frames are replayed exactly as N4's builds replay the corrupted ones (`ttbn._init`: v2, seed = draw),
restored with `restore.restore` and detected by the same Phase 3 checkpoints, fp16 autocast, batch 8.
"""
from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]
import ttbn                                                    # noqa: E402

OUT = {"dev": ROOT / "runs/novel/ttbn_restore", "p04": ROOT / "runs/holdout_p04/ttbn_restore"}
DARK = 0.18                       # auto-exposure's target mean linear luminance: below it, ae changes the frame
DEV_REPORT = ROOT / "docs/eval/ttbn/ttbn_restore_dev_2026-10-11.json"
P04_PRESORTS = ("union_n4ae/fused", "n4ae/vis")
_W: dict = {}


def rec_path(target: str, k: int, cond: str, draw) -> Path:
    where = "clean" if draw is None else ttbn.sub(draw)
    return OUT[target] / f"seed{k}" / where / f"gauss_vis_paired_{cond}_restored_fp16.pkl"


def dark_path(target: str, cond: str, draw) -> Path:
    return OUT[target] / "dark" / f"{cond}_{draw or 'clean'}.json"


def dark_flags(target: str, cond: str, draw, n: int) -> np.ndarray:
    """Per listed frame: dark (only frames the build visited carry a value; the rest are False)."""
    out = np.zeros(n, bool)
    p = dark_path(target, cond, draw)
    if p.is_file():
        for i, v in json.loads(p.read_text(encoding="utf-8"))["lum"].items():
            out[int(i)] = v < DARK
    return out


# ------------------------------------------------------------------ frames (in order, bounded)
def _init(spec):
    import cv2
    cv2.setNumThreads(1)
    ttbn._low()
    from restore import _rows
    from uqfusion.eval.corruptions import make_corruption
    _W.update(spec)
    _W["rows"] = _rows("vis")
    _W["tf"] = (make_corruption(spec["kind"], spec["sev"], spec["seed"], version="v2", modality="vis",
                                images=spec["images"]) if spec.get("kind") else None)


def _one(i: int):
    import cv2
    from restore import gate_feats, restore
    im = cv2.imread(_W["images"][i])
    if im is None:
        raise FileNotFoundError(_W["images"][i])
    if _W["tf"] is not None:
        im = _W["tf"](im, i)
    lum = float(gate_feats(im, "vis", _W["rows"])["lum_lin"])
    m = "ae" if lum < DARK else _W["other"]
    return i, lum, m, (restore(im, m, "vis", _W["rows"]) if m else None)


def _chunk(idx):
    return [_one(i) for i in idx]


def _frames(spec, idx, workers, chunk=4):
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


def _dev_passed() -> None:
    v = json.loads(DEV_REPORT.read_text(encoding="utf-8")).get("N4ae") if DEV_REPORT.is_file() else None
    if v != "PASSES":
        raise SystemExit(f"N4ae's pohang04 test runs only after a development pass (verdict: {v})")


def cmd_build(args) -> int:
    import torch
    import p04_heldout as ph
    from uqfusion.eval.corruptions_v2 import CODE_REV
    from uqfusion.uq.infer import UQPredictor
    t = args.target
    if t == "p04":
        _dev_passed()
    ttbn._low()
    conds = ttbn.DEV_CONDS if t == "dev" else ttbn.P04_CONDS
    draws = ttbn.DEV_DRAWS if t == "dev" else ph.VIS_DRAWS
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()] if t == "dev" else ph.images("vis")
    todo = [(c, d) for c in args.conds for d in ([None] if c == "clean" else draws)
            if not all(rec_path(t, k, c, d).is_file() for k in ttbn.SEEDS)]
    print(f"[n4ae] {t}: {len(lst)} frames, {len(todo)} streams to build", flush=True)
    if not todo:
        return 0
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=0, imgsz=640,
                         conf=0.001) for k in ttbn.SEEDS]
    ph.setup_fp16(preds)
    for c, d in todo:
        kc = conds[c]
        need = ttbn.needed(t, c, d)
        assert len(need) == len(lst)
        idx = np.flatnonzero(need).tolist()
        recs = [[None] * len(lst) for _ in ttbn.SEEDS]
        lum, meth, buf = {}, {}, []
        t0 = time.time()

        def flush():
            if buf:
                with torch.no_grad(), ph.amp(True):
                    out = ph._multi_batch(preds, [im for _, im in buf])
                for j in range(len(ttbn.SEEDS)):
                    for (i, _im), r in zip(buf, out[j]):
                        recs[j][i] = {**ph._f32(r), "image_path": lst[i]}
                buf.clear()

        if idx:
            spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": d or 0, "images": lst,
                    "other": "dcp" if t == "dev" else None}
            for n_done, (i, lu, m, im) in enumerate(_frames(spec, idx, args.workers), 1):
                lum[i], meth[i] = lu, m
                if im is not None:
                    buf.append((i, im))
                    if len(buf) >= ttbn.BATCH:
                        flush()
                if n_done % 2000 == 0:
                    print(f"[n4ae] {c}/{d} {n_done}/{len(idx)} frames ({time.time() - t0:.0f}s)", flush=True)
            flush()
        dp = dark_path(t, c, d)
        dp.parent.mkdir(parents=True, exist_ok=True)
        dp.write_text(json.dumps({"threshold": DARK, "lum": {str(i): v for i, v in lum.items()},
                                  "method": {str(i): v for i, v in meth.items()}}), encoding="utf-8")
        n_dark = sum(v < DARK for v in lum.values())
        for j, k in enumerate(ttbn.SEEDS):
            o = rec_path(t, k, c, d)
            o.parent.mkdir(parents=True, exist_ok=True)
            meta = {"target": t, "cond": c, "draw": d, "needed_frames": len(idx), "dark_frames": int(n_dark),
                    "methods": "ae where dark" + (", dcp elsewhere (descriptive arm)" if t == "dev" else " only"),
                    "precision": "fp16 autocast, channels_last", "batch": ttbn.BATCH,
                    "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                    "corrupt_seed": d if kc else None, "corrupt_version": "v2" if kc else None,
                    "corrupt_code": CODE_REV if kc else None,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": meta, "records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[n4ae] {c}/{d} done: {len(idx)} needed frames, {n_dark} dark, {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ streams
def _put(rec: dict, plain: dict) -> dict:
    assert Path(rec["image_path"]).name == Path(plain["image_path"]).name
    return {**rec, "image_path": plain["image_path"], "feat": plain["feat"]}


def n4ae_stream(target, k, cond, draw, plain, n4, gate, dark):
    """Gated and dark: the auto-exposed frame's record; gated and not dark: N4's; elsewhere: plain (`n4`
    already holds the plain records off the gate)."""
    use = np.asarray(gate, bool) & np.asarray(dark, bool)
    if not use.any():
        return n4
    with open(rec_path(target, k, cond, draw), "rb") as fh:
        rr = pickle.load(fh)["records"]
    out = list(n4)
    for i in np.flatnonzero(use):
        if rr[i] is None:
            raise SystemExit(f"dark gated frame without a restored record ({target}, {k}, {cond}, {draw}, {i})")
        out[i] = _put(rr[i], plain[i])
    return out


# ------------------------------------------------------------------ development
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
        g = ttbn.gate_ctx(ctx, c)
        vis = ctx.vis_by_cond[c]
        dd = None if c == "clean" else draw
        dk = dark_flags("dev", c, dd, len(vis))
        n4 = ttbn.gated_stream("dev", k, c, dd, vis, g)
        ae = n4ae_stream("dev", k, c, dd, vis, n4, g, dk)
        if g.any() and (g & ~dk).any():
            with open(rec_path("dev", k, c, dd), "rb") as fh:
                rr = pickle.load(fh)["records"]
            sel = [_put(rr[i], b) if (g[i] and not dk[i]) else a for i, (a, b) in enumerate(zip(ae, vis))]
        else:
            sel = ae
        recs = {"union": run_systems(u, c, compute_metrics=False)["fused_gated"]}
        recs["union_ttbn"] = recs["union"] if n4 is vis else run_systems(u, c, compute_metrics=False, vis_records=n4)["fused_gated"]
        recs["union_n4ae"] = recs["union_ttbn"] if ae is n4 else run_systems(u, c, compute_metrics=False, vis_records=ae)["fused_gated"]
        recs["union_sel"] = recs["union_n4ae"] if sel is ae else run_systems(u, c, compute_metrics=False, vis_records=sel)["fused_gated"]
        parts = {n: frame_parts(r, ctx.gts) for n, r in recs.items()}
        for s, idx in sl.items():
            res[f"{c}|{s}"] = {n: ship_ap(p, idx) for n, p in parts.items()}
            res[f"{c}|{s}"]["gate"] = float(g[idx].mean())
            res[f"{c}|{s}"]["dark_of_gated"] = float(dk[idx][g[idx]].mean()) if g[idx].any() else 0.0
    return spec, res


def cmd_score_dev(args) -> int:
    out_p = OUT["dev"] / "score_dev.json"
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
    res = json.loads((OUT["dev"] / "score_dev.json").read_text(encoding="utf-8"))
    out, rows, worse, better = {}, [], [], []
    for c in ttbn.DEV_CONDS:
        for ic in ttbn.DEV_IR_CONDS:
            for s in ("day", "night"):
                pk = {n: [] for n in ("a", "a4", "s", "s4", "gate", "dark")}
                for k in ttbn.SEEDS:
                    rs = [res[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in ttbn.DEV_DRAWS]
                    pk["a"].append(np.mean([r["union_n4ae"] - r["union"] for r in rs]))
                    pk["a4"].append(np.mean([r["union_n4ae"] - r["union_ttbn"] for r in rs]))
                    pk["s"].append(np.mean([r["union_sel"] - r["union"] for r in rs]))
                    pk["s4"].append(np.mean([r["union_sel"] - r["union_ttbn"] for r in rs]))
                    pk["gate"].append(np.mean([r["gate"] for r in rs]))
                    pk["dark"].append(np.mean([r["dark_of_gated"] for r in rs]))
                st = {n: ttbn._cell_stats(pk[n]) for n in ("a", "a4", "s", "s4")}
                cell = f"{c}/{ic or 'clean'}/{s}"
                out[cell] = {"n4ae": list(st["a"]), "n4ae_minus_n4": list(st["a4"]), "sel": list(st["s"]),
                             "sel_minus_n4": list(st["s4"]), "gate": float(np.mean(pk["gate"])),
                             "dark_of_gated": float(np.mean(pk["dark"]))}
                a, alo, ahi = st["a"]
                if a <= -0.0014 and ahi < 0:
                    worse.append(cell)
                if a >= 0.0060 and alo > 0:
                    better.append(cell)
                f = lambda x: f"{x[0]:+.4f} [{x[1]:+.4f}, {x[2]:+.4f}]"            # noqa: E731
                rows.append(f"| {cell} | {out[cell]['gate']:.3f} | {out[cell]['dark_of_gated']:.3f} | {f(st['a'])} | "
                            f"{f(st['a4'])} | {f(st['s'])} | {f(st['s4'])} |")
    v = "PASSES" if (not worse and better) else "FAILS"
    L = ["# N4ae (IR-vouched auto-exposure on dark gated frames) on development", "",
         "`docs/prereg-ttbn-restore-2026-10-11.md`. Draws VIS 942–944 / IR 952–954, five systems. Δ against `union` "
         "(ship AP), mean over the three draws per system, then over the five systems; 95% t-interval (df 4). "
         "`− N4` columns are against `union_ttbn`, recomputed in the same job. `sel` (dehaze on gated frames that "
         "are not dark) is descriptive.", "",
         f"**N4ae {v} development.** A-H: {'PASS' if not worse else 'FAIL (' + ', '.join(worse) + ')'}. "
         f"A-G: {'PASS (' + str(len(better)) + ' cells: ' + ', '.join(better) + ')' if better else 'FAIL'}.", "",
         "| cell (VIS/IR/slice) | gate | dark share of gated | N4ae Δ vs union [95%] | N4ae − N4 [95%] | "
         "sel Δ vs union [95%] | sel − N4 [95%] |", "|---|---:|---:|---:|---:|---:|---:|"] + rows
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"N4ae": v, "worse": worse, "better": better, "cells": out},
                                                 indent=1), encoding="utf-8")
    print("\n".join(L[:6]))
    return 0


# ------------------------------------------------------------------ pohang04 (P-A, only after a development pass)
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
    draw = None if s == "clean" else int(s[4:7])
    vd = None if vc == "clean" else draw
    vis = ctx.vis_by_cond[vc]
    dk = dark_flags("p04", vc, vd, len(vis))
    n4 = ttbn.gated_stream("p04", k, vc, vd, vis, g, True)
    ae = n4ae_stream("p04", k, vc, vd, vis, n4, g, dk)
    out = run_systems(dataclasses.replace(ctx, veto=None), vc, compute_metrics=False, vis_records=ae)
    info = {"gate": float(g.mean()), "dark_gated": int((g & dk).sum())}
    for name, recs in (("union_n4ae/fused", out["fused_gated"]), ("n4ae/vis", ae)):
        d = psc.PRE / name / tag / key
        if not (d / "obs.json").is_file():
            info[name] = psc.save_pre(frame_parts(recs, ctx.gts), d)
    return job, info


def _p04_jobs():
    import p04_heldout_score as psc
    jobs, waiting = [], 0
    for vc, ic in [psc.VERDICT_CELL] + psc.CORRUPTED:
        for s in psc.subs_of((vc, ic)):
            if not ttbn.gate_p04(vc, ic, s).any():
                continue
            vd = None if (vc == "clean" or s == "clean") else int(s[4:7])
            ready = dark_path("p04", vc, vd).is_file() and all(rec_path("p04", k, vc, vd).is_file() for k in ttbn.SEEDS)
            for k in ttbn.SEEDS:
                tag, pk = psc.cell_tag((vc, ic)), f"seed{k}_{s}"
                if all((psc.PRE / n / tag / pk / "obs.json").is_file() for n in P04_PRESORTS):
                    continue
                if ready:
                    jobs.append((k, s, ic, vc))
                else:
                    waiting += 1
    return jobs, waiting


def cmd_score_p04(args) -> int:
    _dev_passed()
    jobs, waiting = _p04_jobs()
    print(f"[n4ae-p04] {len(jobs)} jobs, {waiting} waiting for the build", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (j, info) in enumerate(ex.map(_p04_job, jobs), 1):
            print(f"[n4ae-p04] {n}/{len(jobs)} {j} {info} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_report_p04(args) -> int:
    import p04_heldout_score as psc
    L = ["# N4ae on pohang04 (P-A)", "",
         "`docs/prereg-ttbn-restore-2026-10-11.md`. P-T's cells, statistic and replicates: ship AP against the VIS "
         "ground truth, mean over the five systems and the four draws of a corrupted cell; Δ paired; 95% interval "
         "from the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1).", "",
         "| cell (VIS/IR) | gate | dark gated frames | union | N4ae Δ vs union [95%] | N4ae − N4 [95%] |",
         "|---|---:|---:|---:|---:|---:|"]
    res, worse, better, complete = {}, [], [], True
    for c in [psc.VERDICT_CELL] + psc.CORRUPTED:
        tag = psc.cell_tag(c)
        subs = psc.subs_of(c)
        gate = float(np.mean([ttbn.gate_p04(c[0], c[1], s).mean() for s in subs]))
        un = psc.series("union/fused", c)
        if gate == 0.0:
            res[tag] = {"gate": 0.0, "delta": 0.0, "note": "the gate fires on no frame: N4ae is union"}
            L.append(f"| {tag.replace('__', '/')} | 0 | 0 | {'' if un is None else f'{un[0]:.4f}'} | 0 (identical) | |")
            continue
        nd = 0
        for s in subs:
            vd = None if (c[0] == "clean" or s == "clean") else int(s[4:7])
            g = ttbn.gate_p04(c[0], c[1], s)
            nd += int((g & dark_flags("p04", c[0], vd, len(g))).sum())
        a, t4 = psc.series("union_n4ae/fused", c), psc.series("union_ttbn/fused", c)
        if None in (a, un, t4):
            L.append(f"| {tag.replace('__', '/')} | {gate:.3f} | {nd} | pending | | |")
            complete = False
            continue
        d, (lo, hi) = a[0] - un[0], psc.ci(a[1] - un[1])
        d4, (lo4, hi4) = a[0] - t4[0], psc.ci(a[1] - t4[1])
        res[tag] = {"gate": gate, "dark_gated": nd, "union": un[0], "union_n4ae": a[0], "delta": d, "ci": [lo, hi],
                    "minus_n4": d4, "ci_minus_n4": [lo4, hi4]}
        if d <= -0.0014 and hi < 0:
            worse.append(tag)
        if d >= 0.0060 and lo > 0:
            better.append(tag)
        L.append(f"| {tag.replace('__', '/')} | {gate:.3f} | {nd} | {un[0]:.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                 f"{d4:+.4f} [{lo4:+.4f}, {hi4:+.4f}] |")
    v = ("PENDING" if not complete else "REPLICATES" if (not worse and better) else
         f"DOES NOT REPLICATE (worse: {', '.join(worse) or 'none'}; better: {', '.join(better) or 'none'})")
    L[3:3] = ["", f"**P-A: {v}.**"]
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"P-A": v, "cells": res}, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    b = sp.add_parser("build")
    b.add_argument("--target", choices=["dev", "p04"], required=True)
    b.add_argument("--conds", nargs="+", default=None)
    b.add_argument("--workers", type=int, default=4)
    s = sp.add_parser("score-dev")
    s.add_argument("--workers", type=int, default=3)
    r = sp.add_parser("report-dev")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_restore_dev_2026-10-11.md")
    s = sp.add_parser("score-p04")
    s.add_argument("--workers", type=int, default=2)
    r = sp.add_parser("report-p04")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_restore_p04_2026-10-11.md")
    args = ap.parse_args()
    if args.cmd == "build" and args.conds is None:
        args.conds = (["fog", "fog_s1", "lowlight", "glare", "clean"] if args.target == "dev" else
                      ["lowlight", "noise_s2", "fog", "rain_s2", "blur_s3", "clean"])
    return {"build": cmd_build, "score-dev": cmd_score_dev, "report-dev": cmd_report_dev,
            "score-p04": cmd_score_p04, "report-p04": cmd_report_p04}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
