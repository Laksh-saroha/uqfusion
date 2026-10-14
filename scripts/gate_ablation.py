"""Gate ablation for IG-BN (N4), 2026-10-14 (`docs/prereg-gate-ablation-2026-10-14.md`).

Which part of IG-BN's gate does the work: BN-mix itself, the visible health term, the thermal day term,
or would a clock do? Six arms share N4's adaptation (BN-mix, N = 16, non-overlapping 8-frame blocks of the
stream's frame list) and differ only in which frames take the adapted record:

    ungated     every frame
    qvis        q_vis < 1
    irday       ir_p05 <= 41.5 AND ir_d2 <= 222.7
    solar       solar elevation > 0 deg (NOAA, stereo capture time, nearest valid GPS fix)
    qvis_solar  q_vis < 1 AND solar elevation > 0 deg
    n4          q_vis < 1 AND ir_p05 <= 41.5 AND ir_d2 <= 222.7   (IG-BN, as registered)

A block's statistics depend only on its own eight frames, so every arm reads one *complete* adapted stream
per (VIS condition, draw): N4's stored records for the blocks N4 ran, plus records built here for the
blocks it did not run (`build`). N4's arm therefore reproduces N4 exactly (asserted on development).

    py -3.13 scripts/gate_ablation.py rates                       # label-free gate rates (no detection)
    py -3.13 scripts/gate_ablation.py repro --target dev          # rebuild stored blocks, compare
    py -3.13 scripts/gate_ablation.py build --target dev
    py -3.13 scripts/gate_ablation.py score-dev && py -3.13 scripts/gate_ablation.py report-dev
    py -3.13 scripts/gate_ablation.py build --target p04
    py -3.13 scripts/gate_ablation.py score-p04 && py -3.13 scripts/gate_ablation.py boot-p04
    py -3.13 scripts/gate_ablation.py report-p04
    py -3.13 scripts/gate_ablation.py night-stats && py -3.13 scripts/gate_ablation.py night-rates

Large outputs (complement records, pohang04 presorts) live in `runs/gate_ablation/` (a junction to C:).
"""

from __future__ import annotations

import argparse
import json
import os
import pickle
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import ttbn  # noqa: E402

ARMS = ("ungated", "qvis", "irday", "solar", "qvis_solar", "n4")
OUT = ROOT / "runs/gate_ablation"
SOLAR_JSON = OUT / "solar_elevation.json"
DS = ROOT / "Pohang_dataset"
NIGHT_LIST = ROOT / "runs/night_p01test/pairs_vis.txt"
NIGHT_IR_LIST = ROOT / "runs/night_p01test/pairs_ir.txt"
NIGHT_DERIVED = ROOT / "runs/night_p01test/derived"
NIGHT_IR_CONDS = {"fog_s1": ("fog", 1), "fog_s2": ("fog", 2), "fog_s3": ("fog", 3), "noise_s2": ("noise", 2),
                  "noise_s3": ("noise", 3)}
NIGHT_IR_DRAWS = (952, 953, 954)
T_975_DF4 = ttbn.T_975_DF4


# ------------------------------------------------------------------ solar elevation (A9.4's rule)
def _solar_for(paths: list[str]) -> np.ndarray:
    import holdout_p04_step1 as s1
    el, cache = [], {}
    for p in paths:
        p = Path(p)
        run = p.parent.name
        if run not in cache:
            m = DS / "meta" / run
            cache[run] = (s1.load_ts(m / "timestamps" / "stereo.txt"), *s1.load_gps(m / "navigation" / "gps.txt"))
        st, gt, gp = cache[run]
        t = st[p.stem.split("_")[-1]]
        el.append(s1.solar_elevation(t, *gp[s1.nearest(gt, t)]))
    return np.asarray(el, float)


def solar() -> dict:
    """Per-frame solar elevation of the development list, pohang04's VIS list and the pohang01 night pairs."""
    if SOLAR_JSON.is_file():
        return json.loads(SOLAR_JSON.read_text(encoding="utf-8"))
    import holdout_p04_step1 as s1
    import p04_heldout as ph
    san = {r: s1.run_elevation(r)[1] for r in ("pohang00", "pohang01")}
    if not san["pohang00"] > 0 > san["pohang01"]:
        raise SystemExit("solar code does not call pohang00 day and pohang01 night")
    out = {"rule": "solar elevation > 0 deg (NOAA, geometric), stereo capture time, nearest valid GPS fix "
                   "(holdout_p04_step1.solar_elevation)", "sanity_first_frame": san,
           "dev": _solar_for([ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]).tolist(),
           "p04": _solar_for(ph.images("vis")).tolist(),
           "night": _solar_for(ttbn_read(NIGHT_LIST)).tolist()}
    OUT.mkdir(parents=True, exist_ok=True)
    SOLAR_JSON.write_text(json.dumps(out), encoding="utf-8")
    return out


def ttbn_read(p: Path) -> list[str]:
    return [ln.strip() for ln in Path(p).read_text(encoding="utf-8").splitlines() if ln.strip()]


def sun_up(target: str) -> np.ndarray:
    return np.asarray(solar()[target], float) > 0.0


# ------------------------------------------------------------------ gates (frame statistics, seed-free)
def gates_from(q: np.ndarray, irok: np.ndarray, sun: np.ndarray) -> dict:
    return {"ungated": np.ones_like(q, bool), "qvis": q, "irday": irok, "solar": sun,
            "qvis_solar": q & sun, "n4": q & irok}


def gates_dev(draw: int, cond: str, ic) -> dict:
    q = ttbn.vis_unhealthy(ttbn._dev_stats(draw, "vis", cond, "structure"))
    irok = ttbn.ir_day_ok(ttbn._dev_stats(draw, "ir", ic, "brightness"), ttbn._dev_stats(draw, "ir", ic, "structure"))
    return gates_from(q, irok, sun_up("dev"))


def gates_p04(vc: str, ic, s: str) -> dict:
    draw = None if s == "clean" else int(s[4:7])
    q = ttbn.vis_unhealthy(ttbn._p04_stats("vis", vc, draw, "structure"))
    irok = ttbn.ir_day_ok(ttbn._p04_stats("ir", ic, draw, "brightness"), ttbn._p04_stats("ir", ic, draw, "structure"))
    return gates_from(q, irok, sun_up("p04"))


def cmd_rates(args) -> int:
    """Label-free: the share of frames each arm adapts, development (day/night by run) and pohang04."""
    from uqfusion.eval.ctx import NIGHT_RUNS
    import p04_heldout_score as psc
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    night = np.isin([Path(p).parent.name for p in lst], NIGHT_RUNS)
    el = np.asarray(solar()["dev"])
    print(f"[solar] development: day frames elevation {el[~night].min():+.1f}..{el[~night].max():+.1f} deg, "
          f"night {el[night].min():+.1f}..{el[night].max():+.1f}; solar day == run day on "
          f"{np.mean((el > 0) == ~night):.4f} of frames")
    pe = np.asarray(solar()["p04"])
    print(f"[solar] pohang04: elevation {pe.min():+.1f}..{pe.max():+.1f} deg, sun up on {np.mean(pe > 0):.4f}")
    ne = np.asarray(solar()["night"])
    print(f"[solar] pohang01 night pairs: elevation {ne.min():+.1f}..{ne.max():+.1f} deg, sun up on {np.mean(ne > 0):.4f}")
    out = {"dev": {}, "p04": {}}
    for c in ttbn.DEV_CONDS:
        for ic in ttbn.DEV_IR_CONDS:
            for sl, m in (("day", ~night), ("night", night)):
                r = {a: float(np.mean([gates_dev(d, c, ic)[a][m].mean() for d in ttbn.DEV_DRAWS])) for a in ARMS}
                out["dev"][f"{c}/{ic or 'clean'}/{sl}"] = r
    for vc, ic in [psc.VERDICT_CELL] + psc.CORRUPTED:
        r = {a: float(np.mean([gates_p04(vc, ic, s)[a].mean() for s in psc.subs_of((vc, ic))])) for a in ARMS}
        out["p04"][psc.cell_tag((vc, ic))] = r
    (OUT / "rates.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    for t in out:
        print(f"\n{t}: " + " ".join(f"{a:>10}" for a in ARMS))
        for cell, r in out[t].items():
            print(f"{cell:28s} " + " ".join(f"{r[a]:10.3f}" for a in ARMS))
    return 0


# ------------------------------------------------------------------ complete adapted streams
def comp_path(target: str, k: int, cond: str, draw) -> Path:
    where = "clean" if cond == "clean" else ttbn.sub(draw)
    fp = "_fp16" if target == "p04" else ""
    return OUT / target / f"seed{k}" / where / f"gauss_vis_paired_{cond}_bn16{fp}_complement.pkl"


def _stored(target: str, k: int, cond: str, draw):
    p = ttbn.out_path(target, k, cond, draw, target == "p04")
    if not p.is_file():
        return None
    with open(p, "rb") as fh:
        return pickle.load(fh)["records"]


def streams(target: str) -> list[tuple[str, int | None]]:
    import p04_heldout as ph
    conds, draws = (ttbn.DEV_CONDS, ttbn.DEV_DRAWS) if target == "dev" else (ttbn.P04_CONDS, ph.VIS_DRAWS)
    return [(c, d) for c in conds for d in ([None] if c == "clean" else draws)]


def full_stream(target: str, k: int, cond: str, draw) -> list[dict]:
    """N4's stored adapted records where present, the complement's elsewhere; every frame has one."""
    a = _stored(target, k, cond, draw)
    with open(comp_path(target, k, cond, draw), "rb") as fh:
        b = pickle.load(fh)["records"]
    if a is None:
        a = [None] * len(b)
    assert len(a) == len(b)
    out = [x if x is not None else y for x, y in zip(a, b)]
    if any(r is None for r in out):
        raise SystemExit(f"incomplete adapted stream ({target}, {k}, {cond}, {draw})")
    return out


def _preds(fp16: bool):
    import p04_heldout as ph
    from uqfusion.uq.infer import UQPredictor
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=0, imgsz=640,
                         conf=0.001) for k in ttbn.SEEDS]
    nb = [ttbn.patch_bn(p.model) for p in preds]
    if fp16:
        ph.setup_fp16(preds)
        ttbn.MODE["fused"] = True
    print(f"[build] BN layers {nb} ({'fp16, fused statistics' if fp16 else 'fp32'})", flush=True)
    return preds


def _run_blocks(preds, lst, cond, draw, chunks, fp16, workers, target, tag=""):
    import torch
    import p04_heldout as ph
    conds = ttbn.DEV_CONDS if target == "dev" else ttbn.P04_CONDS
    kc = conds[cond]
    recs = [[None] * len(lst) for _ in ttbn.SEEDS]
    spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": draw or 0, "images": lst}
    t0 = time.time()
    ttbn.MODE["n"] = ttbn.PRIOR_N
    for n_done, ch in enumerate(ttbn._frames(spec, chunks, workers), 1):
        with torch.no_grad(), ph.amp(fp16):
            out = ph._multi_batch(preds, [im for _, im in ch])
        for j in range(len(ttbn.SEEDS)):
            for (i, _), r in zip(ch, out[j]):
                recs[j][i] = {**(ph._f32(r) if fp16 else r), "image_path": lst[i]}
        if n_done % 200 == 0:
            print(f"[build{tag}] {cond}/{draw} {n_done}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
    ttbn.MODE["n"] = None
    return recs


def _blocks(n: int) -> list[list[int]]:
    idx = list(range(n))
    return [idx[a:a + ttbn.BATCH] for a in range(0, n, ttbn.BATCH)]


def _list(target: str) -> list[str]:
    import p04_heldout as ph
    return [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()] if target == "dev" else ph.images("vis")


def cmd_build(args) -> int:
    from uqfusion.eval.corruptions_v2 import CODE_REV
    fp16 = args.target == "p04"
    if fp16 and not json.loads(ttbn.FP16_CHECK.read_text(encoding="utf-8")).get("pass"):
        raise SystemExit("pohang04 fp16 needs N4's passed development check")
    lst = _list(args.target)
    todo = [(c, d) for c, d in streams(args.target)
            if not all(comp_path(args.target, k, c, d).is_file() for k in ttbn.SEEDS)]
    print(f"[build] {args.target}: {len(lst)} frames, {len(todo)} streams to complete", flush=True)
    if not todo:
        return 0
    preds = _preds(fp16)
    for c, d in todo:
        st = [_stored(args.target, k, c, d) for k in ttbn.SEEDS]
        have = (np.zeros(len(lst), bool) if st[0] is None else np.array([r is not None for r in st[0]]))
        for s in st[1:]:                                   # the five systems ran the same blocks
            assert (s is None and st[0] is None) or np.array_equal(have, [r is not None for r in s])
        blocks = _blocks(len(lst))
        for b in blocks:                                   # N4 ran whole blocks or none
            assert have[b].all() or not have[b].any(), f"partial block in N4's records ({c}, {d})"
        chunks = [b for b in blocks if not have[b].any()]
        t0 = time.time()
        recs = _run_blocks(preds, lst, c, d, chunks, fp16, args.workers, args.target)
        conds = ttbn.DEV_CONDS if args.target == "dev" else ttbn.P04_CONDS
        kc = conds[c]
        for j, k in enumerate(ttbn.SEEDS):
            o = comp_path(args.target, k, c, d)
            o.parent.mkdir(parents=True, exist_ok=True)
            meta = {"target": args.target, "cond": c, "draw": d, "prior_N": ttbn.PRIOR_N, "batch": ttbn.BATCH,
                    "blocks": "N4's grid; only the blocks N4's stored records lack", "blocks_run": len(chunks),
                    "blocks_from_n4": len(blocks) - len(chunks),
                    "precision": "fp16 autocast, channels_last, cudnn.benchmark, fused statistics" if fp16 else "fp32",
                    "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                    "corrupt_seed": d if kc else None, "corrupt_version": "v2" if kc else None,
                    "corrupt_code": CODE_REV if kc else None,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": meta, "records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[build] {c}/{d} done: {len(chunks)} new blocks, {len(blocks) - len(chunks)} from N4, "
              f"{time.time() - t0:.0f}s", flush=True)
    return 0


def _rec_diff(a: dict, b: dict) -> float:
    m = 0.0
    for key, v in a.items():
        if isinstance(v, np.ndarray) and key != "feat" and v.dtype.kind == "f":
            w = b[key]
            if v.shape != w.shape:
                return float("inf")
            if v.size:
                m = max(m, float(np.abs(v.astype(np.float64) - w.astype(np.float64)).max()))
    return m


def cmd_repro(args) -> int:
    """Rebuild a few blocks N4 already ran, on the same code path, and compare with the stored records."""
    fp16 = args.target == "p04"
    lst = _list(args.target)
    c, d = ("lowlight", ttbn.DEV_DRAWS[0]) if args.target == "dev" else ("lowlight", 941)
    st = [_stored(args.target, k, c, d) for k in ttbn.SEEDS]
    have = np.array([r is not None for r in st[0]])
    blocks = [b for b in _blocks(len(lst)) if have[b].all()]
    pick = [blocks[i] for i in np.linspace(0, len(blocks) - 1, args.n).astype(int)]
    preds = _preds(fp16)
    recs = _run_blocks(preds, lst, c, d, pick, fp16, 2, args.target, " repro")
    diffs, nbox, top = [], [], []
    for j in range(len(ttbn.SEEDS)):
        for b in pick:
            for i in b:
                a_, b_ = recs[j][i], st[j][i]
                diffs.append(_rec_diff(a_, b_))
                nbox.append(len(a_["conf"]) == len(b_["conf"]))
                # boxes above 0.25 confidence: count and largest confidence change of the top box
                top.append((int((a_["conf"] > 0.25).sum()) == int((b_["conf"] > 0.25).sum()),
                            float(abs(a_["conf"].max(initial=0) - b_["conf"].max(initial=0)))))
    fin = [x for x in diffs if np.isfinite(x)]
    res = {"target": args.target, "cond": c, "draw": d, "blocks": [b[0] // ttbn.BATCH for b in pick],
           "frames_compared": len(diffs), "n_exact": int(sum(x == 0.0 for x in diffs)),
           "same_box_count": float(np.mean(nbox)),
           "max_abs_diff_same_count": float(max(fin)) if fin else None,
           "same_count_above_0.25": float(np.mean([t[0] for t in top])),
           "max_top_conf_change": float(max(t[1] for t in top))}
    p = OUT / f"repro_{args.target}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(json.dumps(res, indent=1))
    return 0


# ------------------------------------------------------------------ development scoring
def gate_ctx_arms(ctx, cond: str) -> dict:
    q = np.asarray(ctx.q_vis_by_cond[cond]) < 1.0
    irok = ~np.asarray(ctx.ir_night_raw, bool) & (np.asarray(ctx.ir_d2) <= ctx.ir_bound)
    return q, irok


def select(adapted: list[dict], plain: list[dict], gate: np.ndarray) -> list[dict]:
    """Adapted records where `gate` fires, plain elsewhere (as `ttbn.gated_stream`)."""
    if not gate.any():
        return plain
    out = []
    for a, b, f in zip(adapted, plain, gate):
        if not f:
            out.append(b)
            continue
        assert Path(a["image_path"]).name == Path(b["image_path"]).name
        out.append({**a, "image_path": b["image_path"], "feat": b["feat"]})
    return out


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
    sun = sun_up("dev")
    res = {}
    for c in conds:
        q, irok = gate_ctx_arms(ctx, c)
        gs = gates_from(q, irok, sun)
        fs = gates_dev(draw, c, ir_cond)
        for a in ARMS:
            if not np.array_equal(gs[a], fs[a]):
                raise SystemExit(f"context gate differs from frame-statistics gate ({spec}, {c}, {a})")
        vis = ctx.vis_by_cond[c]
        assert [Path(r["image_path"]).name for r in vis] == [Path(p).name for p in _list("dev")]
        full = full_stream("dev", k, c, None if c == "clean" else draw)
        o0 = run_systems(u, c, compute_metrics=False)
        parts = {"vis": frame_parts(vis, ctx.gts), "ir": frame_parts(o0["ir_in_vis"], ctx.gts),
                 "union": frame_parts(o0["fused_gated"], ctx.gts)}
        done = {}
        for a in ARMS:
            key = gs[a].tobytes()
            if key in done:                                  # identical gate: identical arm
                parts[f"union_{a}"], parts[f"vis_{a}"] = parts[f"union_{done[key]}"], parts[f"vis_{done[key]}"]
                continue
            done[key] = a
            vg = select(full, vis, gs[a])
            parts[f"vis_{a}"] = frame_parts(vg, ctx.gts)
            parts[f"union_{a}"] = (parts["union"] if not gs[a].any() else
                                   frame_parts(run_systems(u, c, compute_metrics=False, vis_records=vg)["fused_gated"],
                                               ctx.gts))
        for s, idx in sl.items():
            res[f"{c}|{s}"] = {n: ship_ap(p, idx) for n, p in parts.items()}
            res[f"{c}|{s}"]["gate"] = {a: float(gs[a][idx].mean()) for a in ARMS}
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


def _t(vals) -> tuple[float, float, float]:
    return ttbn._cell_stats(list(vals))


def cmd_report_dev(args) -> int:
    res = json.loads((OUT / "score_dev.json").read_text(encoding="utf-8"))
    n4 = json.loads((ttbn.DEV_OUT / "score.json").read_text(encoding="utf-8"))
    cells, verdict = {}, {}
    worst_id = 0.0
    for c in ttbn.DEV_CONDS:
        for ic in ttbn.DEV_IR_CONDS:
            for s in ("day", "night"):
                cell = f"{c}/{ic or 'clean'}/{s}"
                rows = {k: [res[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in ttbn.DEV_DRAWS] for k in ttbn.SEEDS}
                # identity: this pipeline's N4 arm and union are N4's registered development scores
                for k in ttbn.SEEDS:
                    for d, r in zip(ttbn.DEV_DRAWS, rows[k]):
                        q = n4[f"{d}|{ic}|{k}"][f"{c}|{s}"]
                        worst_id = max(worst_id, abs(r["union_n4"] - q["union_ttbn"]), abs(r["union"] - q["union"]))
                cells[cell] = {"union": float(np.mean([np.mean([r["union"] for r in rows[k]]) for k in ttbn.SEEDS]))}
                for a in ARMS:
                    dv = [np.mean([r[f"union_{a}"] - r["union"] for r in rows[k]]) for k in ttbn.SEEDS]
                    dn = [np.mean([r[f"union_{a}"] - r["union_n4"] for r in rows[k]]) for k in ttbn.SEEDS]
                    cl = [np.mean([r[f"union_{a}"] - max(r[f"vis_{a}"], r["ir"]) for r in rows[k]]) for k in ttbn.SEEDS]
                    g = float(np.mean([np.mean([r["gate"][a] for r in rows[k]]) for k in ttbn.SEEDS]))
                    cells[cell][a] = {"gate": g, "vs_union": _t(dv), "vs_n4": _t(dn), "claim": _t(cl)}
    if worst_id > 0:
        raise SystemExit(f"N4 arm or union differs from N4's registered development scores by {worst_id}")
    for a in ARMS:
        worse = [c for c, v in cells.items() if v[a]["vs_union"][0] <= -0.0014 and v[a]["vs_union"][2] < 0]
        better = [c for c, v in cells.items() if v[a]["vs_union"][0] >= 0.0060 and v[a]["vs_union"][1] > 0]
        wn = [c for c, v in cells.items() if v[a]["vs_n4"][0] <= -0.0014 and v[a]["vs_n4"][2] < 0]
        bn = [c for c, v in cells.items() if v[a]["vs_n4"][0] >= 0.0060 and v[a]["vs_n4"][1] > 0]
        verdict[a] = {"worse": worse, "better": better, "pass": (not worse) and bool(better),
                      "worse_than_n4": wn, "better_than_n4": bn}
    json_out = {"verdict": verdict, "cells": cells, "identity_max_abs": worst_id}
    o = ROOT / args.out
    o.with_suffix(".json").write_text(json.dumps(json_out, indent=1), encoding="utf-8")
    name = {"ungated": "ungated", "qvis": "q_vis only", "irday": "IR day only", "solar": "solar only",
            "qvis_solar": "q_vis ∧ solar", "n4": "IG-BN (N4)"}
    L = ["# Gate ablation, development", "",
         "`docs/prereg-gate-ablation-2026-10-14.md`. Draws VIS 942–944 / IR 952–954, five systems, 60 cells. "
         "Δ = arm − `union` (ship AP), mean over the three draws per system, then over the five systems; 95% "
         "t-interval (df 4). Rule (§3.4): worse = Δ ≤ −0.0014 with interval below zero; better = Δ ≥ +0.0060 "
         "with interval above zero. The N4 arm reproduces N4's registered development scores exactly "
         f"(max |diff| {worst_id:.1e}).", "",
         "| arm | better | worse | verdict vs `union` | better than IG-BN | worse than IG-BN |",
         "|---|---:|---:|---|---:|---:|"]
    for a in ARMS:
        v = verdict[a]
        L.append(f"| {name[a]} | {len(v['better'])} | {len(v['worse'])} | {'PASS' if v['pass'] else 'FAIL'} | "
                 f"{len(v['better_than_n4'])} | {len(v['worse_than_n4'])} |")
    for a in ARMS:
        v = verdict[a]
        L += ["", f"**{name[a]}.** Worse than `union`: {', '.join(v['worse']) or 'none'}. Better: "
                  f"{', '.join(v['better']) or 'none'}. Worse than IG-BN: {', '.join(v['worse_than_n4']) or 'none'}. "
                  f"Better than IG-BN: {', '.join(v['better_than_n4']) or 'none'}."]
    L += ["", "## Per cell: gate rate and Δ vs `union` [95%]", "",
          "| cell | union | " + " | ".join(name[a] for a in ARMS) + " |",
          "|---|---:|" + "---|" * len(ARMS)]
    for cell, v in cells.items():
        L.append(f"| {cell} | {v['union']:.4f} | " + " | ".join(
            f"{v[a]['gate']:.2f}: {v[a]['vs_union'][0]:+.4f} [{v[a]['vs_union'][1]:+.4f}, {v[a]['vs_union'][2]:+.4f}]"
            for a in ARMS) + " |")
    L += ["", "## Per cell: Δ vs IG-BN [95%] (paired, descriptive except as registered)", "",
          "| cell | " + " | ".join(name[a] for a in ARMS if a != "n4") + " |", "|---|" + "---|" * (len(ARMS) - 1)]
    for cell, v in cells.items():
        L.append(f"| {cell} | " + " | ".join(
            f"{v[a]['vs_n4'][0]:+.4f} [{v[a]['vs_n4'][1]:+.4f}, {v[a]['vs_n4'][2]:+.4f}]" for a in ARMS if a != "n4") + " |")
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:16 + 2 * len(ARMS)]))
    return 0


# ------------------------------------------------------------------ pohang04
PRE = OUT / "p04_pre"
P04_GATES = OUT / "p04_gates.json"


def _p04_cells():
    import p04_heldout_score as psc
    return [psc.VERDICT_CELL] + psc.CORRUPTED


def _p04_job(job):
    import dataclasses
    import holdout_p04_look as look
    import holdout_p04_v2_rescore as v2r
    import p04_heldout_score as psc
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import ap_from_parts, frame_parts
    from uqfusion.eval.ctx import run_systems
    k, s, ic, vc = job
    ttbn._low()
    se._memo_distances()
    tag, key = psc.cell_tag((vc, ic)), f"seed{k}_{s}"
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    sub_ = look.Pohang04() if s == "clean" else v2r.Pohang04V2()
    ctx = look.system_context(sub_, k, s, sorted({vc, "clean"}), ic, dev)
    q, irok = gate_ctx_arms(ctx, vc)
    assert float(ctx.ir_bound) == float(ttbn._sconst()["axes"]["ir_health"]["bound"])
    gs = gates_from(q, irok, sun_up("p04"))
    fs = gates_p04(vc, ic, s)
    for a in ARMS:
        if not np.array_equal(gs[a], fs[a]):
            raise SystemExit(f"context gate differs from frame-statistics gate ({job}, {a})")
    vis = ctx.vis_by_cond[vc]
    draw = None if (s == "clean" or vc == "clean") else int(s[4:7])
    full = full_stream("p04", k, vc, draw)
    u = dataclasses.replace(ctx, veto=None)
    info, done = {}, {}
    for a in ARMS:
        g = gs[a]
        info[a] = {"gate": float(g.mean()), "n_gated": int(g.sum())}
        if not g.any():
            info[a]["same_as"] = "union"
            continue
        kb = g.tobytes()
        if kb in done:
            info[a]["same_as"] = done[kb]
            continue
        done[kb] = a
        recs = run_systems(u, vc, compute_metrics=False, vis_records=select(full, vis, g))["fused_gated"]
        parts = frame_parts(recs, ctx.gts)
        if a == "n4":                 # identity with the registered N4 scoring: same observed AP
            ref = psc.PRE / "union_ttbn/fused" / tag / key / "obs.json"
            e = ap_from_parts(parts)["per_class"].get(psc.SHIP)
            v = float(e["ap50_95"]) if e else float("nan")
            info[a]["obs"] = v
            if ref.is_file():
                r = json.loads(ref.read_text())["ap"]
                info[a]["registered_obs"] = r
                if r != v:
                    raise SystemExit(f"N4 arm differs from the registered N4 scoring ({job}): {v} vs {r}")
            continue
        d = PRE / f"union_{a}" / tag / key
        if not (d / "obs.json").is_file():
            d.mkdir(parents=True, exist_ok=True)
            info[a]["obs"] = psc.save_pre(parts, d)
    return job, info


def cmd_score_p04(args) -> int:
    import p04_heldout_score as psc
    gates = json.loads(P04_GATES.read_text(encoding="utf-8")) if P04_GATES.is_file() else {}
    jobs = [(k, s, ic, vc) for vc, ic in _p04_cells() for s in psc.subs_of((vc, ic)) for k in ttbn.SEEDS
            if f"{k}|{s}|{ic}|{vc}" not in gates]
    print(f"[p04] {len(jobs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (j, info) in enumerate(ex.map(_p04_job, jobs), 1):
            gates[f"{j[0]}|{j[1]}|{j[2]}|{j[3]}"] = info
            tmp = P04_GATES.with_suffix(".tmp")
            tmp.write_text(json.dumps(gates, indent=1), encoding="utf-8")
            os.replace(tmp, P04_GATES)
            print(f"[p04] {n}/{len(jobs)} {j} " + " ".join(f"{a}={info[a]['gate']:.3f}" for a in ARMS)
                  + f" ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_boot_p04(args) -> int:
    """The gate-free scoring's replicates (same weights, same arithmetic, checked against the CPU score)."""
    import torch
    import p04_heldout_score as psc
    from holdout_p04_v2_rescore import load_ship_pre
    from uqfusion.eval.apmetrics import RECALL_GRID
    dev = torch.device("cuda")
    grid_g = torch.from_numpy(RECALL_GRID).to(dev)
    W = np.load(psc.OUT / "boot_weights.npy")
    todo = sorted(p.parent for p in PRE.rglob("obs.json") if not (p.parent / "reps.npy").is_file())
    print(f"[boot] {len(todo)} presorts ({len(W)} replicates each)", flush=True)
    t0, checked = time.time(), False
    for n, d in enumerate(todo, 1):
        pre = load_ship_pre(d)
        if not checked:
            print(f"[boot] checked {psc._check(pre, W, dev, grid_g)} rows equal to the CPU reference", flush=True)
            checked = True
        np.save(d / "reps.npy", psc.boot_batch(pre, W, dev, grid_g))
        torch.cuda.empty_cache()
        if n % 10 == 0 or n == len(todo):
            print(f"[boot] {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def _series(arm: str, cell, gates: dict):
    """(observed, replicates) for one arm on one cell, mean over systems within a draw, then over draws.
    An arm whose gate equals another's (or fires nowhere) takes that arm's (or `union`'s) presort."""
    import p04_heldout_score as psc
    tag = psc.cell_tag(cell)
    vals, reps = [], []
    for s in psc.subs_of(cell):
        vv, rr = [], []
        for k in ttbn.SEEDS:
            a = arm
            info = gates[f"{k}|{s}|{cell[1]}|{cell[0]}"]
            while a not in ("union", "n4") and "same_as" in info[a]:
                a = info[a]["same_as"]
            if a == "n4" and "same_as" in info["n4"]:
                a = info["n4"]["same_as"]
            d = {"union": psc.PRE / "union/fused", "n4": psc.PRE / "union_ttbn/fused"}.get(a, PRE / f"union_{a}")
            d = d / tag / f"seed{k}_{s}"
            if not (d / "reps.npy").is_file():
                return None
            vv.append(json.loads((d / "obs.json").read_text())["ap"])
            rr.append(np.load(d / "reps.npy"))
        vals.append(np.mean(vv))
        reps.append(np.mean(rr, axis=0))
    return float(np.mean(vals)), np.mean(reps, axis=0)


def cmd_report_p04(args) -> int:
    import p04_heldout_score as psc
    gates = json.loads(P04_GATES.read_text(encoding="utf-8"))
    name = {"ungated": "ungated", "qvis": "q_vis only", "irday": "IR day only", "solar": "solar only",
            "qvis_solar": "q_vis ∧ solar", "n4": "IG-BN (N4)"}
    cells, verdict = {}, {}
    for c in _p04_cells():
        tag = psc.cell_tag(c)
        un = _series("union", c, gates)
        n4 = _series("n4", c, gates)
        cells[tag] = {"union": un[0]}
        for a in ARMS:
            x = _series(a, c, gates)
            g = float(np.mean([gates[f"{k}|{s}|{c[1]}|{c[0]}"][a]["gate"] for s in psc.subs_of(c) for k in ttbn.SEEDS]))
            d, (lo, hi) = x[0] - un[0], psc.ci(x[1] - un[1])
            dn, (lon, hin) = x[0] - n4[0], psc.ci(x[1] - n4[1])
            cells[tag][a] = {"gate": g, "ap": x[0], "vs_union": [d, lo, hi], "vs_n4": [dn, lon, hin]}
    for a in ARMS:
        worse = [t for t, v in cells.items() if v[a]["vs_union"][0] <= -0.0014 and v[a]["vs_union"][2] < 0]
        better = [t for t, v in cells.items() if v[a]["vs_union"][0] >= 0.0060 and v[a]["vs_union"][1] > 0]
        wn = [t for t, v in cells.items() if v[a]["vs_n4"][0] <= -0.0014 and v[a]["vs_n4"][2] < 0]
        bn = [t for t, v in cells.items() if v[a]["vs_n4"][0] >= 0.0060 and v[a]["vs_n4"][1] > 0]
        verdict[a] = {"worse": worse, "better": better,
                      "replicates": (not worse) and bool(better), "worse_than_n4": wn, "better_than_n4": bn}
    o = ROOT / args.out
    o.with_suffix(".json").write_text(json.dumps({"verdict": verdict, "cells": cells}, indent=1), encoding="utf-8")
    L = ["# Gate ablation, held-out run (pohang04)", "",
         "`docs/prereg-gate-ablation-2026-10-14.md`. The nine registered cells; ship AP against the VIS ground "
         "truth, mean over the five systems and the four draws of a corrupted cell. Δ paired; 95% interval from "
         "the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1). Same rule as development.", "",
         "| arm | better | worse | vs `union` | better than IG-BN | worse than IG-BN |", "|---|---:|---:|---|---:|---:|"]
    for a in ARMS:
        v = verdict[a]
        L.append(f"| {name[a]} | {len(v['better'])} | {len(v['worse'])} | "
                 f"{'REPLICATES' if v['replicates'] else 'DOES NOT REPLICATE'} | {len(v['better_than_n4'])} | "
                 f"{len(v['worse_than_n4'])} |")
    L += ["", "| cell (VIS/IR) | union | " + " | ".join(name[a] for a in ARMS) + " |", "|---|---:|" + "---|" * len(ARMS)]
    for t, v in cells.items():
        L.append(f"| {t.replace('__', '/')} | {v['union']:.4f} | " + " | ".join(
            f"{v[a]['gate']:.3f}: {v[a]['vs_union'][0]:+.4f} [{v[a]['vs_union'][1]:+.4f}, {v[a]['vs_union'][2]:+.4f}]"
            for a in ARMS) + " |")
    L += ["", "Δ vs IG-BN [95%]:", "", "| cell (VIS/IR) | " + " | ".join(name[a] for a in ARMS if a != "n4") + " |",
          "|---|" + "---|" * (len(ARMS) - 1)]
    for t, v in cells.items():
        L.append(f"| {t.replace('__', '/')} | " + " | ".join(
            f"{v[a]['vs_n4'][0]:+.4f} [{v[a]['vs_n4'][1]:+.4f}, {v[a]['vs_n4'][2]:+.4f}]" for a in ARMS if a != "n4") + " |")
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


# ------------------------------------------------------------------ pohang01 night pairs: label-free gate rate
def _night_stats_path(ic: str, draw: int, kind: str) -> Path:
    return OUT / "night" / kind / ttbn.sub(draw - 10) / f"gauss_ir_paired_{ic}.json"


#: depth maps of the night-test IR frames (v2 fog), built by `build_depth_maps.py --out-root` on C: (A: is full);
#: `corruptions_v2.depth_path` reads the module's DEPTH_ROOT at call time, so each worker points it here
NIGHT_DEPTH = OUT / "depth_v2"


def _night_init(spec):
    import uqfusion.eval.corruptions_v2 as c2
    from uqfusion.eval import parallel_frames as pf
    c2.DEPTH_ROOT = NIGHT_DEPTH
    pf._init(spec)


def _night_chunk(idx):
    from uqfusion.eval import parallel_frames as pf
    return [pf._one(i)[2] for i in idx]


def _night_frames(paths, spec, workers):
    """`parallel_frames.corrupted_frames`' arithmetic (its `_init`/`_one`), statistics only, in order."""
    spec = {**spec, "images": [str(p) for p in paths]}
    chunks = [list(range(a, min(a + 4, len(paths)))) for a in range(0, len(paths), 4)]
    with ProcessPoolExecutor(workers, initializer=_night_init, initargs=(spec,)) as ex:
        for out in ex.map(_night_chunk, chunks):
            yield from out


def cmd_night_stats(args) -> int:
    """IR frame statistics of the 1,109 night pairs under the development IR corruptions (v2, seeds
    952-954), by the arithmetic that wrote the development v2 IR statistics (`parallel_frames`, this
    interpreter). Checked bit-exact on development frames first."""
    import cv2
    from uqfusion.eval.parallel_frames import corrupted_frames
    pi = [ln.strip() for ln in open(ROOT / "runs/derived/paired_val_ir.txt") if ln.strip()]
    for ic, (kind, sev) in (("fog_s1", ("fog", 1)), ("noise_s3", ("noise", 3))):
        for kd in ("brightness", "structure"):
            ref = json.loads(ttbn._dev_stats(943, "ir", ic, kd).read_text(encoding="utf-8"))
            assert (ref["corrupt"], ref["severity"], ref["corrupt_seed"]) == (kind, sev, 953)
        spec = {"corrupt": kind, "severity": sev, "corrupt_seed": 953, "corrupt_version": "v2", "modality": "ir",
                "stats": "ir"}
        got = [st for _, _, st in corrupted_frames(pi[:args.verify_n], spec, workers=0)]
        for kd, j in (("brightness", 0), ("structure", 1)):
            ref = json.loads(ttbn._dev_stats(943, "ir", ic, kd).read_text(encoding="utf-8"))["frames"]
            for i in range(args.verify_n):
                diff = [x for x in ref[i] if x != "image_path" and got[i][j].get(x) != ref[i][x]]
                if diff:
                    raise SystemExit(f"[verify] IR {ic} {kd} frame {i} differs: {diff[:4]}")
        print(f"[verify] IR {ic}: {args.verify_n} development frames bit-exact (brightness, structure)", flush=True)
    paths = ttbn_read(NIGHT_IR_LIST)
    for ic, (kind, sev) in NIGHT_IR_CONDS.items():
        for draw in NIGHT_IR_DRAWS:
            outs = {kd: _night_stats_path(ic, draw, kd) for kd in ("brightness", "structure")}
            if all(p.is_file() for p in outs.values()):
                continue
            spec = {"corrupt": kind, "severity": sev, "corrupt_seed": draw, "corrupt_version": "v2", "modality": "ir",
                    "stats": "ir"}
            br, st = [], []
            t0 = time.time()
            for s_ in _night_frames(paths, spec, args.workers):
                br.append(s_[0])
                st.append(s_[1])
            for kd, fr in (("brightness", br), ("structure", st)):
                p = outs[kd]
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(json.dumps({"modality": "ir", "corrupt": kind, "severity": sev, "corrupt_seed": draw,
                                         "corrupt_version": "v2", "n_frames": len(fr), "numpy": np.__version__,
                                         "opencv": cv2.__version__, "frames": fr}), encoding="utf-8")
            print(f"[night] IR {ic} seed {draw}: {len(br)} frames ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_night_rates(args) -> int:
    vis_st = NIGHT_DERIVED / "structure/clean/gauss_vis_paired_clean.json"
    q = ttbn.vis_unhealthy(vis_st)
    sun = sun_up("night")
    s = ttbn._sconst()["axes"]
    rows, out = [], {}

    def terms(br: Path, st: Path):
        p05 = np.asarray([f["p05"] for f in json.loads(br.read_text(encoding="utf-8"))["frames"]], float)
        day = ~(p05 > float(s["ir_p05"]["threshold"]))
        ok = ttbn._d2(st, s["ir_health"]) <= float(s["ir_health"]["bound"])
        return day, ok

    conds = [("clean", None)] + [(ic, d) for ic in NIGHT_IR_CONDS for d in NIGHT_IR_DRAWS]
    for ic, d in conds:
        if ic == "clean":
            day, ok = terms(NIGHT_DERIVED / "brightness/clean/gauss_ir_paired_clean.json",
                            NIGHT_DERIVED / "structure/clean/gauss_ir_paired_clean.json")
        else:
            day, ok = terms(_night_stats_path(ic, d, "brightness"), _night_stats_path(ic, d, "structure"))
        g = q & day & ok
        key = ic if d is None else f"{ic}/{d}"
        out[key] = {"n": int(len(g)), "q_vis_lt_1": float(q.mean()), "ir_reads_day": float(day.mean()),
                    "ir_healthy": float(ok.mean()), "ir_day_and_healthy": float((day & ok).mean()),
                    "n4_gate": float(g.mean()), "n4_gate_frames": int(g.sum()), "solar_day": float(sun.mean())}
    by_cond = {}
    for ic in ["clean"] + list(NIGHT_IR_CONDS):
        ks = ["clean"] if ic == "clean" else [f"{ic}/{d}" for d in NIGHT_IR_DRAWS]
        by_cond[ic] = {m: float(np.mean([out[k][m] for k in ks])) for m in
                       ("q_vis_lt_1", "ir_reads_day", "ir_healthy", "ir_day_and_healthy", "n4_gate")}
        by_cond[ic]["n4_gate_frames_max"] = int(max(out[k]["n4_gate_frames"] for k in ks))
        b = by_cond[ic]
        rows.append(f"| {ic} | {b['q_vis_lt_1']:.3f} | {b['ir_reads_day']:.4f} | {b['ir_healthy']:.4f} | "
                    f"{b['ir_day_and_healthy']:.4f} | {b['n4_gate']:.4f} | {b['n4_gate_frames_max']} |")
    o = ROOT / args.out
    o.with_suffix(".json").write_text(json.dumps({"per_draw": out, "per_condition": by_cond}, indent=1), encoding="utf-8")
    L = ["# IG-BN's gate on the pohang01 night test pairs (label-free)", "",
         "`docs/prereg-gate-ablation-2026-10-14.md`, part 2. The 1,109 pohang01 test-split pairs of the night test "
         "(`runs/night_p01test/pairs_*.txt`); VIS clean; IR clean and under the development IR corruptions (v2, "
         "seeds 952–954, mean over the three). No detection and no label is read. Solar elevation is below zero on "
         f"{1 - sun.mean():.4f} of the pairs.", "",
         "| IR condition | q_vis < 1 | IR reads day | IR healthy | IR day ∧ healthy | IG-BN gate | max gated frames (of 1,109) |",
         "|---|---:|---:|---:|---:|---:|---:|"] + rows
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("rates")
    r = sp.add_parser("repro")
    r.add_argument("--target", choices=["dev", "p04"], required=True)
    r.add_argument("--n", type=int, default=6)
    b = sp.add_parser("build")
    b.add_argument("--target", choices=["dev", "p04"], required=True)
    b.add_argument("--workers", type=int, default=6)
    for n_ in ("score-dev", "score-p04"):
        s_ = sp.add_parser(n_)
        s_.add_argument("--workers", type=int, default=4)
    sp.add_parser("boot-p04")
    r = sp.add_parser("report-dev")
    r.add_argument("--out", default="docs/eval/ttbn/gate_ablation_dev_2026-10-14.md")
    r = sp.add_parser("report-p04")
    r.add_argument("--out", default="docs/eval/ttbn/gate_ablation_p04_2026-10-14.md")
    n = sp.add_parser("night-stats")
    n.add_argument("--workers", type=int, default=8)
    n.add_argument("--verify-n", type=int, default=24)
    r = sp.add_parser("night-rates")
    r.add_argument("--out", default="docs/eval/ttbn/gate_night_p01_2026-10-14.md")
    args = ap.parse_args()
    return {"rates": cmd_rates, "repro": cmd_repro, "build": cmd_build, "score-dev": cmd_score_dev,
            "report-dev": cmd_report_dev, "score-p04": cmd_score_p04, "boot-p04": cmd_boot_p04,
            "report-p04": cmd_report_p04, "night-stats": cmd_night_stats, "night-rates": cmd_night_rates}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
