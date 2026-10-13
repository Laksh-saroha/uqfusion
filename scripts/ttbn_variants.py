"""Two N4 variants on development (`docs/prereg-ttbn-variants-2026-10-11.md`).

N4g: N4's gate with a glare term in its VIS side, (q_vis < 1 OR (VIS mean > 117.92 AND VIS std > 62.31))
AND IR reads day AND ir_d2 <= 222.7. Adapted records are N4's (same BN-mix, same blocks); the glare
blocks N4 never ran are built here.

N4c: causal statistics. Within each run, blocks of 8 consecutive listed frames; block b normalises every
BatchNorm2d with BN-mix of the source statistics and the statistics block b-1's input had at that layer
during block b-1's own pass; a run's first block uses the source statistics. N4's gate, E4 union.

    py -3.13 scripts/ttbn_variants.py build-glare
    py -3.13 scripts/ttbn_variants.py build-causal
    py -3.13 scripts/ttbn_variants.py score-dev
    py -3.13 scripts/ttbn_variants.py report-dev
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

import ttbn  # noqa: E402

OUT = ROOT / "runs/novel/ttbn_variants"
GLARE_MEAN, GLARE_STD = 117.92185974121094, 62.30767059326172     # max over clean development day frames
CAUSAL_CONDS = ("fog", "fog_s1", "lowlight", "glare")              # clean: N4's gate never fires
STATE = {"on": False}


def out_path(kind: str, k: int, cond: str, draw: int) -> Path:
    return OUT / kind / f"seed{k}" / ttbn.sub(draw) / f"gauss_vis_paired_{cond}_bn16.pkl"


def glare_term(bright: Path) -> np.ndarray:
    fr = json.loads(Path(bright).read_text(encoding="utf-8"))["frames"]
    m = np.array([f["mean"] for f in fr], float)
    s = np.array([f["std"] for f in fr], float)
    return (m > GLARE_MEAN) & (s > GLARE_STD)


def gate_g_stats(draw: int, cond: str, ir_cond) -> np.ndarray:
    """N4g's gate from frame-statistics files."""
    q = ttbn.vis_unhealthy(ttbn._dev_stats(draw, "vis", cond, "structure"))
    g = glare_term(ttbn._dev_stats(draw, "vis", cond, "brightness"))
    return (q | g) & ttbn.ir_day_ok(ttbn._dev_stats(draw, "ir", ir_cond, "brightness"),
                                    ttbn._dev_stats(draw, "ir", ir_cond, "structure"))


def _needed(draw: int, cond: str, gate_fn) -> np.ndarray:
    out = None
    for ic in ttbn.DEV_IR_CONDS:
        g = gate_fn(draw, cond, ic)
        out = g if out is None else out | g
    return out


def _gate_n4_stats(draw, cond, ic):
    return ttbn.vis_unhealthy(ttbn._dev_stats(draw, "vis", cond, "structure")) & ttbn.ir_day_ok(
        ttbn._dev_stats(draw, "ir", ic, "brightness"), ttbn._dev_stats(draw, "ir", ic, "structure"))


# ------------------------------------------------------------------ causal BatchNorm
def patch_bn_causal(model) -> int:
    """While STATE['on']: normalise with BN-mix of the source statistics and the previous block's
    statistics at this layer (`m._prev`), then store this block's statistics for the next block.
    `m._prev is None` (a run's first block): the source statistics."""
    import torch
    import torch.nn.functional as F
    n_bn = 0
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            n_bn += 1
            m._prev = None

            def fwd(x, m=m):
                if not STATE["on"]:
                    return F.batch_norm(x, m.running_mean, m.running_var, m.weight, m.bias, False, 0.0, m.eps)
                xf = x.float()
                mu_t = xf.mean(dim=(0, 2, 3))
                var_t = xf.var(dim=(0, 2, 3), unbiased=False)
                if m._prev is None:
                    mu, var = m.running_mean, m.running_var
                else:
                    pm, pv, n = m._prev
                    a = n / (ttbn.PRIOR_N + n)
                    mu = (1 - a) * m.running_mean + a * pm
                    var = (1 - a) * m.running_var + a * pv
                m._prev = (mu_t.detach(), var_t.detach(), x.shape[0])
                return F.batch_norm(x, mu, var, m.weight, m.bias, False, 0.0, m.eps)
            m.forward = fwd
    return n_bn


def _reset(preds) -> None:
    import torch
    for p in preds:
        for m in p.model.modules():
            if isinstance(m, torch.nn.BatchNorm2d):
                m._prev = None


# ------------------------------------------------------------------ builds
def _build(kind: str, todo, workers: int) -> None:
    import torch
    import p04_heldout as ph
    from uqfusion.eval.corruptions_v2 import CODE_REV
    from uqfusion.uq.infer import UQPredictor
    ttbn._low()
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    runs = [Path(p).parent.name for p in lst]
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=0, imgsz=640,
                         conf=0.001) for k in ttbn.SEEDS]
    nb = [(ttbn.patch_bn if kind == "glare" else patch_bn_causal)(p.model) for p in preds]
    print(f"[{kind}] BN layers {nb}; {len(todo)} streams", flush=True)
    for c, d, need in todo:
        kc = ttbn.DEV_CONDS[c]
        if kind == "glare":           # N4's block grid and arithmetic: blocks of 8 listed frames with a needed frame
            idx = list(range(len(lst)))
            chunks = [idx[a:a + ttbn.BATCH] for a in range(0, len(idx), ttbn.BATCH) if need[a:a + ttbn.BATCH].any()]
            first = set()
        else:                         # per-run grid; every block of a run with a needed frame, in order
            chunks, first = [], set()
            for r in dict.fromkeys(runs):
                ir_ = [i for i, x in enumerate(runs) if x == r]
                if not need[ir_].any():
                    continue
                first.add(len(chunks))
                chunks += [ir_[a:a + ttbn.BATCH] for a in range(0, len(ir_), ttbn.BATCH)]
        recs = [[None] * len(lst) for _ in ttbn.SEEDS]
        t0 = time.time()
        spec = {"kind": kc[0] if kc else None, "sev": kc[1] if kc else 0, "seed": d, "images": lst}
        if kind == "glare":
            ttbn.MODE["n"] = ttbn.PRIOR_N
        else:
            STATE["on"] = True
        for n_done, ch in enumerate(ttbn._frames(spec, chunks, workers)):
            if kind == "causal" and n_done in first:
                _reset(preds)                                   # a run's first block: source statistics
            with torch.no_grad():
                out = ph._multi_batch(preds, [im for _, im in ch])
            for j in range(len(ttbn.SEEDS)):
                for (i, _), r in zip(ch, out[j]):
                    recs[j][i] = {**r, "image_path": lst[i]}
            if (n_done + 1) % 100 == 0:
                print(f"[{kind}] {c}/{d} {n_done + 1}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
        ttbn.MODE["n"] = None
        STATE["on"] = False
        for j, k in enumerate(ttbn.SEEDS):
            o = out_path(kind, k, c, d)
            o.parent.mkdir(parents=True, exist_ok=True)
            meta = {"variant": "N4g" if kind == "glare" else "N4c", "cond": c, "draw": d, "prior_N": ttbn.PRIOR_N,
                    "batch": ttbn.BATCH, "blocks_run": len(chunks), "needed_frames": int(need.sum()),
                    "blocks": ("N4's grid, blocks with a needed frame" if kind == "glare" else
                               "per-run grid, every block of a run with a needed frame, in order; "
                               "block b normalised with block b-1's statistics; a run's first block at source"),
                    "precision": "fp32", "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                    "corrupt_seed": d if kc else None, "corrupt_version": "v2" if kc else None,
                    "corrupt_code": CODE_REV if kc else None,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}
            with open(o.with_suffix(".tmp"), "wb") as fh:
                pickle.dump({"meta": meta, "records": recs[j]}, fh, protocol=pickle.HIGHEST_PROTOCOL)
            os.replace(o.with_suffix(".tmp"), o)
        print(f"[{kind}] {c}/{d} done: {len(chunks)} blocks, {int(need.sum())} needed frames, "
              f"{time.time() - t0:.0f}s", flush=True)


def cmd_build_glare(args) -> int:
    todo = []
    for d in ttbn.DEV_DRAWS:
        need_g = _needed(d, "glare", gate_g_stats)
        if not all(out_path("glare", k, "glare", d).is_file() for k in ttbn.SEEDS):
            todo.append(("glare", d, need_g))
    for c in ttbn.DEV_CONDS:          # the term must add nothing outside glare (checked, not assumed)
        if c == "glare":
            continue
        for d in ttbn.DEV_DRAWS:
            for ic in ttbn.DEV_IR_CONDS:
                if not np.array_equal(gate_g_stats(d, c, ic), _gate_n4_stats(d, c, ic)):
                    raise SystemExit(f"N4g's gate differs from N4's outside glare: {c}/{d}/{ic}")
    print("[glare] N4g's gate equals N4's in every non-glare cell", flush=True)
    _build("glare", todo, args.workers)
    return 0


def cmd_build_causal(args) -> int:
    todo = [(c, d, _needed(d, c, _gate_n4_stats)) for c in CAUSAL_CONDS for d in ttbn.DEV_DRAWS
            if not all(out_path("causal", k, c, d).is_file() for k in ttbn.SEEDS)]
    todo = [t for t in todo if t[2].any()]
    _build("causal", todo, args.workers)
    return 0


# ------------------------------------------------------------------ scoring
def _stream(path: Path, plain, gate):
    """Adapted records where `gate` fires, plain elsewhere (as `ttbn.gated_stream`)."""
    if not np.any(gate):
        return plain
    with open(path, "rb") as fh:
        rr = pickle.load(fh)["records"]
    out = []
    for a, b, f in zip(rr, plain, gate):
        if not f:
            out.append(b)
            continue
        if a is None:
            raise SystemExit(f"gated frame without an adapted record ({path})")
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
    irok = ~np.asarray(ctx.ir_night_raw, bool) & (np.asarray(ctx.ir_d2) <= ctx.ir_bound)
    res = {}
    for c in conds:
        g4 = ttbn.gate_ctx(ctx, c)
        gg = ((np.asarray(ctx.q_vis_by_cond[c]) < 1.0) | glare_term(ttbn._dev_stats(draw, "vis", c, "brightness"))) & irok
        if c != "glare" and not np.array_equal(gg, g4):
            raise SystemExit(f"N4g's gate differs from N4's outside glare ({draw}, {ir_cond}, {k}, {c})")
        if not np.array_equal(gg, gate_g_stats(draw, c, ir_cond)):
            raise SystemExit(f"N4g's context gate differs from its frame-statistics gate ({draw}, {ir_cond}, {k}, {c})")
        vis = ctx.vis_by_cond[c]
        dd = None if c == "clean" else draw
        if c == "glare":
            vis_g = _stream(out_path("glare", k, c, draw), vis, gg)
        else:
            vis_g = ttbn.gated_stream("dev", k, c, dd, vis, gg)
        vis_c = vis if (c == "clean" or not g4.any()) else _stream(out_path("causal", k, c, draw), vis, g4)
        parts = {"union": run_systems(u, c, compute_metrics=False)["fused_gated"],
                 "union_ttbng": run_systems(u, c, compute_metrics=False, vis_records=vis_g)["fused_gated"],
                 "union_ttbnc": run_systems(u, c, compute_metrics=False, vis_records=vis_c)["fused_gated"],
                 "vis_ttbng": vis_g, "vis_ttbnc": vis_c}
        parts = {n: frame_parts(r, ctx.gts) for n, r in parts.items()}
        for s, idx in sl.items():
            res[f"{c}|{s}"] = {n: ship_ap(p, idx) for n, p in parts.items()}
            res[f"{c}|{s}"]["gate_g"] = float(gg[idx].mean())
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
    out = {}
    L = ["# N4 variants on development: N4g (glare term) and N4c (causal statistics)", "",
         "`docs/prereg-ttbn-variants-2026-10-11.md`. Draws VIS 942–944 / IR 952–954, five systems. Δ = arm − "
         "`union` (ship AP), mean over the three draws per system, then over the five systems; 95% t-interval "
         "(df 4). `N4c − N4` = `union_ttbnc` − `union_ttbn` (N4's development scores), descriptive.", ""]
    rows = []
    worse_g, better_g, worse_c, better_c = [], [], [], []
    for c in ttbn.DEV_CONDS:
        for ic in ttbn.DEV_IR_CONDS:
            for s in ("day", "night"):
                pk = {n: [] for n in ("g", "c", "c4", "gate_g", "gate")}
                for k in ttbn.SEEDS:
                    rs = [res[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in ttbn.DEV_DRAWS]
                    r4 = [n4[f"{d}|{ic}|{k}"][f"{c}|{s}"] for d in ttbn.DEV_DRAWS]
                    pk["g"].append(np.mean([r["union_ttbng"] - r["union"] for r in rs]))
                    pk["c"].append(np.mean([r["union_ttbnc"] - r["union"] for r in rs]))
                    pk["c4"].append(np.mean([r["union_ttbnc"] - q["union_ttbn"] for r, q in zip(rs, r4)]))
                    pk["gate_g"].append(np.mean([r["gate_g"] for r in rs]))
                    pk["gate"].append(np.mean([r["gate"] for r in rs]))
                g, glo, ghi = ttbn._cell_stats(pk["g"])
                cc, clo, chi = ttbn._cell_stats(pk["c"])
                c4, c4lo, c4hi = ttbn._cell_stats(pk["c4"])
                cell = f"{c}/{ic or 'clean'}/{s}"
                out[cell] = {"n4g": [g, glo, ghi], "n4c": [cc, clo, chi], "n4c_minus_n4": [c4, c4lo, c4hi],
                             "gate_g": float(np.mean(pk["gate_g"])), "gate": float(np.mean(pk["gate"]))}
                if g <= -0.0014 and ghi < 0:
                    worse_g.append(cell)
                if c == "glare" and g >= 0.0060 and glo > 0:
                    better_g.append(cell)
                if cc <= -0.0014 and chi < 0:
                    worse_c.append(cell)
                if cc >= 0.0060 and clo > 0:
                    better_c.append(cell)
                rows.append(f"| {cell} | {out[cell]['gate_g']:.3f} | {g:+.4f} [{glo:+.4f}, {ghi:+.4f}] | "
                            f"{out[cell]['gate']:.3f} | {cc:+.4f} [{clo:+.4f}, {chi:+.4f}] | {c4:+.4f} [{c4lo:+.4f}, {c4hi:+.4f}] |")
    vg = "PASSES" if (not worse_g and better_g) else "FAILS"
    vc = "PASSES" if (not worse_c and better_c) else "FAILS"
    L += [f"**N4g {vg} development.** G-H: {'PASS' if not worse_g else 'FAIL (' + ', '.join(worse_g) + ')'}. "
          f"G-G: {'PASS (' + ', '.join(better_g) + ')' if better_g else 'FAIL'}.  ",
          f"**N4c {vc} development.** C-H: {'PASS' if not worse_c else 'FAIL (' + ', '.join(worse_c) + ')'}. "
          f"C-G: {'PASS (' + str(len(better_c)) + ' cells: ' + ', '.join(better_c) + ')' if better_c else 'FAIL'}.", "",
          "| cell (VIS/IR/slice) | N4g gate | N4g Δ vs union [95%] | N4 gate | N4c Δ vs union [95%] | N4c − N4 [95%] |",
          "|---|---:|---:|---:|---:|---:|"] + rows
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"N4g": vg, "N4c": vc, "worse_g": worse_g, "better_g": better_g,
                                                   "worse_c": worse_c, "better_c": better_c, "cells": out}, indent=1),
                                      encoding="utf-8")
    print("\n".join(L[:7]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    for name in ("build-glare", "build-causal"):
        b = sp.add_parser(name)
        b.add_argument("--workers", type=int, default=4)
    s = sp.add_parser("score-dev")
    s.add_argument("--workers", type=int, default=3)
    r = sp.add_parser("report-dev")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_variants_dev_2026-10-11.md")
    args = ap.parse_args()
    return {"build-glare": cmd_build_glare, "build-causal": cmd_build_causal, "score-dev": cmd_score_dev,
            "report-dev": cmd_report_dev}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
