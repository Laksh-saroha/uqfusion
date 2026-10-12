"""Gate zoo (2026-10-10): three more ways to gate the fusion, beside shipped and no gate.

`docs/prereg-no-retrain-2026-10-10.md` Amendment 2 (E6). No detector is retrained; every arm reads
the shipped caches.

  shipped   `crossmodal26m` as is
  union     the veto switched off (E4)
  selfconf  OUTPUT-space health instead of image statistics: a stream is vetoed on a frame iff its
            most confident ship box scores < 0.10 while the other stream's scores >= 0.25 (fixed
            round numbers, not tuned). It asks the detector whether it sees anything, which is the
            question the night veto answers with brightness.
  calunion  each stream's ship confidences mapped to P(true positive at IoU 0.5) by an isotonic
            calibrator fitted per system, per stream and per health state (the shipped gate's own
            degradation flags: VIS veil OR dark, IR ir_d2 > ir_bound) on draw 941, leave-one-run-out;
            then the two streams are simply concatenated. A degraded stream is down-weighted by its
            own measured reliability instead of being deleted.
  proben    calunion, plus ProbEn-style Bayesian fusion (Chen et al., ECCV 2022) of cross-stream
            pairs: VIS and IR ship boxes matched greedily at IoU >= 0.30 become one box (VIS
            coordinates) with logit p = logit p_vis + logit p_ir - logit prior.

Amendment 4 (E9), tiered (lexicographic) concatenation. Matching is greedy in confidence order and the
precision envelope is a running max from the tail, so boxes appended BELOW every existing box can
neither steal a match nor lower any earlier precision: appending a tail never lowers AP (up to an
np.interp tie when a 101-grid recall equals a duplicated recall value). Three arms use that:

  lex       every VIS ship box ranks above every IR ship box (VIS 1 + c, IR 0.999 c): fused AP >= VIS
            AP in every cell by construction. No parameter, no gate.
  demote    the shipped gate, but a vetoed stream is DEMOTED to the tail instead of deleted: the shipped
            fused output is the first tier, the vetoed stream's ship boxes the second. AP >= shipped AP
            in every cell by construction.

Not registered, measured on the smoke draw 941 only: `udemote` (union where IR is healthy, VIS first and
IR as the tail where the IR-health flag fires). It lost to both union and lex under IR fog s1 (clean
day 0.3225 vs 0.3553 / 0.3514): switching between fusion outputs frame by frame puts the switched
frames' scores on a different scale, and AP pools every frame into one ranking.

    py -3.13 scripts/gate_zoo.py fit --workers 6
    py -3.13 scripts/gate_zoo.py run --draws 945 946 947 --workers 6
    py -3.13 scripts/gate_zoo.py report --draws 945 946 947
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

VIS_CONDS = ("clean", "fog", "fog_s1", "lowlight", "glare")
IR_CONDS = (None, "fog_s1", "fog_s2", "fog_s3", "noise_s2", "noise_s3")
RUNS = ("pohang00", "pohang01", "pohang02", "pohang03")
SEEDS = (0, 1, 2, 3, 4)
ARMS = ("shipped", "union", "selfconf", "calunion", "proben", "lex", "demote")
FIT_DRAW = 941
SC_LOW, SC_HIGH = 0.10, 0.25
PAIR_IOU = 0.30
T_975_DF4 = 2.7764451051977987
CACHE, STATS = "runs/cache_p3dev_v2", "runs/derived_p3dev_v2"
OUT = ROOT / "runs/gate_zoo"


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def load(k, draw, ic, conds=VIS_CONDS):
    from uqfusion.eval.ctx import load_context
    d = f"{STATS}/brightness/{sub(draw)}"
    return load_context(preset="crossmodal26m", cache_dir=f"{CACHE}/seed{k}/{sub(draw)}", bright_dir=d,
                        structure_dir=f"{STATS}/structure/{sub(draw)}",
                        ir_bright=f"{d}/gauss_ir_paired_{ic or 'clean'}.json",
                        conditions=conds, ir_condition=ic, verbose=False)


def flags(ctx):
    thr = float(ctx.struct_const["axes"]["grad_gini"]["threshold"])
    v = {c: np.asarray((ctx.gini_by_cond[c] < thr) | (ctx.bright_by_cond[c] < float(ctx.c_vis.mu_b)), bool)
         for c in ctx.conditions}
    return v, np.asarray(ctx.ir_d2 > ctx.ir_bound, bool)


def ir_in_vis(ctx):
    from uqfusion.uq.fusion import apply_homography
    return [{**r, "boxes_xyxy": apply_homography(np.asarray(r["boxes_xyxy"]).reshape(-1, 4), h)}
            for r, h in zip(ctx.ir_clean, ctx.h_frames)]


def ship_rows(recs, gts):
    """Per frame: (conf, tp@0.5) of ship boxes."""
    from uqfusion.eval.matching import tp_matrix
    out = []
    for r, g in zip(recs, gts):
        cls = np.asarray(r["cls"]).astype(int).ravel()
        c = np.asarray(r["conf"], float).ravel()
        if len(c):
            tp = tp_matrix({"boxes_xyxy": np.asarray(r["boxes_xyxy"], float).reshape(-1, 4), "conf": c, "cls": cls}, g)[:, 0]
        else:
            tp = np.zeros(0, bool)
        m = cls == 0
        out.append((c[m], tp[m].astype(np.float64)))
    return out


# ---------------------------------------------------------------- calibration fit
def _fit_job(job):
    k, ic = job
    ctx = load(k, FIT_DRAW, ic)
    vflag, iflag = flags(ctx)
    irv = ir_in_vis(ctx)
    rows = []                     # (stream, health, run, conf, tp)
    ir_rows = ship_rows(irv, ctx.gts)
    for c in VIS_CONDS:
        vr = ship_rows(ctx.vis_by_cond[c], ctx.gts)
        for i, ((cv, tv), (ci, ti)) in enumerate(zip(vr, ir_rows)):
            ru = RUNS.index(ctx.runs[i])
            if len(cv):
                rows.append(np.column_stack([np.zeros(len(cv)), np.full(len(cv), vflag[c][i]), np.full(len(cv), ru), cv, tv]))
            if len(ci) and c == "clean":          # the IR stream does not depend on the VIS condition
                rows.append(np.column_stack([np.ones(len(ci)), np.full(len(ci), iflag[i]), np.full(len(ci), ru), ci, ti]))
    return k, ic, np.concatenate(rows)


def cmd_fit(args) -> int:
    from sklearn.isotonic import IsotonicRegression
    t0 = time.time()
    data = {k: [] for k in SEEDS}
    with ProcessPoolExecutor(args.workers) as ex:
        for k, ic, R in ex.map(_fit_job, [(k, ic) for k in SEEDS for ic in IR_CONDS]):
            data[k].append(R)
            print(f"[fit] seed{k} {ic}: {len(R):,} rows ({time.time() - t0:.0f}s)", flush=True)
    cal = {}
    for k in SEEDS:
        R = np.concatenate(data[k])
        for ru in range(len(RUNS)):
            tr = R[R[:, 2] != ru]
            prior = float(np.clip(tr[:, 4].mean(), 1e-4, 1 - 1e-4))
            cal[(k, RUNS[ru], "prior")] = prior
            for s in (0, 1):
                for h in (0, 1):
                    m = (tr[:, 0] == s) & (tr[:, 1] == h)
                    if m.sum() < 50:               # too few rows: fall back to the stream's pooled fit
                        m = tr[:, 0] == s
                    iso = IsotonicRegression(y_min=0.0, y_max=1.0, increasing=True, out_of_bounds="clip")
                    iso.fit(tr[m, 3], tr[m, 4])
                    cal[(k, RUNS[ru], s, h)] = (np.asarray(iso.X_thresholds_), np.asarray(iso.y_thresholds_), int(m.sum()))
        print(f"[fit] seed{k}: {len(R):,} rows, TP rate {R[:, 4].mean():.3%}", flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "calibrators.pkl", "wb") as fh:
        pickle.dump(cal, fh)
    print(f"[fit] wrote {OUT / 'calibrators.pkl'} ({time.time() - t0:.0f}s)", flush=True)
    return 0


# ---------------------------------------------------------------- eval
def _cal(cal, k, run, s, h, conf):
    x, y, _ = cal[(k, run, s, int(h))]
    p = np.interp(conf, x, y) if len(x) else np.zeros_like(conf)
    return np.clip(p, 1e-4, 1 - 1e-4)


def _logit(p):
    return np.log(p) - np.log1p(-p)


def _iou(a, b):
    from uqfusion.eval.matching import iou_matrix
    return iou_matrix(a, b)


def build_cal_records(ctx, c, irv, vflag, iflag, cal, k, bayes: bool):
    out = []
    for i, (vr, ir) in enumerate(zip(ctx.vis_by_cond[c], irv)):
        run = ctx.runs[i]
        vb = np.asarray(vr["boxes_xyxy"], float).reshape(-1, 4)
        vc = np.asarray(vr["conf"], float).ravel()
        vcl = np.asarray(vr["cls"]).astype(int).ravel()
        ib = np.asarray(ir["boxes_xyxy"], float).reshape(-1, 4)
        icf = np.asarray(ir["conf"], float).ravel()
        icl = np.asarray(ir["cls"]).astype(int).ravel()
        vs, iss = vcl == 0, icl == 0
        pv = _cal(cal, k, run, 0, vflag[i], vc[vs])
        pi = _cal(cal, k, run, 1, iflag[i], icf[iss])
        VB, IB = vb[vs], ib[iss]
        sv = _logit(pv) + 1e-6 * vc[vs]
        si = _logit(pi) + 1e-6 * icf[iss]
        keep_i = np.ones(len(IB), bool)
        if bayes and len(VB) and len(IB):
            lp = _logit(np.asarray(cal[(k, run, "prior")]))
            M = _iou(VB, IB)
            for a in np.argsort(-sv, kind="stable"):
                cand = np.flatnonzero(keep_i & (M[a] >= PAIR_IOU))
                if not len(cand):
                    continue
                b = cand[np.argmax(M[a, cand])]
                sv[a] = _logit(pv[a]) + _logit(pi[b]) - lp + 1e-6 * vc[vs][a]
                keep_i[b] = False
        # logits are monotone in p, so ranking by logit == ranking by probability
        boxes = np.vstack([VB, IB[keep_i], vb[~vs]]) if len(vb) + len(ib) else np.zeros((0, 4))
        conf = np.concatenate([sv, si[keep_i], vc[~vs]])   # buoys: VIS only, raw (ship AP is per class)
        cls = np.concatenate([np.zeros(len(VB), int), np.zeros(int(keep_i.sum()), int), vcl[~vs]])
        out.append({"boxes_xyxy": boxes.reshape(-1, 4), "conf": conf, "cls": cls, "image_hw": vr["image_hw"]})
    return out


def _ships(r):
    cl = np.asarray(r["cls"]).astype(int).ravel()
    m = cl == 0
    return np.asarray(r["boxes_xyxy"], float).reshape(-1, 4)[m], np.asarray(r["conf"], float).ravel()[m]


def tiered(head, tail):
    """Per frame: every box of `head` (all classes) above every ship box of `tail` (None = no tail)."""
    out = []
    for h, t in zip(head, tail):
        hb = np.asarray(h["boxes_xyxy"], float).reshape(-1, 4)
        hc = np.asarray(h["conf"], float).ravel()      # fused scores can exceed 1 (support bonus): never clip
        hl = np.asarray(h["cls"]).astype(int).ravel()
        tb, tc = _ships(t) if t is not None else (np.zeros((0, 4)), np.zeros(0))
        out.append({"boxes_xyxy": np.vstack([hb, tb]).reshape(-1, 4),
                    "conf": np.concatenate([1.0 + np.maximum(hc, 0.0), 0.999 * np.clip(tc, 0, 1)]),
                    "cls": np.concatenate([hl, np.zeros(len(tb), int)]), "image_hw": h.get("image_hw")})
    return out


def build_lex_records(vis_recs, irv, ir_first):
    """Tiered concatenation of two streams: the first tier's boxes all outrank the second tier's ships."""
    head = [ir if f else vr for vr, ir, f in zip(vis_recs, irv, ir_first)]
    tail = [vr if f else ir for vr, ir, f in zip(vis_recs, irv, ir_first)]
    out = tiered(head, tail)
    for o, vr in zip(out, vis_recs):
        o["image_hw"] = vr["image_hw"]
    return out


def build_demote_records(out, vis_recs):
    """The shipped output first; the ship boxes of whichever stream it vetoed on a frame as the tail."""
    vv, iv = np.asarray(out["veto_vis"], bool), np.asarray(out["veto_ir"], bool)
    tails = []
    for vr, ir, a, b in zip(vis_recs, out["ir_in_vis"], vv, iv):
        parts = ([vr] if a else []) + ([ir] if b else [])
        if not parts:
            tails.append(None)
            continue
        tb = [_ships(x) for x in parts]
        tails.append({"boxes_xyxy": np.vstack([x[0] for x in tb]), "conf": np.concatenate([x[1] for x in tb]),
                      "cls": np.zeros(sum(len(x[1]) for x in tb), int)})
    res = tiered(out["fused_gated"], tails)
    for o, vr in zip(res, vis_recs):
        o["image_hw"] = vr["image_hw"]
    return res


def _run_job(job):
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, run_systems
    from v2_sensitivity import ship_ap
    draw, ic, k = job
    with open(OUT / "calibrators.pkl", "rb") as fh:
        cal = pickle.load(fh)
    ctx = load(k, draw, ic)
    vflag, iflag = flags(ctx)
    irv = ir_in_vis(ctx)
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    uctx = dataclasses.replace(ctx, veto=None)
    imax = np.array([max([float(x) for x, cl in zip(np.ravel(r["conf"]), np.ravel(r["cls"])) if int(cl) == 0] or [0.0])
                     for r in ctx.ir_clean])
    res, rates = {}, {}
    pir = frame_parts(irv, ctx.gts)
    for c in VIS_CONDS:
        vrec = ctx.vis_by_cond[c]
        vmax = np.array([max([float(x) for x, cl in zip(np.ravel(r["conf"]), np.ravel(r["cls"])) if int(cl) == 0] or [0.0])
                         for r in vrec])
        vv = (vmax < SC_LOW) & (imax >= SC_HIGH)
        iv = (imax < SC_LOW) & (vmax >= SC_HIGH)
        sh = run_systems(ctx, c, compute_metrics=False)
        un = run_systems(uctx, c, compute_metrics=False)["fused_gated"]
        recs = {"shipped": sh["fused_gated"],
                "union": un,
                "selfconf": run_systems(ctx, c, compute_metrics=False, veto_override=(vv.tolist(), iv.tolist()))["fused_gated"],
                "calunion": build_cal_records(ctx, c, irv, vflag[c], iflag, cal, k, bayes=False),
                "proben": build_cal_records(ctx, c, irv, vflag[c], iflag, cal, k, bayes=True),
                "lex": build_lex_records(vrec, irv, np.zeros(len(vrec), bool)),
                "demote": build_demote_records(sh, vrec)}
        parts = {a: frame_parts(r, ctx.gts) for a, r in recs.items()}
        parts["vis"], parts["ir"] = frame_parts(vrec, ctx.gts), pir
        for s, idx in sl.items():
            res[f"{c}|{s}"] = {a: ship_ap(p, idx) for a, p in parts.items()}
            rates[f"{c}|{s}"] = {"sc_veto_vis": float(vv[idx].mean()), "sc_veto_ir": float(iv[idx].mean()),
                                 "vis_unhealthy": float(vflag[c][idx].mean()), "ir_unhealthy": float(iflag[idx].mean()),
                                 "veto_vis": float(np.asarray(sh["veto_vis"], bool)[idx].mean()),
                                 "veto_ir": float(np.asarray(sh["veto_ir"], bool)[idx].mean())}
    return job, res, rates


def cmd_run(args) -> int:
    out_p = OUT / f"run_{args.tag}.json"
    done = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    jobs = [(d, ic, k) for d in args.draws for ic in IR_CONDS for k in SEEDS if f"{d}|{ic}|{k}" not in done]
    print(f"[zoo] {len(jobs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for n, (job, res, rates) in enumerate(ex.map(_run_job, jobs), 1):
            done["|".join(map(str, job))] = {"ap": res, "rates": rates}
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(done), encoding="utf-8")
            os.replace(tmp, out_p)
            print(f"[zoo] {n}/{len(jobs)} {job} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_report(args) -> int:
    raw = json.loads((OUT / f"run_{args.tag}.json").read_text(encoding="utf-8"))

    def ps(c, ic, s, a):
        return np.array([np.mean([raw[f"{d}|{ic}|{k}"]["ap"][f"{c}|{s}"][a] for d in args.draws]) for k in SEEDS])

    def ci(x):
        m, h = float(np.mean(x)), T_975_DF4 * float(np.std(x, ddof=1)) / np.sqrt(len(x))
        return m, m - h, m + h

    fail = lambda x: x[0] <= -0.0060 and x[2] < 0              # noqa: E731
    cells = [(c, ic, s) for ic in IR_CONDS for c in VIS_CONDS for s in ("day", "night")]
    lines = [f"# Gate zoo, `{args.tag}` (draws {', '.join(map(str, args.draws))})", "",
             "`scripts/gate_zoo.py`, registered in `docs/prereg-no-retrain-2026-10-10.md` Amendment 2 (E6). Ship AP, "
             "five Phase 3 systems, per system the draw mean, 95% t-interval over systems (df 4). Δ = arm − shipped. "
             "IMPROVES = no cell worse at 0.0014 with the interval below zero and at least one better at 0.0060 with "
             "the interval above zero. Claim = fused − max(VIS, IR) failing at ≤ −0.0060 with the interval below zero.", "",
             "| arm | IMPROVES | better ≥0.0060 | worse ≤−0.0014 | worse ≤−0.0060 | worst Δ (cell) | best Δ (cell) | claim fails | mean fused |",
             "|---|---|---:|---:|---:|---|---|---:|---|"]
    summ, detail = {}, ["", "## Per cell (mean fused; Δ vs shipped)", "",
                        "| VIS | IR | slice | VIS | IR | " + " | ".join(ARMS) + " |", "|---|---|---|---:|---:|" + "---|" * len(ARMS)]
    for c, ic, s in cells:
        base = ps(c, ic, s, "shipped")
        row = [f"{ps(c, ic, s, 'vis').mean():.4f}", f"{ps(c, ic, s, 'ir').mean():.4f}", f"{base.mean():.4f}"]
        for a in ARMS[1:]:
            d = ci(ps(c, ic, s, a) - base)
            row.append(f"{ps(c, ic, s, a).mean():.4f} ({d[0]:+.4f})")
        detail.append(f"| {c} | {ic or 'clean'} | {s} | " + " | ".join(row) + " |")
    for a in ARMS:
        ds, fails, means = [], 0, []
        for c, ic, s in cells:
            arm, base = ps(c, ic, s, a), ps(c, ic, s, "shipped")
            ds.append((ci(arm - base), f"{c}/{ic or 'clean'}/{s}"))
            fails += fail(ci(arm - np.maximum(ps(c, ic, s, "vis"), ps(c, ic, s, "ir"))))
            means.append(arm.mean())
        b60 = sum(1 for (m, lo, hi), _ in ds if m >= 0.0060 and lo > 0)
        w14 = sum(1 for (m, lo, hi), _ in ds if m <= -0.0014 and hi < 0)
        w60 = sum(1 for (m, lo, hi), _ in ds if m <= -0.0060 and hi < 0)
        w, b = min(ds, key=lambda x: x[0][0]), max(ds, key=lambda x: x[0][0])
        imp = a != "shipped" and w14 == 0 and b60 >= 1
        summ[a] = {"better_0060": b60, "worse_0014": w14, "worse_0060": w60, "IMPROVES": bool(imp), "claim_fails": int(fails),
                   "worst": [w[1], *w[0]], "best": [b[1], *b[0]], "mean_fused": float(np.mean(means)),
                   "cells": {n: list(x) for x, n in ds}}
        lines.append(f"| {a} | {'—' if a == 'shipped' else ('**yes**' if imp else 'no')} | {b60} | {w14} | {w60} | "
                     f"{w[0][0]:+.4f} ({w[1]}) | {b[0][0]:+.4f} ({b[1]}) | {fails} | {np.mean(means):.4f} |")
    rk = sorted({k for v in raw.values() for k in v["rates"]})
    lines += detail + ["", "## Self-confidence veto rates and health flags (mean over draws, IR conditions, systems)", ""]
    for key in rk:
        vals = [v["rates"][key] for v in raw.values() if key in v["rates"]]
        lines.append(f"* {key}: " + ", ".join(f"{m} {np.mean([x[m] for x in vals]):.1%}" for m in vals[0]))
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("\n".join(lines[6:6 + len(ARMS)]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    f = sp.add_parser("fit")
    f.add_argument("--workers", type=int, default=6)
    r = sp.add_parser("run")
    r.add_argument("--draws", type=int, nargs="+", required=True)
    r.add_argument("--tag", default="fresh")
    r.add_argument("--workers", type=int, default=6)
    p = sp.add_parser("report")
    p.add_argument("--draws", type=int, nargs="+", required=True)
    p.add_argument("--tag", default="fresh")
    p.add_argument("--out", default="docs/eval/no_retrain/gate_zoo_fresh.md")
    args = ap.parse_args()
    return {"fit": cmd_fit, "run": cmd_run, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
