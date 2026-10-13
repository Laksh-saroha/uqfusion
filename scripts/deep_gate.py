"""Deep-feature trust (E7, 2026-10-10): LLF-2's gate learner, also shown the frozen backbones' embeddings.

`docs/prereg-no-retrain-2026-10-10.md` Amendment 2. LLF-2 (`llf.py`, variant `trust`) learns a
per-box score from what a gate may use: stream, own confidence (monotone up), cross-stream support
(monotone up), duplication (monotone down) and 16 frame-health image statistics. E7 adds what the
detectors themselves see: the 1,280-d pooled backbone embedding each checkpoint already writes for
every frame (`feat`, the vector the Mahalanobis scorer reads), reduced to 16 principal components
per stream. The PCA is fitted on the checkpoint's own clean TRAIN frames, so the components belong
to one network; the model is therefore fitted per system (seed k) on that system's rows.

Model: XGBoost regressor on the GPU (`device="cuda"`), fixed hyperparameters, the LLF credit target,
the same 20% training rows of draw 941 / 951, monotone constraints as LLF-2. Folds: leave one run out
(primary; the night run is scored by a model that never saw it) and LLF's four night blocks
(secondary). Evaluation on draws 945-947 / 955-957 with LLF-2's P1/P2. Prediction runs on the CPU
inside the pool workers so no worker opens a CUDA context (feedback-vram-spill).

    py -3.13 scripts/deep_gate.py extract
    py -3.13 scripts/deep_gate.py fit
    py -3.13 scripts/deep_gate.py eval --draws 945 946 947 --workers 6
    py -3.13 scripts/deep_gate.py report --draws 945 946 947
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

import llf                                                    # noqa: E402

PCA_K = 16
TRUST = llf.VARIANT_FEATS["trust"]
DEEP = [f"dv{j}" for j in range(PCA_K)] + [f"di{j}" for j in range(PCA_K)]
FEATS = TRUST + DEEP
MONO = tuple(llf.MONO.get(f, 0) for f in FEATS)
XGB = dict(n_estimators=500, learning_rate=0.05, max_depth=8, min_child_weight=200.0, reg_lambda=1.0,
           tree_method="hist", random_state=0)
OUT = ROOT / "runs/deep_gate"
SEEDS = llf.SEEDS


def _feats(path) -> np.ndarray:
    from uqfusion.eval.cache import load_cache
    r, _ = load_cache(path)
    return np.stack([np.asarray(x["feat"], np.float32).ravel() for x in r])


def pca(k: int) -> dict:
    """Per-stream PCA of seed k's clean TRAIN frame embeddings (cached)."""
    p = OUT / f"pca_seed{k}.npz"
    if p.is_file():
        z = np.load(p)
        return {m: (z[f"{m}_mu"], z[f"{m}_W"]) for m in ("vis", "ir")}
    out = {}
    for m in ("vis", "ir"):
        F = _feats(ROOT / f"runs/cache_p3/seed{k}/gauss_{m}_train_clean.pkl").astype(np.float64)
        rng = np.random.default_rng(0)
        if len(F) > 20000:
            F = F[rng.choice(len(F), 20000, replace=False)]
        mu = F.mean(0)
        _, s, vt = np.linalg.svd(F - mu, full_matrices=False)
        W = vt[:PCA_K].T / np.maximum(s[:PCA_K] / np.sqrt(len(F)), 1e-9)   # whitened components
        out[m] = (mu, W)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez(p, **{f"{m}_mu": out[m][0] for m in out}, **{f"{m}_W": out[m][1] for m in out})
    return out


def deep_block(k: int, draw: int, c: str, ic, P=None) -> np.ndarray:
    """(n_frames, 32) projected embeddings of the VIS frame under condition c and the IR frame under ic."""
    P = P or pca(k)
    base = ROOT / llf.CACHE / f"seed{k}" / llf.sub(draw)
    fv = _feats(base / f"gauss_vis_paired_{c}.pkl")
    fi = _feats(base / f"gauss_ir_paired_{ic or 'clean'}.pkl")
    return np.hstack([(fv - P["vis"][0]) @ P["vis"][1], (fi - P["ir"][0]) @ P["ir"][1]]).astype(np.float64)


# ---------------------------------------------------------------- extract (joins LLF's training rows)
def _extract_job(k):
    P = pca(k)
    cols = [llf.FEATS.index(f) for f in TRUST]
    out_X, out_y, out_m = [], [], []
    for ic in llf.IR_CONDS:
        z = np.load(llf.OUT / "train" / f"seed{k}_{ic or 'clean'}.npz")
        X, y, meta = z["X"], z["y"], z["meta"]
        assert (meta[:, 0] == k).all()
        D = np.zeros((len(X), 2 * PCA_K))
        for ci, c in enumerate(llf.VIS_CONDS):
            m = meta[:, 1] == ci
            if m.any():
                D[m] = deep_block(k, llf.TRAIN_DRAW, c, ic, P)[meta[m, 5]]
        out_X.append(np.hstack([X[:, cols], D]))
        out_y.append(y)
        out_m.append(meta)
    p = OUT / "train" / f"seed{k}.npz"
    p.parent.mkdir(parents=True, exist_ok=True)
    np.savez(p, X=np.concatenate(out_X), y=np.concatenate(out_y), meta=np.concatenate(out_m))
    return k, sum(len(x) for x in out_X)


def cmd_extract(args) -> int:
    for k in SEEDS:          # PCA first, serially (one SVD per stream per seed)
        pca(k)
    with ProcessPoolExecutor(args.workers) as ex:
        for k, n in ex.map(_extract_job, SEEDS):
            print(f"[deep] extract seed{k}: {n:,} rows", flush=True)
    return 0


def folds() -> dict:
    night = llf.RUNS.index(llf.NIGHT_RUNS[0])
    sp = {f"loro_{run}": (lambda m, r=r: m[:, 3] != r) for r, run in enumerate(llf.RUNS)}
    sp.update({f"nb_{b}": (lambda m, b=b: (m[:, 3] != night) | (m[:, 4] != b)) for b in range(llf.NIGHT_BLOCKS)})
    return sp


def cmd_fit(args) -> int:
    import xgboost as xgb
    t0 = time.time()
    (OUT / "models").mkdir(parents=True, exist_ok=True)
    for k in SEEDS:
        z = np.load(OUT / "train" / f"seed{k}.npz")
        X, y, meta = z["X"], z["y"], z["meta"]
        for name, f in folds().items():
            out = OUT / "models" / f"seed{k}_{name}.json"
            if out.is_file() and not args.force:
                continue
            sel = f(meta)
            m = xgb.XGBRegressor(device="cuda", monotone_constraints=MONO, **XGB)
            m.fit(X[sel], y[sel])
            m.save_model(out)
            print(f"[deep] fit seed{k} {name}: {int(sel.sum()):,} rows ({time.time() - t0:.0f}s)", flush=True)
    return 0


# ---------------------------------------------------------------- eval
_M: dict = {}


def _model(k, name):
    import xgboost as xgb
    if (k, name) not in _M:
        m = xgb.XGBRegressor()
        m.load_model(OUT / "models" / f"seed{k}_{name}.json")
        m.set_params(device="cpu", n_jobs=2)
        _M[(k, name)] = m
    return _M[(k, name)]


def _eval_job(job):
    k, draw, ic = job
    llf.wait_ready(k, draw, ic)
    ctx = llf.load(k, draw, ic)
    n = len(ctx.runs)
    H = llf._per_frame_h(ctx.h_frames, n)
    night = np.isin(ctx.runs, llf.NIGHT_RUNS)
    sl = {"day": np.flatnonzero(~night), "night": np.flatnonzero(night)}
    blk = llf.night_block(ctx)
    cols = [llf.FEATS.index(f) for f in TRUST]
    P = pca(k)
    out = {}
    for c in llf.VIS_CONDS:
        r = llf.run_systems(ctx, c, compute_metrics=False)
        base = {"shipped": llf.frame_parts(r["fused_gated"], ctx.gts), "vis": llf.frame_parts(ctx.vis_by_cond[c], ctx.gts),
                "ir": llf.frame_parts(r["ir_in_vis"], ctx.gts)}
        fs = llf.frame_stats(ctx, draw, c, ic)
        D = deep_block(k, draw, c, ic, P)
        cands, Xs, Bs = [], [], []
        for i, (vr, ir) in enumerate(zip(ctx.vis_by_cond[c], ctx.ir_clean)):
            cd = llf.candidates(vr, ir, H[i])
            X, B, _ = llf.frame_features(cd, fs[i], vr["image_hw"])
            Xs.append(np.hstack([X[:, cols], np.repeat(D[i][None], len(X), 0)]) if len(X) else np.zeros((0, len(FEATS))))
            cands.append(cd)
            Bs.append(B)
        hws = [vr["image_hw"] for vr in ctx.vis_by_cond[c]]
        lens = np.array([len(x) for x in Xs])
        starts = np.concatenate([[0], np.cumsum(lens)])
        Xall = np.concatenate(Xs)

        def scores(names):
            s = np.zeros(len(Xall))
            names = np.array(names)
            for nm in np.unique(names):
                rows = np.concatenate([np.arange(starts[i], starts[i + 1]) for i in np.flatnonzero(names == nm)])
                if len(rows):
                    s[rows] = _model(k, nm).predict(Xall[rows])
            return [np.clip(s[starts[i]:starts[i + 1]], 1e-6, None) + 1e-7 * Xs[i][:, 1] for i in range(n)]

        parts = dict(base)
        parts["deep"] = llf.frame_parts(llf._fused(cands, Bs, scores([f"loro_{ru}" for ru in ctx.runs]), hws), ctx.gts)
        parts["deep_nb"] = llf.frame_parts(llf._fused(cands, Bs, scores(
            [(f"nb_{blk[i]}" if night[i] else f"loro_{ctx.runs[i]}") for i in range(n)]), hws), ctx.gts)
        for s, idx in sl.items():
            out[f"{c}|{s}"] = {a: llf.ship_ap(p, idx) for a, p in parts.items()}
    return job, out


def cmd_eval(args) -> int:
    os.environ.setdefault("OMP_NUM_THREADS", "2")
    out_p = OUT / f"eval_{args.tag}.json"
    done = json.loads(out_p.read_text(encoding="utf-8")) if out_p.is_file() else {}
    jobs = [(k, d, ic) for d in args.draws for ic in llf.IR_CONDS for k in SEEDS if f"{k}|{d}|{ic}" not in done]
    print(f"[deep] {len(jobs)} eval jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for n, (job, res) in enumerate(ex.map(_eval_job, jobs), 1):
            done["|".join(map(str, job))] = res
            tmp = out_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(done), encoding="utf-8")
            os.replace(tmp, out_p)
            print(f"[deep] {n}/{len(jobs)} {job} ({time.time() - t0:.0f}s)", flush=True)
    return 0


def cmd_report(args) -> int:
    raw = json.loads((OUT / f"eval_{args.tag}.json").read_text(encoding="utf-8"))

    def ps(c, ic, s, a):
        return np.array([np.mean([raw[f"{k}|{d}|{ic}"][f"{c}|{s}"][a] for d in args.draws]) for k in SEEDS])

    def ci(x):
        m, h = float(np.mean(x)), llf.T_975_DF4 * float(np.std(x, ddof=1)) / np.sqrt(len(x))
        return m, m - h, m + h

    fail = lambda x: x[0] <= -0.0060 and x[2] < 0              # noqa: E731
    cells = [(c, ic, s) for ic in llf.IR_CONDS for c in llf.VIS_CONDS for s in ("day", "night")]
    lines = [f"# Deep-feature trust (E7), `{args.tag}` (draws {', '.join(map(str, args.draws))})", "",
             "`scripts/deep_gate.py`, registered in `docs/prereg-no-retrain-2026-10-10.md` Amendment 2. Ship AP, five "
             "systems, draw mean per system, 95% t-interval (df 4). P1 = claim (fused − max(VIS, IR)) not failing on any "
             "cell; P2 = no cell worse than shipped at ≤ −0.0060 with the interval below zero, and at least one better.", "",
             "| arm | P1 holds on | P2 worse cells | P2 better cells | worse at 0.0014 | mean fused | verdict |",
             "|---|---|---:|---:|---:|---|---|"]
    summ = {}
    for a in ("shipped", "deep", "deep_nb"):
        hold = worse = better = w14 = 0
        means = []
        for c, ic, s in cells:
            arm = ps(c, ic, s, a)
            mx = np.maximum(ps(c, ic, s, "vis"), ps(c, ic, s, "ir"))
            hold += not fail(ci(arm - mx))
            d = ci(arm - ps(c, ic, s, "shipped"))
            worse += d[0] <= -0.0060 and d[2] < 0
            better += d[0] >= 0.0060 and d[1] > 0
            w14 += d[0] <= -0.0014 and d[2] < 0
            means.append(arm.mean())
            summ[f"{a}|{c}|{ic}|{s}"] = {"fused": float(arm.mean()), "max": float(mx.mean()), "d_shipped": d}
        p1, p2 = hold == len(cells), (a != "shipped" and worse == 0 and better >= 1)
        summ[f"{a}|summary"] = {"P1_hold": int(hold), "P2_worse": int(worse), "P2_better": int(better), "worse_0014": int(w14),
                                "P1": bool(p1), "P2": bool(p2), "mean_fused": float(np.mean(means))}
        lines.append(f"| {a} | {hold}/{len(cells)} | {worse} | {better} | {w14} | {np.mean(means):.4f} | "
                     f"{'—' if a == 'shipped' else ('PASS' if p1 and p2 else 'FAIL')} |")
    lines += ["", "| VIS | IR | slice | VIS | IR | shipped | deep | deep_nb |", "|---|---|---|---:|---:|---:|---:|---:|"]
    for c, ic, s in cells:
        lines.append(f"| {c} | {ic or 'clean'} | {s} | " + " | ".join(f"{ps(c, ic, s, a).mean():.4f}" for a in
                                                                        ("vis", "ir", "shipped", "deep", "deep_nb")) + " |")
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(summ, indent=1), encoding="utf-8")
    print("\n".join(lines[6:9]))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    e = sp.add_parser("extract")
    e.add_argument("--workers", type=int, default=5)
    f = sp.add_parser("fit")
    f.add_argument("--force", action="store_true")
    v = sp.add_parser("eval")
    v.add_argument("--draws", type=int, nargs="+", required=True)
    v.add_argument("--tag", default="fresh")
    v.add_argument("--workers", type=int, default=6)
    r = sp.add_parser("report")
    r.add_argument("--draws", type=int, nargs="+", required=True)
    r.add_argument("--tag", default="fresh")
    r.add_argument("--out", default="docs/eval/no_retrain/deep_gate_fresh.md")
    args = ap.parse_args()
    return {"extract": cmd_extract, "fit": cmd_fit, "eval": cmd_eval, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
