"""Scoring for `docs/prereg-p04-heldout-gatefree-2026-10-11.md` (the builder is `p04_heldout.py`).

    py -3.13 scripts/p04_heldout_score.py devcal                   # ensemble capability priors, development
    py -3.13 scripts/p04_heldout_score.py parts --cells clean      # presorts + observed APs, every arm
    py -3.13 scripts/p04_heldout_score.py parts --cells corrupted --arms shipped union lex ens5_union
    py -3.13 scripts/p04_heldout_score.py boot                     # GPU bootstrap of every presort not yet done
    py -3.13 scripts/p04_heldout_score.py report

Arms and streams. `shipped`, `union` and `lex` share the single-checkpoint streams (`single`); `ens5_union`
uses the five-checkpoint streams (`ens5`); `full_tta` the five-checkpoint, flipped, IR-restored streams
(`full`). Every arm's fused output is scored, and each stream set's VIS-only and IR-only records (IR mapped
into the VIS frame), against the VIS ground truth. Ensemble streams are the same for every system, so
their presort is stored once per draw (key `all`).

Bootstrap. The look's resample (moving blocks, L = 20, seed 1, 1,000 replicates, `resample_weights`), one
weight matrix for every arm, system and draw. `boot_batch` evaluates eight replicates per GPU pass with
the arithmetic of `holdout_p04_v2_rescore._ap_gpu` per replicate and is checked equal (`==`) to the CPU
`_score` before any replicate is used.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
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

import p04_heldout as ph                                                             # noqa: E402

HP = ROOT / "runs" / "holdout_p04"
OUT = HP / "gatefree"
PRE = OUT / "pre"
SEEDS = ph.SEEDS
VIS_DRAWS = ph.VIS_DRAWS
SHIP = 0
ENS_IOU, IR_NMS = 0.55, 0.70
VERDICT_CELL = ("clean", None)
CORRUPTED = [("clean", "blur_s2"), ("clean", "noise_s2"), ("clean", "fog_s2"), ("blur_s3", None),
             ("noise_s2", None), ("rain_s2", None), ("fog", None), ("lowlight", None)]
ARMS = ("shipped", "union", "lex", "ens5_union", "full_tta", "full_tta_lex")
STREAM_OF = {"shipped": "single", "union": "single", "lex": "single", "ens5_union": "ens5", "full_tta": "full",
             "full_tta_lex": "full"}      # Amendment 2: descriptive
T_975_DF4 = 2.7764451051977987


def cell_tag(c) -> str:
    return f"{c[0]}__{c[1] or 'clean'}"


def subs_of(c) -> list[str]:
    return ["clean"] if c == VERDICT_CELL else [ph.sub(v) for v in VIS_DRAWS]


# ------------------------------------------------------------------ development calibration
def cmd_devcal(args) -> int:
    """The capability prior `seed_ensemble.job` uses for an ensemble arm: each stream's clean ship+buoy
    mAP on the development fit frames, IR divided by `cap_ir_scale`. Identical for every system."""
    import seed_ensemble as se
    from uqfusion.eval.ctx import capability_prior, load_context
    d = f"{se.STATS}/brightness/{se.sub(945)}"
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed0/{se.sub(945)}", bright_dir=d,
                       structure_dir=f"{se.STATS}/structure/{se.sub(945)}",
                       ir_bright=f"{d}/gauss_ir_paired_clean.json", conditions=("clean",), ir_condition=None,
                       verbose=False)
    iflag0 = ctx.ir_d2 > ctx.ir_bound
    out = {"cap_ir_scale": float(ctx.cap_ir_scale), "single_dev_cap": [float(ctx.cap_vis), float(ctx.cap_ir)]}
    for arm, flip, rest in (("ens5_union", False, False), ("full_tta", True, True)):
        se._CACHE.clear()
        vis = se._stream(945, "vis", "clean", SEEDS, None, flip)
        ir = se._stream(945, "ir", None, SEEDS, iflag0 if rest else None, flip)
        cv, ci = capability_prior(vis, ir, ctx.gts, ctx.h_frames, ctx.cap_fit_frames)
        out[arm] = {"cap_vis": float(cv), "cap_ir": float(ci / ctx.cap_ir_scale), "cap_ir_raw": float(ci)}
        print(f"[devcal] {arm}: cap_vis {cv:.5f} cap_ir {ci:.5f} / {ctx.cap_ir_scale} -> {ci / ctx.cap_ir_scale:.6f}",
              flush=True)
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "devcal.json").write_text(json.dumps(out, indent=1), encoding="utf-8")
    return 0


# ------------------------------------------------------------------ pohang04 streams
_C: dict = {}


def _load_pkl(p: Path):
    with open(p, "rb") as fh:
        return pickle.load(fh)


def raw_path(k, mod, cond, s) -> Path:
    if cond == "clean":
        return HP / f"caches/seed{k}/clean/gauss_{mod}_paired_clean.pkl"
    return HP / f"v2/caches/seed{k}/{s}/gauss_{mod}_paired_{cond}.pkl"


def _raw(k, mod, cond, s):
    from uqfusion.eval.cache import load_cache
    key = ("raw", k, mod, cond, s if cond != "clean" else "clean")
    if key not in _C:
        _C[key] = load_cache(raw_path(k, mod, cond, s))[0]
    return _C[key]


def _view(k, mod, cond, s, flip: bool, flag):
    """One checkpoint's record list for a stream: raw (restored where `flag`), or its flipped view;
    IR gets the shipped detector-side NMS. As `seed_ensemble._per_ckpt` / `_flip_ckpt`."""
    from uqfusion.eval.irdedup import nms_records
    draw = None if cond == "clean" else int(s[4:7])
    if flip:
        r = _load_pkl(ph.flip_out(k, mod, cond, draw))["records"]
    else:
        r = _raw(k, mod, cond, s)
        if flag is not None:
            nl = _load_pkl(ph.nlm_out(k, cond, draw))["records"]
            assert set(nl) == set(np.flatnonzero(flag).tolist()), "restored frames differ from the flag"
            # The builder's restored records carry detections only. The frame identity (`image_path`,
            # for the pairing check) and `feat` come from the raw record. `feat` feeds only the
            # Mahalanobis distance, which leaves the gate-free fusion unchanged: on development IR
            # fog_s2 (draw 945, 2,024 restored frames) taking it from the raw rather than the restored
            # frame changed 0 fused records (scratchpad check_feat_effect.py, 2026-10-11).
            r = [{**nl[i], "image_path": a["image_path"], "feat": a["feat"]} if f else a
                 for i, (a, f) in enumerate(zip(r, flag))]
    return nms_records(r, IR_NMS) if mod == "ir" else r


def ens_stream(mod, cond, s, kind: str, flag=None):
    """`ens5` (five raw views) or `full` (five raw views with IR restored where flagged + five flips),
    merged frame by frame with plain WBF at 0.55 (`seed_ensemble.merge`)."""
    from seed_ensemble import merge
    key = ("ens", mod, cond, s if cond != "clean" else "clean", kind)
    if key not in _C:
        per = [_view(k, mod, cond, s, False, flag if kind == "full" else None) for k in SEEDS]
        if kind == "full":
            per += [_view(k, mod, cond, s, True, None) for k in SEEDS]
        _C[key] = [merge(list(fr), ENS_IOU) for fr in zip(*per)]
    return _C[key]


def save_pre(parts, d: Path) -> float:
    from uqfusion.eval.apmetrics import ap_from_parts
    from holdout_p04_v2_rescore import save_ship_pre
    save_ship_pre(parts, np.arange(len(parts)), d)
    e = ap_from_parts(parts)["per_class"].get(SHIP)
    v = float(e["ap50_95"]) if e else float("nan")
    (d / "obs.json").write_text(json.dumps({"ap": v}), encoding="utf-8")
    return v


def _job_done(k, s, ic, vc, arms) -> bool:
    tag = cell_tag((vc, ic))
    need = []
    if any(STREAM_OF[a] == "single" for a in arms):
        need += [f"single/vis/{tag}/seed{k}_{s}", f"single/ir/{tag}/seed{k}_{s}"]
        need += [f"{a}/fused/{tag}/seed{k}_{s}" for a in arms if STREAM_OF[a] == "single"]
    if k == 0:
        for a in arms:
            if STREAM_OF[a] != "single":
                kind = STREAM_OF[a]
                need += [f"{kind}/vis/{tag}/all_{s}", f"{kind}/ir/{tag}/all_{s}", f"{a}/fused/{tag}/all_{s}"]
    return all((PRE / n / "obs.json").is_file() for n in need)


def _parts_job(job):
    """One (system k, sub, IR condition): every requested arm and stream on every VIS condition of the
    group. Ensemble arms are computed on k == 0 only (identical for every system)."""
    import holdout_p04_look as look
    import holdout_p04_v2_rescore as v2r
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import run_systems
    from gate_zoo import build_lex_records
    k, s, ic, vcs, arms = job
    import seed_ensemble as se
    se._memo_distances()      # shipped and union score the same records: compute their distances once (pure cache)
    _C.clear()
    t0 = time.time()
    if all(_job_done(k, s, ic, vc, arms) for vc in vcs):
        return job[:3], {}, 0.0
    devcal = json.loads((OUT / "devcal.json").read_text(encoding="utf-8"))
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    sub_ = look.Pohang04() if s == "clean" else v2r.Pohang04V2()
    ctx = look.system_context(sub_, k, s, sorted(set(vcs) | {"clean"}), ic, dev)
    icond = ic or "clean"
    draw = None if s == "clean" else int(s[4:7])
    flag = ph.ir_flags(icond, draw)
    if not np.array_equal(flag, ctx.ir_d2 > ctx.ir_bound):
        raise SystemExit(f"IR flag differs from the context's ({k}, {s}, {ic})")
    cap0 = (ctx.cap_vis, ctx.cap_ir)
    done = {}
    for vc in vcs:
        cell = (vc, ic)
        tag = cell_tag(cell)

        def put(arm_stream, recs, key=f"seed{k}_{s}"):
            d = PRE / arm_stream / tag / key
            if (d / "obs.json").is_file():
                return
            done[f"{arm_stream}|{tag}|{key}"] = save_pre(frame_parts(recs, ctx.gts), d)

        need_single = any(STREAM_OF[a] == "single" for a in arms)
        if need_single:
            ctx.cap_vis, ctx.cap_ir = cap0
            out = run_systems(ctx, vc, compute_metrics=False)
            vis, irv = ctx.vis_by_cond[vc], out["ir_in_vis"]
            put("single/vis", vis)
            put("single/ir", irv)
            if "shipped" in arms:
                put("shipped/fused", out["fused_gated"])
            if "lex" in arms:
                put("lex/fused", build_lex_records(vis, irv, np.zeros(len(vis), bool)))
            if "union" in arms:
                put("union/fused", run_systems(dataclasses.replace(ctx, veto=None), vc,
                                               compute_metrics=False)["fused_gated"])
        if k == 0:
            for arm, kind in (("ens5_union", "ens5"), ("full_tta", "full")):
                lex_arm = f"{arm}_lex" if arm == "full_tta" else None      # Amendment 2: E9 on the same streams
                if arm not in arms and lex_arm not in arms:
                    continue
                vis = ens_stream("vis", vc, s, kind)
                ir = ens_stream("ir", icond, s, kind, flag if kind == "full" else None)
                ctx.cap_vis, ctx.cap_ir = devcal[arm]["cap_vis"], devcal[arm]["cap_ir"]
                out = run_systems(dataclasses.replace(ctx, veto=None), vc, compute_metrics=False,
                                  vis_records=vis, ir_records=ir)
                put(f"{kind}/vis", vis, f"all_{s}")
                put(f"{kind}/ir", out["ir_in_vis"], f"all_{s}")
                if arm in arms:
                    put(f"{arm}/fused", out["fused_gated"], f"all_{s}")
                if lex_arm in arms:
                    put(f"{lex_arm}/fused", build_lex_records(vis, out["ir_in_vis"], np.zeros(len(vis), bool)),
                        f"all_{s}")
                ctx.cap_vis, ctx.cap_ir = cap0
    return job[:3], done, time.time() - t0


def cmd_parts(args) -> int:
    cells = [VERDICT_CELL] if args.cells == "clean" else CORRUPTED
    arms = tuple(args.arms)
    jobs = []
    for k in SEEDS:
        for s in sorted({x for c in cells for x in subs_of(c)}):
            by_ir: dict = {}
            for vc, ic in cells:
                if s in subs_of((vc, ic)):
                    by_ir.setdefault(ic, []).append(vc)
            for ic, vcs in by_ir.items():
                a = tuple(x for x in arms if k == 0 or STREAM_OF[x] == "single")
                if a:
                    jobs.append((k, s, ic, tuple(vcs), a))
    if args.seeds is not None:
        jobs = [j for j in jobs if j[0] in args.seeds]
    if args.subs is not None:
        jobs = [j for j in jobs if j[1] in args.subs]
    if args.split:
        # one VIS condition per job: each worker holds one condition's caches, so more fit in memory
        jobs = [(k, s, ic, (vc,), a) for (k, s, ic, vcs, a) in jobs for vc in vcs]
    # the single-checkpoint jobs are cheap and many; the ensemble jobs (k == 0) are the slowest, so they start first
    jobs.sort(key=lambda j: (j[0] != 0, j[1]))
    print(f"[parts] {len(jobs)} jobs, arms {arms}", flush=True)
    obs_p = OUT / "observed.json"
    obs = json.loads(obs_p.read_text(encoding="utf-8")) if obs_p.is_file() else {}
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (j, done, dt) in enumerate(ex.map(_parts_job, jobs), 1):
            obs.update(done)
            tmp = obs_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(obs, indent=1), encoding="utf-8")
            os.replace(tmp, obs_p)
            print(f"[parts] {n}/{len(jobs)} {j} {dt:.0f}s (elapsed {time.time() - t0:.0f}s)", flush=True)
    return 0


# ------------------------------------------------------------------ bootstrap
def boot_batch(pre: dict, W: np.ndarray, dev, grid_g) -> np.ndarray:
    """Ship AP for each row of `W` (integer frame multiplicities), several rows per GPU pass. Per row the
    arithmetic is `_ap_gpu`'s: float64 cumsums of 0/1 (exact), IEEE division, exact running max, the
    numpy interpolation on the bracketing points. Padding past a row's length is -inf precision and
    +inf recall, which neither the running max nor the search can reach."""
    import torch
    from uqfusion.eval.apmetrics import NL, RECALL_GRID
    d = pre["classes"][SHIP]
    tp_g = torch.from_numpy(np.array(d["tp"])).to(dev).T.contiguous()           # (NL, m)
    fidx_g = torch.from_numpy(np.ascontiguousarray(d["fidx"])).to(dev)
    gt = np.asarray(pre["gt"][SHIP])
    out = np.empty(len(W))
    step = int(os.environ.get("P04_BOOT_R", "8"))      # replicates per GPU pass: speed and memory only
    for a in range(0, len(W), step):
        Wc = np.ascontiguousarray(W[a:a + step])
        R = len(Wc)
        Wg = torch.from_numpy(Wc).to(dev)
        mult = Wg[:, fidx_g]                                                       # (R, m)
        lens = mult.sum(dim=1)
        L = int(lens.max().item()) if tp_g.shape[1] else 0
        n_gt = Wc @ gt
        if L == 0:
            out[a:a + R] = [float("nan") if n == 0 else 0.0 for n in n_gt]
            continue
        E = torch.zeros((R, NL, L), dtype=torch.bool, device=dev)
        for r in range(R):
            n = int(lens[r])
            if n:
                E[r, :, :n] = torch.repeat_interleave(tp_g, mult[r], dim=1)
        pos = torch.arange(L, device=dev)
        pad = (pos[None, :] >= lens[:, None])[:, None, :]                       # (R, 1, L)
        cum_tp = torch.cumsum(E, dim=2, dtype=torch.float64)
        del E
        ngt = torch.from_numpy(np.maximum(n_gt, 1).astype(np.float64)).to(dev)[:, None, None]
        recall = (cum_tp / ngt).masked_fill(pad, float("inf"))
        # every expanded detection is a TP or an FP, so cum_tp + cum_fp is exactly pos + 1 (integers held
        # exactly in float64): the same divisor, without a second cumulative sum
        prec = (cum_tp / (pos + 1).to(torch.float64)[None, None, :]).masked_fill(pad, float("-inf"))
        del cum_tp
        env = torch.flip(torch.cummax(torch.flip(prec, [2]), dim=2).values, [2])
        del prec
        grid = grid_g.expand(R, NL, -1).contiguous()
        j = torch.searchsorted(recall, grid, right=True) - 1
        last = (lens - 1).clamp(min=0)[:, None, None]
        ends = torch.cat([torch.zeros_like(last), last], dim=2).expand(R, NL, 2)
        ix = torch.cat([ends, torch.minimum(j.clamp(min=0), last), torch.minimum(j + 1, last)], dim=2)
        rx = torch.gather(recall, 2, ix).cpu().numpy()
        ex_ = torch.gather(env, 2, ix).cpu().numpy()
        ix = ix.cpu().numpy()
        del recall, env, j
        for r in range(R):
            if n_gt[r] == 0:
                out[a + r] = float("nan")
                continue
            if int(lens[r]) == 0:
                out[a + r] = float(np.zeros(NL).mean())
                continue
            v = np.empty(NL)
            for k in range(NL):
                _u, first = np.unique(ix[r, k], return_index=True)
                xs, fs = rx[r, k][first], ex_[r, k][first]
                v[k] = np.interp(RECALL_GRID, xs, fs, left=fs[0], right=0).mean()
            out[a + r] = float(v.mean())
    return out


def _check(pre, W, dev, grid_g, cpu: bool = False) -> int:
    from uqfusion.eval.apmetrics import _score
    rows = np.vstack([np.ones(pre["n_frames"], np.int64), W[:3], W[len(W) // 2: len(W) // 2 + 2], W[-3:]])
    if cpu:
        from boot_cpu import boot_cpu
        got = boot_cpu(pre, rows)
    else:
        got = boot_batch(pre, rows, dev, grid_g)
    for w, g in zip(rows, got):
        a = _score(pre, w, SHIP)
        if not (a == g or (np.isnan(a) and np.isnan(g))):
            raise SystemExit(f"batched GPU AP differs from the CPU reference: {a!r} vs {g!r}")
    return len(rows)


def _pre_hash(d: Path) -> str:
    h = hashlib.sha1()
    for f in sorted(d.iterdir()):
        if f.suffix == ".npy" and f.name != "reps.npy":
            h.update(f.name.encode())
            h.update(f.read_bytes())
    return h.hexdigest()


def presort_dirs() -> list[Path]:
    return sorted(p.parent for p in PRE.rglob("obs.json"))


def cmd_boot(args) -> int:
    from holdout_p04_v2_rescore import load_ship_pre, resample_weights
    from uqfusion.eval.apmetrics import RECALL_GRID
    if args.cpu:
        # `boot_cpu`: the same replicates without the GPU (checked == against the GPU path on 4,000
        # replicates and against `_score` by scripts/check_boot_cpu.py), so the GPU stays with the build
        from boot_cpu import boot_cpu
        dev = grid_g = None
    else:
        import torch
        dev = torch.device("cuda")
        frac = float(os.environ.get("P04_BOOT_MEMFRAC", "0") or 0)
        if frac:      # cap the allocator's cache so the build running beside it keeps its memory
            torch.cuda.set_per_process_memory_fraction(frac, 0)
        grid_g = torch.from_numpy(RECALL_GRID).to(dev)
    wp = OUT / "boot_weights.npy"
    if not wp.is_file():
        import holdout_p04_look as look
        paths = [r.strip() for r in (HP / "p04_pairs_vis.txt").read_text(encoding="utf-8").splitlines() if r.strip()]
        # the frame paths `run_ids` reads; pohang04 is one run, so the run split is a single slice
        W = resample_weights(paths, np.arange(len(paths)), look.N_BOOT)
        np.save(wp, W)
    W = np.load(wp)
    todo = [d for d in presort_dirs() if not (d / "reps.npy").is_file()]
    # identical presorts (e.g. the VIS clean stream, which every IR-corrupted cell of a draw shares) are
    # bootstrapped once: same detections, same frames, same weights -> same replicates
    done_by_hash = {}
    for d in presort_dirs():
        if (d / "reps.npy").is_file():
            done_by_hash.setdefault(_pre_hash(d), d)
    print(f"[boot] {len(todo)} presorts to bootstrap ({len(W)} replicates each)", flush=True)
    t0 = time.time()
    checked = False
    for n, d in enumerate(todo, 1):
        h = _pre_hash(d)
        if h in done_by_hash:
            import shutil
            shutil.copyfile(done_by_hash[h] / "reps.npy", d / "reps.npy")
            continue
        pre = load_ship_pre(d)
        if not checked:
            print(f"[boot] checked {_check(pre, W, dev, grid_g, args.cpu)} rows equal to the CPU reference", flush=True)
            checked = True
        reps = boot_cpu(pre, W) if args.cpu else boot_batch(pre, W, dev, grid_g)
        np.save(d / "reps.npy", reps)
        done_by_hash[h] = d
        if not args.cpu:
            torch.cuda.empty_cache()
        if n % 10 == 0 or n == len(todo):
            print(f"[boot] {n}/{len(todo)} ({time.time() - t0:.0f}s)", flush=True)
    return 0


# ------------------------------------------------------------------ report
def series(arm_stream: str, cell) -> tuple[float, np.ndarray] | None:
    """Observed statistic and replicates: mean over systems within a draw, then over draws."""
    base = PRE / arm_stream / cell_tag(cell)
    if not base.is_dir():
        return None
    subs = subs_of(cell)
    vals, reps = [], []
    for s in subs:
        keys = [base / f"seed{k}_{s}" for k in SEEDS] if (base / f"seed0_{s}").is_dir() else [base / f"all_{s}"]
        if not all((d / "reps.npy").is_file() for d in keys):
            return None
        vals.append(np.mean([json.loads((d / "obs.json").read_text())["ap"] for d in keys]))
        reps.append(np.mean([np.load(d / "reps.npy") for d in keys], axis=0))
    return float(np.mean(vals)), np.mean(reps, axis=0)


def ci(r: np.ndarray) -> tuple[float, float]:
    r = r[np.isfinite(r)]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)


def cmd_report(args) -> int:
    cells = [VERDICT_CELL] + CORRUPTED
    L = ["# Gate-free systems on the held-out run (pohang04)", "",
         "`docs/prereg-p04-heldout-gatefree-2026-10-11.md` (Amendment 1: fp16 views). Third scoring of pohang04. "
         "Ship AP against the VIS ground truth, mean over the five systems (and the four draws of a corrupted "
         "cell); Δ = arm − shipped, paired; 95% interval from 1,000 moving-block replicates (L = 20, seed 1). "
         "Claim = fused − max(VIS, IR) of the arm's own streams.", ""]
    res = {}
    complete = True
    L += ["| cell | shipped | union Δ | full_tta Δ | ens5_union Δ | lex Δ | full_tta_lex Δ |",
          "|---|---:|---:|---:|---:|---:|---:|"]
    for c in cells:
        sh = series("shipped/fused", c)
        row = [cell_tag(c).replace("__", "/"), "—" if sh is None else f"{sh[0]:.4f}"]
        for arm in ("union", "full_tta", "ens5_union", "lex", "full_tta_lex"):
            a = series(f"{arm}/fused", c)
            if a is None or sh is None:
                row.append("pending")
                complete = complete and arm not in ("union", "full_tta")
                continue
            dlt = a[0] - sh[0]
            lo, hi = ci(a[1] - sh[1])
            res[f"{arm}|{cell_tag(c)}"] = {"value": a[0], "shipped": sh[0], "delta": dlt, "ci": [lo, hi]}
            row.append(f"{dlt:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        L.append("| " + " | ".join(row) + " |")
    L += ["", "## `full_tta_lex` − `full_tta` (Amendment 2, descriptive)", "",
          "Paired, same replicates: does the lex ordering beat the WBF merge of the same streams?", "",
          "| cell | full_tta | full_tta_lex | Δ | 95% interval |", "|---|---:|---:|---:|---|"]
    for c in cells:
        a, b = series("full_tta/fused", c), series("full_tta_lex/fused", c)
        if a is None or b is None:
            L.append(f"| {cell_tag(c).replace('__', '/')} | pending | | | |")
            continue
        lo, hi = ci(b[1] - a[1])
        res[f"lex_vs_merge|{cell_tag(c)}"] = {"full_tta": a[0], "full_tta_lex": b[0], "delta": b[0] - a[0], "ci": [lo, hi]}
        L.append(f"| {cell_tag(c).replace('__', '/')} | {a[0]:.4f} | {b[0]:.4f} | {b[0] - a[0]:+.4f} | [{lo:+.4f}, {hi:+.4f}] |")
    L += ["", "## Claim: fused − max(VIS, IR), each arm's own streams", "",
          "| cell | arm | fused | VIS | IR | fused − max | 95% interval | fails |", "|---|---|---:|---:|---:|---:|---|---|"]
    for c in cells:
        for arm in ARMS:
            f, v, i = (series(f"{arm}/fused", c), series(f"{STREAM_OF[arm]}/vis", c), series(f"{STREAM_OF[arm]}/ir", c))
            if None in (f, v, i):
                continue
            mx = np.maximum(v[1], i[1])
            dd = f[0] - max(v[0], i[0])
            lo, hi = ci(f[1] - mx)
            fails = dd <= -0.0060 and hi < 0
            res[f"claim|{arm}|{cell_tag(c)}"] = {"fused": f[0], "vis": v[0], "ir": i[0], "d": dd, "ci": [lo, hi],
                                                 "fails": bool(fails)}
            L.append(f"| {cell_tag(c).replace('__', '/')} | {arm} | {f[0]:.4f} | {v[0]:.4f} | {i[0]:.4f} | {dd:+.4f} | "
                     f"[{lo:+.4f}, {hi:+.4f}] | {'FAIL' if fails else ''} |")
    # verdicts
    def verdict(arm):
        ds = [res.get(f"{arm}|{cell_tag(c)}") for c in cells]
        if any(x is None for x in ds):
            return "PENDING", None
        worse = [cell_tag(c) for c, x in zip(cells, ds) if x["delta"] <= -0.0014 and x["ci"][1] < 0]
        better = [cell_tag(c) for c, x in zip(cells, ds) if x["delta"] >= 0.0060 and x["ci"][0] > 0]
        prim = ds[0]["delta"] >= 0.0060 and ds[0]["ci"][0] > 0
        return worse, (better, prim)
    w, _ = verdict("union")
    pu = "PENDING" if w == "PENDING" else ("PASS" if not w else f"FAIL (worse: {', '.join(w)})")
    w2, b2 = verdict("full_tta")
    if w2 == "PENDING":
        pf = "PENDING"
    else:
        better, prim = b2
        pf = ("REPLICATES" if (not w2 and better and prim) else
              f"DOES NOT REPLICATE (worse: {', '.join(w2) or 'none'}; better: {', '.join(better) or 'none'}; "
              f"primary cell {'passes' if prim else 'fails'})")
    L[4:4] = [f"**P-U (`union` non-inferior by day): {pu}.**  ", f"**P-F (`full_tta` replicates): {pf}.**", "",
              "" if complete else "_Partial: cells marked pending are not yet scored; the verdicts wait for all nine._", ""]
    res["P-U"], res["P-F"] = pu, pf
    out = ROOT / args.out
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L[:12]))
    return 0


def main() -> int:
    if os.environ.get("P04_LOW") == "1" and os.name == "nt":
        # BELOW_NORMAL_PRIORITY_CLASS, inherited by the pool's worker processes: scoring yields the CPU
        # to the cache build that feeds the GPU
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("devcal")
    p = sp.add_parser("parts")
    p.add_argument("--cells", choices=["clean", "corrupted"], required=True)
    p.add_argument("--arms", nargs="+", default=list(ARMS), choices=list(ARMS))
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--split", action="store_true", help="one job per VIS condition (less memory per worker)")
    p.add_argument("--seeds", type=int, nargs="+", default=None, help="only these systems' jobs")
    p.add_argument("--subs", nargs="+", default=None, help="only these draws, e.g. draw941_951")
    bo = sp.add_parser("boot")
    bo.add_argument("--cpu", action="store_true", help="numba CPU path (same replicates), leaves the GPU free")
    r = sp.add_parser("report")
    r.add_argument("--out", default="docs/eval/holdout_p04_gatefree_2026-10-11.md")
    args = ap.parse_args()
    return {"devcal": cmd_devcal, "parts": cmd_parts, "boot": cmd_boot, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
