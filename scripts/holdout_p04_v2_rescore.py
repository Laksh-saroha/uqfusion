"""pohang04 corrupted cells re-scored under corruption v2 -- a DISCLOSED SECOND EXPOSURE.

pohang04 was spent by the single look of 2026-09-20 (`holdout_p04_look.py`). Its ten
corrupted cells were descriptive and built with the v1 corruptions, which were found broken
on 2026-10-09 (`docs/eval/corruption_v2/README.md`: fog was a fixed 21-px blur, lowlight an
additive clip, glare painted into the letterbox pad, every filter rewrote the pad, IR got
visible-light filters). On 2026-10-10 Laksh decided to re-score those cells under v2 and
disclose it (`docs/prereg-p04-v2-rescore-2026-10-10.md`, exposure ledger entry of that date).

What this is, and is not
------------------------
* The **`clean/clean` verdict is not touched**: it involves no corruption, it is not
  re-scored, and `LOOK_TAKEN.json` / its tracked mirror are only read, never written.
* Same five Phase 3 systems, preset, calibration injection (`holdout_p04_look.system_context`),
  draws (VIS 941-944, IR = VIS + 10), statistic (seed-mean, draw-mean fused ship AP, VIS GT)
  and interval (moving-block bootstrap L = 20, n_boot = 1000, seed 1) as the look. Only the
  corruption changes. Descriptive: no verdict, no outcome label, nothing adopted.
* v2 refuses IR glare (a thermal sensor does not see a visible-light flare), so the three
  cells with IR `glare_s2` are reported as not modelled; `lowlight/glare_s2` is replaced by
  `lowlight/clean`, and `blur_s3/glare_s2` collapses into `blur_s3/clean`.
* No VIS-only or IR-only arm, as in the look: a per-stream comparison on pohang04 would be a
  new question asked of the spent set, which is not what was authorised.

Trees (outside `runs/cache*` on purpose, see `holdout_p04_build_caches.py`):
    runs/holdout_p04/v2/caches/seed{k}/draw{v}_{v+10}/<stem>.pkl
    runs/holdout_p04/v2/derived/{brightness,structure}/draw{v}_{v+10}/<stem>.json
    runs/holdout_p04/v2/stage/...              hard-linked, one directory per (system, draw)
Clean caches and clean statistics are the look's own (hard-linked, unchanged).

Speed. Caches: `build_cache_multi.py --batch 4` (corrupt once, five checkpoints, statistics
in the same pass; batch 4 measured at |d ship AP| ~1e-5, `docs/eval/corruption_v2/
batched_inference.json`), several units at once. Scoring: (system, draw) contexts in a
process pool, each returning only class-0 presorted parts. Bootstrap: the 1,000 block
resamples are drawn once, exactly as `boot_cell` draws them (same rng, same order), and
evaluated in parallel chunks; restricting the presort to class 0 does not change class 0's
AP (`ap_weighted` scores classes independently). `--selftest` proves both on development
data: replicates and value identical to `holdout_p04_look.boot_cell`.

    py -3.13 scripts/holdout_p04_v2_rescore.py --selftest
    py -3.13 scripts/holdout_p04_v2_rescore.py --build --jobs 4
    py -3.13 scripts/holdout_p04_v2_rescore.py --score --out docs/eval/holdout_p04_v2_rescore_2026-10-10.md
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import holdout_p04_look as look                                                   # noqa: E402
from _ideas_common import write_md                                                # noqa: E402
from uqfusion.eval.apmetrics import _score, ap_from_parts, frame_parts, presort   # noqa: E402
from uqfusion.eval.blockboot import block_resample, run_ids, run_slices           # noqa: E402
from uqfusion.eval.ctx import run_systems                                         # noqa: E402

HP = look.HP
V2 = HP / "v2"
CACHES, DERIVED, STAGE = V2 / "caches", V2 / "derived", V2 / "stage"
SEEDS, VIS_DRAWS, SHIP = look.SEEDS, look.VIS_DRAWS, look.SHIP
BLOCK_LEN, N_BOOT, BOOT_SEED = look.BLOCK_LEN, look.N_BOOT, look.BOOT_SEED
LISTS = {"vis": HP / "p04_pairs_vis.txt", "ir": HP / "p04_pairs_ir.txt"}

#: (modality, stem, kind, severity), in build order: the cells v1 got most wrong first.
STREAMS = [
    ("vis", "gauss_vis_paired_fog", "fog", 2),
    ("vis", "gauss_vis_paired_lowlight", "lowlight", 2),
    ("ir", "gauss_ir_paired_fog_s2", "fog", 2),
    ("vis", "gauss_vis_paired_blur_s3", "blur", 3),
    ("vis", "gauss_vis_paired_noise_s2", "noise", 2),
    ("vis", "gauss_vis_paired_rain_s2", "rain", 2),
    ("ir", "gauss_ir_paired_blur_s2", "blur", 2),
    ("ir", "gauss_ir_paired_noise_s2", "noise", 2),
]
CELLS = [("clean", "blur_s2"), ("clean", "noise_s2"), ("clean", "fog_s2"),
         ("blur_s3", None), ("noise_s2", None), ("rain_s2", None), ("fog", None), ("lowlight", None)]
NOT_MODELLED = {("clean", "glare_s2"): "IR glare not modelled in v2",
                ("lowlight", "glare_s2"): "IR glare not modelled in v2; see lowlight/clean",
                ("blur_s3", "glare_s2"): "IR glare not modelled in v2; see blur_s3/clean"}
#: The look's v1 values, for the side-by-side (docs/eval/holdout_p04_look.json).
LOOK_JSON = ROOT / "docs/eval/holdout_p04_look.json"


def sub(v: int) -> str:
    return f"draw{v}_{v + 10}"


def weights(mod: str, k: int) -> Path:
    return ROOT / f"runs/phase3_stage2/p3_{mod}_seed{k}/weights/best.pt"


def n_pairs() -> int:
    return sum(1 for ln in LISTS["vis"].read_text(encoding="utf-8").splitlines() if ln.strip())


# ------------------------------------------------------------------ build
def cache_ok(out: Path, w: Path, mod: str, kind: str, sev: int, cseed: int, n: int, batch: int) -> bool:
    from uqfusion.eval.cache import load_cache
    from uqfusion.eval.corruptions_v2 import CODE_REV
    if not out.is_file():
        return False
    try:
        recs, m = load_cache(out)
    except Exception:                                    # noqa: BLE001 -- torn file: rebuild
        return False
    return (len(recs) == n and Path(m["weights"][0]).resolve() == w.resolve()
            and Path(m["images_list"]).resolve() == LISTS[mod].resolve()
            and (m.get("corrupt"), m.get("severity"), m.get("corrupt_seed")) == (kind, sev, cseed)
            and m.get("corrupt_version") == "v2" and m.get("corrupt_code") == CODE_REV and m.get("conf") == 0.001
            and m.get("infer_batch", 1) == batch)


def build(args) -> int:
    from uqfusion.config import resolve_gpu_python
    n = n_pairs()
    gp = resolve_gpu_python(None)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    cpus = os.cpu_count() or 4
    workers = args.workers or max(1, (cpus - args.jobs - 1) // args.jobs)
    (V2 / "logs").mkdir(parents=True, exist_ok=True)
    units = [(v, s) for s in STREAMS for v in VIS_DRAWS]          # stream-major: fog first
    t0 = time.time()

    def run(u) -> tuple[str, bool]:
        v, (mod, stem, kind, sev) = u
        cseed = v + 10 if mod == "ir" else v
        todo = [(weights(mod, k), CACHES / f"seed{k}" / sub(v) / f"{stem}.pkl") for k in SEEDS]
        stats = [DERIVED / kd / sub(v) / f"{stem}.json" for kd in ("brightness", "structure")]
        name = f"{sub(v)}/{stem}"
        ok = lambda: (all(cache_ok(o, w, mod, kind, sev, cseed, n, args.batch) for w, o in todo)
                      and all(p.is_file() for p in stats))
        if ok():
            print(f"[build] {name} skip (verified)", flush=True)
            return name, True
        for _, o in todo:
            o.parent.mkdir(parents=True, exist_ok=True)
        # blur and rain go through albumentations, which imports torch (~2 GB commit per worker):
        # four such units at full width exceed this machine's commit limit, and they are cheap.
        w_unit = min(workers, 3) if kind in ("blur", "rain") else workers
        cmd = [gp, "-u", str(ROOT / "scripts/build_cache_multi.py"),
               "--weights", *[str(w) for w, _ in todo], "--outs", *[str(o) for _, o in todo],
               "--images-list", str(LISTS[mod]), "--imgsz", "640", "--conf", "0.001", "--data", mod,
               "--corrupt", kind, "--severity", str(sev), "--corrupt-seed", str(cseed),
               "--corrupt-version", "v2", "--modality", mod, "--workers", str(w_unit),
               "--batch", str(args.batch), "--stats-out", *[str(p) for p in stats]]
        t = time.time()
        print(f"[build] {name} start ({w_unit} workers, batch {args.batch})", flush=True)
        with open(V2 / "logs" / f"{sub(v)}_{stem}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        good = rc == 0 and ok()
        print(f"[build] {name} {'ok' if good else f'FAIL rc={rc}'} in {time.time() - t:.0f}s "
              f"(elapsed {(time.time() - t0) / 3600:.2f} h)", flush=True)
        return name, good

    with ThreadPoolExecutor(args.jobs) as ex:
        res = list(ex.map(run, units))
    bad = [nm for nm, g in res if not g]
    print(f"[build] {len(res) - len(bad)}/{len(res)} units ok in {(time.time() - t0) / 3600:.2f} h"
          + (f"; FAILED {bad}" if bad else ""), flush=True)
    return 1 if bad else 0


# ------------------------------------------------------------------ substrate
class Pohang04V2(look.Substrate):
    """The look's clean caches and statistics, v2 corrupted ones, hard-linked per (system, draw)."""

    def __init__(self):
        super().__init__("pohang04 (corruption v2)", SEEDS, [sub(v) for v in VIS_DRAWS],
                         look.rel(HP / "p04_pairs_manifest.csv"), None)

    def cache_dir(self, seed, s):
        d = STAGE / f"seed{seed}" / s
        for m in ("vis", "ir"):
            look.Pohang04._link(ROOT / f"runs/cache_p3/seed{seed}/gauss_{m}_train_clean.pkl", d / f"gauss_{m}_train_clean.pkl")
            look.Pohang04._link(HP / f"caches/seed{seed}/clean/gauss_{m}_paired_clean.pkl", d / f"gauss_{m}_paired_clean.pkl")
        for _m, stem, _k, _s in STREAMS:
            look.Pohang04._link(CACHES / f"seed{seed}/{s}/{stem}.pkl", d / f"{stem}.pkl")
        return look.rel(d)

    def _derived(self, kind, s):
        d = STAGE / "derived" / s / kind
        for m in ("vis", "ir"):
            look.Pohang04._link(HP / f"derived/{kind}/clean/gauss_{m}_paired_clean.json", d / f"gauss_{m}_paired_clean.json")
        for _m, stem, _k, _s in STREAMS:
            look.Pohang04._link(DERIVED / kind / s / f"{stem}.json", d / f"{stem}.json")
        return look.rel(d)

    def bright_dir(self, s): return self._derived("brightness", s)
    def struct_dir(self, s): return self._derived("structure", s)


class DevCheck(look.Substrate):
    """Development data only (selftest): runs/cache_m, shipped draw, one system."""

    def __init__(self):
        super().__init__("development", (0,), ["shipped"], "runs/derived/paired_val_manifest.csv", "tune")

    def cache_dir(self, seed, s): return "runs/cache_m"
    def bright_dir(self, s): return "runs/derived/brightness"
    def struct_dir(self, s): return "runs/derived/structure"


# ------------------------------------------------------------------ scoring
def cell_tag(c) -> str:
    return f"{c[0]}__{c[1] or 'clean'}"


def save_ship_pre(parts: list[dict], idx: np.ndarray, d: Path) -> None:
    """`presort` restricted to class 0, written as .npy so bootstrap workers can memory-map
    it (one copy in the page cache instead of one per process). Class 0's AP under any
    weights is unchanged by dropping the other classes: `ap_weighted` scores each alone."""
    pre = presort(parts, idx)
    d.mkdir(parents=True, exist_ok=True)
    np.save(d / "tp.npy", pre["classes"][SHIP]["tp"])
    np.save(d / "fidx.npy", pre["classes"][SHIP]["fidx"])
    np.save(d / "gt.npy", pre["gt"][SHIP])


def load_ship_pre(d: Path) -> dict:
    gt = np.load(d / "gt.npy")
    return {"classes": {SHIP: {"tp": np.load(d / "tp.npy", mmap_mode="r"),
                               "fidx": np.load(d / "fidx.npy", mmap_mode="r")}},
            "gt": {SHIP: gt}, "n_frames": int(gt.shape[0])}


def _score_job(job):
    """One (system, draw): every cell's class-0 presort and observed ship AP. Runs in a worker."""
    sub_, seed, s, cells, sel, scratch = job
    dev = look.dev_context(seed, "runs/cache_m" if isinstance(sub_, DevCheck) else f"runs/cache_p3/seed{seed}")
    by_ir: dict = {}
    for vc, ic in cells:
        by_ir.setdefault(ic, []).append(vc)
    out, paths = {}, None
    for ic, vcs in by_ir.items():
        ctx = look.system_context(sub_, seed, s, sorted(set(vcs) | {"clean"}), ic, dev)
        p = [r["image_path"] for r in ctx.vis_by_cond["clean"]]
        if paths is None:
            paths = p
        elif p != paths:
            raise SystemExit("frame order differs between contexts")
        idx = np.arange(len(p)) if sel is None else np.asarray(ctx.sel(sel))
        for vc in vcs:
            r = run_systems(ctx, vc)
            parts = frame_parts(r["fused_gated"], r["gts"])
            e = ap_from_parts(parts, sel=idx)["per_class"].get(SHIP)
            save_ship_pre(parts, idx, scratch / cell_tag((vc, ic)) / f"seed{seed}_{s}")
            out[(vc, ic)] = float(e["ap50_95"]) if e else float("nan")
        del ctx
    return seed, s, out, paths, (None if sel is None else idx)


_W: dict = {}


def _boot_init(scratch, keys, weights_path):
    _W["scratch"], _W["keys"], _W["pre"] = Path(scratch), keys, {}
    _W["w"] = np.load(weights_path, mmap_mode="r")


def _boot_chunk(task):
    cell, t0, t1 = task
    if cell not in _W["pre"]:
        _W["pre"][cell] = {k: load_ship_pre(_W["scratch"] / cell_tag(cell) / f"seed{k[0]}_{k[1]}")
                           for k in _W["keys"]}
    pres = _W["pre"][cell]
    keys = sorted(pres)
    subs = sorted({s for _, s in keys})
    reps = np.empty(t1 - t0)
    for j, t in enumerate(range(t0, t1)):
        w = np.asarray(_W["w"][t])
        vals = {k: _score(pres[k], w, SHIP) for k in keys}
        reps[j] = float(np.mean([np.mean([vals[(sd, s2)] for sd, s2 in keys if s2 == s]) for s in subs]))
    return cell, t0, reps


# ---- the same bootstrap on the GPU (added 2026-10-10, after the CPU run proved ~8 h) --------------
# `ap_weighted` on one class, exactly: the expanded (resampled) TP flags are cumulated in float64,
# where sums of 0/1 are exact integers in any order; division is IEEE round-to-nearest on both
# devices and the running max is exact, so recall and the precision envelope are bit-identical to
# numpy's. The 101-point interpolation then runs in numpy on the bracketing points only: for each
# grid value, the last index with recall <= r and the one after it (plus the first and last index),
# which are the points numpy's own search picks on the full arrays. `_gpu_check` compares against
# `_score` with `==` before any replicate is used, and the run refuses on one mismatch.

def _ap_gpu(tp_g, fidx_g, gt: np.ndarray, w_g, w_np: np.ndarray, grid_g) -> float:
    import torch
    from uqfusion.eval.apmetrics import NL, RECALL_GRID
    n_gt = int(gt @ w_np)
    if n_gt == 0:
        return float("nan")
    if tp_g.shape[1] == 0:
        return float(np.zeros(NL).mean())
    # (NL, m): every scan runs along the contiguous axis (~30x faster on CUDA than along dim 0)
    tp = torch.repeat_interleave(tp_g, w_g[fidx_g], dim=1)
    m = tp.shape[1]
    if m == 0:
        return float(np.zeros(NL).mean())
    cum_tp = torch.cumsum(tp, dim=1, dtype=torch.float64)
    cum_fp = torch.cumsum(~tp, dim=1, dtype=torch.float64)
    recall = cum_tp / n_gt
    prec = cum_tp / torch.clamp(cum_tp + cum_fp, min=1e-9)
    env = torch.flip(torch.cummax(torch.flip(prec, [1]), dim=1).values, [1])
    j = torch.searchsorted(recall, grid_g.expand(NL, -1).contiguous(), right=True) - 1   # last <= r
    ends = torch.tensor([0, m - 1], device=tp.device).expand(NL, 2)
    ix = torch.cat([ends, j.clamp(0, m - 1), (j + 1).clamp(0, m - 1)], dim=1)
    rx, ex_ = torch.gather(recall, 1, ix).cpu().numpy(), torch.gather(env, 1, ix).cpu().numpy()
    ix = ix.cpu().numpy()
    out = np.empty(NL)
    for k in range(NL):
        _u, first = np.unique(ix[k], return_index=True)                       # ascending index
        xs, fs = rx[k][first], ex_[k][first]
        out[k] = np.interp(RECALL_GRID, xs, fs, left=fs[0], right=0).mean()
    return float(out.mean())


def _gpu_tensors(pre: dict, dev):
    import torch
    d = pre["classes"][SHIP]
    return (torch.from_numpy(np.array(d["tp"])).to(dev).T.contiguous(),         # (NL, n) on the GPU
            torch.from_numpy(np.ascontiguousarray(d["fidx"])).to(dev), np.asarray(pre["gt"][SHIP]))


def _gpu_check(pre_dir: Path, cells, keys, W, dev, grid_g) -> int:
    """`_ap_gpu` against the CPU `_score` with `==`, on every cell, three keys, four replicates and
    the observed (all-ones) weights. Returns the number of comparisons; raises on any mismatch."""
    import torch
    n = 0
    for c in cells:
        for key in (keys[0], keys[len(keys) // 2], keys[-1]):
            pre = load_ship_pre(pre_dir / cell_tag(c) / f"seed{key[0]}_{key[1]}")
            g = _gpu_tensors(pre, dev)
            for t in (None, 0, len(W) // 2, len(W) - 1):
                w = np.ones(pre["n_frames"], dtype=np.int64) if t is None else np.asarray(W[t])
                a = _score(pre, w, SHIP)
                b = _ap_gpu(*g, torch.from_numpy(np.ascontiguousarray(w)).to(dev), w, grid_g)
                if not (a == b or (np.isnan(a) and np.isnan(b))):
                    raise SystemExit(f"GPU bootstrap differs from CPU on {cell_tag(c)} {key} t={t}: {a!r} vs {b!r}")
                n += 1
    return n


def gpu_boot(pre_dir: Path, cells, keys, W) -> dict:
    import torch
    from uqfusion.eval.apmetrics import RECALL_GRID
    dev = torch.device("cuda")
    grid_g = torch.from_numpy(RECALL_GRID).to(dev)
    t0 = time.time()
    print(f"[gpu] checked {_gpu_check(pre_dir, cells, keys, W, dev, grid_g)} replicates against the CPU "
          f"path, all identical ({time.time() - t0:.0f}s)", flush=True)
    Wg = torch.from_numpy(np.ascontiguousarray(W)).to(dev)
    keys = sorted(keys)
    subs = sorted({s for _, s in keys})
    reps = {}
    for ci, c in enumerate(cells):
        G = {k: _gpu_tensors(load_ship_pre(pre_dir / cell_tag(c) / f"seed{k[0]}_{k[1]}"), dev) for k in keys}
        r = np.empty(len(W))
        for t in range(len(W)):
            w_np = np.asarray(W[t])
            vals = {k: _ap_gpu(*G[k], Wg[t], w_np, grid_g) for k in keys}
            r[t] = float(np.mean([np.mean([vals[(sd, s2)] for sd, s2 in keys if s2 == s]) for s in subs]))
            if t % 100 == 99:
                print(f"[gpu] {cell_tag(c)} {t + 1}/{len(W)} ({time.time() - t0:.0f}s)", flush=True)
        reps[c] = r
        del G
        torch.cuda.empty_cache()
        print(f"[gpu] cell {ci + 1}/{len(cells)} {cell_tag(c)} done ({time.time() - t0:.0f}s)", flush=True)
    return reps


def resample_weights(paths, idx, n_boot: int) -> np.ndarray:
    """The block resamples `boot_cell` draws: fresh rng(BOOT_SEED) per cell, same frames, so
    every cell sees the same 1,000 weight vectors. Drawn once here."""
    per_run = run_slices(run_ids([paths[i] for i in idx]))
    rng = np.random.default_rng(BOOT_SEED)
    n = len(idx)
    return np.stack([np.bincount(block_resample(rng, per_run, BLOCK_LEN), minlength=n)
                     for _ in range(n_boot)]).astype(np.int64)


def score(sub_, cells, sel, n_boot, workers, boot_workers, scratch: Path,
          device: str = "cpu", reuse_pre: bool = False) -> dict:
    t0 = time.time()
    pre_dir = scratch / "pre"
    jobs = [(sub_, k, s, cells, sel, pre_dir) for k in sub_.seeds for s in sub_.subs]
    obs_by_cell = {c: {} for c in cells}
    keys = [(k, s) for k in sub_.seeds for s in sub_.subs]
    wp = scratch / "boot_weights.npy"
    if reuse_pre:
        # The presorts and weights an earlier run of this function wrote. The observed value per
        # (system, draw) is `_score` at unit weights on the presort, which equals `ap_from_parts`
        # on the same parts: both take class 0's detections in frame order and sort them with the
        # same stable kind (apmetrics.presort / ap_from_parts).
        W = np.load(wp)
        if W.shape[0] != n_boot:
            raise SystemExit(f"{wp} holds {W.shape[0]} replicates, not {n_boot}")
        for c in cells:
            for k in keys:
                d = pre_dir / cell_tag(c) / f"seed{k[0]}_{k[1]}"
                if not (d / "tp.npy").is_file():
                    raise SystemExit(f"--reuse-pre: {d} missing")
                obs_by_cell[c][k] = _score(load_ship_pre(d), None, SHIP)
        print(f"[score] reused {len(cells) * len(keys)} presorts and {wp.name} ({time.time() - t0:.0f}s)", flush=True)
        reps = (gpu_boot(pre_dir, cells, keys, W) if device == "cuda"
                else _cpu_boot(pre_dir, cells, keys, wp, boot_workers, n_boot, t0))
        return _finish(cells, obs_by_cell, reps)
    paths = idx = None
    with ProcessPoolExecutor(min(workers, len(jobs))) as ex:
        for seed, s, out, p, ix in ex.map(_score_job, jobs):
            if paths is None:
                paths, idx = p, (np.arange(len(p)) if ix is None else ix)
            elif p != paths:
                raise SystemExit("frame order differs between systems")
            for c, v in out.items():
                obs_by_cell[c][(seed, s)] = v
            print(f"[score] system {seed} {s} done ({time.time() - t0:.0f}s)", flush=True)

    W = resample_weights(paths, idx, n_boot)
    scratch.mkdir(parents=True, exist_ok=True)
    np.save(wp, W)
    reps = (gpu_boot(pre_dir, cells, keys, W) if device == "cuda"
            else _cpu_boot(pre_dir, cells, keys, wp, boot_workers, n_boot, t0))
    return _finish(cells, obs_by_cell, reps)


def _cpu_boot(pre_dir, cells, keys, wp, boot_workers, n_boot, t0) -> dict:
    chunk = max(1, -(-n_boot // max(1, boot_workers // max(1, len(cells)) or 1)))
    tasks = [(c, a, min(a + chunk, n_boot)) for c in cells for a in range(0, n_boot, chunk)]
    reps = {c: np.empty(n_boot) for c in cells}
    with ProcessPoolExecutor(min(boot_workers, len(tasks)), initializer=_boot_init,
                             initargs=(str(pre_dir), keys, str(wp))) as ex:
        for c, a, r in ex.map(_boot_chunk, tasks):
            reps[c][a:a + len(r)] = r
    print(f"[boot] {len(cells)} cells x {n_boot} replicates ({time.time() - t0:.0f}s)", flush=True)
    return reps


def _finish(cells, obs_by_cell, reps) -> dict:
    res = {}
    for c in cells:
        per = obs_by_cell[c]
        subs = sorted({s for _, s in per})
        obs = float(np.mean([np.mean([per[(sd, s2)] for sd, s2 in per if s2 == s]) for s in subs]))
        kept = reps[c][np.isfinite(reps[c])]
        lo, hi = np.percentile(kept, [2.5, 97.5])
        res[c] = {"value": obs, "ci_lo": float(lo), "ci_hi": float(hi), "n_effective": int(kept.size),
                  "per_system_draw": {f"seed{sd}/{s}": v for (sd, s), v in sorted(per.items())},
                  "replicates": reps[c].tolist()}
    return res


def selftest(args) -> int:
    """Development data: this file's parallel, class-0 path against `look.boot_cell`."""
    cells = [("clean", None), ("fog", None), ("clean", "fog_s2")]
    sub_ = DevCheck()
    mine = score(sub_, cells, "tune", args.selftest_boot, 3, 6, V2 / "selftest")
    dev = {0: look.dev_context(0, "runs/cache_m")}
    ok = True
    for c in cells:
        ctx = look.system_context(sub_, 0, "shipped", sorted({c[0], "clean"}), c[1], dev[0])
        r = run_systems(ctx, c[0])
        ref = look.boot_cell({(0, "shipped"): frame_parts(r["fused_gated"], r["gts"])},
                             [x["image_path"] for x in ctx.vis_by_cond["clean"]], ctx.sel("tune"), args.selftest_boot)
        same = ref["value"] == mine[c]["value"] and ref["replicates"] == mine[c]["replicates"]
        ok &= same
        print(f"[selftest] {c[0]}/{c[1] or 'clean'}: value {mine[c]['value']!r} vs boot_cell {ref['value']!r}, "
              f"{args.selftest_boot} replicates {'IDENTICAL' if same else 'DIFFER'}", flush=True)
    print(f"[selftest] {'PASS' if ok else 'FAIL'}")
    return 0 if ok else 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--score", action="store_true")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--selftest-boot", type=int, default=40)
    ap.add_argument("--jobs", type=int, default=4, help="build: units at once (GPU streams)")
    ap.add_argument("--workers", type=int, default=None, help="build: corruption processes per unit")
    ap.add_argument("--batch", type=int, default=4, help="build: frames per forward pass")
    ap.add_argument("--score-workers", type=int, default=5, help="(system, draw) contexts at once (RAM-bound)")
    ap.add_argument("--boot-workers", type=int, default=24)
    ap.add_argument("--boot", type=int, default=N_BOOT)
    ap.add_argument("--boot-device", choices=("cpu", "cuda"), default="cpu",
                    help="cuda: the bootstrap on the GPU, checked bit-for-bit against the CPU path first")
    ap.add_argument("--reuse-pre", action="store_true",
                    help="score: reuse the presorts and boot_weights.npy an earlier --score run wrote")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    if args.selftest:
        return selftest(args)
    if args.build:
        return build(args)
    if not args.score or not args.out:
        raise SystemExit("--score needs --out")

    # The single look must exist and must stay exactly as it is.
    for m in (look.MARKER, look.MIRROR):
        if not m.is_file() or not json.loads(m.read_text(encoding="utf-8")).get("numbers_written"):
            raise SystemExit(f"{look.rel(m)} missing or incomplete -- this re-score presupposes the look")
    marker_bytes = {m: m.read_bytes() for m in (look.MARKER, look.MIRROR)}
    n = n_pairs()
    bad = [f"{sub(v)}/{stem}/seed{k}" for v in VIS_DRAWS for mod, stem, kind, sev in STREAMS for k in SEEDS
           if not cache_ok(CACHES / f"seed{k}" / sub(v) / f"{stem}.pkl", weights(mod, k), mod, kind, sev,
                           v + 10 if mod == "ir" else v, n, args.batch)]
    if bad:
        raise SystemExit(f"{len(bad)} v2 caches missing or failing meta, e.g. {bad[0]}")

    t0 = time.time()
    res = score(Pohang04V2(), CELLS, None, args.boot, args.score_workers, args.boot_workers, V2 / "boot",
                device=args.boot_device, reuse_pre=args.reuse_pre)
    v1 = json.loads(LOOK_JSON.read_text(encoding="utf-8"))["cells"]
    clean = v1["clean/clean"]
    for m, b in marker_bytes.items():
        if m.read_bytes() != b:
            raise SystemExit(f"{look.rel(m)} changed during the re-score")

    key = lambda vc, ic: f"{vc}/{ic or 'clean'}"
    rows = [f"| clean/clean (verdict, not re-scored) | {clean['value']:.4f} | [{clean['ci_lo']:.4f}, {clean['ci_hi']:.4f}] "
            f"| {clean['value']:.4f} | — |"]
    for vc, ic in look.CELLS:
        if (vc, ic) == look.VERDICT_CELL:
            continue
        o = v1[key(vc, ic)]
        if (vc, ic) in NOT_MODELLED:
            rows.append(f"| {key(vc, ic)} | — | — | {o['value']:.4f} | {NOT_MODELLED[(vc, ic)]} |")
        else:
            r = res[(vc, ic)]
            rows.append(f"| {key(vc, ic)} | {r['value']:.4f} | [{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | "
                        f"{o['value']:.4f} | {r['value'] - o['value']:+.4f} |")
    r = res[("lowlight", None)]
    rows.append(f"| lowlight/clean (new; replaces lowlight/glare_s2) | {r['value']:.4f} | "
                f"[{r['ci_lo']:.4f}, {r['ci_hi']:.4f}] | — | — |")
    L = ["**A disclosed second exposure of pohang04** (decided 2026-10-10; "
         "`docs/prereg-p04-v2-rescore-2026-10-10.md`). The corrupted cells of the single look "
         "re-scored with corruption v2 (`docs/eval/corruption_v2/README.md`); everything else is the "
         f"look's: five Phase 3 systems, preset `{look.PRESET}`, development calibration injected, "
         f"{n} day pairs, fused ship AP against VIS GT, draws VIS 941–944 / IR 951–954, moving-block "
         f"bootstrap L={BLOCK_LEN}, n_boot={args.boot}, seed {BOOT_SEED}. Descriptive only. The "
         "`clean/clean` verdict (NO-GAP, D = +0.0216) involves no corruption and is not re-scored.",
         "## Cells\n\n| cell (VIS/IR) | v2 AP | v2 95% CI | v1 AP (look) | v2 − v1 / note |\n"
         "|---|---:|---|---:|---|\n" + "\n".join(rows),
         "## Per system and draw\n\n" + "\n".join(
             f"- **{key(*c)}**: " + ", ".join(f"{k} {x:.4f}" for k, x in res[c]["per_system_draw"].items())
             for c in CELLS),
         f"Caches built at inference batch {args.batch} (`build_cache_multi.py --batch`); the look's "
         "caches were batch 1. Measured batch-4 drift on a Phase 3 checkpoint: |d ship AP| ~1e-5 "
         "(`docs/eval/corruption_v2/batched_inference.json`). Gate statistics for the corrupted "
         "streams were computed in the build pass under the GPU interpreter (numpy/opencv versions "
         "in each JSON); the look's clean statistics are reused. The two interpreters agree to "
         "≤ 6.3e-7 relative on float statistics and exactly on percentile statistics."]
    from uqfusion.eval.identity import system_identity
    ident = system_identity(look.dev_context(0, "runs/cache_p3/seed0"), preset=look.PRESET, seeds=list(SEEDS),
                            draws=list(VIS_DRAWS), block_len=BLOCK_LEN, n_boot=args.boot,
                            bootstrap_seed=BOOT_SEED, corrupt_version="v2", infer_batch=args.batch)
    write_md(Path(args.out), "pohang04 corrupted cells under corruption v2 (second exposure, disclosed)", L,
             identity=ident)
    Path(args.out).with_suffix(".json").write_text(json.dumps(
        {"cells": {key(*c): {k: x for k, x in r.items() if k != "replicates"} for c, r in res.items()},
         "not_modelled": {key(*c): why for c, why in NOT_MODELLED.items()},
         "v1_look": {k: v1[k]["value"] for k in v1}, "infer_batch": args.batch,
         "elapsed_s": time.time() - t0}, indent=1), encoding="utf-8")
    print(f"[rescore] written to {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
