"""N4g on the held-out run through a new glare cell (P-G, 2026-10-12).

Registered in `docs/prereg-ttbn-glare-p04-2026-10-12.md`. N4g is N4 with a glare term in the VIS side of
the gate, (q_vis < 1 OR (VIS mean > 117.92 AND VIS std > 62.31)) AND IR reads day AND ir_d2 <= 222.7
(`ttbn_variants.py`). pohang04's registered cells had no VIS glare, so this adds glare/clean (v2 glare,
severity 2, draws 941-944, IR clean) and scores N4g there and on the nine P-T cells.

Records. N4g reuses N4's pohang04 records for every block N4 ran; new blocks (every glare block, and the
few blocks of clean and fog where the glare term gates a frame N4's grid never ran) are built here, on N4's
grid and arithmetic. Where N4g's gate equals N4's, N4g's stream is N4's, bit for bit.

    py -3.13 scripts/ttbn_glare_p04.py check-fp16        # development: decides the new blocks' precision
    py -3.13 scripts/ttbn_glare_p04.py build-plain       # plain glare caches + frame statistics, pohang04
    py -3.13 scripts/ttbn_glare_p04.py build-adapted     # N4g's new blocks
    py -3.13 scripts/ttbn_glare_p04.py score             # presorts
    py -3.13 scripts/p04_heldout_score.py boot           # replicates of every new presort
    py -3.13 scripts/ttbn_glare_p04.py report
"""

from __future__ import annotations

import argparse
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
import ttbn_variants as tv  # noqa: E402

GLARE = ("glare", 2)
STEM = "gauss_vis_paired_glare"
OUT = ROOT / "runs/holdout_p04/ttbn_glare"
DEV_CHECK = ROOT / "runs/novel/ttbn_variants/glare_fp16_check.json"
DEV_CHECK_DRAW = 942
#: VIS stream -> the IR conditions it is paired with in the registered cells (P-T's, plus glare/clean)
IR_OF = {**{c: (None,) for c in ttbn.P04_CONDS}, **ttbn.P04_IR_OF, "glare": (None,)}
NEW_CELL = ("glare", None)
#: cells built and scored for N4g (the gate differs from N4's on some draw, or the cell is new)
BUILT = [("clean", "blur_s2"), ("clean", "fog_s2"), ("fog", None), NEW_CELL]
N4G_PRESORTS = ("union_ttbng/fused", "ttbng/vis", "ttbng_lex/fused")
N4_PRESORTS = ("union_ttbn/fused", "ttbn/vis", "ttbn_lex/fused")
FREE_PRESORTS = ("single/vis", "single/ir", "shipped/fused", "union/fused", "lex/fused")


def p04_draws():
    import p04_heldout as ph
    return ph.VIS_DRAWS


# ------------------------------------------------------------------ gates (frame statistics)
def gate_g_p04(vc: str, ic, s: str) -> np.ndarray:
    """N4g's gate of one pohang04 cell and draw, from its frame-statistics files."""
    draw = None if s == "clean" else int(s[4:7])
    q = ttbn.vis_unhealthy(ttbn._p04_stats("vis", vc, draw, "structure"))
    g = tv.glare_term(ttbn._p04_stats("vis", vc, draw, "brightness"))
    return (q | g) & ttbn.ir_day_ok(ttbn._p04_stats("ir", ic, draw, "brightness"),
                                    ttbn._p04_stats("ir", ic, draw, "structure"))


def gate_n4_p04(vc: str, ic, s: str) -> np.ndarray:
    return ttbn.gate_p04(vc, ic, s)


def needed_g(vc: str, draw) -> np.ndarray:
    """Frames N4g gates under any IR condition registered for this VIS stream (any draw, for clean VIS)."""
    import p04_heldout as ph
    out = None
    for d in (ph.VIS_DRAWS if vc == "clean" else (draw,)):
        s = ph.sub(d)
        for ic in IR_OF[vc]:
            g = gate_g_p04(vc, ic, s)
            out = g if out is None else out | g
    return out


def blocks_of(mask: np.ndarray) -> set[int]:
    return {i // ttbn.BATCH for i in np.flatnonzero(mask)}


# ------------------------------------------------------------------ record files
def new_path(k: int, cond: str, draw, fp16: bool) -> Path:
    where = "clean" if cond == "clean" else ttbn.sub(draw)
    return OUT / f"seed{k}" / where / f"gauss_vis_paired_{cond}_bn{int(ttbn.PRIOR_N)}{'_fp16' if fp16 else ''}.pkl"


def n4_ran(cond: str, draw) -> set[int]:
    """Blocks N4's pohang04 build ran for this VIS stream (from system 0's records; all five share the grid)."""
    if cond not in ttbn.P04_CONDS:
        return set()
    p = ttbn.out_path("p04", 0, cond, None if cond == "clean" else draw, True)
    if not p.is_file():
        return set()
    with open(p, "rb") as fh:
        rr = pickle.load(fh)["records"]
    return {i // ttbn.BATCH for i, r in enumerate(rr) if r is not None}


def fp16_ok() -> bool:
    if not DEV_CHECK.is_file():
        raise SystemExit("run check-fp16 first: the new blocks' precision depends on it")
    return bool(json.loads(DEV_CHECK.read_text(encoding="utf-8"))["pass"])


_RC: dict = {}


def _records(p: Path):
    if p not in _RC:
        with open(p, "rb") as fh:
            _RC[p] = pickle.load(fh)["records"]
    return _RC[p]


def gated_stream(k: int, cond: str, draw, plain, gate):
    """Adapted records where `gate` fires (N4's record if N4 ran the block, else the new block's), plain
    elsewhere. Frame identity and `feat` come from the plain record, as in `ttbn.gated_stream`."""
    if not np.any(gate):
        return plain
    n4p = ttbn.out_path("p04", k, cond, None if cond == "clean" else draw, True)
    n4 = _records(n4p) if (cond in ttbn.P04_CONDS and n4p.is_file()) else None
    np_ = new_path(k, cond, draw, fp16_ok())
    new = _records(np_) if np_.is_file() else None
    out = []
    for i, (b, f) in enumerate(zip(plain, gate)):
        if not f:
            out.append(b)
            continue
        a = n4[i] if (n4 is not None and n4[i] is not None) else (new[i] if new is not None else None)
        if a is None:
            raise SystemExit(f"gated frame {i} without an adapted record ({k}, {cond}, {draw})")
        assert Path(a["image_path"]).name == Path(b["image_path"]).name
        out.append({**a, "image_path": b["image_path"], "feat": b["feat"]})
    return out


# ------------------------------------------------------------------ inference (N4's loop)
def _run_blocks(preds, lst, kc, draw, chunks, workers, fp16, tag):
    import torch
    import p04_heldout as ph
    recs = [[None] * len(lst) for _ in ttbn.SEEDS]
    if not chunks:
        return recs
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
            print(f"[{tag}] {n_done}/{len(chunks)} blocks ({time.time() - t0:.0f}s)", flush=True)
    ttbn.MODE["n"] = None
    return recs


def _predictors(fp16: bool):
    import p04_heldout as ph
    from uqfusion.uq.infer import UQPredictor
    preds = [UQPredictor(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt", device=0, imgsz=640,
                         conf=0.001) for k in ttbn.SEEDS]
    for p in preds:
        ttbn.patch_bn(p.model)
    if fp16:
        ph.setup_fp16(preds)
        ttbn.MODE["fused"] = True
    return preds


def _write(path: Path, meta: dict, recs) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path.with_suffix(".tmp"), "wb") as fh:
        pickle.dump({"meta": meta, "records": recs}, fh, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(path.with_suffix(".tmp"), path)


# ------------------------------------------------------------------ development fp16 check
def fp16_dev_path(k: int) -> Path:
    return tv.OUT / "glare_fp16" / f"seed{k}" / ttbn.sub(DEV_CHECK_DRAW) / f"{STEM}_bn16.pkl"


def _check_job(k):
    import dataclasses
    import seed_ensemble as se
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems
    from v2_sensitivity import ship_ap
    ttbn._low()
    se._memo_distances()
    d = DEV_CHECK_DRAW
    b = f"{se.STATS}/brightness/{ttbn.sub(d)}"
    ctx = load_context(preset="crossmodal26m", cache_dir=f"{se.CACHE}/seed{k}/{ttbn.sub(d)}", bright_dir=b,
                       structure_dir=f"{se.STATS}/structure/{ttbn.sub(d)}", ir_bright=f"{b}/gauss_ir_paired_clean.json",
                       conditions=tuple(ttbn.DEV_CONDS), ir_condition=None, verbose=False)
    day = np.flatnonzero(~np.isin(ctx.runs, NIGHT_RUNS))
    irok = ~np.asarray(ctx.ir_night_raw, bool) & (np.asarray(ctx.ir_d2) <= ctx.ir_bound)
    gg = ((np.asarray(ctx.q_vis_by_cond["glare"]) < 1.0)
          | tv.glare_term(ttbn._dev_stats(d, "vis", "glare", "brightness"))) & irok
    if not np.array_equal(gg, tv.gate_g_stats(d, "glare", None)):
        raise SystemExit(f"context gate differs from the frame-statistics gate (system {k})")
    u = dataclasses.replace(ctx, veto=None)
    vis = ctx.vis_by_cond["glare"]
    res = {}
    for name, p in (("fp32", tv.out_path("glare", k, "glare", d)), ("fp16", fp16_dev_path(k))):
        vg = tv._stream(p, vis, gg)
        res[name] = ship_ap(frame_parts(run_systems(u, "glare", compute_metrics=False, vis_records=vg)["fused_gated"],
                                        ctx.gts), day)
    return k, res


def cmd_check_fp16(args) -> int:
    lst = [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    need = tv.gate_g_stats(DEV_CHECK_DRAW, "glare", None)
    chunks = [list(range(a, min(a + ttbn.BATCH, len(lst)))) for a in sorted(b * ttbn.BATCH for b in blocks_of(need))]
    if not all(fp16_dev_path(k).is_file() for k in ttbn.SEEDS):
        ttbn._low()
        preds = _predictors(True)
        recs = _run_blocks(preds, lst, GLARE, DEV_CHECK_DRAW, chunks, args.workers, True, "check")
        for j, k in enumerate(ttbn.SEEDS):
            _write(fp16_dev_path(k), {"variant": "N4g", "cond": "glare", "draw": DEV_CHECK_DRAW, "blocks_run": len(chunks),
                                      "precision": "fp16 autocast, channels_last, cudnn.benchmark, fused statistics"},
                   recs[j])
        del preds
    ref = json.loads((tv.OUT / "score_dev.json").read_text(encoding="utf-8"))
    rows = {}
    with ProcessPoolExecutor(args.score_workers, max_tasks_per_child=1) as ex:
        for k, r in ex.map(_check_job, ttbn.SEEDS):
            stored = ref[f"{DEV_CHECK_DRAW}|None|{k}"]["glare|day"]["union_ttbng"]
            if r["fp32"] != stored:
                raise SystemExit(f"fp32 re-score differs from the stored development score (system {k}): "
                                 f"{r['fp32']!r} vs {stored!r}")
            rows[k] = r
            print(f"[check] system {k}: fp32 {r['fp32']:.5f} fp16 {r['fp16']:.5f}", flush=True)
    a = float(np.mean([rows[k]["fp32"] for k in ttbn.SEEDS]))
    b = float(np.mean([rows[k]["fp16"] for k in ttbn.SEEDS]))
    ok = abs(b - a) <= ttbn.FP16_TOL
    DEV_CHECK.write_text(json.dumps({"pass": bool(ok), "tolerance": ttbn.FP16_TOL, "cell": "glare/clean/day",
                                     "draw": DEV_CHECK_DRAW, "fp32": a, "fp16": b, "diff": b - a, "blocks": len(chunks),
                                     "per_system": {str(k): v for k, v in rows.items()}}, indent=1), encoding="utf-8")
    print(f"[check] union_ttbng glare/clean/day: fp32 {a:.5f} fp16 {b:.5f} diff {b - a:+.5f} -> "
          f"{'PASS: fp16' if ok else 'FAIL: fp32'}", flush=True)
    return 0


# ------------------------------------------------------------------ pohang04: plain glare caches
def cmd_build_plain(args) -> int:
    import holdout_p04_v2_rescore as v2r
    from uqfusion.config import resolve_gpu_python
    n = v2r.n_pairs()
    gp = resolve_gpu_python(None)
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    # Draw 941 ran with the default thread pools: the builder's main process spun ~15 cores and the
    # laptop's shared power budget pulled the GPU from ~1.86 to ~1.4 GHz. Capped at 2 threads it uses
    # ~1.5 cores; checked bit-identical on draw 941 (5 caches x 400 frames, max |diff| 0; statistics equal).
    env.update(OMP_NUM_THREADS="2", MKL_NUM_THREADS="2", OPENBLAS_NUM_THREADS="2")
    (v2r.V2 / "logs").mkdir(parents=True, exist_ok=True)
    kind, sev = GLARE
    for v in (args.draws or p04_draws()):
        todo = [(v2r.weights("vis", k), v2r.CACHES / f"seed{k}" / v2r.sub(v) / f"{STEM}.pkl") for k in ttbn.SEEDS]
        stats = [v2r.DERIVED / kd / v2r.sub(v) / f"{STEM}.json" for kd in ("brightness", "structure")]
        ok = lambda: (all(v2r.cache_ok(o, w, "vis", kind, sev, v, n, 4) for w, o in todo)
                      and all(p.is_file() for p in stats))
        if ok():
            print(f"[plain] {v2r.sub(v)} skip (verified)", flush=True)
            continue
        for _, o in todo:
            o.parent.mkdir(parents=True, exist_ok=True)
        cmd = [gp, "-u", str(ROOT / "scripts/build_cache_multi.py"),
               "--weights", *[str(w) for w, _ in todo], "--outs", *[str(o) for _, o in todo],
               "--images-list", str(v2r.LISTS["vis"]), "--imgsz", "640", "--conf", "0.001", "--data", "vis",
               "--corrupt", kind, "--severity", str(sev), "--corrupt-seed", str(v),
               "--corrupt-version", "v2", "--modality", "vis", "--workers", str(args.workers),
               "--batch", "4", "--stats-out", *[str(p) for p in stats]]
        t = time.time()
        print(f"[plain] {v2r.sub(v)} start ({args.workers} workers, batch 4)", flush=True)
        with open(v2r.V2 / "logs" / f"{v2r.sub(v)}_{STEM}.log", "w", encoding="utf-8") as fh:
            rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
        if rc != 0 or not ok():
            raise SystemExit(f"[plain] {v2r.sub(v)} FAILED rc={rc}")
        print(f"[plain] {v2r.sub(v)} ok in {time.time() - t:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ pohang04: N4g's new blocks
def plan(conds=None) -> list[tuple[str, int | None, np.ndarray, list[int]]]:
    """(VIS stream, draw, N4g-needed frames, new blocks) for every stream N4g needs blocks N4 never ran."""
    out = []
    for vc in (conds or list(ttbn.P04_CONDS) + ["glare"]):
        for d in ([None] if vc == "clean" else p04_draws()):
            need = needed_g(vc, d)
            new = sorted(blocks_of(need) - n4_ran(vc, d))
            if new:
                out.append((vc, d, need, new))
    return out


def cmd_plan(args) -> int:
    for vc, d, need, new in plan(args.conds):
        print(f"{vc} {d}: N4g gates {int(need.sum())} frames; {len(new)} new blocks"
              + (f" {new}" if len(new) <= 10 else ""), flush=True)
    return 0


def cmd_build_adapted(args) -> int:
    import p04_heldout as ph
    from uqfusion.eval.corruptions_v2 import CODE_REV
    fp16 = fp16_ok()
    ttbn._low()
    lst = ph.images("vis")
    todo = [t for t in plan(args.conds) if not all(new_path(k, t[0], t[1], fp16).is_file() for k in ttbn.SEEDS)]
    print(f"[n4g] {len(todo)} streams to build ({'fp16' if fp16 else 'fp32'})", flush=True)
    if not todo:
        return 0
    preds = _predictors(fp16)
    for vc, d, need, new in todo:
        kc = GLARE if vc == "glare" else ttbn.P04_CONDS[vc]
        chunks = [list(range(b * ttbn.BATCH, min((b + 1) * ttbn.BATCH, len(lst)))) for b in new]
        t0 = time.time()
        recs = _run_blocks(preds, lst, kc, d, chunks, args.workers, fp16, f"n4g {vc}/{d}")
        for j, k in enumerate(ttbn.SEEDS):
            _write(new_path(k, vc, d, fp16),
                   {"variant": "N4g", "target": "p04", "cond": vc, "draw": d, "prior_N": ttbn.PRIOR_N,
                    "batch": ttbn.BATCH, "blocks": "N4's grid; only blocks N4's pohang04 build did not run",
                    "new_blocks": new, "needed_frames": int(need.sum()),
                    "precision": "fp16 autocast, channels_last, cudnn.benchmark, fused statistics" if fp16 else "fp32",
                    "corrupt": kc[0] if kc else None, "severity": kc[1] if kc else None,
                    "corrupt_seed": d if kc else None, "corrupt_version": "v2" if kc else None,
                    "corrupt_code": CODE_REV if kc else None,
                    "weights": str(ROOT / f"runs/phase3_stage2/p3_vis_seed{k}/weights/best.pt")}, recs[j])
        print(f"[n4g] {vc}/{d} done: {len(new)} blocks, {time.time() - t0:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ pohang04: scoring
def _substrate():
    import holdout_p04_look as look
    import holdout_p04_v2_rescore as v2r

    class Pohang04V2Glare(v2r.Pohang04V2):
        """The re-score's staging plus the glare stream, linked into the same per-(system, draw) trees."""

        def cache_dir(self, seed, s):
            r = super().cache_dir(seed, s)
            look.Pohang04._link(v2r.CACHES / f"seed{seed}/{s}/{STEM}.pkl", v2r.STAGE / f"seed{seed}" / s / f"{STEM}.pkl")
            return r

        def _derived(self, kind, s):
            r = super()._derived(kind, s)
            look.Pohang04._link(v2r.DERIVED / kind / s / f"{STEM}.json", v2r.STAGE / "derived" / s / kind / f"{STEM}.json")
            return r

    return Pohang04V2Glare()


def _score_job(job):
    import dataclasses
    import holdout_p04_look as look
    import p04_heldout_score as psc
    import seed_ensemble as se
    from gate_zoo import build_lex_records
    from uqfusion.eval.apmetrics import frame_parts
    from uqfusion.eval.ctx import run_systems
    k, s, ic, vc = job
    ttbn._low()
    se._memo_distances()
    tag, key = psc.cell_tag((vc, ic)), f"seed{k}_{s}"
    draw = int(s[4:7])
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    import holdout_p04_v2_rescore as v2r
    sub_ = _substrate() if vc == "glare" else v2r.Pohang04V2()        # P-T's substrate for its own cells
    ctx = look.system_context(sub_, k, s, sorted({vc, "clean"}), ic, dev)
    irok = ~np.asarray(ctx.ir_night_raw, bool) & (np.asarray(ctx.ir_d2) <= ctx.ir_bound)
    g4 = ttbn.gate_ctx(ctx, vc)
    gg = ((np.asarray(ctx.q_vis_by_cond[vc]) < 1.0)
          | tv.glare_term(ttbn._p04_stats("vis", vc, draw, "brightness"))) & irok
    if not np.array_equal(g4, gate_n4_p04(vc, ic, s)) or not np.array_equal(gg, gate_g_p04(vc, ic, s)):
        raise SystemExit(f"context gate differs from the frame-statistics gate ({k}, {s}, {ic}, {vc})")
    u = dataclasses.replace(ctx, veto=None)
    vis = ctx.vis_by_cond[vc]
    info = {"gate_g": float(gg.mean()), "gate": float(g4.mean()), "n_gated_g": int(gg.sum())}

    def put(name, recs):
        d = psc.PRE / name / tag / key
        if not (d / "obs.json").is_file():
            psc.save_pre(frame_parts(recs, ctx.gts), d)

    o0 = run_systems(u, vc, compute_metrics=False)
    irv = o0["ir_in_vis"]
    z = np.zeros(len(vis), bool)
    if (vc, ic) == NEW_CELL:                       # the gate-free arms (as p04_heldout_score) and N4 on the new cell
        osh = run_systems(ctx, vc, compute_metrics=False)
        put("single/vis", vis)
        put("single/ir", osh["ir_in_vis"])
        put("shipped/fused", osh["fused_gated"])
        put("union/fused", o0["fused_gated"])
        put("lex/fused", build_lex_records(vis, osh["ir_in_vis"], z))
        v4 = gated_stream(k, vc, draw, vis, g4)
        put("union_ttbn/fused", run_systems(u, vc, compute_metrics=False, vis_records=v4)["fused_gated"])
        put("ttbn/vis", v4)
        put("ttbn_lex/fused", build_lex_records(v4, irv, z))
    vg = gated_stream(k, vc, draw if vc != "clean" else None, vis, gg)
    put("union_ttbng/fused", run_systems(u, vc, compute_metrics=False, vis_records=vg)["fused_gated"])
    put("ttbng/vis", vg)
    put("ttbng_lex/fused", build_lex_records(vg, irv, z))
    return job, info


def cmd_score(args) -> int:
    import p04_heldout_score as psc
    fp16_ok()
    gp = OUT / "gates.json"
    gates = json.loads(gp.read_text(encoding="utf-8")) if gp.is_file() else {}
    jobs = []
    for vc, ic in [c for c in BUILT if args.cells is None or c[0] in args.cells]:
        names = N4G_PRESORTS + ((FREE_PRESORTS + N4_PRESORTS) if (vc, ic) == NEW_CELL else ())
        for s in psc.subs_of((vc, ic)):
            for k in ttbn.SEEDS:
                tag, pk = psc.cell_tag((vc, ic)), f"seed{k}_{s}"
                if f"{k}|{s}|{ic}|{vc}" in gates and all((psc.PRE / n / tag / pk / "obs.json").is_file() for n in names):
                    continue
                jobs.append((k, s, ic, vc))
    jobs.sort(key=lambda j: ((j[3], j[2]) != NEW_CELL, j[1], j[0]))
    print(f"[score] {len(jobs)} jobs", flush=True)
    t0 = time.time()
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for n, (j, info) in enumerate(ex.map(_score_job, jobs), 1):
            gates[f"{j[0]}|{j[1]}|{j[2]}|{j[3]}"] = info
            tmp = gp.with_suffix(".tmp")
            tmp.write_text(json.dumps(gates, indent=1), encoding="utf-8")
            os.replace(tmp, gp)
            print(f"[score] {n}/{len(jobs)} {j} ({time.time() - t0:.0f}s)", flush=True)
    return 0


# ------------------------------------------------------------------ report
def cmd_report(args) -> int:
    import p04_heldout_score as psc
    cells = [psc.VERDICT_CELL] + psc.CORRUPTED + [NEW_CELL]
    chk = json.loads(DEV_CHECK.read_text(encoding="utf-8"))
    rows, res, worse, better = [], {}, [], []
    for c in cells:
        tag, name = psc.cell_tag(c), psc.cell_tag(c).replace("__", "/")
        subs = psc.subs_of(c)
        g_g = [gate_g_p04(c[0], c[1], s) for s in subs]
        g_4 = [gate_n4_p04(c[0], c[1], s) for s in subs]
        rate_g, rate_4 = float(np.mean([x.mean() for x in g_g])), float(np.mean([x.mean() for x in g_4]))
        same = all(np.array_equal(a, b) for a, b in zip(g_g, g_4))
        un, sh = psc.series("union/fused", c), psc.series("shipped/fused", c)
        if same and c not in BUILT:
            if not any(x.any() for x in g_4):
                res[tag] = {"gate_g": rate_g, "gate": rate_4, "delta": 0.0, "ci": [0.0, 0.0],
                            "note": "no gated frame: union_ttbng is union"}
                rows.append(f"| {name} | {rate_g:.3f} | {rate_4:.3f} | {un[0]:.4f} | 0 (identical) | | 0 | |")
                continue
            tt, src = psc.series("union_ttbn/fused", c), "P-T (gates equal: union_ttbng is union_ttbn)"
            tv_ = psc.series("ttbn/vis", c)
        else:
            tt, src = psc.series("union_ttbng/fused", c), "built"
            tv_ = psc.series("ttbng/vis", c)
        ir = psc.series("single/ir", c)
        if None in (tt, un, sh, tv_, ir):
            raise SystemExit(f"{name}: presorts or replicates missing; run score and boot first")
        d, (lo, hi) = tt[0] - un[0], psc.ci(tt[1] - un[1])
        ds, (los, his) = tt[0] - sh[0], psc.ci(tt[1] - sh[1])
        cl, (cllo, clhi) = tt[0] - max(tv_[0], ir[0]), psc.ci(tt[1] - np.maximum(tv_[1], ir[1]))
        n4 = un if not any(x.any() for x in g_4) else psc.series("union_ttbn/fused", c)
        dg = tt[0] - n4[0]
        res[tag] = {"gate_g": rate_g, "gate": rate_4, "source": src, "union": un[0], "union_ttbng": tt[0], "delta": d,
                    "ci": [lo, hi], "delta_vs_shipped": ds, "ci_vs_shipped": [los, his], "claim": [cl, cllo, clhi],
                    "minus_n4": dg}
        if d <= -0.0014 and hi < 0:
            worse.append(tag)
        if c == NEW_CELL and d >= 0.0060 and lo > 0:
            better.append(tag)
        rows.append(f"| {name} | {rate_g:.3f} | {rate_4:.3f} | {un[0]:.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | "
                    f"{ds:+.4f} [{los:+.4f}, {his:+.4f}] | {dg:+.4f} | {cl:+.4f} [{cllo:+.4f}, {clhi:+.4f}] |")
    g_rate = res[psc.cell_tag(NEW_CELL)]["gate_g"]
    if not worse and better:
        v = "REPLICATES"
    else:
        v = (f"DOES NOT REPLICATE (worse: {', '.join(worse) or 'none'}; glare/clean "
             f"{'gains' if better else 'does not gain'})")
        if not better and g_rate < 0.10:
            v += f"; N4g's gate fires on {g_rate:.1%} of glare/clean frames: the term did not act on pohang04's glare"
    gc = NEW_CELL
    vals = {n: psc.series(n, gc) for n in ("shipped/fused", "union/fused", "lex/fused", "single/vis", "single/ir",
                                           "union_ttbn/fused", "ttbng_lex/fused", "union_ttbng/fused")}
    lx, tl = vals["lex/fused"], vals["ttbng_lex/fused"]
    lo_l, hi_l = psc.ci(tl[1] - lx[1])
    u4, uu = vals["union_ttbn/fused"], vals["union/fused"]
    lo4, hi4 = psc.ci(u4[1] - uu[1])
    L = ["# N4g on the held-out run through a new glare cell (pohang04)", "",
         "`docs/prereg-ttbn-glare-p04-2026-10-12.md`. Ship AP against the VIS ground truth, mean over the five "
         "systems and the four draws of a corrupted cell. Δ = `union_ttbng` − `union`, paired; 95% interval from "
         "the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1). glare/clean is new: VIS v2 glare "
         "severity 2, draws 941–944, IR clean.", "",
         f"**P-G: {v}.**", "",
         f"Precision of N4g's new blocks: {'fp16' if chk['pass'] else 'fp32'} (development check, glare/clean/day, "
         f"draw {chk['draw']}: fp32 {chk['fp32']:.5f}, fp16 {chk['fp16']:.5f}, Δ {chk['diff']:+.5f}, tolerance "
         f"{chk['tolerance']}).", "",
         "| cell (VIS/IR) | N4g gate | N4 gate | union | Δ vs union [95%] | Δ vs shipped [95%] | N4g − N4 | "
         "claim union_ttbng [95%] |", "|---|---:|---:|---:|---:|---:|---:|---:|"] + rows + [
         "", "Cells whose N4g gate equals N4's on every draw take P-T's `union_ttbn` series (the same stream, bit for "
         "bit); the four others are built and scored.", "",
         "## glare/clean, descriptive", "",
         "| arm | ship AP |", "|---|---:|"] + [f"| `{n}` | {x[0]:.4f} |" for n, x in vals.items()] + [
         "", f"* N4 alone (`union_ttbn` − `union`): {u4[0] - uu[0]:+.4f} [{lo4:+.4f}, {hi4:+.4f}]; N4's gate fires on "
             f"{res[psc.cell_tag(gc)]['gate']:.1%} of frames, N4g's on {g_rate:.1%}.",
         f"* `ttbng_lex` − `lex`: {tl[0] - lx[0]:+.4f} [{lo_l:+.4f}, {hi_l:+.4f}].",
         f"* `union` − `shipped`: {uu[0] - vals['shipped/fused'][0]:+.4f}."]
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps({"P-G": v, "fp16_check": chk, "cells": res,
                                                   "glare_clean": {n: x[0] for n, x in vals.items()}}, indent=1),
                                      encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    c = sp.add_parser("check-fp16")
    c.add_argument("--workers", type=int, default=6)
    c.add_argument("--score-workers", type=int, default=3)
    b = sp.add_parser("build-plain")
    b.add_argument("--workers", type=int, default=10)
    b.add_argument("--draws", type=int, nargs="+", default=None)
    pl = sp.add_parser("plan")
    pl.add_argument("--conds", nargs="+", default=None)
    a = sp.add_parser("build-adapted")
    a.add_argument("--workers", type=int, default=6)
    a.add_argument("--conds", nargs="+", default=None, help="only these VIS streams (default: all)")
    s = sp.add_parser("score")
    s.add_argument("--workers", type=int, default=4)
    s.add_argument("--cells", nargs="+", default=None, help="only the built cells with these VIS streams")
    r = sp.add_parser("report")
    r.add_argument("--out", default="docs/eval/ttbn/ttbn_glare_p04_2026-10-12.md")
    args = ap.parse_args()
    return {"check-fp16": cmd_check_fp16, "build-plain": cmd_build_plain, "plan": cmd_plan,
            "build-adapted": cmd_build_adapted, "score": cmd_score, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
