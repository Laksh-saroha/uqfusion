"""Night test on pohang01's test split (`docs/prereg-p01-nighttest-2026-10-11.md`).

    py -3.13 scripts/p01_nighttest.py lists                 # pair lists + manifest (no image, no label read)
    .venv/Scripts/python.exe scripts/p01_nighttest.py stats # frame statistics, development arithmetic
    py -3.13 scripts/p01_nighttest.py caches                # detection caches, five systems, fp32
    py -3.13 scripts/p01_nighttest.py score                 # presorts and observed APs, every arm
    py -3.13 scripts/p01_nighttest.py boot                  # moving-block replicates (CPU)
    py -3.13 scripts/p01_nighttest.py report
    py -3.13 scripts/p01_nighttest.py build-tta             # Amendment 1: flipped + NLM-restored IR views, fp16
    py -3.13 scripts/p01_nighttest.py score-tta             # Amendment 1: full_tta, full_tta_lex (then boot, report)

Frames: pohang01 pairs (`Pohang_dataset/paired/pohang01_pairs.csv`) whose VIS frame is in the VIS test
split and whose IR frame is in the IR test split, sorted by VIS frame name (capture order). Systems are
assembled as the pohang04 look assembles them (`holdout_p04_look.system_context`: development calibration
injected, nothing fitted here). Arms: `shipped`, `union` (veto off), `lex` (E9), `ens5_union` (five
checkpoints per stream, plain WBF 0.55, IR NMS 0.70, development ensemble prior). Clean frames only.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

OUT = ROOT / "runs" / "night_p01test"
PRE = OUT / "pre"
SEEDS = (0, 1, 2, 3, 4)
RUN = "pohang01"
PAIRS = ROOT / "Pohang_dataset" / "paired" / f"{RUN}_pairs.csv"
VIS_IMG = ROOT / "Pohang_dataset" / "visible" / "images" / RUN
IR_IMG = ROOT / "Pohang_dataset" / "infrared" / "images" / RUN
LISTS = {"vis": OUT / "pairs_vis.txt", "ir": OUT / "pairs_ir.txt"}
MANIFEST = OUT / "pairs_manifest.csv"
IMGSZ, CONF = 640, 0.001
ENS_IOU, IR_NMS = 0.55, 0.70
SHIP = 0
ARMS = ("shipped", "union", "lex", "ens5_union", "full_tta", "full_tta_lex")
STREAM_OF = {"shipped": "single", "union": "single", "lex": "single", "ens5_union": "ens5", "full_tta": "full",
             "full_tta_lex": "full"}       # Amendment 1: full_tta, full_tta_lex (descriptive)
TAG = "clean__clean"
TTA = OUT / "tta"
#: the two test stretches (VIS frame index), for the per-stretch values
STRETCHES = {"test A (11,059–11,575)": (0, 20000), "test B (26,161–26,752)": (20000, 10 ** 9)}


def _low() -> None:
    if os.name == "nt":
        import ctypes
        k32 = ctypes.windll.kernel32
        k32.SetPriorityClass(k32.GetCurrentProcess(), 0x4000)      # BELOW_NORMAL, inherited by children


def read_list(p: Path) -> list[str]:
    return [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip()]


def _split(mod: str) -> set[str]:
    d = "visible" if mod == "vis" else "infrared"
    return {Path(ln.strip()).name for ln in (ROOT / f"Pohang_dataset/{d}/test.txt").read_text().splitlines()
            if f"/{RUN}/" in ln.replace("\\", "/")}


# ------------------------------------------------------------------ lists
def cmd_lists(args) -> int:
    vt, it = _split("vis"), _split("ir")
    with open(PAIRS, newline="", encoding="utf-8") as fh:
        rows = sorted((r for r in csv.DictReader(fh) if r["stereo_L_file"] in vt and r["ir_file"] in it),
                      key=lambda r: r["stereo_L_file"])
    vis = [VIS_IMG / r["stereo_L_file"] for r in rows]
    ir = [IR_IMG / r["ir_file"] for r in rows]
    missing = [p for p in vis + ir if not p.is_file()]
    if missing:
        raise SystemExit(f"{len(missing)} paired images missing, e.g. {missing[0]}")
    OUT.mkdir(parents=True, exist_ok=True)
    LISTS["vis"].write_text("\n".join(str(p) for p in vis) + "\n", encoding="utf-8")
    LISTS["ir"].write_text("\n".join(str(p) for p in ir) + "\n", encoding="utf-8")
    with open(MANIFEST, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["idx", "run", "vis_image", "ir_image", "dt_ms"])
        for k, (v, i, r) in enumerate(zip(vis, ir, rows)):
            w.writerow([k, RUN, str(v), str(i), r["dt_ms"]])
    print(f"[lists] {len(rows)} pairs (VIS test {len(vt)}, IR test {len(it)})", flush=True)
    return 0


# ------------------------------------------------------------------ frame statistics (.venv)
def cmd_stats(args) -> int:
    import cv2
    import holdout_p04_frame_stats as fst
    fst.verify(args.verify_n)                     # bit-exact against development files, or SystemExit
    for mod in ("vis", "ir"):
        paths = read_list(LISTS[mod])
        rows, br, st = fst.measure(paths, mod, None, None, None)
        for d, frames in (("brightness", br), ("structure", st)):
            p = OUT / "derived" / d / "clean" / f"gauss_{mod}_paired_clean.json"
            p.parent.mkdir(parents=True, exist_ok=True)
            payload = {"cache": f"runs/night_p01test/caches/seed*/clean/gauss_{mod}_paired_clean.pkl",
                       "modality": mod, "content_rows": list(rows), "corrupt": None, "severity": None,
                       "corrupt_seed": None, "n_frames": len(frames), "numpy": np.__version__,
                       "opencv": cv2.__version__, "frames": frames}
            tmp = p.with_suffix(".json.tmp")
            tmp.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp, p)
        print(f"[stats] {mod}: {len(br)} frames", flush=True)
    return 0


# ------------------------------------------------------------------ caches (GPU)
def _cache_out(k: int, mod: str) -> Path:
    return OUT / "caches" / f"seed{k}" / "clean" / f"gauss_{mod}_paired_clean.pkl"


def _weights(k: int, mod: str) -> Path:
    return ROOT / f"runs/phase3_stage2/p3_{mod}_seed{k}/weights/best.pt"


def _meta_ok(path: Path, k: int, mod: str, n: int) -> str | None:
    from uqfusion.eval.cache import load_cache
    try:
        recs, m = load_cache(path)
    except Exception as exc:                                      # noqa: BLE001
        return f"unreadable ({type(exc).__name__})"
    checks = {"n_records": (len(recs), n), "weights": (Path(m["weights"][0]).resolve(), _weights(k, mod).resolve()),
              "images_list": (Path(m["images_list"]).resolve(), LISTS[mod].resolve()),
              "corrupt": (m.get("corrupt"), None), "conf": (m.get("conf"), CONF)}
    bad = [f"{a} {x!r} != {y!r}" for a, (x, y) in checks.items() if x != y]
    return "; ".join(bad) or None


def cmd_caches(args) -> int:
    _low()
    n = len(read_list(LISTS["vis"]))
    env = dict(os.environ)
    env["PYTHONPATH"] = str(ROOT / "src") + os.pathsep + env.get("PYTHONPATH", "")
    for k in SEEDS:
        for mod in ("vis", "ir"):
            out = _cache_out(k, mod)
            if out.is_file() and _meta_ok(out, k, mod, n) is None:
                print(f"[caches] seed{k} {mod} skip (meta verified)", flush=True)
                continue
            out.parent.mkdir(parents=True, exist_ok=True)
            tmp = out.with_suffix(".pkl.tmp")
            tmp.unlink(missing_ok=True)
            cmd = [sys.executable, "-u", str(ROOT / "scripts/build_cache.py"), "--source", "gaussian",
                   "--weights", str(_weights(k, mod)), "--data", mod, "--images-list", str(LISTS[mod]),
                   "--imgsz", str(IMGSZ), "--conf", str(CONF), "--out", str(tmp)]
            t = time.time()
            with open(out.with_suffix(".log"), "w", encoding="utf-8") as fh:
                rc = subprocess.call(cmd, stdout=fh, stderr=subprocess.STDOUT, cwd=str(ROOT), env=env)
            why = f"build exited {rc}" if rc else _meta_ok(tmp, k, mod, n)
            if why:
                raise SystemExit(f"[caches] seed{k} {mod} FAILED: {why}")
            os.replace(tmp, out)
            print(f"[caches] seed{k} {mod} ok in {time.time() - t:.0f}s", flush=True)
    return 0


# ------------------------------------------------------------------ Amendment 1: full_tta views (GPU)
def tta_out(k: int, mod: str, view: str) -> Path:
    return TTA / f"seed{k}" / f"gauss_{mod}_paired_clean_{view}.pkl"


def ir_flags_night() -> np.ndarray:
    """`ctx.ir_d2 > ctx.ir_bound` as `p04_heldout.ir_flags` computes it, from this test's own IR structure
    statistics (`stats`) and the frozen health model. Checked against every system context in `score-tta`."""
    sconst = json.loads((ROOT / "runs/eval/structure_constants.json").read_text(encoding="utf-8"))
    hm = sconst["axes"]["ir_health"]
    fr = json.loads((OUT / "derived/structure/clean/gauss_ir_paired_clean.json").read_text(encoding="utf-8"))["frames"]
    X = np.stack([[f[k] for k in hm["keys"]] for f in fr]).astype(float)
    for j, k in enumerate(hm["keys"]):
        if k in hm["log1p_keys"]:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Z = (X - np.asarray(hm["mean"])) / np.asarray(hm["std"])
    d2 = np.einsum("ij,jk,ik->i", Z, np.asarray(hm["precision"]), Z)
    return d2 > float(hm["bound"])


def cmd_build_tta(args) -> int:
    """Flipped views (VIS and IR) and NLM-restored IR views where the IR health check fails, five checkpoints,
    built as `p04_heldout.py build-unit` builds pohang04's: fp16 autocast, channels_last, batch 8."""
    import pickle
    import torch
    import p04_heldout as ph
    from restore import _rows
    from uqfusion.uq.infer import UQPredictor
    _low()
    flag = ir_flags_night()
    print(f"[tta] IR health check fails on {int(flag.sum())}/{flag.size} frames", flush=True)
    for mod in ("vis", "ir"):
        outs = [tta_out(k, mod, "flip") for k in SEEDS] + ([tta_out(k, "ir", "nlm") for k in SEEDS] if mod == "ir" else [])
        if all(o.is_file() for o in outs):
            print(f"[tta] {mod}: present", flush=True)
            continue
        lst = read_list(LISTS[mod])
        n = len(lst)
        flagged = set(np.flatnonzero(flag).tolist()) if mod == "ir" else set()
        spec = {"kind": None, "sev": 0, "seed": 0, "mod": mod, "images": lst, "rows": _rows(mod), "flagged": flagged}
        preds = [UQPredictor(_weights(k, mod), device=0, imgsz=IMGSZ, conf=CONF) for k in SEEDS]
        ph.setup_fp16(preds)
        flips = [[None] * n for _ in SEEDS]
        nlms = [{} for _ in SEEDS]
        buf_f, buf_r = [], []
        t0 = time.time()

        def flush_f():
            if buf_f:
                with ph.amp(True):
                    out = ph._multi_batch(preds, [im for _, im, _ in buf_f])
                for j in range(len(SEEDS)):
                    for (i, _im, w), r in zip(buf_f, out[j]):
                        flips[j][i] = ph._unflip(r, w)
                buf_f.clear()

        def flush_r():
            if buf_r:
                with ph.amp(True):
                    out = ph._multi_batch(preds, [im for _, im in buf_r])
                for j in range(len(SEEDS)):
                    for (i, _im), r in zip(buf_r, out[j]):
                        nlms[j][i] = ph._f32(r)
                buf_r.clear()

        for c, (i, fim, rim, w) in enumerate(ph.frames(spec, list(range(n)), args.workers)):
            buf_f.append((i, fim, w))
            if rim is not None:
                buf_r.append((i, rim))
            if len(buf_f) >= ph.BATCH:
                flush_f()
            if len(buf_r) >= ph.BATCH:
                flush_r()
            if (c + 1) % 250 == 0:
                print(f"[tta] {mod} {c + 1}/{n} ({time.time() - t0:.0f}s)", flush=True)
        flush_f()
        flush_r()
        meta = {"images_list": str(LISTS[mod]), "mod": mod, "cond": "clean", "draw": None, "imgsz": IMGSZ,
                "conf": CONF, "corrupt": None, "infer_batch": ph.BATCH, "built_by": "p01_nighttest.py build-tta",
                "torch": torch.__version__, "precision": "fp16 autocast, channels_last, cudnn.benchmark"}
        for j, k in enumerate(SEEDS):
            assert all(r is not None for r in flips[j])
            items = [(tta_out(k, mod, "flip"), {**meta, "weights": str(_weights(k, mod)),
                                                "view": "hflip of the unrestored frame, boxes mapped back"}, flips[j])]
            if mod == "ir":
                assert set(nlms[j]) == flagged
                items.append((tta_out(k, "ir", "nlm"), {**meta, "weights": str(_weights(k, mod)),
                                                        "view": "NLM-restored frame, flagged frames only",
                                                        "flagged": sorted(flagged)}, nlms[j]))
            for o, m, recs in items:
                o.parent.mkdir(parents=True, exist_ok=True)
                tmp = o.with_suffix(".tmp")
                with open(tmp, "wb") as fh:
                    pickle.dump({"meta": m, "records": recs}, fh, protocol=pickle.HIGHEST_PROTOCOL)
                os.replace(tmp, o)
        print(f"[tta] {mod} done in {time.time() - t0:.0f}s", flush=True)
        del preds
        torch.cuda.empty_cache()
    return 0


# ------------------------------------------------------------------ scoring
def _substrate():
    import holdout_p04_look as look

    class P01Test(look.Substrate):
        """Hard-linked staging (as `holdout_p04_look.Pohang04`): one directory per system."""

        def __init__(self):
            super().__init__("pohang01-test", SEEDS, ["clean"], str(MANIFEST.relative_to(ROOT)), None)
            self.stage = OUT / "stage"

        @staticmethod
        def _link(src: Path, dst: Path) -> None:
            dst.parent.mkdir(parents=True, exist_ok=True)
            if dst.exists():
                if os.path.samefile(src, dst):
                    return
                dst.unlink()
            os.link(src, dst)

        def cache_dir(self, seed, sub):
            d = self.stage / f"seed{seed}" / sub
            for m in ("vis", "ir"):
                self._link(ROOT / f"runs/cache_p3/seed{seed}/gauss_{m}_train_clean.pkl", d / f"gauss_{m}_train_clean.pkl")
                self._link(_cache_out(seed, m), d / f"gauss_{m}_paired_clean.pkl")
            return str(d.relative_to(ROOT))

        def _derived(self, kind, sub):
            d = self.stage / "derived" / sub / kind
            for m in ("vis", "ir"):
                self._link(OUT / f"derived/{kind}/clean/gauss_{m}_paired_clean.json", d / f"gauss_{m}_paired_clean.json")
            return str(d.relative_to(ROOT))

        def bright_dir(self, sub): return self._derived("brightness", sub)
        def struct_dir(self, sub): return self._derived("structure", sub)

    return P01Test()


def _save(name: str, key: str, recs, gts) -> float:
    import p04_heldout_score as psc
    from uqfusion.eval.apmetrics import frame_parts
    d = PRE / name / TAG / key
    if (d / "obs.json").is_file():
        return json.loads((d / "obs.json").read_text())["ap"]
    return psc.save_pre(frame_parts(recs, gts), d)


def _score_job(k: int) -> dict:
    import holdout_p04_look as look
    import seed_ensemble as se
    from gate_zoo import build_lex_records
    from uqfusion.eval.ctx import NIGHT_RUNS, run_systems
    _low()
    se._memo_distances()
    sub_ = _substrate()
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    ctx = look.system_context(sub_, k, "clean", ["clean"], None, dev)
    paths = read_list(LISTS["vis"])
    got = [r["image_path"] for r in ctx.vis_by_cond["clean"]]
    if [Path(p).name for p in got] != [Path(p).name for p in paths]:
        raise SystemExit(f"system {k}: frame order differs from the pair list")
    if not np.all(np.isin(ctx.runs, NIGHT_RUNS)):
        raise SystemExit(f"system {k}: a frame outside the night run")
    key, info = f"seed{k}", {}
    out = run_systems(ctx, "clean", compute_metrics=False)
    vis, irv = ctx.vis_by_cond["clean"], out["ir_in_vis"]
    info["ir_night"] = float(np.mean(np.asarray(ctx.ir_night, bool))) if ctx.ir_night is not None else None
    info["shipped_veto_vis"] = float(np.mean(np.asarray(out["veto_vis"], bool)))
    info["single/vis"] = _save("single/vis", key, vis, ctx.gts)
    info["single/ir"] = _save("single/ir", key, irv, ctx.gts)
    info["shipped/fused"] = _save("shipped/fused", key, out["fused_gated"], ctx.gts)
    info["lex/fused"] = _save("lex/fused", key, build_lex_records(vis, irv, np.zeros(len(vis), bool)), ctx.gts)
    u = run_systems(dataclasses.replace(ctx, veto=None), "clean", compute_metrics=False)
    info["union/fused"] = _save("union/fused", key, u["fused_gated"], ctx.gts)
    if k == 0:
        # five checkpoints per stream (`p04_heldout_score.ens_stream`, kind `ens5`), development prior
        from seed_ensemble import merge
        from uqfusion.eval.cache import load_cache
        from uqfusion.eval.irdedup import nms_records
        devcal = json.loads((ROOT / "runs/holdout_p04/gatefree/devcal.json").read_text(encoding="utf-8"))
        per = {m: [load_cache(_cache_out(j, m))[0] for j in SEEDS] for m in ("vis", "ir")}
        per["ir"] = [nms_records(r, IR_NMS) for r in per["ir"]]
        ens = {m: [merge(list(fr), ENS_IOU) for fr in zip(*per[m])] for m in ("vis", "ir")}
        cap0 = (ctx.cap_vis, ctx.cap_ir)
        ctx.cap_vis, ctx.cap_ir = devcal["ens5_union"]["cap_vis"], devcal["ens5_union"]["cap_ir"]
        e = run_systems(dataclasses.replace(ctx, veto=None), "clean", compute_metrics=False,
                        vis_records=ens["vis"], ir_records=ens["ir"])
        ctx.cap_vis, ctx.cap_ir = cap0
        info["ens5/vis"] = _save("ens5/vis", "all", ens["vis"], ctx.gts)
        info["ens5/ir"] = _save("ens5/ir", "all", e["ir_in_vis"], ctx.gts)
        info["ens5_union/fused"] = _save("ens5_union/fused", "all", e["fused_gated"], ctx.gts)
    return {"k": k, **info}


def cmd_score(args) -> int:
    from concurrent.futures import ProcessPoolExecutor
    obs_p = OUT / "observed.json"
    obs = json.loads(obs_p.read_text(encoding="utf-8")) if obs_p.is_file() else {}
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for r in ex.map(_score_job, SEEDS):
            obs[f"seed{r['k']}"] = r
            tmp = obs_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(obs, indent=1), encoding="utf-8")
            os.replace(tmp, obs_p)
            print(f"[score] system {r['k']} done", flush=True)
    return 0


def _score_tta_job(k: int) -> dict:
    """Amendment 1. Every system: the IR flag equals the context's. System 0 only: the `full` streams and the
    two arms (identical for every system), as `p04_heldout_score.ens_stream` / `_parts_job` build them."""
    import pickle
    import holdout_p04_look as look
    import seed_ensemble as se
    from gate_zoo import build_lex_records
    from seed_ensemble import merge
    from uqfusion.eval.cache import load_cache
    from uqfusion.eval.ctx import run_systems
    from uqfusion.eval.irdedup import nms_records
    _low()
    se._memo_distances()
    dev = look.dev_context(k, f"runs/cache_p3/seed{k}")
    ctx = look.system_context(_substrate(), k, "clean", ["clean"], None, dev)
    flag = ir_flags_night()
    if not np.array_equal(flag, ctx.ir_d2 > ctx.ir_bound):
        raise SystemExit(f"system {k}: the IR flag differs from the context's")
    info = {"k": k, "ir_flagged": int(flag.sum())}
    if k != 0:
        return info

    def recs(p):
        with open(p, "rb") as fh:
            return pickle.load(fh)["records"]

    vis = [load_cache(_cache_out(j, "vis"))[0] for j in SEEDS] + [recs(tta_out(j, "vis", "flip")) for j in SEEDS]
    ir = []
    for j in SEEDS:
        raw, nl = load_cache(_cache_out(j, "ir"))[0], recs(tta_out(j, "ir", "nlm"))
        assert set(nl) == set(np.flatnonzero(flag).tolist()), "restored frames differ from the flag"
        ir.append(nms_records([{**nl[i], "image_path": a["image_path"], "feat": a["feat"]} if f else a
                               for i, (a, f) in enumerate(zip(raw, flag))], IR_NMS))
    ir += [nms_records(recs(tta_out(j, "ir", "flip")), IR_NMS) for j in SEEDS]
    ens = {"vis": [merge(list(fr), ENS_IOU) for fr in zip(*vis)], "ir": [merge(list(fr), ENS_IOU) for fr in zip(*ir)]}
    devcal = json.loads((ROOT / "runs/holdout_p04/gatefree/devcal.json").read_text(encoding="utf-8"))
    ctx.cap_vis, ctx.cap_ir = devcal["full_tta"]["cap_vis"], devcal["full_tta"]["cap_ir"]
    e = run_systems(dataclasses.replace(ctx, veto=None), "clean", compute_metrics=False,
                    vis_records=ens["vis"], ir_records=ens["ir"])
    info["full/vis"] = _save("full/vis", "all", ens["vis"], ctx.gts)
    info["full/ir"] = _save("full/ir", "all", e["ir_in_vis"], ctx.gts)
    info["full_tta/fused"] = _save("full_tta/fused", "all", e["fused_gated"], ctx.gts)
    info["full_tta_lex/fused"] = _save("full_tta_lex/fused", "all",
                                       build_lex_records(ens["vis"], e["ir_in_vis"], np.zeros(len(vis[0]), bool)),
                                       ctx.gts)
    return info


def cmd_score_tta(args) -> int:
    from concurrent.futures import ProcessPoolExecutor
    obs_p = OUT / "observed.json"
    obs = json.loads(obs_p.read_text(encoding="utf-8")) if obs_p.is_file() else {}
    with ProcessPoolExecutor(args.workers, max_tasks_per_child=1) as ex:
        for r in ex.map(_score_tta_job, SEEDS):
            obs.setdefault("tta", {})[f"seed{r['k']}"] = r
            tmp = obs_p.with_suffix(".tmp")
            tmp.write_text(json.dumps(obs, indent=1), encoding="utf-8")
            os.replace(tmp, obs_p)
            print(f"[score-tta] system {r['k']} done: {r}", flush=True)
    return 0


# ------------------------------------------------------------------ bootstrap
def cmd_boot(args) -> int:
    import holdout_p04_look as look
    import p04_heldout_score as psc
    from boot_cpu import boot_cpu
    from holdout_p04_v2_rescore import load_ship_pre, resample_weights
    _low()
    wp = OUT / "boot_weights.npy"
    if not wp.is_file():
        paths = read_list(LISTS["vis"])
        np.save(wp, resample_weights(paths, np.arange(len(paths)), look.N_BOOT))
    W = np.load(wp)
    todo = sorted(p.parent for p in PRE.rglob("obs.json") if not (p.parent / "reps.npy").is_file())
    checked = False
    for d in todo:
        pre = load_ship_pre(d)
        if not checked:
            print(f"[boot] checked {psc._check(pre, W, None, None, cpu=True)} rows equal to the CPU reference", flush=True)
            checked = True
        np.save(d / "reps.npy", boot_cpu(pre, W))
        print(f"[boot] {d.relative_to(PRE)}", flush=True)
    return 0


# ------------------------------------------------------------------ report
def series(arm_stream: str):
    base = PRE / arm_stream / TAG
    keys = [base / f"seed{k}" for k in SEEDS] if (base / "seed0").is_dir() else [base / "all"]
    if not all((d / "reps.npy").is_file() for d in keys):
        return None
    v = float(np.mean([json.loads((d / "obs.json").read_text())["ap"] for d in keys]))
    return v, np.mean([np.load(d / "reps.npy") for d in keys], axis=0), \
        [json.loads((d / "obs.json").read_text())["ap"] for d in keys]


def stretch_values(arm_stream: str) -> dict:
    """Ship AP of one arm/stream on each test stretch (mean over systems for per-system arms)."""
    from holdout_p04_v2_rescore import load_ship_pre
    from uqfusion.eval.apmetrics import _score
    base = PRE / arm_stream / TAG
    keys = [base / f"seed{k}" for k in SEEDS] if (base / "seed0").is_dir() else [base / "all"]
    ix = np.array([int(Path(p).stem.split("_")[-1]) for p in read_list(LISTS["vis"])])
    pres = [load_ship_pre(d) for d in keys]
    return {name: float(np.mean([_score(p, ((ix >= a) & (ix < b)).astype(np.int64), 0) for p in pres]))
            for name, (a, b) in STRETCHES.items()}


def ci(r: np.ndarray) -> tuple[float, float]:
    r = r[np.isfinite(r)]
    lo, hi = np.percentile(r, [2.5, 97.5])
    return float(lo), float(hi)


def cmd_report(args) -> int:
    obs = json.loads((OUT / "observed.json").read_text(encoding="utf-8"))
    n = len(read_list(LISTS["vis"]))
    sh = series("shipped/fused")
    res = {}
    L = ["# Night test on pohang01's test split", "",
         f"`docs/prereg-p01-nighttest-2026-10-11.md`. {n} night frame pairs never seen by any fusion system. Ship "
         "AP against the VIS labels, mean over the five Phase 3 systems; Δ = arm − `shipped`, paired; 95% "
         "interval from 1,000 moving-block replicates (L = 20, seed 1). Claim = fused − max(VIS, IR) of the "
         "arm's own streams.", "",
         "| arm | ship AP | Δ vs shipped [95%] | VIS | IR | claim [95%] | per system |",
         "|---|---:|---:|---:|---:|---:|---|"]
    for arm in ARMS:
        f, v, i = series(f"{arm}/fused"), series(f"{STREAM_OF[arm]}/vis"), series(f"{STREAM_OF[arm]}/ir")
        if None in (f, v, i, sh):
            L.append(f"| {arm} | pending | | | | | |")
            continue
        d, (lo, hi) = f[0] - sh[0], ci(f[1] - sh[1])
        cl, (cllo, clhi) = f[0] - max(v[0], i[0]), ci(f[1] - np.maximum(v[1], i[1]))
        res[arm] = {"ap": f[0], "delta": d, "ci": [lo, hi], "vis": v[0], "ir": i[0], "claim": [cl, cllo, clhi],
                    "per_system": f[2]}
        L.append(f"| {arm} | {f[0]:.4f} | {'—' if arm == 'shipped' else f'{d:+.4f} [{lo:+.4f}, {hi:+.4f}]'} | "
                 f"{v[0]:.4f} | {i[0]:.4f} | {cl:+.4f} [{cllo:+.4f}, {clhi:+.4f}] | "
                 f"{', '.join(f'{x:.4f}' for x in f[2])} |")
    u = res.get("union")
    if u is None:
        p1 = p2 = "PENDING"
    else:
        p1 = "REPLICATES" if (u["delta"] >= 0.0060 and u["ci"][0] > 0) else "DOES NOT REPLICATE"
        p2 = "FAIL" if (u["claim"][0] <= -0.0060 and u["claim"][2] < 0) else "PASS"
    irn = [obs[f"seed{k}"]["ir_night"] for k in SEEDS if f"seed{k}" in obs]
    vv = [obs[f"seed{k}"]["shipped_veto_vis"] for k in SEEDS if f"seed{k}" in obs]
    L[4:4] = [f"**P-N1 (`union` beats `shipped` at night): {p1}.**  ",
              f"**P-N2 (`union` keeps the claim at night): {p2}.**", ""]
    L += ["", f"IR reads night on {np.mean(irn):.1%} of frames (mean over systems); `shipped` vetoes VIS on "
              f"{np.mean(vv):.1%}." if irn and vv else ""]
    res["P-N1"], res["P-N2"] = p1, p2
    un = series("union/fused")
    if un is not None and all(a in res for a in ("full_tta", "full_tta_lex")):
        L += ["", "## Amendment 1: `full_tta` and `full_tta_lex` (descriptive, no verdict)", "",
              "Δ against `union`, paired, on the same replicates.", "",
              "| arm | ship AP | Δ vs `union` [95%] | claim [95%] |", "|---|---:|---:|---:|"]
        for arm in ("full_tta", "full_tta_lex"):
            f = series(f"{arm}/fused")
            d, (lo, hi) = f[0] - un[0], ci(f[1] - un[1])
            res[arm]["vs_union"] = [d, lo, hi]
            c = res[arm]["claim"]
            L.append(f"| {arm} | {f[0]:.4f} | {d:+.4f} [{lo:+.4f}, {hi:+.4f}] | {c[0]:+.4f} [{c[1]:+.4f}, {c[2]:+.4f}] |")
        rows = {"shipped": "shipped/fused", "union": "union/fused", "ens5_union": "ens5_union/fused",
                "full_tta": "full_tta/fused", "full_tta_lex": "full_tta_lex/fused", "VIS": "single/vis",
                "VIS, full": "full/vis", "IR": "single/ir", "IR, full": "full/ir"}
        sv = {a: stretch_values(s) for a, s in rows.items()}
        res["stretches"] = sv
        L += ["", "Per test stretch (ship AP; mean over systems where the arm is per system):", "",
              "| stretch | " + " | ".join(rows) + " |", "|---|" + "---:|" * len(rows)]
        for name in STRETCHES:
            L.append(f"| {name} | " + " | ".join(f"{sv[a][name]:.4f}" for a in rows) + " |")
    o = ROOT / args.out
    o.parent.mkdir(parents=True, exist_ok=True)
    o.write_text("\n".join(L) + "\n", encoding="utf-8")
    o.with_suffix(".json").write_text(json.dumps(res, indent=1), encoding="utf-8")
    print("\n".join(L))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    sp.add_parser("lists")
    s = sp.add_parser("stats")
    s.add_argument("--verify-n", type=int, default=40)
    sp.add_parser("caches")
    s = sp.add_parser("build-tta")
    s.add_argument("--workers", type=int, default=3)
    s = sp.add_parser("score")
    s.add_argument("--workers", type=int, default=3)
    s = sp.add_parser("score-tta")
    s.add_argument("--workers", type=int, default=3)
    sp.add_parser("boot")
    r = sp.add_parser("report")
    r.add_argument("--out", default="docs/eval/night_p01test_2026-10-11.md")
    args = ap.parse_args()
    return {"lists": cmd_lists, "stats": cmd_stats, "caches": cmd_caches, "build-tta": cmd_build_tta,
            "score": cmd_score, "score-tta": cmd_score_tta, "boot": cmd_boot, "report": cmd_report}[args.cmd](args)


if __name__ == "__main__":
    raise SystemExit(main())
