"""v1 vs v2 corruptions on the same frames: what each does, measured (2026-10-09).

Same fixed sample as `inspect_corruptions.py` (150 day + 80 night paired val frames,
seed 0; VIS draw 941, IR draw 911), so the v1 rows reproduce that report. Nothing is
scored and no detector runs. GT boxes are read only to measure how much contrast the
TARGETS keep (`box_contrast`: RMS contrast inside each VIS GT box, corrupted / clean,
median over boxes), the quantity a detector actually loses.

Writes to docs/eval/corruption_v2/: stats.json, vis_sheet_{fog,lowlight,glare,other}.png,
ir_sheet.png, timing.json. All frames are processed on a process pool.

    py -3.13 scripts/inspect_corruptions_v2.py --workers 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")     # no network check in 30 fresh workers
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.corruptions import make_corruption       # noqa: E402
from uqfusion.eval.frame_geometry import content_rows       # noqa: E402

VIS_LIST = ROOT / "runs/derived/paired_val_vis.txt"
IR_LIST = ROOT / "runs/derived/paired_val_ir.txt"
VSEED, ISEED = 941, 911
VIS_CONDS = [(k, s) for k in ("fog", "lowlight", "glare") for s in (1, 2, 3)] + [("rain", 2), ("blur", 3), ("noise", 2)]
IR_CONDS = [("fog", 1), ("fog", 2), ("fog", 3), ("blur", 2), ("noise", 2)]
IR_V1_ONLY = [("glare", 2)]
_G: dict = {}


def run_of(p: str) -> str:
    return Path(p).parent.name


def lum(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)


def boxes_of(path: str, lo: int):
    lab = ROOT / "Pohang_dataset/visible/labels" / Path(path).parent.name / (Path(path).stem + ".txt")
    out = []
    if lab.is_file():
        for ln in lab.read_text().splitlines():
            p = ln.split()
            if len(p) >= 5:
                xc, yc, bw, bh = map(float, p[1:5])
                x0, x1 = int((xc - bw / 2) * 640), int(np.ceil((xc + bw / 2) * 640))
                y0, y1 = int((yc - bh / 2) * 640), int(np.ceil((yc + bh / 2) * 640))
                if (x1 - x0) >= 3 and (y1 - y0) >= 3:
                    out.append((max(x0, 0), max(y0, lo), min(x1, 640), y1))
    return out


def measure(im, base, mod, boxes):
    lo, hi = content_rows(mod)
    L, B = lum(im[lo:hi]), lum(base[lo:hi])
    P = lum(np.concatenate([im[:lo], im[hi:]]))
    P0 = lum(np.concatenate([base[:lo], base[hi:]]))
    g8 = lambda x: cv2.cvtColor(x[lo:hi], cv2.COLOR_BGR2GRAY)      # uint8, as inspect_corruptions.py
    lap = cv2.Laplacian(g8(im), cv2.CV_64F).var() / max(cv2.Laplacian(g8(base), cv2.CV_64F).var(), 1e-6)
    r = dict(mean=float(L.mean()), std=float(L.std()), p05=float(np.percentile(L, 5)),
             p95=float(np.percentile(L, 95)), frac_black=float((L <= 2).mean()),
             frac_white=float((L >= 253).mean()), sharp_ratio=float(lap),
             pad_changed=float((np.abs(P - P0) > 0).mean()),
             colour=float(np.abs(im[lo:hi].astype(np.int16)[..., 0] - im[lo:hi].astype(np.int16)[..., 2]).mean()))
    q = lambda x: float(np.percentile(x, 99) - np.percentile(x, 1))
    r["range_kept"] = q(L) / max(q(B), 1e-6)            # IR: the stretch should keep this ~1
    if boxes:
        g, g0 = lum(im), lum(base)
        ratios = [g[y0:y1, x0:x1].std() / max(g0[y0:y1, x0:x1].std(), 1e-3) for x0, y0, x1, y1 in boxes]
        r["box_contrast"] = float(np.median(ratios))
    return r


def _init(vis, ir):
    cv2.setNumThreads(1)
    _G["vis"], _G["ir"] = vis, ir
    _G["tf"] = {}
    for v in ("v1", "v2"):
        for c in VIS_CONDS:
            _G["tf"][(v, "vis", c)] = make_corruption(c[0], c[1], VSEED, version=v, modality="vis", images=vis)
        for c in IR_CONDS + (IR_V1_ONLY if v == "v1" else []):
            _G["tf"][(v, "ir", c)] = make_corruption(c[0], c[1], ISEED, version=v, modality="ir", images=ir)


def task(args):
    i, part = args
    out = {}
    lo_v, _ = content_rows("vis")
    bx = boxes_of(_G["vis"][i], lo_v) if part == "day" else []
    for mod, conds in (("vis", VIS_CONDS), ("ir", IR_CONDS + IR_V1_ONLY)):
        base = cv2.imread(_G[mod][i])
        out[f"{mod}/{part}/clean_s0/v0"] = measure(base, base, mod, bx if mod == "vis" else [])
        for c in conds:
            for v in ("v1", "v2"):
                tf = _G["tf"].get((v, mod, c))
                if tf is None:
                    continue
                t = time.perf_counter()
                im = tf(base, i)
                dt = time.perf_counter() - t
                m = measure(im, base, mod, bx if mod == "vis" else [])
                m["ms"] = dt * 1e3
                lp = getattr(tf, "last_params", {}) or {}
                if "ae_gain" in lp:
                    m["ae_gain"] = float(lp["ae_gain"])
                if v == "v2" and c[0] == "glare" and "src_xy" in lp:
                    # mean shift of the frame AWAY from the source (> 150 px): the AE's darkening
                    lo, hi = content_rows(mod)
                    yy, xx = np.mgrid[0:hi - lo, 0:640]
                    far = np.hypot(xx - lp["src_xy"][0], yy - lp["src_xy"][1]) > 150
                    m["nonsource_mean_ratio"] = float(lum(im[lo:hi])[far].mean() / max(lum(base[lo:hi])[far].mean(), 1e-3))
                out[f"{mod}/{part}/{c[0]}_s{c[1]}/{v}"] = m
    return out


def label(tile, text):
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 20), (0, 0, 0), -1)
    cv2.putText(tile, text, (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def sheet(rows, path, mod, tile_w=320):
    lo, hi = content_rows(mod)
    th = int(round((hi - lo) * tile_w / 640))
    out = []
    for r in rows:
        out.append(np.concatenate([label(cv2.resize(im[lo:hi], (tile_w, th), interpolation=cv2.INTER_AREA), t)
                                   for t, im in r], axis=1))
    w = max(x.shape[1] for x in out)
    out = [np.pad(x, ((0, 3), (0, w - x.shape[1]), (0, 0)), constant_values=255) for x in out]
    cv2.imwrite(str(path), np.concatenate(out, axis=0))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=str(ROOT / "docs/eval/corruption_v2"))
    ap.add_argument("--workers", type=int, default=16,
                    help="each worker imports torch via albumentations (~2 GB commit); 30 hit the commit limit "
                         "while other jobs ran")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis = [ln.strip() for ln in VIS_LIST.read_text().splitlines() if ln.strip()]
    ir = [ln.strip() for ln in IR_LIST.read_text().splitlines() if ln.strip()]
    rng = np.random.default_rng(0)                       # the same sample as inspect_corruptions.py
    day = [i for i, p in enumerate(vis) if run_of(p) != "pohang01"]
    night = [i for i, p in enumerate(vis) if run_of(p) == "pohang01"]
    sample = {"day": sorted(int(x) for x in rng.choice(day, 150, replace=False)),
              "night": sorted(int(x) for x in rng.choice(night, 80, replace=False))}
    jobs = [(i, part) for part, idx in sample.items() for i in idx]
    t0 = time.time()
    res = []
    with ProcessPoolExecutor(args.workers, initializer=_init, initargs=(vis, ir)) as ex:
        futs = [ex.submit(task, j) for j in jobs]
        for n, f in enumerate(as_completed(futs), 1):
            res.append(f.result())
            if n % 25 == 0 or n == len(futs):
                print(f"[inspect] {n}/{len(futs)} frames ({time.time() - t0:.0f}s)", flush=True)
    wall = time.time() - t0
    agg: dict[str, dict] = {}
    for r in res:
        for k, m in r.items():
            for kk, vv in m.items():
                agg.setdefault(k, {}).setdefault(kk, []).append(vv)
    stats = {k: {kk: float(np.mean(vv)) for kk, vv in m.items()} for k, m in sorted(agg.items())}
    (out / "stats.json").write_text(json.dumps({"meta": {"vis_seed": VSEED, "ir_seed": ISEED,
                                                        "sample": {k: len(v) for k, v in sample.items()},
                                                        "workers": args.workers, "wall_s": wall},
                                               "cells": stats}, indent=1), encoding="utf-8")

    # contact sheets: middle paired frame of pohang00, 02, 03 (day) and pohang01 (night)
    _init(vis, ir)
    by_run: dict[str, list[int]] = {}
    for i, p in enumerate(vis):
        by_run.setdefault(run_of(p), []).append(i)
    frames = [by_run[r][len(by_run[r]) // 2] for r in ("pohang00", "pohang02", "pohang03", "pohang01")]
    for kind in ("fog", "lowlight", "glare"):
        rows = []
        for i in frames:
            im = cv2.imread(vis[i])
            for v in ("v1", "v2"):
                rows.append([(f"{run_of(vis[i])} clean", im)] +
                            [(f"{v} {kind} s{s}", _G["tf"][(v, "vis", (kind, s))](im, i)) for s in (1, 2, 3)])
        sheet(rows, out / f"vis_sheet_{kind}.png", "vis")
    rows = []
    for i in frames:
        im = cv2.imread(vis[i])
        for v in ("v1", "v2"):
            rows.append([(f"{run_of(vis[i])} clean", im)] +
                        [(f"{v} {c} s{s}", _G["tf"][(v, "vis", (c, s))](im, i)) for c, s in VIS_CONDS[9:]])
    sheet(rows, out / "vis_sheet_other.png", "vis")
    rows = []
    for i in frames:
        im = cv2.imread(ir[i])
        rows.append([(f"IR {run_of(ir[i])} clean", im)] +
                    [(f"v1 {c} s{s}", _G["tf"][("v1", "ir", (c, s))](im, i)) for c, s in [("fog", 2), ("glare", 2), ("noise", 2)]] +
                    [(f"v2 {c} s{s}", _G["tf"][("v2", "ir", (c, s))](im, i)) for c, s in [("fog", 1), ("fog", 2), ("fog", 3), ("noise", 2)]])
    sheet(rows, out / "ir_sheet.png", "ir", tile_w=256)

    keys = ("mean", "std", "frac_black", "frac_white", "sharp_ratio", "box_contrast", "range_kept", "ae_gain",
            "nonsource_mean_ratio", "pad_changed", "colour", "ms")
    print(f"{'cell':34s} " + " ".join(f"{k[:9]:>9s}" for k in keys))
    for k, m in stats.items():
        print(f"{k:34s} " + " ".join(f"{m[x]:9.3f}" if x in m else f"{'-':>9s}" for x in keys))
    print(f"[inspect] {len(jobs)} frames x {len(VIS_CONDS) * 2 + len(IR_CONDS) * 2 + 1} transforms "
          f"in {wall:.0f}s on {args.workers} workers -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
