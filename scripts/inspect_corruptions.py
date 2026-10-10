"""What do the synthetic corruptions actually do to a frame? (inspection, 2026-10-09)

Regenerates corrupted frames exactly as the cache builder does — `make_corruption(name,
severity, seed)` applied to `cv2.imread(path)` with the frame's index in the paired list — and
writes contact sheets plus per-condition statistics. Nothing is scored; no detector runs.

Measured on the image rows of the letterboxed canvas (VIS rows 151..488, IR rows 64..575) unless
the column says pad. Draws: VIS 941 and IR 911, as in the Phase 3 development cells.

    py -3.13 scripts/inspect_corruptions.py                 # -> docs/eval/corruption_inspect_2026-10-09/
    py -3.13 scripts/inspect_corruptions.py --out some/dir
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.corruptions import _albu, make_corruption  # noqa: E402

VIS_LIST = ROOT / "runs/derived/paired_val_vis.txt"
IR_LIST = ROOT / "runs/derived/paired_val_ir.txt"
VSEED, ISEED = 941, 911
PAD_V, PAD_I = 151, 64
VIS_CONDS = [("fog", 1), ("fog", 2), ("fog", 3), ("lowlight", 1), ("lowlight", 2), ("lowlight", 3),
             ("glare", 1), ("glare", 2), ("glare", 3), ("rain", 2), ("blur", 3), ("noise", 2)]
IR_CONDS = [("fog", 2), ("glare", 2), ("blur", 2), ("noise", 2)]


def run_of(p: str) -> str:
    return Path(p).parent.name


def rows(im, pad):
    return im[pad:im.shape[0] - pad]


def pad_rows(im, pad):
    return np.concatenate([im[:pad], im[im.shape[0] - pad:]])


def lum(im):
    return cv2.cvtColor(im, cv2.COLOR_BGR2GRAY).astype(np.float32)


def sharpness(im, pad):
    return cv2.Laplacian(cv2.cvtColor(rows(im, pad), cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()


def label(tile, text):
    cv2.rectangle(tile, (0, 0), (tile.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(tile, text, (5, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return tile


def sheet(grid, path, tile=300):
    out = [np.concatenate([label(cv2.resize(im, (tile, tile), interpolation=cv2.INTER_AREA), t)
                           for t, im in r], axis=1) for r in grid]
    w = max(r.shape[1] for r in out)
    out = [np.pad(r, ((0, 4), (0, w - r.shape[1]), (0, 0)), constant_values=255) for r in out]
    cv2.imwrite(str(path), np.concatenate(out, axis=0))


def glare_source_rows(severity, seed, n_total, step=8):
    """Row of the flare source on the 640x640 canvas, from albumentations' applied params."""
    import albumentations as A

    t = A.Compose([_albu("glare", severity)], p=1.0, save_applied_params=True)
    blank = np.full((640, 640, 3), 114, np.uint8)
    ys = []
    for i in range(0, n_total, step):
        t.set_random_seed(seed * 100003 + i)        # the same seeding as make_corruption
        ys.append(t(image=blank)["applied_transforms"][0][1]["flare_center"][1])
    return np.array(ys)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "docs/eval/corruption_inspect_2026-10-09"))
    ap.add_argument("--n-day", type=int, default=150)
    ap.add_argument("--n-night", type=int, default=80)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    vis = [l.strip() for l in VIS_LIST.read_text().splitlines() if l.strip()]
    ir = [l.strip() for l in IR_LIST.read_text().splitlines() if l.strip()]
    tf = {c: make_corruption(c[0], c[1], VSEED) for c in VIS_CONDS}
    tfi = {c: make_corruption(c[0], c[1], ISEED) for c in IR_CONDS}

    # contact sheets: the middle paired frame of pohang00, 02, 03 (day) and pohang01 (night)
    by_run: dict[str, list[int]] = {}
    for i, p in enumerate(vis):
        by_run.setdefault(run_of(p), []).append(i)
    frames = [by_run[r][len(by_run[r]) // 2] for r in ("pohang00", "pohang02", "pohang03", "pohang01")]
    for name, conds in (("vis_sheet_A.png", VIS_CONDS[:6]), ("vis_sheet_B.png", VIS_CONDS[6:])):
        grid = []
        for i in frames:
            im = cv2.imread(vis[i])
            grid.append([(f"{run_of(vis[i])} clean", im)] + [(f"{c} s{s}", tf[(c, s)](im, i)) for c, s in conds])
        sheet(grid, out / name)
    grid = []
    for i in frames:
        im = cv2.imread(ir[i])
        grid.append([(f"IR {run_of(ir[i])} clean", im)] + [(f"IR {c} s{s}", tfi[(c, s)](im, i)) for c, s in IR_CONDS])
    sheet(grid, out / "ir_sheet.png")
    im = cv2.imread(vis[frames[0]])
    cv2.imwrite(str(out / "full_fog_s2.png"), np.concatenate([im, tf[("fog", 2)](im, frames[0])], axis=1))

    # statistics on a fixed random sample of day and night paired frames
    rng = np.random.default_rng(0)
    day = [i for i, p in enumerate(vis) if run_of(p) != "pohang01"]
    night = [i for i, p in enumerate(vis) if run_of(p) == "pohang01"]
    sample = {"day": sorted(int(x) for x in rng.choice(day, args.n_day, replace=False)),
              "night": sorted(int(x) for x in rng.choice(night, args.n_night, replace=False))}
    stats = {}
    for part, idx in sample.items():
        base = {i: cv2.imread(vis[i]) for i in idx}
        sharp0 = {i: sharpness(base[i], PAD_V) for i in idx}
        for cond in [("clean", 0)] + VIS_CONDS:
            recs = []
            for i in idx:
                im = base[i] if cond[0] == "clean" else tf[cond](base[i], i)
                L, P = lum(rows(im, PAD_V)), lum(pad_rows(im, PAD_V))
                recs.append(dict(mean=L.mean(), std=L.std(), p05=np.percentile(L, 5),
                                 p95=np.percentile(L, 95), frac_black=(L <= 2).mean(),
                                 frac_white=(L >= 253).mean(), pad_mean=P.mean(),
                                 pad_changed=(np.abs(P - 114) > 3).mean(),
                                 sharpness=sharpness(im, PAD_V), sharp_ratio=sharpness(im, PAD_V) / max(sharp0[i], 1e-6)))
            stats[f"{part}/{cond[0]}_s{cond[1]}"] = {k: float(np.mean([r[k] for r in recs])) for k in recs[0]}
    gl = {}
    for s in (1, 2, 3):
        ys = glare_source_rows(s, VSEED, len(vis))
        gl[f"glare_s{s}"] = {"n": int(len(ys)), "frac_source_in_top_pad": float((ys < PAD_V).mean()),
                             "row_min": int(ys.min()), "row_max": int(ys.max())}
    import albumentations
    meta = {"albumentations": albumentations.__version__, "vis_seed": VSEED, "ir_seed": ISEED,
            "sample": {k: len(v) for k, v in sample.items()}, "sheet_frames": [vis[i] for i in frames]}
    (out / "stats.json").write_text(json.dumps({"meta": meta, "cells": stats, "glare_source": gl}, indent=1))

    print(f"{'cell':20s} {'mean':>6s} {'std':>5s} {'p05':>5s} {'p95':>5s} {'black':>6s} {'sharp':>6s} "
          f"{'padMean':>7s} {'padChg':>6s}")
    for k, a in stats.items():
        print(f"{k:20s} {a['mean']:6.1f} {a['std']:5.1f} {a['p05']:5.1f} {a['p95']:5.1f} "
              f"{a['frac_black']:6.3f} {a['sharp_ratio']:6.3f} {a['pad_mean']:7.1f} {a['pad_changed']:6.2f}")
    for k, g in gl.items():
        print(f"{k}: flare source in the top pad on {100 * g['frac_source_in_top_pad']:.1f}% of {g['n']} frames "
              f"(rows {g['row_min']}..{g['row_max']})")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
