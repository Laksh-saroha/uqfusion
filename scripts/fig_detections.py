"""Figures 9-11 of PAPER_DRAFT2.md: the shipped rule on real frames.

Figure 9 (fig_detections): one day and one night frame, clean and fogged. Figure 10
(fig_detections_conditions): the same two frames under low light and glare, so the two
figures together cover all eight benchmark cells. Figure 11 (fig_detections_scenes): three
further scenes, clean. Columns: the VIS detector, the IR detector (its frame warped into the
VIS canvas by the per-frame homography the fusion uses, so all panels share one geometry),
and the fused output of `crossmodal26m`. Every box comes from the development caches through
`run_systems`, the path that scores Table 3b; nothing is re-run. **Reads no pohang04 frame.**

System: Phase 3 VIS seed 0 + IR seed 0; corruptions are VIS draw 941 at severity 2, and IR
stays clean, as in every Table 3b cell. A corrupted frame is regenerated with
`make_corruption` at the frame's index in the paired list, the same call the cache builder
made. `--corrupt-version v2` (the default since 2026-10-10) draws from the corruption-v2
caches (runs/cache_p3dev_v2, docs/eval/corruption_v2/README.md); `v1` reproduces the
2026-10-09 figures.

How the frames were chosen, never on a detection: `--candidates DIR` writes the candidate
pool, ground truth only. A ship is co-visible when its box centre lies inside the IR field of
view warped into the VIS canvas. Per run, frames with at least two co-visible ships are ranked
by median co-visible ship height times their count (capped at six), keeping one frame per
CAND_GAP capture ordinals; the top CAND_K per run are rendered with ground truth and no
detection. Each run's sheets were reviewed visually for legibility at print size (ship size,
distinct ships, IR coverage, framing, scene variety; parallel AI-agent reviews, 2026-10-09),
which also proposed the crops; FRAMES records the frames and crops taken from those reviews. Night VIS is brightened for display only
(gamma NIGHT_GAMMA); the detector saw the stored frame. Ship class only, the class every
table reports; boxes shown at confidence >= CONF; fused boxes are coloured by the stream they
came from (at IoU 0.85 the merge is concatenation, so each fused box is one input box).

Writes docs/figures/fig_detections*.{pdf,png} and fig_detections.json (frames, crops, flags,
and each shown box's best IoU with the ship ground truth).

Usage (from the repo root):
    python scripts/fig_detections.py
    python scripts/fig_detections.py --candidates <dir>     # the ground-truth-only pool
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import cv2                                                                  # noqa: E402
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.lines import Line2D                                         # noqa: E402
from matplotlib.patches import Polygon, Rectangle                           # noqa: E402

from paper_figures import DOUBLE, INK, IR, VIS, save                        # noqa: E402
from uqfusion.eval.corruptions import make_corruption                       # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems         # noqa: E402

PRESET = "crossmodal26m"
SEED, DRAW, SEV = 0, 941, 2
SUB = f"draw{DRAW}_{DRAW + 10}"
PV = ROOT / "runs/derived/paired_val_vis.txt"
SHIP = 0
CONF = 0.25
NIGHT_GAMMA = 0.45
CROP_W, CROP_H = 320, 170
VIS_ROWS = (151, 489)          # letterboxed VIS content rows (§3.3)
IR_ROWS = (64, 576)            # letterboxed IR content rows
PAD = 114
CONDS = ("clean", "fog", "lowlight", "glare")
CAND_K = {"pohang00": 24, "pohang02": 16, "pohang03": 12, "pohang01": 40}
CAND_GAP = 40

#: name -> (VIS file, crop top-left x, y in the 640 canvas)
FRAMES = {
    "port_day": ("pohang03_L_011704.png", 221, 240),      # three berthed vessels, all inside the IR view
    "ferry_night": ("pohang01_L_014451.png", 147, 251),   # lit ferry and two small boats
    "quay_day": ("pohang00_L_012572.png", 141, 253),      # four vessels along the quay
    "tug_day": ("pohang03_L_011921.png", 221, 245),       # tug beside a large ship
    "moored_night": ("pohang01_L_014898.png", 205, 254),  # row of moored ships
}
FIGS = {
    "fig_detections": [("port_day", "clean"), ("port_day", "fog"), ("ferry_night", "clean"), ("ferry_night", "fog")],
    "fig_detections_conditions": [("port_day", "lowlight"), ("port_day", "glare"),
                                  ("ferry_night", "lowlight"), ("ferry_night", "glare")],
    "fig_detections_scenes": [("quay_day", "clean"), ("tug_day", "clean"), ("moored_night", "clean")],
}
COND_LABEL = {"clean": "clean", "fog": "fog", "lowlight": "low light", "glare": "glare"}


def ships(rec, conf=None):
    b = np.asarray(rec["boxes_xyxy"], dtype=np.float64).reshape(-1, 4)
    m = np.asarray(rec["cls"]).astype(int) == SHIP
    if conf is not None:
        m &= np.asarray(rec["conf"], dtype=np.float64) >= conf
    return b[m], (np.asarray(rec["conf"], dtype=np.float64)[m] if "conf" in rec else None)


def iou(a, b):
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0]); y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2]); y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    area = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])          # noqa: E731
    return inter / np.maximum(area(a)[:, None] + area(b)[None, :] - inter, 1e-9)


def ir_footprint(h) -> np.ndarray:
    """Mask of the VIS canvas that the IR camera's content rows land on."""
    m = np.zeros((640, 640), np.uint8)
    m[IR_ROWS[0]:IR_ROWS[1]] = 1
    return cv2.warpPerspective(m, np.asarray(h, np.float64), (640, 640), flags=cv2.INTER_NEAREST) > 0


def ir_outline(h) -> np.ndarray:
    """Corners of the IR content rows in the VIS canvas."""
    c = np.array([[[0, IR_ROWS[0]], [640, IR_ROWS[0]], [640, IR_ROWS[1]], [0, IR_ROWS[1]]]], np.float64)
    return cv2.perspectiveTransform(c, np.asarray(h, np.float64))[0]


def covisible(gt, h) -> np.ndarray:
    """Ground-truth ship boxes whose centre the IR camera also sees."""
    if not len(gt):
        return gt
    cx, cy = ((gt[:, 0] + gt[:, 2]) / 2).astype(int), ((gt[:, 1] + gt[:, 3]) / 2).astype(int)
    return gt[ir_footprint(h)[np.clip(cy, 0, 639), np.clip(cx, 0, 639)]]


def crop_box(gt) -> tuple[int, int]:
    """Default crop: centred on the mean co-visible ship centre."""
    c = np.array([(gt[:, 0] + gt[:, 2]).mean() / 2, (gt[:, 1] + gt[:, 3]).mean() / 2])
    x0 = int(np.clip(round(c[0] - CROP_W / 2), 0, 640 - CROP_W))
    y0 = int(np.clip(round(c[1] - CROP_H / 2), VIS_ROWS[0], VIS_ROWS[1] - CROP_H))
    return x0, y0


def for_display(im, run):
    return np.clip(255 * (im / 255.0) ** NIGHT_GAMMA, 0, 255).astype(np.uint8) if run in NIGHT_RUNS else im


VERSION = "v2"               # set from --corrupt-version in main()


def roots() -> tuple[str, str]:
    return ("runs/cache_p3dev_v2", "runs/derived_p3dev_v2") if VERSION == "v2" else ("runs/cache_p3dev", "runs/derived_p3dev")


def context(conds=CONDS):
    cr, sr = roots()
    ctx = load_context(preset=PRESET, cache_dir=f"{cr}/seed{SEED}/{SUB}",
                       bright_dir=f"{sr}/brightness/{SUB}",
                       structure_dir=f"{sr}/structure/{SUB}", conditions=conds, verbose=False)
    if any("pohang04" in r for r in ctx.runs):
        raise SystemExit("a pohang04 frame reached the development context")
    pv = [Path(ln.strip()).name for ln in PV.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for c in conds:
        if [Path(r["image_path"]).name for r in ctx.vis_by_cond[c]] != pv:
            raise SystemExit(f"{c}: cache frame order differs from {PV.name}; the corruption index would be wrong")
    return ctx, {n: i for i, n in enumerate(pv)}


def candidates(out: Path) -> None:
    """The candidate pool, ground truth only: no detection is read, drawn or written."""
    ctx, _ = context(("clean",))
    out.mkdir(parents=True, exist_ok=True)
    ordn = np.array([int(Path(r["image_path"]).stem.split("_")[-1]) for r in ctx.vis_by_cond["clean"]])
    pool = []
    for i, g in enumerate(ctx.gts):
        cv = covisible(ships(g)[0], ctx.h_frames[i])
        if len(cv) >= 2:
            pool.append((float(np.median(cv[:, 3] - cv[:, 1])) * min(len(cv), 6), i, cv))
    for run, k in CAND_K.items():
        taken = []
        for _, i, cv in sorted((p for p in pool if ctx.runs[p[1]] == run), key=lambda p: -p[0]):
            if all(abs(ordn[i] - ordn[j]) >= CAND_GAP for j in taken):
                taken.append(i)
                x0, y0 = crop_box(cv)
                vis = for_display(cv2.imread(str(ctx.vis_by_cond["clean"][i]["image_path"])), run)
                ir = cv2.warpPerspective(cv2.imread(str(ctx.ir_clean[i]["image_path"])),
                                         np.asarray(ctx.h_frames[i], np.float64), (640, 640),
                                         borderValue=(PAD, PAD, PAD))
                strip = vis[VIS_ROWS[0]:VIS_ROWS[1]].copy()
                cv2.rectangle(strip, (x0, y0 - VIS_ROWS[0]), (x0 + CROP_W, y0 - VIS_ROWS[0] + CROP_H),
                              (255, 255, 255), 1)
                crops = [img[y0:y0 + CROP_H, x0:x0 + CROP_W].copy() for img in (vis, ir)]
                for x1, y1, x2, y2 in ships(ctx.gts[i])[0].astype(int):
                    cv2.rectangle(strip, (x1, y1 - VIS_ROWS[0]), (x2, y2 - VIS_ROWS[0]), (0, 255, 255), 1)
                    for c in crops:
                        cv2.rectangle(c, (x1 - x0, y1 - y0), (x2 - x0, y2 - y0), (0, 255, 255), 1)
                sheet = np.vstack([cv2.resize(strip, (1280, 676)), cv2.resize(np.hstack(crops), (1280, 340))])
                cv2.imwrite(str(out / f"{run}_{Path(ctx.vis_by_cond['clean'][i]['image_path']).name}"), sheet)
            if len(taken) == k:
                break
        print(f"[candidates] {run}: {len(taken)} frames -> {out}")


def draw_boxes(ax, boxes, color, x0, y0, gt=False):
    for x1, y1, x2, y2 in boxes:
        xy, w, h = (x1 - x0, y1 - y0), x2 - x1, y2 - y1
        if gt:
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec="white", lw=1.3))
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec=INK, lw=1.3, ls=(0, (2.2, 2.2))))
        else:
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec="white", lw=2.2, alpha=0.9))
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec=color, lw=1.3))


def figure(name, rows, ctx, index, out, record) -> None:
    paths = [ln.strip() for ln in PV.read_text(encoding="utf-8").splitlines() if ln.strip()]
    corrupt = {c: make_corruption(c, SEV, DRAW, version=VERSION, modality="vis", images=paths)
               for c in CONDS if c != "clean"}
    fig, axes = plt.subplots(len(rows), 3, figsize=(DOUBLE, DOUBLE * len(rows) * CROP_H / (3 * CROP_W) + 0.55),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.05}, squeeze=False)
    for r, (frame, cond) in enumerate(rows):
        fname, x0, y0 = FRAMES[frame]
        i = index[fname]
        run = str(ctx.runs[i])
        sl = "night" if run in NIGHT_RUNS else "day"
        vrec, irec, res = ctx.vis_by_cond[cond][i], ctx.ir_clean[i], out[cond]
        gt = ships(ctx.gts[i])[0]
        im = cv2.imread(str(vrec["image_path"]))
        if cond != "clean":
            im = corrupt[cond](im, i)
        im = for_display(im, run)
        h = np.asarray(ctx.h_frames[i], np.float64)
        ir_im = cv2.warpPerspective(cv2.imread(str(irec["image_path"])), h, (640, 640), borderValue=(PAD, PAD, PAD))
        vetoed = bool(res["veto_vis"][i])
        b_vis, _ = ships(vrec, CONF)
        b_ir, _ = ships(res["ir_in_vis"][i], CONF)
        b_fu, _ = ships(res["fused_gated"][i], CONF)
        # origin of each fused box: the input box it reproduces (concatenation at IoU 0.85)
        all_v, all_i = ships(vrec)[0], ships(res["ir_in_vis"][i])[0]
        ov = iou(b_fu, all_v).max(axis=1) if len(all_v) else np.zeros(len(b_fu))
        oi = iou(b_fu, all_i).max(axis=1) if len(all_i) else np.zeros(len(b_fu))
        from_ir = oi > ov
        panels = ((im, b_vis, None), (ir_im, None, b_ir), (im, b_fu[~from_ir], b_fu[from_ir]))
        for c, (img, bv, bi) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(cv2.cvtColor(img[y0:y0 + CROP_H, x0:x0 + CROP_W], cv2.COLOR_BGR2RGB),
                      interpolation="lanczos")
            ax.set_xlim(-0.5, CROP_W - 0.5); ax.set_ylim(CROP_H - 0.5, -0.5)
            ax.set_xticks([]); ax.set_yticks([]); ax.grid(False)
            for s in ax.spines.values():
                s.set_visible(True); s.set_color(INK); s.set_linewidth(0.5)
            if c == 1:
                ax.add_patch(Polygon(ir_outline(h) - [x0, y0], closed=True, fill=False, ec=INK, lw=0.9,
                                     ls=(0, (1, 1.5))))
            draw_boxes(ax, gt, None, x0, y0, gt=True)
            if bv is not None:
                draw_boxes(ax, bv, VIS, x0, y0)
            if bi is not None:
                draw_boxes(ax, bi, IR, x0, y0)
            if c == 2:
                ax.text(0.985, 0.04, "VIS vetoed" if vetoed else "both streams kept", transform=ax.transAxes,
                        ha="right", va="bottom", fontsize=7, color=INK,
                        bbox={"boxstyle": "round,pad=0.25", "fc": "white", "ec": "none", "alpha": 0.85})
        label = f"{COND_LABEL[cond]} · {sl}" + (f"\n{run}" if name == "fig_detections_scenes" else "")
        axes[r, 0].set_ylabel(label, fontsize=8, color=INK, labelpad=4)
        record.append({"figure": name, "row": f"{frame}/{cond}", "frame_index": i, "run": run, "vis": fname,
                       "ir": Path(irec["image_path"]).name, "crop_xy": [x0, y0], "veto_vis": vetoed,
                       "gt_ships": int(len(gt)),
                       "shown": {"vis": int(len(b_vis)), "ir": int(len(b_ir)),
                                 "fused_from_vis": int((~from_ir).sum()), "fused_from_ir": int(from_ir.sum())},
                       "best_iou_to_gt": {k: [round(float(x), 2) for x in
                                              (iou(bb, gt).max(axis=1) if len(bb) and len(gt) else [])]
                                          for k, bb in (("vis", b_vis), ("ir", b_ir))}})
        print(f"[fig] {name} {frame}/{cond}: {fname} veto={vetoed} gt={len(gt)} vis={len(b_vis)} "
              f"ir={len(b_ir)} fused={len(b_fu)} (from IR {int(from_ir.sum())})")
    for c, t in enumerate(("VIS detector", "IR detector (warped to the VIS view)", "Fused, shipped rule")):
        axes[0, c].set_title(t, fontsize=8, color=INK, pad=3)
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.3, ls=(0, (2.2, 2.2)), label="ground truth (ship)"),
                        Line2D([], [], color=VIS, lw=1.6, label="VIS box"),
                        Line2D([], [], color=IR, lw=1.6, label="IR box"),
                        Line2D([], [], color=INK, lw=0.9, ls=(0, (1, 1.5)), label="edge of the IR field of view")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005), fontsize=7.5)
    fig.subplots_adjust(left=0.035 if name != "fig_detections_scenes" else 0.05, right=0.995,
                        top=1 - 0.13 / fig.get_figheight(), bottom=0.3 / fig.get_figheight())
    save(fig, name)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--candidates", type=Path, help="write the ground-truth-only candidate pool here and stop")
    ap.add_argument("--corrupt-version", choices=["v1", "v2"], default="v2")
    args = ap.parse_args()
    global VERSION
    VERSION = args.corrupt_version
    if args.candidates:
        candidates(args.candidates)
        return 0
    ctx, index = context()
    out = {c: run_systems(ctx, c) for c in CONDS}
    record = {"system": f"Phase 3 VIS seed {SEED} + IR seed {SEED}", "preset": PRESET,
              "corruption": {"draw": DRAW, "severity": SEV, "stream": "VIS only", "version": VERSION,
                             "cache_root": roots()[0]}, "conf_display": CONF,
              "night_display_gamma": NIGHT_GAMMA, "frames": {k: list(v) for k, v in FRAMES.items()}, "panels": []}
    for name, rows in FIGS.items():
        figure(name, rows, ctx, index, out, record["panels"])
    (ROOT / "docs/figures/fig_detections.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("wrote docs/figures/fig_detections.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
