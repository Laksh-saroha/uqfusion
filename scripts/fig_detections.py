"""Figure 8 of PAPER_DRAFT2.md: the shipped rule on one day frame and one night frame.

Rows: clean/day, fog/day, clean/night, fog/night. Columns: the VIS detector, the IR detector
(its frame warped into the VIS canvas by the per-frame homography the fusion uses, so all
three panels share one geometry), and the fused output of `crossmodal26m`. Every box comes
from the development caches and `run_systems`, the path that scores Table 3b; nothing is
re-run. **Reads no pohang04 frame.**

System: Phase 3 VIS seed 0 + IR seed 0; fog is VIS draw 941 (severity 2), IR stays clean,
as in every Table 3b cell. The fogged frame is regenerated with `make_corruption` at the
frame's index in the paired list, the same call the cache builder made.

Frames are chosen on ground truth and geometry alone, never on a detection. A ship is
co-visible when its box centre lies inside the IR camera's field of view warped into the VIS
canvas. Among paired development frames with at least MIN_SHIPS co-visible ships of median
height at least MIN_MED_H px, one day and one night frame are drawn uniformly at random with
`default_rng(PICK_SEED)`. Each panel is a CROP_W x CROP_H crop of the 640 x 640 canvas
centred on the mean co-visible ship centre. Ship class only,
the class every table reports. Boxes shown at confidence >= CONF; fused boxes are coloured by
the stream they came from (at IoU 0.85 the merge is concatenation, so each fused box is one
input box).

Writes docs/figures/fig_detections.{pdf,png} and fig_detections.json (frames, crop, flags, and
each shown box's best IoU with the ship ground truth).

Usage (from the repo root):
    python scripts/fig_detections.py
"""

from __future__ import annotations

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
SEED, DRAW, FOG_SEV = 0, 941, 2
SUB = f"draw{DRAW}_{DRAW + 10}"
PV = ROOT / "runs/derived/paired_val_vis.txt"
SHIP = 0
CONF = 0.25
PICK_SEED, MIN_SHIPS, MIN_MED_H = 0, 3, 15.0
CROP_W, CROP_H = 320, 170
VIS_ROWS = (151, 489)          # letterboxed VIS content rows (§3.3)
IR_ROWS = (64, 576)            # letterboxed IR content rows
PAD = 114
ROWS = (("clean", "day"), ("fog", "day"), ("clean", "night"), ("fog", "night"))


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


def covisible(gt, h) -> np.ndarray:
    """Ground-truth ship boxes whose centre the IR camera also sees."""
    if not len(gt):
        return gt
    cx, cy = ((gt[:, 0] + gt[:, 2]) / 2).astype(int), ((gt[:, 1] + gt[:, 3]) / 2).astype(int)
    return gt[ir_footprint(h)[np.clip(cy, 0, 639), np.clip(cx, 0, 639)]]


def ir_outline(h) -> np.ndarray:
    """Corners of the IR content rows in the VIS canvas."""
    c = np.array([[[0, IR_ROWS[0]], [640, IR_ROWS[0]], [640, IR_ROWS[1]], [0, IR_ROWS[1]]]], np.float64)
    return cv2.perspectiveTransform(c, np.asarray(h, np.float64))[0]


def pick(ctx) -> dict[str, int]:
    ok = np.zeros(len(ctx.gts), bool)
    for i, g in enumerate(ctx.gts):
        b = covisible(ships(g)[0], ctx.h_frames[i])
        ok[i] = len(b) >= MIN_SHIPS and np.median(b[:, 3] - b[:, 1]) >= MIN_MED_H
    night = np.isin(ctx.runs, NIGHT_RUNS)
    rng = np.random.default_rng(PICK_SEED)
    out = {}
    for s, m in (("day", ~night), ("night", night)):
        cand = np.flatnonzero(ok & m)
        print(f"[fig8] {s}: {cand.size} eligible frames")
        out[s] = int(rng.choice(cand))
    return out


def crop_box(gt) -> tuple[int, int]:
    c = np.array([(gt[:, 0] + gt[:, 2]).mean() / 2, (gt[:, 1] + gt[:, 3]).mean() / 2])
    x0 = int(np.clip(round(c[0] - CROP_W / 2), 0, 640 - CROP_W))
    y0 = int(np.clip(round(c[1] - CROP_H / 2), VIS_ROWS[0], VIS_ROWS[1] - CROP_H))
    return x0, y0


def draw_boxes(ax, boxes, color, x0, y0, gt=False):
    for x1, y1, x2, y2 in boxes:
        xy, w, h = (x1 - x0, y1 - y0), x2 - x1, y2 - y1
        if gt:
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec="white", lw=1.3))
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec=INK, lw=1.3, ls=(0, (2.2, 2.2))))
        else:
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec="white", lw=2.2, alpha=0.9))
            ax.add_patch(Rectangle(xy, w, h, fill=False, ec=color, lw=1.3))


def main() -> int:
    ctx = load_context(preset=PRESET, cache_dir=f"runs/cache_p3dev/seed{SEED}/{SUB}",
                       bright_dir=f"runs/derived_p3dev/brightness/{SUB}",
                       structure_dir=f"runs/derived_p3dev/structure/{SUB}",
                       conditions=("clean", "fog"), verbose=False)
    if any("pohang04" in r for r in ctx.runs):
        raise SystemExit("a pohang04 frame reached the development context")
    pv = [ln.strip() for ln in PV.read_text(encoding="utf-8").splitlines() if ln.strip()]
    for c in ("clean", "fog"):
        got = [str(r["image_path"]) for r in ctx.vis_by_cond[c]]
        if [Path(p).name for p in got] != [Path(p).name for p in pv]:
            raise SystemExit(f"{c}: cache frame order differs from {PV.name}; the fog index would be wrong")
    out = {c: run_systems(ctx, c) for c in ("clean", "fog")}
    frames = pick(ctx)
    fog = make_corruption("fog", FOG_SEV, DRAW)

    fig, axes = plt.subplots(4, 3, figsize=(DOUBLE, DOUBLE * 4 * CROP_H / (3 * CROP_W) + 0.55),
                             gridspec_kw={"wspace": 0.03, "hspace": 0.05})
    record = {"system": f"Phase 3 VIS seed {SEED} + IR seed {SEED}", "preset": PRESET,
              "fog": {"draw": DRAW, "severity": FOG_SEV, "stream": "VIS only"}, "conf_display": CONF,
              "rule": {"pick_seed": PICK_SEED, "min_ships": MIN_SHIPS, "min_median_ship_height_px": MIN_MED_H,
                       "crop": [CROP_W, CROP_H]}, "panels": []}
    for r, (cond, sl) in enumerate(ROWS):
        i = frames[sl]
        vrec, irec = ctx.vis_by_cond[cond][i], ctx.ir_clean[i]
        gt = ships(ctx.gts[i])[0]
        x0, y0 = crop_box(covisible(gt, ctx.h_frames[i]))
        im = cv2.imread(str(vrec["image_path"]))
        if cond == "fog":
            im = fog(im, i)
        h = np.asarray(ctx.h_frames[i], np.float64)
        ir_im = cv2.warpPerspective(cv2.imread(str(irec["image_path"])), h, (640, 640),
                                    borderValue=(PAD, PAD, PAD))
        res = out[cond]
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
        axes[r, 0].set_ylabel(f"{cond} · {sl}", fontsize=8, color=INK, labelpad=4)
        record["panels"].append({"row": f"{cond}/{sl}", "frame_index": i, "run": str(ctx.runs[i]),
                                 "vis": Path(vrec["image_path"]).name, "ir": Path(irec["image_path"]).name,
                                 "crop_xy": [x0, y0], "veto_vis": vetoed, "gt_ships": int(len(gt)),
                                 "shown": {"vis": int(len(b_vis)), "ir": int(len(b_ir)),
                                           "fused_from_vis": int((~from_ir).sum()),
                                           "fused_from_ir": int(from_ir.sum())},
                                 "best_iou_to_gt": {k: [round(float(x), 2) for x in
                                                        (iou(bb, gt).max(axis=1) if len(bb) and len(gt) else [])]
                                                    for k, bb in (("vis", b_vis), ("ir", b_ir))}})
        print(f"[fig8] {cond}/{sl}: frame {i} {Path(vrec['image_path']).name} veto={vetoed} "
              f"gt={len(gt)} vis={len(b_vis)} ir={len(b_ir)} fused={len(b_fu)} (from IR {int(from_ir.sum())})")
    for c, t in enumerate(("VIS detector", "IR detector (warped to the VIS view)", "Fused, shipped rule")):
        axes[0, c].set_title(t, fontsize=8, color=INK, pad=3)
    fig.legend(handles=[Line2D([], [], color=INK, lw=1.3, ls=(0, (2.2, 2.2)), label="ground truth (ship)"),
                        Line2D([], [], color=VIS, lw=1.6, label="VIS box"),
                        Line2D([], [], color=IR, lw=1.6, label="IR box"),
                        Line2D([], [], color=INK, lw=0.9, ls=(0, (1, 1.5)), label="edge of the IR field of view")],
               loc="lower center", ncol=4, bbox_to_anchor=(0.5, -0.005), fontsize=7.5)
    fig.subplots_adjust(left=0.035, right=0.995, top=0.965, bottom=0.05)
    save(fig, "fig_detections")
    (ROOT / "docs/figures/fig_detections.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print("wrote docs/figures/fig_detections.json")
    return 0


if __name__ == "__main__":
    sys.exit(main())
