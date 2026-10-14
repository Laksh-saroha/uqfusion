"""Figure: IG-BN on one held-out frame (pohang04) under fog, low light and sensor noise.

Two columns per row: the frozen visible detector (the VIS stream `union` fuses) and the same detector with
IG-BN. Every box is read from the records that were scored, nothing is re-run: the plain stream from the
corruption-v2 held-out caches (runs/holdout_p04/v2/caches) and the adapted stream from the IG-BN build
(runs/holdout_p04/ttbn, fp16, the precision the held-out scorings used). Seed 0 (VIS seed 0), draw 941,
thermal camera clean, as in the registered cells. The corrupted frame is regenerated for display with
`make_corruption` at the frame's index, the call the cache builder made.

How the frame was chosen, never on a detection: frames where the gate fires under all three conditions,
with at least three ground-truth ships, ranked by median ship height times their count (capped at six);
the top-ranked frame is shown. The crop is centred on the ground-truth ships. Low light is brightened for
display only (gamma 0.45).

Writes docs/figures/fig_tta_detections.{pdf,png} and fig_tta_detections.json.

    python scripts/fig_tta_detections.py
"""

from __future__ import annotations

import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import cv2                                                                  # noqa: E402
import matplotlib.pyplot as plt                                             # noqa: E402
from matplotlib.lines import Line2D                                         # noqa: E402
from matplotlib.patches import Rectangle                                    # noqa: E402

import p04_heldout as ph                                                    # noqa: E402
import ttbn                                                                 # noqa: E402
from paper_figures import DOUBLE, FUSED, INK, VIS, save                     # noqa: E402
from uqfusion.eval.corruptions import make_corruption                       # noqa: E402

SEED, DRAW, SHIP, CONF, IOU_HIT = 0, 941, 0, 0.25, 0.5
ROWS = (("fog", "Fog"), ("lowlight", "Low light"), ("noise_s2", "Sensor noise"))
DISPLAY_GAMMA = {"lowlight": 0.45}                              # display only; the detector saw the stored frame
CROP_H = 210
SUB = f"draw{DRAW}_{DRAW + 10}"


def gt_ships(path: str) -> np.ndarray:
    lab = Path(path.replace("\\images\\", "\\labels\\").replace("/images/", "/labels/")).with_suffix(".txt")
    rows = [ln.split() for ln in lab.read_text().splitlines() if ln.strip()] if lab.is_file() else []
    b = np.array([[float(v) for v in r[1:5]] for r in rows if int(r[0]) == SHIP]).reshape(-1, 4) * 640
    return np.c_[b[:, 0] - b[:, 2] / 2, b[:, 1] - b[:, 3] / 2, b[:, 0] + b[:, 2] / 2, b[:, 1] + b[:, 3] / 2]


def ships(rec) -> tuple[np.ndarray, np.ndarray]:
    m = (np.asarray(rec["cls"]).astype(int) == SHIP) & (np.asarray(rec["conf"], np.float64) >= CONF)
    return np.asarray(rec["boxes_xyxy"], np.float64).reshape(-1, 4)[m], np.asarray(rec["conf"], np.float64)[m]


def iou(a, b) -> np.ndarray:
    x1, y1 = np.maximum(a[:, None, 0], b[None, :, 0]), np.maximum(a[:, None, 1], b[None, :, 1])
    x2, y2 = np.minimum(a[:, None, 2], b[None, :, 2]), np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    ar = lambda z: (z[:, 2] - z[:, 0]) * (z[:, 3] - z[:, 1])               # noqa: E731
    return inter / (ar(a)[:, None] + ar(b)[None, :] - inter + 1e-9)


def found(boxes, conf, gt) -> int:
    """Ground-truth ships matched at IoU >= 0.5, greedily in confidence order."""
    if not len(boxes) or not len(gt):
        return 0
    m, used = iou(boxes[np.argsort(-conf)], gt), set()
    for row in m:
        j = next((j for j in np.argsort(-row) if row[j] >= IOU_HIT and j not in used), None)
        if j is not None:
            used.add(j)
    return len(used)


def load(path: Path) -> list:
    with open(path, "rb") as fh:
        return pickle.load(fh)["records"]


def main() -> int:
    lst = ph.images("vis")
    plain = {c: load(ph.HP / "v2" / "caches" / f"seed{SEED}" / SUB / f"gauss_vis_paired_{c}.pkl") for c, _ in ROWS}
    adapt = {c: load(ttbn.out_path("p04", SEED, c, DRAW, fp16=True)) for c, _ in ROWS}
    gate = {c: ttbn.gate_p04(c, None, SUB) for c, _ in ROWS}
    ok = np.logical_and.reduce([gate[c] for c, _ in ROWS])
    ok &= np.array([all(adapt[c][i] is not None for c, _ in ROWS) for i in range(len(lst))])

    scored = []                                                 # ground truth only, never a detection
    for i in np.flatnonzero(ok):
        g = gt_ships(lst[i])
        if len(g) >= 3:
            scored.append((float(np.median(g[:, 3] - g[:, 1])) * min(len(g), 6), int(i)))
    if not scored:
        raise SystemExit("no candidate frame")
    i = max(scored)[1]
    gt = gt_ships(lst[i])
    cy = float(np.clip((gt[:, 1].min() + gt[:, 3].max()) / 2, CROP_H / 2, 640 - CROP_H / 2))
    y0 = int(round(cy - CROP_H / 2))

    gap, leg = 0.04, 0.22                                       # inches between panels; legend strip
    pw = (DOUBLE - gap) / 2
    ph_ = pw * CROP_H / 640
    H = len(ROWS) * ph_ + (len(ROWS) - 1) * gap + leg
    fig, axes = plt.subplots(len(ROWS), 2, figsize=(DOUBLE, H))
    fig.subplots_adjust(left=0, right=1, top=1, bottom=leg / H, wspace=gap / pw, hspace=gap / ph_)
    record = {"frame": Path(lst[i]).name, "index": int(i), "seed": SEED, "draw": DRAW, "crop_y0": y0,
              "crop_h": CROP_H, "display_gamma": DISPLAY_GAMMA, "candidates": len(scored), "gt_ships": int(len(gt)), "conf": CONF, "panels": {}}
    for r, (c, label) in enumerate(ROWS):
        kind, sev = ph.VIS_CONDS[c]
        tf = make_corruption(kind, sev, DRAW, version="v2", modality="vis", images=lst)
        im = cv2.cvtColor(tf(cv2.imread(lst[i]), i), cv2.COLOR_BGR2RGB)[y0:y0 + CROP_H]
        if c in DISPLAY_GAMMA:
            im = (255 * (im / 255.0) ** DISPLAY_GAMMA[c]).astype(np.uint8)
        assert Path(plain[c][i]["image_path"]).name == Path(adapt[c][i]["image_path"]).name == Path(lst[i]).name
        for col, (name, rec, color) in enumerate((("No adaptation", plain[c][i], VIS),
                                                   ("IG-BN", adapt[c][i], FUSED))):
            ax = axes[r, col]
            ax.imshow(im, extent=(0, 640, y0 + CROP_H, y0))
            for x1, y1, x2, y2 in gt:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec="white", lw=0.9, ls=(0, (2, 1.5))))
            b, s = ships(rec)
            for x1, y1, x2, y2 in b:
                ax.add_patch(Rectangle((x1, y1), x2 - x1, y2 - y1, fill=False, ec=color, lw=1.2))
            n = found(b, s, gt)
            ax.text(6, y0 + 6, f"{label} · {name}: {n} of {len(gt)} ships found, {len(b)} boxes", fontsize=7,
                    va="top", ha="left", color="white", bbox={"fc": "black", "alpha": 0.6, "lw": 0, "pad": 1.6})
            ax.set_xlim(0, 640)
            ax.set_ylim(y0 + CROP_H, y0)
            ax.set_xticks([])
            ax.set_yticks([])
            ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
            record["panels"][f"{c}/{'plain' if col == 0 else 'igbn'}"] = {"boxes": int(len(b)), "found": n}
    fig.legend(handles=[Line2D([], [], color="white", mec=INK, lw=0, marker="s", mfc="none", label="ground truth (dashed)"),
                        Line2D([], [], color=VIS, lw=1.5, label="frozen detector"),
                        Line2D([], [], color=FUSED, lw=1.5, label="IG-BN")],
               loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0), borderaxespad=0.1)
    save(fig, "fig_tta_detections")
    (ROOT / "docs/figures/fig_tta_detections.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
