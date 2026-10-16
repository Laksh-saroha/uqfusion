"""Figure: the question every gate answers. A dark visible frame may be a failing camera or a dark scene.

Three development frames (paired validation set): one pohang00 day frame, clean; the same frame under the
registered low-light corruption (v2, severity 2, draw 942, the call the statistics were built with); one
pohang01 night frame, clean. Each column shows the visible frame, its thermal partner, and the three readings
the gates use, read from the stored frame statistics of draw 942 (nothing is re-measured):
* the visible health novelty d² against its bound 124.3 (`q_vis` < 1 iff d² > 124.3);
* the thermal 5th-percentile intensity `ir_p05` against the day threshold 41.5, and the thermal health d²;
* the solar elevation at capture time (`gate_ablation.solar`).

How the frames were chosen, never on a detection or a statistic: in each run, the paired development frame
with the most ground-truth ships (earliest on a tie). Visible frames are shown without their letterbox pad and thermal
frames as a centre band, for display only. Writes docs/figures/fig_ambiguity.{pdf,png} and fig_ambiguity.json.

    python scripts/fig_ambiguity.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "src"), str(ROOT / "scripts")]

import cv2                                                                  # noqa: E402
import matplotlib.pyplot as plt                                             # noqa: E402

import ttbn                                                                 # noqa: E402
from fig_tta_detections import gt_ships                                     # noqa: E402
from gate_ablation import solar                                             # noqa: E402
from paper_figures import DOUBLE, INK, INK2, save                           # noqa: E402
from uqfusion.eval.corruptions import make_corruption                       # noqa: E402

DRAW = 942
IR_ASPECT = 2.6        # thermal frames shown as a centre band, for display only
IR_LIST = ROOT / "runs/derived/paired_val_ir.txt"


def pick(lst: list[str], run: str) -> int:
    best = max((len(gt_ships(p)), -i) for i, p in enumerate(lst) if f"{run}_" in Path(p).name)
    return -best[1]


def main() -> int:
    lst = ttbn.ttbn_read(ttbn.DEV_LIST) if hasattr(ttbn, "ttbn_read") else \
        [ln.strip() for ln in open(ttbn.DEV_LIST) if ln.strip()]
    irl = [ln.strip() for ln in open(IR_LIST) if ln.strip()]
    sc = ttbn._sconst()["axes"]
    vh, ih = sc["vis_health"], sc["ir_health"]
    thr, vb, ib = float(sc["ir_p05"]["threshold"]), float(vh["bound"]), float(ih["bound"])
    d2v = {c: ttbn._d2(ttbn._dev_stats(DRAW, "vis", c, "structure"), vh) for c in ("clean", "lowlight")}
    p05 = np.asarray([f["p05"] for f in json.loads(ttbn._dev_stats(DRAW, "ir", None, "brightness")
                                                   .read_text(encoding="utf-8"))["frames"]], float)
    d2i = ttbn._d2(ttbn._dev_stats(DRAW, "ir", None, "structure"), ih)
    sun = np.asarray(solar()["dev"], float)
    assert len(d2v["clean"]) == len(lst) == len(irl) == len(p05) == len(sun)

    day, night = pick(lst, "pohang00"), pick(lst, "pohang01")
    cols = (("Clean day", day, "clean"), ("Low light by day: a failing camera", day, "lowlight"),
            ("Clean night: a dark scene", night, "clean"))
    tf = make_corruption("lowlight", 2, DRAW, version="v2", modality="vis", images=lst)

    gap, text_h = 0.04, 0.56
    pw = (DOUBLE - 2 * gap) / 3
    vh_in = pw * 1080 / 2048
    ih_in = pw / IR_ASPECT
    H = vh_in + ih_in + gap + text_h + 0.18
    fig = plt.figure(figsize=(DOUBLE, H))
    record = {"draw": DRAW, "frames": {}}
    for c, (title, i, cond) in enumerate(cols):
        x0 = c * (pw + gap) / DOUBLE
        im = cv2.imread(lst[i])
        if cond == "lowlight":
            im = tf(im, i)
        rows = np.flatnonzero(np.any(np.abs(im.astype(int) - 114) > 2, axis=(1, 2)))   # drop the letterbox pad
        im = im[rows.min():rows.max() + 1]
        ax = fig.add_axes((x0, (text_h + ih_in + gap) / H, pw / DOUBLE, vh_in / H))
        ax.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), aspect="auto")
        ax.set_axis_off()
        ax.set_title(title, fontsize=7.5, color=INK, pad=3)
        ir = cv2.imread(irl[i], cv2.IMREAD_GRAYSCALE)
        h = min(ir.shape[0], int(round(ir.shape[1] / IR_ASPECT)))
        y0 = (ir.shape[0] - h) // 2
        ax2 = fig.add_axes((x0, text_h / H, pw / DOUBLE, ih_in / H))
        ax2.imshow(ir[y0:y0 + h], cmap="gray", vmin=0, vmax=255, aspect="auto")
        ax2.set_axis_off()
        flagged = d2v[cond][i] > vb
        reads_day = p05[i] <= thr
        healthy = d2i[i] <= ib
        adapt = flagged and reads_day and healthy
        lines = [f"visible health d² = {d2v[cond][i]:,.0f}: {'flagged' if flagged else 'healthy'}",
                 f"thermal p05 = {p05[i]:.0f}: reads {'day' if reads_day else 'night'}"
                 f"{'' if healthy else ' (thermal unhealthy)'}",
                 f"sun {sun[i]:+.0f}°".replace("-", "−")]
        ty = (text_h - 0.06) / H
        fig.text(x0, ty, "\n".join(lines), fontsize=6.8, color=INK2, va="top", ha="left", linespacing=1.35)
        fig.text(x0, ty - len(lines) * 1.35 * 6.8 / 72 / H, f"IG-BN and clock gate: {'adapt' if adapt else 'do not adapt'}",
                 fontsize=6.8, color=INK, fontweight="bold", va="top", ha="left")
        if c == 0:                                   # name the rows once, inside the first column
            for a, name in ((ax, "Visible"), (ax2, "Thermal")):
                a.text(0.012, 0.95, name, transform=a.transAxes, fontsize=6.6, color="white", va="top", ha="left",
                       bbox=dict(boxstyle="round,pad=0.25", fc="black", ec="none", alpha=0.55))
        record["frames"][title] = {"vis": Path(lst[i]).name, "ir": Path(irl[i]).name, "cond": cond,
                                   "vis_d2": float(d2v[cond][i]), "ir_p05": float(p05[i]), "ir_d2": float(d2i[i]),
                                   "solar_deg": float(sun[i]), "gt_ships": int(len(gt_ships(lst[i])))}
    record["bounds"] = {"vis_d2": vb, "ir_p05_day_max": thr, "ir_d2": ib}
    save(fig, "fig_ambiguity")
    (ROOT / "docs/figures/fig_ambiguity.json").write_text(json.dumps(record, indent=1), encoding="utf-8")
    print(json.dumps(record, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
