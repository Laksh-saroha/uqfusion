"""Figure: the three answers, and adapting every frame, across every test (Δ ship AP against `union`, 95% CI).

Rows are registered cells, read from the result records (nothing is re-scored):
* development (draws 942–944, five systems, t-intervals): `gate_ablation_dev_2026-10-14.json` (ungated, IG-BN,
  clock gate = `q_vis` ∧ sun) and `gatefree_dev_2026-10-16.json` (shallow BN-mix, GF31);
* held-out daylight (pohang04, moving-block intervals): `gate_ablation_p04_2026-10-14.json` and
  `gatefree_p04_2026-10-16.json`;
* night test (pohang01 test split): `gatefree_night_p01_2026-10-16.json`; IG-BN and the clock gate adapt no frame
  there, so they equal `union` exactly (drawn at zero, no interval).
Values beyond the axis are drawn at its edge with their value written.

Writes docs/figures/fig_three_answers.{pdf,png} and fig_three_answers.json.

    python scripts/fig_three_answers.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "scripts")]

import matplotlib.pyplot as plt                                              # noqa: E402
from matplotlib.lines import Line2D                                          # noqa: E402

from paper_figures import BAND, DOUBLE, INK, INK2, MUTED, RULE, read_json, save, signed  # noqa: E402

E = "docs/eval/ttbn/"
ARMS = (("ungated", "Ungated BN-mix", "#2a78d6", "o"), ("n4", "IG-BN (thermal gate)", "#eb6834", "s"),
        ("clock", "Clock gate", "#1baf7a", "D"), ("gf31", "Shallow BN-mix", "#eda100", "^"))
XMIN, XMAX = -0.035, 0.09
GROUPS = (
    ("Development", (("Clean, day", "clean/clean/day"), ("Clean, night", "clean/clean/night"),
                     ("Low light, night", "lowlight/clean/night"), ("Fog, day", "fog/clean/day"),
                     ("Low light, day", "lowlight/clean/day"),
                     ("Mild fog, day; thermal fog", "fog_s1/fog_s2/day"),
                     ("Low light, day; thermal noise", "lowlight/noise_s3/day"))),
    ("Held-out daylight", (("Clean", "clean__clean"), ("Fog", "fog__clean"), ("Low light", "lowlight__clean"),
                           ("Sensor noise", "noise_s2__clean"))),
    ("Night test", (("Clean, night", "night"),)),
)


def values() -> dict:
    gd, gp = read_json(E + "gate_ablation_dev_2026-10-14.json")["cells"], read_json(E + "gate_ablation_p04_2026-10-14.json")["cells"]
    fd = read_json(E + "gatefree_dev_2026-10-16.json")["cells"]["sh31_hinge50"]
    fp = read_json(E + "gatefree_p04_2026-10-16.json")["cells"]
    nt = read_json(E + "gatefree_night_p01_2026-10-16.json")
    out = {}
    for group, rows in GROUPS:
        for label, key in rows:
            if group == "Development":
                c = gd[key]
                v = {"ungated": c["ungated"]["vs_union"], "n4": c["n4"]["vs_union"], "clock": c["qvis_solar"]["vs_union"],
                     "gf31": fd[key]["d"]}
            elif group == "Held-out daylight":
                c = gp[key]
                v = {"ungated": c["ungated"]["vs_union"], "n4": c["n4"]["vs_union"], "clock": c["qvis_solar"]["vs_union"],
                     "gf31": fp[key]["vs_union"]}
            else:
                v = {"ungated": nt["mix16"]["vs_union"], "n4": [0.0, 0.0, 0.0], "clock": [0.0, 0.0, 0.0],
                     "gf31": nt["gf31"]["vs_union"]}
            out[f"{group}|{label}"] = {a: [float(x) for x in v[a]] for a in v}
    return out


def main() -> int:
    vals = values()
    n_rows = sum(len(r) for _, r in GROUPS)
    row_h, head = 0.2, 0.23
    H = n_rows * row_h + len(GROUPS) * head + 0.55
    fig, ax = plt.subplots(figsize=(DOUBLE, H))
    y, ticks, labels = 0.0, [], []
    off = {a: -(j - 1.5) * 0.17 for j, (a, *_) in enumerate(ARMS)}
    for gi, (group, rows) in enumerate(GROUPS):
        y -= head / row_h
        ax.text(XMIN - 0.0005, y + 0.1, group, fontsize=7.8, fontweight="bold", color=INK, ha="right", va="center",
                transform=ax.transData, clip_on=False)
        top = y
        for label, _ in rows:
            y -= 1
            ticks.append(y)
            labels.append(label)
            v = vals[f"{group}|{label}"]
            over = [v[a][0] for a, *_ in ARMS if v[a][0] > XMAX]
            if over:
                ax.text(XMAX - 0.004, y, f"all four {signed(min(over), 3)} to {signed(max(over), 3)} →", fontsize=6.8,
                        color=INK2, ha="right", va="center")
            for a, _, col, mk in ARMS:
                d, lo, hi = v[a]
                yy = y + off[a]
                same = group == "Night test" and a in ("n4", "clock")
                if not same and lo <= XMAX:
                    ax.plot([max(lo, XMIN), min(hi, XMAX)], [yy, yy], color=col, lw=1.4, solid_capstyle="round", zorder=2)
                if d > XMAX:
                    continue
                xd = max(d, XMIN)
                ax.scatter([xd], [yy], marker=mk, s=26, color="white" if same else col, edgecolor=col if same else "white",
                           linewidth=1.0 if same else 0.6, zorder=3)
        if gi % 2 == 0:
            ax.axhspan(y - 0.5, top + 0.5 - 1 + 1, color=BAND, zorder=0, lw=0)
    ax.axvline(0, color=MUTED, lw=0.9, zorder=1)
    ax.set_yticks(ticks)
    ax.set_yticklabels(labels)
    ax.set_ylim(y - 0.6, 0.2)
    ax.set_xlim(XMIN, XMAX)
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: "0" if abs(v) < 1e-9 else signed(v, 2)))
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("Δ ship AP against union (95% CI)")
    ax.tick_params(axis="y", length=0, pad=4)
    ax.spines["left"].set_visible(False)
    handles = [Line2D([], [], color=c, marker=m, lw=1.4, markersize=5, markeredgecolor="white") for _, _, c, m in ARMS]
    ax.legend(handles, [n for _, n, _, _ in ARMS], loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=4,
              handlelength=1.6, columnspacing=1.4)
    for s in ("bottom",):
        ax.spines[s].set_color(RULE)
    save(fig, "fig_three_answers")
    (ROOT / "docs/figures/fig_three_answers.json").write_text(json.dumps(vals, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
