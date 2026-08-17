"""Phase 1 figures: accuracy and batch-1 latency, by model and by family.

Four PNGs into phase1_benchmark/figures/:

    fig1_accuracy_ranking.png   31 variants ranked by mAP50-95
    fig2_accuracy_by_family.png the same numbers grouped into family ladders
    fig3a_fps_by_model.png      31 variants ranked by fp32 FPS
    fig3b_fps_by_family.png     the same, grouped into family ladders

Why the ranked charts are single-hue and only the grouped ones carry family
colour: colour has to survive colour-vision deficiency, and that is a computable
property, not a taste call. Six categorical hues are separable only while
same-coloured marks stay contiguous. Rank the bars by value and the families
interleave, so every family pair can end up side by side -- and no six hues clear
the thresholds under that condition. Measured with the dataviz validator
(OKLab dE x100, CVD >= 8, normal-vision >= 15):

    reference palette, 6 slots, adjacent pairs   PASS (worst CVD 9.1, normal 19.6)
    reference palette, 6 slots, all pairs        FAIL (green<->orange CVD 3.2 protan;
                                                       magenta<->orange normal 12.9)
    all 28 six-subsets of the reference 8        0 pass all-pairs in both modes
    all 7 six-subsets of Okabe-Ito               0 pass all-pairs in both modes

A protanopic reader genuinely cannot separate YOLOv9 from YOLOv10 in a
family-coloured ranked chart. So the ranked figures use one hue plus emphasis --
the family is already written on every axis label -- and the grouped figures keep
family colour, where it is legitimate because each family is one contiguous block
and only the five adjacent pairs ever occur.

    python scripts/make_phase1_figures.py
"""

from __future__ import annotations

import csv
import statistics as st
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

REPO = Path(__file__).resolve().parents[1]
RECORD = REPO / "phase1_benchmark" / "results.csv"
FPS_CSV = REPO / "phase1_benchmark" / "fps.csv"
OUT = REPO / "phase1_benchmark" / "figures"

# Reference categorical palette, slots 1-6, in documented order. Only ever used
# where family blocks are contiguous (see module docstring).
FAMILY_COLOR = {
    "yolov8": "#2a78d6", "yolov9": "#eb6834", "yolov10": "#1baf7a",
    "yolo11": "#eda100", "yolo12": "#e87ba4", "yolo26": "#008300",
}
FAMILY_LABEL = {
    "yolov8": "YOLOv8", "yolov9": "YOLOv9", "yolov10": "YOLOv10",
    "yolo11": "YOLO11", "yolo12": "YOLO12", "yolo26": "YOLO26",
}
FAMILY_ORDER = ["yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26"]
SCALE_ORDER = ["n", "t", "s", "m", "b", "c", "l", "e", "x"]

ACCENT = "#2a78d6"      # the one hue the ranked charts use
EMPHASIS = "#008300"    # YOLO26 -- the family this project contributed
INK = "#0b0b0b"
INK_2 = "#52514e"
GRID = "#d8d7d2"
SURFACE = "#fcfcfb"


def family_of(variant: str) -> str:
    for fam in sorted(FAMILY_COLOR, key=len, reverse=True):
        if variant.startswith(fam):
            return fam
    raise ValueError(f"no family for {variant}")


def scale_of(variant: str) -> str:
    return variant[len(family_of(variant)):]


def load_accuracy() -> dict[str, dict]:
    rows = [r for r in csv.DictReader(open(RECORD, encoding="utf-8"))
            if r["admissible"] == "yes"]
    by: dict[str, list[dict]] = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(r)
    out = {}
    for v, rs in by.items():
        m = [float(r["map50_95"]) for r in rs]
        out[v] = {"mean": st.mean(m), "sd": st.stdev(m) if len(m) > 1 else 0.0,
                  "n": len(m), "gflops": float(rs[0]["gflops"]),
                  "params": float(rs[0]["params_m"]), "grid": rs[0]["grid"]}
    return out


def load_fps() -> dict[str, dict]:
    if not FPS_CSV.is_file():
        return {}
    rows = [r for r in csv.DictReader(open(FPS_CSV, encoding="utf-8"))
            if r["half"] == "False"]
    by: dict[str, list[float]] = {}
    for r in rows:
        by.setdefault(r["variant"], []).append(float(r["fps"]))
    return {v: {"mean": st.mean(f), "sd": st.stdev(f) if len(f) > 1 else 0.0, "n": len(f)}
            for v, f in by.items()}


def style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, length=0)
    ax.xaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_axisbelow(True)


def ranked_bars(data: dict[str, dict], value_fmt: str, xlabel: str, title: str,
                subtitle: str, out: Path) -> None:
    """Ranked horizontal bars, one hue plus emphasis.

    Colour encodes nothing here -- the ranking is the message and the family is
    already on the axis label, so a second colour channel would add CVD risk for
    no information.
    """
    items = sorted(data.items(), key=lambda kv: kv[1]["mean"])
    names = [k for k, _ in items]
    means = [v["mean"] for _, v in items]
    sds = [v["sd"] for _, v in items]
    colors = [EMPHASIS if family_of(n) == "yolo26" else ACCENT for n in names]

    fig, ax = plt.subplots(figsize=(9.5, 11), dpi=200)
    fig.patch.set_facecolor(SURFACE)
    y = range(len(names))
    ax.barh(list(y), means, height=0.74, color=colors,
            xerr=sds, error_kw=dict(ecolor=INK_2, elinewidth=1.1, capsize=2.5, alpha=0.85))
    ax.set_yticks(list(y))
    ax.set_yticklabels(names, fontsize=9.5, color=INK)
    ax.set_ylim(-0.8, len(names) - 0.2)

    span = max(means)
    for i, (m, s) in enumerate(zip(means, sds)):
        ax.text(m + s + span * 0.012, i, format(m, value_fmt),
                va="center", ha="left", fontsize=9, color=INK_2)
    ax.set_xlim(0, span * 1.16)
    style(ax)
    ax.set_xlabel(xlabel, fontsize=10.5, color=INK_2, labelpad=9)
    ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=22, loc="left")
    ax.text(0, 1.012, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2)
    ax.legend(handles=[Patch(facecolor=EMPHASIS, label="YOLO26"),
                       Patch(facecolor=ACCENT, label="other families")],
              loc="lower right", frameon=False, fontsize=9.5, labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def family_ladder(data: dict[str, dict], value_fmt: str, ylabel: str, title: str,
                  subtitle: str, out: Path) -> None:
    """Each family's capacity ladder, side by side, in family colour.

    Dots rather than bars, and the y-axis does not start at zero. That is the
    whole reason this form is used: mAP spans 0.249-0.305, a 1.23x range, so bars
    from a zero baseline put every family in the top fifth of the canvas and the
    differences this figure exists to show vanish. A bar encodes magnitude by
    length and so must start at zero; a dot encodes position and may be zoomed.

    Family colour is legitimate here because each family occupies its own
    contiguous x-block. Only the five adjacent family pairs ever sit together,
    and the palette passes on that pairlist -- unlike the ranked figures, where
    sorting interleaves the families and no six hues clear the thresholds.
    """
    groups = []
    for fam in FAMILY_ORDER:
        members = [v for v in data if family_of(v) == fam]
        members.sort(key=lambda v: SCALE_ORDER.index(scale_of(v)))
        if members:
            groups.append((fam, members))

    lo = min(v["mean"] - v["sd"] for v in data.values())
    hi = max(v["mean"] + v["sd"] for v in data.values())
    pad = (hi - lo) * 0.16
    fig, ax = plt.subplots(figsize=(13.5, 6.6), dpi=200)
    fig.patch.set_facecolor(SURFACE)

    x, ticks, labels, gap = 0.0, [], [], 1.4
    for fam, members in groups:
        xs = [x + i for i in range(len(members))]
        ys = [data[v]["mean"] for v in members]
        es = [data[v]["sd"] for v in members]
        c = FAMILY_COLOR[fam]
        ax.plot(xs, ys, color=c, linewidth=2, zorder=2, solid_capstyle="round")
        ax.errorbar(xs, ys, yerr=es, fmt="none", ecolor=c, elinewidth=1.4,
                    capsize=3, alpha=0.9, zorder=3)
        # a surface ring keeps overlapping marks readable
        ax.scatter(xs, ys, s=95, color=c, edgecolors=SURFACE, linewidths=1.8, zorder=4)
        # anchor the label above the error-bar cap, not the marker, or it lands on top
        # of its own whisker wherever the spread is wide
        for xi, yi, ei in zip(xs, ys, es):
            ax.annotate(format(yi, value_fmt), (xi, yi + ei), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=7.6, color=INK_2)
        ticks += xs
        labels += [scale_of(v) for v in members]
        ax.text(sum(xs) / len(xs), lo - pad * 0.72, FAMILY_LABEL[fam], ha="center",
                va="top", fontsize=11.5, fontweight="bold", color=c)
        x += len(members) + gap

    ax.set_xticks(ticks)
    ax.set_xticklabels(labels, fontsize=9, color=INK_2)
    ax.set_xlim(-1.0, x - gap)
    ax.set_ylim(lo - pad, hi + pad)
    style(ax)
    ax.xaxis.grid(False)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8, alpha=0.7)
    ax.set_ylabel(ylabel, fontsize=10.5, color=INK_2, labelpad=9)
    ax.set_title(title, fontsize=14, fontweight="bold", color=INK, pad=26, loc="left")
    ax.text(0, 1.028, subtitle, transform=ax.transAxes, fontsize=9.5, color=INK_2)
    ax.text(1, 1.028, "y-axis does not start at zero", transform=ax.transAxes,
            fontsize=8.5, color=INK_2, ha="right", style="italic")
    fig.tight_layout()
    fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.relative_to(REPO)}")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    acc = load_accuracy()
    fps = load_fps()
    n_note = "seed mean ± sd, n=3 (yolo12x n=2)"

    print("[figures]")
    ranked_bars(acc, ".3f", "mAP@50-95   ({})".format(n_note),
                "Phase 1 — YOLO backbone benchmark  (Pohang VIS, ship only)",
                f"{len(acc)} variants · ultralytics 8.4.90 · imgsz 640 · patience 20",
                OUT / "fig1_accuracy_ranking.png")
    family_ladder(acc, ".3f", "mAP@50-95   ({})".format(n_note),
                "Accuracy by family and capacity  (Pohang VIS, ship only)",
                "each family's size ladder, small → large",
                OUT / "fig2_accuracy_by_family.png")

    if not fps:
        print("  fps.csv absent — skipping the latency figures")
        return 0
    shared = {v: d for v, d in fps.items() if v in acc}
    fps_note = "batch 1, fp32, seed mean ± sd, GPU clock pinned 1500 MHz"
    ranked_bars(shared, ".1f", f"frames per second   ({fps_note})",
                "Phase 1 — batch-1 throughput  (RTX 4080 Laptop)",
                f"{len(shared)} variants · 500 timed frames after 50 warmup · imgsz 640",
                OUT / "fig3a_fps_by_model.png")
    family_ladder(shared, ".1f", f"frames per second   ({fps_note})",
                "Throughput by family and capacity  (RTX 4080 Laptop, batch 1)",
                "each family's size ladder, small → large",
                OUT / "fig3b_fps_by_family.png")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
