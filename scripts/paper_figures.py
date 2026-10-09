"""Figures for PAPER_DRAFT2.md, drawn from the files the paper's tables cite.

Every number is read from a source file where one exists; the two exceptions are noted at
their figure (the Table 3a gate history and the night-restore verdict, which exist only as
Markdown records). Output: docs/figures/<name>.pdf (vector, for the manuscript) and .png
(300 dpi, for preview). Static print figures, so light mode only and no hover layer.

Palette: the dataviz reference categorical slots 1-3 in fixed order, one per entity and the
same entity in every figure (VIS blue, IR orange, fused aqua); validated all-pairs on the
light surface (worst CVD dE 9.2, normal-vision dE 24.0). Aqua sits at 2.74:1 against the
surface, so every aqua mark carries a direct label and its own marker shape.

Usage (from the repo root):
    python scripts/paper_figures.py              # all figures
    python scripts/paper_figures.py gate_history lift_screen    # some
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from statistics import mean

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "figures"

VIS, IR, FUSED = "#2a78d6", "#eb6834", "#1baf7a"
INK, INK2, MUTED = "#0b0b0b", "#52514e", "#8a8984"
GRID, RULE, BAND = "#e6e5e0", "#c3c2b7", "#f0efec"
SINGLE, DOUBLE = 3.5, 7.16                       # column widths, inches

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7.5,
    "text.color": INK, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "axes.edgecolor": RULE, "axes.linewidth": 0.8, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "xtick.major.size": 0, "ytick.major.size": 0,
    "legend.frameon": False, "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "pdf.fonttype": 42, "lines.solid_capstyle": "round",
})


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext, kw in (("pdf", {}), ("png", {"dpi": 300})):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", pad_inches=0.03, **kw)
    plt.close(fig)
    print(f"wrote docs/figures/{name}.pdf/.png")


def read_json(rel: str) -> dict:
    return json.loads((ROOT / rel).read_text(encoding="utf-8"))


def signed(v: float, nd: int = 4) -> str:
    """+0.0123 / −0.0123 with a true minus sign."""
    return f"{v:+.{nd}f}".replace("-", "−")


# --------------------------------------------------------------------------------------------
# Figure 1: the shipped decision layer (preset `crossmodal26m`), drawn from
# src/uqfusion/eval/ctx.py and the constants it loads (runs/eval/structure_constants.json,
# runs/eval/brightness_constants.json).
# --------------------------------------------------------------------------------------------

def fig1() -> None:
    sc = read_json("runs/eval/structure_constants.json")["axes"]
    bc = read_json("runs/eval/brightness_constants.json")["vis"]
    gini, lov = sc["grad_gini"]["threshold"], sc["lap_over_var"]["threshold"]
    irp, bound = sc["ir_p05"]["threshold"], sc["ir_health"].get("bound_switch", sc["ir_health"]["bound"])
    nkeys, mu_b = len(sc["ir_health"]["keys"]), bc["mu_b"]

    fig, ax = plt.subplots(figsize=(DOUBLE, 3.7))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 56)
    ax.axis("off")
    T, B = 7.0, 6.2                                             # title / body font size

    def box(x, y, w, h, title, body="", edge=RULE, fill="white", dashed=False):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.8",
                                    lw=0.9, ec=edge, fc=fill, ls=(0, (3, 2)) if dashed else "-"))
        if title:
            ax.text(x + 1.0, y + h - 1.1, title, ha="left", va="top", fontsize=T,
                    fontweight="bold", color=INK)
        if body:
            ax.text(x + 1.0, y + h - 4.2, body, ha="left", va="top", fontsize=B, color=INK2,
                    linespacing=1.4)

    def arrow(x0, y0, x1, y1, color=MUTED):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=0.9, mutation_scale=7,
                                    shrinkA=0, shrinkB=0))

    def stream(y0, y1, label, color):
        ax.add_patch(FancyBboxPatch((0.4, y0), 2.4, y1 - y0,
                                    boxstyle="round,pad=0,rounding_size=0.6", lw=0, fc=color))
        ax.text(1.6, (y0 + y1) / 2, label, rotation=90, ha="center", va="center", fontsize=6.8,
                color="white", fontweight="bold")

    # left column: each stream feeds its detector and its frame statistics
    stream(32, 54, "VIS frame", VIS)
    stream(7, 29.5, "IR frame", IR)
    box(5, 46, 23, 8, "VIS detector", "yolo26m, ship + buoy", edge=VIS)
    box(5, 32, 23, 12, "VIS statistics",
        f"dark:  p05 < {mu_b:g}\nveil:  grad_gini < {gini:.3f}\nconc.:  lap/var > {lov:.2f}",
        edge=VIS)
    box(5, 17.5, 23, 12, "IR statistics",
        f"night vote:  p05 > {irp:g}\nhealthy:  d² ≤ {bound:.1f}\n"
        f"({nkeys} statistics, p99 bound)", edge=IR)
    box(5, 7, 23, 8.5, "IR detector", "yolo26m-P2, ship only;\nNMS at IoU 0.70", edge=IR)
    for y in (50, 38, 23.5, 11.25):
        arrow(2.8, y, 5, y)

    # middle: the veto
    box(32, 17.5, 31, 26.5, "Veto VIS when",
        "night ∧ (dark ∨ veil)\n\n"
        "or, if IR votes night but is unhealthy:\n"
        "vote ∧ (conc. ∨ (dark ∧ veil))\n\n"
        "night = IR night vote ∧ IR healthy\n\n"
        "a vetoed stream is removed from the\nmerge input, not down-weighted")
    arrow(28, 38, 32, 38)
    arrow(28, 23.5, 32, 23.5)

    # right: merge, support, output
    box(67, 32, 32.5, 22, "Merge the surviving streams",
        "one survivor: passed through as is\n"
        "two: WBF at IoU 0.85 with constant\n"
        "capability weights, w_vis = 0.9926\n"
        "on every frame and condition\n"
        "0.05% of VIS boxes find an IR partner,\n"
        "so the merge is concatenation")
    box(67, 15, 32.5, 13, "Cross-modal support",
        "a box with a partner at IoU 0.30:\nscore × (1 + γ), γ = 0.5\n"
        "scores only; coordinates never move")
    box(67, 6.5, 32.5, 5, "Fused detections", fill=BAND, edge=BAND)

    arrow(28, 50, 67, 50, color=VIS)
    ax.text(47.5, 50.6, "VIS detections", ha="center", va="bottom", fontsize=B, color=INK2)
    ax.plot([28, 65, 65], [11.25, 11.25, 34.0], color=IR, lw=0.9)
    arrow(65, 34.0, 67, 34.0, color=IR)
    ax.text(46, 10.6, "IR detections", ha="center", va="top", fontsize=B, color=INK2)
    arrow(63, 38, 67, 38)
    arrow(83.25, 32, 83.25, 28)
    arrow(83.25, 15, 83.25, 11.5)

    # bottom strip: what the design had and the shipped preset switches off
    box(5, 0.4, 94.5, 4.6, "", edge=MUTED, dashed=True)
    ax.text(6, 2.7, "Inert in the shipped preset:", ha="left", va="center", fontsize=T,
            fontweight="bold", color=INK2)
    ax.text(34, 2.7, "σ head (sigma_weighted off, score α = 0)  ·  "
            "Mahalanobis OOD (μ_d = 1e9, λ = 0)", ha="left", va="center",
            fontsize=B, color=INK2)
    save(fig, "fig_decision_layer")


# --------------------------------------------------------------------------------------------
# Figure 3: per-stream calibration of the three UQ arms on the day slice (Table 2's primary
# basis), recomputed from the same caches Table 2 was scored on (runs/cache_uqslice/, with
# the declared VIS overrides) through scripts/slice_uq_day_night.py's own Arm loader and
# uqfusion.eval.metrics. Every scalar the panels imply is checked against Table 2's source,
# docs/eval/uq_day_night_slice_u2_nanpolicy_2026-09-09.json, before anything is drawn.
# --------------------------------------------------------------------------------------------

UQ_CACHES = {  # modality -> (label in the source JSON, label on the figure, cache, colour, marker)
    "VIS": [("sigma-head", "σ head", "sigma_vis_seed0_nightfull.pkl", "#4a3aa7", "o"),
            ("MC-Dropout", "MC-Dropout", "mc_vis_nightfull.pkl", "#e87ba4", "s"),
            ("ensemble(n=5)", "ensemble (5)", "ens_vis_nightfull.pkl", "#008300", "^")],
    "IR": [("sigma-head", "σ head", "sigma_ir.pkl", "#4a3aa7", "o"),
           ("MC-Dropout", "MC-Dropout", "mc_ir.pkl", "#e87ba4", "s"),
           ("ensemble(n=5)", "ensemble (5)", "ens_ir.pkl", "#008300", "^")],
}


def fig3() -> None:
    import numpy as np
    sys.path.insert(0, str(ROOT / "scripts"))
    import slice_uq_day_night as sl
    from uqfusion.eval.metrics import coverage_interval_ece, d_ece, sparsification

    ref = {m["modality"]: m for m in read_json("docs/eval/uq_day_night_slice_u2_nanpolicy_2026-09-09.json")}
    fig, axes = plt.subplots(2, 3, figsize=(DOUBLE, 4.5), gridspec_kw={"wspace": 0.36, "hspace": 0.45})
    for row, mod in enumerate(("VIS", "IR")):
        a_rel, a_cov, a_sp = axes[row]
        for a in (a_rel, a_cov):
            a.plot([0, 1], [0, 1], color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
        a_sp.axhline(0, color=INK2, lw=0.7, zorder=1)
        for src, lab, cache, col, mk in UQ_CACHES[mod]:
            arm = sl.Arm(lab, sl.CACHE / cache)
            is_night = np.array([sl.NIGHT_RUN in str(p) for p in arm.paths])
            i = arm.gather(np.flatnonzero(~is_night))
            conf, matched = arm.conf[i], arm.matched[i]
            cov = coverage_interval_ece(arm.err[i], arm.sigma[i])
            sp = sparsification(arm.u[i], arm.risk[i])
            want = ref[mod]["point"][src]["day"]
            got = {"d_ece": d_ece(conf, matched), "interval_ece": cov["interval_ece"],
                   "ause": sp["ause"], "aurc": sp["aurc"]}
            # AUSE/AURC read a sort of u, and u has exact ties wherever ensemble/MC members
            # agree (only 57-68% of ensemble day boxes have a unique u). Table 2 was scored on
            # 2026-09-09 with numpy's default sort; the stable sort was pinned on 2026-09-10
            # (9c0f921). Measured over tie orders (stable, quicksort, 20 random permutations):
            # VIS ensemble AUSE 0.0948-0.0951, IR ensemble 0.0769-0.0777, MC < 2e-5, sigma head
            # 0. So those two are checked to 5e-4, inside which no Table 2 ordering changes, and
            # d-ECE / interval-ECE exactly.
            for k, v in got.items():
                if abs(v - want[k]) > (5e-4 if k in ("ause", "aurc") else 1e-9):
                    raise SystemExit(f"fig3: {mod} {src} day {k} = {v} but Table 2's source says {want[k]}")

            edges = np.linspace(0, 1, 11)                          # d_ece's own 10 bins
            idx = np.clip(np.digitize(conf, edges) - 1, 0, 9)
            bx = [conf[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
            by = [matched[idx == b].mean() for b in range(10) if (idx == b).sum() >= 20]
            a_rel.plot(bx, by, color=col, lw=1.2, marker=mk, ms=3.6, mec="white", mew=0.5, zorder=3)

            ks = sorted(cov["coverage"])
            a_cov.plot([cov["coverage"][k]["nominal"] for k in ks],
                       [cov["coverage"][k]["empirical"] for k in ks],
                       color=col, lw=1.2, marker=mk, ms=3.6, mec="white", mew=0.5, zorder=3)

            err = np.asarray(sp["risk_by_u"]) - np.asarray(sp["risk_oracle"])
            a_sp.plot(sp["fractions"], err, color=col, lw=1.2, marker=mk, ms=3.0, mec="white",
                      mew=0.4, zorder=3, markevery=3)
        n_day = int(ref[mod]["n_day"])
        a_rel.set_title(f"{mod}: confidence reliability", loc="left", color=INK)
        a_cov.set_title(f"{mod}: interval coverage (TP only)", loc="left", color=INK)
        a_sp.set_title(f"{mod}: sparsification error", loc="left", color=INK)
        a_rel.set(xlim=(0, 1.02), ylim=(0, 1.03), xlabel="confidence", ylabel="precision at IoU 0.5")
        a_cov.set(xlim=(0.3, 1.03), ylim=(0, 1.03), xlabel="nominal Gaussian coverage",
                  ylabel="empirical coverage")
        a_sp.set(xlim=(-0.02, 0.97), xlabel="fraction of boxes removed",
                 ylabel="risk gap to oracle order")
        a_rel.text(0.98, 0.03, f"{n_day:,} day frames", ha="right", va="bottom", fontsize=6.2, color=INK2)
    handles = [plt.Line2D([], [], color=c, marker=m, ms=4, lw=1.2, mec="white", mew=0.5, label=l)
               for _, l, _, c, m in UQ_CACHES["VIS"]]
    fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 1.0), ncol=3,
               handletextpad=0.4, columnspacing=1.6)
    fig.subplots_adjust(top=0.88)
    save(fig, "fig_uq_calibration")


# --------------------------------------------------------------------------------------------
# Figure 2: worst-cell gap across the six gate rewrites. Table 3a of the draft is the record
# (compiled in PAPER_CONTEXT_COMPILED.md Part 6); no machine-readable file holds the history.
# --------------------------------------------------------------------------------------------

STAGES = [  # stage, date, worst cell, ship-AP gap to max(VIS, IR), hollow = within CI
    # a, b: the 08-19/20 records ranked cells against ir_only (fog/night); against max(VIS, IR)
    # their worst cell is lowlight/day (x_fusion_ci.md §4, final_system.md §2; class_set_audit).
    ("a", "08-19", "lowlight/day", -0.0180, False),
    ("b", "08-20", "lowlight/day", -0.0180, False),
    ("c", "09-01", "lowlight/day", -0.0180, False),
    ("d", "09-01", "", 0.0, False),
    ("e", "09-01", "fog/clean", -0.0632, False),
    ("e′", "09-01", "", 0.0, False),
]


def fig2() -> None:
    fig, ax = plt.subplots(figsize=(SINGLE, 2.35))
    xs = list(range(len(STAGES)))
    ax.axhline(0, color=INK2, lw=0.8, zorder=1)
    ax.plot(xs, [s[3] for s in STAGES], color=VIS, lw=1.2, zorder=2)
    for x, (st, date, cell, gap, hollow) in zip(xs, STAGES):
        ax.scatter([x], [gap], s=30, zorder=3, color="white" if hollow else VIS, edgecolor=VIS,
                   linewidth=1.4)
        if gap < 0:
            # a run of stages sharing one worst cell and gap is labelled once, at its centre
            same = lambda i: 0 <= i < len(STAGES) and STAGES[i][2:4] == (cell, gap)
            if same(x - 1):
                continue
            end = x
            while same(end + 1):
                end += 1
            span = f" ({STAGES[x][0]}–{STAGES[end][0]})" if end > x else ""
            text = f"{signed(gap)}\n{cell}{span}" + ("\n(within CI)" if hollow else "")
            ax.text((x + end) / 2, gap - 0.004, text, ha="center", va="top", fontsize=6.2,
                    color=INK2, linespacing=1.15)
        else:
            ax.text(x, gap + 0.004, f"tie\n{cell}" if cell else "0", ha="center", va="bottom",
                    fontsize=6.2, color=INK2, linespacing=1.15)
    ax.set_xticks(xs, [f"{s[0]}\n{s[1]}" for s in STAGES])
    ax.set_ylim(-0.08, 0.0145)
    ax.set_xlim(-0.5, len(STAGES) - 0.5)
    ax.set_ylabel("worst-cell gap to max(VIS, IR)")
    ax.set_xlabel("gate rewrite (2026 date)")
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda v, _: signed(v, 2) if v else "0"))
    ax.grid(axis="x", visible=False)
    save(fig, "fig_gate_history")


# --------------------------------------------------------------------------------------------
# Figure for Table 3b: the shipped rule on the five Phase 3 systems, per cell.
# Sources: docs/eval/p3_night_check_2026-09-27.json (clean) and
# docs/eval/p3_corrupt_cells_2026-09-27.json (fog, lowlight, glare at severity 2).
# --------------------------------------------------------------------------------------------

def fig3b() -> None:
    nc = read_json("docs/eval/p3_night_check_2026-09-27.json")["slices"]
    cc = read_json("docs/eval/p3_corrupt_cells_2026-09-27.json")["cells"]
    rows = []
    for tod in ("day", "night"):
        a = nc[tod]["ap_mean"]
        rows.append((f"clean / {tod}", a["vis"], a["ir"], a["on"], nc[tod]["claim_fails"]))
        for cond in ("fog", "lowlight", "glare"):
            c = cc[f"{cond}/{tod}"]
            m = c["ap_mean"]
            rows.append((f"{cond} / {tod}", m["vis"], m["ir"], m["fused"], c["claim_fails"]))

    fig, ax = plt.subplots(figsize=(SINGLE, 3.0))
    ys = [len(rows) - i - (1 if i >= 4 else 0) for i in range(len(rows))]   # gap between day and night
    for y, (name, v, i, f, fails) in zip(ys, rows):
        lo, hi = min(v, i, f), max(v, i, f)
        ax.plot([lo, hi], [y, y], color=GRID, lw=2.2, zorder=1)
        ax.scatter([v], [y], s=26, marker="o", color=VIS, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([i], [y], s=24, marker="s", color=IR, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([f], [y], s=70, marker="o", facecolor="none", edgecolor=FUSED, linewidth=1.6, zorder=4)
        if fails:
            ax.text(f + 0.012, y + 0.33, f"fused − VIS {signed(f - v)}", fontsize=6.3,
                    color=INK2, va="bottom", ha="left")
    ax.set_yticks(ys, [r[0] for r in rows])
    ax.set_xlim(-0.005, 0.40)
    ax.set_ylim(min(ys) - 0.8, max(ys) + 1.3)
    ax.set_xlabel("ship AP50-95 (local AP), mean of 5 seed pairs")
    ax.grid(axis="y", visible=False)
    ax.axhline((ys[3] + ys[4]) / 2, color=RULE, lw=0.6)
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=VIS, label="VIS only"),
               plt.Line2D([], [], ls="", marker="s", ms=4.6, color=IR, label="IR only"),
               plt.Line2D([], [], ls="", marker="o", ms=7.5, mfc="none", mec=FUSED, mew=1.6,
                          label="fused (shipped)")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=3,
              handletextpad=0.3, columnspacing=1.0)
    save(fig, "fig_phase3_cells")


# --------------------------------------------------------------------------------------------
# Figure 4: the night-label restore. Values from runs/eval/night_restore_verdict.md (the
# pre-registered verdict; bands from docs/prereg-night-label-restore.md).
# --------------------------------------------------------------------------------------------

def fig4() -> None:
    old, new, ci = 0.0000, 0.2520, (0.2473, 0.2567)
    dead, alive = 0.005, 0.02
    fig, ax = plt.subplots(figsize=(SINGLE, 2.4))
    ax.axhspan(0, dead, color="#e7e6e2", lw=0, zorder=0)
    ax.axhspan(dead, alive, color=BAND, lw=0, zorder=0)
    ax.axhline(alive, color=INK2, lw=0.7, ls=(0, (3, 2)), zorder=1)
    ax.text(0.5, alive + 0.004, "ALIVE threshold 0.02\n(pre-registered)", fontsize=6.2, color=INK2,
            va="bottom", ha="center", linespacing=1.15)
    xs = [0, 1]
    ax.bar(xs, [old, new], width=0.42, color=VIS, zorder=2)
    ax.errorbar([1], [new], yerr=[[new - ci[0]], [ci[1] - new]], fmt="none", ecolor=INK,
                elinewidth=0.9, capsize=3, zorder=3)
    ax.text(0, alive + 0.004, "0.0000", ha="center", va="bottom", fontsize=7, color=INK)
    ax.text(1, ci[1] + 0.006, f"{new:.4f}\n[{ci[0]:.4f}, {ci[1]:.4f}]", ha="center", va="bottom",
            fontsize=7, color=INK, linespacing=1.2)
    ax.set_xticks(xs, ["shipped checkpoint\n(filtered labels)", "fine-tuned on\nrestored labels"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, 0.315)
    ax.set_ylabel("night VIS mAP50-95")
    ax.grid(axis="x", visible=False)
    save(fig, "fig_night_restore")


# --------------------------------------------------------------------------------------------
# Figure 5: the lift screen, clean condition. Source: runs/eval/signal_lift_26m.json.
# --------------------------------------------------------------------------------------------

LIFT_SIGNALS = [  # (row label in the source, label on the figure)
    ("conf above its frame median", "confidence above\nframe median"),
    ("sigma below its frame median (tight box)", "σ below frame\nmedian"),
    ("cross-modal IoU0.30", "cross-modal partner\nat IoU 0.30"),
    ("temporal k2 IoU.30", "temporal support\n(k = 2, IoU 0.30)"),
]


def fig5() -> None:
    rows = read_json("runs/eval/signal_lift_26m.json")["rows"]
    lift = {(r["cond"], r["signal"], str(r["iou"])): float(r["lift"]) for r in rows}
    fig, ax = plt.subplots(figsize=(SINGLE, 2.25))
    ys = list(range(len(LIFT_SIGNALS)))[::-1]
    ax.axvline(1.0, color=INK2, lw=0.8, zorder=1)
    for y, (src, lab) in zip(ys, LIFT_SIGNALS):
        l50, l75 = lift[("clean", src, "50")], lift[("clean", src, "75")]
        ax.plot([1.0, l50], [y, y], color=VIS, lw=2, zorder=2)
        ax.scatter([l50], [y], s=30, color=VIS, zorder=3, edgecolor="white", linewidth=0.8)
        ax.scatter([l75], [y], s=26, facecolor="white", edgecolor=VIS, linewidth=1.2, zorder=3)
        ax.text(max(l50, l75) + 0.18, y, f"{l50:.2f}×", va="center", ha="left", fontsize=7, color=INK)
    ax.set_yticks(ys, [s[1] for s in LIFT_SIGNALS], fontsize=7)
    ax.set_xlim(0.6, 5.9)
    ax.set_ylim(-0.6, len(LIFT_SIGNALS) - 0.4)
    ax.set_xlabel("lift = P(TP | signal) / P(TP | no signal)")
    ax.grid(axis="y", visible=False)
    ax.text(1.06, -0.55, "no information", fontsize=6.2, color=INK2, va="bottom", ha="left")
    handles = [plt.Line2D([], [], ls="", marker="o", ms=5, color=VIS, label="TP at IoU 0.50"),
               plt.Line2D([], [], ls="", marker="o", ms=4.8, mfc="white", mec=VIS, mew=1.2,
                          label="TP at IoU 0.75")]
    ax.legend(handles=handles, loc="lower right", bbox_to_anchor=(1.0, 0.0), handletextpad=0.2)
    save(fig, "fig_lift_screen")


# --------------------------------------------------------------------------------------------
# Figure 6: best-epoch vs epoch-mean on the two same-machine UQ runs of D31.
# Sources: runs/ensemble/ens_vis_seed0_ft_control/results.csv,
#          runs/mc_dropout/mc_vis_seed0_ft_refit/results.csv.
# --------------------------------------------------------------------------------------------

def epoch_curve(rel: str) -> list[float]:
    rows = list(csv.DictReader(open(ROOT / rel / "results.csv", encoding="utf-8")))
    k = next(c for c in rows[0] if "mAP50-95" in c)
    return [float(r[k]) for r in rows]


def fig6() -> None:
    runs = [("ensemble member (control)", "runs/ensemble/ens_vis_seed0_ft_control", VIS, "o"),
            ("MC-Dropout", "runs/mc_dropout/mc_vis_seed0_ft_refit", IR, "s")]
    fig, ax = plt.subplots(figsize=(SINGLE, 2.5))
    for name, rel, col, mk in runs:
        ys = epoch_curve(rel)
        xs = list(range(1, len(ys) + 1))
        best = max(range(len(ys)), key=lambda i: ys[i])      # first maximum, as best.pt keeps
        mu = mean(ys)
        ax.plot(xs, ys, color=col, lw=1.4, zorder=2)
        ax.scatter(xs, ys, s=12, color=col, marker=mk, zorder=3, edgecolor="white", linewidth=0.5)
        ax.scatter([xs[best]], [ys[best]], s=58, facecolor="none", edgecolor=col, linewidth=1.4, zorder=4)
        ax.hlines(mu, 0.7, len(ys) + 0.3, color=col, lw=1.0, ls=(0, (4, 2.5)), zorder=1)
        ax.text(len(ys) + 0.45, mu, f"mean {mu:.4f}", va="center", ha="left", fontsize=6.4, color=INK2)
        ax.text(xs[best] + 0.25, ys[best] + 0.0012, f"best {ys[best]:.4f}", va="bottom",
                ha="left", fontsize=6.4, color=INK2)
    ax.set_xlim(0.6, 12.6)
    ax.set_xticks(range(1, 11))
    ax.set_ylim(0.222, 0.2645)
    ax.set_xlabel("fine-tune epoch")
    ax.set_ylabel("val mAP50-95 (Ultralytics)")
    handles = [plt.Line2D([], [], color=c, marker=m, ms=3.5, lw=1.4, label=n) for n, _, c, m in runs]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2,
              handletextpad=0.4, columnspacing=1.2)
    save(fig, "fig_checkpoint_selection")


FIGS = {  # paper order: Figure 1 ... Figure 7
    "decision_layer": fig1, "uq_calibration": fig3, "lift_screen": fig5,
    "checkpoint_selection": fig6, "gate_history": fig2, "phase3_cells": fig3b,
    "night_restore": fig4,
}


def main(argv: list[str]) -> int:
    names = argv or list(FIGS)
    unknown = [n for n in names if n not in FIGS]
    if unknown:
        raise SystemExit(f"unknown figure(s) {unknown}; choose from {list(FIGS)}")
    for n in names:
        FIGS[n]()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
