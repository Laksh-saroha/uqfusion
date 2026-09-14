"""Phase 3 Stage 1 — the correspondence x mechanism crossing (cell D).

Executes `docs/prereg-phase3-retrain-2026-09-10.md` §4 **as written**. The rule is
fixed there; this file only transcribes it. Read §4 before this file.

The 2x2, on the shipped preset (`crossmodal26m`) over `runs/cache_m`, predictions,
gate, veto, capability prior and every other constant held fixed:

                    sigma inert            sigma live
    iou_thr 0.85    A  shipped             B  published: bit-identical to A
    iou_thr 0.55    C  published: -0.0138  D  never measured -- the experiment

Transcription choices, each forced by the prereg text and recorded here so they can be
checked against it rather than trusted:

* **"D non-inferior to A ... with the §8 block-bootstrap CI"** is read as the standard
  non-inferiority test: the upper end of the 95% CI of AP(A) - AP(D) lies below the 0.0060
  margin. A point estimate below 0.0060 with a CI reaching past it does not pass.
* **Decided on TUNE_RUNS (pohang00), reported on TEST_RUNS (pohang02+03)** — §4.3. TEST
  is printed next to TUNE and never enters the verdict. Both are day-only by
  construction; night (pohang01) is reported separately, as §8 requires, and decides
  nothing.
* **"On at least three of the four conditions"** — the four are `DEFAULT_CONDITIONS`
  (clean, fog, lowlight, glare), the same four the A/B/C numbers were published over.
* **Ship AP** (`cls=0`) throughout, as `local_ap50_95` per class — §4.2.
* **The §6 knob gate is REPORTED, not applied.** §4.2 asks for distinct `w_vis` values
  and IQR. `sigma_weighted` does not act through `w_vis` — it moves fused coordinates —
  so a constant `w_vis` here does not mean the arm is inert. Whether the sigma path
  actually changed anything is measured directly instead: the number of frames whose
  fused output differs between the sigma-live and sigma-inert cell at each threshold.

A, B and C are reproduction checks. Their published values come from `crossmodal-gate-
2026-09-01.md` §3c, measured under the bare `crossmodal` preset on the yolo26s caches and
selected on fit-run clean frames (n=1200); this run is on the shipped `crossmodal26m`
over the yolo26m caches. A mismatch is reported as a finding, not reconciled.

Usage:
    python scripts/stage1_crossing.py --out runs/eval/stage1_crossing_2026-09-14.md
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from _ideas_common import write_md                                         # noqa: E402
from uqfusion.eval.apmetrics import _score, ap_from_parts, frame_parts, presort  # noqa: E402
from uqfusion.eval.blockboot import block_bootstrap_delta, block_resample, run_ids, run_slices  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems                    # noqa: E402
from uqfusion.eval.identity import system_identity                         # noqa: E402

# --- everything §4 and §8 pin, in one block so it can be checked -------------------
PRESET = "crossmodal26m"
CACHE_DIR = "runs/cache_m"
THR_SHIPPED, THR_RELAXED = 0.85, 0.55     # §4.1: nothing below 0.55, whatever D shows
BLOCK_LEN, N_BOOT, SEED = 20, 1000, 0     # §8
MARGIN = 0.0060                           # §8 absolute floor, the non-inferiority margin
FLOORS = (0.0014, 0.0031, 0.0060, 0.0100) # §8: counts at four floors, verdict at 0.0060
MIN_CONDITIONS = 3                        # §4.3: of 4
SHIP = 0
PUBLISHED_C_MINUS_A = -0.0138

CELLS = {"A": (THR_SHIPPED, False), "B": (THR_SHIPPED, True),
         "C": (THR_RELAXED, False), "D": (THR_RELAXED, True)}


def ship_ap(parts, sel) -> float:
    e = ap_from_parts(parts, sel=sel)["per_class"].get(SHIP)
    return float(e["ap50_95"]) if e else float("nan")


def partner_rate(vis_recs, ir_in_vis, sel, iou_thr: float) -> float:
    """Share of VIS boxes with an IR box (mapped into the VIS canvas) at IoU >= iou_thr.

    Same definition as `eval_night_veto.partner_rate`, which is the one the 0.05% figure
    in §2(b) was quoted from. All VIS classes, no confidence cut.
    """
    num = den = 0
    for i in sel:
        a = np.asarray(vis_recs[i]["boxes_xyxy"], dtype=float).reshape(-1, 4)
        b = np.asarray(ir_in_vis[i]["boxes_xyxy"], dtype=float).reshape(-1, 4)
        den += len(a)
        if not len(a) or not len(b):
            continue
        x1 = np.maximum(a[:, None, 0], b[None, :, 0])
        y1 = np.maximum(a[:, None, 1], b[None, :, 1])
        x2 = np.minimum(a[:, None, 2], b[None, :, 2])
        y2 = np.minimum(a[:, None, 3], b[None, :, 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
        bb = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
        iou = inter / np.maximum(aa[:, None] + bb[None, :] - inter, 1e-9)
        num += int((iou.max(axis=1) >= iou_thr).sum())
    return num / den if den else float("nan")


def frames_differing(fused_a, fused_b, sel) -> int:
    """Frames whose fused output is not bit-identical between two cells."""
    n = 0
    for i in sel:
        ra, rb = fused_a[i], fused_b[i]
        same = all(np.array_equal(np.asarray(ra.get(k, [])), np.asarray(rb.get(k, [])))
                   for k in ("boxes_xyxy", "conf", "cls"))
        n += not same
    return n


def interaction_boot(parts: dict, paths, sel, n_boot: int, seed: int) -> dict:
    """(D - C) - (B - A), with all four cells scored on the SAME block resample."""
    idx = np.asarray(sel)
    per_run = run_slices(run_ids([paths[i] for i in idx]))
    pre = {k: presort(parts[k], idx) for k in "ABCD"}
    n = pre["A"]["n_frames"]

    def stat(w):
        s = {k: _score(pre[k], w, SHIP) for k in "ABCD"}
        return (s["D"] - s["C"]) - (s["B"] - s["A"])

    obs = stat(None)
    rng = np.random.default_rng(seed)
    vals = np.empty(n_boot)
    for t in range(n_boot):
        take = block_resample(rng, per_run, BLOCK_LEN)
        vals[t] = stat(np.bincount(take, minlength=n))
    kept = vals[np.isfinite(vals)]
    lo, hi = (np.percentile(kept, [2.5, 97.5]) if kept.size >= 2 else (np.nan, np.nan))
    return {"interaction": float(obs), "ci_lo": float(lo), "ci_hi": float(hi),
            "n_effective": int(kept.size)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/eval/stage1_crossing_2026-09-14.md")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    t0 = time.time()

    ctxs = {thr: load_context(preset=PRESET, cache_dir=CACHE_DIR, iou_thr=thr, verbose=(thr == THR_SHIPPED))
            for thr in (THR_SHIPPED, THR_RELAXED)}
    c85, c55 = ctxs[THR_SHIPPED], ctxs[THR_RELAXED]
    # §4.1: everything but the threshold is held fixed. Checked, not assumed.
    for attr in ("cap_vis", "cap_ir", "support_iou", "support_gamma", "single_passthrough",
                 "veto", "veto_rule", "veil_requires_night", "conditions"):
        if getattr(c85, attr) != getattr(c55, attr):
            raise SystemExit(f"contexts differ on {attr}: {getattr(c85, attr)!r} vs {getattr(c55, attr)!r}")
    if not np.array_equal(c85.runs, c55.runs):
        raise SystemExit("contexts differ on frame order")

    conditions = list(c85.conditions)
    sel = {"tune": c85.sel("tune"), "test": c85.sel("test"), "night": c85.sel("night")}
    print(f"[s1] conditions {conditions} | tune {len(sel['tune'])} test {len(sel['test'])} "
          f"night {len(sel['night'])}", flush=True)

    parts, fused, wvis, paths, irv = {}, {}, {}, {}, {}
    for cond in conditions:
        paths[cond] = [r["image_path"] for r in c85.vis_by_cond[cond]]
        for cell, (thr, sig) in CELLS.items():
            out = run_systems(ctxs[thr], cond, sigma_weighted=sig)
            parts[(cond, cell)] = frame_parts(out["fused_gated"], out["gts"])
            fused[(cond, cell)] = out["fused_gated"]
            wvis[(cond, cell)] = np.asarray(out["w_vis_gated"], dtype=float)
            irv[cond] = out["ir_in_vis"]
            print(f"[s1] {cond:9s} {cell} iou {thr} sigma {sig!s:5s} "
                  f"tune {ship_ap(parts[(cond, cell)], sel['tune']):.4f} ({time.time() - t0:.0f}s)",
                  flush=True)

    # ------------------------------------------------------------------ measurements
    ap_tab = {(cond, cell, s): ship_ap(parts[(cond, cell)], sel[s])
              for cond in conditions for cell in CELLS for s in sel}
    ni = {}      # non-inferiority, A - D
    for cond in conditions:
        for s in sel:
            r = block_bootstrap_delta(parts[(cond, "A")], parts[(cond, "D")], paths[cond],
                                      BLOCK_LEN, sel=sel[s], n_boot=args.boot, seed=SEED, cls=SHIP)
            ni[(cond, s)] = r
            print(f"[s1] A-D {cond:9s} {s:5s} {r['delta']:+.4f} [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]",
                  flush=True)
    inter = {(cond, s): interaction_boot({k: parts[(cond, k)] for k in "ABCD"}, paths[cond],
                                         sel[s], args.boot, SEED)
             for cond in conditions for s in ("tune", "test")}
    moved = {(cond, pair, s): frames_differing(fused[(cond, pair[0])], fused[(cond, pair[1])], sel[s])
             for cond in conditions for pair in (("B", "A"), ("D", "C")) for s in sel}
    partners = {(cond, thr, s): partner_rate(c85.vis_by_cond[cond], irv[cond], sel[s], thr)
                for cond in conditions for thr in (THR_SHIPPED, THR_RELAXED) for s in sel}
    knob = {}
    for cond in conditions:
        for cell in CELLS:
            w = wvis[(cond, cell)][sel["tune"]]
            q1, q3 = np.percentile(w, [25, 75]) if len(w) else (np.nan, np.nan)
            knob[(cond, cell)] = (int(len(np.unique(w))), float(q3 - q1))

    # ------------------------------------------------------------------- verdict (TUNE)
    def passes(cond, s, margin):
        return ni[(cond, s)]["ci_hi"] < margin
    counts = {f: sum(passes(c, "tune", f) for c in conditions) for f in FLOORS}
    n_pass = counts[MARGIN]
    verdict = "POSITIVE" if n_pass >= MIN_CONDITIONS else "S1-NULL"

    # -------------------------------------------------------------------------- report
    L = []
    L.append("Executes [`docs/prereg-phase3-retrain-2026-09-10.md`](../../docs/prereg-phase3-retrain-2026-09-10.md) "
             "§4, committed before this ran. The rule is fixed there; this report applies it. "
             f"Preset `{PRESET}`, caches `{CACHE_DIR}`, ship AP, block bootstrap L={BLOCK_LEN}, "
             f"n_boot={args.boot}, seed {SEED}.")
    L.append(f"## Verdict — **{verdict}**\n\n"
             f"D is non-inferior to A (upper 95% CI of AP(A) − AP(D) < {MARGIN}) on "
             f"**{n_pass} of {len(conditions)}** conditions on TUNE (pohang00); "
             f"{MIN_CONDITIONS} are required.\n\n"
             "Passing conditions by margin: "
             + ", ".join(f"**{f:.4f}: {counts[f]}/4**" if f == MARGIN else f"{f:.4f}: {counts[f]}/4"
                         for f in FLOORS)
             + f" — verdict taken at {MARGIN:.4f}.")

    rows = ["| condition | split | A | B | C | D | A − D | 95% CI | non-inferior |",
            "|---|---|---:|---:|---:|---:|---:|---|---|"]
    for cond in conditions:
        for s in ("tune", "test", "night"):
            r = ni[(cond, s)]
            tag = "**decides**" if s == "tune" else ("report" if s == "test" else "report (night)")
            rows.append(f"| {cond} | {s} ({tag}) | "
                        + " | ".join(f"{ap_tab[(cond, k, s)]:.4f}" for k in "ABCD")
                        + f" | {r['delta']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] | "
                        + ("yes" if r["ci_hi"] < MARGIN else "no") + " |")
    L.append("## The four cells, ship AP\n\n" + "\n".join(rows))

    rows = ["| condition | split | (D − C) − (B − A) | 95% CI |", "|---|---|---:|---|"]
    for cond in conditions:
        for s in ("tune", "test"):
            r = inter[(cond, s)]
            rows.append(f"| {cond} | {s} | {r['interaction']:+.4f} | [{r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}] |")
    L.append("## Interaction, §4.3\n\nReported, not the decision rule. Same block resample "
             "scores all four cells.\n\n" + "\n".join(rows))

    rows = ["| condition | split | B − A (published: bit-identical) | frames B ≠ A | C − A (published: −0.0138) | frames D ≠ C |",
            "|---|---|---:|---:|---:|---:|"]
    for cond in conditions:
        for s in sel:
            rows.append(f"| {cond} | {s} | {ap_tab[(cond, 'B', s)] - ap_tab[(cond, 'A', s)]:+.4f} | "
                        f"{moved[(cond, ('B', 'A'), s)]}/{len(sel[s])} | "
                        f"{ap_tab[(cond, 'C', s)] - ap_tab[(cond, 'A', s)]:+.4f} | "
                        f"{moved[(cond, ('D', 'C'), s)]}/{len(sel[s])} |")
    L.append("## Reproduction checks, and whether the sigma path moved anything\n\n"
             "The published A/B/C values were measured under bare `crossmodal` on the yolo26s "
             "caches, selected on fit-run clean frames (n=1200). This run is `crossmodal26m` on "
             "yolo26m. A mismatch is a finding and is not reconciled here.\n\n"
             "`frames X ≠ Y` counts frames whose fused boxes, scores or classes are not "
             "bit-identical between the two cells. It is the direct test of whether "
             "`sigma_weighted` did anything at that threshold.\n\n" + "\n".join(rows))

    rows = ["| condition | split | partner rate @0.85 | partner rate @0.55 |", "|---|---|---:|---:|"]
    for cond in conditions:
        for s in sel:
            rows.append(f"| {cond} | {s} | {partners[(cond, THR_SHIPPED, s)]:.2%} | "
                        f"{partners[(cond, THR_RELAXED, s)]:.2%} |")
    L.append("## Partner rate, §4.2\n\nShare of VIS boxes with an IR partner in the VIS canvas "
             "at the cell's threshold. All classes, no confidence cut.\n\n" + "\n".join(rows))

    rows = ["| condition | cell | distinct `w_vis` (tune) | IQR |", "|---|---|---:|---:|"]
    for cond in conditions:
        for cell in CELLS:
            k, iqr = knob[(cond, cell)]
            rows.append(f"| {cond} | {cell} | {k} | {iqr:.6f} |")
    L.append("## `w_vis` knob gate, §4.2 / §6 — reported, not applied\n\n"
             "`sigma_weighted` acts on fused coordinates, not on `w_vis`, so a single distinct "
             "value here does not make a cell inert; the `frames D ≠ C` column above is the "
             "test of that.\n\n" + "\n".join(rows))

    L.append("## What this does not settle\n\n"
             "* Development data only (§4.3). This is an exposure and is recorded in the "
             "exposure ledger.\n"
             f"* One trained model per stream (`{CACHE_DIR}`, pre-Phase-3 checkpoints). Training-"
             "seed variance is not in these intervals.\n"
             "* `runs/cache_m/gauss_vis_train_clean.pkl` is fitted on `maha_fit_vis.txt`, which "
             "holds 819 pohang04 frames (holdout audit). `crossmodal26m` drives weights from the "
             "capability prior alone, and no pohang04 frame is scored here; the reference is "
             "recorded, not claimed inert.")

    ident = system_identity(c85, preset=PRESET, cache_dir=CACHE_DIR, thresholds=[THR_SHIPPED, THR_RELAXED],
                            block_len=BLOCK_LEN, n_boot=args.boot, seed=SEED, margin=MARGIN)
    write_md(Path(args.out), "Phase 3 Stage 1 — correspondence × mechanism crossing", L, identity=ident)
    Path(args.out).with_suffix(".json").write_text(json.dumps({
        "verdict": verdict, "n_pass_tune": n_pass, "counts_by_floor": {str(k): v for k, v in counts.items()},
        "ap": {f"{c}/{k}/{s}": v for (c, k, s), v in ap_tab.items()},
        "noninferiority": {f"{c}/{s}": {k: r[k] for k in ("delta", "ci_lo", "ci_hi", "n_effective")}
                           for (c, s), r in ni.items()},
        "interaction": {f"{c}/{s}": r for (c, s), r in inter.items()},
        "frames_differing": {f"{c}/{p[0]}-{p[1]}/{s}": v for (c, p, s), v in moved.items()},
        "partner_rate": {f"{c}/{t}/{s}": v for (c, t, s), v in partners.items()},
        "w_vis": {f"{c}/{k}": {"distinct": d, "iqr": q} for (c, k), (d, q) in knob.items()},
    }, indent=2), encoding="utf-8")
    print(f"[s1] VERDICT {verdict} ({n_pass}/{len(conditions)} on tune) in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    raise SystemExit(main())
