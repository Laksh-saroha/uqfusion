"""Per-frame DETECTOR-EVIDENCE statistics, and what they do to the eight cells.

**Why.** Every veto axis in the adopted gate is an INPUT-IMAGE statistic: `p05`
asks "did photons arrive?", `lap_var` asks "did the photons carry edges?". Both
are properties of the picture, and `docs/gated-fusion-handoff.md` §7.2 claims --
with a measurement behind it -- that no such statistic can separate lowlight/day
(VIS alive at 0.0346) from clean/night (VIS dead at 0.0000), because on every
structure metric tried lowlight/day sits BETWEEN two cells that must both be
vetoed.

That claim is about image statistics. It says nothing about the OUTPUT side. The
gate is supposed to answer "can this sensor's detector produce usable detections
on this frame?", and the most direct evidence for that question is the detector's
own response -- which is already sitting in the caches and which the gate
currently throws away (`r_box` reads sigma, never conf, and the §6 ablation shows
`r_box` is inert).

This script only MEASURES. It writes a report and changes nothing.

Usage:
    python scripts/probe_detector_evidence.py --out runs/eval/probe_detector_evidence.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.cache import load_cache          # noqa: E402
from uqfusion.eval.hysteresis import temporal_order  # noqa: E402

CONDITIONS = ("clean", "fog", "lowlight", "glare")
NIGHT_RUN = "pohang01"
FIT_RUNS = ("pohang00", "pohang02", "pohang03")
STATS = ("max_conf", "sum_conf", "top5_conf", "n_c10", "n_c25", "feat_norm")


def evidence(rec: dict) -> dict:
    """Everything a frame's detection set says about whether the sensor is working.

    All of these are computable at inference time from the detector alone -- no
    labels, no second stream, no image re-read.
    """
    c = np.asarray(rec["conf"], dtype=np.float64).reshape(-1)
    fn = float(np.linalg.norm(np.asarray(rec["feat"], dtype=np.float64)))
    if len(c) == 0:
        return {"max_conf": 0.0, "sum_conf": 0.0, "n_dets": 0, "top5_conf": 0.0,
                "n_c10": 0, "n_c25": 0, "n_c50": 0, "mean_conf": 0.0, "feat_norm": fn}
    s = np.sort(c)[::-1]
    return {
        "max_conf": float(s[0]),
        "sum_conf": float(c.sum()),
        "n_dets": int(len(c)),
        "top5_conf": float(s[:5].mean()),
        "n_c10": int((c >= 0.10).sum()),
        "n_c25": int((c >= 0.25).sum()),
        "n_c50": int((c >= 0.50).sum()),
        "mean_conf": float(c.mean()),
        "feat_norm": fn,
    }


def window_max(v: np.ndarray, order: dict, k: int) -> np.ndarray:
    """Max of `v` over a centred window of k frames, within each run, in capture order.

    A sensor being blind is a property of a STRETCH of a recording, not of one
    frame: an empty-sea clean day frame legitimately has nothing to detect, and a
    per-frame evidence test would veto it. Both adopted filters already encode
    that fact about the signal; this encodes it about the statistic instead, which
    is the same argument applied one step earlier.
    """
    if k <= 1:
        return v.copy()
    out = v.copy()
    half = k // 2
    for idx in order.values():
        seq = v[idx]
        pad = np.pad(seq, (half, half), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, k)[:len(seq)]
        out[idx] = win.max(axis=1)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache-dir", default="runs/cache")
    ap.add_argument("--out", default="runs/eval/probe_detector_evidence.md")
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 15, 31, 61])
    args = ap.parse_args()

    cd = ROOT / args.cache_dir
    per_cond, runs, order = {}, None, None
    for cond in CONDITIONS:
        recs, _ = load_cache(cd / f"gauss_vis_paired_{cond}.pkl")
        per_cond[cond] = [evidence(r) for r in recs]
        if runs is None:
            runs = np.asarray([Path(r["image_path"]).parent.name for r in recs])
            order = temporal_order(recs)
        print(f"[probe] {cond}: {len(recs)} frames", flush=True)

    night = runs == NIGHT_RUN
    keys = sorted(per_cond["clean"][0])
    cells = [(c, lab, m) for c in CONDITIONS for lab, m in (("day", ~night), ("night", night))]

    L = ["# Detector-evidence probe (measurement only)", "",
         "Per-frame statistics of the VIS detector's OWN output, by cell. Every "
         "column is available at inference time with no labels and no second "
         "stream. `feat_norm` is the pooled-neck-feature L2 norm, included as the "
         "one non-detection column so the comparison is not purely about conf.", "",
         "## 1. Median per cell", "",
         "| cell | n | " + " | ".join(keys) + " |",
         "|---|---:|" + "---:|" * len(keys)]
    per_cell_median = {}
    for cond, lab, m in cells:
        vals = {k: np.asarray([e[k] for e in per_cond[cond]], dtype=float)[m] for k in keys}
        per_cell_median[f"{cond}/{lab}"] = {k: float(np.median(vals[k])) for k in keys}
        L.append(f"| {cond}/{lab} | {int(m.sum())} | "
                 + " | ".join(f"{np.median(vals[k]):.4g}" for k in keys) + " |")

    L += ["", "## 2. Windowed max over capture order, by cell", "",
          "`w=1` is the per-frame value. The threshold is a NOVELTY bound fitted on "
          "CLEAN FIT-RUN frames only (pohang00/02/03) -- the same protocol as "
          "`tau_lap`: a hard minimum, plus the p01 quantile for reference. No "
          "corrupted frame and no held-out night frame informs it.", ""]

    results = {}
    for key in STATS:
        for w in args.windows:
            wc = {c: window_max(np.asarray([e[key] for e in per_cond[c]], dtype=float), order, w)
                  for c in CONDITIONS}
            ref = wc["clean"][np.isin(runs, FIT_RUNS)]
            thr_min, thr_p01 = float(ref.min()), float(np.quantile(ref, 0.01))
            row = {"stat": key, "window": w, "thr_min": thr_min, "thr_p01": thr_p01, "cells": {}}
            for cond, lab, m in cells:
                v = wc[cond][m]
                row["cells"][f"{cond}/{lab}"] = {
                    "median": float(np.median(v)),
                    "veto_min": float((v < thr_min).mean()),
                    "veto_p01": float((v < thr_p01).mean()),
                }
            results[(key, w)] = row

    L += ["| stat | w | thr(min) | " + " | ".join(f"{c}/{l}" for c, l, _ in cells) + " |",
          "|---|---:|---:|" + "---:|" * len(cells)]
    for (key, w), row in results.items():
        L.append(f"| {key} | {w} | {row['thr_min']:.4g} | "
                 + " | ".join(f"{row['cells'][f'{c}/{l}']['veto_min']:.1%}" for c, l, _ in cells)
                 + " |")

    L += ["", "### median of the windowed statistic", "",
          "| stat | w | " + " | ".join(f"{c}/{l}" for c, l, _ in cells) + " |",
          "|---|---:|" + "---:|" * len(cells)]
    for (key, w), row in results.items():
        L.append(f"| {key} | {w} | "
                 + " | ".join(f"{row['cells'][f'{c}/{l}']['median']:.4g}" for c, l, _ in cells)
                 + " |")

    L += ["", "## 3. Which (stat, window, rule) satisfies the full constraint", "",
          "PASS = clean/day and glare/day at 0.0%, lowlight/day at 0.0%, fog/day at "
          "100.0%, and all four night cells at 100.0%. That is the complete "
          "eight-cell target the handoff says no image statistic reaches.", "",
          "| stat | w | rule | clean/day | glare/day | lowlight/day | fog/day | nights | PASS |",
          "|---|---:|---|---:|---:|---:|---:|---|---|"]
    passes = []
    for (key, w), row in results.items():
        for rule in ("veto_min", "veto_p01"):
            c = row["cells"]
            nights = [c[f"{x}/night"][rule] for x in CONDITIONS]
            ok = (c["clean/day"][rule] == 0.0 and c["glare/day"][rule] == 0.0
                  and c["lowlight/day"][rule] == 0.0 and c["fog/day"][rule] == 1.0
                  and all(n == 1.0 for n in nights))
            L.append(f"| {key} | {w} | {rule} | {c['clean/day'][rule]:.1%} | "
                     f"{c['glare/day'][rule]:.1%} | {c['lowlight/day'][rule]:.1%} | "
                     f"{c['fog/day'][rule]:.1%} | {min(nights):.1%}-{max(nights):.1%} | "
                     f"{'**PASS**' if ok else ''} |")
            if ok:
                passes.append({"stat": key, "window": w, "rule": rule,
                               "thr": row["thr_min"] if rule == "veto_min" else row["thr_p01"]})

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"per_cell_median": per_cell_median,
         "windowed": [{**v, "key": f"{k[0]}@{k[1]}"} for k, v in results.items()],
         "passes": passes}, indent=2), encoding="utf-8")
    print(f"[probe] wrote {out}  ({len(passes)} passing rules)")
    for p in passes:
        print(f"[probe] PASS  {p['stat']} window={p['window']} {p['rule']} thr={p['thr']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
