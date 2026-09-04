"""Can any INPUT-IMAGE statistic keep lowlight/day while vetoing clean/night?

`docs/gated-fusion-handoff.md` §7.2 says no, and invites an argument. The claim
rests on `lap_var`, where lowlight/day (152) sits between clean/night (1334) and
lowlight/night (23), so no monotone threshold can separate them. The reason is
structural rather than empirical: `lap_var` is a variance of a LINEAR operator, so
under the lowlight corruption -- `RandomBrightnessContrast(brightness_limit=
(-0.9,-0.7))`, i.e. multiplication by roughly 0.1-0.3 -- it falls by the SQUARE of
that factor from amplitude alone, with the scene's structure untouched. It is
measuring gain, not structure.

`frame_structure.py` computes statistics chosen to be invariant to that gain
(`lap_over_var`, `grad_gini`, `tex_cover`, `tile_std_p50`, `spec_slope`,
`edge_density`, `log_range`). This script asks, of each of them, the exact
question §7.2 poses:

    is there a threshold that vetoes 100% of clean/night and 0% of lowlight/day,
    while leaving clean/day and glare/day at 0% and still catching fog?

and reports the separation margin so a near-miss is visible as a near-miss rather
than as a bare fail. Measurement only.

Usage:
    python scripts/probe_structure_separation.py --out runs/eval/probe_structure.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
NIGHT_RUN = "pohang01"
FIT_RUNS = ("pohang00", "pohang02", "pohang03")
PHOTOMETRIC = {"mean", "p05", "p50", "p95", "std", "range", "frac_dark", "lap_var"}


def load(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))["frames"]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--struct-dir", default="runs/derived/structure")
    ap.add_argument("--out", default="runs/eval/probe_structure.md")
    args = ap.parse_args()

    sd = ROOT / args.struct_dir
    conds = [c for c in ("clean", "fog", "lowlight", "glare")
             if (sd / f"gauss_vis_paired_{c}.json").is_file()]
    if "clean" not in conds:
        raise SystemExit("need at least gauss_vis_paired_clean.json")
    frames = {c: load(sd / f"gauss_vis_paired_{c}.json") for c in conds}
    runs = np.asarray([f["run"] for f in frames["clean"]])
    night = runs == NIGHT_RUN
    cells = [(c, lab, m) for c in conds for lab, m in (("day", ~night), ("night", night))]
    keys = sorted(k for k, v in frames["clean"][0].items() if isinstance(v, (int, float)))

    train_p = sd / "gauss_vis_train_clean.json"
    train = load(train_p) if train_p.is_file() else None

    vals = {(c, k): np.asarray([f[k] for f in frames[c]], dtype=float)
            for c in conds for k in keys}

    L = ["# Input-image structure statistics vs the §7.2 constraint", "",
         f"Conditions present: {', '.join(conds)}. Statistics marked (photometric) "
         "are the pre-existing block, reproduced so the new columns are read "
         "against them rather than in isolation.", "",
         "## 1. Median per cell", "",
         "| stat | " + " | ".join(f"{c}/{l}" for c, l, _ in cells) + " |",
         "|---|" + "---:|" * len(cells)]
    for k in keys:
        tag = " (photometric)" if k in PHOTOMETRIC else ""
        L.append(f"| {k}{tag} | "
                 + " | ".join(f"{np.median(vals[(c, k)][m]):.4g}" for c, l, m in cells) + " |")

    # ---- the §7.2 separation question --------------------------------------
    # KEEP  = must stay unvetoed: every frame of lowlight/day.
    # VETO  = must be vetoed: every frame of clean/night.
    # A statistic separates them iff the two sets do not overlap in the direction
    # the veto runs (veto when LOW, or veto when HIGH -- both are tried, since a
    # statistic like grad_gini is inverted relative to a focus measure).
    L += ["", "## 2. §7.2: separate lowlight/day (keep) from clean/night (veto)", "",
          "`sep` is the margin between the two sets in the stated direction: "
          "positive means a threshold exists that vetoes EVERY clean/night frame "
          "and NO lowlight/day frame. Reported as a ratio of the gap to the "
          "keep-set's spread, so it is comparable across statistics with "
          "different units.", "",
          "| stat | direction | keep(lowlight/day) min..max | veto(clean/night) min..max | sep | separates |",
          "|---|---|---|---|---:|---|"]
    seps = []
    if "lowlight" in conds:
        keep = {k: vals[("lowlight", k)][~night] for k in keys}
        veto = {k: vals[("clean", k)][night] for k in keys}
        for k in keys:
            a, b = keep[k], veto[k]
            spread = max(a.max() - a.min(), 1e-12)
            # veto-when-LOW works iff every veto frame is below every keep frame
            lo_gap = (a.min() - b.max()) / spread
            hi_gap = (b.min() - a.max()) / spread
            direction, gap = (("veto when LOW", lo_gap) if lo_gap >= hi_gap
                              else ("veto when HIGH", hi_gap))
            seps.append({"stat": k, "direction": direction, "sep": float(gap),
                         "separates": bool(gap > 0)})
            L.append(f"| {k} | {direction} | {a.min():.4g}..{a.max():.4g} | "
                     f"{b.min():.4g}..{b.max():.4g} | {gap:+.3f} | "
                     f"{'**YES**' if gap > 0 else 'no'} |")

    # ---- full eight-cell test for the statistics that survive --------------
    # A statistic that separates the §7.2 pair still has to behave on the other
    # six cells, fitted the same way tau_lap was: a hard novelty bound on CLEAN
    # FIT-RUN frames, no corrupted frame and no held-out night frame involved.
    L += ["", "## 3. Novelty-bound veto rates over every available cell", "",
          "Threshold = hard min (veto-when-LOW) or hard max (veto-when-HIGH) of "
          "the statistic over CLEAN frames of pohang00/02/03 -- the `tau_lap` "
          "protocol exactly. Target: clean/day 0%, glare/day 0%, lowlight/day 0%, "
          "fog/* 100%, every night cell 100%.", ""]
    if train is not None:
        fitm = np.asarray([f["run"] in FIT_RUNS for f in train])
        L += ["| stat | direction | thr | "
              + " | ".join(f"{c}/{l}" for c, l, _ in cells) + " | PASS |",
              "|---|---|---:|" + "---:|" * len(cells) + "---|"]
        passes = []
        for s in seps:
            k, d = s["stat"], s["direction"]
            tv = np.asarray([f[k] for f in train], dtype=float)[fitm]
            thr = float(tv.min()) if d == "veto when LOW" else float(tv.max())
            rates = {}
            for c, lab, m in cells:
                v = vals[(c, k)][m]
                rates[f"{c}/{lab}"] = float((v < thr).mean() if d == "veto when LOW"
                                            else (v > thr).mean())
            ok = (rates.get("clean/day", 1) == 0.0 and rates.get("glare/day", 1) == 0.0
                  and rates.get("lowlight/day", 1) == 0.0
                  and all(v == 1.0 for kk, v in rates.items() if kk.endswith("/night"))
                  and rates.get("fog/day", 0) == 1.0)
            if ok:
                passes.append({"stat": k, "direction": d, "thr": thr})
            L.append(f"| {k} | {d} | {thr:.4g} | "
                     + " | ".join(f"{rates[f'{c}/{l}']:.1%}" for c, l, _ in cells)
                     + f" | {'**PASS**' if ok else ''} |")
    else:
        passes = []
        L.append("_(no train_clean structure file yet — section skipped)_")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"conditions": conds, "separation": seps, "passes": passes}, indent=2),
        encoding="utf-8")
    print(f"[struct] wrote {out}")
    for s in seps:
        if s["separates"]:
            print(f"[struct] SEPARATES §7.2  {s['stat']} ({s['direction']}) sep {s['sep']:+.3f}")
    for p in passes:
        print(f"[struct] FULL PASS  {p['stat']} {p['direction']} thr {p['thr']:.4g}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
