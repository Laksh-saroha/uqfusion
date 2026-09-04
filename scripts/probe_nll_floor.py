"""Why NLL reads 1.9e16 on MC/ensemble, and what floor makes it mean something.

THE MECHANISM, stated before any number is chosen. `cluster_records` sets a
cluster's `sigma_ltrb` to the per-coordinate std over its members (ddof=0). Two
members that agree to the last float on an edge give **std exactly 0.0**. The
Gaussian NLL

    0.5 * log(2 pi sigma^2) + err^2 / (2 sigma^2)

is +inf there for any err != 0, and `metrics._EPS = 1e-9` converts that infinity
into a finite ~1e17 per edge. Averaged over ~25k edges that is the 1.9e16 in the
day/night slice. The number is therefore a function of the CLIP CONSTANT, not of
the model -- change `_EPS` to 1e-12 and it moves by six orders of magnitude.

The Gaussian sigma head cannot do this: it emits a strictly positive sigma (min
0.048 px on VIS). Only the sample-std arms can, which is why exactly the two
arms with a sample-std sigma blew up and the two sigma-head arms did not.

WHAT THIS SCRIPT DOES, and deliberately does not do. It measures, per arm and on
TP edges only (FP rows carry NaN err and are dropped by the metric):

  * the share of edges with sigma exactly 0, and where the mass sits below 1 px;
  * |err| ON those zero-sigma edges -- if members agree AND the box is right the
    contribution is finite, and the blow-up is carried by a few edges where the
    members agreed and were wrong together;
  * NLL and interval-ECE across a LADDER of sigma floors, so the floor is chosen
    against evidence of where each arm stabilises rather than picked to taste;
  * the same ladder for the sigma-head arms, because a floor that moves THEM
    changes a published number and that has to be visible before it is adopted.

It changes no metric and writes no cache. `write_md` refuses an existing --out.

Usage:
    python scripts/probe_nll_floor.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _ideas_common import fmt, md_table, write_md                    # noqa: E402
from uqfusion.eval.cache import load_cache                           # noqa: E402
from uqfusion.eval.matching import load_gt, match_image               # noqa: E402
from uqfusion.eval.metrics import COVERAGE_K                          # noqa: E402

#: The five arms of the day/night slice. `sigma_*` are the trained Gaussian head;
#: `mc_*` and `ens_*` carry a cluster sample std and are the ones at issue.
ARMS = [("sigma_vis", "VIS σ-head"), ("mc_vis", "VIS MC-dropout"),
        ("sigma_ir", "IR σ-head"), ("mc_ir", "IR MC-dropout"),
        ("ens_ir", "IR ensemble")]

#: Pixel floors. 0.0 reproduces today's behaviour (the 1e-9 numerical guard).
#: 0.5 px is the coordinate quantisation of the labels themselves -- no box edge
#: is resolvable below it, so a sigma under 0.5 asserts precision the GT cannot
#: support. The ladder exists so that claim is checked rather than assumed.
FLOORS = [0.0, 1e-3, 1e-2, 0.05, 0.1, 0.25, 0.5, 1.0]
_EPS = 1e-9


def nll_at(err: np.ndarray, sig: np.ndarray, floor: float) -> float:
    s = np.clip(sig, max(floor, _EPS), None)
    return float((0.5 * np.log(2 * np.pi * s ** 2) + err ** 2 / (2 * s ** 2)).mean())


def iece_at(err: np.ndarray, sig: np.ndarray, floor: float) -> float:
    s = np.clip(sig, max(floor, _EPS), None)
    a = np.abs(err)
    import math
    return float(np.mean([abs(float((a <= k * s).mean()) - math.erf(k / math.sqrt(2)))
                          for k in COVERAGE_K]))


def tp_edges(path: Path, iou_match: float = 0.5):
    """Flattened (err, sigma) over TP edges only — the rows the metric keeps."""
    recs, _ = load_cache(path)
    errs, sigs = [], []
    for rec in recs:
        if len(rec["conf"]) == 0:
            continue
        gt = load_gt(rec["image_path"], rec["image_hw"])
        m = match_image(rec, gt, iou_thr=iou_match)
        e = np.asarray(m["err_edges"], dtype=float).reshape(-1, 4)
        s = np.asarray(rec["sigma_ltrb"], dtype=float).reshape(-1, 4)
        ok = np.isfinite(e).all(axis=1)
        errs.append(e[ok])
        sigs.append(s[ok])
    return np.concatenate(errs).ravel(), np.concatenate(sigs).ravel()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cache-dir", default="runs/cache_uqslice")
    ap.add_argument("--out", default="runs/eval/nll_floor.md")
    args = ap.parse_args()

    cd = ROOT / args.cache_dir
    data = {}
    for stem, label in ARMS:
        p = cd / f"{stem}.pkl"
        if not p.is_file():
            print(f"[skip] missing {p}")
            continue
        err, sig = tp_edges(p)
        data[stem] = (label, err, sig)
        print(f"[read] {stem}: {err.size} TP edges, {np.mean(sig == 0):.4%} at sigma=0")

    if not data:
        print("[fail] no arms read")
        return 1

    shape = md_table(
        ["arm", "TP edges", "σ = 0 exactly", "σ < 0.05 px", "σ < 0.5 px",
         "median σ (px)", "median |err| on σ=0 edges", "max |err| on σ=0 edges"],
        [[label, f"{err.size:,}", f"{np.mean(sig == 0):.4%}",
          f"{np.mean(sig < 0.05):.3%}", f"{np.mean(sig < 0.5):.2%}",
          fmt(float(np.median(sig)), 4),
          fmt(float(np.median(np.abs(err[sig == 0]))), 4) if (sig == 0).any() else "—",
          fmt(float(np.max(np.abs(err[sig == 0]))), 3) if (sig == 0).any() else "—"]
         for _s, (label, err, sig) in data.items()])

    ladder = md_table(
        ["σ floor (px)"] + [lab for lab, _e, _s in data.values()],
        [[("0 (today)" if f == 0.0 else f"{f:g}")]
         + [f"{nll_at(e, s, f):.4g}" for _l, e, s in data.values()]
         for f in FLOORS])

    iece = md_table(
        ["σ floor (px)"] + [lab for lab, _e, _s in data.values()],
        [[("0 (today)" if f == 0.0 else f"{f:g}")]
         + [fmt(iece_at(e, s, f), 4) for _l, e, s in data.values()]
         for f in FLOORS])

    # How much of today's NLL is carried by the exact-zero edges alone?
    carried = []
    for _s, (label, err, sig) in data.items():
        z = sig == 0
        if not z.any():
            carried.append([label, "—", "—", "—"])
            continue
        full = nll_at(err, sig, 0.0)
        drop = nll_at(err[~z], sig[~z], 0.0)
        carried.append([label, f"{full:.4g}", f"{drop:.4g}",
                        f"{100 * (1 - drop / full):.4f}%" if full else "—"])
    carried_t = md_table(
        ["arm", "NLL as computed today", "NLL with σ=0 edges dropped",
         "share of today's NLL carried by them"], carried)

    secs = [
        "**The metric is reporting the clip constant, not the model.** "
        "`cluster_records` takes a cluster's σ to be the per-coordinate std over "
        "its members (ddof=0), so two members agreeing to the last float give "
        "σ **exactly 0**. Gaussian NLL is +∞ there for any non-zero error, and "
        "`metrics._EPS = 1e-9` renders that infinity as ~1e17 per edge. Set "
        "`_EPS` to 1e-12 instead and every MC/ensemble NLL in this project moves "
        "by six orders of magnitude without a single weight changing.",
        "",
        "The σ-head arms cannot do this — they emit a strictly positive σ — which "
        "is why exactly the two sample-std arms blew up and the two σ-head arms "
        "did not. This is a property of the estimator, not a bug in any one run.",
        "",
        "## Where the σ mass sits (TP edges only; FP rows carry NaN err and are dropped)",
        "", shape, "",
        "## NLL across a ladder of σ floors",
        "",
        "A floor is a claim about resolution: no box edge is resolvable below the "
        "quantisation of the labels, so a σ under ~0.5 px asserts precision the GT "
        "cannot support. The ladder is here so that claim is tested rather than "
        "assumed — and so the cost to the σ-head arms, whose numbers are already "
        "published, is visible before anything is adopted.",
        "", ladder, "",
        "## Interval-ECE across the same ladder",
        "",
        "`coverage_interval_ece` clips with the same `_EPS`, so it is contaminated "
        "too — but boundedly, because a coverage share cannot exceed 1. At σ=0 the "
        "nominal interval has zero width and only an exactly-zero error falls "
        "inside it, so those edges read as total under-coverage.",
        "", iece, "",
        "## How much of today's number is those edges alone",
        "", carried_t, "",
        "## What this does NOT settle",
        "",
        "1. **`nll` is a DECISION input** of `docs/prereg-uq-day-night-slice.md` "
        "(rule 4). Changing how it is computed can move a registered verdict, so "
        "the slice must be re-run as a **declared amendment** carrying both "
        "verdicts — not silently re-scored.",
        "2. **A floor is not the only defensible repair.** Bessel correction "
        "(ddof=1) does not help — it rescales a zero to a zero. An additive "
        "variance floor σ² + σ₀² with σ₀ fitted on clean data is the principled "
        "alternative and is not evaluated here.",
        "3. **The singleton convention is untouched.** Support-1 clusters take "
        "σ = box size, which is 19.7% of VIS MC and 45.1% of IR ensemble edges. "
        "That is pre-registered and disclosed, and it pushes NLL the other way.",
        "4. Nothing here says which arm is better calibrated. It says today's "
        "NLL cannot be used to ask.",
    ]
    out = write_md(ROOT / args.out, "NLL σ-floor probe", secs)
    print(f"[out] {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
