"""R-D1 / F09 — does predicted uncertainty improve the fusion?

Runs the pre-registered ablation in `docs/prereg-uq-mechanism-ablation.md`
(committed `a8f087c`, amendment 1 `f713961`). **Read the pre-registration before
reading this file**; the decision rule is fixed there and this script only executes
it.

The design in one line: hold predictions and fusion options FIXED, vary only the
`sigma_ltrb` array handed to the fusion, and compare real sigma against sigma of the
same shape attached to the wrong boxes.

Why the shuffled control and not the constant one: inverse-variance averaging with
all sigmas equal reduces analytically to the score-weighted mean stock WBF already
computes, so a tie against constant sigma is arithmetic, not evidence. Shuffling
preserves the distribution, the scale and the numerical conditioning and destroys
only the property under test.

CPU only, no retraining, no new prediction cache: the arms read
`runs/cache_m/gauss_{vis,ir}_paired_*.pkl` exactly as they are.

Usage:
    python scripts/ablate_uq_mechanism.py --out runs/eval/uq_mechanism_ablation.md
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from _ideas_common import write_md
from uqfusion.eval.apmetrics import frame_parts
from uqfusion.eval.blockboot import block_bootstrap_delta
from uqfusion.eval.ctx import load_context, run_systems
from uqfusion.eval.identity import system_identity

# --- everything the pre-registration pins, in one block so it can be checked ------
ALPHA = 1.0             # sigma_score_alpha for the score arms; NOT tuned (prereg §3)
BLOCK_LEN = 20          # amendment 1: largest defensible block (shortest run / 5 = 23)
N_BOOT = 1000
SEED = 0
FLOORS = (0.0014, 0.0031, 0.0060, 0.0100)
ADOPTED_FLOOR = 0.0060  # amendment 1: 0.0031 x 1.95, the dependence correction
MIN_CONDITIONS = 3      # of 4, prereg §5

ARMS = {
    "S0": ("shipped (no sigma anywhere)", None, {}),
    "S1": ("real sigma, coordinates", "real", {"sigma_weighted": True}),
    "S2": ("constant sigma, coordinates", "const", {"sigma_weighted": True}),
    "S3": ("shuffled within frame, coordinates", "shuf_frame", {"sigma_weighted": True}),
    "S4": ("shuffled across cache, coordinates", "shuf_cache", {"sigma_weighted": True}),
    "S5": ("real sigma, score", "real", {"sigma_score_alpha": ALPHA}),
    "S6": ("constant sigma, score", "const", {"sigma_score_alpha": ALPHA}),
    "S7": ("shuffled within frame, score", "shuf_frame", {"sigma_score_alpha": ALPHA}),
}

# The two comparisons the verdict is taken on, and the two that inform it.
PRIMARY = (("S1", "S3", "coordinate path"), ("S5", "S7", "score path"))
SECONDARY = (("S1", "S4", "coordinate: per-box vs between-frame signal"),
             ("S1", "S2", "coordinate: real vs constant (expected inert)"),
             ("S5", "S6", "score: real vs constant"),
             ("S1", "S0", "coordinate: does enabling the mechanism at all move it"))


def _sigma_stack(records: list[dict]) -> tuple[np.ndarray, list[int]]:
    """All sigma rows across a cache, plus the per-record row counts."""
    rows, counts = [], []
    for r in records:
        s = np.asarray(r["sigma_ltrb"], dtype=np.float64).reshape(-1, 4)
        rows.append(s)
        counts.append(len(s))
    return (np.concatenate(rows) if rows else np.zeros((0, 4))), counts


def substitute(records: list[dict], mode: str | None, rng: np.random.Generator) -> tuple[list[dict], dict]:
    """A copy of `records` with `sigma_ltrb` replaced. Nothing else is touched.

    Returns the records and a small audit dict, because two of these modes can be
    partly or wholly inert and a silent no-op would be indistinguishable from a null
    result -- which is the failure mode this whole exercise exists to avoid.
    """
    if mode is None or mode == "real":
        return records, {"mode": mode or "none", "n_rows": _sigma_stack(records)[0].shape[0],
                         "n_permuted": 0, "note": "unchanged"}

    stack, counts = _sigma_stack(records)
    out = []
    if mode == "const":
        med = np.median(stack, axis=0) if len(stack) else np.ones(4)
        for r, n in zip(records, counts):
            out.append({**r, "sigma_ltrb": np.tile(med, (n, 1))})
        return out, {"mode": mode, "n_rows": int(stack.shape[0]), "n_permuted": 0,
                     "median_sigma": [float(x) for x in med]}

    if mode == "shuf_cache":
        perm = rng.permutation(stack.shape[0])
        shuffled = stack[perm]
        moved = int((perm != np.arange(stack.shape[0])).sum())
        i = 0
        for r, n in zip(records, counts):
            out.append({**r, "sigma_ltrb": shuffled[i:i + n]})
            i += n
        return out, {"mode": mode, "n_rows": int(stack.shape[0]), "n_permuted": moved,
                     "note": "rows permuted across the whole cache"}

    if mode == "shuf_frame":
        # A frame with 0 or 1 detections cannot be permuted. That share is REPORTED
        # rather than assumed small: if most frames are singletons, this control is
        # weak and the reader has to be able to see that.
        moved = single = 0
        for r, n in zip(records, counts):
            s = np.asarray(r["sigma_ltrb"], dtype=np.float64).reshape(-1, 4)
            if n < 2:
                single += n
                out.append({**r, "sigma_ltrb": s})
                continue
            perm = rng.permutation(n)
            moved += int((perm != np.arange(n)).sum())
            out.append({**r, "sigma_ltrb": s[perm]})
        return out, {"mode": mode, "n_rows": int(stack.shape[0]), "n_permuted": moved,
                     "n_unpermutable_singletons": single,
                     "note": "rows permuted within each frame"}

    raise ValueError(f"unknown sigma mode {mode!r}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/uq_mechanism_ablation.md")
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--preset", default="crossmodal")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()

    t0 = time.time()
    ctx = load_context(preset=args.preset, cache_dir=args.cache_dir, verbose=False)
    conditions = list(ctx.vis_by_cond)
    print(f"[abl] preset {args.preset}  conditions {conditions}")

    # Reproduce the finding this ablation exists because of, in the run itself.
    inert = {
        "mu_d_vis": float(ctx.c_vis.mu_d), "lam_vis": float(ctx.c_vis.lam),
        "mu_d_ir": float(ctx.c_ir.mu_d), "lam_ir": float(ctx.c_ir.lam),
        "sigma_score_alpha_default": float(ctx.sigma_score_alpha),
    }
    print(f"[abl] shipped soft terms: {inert}")

    ap_rows: dict[tuple[str, str], float] = {}
    parts: dict[tuple[str, str], list] = {}
    paths: dict[str, list] = {}
    audits: dict[str, dict] = {}

    for cond in conditions:
        vis = ctx.vis_by_cond[cond]
        ir = ctx.ir_clean
        paths[cond] = [r["image_path"] for r in vis]
        for arm, (label, mode, opts) in ARMS.items():
            rng = np.random.default_rng(SEED)
            v2, av = substitute(vis, mode, rng)
            i2, ai = substitute(ir, mode, rng)
            audits[f"{cond}/{arm}/vis"] = av
            audits[f"{cond}/{arm}/ir"] = ai
            out = run_systems(ctx, cond, vis_records=v2, ir_records=i2, **opts)
            fused = out["fused_gated"]
            gts = out["gts"]
            ap_rows[(cond, arm)] = float(out["gated_fusion"]["map50_95"])
            parts[(cond, arm)] = frame_parts(fused, gts)
            print(f"[abl] {cond:9s} {arm} {label:38s} AP {ap_rows[(cond, arm)]:.6f} "
                  f"({time.time() - t0:.0f}s)")

    # ---------------------------------------------------------------- comparisons
    def compare(a: str, b: str) -> list[dict]:
        rows = []
        for cond in conditions:
            r = block_bootstrap_delta(parts[(cond, a)], parts[(cond, b)], paths[cond],
                                      BLOCK_LEN, n_boot=args.boot, seed=SEED)
            rows.append({"cond": cond, "delta": r["delta"], "se": r["se"],
                         "ci_lo": r["ci_lo"], "ci_hi": r["ci_hi"],
                         "spans_zero": r["spans_zero"]})
            print(f"[abl] {a}-{b} {cond:9s} delta {r['delta']:+.6f} "
                  f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}] "
                  f"{'spans zero' if r['spans_zero'] else 'excludes zero'}")
        return rows

    results = {f"{a}-{b}": compare(a, b) for a, b, _ in PRIMARY + SECONDARY}

    # ------------------------------------------------------------------- verdict
    def verdict(key: str) -> tuple[str, dict]:
        rows = results[key]
        counts = {f: sum(1 for r in rows if r["delta"] >= f and not r["spans_zero"])
                  for f in FLOORS}
        n = counts[ADOPTED_FLOOR]
        worse = sum(1 for r in rows if r["delta"] <= -ADOPTED_FLOOR and not r["spans_zero"])
        if n >= MIN_CONDITIONS:
            v = "POSITIVE"
        elif worse >= MIN_CONDITIONS:
            v = "NEGATIVE"
        else:
            v = "NULL"
        return v, counts

    verdicts = {f"{a}-{b}": verdict(f"{a}-{b}") for a, b, _ in PRIMARY}

    # ---------------------------------------------------------------- the report
    L = []
    L.append("Pre-registered in [`docs/prereg-uq-mechanism-ablation.md`]"
             "(../../docs/prereg-uq-mechanism-ablation.md) (`a8f087c`, amendment 1 "
             "`f713961`) **before this ran**. The decision rule below was fixed there; "
             "this report only applies it.\n")
    L.append("## The finding this exists because of\n")
    L.append("Read off the resolved context at run time, not quoted from a document:\n")
    L.append("| term | value |\n|---|---|")
    for k, v in inert.items():
        L.append(f"| `{k}` | `{v!r}` |")
    L.append("\nWith `mu_d` at 1e9 and `lam` at 0, the Mahalanobis soft weight is inert and "
             "the fusion weight is a constant. No predicted uncertainty reaches the fusion "
             "decision in the shipped preset.\n")

    L.append("## Arms\n")
    L.append("| arm | what | AP " + " | AP ".join(conditions) + " |")
    L.append("|---|---|" + "---:|" * len(conditions))
    for arm, (label, _, _) in ARMS.items():
        cells = " | ".join(f"{ap_rows[(c, arm)]:.6f}" for c in conditions)
        L.append(f"| **{arm}** | {label} | {cells} |")

    L.append("\n## Primary comparisons — the verdict rests on these\n")
    for a, b, why in PRIMARY:
        key = f"{a}-{b}"
        v, counts = verdicts[key]
        L.append(f"\n### {a} − {b} · {why} · **{v}**\n")
        L.append("| condition | delta | se | 95% CI (block L=20) | excludes zero |")
        L.append("|---|---:|---:|---|---|")
        for r in results[key]:
            L.append(f"| {r['cond']} | {r['delta']:+.6f} | {r['se']:.6f} | "
                     f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}] | "
                     f"{'no' if r['spans_zero'] else '**yes**'} |")
        L.append("\nConditions meeting both criteria, by floor: "
                 + ", ".join(f"**{f:.4f}: {counts[f]}/4**" if f == ADOPTED_FLOOR
                             else f"{f:.4f}: {counts[f]}/4" for f in FLOORS)
                 + f" — the verdict is taken at {ADOPTED_FLOOR:.4f} "
                   f"(needs {MIN_CONDITIONS}/4).")

    L.append("\n## Secondary comparisons — context, not verdict\n")
    for a, b, why in SECONDARY:
        key = f"{a}-{b}"
        L.append(f"\n**{a} − {b}** · {why}\n")
        L.append("| condition | delta | 95% CI | excludes zero |")
        L.append("|---|---:|---|---|")
        for r in results[key]:
            L.append(f"| {r['cond']} | {r['delta']:+.6f} | "
                     f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}] | "
                     f"{'no' if r['spans_zero'] else '**yes**'} |")

    L.append("\n## Was the control actually applied?\n")
    L.append("A permutation that silently did nothing would be indistinguishable from a "
             "null result, so the audit is published rather than assumed.\n")
    L.append("| arm/stream | mode | sigma rows | rows moved | singletons |")
    L.append("|---|---|---:|---:|---:|")
    for k in sorted(audits):
        if not k.startswith(conditions[0] + "/"):
            continue
        a = audits[k]
        L.append(f"| {k.split('/', 1)[1]} | {a['mode']} | {a['n_rows']} | "
                 f"{a['n_permuted']} | {a.get('n_unpermutable_singletons', 0)} |")

    write_md(Path(args.out), "R-D1 - does predicted uncertainty improve the fusion?",
             [chr(10).join(L)],
             identity=system_identity(ctx, alpha=ALPHA, block_len=BLOCK_LEN,
                                      n_boot=args.boot, seed=SEED,
                                      adopted_floor=ADOPTED_FLOOR,
                                      cache_dir=args.cache_dir))
    print(f"\n[abl] verdicts: " + "; ".join(f"{k} {v[0]}" for k, v in verdicts.items()))
    print(f"[abl] wrote {args.out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
