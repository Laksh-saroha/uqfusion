"""Stage 0 of `docs/prereg-night-veto-v2.md` — screen, and adopt nothing.

This script implements that document's Stage 0 literally. It selects no rule,
scores no fusion cell and writes no constant. Its only output is an AUROC per
candidate instrument and a PROCEED / ABANDON call against the bar the
registration fixed at **0.90** before any candidate was run.

THE DISCRIMINATION. Over the 1,032 night frames, positive = the five VIS-HEALTHY
cells where V1 measured the veto to be harmful, negative = the five VIS-DEGRADED
cells where V1 measured it to be helpful. `noise_s2`/clean-IR is excluded: V1 put
it in neither band (delta -0.0000).

WHY 0.90 AND NOT 0.5. The incumbent rule scores **exactly 0.5** here, because it
fires on 100% of night frames in every cell and therefore orders them not at all.
Any bar above 0.5 is "better than what ships"; 0.90 is the margin demanded of a
change deployable only alongside a full re-baseline.

ORIENTATION IS FIXED A PRIORI, not chosen from the result:

    sigma_frame   healthy = LOW  uncertainty   -> score = -sigma
    n_det         healthy = MORE detections    -> score = +n
    conf_mean     healthy = HIGH confidence    -> score = +c
    q_refit       healthy = HIGH q             -> score = +q

An instrument that lands near 0 is a good discriminator with the sign reversed.
It is reported as such and it still FAILS the bar as oriented, because flipping a
sign after seeing the number is a selection step and this is a screening stage.

TWO CONVENTIONS the registration does not cover, disclosed rather than buried:

  * A night frame with **no detections** has no mean sigma and no mean confidence.
    Both are set to the worst observed value -- "the detector returned nothing" is
    a statement about health, not missing data.
  * The five positive cells share one VIS condition (`clean`), so they carry
    IDENTICAL instrument values; `blur_s3` appears twice among the negatives. That
    is what "labelled by cell" means when the instruments cannot see the IR
    stream. The pooled AUROC is the registered number; a de-duplicated variant
    over unique VIS conditions is reported as a sensitivity and is NOT the
    decision.

Usage:
    python scripts/stage0_night_veto_v2.py
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _ideas_common import fmt, md_table, write_md                     # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context                # noqa: E402
from uqfusion.uq.reliability import per_box_uncertainty               # noqa: E402

PREREG = "docs/prereg-night-veto-v2.md"
BAR = 0.90                      # registered before any candidate was run
STRUCT_DIR = ROOT / "runs/derived/structure"
FIT_RUNS = ("pohang00", "pohang02", "pohang03")
NIGHT_RUN = "pohang01"

#: (VIS condition, IR condition, label). 1 = VIS healthy (V1: veto harmful),
#: 0 = VIS degraded (V1: veto helpful). `noise_s2`/None is deliberately absent.
CELLS = [("clean", None, 1), ("clean", "glare_s2", 1), ("clean", "blur_s2", 1),
         ("clean", "noise_s2", 1), ("clean", "fog_s2", 1),
         ("blur_s3", None, 0), ("rain_s2", None, 0), ("fog", None, 0),
         ("lowlight", "glare_s2", 0), ("blur_s3", "glare_s2", 0)]

DRAWS = ["runs/cache_nv_draw1", "runs/cache_nv_draw901",
         "runs/cache_nv_draw902", "runs/cache_nv_draw903"]

INSTRUMENTS = ("sigma_frame", "n_det", "conf_mean", "q_refit")


def committed(rel: str) -> bool:
    try:
        tracked = subprocess.run(["git", "ls-files", "--error-unmatch", rel],
                                 cwd=ROOT, capture_output=True).returncode == 0
        dirty = subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel],
                               cwd=ROOT, capture_output=True).returncode != 0
        return tracked and not dirty
    except OSError:
        return False


def auroc(score: np.ndarray, y: np.ndarray) -> float:
    """Rank-based AUROC with ties handled by mid-rank. No sklearn dependency."""
    ok = np.isfinite(score)
    s, yy = score[ok], y[ok]
    if len(np.unique(yy)) < 2:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    ranks[order] = np.arange(1, len(s) + 1, dtype=float)
    ss = s[order]
    i = 0                                     # average ranks within tie groups
    while i < len(ss):
        j = i
        while j + 1 < len(ss) and ss[j + 1] == ss[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = ranks[order[i:j + 1]].mean()
        i = j + 1
    n1 = float((yy == 1).sum())
    n0 = float((yy == 0).sum())
    return float((ranks[yy == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def refit_vis_health() -> dict:
    """`vis_health` with clean `pohang01` pooled into its reference.

    One line different from `fit_structure_gate.py`: the fit set is FIT_RUNS plus
    the night run instead of FIT_RUNS alone. Same input file, same helper, same
    bound rule -- so a difference in the result is the pooling and nothing else.
    """
    sys.path.insert(0, str(ROOT / "scripts"))
    from fit_structure_gate import _ir_health_matrix, IR_HEALTH_KEYS, IR_LOG_KEYS

    tf = json.loads((STRUCT_DIR / "gauss_vis_train_clean.json").read_text(
        encoding="utf-8"))["frames"]
    keep = [f for f in tf if f["run"] in FIT_RUNS + (NIGHT_RUN,)]
    X = _ir_health_matrix(keep)
    mu, sd = X.mean(0), X.std(0) + 1e-9
    Z = (X - mu) / sd
    prec = np.linalg.inv(np.cov(Z.T) + 1e-6 * np.eye(X.shape[1]))
    d2 = np.einsum("ij,jk,ik->i", Z, prec, Z)
    return {"keys": list(IR_HEALTH_KEYS), "log1p_keys": list(IR_LOG_KEYS),
            "mean": mu, "std": sd, "precision": prec, "bound": float(d2.max()),
            "fit_n": len(keep),
            "n_night_pooled": sum(1 for f in keep if f["run"] == NIGHT_RUN)}


def q_for(vh: dict, cond: str) -> np.ndarray:
    fr = json.loads((STRUCT_DIR / f"gauss_vis_paired_{cond}.json").read_text(
        encoding="utf-8"))["frames"]
    X = np.stack([[f[k] for k in vh["keys"]] for f in fr]).astype(float)
    for j, k in enumerate(vh["keys"]):
        if k in vh["log1p_keys"]:
            X[:, j] = np.log1p(np.clip(X[:, j], 0, None))
    Z = (X - vh["mean"]) / vh["std"]
    d2 = np.einsum("ij,jk,ik->i", Z, vh["precision"], Z)
    return np.clip(vh["bound"] / np.maximum(d2, 1e-12), 0, 1)


def instruments_for(ctx, cond: str, night: np.ndarray, vh: dict) -> dict:
    recs = ctx.vis_by_cond[cond]
    sig, nd, cm = [], [], []
    for i in night:
        r = recs[i]
        n = len(r["conf"])
        nd.append(n)
        if n == 0:
            sig.append(np.nan)
            cm.append(np.nan)
            continue
        u = per_box_uncertainty(np.asarray(r["sigma_ltrb"]).reshape(-1, 4),
                                np.asarray(r["boxes_xyxy"]).reshape(-1, 4))
        sig.append(float(np.mean(u)))
        cm.append(float(np.mean(r["conf"])))
    return {"sigma_frame": np.asarray(sig, dtype=float),
            "n_det": np.asarray(nd, dtype=float),
            "conf_mean": np.asarray(cm, dtype=float),
            "q_refit": q_for(vh, cond)[night]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="runs/eval/night_veto_v2_stage0.md")
    args = ap.parse_args()

    if not committed(PREREG):
        print(f"[gate] {PREREG} is not committed clean -- refusing to run.")
        return 1
    print(f"[gate] prereg committed; bar AUROC >= {BAR}")

    vh = refit_vis_health()
    print(f"[refit] vis_health with night pooled: n={vh['fit_n']} "
          f"({vh['n_night_pooled']} night), bound {vh['bound']:.1f}")

    conds = sorted({c for c, _i, _y in CELLS})
    per_draw = {k: [] for k in INSTRUMENTS}
    dedup = {k: [] for k in INSTRUMENTS}
    shape_rows, winner_scores = None, None
    for d in DRAWS:
        cd = ROOT / d
        if not cd.is_dir():
            print(f"[fail] missing {cd} -- Stage 0 needs V1's four draws")
            return 1
        ctxs = {ic: load_context(preset="crossmodal26m", cache_dir=str(cd),
                                 conditions=tuple(conds), ir_condition=ic,
                                 verbose=False)
                for ic in sorted({i for _c, i, _y in CELLS}, key=lambda x: (x is None, x))}
        c0 = ctxs[None]
        night = np.flatnonzero(np.isin(c0.runs, NIGHT_RUNS))
        cache = {c: instruments_for(c0, c, night, vh) for c in conds}

        pooled = {k: ([], []) for k in INSTRUMENTS}
        for vc, _ic, y in CELLS:
            for k in INSTRUMENTS:
                pooled[k][0].append(cache[vc][k])
                pooled[k][1].append(np.full(len(night), y))
        for k in INSTRUMENTS:
            v = np.concatenate(pooled[k][0])
            y = np.concatenate(pooled[k][1])
            worst = np.nanmax(v) if k == "sigma_frame" else np.nanmin(v)
            v = np.where(np.isfinite(v), v, worst)
            s = -v if k == "sigma_frame" else v          # orientation fixed a priori
            per_draw[k].append(auroc(s, y))

            # sensitivity: unique VIS conditions only, one vote each
            uv, uy = [], []
            for vc in conds:
                lab = next(yy for c2, _i2, yy in CELLS if c2 == vc)
                x = cache[vc][k]
                x = np.where(np.isfinite(x), x, worst)
                uv.append(-x if k == "sigma_frame" else x)
                uy.append(np.full(len(night), lab))
            dedup[k].append(auroc(np.concatenate(uv), np.concatenate(uy)))

        if shape_rows is None:
            shape_rows = [[vc, f"{np.nanmean(cache[vc]['sigma_frame']):.4f}",
                           f"{np.nanmean(cache[vc]['n_det']):.1f}",
                           f"{np.nanmean(cache[vc]['conf_mean']):.4f}",
                           f"{np.nanmean(cache[vc]['q_refit']):.4f}",
                           f"{np.mean(cache[vc]['n_det'] == 0):.1%}"]
                          for vc in conds]
        print(f"[draw] {cd.name}: " + "  ".join(
            f"{k} {per_draw[k][-1]:.4f}" for k in INSTRUMENTS), flush=True)

    auc = {k: float(np.mean(per_draw[k])) for k in INSTRUMENTS}
    sd = {k: float(np.std(per_draw[k], ddof=1)) for k in INSTRUMENTS}
    passing = [k for k in INSTRUMENTS if auc[k] >= BAR]
    if passing:
        winner = max(passing, key=lambda k: auc[k])
        call = f"PROCEED with `{winner}`"
    else:
        winner = None
        call = "**ABANDON** — no instrument clears the bar; the veto stands"

    reversed_ = [k for k in INSTRUMENTS if auc[k] <= 1 - BAR]

    secs = [
        f"**Pre-registration:** [`{PREREG}`]({PREREG}), committed before any "
        f"candidate was run. Bar **AUROC >= {BAR}**, fixed there and not moved. "
        f"4 paired draws, {len(CELLS)} cells, 1,032 night frames each "
        f"({len(CELLS) * 1032:,} pooled points per instrument).",
        "",
        f"## Stage 0 call — {call}",
        "",
        md_table(["instrument", "orientation (a priori)", "AUROC (draw-avg)",
                  "sd across draws", "vs bar"],
                 [[f"`{k}`",
                   {"sigma_frame": "healthy = LOW sigma", "n_det": "healthy = MORE dets",
                    "conf_mean": "healthy = HIGH conf", "q_refit": "healthy = HIGH q"}[k],
                   fmt(auc[k], 4), fmt(sd[k], 4),
                   "**PASS**" if auc[k] >= BAR else f"fail ({auc[k] - BAR:+.4f})"]
                  for k in INSTRUMENTS]),
        "",
        ("**No candidate clears 0.90.** The registration names abandonment as a "
         "live outcome and it is the outcome. The night veto stands, V2 is not "
         "advanced to Stage 1, and the bar is not revisited — lowering it after "
         "seeing these numbers is precisely the move the document exists to "
         "prevent." if winner is None else
         f"`{winner}` clears the bar. Stage 1 runs on it alone: no ensembling of "
         f"candidates, and the threshold is the Youden point of its ROC, read off "
         f"the curve rather than tuned against mAP."),
        "",
        ("" if not reversed_ else
         "**Reversed discriminators.** " + ", ".join(f"`{k}` ({auc[k]:.4f})" for k in reversed_)
         + " land below " + f"{1 - BAR:.2f}" + ", i.e. they separate the classes well "
         "with the sign opposite to the one declared a priori. They still FAIL: "
         "flipping a sign after seeing the number is a selection step, and this is "
         "a screening stage. A future registration may declare the opposite "
         "orientation up front and re-ask."),
        "",
        "## What the instruments look like on night frames",
        "",
        md_table(["VIS condition", "mean sigma_frame", "mean n_det",
                  "mean conf", "mean q_refit", "frames with 0 detections"],
                 shape_rows),
        "",
        "## Sensitivity — de-duplicated, and NOT the decision",
        "",
        "The five positive cells share one VIS condition, so they carry identical "
        "instrument values; `blur_s3` appears twice among the negatives. That is "
        "what \"labelled by cell\" means for instruments that cannot see the IR "
        "stream. Collapsing to unique VIS conditions, one vote each:",
        "",
        md_table(["instrument", "AUROC (registered, pooled)", "AUROC (de-duplicated)"],
                 [[f"`{k}`", fmt(auc[k], 4), fmt(float(np.mean(dedup[k])), 4)]
                  for k in INSTRUMENTS]),
        "",
        "## The refitted health axis",
        "",
        f"`vis_health` refitted with clean `{NIGHT_RUN}` pooled into its reference: "
        f"n = {vh['fit_n']} frames of which **{vh['n_night_pooled']} are night**, "
        f"novelty bound {vh['bound']:.1f}. One line differs from "
        "`fit_structure_gate.py` — the fit set is FIT_RUNS plus the night run "
        "instead of FIT_RUNS alone — so any difference is the pooling and nothing "
        "else. It is fitted here for screening only and **is not written to** "
        "`runs/eval/structure_constants.json`.",
        "",
        "## What this stage did not do",
        "",
        "1. It adopted nothing, scored no fusion cell and wrote no constant.",
        "2. The labels come from V1's cell-level result on **these same 1,032 "
        "night frames**. `pohang01` is the only night run in the dataset and no "
        "split can create a second one. Cell-level labels, a bar fixed in advance "
        "and a registered abandonment outcome are the only defences available.",
        "3. The negative class is four **synthetic** VIS corruptions. An "
        "instrument that separates synthetic fog from clean night need not "
        "separate real ones.",
    ]
    out = write_md(ROOT / args.out, "Night veto V2 — Stage 0 screening", secs)
    js = out.with_suffix(".json")
    js.write_text(json.dumps({
        "bar": BAR, "auroc": auc, "sd": sd, "per_draw": per_draw,
        "dedup": {k: float(np.mean(v)) for k, v in dedup.items()},
        "winner": winner, "call": "PROCEED" if winner else "ABANDON",
    }, indent=1), encoding="utf-8")
    print(f"\n[stage0] {call}")
    print(f"[out] {out}\n[out] {js}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
