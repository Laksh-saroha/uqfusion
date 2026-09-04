"""Shared helpers for the 2026-09-01 architecture-ideas probes.

Every script under `docs/architecture-ideas-2026-09-01.md` needs the same four
things: the day/night run split, an AP over a frame subset, the oracle
re-ranking ceiling, and a leave-one-run-out fold generator. They live here so
the eight probes cannot drift apart on the definition of any of them --
`probe_oracle_headroom.py` and `fit_rerank.py` disagreeing about what "oracle"
means would make their two tables silently incomparable.

Nothing here writes to disk. Nothing here imports torch.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import NL, _ap_from_sorted, ap_from_parts, frame_parts  # noqa: E402
from uqfusion.eval.cache import load_cache                                           # noqa: E402
from uqfusion.eval.matching import load_gt                                           # noqa: E402

#: Mirrors `uqfusion.eval.ctx`; duplicated rather than imported so a probe can
#: run against a VIS-only substrate that has no paired manifest to build a ctx from.
NIGHT_RUNS = ("pohang01",)
DAY_RUNS = ("pohang00", "pohang02", "pohang03", "pohang04")
IOU_LEVELS = np.round(np.arange(0.5, 1.0, 0.05), 2)


def run_of(image_path) -> str:
    """`pohang00` from `.../pohang00/pohang00_L_006767.png`. Basename, not parent,
    so it works for both the VIS (`_L_`) and IR naming schemes."""
    return os.path.basename(str(image_path)).split("_")[0]


def runs_of(records) -> np.ndarray:
    return np.asarray([run_of(r["image_path"]) for r in records])


def load_records(path):
    recs, meta = load_cache(path)
    return recs, meta


def gts_for(records) -> list[dict]:
    return [load_gt(r["image_path"], r["image_hw"]) for r in records]


def day_night(records) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(runs, day_idx, night_idx)."""
    runs = runs_of(records)
    night = np.isin(runs, NIGHT_RUNS)
    return runs, np.flatnonzero(~night), np.flatnonzero(night)


def loro_folds(runs: np.ndarray, idx: np.ndarray) -> list[tuple[str, np.ndarray, np.ndarray]]:
    """Leave-one-run-out over the runs present in `idx`.

    Split by RUN and never by frame: consecutive frames of one canal transit are
    near-duplicates, so a frame split leaks the answer across the boundary. This
    is the C8 discipline, generalised from one fixed tune/test pair to a fold per
    run so every run serves as both.
    """
    present = [r for r in DAY_RUNS if (runs[idx] == r).any()]
    folds = []
    for held in present:
        te = idx[runs[idx] == held]
        tr = idx[runs[idx] != held]
        if len(te) and len(tr):
            folds.append((held, tr, te))
    return folds


def subsample(records, limit):
    """Evenly spaced subsample, NOT a head slice.

    `records[:limit]` takes the first N frames, and the caches are ordered by run,
    so every small `--limit` collapsed onto pohang00 alone -- which silently left
    `loro_folds` with zero folds and turned a preflight into a test of nothing.
    (It also produced an out-of-fold R^2 of exactly 1.0000 in
    `probe_sigma_residual.py`, from a division with an empty numerator.) Even
    spacing keeps every run represented.
    """
    if not limit or limit >= len(records):
        return records
    idx = np.unique(np.linspace(0, len(records) - 1, int(limit)).astype(int))
    return [records[i] for i in idx]


def lift(flag: np.ndarray, tp: np.ndarray, conf: np.ndarray | None = None,
         n_bins: int = 5) -> tuple[float, float]:
    """(raw lift, confidence-matched lift).

    `lift = P(TP | signal) / P(TP | no signal)` is the F1 screen, and on its own it
    is easy to fool: any signal correlated with confidence inherits confidence's
    own 4.8x. The two-checkpoint probe measured a raw 211x this way, almost all of
    which was "this is a high-confidence box".

    The matched version bins by confidence quantile and averages the within-bin
    lifts, so it answers the question a re-scorer actually faces: among boxes the
    detector already ranks equally, does this signal separate them?
    """
    if flag.all() or not flag.any():
        return float("nan"), float("nan")
    raw = tp[flag].mean() / max(tp[~flag].mean(), 1e-9)
    if conf is None:
        return float(raw), float("nan")
    qs = np.quantile(conf, np.linspace(0, 1, n_bins + 1))
    qs[-1] += 1e-9
    vals, wts = [], []
    for k in range(n_bins):
        m = (conf >= qs[k]) & (conf < qs[k + 1])
        f, nf = m & flag, m & ~flag
        if f.sum() < 20 or nf.sum() < 20 or tp[nf].mean() <= 0:
            continue
        vals.append(tp[f].mean() / tp[nf].mean())
        wts.append(m.sum())
    if not vals:
        return float(raw), float("nan")
    return float(raw), float(np.average(vals, weights=wts))


def ap_of(records, gts, sel=None) -> dict:
    return ap_from_parts(frame_parts(records, gts), sel=sel)


def pooled(parts, sel):
    """Concatenate (tp, conf, cls) over a frame subset, plus per-class GT counts."""
    ngt: dict[int, int] = {}
    tps, confs, clss, frame = [], [], [], []
    for i in sel:
        p = parts[i]
        for c, k in zip(p["gt_cls"], p["gt_cnt"]):
            ngt[int(c)] = ngt.get(int(c), 0) + int(k)
        if len(p["conf"]):
            tps.append(p["tp"])
            confs.append(p["conf"])
            clss.append(p["cls"])
            frame.append(np.full(len(p["conf"]), i))
    if not tps:
        return (np.zeros((0, NL), bool), np.zeros(0), np.zeros(0, int), np.zeros(0, int), ngt)
    return (np.concatenate(tps), np.concatenate(confs), np.concatenate(clss),
            np.concatenate(frame), ngt)


def ap_from_scores(parts, sel, scores_by_frame) -> float:
    """mAP@50-95 when frame i's detections are ranked by `scores_by_frame[i]`
    instead of by confidence. Boxes and TP flags are untouched, so this isolates
    the ORDERING -- which is the only thing any re-scoring lever can change."""
    tp, _conf, cls, frame, ngt = pooled(parts, sel)
    if not ngt or not len(cls):
        return 0.0
    s = np.concatenate([np.asarray(scores_by_frame[i], float) for i in sel
                        if len(parts[i]["conf"])])
    rows = []
    for c in sorted(ngt):
        m = cls == c
        rows.append(_ap_from_sorted(tp[m][np.argsort(-s[m])], ngt[c]))
    return float(np.stack(rows).mean())


def oracle_curves(parts, sel) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """(actual, oracle) AP curves per IoU level, averaged over classes.

    ORACLE = sort each class's pooled detections by whether they are a TP AT THAT
    LEVEL, confidence breaking ties. It is the upper bound on every score-based
    intervention at once -- the support term, the capability weights, calibration,
    sigma-in-score, and any future re-scorer -- because all of them can only
    permute this same list. It is NOT an upper bound on anything that changes the
    boxes themselves.
    """
    tp, conf, cls, _frame, ngt = pooled(parts, sel)
    if not ngt or not len(cls):
        z = np.zeros(NL)
        return z, z, {}, {}
    A, O, per_class, recall = [], [], {}, {}
    for c in sorted(ngt):
        m = cls == c
        t, cf = tp[m], conf[m]
        a = _ap_from_sorted(t[np.argsort(-cf)], ngt[c])
        o = np.empty(NL)
        for k in range(NL):
            # lexsort's last key is primary: TP first, then confidence within it.
            o[k] = _ap_from_sorted(t[np.lexsort((-cf, -t[:, k].astype(float)))], ngt[c])[k]
        A.append(a)
        O.append(o)
        per_class[int(c)] = {"actual": a, "oracle": o, "n_gt": ngt[c], "n_pred": int(m.sum())}
        recall[int(c)] = t.sum(0) / max(ngt[c], 1)
    return np.stack(A).mean(0), np.stack(O).mean(0), per_class, recall


def md_table(header: list[str], rows: list[list], align: str | None = None) -> str:
    if align is None:
        align = "l" + "r" * (len(header) - 1)
    sep = {"l": "---", "r": "---:", "c": ":---:"}
    out = ["| " + " | ".join(str(h) for h in header) + " |",
           "|" + "|".join(sep[a] for a in align) + "|"]
    for r in rows:
        out.append("| " + " | ".join(str(x) for x in r) + " |")
    return "\n".join(out)


def write_md(path, title: str, sections: list[str]) -> Path:
    p = ROOT / path if not Path(path).is_absolute() else Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        raise FileExistsError(
            f"{p} exists -- the project rule is that every new result goes to a NEW "
            f"filename so nothing published is overwritten. Pass a different --out.")
    p.write_text("# " + title + "\n\n" + "\n\n".join(sections) + "\n", encoding="utf-8")
    print(f"[out] {p}")
    return p


def fmt(x, nd=4) -> str:
    return f"{x:.{nd}f}" if np.isfinite(x) else "--"


def sgn(x, nd=4) -> str:
    return f"{x:+.{nd}f}" if np.isfinite(x) else "--"
