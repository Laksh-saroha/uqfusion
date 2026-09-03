"""Day/night slice of the UQ calibration table -- the U1 pre-registration.

Every threshold, band and rule here is read off `docs/prereg-uq-day-night-slice.md`
(committed a4f9364, before any cache was built). Nothing is trained and no label is
touched: this re-scores cached predictions on two partitions of one inference pass.

The question. The "UQ arms need no retraining" argument is borrowed from Phase 1,
where a uniform night handicap cancelled out of an mAP ranking. Per D31 the UQ arms
are NOT ranked on mAP -- the comparison is carried by d-ECE, NLL, AUSE and AURC, and
those are scored against a reality the emptied night labels misstate. The paired val
list is 46.2% pohang01, so nearly half of every pooled number sits on that reality.

The statistic, per decision metric M and arm a:
    D(a)      = M_pooled(a) - M_day(a)          the night pull on that arm
    spread(M) = max_a D(a) - min_a D(a)         how DIFFERENTIALLY night moves them
    sep(M)    = max_a M_day(a) - min_a M_day(a) the separation being reported
    r(M)      = spread(M) / sep(M)
A uniform handicap -- the borrowed Phase 1 condition -- gives spread = 0 and r = 0.

Usage:
    python scripts/slice_uq_day_night.py --out runs/eval/uq_day_night_slice.md
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import ROOT, fmt, md_table, sgn, write_md  # noqa: E402

from uqfusion.eval.cache import load_cache  # noqa: E402
from uqfusion.eval.matching import load_gt, map50_95, match_image  # noqa: E402
from uqfusion.eval.metrics import (  # noqa: E402
    coverage_interval_ece,
    d_ece,
    gaussian_nll,
    sparsification,
)
from uqfusion.uq.reliability import per_box_uncertainty  # noqa: E402

# ----------------------------------------------------------------- prereg constants
NIGHT_RUN = "pohang01"                       # pohang01 is entirely night
DECISION = ("d_ece", "nll", "interval_ece", "ause", "aurc")   # rule 4; mAP is NOT one
BAND_CLEAN, BAND_CONTAM = 0.25, 1.0          # rule 8
FLOOR_REL = 0.02                             # rule 7, relative arm of the floor
N_BOOT, BOOT_SEED = 2000, 0                  # rule 6
SIGMA_KEY = "sigma_ltrb"                     # rule 3

CACHE = ROOT / "runs/cache_uqslice"
ARMS = {
    "VIS": {"sigma-head": "sigma_vis.pkl", "MC-Dropout": "mc_vis.pkl"},
    "IR":  {"sigma-head": "sigma_ir.pkl", "MC-Dropout": "mc_ir.pkl",
            "ensemble(n=5)": "ens_ir.pkl"},
}


class Arm:
    """One cache reduced to per-frame flat arrays, so a bootstrap draw is a gather."""

    def __init__(self, label: str, path: Path):
        self.label = label
        records, meta = _load(path)
        self.meta = meta
        self.paths = [r["image_path"] for r in records]
        conf, matched, err, sig, unc, risk, counts = [], [], [], [], [], [], []
        gts = []
        for rec in records:
            gt = load_gt(rec["image_path"], rec["image_hw"])
            gts.append(gt)
            n = len(rec["conf"])
            counts.append(n)
            if n == 0:
                continue
            m = match_image(rec, gt, iou_thr=0.5)
            conf.append(rec["conf"])
            matched.append(m["matched"])
            err.append(m["err_edges"])
            sig.append(rec[SIGMA_KEY])
            unc.append(per_box_uncertainty(rec[SIGMA_KEY], rec["boxes_xyxy"]))
            risk.append(1.0 - m["iou_realized"])
        self.conf = np.concatenate(conf)
        self.matched = np.concatenate(matched)
        self.err = np.concatenate(err)
        self.sigma = np.concatenate(sig)
        self.u = np.concatenate(unc)
        self.risk = np.concatenate(risk)
        self.counts = np.asarray(counts, dtype=np.int64)
        self.starts = np.concatenate([[0], np.cumsum(self.counts)[:-1]])
        self.records, self.gts = records, gts

    def gather(self, frames: np.ndarray) -> np.ndarray:
        """Detection indices belonging to `frames` (a frame list, possibly with repeats)."""
        c = self.counts[frames]
        if c.sum() == 0:
            return np.empty(0, dtype=np.int64)
        offs = np.concatenate([[0], np.cumsum(c)[:-1]])
        return np.repeat(self.starts[frames] - offs, c) + np.arange(c.sum())

    def metrics(self, frames: np.ndarray) -> dict:
        i = self.gather(frames)
        if i.size == 0:
            return {k: float("nan") for k in DECISION}
        out = {"d_ece": d_ece(self.conf[i], self.matched[i]),
               "nll": gaussian_nll(self.err[i], self.sigma[i]),
               "interval_ece": coverage_interval_ece(self.err[i], self.sigma[i])["interval_ece"]}
        out.update({k: v for k, v in sparsification(self.u[i], self.risk[i]).items()
                    if k in ("ause", "aurc")})
        return out

    def maps(self, frames: np.ndarray) -> dict:
        recs = [self.records[j] for j in frames]
        gts = [self.gts[j] for j in frames]
        return map50_95(recs, gts)


def _load(path: Path):
    obj = load_cache(path)
    if isinstance(obj, tuple):
        records, meta = obj[0], (obj[1] if len(obj) > 1 else {})
    else:
        records, meta = obj["records"], obj.get("meta", {})
    if isinstance(meta, dict) and "meta" in meta and "source" not in meta:
        meta = meta["meta"]          # caches nest one level
    return records, meta


def verify(arms: list[Arm], modality: str) -> list[str]:
    """Rule 1: identical cache settings and identical frame order across arms."""
    notes = []
    ref = arms[0]
    for key in ("images_list", "imgsz", "conf", "corrupt"):
        vals = {a.label: a.meta.get(key) for a in arms}
        assert len(set(map(str, vals.values()))) == 1, f"{modality}: {key} differs across arms: {vals}"
        notes.append(f"{key}={ref.meta.get(key)}")
    assert ref.meta.get("conf") == 0.001, f"{modality}: conf is {ref.meta.get('conf')}, prereg pins 0.001"
    assert ref.meta.get("corrupt") in (None, "none"), f"{modality}: cache is corrupted"
    for a in arms[1:]:
        assert len(a.paths) == len(ref.paths), f"{modality}: {a.label} frame count differs"
        for p, q in zip(a.paths, ref.paths):
            assert Path(p) == Path(q), f"{modality}: frame order differs at {p} vs {q}"
    return notes


def bands(r: float) -> str:
    if not np.isfinite(r):
        return "NO-SIGNAL"
    return "CLEAN" if r < BAND_CLEAN else ("SUSPECT" if r < BAND_CONTAM else "CONTAMINATED")


WORST = {"CLEAN": 0, "SUSPECT": 1, "CONTAMINATED": 2}


def order(vals: dict[str, float]) -> tuple[str, ...]:
    """Arm ordering, best first. All five decision metrics are lower-is-better."""
    return tuple(k for k, _ in sorted(vals.items(), key=lambda kv: kv[1]))


def flips(day: dict[str, float], pooled: dict[str, float], floor: float) -> dict:
    """Arm pairs whose relative order changed, split by whether the change is resolvable.

    Rule 9 as first written fired on ANY ordering change, with the magnitude floor
    guarding only `sep`. That let a swap between two arms separated by less than the
    noise force CONTAMINATED -- the sign-test-on-noise failure this project has already
    hit twice. A flip counts only if the two arms that swapped are separated by more
    than the floor in BOTH orderings: clearly apart one way on day, clearly apart the
    other way pooled. Anything else is an unresolvable swap and is reported, not acted on.
    """
    resolved, unresolved = [], []
    labs = list(day)
    for i, a in enumerate(labs):
        for b in labs[i + 1:]:
            d, p = day[a] - day[b], pooled[a] - pooled[b]
            if not (np.isfinite(d) and np.isfinite(p)) or (d > 0) == (p > 0):
                continue
            rec = {"pair": (a, b), "gap_day": abs(d), "gap_pooled": abs(p)}
            (resolved if min(abs(d), abs(p)) >= floor else unresolved).append(rec)
    return {"resolved": resolved, "unresolved": unresolved}


def analyse(modality: str, arms: list[Arm]) -> dict:
    n = len(arms[0].paths)
    is_night = np.array([NIGHT_RUN in str(p) for p in arms[0].paths])
    all_f = np.arange(n)
    day_f, night_f = all_f[~is_night], all_f[is_night]

    point = {a.label: {"pooled": a.metrics(all_f), "day": a.metrics(day_f),
                       "night": a.metrics(night_f)} for a in arms}
    maps = {a.label: {"pooled": a.maps(all_f), "day": a.maps(day_f),
                      "night": a.maps(night_f)} for a in arms}

    def stats(pt: dict) -> dict:
        o = {}
        for m in DECISION:
            day = {lab: pt[lab]["day"][m] for lab in pt}
            pool = {lab: pt[lab]["pooled"][m] for lab in pt}
            pull = {lab: pool[lab] - day[lab] for lab in pt}
            dv, pv = list(day.values()), list(pull.values())
            o[m] = {"sep": max(dv) - min(dv), "spread": max(pv) - min(pv),
                    "pull": pull, "day": day, "pooled": pool,
                    "order_day": order(day), "order_pooled": order(pool)}
        return o

    obs = stats(point)

    rng = np.random.default_rng(BOOT_SEED)
    boot = {m: {"sep": [], "spread": []} for m in DECISION}
    for _ in range(N_BOOT):
        draw = rng.integers(0, n, n)                      # rule 6: paired across arms
        dnight = is_night[draw]
        bpt = {a.label: {"pooled": a.metrics(draw), "day": a.metrics(draw[~dnight])}
               for a in arms}
        for m in DECISION:
            day = {lab: bpt[lab]["day"][m] for lab in bpt}
            pull = {lab: bpt[lab]["pooled"][m] - day[lab] for lab in bpt}
            dv, pv = list(day.values()), list(pull.values())
            boot[m]["sep"].append(max(dv) - min(dv))
            boot[m]["spread"].append(max(pv) - min(pv))

    for m in DECISION:
        se_sep = float(np.nanstd(boot[m]["sep"], ddof=1))
        se_spr = float(np.nanstd(boot[m]["spread"], ddof=1))
        scale = float(np.mean([abs(v) for v in obs[m]["pooled"].values()]))
        floor = max(2.0 * se_sep, FLOOR_REL * scale)      # rule 7
        sep, spr = obs[m]["sep"], obs[m]["spread"]
        passes = sep >= floor
        r = spr / sep if (passes and sep > 0) else float("nan")
        fl = flips(obs[m]["day"], obs[m]["pooled"], floor)     # rule 9, floored
        flip = bool(fl["resolved"])
        band = "NO-SIGNAL" if not passes else bands(r)
        if flip and passes:
            band = "CONTAMINATED"
        obs[m].update({"se_sep": se_sep, "se_spread": se_spr, "scale": scale,
                       "floor": floor, "passes_floor": bool(passes), "r": r,
                       "order_flip": flip, "flips": fl, "band": band})

    scored = [obs[m]["band"] for m in DECISION if obs[m]["passes_floor"]]
    verdict = max(scored, key=lambda b: WORST[b]) if scored else "NO-SIGNAL"  # rule 10

    # Rule 9 AS REGISTERED had no floor on the flip: ANY ordering change forced
    # CONTAMINATED. The floor was added on 2026-09-03 AFTER seeing it fire on a swap
    # inside the noise -- i.e. after the fact, which is the one thing a prereg exists to
    # stop. Both verdicts are therefore carried, and the registered one is never dropped.
    asreg = []
    for m in DECISION:
        if not obs[m]["passes_floor"]:
            continue
        f = obs[m]["flips"]
        asreg.append("CONTAMINATED" if (f["resolved"] or f["unresolved"]) else obs[m]["band"])
    verdict_asreg = max(asreg, key=lambda b: WORST[b]) if asreg else "NO-SIGNAL"

    return {"modality": modality, "n": n, "n_day": int((~is_night).sum()),
            "n_night": int(is_night.sum()), "point": point, "maps": maps,
            "stats": obs, "verdict": verdict, "verdict_as_registered": verdict_asreg,
            "arms": [a.label for a in arms]}


def section(res: dict) -> str:
    m_lab = res["modality"]
    rows = []
    for lab in res["arms"]:
        for part in ("pooled", "day", "night"):
            rows.append([lab if part == "pooled" else "", part]
                        + [fmt(res["point"][lab][part][m]) for m in DECISION]
                        + [fmt(res["maps"][lab][part].get("map50_95", float("nan")))])
    body = md_table(["arm", "subset", *DECISION, "map50_95"], rows)

    srows = []
    for m in DECISION:
        s = res["stats"][m]
        nf = len(s["flips"]["resolved"])
        nu = len(s["flips"]["unresolved"])
        fcell = (f"**{nf}**" if nf else "no") + (f" (+{nu} unresolvable)" if nu else "")
        srows.append([m, fmt(s["sep"]), fmt(s["se_sep"]), fmt(s["floor"]),
                      "yes" if s["passes_floor"] else "**no**", fmt(s["spread"]),
                      fmt(s["r"]) if np.isfinite(s["r"]) else "--",
                      fcell, f"**{s['band']}**"])
    stat = md_table(["metric", "sep (day)", "se(sep)", "floor", "clears floor",
                     "spread", "r", "order flips", "band"], srows)
    detail = [f"- `{m}` {w} swap {r['pair'][0]} vs {r['pair'][1]}: "
              f"gap day {fmt(r['gap_day'])}, pooled {fmt(r['gap_pooled'])}, "
              f"floor {fmt(res['stats'][m]['floor'])}"
              for m in DECISION for w, lst in (("RESOLVED", res["stats"][m]["flips"]["resolved"]),
                                               ("unresolvable", res["stats"][m]["flips"]["unresolved"]))
              for r in lst]
    if detail:
        stat += ("\n\nOrdering changes, and whether they clear the floor in **both** "
                 "orderings (rule 9):\n\n" + "\n".join(detail))

    prows = [[m] + [sgn(res["stats"][m]["pull"][lab]) for lab in res["arms"]]
             for m in DECISION]
    pull = md_table(["metric", *res["arms"]], prows)

    amend = ("" if res["verdict"] == res["verdict_as_registered"] else
             f" — **as registered: {res['verdict_as_registered']}** (rule 9 unfloored)")
    return (f"## {m_lab} — verdict **{res['verdict']}**{amend}\n\n"
            f"{res['n']} frames = {res['n_day']} day + {res['n_night']} night "
            f"({100 * res['n_night'] / res['n']:.1f}% night).\n\n"
            f"### Metrics by subset\n\n{body}\n\n"
            f"### The night pull `D(a) = pooled − day`, per arm\n\n{pull}\n\n"
            f"A uniform handicap makes this row constant; `spread` is its range.\n\n"
            f"### The statistic\n\n{stat}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/uq_day_night_slice.md")
    ap.add_argument("--boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    globals()["N_BOOT"] = args.boot

    results, notes = [], []
    for modality, spec in ARMS.items():
        paths = {lab: CACHE / f for lab, f in spec.items()}
        missing = [str(p) for p in paths.values() if not p.is_file()]
        if missing:
            print(f"[skip] {modality}: missing {missing}")
            continue
        arms = [Arm(lab, p) for lab, p in paths.items()]
        notes.append(f"**{modality}** — " + ", ".join(verify(arms, modality)))
        print(f"[{modality}] scoring {len(arms)} arms x 3 subsets + {args.boot} bootstrap draws")
        res = analyse(modality, arms)
        print(f"[{modality}] verdict {res['verdict']}")
        results.append(res)

    vis = next((r for r in results if r["modality"] == "VIS"), None)
    ir = next((r for r in results if r["modality"] == "IR"), None)
    head = [
        "Scored by `scripts/slice_uq_day_night.py` against the bands fixed in "
        "`docs/prereg-uq-day-night-slice.md` (commit `a4f9364`), written before any "
        "cache in `runs/cache_uqslice/` existed. No weight was trained and no label "
        "was touched.",
        "**Stage A.** The VIS ensemble arm is absent: seeds 1–4 weights are on "
        "`dgxanode01` and are not in the archive. Per the prereg, Stage A can return "
        "CONTAMINATED but **cannot** return CLEAN — two arms agreeing says nothing "
        "about the third.",
    ]
    if vis and ir:
        # The prereg registered only two readings -- "IR clean, VIS dirty" and "both the
        # same band" -- and an earlier version of this function collapsed everything else
        # into the first of them. That is wrong in the direction that matters: IR labels
        # were NEVER filtered, so IR moving at night cannot be label contamination, and a
        # dirtier IR than VIS is evidence AGAINST the label story rather than for it.
        clean = {"CLEAN", "NO-SIGNAL"}
        v, i = vis["verdict"], ir["verdict"]
        if v == i:
            why = ("**They land in the same band**, so on the prereg's own reading the "
                   "distortion is NOT label-driven — night is intrinsically harder to "
                   "calibrate on, and retraining would not repair it.")
        elif v not in clean and i in clean:
            why = ("VIS is dirty where IR is clean, and only VIS labels were filtered. "
                   "This is the registered branch in which the distortion tracks the labels.")
        else:
            why = ("**IR is the dirtier arm, and IR labels were never filtered** — so this "
                   "divergence cannot be label contamination. It is the branch the prereg "
                   "did not register. The likely reading is that IR genuinely sees at "
                   "night, night is a different regime for it, and the arms diverge there "
                   "for real reasons; `r` cannot separate that from contamination. Treat "
                   "the IR verdict as a finding about night, not about labels.")
        head.append(f"**VIS {v}, IR {i}.** IR is the negative control: its labels were "
                    f"never filtered. {why}")
    if any(r["verdict"] != r["verdict_as_registered"] for r in results):
        head.insert(1, (
            "> **Amendment, declared.** Rule 9 as registered forced CONTAMINATED on *any* "
            "day-vs-pooled ordering change. On 2026-09-03, **after seeing it fire on a swap "
            "inside the noise**, it was floored: a flip now counts only if the two arms are "
            "separated by more than the floor in **both** orderings. Amending a rule after "
            "seeing it fire is the exact move a pre-registration exists to prevent, so both "
            "verdicts are reported and the registered one is never dropped. The unfloored "
            "rule is the same sign-test-on-noise this project has already been burned by "
            "twice, which is why the amendment was made rather than the result accepted."))
    secs = head + ["## Cache verification\n\n" + "\n\n".join(notes)] + [section(r) for r in results]
    secs.append("---\n\n_Rules: bands CLEAN < 0.25 ≤ SUSPECT < 1.0 ≤ CONTAMINATED on "
                "`r = spread/sep`; a metric enters the verdict only if "
                "`sep ≥ max(2·se_sep, 0.02·scale)`; an ordering flip forces CONTAMINATED "
                "only when the swapped pair clears the floor in both orderings (amended — "
                "as registered, any flip counted); the verdict is the worst band over "
                "metrics that clear the floor._")
    out = write_md(args.out, "UQ calibration table — day and night, unpooled", secs)
    Path(str(out).replace(".md", ".json")).write_text(
        json.dumps(results, indent=1, default=float), encoding="utf-8")
    return 0


if __name__ == "__main__":
    try:
        sys.stdout.reconfigure(errors="replace")
    except Exception:
        pass
    sys.exit(main())
