"""Gate laboratory: cache the two fusion outcomes once, then iterate on rules for free.

`sweep_veto_rules.py` proved the shortcut exact (a veto mask only ever picks
between `WBF(VIS, IR)` and `WBF(IR)` per frame, because the fusion weights are
computed before any veto is consulted). This script persists that computation, so
the ~6-minute WBF replay happens once and every later question -- a new rule, a
new filter, a new threshold, a new cell -- costs milliseconds.

    python scripts/gate_lab.py --build           # ~6 min, writes runs/derived/gate_lab.pkl
    python scripts/gate_lab.py --out runs/eval/gate_lab_rules.md

The cache holds per-frame match parts, not fused boxes: parts are what AP pools,
they are ~10x smaller, and they cannot be mistaken for a detection set someone
might try to re-score under a different matcher.

**Nothing here overwrites anything.** The cache lives at a new path and every
report goes to a filename the caller supplies.
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import FIT_RUNS, NIGHT_RUNS, load_context, run_systems    # noqa: E402
from uqfusion.eval.hysteresis import filter_veto                                 # noqa: E402

SHIP, BUOY = 0, 1
CACHE = ROOT / "runs" / "derived" / "gate_lab_v2.pkl"

#: Soft-weight arms. The veto is orthogonal to all of them and is applied later,
#: so every arm is cached in both its unvetoed and its VIS-vetoed form.
#:
#: `no_maha` and `cap_only` are the handoff §6 ablations, and they are in the
#: cache rather than in a separate script because the lowlight/day question turns
#: out to be about them: the Mahalanobis term does not merely fail to help there,
#: it is the mechanism of the loss. lowlight pushes VIS's D to ~248 against a mu_d
#: of 71, so r_frame_vis collapses to ~0.009 while clean IR keeps ~0.82 -- and
#: after the capability prior that leaves w_vis at roughly 0.28 against IR's 0.72.
#: The gate hands the frame to IR on a cell where VIS scores 0.0346 and IR 0.0177.
VARIANTS = ("adopted", "no_maha", "cap_only")


def variant_ctx(ctx, name: str):
    if name == "adopted":
        return ctx
    if name == "no_maha":
        return replace(ctx, c_vis=replace(ctx.c_vis, mu_d=1e9),
                       c_ir=replace(ctx.c_ir, mu_d=1e9))
    if name == "cap_only":
        return replace(ctx, c_vis=replace(ctx.c_vis, mu_d=1e9, lam=0.0),
                       c_ir=replace(ctx.c_ir, mu_d=1e9, lam=0.0))
    raise ValueError(name)


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------

def build(cache_path: Path, variants=VARIANTS) -> dict:
    t0 = time.time()
    ctx = load_context()
    n = ctx.n()
    payload = {
        "conditions": list(ctx.conditions),
        "variants": list(variants),
        "runs": ctx.runs.tolist(),
        "order": {k: v.tolist() for k, v in ctx.order.items()},
        "cap_vis": ctx.cap_vis, "cap_ir": ctx.cap_ir,
        "mu_b": ctx.c_vis.mu_b, "tau_b": ctx.c_vis.tau_b, "veto": ctx.veto,
        "tau_lap": ctx.tau_lap,
        "bright": {c: v.tolist() for c, v in ctx.bright_by_cond.items()},
        "lap": {c: v.tolist() for c, v in ctx.struct_by_cond.items()},
        "evidence": {}, "parts": {}, "w_vis": {},
    }
    for cond in ctx.conditions:
        recs = ctx.vis_by_cond[cond]
        payload["evidence"][cond] = _evidence(recs)
        for name in variants:
            vctx = variant_ctx(ctx, name)
            res_b = run_systems(vctx, cond, veto_override=([False] * n, [False] * n))
            res_v = run_systems(vctx, cond, veto_override=([True] * n, [False] * n))
            payload["parts"][(name, cond)] = {
                "both": frame_parts(res_b["fused_gated"], ctx.gts),
                "veto": frame_parts(res_v["fused_gated"], ctx.gts),
            }
            payload["w_vis"][(name, cond)] = list(map(float, res_b["w_vis_gated"]))
            if name == variants[0]:
                payload["parts"][("_single", cond)] = {
                    "vis": frame_parts(recs, ctx.gts),
                    "ir": frame_parts(res_b["ir_in_vis"], ctx.gts),
                }
            print(f"[lab] {cond}/{name} cached ({time.time() - t0:.0f}s)", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    with open(cache_path, "wb") as f:
        pickle.dump(payload, f, protocol=4)
    print(f"[lab] wrote {cache_path} "
          f"({cache_path.stat().st_size / 1e6:.0f} MB, {time.time() - t0:.0f}s)")
    return payload


def _evidence(records: list[dict]) -> dict[str, list[float]]:
    """The VIS detector's own response per frame -- available at inference time."""
    out = {"max_conf": [], "sum_conf": [], "n_c25": [], "top5_conf": []}
    for r in records:
        c = np.asarray(r["conf"], dtype=float).reshape(-1)
        s = np.sort(c)[::-1] if len(c) else c
        out["max_conf"].append(float(s[0]) if len(s) else 0.0)
        out["sum_conf"].append(float(c.sum()))
        out["n_c25"].append(float((c >= 0.25).sum()))
        out["top5_conf"].append(float(s[:5].mean()) if len(s) else 0.0)
    return out


# ---------------------------------------------------------------------------
# lab
# ---------------------------------------------------------------------------

class Lab:
    def __init__(self, payload: dict):
        p = payload
        self.conditions = tuple(p["conditions"])
        self.variants = tuple(p.get("variants", ("adopted",)))
        self.runs = np.asarray(p["runs"])
        self.order = {k: np.asarray(v) for k, v in p["order"].items()}
        self.n = len(self.runs)
        self.night = np.isin(self.runs, NIGHT_RUNS)
        self.fit = np.isin(self.runs, FIT_RUNS)
        self.splits = {"day": ~self.night, "night": self.night}
        self.cells = [(c, s) for c in self.conditions for s in ("day", "night")]
        self.bright = {c: np.asarray(v) for c, v in p["bright"].items()}
        self.lap = {c: np.asarray(v) for c, v in p["lap"].items()}
        self.ev = {c: {k: np.asarray(v) for k, v in d.items()}
                   for c, d in p["evidence"].items()}
        self.parts = p["parts"]
        self.w_vis = {k: np.asarray(v) for k, v in p.get("w_vis", {}).items()}
        self.mu_b, self.tau_b, self.veto = p["mu_b"], p["tau_b"], p["veto"]
        self.tau_lap = p["tau_lap"]
        self.cap_vis, self.cap_ir = p["cap_vis"], p["cap_ir"]

    # -- scoring -----------------------------------------------------------
    def mixed(self, cond: str, mask: np.ndarray, sel: np.ndarray,
              variant: str = "adopted") -> list:
        d = self.parts[(variant, cond)]
        pb, pv = d["both"], d["veto"]
        return [pv[i] if mask[i] else pb[i] for i in np.flatnonzero(sel)]

    def ap(self, parts: list, cls: int | None = SHIP) -> float:
        r = ap_from_parts(parts)
        if cls is None:
            return r["map50_95"]
        e = r["per_class"].get(cls)
        return float(e["ap50_95"]) if e else 0.0

    def score(self, cond: str, mask: np.ndarray, split: str, cls: int | None = SHIP,
              variant: str = "adopted") -> float:
        return self.ap(self.mixed(cond, mask, self.splits[split], variant), cls)

    def ref(self, cond: str, which: str, split: str, cls: int | None = SHIP) -> float:
        sel = np.flatnonzero(self.splits[split])
        return self.ap([self.parts[("_single", cond)][which][i] for i in sel], cls)

    # -- signals -----------------------------------------------------------
    def window(self, v: np.ndarray, k: int, how: str = "max") -> np.ndarray:
        if k <= 1:
            return v.copy()
        out = v.copy()
        half = k // 2
        for idx in self.order.values():
            seq = v[idx]
            pad = np.pad(seq, (half, half), mode="edge")
            win = np.lib.stride_tricks.sliding_window_view(pad, k)[:len(seq)]
            out[idx] = win.max(axis=1) if how == "max" else np.median(win, axis=1)
        return out

    def filt(self, m: np.ndarray, mode: str, k: int) -> np.ndarray:
        return np.asarray(filter_veto(m.tolist(), self.order, k, mode), dtype=bool)

    def novelty_thr(self, key: str, k: int) -> float:
        """Hard lower bound of the windowed statistic over CLEAN FIT-RUN frames.

        The same protocol `fit_veil_gate.py` uses for `tau_lap`: pohang00/02/03,
        clean only, pohang01 and every corrupted condition held out. The rule the
        threshold encodes is "this sensor is producing less evidence than it ever
        produced on data we fitted on", which is a statement about the reference
        distribution and not about any particular corruption.
        """
        return float(self.window(self.ev["clean"][key], k, "max")[self.fit].min())

    # -- rules -------------------------------------------------------------
    def photometric(self, cond: str) -> np.ndarray:
        r = 1.0 / (1.0 + np.exp(-(self.bright[cond] - self.mu_b) / max(self.tau_b, 1e-9)))
        return self.filt(r < self.veto, "dilate", 15)

    def veil(self, cond: str) -> np.ndarray:
        return self.filt(self.lap[cond] < self.tau_lap, "majority", 15)

    def evidence(self, cond: str, key: str, k: int) -> np.ndarray:
        return self.window(self.ev[cond][key], k, "max") < self.novelty_thr(key, k)

    def evidence_rescue(self, cond: str, key: str, k: int, rkey: str = "max_conf") -> np.ndarray:
        """Blind STRETCH, minus frames whose own evidence clears the clean bound.

        Two levels, and the split is the point. Whether a sensor has gone blind is
        a property of a stretch of recording -- it is low-variance, and the
        windowed statistic measures it well. Whether THIS frame is one the
        detector still handled is a per-frame question, and answering it with the
        windowed statistic throws away exactly the frames worth keeping. So the
        stretch decides the default and a frame may overrule it, but only on
        evidence stronger than anything the clean fit runs ever fell below, which
        is a bound a genuinely blind sensor cannot clear.
        """
        return self.evidence(cond, key, k) & ~(self.ev[cond][rkey] >= self.novelty_thr(rkey, 1))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--cache", default=str(CACHE))
    ap.add_argument("--out", default="runs/eval/gate_lab_rules.md")
    ap.add_argument("--n-boot", type=int, default=0,
                    help="bootstrap the best rule against the adopted one (0 = skip)")
    args = ap.parse_args()

    cache_path = Path(args.cache)
    if args.build or not cache_path.is_file():
        payload = build(cache_path)
    else:
        with open(cache_path, "rb") as f:
            payload = pickle.load(f)
        print(f"[lab] loaded {cache_path}")
    lab = Lab(payload)

    # ---- rule catalogue ---------------------------------------------------
    rules: dict[str, dict[str, np.ndarray]] = {}

    def add(name, fn):
        rules[name] = {c: fn(c) for c in lab.conditions}

    add("A photometric only", lab.photometric)
    add("B photometric OR veil", lambda c: lab.photometric(c) | lab.veil(c))
    add("Z no veto", lambda c: np.zeros(lab.n, bool))
    add("C veil only", lab.veil)
    for key in ("max_conf", "sum_conf", "top5_conf", "n_c25"):
        for k in (1, 15, 31, 61):
            add(f"D evidence {key} w{k}", lambda c, key=key, k=k: lab.evidence(c, key, k))
    for key in ("max_conf", "sum_conf", "top5_conf", "n_c25"):
        for k in (15, 31, 61):
            add(f"E evidence {key} w{k} + rescue",
                lambda c, key=key, k=k: lab.evidence_rescue(c, key, k))
    for key in ("sum_conf", "max_conf"):
        for k in (15, 31):
            add(f"F photometric OR veil OR evidence {key} w{k}",
                lambda c, key=key, k=k: lab.photometric(c) | lab.veil(c) | lab.evidence(c, key, k))
            add(f"G (photometric OR veil OR evidence {key} w{k}) + rescue",
                lambda c, key=key, k=k: (
                    (lab.photometric(c) | lab.veil(c) | lab.evidence(c, key, k))
                    & ~(lab.ev[c]["max_conf"] >= lab.novelty_thr("max_conf", 1))))

    # ---- references and the bar ------------------------------------------
    refs, bar = {}, {}
    for cond, s in lab.cells:
        v = lab.ref(cond, "vis", s)
        i = lab.ref(cond, "ir", s)
        refs[(cond, s)] = {"vis": v, "ir": i,
                           "never": lab.score(cond, np.zeros(lab.n, bool), s),
                           "always": lab.score(cond, np.ones(lab.n, bool), s),
                           "w_vis": {vn: float(np.mean(lab.w_vis[(vn, cond)][lab.splits[s]]))
                                     for vn in lab.variants if (vn, cond) in lab.w_vis}}
        bar[(cond, s)] = max(v, i)

    rows = []
    for variant in lab.variants:
        for name, masks in rules.items():
            cells = {}
            for cond, s in lab.cells:
                cells[f"{cond}/{s}"] = {
                    "ap": lab.score(cond, masks[cond], s, variant=variant),
                    "rate": float(masks[cond][lab.splits[s]].mean())}
            gaps = [cells[f"{c}/{s}"]["ap"] - bar[(c, s)] for c, s in lab.cells]
            rows.append({"rule": f"{name}  [{variant}]", "variant": variant, "base": name,
                         "cells": cells, "worst_gap": float(min(gaps)),
                         "sum_gap": float(sum(gaps))})
    ranked = sorted(rows, key=lambda r: (-r["worst_gap"], -r["sum_gap"]))

    L = ["# Gate laboratory — veto rules over the eight cells (ship AP)", "",
         "Ship AP (class 0) only — IR is nc=1 ship-only, so ship is the one class "
         "both streams can produce (handoff §5). Every rule is scored on the same "
         "cached fusion outcomes, so a difference between two rows is the rule and "
         "nothing else.", "",
         "`bar` = max(VIS, IR) per cell: what the gate has to beat to have earned "
         "its place. Rules are ranked by their WORST cell, because a gate is a "
         "safety mechanism and its worth is set by the condition it handles least "
         "well, not by its average.", "",
         "## 0. References", "",
         "`never`/`always` are under the ADOPTED soft weights. `mean w_vis` is "
         "printed per soft-weight arm because it is the mechanism behind the "
         "lowlight/day result: the gate can only prefer the stream its weights "
         "point at.", "",
         "| cell | VIS | IR | never veto | always veto | bar | "
         + " | ".join(f"w_vis[{v}]" for v in lab.variants) + " |",
         "|---|---:|---:|---:|---:|---:|" + "---:|" * len(lab.variants)]
    for cond, s in lab.cells:
        r = refs[(cond, s)]
        L.append(f"| {cond}/{s} | {r['vis']:.4f} | {r['ir']:.4f} | {r['never']:.4f} | "
                 f"{r['always']:.4f} | {bar[(cond, s)]:.4f} | "
                 + " | ".join(f"{r['w_vis'].get(v, float('nan')):.3f}" for v in lab.variants)
                 + " |")

    L += ["", "## 1. Ship AP, ranked by worst cell", "",
          "| rule | " + " | ".join(f"{c}/{s}" for c, s in lab.cells) + " | worst gap |",
          "|---|" + "---:|" * (len(lab.cells) + 1)]
    for r in ranked:
        L.append(f"| {r['rule']} | "
                 + " | ".join(f"{r['cells'][f'{c}/{s}']['ap']:.4f}" for c, s in lab.cells)
                 + f" | {r['worst_gap']:+.4f} |")

    L += ["", "## 2. Veto rate, same order", "",
          "| rule | " + " | ".join(f"{c}/{s}" for c, s in lab.cells) + " |",
          "|---|" + "---:|" * len(lab.cells)]
    for r in ranked:
        L.append(f"| {r['rule']} | "
                 + " | ".join(f"{r['cells'][f'{c}/{s}']['rate']:.1%}" for c, s in lab.cells) + " |")

    L += ["", "## 3. Gap to bar per cell (AP − max(VIS, IR)), same order", "",
          "| rule | " + " | ".join(f"{c}/{s}" for c, s in lab.cells) + " |",
          "|---|" + "---:|" * len(lab.cells)]
    for r in ranked:
        L.append(f"| {r['rule']} | "
                 + " | ".join(f"{r['cells'][f'{c}/{s}']['ap'] - bar[(c, s)]:+.4f}"
                             for c, s in lab.cells) + " |")

    boot = []
    if args.n_boot:
        best = ranked[0]
        ref_rule, ref_var = "B photometric OR veil", "adopted"
        for cond, s in lab.cells:
            sel = lab.splits[s]
            a = lab.mixed(cond, rules[best["base"]][cond], sel, best["variant"])
            b = lab.mixed(cond, rules[ref_rule][cond], sel, ref_var)
            bd = bootstrap_delta(a, b, None, n_boot=args.n_boot, cls=SHIP)
            boot.append({"cell": f"{cond}/{s}", **bd})
        L += ["", f"## 4. `{best['rule']}` vs `{ref_rule}  [{ref_var}]`, paired "
              f"bootstrap (n={args.n_boot}, ship AP)", "",
              "| cell | best | adopted | delta | 95% CI |", "|---|---:|---:|---:|---|"]
        for b in boot:
            L.append(f"| {b['cell']} | {b['a']:.4f} | {b['b']:.4f} | {b['delta']:+.4f} | "
                     f"[{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]"
                     f"{' (spans 0)' if b['spans_zero'] else ''} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"reference": {f"{c}/{s}": refs[(c, s)] for c, s in lab.cells},
         "bar": {f"{c}/{s}": bar[(c, s)] for c, s in lab.cells},
         "rules": ranked, "bootstrap": boot}, indent=2), encoding="utf-8")
    print(f"[lab] wrote {out}")
    for r in ranked[:6]:
        print(f"[lab] {r['worst_gap']:+.4f}  {r['rule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
