"""Sweep candidate VIS-veto rules over the eight cells, cheaply and exactly.

**The trick that makes this affordable.** `eval_final_system.py` re-runs WBF over
2,232 frames for every arm, ~25 s per condition per variant, which is why the
adopted table carries four arms and not forty. But the veto does not change the
fusion *weights* -- those come from R and the capability prior and are computed
before any veto is consulted. It changes only WHICH of two already-determined
outcomes each frame gets:

    fused_both[i]   -- WBF over (VIS, IR) at the gated weights
    fused_veto[i]   -- WBF over (IR) alone

So both are computed ONCE per condition, `frame_parts` is taken of each, and
every candidate rule is then a per-frame pick between two cached match arrays.
Rule evaluation drops from ~25 s to ~2 ms and a forty-rule sweep becomes
interactive. This is exact, not an approximation: the fused detections a rule
produces are byte-for-byte the ones `run_systems` would have produced under the
same mask, which `--verify` asserts against a real `run_systems` call.

**What it is for.** `docs/gated-fusion-handoff.md` §7.2 claims no frame statistic
keeps lowlight/day without breaking the clean/day and glare/day guard cells. That
claim was tested on histogram and focus statistics of the INPUT IMAGE. This sweep
adds the detector's own output as a third axis and reports, for every rule, all
eight cells at once plus the coordinate-ascent oracle -- the best any per-frame
veto could do -- so a rule can be judged against the ceiling rather than against
the adopted system alone.

Writes a new report; overwrites nothing. Usage:
    python scripts/sweep_veto_rules.py --out runs/eval/veto_rule_sweep.md
    python scripts/sweep_veto_rules.py --verify --conditions clean
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

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts          # noqa: E402
from uqfusion.eval.ctx import FIT_RUNS, NIGHT_RUNS, load_context, run_systems  # noqa: E402
from uqfusion.eval.hysteresis import filter_veto                        # noqa: E402

SHIP, BUOY = 0, 1


# ---------------------------------------------------------------------------
# per-frame signals
# ---------------------------------------------------------------------------

def detector_evidence(records: list[dict]) -> dict[str, np.ndarray]:
    """The VIS detector's own response, per frame.

    Deliberately kept to quantities a deployed system already has in hand: the
    confidences of the boxes it just emitted. No labels, no second stream, no
    second pass over the image.
    """
    max_conf, sum_conf, n_c25 = [], [], []
    for r in records:
        c = np.asarray(r["conf"], dtype=float).reshape(-1)
        max_conf.append(float(c.max()) if len(c) else 0.0)
        sum_conf.append(float(c.sum()))
        n_c25.append(float((c >= 0.25).sum()))
    return {"max_conf": np.asarray(max_conf), "sum_conf": np.asarray(sum_conf),
            "n_c25": np.asarray(n_c25)}


def window_stat(v: np.ndarray, order: dict, k: int, how: str = "max") -> np.ndarray:
    """Aggregate `v` over a centred window of k frames, within run, in capture order."""
    if k <= 1:
        return v.copy()
    out = v.copy()
    half = k // 2
    for idx in order.values():
        seq = v[idx]
        pad = np.pad(seq, (half, half), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, k)[:len(seq)]
        out[idx] = win.max(axis=1) if how == "max" else np.median(win, axis=1)
    return out


# ---------------------------------------------------------------------------
# rules
# ---------------------------------------------------------------------------

def photometric_mask(bright: np.ndarray, mu_b: float, tau_b: float, veto: float,
                     order: dict) -> np.ndarray:
    """The adopted darkness axis: r_bright < veto, then dilate-15."""
    r = 1.0 / (1.0 + np.exp(-(bright - mu_b) / max(tau_b, 1e-9)))
    return np.asarray(filter_veto((r < veto).tolist(), order, 15, "dilate"), dtype=bool)


def veil_mask(lap: np.ndarray, tau_lap: float, order: dict) -> np.ndarray:
    """The adopted veil axis: lap_var < tau_lap, then majority-15."""
    return np.asarray(filter_veto((lap < tau_lap).tolist(), order, 15, "majority"), dtype=bool)


def evidence_mask(ev: np.ndarray, thr: float, order: dict, k: int,
                  filt: tuple[str, int] | None) -> np.ndarray:
    """The proposed detection-evidence axis: windowed evidence below a novelty bound.

    The window is inside the statistic, not after the threshold, and that is the
    substantive choice. An empty-sea clean day frame legitimately produces no
    detections; a per-frame evidence test would veto it and the guard cell would
    move. Taking the window max FIRST asks "did this sensor produce evidence
    anywhere near here?", which is the question that distinguishes a quiet scene
    from a blind sensor.
    """
    m = window_stat(ev, order, k, "max") < thr
    if filt is not None:
        m = np.asarray(filter_veto(m.tolist(), order, filt[1], filt[0]), dtype=bool)
    return m


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/veto_rule_sweep.md")
    ap.add_argument("--conditions", nargs="+", default=None)
    ap.add_argument("--verify", action="store_true",
                    help="assert the two-outcome shortcut reproduces run_systems exactly")
    ap.add_argument("--oracle-passes", type=int, default=2)
    args = ap.parse_args()

    t0 = time.time()
    kw = {"conditions": tuple(args.conditions)} if args.conditions else {}
    ctx = load_context(**kw)
    conds = ctx.conditions
    night = np.isin(ctx.runs, NIGHT_RUNS)
    fitsel = np.isin(ctx.runs, FIT_RUNS)
    splits = {"day": ~night, "night": night}
    cells = [(c, s) for c in conds for s in ("day", "night")]
    n = ctx.n()
    order = ctx.order

    # ---- the two outcomes per condition, computed once ---------------------
    parts_both, parts_veto, parts_vis, parts_ir = {}, {}, {}, {}
    for cond in conds:
        res_b = run_systems(ctx, cond, veto_override=([False] * n, [False] * n))
        res_v = run_systems(ctx, cond, veto_override=([True] * n, [False] * n))
        parts_both[cond] = frame_parts(res_b["fused_gated"], ctx.gts)
        parts_veto[cond] = frame_parts(res_v["fused_gated"], ctx.gts)
        parts_vis[cond] = frame_parts(ctx.vis_by_cond[cond], ctx.gts)
        parts_ir[cond] = frame_parts(res_b["ir_in_vis"], ctx.gts)
        print(f"[sweep] {cond}: both/veto outcomes cached ({time.time() - t0:.0f}s)", flush=True)

    def score(cond: str, mask: np.ndarray, sel: np.ndarray, cls=SHIP) -> float:
        # Only the selected frames are assembled: the oracle calls this once per
        # candidate frame flip, so building all 2,232 parts each time would
        # dominate the runtime with rows the AP never pools.
        idx = np.flatnonzero(sel)
        pb, pv = parts_both[cond], parts_veto[cond]
        p = [pv[i] if mask[i] else pb[i] for i in idx]
        r = ap_from_parts(p)
        if cls is None:
            return r["map50_95"]
        e = r["per_class"].get(cls)
        return float(e["ap50_95"]) if e else 0.0

    if args.verify:
        # The shortcut must equal a real run_systems call under the same mask.
        rng = np.random.default_rng(0)
        m = rng.random(n) < 0.4
        cond = conds[0]
        ref = run_systems(ctx, cond, veto_override=(m.tolist(), [False] * n))
        ref_p = frame_parts(ref["fused_gated"], ctx.gts)
        a = ap_from_parts(ref_p, np.flatnonzero(splits["day"]))["per_class"].get(SHIP)
        b = score(cond, m, splits["day"])
        a = float(a["ap50_95"]) if a else 0.0
        print(f"[verify] run_systems {a:.10f}  shortcut {b:.10f}  |d| {abs(a - b):.2e}")
        assert abs(a - b) < 1e-12, "the two-outcome shortcut is NOT exact"
        print("[verify] OK — shortcut is exact")

    # ---- reference rows ----------------------------------------------------
    ref_rows = {}
    for cond, s in cells:
        sel = splits[s]
        ref_rows[(cond, s)] = {
            "vis": float((ap_from_parts(parts_vis[cond], np.flatnonzero(sel))["per_class"].get(SHIP)
                          or {"ap50_95": 0.0})["ap50_95"]),
            "ir": float((ap_from_parts(parts_ir[cond], np.flatnonzero(sel))["per_class"].get(SHIP)
                         or {"ap50_95": 0.0})["ap50_95"]),
            "never": score(cond, np.zeros(n, bool), sel),
            "always": score(cond, np.ones(n, bool), sel),
        }

    # ---- candidate rules ---------------------------------------------------
    bright = ctx.bright_by_cond
    lap = ctx.struct_by_cond
    ev = {c: detector_evidence(ctx.vis_by_cond[c]) for c in conds}

    # Every threshold is a NOVELTY bound on CLEAN FIT-RUN frames: the value below
    # which no clean fit frame ever falls. Same protocol as tau_lap; pohang01 and
    # every corrupted condition are held out of the fit.
    thr = {}
    for key in ("max_conf", "sum_conf", "n_c25"):
        for k in (1, 15, 31):
            ref = window_stat(ev["clean"][key], order, k, "max")[fitsel]
            thr[(key, k)] = float(ref.min())

    rules: dict[str, dict[str, np.ndarray]] = {}

    def add(name: str, fn) -> None:
        rules[name] = {c: fn(c) for c in conds}

    add("A. photometric only (pre-2026-09-01)",
        lambda c: photometric_mask(bright[c], ctx.c_vis.mu_b, ctx.c_vis.tau_b, ctx.veto, order))
    if lap:
        add("B. photometric OR veil (adopted 2026-09-01)",
            lambda c: (photometric_mask(bright[c], ctx.c_vis.mu_b, ctx.c_vis.tau_b, ctx.veto, order)
                       | veil_mask(lap[c], ctx.tau_lap, order)))
        add("C. veil only",
            lambda c: veil_mask(lap[c], ctx.tau_lap, order))
    for key in ("max_conf", "sum_conf", "n_c25"):
        for k in (1, 15, 31):
            for fname, filt in (("raw", None), ("maj15", ("majority", 15))):
                add(f"D. evidence {key} w{k} {fname}",
                    lambda c, key=key, k=k, filt=filt: evidence_mask(
                        ev[c][key], thr[(key, k)], order, k, filt))
    for key in ("max_conf", "sum_conf"):
        for k in (1, 15):
            add(f"E. photometric OR evidence {key} w{k}",
                lambda c, key=key, k=k: (
                    photometric_mask(bright[c], ctx.c_vis.mu_b, ctx.c_vis.tau_b, ctx.veto, order)
                    | evidence_mask(ev[c][key], thr[(key, k)], order, k, None)))
            if lap:
                add(f"F. all three ({key} w{k})",
                    lambda c, key=key, k=k: (
                        photometric_mask(bright[c], ctx.c_vis.mu_b, ctx.c_vis.tau_b, ctx.veto, order)
                        | veil_mask(lap[c], ctx.tau_lap, order)
                        | evidence_mask(ev[c][key], thr[(key, k)], order, k, None)))

    # ---- oracle ceiling ----------------------------------------------------
    # Coordinate ascent on ship AP, per cell, starting from the never-veto mask.
    # This is an ORACLE: it reads the labels. It is not a proposal, it is the
    # ceiling any per-frame veto rule is competing against, and a rule that
    # already sits near it has nothing left to win.
    oracle = {}
    for cond, s in cells:
        sel = splits[s]
        idx = np.flatnonzero(sel)
        m = np.zeros(n, bool)
        best = score(cond, m, sel)
        for _ in range(args.oracle_passes):
            improved = False
            for i in idx:
                m[i] = ~m[i]
                v = score(cond, m, sel)
                if v > best + 1e-12:
                    best, improved = v, True
                else:
                    m[i] = ~m[i]
            if not improved:
                break
        oracle[(cond, s)] = {"ap": best, "rate": float(m[sel].mean())}
        print(f"[sweep] oracle {cond}/{s}: {best:.4f} at {m[sel].mean():.1%} veto "
              f"({time.time() - t0:.0f}s)", flush=True)

    # ---- score every rule --------------------------------------------------
    rows = []
    for name, masks in rules.items():
        row = {"rule": name, "cells": {}}
        for cond, s in cells:
            sel = splits[s]
            row["cells"][f"{cond}/{s}"] = {
                "ap": score(cond, masks[cond], sel),
                "rate": float(masks[cond][sel].mean()),
            }
        rows.append(row)
    print(f"[sweep] {len(rows)} rules scored ({time.time() - t0:.0f}s)", flush=True)

    # ---- report ------------------------------------------------------------
    # The bar every rule is held to: max(VIS, IR) on that cell. A gate that beats
    # neither single stream has not earned the fusion.
    bar = {k: max(v["vis"], v["ir"]) for k, v in ref_rows.items()}

    L = ["# VIS-veto rule sweep — ship AP over the eight cells", "",
         "Ship AP (class 0) only: IR is nc=1 ship-only (D28/A-1), so ship is the "
         "single class both streams can produce and the macro column answers no "
         "one question (handoff §5). Every rule below is evaluated on exactly the "
         "same cached fusion outcomes, so differences are the rule and nothing "
         "else.", "",
         "`bar` = max(VIS, IR) on that cell — the score a gate has to beat to have "
         "earned anything. `oracle` is coordinate ascent on the labels: the ceiling "
         "for ANY per-frame veto rule, not a proposal.", "",
         "## 0. References", "",
         "| cell | VIS ship | IR ship | never veto | always veto | bar | oracle | oracle veto% |",
         "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for cond, s in cells:
        r, o = ref_rows[(cond, s)], oracle[(cond, s)]
        L.append(f"| {cond}/{s} | {r['vis']:.4f} | {r['ir']:.4f} | {r['never']:.4f} | "
                 f"{r['always']:.4f} | {bar[(cond, s)]:.4f} | {o['ap']:.4f} | {o['rate']:.1%} |")

    L += ["", "## 1. Ship AP per rule", "",
          "| rule | " + " | ".join(f"{c}/{s}" for c, s in cells) + " | min(AP−bar) |",
          "|---|" + "---:|" * (len(cells) + 1)]
    for row in rows:
        gaps = [row["cells"][f"{c}/{s}"]["ap"] - bar[(c, s)] for c, s in cells]
        row["worst_gap"] = float(min(gaps))
        L.append(f"| {row['rule']} | "
                 + " | ".join(f"{row['cells'][f'{c}/{s}']['ap']:.4f}" for c, s in cells)
                 + f" | {min(gaps):+.4f} |")

    L += ["", "## 2. Veto rate per rule", "",
          "| rule | " + " | ".join(f"{c}/{s}" for c, s in cells) + " |",
          "|---|" + "---:|" * len(cells)]
    for row in rows:
        L.append(f"| {row['rule']} | "
                 + " | ".join(f"{row['cells'][f'{c}/{s}']['rate']:.1%}" for c, s in cells) + " |")

    ranked = sorted(rows, key=lambda r: -r["worst_gap"])
    L += ["", "## 3. Ranked by worst cell (AP − bar), best first", "",
          "The ranking metric is the WORST cell, not the mean: a gate is a safety "
          "mechanism and its value is set by the condition it handles least well.", "",
          "| rank | rule | worst gap | worst cell |", "|---:|---|---:|---|"]
    for i, row in enumerate(ranked, 1):
        wc = min(cells, key=lambda k: row["cells"][f"{k[0]}/{k[1]}"]["ap"] - bar[k])
        L.append(f"| {i} | {row['rule']} | {row['worst_gap']:+.4f} | {wc[0]}/{wc[1]} |")

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"thresholds": {f"{k[0]}@{k[1]}": v for k, v in thr.items()},
         "reference": {f"{c}/{s}": ref_rows[(c, s)] for c, s in cells},
         "oracle": {f"{c}/{s}": oracle[(c, s)] for c, s in cells},
         "rules": rows}, indent=2), encoding="utf-8")
    print(f"[sweep] wrote {out} in {time.time() - t0:.0f}s")
    for row in ranked[:5]:
        print(f"[sweep] {row['worst_gap']:+.4f}  {row['rule']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
