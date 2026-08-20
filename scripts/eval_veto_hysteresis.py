"""Temporal hysteresis on the veto switch — aimed at the one cell §4.5 leaves open.

The adopted veto fires per frame from an instantaneous brightness reading. But
darkness is a property of a contiguous stretch of a recording, not of one frame.
§4.5 measures the consequence: fog lifts `p05` above `mu_b` on 71% of night
frames, so on fog/night the veto fires on only 29% of them and the cell reaches
0.0789 against `ir_only`'s 0.0810 — the single cell of eight where the gate does
not close. The switch is flickering on a condition that does not flicker.

Two filters over the veto decision, both applied within a recording run in
temporal order (frames are consecutive in the manifest, and the frame index is
recovered from the file stem rather than assumed):

* **majority** — veto if more than half the window says veto. Denoises in both
  directions and cannot extend a veto far past its evidence.
* **dilate** — veto if ANY frame in the window says veto. This is hysteresis
  proper: once the sensor is shown to be dark, brief brightenings do not restore
  trust. Strictly more aggressive, so it is the one that can hurt daylight, and
  the daylight cells are the check on it.

Only the switch is filtered. Smoothing `brightness_vis` instead would also move
`r_bright` inside the soft weight and make the arm a test of two changes; §0.7
already measured that smoothing a continuous weight is inert, and §4.1 explains
why it must be (weights normalise, WBF rescales rather than drops). This is a
different operation on a different object.

`k=1` is the adopted rule and must reproduce it exactly — asserted, not assumed.

Usage:
    python scripts/eval_veto_hysteresis.py [--windows 1 3 5 9 15 31]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.ctx import load_context, run_systems  # noqa: E402


def temporal_order(records) -> dict[str, np.ndarray]:
    """{run: frame indices in ascending capture order}, from the file stem."""
    runs, nums = [], []
    for r in records:
        p = Path(r["image_path"])
        runs.append(p.parent.name)
        m = re.search(r"(\d+)$", p.stem)
        if not m:
            raise ValueError(f"cannot recover a frame number from {p.stem!r}")
        nums.append(int(m.group(1)))
    runs, nums = np.asarray(runs), np.asarray(nums)
    out = {}
    for run in sorted(set(runs.tolist())):
        idx = np.flatnonzero(runs == run)
        out[run] = idx[np.argsort(nums[idx], kind="mergesort")]
    return out


def filter_veto(veto: list[bool], order: dict[str, np.ndarray], k: int, mode: str) -> list[bool]:
    """Apply a length-k filter to the veto flags, within each run, in time order."""
    v = np.asarray(veto, dtype=bool)
    if k <= 1:
        return v.tolist()
    half = k // 2
    out = v.copy()
    for idx in order.values():
        seq = v[idx].astype(np.int32)
        n = len(seq)
        pad = np.pad(seq, (half, half), mode="edge")
        win = np.lib.stride_tricks.sliding_window_view(pad, k)[:n]
        if mode == "majority":
            out[idx] = win.sum(axis=1) * 2 > k
        elif mode == "dilate":
            out[idx] = win.max(axis=1) > 0
        else:
            raise ValueError(f"unknown mode {mode!r}")
    return out.tolist()


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", type=int, nargs="+", default=[1, 3, 5, 9, 15, 31, 61])
    ap.add_argument("--modes", nargs="+", default=["majority", "dilate"])
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--out", default="runs/eval/x_veto_hysteresis.md")
    ap.add_argument("--conditions", nargs="+", default=None,
                    help="restrict the condition sweep (pre-flight uses --conditions clean)")
    args = ap.parse_args()

    t0 = time.time()
    ctx = load_context(**({'conditions': tuple(args.conditions)} if args.conditions else {}))
    order = temporal_order(ctx.vis_by_cond["clean"])
    splits = {"day": ctx.sel("day"), "night": ctx.sel("night")}
    print(f"[hys] temporal order over {len(order)} runs: "
          + ", ".join(f"{k}:{len(v)}" for k, v in order.items()))

    base, base_parts, rates = {}, {}, {}
    for cond in ctx.conditions:
        res = run_systems(ctx, cond)
        base[cond] = (list(res["veto_vis"]), list(res["veto_ir"]))
        base_parts[cond] = frame_parts(res["fused_gated"], ctx.gts)
        for sname, sel in splits.items():
            rates[(cond, sname, 1, "adopted")] = float(np.mean(np.asarray(res["veto_vis"])[sel]))

    rows, parts_keep = [], {}
    for mode in args.modes:
        for k in args.windows:
            if k == 1 and mode != args.modes[0]:
                continue  # k=1 is the adopted rule; run it once
            for cond in ctx.conditions:
                vv = filter_veto(base[cond][0], order, k, mode)
                vi = list(base[cond][1])
                if k == 1:
                    assert vv == list(base[cond][0]), "k=1 must reproduce the adopted decisions"
                res = run_systems(ctx, cond, veto_below=None, veto_override=(vv, vi))
                p = frame_parts(res["fused_gated"], ctx.gts)
                parts_keep[(mode, k, cond)] = p
                if k == 1:
                    # the override path at k=1 must reproduce the fitted path exactly
                    a = ap_from_parts(p)["map50_95"]
                    b = ap_from_parts(base_parts[cond])["map50_95"]
                    assert abs(a - b) < 1e-12, f"override path diverged at k=1: {a} vs {b}"
                for sname, sel in splits.items():
                    rows.append({"mode": mode, "k": k, "condition": cond, "split": sname,
                                 "map": ap_from_parts(p, sel)["map50_95"],
                                 "veto_rate": float(np.mean(np.asarray(vv)[sel]))})
            print(f"[hys] {mode:9s} k={k:3d} " + "  ".join(
                f"{r['condition'][:4]}/{r['split'][:1]}={r['map']:.4f}"
                for r in rows if r["mode"] == mode and r["k"] == k), flush=True)

    # bootstrap the open cell (fog/night) and the guard cell (clean/day).
    # `fog` is the target; a restricted --conditions pre-flight falls back to
    # whatever is present so the script still exercises this path.
    target = "fog" if "fog" in ctx.conditions else ctx.conditions[0]
    guard = "clean" if "clean" in ctx.conditions else ctx.conditions[0]
    boots = {}
    for mode in args.modes:
        for k in args.windows:
            if (mode, k, target) not in parts_keep:
                continue
            boots[(mode, k, target, "night")] = bootstrap_delta(
                parts_keep[(mode, k, target)], base_parts[target], splits["night"], n_boot=args.n_boot)
            boots[(mode, k, guard, "day")] = bootstrap_delta(
                parts_keep[(mode, k, guard)], base_parts[guard], splits["day"], n_boot=args.n_boot)
    print(f"[hys] bootstrap done ({time.time() - t0:.0f}s)", flush=True)

    ir_only = {}
    res0 = run_systems(ctx, "clean")
    for sname, sel in splits.items():
        ir_only[sname] = ap_from_parts(frame_parts(res0["ir_in_vis"], ctx.gts), sel)["map50_95"]

    L = ["# Temporal hysteresis on the veto switch", "",
         f"Filters applied to the veto decision within a run, in capture order. "
         f"`k=1` is the adopted per-frame rule and is asserted to reproduce it exactly. "
         f"IR is never vetoed by design, so only the VIS switch moves.",
         "",
         f"Target: **fog/night**, the one cell §4.5 leaves open "
         f"(0.0789 against `ir_only` {ir_only['night']:.4f}). "
         f"Guard: **clean/day**, which a too-aggressive veto would damage.",
         "",
         "## 1. mAP by filter and window", "",
         "| mode | k | " + " | ".join(f"{c}/{s[:1]}" for c in ctx.conditions for s in splits) + " |",
         "|---|---:|" + "---:|" * (len(ctx.conditions) * len(splits))]
    for mode in args.modes:
        for k in args.windows:
            sel_rows = [r for r in rows if r["mode"] == mode and r["k"] == k]
            if not sel_rows:
                continue
            cells = []
            for c in ctx.conditions:
                for s in splits:
                    m = next((r["map"] for r in sel_rows if r["condition"] == c and r["split"] == s), None)
                    cells.append(f"{m:.4f}" if m is not None else "—")
            L.append(f"| {mode} | {k} | " + " | ".join(cells) + " |")

    L += ["", "## 2. VIS veto rate by filter and window", "",
          "| mode | k | " + " | ".join(f"{c}/{s[:1]}" for c in ctx.conditions for s in splits) + " |",
          "|---|---:|" + "---:|" * (len(ctx.conditions) * len(splits))]
    for mode in args.modes:
        for k in args.windows:
            sel_rows = [r for r in rows if r["mode"] == mode and r["k"] == k]
            if not sel_rows:
                continue
            cells = []
            for c in ctx.conditions:
                for s in splits:
                    m = next((r["veto_rate"] for r in sel_rows if r["condition"] == c and r["split"] == s), None)
                    cells.append(f"{100 * m:.0f}%" if m is not None else "—")
            L.append(f"| {mode} | {k} | " + " | ".join(cells) + " |")

    L += ["", "## 3. The open cell and the guard cell, against the adopted rule", "",
          f"| mode | k | {target}/night delta | 95% CI | {guard}/day delta | 95% CI |",
          "|---|---:|---:|---|---:|---|"]
    for mode in args.modes:
        for k in args.windows:
            if (mode, k, target, "night") not in boots:
                continue
            bf = boots[(mode, k, target, "night")]
            bc = boots[(mode, k, guard, "day")]
            L.append(f"| {mode} | {k} | {bf['delta']:+.4f} | [{bf['ci_lo']:+.4f}, {bf['ci_hi']:+.4f}] | "
                     f"{bc['delta']:+.4f} | [{bc['ci_lo']:+.4f}, {bc['ci_hi']:+.4f}] |")
    L += ["", f"`ir_only` reference: day {ir_only['day']:.4f}, night {ir_only['night']:.4f}. "
              f"A fog/night result at or above {ir_only['night']:.4f} closes the last cell."]

    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps({
        "rows": rows, "ir_only": ir_only,
        "bootstrap": {"|".join(map(str, k)): v for k, v in boots.items()}}, indent=2), encoding="utf-8")
    print(f"[hys] wrote {out} in {time.time() - t0:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
