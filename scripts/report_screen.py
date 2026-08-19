"""Comparison table for the stage-1 small-object screen (runs/screen1).

Every arm runs a fixed epoch budget with early stopping disabled, so the honest
summary is each arm's BEST epoch and its value there, read from results.csv rather
than from best.pt (which agrees, but only because 8.4.90's fitness is mAP50-95
alone -- see train_gaussian.restore_early_stopping).

The seed-1 replicate of the control is the only thing that makes the other rows
readable: |seed0 - seed1| is a one-sample estimate of run-to-run spread, and any
arm inside that band is unranked, not "no effect".

Usage:  python scripts/report_screen.py [--dir runs/screen1] [--control s1_base_seed0]
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def curve(run_dir: Path) -> list[tuple[int, float, float]]:
    f = run_dir / "results.csv"
    if not f.is_file():
        return []
    out = []
    for r in csv.DictReader(open(f, encoding="utf-8")):
        try:
            out.append((int(float(r["epoch"])),
                        float(r["metrics/mAP50-95(B)"]),
                        float(r["metrics/mAP50(B)"])))
        except (KeyError, ValueError, TypeError):
            continue
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dir", default="runs/screen1")
    ap.add_argument("--control", default="s1_base_seed0")
    ap.add_argument("--noise-arm", default="s1_base_seed1")
    ap.add_argument("--queue", default="runs/queue_screen/queue.json")
    args = ap.parse_args()

    root = ROOT / args.dir
    notes, order = {}, []
    qf = ROOT / args.queue
    if qf.is_file():
        q = json.loads(qf.read_text(encoding="utf-8"))
        for r in q["runs"]:
            notes[r["id"]] = r.get("_note", "")
            order.append(r["id"])

    rows = {}
    for rid in order or sorted(p.name for p in root.iterdir() if p.is_dir()):
        c = curve(root / rid)
        if not c:
            continue
        be, bv, b50 = max(c, key=lambda t: t[1])
        # Peak-epoch is a single-epoch max over a jittery curve — the control swings
        # 0.0579 to 0.0676 inside five epochs, which is 6x the seed-to-seed gap. It
        # also truncates any arm still climbing at the budget edge. The mean of the
        # last five epochs is the ranking statistic; peak is kept for continuity with
        # the Phase 2 table, which reports best.pt.
        tail = [v for _, v, _ in c[-5:]]
        rows[rid] = {"epochs": len(c), "best_ep": be, "map": bv, "map50": b50,
                     "tail": sum(tail) / len(tail), "final": c[-1][1],
                     "at_edge": be >= len(c) - 5}

    if args.control not in rows:
        print(f"control '{args.control}' has no results yet in {root}")
        return 1
    ctrl = rows[args.control]["tail"]
    ctrl_peak = rows[args.control]["map"]

    noise = None
    if args.noise_arm in rows:
        noise = abs(rows[args.noise_arm]["tail"] - ctrl)

    print(f"\nstage-1 screen -- {root}")
    print(f"control = {args.control}: last-5 mean {ctrl:.5f} (peak {ctrl_peak:.5f})\n")
    hdr = (f"{'arm':18s} {'peak':>8s} {'@ep':>4s} {'last5':>8s} {'d vs ctrl':>10s} "
           f"{'d%':>7s}  verdict")
    print(hdr)
    print("-" * len(hdr))
    for rid in rows:
        r = rows[rid]
        d = r["tail"] - ctrl
        if rid == args.control:
            verdict = "control"
        elif noise is not None:
            verdict = "inside noise" if abs(d) <= noise else ("GAIN" if d > 0 else "LOSS")
        else:
            verdict = "noise probe" if rid == args.noise_arm else ""
        if r["at_edge"] and rid != args.control:
            verdict += "  [peak at edge: still climbing]"
        print(f"{rid:18s} {r['map']:8.5f} {r['best_ep']:4d} {r['tail']:8.5f} "
              f"{d:+10.5f} {100*d/ctrl:+6.1f}%  {verdict}")

    if noise is not None:
        print(f"\nnoise floor |seed0 - seed1| on last-5 mean = {noise:.5f} "
              f"({100*noise/ctrl:.1f}% of control). ONE pair: an order of magnitude for "
              f"run-to-run spread, not a confidence interval. The same pair on PEAK gives "
              f"{abs(rows[args.noise_arm]['map'] - ctrl_peak):.5f}, and the control alone "
              f"swings more than that inside five epochs -- which is why peak is not the "
              f"ranking statistic.")
    else:
        print(f"\nno noise estimate yet ({args.noise_arm} incomplete) -- deltas unranked.")

    print("\nwhat each arm changed:")
    for rid in rows:
        if notes.get(rid):
            print(f"  {rid:18s} {notes[rid]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
