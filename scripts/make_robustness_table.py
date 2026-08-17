"""Cross-check every Table 1 number against the run's own training curve.

Table 1 reports an explicit `best.pt` validation (grid.py:231). That number
depends on which checkpoint is on disk — which is exactly what the concurrent
-writer collision in the server's yolo12x directories put in doubt. Each run's
`results.csv` carries a second, independent record: the in-training validator's
metrics at every epoch. If the two agree across the grid, the ranking does not
rest on the checkpoint lottery.

Both numbers now live side by side in the consolidated record
(`phase1_benchmark/results.csv`, columns `map50_95` and `best_map50_95_curve`),
so this script no longer has to locate run directories or disambiguate rows by
wall clock — it reads one file. Regenerate the record itself with
`scripts/consolidate_phase1.py`.

    python scripts/make_robustness_table.py --out docs/phase1-robustness-table.md
"""

from __future__ import annotations

import argparse
import csv
import statistics as st
from pathlib import Path

RECORD = Path("phase1_benchmark/results.csv")


def _mixed_version(row: dict) -> bool:
    """True if the run TRAINED under a different ultralytics than it was scored under.

    grid.py stamps '8.4.7+val8.4.90' for such a row. Its Table 1 metric is on the
    8.4.90 scale but every number in its results.csv is on 8.4.7's, which reads
    ~0.034-0.039 mAP50-95 high — so its curve-peak is not comparable and must not
    enter the curve-peak means.
    """
    return "+val" in (row.get("ultralytics_version") or "")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--csv", default=str(RECORD))
    p.add_argument("--grid", default="main", choices=["main", "pilot", "all"],
                   help="which training campaign to report; the record holds both and "
                        "they ran on different splits (docs/phase1-pilot-grid.md)")
    p.add_argument("--out", default=None, help="write markdown here as well as stdout")
    args = p.parse_args()

    rows = list(csv.DictReader(open(args.csv, encoding="utf-8")))
    if args.grid != "all":
        rows = [r for r in rows if r["grid"] == args.grid]

    lines = [f"Campaign: **{args.grid}**"
             + ("  — both grids, which ran on different splits; see "
                "`docs/phase1-pilot-grid.md`" if args.grid == "all" else ""), "",
             "| run | Table 1 mAP50-95 | curve peak | delta | best ep | last ep | gap | stop reason | admissible |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in rows:
        note = r["admissible"]
        if _mixed_version(r):
            note += " [CURVE ON 8.4.7 SCALE]"
        delta = float(r["map50_95"]) - float(r["best_map50_95_curve"])
        lines.append(
            f"| {r['run_id']} | {float(r['map50_95']):.5f} | {float(r['best_map50_95_curve']):.5f} | "
            f"{delta:+.5f} | {r['best_epoch']} | {r['last_epoch']} | {r['patience_gap']} | "
            f"{r['stop_reason']} | {note} |")

    comparable = [r for r in rows if not _mixed_version(r)]
    deltas = [float(r["map50_95"]) - float(r["best_map50_95_curve"]) for r in comparable]
    lines += ["", f"Explicit-val minus curve-peak across {len(deltas)} runs: "
                  f"mean {st.mean(deltas):+.5f}, sd {st.stdev(deltas):.5f}, "
                  f"range {min(deltas):+.5f}..{max(deltas):+.5f}", ""]

    # the offset splits by model family, not by machine — worth showing separately
    lines += ["### Offset by family", "", "| family | n | mean delta |", "|---|---|---|"]
    for fam in ("yolov8", "yolov9", "yolov10", "yolo11", "yolo12", "yolo26"):
        d = [float(r["map50_95"]) - float(r["best_map50_95_curve"])
             for r in comparable if r["variant"].startswith(fam)]
        if d:
            lines.append(f"| `{fam}*` | {len(d)} | {st.mean(d):+.5f} |")
    lines.append("")

    def means(pred):
        by: dict[str, list[tuple[float, float]]] = {}
        for r in comparable:
            if not pred(r):
                continue
            by.setdefault(r["variant"], []).append(
                (float(r["map50_95"]), float(r["best_map50_95_curve"])))
        return by

    for label, pred in [("ALL rows", lambda r: True),
                        ("ADMISSIBLE rows only", lambda r: r["admissible"] == "yes")]:
        by = means(pred)
        lines += [f"### Seed-means, {label}", "",
                  "| variant | n | Table 1 mean | curve-peak mean |", "|---|---|---|---|"]
        for v, pairs in sorted(by.items(), key=lambda kv: -st.mean(x[0] for x in kv[1])):
            lines.append(f"| {v} | {len(pairs)} | {st.mean(x[0] for x in pairs):.4f} | "
                         f"{st.mean(x[1] for x in pairs):.4f} |")
        order_t1 = [v for v, _ in sorted(by.items(), key=lambda kv: -st.mean(x[0] for x in kv[1]))]
        order_cv = [v for v, _ in sorted(by.items(), key=lambda kv: -st.mean(x[1] for x in kv[1]))]
        lines += ["", f"ranking identical under both metrics: **{order_t1 == order_cv}**"]
        if order_t1 != order_cv:
            lines += [f"- Table 1 order: {' > '.join(order_t1)}", f"- curve-peak order: {' > '.join(order_cv)}"]
        lines.append("")

    text = "\n".join(lines)
    print(text)
    if args.out:
        Path(args.out).write_text(text + "\n", encoding="utf-8")
        print(f"\n[robustness] wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
