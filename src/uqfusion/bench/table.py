"""Table 1 generation (scope §9.5): benchmark CSV(s) -> markdown, mean ± std over seeds.

Reads only the committed CSV artifacts, so the table is regenerable by anyone
from the repo + results — the Phase 1 verification gate.
"""

from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

METRIC_COLUMNS = [("precision", "Precision"), ("recall", "Recall"),
                  ("map50", "mAP@50"), ("map50_95", "mAP@50–95")]


def _mean_std(values: list[float]) -> str:
    if not values:
        return "—"
    m = mean(values)
    s = stdev(values) if len(values) > 1 else 0.0
    return f"{m:.3f} ± {s:.3f}"


def load_results(results_csv: str | Path) -> dict[str, list[dict]]:
    by_variant: dict[str, list[dict]] = defaultdict(list)
    with open(results_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            by_variant[row["variant"]].append(row)
    return by_variant


def load_fps(fps_csv: str | Path | None) -> dict[str, str]:
    """variant -> display string; prefers fp16 rows when both precisions exist."""
    if not fps_csv or not Path(fps_csv).is_file():
        return {}
    best: dict[str, dict] = {}
    with open(fps_csv, "r", encoding="utf-8", newline="") as f:
        for row in csv.DictReader(f):
            v = row["variant"]
            if v not in best or (row["half"] == "True" and best[v]["half"] != "True"):
                best[v] = row
    return {
        v: f"{row['fps']} ({row['wall_ms_per_img']} ms{', fp16' if row['half'] == 'True' else ''})"
        for v, row in best.items()
    }


def make_table1(
    results_csv: str | Path,
    fps_csv: str | Path | None = None,
    out_md: str | Path | None = None,
) -> str:
    by_variant = load_results(results_csv)
    if not by_variant:
        raise ValueError(f"no rows in {results_csv}")
    fingerprints = {(r.get("split_fingerprint") or "") for rows in by_variant.values() for r in rows}
    if len(fingerprints) > 1:
        raise ValueError(
            f"{results_csv} mixes rows from different splits "
            f"(fingerprints: {sorted(fingerprints)}) — a mean over them is meaningless. "
            "Separate the CSVs before building Table 1."
        )
    fingerprint = next(iter(fingerprints))
    class_tags = {(r.get("classes") or "all") for rows in by_variant.values() for r in rows}
    if len(class_tags) > 1:
        raise ValueError(
            f"{results_csv} mixes rows with different class filters "
            f"({sorted(class_tags)}) — metrics are not comparable. "
            "Separate the CSVs before building Table 1."
        )
    classes_tag = next(iter(class_tags))
    fps = load_fps(fps_csv)

    header = (
        "| Model | Precision | Recall | mAP@50 | mAP@50–95 | FPS (ms/img) | Params (M) | GFLOPs | Seeds |\n"
        "|---|---|---|---|---|---|---|---|---|"
    )
    lines = [header]
    for variant in sorted(by_variant):
        rows = by_variant[variant]
        cells = [variant]
        for key, _ in METRIC_COLUMNS:
            cells.append(_mean_std([float(r[key]) for r in rows if r.get(key) not in ("", None)]))
        cells.append(fps.get(variant, "—"))
        cells.append(next((r["params_m"] for r in rows if r.get("params_m")), "—"))
        cells.append(next((r["gflops"] for r in rows if r.get("gflops")), "—"))
        cells.append(str(len(rows)))
        lines.append("| " + " | ".join(cells) + " |")

    meta = rows[-1]  # any row carries the environment stamp
    lines.append("")
    lines.append(
        f"*Generated from `{Path(results_csv).name}` — ultralytics {meta.get('ultralytics_version', '?')}, "
        f"torch {meta.get('torch_version', '?')}, commit {meta.get('git_commit', '?')}, "
        f"split fingerprint {fingerprint or 'unstamped (pre-provenance CSV)'}, "
        f"classes {classes_tag}. "
        f"Mean ± std over seeds; identical data/split/imgsz/epochs/batch across rows "
        f"(plan C5; single fingerprint enforced).*"
    )
    table = "\n".join(lines)

    if out_md:
        out_md = Path(out_md)
        out_md.parent.mkdir(parents=True, exist_ok=True)
        out_md.write_text(table + "\n", encoding="utf-8")
        print(f"[table1] written -> {out_md}")
    return table
