"""Split-leakage audit (decision D6/D6-rev; plan B6-2).

10 Hz video makes random frame-level splits leak near-duplicate frames across
train/val/test and inflate every reported number. This audit verifies a split
that already exists (Laksh's custom split, answer A2-4) instead of trusting it:

  1. no image appears in more than one split (exact-path and stem collisions);
  2. within each recording run, no train frame lies within `min_gap_frames`
     (time-scaled) of a val/test frame — block splits pass, random splits fail
     loudly;
  3. run x split occupancy matrix, so distribution skew (e.g. night frames all
     in train) is visible rather than hidden.

Temporal gaps are measured in filename-derived time units: the per-run median
inter-frame delta defines one "frame", so the check works for epoch-timestamp
stems and plain frame indices alike (see dataset_requirement.md).
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path
from statistics import median

from uqfusion.data.lists import frame_ordinal, run_key, split_image_list

SPLITS = ("train", "val", "test")


def audit_split(data: dict, min_gap_frames: int = 100) -> dict:
    """Run all checks on one dataset yaml. Returns a report dict; report["ok"] is the verdict."""
    present = [s for s in SPLITS if data.get(s) is not None]
    images = {s: split_image_list(data, s) for s in present}

    report: dict = {
        "yaml": data["_yaml_path"],
        "splits_present": present,
        "sizes": {s: len(images[s]) for s in present},
        "missing_splits": [s for s in SPLITS if s not in present],
        "duplicates": [],
        "temporal_violations": [],
        "runs_without_ordinals": [],
        "run_matrix": {},
        "ok": True,
    }

    # -- 1. cross-split duplicates (path level, then stem level within a run) --
    seen: dict[str, str] = {}
    for split in present:
        for img in images[split]:
            key = str(img).lower()
            if key in seen and seen[key] != split:
                report["duplicates"].append(f"{img} in both '{seen[key]}' and '{split}'")
            seen[key] = split
    stem_seen: dict[tuple[str, str], str] = {}
    for split in present:
        for img in images[split]:
            key = (run_key(img), img.stem.lower())
            if key in stem_seen and stem_seen[key] != split:
                report["duplicates"].append(
                    f"stem '{key[1]}' of run '{key[0]}' in both '{stem_seen[key]}' and '{split}'"
                )
            stem_seen[key] = split

    # -- 2 & 3. per-run temporal proximity + occupancy matrix ------------------
    by_run: dict[str, list[tuple[float, str, Path]]] = defaultdict(list)
    matrix: dict[str, dict[str, int]] = defaultdict(lambda: {s: 0 for s in present})
    for split in present:
        for img in images[split]:
            run = run_key(img)
            matrix[run][split] += 1
            ordinal = frame_ordinal(img)
            if ordinal is not None:
                by_run[run].append((ordinal, split, img))
    report["run_matrix"] = {run: dict(counts) for run, counts in sorted(matrix.items())}

    for run, frames in sorted(by_run.items()):
        if len(frames) < 2:
            continue
        n_with_ordinal = len(frames)
        n_total = sum(matrix[run].values())
        if n_with_ordinal < n_total:
            report["runs_without_ordinals"].append(
                f"{run}: {n_total - n_with_ordinal}/{n_total} frames have no numeric stem — temporal check partial"
            )
        frames.sort(key=lambda t: t[0])
        deltas = [b[0] - a[0] for a, b in zip(frames, frames[1:]) if b[0] > a[0]]
        if not deltas:
            continue
        min_gap_units = median(deltas) * min_gap_frames
        train_like = {"train"}
        eval_like = {"val", "test"}
        for a, b in zip(frames, frames[1:]):
            crosses = (a[1] in train_like and b[1] in eval_like) or (a[1] in eval_like and b[1] in train_like)
            if crosses and (b[0] - a[0]) < min_gap_units:
                report["temporal_violations"].append(
                    f"{run}: gap {(b[0] - a[0]):.6g} < required {min_gap_units:.6g} between "
                    f"{a[1]} frame {a[2].name} and {b[1]} frame {b[2].name}"
                )

    if report["duplicates"] or report["temporal_violations"]:
        report["ok"] = False
    return report


def format_report(report: dict, max_examples: int = 20) -> str:
    lines = [f"# Split audit — {report['yaml']}", ""]
    lines.append(f"Splits present: {', '.join(report['splits_present'])}  "
                 f"(sizes: {report['sizes']})")
    if report["missing_splits"]:
        lines.append(f"**MISSING SPLITS: {report['missing_splits']}** — "
                     "a three-way train/val/test split is required (dataset_requirement.md §3); "
                     "val drives Table 1 ranking + early stopping, test stays untouched until Phase 4.")
    lines.append("")
    lines.append("## Run x split occupancy")
    for run, counts in report["run_matrix"].items():
        lines.append(f"  {run}: " + ", ".join(f"{s}={n}" for s, n in counts.items()))
    lines.append("")

    for key, title in (("duplicates", "Cross-split duplicates"),
                       ("temporal_violations", f"Temporal-proximity violations")):
        items = report[key]
        lines.append(f"## {title}: {len(items)}")
        for item in items[:max_examples]:
            lines.append(f"  - {item}")
        if len(items) > max_examples:
            lines.append(f"  ... and {len(items) - max_examples} more")
        lines.append("")

    for warn in report["runs_without_ordinals"]:
        lines.append(f"WARNING: {warn}")

    lines.append("")
    lines.append("VERDICT: " + ("PASS — split is usable under D6-rev" if report["ok"]
                                else "FAIL — fix the split before any training run"))
    return "\n".join(lines)
