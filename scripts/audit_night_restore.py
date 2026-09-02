"""Idea I5 -- what is actually recoverable at night, before anything is retrained.

VIS scores exactly **0.0000** on all 1032 night frames with both detectors. That
is a dataset decision, not a fusion result: `filter_night_boxes.py --cut-dark
pohang01:100` removed 132k boxes from the VIS TRAIN labels on purpose, because
they were annotations projected from other sensors for ships the camera cannot
see. The consequence is architectural -- the gate's entire night apparatus (the
photometric axis, the `night AND (dark OR veil)` repair, the `lap_over_var`
fallback, the authority bound) exists to defend a stream that is identically
zero, and 46% of the benchmark cannot test any fusion property at all.

**This script retrains nothing and writes no labels.** It audits, so the
expensive decision is made on numbers:

  1. How many boxes are recoverable, from the `*.pre_visfilter` backups the
     filter left behind, and how they distribute over runs and frames.
  2. Whether they are *learnable* -- the filter's own per-box scores
     (`runs/visfilter/box_scores.csv`) say how dark, how flat and how
     low-contrast each dropped box was. Restoring boxes the camera genuinely
     cannot see re-creates the label noise the filter was written to remove.
  3. What the VAL night GT already contains, since val labels were never
     filtered -- that is the ceiling any night model would be scored against.

The three outcomes:

  * a large recoverable set with real photometric structure -> a night fine-tune
    is worth its GPU hours, and the night half of the benchmark becomes live;
  * recoverable boxes that are uniformly black and flat -> the filter was right,
    night VIS is physically dead, and the correct move is to STOP defending it:
    delete the night apparatus rather than improve it;
  * a middle band -> restore only above a threshold, as ignore-regions rather
    than positives.

Usage:
    python scripts/audit_night_restore.py
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _ideas_common import NIGHT_RUNS, fmt, md_table, write_md   # noqa: E402

ROOT = Path(__file__).resolve().parents[1]


def count_boxes(p: Path) -> int:
    try:
        return sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
    except OSError:
        return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--labels-root", default="Pohang_dataset/visible/labels")
    ap.add_argument("--manifest", default="runs/visfilter/visfilter_manifest.json")
    ap.add_argument("--box-scores", default="runs/visfilter/box_scores.csv")
    ap.add_argument("--out", default="runs/eval/night_restore_audit.md")
    args = ap.parse_args()
    t0 = time.time()

    root = ROOT / args.labels_root
    backups = sorted(root.rglob("*.pre_visfilter"))
    secs = [f"Labels root: `{args.labels_root}`  \n"
            f"{len(backups)} `*.pre_visfilter` backups found -- the filter is reversible."]

    # ---- 1. what was dropped, per run ------------------------------------
    per_run: dict[str, list[int]] = {}
    for b in backups:
        run = b.parent.name
        before = count_boxes(b)
        after = count_boxes(b.with_suffix(""))
        per_run.setdefault(run, [0, 0, 0, 0])
        a = per_run[run]
        a[0] += 1
        a[1] += before
        a[2] += after
        a[3] += 1 if after == 0 and before > 0 else 0
    rows = [[r, v[0], v[1], v[2], v[1] - v[2],
             fmt((v[1] - v[2]) / max(v[1], 1), 3), v[3],
             "night" if r in NIGHT_RUNS else "day"]
            for r, v in sorted(per_run.items())]
    tot = [sum(v[i] for v in per_run.values()) for i in range(4)]
    rows.append(["**total**", tot[0], tot[1], tot[2], tot[1] - tot[2],
                 fmt((tot[1] - tot[2]) / max(tot[1], 1), 3), tot[3], ""])
    secs.append("## 1. What the filter dropped, per run\n\n"
                "Only TRAIN labels were touched; val/test were left honest.\n\n"
                + md_table(["run", "files", "boxes before", "boxes after", "dropped",
                            "drop rate", "emptied frames", "day/night"], rows))

    # ---- 2. are the dropped boxes learnable? -----------------------------
    bs = ROOT / args.box_scores
    if bs.is_file():
        import csv
        cols: dict[str, list[float]] = {}
        flag = []
        with open(bs, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                for k in ("box_mean", "grad", "contrast"):
                    if k in row:
                        try:
                            cols.setdefault(k, []).append(float(row[k]))
                        except (TypeError, ValueError):
                            cols.setdefault(k, []).append(np.nan)
                f = str(row.get("flagged", "")).strip().lower()
                flag.append(f in ("1", "true", "yes"))
        flag = np.asarray(flag, bool)
        rows = []
        for k, v in cols.items():
            v = np.asarray(v, float)
            if len(v) != len(flag):
                continue
            fl, kp = v[flag], v[~flag]
            rows.append([k, len(fl), fmt(np.nanmedian(fl), 2), fmt(np.nanpercentile(fl, 90), 2),
                         len(kp), fmt(np.nanmedian(kp), 2), fmt(np.nanpercentile(kp, 10), 2)])
        secs.append("## 2. Were the dropped boxes learnable?\n\n"
                    "The filter's own per-box scores. `FLAGGED` required ALL THREE of "
                    "intensity, gradient and local contrast to fail, so it was already "
                    "conservative. If the flagged p90 sits below the kept p10 on every "
                    "axis, the two populations do not overlap and the filter was right.\n\n"
                    + md_table(["axis", "n flagged", "flagged median", "flagged p90",
                                "n kept", "kept median", "kept p10"], rows))
    else:
        secs.append(f"## 2. Were the dropped boxes learnable?\n\n`{args.box_scores}` "
                    "absent -- re-run `scripts/filter_night_boxes.py` in audit mode to "
                    "regenerate the per-box scores.")

    # ---- 3. the val night ceiling ----------------------------------------
    try:
        sys.path.insert(0, str(ROOT / "src"))
        from uqfusion.config import load_config, resolve_data_yaml
        from uqfusion.data.lists import load_data_yaml, split_image_list
        cfg = load_config()
        val = split_image_list(load_data_yaml(resolve_data_yaml(cfg, "vis")), "val")
        night = [p for p in val if Path(p).name.split("_")[0] in NIGHT_RUNS]
        nb = Counter()
        total = 0
        for p in night:
            lp = Path(str(p).replace("images", "labels")).with_suffix(".txt")
            n = count_boxes(lp)
            total += n
            nb[min(n, 5)] += 1
        rows = [[f"{k}{'+' if k == 5 else ''}", v] for k, v in sorted(nb.items())]
        secs.append("## 3. The night VAL ceiling (labels never filtered)\n\n"
                    f"{len(night)} night val frames carrying **{total}** GT boxes. This is "
                    "what a night model would be scored against; VIS currently recovers "
                    "0.0000 of it.\n\n"
                    + md_table(["boxes in frame", "frames"], rows))
    except Exception as e:  # noqa: BLE001
        secs.append(f"## 3. The night VAL ceiling\n\nCould not read the val split: "
                    f"{type(e).__name__}: {e}")

    mf = ROOT / args.manifest
    if mf.is_file():
        m = json.loads(mf.read_text(encoding="utf-8"))
        keys = [k for k in ("thresholds", "mode", "cut", "n_files", "n_dropped") if k in m]
        secs.append("## 4. The filter's own manifest\n\n```json\n"
                    + json.dumps({k: m[k] for k in keys}, indent=2) + "\n```")

    secs.append(
        "## 5. Decision rule\n\n"
        "* **Flagged and kept populations overlap, and the recoverable count is large** "
        "-> restore above a threshold as ignore-regions (not positives) and fine-tune. "
        "The night half of the benchmark becomes testable and the authority bound "
        "becomes priceable (experiment log SS10.3).\n"
        "* **The populations are cleanly separated** -> the filter was right, night VIS "
        "is physically dead, and the correct architectural move is the OPPOSITE of a "
        "retrain: delete the night apparatus and state that night is single-sensor by "
        "physics rather than by omission.\n\n"
        "Either way this is a decision the audit can make. **Nothing here writes a label "
        "or launches a training run**; `filter_night_boxes.py --restore` is the "
        "documented undo if a restore is chosen.")
    secs.append(f"---\n\n_Generated by `scripts/audit_night_restore.py` in "
                f"{time.time() - t0:.1f}s._")
    write_md(args.out, "Night label restore -- audit before retrain (I5)", secs)
    return 0


if __name__ == "__main__":
    sys.exit(main())
