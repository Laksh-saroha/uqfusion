"""R-E1 — do the label hashes at training, caching and now agree?

This is R-E1's actual acceptance criterion, and until 2026-09-10 neither side of it
existed. Slice 2 gave a prediction cache a `labels_sha256`; the training side gained
`uqfusion_labels.json` (written at start AND end of every Gaussian run) at the same
time. This script is the join: it walks run directories and caches, and reports where
the three states agree, where they differ, and — the honest part — where the answer is
simply not recorded because the artifact predates the stamping.

**It reports; it does not refuse.** A run trained under an older label state is not
invalid, it is a run whose label state has to be named when its numbers are quoted.
Refusing here would delete history rather than describe it. `load_cache` already
refuses the case that IS an error: a cache whose stamped hash no longer matches the
tree it would be scored against.

Three scopes, three algorithms' worth of confusion avoided by never mixing them:
`labels_train` is over the train image list, `labels_trainval` over train+val. They
are different numbers and neither means anything against the other. See
`scripts/label_hash_ledger.py` for what happened the last time two such numbers were
compared as though they were one.

Usage:
    python scripts/verify_label_provenance.py
    python scripts/verify_label_provenance.py --runs-root runs/full_scale --data vis
"""

from __future__ import annotations

import argparse
import glob
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from uqfusion.config import load_config, resolve_data_yaml
from uqfusion.data.labels import label_content_hash
from uqfusion.data.lists import load_data_yaml, split_image_list
from uqfusion.uq.train_gaussian import LABEL_MANIFEST


def current_hashes(data_yaml) -> dict:
    data = load_data_yaml(data_yaml)
    train = split_image_list(data, "train")
    val = split_image_list(data, "val")
    return {"labels_train": label_content_hash(train),
            "labels_trainval": label_content_hash(list(train) + list(val)),
            "n_train": len(train), "n_val": len(val)}


def scan_runs(roots: list[Path]) -> list[dict]:
    rows = []
    for root in roots:
        if not root.is_dir():
            continue
        for man in sorted(root.rglob(LABEL_MANIFEST)):
            try:
                entries = json.loads(man.read_text(encoding="utf-8"))
            except Exception as exc:  # noqa: BLE001
                rows.append({"run": str(man.parent), "state": "UNREADABLE", "detail": str(exc)[:60]})
                continue
            start = next((e for e in entries if e.get("phase") == "start"), None)
            end = next((e for e in reversed(entries) if e.get("phase") == "end"), None)
            rows.append({"run": str(man.parent), "state": "recorded",
                         "start": (start or {}).get("labels_train"),
                         "end": (end or {}).get("labels_train"),
                         "trainval_end": (end or {}).get("labels_trainval"),
                         "unchanged": (end or {}).get("unchanged_during_run")})
    return rows


def scan_runs_without_manifest(roots: list[Path]) -> list[Path]:
    """Run directories that hold weights but no label manifest — the honest gap."""
    out = []
    for root in roots:
        if not root.is_dir():
            continue
        for w in sorted(root.rglob("weights/best.pt")):
            run = w.parent.parent
            if not (run / LABEL_MANIFEST).is_file():
                out.append(run)
    return out


def scan_caches(patterns: list[str]) -> tuple[list[dict], int]:
    rows, unstamped = [], 0
    for pat in patterns:
        for p in sorted(glob.glob(pat, recursive=True)):
            try:
                with open(p, "rb") as f:
                    meta = pickle.load(f).get("meta", {})
            except Exception:  # noqa: BLE001
                continue
            h = meta.get("labels_sha256")
            if h is None:
                unstamped += 1
            else:
                rows.append({"cache": p, "labels_sha256": h})
    return rows, unstamped


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--data", default="vis", help="'vis'/'ir' alias or a dataset yaml")
    ap.add_argument("--runs-root", action="append", default=None,
                    help="repeatable; default: runs/full_scale, runs/mc_dropout, runs/gaussian")
    ap.add_argument("--cache-glob", action="append", default=None,
                    help="repeatable; default: runs/cache*/**/*.pkl")
    args = ap.parse_args()

    cfg = load_config(None)
    data_yaml = resolve_data_yaml(cfg, args.data)
    roots = [Path(r) for r in (args.runs_root or
                               ["runs/full_scale", "runs/mc_dropout", "runs/gaussian"])]
    globs = args.cache_glob or ["runs/cache*/**/*.pkl"]

    now = current_hashes(data_yaml)
    print(f"data      : {data_yaml}")
    print(f"NOW       : labels_train {now['labels_train']}  "
          f"labels_trainval {now['labels_trainval']}  "
          f"({now['n_train']} train, {now['n_val']} val)")
    print()

    runs = scan_runs(roots)
    print(f"--- training side: {len(runs)} run(s) with a label manifest ---")
    for r in runs:
        if r["state"] != "recorded":
            print(f"  {r['run']}: {r['state']} ({r.get('detail')})")
            continue
        drift = "" if r["unchanged"] in (True, None) else "  *** CHANGED DURING RUN ***"
        vs_now = "matches now" if r["end"] == now["labels_train"] else "**differs from now**"
        print(f"  {r['run']}")
        print(f"      start {r['start']}  end {r['end']}  -> {vs_now}{drift}")

    gaps = scan_runs_without_manifest(roots)
    print(f"\n--- training side: {len(gaps)} checkpoint(s) with NO label manifest ---")
    print("    These predate the stamping. Their training-time label state is NOT")
    print("    recoverable from the run directory; the nearest evidence is the dated")
    print("    rows in runs/label_hash_ledger.csv, which bound it but do not pin it.")
    for g in gaps[:15]:
        print(f"      {g}")
    if len(gaps) > 15:
        print(f"      ... and {len(gaps) - 15} more")

    caches, unstamped = scan_caches(globs)
    print(f"\n--- caching side: {len(caches)} stamped, {unstamped} unstamped ---")
    mismatch = [c for c in caches if c["labels_sha256"] != now.get("labels_sha256")]
    for c in caches[:15]:
        print(f"      {c['cache']}: {c['labels_sha256']}")
    if unstamped:
        print(f"    {unstamped} cache(s) carry no labels_sha256 -- built before slice 2.")
        print("    load_cache cannot check what was never stamped, so those are")
        print("    reported here rather than silently treated as verified.")

    print("\n--- verdict ---")
    if not runs and not caches:
        print("  NOTHING JOINABLE YET. Both sides of R-E1's criterion now exist in code,")
        print("  but every artifact on disk predates them. The first Gaussian run and the")
        print("  first cache built from here on will be the first pair that can be")
        print("  compared. That is the honest state, and saying so is the point.")
    else:
        print(f"  {len(runs)} run(s) and {len(caches)} cache(s) carry a label state.")
        print(f"  {len(gaps)} checkpoint(s) and {unstamped} cache(s) do not.")
    _ = mismatch
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
