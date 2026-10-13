"""Does each Phase-1 benchmark row resolve to one manifest and a compatible metric?

R-F3 / F15 acceptance check. The review's wording is precise and testable:

    "Each table row resolves to one manifest and a compatible metric. A
    comparison can be reproduced without recovering undocumented pilot data."

This walks `phase1_benchmark/results.csv` and answers it per campaign rather
than asserting it. It reports; it gates nothing, because the historical rows
cannot be repaired -- the machine that produced the pilot was wiped. The point
is to make the gap a number in a file instead of a caveat in prose.

    python scripts/audit_phase1_manifests.py [--csv ...] [--json ...]

Three separate questions, kept separate on purpose:

  * **identity** -- is the row unique, and does anything repeat across
    campaigns? (a pooled ranking that double-counts a variant is a different
    defect from one that mixes conditions)
  * **manifest** -- can the row's training data and code be recovered from
    what the row records? A path string is not a manifest if the file at that
    path was rewritten between campaigns.
  * **metric compatibility** -- were the rows scored on one scale?
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = "phase1_benchmark/results.csv"

# Fields a row would need for its training data and code to be recoverable.
MANIFEST_FIELDS = ("data_yaml", "classes", "git_commit", "ultralytics_version",
                   "torch_version", "trained_on", "batch")
# Values that look like data but name nothing recoverable.
NULL_TOKENS = {"", "unknown", "none", "n/a", "na", "server_wiped"}


def load(path: Path) -> list[dict[str, str]]:
    with io.open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--csv", default=DEFAULT_CSV)
    ap.add_argument("--json", default=None)
    args = ap.parse_args(argv)

    path = ROOT / args.csv
    rows = load(path)
    grids = sorted({r["grid"] for r in rows})
    out: dict[str, object] = {"csv": args.csv, "n_rows": len(rows)}

    print(f"{args.csv}: {len(rows)} rows, campaigns {dict(Counter(r['grid'] for r in rows))}")
    print()

    # ---- identity ----------------------------------------------------------
    print("== identity ==")
    dup_full = {k: v for k, v in Counter(
        (r["grid"], r["variant"], r["seed"]) for r in rows).items() if v > 1}
    dup_cross = {k: v for k, v in Counter(
        (r["variant"], r["seed"]) for r in rows).items() if v > 1}
    per_grid_variants = {g: {r["variant"] for r in rows if r["grid"] == g} for g in grids}
    shared = set.intersection(*per_grid_variants.values()) if len(grids) > 1 else set()
    print(f"  repeated (grid, variant, seed) : {dup_full or 'none'}")
    print(f"  repeated (variant, seed)       : {dup_cross or 'none'}")
    print(f"  variants in more than one grid : {sorted(shared) or 'none'}")
    for g in grids:
        print(f"  {g:<6} {len(per_grid_variants[g]):>3} variants, "
              f"{sum(1 for r in rows if r['grid'] == g):>3} rows")
    total_variants = len(set().union(*per_grid_variants.values()))
    print(f"  ladder covered by POOLING only : {total_variants} variants "
          f"({' + '.join(str(len(per_grid_variants[g])) for g in grids)})")
    out["identity"] = {
        "duplicate_rows": {str(k): v for k, v in dup_full.items()},
        "duplicate_variant_seed": {str(k): v for k, v in dup_cross.items()},
        "variants_in_multiple_grids": sorted(shared),
        "variants_per_grid": {g: len(v) for g, v in per_grid_variants.items()},
        "variants_total_pooled": total_variants,
    }
    print()

    # ---- manifest ----------------------------------------------------------
    print("== manifest recoverability ==")
    unrecoverable: dict[str, list[str]] = defaultdict(list)
    for r in rows:
        for f in MANIFEST_FIELDS:
            if (r.get(f) or "").strip().lower() in NULL_TOKENS:
                unrecoverable[f].append(r["run_id"])
    for f in MANIFEST_FIELDS:
        ids = unrecoverable.get(f, [])
        if ids:
            by_grid = Counter(r["grid"] for r in rows if r["run_id"] in set(ids))
            print(f"  {f:<20} {len(ids):>3}/{len(rows)} rows name nothing recoverable  {dict(by_grid)}")
    if not unrecoverable:
        print("  every row names a recoverable value in every manifest field")

    # A path string shared by two campaigns whose splits differ is not an identity.
    shared_yaml = {v for v in {r["data_yaml"] for r in rows}
                   if len({r["grid"] for r in rows if r["data_yaml"] == v}) > 1}
    print()
    for v in sorted(shared_yaml):
        n = sum(1 for r in rows if r["data_yaml"] == v)
        print(f"  !! data_yaml '{v}' is claimed identically by {n} rows across "
              f"{len({r['grid'] for r in rows if r['data_yaml'] == v})} campaigns.")
        print("     The file at that path was regenerated by the 2026-07-14 resplit, so the")
        print("     STRING matches while the SPLIT does not. A path is not a fingerprint.")
    has_fp = any(c in (rows[0] if rows else {}) for c in
                 ("split_fingerprint", "label_fingerprint_trainval", "recipe_fingerprint"))
    print(f"  fingerprint column present     : {has_fp}")
    out["manifest"] = {
        "unrecoverable_counts": {f: len(v) for f, v in unrecoverable.items()},
        "data_yaml_shared_across_campaigns": sorted(shared_yaml),
        "fingerprint_column_present": has_fp,
    }
    print()

    # ---- metric compatibility ---------------------------------------------
    print("== metric compatibility ==")
    vers = Counter((r.get("ultralytics_version") or "").strip() for r in rows)
    for v, n in vers.most_common():
        print(f"  ultralytics {v or '<EMPTY>':<18} {n:>3} rows")
    # "trained@X+valY" means the row was RE-SCORED on Y, so its reported metric is
    # on Y's scale even though training ran under X. That is a training-condition
    # difference, not a metric incompatibility -- record it as the former, because
    # overstating a defect costs the same credibility as hiding one.
    mixed = [r["run_id"] for r in rows if "+" in (r.get("ultralytics_version") or "")]
    scored = {(r.get("ultralytics_version") or "").split("+val")[-1] for r in rows}
    print(f"  distinct SCORING scales                     : {sorted(scored)}")
    print(f"  rows trained under a different version than scored: {mixed or 'none'}")
    if mixed:
        print("    -> reported metric is on the scoring scale; the in-training curve is not.")
    out["metric"] = {"versions": dict(vers), "scoring_scales": sorted(scored),
                     "retrained_under_other_version": mixed}
    print()

    # ---- verdict -----------------------------------------------------------
    print("== verdict against the acceptance check ==")
    ok_identity = not dup_full and not dup_cross and not shared
    ok_manifest = not unrecoverable and not shared_yaml
    ok_metric = len(scored) == 1
    print(f"  unique rows, no variant double-counted     : {'PASS' if ok_identity else 'FAIL'}")
    print(f"  every row resolves to one manifest         : {'PASS' if ok_manifest else 'FAIL'}")
    print(f"  one metric SCALE for reported numbers      : {'PASS' if ok_metric else 'FAIL'}")
    print(f"  one TRAINING software version              : "
          f"{'PASS' if not mixed and len(vers) == 1 else 'FAIL'}")
    print(f"  reproducible without undocumented pilot data: "
          f"{'FAIL - the pilot machine was wiped' if 'server_wiped' in {(r.get('trained_on') or '').strip() for r in rows} else 'PASS'}")
    out["verdict"] = {"identity": ok_identity, "manifest": ok_manifest, "metric": ok_metric}

    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2), encoding="utf-8")
        print(f"\nwrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
