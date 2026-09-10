"""G5 -- assert the held-out run appears in no training or selection list.

Pre-registered at `docs/prereg-phase3-retrain-2026-09-10.md` section 3, gate G5.
Stop rule 5 of that document has no remedy: a pohang04 number looked at before the
freeze commit voids the holdout permanently. This script is the mechanical guard,
and section 3 requires it to run **as a precondition on each training invocation**,
not once by hand -- so it is a library call first and a CLI second.

Scope, measured 2026-09-10 rather than assumed. The pre-registration named five
lists. There are sixteen, and the master `visible/train.txt` -- the one every
derived stride list is cut from -- was not among the five:

    Pohang_dataset/visible/train.txt              19,687 pohang04 rows
    runs/derived/data_vis_train_stride2.txt        9,841
    runs/derived/data_vis_train_stride5.txt        3,943
    runs/derived/fullres_vis_train_stride5.txt     3,937
    ... and twelve more

Enumerating by hand is what produced a list of five, so this script does not take a
list of files. It walks the split roots and checks every list it finds, which means a
list added later is covered without anyone remembering to add it here.

IR lists carry zero pohang04 rows and always will -- the run has no thermal frames
extracted -- but they are checked anyway, because the whole point of Stage 4 is that
thermal frames are about to be created.

Exit codes: 0 clean, 1 contaminated, 2 could not check (which is not a pass).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

#: Directories walked for split/selection lists. A `.txt` under an `images/` or
#: `labels/` directory is annotation data, not a list, and is skipped.
SPLIT_ROOTS = (
    ROOT / "Pohang_dataset",
    ROOT / "runs" / "derived",
    ROOT / "runs" / "derived_day",
    ROOT / "runs" / "bench_bundle",
)

#: The run held out by the pre-registration. A tuple so a second holdout is a
#: one-line change rather than a rewrite.
HELD_OUT = ("pohang04",)

#: `images/` and `labels/` hold annotation data; `meta/` holds the upstream release
#: record (timestamps, pairs.csv, class audits). None are split lists, and all three
#: mention run names constantly -- checking them manufactures a failure that cannot
#: be cleared, which would train operators to ignore this gate.
SKIP_PARTS = ("images", "labels", "meta")


def split_lists(roots=SPLIT_ROOTS) -> list[Path]:
    """Every file that looks like an image-list, under `roots`."""
    out: list[Path] = []
    for root in roots:
        if not root.is_dir():
            continue
        for p in root.rglob("*.txt"):
            if any(part in SKIP_PARTS for part in p.relative_to(root).parts):
                continue
            out.append(p)
    return sorted(out)


def scan(paths: list[Path], held_out=HELD_OUT) -> list[tuple[Path, str, int, int]]:
    """(path, run, n_hits, n_rows) for every list holding a held-out frame."""
    hits = []
    for p in paths:
        try:
            rows = [ln for ln in p.read_text(encoding="utf-8",
                                             errors="replace").splitlines() if ln.strip()]
        except OSError:
            continue
        for run in held_out:
            n = sum(1 for ln in rows if run in ln)
            if n:
                hits.append((p, run, n, len(rows)))
    return hits


def assert_clean(roots=SPLIT_ROOTS, held_out=HELD_OUT) -> None:
    """Raise unless no list under `roots` mentions a held-out run.

    Call this at the top of a training entry point. It raises rather than exits so
    a caller can log it; the CLI below turns it into an exit code.
    """
    hits = scan(split_lists(roots), held_out)
    if hits:
        worst = "\n".join(f"    {p.relative_to(ROOT)}: {n} {run} rows of {tot}"
                          for p, run, n, tot in hits)
        raise RuntimeError(
            f"G5 FAILED -- held-out run present in {len(hits)} list(s):\n{worst}\n"
            "  See docs/prereg-phase3-retrain-2026-09-10.md section 3 (G5). "
            "No training starts while this fails."
        )


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", action="append", default=None,
                    help=f"run to hold out (repeatable); default {HELD_OUT}")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the verdict line")
    args = ap.parse_args()

    held_out = tuple(args.run) if args.run else HELD_OUT
    paths = split_lists()
    if not paths:
        print("G5 INDETERMINATE: no split lists found -- checked nothing, "
              "which is not a pass.", file=sys.stderr)
        return 2

    hits = scan(paths, held_out)
    if not args.quiet:
        print(f"[G5] {len(paths)} lists checked, holding out {', '.join(held_out)}")
        for p, run, n, tot in sorted(hits, key=lambda h: -h[2]):
            print(f"     {n:>7,} / {tot:>7,}  {run}  {p.relative_to(ROOT)}")

    if hits:
        total = sum(h[2] for h in hits)
        print(f"[G5] FAIL: {total:,} held-out rows across {len(hits)} of "
              f"{len(paths)} lists. No training starts.")
        return 1
    print(f"[G5] PASS: 0 held-out rows across {len(paths)} lists.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
