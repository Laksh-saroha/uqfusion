"""G5 -- write pohang04-free copies of the VIS split lists, and the yaml over them.

Pre-registered at `docs/prereg-phase3-retrain-2026-09-10.md` section 5.1
("`data_vis_stride2.yaml` regenerated **without** pohang04") and gate G5.

**This filters; it does not regenerate.** The distinction matters more than it looks.
Rebuilding stride-2 from the master list would re-run the stride selection over a
smaller pool and pick a *different* subset of pohang00-03 frames than every measurement
already in the record was taken on. Dropping the pohang04 rows from the existing
stride-2 list leaves every surviving frame exactly where it was, so the only difference
between the old training set and the new one is the held-out run -- which is the only
difference Phase 3 is entitled to introduce here.

**Nothing is overwritten.** Outputs are new `_p04out` filenames beside the originals,
and the originals stay valid for reproducing pre-Phase-3 results. Each input keeps its
own path style: `visible/val.txt` stores relative paths and the derived lists store
absolute ones, so lines are dropped, never rewritten.

The IR side needs no equivalent. `runs/derived/ir_shiponly/{train_stride2,val,test}.txt`
carry zero pohang04 rows and will until Stage 4 extracts thermal frames -- the run has
no IR labels at all (Amendment 2).

Composition is reported rather than assumed. pohang04 is daylight throughout, so removing
it does not shrink val evenly: the night share rises, and section 8 requires day and night
to be reported separately, never pooled.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

HELD_OUT = "pohang04"
SUFFIX = "_p04out"

#: (source list, output list). Only the three lists `data_vis_stride2.yaml` actually
#: points at -- the other contaminated lists are selection sets for earlier phases and
#: are not inputs to Stage 2. G5's gate still covers all of them.
LISTS = (
    (ROOT / "runs/derived/data_vis_train_stride2.txt",
     ROOT / f"runs/derived/data_vis_train_stride2{SUFFIX}.txt"),
    (ROOT / "Pohang_dataset/visible/val.txt",
     ROOT / f"runs/derived/vis_val{SUFFIX}.txt"),
    (ROOT / "Pohang_dataset/visible/test.txt",
     ROOT / f"runs/derived/vis_test{SUFFIX}.txt"),
)

YAML_OUT = ROOT / f"runs/derived/data_vis_stride2{SUFFIX}.yaml"

#: Night reference. Every night frame in val comes from pohang01; the day/night call is
#: the one Phase 1 made, reused rather than recomputed so the two agree by construction.
NIGHT_REF = ROOT / "runs/derived/phase1val_night.txt"


def frame_key(line: str) -> str:
    """`pohang01_L_007724` from any path style."""
    return Path(line.strip().replace("\\", "/")).stem


def run_of(line: str) -> str:
    k = frame_key(line)
    return k.split("_")[0] if "_" in k else "?"


def absolutise(line: str, src: Path) -> str:
    """Resolve a list line against the list it came from, and emit an absolute path.

    This is not cosmetic. `Pohang_dataset/visible/val.txt` stores relative paths
    (`./images/pohang00/...`) which a loader resolves against the *list file's* directory.
    Writing those lines unchanged into `runs/derived/` repoints every one of them at
    `runs/derived/images/...`, which does not exist -- the filtered list then looks
    perfectly well-formed and loads zero images. Measured: it got through G5 and failed
    only when a training run tried to read it.
    """
    p = Path(line.strip())
    return str(p if p.is_absolute() else (src.parent / p).resolve())


def filter_list(src: Path, dst: Path, held_out: str, dry: bool) -> dict:
    raw = src.read_text(encoding="utf-8").splitlines()
    kept = [absolutise(ln, src) for ln in raw if ln.strip() and held_out not in ln]
    dropped = len([ln for ln in raw if ln.strip()]) - len(kept)
    if not dry:
        dst.parent.mkdir(parents=True, exist_ok=True)
        dst.write_text("\n".join(kept) + "\n", encoding="utf-8")
    runs: dict[str, int] = {}
    for ln in kept:
        runs[run_of(ln)] = runs.get(run_of(ln), 0) + 1
    return {"src": src, "dst": dst, "in": len(raw), "kept": len(kept),
            "dropped": dropped, "runs": runs}


def night_split(lines: list[str], night: set[str]) -> tuple[int, int]:
    n = sum(1 for ln in lines if frame_key(ln) in night)
    return len(lines) - n, n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="report what would change; write nothing")
    args = ap.parse_args()

    missing = [s for s, _ in LISTS if not s.is_file()]
    if missing:
        for m in missing:
            print(f"missing input list: {m}", file=sys.stderr)
        return 2

    night = {frame_key(ln) for ln in NIGHT_REF.read_text(encoding="utf-8").splitlines()
             if ln.strip()} if NIGHT_REF.is_file() else set()

    results = [filter_list(s, d, HELD_OUT, args.dry_run) for s, d in LISTS]

    print(f"{'list':<44}{'in':>9}{'dropped':>10}{'kept':>9}")
    for r in results:
        print(f"{r['dst'].name:<44}{r['in']:>9,}{r['dropped']:>10,}{r['kept']:>9,}")

    print("\nper-run composition after filtering:")
    for r in results:
        parts = "  ".join(f"{k} {v:,}" for k, v in sorted(r["runs"].items()))
        print(f"  {r['dst'].name:<42}{parts}")

    if night:
        print("\nday/night shift (night reference: phase1val_night.txt):")
        for r in results:
            before = [ln for ln in r["src"].read_text(encoding="utf-8").splitlines()
                      if ln.strip()]
            after = [ln for ln in before if HELD_OUT not in ln]
            d0, n0 = night_split(before, night)
            d1, n1 = night_split(after, night)
            if n0 == 0 and n1 == 0:
                continue
            print(f"  {r['dst'].name}")
            print(f"     before: day {d0:,}  night {n0:,}  ({n0 / max(len(before), 1):.1%} night)")
            print(f"     after : day {d1:,}  night {n1:,}  ({n1 / max(len(after), 1):.1%} night)")

    if not args.dry_run:
        YAML_OUT.write_text(
            "# Stage 2 VIS data, pohang04 held out (G5).\n"
            "# Filtered from data_vis_stride2.yaml's lists, NOT regenerated: every\n"
            "# surviving frame keeps the stride-2 selection earlier results were taken\n"
            "# on, so the held-out run is the only difference.\n"
            "# docs/prereg-phase3-retrain-2026-09-10.md sections 3 (G5) and 5.1.\n"
            f"train: {LISTS[0][1]}\n"
            f"val: {LISTS[1][1]}\n"
            f"test: {LISTS[2][1]}\n"
            "names:\n  0: ship\n  1: buoy\n",
            encoding="utf-8")
        print(f"\nwrote {YAML_OUT}")
    else:
        print("\n[dry-run] nothing written")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
