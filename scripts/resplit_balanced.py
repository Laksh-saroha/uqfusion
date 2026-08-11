"""Re-split Pohang into interleaved temporal blocks that pass BOTH gates:
leakage (D6-rev temporal buffer) and balance (class share + box density).

Why: the original tail-block split (prepare_pohang.py) put val/test in fixed
positional windows (80-90% / 90-100% of every run). Those windows sample one
mission phase, so val came out buoy-dominated (77% vs 5% global) and sparse
(1.35 vs 8.10 boxes/frame) — verify_split_balance FAIL. A backbone selected on
that val set would be selected on the wrong signal.

Fix: cut each run's shared VIS+IR ordinal timeline into K equal blocks and
assign them in a repeating cycle of 10 (block%10==4 -> val, ==9 -> test, else
train; 80/10/10 by construction). Val/test blocks are scattered across the
whole mission, so every split samples every phase. A guard band (half on each
side) is dropped at every train<->eval boundary, so the leakage audit still
passes. K starts coarse and is raised until both gates pass (finer blocks =
better phase mixing, but more guard loss — the search finds the smallest K
that works).

Both modalities use the SAME per-run block plan (paired VIS/IR frames never
split across sets), same rule as the original prep. L/R stereo frames share a
frame index -> same ordinal -> same block, so stereo pairs can't leak either.

Guard bands are PER MODALITY (leakage is a within-modality constraint: each
detector trains on one stream, and the audit measures gaps per modality in its
own median-delta units). Sparse IR runs need wide guards; dense VIS does not.
A guard drop can never create a cross-modality train/eval conflict because the
block->split assignment itself is shared — dropping a frame only removes it.
Each run's K is capped so its blocks stay wider than its largest modality
guard (sparse runs get coarser blocks instead of losing whole eval blocks),
and a coverage gate requires every sizable run to contribute val AND test
frames in every modality it appears in.

List-only operation: rewrites {train,val,test}.txt (with the './' prefix
Ultralytics needs). No image/label files are moved. Originals backed up once
to *.txt.pre_resplit.

Paths come from config.yaml (--data-vis/--data-ir accept the usual aliases or
explicit yaml paths), so the script runs unchanged on any machine.

Usage:
    python scripts/resplit_balanced.py               # dry-run: plan + gates, no writes
    python scripts/resplit_balanced.py --execute     # write lists + re-audit on disk
    python scripts/resplit_balanced.py --audit       # verify the CURRENT on-disk split
                                                     # (leakage + balance + coverage), no writes
    python scripts/resplit_balanced.py --k 40        # force K instead of the ladder

Exit 0 = gates pass (and, with --execute, lists written + re-verified); 1 = fail.

AFTER a successful --execute you must regenerate every derived artifact built
from the old lists (e.g. runs/derived/data_vis_stride2.yaml via
scripts/make_stride_subset.py) and re-run any training that used them.
"""

from __future__ import annotations

import argparse
import shutil
import sys
from bisect import bisect_right
from pathlib import Path
from statistics import median

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from uqfusion.config import load_config, resolve_data_yaml          # noqa: E402
from uqfusion.data.audit import audit_split, format_report          # noqa: E402
from uqfusion.data.lists import (                                   # noqa: E402
    IMAGE_SUFFIXES, dataset_root, frame_ordinal, load_data_yaml, run_key,
)

SPLITS = ("train", "val", "test")
CYCLE = 10           # block assignment repeats every 10 blocks
VAL_SLOT, TEST_SLOT = 4, 9   # -> 80/10/10, val and test blocks never adjacent
K_LADDER = (20, 30, 40, 50, 60, 80, 100)
GUARD_MARGIN = 1.3   # guard = required_gap * margin (same as prepare_pohang)
GUARD_FLOOR = 150.0  # minimum guard width in ordinal units
BLOCK_GUARD_RATIO = 2.5   # per-run K cap: block width >= ratio * largest modality guard
CLASS_DEV = 0.40     # max relative deviation of a split's class share vs global
DENSITY_DEV = 0.50   # max relative deviation of a split's boxes/frame vs global
COVERAGE_MIN_RUN = 1500   # runs with >= this many frames in a modality must hit val AND test


# ---- collection ------------------------------------------------------------

def collect_frames(mod_root: Path) -> dict[str, list[tuple[float, Path]]]:
    """{run: [(ordinal, image_path), ...]} from the per-run images/<run>/ tree."""
    images_dir = mod_root / "images"
    if not images_dir.is_dir():
        raise SystemExit(f"expected per-run layout {images_dir}/<run>/ — not found")
    out: dict[str, list[tuple[float, Path]]] = {}
    for run_dir in sorted(p for p in images_dir.iterdir() if p.is_dir()):
        frames = []
        for img in run_dir.iterdir():
            if img.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            o = frame_ordinal(img)
            if o is None:
                raise SystemExit(f"no numeric ordinal in filename: {img}")
            frames.append((o, img))
        if frames:
            out[run_key(run_dir / "x")] = sorted(frames)
    if not out:
        raise SystemExit(f"no run directories with images under {images_dir}")
    return out


def label_path(mod_root: Path, img: Path) -> Path:
    return mod_root / "labels" / img.parent.name / (img.stem + ".txt")


def read_label_counts(mod_root: Path, byrun) -> tuple[dict[Path, dict[int, int]], int]:
    """{image_path: {class_id: n_boxes}}; also returns count of missing label files."""
    counts: dict[Path, dict[int, int]] = {}
    missing = 0
    for frames in byrun.values():
        for _, img in frames:
            lbl = label_path(mod_root, img)
            per: dict[int, int] = {}
            if lbl.is_file():
                for line in lbl.read_text(encoding="utf-8").splitlines():
                    parts = line.split()
                    if parts:
                        cid = int(float(parts[0]))
                        per[cid] = per.get(cid, 0) + 1
            else:
                missing += 1
            counts[img] = per
    return counts, missing


def median_delta(ordinals: list[float]) -> float:
    o = sorted(set(ordinals))
    deltas = [b - a for a, b in zip(o, o[1:])]
    return median(deltas) if deltas else 1.0


# ---- planning --------------------------------------------------------------

def run_k_cap(union_ordinals: list[float], max_guard: float) -> int:
    """Largest K (multiple of CYCLE, >= CYCLE) whose blocks stay wider than the
    run's largest modality guard — sparse runs get coarser blocks instead of
    having whole eval blocks swallowed by guard bands."""
    span = max(union_ordinals) - min(union_ordinals)
    cap = int(span / (BLOCK_GUARD_RATIO * max_guard)) if max_guard > 0 else 10**6
    return max(CYCLE, (cap // CYCLE) * CYCLE)


def plan_run(union_ordinals: list[float], k: int) -> list[float] | None:
    """K equal-frequency block cut points (len k-1) over the unique ordinal timeline."""
    uniq = sorted(set(union_ordinals))
    if len(uniq) < 4 * k:   # blocks would be too thin to survive guards
        return None
    cuts = []
    for j in range(1, k):
        idx = j * len(uniq) // k
        cuts.append((uniq[idx - 1] + uniq[idx]) / 2.0)
    return cuts


def block_split(block: int) -> str:
    slot = block % CYCLE
    return "val" if slot == VAL_SLOT else "test" if slot == TEST_SLOT else "train"


def assign_frame(o: float, cuts: list[float], guard: float) -> str | None:
    """Split for ordinal o, or None when it falls in a guard band."""
    b = bisect_right(cuts, o)
    split = block_split(b)
    half = guard / 2.0
    for edge_idx, neighbor in ((b - 1, b - 1), (b, b + 1)):
        if 0 <= edge_idx < len(cuts) and abs(o - cuts[edge_idx]) < half:
            other = block_split(neighbor)
            if other != split and ("train" in (split, other)):
                return None
    return split


def build_lists(byrun, cuts_by_run, mod_guards) -> dict[str, list[str]]:
    """{split: ['images/<run>/<file>', ...]} — guard-band frames dropped.

    mod_guards: {run: guard_units} for THIS modality (leakage is per modality;
    the shared cuts keep paired frames' split assignments identical)."""
    lists: dict[str, list[str]] = {s: [] for s in SPLITS}
    for run, frames in byrun.items():
        cuts, guard = cuts_by_run[run], mod_guards[run]
        for o, img in frames:
            split = assign_frame(o, cuts, guard)
            if split is not None:
                lists[split].append(f"images/{run}/{img.name}")
    return {s: sorted(v) for s, v in lists.items()}


# ---- gates -----------------------------------------------------------------

def leakage_gate(mod_root: Path, lists, min_gap: int) -> dict:
    """The real audit_split() on projected lists (filenames only)."""
    tmp = mod_root / "_resplit_plan"
    tmp.mkdir(exist_ok=True)
    try:
        for split, items in lists.items():
            (tmp / f"{split}.txt").write_text("\n".join(items) + "\n", encoding="utf-8")
        data = {
            "_yaml_path": str((mod_root / "_resplit_plan.yaml").resolve()),
            "path": str(mod_root.resolve()),
            "train": "_resplit_plan/train.txt",
            "val": "_resplit_plan/val.txt",
            "test": "_resplit_plan/test.txt",
        }
        return audit_split(data, min_gap)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def balance_stats(mod_root: Path, lists, counts, names) -> dict:
    """Per-split frame/box statistics for the balance gate + report table."""
    stats = {}
    for split, items in lists.items():
        cls_boxes = {c: 0 for c in names}
        n_frames, n_empty = len(items), 0
        for rel in items:
            img = mod_root / rel
            per = counts.get(img, {})
            if not per:
                n_empty += 1
            for cid, n in per.items():
                cls_boxes[cid] = cls_boxes.get(cid, 0) + n
        total = sum(cls_boxes.values())
        stats[split] = {
            "frames": n_frames, "boxes": total, "cls": cls_boxes,
            "bpf": (total / n_frames) if n_frames else 0.0, "empty": n_empty,
        }
    g_cls = {c: sum(stats[s]["cls"].get(c, 0) for s in SPLITS) for c in names}
    g_boxes, g_frames = sum(g_cls.values()), sum(stats[s]["frames"] for s in SPLITS)
    stats["_global"] = {
        "frames": g_frames, "boxes": g_boxes, "cls": g_cls,
        "bpf": (g_boxes / g_frames) if g_frames else 0.0,
    }
    return stats


def coverage_gate(byrun, run_matrix) -> list[str]:
    """Every run with >= COVERAGE_MIN_RUN frames in this modality must land
    frames in val AND test — no silently vanished mission (run_matrix comes
    from the leakage audit, so it reflects the post-guard lists)."""
    fails = []
    for run, frames in byrun.items():
        if len(frames) < COVERAGE_MIN_RUN:
            continue
        counts = run_matrix.get(run, {})
        for split in ("val", "test"):
            if counts.get(split, 0) == 0:
                fails.append(f"coverage: run {run} ({len(frames)} frames) has no {split} frames")
    return fails


def balance_gate(stats, names) -> list[str]:
    """Empty list = PASS; otherwise the reasons, verify_split_balance-style."""
    fails = []
    g = stats["_global"]
    for split in SPLITS:
        s = stats[split]
        if s["frames"] == 0:
            fails.append(f"{split}: empty split")
            continue
        for cid, cname in names.items():
            g_share = g["cls"][cid] / g["boxes"] if g["boxes"] else 0.0
            share = s["cls"].get(cid, 0) / s["boxes"] if s["boxes"] else 0.0
            if g_share > 0 and abs(share - g_share) / g_share > CLASS_DEV:
                fails.append(f"{split}: class '{cname}' share {share:.1%} vs global "
                             f"{g_share:.1%} (>{CLASS_DEV:.0%} relative deviation)")
        if g["bpf"] > 0 and abs(s["bpf"] - g["bpf"]) / g["bpf"] > DENSITY_DEV:
            fails.append(f"{split}: boxes/frame {s['bpf']:.2f} vs global {g['bpf']:.2f} "
                         f"(>{DENSITY_DEV:.0%} relative deviation)")
    return fails


def print_balance(stats, names, title: str) -> None:
    print(f"  {title}")
    print(f"    {'split':<6} {'frames':>8} " +
          " ".join(f"{names[c]:>8} {'%':>5}" for c in sorted(names)) +
          f" {'box/frm':>8} {'empty':>6}")
    for split in SPLITS:
        s = stats[split]
        row = f"    {split:<6} {s['frames']:>8} "
        for c in sorted(names):
            share = s["cls"].get(c, 0) / s["boxes"] if s["boxes"] else 0.0
            row += f"{s['cls'].get(c, 0):>8} {share:>5.1%} "
        print(row + f"{s['bpf']:>8.2f} {s['empty']:>6}")
    g = stats["_global"]
    print(f"    global bpf={g['bpf']:.2f}, shares: " +
          ", ".join(f"{names[c]}={g['cls'][c] / g['boxes']:.1%}" for c in sorted(names)))


def run_balance_stats(mod_root: Path, lists, counts) -> dict:
    """{split: {run: {frames, cls}}} — where each split's boxes come from, run by run."""
    per: dict[str, dict[str, dict]] = {s: {} for s in SPLITS}
    for split, items in lists.items():
        for rel in items:
            img = mod_root / rel
            d = per[split].setdefault(run_key(img), {"frames": 0, "cls": {}})
            d["frames"] += 1
            for cid, n in counts.get(img, {}).items():
                d["cls"][cid] = d["cls"].get(cid, 0) + n
    return per


def print_run_balance(per, names, title: str) -> None:
    """Per-run class table: reveals a split whose global share hides that one run
    supplies all of a class (informational — some runs genuinely lack a class,
    so this is not gated; the coverage gate handles outright missing runs)."""
    print(f"  {title}")
    for split in SPLITS:
        for run in sorted(per[split]):
            d = per[split][run]
            total = sum(d["cls"].values())
            cells = " ".join(
                f"{names[c]}={d['cls'].get(c, 0)}"
                f"({(d['cls'].get(c, 0) / total if total else 0):.0%})"
                for c in sorted(names))
            print(f"    {split:<6} {run}: frames={d['frames']:>6} boxes={total:>7}  {cells}")


# ---- disk verification (shared by --audit and post---execute) ---------------

def read_disk_lists(mod_root: Path) -> dict[str, list[str]]:
    lists = {}
    for split in SPLITS:
        txt = mod_root / f"{split}.txt"
        if not txt.is_file():
            raise SystemExit(f"missing split list: {txt}")
        lists[split] = [line.strip().lstrip("./").replace("\\", "/")
                        for line in txt.read_text(encoding="utf-8").splitlines()
                        if line.strip()]
    return lists


def verify_disk(mods: dict, cfg: dict, min_gap: int) -> bool:
    """All three gates against the lists as they exist on disk, through the real
    dataset yamls — exactly what training will see. Returns overall PASS."""
    all_ok = True
    for name, m in mods.items():
        data = load_data_yaml(resolve_data_yaml(cfg, m["alias"]))
        report = audit_split(data, min_gap)
        disk_lists = read_disk_lists(m["root"])
        stats = balance_stats(m["root"], disk_lists, m["counts"], m["names"])
        fails = balance_gate(stats, m["names"]) + coverage_gate(m["byrun"], report["run_matrix"])
        leak_ok = report["ok"]
        print(f"\n[{name}] leakage: {'PASS' if leak_ok else 'FAIL'}, "
              f"balance+coverage: {'PASS' if not fails else 'FAIL'}")
        print_balance(stats, m["names"], "balance (on disk):")
        print_run_balance(run_balance_stats(m["root"], disk_lists, m["counts"]),
                          m["names"], "per-run class breakdown:")
        if not leak_ok:
            print(format_report(report, max_examples=5))
        for reason in fails:
            print(f"  {reason}")
        all_ok &= leak_ok and not fails
    return all_ok


# ---- execution -------------------------------------------------------------

def write_lists(mod_root: Path, lists) -> None:
    for split, items in lists.items():
        target = mod_root / f"{split}.txt"
        backup = mod_root / f"{split}.txt.pre_resplit"
        if target.is_file() and not backup.exists():
            shutil.copy2(target, backup)
        # './' prefix: the form Ultralytics resolves relative to the txt's folder
        target.write_text("".join(f"./{p}\n" for p in items), encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="config.yaml path (default: repo root)")
    ap.add_argument("--data-vis", default="vis", help="VIS dataset: alias or yaml path")
    ap.add_argument("--data-ir", default="ir", help="IR dataset: alias or yaml path")
    ap.add_argument("--k", type=int, default=None, help="force blocks-per-run (skip the ladder)")
    ap.add_argument("--min-gap-frames", type=int, default=None,
                    help="override data_audit.min_gap_frames from config")
    ap.add_argument("--label-cache", default=None,
                    help="optional json cache of per-frame label counts (speeds up re-runs; "
                         "delete it if labels ever change)")
    ap.add_argument("--execute", action="store_true", help="write the lists (default: dry-run)")
    ap.add_argument("--audit", action="store_true",
                    help="verify the CURRENT on-disk split (all gates + per-run class "
                         "breakdown) and exit — no planning, no writes")
    args = ap.parse_args()
    if args.audit and args.execute:
        ap.error("--audit and --execute are mutually exclusive")

    cfg = load_config(args.config)
    min_gap = args.min_gap_frames or cfg.get("data_audit", {}).get("min_gap_frames", 100)

    cache = {}
    if args.label_cache and Path(args.label_cache).is_file():
        import json
        cache = json.loads(Path(args.label_cache).read_text(encoding="utf-8"))
        print(f"[cache] loaded label counts from {args.label_cache}")

    mods = {}
    for name, alias in (("VIS", args.data_vis), ("IR", args.data_ir)):
        data = load_data_yaml(resolve_data_yaml(cfg, alias))
        root = dataset_root(data)
        names = {int(k): str(v) for k, v in data.get("names", {}).items()}
        print(f"[{name}] root={root}")
        byrun = collect_frames(root)
        if name in cache:
            counts = {root / rel: {int(c): n for c, n in per.items()}
                      for rel, per in cache[name]["counts"].items()}
            missing = cache[name]["missing"]
        else:
            counts, missing = read_label_counts(root, byrun)
            cache[name] = {"counts": {img.relative_to(root).as_posix(): per
                                      for img, per in counts.items()},
                           "missing": missing}
        n = sum(len(f) for f in byrun.values())
        print(f"[{name}] {n} frames across {len(byrun)} runs"
              + (f" ({missing} missing label files)" if missing else ""))
        mods[name] = {"root": root, "byrun": byrun, "counts": counts,
                      "names": names, "alias": alias}

    if args.label_cache and not Path(args.label_cache).is_file():
        import json
        Path(args.label_cache).parent.mkdir(parents=True, exist_ok=True)
        Path(args.label_cache).write_text(json.dumps(cache), encoding="utf-8")
        print(f"[cache] wrote label counts to {args.label_cache}")

    # Shared per-run timeline: union of both modalities' ordinals (paired frames
    # get identical assignments; runs missing a modality use the one present).
    all_runs = sorted(set().union(*(m["byrun"].keys() for m in mods.values())))
    union_ords = {r: sorted(set(o for m in mods.values()
                                for o, _ in m["byrun"].get(r, [])))
                  for r in all_runs}
    if args.audit:
        print("\n=== AUDIT OF CURRENT ON-DISK SPLIT ===")
        ok = verify_disk(mods, cfg, min_gap)
        print(f"\nVERDICT: {'PASS — current split satisfies all gates' if ok else 'FAIL — re-split needed'}")
        return 0 if ok else 1

    # Per-(modality, run) guards: leakage is measured per modality in its own
    # median-delta units, so dense VIS never pays sparse IR's wide guard.
    mod_guards: dict[str, dict[str, float]] = {name: {} for name in mods}
    for name, m in mods.items():
        for r, frames in m["byrun"].items():
            req = median_delta([o for o, _ in frames]) * min_gap
            mod_guards[name][r] = max(GUARD_FLOOR, req * GUARD_MARGIN)
    for name in mods:
        print(f"[plan] {name} guard bands (ordinal units): " +
              ", ".join(f"{r}={g:.0f}" for r, g in sorted(mod_guards[name].items())))

    # Per-run K cap from the widest guard any modality needs in that run.
    k_caps = {}
    for r in all_runs:
        widest = max(mod_guards[name][r] for name in mods if r in mod_guards[name] and r in mods[name]["byrun"])
        k_caps[r] = run_k_cap(union_ords[r], widest)
    print("[plan] per-run K caps: " + ", ".join(f"{r}={k_caps[r]}" for r in all_runs))

    ladder = [args.k] if args.k else list(K_LADDER)
    chosen = None
    for k in ladder:
        cuts_by_run, k_by_run, bad = {}, {}, None
        for r in all_runs:
            k_r = min(k, k_caps[r])
            cuts = plan_run(union_ords[r], k_r)
            if cuts is None:
                bad = f"run {r} too short for K={k_r}"
                break
            cuts_by_run[r] = cuts
            k_by_run[r] = k_r
        if bad:
            print(f"[K={k}] SKIP — {bad}")
            continue

        ok, results = True, {}
        for name, m in mods.items():
            run_guards = {r: mod_guards[name][r] for r in m["byrun"]}
            lists = build_lists(m["byrun"], cuts_by_run, run_guards)
            leak = leakage_gate(m["root"], lists, min_gap)
            stats = balance_stats(m["root"], lists, m["counts"], m["names"])
            fails = balance_gate(stats, m["names"]) + coverage_gate(m["byrun"], leak["run_matrix"])
            results[name] = (lists, leak, stats, fails)
            if not leak["ok"] or fails:
                ok = False
        line = " | ".join(
            f"{name}: leak={'PASS' if res[1]['ok'] else 'FAIL(' + str(len(res[1]['temporal_violations']) + len(res[1]['duplicates'])) + ')'}"
            f" balance={'PASS' if not res[3] else 'FAIL(' + str(len(res[3])) + ')'}"
            for name, res in results.items())
        effective = sorted(set(k_by_run.values()))
        print(f"[K={k}] per-run K {effective} | {line}")
        if ok:
            chosen = (k, k_by_run, results)
            break
        for name, res in results.items():
            for reason in res[3][:4]:
                print(f"        {name} {reason}")

    if chosen is None:
        print("\nNO K PASSED ALL GATES — inspect the reasons above "
              "(try a custom --k or revisit thresholds).")
        return 1

    k, k_by_run, results = chosen
    print(f"\n=== PLAN ACCEPTED: K={k} (per-run: "
          + ", ".join(f"{r}={k_by_run[r]}" for r in all_runs)
          + f"), cycle {CYCLE} (val slot {VAL_SLOT}, test slot {TEST_SLOT}), "
          f"guard split half/half ===")
    for name, (lists, leak, stats, _) in results.items():
        kept = sum(len(v) for v in lists.values())
        total = sum(len(f) for f in mods[name]["byrun"].values())
        print(f"\n[{name}] kept {kept}/{total} frames "
              f"({total - kept} dropped in guard bands, {100 * (total - kept) / total:.1f}%)")
        print(f"  sizes: " + ", ".join(f"{s}={len(lists[s])}" for s in SPLITS))
        print_balance(stats, mods[name]["names"], "balance (projected):")
        print("  run matrix: " + str(leak["run_matrix"]))

    if not args.execute:
        print("\n(dry run — rerun with --execute to write the lists)")
        return 0

    print("\n=== EXECUTING (list rewrite only; images/labels untouched) ===")
    for name, (lists, _, _, _) in results.items():
        write_lists(mods[name]["root"], lists)
        print(f"[{name}] wrote train/val/test.txt (originals -> *.txt.pre_resplit)")

    # Re-verify from disk through the real yamls — what training will actually see.
    print("\n=== POST-WRITE VERIFICATION (from disk) ===")
    all_ok = verify_disk(mods, cfg, min_gap)

    if all_ok:
        print("\nVERDICT: PASS — split rewritten and verified (leakage + balance).")
        print("NEXT: regenerate derived artifacts (scripts/make_stride_subset.py) and "
              "re-run scripts/audit_split.py if you want the standalone report files.")
        return 0
    print("\nVERDICT: FAIL after write — restore *.txt.pre_resplit and investigate.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
