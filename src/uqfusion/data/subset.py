"""Derived training subsets: stride subsampling (approved A2-6) and tiny smoke subsets.

Never modifies the source dataset. Writes a derived txt list plus a derived
dataset yaml (same val/test/names, new train list) under the outputs tree, and
returns the derived yaml path to feed straight into training.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import yaml

from uqfusion.data.lists import frame_ordinal, run_key, split_image_list


def make_stride_subset(data: dict, stride: int, out_dir: str | Path, tag: str | None = None) -> Path:
    """Keep every `stride`-th training TIME STEP per run (all frames sharing it).

    Frames are grouped by their per-run frame ordinal before striding: L/R
    stereo frames share a frame index, so striding raw frame lists would just
    alternate cameras — keeping the temporal near-duplicates A2-6 exists to
    remove while halving viewpoint diversity. Striding unique ordinals thins
    time and keeps stereo pairs together. Frames without a numeric stem have
    no timeline; they are strided by name order as a documented fallback.
    """
    if stride < 1:
        raise ValueError(f"stride must be >= 1, got {stride}")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = tag or f"stride{stride}"

    by_run: dict[str, list[Path]] = defaultdict(list)
    for img in split_image_list(data, "train"):
        by_run[run_key(img)].append(img)

    kept: list[Path] = []
    for run in sorted(by_run):
        groups: dict[float, list[Path]] = defaultdict(list)
        no_ordinal: list[Path] = []
        for p in by_run[run]:
            o = frame_ordinal(p)
            (no_ordinal if o is None else groups[o]).append(p)
        for o in sorted(groups)[::stride]:
            kept.extend(sorted(groups[o], key=lambda p: p.name))
        kept.extend(sorted(no_ordinal, key=lambda p: p.name)[::stride])

    stem = Path(data["_yaml_path"]).stem  # e.g. data_vis
    list_path = out_dir / f"{stem}_train_{tag}.txt"
    with open(list_path, "w", encoding="utf-8") as f:
        for img in kept:
            f.write(str(img) + "\n")  # absolute paths: derived lists are machine-local artifacts

    derived = {k: v for k, v in data.items() if not k.startswith("_")}
    derived.pop("path", None)  # absolute lines in the txt make the root key unnecessary/harmful
    derived["train"] = str(list_path)
    # val/test entries must survive as resolvable paths without the original root:
    for split in ("val", "test"):
        if derived.get(split) is not None:
            entries = derived[split] if isinstance(derived[split], list) else [derived[split]]
            resolved = []
            for entry in entries:
                p = Path(str(entry))
                from uqfusion.data.lists import dataset_root

                resolved.append(str(p if p.is_absolute() else (dataset_root(data) / p).resolve()))
            derived[split] = resolved if len(resolved) > 1 else resolved[0]

    yaml_path = out_dir / f"{stem}_{tag}.yaml"
    with open(yaml_path, "w", encoding="utf-8") as f:
        yaml.safe_dump(derived, f, sort_keys=False)

    n_total = sum(len(v) for v in by_run.values())
    print(f"stride subset: kept {len(kept)}/{n_total} training frames "
          f"({len(by_run)} runs, stride {stride}) -> {yaml_path}")
    return yaml_path
