"""Resolve image lists from Ultralytics dataset yamls.

The dataset contract (see dataset_requirement.md) is that ``data_vis.yaml`` /
``data_ir.yaml`` expose ``train``/``val``/``test`` entries as txt file lists
(one image path per line) or directories. These helpers resolve either form to
absolute Paths without touching Ultralytics internals, so the audit and
subsetting tools work on any machine before any training starts.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# Pohang runs are pohang00..pohang04; MIT runs are whatever directory naming the
# conversion uses — fall back to the image's parent directory as the run key.
_RUN_PATTERN = re.compile(r"(pohang\d{2})", re.IGNORECASE)


def load_data_yaml(yaml_path: str | Path) -> dict:
    yaml_path = Path(yaml_path)
    with open(yaml_path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"{yaml_path} is not a mapping — not an Ultralytics dataset yaml")
    data["_yaml_path"] = str(yaml_path.resolve())
    return data


def dataset_root(data: dict) -> Path:
    """The root against which relative entries resolve (yaml `path:` key, else yaml dir)."""
    yaml_dir = Path(data["_yaml_path"]).parent
    root = Path(str(data.get("path", ".")))
    return root if root.is_absolute() else (yaml_dir / root).resolve()


def split_image_list(data: dict, split: str) -> list[Path]:
    """All image paths for a split. Accepts txt list files, directories, or lists thereof."""
    if split not in data or data[split] is None:
        raise KeyError(f"dataset yaml {data['_yaml_path']} has no '{split}' entry")
    root = dataset_root(data)
    entries = data[split] if isinstance(data[split], list) else [data[split]]
    images: list[Path] = []
    for entry in entries:
        p = Path(str(entry))
        p = p if p.is_absolute() else (root / p).resolve()
        if p.suffix.lower() == ".txt":
            if not p.is_file():
                raise FileNotFoundError(f"split list not found: {p}")
            with open(p, "r", encoding="utf-8") as f:
                lines = [ln.strip() for ln in f if ln.strip()]
            # Ultralytics resolves a './x' entry against the LIST FILE's directory
            # (data/dataset.py::get_img_files), not against the yaml root. Those differ
            # whenever a yaml points at lists outside its own tree — as the derived
            # stride yamls do — and resolving against the root then yields paths that do
            # not exist. Pick the base once from the first entry rather than per line.
            base = p.parent
            first = next((ln for ln in lines if not Path(ln).is_absolute()), None)
            if first is not None and not (base / first).exists() and (root / first).exists():
                base = root
            for line in lines:
                q = Path(line)
                images.append(q if q.is_absolute() else (base / q).resolve())
        elif p.is_dir():
            images.extend(sorted(x for x in p.rglob("*") if x.suffix.lower() in IMAGE_SUFFIXES))
        else:
            raise FileNotFoundError(f"'{split}' entry is neither a txt list nor a directory: {p}")
    return images


def run_key(image_path: Path) -> str:
    """Recording-run identifier for a frame (temporal-leakage checks group by this)."""
    m = _RUN_PATTERN.search(str(image_path))
    if m:
        return m.group(1).lower()
    return image_path.parent.name


def frame_ordinal(image_path: Path) -> float | None:
    """Numeric ordering key from the filename stem (timestamp or frame index).

    Handles '1544674707.213211' (epoch-seconds stems), '000123' (frame index),
    and 'pohang00_000123' (trailing index). Returns None when no number exists —
    callers must then skip temporal checks for that frame and say so.
    """
    stem = image_path.stem
    try:
        return float(stem)
    except ValueError:
        pass
    digit_runs = re.findall(r"\d+", stem)
    if digit_runs:
        return float(digit_runs[-1])
    return None
