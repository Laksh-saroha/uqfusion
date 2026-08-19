"""Architecture variants that ship no COCO checkpoint of their own.

`train_gaussian` normally starts a run from `{variant}.pt`. Two of the levers in
the small-object screen have no such file — they are architecture edits, not
released models — so they must be built from a yaml and seeded from the nearest
released checkpoint instead.

**Why a remap and not `YOLO(yaml).load(pt)`.** Ultralytics' `BaseModel.load`
intersects state dicts by KEY NAME and shape. Inserting a level into the neck
renumbers every layer after it, so the plain call silently drops the whole
bottom-up PAN and the Detect head: measured 62.0% of parameters transferred for
`yolo26s-p2` and 59.5% for `yolo26s-p2feat`. Declaring the index shift lifts
those to 92.5% and 98.3%. A 30-point difference in how much COCO the arm starts
from is easily larger than the effect the arm is meant to measure, so an
unremapped start would not be a test of the architecture — it would be a test of
initialisation.

Every spec therefore carries `min_transfer`, and `build_variant` RAISES below it.
A remap that quietly stops matching (an ultralytics yaml revision, a different
base checkpoint) has to fail loudly rather than train a plausible wrong thing —
the same reason `export_ir_percentile.py` gates on a round-trip against the
existing tree.
"""

from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
MODELS_DIR = ROOT / "configs" / "models"

# identity_through: last layer index whose numbering is unchanged from the base.
# shift:            added to every index after it.
# det_index:        the base's Detect layer, whose per-level branch lists also shift.
# det_level_shift:  how many levels were inserted BELOW the base's finest level.
VARIANT_SPECS: dict[str, dict] = {
    "yolo26s-p2": {
        "yaml": "yolo26s-p2.yaml",
        "base": "yolo26s.pt",
        "identity_through": 16,
        "shift": 6,
        "det_index": 23,
        "det_level_shift": 1,
        "min_transfer": 90.0,
        "note": "verbatim ultralytics 8.4.90 yolo26-p2.yaml: Detect(P2,P3,P4,P5). "
                "Drops ch[0] 128->64, so pair it with gaussian.sigma_width=32 or the "
                "arm also halves the variance branch.",
    },
    "yolo26s-p2feat": {
        "yaml": "yolo26s-p2feat.yaml",
        "base": "yolo26s.pt",
        "identity_through": 16,
        "shift": 3,
        "det_index": 23,
        "det_level_shift": 0,
        "min_transfer": 97.0,
        "note": "backbone P2/4 strided into the P3 neck feature. Detect stays 3-level, "
                "so anchor count and every head width are unchanged from yolo26s.",
    },
}


def is_variant(name: str) -> bool:
    return name in VARIANT_SPECS


def _remap_key(key: str, spec: dict) -> str:
    m = re.match(r"model\.(\d+)\.(.*)", key)
    if not m:
        return key
    i, rest = int(m.group(1)), m.group(2)
    if i <= spec["identity_through"]:
        return key
    if i == spec["det_index"] and spec["det_level_shift"]:
        # Detect holds one Sequential per level in cv2/cv3 (and their one2one twins);
        # inserting a finer level pushes each existing level up by det_level_shift.
        rest = re.sub(
            r"^((?:one2one_)?cv[23])\.(\d+)\.",
            lambda s: f"{s.group(1)}.{int(s.group(2)) + spec['det_level_shift']}.",
            rest,
        )
    return f"model.{i + spec['shift']}.{rest}"


def remap_state_dict(csd: dict, target: dict, spec: dict) -> tuple[dict, float]:
    """Base checkpoint -> variant state dict. Returns (matched subset, % of target params)."""
    matched = {}
    for k, v in csd.items():
        kk = _remap_key(k, spec)
        tv = target.get(kk)
        if tv is not None and tv.shape == v.shape:
            matched[kk] = v
    total = sum(v.numel() for v in target.values())
    return matched, 100.0 * sum(v.numel() for v in matched.values()) / max(total, 1)


def build_variant(name: str, verbose: bool = True):
    """A YOLO carrying the variant architecture seeded from its base checkpoint.

    Returns a `ultralytics.YOLO` whose `.ckpt` is set, so `Model.train` forwards the
    weights into the trainer rather than starting the run from scratch.
    """
    from ultralytics import YOLO
    from ultralytics.nn.tasks import load_checkpoint  # 8.4.90 name (was attempt_load_one_weight)
    from ultralytics.utils import LOGGER

    spec = VARIANT_SPECS[name]
    yaml_path = MODELS_DIR / spec["yaml"]
    if not yaml_path.is_file():
        raise FileNotFoundError(f"variant yaml missing: {yaml_path}")
    base_path = ROOT / spec["base"]
    if not base_path.is_file():
        raise FileNotFoundError(f"base checkpoint missing: {base_path}")

    model = YOLO(str(yaml_path))
    model.load(str(base_path))  # sets .ckpt (name-matched subset only)

    base_model, _ = load_checkpoint(str(base_path))
    matched, pct = remap_state_dict(base_model.float().state_dict(),
                                    model.model.state_dict(), spec)
    model.model.load_state_dict(matched, strict=False)

    if pct < spec["min_transfer"]:
        raise RuntimeError(
            f"{name}: only {pct:.1f}% of parameters transferred from {spec['base']}, "
            f"below the declared floor of {spec['min_transfer']:.1f}%. The index remap no "
            f"longer matches this architecture — fix the spec rather than training on it."
        )
    if verbose:
        LOGGER.info(f"[variant] {name}: {pct:.1f}% of parameters seeded from {spec['base']} "
                    f"({len(matched)} tensors, index shift +{spec['shift']})")
    return model, pct
