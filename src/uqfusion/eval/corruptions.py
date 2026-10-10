"""Named adverse-condition corruptions (scope §5.2, §11.3 B.6) via albumentations.

Each corruption is a deterministic-per-frame transform: `make_corruption(name,
severity, seed)` returns `transform(im_bgr, index) -> im_bgr` for cache
building. The per-frame seed is derived from (seed, index), so the SAME
corrupted frames are reproducible — and the anti-leakage rule (plan B5-5)
is enforced by convention: gate tuning and final testing must use DIFFERENT
`seed`/severity values, recorded in the cache meta.

Severity in [1..3] maps to mild/moderate/heavy parameterizations.
"""

from __future__ import annotations

import numpy as np

CORRUPTIONS = ("fog", "rain", "lowlight", "glare", "blur", "noise")


def _albu(name: str, severity: int):
    import albumentations as A

    s = int(np.clip(severity, 1, 3))
    if name == "fog":
        coef = [(0.3, 0.5), (0.5, 0.8), (0.8, 1.0)][s - 1]
        return A.RandomFog(fog_coef_range=coef, alpha_coef=0.1, p=1.0)
    if name == "rain":
        return A.RandomRain(blur_value=[3, 5, 7][s - 1], brightness_coefficient=0.8, p=1.0)
    if name == "lowlight":
        limit = [(-0.5, -0.3), (-0.7, -0.5), (-0.9, -0.7)][s - 1]
        return A.RandomBrightnessContrast(brightness_limit=limit, contrast_limit=(-0.2, 0.0), p=1.0)
    if name == "glare":
        return A.RandomSunFlare(src_radius=[120, 200, 320][s - 1], p=1.0)
    if name == "blur":
        return A.MotionBlur(blur_limit=[(5, 9), (9, 15), (15, 25)][s - 1], p=1.0)
    if name == "noise":
        std = [(0.05, 0.1), (0.1, 0.2), (0.2, 0.3)][s - 1]
        return A.GaussNoise(std_range=std, p=1.0)
    raise ValueError(f"unknown corruption '{name}' — options: {CORRUPTIONS}")


def make_corruption(name: str, severity: int = 2, seed: int = 0, *, version: str = "v1",
                    modality: str = "vis", images: list | None = None, params: dict | None = None):
    """Returns transform(im_bgr, index) -> corrupted im_bgr (deterministic per index).

    `version="v1"` (the default, and what every cache built before 2026-10-09 used) is the
    albumentations parameterisation above, bit-for-bit unchanged. `version="v2"` is the
    physically modelled, content-only rebuild in `corruptions_v2.py`; it needs `modality`
    and, for fog and glare, the ordered frame list `images` the index refers to; `params`
    overrides v2 constants for a sensitivity row (v2 only).
    """
    if version == "v2":
        from uqfusion.eval.corruptions_v2 import make_corruption_v2

        return make_corruption_v2(name, severity, seed, modality=modality, images=images, params=params)
    if params:
        raise ValueError("params= is a corruption v2 option")
    if version != "v1":
        raise ValueError(f"unknown corruption version '{version}' -- options: v1, v2")
    import albumentations as A

    t = A.Compose([_albu(name, severity)], p=1.0)

    def transform(im_bgr: np.ndarray, index: int) -> np.ndarray:
        t.set_random_seed(seed * 100003 + index)
        return t(image=im_bgr)["image"]

    transform.spec = {"corrupt": name, "severity": severity, "corrupt_seed": seed, "corrupt_version": "v1"}
    return transform


def corruption_from_meta(meta: dict, images: list | None = None, modality: str | None = None):
    """The transform a cache (or statistics file) was built with, or None for a clean one.

    Caches written before v2 carry no `corrupt_version`; they are v1 by construction. A v2
    cache stamped with a different `corrupt_code` was made by an older revision of
    corruptions_v2.py and cannot be replayed by this one: that raises.
    """
    if not meta.get("corrupt"):
        return None
    tf = make_corruption(meta["corrupt"], meta["severity"], meta["corrupt_seed"],
                         version=meta.get("corrupt_version") or "v1",
                         modality=modality or meta.get("modality") or "vis", images=images,
                         params=meta.get("corrupt_params"))
    want = meta.get("corrupt_code")
    if (meta.get("corrupt_version") == "v2" and tf.spec.get("corrupt_code") != want):
        raise ValueError(f"cache built by corruptions_v2 revision {want!r}, this is "
                         f"{tf.spec.get('corrupt_code')!r}: rebuild it rather than replay it")
    return tf
