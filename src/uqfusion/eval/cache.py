"""Prediction caches (plan B5-2): run every model over the eval frames ONCE,
then all calibration metrics and every gate-level ablation are CPU
post-processing over the cached records — no retraining, no re-inference.

Format: one pickle per (source, split, condition): {"meta": {...}, "records":
[record, ...]} where records follow the uqfusion.uq.infer schema plus
"image_path". Meta stamps weights, thresholds, corruption, and git commit so a
cache can always be traced to what produced it (scope §11.1).
"""

from __future__ import annotations

import pickle
from pathlib import Path

from uqfusion.bench.grid import _git_commit
from uqfusion.eval.identity import (
    frames_sha256, labels_sha256, pair_ids, system_identity,
)


def build_cache(
    predictor,
    images: list,
    out_path: str | Path,
    meta: dict | None = None,
    transform=None,
    log_every: int = 50,
) -> Path:
    """Run `predictor` over `images` (optionally through `transform(im_bgr, index)`,
    e.g. a corruption) and pickle the records."""
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    records = []
    for i, img in enumerate(images):
        if transform is None:
            rec = predictor(img)
        else:
            im = cv2.imread(str(img))
            if im is None:
                raise FileNotFoundError(f"could not read image: {img}")
            rec = predictor(transform(im, i))
        rec["image_path"] = str(img)
        records.append(rec)
        if log_every and (i + 1) % log_every == 0:
            print(f"[cache] {i + 1}/{len(images)} frames")

    # R-E1 slice 2: a cache now carries its own identity, not just its provenance.
    # `frames_sha256` hashes the ORDERED pair ids, so a reordered or cross-substrate
    # cache stops being interchangeable with this one; `labels_sha256` hashes the GT
    # label bytes, which `split_fingerprint` cannot see (it hashes frame NAMES, so the
    # night-cut edit left it unchanged -- reproduced in the report). `load_cache`
    # re-derives both and refuses on a mismatch.
    payload = {"meta": {**(meta or {}), "n_frames": len(records), "git_commit": _git_commit(),
                        "pair_ids": pair_ids(records),
                        "frames_sha256": frames_sha256(records),
                        "labels_sha256": labels_sha256(records),
                        "identity": system_identity()},
               "records": records}
    with open(out_path, "wb") as f:
        pickle.dump(payload, f)
    print(f"[cache] {len(records)} records -> {out_path}")
    return out_path


# Arrays that must agree in length within one record. `sigma_ltrb` is optional (only
# the Gaussian and MC/ensemble sources carry it) and is checked when present.
_PER_DETECTION = ("conf", "cls", "boxes_xyxy", "sigma_ltrb", "dfl_sigma_ltrb")


def _check_records(records: list, path) -> None:
    if not isinstance(records, list):
        raise ValueError(f"{path}: 'records' is {type(records).__name__}, not a list")
    for i, r in enumerate(records):
        if not isinstance(r, dict):
            raise ValueError(f"{path}: record {i} is {type(r).__name__}, not a dict")
        if "image_path" not in r:
            raise ValueError(f"{path}: record {i} has no 'image_path' -- it cannot be "
                             "identified, paired, or scored against a label file")
        lens = {k: len(r[k]) for k in _PER_DETECTION if k in r}
        if len(set(lens.values())) > 1:
            raise ValueError(f"{path}: record {i} is internally inconsistent -- "
                             f"per-detection arrays disagree in length: {lens}")


def load_cache(path: str | Path, *, validate: bool = True) -> tuple[list[dict], dict]:
    """Load a prediction cache, refusing a damaged or misidentified one.

    **R-E1 slice 2.** Until now this function validated nothing: a payload claiming
    `n_frames` 99999 while holding 10 records loaded without complaint, and so did
    records stripped of every prediction key. Both reproduced before this was written.

    What is checked, in increasing order of what it can catch:

    * the payload is a cache at all (a dict with `meta` and `records`);
    * `meta["n_frames"]` agrees with the number of records actually present;
    * every record carries an `image_path` and its per-detection arrays agree in
      length -- a truncated write cannot pass;
    * when the cache stamps `frames_sha256` / `labels_sha256`, both are RE-DERIVED and
      must match. That is the part that refuses a reordered cache or one whose labels
      have moved underneath it.

    The last check applies only to caches built after this change. All 266 caches
    already on disk stamp neither, so they keep loading exactly as before and no
    recorded number moves -- this adds a gate for new work rather than retroactively
    invalidating old work, which is not something a validator gets to decide.

    `validate=False` is the deliberate escape hatch (inspecting a cache you already
    know is broken). It is a parameter rather than a silent fallback so that skipping
    the check is visible at the call site.
    """
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if not validate:
        return payload["records"], payload["meta"]

    if not isinstance(payload, dict) or "records" not in payload or "meta" not in payload:
        raise ValueError(f"{path}: not a prediction cache -- expected a dict with "
                         f"'meta' and 'records', got {type(payload).__name__} "
                         f"with keys {sorted(payload) if isinstance(payload, dict) else '-'}")
    records, meta = payload["records"], payload["meta"]
    _check_records(records, path)

    claimed = meta.get("n_frames")
    if claimed is not None and int(claimed) != len(records):
        raise ValueError(f"{path}: meta claims n_frames={claimed} but holds "
                         f"{len(records)} records -- the cache is truncated or the "
                         "meta belongs to a different run")

    stamped = meta.get("frames_sha256")
    if stamped is not None and stamped != frames_sha256(records):
        raise ValueError(
            f"{path}: frames_sha256 {stamped} does not match the records it holds "
            f"({frames_sha256(records)}). The ordered frame identity changed after the "
            "cache was written -- records were reordered, substituted, or appended. "
            "Rebuild it; do not re-stamp it.")

    stamped = meta.get("labels_sha256")
    if stamped is not None:
        now = labels_sha256(records)
        if stamped != now:
            raise ValueError(
                f"{path}: labels_sha256 {stamped} != {now} recomputed now. The GT "
                "LABEL FILES changed after this cache was built, so its predictions "
                "and the labels they would be scored against come from different "
                "annotation states. This is the check `split_fingerprint` cannot make: "
                "it hashes frame names, so the night-cut edit left it unchanged. "
                "Rebuild the cache against the current labels, or score it against the "
                "release it was built on.")
    return records, meta
