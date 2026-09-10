"""R-E1 / F14: what produced a result file.

The review's finding is that nothing in this project can identify a run. Two
concrete instances, both found the hard way rather than theorised:

* `preset="crossmodal"` named **three different systems** on 2026-09-01 -- five
  artifacts, three `cap_ir` values -- because `ir_nms` and `cap_ir_scale` were local
  variables in `load_context`, applied and then discarded. `cap_ir` was their only
  witness, and only by its numeric value.
  (`docs/exposure-ledger-2026-09-09.md` section 6.)
* `map50_95` named an AP convention that was never declared, and had called itself
  "COCO-style" while implementing linear interpolation.
  (`docs/ap-convention-rule-2026-09-10.md`.)

Both are the same defect: **a result recorded a NAME, and the name was a moving
target.** The fix that generalises is to record the VALUES.

This module started as the smallest useful form of that: *for this result file, what
was the system and what was the source?* **Slice 2 (2026-09-10) added the half that
can refuse** -- ordered frame ids, an ordered-frame hash, a label-content hash, and a
VIS<->IR pairing check, all validated by `cache.load_cache`,
`fusion_eval.evaluate_systems`, `iralign.aligned_homographies` and
`learned_gate.LearnedGate.fit`. See `docs/cache-identity-2026-09-10.md`.

Still not the full immutable manifest R-E1 asks for: no checkpoint content hash,
nothing validated on RESUME, and no comparison of a cache's `labels_sha256` against
the hash recorded at training time (which is R-E1's actual acceptance criterion).

`git_revision()` deliberately reports a **dirty hash** as well as HEAD, because R-E1
notes that HEAD alone misses uncommitted source -- and most of this project's analysis
scripts are run before they are committed.
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[3]

# Fields copied off a FusionContext when one is supplied. Every one of these changes
# the numbers. `ir_nms` and `cap_ir_scale` are here because their absence is what made
# the crossmodal artifacts unidentifiable; the rest are the config block
# `eval_final_system.py` already wrote, promoted to a shared definition so a second
# script cannot record a different subset and call it the same thing.
CTX_FIELDS = (
    "preset", "preset_resolved", "role", "ir_nms", "cap_ir_scale", "cap_vis", "cap_ir", "iou_thr",
    "veto", "veto_rule", "veto_filter", "veto_health_mode", "single_passthrough",
    "cap_note",
)


def _git(*args: str) -> str:
    try:
        return subprocess.run(("git", *args), cwd=ROOT, capture_output=True,
                              text=True, timeout=10).stdout.strip()
    except Exception:
        return ""


@lru_cache(maxsize=1)
def git_revision() -> dict[str, Any]:
    """HEAD plus a hash of the uncommitted diff.

    R-E1 is explicit that git HEAD misses uncommitted source. A `dirty_sha256` of
    ``git diff HEAD`` closes that: two runs at the same HEAD with different working
    trees get different identities, which is the property that was missing.

    Cached -- the working tree does not change inside a single run, and this is called
    once per result file.
    """
    head = _git("rev-parse", "HEAD")
    diff = _git("diff", "HEAD")
    return {
        "git_head": head or None,
        "git_dirty": bool(diff),
        "dirty_sha256": (hashlib.sha256(diff.encode("utf-8")).hexdigest()[:16]
                         if diff else None),
        "git_branch": _git("rev-parse", "--abbrev-ref", "HEAD") or None,
    }


def _jsonable(v: Any) -> Any:
    """Coerce a recorded input to something `json.dumps` will accept.

    Paths become strings and tuples become lists. Anything else exotic becomes its
    `repr` rather than raising -- a provenance block that crashes the write is worse
    than one carrying an ugly string, and the alternative to an ugly string here is
    the silence that caused F14 in the first place.
    """
    if v is None or isinstance(v, (bool, int, float, str)):
        return v
    if isinstance(v, Path):
        return str(v)
    if isinstance(v, (tuple, list, set)):
        return [_jsonable(x) for x in v]
    if isinstance(v, dict):
        return {str(k): _jsonable(x) for k, x in v.items()}
    return repr(v)


def system_identity(ctx: Any = None, **extra: Any) -> dict[str, Any]:
    """Everything needed to say what produced a number. Safe to call with no context.

    `ctx` is a `FusionContext` when the caller has one. Anything else worth pinning
    (cache names, seeds, n_boot, thresholds a script chose itself) goes in `extra` --
    the point is that it lands in the file rather than staying in someone's head.
    """
    from uqfusion.eval.apmetrics import declared_policies

    ident: dict[str, Any] = {
        "written_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        **git_revision(),
        **declared_policies(),
    }
    if ctx is not None:
        for f in CTX_FIELDS:
            if hasattr(ctx, f):
                ident[f] = _jsonable(getattr(ctx, f))
        # every resolved `load_context` argument, under one key so the top level stays
        # readable. This is the part that would have made F14 impossible.
        if getattr(ctx, "inputs", None):
            ident["load_context_inputs"] = _jsonable(ctx.inputs)
    ident.update(extra)
    return ident


def identity_markdown(ident: dict[str, Any] | None = None) -> str:
    """The same thing as a markdown section, for `_ideas_common.write_md`."""
    ident = system_identity() if ident is None else ident
    rows = "\n".join(f"| `{k}` | {'--' if v is None else f'`{v}`'} |"
                     for k, v in ident.items())
    dirty = ident.get("git_dirty")
    warn = ("\n\n**The working tree was dirty when this ran.** `git_head` alone does "
            "not identify the source that produced these numbers; `dirty_sha256` is a "
            "hash of `git diff HEAD` and is the part that does."
            if dirty else "")
    return ("## Provenance\n\n"
            "Written automatically (R-E1/F14). A result that records a preset *name* "
            "cannot say which system produced it -- `preset=\"crossmodal\"` named three "
            "different systems on 2026-09-01. These are values.\n\n"
            "| field | value |\n|---|---|\n" + rows + warn)


# ---------------------------------------------------------------- frame identity
# R-E1 slice 2. Slice 1's own docstring said what was missing: "no ordered pair IDs
# ... and nothing here is validated on cache load". These are the ordered pair IDs,
# and `cache.load_cache` / `fusion_eval.evaluate_systems` are where they get checked.
#
# The point is to REFUSE rather than describe. Reproduced on the real caches before
# any of this was written (`docs/cache-identity-2026-09-10.md`): handing
# `evaluate_systems` a fully REVERSED IR cache moves gated fusion by **-0.025630**
# and raises nothing, because the only guard was `len(vis) == len(ir)`. Worse, a
# one-frame SHIFT moves it by **-0.000968**, which is *below* the 0.0014-0.0031
# paired noise floor -- a misalignment that no amount of statistics can find.


def pair_id(image_path: str | Path) -> str | None:
    """Ordered identity of one frame WITHIN one modality: ``run/ordinal``.

    **This is not the VIS<->IR pairing key, and assuming it was is the mistake this
    module made first.** `run/ordinal` looks modality-independent -- `pohang00_L_006767`
    and `pohang00_006767` both reduce to `pohang00/6767` -- and in `pohang00` it happens
    to be right. It is wrong from `pohang01` onward, where IR runs one ahead, and
    `scripts/build_pairs.py` has always said so: 16,544 of 28,388 pair rows carry
    different indices on the two sides. Pairing lives in `_pair_table`; this is for
    hashing a single cache's own frame ORDER.

    Returns None when the stem carries no number -- the synthetic smoke fixtures have
    no ordinals, and refusing there would make the check untestable.
    """
    from uqfusion.data.lists import frame_ordinal, run_key

    p = Path(str(image_path))
    o = frame_ordinal(p)
    if o is None:
        return None
    return f"{run_key(p)}/{int(o)}"


def pair_ids(records: list[dict]) -> list[str | None]:
    """`pair_id` for each record, in cache order. Order is part of the identity."""
    return [pair_id(r["image_path"]) if "image_path" in r else None for r in records]


def frames_sha256(records: list[dict]) -> str:
    """Content hash of the ORDERED pair-id sequence.

    Ordered, because a reordered cache is a different cache: it produced a different
    pairing and therefore different numbers. A set hash would call the reversal above
    identical to the original.
    """
    h = hashlib.sha256()
    for i, pid in enumerate(pair_ids(records)):
        h.update(f"{i}:{pid}\n".encode("utf-8"))
    return h.hexdigest()[:16]


def labels_sha256(records: list[dict]) -> str | None:
    """Content hash of the GT label FILES behind a cache, in cache order.

    R-E1's acceptance asks that label hashes at training, caching and evaluation
    agree, and `split_fingerprint` cannot supply that: it hashes frame NAMES, so the
    night-cut edit that removed 132,688 boxes left it unchanged (reproduced -- see the
    report). This hashes the bytes, so it moves when a label moves.

    Returns None when no label file exists for any frame, which is a real state (an
    IR cache scored against VIS labels) rather than an error.
    """
    from uqfusion.eval.matching import label_path_for

    h = hashlib.sha256()
    seen = 0
    for r in records:
        if "image_path" not in r:
            continue
        lp = Path(label_path_for(r["image_path"]))
        if lp.is_file():
            h.update(lp.read_bytes())
            seen += 1
        else:
            h.update(b"<missing>")
    return h.hexdigest()[:16] if seen else None


@lru_cache(maxsize=1)
def _pair_table() -> dict[str, str]:
    """`stereo_L_file -> ir_file` from the dataset authors' timestamp-matched CSVs.

    **This, not the frame number, is what pairs VIS to IR.** `scripts/build_pairs.py`
    has said so since it was written: *16,544 of the 28,388 pair rows have a DIFFERENT
    index on the VIS and IR side, so a filename join silently mismatches 58% of the
    set*. The first version of this check paired on `run/ordinal` equality and refused
    the LEGITIMATE caches on 1,396 of 2,232 frames (62.5%) -- it had reproduced that
    documented fact and mistaken it for a defect. The ordinals genuinely differ:
    `pohang00` matches index for index, `pohang01` runs IR = VIS + 1.

    Empty dict when the dataset is not present (a CI or smoke machine). The pairing
    check then reports every frame as unidentifiable rather than passing silently.
    """
    try:
        from uqfusion.config import load_config
        root = Path(load_config(None)["datasets"]["pohang"]["root"]) / "paired"
    except Exception:
        return {}
    table: dict[str, str] = {}
    for f in sorted(root.glob("*_pairs.csv")):
        with open(f, newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                table[row["stereo_L_file"]] = row["ir_file"]
    return table


def paired_id_report(vis_records: list[dict], ir_records: list[dict]) -> dict[str, Any]:
    """Compare two caches frame by frame WITHOUT raising. `assert_paired` uses it.

    A frame counts as MISMATCHED only when the pair table knows the VIS filename and
    names a different IR file than the one at that index. A VIS frame the table does
    not know (synthetic smoke fixtures, a dataset that is not present) counts as
    unidentifiable, and that count is reported rather than swallowed -- a run where it
    is large was never really checked.
    """
    table = _pair_table()
    mism, undef = [], 0
    for k, (rv, ri) in enumerate(zip(vis_records, ir_records)):
        vname = Path(str(rv.get("image_path", ""))).name
        iname = Path(str(ri.get("image_path", ""))).name
        want = table.get(vname)
        if want is None:
            undef += 1
        elif want != iname:
            mism.append((k, vname, iname, want))
    return {"n_vis": len(vis_records), "n_ir": len(ir_records),
            "n_mismatched": len(mism), "n_unidentifiable": undef,
            "pair_table_rows": len(table),
            "first_mismatch": (mism[0][0] if mism else None),
            "first_mismatch_detail": (mism[0] if mism else None),
            "vis_frames_sha256": frames_sha256(vis_records),
            "ir_frames_sha256": frames_sha256(ir_records)}


def assert_paired(vis_records: list[dict], ir_records: list[dict], where: str = "") -> dict[str, Any]:
    """Refuse a mis-paired pair of caches BEFORE any metric runs.

    Raises on unequal length (what was already checked) and on any frame whose IR file
    is not the one the dataset's own timestamp matching assigns to that VIS file (what
    was not). Returns the report, including `n_unidentifiable`, so a caller can tell a
    real pass from a vacuous one.
    """
    rep = paired_id_report(vis_records, ir_records)
    tag = f" [{where}]" if where else ""
    if rep["n_vis"] != rep["n_ir"]:
        raise ValueError(f"paired caches differ in length{tag}: "
                         f"vis {rep['n_vis']} vs ir {rep['n_ir']}")
    if rep["n_mismatched"]:
        k, vname, got, want = rep["first_mismatch_detail"]
        raise ValueError(
            f"paired caches are NOT frame-aligned{tag}: {rep['n_mismatched']} of "
            f"{rep['n_vis']} frames disagree with the dataset pair table; first at "
            f"index {k}: {vname} pairs with {want}, but the IR cache holds {got}. "
            f"Equal lengths are not equal frames -- a reversed IR cache moves gated "
            f"fusion by -0.0256 and a one-frame shift by -0.0010, which is below the "
            f"0.0014-0.0031 noise floor and so cannot be found statistically. Rebuild "
            f"both lists with scripts/build_pairs.py rather than passing this check.")
    return rep
