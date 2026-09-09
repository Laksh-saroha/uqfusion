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

This module is the smallest useful form of that. It is not the full immutable manifest
R-E1 asks for -- there are no ordered pair IDs, no annotation-release hash, no
checkpoint content hash, and nothing here is validated on cache load. It answers one
question: *for this result file, what was the system and what was the source?*

`git_revision()` deliberately reports a **dirty hash** as well as HEAD, because R-E1
notes that HEAD alone misses uncommitted source -- and most of this project's analysis
scripts are run before they are committed.
"""

from __future__ import annotations

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
