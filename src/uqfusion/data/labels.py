"""Label-tree identity: one definition of "which labels are these?".

**R-E1 slice 3 (F14).** Before this module the project held **three** independent
implementations of the images->labels path swap and **two** byte-identical copies of
the content hash, in `eval/matching.py`, `scripts/filter_night_boxes.py` and
`scripts/verify_dataset_state.py`. They were written at different times by different
reasoning:

* `matching.label_path_for` replaces **every** path component named `images`;
* `filter_night_boxes._label_path` takes the rightmost `images` **substring**
  (`rfind`), so a run directory containing the letters "images" would match;
* `verify_dataset_state.label_path` takes the last `images` **path separator** group.

Measured on all 133,140 real train+val image paths across both modalities:
**0 disagreements between any pair.** So this is a latent trap rather than a live
defect -- exactly the F14 pattern, and exactly the reason to collapse it to one
definition while it is still latent instead of after it bites.

`label_content_hash` is byte-for-byte the algorithm the label ledger already uses, so
its numbers stay comparable to `runs/label_hash_ledger.csv` and to
`verify_dataset_state.py --expect-label-hash`. It is deliberately NOT a new algorithm:
`docs/` already records what happens when one number is read against another that was
never comparable to it (`df0cb307adc9` vs `b92739202127`).

**The scope is part of the identity and must travel in the name.** The same algorithm
over train, over train+val, and over the whole tree gives three different numbers, none
of which means anything against the others. Callers pass an explicit image list.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from uqfusion.data.lists import run_key


def label_path(image_path: str | Path) -> Path:
    """Ultralytics rule: the LAST `images` path component becomes `labels`, suffix .txt.

    Component-wise rather than substring-wise, so a directory whose name merely
    *contains* "images" is not rewritten. That is the strictest of the three
    implementations this replaces; on the real corpus all three agree.
    """
    p = Path(image_path)
    parts = list(p.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            break
    else:
        raise ValueError(f"no 'images' path component in {image_path}")
    return Path(*parts).with_suffix(".txt")


def label_content_hash(images: list[Path]) -> str:
    """Content hash of the label files behind `images`. 12 hex chars.

    The ledger's algorithm, unchanged: for each image in sorted order, hash
    `run/labelname\\n`, then the label bytes if the file exists, then a NUL. Filename
    and bytes both enter, so a rename and an edit are both visible; a missing label
    contributes its name and the separator, so deleting a file changes the hash.
    """
    h = hashlib.sha256()
    for img in sorted(images):
        lp = label_path(img)
        h.update(f"{run_key(img)}/{lp.name}\n".encode())
        if lp.is_file():
            h.update(lp.read_bytes())
        h.update(b"\x00")
    return h.hexdigest()[:12]
