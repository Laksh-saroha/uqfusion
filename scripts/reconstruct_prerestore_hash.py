"""G1 -- rebuild the pre-restore VIS train label state in memory and hash it.

The question this answers is narrow and was worth answering exactly once: does
`dgxanode01`'s recorded label hash differ from this machine's *only* by the night
restore of 2026-09-02, or is there also drift nobody has accounted for?

The two hashes cannot simply be diffed -- the server holds labels we cannot walk. So
instead of comparing hashes, this reconstructs the server's tree from artifacts that
are already on this disk and hashes the reconstruction. If it reproduces the server's
recorded number, the difference is fully explained and nothing else moved.

The reconstruction is cheap because `cut_dark` was a blunt operation. It did not edit
the affected files, it emptied them (`filter_night_boxes.py:358`, "full cut: frame
becomes background") -- 17,502 of them, each named in
`runs/visfilter/visfilter_manifest.json`. So the pre-restore tree is today's tree with
exactly those files blank, and every other file untouched by either operation.

READ-ONLY. Nothing is written to the label tree, and the `.pre_visfilter` backups are
not consulted -- the manifest's file list is enough, and not opening the backups keeps
this incapable of disturbing them.

Result, 2026-09-10: reproduces `287b11c50b5a` with 616,891 boxes, against the manifest's
recorded `train_label_hash_after` of `287b11c50b5a`. See
`docs/g1-label-reconciliation-2026-09-10.md`.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from uqfusion.config import load_config, resolve_data_yaml       # noqa: E402
from uqfusion.data.labels import label_path, run_key             # noqa: E402
from uqfusion.data.lists import load_data_yaml, split_image_list  # noqa: E402

MANIFEST = ROOT / "runs" / "visfilter" / "visfilter_manifest.json"


def main() -> int:
    if not MANIFEST.is_file():
        print(f"missing {MANIFEST} -- runs/ is gitignored; this needs the local copy",
              file=sys.stderr)
        return 2

    man = json.loads(MANIFEST.read_text(encoding="utf-8"))
    emptied = {Path(k).resolve() for k in man["dropped"]}
    print(f"manifest: {len(emptied):,} files emptied by cut_dark "
          f"({man['boxes_dropped']:,} boxes)")

    cfg = load_config()
    data = load_data_yaml(resolve_data_yaml(cfg, "vis"))
    imgs = split_image_list(data, "train")

    # `label_content_hash`'s algorithm, inlined because we substitute content for
    # some files rather than reading them all off disk.
    h = hashlib.sha256()
    n_blank = boxes = 0
    for img in sorted(imgs):
        lp = label_path(img)
        h.update(f"{run_key(img)}/{lp.name}\n".encode())
        if lp.resolve() in emptied:
            n_blank += 1                       # cut_dark left this file empty
        elif lp.is_file():
            b = lp.read_bytes()
            h.update(b)
            boxes += sum(1 for ln in b.decode().splitlines() if ln.strip())
        h.update(b"\x00")

    got = h.hexdigest()[:12]
    ref = man["train_label_hash_after"]
    print(f"train images       : {len(imgs):,}  ({n_blank:,} blanked)")
    print(f"reconstructed boxes: {boxes:,}")
    print(f"reconstructed hash : {got}")
    print(f"server recorded    : {ref}")
    ok = got == ref
    print("VERDICT:", "MATCH -- the only difference is the night restore"
          if ok else "MISMATCH -- there is drift beyond the night restore")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
