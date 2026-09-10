"""R-E1 slice 3 acceptance gate — labels and recipe must enter a run's identity.

Written 2026-09-10 for finding F14. Slice 2 made a prediction cache able to refuse;
this covers the two defects it reproduced but deliberately deferred, because both
needed a `RESULT_FIELDS` column:

  D3. `split_fingerprint` hashes frame NAMES, so every label edit this project has
      made is invisible to it. Reproduced on the real function: deleting a box and
      changing a class id both leave it at the same 12 hex characters.
  D4. the grid's completed-run lookup omits the recipe, so a CSV holding
      (yolo26m, 0) at 25 epochs skipped a re-request at 50 and at 100 and went on
      reporting the 25-epoch number.

Cases:
  1. D3 reproduced: split_fingerprint does not move when labels change
  2. the label hash DOES move on the same edits, and on a deletion
  3. the shared label_content_hash is the ledger's algorithm, unchanged
  4. the three former `label_path` implementations agree with the shared one
  5. recipe_identity moves on every scientifically relevant knob, and only those
  6. the checkpoint hash is CONTENT, not path
  7. D4 reproduced and closed: plan_grid refuses a different recipe
  8. a different label state refuses the whole CSV
  9. the JSON recipe column survives a real CSV round-trip

Usage:  python scripts/smoke_recipe_identity.py
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from uqfusion.bench.grid import (
    RESULT_FIELDS,
    label_fingerprint,
    plan_grid,
    recipe_identity,
    split_fingerprint,
)
from uqfusion.data.labels import label_content_hash, label_path

FAIL = []


def check(name, cond, detail=""):
    print(f"[smoke] {name:<48} {'PASS' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAIL.append(name)


# ------------------------------------------------------- a synthetic labelled split
root = Path(tempfile.mkdtemp(prefix="smoke_recipe_")) / "ds"
(root / "images").mkdir(parents=True)
(root / "labels").mkdir(parents=True)
for i in range(4):
    (root / "images" / f"pohang00_L_00{i}.png").write_bytes(b"\x89PNG fake")
    (root / "labels" / f"pohang00_L_00{i}.txt").write_text("0 0.5 0.5 0.1 0.1\n", encoding="utf-8")
(root / "data.yaml").write_text("path: .\ntrain: images\nval: images\nnames:\n  0: ship\n",
                                encoding="utf-8")
yaml_path = str(root / "data.yaml")
imgs = sorted((root / "images").glob("*.png"))

split0 = split_fingerprint(yaml_path)
label0 = label_content_hash(imgs)

# -------------------------------------------------------------- 1/2 the label edits
(root / "labels" / "pohang00_L_000.txt").write_text("", encoding="utf-8")   # box deleted
split1, label1 = split_fingerprint(yaml_path), label_content_hash(imgs)
(root / "labels" / "pohang00_L_001.txt").write_text("1 0.5 0.5 0.1 0.1\n", encoding="utf-8")
split2, label2 = split_fingerprint(yaml_path), label_content_hash(imgs)
(root / "labels" / "pohang00_L_002.txt").unlink()                            # file removed
split3, label3 = split_fingerprint(yaml_path), label_content_hash(imgs)

check(
    "1 D3 reproduced: split_fingerprint is label-blind",
    split0 == split1 == split2 == split3,
    f"{split0} through a deleted box, a changed class id and a deleted file",
)
check(
    "2 the label hash moves on a deleted box",
    label1 != label0,
    f"{label0} -> {label1}",
)
check(
    "2 the label hash moves on a changed class id",
    label2 not in (label0, label1),
    f"-> {label2}",
)
check(
    "2 the label hash moves on a deleted label FILE",
    label3 not in (label0, label1, label2),
    f"-> {label3}  (a missing file still contributes its name and separator)",
)
check(
    "2 label_fingerprint is the same hash over train+val",
    label_fingerprint(yaml_path) == label_content_hash(imgs + imgs),
    "this fixture points train and val at one directory, so the list is imgs twice; "
    "the scope lives in the column NAME because the same algorithm over train alone "
    "gives a different number that means nothing against this one",
)

# --------------------------------------------------- 3 the ledger algorithm, intact
fixture = root / "images" / "pohang00_L_000.png"
expect_parts = f"pohang00/{label_path(fixture).name}\n"
import hashlib  # noqa: E402 — local to this one assertion

h = hashlib.sha256()
h.update(expect_parts.encode())
h.update((root / "labels" / "pohang00_L_000.txt").read_bytes())
h.update(b"\x00")
check(
    "3 label_content_hash is name + bytes + NUL",
    label_content_hash([fixture]) == h.hexdigest()[:12],
    "unchanged from the ledger, so runs/label_hash_ledger.csv stays comparable",
)

# ---------------------------------------------- 4 one label_path, three former ones
def lp_matching(img):        # eval/matching.label_path_for — every 'images' component
    p = Path(img)
    return Path(*[("labels" if part == "images" else part) for part in p.parts]).with_suffix(".txt")


def lp_filter(img):          # filter_night_boxes — rightmost 'images' SUBSTRING
    s = str(img)
    i = s.rfind("images")
    return Path(s[:i] + "labels" + s[i + len("images"):]).with_suffix(".txt")


def lp_verify(img):          # verify_dataset_state — last 'images' separator group
    s = str(img)
    sa, sb = f"{os.sep}images{os.sep}", f"{os.sep}labels{os.sep}"
    if sa not in s:
        sa, sb = "/images/", "/labels/"
    head, _, tail = s.rpartition(sa)
    return Path(head + sb + tail).with_suffix(".txt")


cases = [Path("d") / "visible" / "images" / "pohang00" / "pohang00_L_000006.png",
         Path("d") / "infrared" / "images" / "pohang01" / "pohang01_000408.png",
         root / "images" / "pohang00_L_000.png"]
agree = all(label_path(c) == lp_matching(c) == lp_filter(c) == lp_verify(c) for c in cases)
check(
    "4 one label_path, agreeing with all three former ones",
    agree,
    "0 disagreements over the 133,140 real train+val paths, measured before unifying",
)
# a directory whose NAME merely ends in "images" is where the substring reading breaks
tricky = Path("d") / "images" / "run_images" / "x.png"
want = Path("d") / "labels" / "run_images" / "x.txt"
check(
    "4 the shared one is component-wise, not substring",
    label_path(tricky) == lp_matching(tricky) == lp_verify(tricky) == want
    and lp_filter(tricky) != want,
    f"substring version rewrites the wrong component: {lp_filter(tricky)} - latent, "
    "never hit on this corpus, which is why it was worth collapsing while still latent",
)

# --------------------------------------------------------------- 5/6 the recipe
base = dict(epochs=25, imgsz=640, batch=32, mosaic=None, close_mosaic=None,
            optimizer="auto", patience=20, amp=True, deterministic=True,
            weights=None, train_overrides=None)
fp0, _ = recipe_identity(**base)
moved = {}
for k, v in (("epochs", 50), ("imgsz", 1280), ("batch", 16), ("mosaic", 1.0),
             ("close_mosaic", 10), ("optimizer", "SGD"), ("patience", 5),
             ("amp", False), ("deterministic", False),
             ("train_overrides", {"lr0": 0.001})):
    moved[k], _ = recipe_identity(**{**base, k: v})
check(
    "5 every recipe knob changes the fingerprint",
    all(v != fp0 for v in moved.values()) and len(set(moved.values())) == len(moved),
    f"{len(moved)} knobs, {len(set(moved.values()))} distinct fingerprints, base {fp0}",
)
fp_host, _ = recipe_identity(**base, workers=8, device="cuda:0")
check(
    "5 host-local knobs do NOT change it",
    fp_host == fp0,
    "workers/device are excluded on purpose - including them would break resume "
    "across machines, which is the one thing this lookup exists to do",
)

ckpt = root / "best.pt"
ckpt.write_bytes(b"weights version A")
fp_a, json_a = recipe_identity(**{**base, "weights": ckpt})
ckpt.write_bytes(b"weights version B")            # SAME path, different content
fp_b, _ = recipe_identity(**{**base, "weights": ckpt})
check(
    "6 the checkpoint hash is content, not path",
    fp_a != fp_b and "weights_sha256" in json_a,
    "best.pt is overwritten by every run that produces it, so a path is not an identity",
)

# ------------------------------------------------------------- 7/8 the grid gating
dataset = ("split_aaa", "labels_bbb")
row = {"dataset": dataset, "recipe": "recipe_111"}
done = {("yolo26m", "0"): row}

stale, conflict = plan_grid(done, dataset, "recipe_111", ["yolo26m"], [0])
check(
    "7 same dataset and recipe -> neither stale nor conflict",
    not stale and not conflict,
    "an interrupted grid still resumes",
)
stale, conflict = plan_grid(done, dataset, "recipe_222", ["yolo26m"], [0])
check(
    "7 D4 closed: a different recipe is refused, not skipped",
    not stale and conflict == [("yolo26m", "0")],
    "25 epochs in the CSV can no longer answer a request for 50",
)
stale, conflict = plan_grid(done, dataset, "recipe_222", ["yolo26s"], [0])
check(
    "7 a recipe change on a variant we are not running is ignored",
    not stale and not conflict,
    "the conflict is scoped to the (variant, seed) this grid will touch",
)
stale, _ = plan_grid(done, ("split_aaa", "labels_CHANGED"), "recipe_111", ["yolo26m"], [0])
check(
    "8 a different label state refuses the whole CSV",
    stale == [("yolo26m", "0")],
    "two label states must never share one results file",
)
stale, _ = plan_grid({("yolo26m", "0"): {"dataset": ("split_aaa", ""), "recipe": ""}},
                     dataset, "recipe_111", ["yolo26m"], [0])
check(
    "8 rows predating label stamping are stale",
    stale == [("yolo26m", "0")],
    "they were written against a tree that has since changed twice",
)

# The `recipe` column is JSON, so it carries commas and quotes into a CSV. Checked
# end to end rather than assumed: written through `_append_row`, read back through
# `_completed`, and the skip decision taken on what came back.
import csv as _csv          # noqa: E402
import tempfile as _tmp     # noqa: E402

from uqfusion.bench.grid import _append_row, _completed   # noqa: E402

rt = Path(_tmp.gettempdir()) / "smoke_recipe_roundtrip.csv"
rt.unlink(missing_ok=True)
fp_rt, blob_rt = recipe_identity(
    **{**base, "train_overrides": {"lr0": 0.001, "note": 'a,b "c"'}})
row = {k: "" for k in RESULT_FIELDS}
row.update({"variant": "yolo26m", "seed": 0, "split_fingerprint": "split_aaa",
            "label_fingerprint_trainval": "labels_bbb", "classes": "all",
            "recipe_fingerprint": fp_rt, "recipe": blob_rt})
_append_row(rt, row)
back = list(_csv.DictReader(open(rt, newline="", encoding="utf-8")))[0]
done_rt = _completed(rt, "all")
skip_same = plan_grid(done_rt, dataset, fp_rt, ["yolo26m"], [0])
skip_diff = plan_grid(done_rt, dataset, "other_recipe", ["yolo26m"], [0])
rt.unlink(missing_ok=True)
check(
    "9 the JSON recipe column survives the CSV",
    back["recipe"] == blob_rt,
    "commas and quotes included",
)
check(
    "9 the round-tripped row drives the decision",
    skip_same == ([], []) and skip_diff == ([], [("yolo26m", "0")]),
    "same recipe resumes, different recipe refuses - through a real CSV",
)

check(
    "8 the three columns are in the schema",
    all(c in RESULT_FIELDS for c in
        ("label_fingerprint_trainval", "recipe_fingerprint", "recipe")),
    f"{len(RESULT_FIELDS)} fields",
)

print()
if FAIL:
    raise SystemExit(f"smoke_recipe_identity: {len(FAIL)} FAILED -> {FAIL}")
print("smoke_recipe_identity: labels and recipe are part of a run's identity")
