"""Timestamped ledger of the VIS label-tree state — the check OQ-13 did not have.

On 2026-09-03 at 21:19, 7,591 `pohang01` train label files were rewritten to
their pre-filter content. No script in this project writes labels, the shell was
idle and no Python was running. It was caught only because a later hash gate
happened to run; the *window* in which it happened is known only from file
mtimes, and nothing would have caught it had it landed mid-training.

`scripts/verify_dataset_state.py --expect-label-hash` answers "is the tree right
NOW" and answers it well. What was missing is a RECORD: a durable, append-only
timeline, so that a recurrence has a bounded window instead of a guess, and so a
long run can be bracketed by a before and an after.

Three things are recorded, not one. The content hash is the decision, but a
rewrite that restores byte-identical content still moves **mtime**, and a
partial rewrite still moves the **file and box counts** — so all three travel
together and a change in any of them is visible.

Scopes are not interchangeable ([[project-night-restore-verdict]]): `train` is
what `verify_dataset_state.py --expect-label-hash` compares (`8ed69b5974ed` in
the restored tree); `all` is train+val+test, what `restore_night_perbox.py`
prints (`b92739202127`). Both are recorded, each labelled.

Exit code is 1 when the hash differs from the last recorded row, so this drops
straight into a script as a guard:

    python scripts/label_hash_ledger.py --note "before gauss_vis_nightfull"
    ... long run ...
    python scripts/label_hash_ledger.py --note "after gauss_vis_nightfull"   # rc 1 == drift

`--expect` compares against a literal instead of against history, for a run that
wants to name the tree it requires. `--accept "<reason>"` records a change the
operator knows about and exits 0; the reason is stored, so an accepted drift is
never silent.

The ledger is append-only. Nothing else is written and no label is touched.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import socket
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from filter_night_boxes import label_content_hash                   # noqa: E402
from uqfusion.config import load_config, resolve_data_yaml          # noqa: E402
from uqfusion.data.lists import load_data_yaml, split_image_list    # noqa: E402

LEDGER = ROOT / "runs/label_hash_ledger.csv"
FIELDS = ["utc", "host", "git_head", "scope", "hash", "n_label_files",
          "n_boxes", "n_images", "newest_label_mtime_utc", "verdict", "note"]


def label_paths(imgs) -> list[Path]:
    return [Path(str(p).replace("images", "labels")).with_suffix(".txt") for p in imgs]


def survey(imgs) -> dict:
    """Content hash plus the two things a hash alone cannot show."""
    lps = label_paths(imgs)
    present = [p for p in lps if p.is_file()]
    boxes = sum(sum(1 for ln in p.read_text(encoding="utf-8").splitlines() if ln.strip())
                for p in present)
    newest = max((p.stat().st_mtime for p in present), default=0.0)
    return {
        "hash": label_content_hash(list(imgs))[:12],
        "n_label_files": len(present),
        "n_boxes": boxes,
        "n_images": len(lps),
        "newest_label_mtime_utc": dt.datetime.fromtimestamp(
            newest, dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ") if newest else "",
    }


def last_row(scope: str) -> dict | None:
    if not LEDGER.is_file():
        return None
    with LEDGER.open(encoding="utf-8", newline="") as fh:
        rows = [r for r in csv.DictReader(fh) if r.get("scope") == scope]
    return rows[-1] if rows else None


def git_head() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT,
                              capture_output=True, text=True).stdout.strip() or "?"
    except OSError:
        return "?"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="vis")
    ap.add_argument("--scope", choices=("train", "all", "both"), default="both")
    ap.add_argument("--expect", default=None,
                    help="compare against this hash instead of against the last row")
    ap.add_argument("--accept", default=None, metavar="REASON",
                    help="record a change the operator knows about; exits 0")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    cfg = load_config()
    data = load_data_yaml(resolve_data_yaml(cfg, args.data))
    scopes = ("train", "all") if args.scope == "both" else (args.scope,)

    LEDGER.parent.mkdir(parents=True, exist_ok=True)
    new = not LEDGER.is_file()
    rc = 0
    with LEDGER.open("a", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            w.writeheader()
        for scope in scopes:
            imgs = (split_image_list(data, "train") if scope == "train" else
                    [p for s in ("train", "val", "test") for p in split_image_list(data, s)])
            s = survey(imgs)
            prev = last_row(scope)
            ref = args.expect or (prev["hash"] if prev else None)
            if ref is None:
                verdict = "FIRST"
            elif s["hash"] == ref:
                verdict = "MATCH"
            elif args.accept:
                verdict = "ACCEPTED"
            else:
                verdict = "DRIFT"
                rc = 1
            note = args.note
            if args.accept:
                note = f"{note} | accepted: {args.accept}".strip(" |")
            if verdict in ("DRIFT", "ACCEPTED") and prev:
                note = (f"{note} | was {prev['hash']} "
                        f"{prev['n_boxes']} boxes at {prev['utc']}").strip(" |")
            w.writerow({
                "utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "host": socket.gethostname(), "git_head": git_head(),
                "scope": scope, "verdict": verdict, "note": note, **s})
            mark = {"MATCH": "ok", "FIRST": "first record", "ACCEPTED": "accepted",
                    "DRIFT": "**DRIFT**"}[verdict]
            print(f"[ledger] {scope:5s} {s['hash']}  {s['n_label_files']:>6} files  "
                  f"{s['n_boxes']:>7} boxes  newest {s['newest_label_mtime_utc']}  {mark}")
            if verdict == "DRIFT":
                print(f"[ledger] expected {ref} -- the label tree changed since "
                      f"{prev['utc'] if prev else 'the reference'}. Do NOT train on it "
                      f"until this is explained (OQ-13).")
    print(f"[ledger] {LEDGER}")
    return rc


if __name__ == "__main__":
    sys.exit(main())
