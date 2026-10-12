"""Label-file bytes from a validated snapshot, so a scoring process does not open 12,000 files (UQF_FAST_IO=1).

On this machine opening a label file costs 2-13 ms (antivirus scan on open, contention), and a pohang04
scoring job opens ~44,000 of them: `load_gt` for the context's GT and `identity.labels_sha256` for every
cache's identity check. That was a third of a job's time (profile 2026-10-11).

A snapshot holds, per label directory, every file's (size, mtime_ns, bytes). A process lists the directory
once (`os.scandir`: on Windows the listing carries size and mtime, no per-file call) and serves a file from
the snapshot only if its current size and mtime_ns equal the snapshot's; any other file is read from disk
as before. A file absent from the listing does not exist. The bytes served are the file's bytes, so
`load_gt` and `labels_sha256` compute exactly what they compute from disk.

    py -3.13 -m uqfusion.eval.labelcache Pohang_dataset/visible/labels Pohang_dataset/infrared/labels
"""

from __future__ import annotations

import hashlib
import os
import pickle
import sys
from pathlib import Path

CACHE_DIR = Path(__file__).resolve().parents[3] / "runs" / ".label_cache"
_STATE: dict = {}
_FALLBACK = object()


def _key(name: str) -> str:
    return os.path.normcase(name)            # Windows paths are case-insensitive


def _snap_path(d: Path) -> Path:
    h = hashlib.sha1(os.path.normcase(str(Path(d).resolve())).encode("utf-8")).hexdigest()[:20]
    return CACHE_DIR / f"{h}.pkl"


def _dir_state(d: Path):
    """(names in the directory, {name: bytes} valid in the snapshot), None if the directory does not exist,
    or `_FALLBACK` if it cannot be listed."""
    key = os.path.normcase(str(d))
    if key in _STATE:
        return _STATE[key]
    try:
        with os.scandir(d) as it:
            cur = {_key(e.name): e for e in it if e.is_file()}
    except (FileNotFoundError, NotADirectoryError):
        _STATE[key] = None
        return None
    except OSError:
        _STATE[key] = _FALLBACK
        return _FALLBACK
    valid = {}
    p = _snap_path(d)
    if p.is_file():
        with open(p, "rb") as fh:
            saved = pickle.load(fh)                       # {name: (size, mtime_ns, bytes)}
        for k, e in cur.items():
            s = saved.get(k)
            if s is not None:
                st = e.stat()
                if st.st_size == s[0] and st.st_mtime_ns == s[1]:
                    valid[k] = s[2]
    _STATE[key] = (set(cur), valid)
    return _STATE[key]


def read_label_bytes(lp: Path) -> bytes | None:
    """The bytes of label file `lp`, or None if it does not exist."""
    lp = Path(lp)
    st = _dir_state(lp.parent)
    if st is None:
        return None
    if st is _FALLBACK:
        return lp.read_bytes() if lp.is_file() else None
    names, valid = st
    k = _key(lp.name)
    if k not in names:
        return None
    b = valid.get(k)
    return b if b is not None else lp.read_bytes()


def build(dirs) -> None:
    """Snapshot every file of each directory (recursing one level into run sub-directories)."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    todo = []
    for d in map(Path, dirs):
        todo.append(d)
        todo += [p for p in d.iterdir() if p.is_dir()]
    for d in todo:
        saved = {}
        with os.scandir(d) as it:
            for e in it:
                if e.is_file():
                    st = e.stat()                         # before the read: a later change misses, never matches
                    with open(e.path, "rb") as fh:
                        saved[_key(e.name)] = (st.st_size, st.st_mtime_ns, fh.read())
        if not saved:
            continue
        p = _snap_path(d)
        tmp = p.with_suffix(".tmp")
        with open(tmp, "wb") as fh:
            pickle.dump(saved, fh, protocol=pickle.HIGHEST_PROTOCOL)
        os.replace(tmp, p)
        print(f"[labelcache] {d}: {len(saved)} files -> {p.name}", flush=True)


if __name__ == "__main__":
    build(sys.argv[1:])
