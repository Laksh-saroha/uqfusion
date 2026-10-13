"""Read + corrupt frames on every core, hand them to the GPU loop in order.

Cache building was single-threaded: `cv2.imread` -> corruption -> model, one frame at a
time, so a v1 fog cache spent ~234 ms/frame on one core while the GPU (~18 ms/frame per
model) and the other 31 logical cores idled. Here the read and the corruption run in a
process pool and the main process only does inference.

Order and determinism: the corruption is seeded per (seed, frame index), never by worker
or arrival order, so a pooled build is bit-identical to the serial one (checked by
`build_cache_multi.py --verify-against`). Frames come back strictly in list order.

Memory is bounded: at most `window` chunks are in flight, so a slow GPU never lets
2,232 decoded 640x640 frames (2.7 GB) pile up in the parent.

Optionally each worker also computes the gate's frame statistics (brightness and
structure, content rows only) on the corrupted frame, so the statistics no longer need
a second replay of the corruption.
"""

from __future__ import annotations

import os
import sys
from collections import deque
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

_W: dict = {}
ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")   # workers inherit it: no network check per process


def default_workers(reserve: int = 2) -> int:
    return max(1, (os.cpu_count() or 2) - reserve)


def _stats_funcs():
    sys.path.insert(0, str(ROOT / "scripts"))
    import cv2

    import frame_brightness as fb
    import frame_structure as fs

    fs.cv2 = cv2      # frame_structure imports cv2 inside main(); its frame_stats needs it too
    return fb, fs


def _init(spec: dict) -> None:
    import cv2

    cv2.setNumThreads(1)
    from uqfusion.eval.corruptions import make_corruption

    _W["images"] = spec["images"]
    _W["tf"] = (make_corruption(spec["corrupt"], spec["severity"], spec["corrupt_seed"],
                                version=spec.get("corrupt_version", "v1"), params=spec.get("params"),
                                modality=spec.get("modality", "vis"), images=spec["images"])
                if spec.get("corrupt") else None)
    _W["stats"] = spec.get("stats")
    if _W["stats"]:
        fb, fs = _stats_funcs()
        lo, hi = fb.content_rows(_W["stats"])
        if (lo, hi) != fs.content_rows(_W["stats"]):
            raise RuntimeError("frame_brightness and frame_structure disagree on content rows")
        _W["fb"], _W["fs"], _W["rows"] = fb, fs, (lo, hi)


def _one(i: int):
    import cv2

    p = _W["images"][i]
    im = cv2.imread(str(p))
    if im is None:
        raise FileNotFoundError(f"could not read image: {p}")
    if _W["tf"] is not None:
        im = _W["tf"](im, i)
    st = None
    if _W["stats"]:
        lo, hi = _W["rows"]
        g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[lo:hi]
        tag = {"image_path": str(p), "run": Path(p).parent.name}
        st = ({**_W["fb"].frame_stats(g), **tag}, {**_W["fs"].frame_stats(g), **tag})
    return i, im, st


def _chunk(idx: list[int]):
    return [_one(i) for i in idx]


def corrupted_frames(images: list, spec: dict | None = None, workers: int | None = None,
                     chunk: int = 4, window: int | None = None):
    """Yield (index, frame_bgr, stats_or_None) for every image, in order.

    spec: {"corrupt", "severity", "corrupt_seed", "corrupt_version", "modality", "stats"}
    (corrupt None/absent = clean frames; stats = "vis"/"ir" to also compute gate stats).
    workers=0 runs in-process, which is the reference the pooled path must reproduce.
    """
    spec = {**(spec or {}), "images": [str(p) for p in images]}
    n = len(images)
    if workers == 0:
        _init(spec)
        for i in range(n):
            yield _one(i)
        return
    workers = workers or default_workers()
    window = window or 4 * workers
    chunks = [list(range(a, min(a + chunk, n))) for a in range(0, n, chunk)]
    with ProcessPoolExecutor(workers, initializer=_init, initargs=(spec,)) as ex:
        q: deque = deque()
        it = iter(chunks)
        for c in it:
            q.append(ex.submit(_chunk, c))
            if len(q) >= window:
                break
        while q:
            for rec in q.popleft().result():
                yield rec
            nxt = next(it, None)
            if nxt is not None:
                q.append(ex.submit(_chunk, nxt))
