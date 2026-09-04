"""Can IR check its OWN health well enough to be trusted with the night switch?

`probe_ir_night_robustness.py` found the hole: the deployed rule `ir_p05 > 34`
misreads a fogged IR day frame as night on 75-96% of frames (glare: 31-41%), and
a false night vetoes VIS on a clear day -- trading 0.3683 for 0.0177. The other
direction, IR lowlight, misses night 100% of the time, which costs at most the
-0.0022..-0.0054 that `no_veto` costs a night cell. The asymmetry is enormous, so
the fail-safe direction is obvious: when IR cannot be trusted, DO NOT let it veto
VIS.

That needs an IR self-check, and the check has an awkward requirement. It must

    fire   on corrupted IR (any kind), day or night
    stay quiet on CLEAN IR -- and crucially on clean NIGHT IR, because that is
           precisely when the night switch has to work

A novelty bound fitted on clean DAY IR cannot do this: night IR is genuinely
different from day IR, which is the entire basis of the night test, so such a
bound fires on the night frames it must not touch. The check therefore has to use
a statistic that is stable across day and night on clean data while still moving
under corruption. This probe measures which statistics, if any, have that shape.

Measurement only. Read-only on the dataset.

Usage:
    python scripts/probe_ir_selfcheck.py --stride 8 --kinds fog glare
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from frame_structure import frame_stats               # noqa: E402
from uqfusion.eval.cache import load_cache            # noqa: E402
from uqfusion.eval.corruptions import make_corruption  # noqa: E402

NIGHT_RUN = "pohang01"
FIT_RUNS = ("pohang00", "pohang02", "pohang03")
NATIVE_IR = (640, 512)


def content_rows(canvas: int = 640) -> tuple[int, int]:
    w0, h0 = NATIVE_IR
    scale = min(canvas / w0, canvas / h0)
    nh = round(h0 * scale)
    top = (canvas - nh) // 2
    return top, top + nh


def main() -> int:
    import cv2

    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--cache", default="runs/cache/gauss_ir_paired_clean.pkl")
    ap.add_argument("--out", default="runs/derived/ir_selfcheck")
    ap.add_argument("--stride", type=int, default=8)
    ap.add_argument("--kinds", nargs="+", default=["fog", "glare", "rain", "blur",
                                                   "noise", "lowlight"])
    ap.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    recs = load_cache(ROOT / args.cache)[0][::args.stride]
    runs = np.asarray([Path(r["image_path"]).parent.name for r in recs])
    lo, hi = content_rows()
    t0 = time.time()

    ims = []
    for r in recs:
        im = cv2.imread(r["image_path"])                # READ-ONLY
        if im is None:
            raise FileNotFoundError(r["image_path"])
        ims.append(im)
    print(f"[irself] {len(ims)} IR frames loaded ({time.time() - t0:.0f}s)", flush=True)

    def stats_of(images) -> dict[str, np.ndarray]:
        rows = [frame_stats(cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[lo:hi]) for im in images]
        return {k: np.asarray([r[k] for r in rows], dtype=float) for k in rows[0]}

    out = {"runs": runs.tolist(), "stride": args.stride, "seed": args.seed, "arms": {}}
    out["arms"]["clean|0"] = {k: v.tolist() for k, v in stats_of(ims).items()}
    print(f"[irself] clean done ({time.time() - t0:.0f}s)", flush=True)
    for kind in args.kinds:
        for s in args.severities:
            tf = make_corruption(kind, s, args.seed)
            v = stats_of([tf(im, i * args.stride) for i, im in enumerate(ims)])
            out["arms"][f"{kind}|{s}"] = {k: x.tolist() for k, x in v.items()}
            print(f"[irself] {kind} s{s} done ({time.time() - t0:.0f}s)", flush=True)

    od = ROOT / args.out
    od.mkdir(parents=True, exist_ok=True)
    tag = "-".join(args.kinds)
    p = od / f"ir_stats_stride{args.stride}_{tag}.json"
    p.write_text(json.dumps(out), encoding="utf-8")
    print(f"[irself] wrote {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
