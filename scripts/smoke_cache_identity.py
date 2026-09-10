"""R-E1 slice 2 acceptance gate — a cache must be able to REFUSE.

Written 2026-09-10 for finding F14. Slice 1 (`f3364c9`) recorded what produced a
result; its own docstring listed what it did not do: *"no ordered pair IDs ... and
nothing here is validated on cache load"*. This is that part, and this script is
what keeps it honest.

Every defect below was reproduced on the real caches before any code moved
(`docs/cache-identity-2026-09-10.md`):

  * `load_cache` accepted a payload claiming `n_frames` 99999 while holding 10
    records, and records stripped of every prediction key;
  * `evaluate_systems` checked equal LENGTHS, so a fully reversed IR cache passed
    and moved gated fusion by **-0.025630** — and a ONE-FRAME shift moved it by
    **-0.000968**, below the 0.0014–0.0031 paired noise floor, where no statistical
    check could ever find it.

Cases:
  1. `pair_id` is a within-modality frame identity, and is NOT the pairing key
  2. the dataset pair table is the pairing key, and it disagrees with frame number
  3. an aligned pair passes with zero unidentifiable frames
  4. reversal, shift-1 and shift-100 are all refused
  5. `load_cache` refuses every damaged payload it used to accept
  6. a stamped `frames_sha256` / `labels_sha256` refuses tampering
  7. real caches on disk still load unchanged

Usage:  python scripts/smoke_cache_identity.py
"""

from __future__ import annotations

import glob
import pickle
import random
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from uqfusion.eval.cache import load_cache
from uqfusion.eval.identity import (
    _pair_table,
    assert_paired,
    frames_sha256,
    pair_id,
    paired_id_report,
)

FAIL = []


def check(name, cond, detail=""):
    print(f"[smoke] {name:<46} {'PASS' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAIL.append(name)


def refuses(fn, *a, **kw):
    """True when `fn` raises ValueError — the only acceptable outcome for bad input."""
    try:
        fn(*a, **kw)
        return False
    except ValueError:
        return True


# ------------------------------------------------------------ 1 within-modality id
check(
    "1 pair_id is run/ordinal",
    pair_id(r"X:\d\visible\images\pohang00\pohang00_L_006767.png") == "pohang00/6767"
    and pair_id(r"X:\d\infrared\images\pohang00\pohang00_006767.png") == "pohang00/6767",
    "same on both modalities in pohang00 - which is exactly what made it misleading",
)
check(
    "1 no ordinal -> None, not an error",
    pair_id("synthetic_frame.png") is None,
    "smoke fixtures have no ordinals; refusing there would make this untestable",
)

# --------------------------------------------------- 2 the pair table is the truth
table = _pair_table()
if not table:
    check("2 pair table present", False, "no *_pairs.csv found — dataset absent?")
else:
    offset = {}
    for v, i in table.items():
        run = v.split("_")[0]
        dv = int("".join(c for c in Path(v).stem.split("_")[-1] if c.isdigit()))
        di = int("".join(c for c in Path(i).stem.split("_")[-1] if c.isdigit()))
        offset.setdefault(run, set()).add(di - dv)
    disagree = sum(1 for v, i in table.items()
                   if Path(v).stem.split("_")[-1] != Path(i).stem.split("_")[-1])
    check(
        "2 pair table loaded",
        len(table) > 20000,
        f"{len(table)} rows",
    )
    check(
        "2 frame number is NOT the pairing key",
        disagree > 0,
        f"{disagree} of {len(table)} rows carry different indices; per-run IR-VIS offsets "
        + ", ".join(f"{r}:{sorted(o)}" for r, o in sorted(offset.items())),
    )

# ------------------------------------------------------ 3/4 alignment is enforced
names = [(v, i) for v, i in sorted(table.items()) if v.startswith("pohang01")][:200]
if len(names) < 50:
    check("3 fixture built from the pair table", False, f"only {len(names)} rows")
else:
    vis = [{"image_path": f"/d/visible/images/pohang01/{v}", "conf": [], "cls": [],
            "boxes_xyxy": []} for v, _ in names]
    ir = [{"image_path": f"/d/infrared/images/pohang01/{i}", "conf": [], "cls": [],
           "boxes_xyxy": []} for _, i in names]
    rep = paired_id_report(vis, ir)
    check(
        "3 aligned pair passes, non-vacuously",
        rep["n_mismatched"] == 0 and rep["n_unidentifiable"] == 0,
        f"{rep['n_vis']} frames, 0 mismatched, 0 unidentifiable",
    )
    check(
        "4 reversal refused",
        refuses(assert_paired, vis, list(reversed(ir))),
        "gated fusion moves -0.025630 when this slips through",
    )
    check(
        "4 one-frame shift refused",
        refuses(assert_paired, vis, ir[1:] + ir[:1]),
        "moves -0.000968 - BELOW the noise floor, so only identity can catch it",
    )
    check(
        "4 shift-100 refused",
        refuses(assert_paired, vis, ir[100:] + ir[:100]),
        "moves -0.027837",
    )
    check(
        "4 length mismatch still refused",
        refuses(assert_paired, vis, ir[:-1]),
        "the check that already existed must survive the new one",
    )
    # a frame the table does not know must be COUNTED, never silently passed
    unknown = [dict(r, image_path="/d/visible/images/x/no_such_frame_1.png") for r in vis]
    rep_u = paired_id_report(unknown, ir)
    check(
        "4 unknown frames counted, not passed",
        rep_u["n_mismatched"] == 0 and rep_u["n_unidentifiable"] == len(vis),
        f"n_unidentifiable {rep_u['n_unidentifiable']} - a caller can see the pass was vacuous",
    )

# --------------------------------------------------------- 5/6 load_cache refuses
tmp = Path(tempfile.gettempdir()) / "smoke_cache_identity.pkl"


def write(payload):
    with open(tmp, "wb") as f:
        pickle.dump(payload, f)
    return tmp


good_rec = {"image_path": "/d/visible/images/pohang00/pohang00_L_000006.png",
            "conf": [0.5], "cls": [0], "boxes_xyxy": [[0, 0, 1, 1]]}

check(
    "5 non-cache payload refused",
    refuses(load_cache, write({"model": "not a cache"})),
    "gate_lab.pkl and friends now refuse instead of half-loading",
)
check(
    "5 n_frames disagreement refused",
    refuses(load_cache, write({"meta": {"n_frames": 99999}, "records": [good_rec]})),
    "claimed 99999, holds 1 - this used to load",
)
check(
    "5 record without image_path refused",
    refuses(load_cache, write({"meta": {"n_frames": 1}, "records": [{"conf": [0.5]}]})),
    "it could not be identified, paired, or scored",
)
check(
    "5 inconsistent per-detection arrays refused",
    refuses(load_cache, write({"meta": {"n_frames": 1},
                               "records": [dict(good_rec, cls=[0, 1])]})),
    "conf 1 vs cls 2 - a truncated write",
)
check(
    "5 a well-formed cache still loads",
    load_cache(write({"meta": {"n_frames": 1}, "records": [good_rec]}))[0] == [good_rec],
    "no false positives on the shape every existing cache has",
)

stamped = {"meta": {"n_frames": 1, "frames_sha256": frames_sha256([good_rec])},
           "records": [good_rec]}
check(
    "6 stamped frames_sha256 verifies",
    load_cache(write(stamped))[1]["frames_sha256"] == frames_sha256([good_rec]),
    stamped["meta"]["frames_sha256"],
)
tampered = {"meta": dict(stamped["meta"]),
            "records": [dict(good_rec, image_path="/d/visible/images/pohang00/pohang00_L_000007.png")]}
check(
    "6 reordered/substituted records refused",
    refuses(load_cache, write(tampered)),
    "the stamp no longer matches the records it describes",
)
check(
    "6 labels_sha256 mismatch refused",
    refuses(load_cache, write({"meta": {"n_frames": 1, "labels_sha256": "0" * 16},
                               "records": [good_rec]})),
    "the GT files moved under the cache — what split_fingerprint cannot see",
)
check(
    "6 validate=False is the only way past",
    load_cache(write(tampered), validate=False)[1]["frames_sha256"] is not None,
    "an explicit parameter, so skipping the check is visible at the call site",
)
tmp.unlink(missing_ok=True)

# ------------------------------------------------------------- 7 real caches load
real = sorted(glob.glob("runs/cache*/**/*.pkl", recursive=True))
random.Random(0).shuffle(real)
sample = real[:12]
bad = []
for p in sample:
    try:
        load_cache(p)
    except Exception as e:  # noqa: BLE001 — the point is to report, not to crash
        bad.append((p, str(e)[:70]))
check(
    "7 existing caches unaffected",
    sample and not bad,
    f"{len(sample)} sampled, {len(bad)} refused" + (f" -> {bad[0]}" if bad else ""),
)

print()
if FAIL:
    raise SystemExit(f"smoke_cache_identity: {len(FAIL)} FAILED -> {FAIL}")
print("smoke_cache_identity: a cache can now refuse")
