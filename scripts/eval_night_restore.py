"""Score the night-label restore against its pre-registered rule (I5).

`docs/prereg-night-label-restore.md`, committed `030244e` before any label was
touched. This script does not choose anything -- every threshold below is read
off that document, and the arms are fixed:

* **new** -- `runs/full_scale/gauss_vis_nightrestore/weights/best.pt`, fine-tuned
  on the restored labels (+94,553 boxes back into `pohang01` VIS train).
* **old** -- `runs/full_scale/gauss_vis_seed0/weights/best.pt`, the shipped
  checkpoint, trained when every `pohang01` train label was empty.

**Primary endpoint** (prereg rule 2): VIS-only `mAP@50-95` on the 2,068 night val
frames. Night val GT is **16,179 boxes, all class 0** -- there is not one buoy in
it -- and `presort`/`ap_from_parts` take their class list from GT, so the
project's own `map50_95` is already ship-only on this subset. No class filter is
applied or needed, and no buoy zero is averaged in.

    DEAD  < 0.005 | WEAK 0.005-0.02 | ALIVE >= 0.02

**Guard** (prereg rule 3): VIS-only `mAP@50-95` on the 1,200 paired day frames as
a PAIRED delta with its own bootstrap sd. Restore is REJECTED if

    delta_day < -max(2 * se_paired, 0.002)

whatever the night number does. The absolute 0.002 floor is in the prereg because
a margin alone degenerates into a sign test (log SS4.7, SS4.10).

Two notes on what the guard set is. The 2,232-frame paired list is 1,032
`pohang01` + 1,200 day (836 `pohang00`, 247 `pohang02`, 117 `pohang03`), so 364
of the guard frames are TEST. That is consistent with prereg rule 4 and with the
project's C7/C8 discipline: TEST is being used to REJECT, never to select, and
nothing here is tuned. And the old arm's day predictions are not recomputed --
`runs/cache_m/gauss_vis_paired_clean.pkl` already holds them, and its meta is
asserted to match (same weights, imgsz 640, conf 0.001) before it is trusted.

**Reported but explicitly NOT decision inputs** (prereg rule 5): night `mAP@50`,
per-class day AP, the night delta interval, and the paired-night cross-check. The
fused benchmark is not run at all: `veto_vis` fires on 100% of night frames, so
the fused night number is `ir_only` by construction and cannot move.

Caches go to a NEW directory (`runs/cache_nightrestore/`); nothing under
`runs/cache/`, `runs/cache_m/` or `runs/eval/` is overwritten, and `write_md`
refuses an existing `--out`.

Usage:
    python scripts/eval_night_restore.py
    python scripts/eval_night_restore.py --boot 2000 --out runs/eval/night_restore_verdict.md
"""

from __future__ import annotations

import argparse
import re
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))

from _ideas_common import fmt, gts_for, md_table, sgn, write_md      # noqa: E402
from uqfusion.eval.apmetrics import ap_from_parts, bootstrap_delta, frame_parts  # noqa: E402
from uqfusion.eval.cache import build_cache, load_cache             # noqa: E402

NEW_W = ROOT / "runs/full_scale/gauss_vis_nightrestore/weights/best.pt"
OLD_W = ROOT / "runs/full_scale/gauss_vis_seed0/weights/best.pt"
OLD_PAIRED = ROOT / "runs/cache_m/gauss_vis_paired_clean.pkl"
PAIRED_LIST = ROOT / "runs/derived/paired_val_vis.txt"
VAL_LIST = ROOT / "Pohang_dataset/visible/val.txt"
CACHE_DIR = ROOT / "runs/cache_nightrestore"

NIGHT_RUN = "pohang01"
BAND_DEAD, BAND_ALIVE = 0.005, 0.02
GUARD_FLOOR = 0.002


def run_of(p: str) -> str:
    m = re.search(r"(pohang\d\d)", str(p).replace("\\", "/"))
    return m.group(1) if m else "?"


def night_frames() -> list[str]:
    """The 2,068 `pohang01` val frames, as absolute paths in val.txt order."""
    root = VAL_LIST.parent.resolve()
    rows = [l.strip() for l in VAL_LIST.read_text(encoding="utf-8").splitlines()
            if l.strip()]
    return [str((root / r.lstrip("./")).resolve()) for r in rows
            if NIGHT_RUN in r]


def paired_frames() -> list[str]:
    return [l.strip() for l in PAIRED_LIST.read_text(encoding="utf-8").splitlines()
            if l.strip()]


def cached(weights: Path, images: list[str], tag: str, imgsz: int, conf: float):
    """Predictions for one (checkpoint, frame list). Reused if already built."""
    out = CACHE_DIR / f"{tag}.pkl"
    if out.is_file():
        recs, meta = load_cache(out)
        assert len(recs) == len(images), \
            f"{out} holds {len(recs)} records, expected {len(images)}"
        assert meta.get("imgsz") == imgsz and meta.get("conf") == conf, \
            f"{out} was built at imgsz {meta.get('imgsz')} conf {meta.get('conf')}"
        print(f"[cache] reuse {out.name} ({len(recs)} frames)", flush=True)
        return recs
    from uqfusion.uq.infer import UQPredictor
    t = time.time()
    print(f"[cache] building {out.name}: {len(images)} frames "
          f"from {weights.relative_to(ROOT)}", flush=True)
    predictor = UQPredictor(str(weights), device=0, imgsz=imgsz, conf=conf)
    build_cache(predictor, images, out,
                meta={"weights": [str(weights.relative_to(ROOT))], "imgsz": imgsz,
                      "conf": conf, "tag": tag}, log_every=500)
    print(f"[cache] {out.name} in {time.time() - t:.0f}s", flush=True)
    return load_cache(out)[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--new", default=str(NEW_W))
    ap.add_argument("--old", default=str(OLD_W))
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--conf", type=float, default=0.001,
                    help="must match runs/cache_m/gauss_vis_paired_clean.pkl")
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="runs/eval/night_restore_verdict.md")
    args = ap.parse_args()
    t0 = time.time()

    new_w, old_w = Path(args.new), Path(args.old)
    assert new_w.is_file(), (
        f"missing {new_w} -- the fine-tune has not produced a best.pt yet. "
        f"This script is the pre-registered scorer; it does not train.")
    assert old_w.is_file(), f"missing shipped weights {old_w}"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # ---- old arm on the paired frames: reuse, but verify before trusting -----
    assert OLD_PAIRED.is_file(), f"missing {OLD_PAIRED}"
    old_pair_recs, old_pair_meta = load_cache(OLD_PAIRED)
    assert old_pair_meta.get("weights") == [str(old_w.relative_to(ROOT)).replace("\\", "/")], \
        f"{OLD_PAIRED} was built from {old_pair_meta.get('weights')}, not {old_w}"
    assert old_pair_meta.get("imgsz") == args.imgsz, \
        f"{OLD_PAIRED} imgsz {old_pair_meta.get('imgsz')} != {args.imgsz}"
    assert old_pair_meta.get("conf") == args.conf, \
        f"{OLD_PAIRED} conf {old_pair_meta.get('conf')} != {args.conf}"
    assert old_pair_meta.get("corrupt") in (None, "none"), \
        f"{OLD_PAIRED} is a corrupted cache ({old_pair_meta.get('corrupt')})"

    pf = paired_frames()
    assert len(pf) == len(old_pair_recs) == 2232, \
        f"paired list {len(pf)} vs cache {len(old_pair_recs)}"
    for a, b in zip(pf, (r["image_path"] for r in old_pair_recs)):
        assert Path(a) == Path(b), f"paired list order differs from cache: {a} vs {b}"

    nf = night_frames()
    assert len(nf) == 2068, f"night val is {len(nf)} frames, expected 2068"

    new_pair_recs = cached(new_w, pf, "new_paired_clean", args.imgsz, args.conf)
    new_night_recs = cached(new_w, nf, "new_val_night", args.imgsz, args.conf)
    old_night_recs = cached(old_w, nf, "old_val_night", args.imgsz, args.conf)

    # ---- parts --------------------------------------------------------------
    night_gts = gts_for(new_night_recs)
    n_gt_boxes = int(sum(len(g["cls"]) for g in night_gts))
    n_gt_cls = sorted({int(c) for g in night_gts for c in g["cls"]})
    assert n_gt_cls == [0], f"night val GT classes {n_gt_cls}, expected ship only"

    p_new_n = frame_parts(new_night_recs, night_gts)
    p_old_n = frame_parts(old_night_recs, night_gts)

    pair_gts = gts_for(new_pair_recs)
    p_new_p = frame_parts(new_pair_recs, pair_gts)
    p_old_p = frame_parts(old_pair_recs, pair_gts)
    runs = np.array([run_of(r["image_path"]) for r in new_pair_recs])
    day_idx = np.flatnonzero(runs != NIGHT_RUN)
    night_idx = np.flatnonzero(runs == NIGHT_RUN)
    assert len(day_idx) == 1200, f"{len(day_idx)} paired day frames, expected 1200"

    # ---- primary endpoint ---------------------------------------------------
    a_new = ap_from_parts(p_new_n)
    a_old = ap_from_parts(p_old_n)
    night_new, night_old = a_new["map50_95"], a_old["map50_95"]
    band = ("DEAD" if night_new < BAND_DEAD else
            "WEAK" if night_new < BAND_ALIVE else "ALIVE")
    d_night = bootstrap_delta(p_new_n, p_old_n, n_boot=args.boot, seed=args.seed)
    print(f"[night] new {night_new:.4f}  old {night_old:.4f} -> {band}", flush=True)

    # ---- guard --------------------------------------------------------------
    d_day = bootstrap_delta(p_new_p, p_old_p, sel=day_idx,
                            n_boot=args.boot, seed=args.seed)
    floor = -max(2.0 * d_day["se"], GUARD_FLOOR)
    rejected = d_day["delta"] < floor
    which = "2 x se" if 2.0 * d_day["se"] > GUARD_FLOOR else "absolute 0.002"
    print(f"[guard] day delta {d_day['delta']:+.4f} vs floor {floor:+.4f} "
          f"-> {'REJECTED' if rejected else 'PASS'}", flush=True)

    d_pair_night = bootstrap_delta(p_new_p, p_old_p, sel=night_idx,
                                   n_boot=args.boot, seed=args.seed)
    day_new = ap_from_parts(p_new_p, sel=day_idx)
    day_old = ap_from_parts(p_old_p, sel=day_idx)

    verdict = ("REJECTED by the day guard" if rejected
               else f"{band} on night, day guard PASSES")

    # ---- report -------------------------------------------------------------
    def pc(r, c):
        d = r["per_class"].get(c)
        return fmt(d["ap50_95"]) if d else "--"

    band_rows = [
        ["DEAD", "< 0.005", "filter was right; night VIS is blind by physics"],
        ["WEAK", "0.005 - 0.02", "learnable but marginal; record, do not touch the veto"],
        ["ALIVE", ">= 0.02", "night VIS is real; the 100% veto needs its own prereg"],
    ]

    secs = [
        f"**Verdict: {verdict}.**\n\n"
        f"Scored against `docs/prereg-night-label-restore.md` (committed `030244e`, "
        f"before any label was changed). Nothing in this run selects a parameter; "
        f"every threshold below was fixed in advance.\n\n"
        f"* new = `{new_w.relative_to(ROOT)}`\n"
        f"* old = `{old_w.relative_to(ROOT)}` (shipped)\n"
        f"* imgsz {args.imgsz}, conf {args.conf}, {args.boot} paired bootstrap "
        f"resamples, seed {args.seed}",

        "## 1. Primary endpoint — VIS alone on night val\n\n"
        f"2,068 `pohang01` val frames, **{n_gt_boxes:,} GT boxes, all class 0**. Val "
        "labels were never filtered. Because `presort` takes its class list from GT, "
        "`map50_95` here *is* ship AP — no buoy zero is averaged in.\n\n"
        + md_table(["arm", "mAP@50-95", "mAP@50", "ship AP@50-95"],
                   [["new (restored labels)", fmt(night_new), fmt(a_new["map50"]),
                     pc(a_new, 0)],
                    ["old (shipped)", fmt(night_old), fmt(a_old["map50"]),
                     pc(a_old, 0)]])
        + f"\n\ndelta **{sgn(d_night['delta'])}** "
          f"[{fmt(d_night['ci_lo'])}, {fmt(d_night['ci_hi'])}], "
          f"se {fmt(d_night['se'])}, sign-flip {d_night['p_sign_flip']:.3f}"
          f"{' (spans zero)' if d_night['spans_zero'] else ''}. "
          "The interval is reported, not a decision input — the bands are absolute.\n\n"
        + md_table(["band", "range", "what it licenses"], band_rows,
                   align="lll")
        + f"\n\nNight VIS scores **{fmt(night_new)}** → **{band}**.",

        "## 2. Guard — did the day model regress?\n\n"
        "1,200 paired day frames (836 `pohang00`, 247 `pohang02`, 117 `pohang03`), "
        "scored as a PAIRED delta so frame-composition noise cancels. 364 frames are "
        "TEST, used here to reject and never to select — nothing in this run is tuned.\n\n"
        + md_table(["arm", "day mAP@50-95", "ship AP", "buoy AP"],
                   [["new", fmt(day_new["map50_95"]), pc(day_new, 0), pc(day_new, 1)],
                    ["old", fmt(day_old["map50_95"]), pc(day_old, 0), pc(day_old, 1)]])
        + f"\n\ndelta **{sgn(d_day['delta'])}** "
          f"[{fmt(d_day['ci_lo'])}, {fmt(d_day['ci_hi'])}], se {fmt(d_day['se'])}.\n\n"
          f"Rejection floor `-max(2 x se, 0.002)` = **{sgn(floor)}** (binding term: "
          f"{which}). Observed {sgn(d_day['delta'])} → "
          f"**{'REJECTED' if rejected else 'PASS'}**.",

        "## 3. Reported, not decided\n\n"
        "Prereg rule 5. None of this moves the verdict.\n\n"
        + md_table(["quantity", "new", "old", "delta"],
                   [["night val mAP@50", fmt(a_new["map50"]), fmt(a_old["map50"]),
                     sgn(a_new["map50"] - a_old["map50"])],
                    ["paired night subset (1,032 fr)", fmt(d_pair_night["a"]),
                     fmt(d_pair_night["b"]), sgn(d_pair_night["delta"])],
                    ["paired day subset (1,200 fr)", fmt(d_day["a"]),
                     fmt(d_day["b"]), sgn(d_day["delta"])]])
        + "\n\nThe fused benchmark is deliberately not run. `veto_vis` fires on 100% "
          "of night frames, so the fused night number is `ir_only` by construction and "
          "cannot move whatever this detector learned. Scoring the restore on it would "
          "manufacture a null that means nothing.",

        "## 4. What a null here does and does not settle\n\n"
        "Stated in the prereg in advance, repeated so the number is not over-read: "
        "this is a **fine-tune from a checkpoint trained on empty night labels**, so "
        "the initialisation already encodes \"night frames contain nothing\". An "
        "**ALIVE** verdict is strong evidence because it had to overcome that prior. "
        "A **DEAD** verdict is **provisional** — it cannot separate \"night VIS is "
        "blind\" from \"25 epochs could not undo the initialisation\". Settling that "
        "needs a from-scratch run (~13.7 h), which the prereg does not authorise.",

        f"---\n\n_Generated by `scripts/eval_night_restore.py` in "
        f"{time.time() - t0:.1f}s. Caches under `runs/cache_nightrestore/`; "
        f"nothing published was overwritten._",
    ]
    write_md(args.out, "Night label restore — pre-registered verdict", secs)
    print(f"[done] {verdict}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
