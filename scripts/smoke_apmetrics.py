"""Pins `apmetrics` to `matching.map50_95` — MUST pass before any CI is quoted.

Three layers have to agree or the bootstrap is measuring its own arithmetic:

  A. `ap_from_parts` == `matching.map50_95`, pooled and on frame subsets
  B. `ap_weighted` at unit weights == `ap_from_parts`
  C. `ap_weighted` at integer weights == `ap_from_parts` on the physically
     duplicated frame list (the literal resample)

C is the one worth stating: a bootstrap draw is a dataset with frames repeated,
and AP is not linear in the frames, so "weight it" and "duplicate it" are only
the same thing if the duplication happens in conf order. It does, and this
asserts it rather than assuming it.

Runs on synthetic records — no caches, no GPU, ~2 s.

Usage:  python scripts/smoke_apmetrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, ap_weighted, frame_parts, presort  # noqa: E402
from uqfusion.eval.matching import map50_95  # noqa: E402

TOL = 1e-12


def synth(n_frames=60, seed=0):
    """Frames with real overlap structure: GT boxes plus jittered predictions,
    some frames empty, both classes present, confidences overlapping."""
    rng = np.random.default_rng(seed)
    records, gts = [], []
    for _ in range(n_frames):
        n_gt = int(rng.integers(0, 6))
        gb = np.column_stack([
            rng.uniform(0, 560, n_gt), rng.uniform(0, 560, n_gt),
            np.zeros(n_gt), np.zeros(n_gt)])
        gb[:, 2] = gb[:, 0] + rng.uniform(10, 70, n_gt)
        gb[:, 3] = gb[:, 1] + rng.uniform(10, 70, n_gt)
        gcls = rng.integers(0, 2, n_gt)
        gts.append({"boxes_xyxy": gb.astype(float), "cls": gcls.astype(int)})

        n_p = int(rng.integers(0, 9))
        if n_gt and n_p:
            pick = rng.integers(0, n_gt, n_p)
            pb = gb[pick] + rng.normal(0, 9, (n_p, 4))
            pcls = np.where(rng.random(n_p) < 0.85, gcls[pick], 1 - gcls[pick])
        else:
            pb = np.column_stack([
                rng.uniform(0, 560, n_p), rng.uniform(0, 560, n_p),
                np.zeros(n_p), np.zeros(n_p)])
            pb[:, 2] = pb[:, 0] + rng.uniform(10, 70, n_p)
            pb[:, 3] = pb[:, 1] + rng.uniform(10, 70, n_p)
            pcls = rng.integers(0, 2, n_p)
        records.append({"boxes_xyxy": pb.astype(float),
                        "cls": np.asarray(pcls, dtype=int),
                        "conf": rng.random(n_p)})
    return records, gts


def single_class_synth(n_frames=14, seed=3):
    """Frames each holding exactly ONE class, so a resample can drop a class entirely.

    `synth()` mixes both classes into most frames, which is why no draw there ever
    loses one. Here class 1 (the buoy analogue) appears in 2 of 14 frames -- the real
    shape of this dataset, and the shape that makes a bootstrap draw able to delete it.
    """
    rng = np.random.default_rng(seed)
    records, gts = [], []
    for i in range(n_frames):
        c = 1 if i % 7 == 0 else 0          # class 1 in 2 of 14 frames
        n_gt = int(rng.integers(1, 4))
        gb = np.column_stack([rng.uniform(0, 500, n_gt), rng.uniform(0, 500, n_gt),
                              np.zeros(n_gt), np.zeros(n_gt)])
        gb[:, 2] = gb[:, 0] + rng.uniform(20, 60, n_gt)
        gb[:, 3] = gb[:, 1] + rng.uniform(20, 60, n_gt)
        gts.append({"boxes_xyxy": gb, "cls": np.full(n_gt, c, dtype=int)})
        n_p = int(rng.integers(1, 5))
        pick = rng.integers(0, n_gt, n_p)
        pb = gb[pick] + rng.normal(0, 8, (n_p, 4))
        records.append({"boxes_xyxy": pb, "cls": np.full(n_p, c, dtype=int),
                        "conf": rng.random(n_p)})
    return records, gts


def tied_conf_synth(n_frames=40, seed=5):
    """Detections drawn from four distinct confidences, so ties dominate the sort."""
    rng = np.random.default_rng(seed)
    records, gts = [], []
    for _ in range(n_frames):
        n_gt = int(rng.integers(1, 5))
        gb = np.column_stack([rng.uniform(0, 500, n_gt), rng.uniform(0, 500, n_gt),
                              np.zeros(n_gt), np.zeros(n_gt)])
        gb[:, 2] = gb[:, 0] + rng.uniform(20, 60, n_gt)
        gb[:, 3] = gb[:, 1] + rng.uniform(20, 60, n_gt)
        gcls = rng.integers(0, 2, n_gt)
        gts.append({"boxes_xyxy": gb, "cls": gcls.astype(int)})
        n_p = int(rng.integers(2, 8))
        pick = rng.integers(0, n_gt, n_p)
        pb = gb[pick] + rng.normal(0, 12, (n_p, 4))
        records.append({"boxes_xyxy": pb, "cls": gcls[pick].astype(int),
                        "conf": rng.choice([0.9, 0.8, 0.7, 0.6], size=n_p)})
    return records, gts


def main() -> int:
    records, gts = synth()
    parts = frame_parts(records, gts)

    ref = map50_95(records, gts)
    mine = ap_from_parts(parts)
    for k in ("map50_95", "map50"):
        d = abs(ref[k] - mine[k])
        assert d <= TOL, f"A pooled {k}: {mine[k]} vs {ref[k]} (d={d:.3e})"
    print(f"[smoke] A pooled: map50-95 {mine['map50_95']:.8f} == reference OK")

    # A2: the per-class breakdowns must agree too, key for key. Table 3 is read
    # per class (IR is nc=1 ship-only, so buoy AP comes from VIS alone and a macro
    # mean hides it), and two AP implementations that disagree per class while
    # agreeing on the mean would be undetectable in the headline number.
    assert set(ref["per_class"]) == set(mine["per_class"]), (
        f"A2 per-class keys differ: matching {sorted(ref['per_class'])} vs "
        f"apmetrics {sorted(mine['per_class'])}"
    )
    assert ref["per_class"], "A2 fixture produced no classes — the check is vacuous"
    for c in sorted(ref["per_class"]):
        for k in ("ap50_95", "ap50"):
            d = abs(ref["per_class"][c][k] - mine["per_class"][c][k])
            assert d <= TOL, f"A2 class {c} {k}: {mine['per_class'][c][k]} vs {ref['per_class'][c][k]} (d={d:.3e})"
        for k in ("n_gt", "n_pred"):
            assert ref["per_class"][c][k] == mine["per_class"][c][k], (
                f"A2 class {c} {k}: {mine['per_class'][c][k]} vs {ref['per_class'][c][k]}"
            )
    print("[smoke] A2 per-class: " + ", ".join(
        f"cls{c} ap50-95 {ref['per_class'][c]['ap50_95']:.6f} (n_gt {ref['per_class'][c]['n_gt']})"
        for c in sorted(ref["per_class"])) + " == reference OK")

    rng = np.random.default_rng(7)
    for t in range(5):
        sel = np.sort(rng.choice(len(parts), size=len(parts) // 2, replace=False))
        r = map50_95([records[i] for i in sel], [gts[i] for i in sel])
        m = ap_from_parts(parts, sel)
        d = abs(r["map50_95"] - m["map50_95"])
        assert d <= TOL, f"A subset {t}: {m['map50_95']} vs {r['map50_95']} (d={d:.3e})"
    print("[smoke] A subsets: 5 random halves match the reference exactly OK")

    pre = presort(parts)
    u = ap_weighted(pre)
    d = abs(u["map50_95"] - mine["map50_95"])
    assert d <= TOL, f"B unit weights: {u['map50_95']} vs {mine['map50_95']} (d={d:.3e})"
    assert set(u["per_class"]) == set(mine["per_class"]), "B class sets differ"
    for c in u["per_class"]:
        assert abs(u["per_class"][c]["ap50_95"] - mine["per_class"][c]["ap50_95"]) <= TOL
        assert u["per_class"][c]["n_gt"] == mine["per_class"][c]["n_gt"]
    print(f"[smoke] B unit weights == reference path, per class {list(u['per_class'])} OK")

    for t in range(4):
        w = rng.integers(0, 4, len(parts))
        if w.sum() == 0:
            continue
        fast = ap_weighted(pre, w)["map50_95"]
        # the literal resample: frame i physically present w[i] times
        dup = np.concatenate([np.full(int(k), i) for i, k in enumerate(w) if k > 0])
        slow = ap_from_parts(parts, dup)["map50_95"]
        d = abs(fast - slow)
        assert d <= TOL, f"C weights {t}: fast {fast} vs duplicated {slow} (d={d:.3e})"
    print("[smoke] C integer weights == physically duplicated frame list OK")

    empty = ap_weighted(presort(parts, np.zeros(0, dtype=int)))
    assert empty["map50_95"] == 0.0, "empty selection must be 0.0, not a crash"
    print("[smoke] D empty selection handled OK")

    # ---------------------------------------------------------------- R-A2
    # E: SPARSE-CLASS RESAMPLES -- the case B and C above never reach.
    #
    # B and C draw from `synth()`, where 60 frames both classes appear in mean that
    # every resample keeps both. So they passed while the two paths disagreed by 0.5
    # mAP on a resample that drops a class: `ap_from_parts` derives its class set FROM
    # the resample and scores over {ship}; `presort` froze the class set from the full
    # selection, so `ap_weighted` scored the vanished class 0 and halved the macro mean.
    # This section builds fixtures where a class CAN vanish and asserts the paths agree.
    per_class_frames = single_class_synth()
    sparse_parts = frame_parts(*per_class_frames)
    pre_sparse = presort(sparse_parts)
    rng2 = np.random.default_rng(11)
    seen_drop = 0
    for t in range(200):
        w = rng2.integers(0, 3, len(sparse_parts))
        if w.sum() == 0:
            continue
        fast = ap_weighted(pre_sparse, w)
        dup = np.concatenate([np.full(int(k), i) for i, k in enumerate(w) if k > 0])
        slow = ap_from_parts(sparse_parts, dup)
        d = abs(fast["map50_95"] - slow["map50_95"])
        assert d <= TOL, (
            f"E draw {t}: fast {fast['map50_95']} vs literal {slow['map50_95']} "
            f"(d={d:.3e}); dropped={fast['dropped_classes']}")
        # the class sets that ENTER THE MEAN must match, not merely the mean
        fast_scored = {c for c, v in fast["per_class"].items() if not v["excluded"]}
        assert fast_scored == set(slow["per_class"]), (
            f"E draw {t}: scored classes {sorted(fast_scored)} vs reference "
            f"{sorted(slow['per_class'])}")
        if fast["dropped_classes"]:
            seen_drop += 1
    assert seen_drop >= 10, (
        f"E is vacuous: only {seen_drop} of 200 draws dropped a class. The fixture "
        f"must actually exercise the missing-class path or this proves nothing.")
    print(f"[smoke] E sparse-class resamples: 200 draws, {seen_drop} dropped a class, "
          f"fast == literal every time OK")

    # F: TIE DETERMINISM. np.argsort defaults to an UNSTABLE introsort, so detections
    # sharing a confidence were ordered arbitrarily and AP moved with that order. The
    # fixture below is deliberately tie-dense (4 distinct confidences), where the effect
    # reaches 0.0067 mAP. On a real cache it is 2.7e-7, because 99.9% of real confidence
    # values are unique -- so this guards reproducibility, it does not repair a result.
    tied_parts = frame_parts(*tied_conf_synth())
    a = ap_from_parts(tied_parts)["map50_95"]
    for _ in range(5):
        assert ap_from_parts(tied_parts)["map50_95"] == a, "F: reference path not deterministic"
    b = ap_weighted(presort(tied_parts))["map50_95"]
    assert abs(a - b) <= TOL, f"F: tied-conf fast {b} vs reference {a} (d={abs(a - b):.3e})"
    print(f"[smoke] F heavy conf ties: both paths agree at {a:.8f}, repeatable OK")
    # ---- G. strict GT loading (R-B5 / F14) ---------------------------------
    # A MISSING label file is not an EMPTY one. Until 2026-09-10 both returned zero
    # boxes, so a wholly absent labels/ tree evaluated as a valid all-background
    # dataset and every detection in it scored as a false positive -- reproduced on a
    # fixture before the fix. Measured across the corpus FIRST: 0 missing label files
    # in 133,140 images, 3,880 legitimately EMPTY ones, and 0 malformed lines in
    # 1,008,459 non-blank lines, so refusing costs nothing today and exists to catch
    # a path or release mistake tomorrow.
    import tempfile as _tempfile

    from uqfusion.eval.matching import (EMPTY_LABEL_POLICY, MALFORMED_LINE_POLICY,
                                        MISSING_LABEL_POLICY, load_gt)

    assert (MISSING_LABEL_POLICY, EMPTY_LABEL_POLICY, MALFORMED_LINE_POLICY) == (
        "refuse", "allow", "refuse"), "G: declared policies moved"
    g_root = Path(_tempfile.mkdtemp()) / "images" / "pohang00"
    g_root.mkdir(parents=True)
    g_lab = g_root.parent.parent / "labels" / "pohang00"
    g_lab.mkdir(parents=True)
    (g_lab / "empty.txt").write_text("", encoding="utf-8")
    (g_lab / "good.txt").write_text(
        "0 0.5 0.5 0.1 0.1\n1 0.2 0.2 0.05 0.05\n\n", encoding="utf-8")
    (g_lab / "short.txt").write_text("0 0.5 0.5\n", encoding="utf-8")
    (g_lab / "garbage.txt").write_text("banana\n", encoding="utf-8")

    try:
        load_gt(g_root / "absent.png", (100, 100))
        raise AssertionError("G: a missing label file must refuse")
    except FileNotFoundError:
        pass
    assert len(load_gt(g_root / "absent.png", (100, 100),
                       allow_missing=True)["cls"]) == 0, (
        "G: allow_missing must still return an empty frame")
    assert len(load_gt(g_root / "empty.png", (100, 100))["cls"]) == 0, (
        "G: an EMPTY label file is legitimate -- 3,880 of them exist")
    assert len(load_gt(g_root / "good.png", (100, 100))["cls"]) == 2, (
        "G: a blank trailing line must not be an error")
    for _bad in ("short", "garbage"):
        try:
            load_gt(g_root / f"{_bad}.png", (100, 100))
            raise AssertionError(f"G: malformed line in {_bad}.txt must refuse")
        except ValueError:
            pass
    print("[smoke] G strict GT: missing refuses, empty allowed, malformed refuses OK")


    print("\nAPMETRICS SMOKE OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
