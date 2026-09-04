"""Check the two assumptions the 26m rebuild rests on, before anything is built on it.

**1. The image statistics are detector-independent.** `runs/cache_m/` reuses the
stems of `runs/cache/` so that `runs/derived/{brightness,structure}/*.json` can be
read unchanged. That is only legitimate if those files describe the PIXELS, which
are a function of the image list and the corruption (kind, severity, seed) alone.
This asserts the metas match field by field, then re-derives the statistics from
the 26m cache for one VIS and one IR condition and compares them numerically. An
assumption that is cheap to check should not be left as prose.

**2. The stage choice.** D31 selects a checkpoint by validation mAP@50-95, and the
same rule applied one level up picks VIS base (0.24961 over ft 0.24111) and IR ft
(0.14214 over base 0.11981). The paired fusion metric is deliberately not consulted
for that choice -- but it IS reported here, on all four clean caches, so a reader
can see what the rule cost or bought. If the rule and the report disagree, the rule
still wins and the disagreement is on the record.

Usage:
    python scripts/verify_cache_m.py --out runs/eval/cache_m_verify.md
"""

from __future__ import annotations

import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))

from uqfusion.eval.apmetrics import ap_from_parts, frame_parts  # noqa: E402
from uqfusion.eval.cache import load_cache                      # noqa: E402
from uqfusion.eval.matching import load_gt                      # noqa: E402

SHIP = 0
META_KEYS = ("images_list", "imgsz", "conf", "corrupt", "severity", "corrupt_seed", "source")


def meta_of(p: Path) -> dict:
    with open(p, "rb") as fh:
        d = pickle.load(fh)
    m = d.get("meta", d) if isinstance(d, dict) else {}
    return {k: m.get(k) for k in META_KEYS}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="runs/eval/cache_m_verify.md")
    args = ap.parse_args()
    L = ["# `runs/cache_m/` — is the 26m rebuild safe to build on?", ""]
    ok = True

    # ---- 1a. metas agree everywhere except the weights -----------------------
    L += ["## 1a. Every cache describes the same pixels as its 26s twin", "",
          "| stem | meta identical | 26s n | 26m n |", "|---|:--:|---:|---:|"]
    for p in sorted((ROOT / "runs/cache_m").glob("*.pkl")):
        old = ROOT / "runs/cache" / p.name
        if not old.is_file():
            L.append(f"| {p.stem} | *(no 26s twin)* | — | — |")
            continue
        a, b = meta_of(old), meta_of(p)
        same = a == b
        ok &= same
        na, nb = len(load_cache(old)[0]), len(load_cache(p)[0])
        ok &= (na == nb)
        L.append(f"| {p.stem} | {'yes' if same else '**NO** ' + str({k: (a[k], b[k]) for k in a if a[k] != b[k]})} | {na} | {nb} |")

    # ---- 1b. the derived statistics really are the same ----------------------
    # Re-derive from the 26m cache and compare to the file the gate will read.
    L += ["", "## 1b. Re-derived statistics vs the files the gate reads", "",
          "Recomputed from the **26m** cache and compared against "
          "`runs/derived/*` written from the **26s** cache. Any nonzero max "
          "absolute difference means the stems may not be shared.", "",
          "| cache | modality | stat | max abs diff |", "|---|---|---|---:|"]
    import cv2                                                      # noqa: E402
    from uqfusion.eval.corruptions import make_corruption           # noqa: E402
    sys.path.insert(0, str(ROOT / "scripts"))
    import frame_brightness as fb                                   # noqa: E402
    import frame_structure as fs                                    # noqa: E402
    for stem, mod in (("gauss_vis_paired_fog", "vis"), ("gauss_ir_paired_glare_s2", "ir")):
        p = ROOT / "runs/cache_m" / f"{stem}.pkl"
        if not p.is_file():
            L.append(f"| {stem} | {mod} | — | *(cache missing)* |")
            continue
        recs, meta = load_cache(p)
        corr = (make_corruption(meta["corrupt"], meta["severity"], meta["corrupt_seed"])
                if meta.get("corrupt") else None)
        lo, hi = fs.content_rows(mod)
        idx = np.linspace(0, len(recs) - 1, 40).astype(int)          # 40 frames is plenty
        got_s, got_b = [], []
        for i in idx:
            im = cv2.imread(recs[int(i)]["image_path"])              # READ-ONLY
            if corr is not None:
                im = corr(im, int(i))                                # same (seed, index)
            g = cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[lo:hi]
            got_s.append(fs.frame_stats(g))
            got_b.append(fb.frame_stats(g))
        for ref_dir, got, keys in (("structure", got_s, ("grad_gini", "lap_over_var")),
                                   ("brightness", got_b, ("p05",))):
            ref = ROOT / "runs/derived" / ref_dir / f"{stem}.json"
            if not ref.is_file():
                L.append(f"| {stem} | {mod} | *({ref_dir} missing)* | — |")
                continue
            want = json.loads(ref.read_text(encoding="utf-8"))["frames"]
            for key in keys:
                if key not in got[0] or key not in want[0]:
                    continue
                d = float(np.max(np.abs(np.asarray([g[key] for g in got])
                                        - np.asarray([want[int(i)][key] for i in idx]))))
                ok &= (d < 1e-9)
                L.append(f"| {stem} | {mod} | `{key}` | {d:.3e} |")

    # ---- 2. the stage report -------------------------------------------------
    L += ["", "## 2. Stage report — what D31's rule bought, on the paired val", "",
          "Selection was made on training-validation mAP, not on this table. "
          "Ship AP on the paired val, single stream, day and night.", "",
          "| stream | stage | val mAP50-95 | paired day | paired night |",
          "|---|---|---:|---:|---:|"]
    valmap = {("vis", ""): 0.24961, ("vis", "_ft"): 0.24111,
              ("ir", ""): 0.11981, ("ir", "_ft"): 0.14214}
    probe = ROOT / "runs/cache_m_stageprobe"
    gts = runs = None
    for mod in ("vis", "ir"):
        for stage in ("", "_ft"):
            cands = [probe / f"gauss_{mod}_paired_clean{stage}.pkl",
                     ROOT / "runs/cache_m" / f"gauss_{mod}_paired_clean{stage}.pkl"]
            p = next((c for c in cands if c.is_file()), None)
            if p is None:
                L.append(f"| {mod} | {stage or 'base'} | {valmap[(mod, stage)]:.5f} | — | — |")
                continue
            recs, _ = load_cache(p)
            if gts is None:
                vp, _ = load_cache(ROOT / "runs/cache_m/gauss_vis_paired_clean.pkl")
                gts = [load_gt(r["image_path"], r["image_hw"]) for r in vp]
                runs = np.asarray([Path(r["image_path"]).parent.name for r in vp])
            night = runs == "pohang01"
            # IR boxes are in the IR plane; scored here only against IR's own frame
            # for a like-for-like stage comparison, never against the VIS GT.
            if mod == "ir":
                g = [load_gt(r["image_path"], r["image_hw"]) for r in recs]
            else:
                g = gts
            parts = frame_parts(recs, g)

            def apf(sel):
                e = ap_from_parts([parts[i] for i in np.flatnonzero(sel)])["per_class"].get(SHIP)
                return float(e["ap50_95"]) if e else 0.0
            L.append(f"| {mod} | {stage or 'base'} | {valmap[(mod, stage)]:.5f} | "
                     f"{apf(~night):.4f} | {apf(night):.4f} |")

    L += ["", f"## Verdict", "",
          f"- Assumption checks: **{'all pass' if ok else 'FAILED — see above'}**.",
          "- Chosen: VIS `gauss_vis_seed0` (base), IR `gauss_ir_seed0_ft`."]
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"\n[verify] wrote {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
