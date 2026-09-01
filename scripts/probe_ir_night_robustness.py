"""Does the cross-modal night test survive a DEGRADED IR sensor?

`docs/crossmodal-gate-2026-09-01.md` §4.1 names this as the result's largest
exposure. The `crossmodal` preset vetoes VIS when `ir_p05 > 34.0` -- "the other
sensor says it is night" -- and every one of the eight benchmark cells leaves IR
untouched, so on that benchmark the axis consults a sensor that is never wrong.
A deployment does not get that guarantee.

Two failure directions, and they are not equally bad:

  FALSE NIGHT   degraded IR pushes p05 above the threshold on a DAY frame, so VIS
                is vetoed on a clear day. This is the dangerous one: it throws
                away a stream scoring 0.3683 in favour of one scoring 0.0177.
  MISSED NIGHT  degraded IR drops p05 below the threshold on a NIGHT frame, so VIS
                is kept. Costs at most what `no_veto` costs on a night cell,
                measured at -0.0022 to -0.0054.

This probe answers the image-statistics half of the question, which needs no GPU:
apply each corruption to the paired IR frames, recompute `p05`, and report what
the DEPLOYED threshold -- the one fitted on CLEAN IR, held fixed -- then does. It
does not answer what a degraded IR DETECTOR contributes to fusion; that needs new
paired IR caches and a GPU pass, and is the other half of the open item.

Read-only on the dataset; writes one report to a caller-supplied filename.

Usage:
    python scripts/probe_ir_night_robustness.py --out runs/eval/ir_night_robustness.md
    python scripts/probe_ir_night_robustness.py --stride 4          # quick pass
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

from uqfusion.eval.cache import load_cache            # noqa: E402
from uqfusion.eval.corruptions import CORRUPTIONS, make_corruption  # noqa: E402

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
    ap.add_argument("--constants", default="runs/eval/structure_constants.json")
    ap.add_argument("--out", default="runs/eval/ir_night_robustness.md")
    ap.add_argument("--stride", type=int, default=1)
    ap.add_argument("--severities", type=int, nargs="+", default=[1, 2, 3])
    ap.add_argument("--kinds", nargs="+", default=list(CORRUPTIONS),
                    help="restrict the corruption kinds, so the arms can run in parallel")
    ap.add_argument("--seed", type=int, default=7,
                    help="NOT the seed the paired VIS caches used (1): plan B5-5 "
                         "keeps gate tuning and robustness testing on different draws")
    args = ap.parse_args()

    thr = float(json.loads((ROOT / args.constants).read_text(encoding="utf-8"))
                ["axes"]["ir_p05"]["threshold"])
    recs, _ = load_cache(ROOT / args.cache)
    recs = recs[::args.stride]
    runs = np.asarray([Path(r["image_path"]).parent.name for r in recs])
    night = runs == NIGHT_RUN
    fit = np.isin(runs, FIT_RUNS)
    lo, hi = content_rows()
    print(f"[irrob] {len(recs)} IR frames ({int((~night).sum())} day / "
          f"{int(night.sum())} night), deployed threshold p05 > {thr:.1f}")

    grays = []
    for r in recs:
        im = cv2.imread(r["image_path"])                 # READ-ONLY
        if im is None:
            raise FileNotFoundError(r["image_path"])
        grays.append(im)

    def p05_of(images) -> np.ndarray:
        return np.asarray([float(np.percentile(
            cv2.cvtColor(im, cv2.COLOR_BGR2GRAY)[lo:hi].astype(np.float32), 5))
            for im in images])

    t0 = time.time()
    rows = [{"corruption": "clean", "severity": 0, "p05": p05_of(grays)}]
    print(f"[irrob] clean done ({time.time() - t0:.0f}s)", flush=True)
    for kind in args.kinds:
        for s in args.severities:
            tf = make_corruption(kind, s, args.seed)
            v = p05_of([tf(im, i * args.stride) for i, im in enumerate(grays)])
            rows.append({"corruption": kind, "severity": s, "p05": v})
            print(f"[irrob] {kind} s{s} done ({time.time() - t0:.0f}s)", flush=True)

    L = ["# Cross-modal night test under a DEGRADED IR sensor", "",
         f"The deployed rule is `ir_p05 > {thr:.1f}`, fitted once on CLEAN IR of "
         f"the fit runs and then held fixed — so the columns below are what the "
         f"SHIPPED threshold does when the sensor it consults is damaged. "
         f"Corruption seed {args.seed} (the paired VIS caches used seed 1; plan "
         f"B5-5 keeps tuning and testing on different draws). "
         f"n = {len(recs)} frames, stride {args.stride}.", "",
         "`false night` is the dangerous direction: a DAY frame misread as night "
         "vetoes VIS on a clear day, trading 0.3683 for 0.0177. `missed night` is "
         "the benign one, bounded by what `no_veto` costs a night cell "
         "(−0.0022 to −0.0054).", "",
         "| corruption | sev | day p05 med | night p05 med | **false night** | **missed night** | still separable? |",
         "|---|---:|---:|---:|---:|---:|---|"]
    out_rows = []
    for r in rows:
        v = r["p05"]
        d, ng = v[~night], v[night]
        false_night = float((d > thr).mean())
        missed = float((ng <= thr).mean())
        # Could a threshold REFIT on this corruption's own fit-run day frames
        # still separate day from night? That is the best case for a system that
        # knows how its IR is degraded — an upper bound on what a repair could buy.
        refit = float(v[fit].max())
        separable = bool(ng.min() > refit)
        out_rows.append({"corruption": r["corruption"], "severity": r["severity"],
                         "day_med": float(np.median(d)), "night_med": float(np.median(ng)),
                         "false_night": false_night, "missed_night": missed,
                         "refit_thr": refit, "separable_after_refit": separable,
                         "night_min": float(ng.min()), "day_max": float(d.max())})
        flag = "" if false_night == 0.0 else "  ⚠"
        L.append(f"| {r['corruption']} | {r['severity'] or '—'} | {np.median(d):.1f} | "
                 f"{np.median(ng):.1f} | {false_night:.1%}{flag} | {missed:.1%} | "
                 f"{'yes' if separable else '**NO**'} |")

    worst = max(out_rows, key=lambda r: r["false_night"])
    n_bad = sum(1 for r in out_rows if r["false_night"] > 0)
    L += ["", "## Verdict", "",
          f"- {n_bad} of {len(out_rows)} arms produce ANY false night.",
          f"- Worst false-night rate: **{worst['false_night']:.1%}** "
          f"({worst['corruption']} s{worst['severity']}).",
          f"- Arms where a refit threshold could no longer separate day from "
          f"night at all: {sum(1 for r in out_rows if not r['separable_after_refit'])}.",
          "",
          "A non-zero `false night` column means the deployed gate would veto VIS "
          "on clear-day frames whenever IR degrades that way, and the cost of each "
          "such frame is the clean/day-to-IR gap. That is the number that decides "
          "whether the cross-modal axis needs an IR-side self-check before it is "
          "allowed to hold the switch."]

    out = ROOT / (args.out if len(args.kinds) == len(CORRUPTIONS)
                  else args.out.replace(".md", "_" + "-".join(args.kinds) + ".md"))
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    out.with_suffix(".json").write_text(json.dumps(
        {"threshold": thr, "seed": args.seed, "n": len(recs), "stride": args.stride,
         "rows": out_rows}, indent=2), encoding="utf-8")
    print(f"[irrob] wrote {out}")
    print(f"[irrob] worst false-night {worst['false_night']:.1%} "
          f"({worst['corruption']} s{worst['severity']}); {n_bad}/{len(out_rows)} arms affected")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
