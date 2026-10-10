"""R-D1 score path against NO sigma (S0), on ship AP and the macro. Descriptive, post hoc.

The pre-registered comparison is real vs shuffled sigma (S5 - S7). Its POSITIVE outcome
says the shipped preset "should be re-examined for adopting it", so the question that
follows is whether real-sigma score re-ranking beats leaving sigma out (S5 - S0), and
how far shuffled sigma falls below it (S7 - S0). Neither comparison is in either
pre-registration; this script cannot change a verdict.

Arms are rebuilt exactly as `ablate_uq_mechanism.py` builds them (same `ARMS`,
`substitute`, a fresh `default_rng(SEED)` per arm, VIS then IR), so S0/S5/S7 here are
bit-identical to the arm tables of `runs/eval/rd1_*_2026-10-08.md`.

Usage:
    python scripts/diag_rd1_ship_vs_s0.py --preset crossmodal26m --out runs/eval/x.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ablate_uq_mechanism import ARMS, BLOCK_LEN, N_BOOT, SEED, substitute
from uqfusion.eval.apmetrics import ap_from_parts, frame_parts
from uqfusion.eval.blockboot import block_bootstrap_delta
from uqfusion.eval.ctx import load_context, run_systems

COMPARISONS = (("S5", "S0"), ("S7", "S0"), ("S6", "S0"))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--preset", required=True)
    ap.add_argument("--cache-dir", default="runs/cache_m")
    ap.add_argument("--out", required=True)
    ap.add_argument("--bright-dir", default="runs/derived/brightness",
                    help="runs/derived_m_v2/brightness for the corruption-v2 re-run")
    ap.add_argument("--structure-dir", default="runs/derived/structure")
    args = ap.parse_args()

    t0 = time.time()
    ctx = load_context(preset=args.preset, cache_dir=args.cache_dir, bright_dir=args.bright_dir,
                       structure_dir=args.structure_dir, verbose=False)
    res: dict = {"preset": args.preset, "cache_dir": args.cache_dir, "ap": {}, "delta": {}}
    for cond in ctx.vis_by_cond:
        vis, ir = ctx.vis_by_cond[cond], ctx.ir_clean
        paths = [r["image_path"] for r in vis]
        parts = {}
        for arm in ("S0", "S5", "S6", "S7"):
            _, mode, opts = ARMS[arm]
            rng = np.random.default_rng(SEED)
            v2, _ = substitute(vis, mode, rng)
            i2, _ = substitute(ir, mode, rng)
            out = run_systems(ctx, cond, vis_records=v2, ir_records=i2, **opts)
            parts[arm] = frame_parts(out["fused_gated"], out["gts"])
            r = ap_from_parts(parts[arm])
            res["ap"][f"{cond}/{arm}"] = {"macro": r["map50_95"],
                                          "ship": r["per_class"][0]["ap50_95"],
                                          "buoy": r["per_class"].get(1, {}).get("ap50_95")}
            print(f"[s0] {cond:9s} {arm} {res['ap'][f'{cond}/{arm}']} ({time.time() - t0:.0f}s)",
                  flush=True)
        for a, b in COMPARISONS:
            for name, cls in (("macro", None), ("ship", 0)):
                r = block_bootstrap_delta(parts[a], parts[b], paths, BLOCK_LEN,
                                          n_boot=N_BOOT, seed=SEED, cls=cls)
                res["delta"][f"{cond}/{a}-{b}/{name}"] = {
                    k: r[k] for k in ("delta", "ci_lo", "ci_hi", "se", "spans_zero", "n_undefined")}
                print(f"[s0] {cond:9s} {a}-{b} {name:5s} {r['delta']:+.6f} "
                      f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}] ({time.time() - t0:.0f}s)", flush=True)
    p = Path(args.out)
    if p.exists():
        raise FileExistsError(f"{p} exists; pass a new --out")
    p.write_text(json.dumps(res, indent=1), encoding="utf-8")
    print(f"[s0] wrote {p} in {time.time() - t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
