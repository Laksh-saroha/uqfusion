"""Where does R-D1's fog score-path gain under `crossmodal26m` come from? Descriptive only.

The R-D1 re-run (`docs/prereg-uq-mechanism-ablation-26m.md`) found S5 − S7 = +0.0062 on fog,
CI clear of zero: one condition of four, so the verdict is unchanged (NULL needs 3 of 4 to
flip). This decomposes that one number. It cannot change the verdict and is not a test.

The score path multiplies each box's score by (frame-median sigma / sigma)^alpha WITHIN ITS
OWN STREAM. So it can act two ways:

  within-stream   re-ranking one detector's boxes against each other
  cross-stream    changing how VIS and IR boxes interleave in the union's ranking

Arms, all fog, same seeds and substitution as `ablate_uq_mechanism.py` (S5 real sigma,
S7 sigma shuffled within frame, alpha 1.0):

  full      the system as R-D1 scored it
  vis_only  identical, but every IR record emptied -- whatever S5 − S7 remains is
            re-ranking inside the VIS stream

Sliced by day / night (NIGHT_RUNS). Under `crossmodal26m` VIS is vetoed on every night frame,
so night is IR re-ranking by construction; day keeps both streams.

Usage:
    python scripts/diag_rd1_fog_score_path.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

from ablate_uq_mechanism import ALPHA, BLOCK_LEN, SEED, substitute       # noqa: E402
from uqfusion.eval.apmetrics import ap_from_parts, frame_parts           # noqa: E402
from uqfusion.eval.blockboot import block_bootstrap_delta                # noqa: E402
from uqfusion.eval.ctx import NIGHT_RUNS, load_context, run_systems      # noqa: E402

COND = "fog"


PER_BOX = ("boxes_xyxy", "conf", "cls", "sigma_ltrb")   # `feat` is per frame and is kept


def empty(records: list[dict]) -> list[dict]:
    """Same records, zero detections. Every per-box array keeps its trailing shape."""
    return [{**r, **{k: np.asarray(r[k])[:0] for k in PER_BOX}} for r in records]


def main() -> int:
    t0 = time.time()
    ctx = load_context(preset="crossmodal26m", cache_dir="runs/cache_m", conditions=(COND,), verbose=False)
    vis, ir = ctx.vis_by_cond[COND], ctx.ir_clean
    paths = [r["image_path"] for r in vis]
    night = np.isin(ctx.runs, NIGHT_RUNS)
    sl = {"all": np.arange(len(vis)), "day": np.flatnonzero(~night), "night": np.flatnonzero(night)}

    parts, veto = {}, None
    for variant in ("full", "vis_only"):
        for arm, mode in (("S5", "real"), ("S7", "shuf_frame")):
            rng = np.random.default_rng(SEED)            # as ablate_uq_mechanism: fresh per arm
            v2, _ = substitute(vis, mode, rng)
            i2, _ = substitute(ir, mode, rng)
            if variant == "vis_only":
                i2 = empty(i2)
            out = run_systems(ctx, COND, vis_records=v2, ir_records=i2, sigma_score_alpha=ALPHA)
            parts[(variant, arm)] = frame_parts(out["fused_gated"], out["gts"])
            if veto is None:
                veto = np.asarray(out["veto_vis"], dtype=bool)
            print(f"[diag] {variant:8s} {arm} done ({time.time() - t0:.0f}s)", flush=True)

    print(f"\n[diag] VIS veto rate: day {veto[sl['day']].mean():.3f}  night {veto[sl['night']].mean():.3f}")
    print(f"[diag] metric: mAP50-95 over classes, as R-D1 scores it; block bootstrap L={BLOCK_LEN}, n_boot 1000\n")
    print(f"{'variant':9s} {'slice':6s} {'frames':>6s} {'S5':>9s} {'S7':>9s} {'S5-S7':>9s}  95% CI")
    rows = {}
    for variant in ("full", "vis_only"):
        for s, idx in sl.items():
            a = ap_from_parts(parts[(variant, "S5")], sel=idx)["map50_95"]
            b = ap_from_parts(parts[(variant, "S7")], sel=idx)["map50_95"]
            r = block_bootstrap_delta(parts[(variant, "S5")], parts[(variant, "S7")], paths, BLOCK_LEN,
                                      sel=idx, n_boot=1000, seed=SEED)
            rows[(variant, s)] = r["delta"]
            print(f"{variant:9s} {s:6s} {len(idx):6d} {a:9.6f} {b:9.6f} {r['delta']:+9.6f}  "
                  f"[{r['ci_lo']:+.6f}, {r['ci_hi']:+.6f}]", flush=True)
    d_full, d_vis = rows[("full", "day")], rows[("vis_only", "day")]
    print(f"\n[diag] day: full {d_full:+.6f}, VIS re-ranking alone {d_vis:+.6f}, "
          f"cross-stream remainder {d_full - d_vis:+.6f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
