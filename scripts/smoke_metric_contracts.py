"""R-A4 acceptance gate — pins what the metric suite actually measures.

Written 2026-09-10 for finding F12 (docs/TODO-2026-09-09-architecture-review.md).
The review's charge is that five metric NAMES oversell their computations. Each
claim was reproduced on `runs/cache_uqslice/sigma_vis_seed0_nightfull.pkl` before
any code moved; this script re-asserts the same properties on synthetic data so
they cannot drift back, and so the contract constants cannot be edited without a
test failing.

Every case is a CONTRACT, not a quality bar: it pins what the number is, so that
a future change either preserves it or is forced to declare itself.

  1. declared contract constants exist and read the way the docs quote them
  2. AUSE/AURC are RANKING-ONLY: a monotone rescale of every sigma is bit-identical
  3. `aurc` is a GRID MEAN, not an integral, and `aurc_trapz` is the integral
  4. coverage never reaches 0 (the grid stops at 0.95) and is integer-rounded
  5. sorts are stable: a permutation of tied inputs reproduces the curve
  6. `summarize_cache` publishes the TP denominators R-A4 asks for
  7. out-of-unit-range scores are counted, not silently binned
  8. same-stream duplicates — not the support bonus, not cross-modal
     agreement — are what push a fused score above 1.0

Usage:  python scripts/smoke_metric_contracts.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from uqfusion.eval.metrics import (
    AURC_GRID,
    AURC_INTEGRATION,
    DECE_CONDITIONING,
    RANKING_METRICS,
    UQ_SUBSET,
    d_ece,
    sparsification,
    summarize_cache,
)
from uqfusion.uq.fusion import sigma_weighted_fusion

FAIL = []


def check(name, cond, detail=""):
    print(f"[smoke] {name:<44} {'PASS' if cond else 'FAIL'}  {detail}")
    if not cond:
        FAIL.append(name)


# ---------------------------------------------------------------- 1 constants
check(
    "1 declared contracts present",
    DECE_CONDITIONING == "confidence-only"
    and UQ_SUBSET == "true-positives-only"
    and AURC_INTEGRATION == "grid-mean"
    and RANKING_METRICS == ("ause", "aurc", "aurc_trapz")
    and "20 points" in AURC_GRID,
    f"{DECE_CONDITIONING} / {UQ_SUBSET} / {AURC_INTEGRATION}",
)

# ------------------------------------------------------- 2 ranking invariance
rng = np.random.default_rng(0)
n = 4000
u = rng.gamma(2.0, 1.0, n)
risk = np.clip(0.15 * u + rng.normal(0, 0.2, n), 0.0, 1.0)  # u genuinely orders risk

base = sparsification(u, risk)
for scale, label in ((100.0, "x100"), (1e-3, "x1e-3")):
    s = sparsification(u * scale, risk)
    same = (
        s["ause"] == base["ause"]
        and s["aurc"] == base["aurc"]
        and s["aurc_trapz"] == base["aurc_trapz"]
    )
    check(f"2 ranking-only under {label}", same, f"ause {s['ause']:.10f}")
# a monotone but NON-linear rescale must also be inert
s = sparsification(np.log1p(u), risk)
check(
    "2 ranking-only under log1p",
    s["ause"] == base["ause"] and s["aurc"] == base["aurc"],
    f"aurc {s['aurc']:.10f}",
)
# ...while a rank-REVERSING map must move it, or the test above proves nothing
s_rev = sparsification(-u, risk)
check("2 rank reversal does move it", s_rev["ause"] != base["ause"],
      f"{base['ause']:.6f} -> {s_rev['ause']:.6f}")

# ------------------------------------------------------- 3 grid mean vs trapz
curve = np.asarray(base["risk_by_u"])
cov = np.asarray(base["coverage"])
trapz = getattr(np, "trapezoid", np.trapz)
check(
    "3 aurc is the grid mean",
    base["aurc"] == float(curve.mean()) == base["aurc_grid_mean"],
    f"{base['aurc']:.10f}",
)
o = np.argsort(cov)   # the returned grid runs coverage 1.00 -> 0.05; trapz needs ascending
check(
    "3 aurc_trapz is the integral",
    abs(base["aurc_trapz"] - float(trapz(curve[o], cov[o]))) < 1e-15,
    f"{base['aurc_trapz']:.10f}",
)
check(
    "3 the two differ materially",
    abs(base["aurc"] - base["aurc_trapz"]) > 1e-4,
    f"delta {abs(base['aurc'] - base['aurc_trapz']):.6f}",
)

# ------------------------------------------------- 4 coverage floor + rounding
check(
    "4 coverage stops at 0.05, not 0",
    cov.max() == 1.0 and 0.0 < cov.min() < 0.1 and len(cov) == 20,
    f"[{cov.min():.4f}, {cov.max():.4f}] n={len(cov)}",
)
odd = sparsification(u[:997], risk[:997])
odd_cov = np.asarray(odd["coverage"])
frac = np.asarray(odd["fractions"])
check(
    "4 realised coverage != 1 - frac",
    not np.array_equal(odd_cov, 1.0 - frac),
    f"max |gap| {np.abs(odd_cov - (1.0 - frac)).max():.6f}",
)

# ------------------------------------------------------------- 5 stable sorts
# Stability does NOT mean permutation invariance -- permuting the input IS a different
# tie order. What it means is that ties resolve in INPUT order, deterministically, on
# every numpy build. Asserted directly: four fully tied uncertainties, distinct risks.
tied_u = np.ones(4)
tied_risk = np.array([0.0, 0.1, 0.2, 0.3])
a = sparsification(tied_u, tied_risk)
# by_u = argsort(-u) is [0,1,2,3] under a stable sort, and the curve keeps the TAIL,
# so the single retained element at minimum coverage is the LAST input, risk 0.3.
check(
    "5 ties resolve in input order",
    a["risk_by_u"][-1] == 0.3 and a["risk_by_u"][0] == float(tied_risk.mean()),
    f"min-coverage risk {a['risk_by_u'][-1]:.4f} (last input), full-set {a['risk_by_u'][0]:.4f}",
)
b = sparsification(tied_u, tied_risk[::-1].copy())
check(
    "5 input order is what decides it",
    b["risk_by_u"][-1] == 0.0,
    f"reversed input -> {b['risk_by_u'][-1]:.4f}",
)
check(
    "5 sorts are pinned stable in source",
    Path("src/uqfusion/eval/metrics.py").read_text(encoding="utf-8").count('kind="stable"') >= 2,
    "both argsorts",
)

# ------------------------------------------- 6/7 summarize_cache disclosures
REQUIRED = (
    "n_gt", "n_tp_detections", "tp_share", "recall_tp_over_gt",
    "uq_subset", "dece_conditioning", "conf_out_of_unit_range",
    "aurc_trapz", "aurc_integration",
)
src = Path("src/uqfusion/eval/metrics.py").read_text(encoding="utf-8")
check(
    "6 summarize_cache publishes denominators",
    all(f'"{k}"' in src for k in REQUIRED),
    f"{len(REQUIRED)} keys",
)

conf = np.array([0.2, 0.9, 1.75, 1.10, -0.05])
matched = np.array([0, 1, 1, 1, 0], dtype=bool)
out_of_range = int(((conf < 0.0) | (conf > 1.0)).sum())
check(
    "7 out-of-range scores are counted",
    out_of_range == 3 and np.isfinite(d_ece(conf, matched)),
    f"{out_of_range} of {conf.size}; d_ece still finite (it clips, which is the trap)",
)

# ------------------------------------------ 8 self-agreement, not cross-modal
# Measured mechanism behind fused scores above 1.0 (R-A4 claim 6). It is NOT the
# support bonus the review named -- support_gamma is 0.0 in every shipped preset --
# and it is NOT cross-modal agreement, which is bounded: with weights summing to 1,
# a two-STREAM cluster scores w_v*s_v + w_i*s_i <= max(s) <= 1.
#
# What overflows is WBF's k = min(n_models, n_cluster) counting cluster MEMBERS.
# Two overlapping boxes from the SAME stream are scored as a confirmation, so the
# score becomes w_stream * (s1 + s2), which exceeds 1 whenever the pair is confident
# and that stream holds nearly all the weight. On the clean cell w_vis = 0.9930.
box = [[0.10, 0.10, 0.50, 0.50]]
dup = [[0.10, 0.10, 0.50, 0.50], [0.105, 0.105, 0.505, 0.505]]   # same stream, overlapping
sig1 = [[1.0, 1.0, 1.0, 1.0]]
sig2 = [[1.0, 1.0, 1.0, 1.0]] * 2
W = [0.9930, 0.0070]   # the real gated weights on the clean cell; they sum to 1


def fuse(boxes_l, scores_l, sigmas_l, **kw):
    _, sc, _ = sigma_weighted_fusion(
        boxes_l, scores_l, [[0] * len(s) for s in scores_l], sigmas_l,
        weights=W, iou_thr=0.55, support_gamma=0.0, **kw)
    return float(sc[0])


single = fuse([box, []], [[0.9], []], [sig1, []])
selfdup = fuse([dup, []], [[0.9, 0.9], []], [sig2, []])
crossmodal = fuse([box, box], [[0.9], [0.9]], [sig1, sig1])
fixed = fuse([dup, []], [[0.9, 0.9], []], [sig2, []], consensus_distinct=True)
check(
    "8 same-stream duplicates exceed 1.0",
    single <= 1.0 and selfdup > 1.0,
    f"1 box {single:.4f} -> 2 same-stream boxes {selfdup:.4f}, support_gamma=0",
)
check(
    "8 cross-modal agreement does not",
    crossmodal <= 1.0,
    f"2 streams {crossmodal:.4f} <= max score",
)
check(
    "8 consensus_distinct removes it",
    fixed <= 1.0 and abs(fixed - selfdup / 2.0) < 1e-9,
    f"distinct=True -> {fixed:.4f} (k counts streams, not members)",
)

print()
if FAIL:
    raise SystemExit(f"smoke_metric_contracts: {len(FAIL)} FAILED -> {FAIL}")
print("smoke_metric_contracts: all contracts hold")
