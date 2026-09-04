# Veil veto adopted; the VIS-veto rule family measured to its ceiling (2026-09-01)

Code: commit `8ca28cb`. Raw outputs: `runs/eval/final_system_veil.md`,
`runs/eval/veto_rule_sweep.md`, `runs/eval/probe_structure.md`,
`runs/eval/probe_detector_evidence.md` — all under `runs/`, which is **gitignored**,
which is why the tables are reproduced here.

**This document does not modify `docs/gated-fusion-handoff.md`.** §7.1 there is
deliberately left provisional per instruction, and §7.2 is left standing even
though §4 below shows its stated reasoning is wrong. The handoff is the other
chat's starting point; this file is the record.

---

## 1. What was broken

The photometric veto reads `p05` — a histogram statistic. Fog is invisible to
every histogram statistic: it raises the black floor without darkening the
frame, so a fogged frame's percentiles look like a healthy daylight frame's.

The gate therefore never fired on fog/day, and fusion turned a working IR stream
into a worse one:

| fog/day, ship AP | |
|---|---:|
| VIS alone | 0.0020 |
| IR alone | 0.0177 |
| gated (photometric only) | **0.0126** |

A −0.0051 regression against simply using IR, caused by the gate failing to fire
rather than by fusion being wrong in principle.

## 2. The veil term

`lap_var` — variance of the Laplacian — is the odd statistic out: it measures
local structure rather than the intensity distribution, so a contrast-destroying
veil moves it when nothing else responds. Per-cell medians (`probe_structure.md`):

| stat | clean/day | lowlight/day | glare/day | clean/night | lowlight/night |
|---|---:|---:|---:|---:|---:|
| lap_var | 3170 | 152 | 2725 | 1334 | 23.3 |
| p05 | 35 | 0 | 36 | 2.5 | 0 |

Fitted as a **hard novelty bound**, not a quantile:

```
tau_lap      = 508.674
stat         = lap_var
rule         = minimum of lap_var over CLEAN frames of the fit runs
               (pohang00, pohang02, pohang03); no corrupted frame informs it
fit n        = 2399 frames
filter       = ("majority", 15)
```

A `p01` quantile gives 611.9 and fires on **7.0%** of the clean/day guard cell —
not from degradation, but because pohang02's clean footage is genuinely less
textured than pohang00's. Per-run clean minima: pohang00 2802.8, pohang02 529.9,
pohang03 1406.3. The quantile is measuring inter-run scene variation, not damage.
The minimum is the only defensible bound.

### The two axes must be filtered separately

Adopted switch is `dilate15(dark) OR majority15(veil)`, **not**
`dilate15(dark OR veil)`. This is not cosmetic. Dilating the combined switch
turns glare/day's 4 flagged frames into 58 — 4.8% of a guard cell where VIS
scores 0.2892 against IR's 0.0177, worth roughly −0.013 AP, more than double the
entire fog gain.

The reason is that the filters do opposite jobs. Dilation only ever *adds*
vetoes, which is right for an under-firing brightness term and wrong for an
over-firing veil term, which needs denoising instead. Pinned by smoke check G in
`scripts/smoke_veil_gate.py`.

## 3. Measured result

Ship AP. Diff against `runs/eval/final_system_2026-09-01_photometric-only.md`
touches **fog rows only** — every other cell is bit-identical.

| cell | before | after | Δ vs IR before | Δ vs IR after | veto before | veto after |
|---|---:|---:|---:|---:|---:|---:|
| fog/day | 0.0126 | **0.0166** | −0.0051 [−0.0062, −0.0030] | **−0.0011** [−0.0018, −0.0004] | 0% | 100% |
| fog/night | 0.0809 | **0.0813** | −0.0001 | **+0.0003** (spans 0) | 71% | 100% |

Unchanged, guard cells included: clean/day 0.3715, glare/day 0.2928,
clean/night 0.0813, glare/night 0.0813, lowlight/day 0.0166, lowlight/night 0.0813.

Against `visible_only`, fog/day went +0.0106 → **+0.0146** [+0.0118, +0.0178].

**The ablation now carries the evidence it previously could not.** `no_veto` on
fog/day went from "identical" to **−0.0040** [−0.0051, −0.0019]: before the veil
term, ablating the veto changed nothing there because the veto was not firing at
all. `no_maha` and `cap_only` on both fog cells went to "identical" — with VIS
fully vetoed there are no soft weights left to ablate.

### The residual −0.0011 is not the gate

A fully vetoed *day* cell lands at 0.0166 against IR's raw 0.0177. That gap is
**single-list WBF post-processing**, measured: with one stream vetoed, WBF over a
single input list still clips boxes to the canvas, still merges boxes overlapping
above `iou_thr`, and still replaces a merged cluster's scores with their mean.
Day 0.0177 → 0.0166; at night the identical post-process is worth **+0.0003**,
because merging genuinely helps a stream that emits duplicates there. Same
artifact, both signs. `fuse_detections(single_passthrough=True)` bypasses it;
default `False` because the adopted table was measured with it off.

## 4. The 29-rule sweep — and what it actually proves

29 VIS-veto rules scored on identical cached fusion outcomes, ranked by **worst
cell** against `bar = max(VIS, IR)`, because a gate is a safety mechanism and its
value is set by the condition it handles least well.

| cell | never veto | always veto | bar | oracle | oracle veto% |
|---|---:|---:|---:|---:|---:|
| clean/day | 0.3715 | 0.0166 | 0.3683 | 0.3722 | 8.7% |
| clean/night | 0.0763 | 0.0813 | 0.0810 | 0.0783 | 0.1% |
| fog/day | 0.0126 | 0.0166 | 0.0177 | **0.0204** | 41.4% |
| fog/night | 0.0772 | 0.0813 | 0.0810 | 0.0845 | 48.5% |
| lowlight/day | 0.0153 | 0.0166 | **0.0346** | **0.0214** | 21.4% |
| lowlight/night | 0.0775 | 0.0813 | 0.0810 | 0.0833 | 41.2% |
| glare/day | 0.2928 | 0.0166 | 0.2892 | 0.2931 | 8.7% |
| glare/night | 0.0640 | 0.0813 | 0.0810 | 0.0685 | 6.0% |

22 of 29 rules tie at exactly **−0.0180 worst-gap**, and all 29 have the same
worst cell: **lowlight/day**. The adopted rule is on that tied frontier. Rules
using detector evidence (`max_conf`, `sum_conf`, `n_c25`) instead of photometry
do not beat it; the widest-window variants are strictly worse (−0.0207).

Rule C (veil only) justifies keeping both veto terms: identical to the adopted
rule everywhere except clean/night 0.0751 and glare/night 0.0649, where the
photometric term is what saves it.

### Correction: lowlight/day is not "unfixable"

An earlier reading of this sweep — including in my own reporting — was that no
per-frame rule can win lowlight/day. **That overstates it, and the reference rows
above say why.**

The sweep varies the **VIS** veto only. On lowlight/day:

- fuse both streams → 0.0153
- veto VIS, keep IR → 0.0166
- **VIS alone → 0.0346**

The bar of 0.0346 *is* VIS alone. Reaching it requires removing **IR** from the
merge, and an IR veto is **not a member of the swept family**. So what the sweep
establishes is narrower and sharper than "unfixable":

> No VIS-veto rule can win lowlight/day, because on that cell the corrective
> action is to drop IR, not VIS.

The oracle's 0.0214 is the ceiling of the *wrong control surface*. This is a
hypothesis with strong numerical support, **not a measured result** — an IR-veto
rule has not been evaluated. `fuse_detections` already accepts `veto_ir` and
`evaluate_systems` accepts a two-sided `veto_override`, so the experiment is
cheap. It is the single highest-value open item.

### The §7.2 premise is also wrong

Handoff §7.2 argues lowlight/day cannot be separated from clean/night by any
frame statistic. `probe_structure.md` §2 tests exactly that and finds **five**
statistics that separate them with positive margin — `p05` and `p50` with
complete separation (lowlight/day `p05` spans 0..0; clean/night spans 1..4):

| stat | direction | keep lowlight/day | veto clean/night | separates |
|---|---|---|---|---|
| p05 | veto when HIGH | 0..0 | 1..4 | **complete** |
| p50 | veto when HIGH | 0..0 | 2..13 | **complete** |
| grad_gini | veto when LOW | 0.777..0.981 | 0.593..0.747 | yes |
| lap_over_var | veto when HIGH | 0.018..13.52 | 14.22..24.85 | yes |
| spec_slope | veto when HIGH | −3.73..−2.13 | −2.05..−1.56 | yes |

So the separability argument fails — but the conclusion survives on the *other*
ground established above: separating them better does not help, because keeping
VIS in the fusion still loses to VIS alone. Two different reasons, and only the
second one holds.

### Caveat on "oracle"

The oracle is coordinate ascent on the labels and lands **below** the adopted
rule on clean/night (0.0783 vs 0.0813) and glare/night (0.0685 vs 0.0813). A true
ceiling cannot sit below an achievable rule, so on at least those cells it is a
local optimum. Read every oracle figure as **"best found by ascent" — a lower
bound on the ceiling, not the ceiling.** The lowlight/day argument is unaffected:
that gap is 0.0132, far outside this slack.

## 5. A process failure worth recording

The first veil eval ran 30 minutes and produced output **byte-identical to the
baseline**. `ctx.py` resolved `tau_lap` *after* the loop that gates
`struct_by_cond` population on `tau_lap` being set, so the structure data was
never loaded and the feature was silently inert.

**All 7 smoke checks passed throughout.** Every one of them exercises
`raw_veto_flags` / `filter_veto` directly; none touched the `build_context`
assembly path where the bug lived. A green suite sat next to a dead feature.

Fixes: ordering corrected, and `build_context` now **prints** whether the veil
term is active, so an inert run announces itself in the first two seconds instead
of after half an hour of identical numbers:

```
[ctx] veil term ACTIVE: lap_var < 508.7, filter=('majority', 15), conditions [...]
[ctx] veil term OFF (no tau_lap in brightness_constants.json) - photometric-only gate
```

The bad output is preserved as `runs/eval/final_system_veil_INERT-BUG.md`.

## 6. Open items, ranked

1. **Evaluate an IR-veto rule on lowlight/day** (§4). Highest value; the
   machinery already exists.
2. **fog/day headroom.** Oracle 0.0204 vs adopted 0.0166 and bar 0.0177 — the
   oracle clears the bar, so a better fog rule can still pay ~+0.0038.
3. **`single_passthrough`** removes a −0.0011 floor on vetoed day cells at a cost
   of −0.0003 on vetoed night cells. Not adopted; the adopted table was measured
   with it off, and changing it would invalidate the comparison without a rerun.
4. **`score_scale` / soft trust** (`scripts/eval_soft_trust.py`) is implemented
   and unmeasured. The old claim that down-weighting cannot remove a failed
   stream is narrower than it sounded: WBF divides cluster scores by
   `sum(weights)`, so with normalized weights only the *ratio* survives and the
   gate's estimate of absolute frame quality is discarded at the last step.
   Riding trust on the scores with equal weights leaves the normalization nothing
   to cancel. Whether that beats the hard veto is untested.
