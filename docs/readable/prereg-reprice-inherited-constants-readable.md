# Pre-registration — re-pricing three inherited constants

**Written 2026-09-02, before any re-priced number exists.** Same discipline as
`docs/prereg-snms-draw-averaged-gate.md`.

## Why

§4.5 of `docs/experiment-log-2026-09-02.md` showed a single corruption draw cannot resolve
±0.001 on low-scoring cells. §4.9 measured the real per-cell floor for a *paired* delta:
**0.0014–0.0031** on cells that carry information, **0.0000** on two that carry none.

Three shipped constants were selected under the single-draw bar, before either number
existed:

| constant | shipped | origin |
|---|---|---|
| `cap_ir_scale` | 4.0 | set by preset; never re-selected under run-disjoint discipline |
| `iou_thr` | 0.85 | `merge_support_split.md` (I9b) — 0.95 scored −0.0001, inside the floor |
| veil repair (`veil_requires_night`) | True | `veil_veto_repair_26m.md`, single draw |

§4.7's scope limit deferred this; this is that work.

## The arms

Single-axis, one constant moved at a time from shipped `crossmodal26m` (bold = shipped):

* `cap_ir_scale` ∈ {1.0, 2.0, **4.0**, 8.0}
* `iou_thr` ∈ {0.70, **0.85**, 0.95}
* `veil_requires_night` ∈ {**True**, False} — moved together with `night_weak_fallback`,
  since §5 of the veil record introduced them as one repair; separating them would price a
  system never proposed.

All three are consumed at `run_systems` time, so arms are built with `dataclasses.replace`
on one loaded context per (draw, IR condition). **Asserted, not assumed:** the run checks a
replace-built `cap_ir_scale = 1.0` arm reproduces a genuinely loaded one to 1e-12 on one
cell, and aborts otherwise.

## The rule (fixed before the run)

1. **Draws.** N = 4 — shipped draw (VIS 1 / IR 7) plus seeds 901/902/903, all nine
   corrupted caches rebuilt per draw. Draws 904/905 excluded: only their two contested
   caches were re-drawn.
2. **Statistic.** Per arm A and per cell (11), delta `AP(A) − AP(shipped)` averaged across
   the 4 draws, day and night separately.
3. **Margin.** Per cell *and per arm*, `margin = 2 × hypot(sd_draw(delta),
   sd_paired_bootstrap(delta))`, both measured in this run. §4.9 numbers are **not**
   imported: the paired bootstrap sd depends on how much the arm moves, so it is measured
   per arm.
4. **Per-cell verdict.** `BETTER` if delta > +margin, `WORSE` if delta < −margin, else
   `INDISTINGUISHABLE`.
5. **Decision.** An alternative **dominates** iff it is not `WORSE` on any informative cell
   **and** `BETTER` on at least one. A constant is **MIS-PRICED** if any alternative
   dominates it; otherwise it **STANDS** — meaning "not shown wrong", never "optimal".
6. **Informative cells.** A cell where *every* arm's delta is exactly 0.0000 is
   uninformative and cannot reject anything. §4.9 identifies `noise_s2/clean` and
   `lowlight/glare_s2` as having a measured delta floor of 0.0000. Such cells are reported
   and **excluded from the count**, never tallied as passing.
7. **TEST rejects, never selects.** If an alternative dominates but is `WORSE` on held-out
   TEST (`pohang02`, `pohang03`) on the clean cell, it is rejected. TEST is not consulted
   to choose among survivors — the C7 rule.
8. **Reported but NOT decision inputs:** per-draw tables, the sd columns, bootstrap CIs,
   per-class splits.

## The loophole, stated in advance

A margin measured from the arm's own variance means a **noisier arm gets a wider margin**,
making "not `WORSE`" easier to satisfy. The asymmetry is not in the arm's favour: `BETTER`
also requires clearing the wider margin, so a noisy arm finds it *harder* to dominate and
*easier* to survive. Consequence accepted in advance: this design can fail to reject a
bad-but-noisy alternative, but cannot promote one. Since the question is "was the shipped
constant wrong", not "which value is best", that is the safe direction. Merely surviving is
**not** endorsement.

## Outcomes

* **STANDS** → decided on a noisy instrument but holds up. Stop re-litigating that axis.
* **MIS-PRICED** → a real regression shipped, as the veil veto did at −0.0632. The fix is a
  new pre-registration for the replacement value, not an immediate swap: rule 7 forbids
  this run from selecting the replacement.

## Scope limit

Three constants only. Does **not** re-open the soft-NMS rejection (§4.6/§4.7), and does not
license changing any value in the same run that identifies it as wrong.
