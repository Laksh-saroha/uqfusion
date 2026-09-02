# Pre-registration — re-pricing three inherited constants

**Written 2026-09-02, before any re-priced number exists.** Committed ahead of the
run so the rule is verifiable in git history rather than asserted afterwards.
Follows the same discipline as `docs/prereg-snms-draw-averaged-gate.md`.

## Why

§4.5 of `docs/experiment-log-2026-09-02.md` showed that a single corruption draw
cannot resolve ±0.001 on the low-scoring cells. §4.9 then measured the real
per-cell floor for a *paired* delta: **0.0014–0.0031** on the cells that carry
information, and **0.0000** on two cells that carry none.

Three constants in the shipped system were selected under the single-draw bar,
before either number existed:

| constant | shipped | where it came from |
|---|---|---|
| `cap_ir_scale` | 4.0 | set by preset; never re-selected under run-disjoint discipline |
| `iou_thr` | 0.85 | `merge_support_split.md` (I9b) — 0.95 scored −0.0001, inside the floor |
| veil repair (`veil_requires_night`) | True | `veil_veto_repair_26m.md`, single draw |

§4.7's scope limit said re-pricing these was separate work. This is that work.

## The arms

Single-axis, one constant moved at a time from the shipped `crossmodal26m`:

* `cap_ir_scale` ∈ {1.0, 2.0, **4.0**, 8.0}
* `iou_thr` ∈ {0.70, **0.85**, 0.95}
* `veil_requires_night` ∈ {**True**, False} — moved together with
  `night_weak_fallback`, because §5 of the veil record introduced them as one
  repair and separating them would price a system that was never proposed.

Bold is shipped. All three are consumed at `run_systems` time, so arms are built
with `dataclasses.replace` on one loaded context per (draw, IR condition). **This
is asserted, not assumed:** the run checks that a replace-built
`cap_ir_scale = 1.0` arm reproduces a genuinely loaded one to 1e-12 on one cell,
and aborts if it does not.

## The rule

Fixed before the run, not to be edited afterwards:

1. **Draws.** N = 4 — the shipped draw (VIS 1 / IR 7) plus seeds 901/902/903, all
   nine corrupted caches rebuilt per draw. Draws 904/905 are excluded: only their
   two contested caches were re-drawn.
2. **Statistic.** For each arm A and each of the 11 cells, the delta
   `AP(A) − AP(shipped)` is averaged across the 4 draws, day and night separately.
3. **Margin.** Per cell *and per arm*,
   `margin = 2 × hypot(sd_draw(delta), sd_paired_bootstrap(delta))`, both measured
   in this run. The §4.9 numbers are **not** imported: the paired bootstrap sd
   depends on how much the arm actually moves, so it must be measured per arm.
4. **Per-cell verdict.** `BETTER` if delta > +margin, `WORSE` if delta < −margin,
   otherwise `INDISTINGUISHABLE`.
5. **Decision.** An alternative **dominates** the shipped value iff it is not
   `WORSE` on any informative cell **and** is `BETTER` on at least one. A constant
   is **MIS-PRICED** if any alternative dominates it; otherwise the shipped value
   **STANDS** — which means "not shown wrong", never "optimal".
6. **Informative cells.** A cell on which *every* arm's delta is exactly 0.0000 is
   uninformative: no system responds to it and it cannot reject anything. §4.9
   already identifies `noise_s2/clean` and `lowlight/glare_s2` as having a measured
   delta floor of 0.0000. Such cells are reported and **excluded from the count**,
   never tallied as passing cells.
7. **TEST rejects, it never selects.** If an alternative dominates but is `WORSE`
   on the held-out TEST runs (`pohang02`, `pohang03`) on the clean cell, it is
   rejected. TEST is not consulted to choose among survivors — the C7 rule.
8. **Reported but NOT decision inputs:** per-draw tables, the sd columns
   themselves, bootstrap CIs, and per-class splits.

## The loophole this rule has, stated in advance

A margin measured from the arm's own variance means a **noisier arm gets a wider
margin**, which makes "not `WORSE`" easier to satisfy. That asymmetry is real and
it is not symmetric in the arm's favour: `BETTER` also requires clearing the wider
margin, so a noisy arm finds it *harder* to dominate and *easier* to survive.

The consequence to accept in advance: this design can fail to reject a bad-but-noisy
alternative, but it cannot promote one. Since the question here is "was the shipped
constant wrong", not "which value is best", that is the safe direction for the
error to point. Any alternative that merely survives is **not** thereby endorsed.

## What each outcome means

* **STANDS** → the constant was decided on a noisy instrument but the decision
  holds up. Record it and stop re-litigating that axis.
* **MIS-PRICED** → a real regression shipped, as the veil veto did at −0.0632.
  The fix is a new pre-registration for the replacement value, not an immediate
  swap: this run identifies that the shipped value is wrong, and rule 7 forbids it
  from selecting the replacement.

## Scope limit

This re-prices three constants. It does **not** re-open the soft-NMS rejection
(§4.6/§4.7), and it does not license changing any value in the same run that
identifies it as wrong.
