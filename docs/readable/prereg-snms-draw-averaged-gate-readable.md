# Pre-registration — draw-averaged adoption gate for VIS soft-NMS

**Written 2026-09-02, before any draw-averaged number exists.** Committed ahead of the
run so the rule is verifiable in git history.

## Why the previous gate must be redone

`runs/eval/vis_soft_nms_adoption_v2.md` rejected VIS soft-NMS because one cell of eleven
— `blur_s3/glare_s2` — came in at **−0.0004**, against a bar of "at or above the shipped
system on every cell".

`runs/eval/snms_cell_redraw.md` re-drew that cell's two corruptions at five further seeds,
kind and severity fixed:

| draw | delta day |
|---|---:|
| 1/7 (shipped) | −0.0004 |
| 901/911 | +0.0021 |
| 902/912 | −0.0005 |
| 903/913 | +0.0012 |
| 904/914 | +0.0012 |
| 905/915 | +0.0020 |

Mean +0.0010, sd 0.0011, negative on 2 of 6. The shipped *baseline* on that cell swings
0.0260–0.0289 across draws — ~7× the −0.0004 that tripped the bar. The `clean/clean`
control is +0.0012 on all six rows, identical to four decimals, so the movement is the
corruption seed and nothing else.

**The conclusion is about the instrument, not soft-NMS.** A single-draw every-cell test on
cells scoring ~0.027 cannot resolve ±0.001; it accepts and rejects by coin flip *in both
directions*. That applies to every adoption decision made on the low-scoring corrupted
cells. So the bar is re-specified — **not** loosened, just with draw noise averaged out.

## The rule (fixed before the run, not editable afterwards)

1. **Draws.** N = 4: the shipped draw (VIS seed 1 / IR seed 7) plus VIS seeds 901/902/903
   with IR seeds 911/912/913. Corruption *kind* and *severity* fixed at shipped values on
   every draw; only the seed moves.
2. **Width.** `vis_soft_nms` = **0.5**, unchanged — the value `probe_within_modality.py`
   measured before the TUNE/TEST split was looked at. Not re-tuned; no width comparison
   feeds the decision.
3. **Statistic.** Per benchmark cell (11), `delta_day` and `delta_night` averaged across
   the 4 draws.
4. **ADOPT iff** draw-averaged `delta_day` ≥ 0 on **every** cell **and** draw-averaged
   `delta_night` ≥ 0 on **every** cell.
5. **TEST rejects, never selects.** If the draw-averaged delta on held-out TEST runs
   (`pohang02`, `pohang03`) is negative on the clean cell, the arm is rejected regardless
   of rule 4. TEST is consulted for nothing else, and selects no parameter — the C7 rule.
6. **Reported but NOT decision inputs:** per-cell sd across draws, count of draws each cell
   is negative on, per-draw tables, frame-level bootstrap CIs. Adding any of them to the
   decision after seeing the numbers is the failure this document prevents.

## Outcomes

* **Passes** → adopt: `crossmodal26m_snms` becomes the shipped preset and the headline
  table is re-baselined under it. The four published `crossmodal26m` numbers stay
  reproducible under their own preset.
* **Fails** → soft-NMS stays off, and the failure is a real reproducible cost rather than a
  single draw. Record and move on.

Either way the **instrument** change stands: the draw-averaged form is the correct bar for
cells at this scale, and future arms should be gated with it.

## Scope limit

This re-prices one arm. It does **not** retroactively re-open decisions already made under
the single-draw bar (`cap_ir_scale` ×4, the veil-veto repair, `iou_thr` 0.85). Those were
decided under a bar now known to be noisy at the low-scoring cells, and several turned on
margins larger than this noise — re-pricing them is separate work.
