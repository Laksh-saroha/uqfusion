# Pre-registration: two degraded-IR repairs to `crossmodal26m`, development (2026-10-10)

Written before any score of either repair at VIS draws 942–944 / IR draws 952–954 exists. The
IR caches for those draws were still being built when this was written
(`runs/v2_ir_draws_942_944.log`); no AP has been computed on them under any preset.

## 1. What is being repaired

Under corruption v2, two parts of the shipped preset misbehave when the IR stream is degraded
(`docs/eval/corruption_v2/README.md` §8.6–8.9b, all draw 941 / 951, development, descriptive):

* **The weak-IR fallback** (`ir_night_raw AND NOT ir_ok AND (concentrated OR (dark AND veil))`)
  vetoes a low-light VIS stream on up to 96.3% of day frames when IR is fogged or noisy, costing
  −0.0067 to −0.1643 ship AP. v2 low light is both dark and veiled, so `dark AND veil` confirms a
  false IR night call.
* **The cross-modal support bonus** (score ×1.5 at IoU ≥ 0.30 with a box from the other stream)
  costs 0.0113 under IR fog s3: scattered IR boxes vouch for VIS false positives. It is worth
  +0.0068 with clean IR and +0.0157 under IR noise s2.

## 2. The rules

* **R1, fallback without veil.** The fallback confirms with `concentrated` alone:
  `ir_night_raw AND NOT ir_ok AND concentrated`. (`FusionContext.fallback_veil = False`.)
* **R2, support only from a merge-healthy IR.** The support bonus applies on a frame only where IR
  passes its merge-health flag `healthy_ir` (`q_ir >= 0.5`, the pre-registered midpoint the IR merge
  veto already uses). IR stays in the merge on every frame; only the bonus is withheld.
  (`FusionContext.support_requires_ir_health = True`.) The flag was chosen before any R2 number
  existed, on the argument that the bonus is a merge operation and `healthy_ir` is the merge's own
  health instrument; the tighter authority bound `ir_ok` was not chosen because it flags 100% of IR
  noise s1–s3 frames (Table 6), where the bonus is measured positive.
* **R1+R2** is `preset="crossmodal26m_r"`.

Both switches default to the shipped behaviour; `crossmodal26m` is unchanged and bit-identical
(checked before any repaired number was read: fused boxes and scores hashed under the pre-edit
and post-edit code on three IR conditions × two VIS conditions).

## 3. Data, arms, statistic

* Five Phase 3 systems (VIS seed k + IR seed k), `runs/cache_p3dev_v2`, development paired frames
  (1,200 day from pohang00/02/03, 1,032 night from pohang01).
* **Verdict draws: VIS 942, 943, 944 with IR 952, 953, 954.** Draw 941 / 951 is excluded from the
  verdict because R1 was chosen after seeing it (`fallback_fix.md`) and the support-off arm was seen
  there too; it is reported beside the verdict, descriptively.
* Cells: VIS {clean, fog, fog_s1, glare, lowlight} × IR {clean, fog_s1, fog_s2, fog_s3, noise_s2,
  noise_s3} × {day, night} = 60 cells.
* Arms: shipped (`crossmodal26m`), R1, R2, R1+R2, each toggled on the same loaded context.
* Statistic: fused ship AP (local AP, `ship_ap`). Per system, the draw mean over the three verdict
  draws; Δ = arm − shipped, paired by system; 95% t-interval over the five systems (df 4).

## 4. Decision rule, per arm

* **(a) No harm.** No cell with mean Δ ≤ −0.0014 and the interval's upper bound below zero.
  (0.0014 is the lower end of the paired noise floor; the count is also reported at 0.0008, 0.0024
  and 0.0060, as `project-gate-magnitude-floor` requires, but only 0.0014 decides.)
* **(b) Clean-IR invariance.** On the 10 clean-IR cells, |mean Δ| < 0.0014. The repaired preset may
  not move the published Table 3b grid.
* **(c) Benefit.** At least one cell with mean Δ ≥ +0.0060 and the interval's lower bound above zero.

**ADOPT** an arm if (a), (b) and (c) all hold. If both R1 and R2 adopt, `crossmodal26m_r` is
R1+R2, provided R1+R2 also satisfies (a) and (b); if exactly one adopts, `crossmodal26m_r` is
redefined to that rule alone; if neither adopts, `crossmodal26m_r` is removed and both are reported
as rejected.

Also reported, not deciding: the Table 3b claim (fused − max(VIS, IR) fails at ≤ −0.0060 with the
interval below zero) counted over the 60 cells for every arm, the veto rates, and draw 941 alone.

## 5. What ADOPT means

`crossmodal26m_r` becomes the recommended configuration for a degraded IR stream, reported as a
registered **development** result. It is not validated on held-out data, because none remains:
pohang04 has been exposed twice and is not touched here. The pohang04 look, Table 3b and every
result reported under `crossmodal26m` stay as they are. **No pohang04 frame is read or scored.**

Script: `scripts/v2_degraded_ir_repairs.py`. Output: `docs/eval/corruption_v2/degraded_ir_repairs.{md,json}`.
