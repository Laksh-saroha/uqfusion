# The night veto: three registrations, and why the axis is closed

**2026-09-04.** V1, V2 and V3 all ran; none adopted. The veto stands, and
`docs/prereg-night-veto-v3.md` §6 closes the axis. This is the durable record — the
reports themselves live under `runs/eval/`, which is not tracked.

| | rule | clean-night delta | verdict | what stopped it |
|---|---|---:|---|---|
| **V1** | remove the night veto | +0.1785 | INCONCLUSIVE | guard 2: 2 night cells below `ir_only` |
| **V2** | `night AND (q < thr)` | +0.1785 | INCONCLUSIVE | clause (b): 5 night cells below `max(VIS, IR)` |
| **V3** | `(night OR disarmed-night) AND (q < thr)` | +0.1785 | **VOID** | day guard: one cell's day veto 0.8% → 19.9% |

Secondary V1 arm (`gauss_vis_nightrestore`, reported not decision): +0.1962, same verdict,
same two guard-2 cells. The finding does not depend on the checkpoint.

## What was learned

**1. The veto was right for a retracted reason.** `filter_night_boxes.py --cut-dark`
deleted 94,553 boxes and made night VIS score 0.0000; the veto answered that, not darkness.
On restored labels VIS-only night measures **0.2233**, and removing the veto on a clean
night frame is worth **+0.1785** — 33× the decision floor.

**2. Darkness was never the right variable.** V1's cells split perfectly: all five
VIS-healthy night cells say ADOPT, all five VIS-degraded ones say REJECT. The veto gates on
darkness; what predicts whether VIS should be dropped is VIS health. Under the crippled
detector those were the same variable — not any more.

**3. The health instrument works; one line of fitting was the whole problem.** `vis_health`
as shipped flags 100% of night frames unhealthy in *every* condition including clean, with
the ordering inverted — it is fitted on FIT_RUNS, which excludes `pohang01`, so night is
novel by construction. Pool 782 clean night frames into its reference and the same
statistic reaches **AUROC 0.9921**. The 782 reference frames and the 1,032 evaluated frames
are disjoint (`maha_fit_vis` vs `paired_val_vis`, zero overlap).

**4. V3 fixed night completely and broke day safety doing it.** Clause (b) went from 5
breaches under V2 to **0**; the veto fires 100% → 0% on all five clean-VIS night cells and
stays 96–100% on the degraded ones — exactly as intended. But on `lowlight` / IR `glare_s2`,
day veto rose **0.8% → 19.9%**.

Measured, day frames, IR `glare_s2`:

| term | rate |
|---|---:|
| `ir_night_raw` — the p05 test alone | **19.9%** |
| `~ir_ok` — IR fails the authority bound | 82.1% |
| `ir_night = raw AND ok` — V1/V2's trigger | **0.6%** |
| `raw AND ~ok` — the term **V3 adds** | **19.3%** |

A glare-corrupted IR misreads one clear day in five as night. The **authority bound** catches
that, cutting 19.9% → 0.6%. V3 deliberately re-admits the calls the bound rejected and asks
health to confirm them. With VIS clean, health blocks it (unhealthy 0.0% → day veto 0.0%,
clause (c) passes). With VIS `lowlight`, health *confirms* it (unhealthy 100.0% → day veto
19.9%).

**The two votes stopped being independent.** "VIS is unhealthy" is true of a corrupted
daylight frame and a real night frame alike, so health cannot serve as the second,
independent confirmation the safety property needs. Same shape of error as §7.2 and V1 — a
proxy asked to carry a claim it does not contain — reached from the opposite direction.

## An imprecision in V3's registration, stated rather than exploited

V3's day guard says a day cell beyond floor makes the run "VOID rather than interpreted",
language inherited from V2 where VOID meant *the implementation escaped the night path*.
Here nothing escaped: V3's trigger contains `ir_night_raw AND ~ir_ok`, which is not
night-confined by construction, so the day movement is the registered rule behaving as
written. The guard conflated "implementation leak" with "any day movement".

**This changes nothing.** VOID or substantive day-safety failure, V3 is not an ADOPT and §6
closes the axis either way. Recorded because the next registration should write a day guard
that distinguishes the two, and because noticing an imprecision only when it might help is
exactly the failure this project guards against.

## Status

* **The veto stands**, unchanged, in `crossmodal26m`.
* **No V4.** No fourth clause, no re-selection of the instrument, no moving the floor or
  clause (b). Registered in `docs/prereg-night-veto-v3.md` §6 before the V3 number was
  known, because each round removed exactly the obstacle the last one hit.
* **Nothing here deploys.** Under shipped `gauss_vis_seed0`, VIS scores 0.0000 at night and
  every version of this rule reaches the same answer by a longer route. All three
  registrations measured a system whose VIS checkpoint is not the shipped one.
* **The open question is the detector, not the rule.** `docs/rebaseline-proposal-2026-09-04.md`
  prices it: ~2.5 h caching and refitting, plus ~40 h for a night-capable VIS ensemble that
  does not exist — the four `ens_vis_seed{1..4}_ft` runs all started 8–10 days before the
  labels were restored and are void for this purpose.

## Caveats travelling with every number above

1. `pohang01` is the only night run in the dataset. Stage 0 selected its instrument on the
   same 1,032 night frames Stage 1 and V3 scored; no split can create a second night recording.
2. `q_refit` is **saturated** — exactly 1.0 for 100% of clean-night frames and 1.59% of
   degraded-night ones. A binary novelty-bound test, not a graded score, so AUROC 0.9921
   reads richer than the instrument is.
3. `gauss_vis_nightfull` is seed 0, early-stopped at epoch 11 of 100 with LR still annealing.
4. Fusion at `iou_thr` 0.85 is ~99.9% concatenation — 0.143% of VIS boxes had an IR partner
   on the clean night cell. Every gain above is union recall, not sensor consensus.
5. Night val contains zero buoys; all 16,179 night GT boxes are class 0, so the endpoint is
   ship AP throughout.
