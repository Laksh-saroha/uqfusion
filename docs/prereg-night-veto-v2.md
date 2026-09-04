# Pre-registration — should the night veto gate on VIS HEALTH instead of darkness? (V2)

**Written 2026-09-04, before any V2 arm has been scored.** Committed ahead of the
run so the rule is verifiable in git history. This is a **new** registration, not
an amendment to `docs/prereg-night-veto.md` and not a re-run of it. V1 returned
INCONCLUSIVE, the veto stands, and nothing below reopens that verdict.

## The question, and why V1 could not answer it

`runs/eval/night_veto_verdict.md` split cleanly along a line the V1 bands could
not express:

| night cells | delta (veto OFF − ON) | band |
|---|---|---|
| VIS **healthy** (clean VIS, any IR) | +0.1785 … +0.2255 | ADOPT, 5 of 5 |
| VIS **degraded** (blur / rain / fog / lowlight) | −0.0102 … −0.0737 | REJECT, 5 of 5 |

The veto gates on **darkness**. The variable that actually predicts whether VIS
should be dropped is **VIS health**. Under `gauss_vis_seed0` those were the same
variable — a detector trained on empty night labels is degraded exactly when it
is dark — so nothing distinguished them and nothing needed to. Under a
night-capable detector they come apart, and the rule has not followed.

**V2 asks: is there an instrument that can tell those two cases apart at night,
and does gating on it beat gating on darkness?**

## The instrument this registration REFUSES, and the measurement that refuses it

`crossmodal26m` already loads a VIS health score, `q_vis_by_cond`, from the
`vis_health` axis of `structure_constants.json`. The obvious V2 rule — "veto when
IR says night AND `q_vis` says unhealthy" — is **excluded before the fact**, on
this measurement (`runs/cache_nv_draw1`, `gauss_vis_nightfull`, 1,032 night
frames; q = 1 healthy, 0 novel):

| VIS condition | q̄ night | q̄ day | q < 0.5, night | q < 0.5, day |
|---|---:|---:|---:|---:|
| clean | 0.0254 | 1.0000 | **100.0%** | 0.0% |
| blur_s3 | 0.0359 | 0.9661 | 100.0% | 1.4% |
| rain_s2 | 0.0273 | 0.5172 | 100.0% | 41.1% |
| fog | **0.1931** | 0.2330 | 100.0% | 100.0% |
| lowlight | 0.0090 | 0.0437 | 100.0% | 100.0% |

`vis_health` is fitted on clean **FIT-RUN** frames, and FIT_RUNS excludes
`pohang01` — which is the entire night set. Night is therefore novel *by
construction*: the axis flags 100% of night frames unhealthy in every condition
including clean, and the ordering is **inverted** — fogged night scores healthier
(0.1931) than clean night (0.0254).

So "night AND unhealthy" is arithmetically identical to "night", and a V2 built
on `q_vis` would collapse into V1 while appearing to be a new rule. This is the
§7.2 error one level up: an instrument that cannot tell a dark world from a
broken sensor, being asked to do exactly that. **Recorded here so that it cannot
be tried later and reported as a finding.**

## Stage 0 — screening, which adopts nothing

Candidate instruments, fixed now. Each must be computable from artefacts that
already exist, because an instrument requiring new fitting is a new constant and
would need its own registration:

1. **`sigma_frame`** — mean `per_box_uncertainty` over the frame's detections.
   The project's stated premise is uncertainty-gated fusion; this is the arm that
   premise predicts should work.
2. **`n_det`** — detection count at conf 0.001.
3. **`conf_mean`** — mean detection confidence.
4. **`vis_health_night_refit`** — `vis_health` refitted with clean `pohang01`
   pooled into its reference. Included because the defect above is a *fitting*
   defect, not a defect of the statistic; excluded from adoption if it wins on
   the basis in "What Stage 0 cannot launder" below.

**Screening metric, fixed:** AUROC of the instrument over the 1,032 night frames,
labelled by cell — positive = the five VIS-healthy cells (where V1 measured the
veto to be harmful), negative = the five VIS-degraded cells (where V1 measured it
to be helpful). Draw-averaged over the same 4 paired draws as V1.

**Bar: AUROC ≥ 0.90.** Justified, not chosen for convenience: **the incumbent
rule scores exactly 0.5 on this discrimination**, because it fires on 100% of
night frames in every cell and so orders them not at all. Any bar above 0.5 is
"better than what ships". 0.90 is the margin demanded of a change that can only
be deployed alongside a full re-baseline (`docs/rebaseline-proposal-2026-09-04.md`).

**If no candidate clears 0.90, V2 is ABANDONED and the veto stands.** That is a
live outcome and the most likely one. It is not a reason to lower the bar, add a
candidate, or move to a per-frame oracle.

**Selection rule if more than one clears:** the highest AUROC wins; ties inside
0.01 go to the instrument with fewer inputs. Stage 1 runs on exactly one
instrument. No ensembling of candidates, no threshold search beyond the single
Youden point of the winning ROC, and that threshold is read off the ROC — not
tuned against mAP.

## Stage 1 — the gate

Rule under test:

    veto_vis  =  ir_night  AND  (instrument says VIS is degraded)

replacing `crossmodal26m`'s `night & (dark | veil)`. Everything else is
untouched: same preset, `iou_thr` 0.85, same IR stream, same VIS checkpoint
(`runs/full_scale/gauss_vis_nightfull/weights/best.pt`), 4 paired draws
(VIS 1/901/902/903, IR 7/911/912/913), n_boot 2000.

**Endpoint:** draw-averaged night `mAP@50-95` per cell, V2 rule versus V1 (veto
ON, as shipped).

**Floor:** `floor = max(2 * se, 0.002)`, `se` the mean of the per-draw paired
bootstrap se. Same construction as V1, and for the same reason: a measured margin
alone degenerates into a sign test.

**ADOPT requires BOTH:**

* **(a) Gain where V1 said gain.** Draw-averaged night delta `>= +floor` on the
  clean-VIS/clean-IR cell.
* **(b) No cell below `max(VIS_only, IR_only) − floor`, on any night cell.**
  This is V1's guard 2 generalised, and it is the clause V1 failed. A rule that
  wins on average and loses when a sensor degrades is not an improvement, and the
  whole point of V2 is to be the rule that does not have to make that trade.

**REJECT** if (a) fails by `<= -floor`. **INCONCLUSIVE** otherwise — and
INCONCLUSIVE, as in V1, **leaves the veto standing**. The default is shipped
behaviour.

**Guard — day.** Every day cell must land within `floor` of its V1 value. The V2
rule is night-gated, so a day movement beyond that means the implementation
escaped the night path. Unlike V1's rule 5, this is stated as a tolerance rather
than as bit-identity: V1's own run showed the veto firing on 0.8% of day frames
in one cell, so bit-identity was never the right bar and demanding it again would
void a valid run.

**Reported, and NOT decision inputs:** per-class AP, `mAP@50`, the partner rate
at 0.85 on every night cell, the fine-tuned secondary arm, every day cell, and
the Stage 0 ROC curves.

## What Stage 0 cannot launder

The screening labels come from V1's cell-level result on **the same 1,032 night
frames** Stage 1 scores. `pohang01` is the only night run in the dataset; there is
no second night recording to hold out and no split can create one. So V2 selects
its instrument on the data it is then evaluated on, and **that is a real weakness
that no amount of procedure removes.**

Three things bound it, and they are the only defences available:

1. Labels are **cell-level, not per-frame**. Ten labels, not 1,032. An instrument
   cannot memorise a frame it never sees labelled.
2. The bar is set **before** any candidate is run, and at a level (0.90) far above
   the incumbent's 0.5.
3. **Abandonment is a registered outcome**, so "no instrument works" is a result
   this document can produce rather than a failure it must avoid.

Any ADOPT verdict must carry this paragraph in the paper.

## What an ADOPT verdict does NOT license

**Deploying V2 on the shipped system.** Under `gauss_vis_seed0` VIS scores 0.0000
at night, the veto is free, and V2 would be a more complicated way to reach the
same answer. **V2 is only meaningful together with the detector re-baseline**, and
`docs/rebaseline-proposal-2026-09-04.md` prices that at ~2.5 h of caching plus
~40 h for a night-capable VIS ensemble that does not yet exist — the four
`ens_vis_seed{1..4}_ft` runs on dgxanode01 all started 8–10 days before the labels
were restored and are void for this purpose (verified 2026-09-04).

So ADOPT here means: *when* the re-baseline happens, V2 is the night rule. It does
not authorise the re-baseline and it does not change the shipped system on its own.

Also out of scope and not reopened: the cross-modal gate mechanism, `cap_ir_scale`,
`iou_thr`, the veil repair, and V1's verdict.

## Weaknesses, stated in advance

1. **One seed, and an early stop.** `gauss_vis_nightfull` is seed 0 and stopped at
   epoch 11 of a 100-epoch schedule with the LR still annealing (pg0 ≈ 0.021). It
   is not a fully converged model, and every V2 number inherits that.
2. **Night val contains zero buoys.** All 16,179 night GT boxes are class 0, so
   the endpoint is ship AP and V2 says nothing about buoy at night.
3. **Fusion at 0.85 is 99.9% concatenation** — 0.143% of VIS boxes had an IR
   partner on the clean night cell in V1. Any V2 gain is union recall, and the
   report must quote the partner rate rather than imply consensus.
4. **The veil axis travels with the night arm.** `veil_requires_night` is True
   under `crossmodal26m`, so the V2 rule replaces both terms at once. V2 is
   therefore "no VIS veto unless the instrument fires", not "night arm only".
5. **The corruption cells are synthetic.** Stage 0's negative class is four
   synthetic VIS corruptions, not degradation observed in the wild, and an
   instrument that separates synthetic fog from clean night may not separate real
   ones.
