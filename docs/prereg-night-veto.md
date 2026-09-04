# Pre-registration — does the night veto still earn its place? (V1)

**Written 2026-09-03, before the from-scratch VIS run is launched.** Committed ahead
of the run so the rule is verifiable in git history. This is the pre-registration
that `docs/prereg-night-label-restore.md` said an ALIVE verdict would license, and
nothing more than that document licensed.

## The question

`veto_vis` fires on **100% of night frames**. The justification on record was that
VIS recovers `0.0000` at night. §7.2 of `docs/experiment-log-2026-09-02.md` showed
that number was manufactured by `filter_night_boxes.py --cut-dark` deleting 132,688
training boxes: restore them and night VIS goes **0.0000 → 0.2520**.

So the veto is a correct response to a detector this project crippled, not to
darkness. **Whether it is still the right response to an uncrippled detector has
never been measured.** Every night cell in every fusion table to date is `ir_only`
scored against `ir_only`, which cannot answer it.

## Why a from-scratch run comes first, and is part of this registration

`gauss_vis_nightrestore` is a fine-tune from a checkpoint whose training data
asserted that night frames are empty. For the *restore* question that was fine — an
ALIVE verdict had to overcome that prior, so the prior only made the test harder.

**For the veto question the checkpoint is the treatment, not the test.** Deciding the
veto against a detector still carrying a "night is empty" initialisation would repeat
precisely the error this project already made once: reading a training artefact as a
sensor property. The primary arm therefore has to be a cold start.

**The run.** `runs/full_scale/gauss_vis_nightfull/`, cold-started from COCO
`yolo26m.pt`, on the restored labels, with `gauss_vis_seed0`'s own recipe unchanged —
`data_vis_stride2.yaml`, `imgsz` 640, `batch` 16, `epochs` 100, `patience` 20, seed 0.
Holding the recipe fixed is the point: the **only** difference from the shipped
detector is the 94,553 restored boxes. Measured cost on the local 4080 at that recipe
is 49,131 s (13.65 h) to an early stop at epoch 45.

Nothing under `runs/full_scale/gauss_vis_seed0/` or `gauss_vis_nightrestore/` is
written. The fine-tuned checkpoint is retained as a **secondary** arm; if the two
disagree, the cold start is the one that decides.

## The endpoint

Fused **night** `mAP@50-95` on `pohang01`, veto **ON** versus veto **OFF**, both arms
using the same from-scratch VIS checkpoint and the same IR stream:

    delta = fused_night(veto OFF) − fused_night(veto ON)

Scored through the existing machinery (`scripts/run_fusion_eval.py`, preset
`crossmodal26m`) with the VIS weights swapped and nothing else touched.

## The rule

Fixed before the run, not to be edited afterwards.

1. **Draw averaging is mandatory.** A single corruption draw cannot resolve the
   margins at issue: on cells scoring ~0.027 the baseline itself swings 7x the
   decision margin. Minimum **4 draws**, paired (identical draw seeds for both arms),
   averaged before the delta is formed — the `scripts/gate_snms_draw_avg.py`
   precedent.
2. **Paired bootstrap** over frames for `se`, resampling the same frames for both
   arms so level noise cancels.
3. **Magnitude floor.** `floor = max(2 * se, 0.002)`. A measured margin alone
   degenerates into a sign test at ~1e-5, which has already produced two decisions
   this project had to reopen.
4. **Bands.**
   * **ADOPT** — `delta >= +floor`. The night veto is removed; VIS re-enters the
     night fusion input list.
   * **REJECT** — `delta <= -floor`. The veto stands on its merits rather than on a
     retracted justification, and that is itself a publishable finding.
   * **INCONCLUSIVE** — anything between. **The veto stands.** The default is the
     shipped behaviour; an unresolved measurement is not a licence to change it.
5. **Guard 1 — day must be bit-identical.** `veto_vis` fires only on night frames, so
   every day cell must reproduce to the last decimal under both arms. Any day
   movement means the implementation touched something outside the night path and the
   run is **void**, not interpreted.
6. **Guard 2 — degraded IR at night.** With the veto off, both streams are live at
   night. Every night cell in which IR is corrupted is reported individually, and if
   any falls below its own `ir_only` value by more than `floor`, the verdict is capped
   at **INCONCLUSIVE** whatever the clean-night delta does. A fusion rule that is
   better on average and worse when a sensor degrades is not an improvement.
7. **Reported but NOT decision inputs:** per-class AP, `mAP@50`, the fine-tuned
   secondary arm, and every day cell.

## There is no held-out night data, and this cannot be cross-validated

`pohang01` is the only night run in the dataset, and it is held out of every gate fit
precisely because of that. It follows that this decision is made on the same night
data that any future night work will be scored on, and **no split can rescue it**.
Registering the bands in advance is the only protection available here, which is why
this document exists before the checkpoint does. Any ADOPT verdict must carry that
caveat in the paper.

## What an ADOPT verdict does NOT license

**Adopting a new VIS detector.** Every headline number in this project reproduces
from `runs/full_scale/gauss_vis_seed0/`. Swapping in a night-capable checkpoint
re-baselines the entire benchmark — Phase 1's ranking, the UQ arms, the gate stack,
all 17 VIS caches and the 4,000-frame Mahalanobis reference, whose features move with
the model. That is a far larger change than a veto flag and it must not be smuggled in
under this registration. ADOPT licenses removing the night veto **and a costed
proposal** for the re-baseline; it does not license the re-baseline itself.

Also out of scope, all re-priced on 2026-09-02 and not reopened here: the cross-modal
gate mechanism, `cap_ir_scale`, `iou_thr`, and the veil repair.

## Weaknesses, stated in advance

1. **Night val contains zero buoys.** All 16,179 night GT boxes are class 0, so the
   endpoint is ship AP. The veto's effect on buoy at night is unmeasurable with this
   val set and must not be asserted in either direction.
2. **Fusion at `iou_thr` 0.85 is 99.9% concatenation, not consensus** — only 0.05% of
   VIS boxes have an IR partner. Any ADOPT gain is therefore most likely recall from
   the union of two box sets, not agreement between two sensors. The report must say
   which, by quoting the partner rate on the night cells rather than assuming it.
3. **One seed.** The from-scratch run is seed 0 only, so a null cannot be separated
   from seed noise at the ~0.002 level. This is accepted because a three-seed cold
   start is ~41 h and the floor in rule 3 is set above the single-seed noise, not
   below it.
4. **The IR stream is uncorrupted in every benchmark cell**, so guard 2 tests
   degraded IR only in the synthetic corruption cells, not in the wild.
