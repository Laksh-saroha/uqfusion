# Pre-registration — does the night veto still earn its place? (V1)

**Written 2026-09-03, before the from-scratch VIS run is launched.** This is the
pre-registration that `docs/prereg-night-label-restore.md` said an ALIVE verdict would
license, and nothing more.

## The question

`veto_vis` fires on **100% of night frames**, justified on record by VIS recovering `0.0000`
at night. §7.2 of `docs/experiment-log-2026-09-02.md` showed that number was manufactured by
`filter_night_boxes.py --cut-dark` deleting 132,688 training boxes: restore them and night
VIS goes **0.0000 → 0.2520**.

So the veto answers a detector this project crippled, not darkness. **Whether it is still
right for an uncrippled detector has never been measured** — every night cell in every
fusion table to date is `ir_only` scored against `ir_only`.

## Why a from-scratch run comes first, and is part of this registration

`gauss_vis_nightrestore` is a fine-tune from a checkpoint whose training data asserted night
frames are empty. For the *restore* question that was fine — the prior only made the test
harder. **For the veto question the checkpoint is the treatment, not the test.** Deciding
against a detector still carrying a "night is empty" initialisation would repeat the exact
error already made once: reading a training artefact as a sensor property.

**The run.** `runs/full_scale/gauss_vis_nightfull/`, cold-started from COCO `yolo26m.pt` on
restored labels, with `gauss_vis_seed0`'s recipe unchanged — `data_vis_stride2.yaml`,
`imgsz` 640, `batch` 16, `epochs` 100, `patience` 20, seed 0. Holding the recipe fixed is
the point: the **only** difference from the shipped detector is the 94,553 restored boxes.
Measured cost on the local 4080: 49,131 s (13.65 h) to an early stop at epoch 45.

Nothing under `gauss_vis_seed0/` or `gauss_vis_nightrestore/` is written. The fine-tuned
checkpoint is retained as a **secondary** arm; if the two disagree, the cold start decides.

## The endpoint

Fused **night** `mAP@50-95` on `pohang01`, veto **ON** vs **OFF**, both arms using the same
from-scratch VIS checkpoint and the same IR stream:

    delta = fused_night(veto OFF) − fused_night(veto ON)

Scored through `scripts/run_fusion_eval.py`, preset `crossmodal26m`, VIS weights swapped and
nothing else touched.

## The rule (fixed before the run)

1. **Draw averaging is mandatory.** A single corruption draw cannot resolve these margins:
   on cells scoring ~0.027 the baseline swings 7× the decision margin. Minimum **4 draws**,
   paired (identical draw seeds for both arms), averaged before the delta is formed — the
   `scripts/gate_snms_draw_avg.py` precedent.
2. **Paired bootstrap** over frames for `se`, resampling the same frames for both arms so
   level noise cancels.
3. **Magnitude floor.** `floor = max(2 * se, 0.002)`. A measured margin alone degenerates
   into a sign test at ~1e-5, which has already forced two decisions to be reopened.
4. **Bands.**
   * **ADOPT** — `delta >= +floor`. Night veto removed; VIS re-enters the night fusion input
     list.
   * **REJECT** — `delta <= -floor`. The veto stands on its merits rather than a retracted
     justification — itself a publishable finding.
   * **INCONCLUSIVE** — anything between. **The veto stands.** An unresolved measurement is
     not a licence to change shipped behaviour.
5. **Guard 1 — day must be bit-identical.** `veto_vis` fires only on night frames, so every
   day cell must reproduce to the last decimal under both arms. Any day movement means the
   implementation touched something outside the night path: the run is **void**, not
   interpreted.
6. **Guard 2 — degraded IR at night.** With the veto off, both streams are live at night.
   Every night cell with corrupted IR is reported individually; if any falls below its own
   `ir_only` value by more than `floor`, the verdict is capped at **INCONCLUSIVE** whatever
   clean-night does. A fusion rule better on average and worse when a sensor degrades is not
   an improvement.
7. **Reported but NOT decision inputs:** per-class AP, `mAP@50`, the fine-tuned secondary
   arm, every day cell.

## There is no held-out night data

`pohang01` is the only night run in the dataset, held out of every gate fit precisely
because of that. This decision is therefore made on the same night data any future night
work will be scored on, and **no split can rescue it**. Registering the bands in advance is
the only protection available, which is why this document exists before the checkpoint does.
Any ADOPT verdict must carry that caveat in the paper.

## What an ADOPT verdict does NOT license

**Adopting a new VIS detector.** Every headline number reproduces from
`runs/full_scale/gauss_vis_seed0/`. Swapping in a night-capable checkpoint re-baselines the
entire benchmark — Phase 1's ranking, the UQ arms, the gate stack, all 17 VIS caches and the
4,000-frame Mahalanobis reference, whose features move with the model. That must not be
smuggled in under this registration. ADOPT licenses removing the night veto **and a costed
proposal** for the re-baseline, not the re-baseline itself.

Also out of scope, all re-priced 2026-09-02 and not reopened: the cross-modal gate
mechanism, `cap_ir_scale`, `iou_thr`, the veil repair.

## Weaknesses, stated in advance

1. **Night val contains zero buoys.** All 16,179 night GT boxes are class 0, so the endpoint
   is ship AP. The veto's effect on buoy at night is unmeasurable here and must not be
   asserted in either direction.
2. **Fusion at `iou_thr` 0.85 is 99.9% concatenation, not consensus** — only 0.05% of VIS
   boxes have an IR partner. Any ADOPT gain is most likely union recall, not sensor
   agreement. The report must say which, by quoting the partner rate on the night cells.
3. **One seed.** Cold start is seed 0 only, so a null cannot be separated from seed noise at
   ~0.002. Accepted because a three-seed cold start is ~41 h and the rule-3 floor is set
   above single-seed noise, not below it.
4. **The IR stream is uncorrupted in every benchmark cell**, so guard 2 tests degraded IR
   only in synthetic corruption cells, not in the wild.
