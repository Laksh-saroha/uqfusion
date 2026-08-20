# TODO — full-scale training and publication (post-finalization)

**Created 2026-08-20**, after the architecture finalization
([`architecture-final-2026-08-20.md`](architecture-final-2026-08-20.md),
decisions D27–D30). **Supersedes the open items of
[`TODO-improvements.md`](TODO-improvements.md)** — that file remains the
2026-08-19 record; everything still outstanding from it is carried here with
its old ID in brackets.

The frozen architecture in one line: *per-modality yolo26s σ-head detectors
(IR = p2feat @640) → WBF in the VIS frame, weighted by run-disjoint capability
× soft reliability (Mahalanobis × r_box) — with VIS thrown out of the merge
entirely whenever its content brightness says no photons arrived (p05 < mu_b,
dilated over 15 frames in capture order).*

## The architecture, in plain language

**Two detectors, one merge rule, one kill switch.**

- **Two detectors.** VIS and IR each get their own yolo26s + Gaussian σ head
  (IR with the p2feat neck — the one screening change that clearly mattered,
  +34% at night). Each also scores how *strange* the current frame looks
  against its own clean training data (Mahalanobis on pooled neck features).
- **One merge rule.** IR boxes are mapped into the VIS image (per-run
  homography) and merged by WBF (`iou_thr` 0.85, σ-weighted coordinates).
  Each sensor's weight ≈ *how good it is on a clear day* (capability prior,
  computed on the daytime runs only) × *how normal the frame looks to it*
  (Mahalanobis × r_box).
- **One kill switch.** Before merging, ask of the VIS frame: *did any light
  reach the sensor?* The 5th-percentile brightness of the image content — a
  statistic glare and fog cannot fake upward — is compared to a threshold
  placed in the empty gap between dark and lit frames (`mu_b` = 10.5, margin
  rule, no optimizer). Below it, VIS **leaves the merge entirely** rather than
  being down-weighted. The switch is sticky: if any frame in the last 15 was
  dark, VIS stays out (darkness is a property of a stretch of recording, not
  one frame). IR is never vetoed — a dark thermal frame is cold water, which
  is IR's job.

**Why a switch instead of weights:** down-weighting cannot save a blind
stream — weights renormalize, WBF rescales scores instead of dropping boxes,
and mAP is rank-based — so a blind camera drags the working one down at any
weight. The governing rule: *a signal gets switch authority only if it is
monotone in sensor health.* Brightness is; the Mahalanobis score is not (glare
looks strange while the detector still works), so it may only nudge weights.

**What each piece earns** (`runs/eval/final_system.md`): the veto carries
every night cell (up to −0.017 without it); the Mahalanobis soft weight
recovers the fog frames the switch misses (−0.003 without it); and glare/day
is the one cell where fusion beats *both* single sensors (+0.0022 over VIS,
CI excluding zero). The retired parts — soft brightness term, score
calibration, top-k caps — were measured to do nothing.

---

## A. Blocking the full-scale launch

- [ ] **A-1. Stage-3 per-class ship AP** (`scripts/perclass_ap.py` on
  `runs/screen3/s3_p2feat_640_b10/weights/best.pt`). GPU, minutes — **blocked
  until `s4_vis_rect` finishes** (was 20/25 epochs on 2026-08-20). Decides the
  IR class set by the pre-registered rule: **adopt nc=1 iff ship-AP delta
  (last-5 mean) > +0.003** vs `s4_ir_shiponly`'s 0.1272. If adopted, remap the
  IR ship class index to match the VIS label space before fusion.
- [ ] **A-2. Laksh sign-off** (architecture-final §8):
  - [ ] matrix stays on `yolo26s`; `26m`/full-res becomes one follow-up arm
        (narrows D25)
  - [ ] ensemble M=5 vs M=3, and DGX MIG confirmed as host (OQ-12)
  - [ ] the publication reframing away from "uncertainty-gated fusion wins"
        (architecture-final §5)
- [ ] **A-3. `s4_vis_rect` readout.** Exploratory only — it carries the
  shuffle-off confound (`rect=True` forces `shuffle=False`). Decide: run a
  shuffle-controlled non-rect control at 896, or drop the rect line entirely.
  It does NOT gate the launch.

## B. Cheap CPU, before publication

- [ ] **B-1. Day-only Mahalanobis reference refit** (~20 min). Rebuild the
  reference set from daylight frames only, re-measure D on pohang01.
  Falsification test 1: if this alone fixes night, the story simplifies to
  "mis-composed reference set" and the photometric term becomes a redundancy
  check. Needed for the limitations section either way.
- [ ] **B-2. Leave-one-fit-run-out on `mu_b`** (~1 h) [was §9.4-20]. One run
  (pohang03, p05 min 21.0) defines half the margin rule. If dropping it moves
  `mu_b` a lot, the rule needs a floor tied to the clean distribution.
- [ ] **B-3. Fixed-GT risk–coverage (or AURC) for `R_sys`** (1–2 h) [was 0.6].
  The recorded non-monotonicity compares mAP over shifting frame subsets;
  redo against a common denominator before recording abstain as a negative.
- [ ] **B-4. Dedup-threshold sweep** (1–2 h) [was 0.4]. **No union-label
  number may be quoted until this runs.** §5.3 all but answers it (only 0.11%
  of VIS boxes reach IoU 0.85 with an IR box), so expect the +76.8% union box
  count to collapse to double-counted objects.

## C. The full-scale matrix itself (GPU; recipe frozen in D28/D29)

- [ ] **C-1. Training matrix** — per modality: Gaussian σ (1) + MC-Dropout
  (1, D19) + ensemble seed replicates (M per A-2) = 7 runs × 2 modalities =
  **14 runs** at 100 epochs, patience 20. IR: `yolo26s-p2feat` @640, min–max
  8-bit map, stride 1, class set per A-1. VIS: `yolo26s` @640 on the prepared
  tree, stride 2. Launch through `resolve_gpu_python()` — never
  `sys.executable` (D30).
- [ ] **C-2. Caches + constants refits** on the new checkpoints, by the same
  pre-registered rules: D-6 ladder for `mu_d`/`tau`, margin rule for `mu_b`,
  pohang01 held out of every fit, tune/report corruption seeds disjoint (D23).
  **B-1's outcome decides the reference-set composition (day-only vs current).**
- [ ] **C-3. σ calibration check** [was §9.1-5]. Predicted σ vs realised
  localisation error on matched detections. Precondition for reporting
  σ-weighted WBF as anything but implemented-and-blocked.
- [ ] **C-4. Table 2** — `evaluate_uq.py` over Gaussian / DFL-row (VIS-family
  only) / MC / ensemble caches, clean + corrupted. This is now the paper's
  load-bearing UQ table.
- [ ] **C-5. Per-baseline fusion rows** (CPU, cheap once caches exist): the
  same frozen fusion layer with MC / ensemble uncertainty as the soft signal.
  Expected result: the fusion conclusions are UQ-source-invariant — a
  supporting result for the selection framing.
- [ ] **C-6. Regenerate Table 3 + `final_system.md`** on the full-scale
  checkpoints (the finalized defaults produce the right system; the 08-19
  variant needs `--bright-soft --capability-runs all --veto-dilate 1`).

## D. Optional performance levers (not blocking, in value order)

- [ ] **D-1. IR empty-frame ablation** [was §9.3-15]: pohang01 is 93.1% empty
  after the night-box filter (19.5% of VIS train). Check help/hurt on one arm.
- [ ] **D-2. `yolo26m` scale-check arm** — one Gaussian run after the matrix
  (per A-2 sign-off; Phase 1 measured +0.02 for 26s→26m on VIS).
- [ ] **D-3. Time-varying registration correction** (A2 resurrected) — only
  worth it if σ-weighted WBF should ever matter; a per-run constant is
  measured insufficient (within-run drift up to 10 px). Otherwise report the
  0.11%-overlap concatenation finding as a limitation.
- [ ] **D-4. Small-object augmentation tuning / longer patience** [were
  B6/B7]. Screen on one arm only if the matrix under-delivers.
- [ ] **D-5. OQ-11 (full-res / imgsz 1280 VIS)** — excluded from the matrix;
  revisit only as a follow-up arm with D-2.

## E. Evaluation integrity, standing (carry into the manuscript)

- [ ] Held-out corruption family: fit the ladder constants on 5 families,
  report the 6th.
- [ ] Report per-class AP everywhere; the class-set decision (A-1) is
  rule-governed and precedes the results.
- [ ] Fog severity 2 is "sensor destroyed" (VIS 0.0012), not "degraded" —
  the interesting range is severity 1.
- [ ] State up front: paired subset over-weights pohang01 2.5× vs full val
  (never compare to Phase 1/2 numbers), and night frames are in val by design
  (`filter_night_boxes.py` touches train only).
- [ ] pohang04 has no IR, so a second held-out run exists only for VIS-side
  claims — scope the generalisation statement accordingly (n=1 held-out run
  for fusion).

## F. Housekeeping

- [ ] `.gitignore` negation `!runs/eval/*.md` (+ `.json`) so the small result
  files the docs cite survive a fresh clone (record §10's suggestion — still
  not done).
- [ ] When A-1/A-2 resolve: write the full-scale queue spec
  (`runs/queue_full/queue.json`) and a HOW_TO_RUN §6 for the matrix.

---

**Done and closed since the 08-19 list** (do not re-run): photometric soft
term (retired — veto-only), veto hysteresis (adopted dilate-15), run-disjoint
capability prior (adopted), gated-vs-VIS bootstrap (glare/day is a real win,
clean/day a tie), soft-weight ablation (Mahalanobis kept — fog/night −0.0028
if removed), score calibration (null), top-k truncation (null), CLAHE (null),
IR rect (confounded, rejected), imgsz 960 IR (noise), A2 per-run constant
(invalidated), GPU interpreter trap (pinned, D30).
