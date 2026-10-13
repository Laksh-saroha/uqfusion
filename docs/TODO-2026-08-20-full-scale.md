# TODO — full-scale training and publication (post-finalization)

**Created 2026-08-20**, after the architecture finalization
([`architecture-final-2026-08-20.md`](architecture-final-2026-08-20.md), decisions D27–D30).
**Supersedes the open items of [`TODO-improvements.md`](TODO-improvements.md)** — that file remains
the 2026-08-19 record; everything still outstanding from it is carried here with its old ID in
brackets.

The frozen architecture in one line: *per-modality yolo26s σ-head detectors (IR = p2feat @640) →
WBF in the VIS frame, weighted by run-disjoint capability × soft reliability (Mahalanobis × r_box)
— with VIS thrown out of the merge entirely whenever its content brightness says no photons arrived
(p05 < mu_b, dilated over 15 frames in capture order).*

## The architecture, in plain language

**Two detectors, one merge rule, one kill switch.**

- **Two detectors.** VIS and IR each get their own yolo26s + Gaussian σ head (IR with the p2feat
  neck — the one screening change that clearly mattered, +34% at night). Each also scores how
  *strange* the current frame looks against its own clean training data (Mahalanobis on pooled neck
  features).
- **One merge rule.** IR boxes are mapped into the VIS image (per-run homography) and merged by WBF
  (`iou_thr` 0.85, σ-weighted coordinates). Each sensor's weight ≈ *how good it is on a clear day*
  (capability prior, computed on the daytime runs only) × *how normal the frame looks to it*
  (Mahalanobis × r_box).
- **One kill switch.** Before merging, ask of the VIS frame: *did any light reach the sensor?* The
  5th-percentile brightness of the image content — a statistic glare and fog cannot fake upward —
  is compared to a threshold placed in the empty gap between dark and lit frames (`mu_b` = 10.5,
  margin rule, no optimizer). Below it, VIS **leaves the merge entirely** rather than being
  down-weighted. The switch is sticky: if any frame in the last 15 was dark, VIS stays out (darkness
  is a property of a stretch of recording, not one frame). IR is never vetoed — a dark thermal frame
  is cold water, which is IR's job.

**Why a switch instead of weights:** down-weighting cannot save a blind stream — weights
renormalize, WBF rescales scores instead of dropping boxes, and mAP is rank-based — so a blind
camera drags the working one down at any weight. The governing rule: *a signal gets switch authority
only if it is monotone in sensor health.* Brightness is; the Mahalanobis score is not (glare looks
strange while the detector still works), so it may only nudge weights.

**What each piece earns** (`runs/eval/final_system.md`): the veto carries every night cell (up to
−0.017 without it); the Mahalanobis soft weight recovers the fog frames the switch misses (−0.003
without it); glare/day is the one cell where fusion beats *both* single sensors (+0.0022 over VIS,
CI excluding zero). The retired parts — soft brightness term, score calibration, top-k caps — were
measured to do nothing.

---

## A. Blocking the full-scale launch

- [x] **A-1. The ship-AP measurement** — the one number that decides the IR class set. GPU, one val
  pass, minutes — **was blocked on the GPU being busy with `s4_vis_rect`; that run is now stopped
  (A-3, 23/25 epochs, not resumed), so this ran.**

  ```bash
  "C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe" scripts/perclass_ap.py runs/screen3/s3_p2feat_640_b10
  ```

  (GPU interpreter, not `.venv` — D30. Prints `all | ship | buoy` for `best.pt` on
  `data_ir_stride2.yaml`.)

  **What it decides.** `s4_ir_shiponly` trained with nc=1, so its mAP *is* ship AP: **best 0.1359 /
  last-5 0.1272**. The stage-3 2-class checkpoint's headline (best 0.0659 / last-5 0.0607) is
  macro-averaged with a ~0.0002 buoy class, so its ship AP is implied ≈ 2×mAP ≈ **0.121** — but
  implied is not measured, and this pass measures it.

  **Pre-registered rule (D28): adopt nc=1 for IR iff**
  `ship AP(s4_ir_shiponly) − ship AP(s3_p2feat_640_b10) > +0.003` (the CLAHE-null magnitude is the
  noise yardstick). Compare best-vs-best. Expected delta if the implied estimate holds: ≈ +0.006 →
  adopt; if the measured stage-3 ship AP comes in ≥ 0.133, keep nc=2.

  **Result (`scripts/perclass_ap.py`, best.pt): `s3_p2feat_640_b10` all 0.06584 | ship 0.13149 |
  buoy 0.00019.** Delta vs `s4_ir_shiponly` ship AP (best 0.1359, also nc=1/best.pt) = **+0.00441 >
  +0.003**, and measured stage-3 ship AP (0.13149) came in below the 0.133 keep-nc=2 line — both
  framings of the rule agree. **Decision: adopt nc=1 for IR.**

  **Now due (were gated on this):** remap the IR ship class index to match the VIS label space
  before fusion (IR then emits no buoys — which it effectively never did, buoy AP 0.00022), and
  regenerate the IR data yamls for the C-1 matrix from the ship-only lists
  (`runs/derived/data_ir_shiponly*.yaml`, stride 1).

- [ ] **A-2. Laksh sign-off** (architecture-final §8):
  - [x] **Backbone: `yolo26m` (not `26s`) at imgsz 640 for the matrix.** `26m`/full-res together
        stays one follow-up arm afterward, as originally recommended — just with `26m` now also the
        matrix backbone. Full-res on the laptop was tested and rejected for the matrix itself.
  - [x] **Ensemble M=5**, confirmed.
  - [x] **Host: laptop, not DGX** — "no option to run on DGX host." This reverses the architecture
        doc's explicit recommendation (`DGX MIG slice is the only viable host`, OQ-12), written
        because of a specific measured risk: Windows/WDDM pages an over-large batch instead of
        raising OOM, so a bad batch choice finishes each epoch and silently costs 5–17× the
        wall-clock instead of erroring. Mitigated, not eliminated, by measuring real batch ceilings.
  - [ ] the publication reframing away from "uncertainty-gated fusion wins" (architecture-final §5)
        — **still open**.

  **Batch sizes, measured on the laptop (`scripts/tune_batch.py`, real data, real model, ranked by
  throughput not just "did it run"):**
  - **IR: batch 10.** Not a memory ceiling — the small-object screen directly tested 16 vs 10 and 10
    won on accuracy (+7.0% ship AP, `screen-small-object-2026-08-19.md` §2). Unchanged.
  - **VIS @ 640: batch 16.** Clean at 50.1 img/s, 9.37 GB torch / 10.53 GB driver peak (1.5 GB
    headroom). Batch 24 pages (torch reserved 13.53 GB past the 12 GB card, throughput collapsed 5×
    to 10.1 img/s); batch 32 confirmed paging live (17.7–17.8 GB, 6–7 s/it) and was killed.
    **Chosen for throughput, not accuracy** — unlike IR's batch 10, nobody has screened whether 16
    is mAP-optimal for VIS. Decision: **keep 16**, accepted as an unvalidated limitation rather than
    spend hours on an accuracy screen (Laksh's call, 2026-08-20).
  - **VIS @ full-res 1280 (follow-up arm only): batch 4.** Clean at 12.2 img/s, 10.21 GB torch /
    11.6 GB driver peak — only 0.4 GB headroom. Batch 6 already pages (torch reserved 14.85 GB,
    throughput collapsed 7×); batch 8 confirmed paging live at the exact ~17 GB figure
    `handoff-2026-08-17.md` OQ-12 had estimated.

  **Wall-clock, VIS side, 7 runs (Gaussian + MC-Dropout + M=5 ensemble) sequential on one GPU, no
  DGX fallback:**

  | config | epoch time | central (~30 ep, early-stop precedent) | worst case (100 ep) |
  |---|---|---|---|
  | `26m` @ 640, batch 16 | ~19–20 min (train measured 16 min + estimated val) | **~3 days** | ~9.7 days |
  | `26m` @ 1280, batch 4 (follow-up arm) | ~34–36 min | ~5.1 days | ~17 days |

  Epoch-count is the weakest part of this estimate — borrowed from Phase 1's `yolo26m` early-stop
  precedent (epochs 26/29/33 at a different resolution/dataset), not measured at this config.
  Val-time is estimated, not directly timed by the probe (which only times train iterations).

- [x] **A-3. `s4_vis_rect` readout.** Exploratory only — it carries the shuffle-off confound
  (`rect=True` forces `shuffle=False`). It does NOT gate the launch.

  **Result (`docs/followup-analysis-2026-08-20.md` §8.2): dropped, on the partial run — no control
  run.** Stopped at 23/25 epochs by choice. mAP50-95 peaked at epoch 10 (0.24884) and declined
  through epoch 23 (last-5 mean 0.23125) — already past its best. Peak sits at/below the non-rect
  640 baseline (0.25049) and the late decline pushes it further negative, same direction as the IR
  rect arm's clean −0.0060 (same `shuffle=False` confound). No positive signal to justify a
  controlled 896 rerun. **VIS stays imgsz 640, non-rect, per the frozen C-1 recipe.**

## B. Cheap CPU, before publication

- [x] **B-1. Day-only Mahalanobis reference refit** (~20 min). Rebuild the reference set from
  daylight frames only, re-measure D on pohang01. Falsification test 1: if this alone fixes night,
  the story simplifies to "mis-composed reference set" and the photometric term becomes a redundancy
  check. Needed for the limitations section either way.

  **Result (`scripts/refit_maha_dayonly.py`, `runs/eval/x_maha_dayonly_refit.md`): FIXED,
  decisively.** Dropping the 973 train-cache frames with p05 < mu_b (10.5) — mostly pohang01 night
  frames, 771/782 of them — flips the inversion outright: D_night goes from 28.58 (below D_day
  30.80, i.e. night reads as *cleaner*) to **88.97** (nearly 3× D_day 30.86, correctly the
  strangest thing in the set) once the reference set no longer contains darkness. The mis-composed
  reference set was the whole story: the photometric veto is a redundancy check on a scorer that,
  correctly fit, would have caught the blind sensor on its own.

- [x] **B-2. Leave-one-fit-run-out on `mu_b`** (~1 h) [was §9.4-20]. One run (pohang03, p05 min
  21.0) defines half the margin rule.

  **Result (`scripts/loro_mu_b.py`, `runs/eval/x_loro_mu_b.md`): confirmed unstable.** Dropping
  pohang00 or pohang02 leaves `mu_b` untouched (still 10.500 — pohang03 sets the clean-min endpoint
  regardless). Dropping **pohang03** swings `mu_b` from 10.500 to **16.000** (+5.5, tau_b
  2.625→4.000) — exactly the run the TODO flagged. Downstream effect on guard/target cells is small
  (fog/night +0.0004, CI excluding zero; clean/day unmoved) because the veto is a hard binary switch
  and both mu_b values still separate day from dark cleanly, but the margin rule itself (bare
  min/max over 3 runs) is not self-stabilizing and needs a floor tied to the clean distribution
  (e.g. a percentile, not a min) before being called robust.

- [x] **B-3. Fixed-GT risk–coverage (or AURC) for `R_sys`** (1–2 h) [was 0.6]. The recorded
  non-monotonicity compares mAP over shifting frame subsets; redo against a common denominator
  before recording abstain as a negative.

  **Result (`scripts/eval_risk_coverage_fixed_gt.py`, `runs/eval/x_risk_coverage_fixed_gt.md`): the
  artifact was real, but so was the negative.** Fixing the GT denominator resolves the
  non-monotonicity on 3/4 conditions (0/4 were monotone under the old shifting-denominator method).
  But `R_sys`-ordered abstention still loses to a random abstention order on **every** condition
  (higher fixed-GT AURC than the 20-shuffle random control, all 4 cells) — `R_sys` is not merely
  noisy-looking due to a measurement bug, it is measurably worse than doing nothing. Abstain as a
  negative stands.

- [x] **B-4. Dedup-threshold sweep** (1–2 h) [was 0.4]. **No union-label number may be quoted until
  this runs.** §5.3 all but answers it (only 0.11% of VIS boxes reach IoU 0.85 with an IR box).

  **Result (`scripts/sweep_dedup_iou.py`, `runs/eval/x_dedup_iou_sweep.md`): confirmed.** Added-box
  count at dedup_iou 0.10 (3,049, +15.9%) is an 84% drop from dedup_iou 0.85 (18,628, +97.4%); only
  8.7% of IR GT boxes have literally zero overlap with any same-class VIS box in frame (the only
  threshold-independent, unambiguously-new additions). The recorded +76.8% (dedup_iou 0.5) sits in
  between and is mostly registration slop, not new objects. No single dedup_iou is self-evidently
  correct — the sweep, not one number, is what the paper can cite until a registration-accuracy
  argument picks a threshold.

## C. The full-scale matrix itself (GPU; recipe frozen in D28/D29)

- [ ] **C-1. Training matrix** — per modality: Gaussian σ (1) + MC-Dropout (1, D19) + ensemble seed
  replicates (M per A-2) = 7 runs × 2 modalities = **14 runs** at 100 epochs, patience 20. IR:
  `yolo26s-p2feat` @640, min–max 8-bit map, stride 1, class set per A-1. VIS: `yolo26s` @640 on the
  prepared tree, stride 2. Launch through `resolve_gpu_python()` — never `sys.executable` (D30).
- [ ] **C-2. Caches + constants refits** on the new checkpoints, by the same pre-registered rules:
  D-6 ladder for `mu_d`/`tau`, margin rule for `mu_b`, pohang01 held out of every fit, tune/report
  corruption seeds disjoint (D23). **B-1's outcome decides the reference-set composition.**
- [ ] **C-3. σ calibration check** [was §9.1-5]. Predicted σ vs realised localisation error on
  matched detections. Precondition for reporting σ-weighted WBF as anything but
  implemented-and-blocked.
- [ ] **C-4. Table 2** — `evaluate_uq.py` over Gaussian / DFL-row (VIS-family only) / MC / ensemble
  caches, clean + corrupted. Now the paper's load-bearing UQ table.
- [ ] **C-5. Per-baseline fusion rows** (CPU, cheap once caches exist): the same frozen fusion layer
  with MC / ensemble uncertainty as the soft signal. Expected: the fusion conclusions are
  UQ-source-invariant — a supporting result for the selection framing.
- [ ] **C-6. Regenerate Table 3 + `final_system.md`** on the full-scale checkpoints (the finalized
  defaults produce the right system; the 08-19 variant needs
  `--bright-soft --capability-runs all --veto-dilate 1`).

## D. Optional performance levers (not blocking, in value order)

- [ ] **D-1. IR empty-frame ablation** [was §9.3-15]: pohang01 is 93.1% empty after the night-box
  filter (19.5% of VIS train). Check help/hurt on one arm.
- [ ] **D-2. `yolo26m` scale-check arm** — one Gaussian run after the matrix (per A-2 sign-off;
  Phase 1 measured +0.02 for 26s→26m on VIS).
- [ ] **D-3. Time-varying registration correction** (A2 resurrected) — only worth it if σ-weighted
  WBF should ever matter; a per-run constant is measured insufficient (within-run drift up to
  10 px). Otherwise report the 0.11%-overlap concatenation finding as a limitation.
- [ ] **D-4. Small-object augmentation tuning / longer patience** [were B6/B7]. Screen on one arm
  only if the matrix under-delivers.
- [ ] **D-5. OQ-11 (full-res / imgsz 1280 VIS)** — excluded from the matrix; revisit only as a
  follow-up arm with D-2.
- [ ] **D-6. Close the `p2feat` missing cell** [screen §8 item 1] — run `p2feat` @ 640 @ batch 10,
  25 ep (~75 min). Decides whether the `p2feat`+960 ceiling (0.15440 ship AP) needs 960
  specifically or projects from batch alone (§2 projects ≈0.152 without it). Not blocking — IR imgsz
  is already frozen at 640 — but it validates that the frozen choice isn't leaving ceiling on the
  table before the number goes in the paper.

## E. Evaluation integrity, standing (carry into the manuscript)

- [ ] Held-out corruption family: fit the ladder constants on 5 families, report the 6th.
- [ ] Report per-class AP everywhere; the class-set decision (A-1) is rule-governed and precedes the
  results.
- [ ] Fog severity 2 is "sensor destroyed" (VIS 0.0012), not "degraded" — the interesting range is
  severity 1.
- [ ] State up front: the paired subset over-weights pohang01 2.5× vs full val (never compare to
  Phase 1/2 numbers), and night frames are in val by design (`filter_night_boxes.py` touches train
  only).
- [ ] pohang04 has no IR, so a second held-out run exists only for VIS-side claims — scope the
  generalisation statement accordingly (n=1 held-out run for fusion).

## F. Housekeeping

- [ ] `.gitignore` negation `!runs/eval/*.md` (+ `.json`) so the small result files the docs cite
  survive a fresh clone (record §10's suggestion — still not done).
- [ ] When A-1/A-2 resolve: write the full-scale queue spec (`runs/queue_full/queue.json`) and a
  HOW_TO_RUN §6 for the matrix.

---

**Done and closed since the 08-19 list** (do not re-run): photometric soft term (retired —
veto-only), veto hysteresis (adopted dilate-15), run-disjoint capability prior (adopted),
gated-vs-VIS bootstrap (glare/day is a real win, clean/day a tie), soft-weight ablation (Mahalanobis
kept — fog/night −0.0028 if removed), score calibration (null), top-k truncation (null), CLAHE
(null), IR rect (confounded, rejected), imgsz 960 IR (noise), A2 per-run constant (invalidated), GPU
interpreter trap (pinned, D30).
