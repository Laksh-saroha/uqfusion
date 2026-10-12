# Pre-registration: three changes that retrain nothing, development (2026-10-10)

Asked: "try something drastic, anything that does not require retraining the backbone". Written
before any score of these systems on VIS draws 945–947 / IR draws 955–957 exists. Those draws were
still being built when this was written (`runs/v2_fresh_draws_945_947.log`), and LLF-2
(`docs/prereg-llf2-2026-10-10.md`) is registered on the same draws. Every system below is judged
by its own rule; none is chosen among the others after the fact, and the report lists all of them.

What was seen before writing. One smoke run of E1 on draw 941, system 0 only, IR noise s2, three
VIS conditions (`scripts/seed_ensemble.py`, 10:35 PM), which showed the mechanics run and that
five-checkpoint merging raises clean-day VIS ship AP (0.3507 → 0.3689 for that system). E2's
methods were timed on two frames. Nothing was seen on draws 945–947.

## E1. Checkpoint ensembles (`scripts/seed_ensemble.py`)

The five Phase 3 systems are five independent trainings of the same VIS and IR detectors. An
ensemble stream merges, frame by frame, the records of several checkpoints of one modality with
plain WBF (IoU 0.55, equal weights, no σ: the arm the class-set audit found positive for two
checkpoints of one model; not tuned here). The IR stream keeps its shipped detector-side NMS (0.70)
per checkpoint before the merge. The shipped gate (`crossmodal26m`) runs on the ensemble streams
unchanged; its capability prior is recomputed on the ensemble's clean streams over the same frames.

* **Arms.** `single` (system k as shipped), `ens2` (checkpoints {k, k+1 mod 5} per modality,
  twice the detector compute), `ens5` (all five, five times the compute). Every comparison is paired
  by system k. Under `ens5` the five systems share their streams and differ only in calibration
  that is identical across k, so its interval is the single systems' spread, not the ensemble's.
* **Cells.** VIS {clean, fog, fog s1, low light, glare} × IR {clean, fog s1–s3, noise s2–s3} ×
  {day, night} = 60, ship AP, per system the mean over draws 945–947, 95% t-interval over the five
  systems (df 4).
* **E1-P (primary, per arm).** Δ fused = arm − single. **IMPROVES** if no cell has mean Δ ≤ −0.0014
  with the interval's upper bound below zero, and at least one cell has mean Δ ≥ +0.0060 with the
  lower bound above zero. Counts are also reported at 0.0008, 0.0024 and 0.0060.
* **E1-S (secondary).** Δ VIS and Δ IR (single-stream ship AP), and the Table 3b claim
  (fused − max(VIS, IR) fails at ≤ −0.0060 with the interval below zero) counted for each arm.
  The claim is about the gate, so an ensemble that lifts everything can leave it unchanged.

## E2. Restoration before detection (`scripts/restore.py`)

Each corrupted frame is replayed exactly as the cache builder replays it, passed through one
classical, image-only restoration, and detected by the same checkpoints at the same settings
(imgsz 640, conf 0.001, batch 1). No restoration reads the corruption's parameters, depth maps or
labels. The gate's frame statistics stay those of the corrupted frame.

* **Candidates.** VIS: `ae` (linear-light auto-exposure to mean 0.18, brighten only, gain ≤ 8),
  `clahe` (L*, clip 2, 8×8), `dcp` (dark-channel-prior dehaze, He et al. 2009: patch 15, ω 0.95,
  t ≥ 0.15, guided-filter refinement r 40, ε 1e-3), `bil_dcp` (bilateral 5/25/5, then `dcp`).
  IR: `nlm` (non-local means, h = 1.2 × the Immerkaer noise sd clipped to [3, 40]), `median3`,
  `clahe` (clip 2, 8×8), `nlm_clahe`.
* **Selection (declared before the development run).** Draw 941 / 951 (the descriptive draw, not
  a verdict draw anywhere), every fourth paired frame (558), systems 0–2, `restore.py dev`. Per
  modality the chosen method maximises the mean single-stream Δ ship AP (restored − raw) over that
  modality's corrupted cells (VIS: fog, fog s1, low light, glare; IR: fog s1–s3, noise s2–s3; day
  and night each a cell), among methods whose clean-cell Δ (day and night) is ≥ −0.0014 when
  applied by the gate below. If no method passes for a modality, that modality is not restored and
  E2 reports it as rejected.
* **Gate.** A frame of a stream is restored only where the shipped gate's own degradation
  statistic for that stream fires: VIS `veil OR dark` (`grad_gini` below its threshold, or the
  brightness statistic below `mu_b`); IR `ir_d2 > ir_bound` (the IR merge-health bound). Otherwise
  the raw detections are used. `always` (every frame restored) is reported beside it.
* **Evaluation.** Draws 945–947, the same 60 cells, five systems, single-stream and fused
  (`crossmodal26m` on the restored streams). **E2-P:** Δ fused = restored − raw, same IMPROVES rule
  as E1-P. **E2-S:** single-stream Δ, the claim count, and `always` beside `gated`.

## E3. The shipped constants under v2 (`scripts/v2_constants_audit.py`)

Descriptive, on development draws 942–944 (already used for the degraded-IR repairs and the LLF
verdict): sixteen single-constant moves (capability ratio ×1/2/8/16, merge IoU 0.55/0.70/0.95,
support off/0.25/1.0, support IoU 0.20/0.50, weak-IR fallback off, IR merge veto on, veil not
night-conditional, and the veto switched off entirely), each against shipped on the 60 cells.
**No constant changes on this evidence.** A move that is better on at least one cell at 0.0060 and
worse on none at 0.0014 there may be registered for draws 945–947 in an amendment written before
those draws are scored for it. The night axis is closed by registration ("no V4"); the veto-off row
prices the gate and proposes nothing.

## What a pass means

Development results on frames that every system here has been developed on; fresh corruption draws
protect against fitting the corruption noise, not against fitting the frames. A pass makes a change
a recommended configuration, reported as such. The shipped system, Table 3b and the pohang04 look
stay as they are. **No pohang04 frame is read or scored.**

## Amendment 1 (11:00 PM, same day, before any score on draws 945–947)

Seen since the registration: a smoke run of E3 (draw 942, system 0, IR fog s3, VIS clean and low
light, four arms). With the IR stream fogged the shipped gate scored 0.0063 on clean night against
VIS 0.2431, and 0.0548 on low-light day against VIS 0.1658; switching the veto off gave 0.2484 and
0.1651. This is the known weak-IR-fallback and night-veto cost (README §8.6), and it motivates two
more systems. Neither is a V4 of the night axis: `crossmodal26m` stays frozen and is the comparator.

* **E4. Union (no gate).** Same streams as shipped, the veto switched off everywhere (`veto=None`):
  every frame merges both streams with the shipped weights, merge IoU and support term. A separate,
  gate-free system.
* **E5. Everything that retrains nothing, together.** Five-checkpoint ensemble streams (E1 `ens5`),
  each checkpoint's stream restored where the gate's degradation statistic fires (E2 `gated`, with
  the methods E2's declared rule selects), merged, and fused without a gate (E4).

Both are scored with `scripts/seed_ensemble.py` on draws 945–947, 60 cells, five systems, against
`single` (shipped), with E1-P's IMPROVES rule (no cell worse at 0.0014 with the interval below zero,
at least one better at 0.0060 with the interval above zero), and with the claim count. Also reported,
to decompose E5: `ens5` with no gate, and E5 with the shipped gate. If E2 rejects both modalities,
E5 is `ens5` with no gate.

## Amendment 2 (11:20 PM, same day, before any score on draws 945–947)

Asked: "try some more drastic things, maybe some more ways for gated fusion". Two more registered
systems, both on draws 945–947, 60 cells, five systems, nothing tuned after this point.

* **E6. Gate zoo** (`scripts/gate_zoo.py`). Three new gates beside `shipped` and E4's `union`:
  * `selfconf`: output-space health. A stream is vetoed on a frame iff its most confident ship box
    scores < 0.10 while the other stream's scores ≥ 0.25 (fixed round numbers, not tuned).
  * `calunion`: each stream's ship confidences mapped to P(TP at IoU 0.5) by isotonic calibrators
    fitted per system, per stream and per health state (VIS `veil OR dark`; IR `ir_d2 > ir_bound`) on
    draw 941 / 951, leave one run out; the two calibrated streams are concatenated (no veto, no WBF).
  * `proben`: `calunion` plus Bayesian fusion of cross-stream pairs (ProbEn, Chen et al. 2022): VIS and
    IR ship boxes matched greedily at IoU ≥ 0.30 become one box with VIS coordinates and
    logit p = logit p_VIS + logit p_IR − logit prior (prior: the calibration rows' TP rate).
  Each arm against `shipped` by E1-P's IMPROVES rule, with the claim count.
* **E7. Deep-feature trust** (`scripts/deep_gate.py`). LLF-2's learner (same candidates, credit target,
  training rows of draw 941, monotone constraints, LORO folds, night-block secondary) given 32 more
  features: 16 whitened principal components of each frozen backbone's 1,280-d frame embedding, the
  PCA fitted per checkpoint on its clean TRAIN frames. Fitted per system with XGBoost on the GPU
  (500 trees, learning rate 0.05, depth 8, min child weight 200, λ 1, seed 0). Judged by LLF-2's P1
  and P2.

Multiplicity, stated once: E1–E7 and LLF-2 are eleven-odd systems scored on the same fresh draws.
Each verdict is its own rule's; the report does not pick a winner among them as a confirmatory result.

## Amendment 3 (11:30 PM, same day, before any score on draws 945–947)

* **E8. Test-time augmentation.** Each checkpoint also detects on the horizontally flipped,
  unrestored frame (`restore.py build --flip`, in the same corruption replay as E2); boxes are
  mapped back (checked: IoU 0.92–0.99 with the unflipped detections of the same frames) and the two
  views are merged with the checkpoint views by the same plain WBF (IoU 0.55). Arms: `tta` (system k,
  both views, shipped gate) and `full_tta` (E5 with both views of every checkpoint: ten lists per
  stream). Same IMPROVES rule against `single`, scored with `scripts/seed_ensemble.py`.

Seen since Amendment 2: the E1/E4 smoke on draw 941 (descriptive draw), IR fog s3, VIS clean and low
light, all five systems: shipped / union / ens5+union fused 0.3389 / 0.3389 / 0.3586 on clean day,
0.0070 / 0.2556 / 0.2747 on clean night, 0.0577 / 0.1684 / 0.1980 on low-light day. Nothing on draws
945–947.

## Amendment 4 (1:25 AM, 2026-10-11, before any score of these arms on draws 945–947)

* **E9. Tiered concatenation and demotion instead of deletion.** The scorer matches greedily in
  confidence order (`matching.tp_matrix`) and takes the precision envelope as a running max from the
  tail (`apmetrics._envelope`). Boxes appended *below every existing box* therefore cannot steal a
  match and cannot lower any earlier precision, so appending a tail never lowers AP (up to an
  `np.interp` tie when a 101-point grid recall equals a duplicated recall value). Two gates are built
  on that, and two compositions:
  * `lex` (`scripts/gate_zoo.py`): every VIS ship box ranks above every IR ship box (VIS `1 + c`, IR
    `0.999 c`), no merge, no parameter, no gate. Prediction, structural: fused ≥ VIS in every cell.
  * `demote` (`scripts/gate_zoo.py`): the shipped `crossmodal26m` output as the first tier, and the
    ship boxes of whichever stream it vetoed on a frame as the tail, instead of deleting them.
    Prediction, structural: fused ≥ shipped in every cell.
  * `ens5_lex` and `full_tta_lex` (`scripts/seed_ensemble.py`): `lex` on the five-checkpoint streams
    (E1) and on the five-checkpoint, gated-restored, flip-augmented streams (E5 + E8).
  Judged exactly as E4–E8: the IMPROVES rule against shipped (`single` for the ensemble arms) and the
  claim test. A cell where `lex` < VIS or `demote` < shipped by more than 1e-4 is reported as a bug,
  not as a result.

Seen before writing, all on the smoke draw 941 (descriptive, five systems, IR clean and IR fog s3; the
table is in `docs/eval/no_retrain/smoke_941_lex.md`). With IR clean, union beats `lex` in every cell
(clean day 0.3566 vs 0.3500, clean night 0.2895 vs 0.2725). With IR fog s3, `lex` ≥ VIS in all ten
cells and union does not (clean day 0.3389 vs VIS 0.3485). `demote` ≥ shipped in all twenty cells
(night 0.1079 vs 0.0687 with IR clean). A first `demote` run clipped fused scores to [0, 1] and scored
*below* shipped; the clip tied boxes the support bonus had pushed above 1, and was removed. Two more
arms were tried on 941 and are **not** registered: `lex_sc` (IR first where VIS's top score < 0.10 and
IR's ≥ 0.25) lost to `lex` in every fog cell, and `udemote` (union where IR is healthy, VIS first and
IR as the tail where the IR-health flag fires) lost to both union and `lex` under IR fog s1 (clean day
0.3225 vs 0.3553 / 0.3514). Seen on draws 945–947 before writing: the first two E7 jobs (draw 945, IR
clean, systems 0 and 1: VIS, IR, shipped and E7 ship AP), and the restoration selection
(`runs/restore/selection.json`: VIS none, IR `nlm`), which reads draw 941 only. Nothing else on draws
945–947.

*Note added 1:30 AM, after launch, changing nothing registered:* the VIS-clean × IR-clean cells do not
depend on the draw (no corruption is replayed), so the draw-941 smoke values for those two cells, here
and in Amendment 3, **are** the draw 945–947 values. Every arm in Amendments 1–4 was therefore chosen
having seen its own clean × clean day and night numbers; the other 58 cells were unseen.
