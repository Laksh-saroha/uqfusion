# Pre-registration: learned late fusion (LLF), development (2026-10-10)

Written before any LLF model was trained and before any LLF number existed. The evaluation IR caches
(VIS draws 942–944 with IR 952–954 degraded) were still being built.

## 1. Why a new decision layer

The shipped decision layer of `crossmodal26m` is a hand-written gate: an IR night vote, a darkness
and veil veto, a weak-IR fallback, a cross-modal support bonus. It has been rewritten six times, each
time forced by a measured failure of the previous version (paper §6.3, Table 3a). On the Phase 3
systems under corruption v2 it holds the claim fused ≥ max(VIS, IR) on 5 of 8 Table 3b cells. It
discards a working VIS stream at night (−0.1847 on clean night, §6.6), and under a degraded IR the
fallback and the support bonus cost up to 0.16 (§6.7). The night axis of that rule was closed by
registration (`docs/night-veto-axis-closed-2026-09-04.md`: no V4). This registration does not reopen
it: LLF is not a fourth clause on the veto. It replaces the decision layer with a separate, learned
system, and it is reported as such.

## 2. The system

`scripts/llf.py`. Late fusion with one learned score per box, in the spirit of CLOCs (Pang et al.,
IROS 2020):

* **Candidates.** All VIS ship boxes, plus the top 100 IR boxes by confidence, warped into the VIS
  canvas by the same per-frame homography the shipped fusion uses. VIS buoy boxes pass through
  unchanged. No coordinate is changed and no box is merged.
* **Features** (34, fixed here). *Box:* stream, confidence, log confidence, rank in its stream,
  confidence relative to its stream's best, log area, aspect, centre x/y, σ normalised by box size,
  the best same-stream IoU, and whether a more confident same-stream box overlaps at IoU ≥ 0.5.
  *Cross-stream:* the best IoU with the other stream, that box's confidence, the strongest support at
  IoU ≥ 0.30, the count of such boxes, the centre distance to the nearest other-stream box in units of
  the box's own size, and whether a more confident other-stream box overlaps at IoU ≥ 0.30.
  *Frame health:* VIS p05, p50 and std, log Laplacian variance, `lap_over_var`, `grad_gini`, edge
  density, VIS health `q_vis`, IR p05, p50 and std, log IR novelty `ir_d2`, box counts at confidence
  ≥ 0.25 per stream, and the mean of the top three confidences per stream. No run, frame index,
  condition label or corruption parameter is a feature.
* **Target.** AP50-95 credit under a one-to-one assignment. For each ground-truth ship, the
  candidate from either stream with the highest IoU, if that IoU is ≥ 0.5, earns the fraction of the ten
  thresholds 0.50–0.95 it clears. Every other candidate earns 0. Ground truth is the VIS labels, as
  in every other cell.
* **Model.** `HistGradientBoostingRegressor` (500 iterations, learning rate 0.05, 63 leaves, minimum
  leaf 200, L2 1.0, seed 0), monotone increasing in confidence. These values are fixed here; nothing is
  tuned.
* **Score.** The prediction, clipped at 1e-6, plus 1e-7 × confidence to break ties.

## 3. Data and folds

* **Training rows:** VIS draw 941 / IR draw 951 caches of the five Phase 3 systems, all 30 cells (VIS
  clean, fog, fog_s1, glare, lowlight × IR clean, fog s1–s3, noise s2–s3), development paired frames.
  20% of (system, cell, frame) units are kept, chosen by a hash.
* **Evaluation:** VIS draws 942–944 with IR 952–954, every frame, all 60 cells (the 30 cells × day /
  night). Training and evaluation share no corruption draw.
* **Primary folds, leave-one-run-out:** a frame of run R is scored by the model trained without R.
  The night run pohang01 is therefore scored by a model that never saw a real night frame. Day cells
  are scored on all 1,200 day frames, each frame out of fold.
* **Secondary analyses (descriptive):**
  1. night seen in training: pohang01 is split into four capture-order blocks, and each block is
     scored by a model trained on the day runs and the other three blocks;
  2. leave one corruption family out: VIS fog, VIS low light, VIS glare, IR fog or IR noise is absent
     from training as well as the test run, and only the cells of that family are scored;
  3. ablations: without the frame-health features, and without the cross-stream features.

## 4. Statistic and decision

Per system, the draw mean over 942–944 of ship AP (local AP), day and night. Paired by system;
95% t-interval over the five systems (df 4). The same claim as Table 3b: fused − max(VIS, IR) fails at
≤ −0.0060 with the interval below zero.

* **P1:** LLF holds the claim on **all 60 cells**.
* **P2:** LLF is never worse than `crossmodal26m` beyond 0.0060 (mean ≤ −0.0060 with the interval
  below zero) and is better on at least one cell (mean ≥ +0.0060, interval above zero). Counts are
  also reported at 0.0014.

The headline is **P1 and P2 together**. If both pass, LLF is reported as a development solution to
the gated-fusion problem. If either fails, it is reported as the failure it is, with the cells.

The mechanics check runs on every cell before any number is used. The VIS candidates at their own
confidence, passed through the same record builder and AP path with IR dropped, must reproduce
VIS-only ship AP to 1e-12, or the run stops.

## 5. What this can and cannot show

Development only. No untouched data remains: pohang04 has been exposed twice and is not read here.
The design was fixed by someone who has seen every earlier development result, on the same frames.
Leave-one-run-out keeps the scored run out of the fit; it does not make those frames unseen by the
designer. The night run is one run; the primary night cells measure generalisation from day frames
and synthetic low light to real night. Corruptions are v2 synthetic, so held-out families test
generalisation to an unseen synthetic family, not to real weather. **No pohang04 frame is read or
scored.**

Output: `docs/eval/llf/llf_2026-10-10.{md,json}`; models and rows under `runs/llf/` (untracked).

## Amendment 1, 2026-10-10, before any LLF evaluation number existed

**A second arm, LLF-Net** (`scripts/llf_net.py`). Same candidates, same 34 features, same target,
the same training rows, and the same leave-one-run-out folds and evaluation draws as LLF. Instead of
scoring each box from features of its own pairs, a set transformer reads every candidate in the frame
(VIS and IR together) and scores each one with attention over all the others. It can therefore learn
duplicate suppression and cross-stream trust jointly rather than through hand-built pair features.

* Features are standardised on each fold's training rows and clipped to ±10.
* Architecture: linear 34 → 128, three pre-norm transformer encoder layers (4 heads, feed-forward
  256, dropout 0.1), no positional encoding, then a linear head and a sigmoid.
* Loss: binary cross-entropy on the [0, 1] credit target.
* Training: AdamW (lr 3e-4, weight decay 1e-4), batches of 64 frames, 15 epochs, cosine schedule,
  mixed precision, seed 0, on the GPU. These values are fixed here; nothing is tuned and there is no
  early stopping.
* Unlike LLF, it is **not** monotone in confidence.
* Only the primary folds are run. LLF-Net is reported with its own P1 and P2. The headline remains
  LLF, the arm registered first; if the two disagree, both are reported.

The model-fitting order changed once before this amendment and before any evaluation. The first LLF
fits were stopped to try a GPU library and then resumed unchanged with the registered sklearn model:
25 of 36 were already fitted, and none was refitted or replaced.
