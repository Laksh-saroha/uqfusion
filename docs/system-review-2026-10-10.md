# System review, 2026-10-10: every open problem, and what was done about it

Asked: "try to fix all the problems we are facing" and "go over the entire system and solve the gated
fusion problem; maybe try a new architecture". This file lists every open problem: the paper's §9
limitations, handoff-2026-10-10 §5 and the problems found along the way. For each it records what
was done, and where the answer is "nothing", why. Results are filled in as they are measured.

Status key: **FIXED** (measured, written up), **RUNNING**, **REGISTERED** (prereg written, not yet
scored), **DISCLOSED** (cannot be fixed with the data that exists; stated in the paper),
**DECLINED** (fixing it would break a registered commitment or spend data that does not exist).

## A. The gated-fusion problem (the decision layer)

| # | Problem | Status | What was done |
|---|---|---|---|
| A1 | Hand-written gate, six rewrites, each forced by a failure (§6.3) | ANSWERED (registered, development) | Nine replacements were registered and scored on fresh corruption draws (VIS 945–947, IR 955–957, 60 cells, five systems; §E below). No replacement meets every registered criterion. Two register a clean result each. `union` (no gate) IMPROVES on shipped, with 0 cells worse and 30 better, and holds the claim on 59/60 cells. `lex` (every VIS box ranked above every IR box) holds the claim on 60/60 cells, no interval below zero, by construction against VIS. The learned gates (LLF 33/60, LLF-2 52/60, deep-feature trust 60/60 only through wide intervals) and three new hand-designed gates all fail at least one criterion. |
| A2 | Night rule discards a working VIS (−0.1847 clean night) | DECLINED for `crossmodal26m`; solved by separate systems | The rule's night axis stays closed (no V4). Every registered replacement except `demote` removes the night cost: `union`, `lex`, LLF-2 and E7 are all +0.16 to +0.25 above shipped on clean-VIS night cells. |
| A3 | Weak-IR fallback vetoes low-light VIS under degraded IR (up to −0.16) | FIXED (registered, development) | R1, the fallback confirming with `concentrated` alone, was ADOPTED on draws 942–944: 0 cells harmed, 5 gaining ≥ 0.0060 (up to +0.1439 on low light with IR noise s3), claim 35/60 vs 32. `crossmodal26m_r` is R1 (`docs/eval/corruption_v2/degraded_ir_repairs.md`; README §8.11). Severity-3 low-light day still fails (−0.022). |
| A4 | Support bonus hurts under fogged IR (−0.0113) | REJECTED | R2, the bonus only where IR is merge-healthy, harmed 8 cells (up to −0.0158, clean VIS with IR noise s2) and gained none. The bonus stays as shipped. |
| A5 | Degraded-IR results rested on one corruption draw | FIXED (caches) | IR fog s1–s3 and noise s2–s3 caches were built for draws 942–944. |
| A6 | Fusion is concatenation, not consensus (0.05% of VIS boxes have an IR partner at IoU 0.85) | MEASURED | Cross-modal support is worth up to +0.0165 by day where IR is healthy (`union` − `lex`, clean VIS with IR noise s2) and costs 0.0099 where IR is fogged (clean VIS with IR fog s3). |
| A7 | Frame-level gates mix score scales (found 10-11) | MEASURED, descriptive | AP pools all frames into one ranking. Switching VIS-only output onto a *random* 75% of frames scores 0.3420, below VIS (0.3485) and union (0.3566), because the union scores the same boxes 1.13× higher (`docs/eval/no_retrain/switching_941.md`). Every frame-level gate pays this on top of its decisions. |

## B. Measurement

| # | Problem | Status | What was done |
|---|---|---|---|
| B1 | Throughput is detector-only, and two-stream is derived (§9 item 18) | MEASURED | `scripts/measure_e2e_fps.py`, 500 pairs, batch 1, clock unpinned: shipped end to end 9.1 FPS (109.9 ms: read 8.8, VIS det 18.9, IR det 19.4, image statistics 60.6, gate + WBF 2.2); with LLF 8.9 FPS; one detector's `predict()` 58.5 FPS on the same frames, so end to end is 0.31 of the derived 29.3. The NumPy image statistics, not the detectors, bound throughput (`docs/eval/e2e_fps_2026-10-11.md`). |
| B2 | Table 3a, Tables 4/4b/4c/5, Table L and the constants re-price are v1 records | partly DISCLOSED | R-D1 (Tables 4, 4b) and Stage 1 (Table 5) were re-run on v2 as robustness checks on 10-10. Table 3a is a history of rewrites on pre-restore checkpoints whose VIS was blind at night, so re-measuring it on v2 would describe a system nobody runs. |
| B3 | `make_corruption` silently defaulted to the broken v1 | FIXED | `version` is now required; every v1 caller states `version="v1"`. |
| B4 | Scoring was slow: AP matching is a pure-Python greedy loop, and `evaluate_systems` always scored four systems that callers re-score themselves | FIXED | The greedy match is now numba-compiled. It is identical on 40,176 real frames (502,998 TP flags) and 20,000 tie-heavy synthetic frames, and 112× faster (`scripts/check_tp_numba.py`; `UQF_NO_NUMBA=1` restores the reference loop). `evaluate_systems(compute_metrics=False)` skips the naive-fusion control and the four AP calls, leaving vetoes, weights and fused boxes unchanged. |

## C. Data and protocol (cannot be fixed with the data that exists)

| # | Problem | Status | Why |
|---|---|---|---|
| C1 | One held-out run, now spent (§9 item 1) | DISCLOSED | No untouched Pohang data remains, and new data would have to be collected. |
| C2 | One night run; in-sample night constants (items 4, 5) | DISCLOSED | A second night run does not exist. LLF's primary night cells train without it. |
| C4 | Glare rests on one real sun event (item 5; handoff §5.3) | DISCLOSED (searched) | All 28,881 TRAIN day frames scored by `calibrate_glare_ae.py` (`glare_ae_frames.json`) were searched for segments with more than 2% of the sky clipped. There are 12 segments of 20 or more frames. Inspected on before/at/peak contact sheets, 11 are overexposed sky: bridge transits (pohang00 5294, 5899, 7893; pohang02 5521, 5661, 6095, 8298), run starts at the pier (pohang00 6, pohang02 20, pohang03 582), and bright overcast (pohang02 17235). Only pohang00 18971 has the sun in frame, so TRAIN holds no second event to validate the fit. Development frames are where glare is scored, so calibrating on them would contaminate the cells. |
| C5 | Registration residual 3–6 px (item 7) | DECLINED | Fixing registration was measured not to help (`project-fusion-mechanism`); LLF reads misregistration through size-normalised centre distance instead. |
| C6 | Parity non-inferiority test not run (item 8) | DISCLOSED | Needs matched baseline retraining of five seeds, which is days of GPU time and changes no conclusion. |
| C7 | Mosaic-off construction (item 10), IR ladder power (item 12) | DISCLOSED | Both need retraining campaigns; neither changes a headline. |

## D. Other

| # | Problem | Status | What was done |
|---|---|---|---|
| D1 | 14 Table 1 runs are author-reported, with no continuation logs (item 13) | OPEN, Laksh's call | The local queue (`bench_ext_local.py`) can train them for real logs. They are the largest models (yolo11x, yolo12l/x, yolo26x, yolov10l/x, yolov9e); at their seed-0 timings this is about 80 GPU-hours. Not started: the author chose on 10-08 to record them as author-reported. |
| D2 | TMLR PDF is 47 pages | OPEN | Cut after the new sections land. |

## E. No-retraining changes, registered (`docs/prereg-no-retrain-2026-10-10.md`, Amendments 1–4)

All are scored on the fresh draws VIS 945–947 / IR 955–957, five Phase 3 systems, ship AP, 95%
t-intervals over systems. The rules: IMPROVES = no cell worse than shipped at 0.0014 with the interval
below zero, and at least one better at 0.0060 with the interval above zero. Claim = fused − max(VIS, IR)
failing at ≤ −0.0060 with the interval below zero. Shipped fails the claim on 28 of 60 cells.

| # | Change | Verdict | Numbers |
|---|---|---|---|
| E4 | `union`: the veto switched off | **IMPROVES**; claim fails 1/60 | 30 cells better at 0.0060, 0 worse at 0.0014. Only failure: clean VIS + IR fog s3, day, −0.0099 [−0.0131, −0.0068]. Mean fused 0.1544 vs shipped 0.0894 (`docs/eval/no_retrain/gate_zoo_fresh.md`). |
| E6 | `selfconf`, `calunion`, `proben` | not adopted | Each worse than shipped on 18–21 cells at 0.0014 (worst −0.0212, −0.0342, −0.0404, all day); claim fails 3, 9, 8. |
| E7 | Deep-feature trust (LLF-2 + 32 backbone PCs, XGBoost) | FAIL (P2) | P1 60/60, but 11 cells sit ≥ 0.0060 below max(VIS, IR) and hold only because the per-system models disagree (intervals ±0.02); P2: 5 cells worse than shipped (−0.0236 worst) (`docs/eval/no_retrain/deep_gate_fresh.md`). |
| E9 | `lex`: VIS tier above IR tier | claim holds 60/60; not IMPROVES | ≥ VIS in every system, draw and cell (0 violations). Worst claim margin −0.0042 [−0.0104, +0.0021]. Worse than shipped on 13 cells at 0.0014 (up to −0.0165, where IR's support helps). |
| E9 | `demote`: vetoed stream as the tail | IMPROVES; claim fails 22/60 | ≥ shipped in every system, draw and cell. Night still IR-led (clean night −0.1455 vs VIS). |
| — | LLF-2 (`docs/prereg-llf2-2026-10-10.md`) | FAIL (P1, P2) | Claim 52/60; clean day −0.0073, six low-light day cells −0.0068 to −0.0142; 12 cells worse than shipped (`docs/eval/llf/llf2_2026-10-10.md`). |
| E1 | Checkpoint ensembles `ens2`, `ens5` (plain WBF 0.55) | **IMPROVES** (both) | ens5 lifts VIS +0.0192 mean (+0.0037 to +0.0316) and IR +0.0073. Under the shipped gate `ens5` is better on 43 cells, worse on none, but the claim still fails on 28. Composed: `ens5_union` better on 56, worse on none, claim fails 1 (clean VIS + IR fog s3 day, −0.0102), mean 0.1751 vs 0.0894; `ens5_lex` better than shipped on all 60 cells (min +0.0036), claim fails 1 (fog VIS + IR noise s2 day, −0.0073). ens5 arms share streams across systems, so their claim intervals are zero-width (`docs/eval/no_retrain/ensembles_fresh.md`). 5× inference. |
| E2 | Restoration (selected: VIS none, IR `nlm`) | does not IMPROVE | Selection on draw 941: every VIS method fails the clean-night constraint (−0.0021 to −0.0460); IR `nlm` +0.0139 mean gated gain (`runs/restore/selection.json`). Gated NLM under the shipped gate is better on 10 cells (up to +0.0477, noisy-IR nights where the shipped output is IR alone) and worse on 2 (IR noise s2 by day with clean or low-light VIS, −0.0064 and −0.0080, although the restored IR itself gains +0.0094); claim fails 26 vs 28 (`docs/eval/restore/restore_2026-10-10.md`). |
| E5 | `full` = ens5 + gated restoration + union; `ens5_restore` | **IMPROVES** (both) | `full` better than shipped on 57 cells, worse on none, claim fails 1, mean 0.1825. `ens5_restore` (shipped gate) better on 49, claim fails 27, mean 0.1157 (`docs/eval/no_retrain/ensembles_fresh.md`). |
| E8 | TTA (horizontal flip): `tta`, `full_tta`, `full_tta_lex` | **IMPROVES** (all three) | Flip TTA lifts VIS +0.0075 alone (`tta`: 14 better, claim fails 28, mean 0.0929) and +0.0057 on top of ens5. `full_tta` is the best system measured: better on 57, worse on none, claim fails 1 (clean VIS + IR fog s3 day, −0.0082), mean 0.1862; clean day 0.3801, clean night 0.3192 (shipped 0.3566, 0.0687). `full_tta_lex`: 57 better, claim fails 2, mean 0.1803. 10 detector passes per stream per frame. |
| E3 | Constants audit (draws 942–944, descriptive) | fusion constants STAND; weak-IR fallback wrong on v2 | cap_ir_scale ×1–×16, iou_thr 0.55–0.95, support γ/IoU: nothing better beyond 0.0060 (iou 0.55/0.70/0.95 worse on 11/8/10 at 0.0014). Support off: 5 better / 9 worse. IR merge veto on: 11 worse. Veil by day: 11 worse (−0.1574). Weak-IR fallback off: 25 better, 0 worse (5 low-light days +0.0094 to +0.1684, 20 degraded-IR nights), claim fails 28 → 5 — night axis, closed. Veto off reproduces `union` (30/0, claim 1) (`docs/eval/corruption_v2/constants_audit.md`). |
