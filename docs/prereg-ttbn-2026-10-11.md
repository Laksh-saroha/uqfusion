# Pre-registration: IR-vouched test-time BatchNorm adaptation of the VIS detector (N4), 2026-10-11

Written before any TT-BN number exists on development draws 942–944 and before any TT-BN detection is
computed on pohang04. Logged in `docs/exposure-ledger-2026-09-09.md` at the same time. Code:
`scripts/ttbn.py`.

## Idea

A corrupted camera shifts the activation statistics of every layer. The detector normalises them with
the statistics of its training data. BN-mix (Schneider et al., NeurIPS 2020) re-estimates them on a
small test batch and mixes them with the source statistics. It changes no weight and reads no label.
Adapting everywhere hurts: in the screen it cost −0.007 on clean day frames and up to −0.031 at night.
So adaptation is switched on only where the VIS camera reports itself unhealthy and the thermal camera
vouches that the scene is in daylight. A dark or veiled VIS frame that IR places in daylight is
underexposed or fogged, not night. Whether VIS darkness comes from the scene or from the camera is a
question VIS cannot answer alone; this is the cross-modal part.

## What was seen before writing

* **Screen** (draw 941, systems 0–1, five VIS and five IR conditions, prior N ∈ {16, 64}, no gate,
  `scratchpad/screen_ttbn*.py`). VIS, N = 16, fused ship AP (union): lowlight day +0.0635/+0.0767, fog_s1
  day +0.0340/+0.0258, glare day +0.0170/+0.0267, fog day +0.0088/+0.0083. Losses: clean day
  −0.0080/−0.0068 and night down to −0.0311. N = 64 halves both the gains and the losses. Adapting IR
  was mixed (noise_s3 +0.009/+0.012, noise_s2 −0.003…−0.007) and is not registered.
* **Gate rates, label-free.**
  * Development draws 943/944: frame statistics only, checked equal to loaded contexts.
  * pohang04: the existing frame-statistics files (no detection, no label read).
  * The screen's gate used the IR *authority* bound (`ir_d2 ≤ bound_switch` = 64.9, the p99).
  * On development VIS lowlight, draw 943, day frames, that bound gives 0.981 with IR clean but collapses
    under any IR corruption (fog_s1 0.064, fog_s2 0.028, noise 0.000).
  * The IR *merge* bound (`ir_d2 ≤ bound` = 222.7, the fitted maximum) gives day 1.000, 0.425, 0.147,
    0.041, 0.000, 0.000 across IR clean, fog_s1–s3 and noise_s2–s3, and night **0.000 in every IR
    condition**.
  * Without a health term, night leaks up to 0.022 (IR fog_s2).
  * On pohang04 the clean IR frames are far from the development IR (health d2 median 106.4 against a
    development clean-day median of 7.9). The authority bound passes 9.5% of them and the merge bound
    86.7%. IR reads night on 4.0% of them.
  * **The gate below uses the merge bound.** This change from the screen was made after seeing these
    label-free pohang04 rates, and is disclosed as such.

## Method (`union_ttbn`)

1. **Adapted VIS detector.** Each of the five Phase 3 VIS checkpoints. Every BatchNorm2d normalises
   with μ = (N μ_src + n μ_b)/(N + n) and σ² = (N σ²_src + n σ²_b)/(N + n), where:
   * N = 16 and n is the batch size (8);
   * μ_b and σ²_b are the biased statistics of the batch over its frames and all spatial positions,
     computed layer by layer as the batch passes through;
   * batches are non-overlapping blocks of 8 consecutive frames of the stream's frame list. A frame's
     statistics include up to seven later frames: 0.7 s of look-ahead at 10 Hz, disclosed as a latency.
   * Inference is at fp32 on development; for pohang04 see Precision.
2. **Gate, per frame, label-free:** `q_vis < 1` (the shipped VIS health score under the frame's VIS
   condition) AND `ir_p05 ≤ 41.5` (IR reads day, the shipped night threshold) AND `ir_d2 ≤ 222.7` (the
   shipped IR merge-health bound). Adapted VIS records are used where the gate fires and plain records
   elsewhere. IR is never adapted.
3. **Fusion:** E4 `union` (`crossmodal26m` with `veto=None`) of that VIS stream with the plain IR
   stream. Nothing else changes.
4. **Exact shortcut:** a block whose eight frames are all ungated, under every IR condition scored, is
   not run adapted. Its adapted records would never be used, and a block's statistics depend only on
   its own frames.

`shipped`, `union` and every other system are unchanged. No constant is fitted on pohang04.

## Development verdict (draws VIS 942–944 / IR 952–954, five systems)

These draws have never been scored with TT-BN. N = 16 and the gate's terms were chosen on draw 941 (the
gate's IR bound also on label-free rates, above).

* **Cells:** 5 VIS conditions (clean, fog, fog_s1, lowlight, glare) × 6 IR conditions (clean,
  fog_s1–s3, noise_s2–s3) × {day, night} = 60.
* **Statistic:** ship AP. For system k, Δ_k = mean over the three draws of AP(`union_ttbn`) − AP(`union`).
  The cell value is the mean over the five systems, with a 95% t-interval (df 4).
* **D-H (no harm):** no cell has Δ ≤ −0.0014 with its interval below zero.
* **D-G (gain):** at least one cell has Δ ≥ +0.0060 with its interval above zero.
* **N4 passes development iff D-H and D-G.**
* Descriptive: gate rate per cell, Δ against `shipped`, and the claim fused − max(VIS, IR) for
  `union_ttbn` (VIS = the gated stream) beside `union`'s.

## pohang04 test (only if N4 passes development)

* **Cells:**
  * the registered cells of `docs/prereg-p04-heldout-gatefree-2026-10-11.md` where the gate fires;
  * VIS fog, lowlight, blur_s3, noise_s2, rain_s2 with IR clean, draws 941–944 (gate rates 0.68,
    0.85, 0.09, 0.85, 0.19).
  * clean/clean and the three IR-corrupted cells: the gate is computed there too. Where it fires on no
    frame, Δ = 0 exactly and is reported as such; otherwise those cells are built and scored like the
    others.
* **Statistic:** ship AP, mean over the five systems and the four draws. Δ = `union_ttbn` − `union`,
  paired; also against `shipped`. Interval: 2.5th and 97.5th percentiles of the same 1,000 moving-block
  replicates (L = 20, seed 1) the gate-free scoring uses (`runs/holdout_p04/gatefree/boot_weights.npy`).
* **P-T, REPLICATES** iff no cell has Δ ≤ −0.0014 with its interval below zero AND at least one cell has
  Δ ≥ +0.0060 with its interval above zero. Otherwise DOES NOT REPLICATE.

## Precision

* **Development:** fp32.
* **pohang04:** fp16 autocast (`p04_heldout.setup_fp16`), if and only if a check on development passes.
  * The check: VIS fog, fog_s1 and lowlight on draw 942 with IR clean, all five systems, day frames,
    built in fp16 and scored against the fp32 build.
  * It passes if `union_ttbn` ship AP moves by at most 0.0010 in each of the three cells.
  * Otherwise pohang04 is built in fp32.
  * The check's numbers are reported with the result.

## Amendment 1 (2:49 PM, before any fp16 TT-BN detection exists): the fp16 statistics kernel

In the fp16 build the batch statistics are computed by `torch.batch_norm_stats` (one fused pass, fp32
accumulation; 6.4× faster than an fp32 copy plus two reductions, equal to them to ~5e-7 relative on a
test tensor). The fp16 development check prices this together with the precision. The fp32 build is
unchanged.

## Amendment 2 (5:05 PM, before any `ttbn_lex` number exists): `ttbn_lex`, descriptive

Added at Laksh's request. `ttbn_lex` is E9 `lex` (`gate_zoo.build_lex_records`) built from:
* the gated VIS stream of `union_ttbn` (adapted records where the gate fires, plain records elsewhere);
* the plain IR stream mapped to the VIS canvas.

Every VIS ship box ranks above every IR ship box, and there is no merge.

*Paired comparison:* Δ = `ttbn_lex` − `lex`, where `lex` is the same construction on the plain VIS
stream. Also reported: Δ against `shipped`, and the claim fused − max(VIS, IR) with VIS the gated stream.
* **Development:** draws 942–944, the 60 cells, five systems, mean over draws, 95% t-interval (df 4).
  `lex` has never been scored on these draws.
* **pohang04:** the cells and replicates of the P-T test. The plain `lex` is the gate-free scoring's arm.

*What was seen first.*
* Development: every cell's gated VIS stream, through the claim column of the N4 verdict.
* pohang04:
  * the plain `lex` of every cell;
  * the lowlight/clean partial of `union_ttbn` (Δ vs `union` +0.0186, claim −0.0054), which places
    its gated VIS stream.

`ttbn_lex` is at least its gated VIS stream (appending boxes below every other box never lowers AP), so
those values bound it from below. **No verdict is registered.** On each target the counts that D-H/D-G
(development) and P-T (pohang04) would give for Δ are reported as description. N4's own verdicts are
unchanged.
