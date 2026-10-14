# Pre-registration: which part of IG-BN's gate does the work (gate ablation, GA), 2026-10-14

Written at 3:50 PM on 10-14, before any ablation arm has a score on development or on pohang04. Logged in
`docs/exposure-ledger-2026-09-09.md` at the same time. Base method: IG-BN (N4,
`docs/prereg-ttbn-2026-10-11.md`). Code: `scripts/gate_ablation.py`.

**Why.** A simulated JOE review (2026-10-14) asked for evidence that the thermal camera, and not BN
re-estimation itself or a clock, is what makes IG-BN's gate work. PAPER.md §7 names this ablation as the
first registration to add. Until now the only evidence that the gate avoids losses was an unregistered
screen (draw 941, two systems).

## Arms

Every arm uses N4's adaptation unchanged: BN-mix with N = 16 on non-overlapping blocks of 8 consecutive
frames of the stream's frame list, the five Phase 3 VIS checkpoints, E4 `union` with the plain IR stream.
The arms differ only in which frames take the adapted record:

| arm | code | adapts a frame iff |
|---|---|---|
| ungated | A0 | always |
| visible term only | A1 | `q_vis < 1` |
| thermal day term only | A2 | `ir_p05 ≤ 41.5` AND `ir_d2 ≤ 222.7` |
| solar only | A3 | solar elevation > 0° |
| visible term + clock | A4 | `q_vis < 1` AND solar elevation > 0° |
| IG-BN | N4 | `q_vis < 1` AND `ir_p05 ≤ 41.5` AND `ir_d2 ≤ 222.7` |

* All constants are N4's (the shipped VIS health bound 124.3, IR night threshold 41.5, IR merge bound 222.7).
* **Solar elevation** is A9.4's rule from the pohang04 look (`holdout_p04_step1.solar_elevation`): NOAA
  geometric elevation, no refraction, at the VIS frame's stereo capture time (`meta/<run>/timestamps/stereo.txt`)
  and the nearest valid GPS fix (`navigation/gps.txt`, no-fix rows dropped). The code is sanity-checked
  first (pohang00 day, pohang01 night). A4 is the direct substitute of a clock for the thermal camera: it
  keeps IG-BN's visible term and replaces both thermal terms with the sun.
* **Records.** A block's statistics depend only on its own eight frames, so every arm reads one complete
  adapted stream per (VIS condition, draw): N4's stored records for every block N4 ran, plus records built
  for the blocks N4 did not run (it skipped blocks with no frame it gates). The new blocks are built with
  N4's code path and precision (development fp32; pohang04 fp16 autocast with fused statistics, as P-T).
  Every arm's adapted frames that fall in N4's blocks use N4's records bit for bit, so the N4 arm reproduces
  N4's registered scores; this is asserted (development: every cell, every draw and system; pohang04: each
  job's observed AP against the stored P-T presort).
* Arms whose gates are identical frame for frame in a job share one computation; an arm whose gate fires
  nowhere is `union` exactly.

## What was seen before writing

* Every development and pohang04 number to date, including N4's 60-cell development table, its variants
  and every pohang04 scoring (P-U, P-F, P-T, P-A, P-FT, P-C2, P-G, P-FG) and the pohang01 night test.
* The unregistered screen of `docs/prereg-ttbn-2026-10-11.md` (ungated BN-mix, draw 941, systems 0–1):
  clean day −0.0080/−0.0068, night down to −0.0311, low-light day +0.0635/+0.0767.
* **Label-free, today** (`runs/gate_ablation/rates.json`; no detection, no label):
  * Solar elevation. Development day frames +32.9° to +50.4°, night frames −7.2° to −5.2°: the solar rule
    equals the run-level day/night split on every development frame. pohang04: +56.0° to +63.0°, sun up on
    every frame. pohang01 night-test pairs: −10.4° to −6.2°.
  * Consequences, by construction:
    * development: A3 adapts exactly the day frames; A4 equals N4 frame for frame wherever the IR stream
      is clean (where every day frame passes both thermal terms; to be checked frame-exact in scoring) and
      on every night cell (both fire on no night frame), and on the clean-VIS cells (`q_vis` never fires
      on clean day). A4 differs from N4 only on the 20 day cells with VIS fog, fog_s1, low light or glare
      and a corrupted IR stream, where A4 adapts more (it ignores the IR stream);
    * pohang04: A3 ≡ A0 and A4 ≡ A1 on every frame. The held-out run cannot separate a clock from no
      day–night term; it can separate the thermal terms from both.
  * Gate rates, development (mean over draws 942–944): A0 1.000 everywhere. A1: day 0.996 (fog), 0.626
    (fog_s1), 1.000 (low light), 0.002 (glare), 0 (clean VIS); night 1.000 on every cell. A2: day 1.000,
    0.425, 0.149, 0.045, 0, 0 across IR clean, fog_s1–s3, noise_s2–s3; night 0. A3: day 1.000, night 0.
    A4: A1's day rates, night 0. N4: as registered (e.g. low light 1.000, 0.425, 0.149, 0.045, 0, 0).
  * Gate rates, pohang04 (mean over draws), cells clean/clean, clean/blur_s2, clean/noise_s2, clean/fog_s2,
    blur_s3, noise_s2, rain_s2, fog, low light (IR clean in the last five):
    * A1 = A4: 0.003, 0.003, 0.003, 0.003, 0.136, 1.000, 0.244, 0.803, 1.000.
    * A2: 0.852, 0.749, 0.000, 0.518, 0.852, 0.852, 0.852, 0.852, 0.852.
    * N4: 0, 0, 0, 0, 0.092, 0.852, 0.186, 0.678, 0.852.
    * The thermal terms withhold adaptation from about 15% of held-out frames (mostly the merge bound: the
      held-out IR frames sit far from development's).
* **Reproducibility of the new blocks** (no score, no label): rebuilding four of N4's development blocks
  (low light, draw 942) on N4's fp32 code path gives N4's records bit for bit (160/160 frames). The
  pohang04 fp16 path is not bit-reproducible (cuDNN autotuning picks kernels per process): rebuilding six
  of N4's pohang04 blocks (low light, draw 941) gives the same box count on 98.3% of frames and moves the
  top box's confidence by at most 0.0021. Its size in ship AP is priced on development before any pohang04
  ablation number exists (rebuild N4's fp16 development blocks of low light and fog, draw 942, and score
  `union_ttbn`, IR clean, day, both ways); the result is entered as an amendment below before the pohang04
  build. This noise touches only frames in blocks N4 did not run, it is the same for every arm that adapts
  them, and it is not a bias toward any arm.

## Development (draws VIS 942–944 / IR 952–954, five systems)

These draws were scored by N4 and its variants; this is a further use, disclosed. The ungated screen used
draw 941.

* **Cells:** N4's 60 (5 VIS conditions × 6 IR conditions × {day, night}).
* **Statistic:** ship AP. Per system k, Δ_k = mean over the three draws of AP(arm) − AP(`union`); the cell
  value is the mean over the five systems, with a 95% t-interval (df 4). The same for arm − N4 (paired).
* **Rule (§3.4), per arm against `union`:** *worse* = Δ ≤ −0.0014 with interval below zero; *better* =
  Δ ≥ +0.0060 with interval above zero. An arm PASSES iff no cell is worse and at least one is better.
  The same thresholds against N4 define *worse than N4* and *better than N4*.

### Registered predictions (development)

* **GA-1, adaptation needs a gate:** A0 FAILS (worse cells on clean-VIS day and on night).
* **GA-2, a day–night term is needed:** A1 FAILS (worse cells at night).
* **GA-3, the visible term is needed:** A2 FAILS (worse cells on clean-VIS day where the IR stream vouches
  for daylight).
* **GA-4:** A3 FAILS (clean-VIS day), for the same reason as GA-3.
* **GA-5, thermal camera against clock:** A4 PASSES, and against N4 it is never worse and is better in at
  least one of the 20 cells where it differs (low light or fog by day with a corrupted IR stream, where N4's
  gate shrinks and A4's does not).
* **N4** reproduces its registered result (PASS; worst Δ −0.00001).

### How the outcome is read (fixed now)

* GA-1, GA-2, GA-3: the named term is *shown necessary on development* iff the arm without it FAILS on a
  worse cell; otherwise *not shown necessary*.
* GA-5 (A4 against N4, development and pohang04 read separately):
  * **CLOCK BETTER** iff A4 has no cell worse than N4 and at least one cell better than N4;
  * **CLOCK MATCHES** iff no cell worse and none better;
  * **THERMAL BETTER** iff at least one cell worse than N4 and none better;
  * **MIXED** otherwise.
  CLOCK BETTER or CLOCK MATCHES means the thermal day term has not been shown to add anything over a clock
  on this data, and the paper will say so.

## pohang04 (the nine registered cells of P-T)

A new exposure: the tenth scoring of pohang04, under the standing rule of 10-11.

* **Arms scored:** A0, A1, A2 and N4. A3 ≡ A0 and A4 ≡ A1 on every pohang04 frame (sun up on all), so
  their values are A0's and A1's, reported under their own names. N4's values are P-T's (asserted).
* **Statistic:** ship AP against the VIS ground truth, mean over the five systems and the four draws of a
  corrupted cell. Δ = arm − `union` and arm − N4, paired; 95% interval from the gate-free scoring's 1,000
  moving-block replicates (L = 20, seed 1, `runs/holdout_p04/gatefree/boot_weights.npy`), as P-T.
* **Rule:** as development; an arm REPLICATES iff no cell is worse than `union` and at least one is better.
* **Predictions:**
  * A0 (= A3) DOES NOT REPLICATE: worse on the four clean-VIS cells (it adapts every clean frame).
  * A2 DOES NOT REPLICATE: worse on clean/clean, clean/blur_s2 and clean/fog_s2 (it adapts 52–85% of
    clean frames).
  * A1 (= A4) REPLICATES; against N4 never worse and better on at least one of low light, noise and fog,
    because it adapts the ~15% of frames N4's thermal terms withhold. On the four clean-VIS cells A1 adapts
    0.3% of frames, which is not expected to move ship AP.
  * GA-5 on pohang04: CLOCK BETTER.

## Part 2: IG-BN's gate on the pohang01 night test pairs (label-free, descriptive)

* **Frames:** the 1,109 pohang01 test-split pairs of `docs/prereg-p01-nighttest-2026-10-11.md`
  (`runs/night_p01test/pairs_{vis,ir}.txt`). VIS clean.
* **IR conditions:** clean (the night test's existing frame statistics) and the development IR
  corruptions, v2 fog s1–s3 and noise s2–s3, at seeds 952, 953 and 954 (the development IR draws, here
  replayed over this frame list). Their IR frame statistics are computed by the arithmetic that wrote the
  development v2 IR statistics (`parallel_frames.corrupted_frames`, GPU interpreter), checked bit-exact
  on development frames (IR fog_s1 and noise_s3, seed 953) before use.
* **Reported:** per IR condition (mean over the three seeds) the share of pairs with `q_vis < 1`, with IR
  reading day, with IR healthy, with both, and IG-BN's gate rate, plus the largest number of gated frames
  in any seed. No detection is run and no label is read; no AP is computed.
* **Prediction:** IG-BN's gate fires on no pair in any condition (as on development night), so on these
  pairs IG-BN is `union` and the night test's `union` result is IG-BN's. Any firing is reported with its
  count. A3 and A4 fire on none of them by construction (sun below the horizon on all).

## Outputs

`docs/eval/ttbn/gate_ablation_dev_2026-10-14.md`, `docs/eval/ttbn/gate_ablation_p04_2026-10-14.md`,
`docs/eval/ttbn/gate_night_p01_2026-10-14.md` (each with a `.json`). Large intermediates (complement
records, pohang04 presorts) in `runs/gate_ablation/` (a junction to C:, because A: is full).

## Amendment 1 (4:21 PM, before any pohang04 ablation block is built): the fp16 rebuild noise, priced

As registered above, N4's fp16 development blocks of low light and fog (draw 942; 151 blocks each) were rebuilt
on the pohang04 code path (fp16 autocast, channels_last, cuDNN autotuning, fused statistics) and `union_ttbn`
was scored both ways (IR clean, day, five systems; `runs/gate_ablation/fp16_rebuild_noise.json`):

| cell | stored | rebuilt | Δ (mean of five) | largest |Δ| of one system |
|---|---:|---:|---:|---:|
| lowlight/clean/day | 0.25071 | 0.25070 | −0.000009 | 0.00020 |
| fog/clean/day | 0.03269 | 0.03268 | −0.000012 | 0.00002 |

Rebuilding fp16 blocks moves a cell by about 1e-5, two orders below the 0.0014 floor. The pohang04 build goes
ahead with N4's stored records plus newly built blocks, as registered. No ablation number exists on pohang04;
development scoring has started (no result read).

## Amendment 2 (5:39 PM, before any gap-arm number exists): Yoo et al.'s feature-gap gate

Added at Laksh's request after the novelty check (`docs/novelty-context-gated-tta-2026-10-14.md`, `afedfb6`): the
claim that a model-internal signal cannot separate illumination from degradation rested on one internal signal
(`q_vis`). The obvious alternative is the feature-distribution gap of Yoo et al. (CVPR 2024, arXiv 2312.08875,
Sec. 3.4), which skips adaptation when test features sit near the training features. It might stay quiet at
night, because night is in the detector's training distribution, and it needs no second sensor and no clock.

**The statistic (Yoo et al. Eq. 1, 2, 5, with the choices their text leaves open fixed here).**
* *Features:* the 1,280-dimensional vector every plain detection cache already stores per frame (`feat`): the
  global average over space of the three inputs of system k's detection head (neck outputs P3–P5),
  concatenated. Yoo et al. take "the backbone" output and do not state the pooling; this is the nearest pooled
  representation we hold, and it is the one the shipped Mahalanobis scorer reads. No new detection is run.
* *Training statistics:* μ_tr and a diagonal Σ_tr (per-channel variance, floored at 1e-6 × its mean) over the
  3,181 training-split VIS frames of the Mahalanobis fit list (`runs/derived/maha_fit_vis_p04out.txt`,
  `runs/cache_p3/seed{k}/gauss_vis_train_clean.pkl`): pohang00 679, pohang01 (night) 782, pohang02 848, pohang03
  872; all in the training split, none in the validation split or the development frames (checked). Yoo et al.
  used 2,000 training samples.
* *Test mean (Eq. 1):* an EMA over N4's 8-frame blocks in list order, μ_te ← (1 − α) μ_te + α · (block mean),
  α = 0.01 (their value), started at μ_tr and restarted where a block begins in another run. A block's decision
  applies to its eight frames.
* *Gap (Eq. 2):* L_img = KL(N(μ_tr, Σ_tr) ‖ N(μ_te, Σ_tr)) = ½ Σ_c (μ_te,c − μ_tr,c)² / σ²_tr,c.
* *In-domain gap (Eq. 5):* D_in = KL between the diagonal Gaussians of two disjoint random halves of the 3,181
  reference frames (permutation seed 0).
* The gate is per system: each detector's own features.

**Arms.**
| arm | adapts a block iff |
|---|---|
| G-Y, gap (Yoo's rule) | L_img / D_in > 1.1 OR L_img / L_ema > 1.05, with L_ema ← 0.99 L_ema + 0.01 L_img (started at the first L_img of a run); their τ1 and τ2 |
| G-C, gap (calibrated bound) | L_img > B_k, B_k = the maximum of L_img over system k's 3,181 reference frames replayed as a stream (list order, blocks of 8, run restarts), as the other bounds are maxima over clean frames; Yoo's criterion 1 only (their Sec. 4.5 option τ2 = ∞) |
| `q_vis` ∧ G-Y | `q_vis` < 1 AND G-Y |
| `q_vis` ∧ G-C | `q_vis` < 1 AND G-C |

The last two ask the question the novelty check raised: can a model-internal gap stand in for the day term
(the thermal terms or the clock) beside the visible term? Everything else (adaptation, records, fusion,
statistics, rule) is as registered for the other arms.

**Seen before writing (label-free, `runs/gate_ablation/gap_rates.json`; no detection run, no label, no score):**
* D_in per system 1.62, 2.35, 2.32, 2.65, 4.49; B_k 72.2, 72.9, 73.4, 64.7, 81.7 (18–45 × D_in). The reference
  list samples every 10–15 frames, so its replayed stream is less autocorrelated than a 10 Hz test stream.
* Firing rates, development (mean over systems and draws): G-Y fires on 0.993 of day frames in every VIS
  condition and on 1.000 of night frames; G-C on 0.61 (clean), 0.86 (fog), 0.82 (fog_s1), 0.72 (low light), 0.69
  (glare) by day and 0.895 (clean) to 1.000 at night. With `q_vis`: by day `q_vis`'s own rates (G-C trims them, e.g.
  low light 0.72); at night 1.000 (G-Y) and 0.895–1.000 (G-C).
* pohang04: G-Y 0.999 and G-C 0.984–0.992 in every cell; with `q_vis`, close to `q_vis` alone.
* pohang01 night pairs (VIS clean): G-Y 0.993, G-C 0.665; with `q_vis` the same.
* So on this detector the gap does not stay quiet at night; the scores below price what that costs.

**Predictions (development, same 60 cells and draws, same rule).**
* G-Y FAILS: worse cells on clean-VIS day and on night, like the ungated arm.
* G-C FAILS: worse cells at night (and on clean day, where it fires on 61%).
* `q_vis` ∧ G-Y FAILS and `q_vis` ∧ G-C FAILS: worse cells at night, like `q_vis` alone.
* **GA-6 reading** (fixed now): a model-internal feature gap *can stand in for the day term on this data* iff
  `q_vis` ∧ G-Y or `q_vis` ∧ G-C PASSES on development; each is also compared with IG-BN and with the clock gate
  (`q_vis` ∧ solar) by the better/worse counts of the rule, reported as CLOCK/GAP BETTER, MATCHES or MIXED as in
  GA-5.

**pohang04** (the eleventh scoring, under the standing rule): the four arms on the nine cells, as registered
for the other arms, against `union`, IG-BN and the clock gate. pohang04 is all daylight, so it cannot test the
night half. Predictions: G-Y and G-C DO NOT REPLICATE (worse on the clean-VIS cells, where they adapt ~99% of
frames, like the ungated arm); `q_vis` ∧ G-Y and `q_vis` ∧ G-C REPLICATE, close to `q_vis` alone.

**pohang01 night pairs:** the four arms' firing rates above, reported in
`docs/eval/ttbn/gate_night_p01_2026-10-14.md` (label-free).
