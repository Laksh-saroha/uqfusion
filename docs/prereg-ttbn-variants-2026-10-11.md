# Pre-registration: two N4 variants, a glare term (N4g) and causal statistics (N4c), 2026-10-11

Written before any N4g or N4c number exists on development draws 942–944, and before any N4c detection
exists anywhere. Logged in `docs/exposure-ledger-2026-09-09.md` at the same time. Base method: N4,
`docs/prereg-ttbn-2026-10-11.md` (passed development 3:13 PM; pohang04 test running). Code:
`scripts/ttbn_variants.py`.

## N4g: a glare term in the VIS side of the gate

**Why.** N4's gate needs the VIS camera to report itself unhealthy (`q_vis < 1`). Under glare it almost
never does: the gate fires on 0.0–0.2% of glare day frames on draws 942–944. Ungated BN-mix gains there:
the screen on draw 941, systems 0–1, gave +0.0170/+0.0267 on glare day frames.

**What was seen before writing.**
* Draw 941, label-free: the clean-frame range of every VIS brightness and structure statistic, and which
  of them separate glare frames from clean, fog, fog_s1 and lowlight frames.
* Draw 941, scored: the gated screen. With `p95 > 220` as the glare term:
  * glare day: +0.0170/+0.0267 (gate 100%);
  * clean, fog, fog_s1 and lowlight: identical to N4.
* Label-free statistics on pohang04 changed the term after the screen:
  * `p95 > 220` fires on 55% of clean pohang04 frames, and `std > 62.31` on 22%. pohang04 is brighter
    than development (median `p95` 224). Each of these detects brightness, not glare.
  * The term below fires on 0.10% of clean pohang04 frames. After the gate's other terms it adds no gated
    frame in any registered pohang04 cell except one frame of VIS fog, draw 941.
  * This choice used pohang04's label-free statistics and is disclosed as such. No pohang04 detection or
    label was read for it.

**Method.**
* Glare term: VIS `mean > 117.92` AND VIS `std > 62.31`. Each threshold is the maximum of that statistic
  over the clean development day frames (all in `FIT_RUNS`; draw-independent).
* Gate: (`q_vis < 1` OR glare term) AND IR reads day AND `ir_d2 ≤ 222.7`.
* Everything else as N4: N = 16, blocks of 8, adapted VIS records where the gate fires, E4 `union`.

On draws 942–944 the glare term fires on 99.8–99.9% of glare day frames and on no clean, fog, fog_s1 or
lowlight day frame. So `union_ttbng` equals `union_ttbn` in every non-glare cell, which is checked, not
assumed.

**Development verdict** (draws VIS 942–944 / IR 952–954, five systems, the 60 cells of N4; Δ =
`union_ttbng` − `union`, mean over draws per system, 95% t-interval over systems, df 4):
* **G-H (no harm):** no cell has Δ ≤ −0.0014 with its interval below zero.
* **G-G (gain where the term acts):** at least one of the 12 VIS-glare cells has Δ ≥ +0.0060 with its
  interval above zero.
* N4g passes iff G-H and G-G.

No pohang04 test is registered: none of its registered cells is a glare cell. The label-free rates above
are the held-out evidence that the term stays quiet on clean frames; that is all they show.

## N4c: causal statistics

**Why.** N4 normalises each block of 8 frames with that block's own statistics. A frame's statistics
include up to 7 later frames, 0.7 s of look-ahead at 10 Hz, disclosed as a latency. N4c removes the
look-ahead.

**Method.**
* **Blocks:** within each run (capture sequence), non-overlapping blocks of 8 consecutive frames of the
  stream's frame list. The grid starts at the run's first listed frame.
* **Normalisation:** block b of a run normalises every BatchNorm2d with
  μ = (N μ_src + n μ_{b−1})/(N + n), and σ² likewise. Here:
  * μ_{b−1} and σ²_{b−1} are the biased batch statistics of that layer's input, measured during block
    b−1's own pass;
  * N = 16 and n = 8.
* **First block of each run:** normalised with the source statistics.
* A frame's normalisation therefore depends only on frames before its block: no look-ahead.
* **Blocks run:** every block of a run that has at least one gated frame, under any IR condition scored,
  in order.
* **Gate and fusion:** N4's gate (`q_vis < 1` & IR reads day & `ir_d2 ≤ 222.7`) and E4 `union`.
* **Precision:** fp32.
* On development, consecutive listed frames of a run can be minutes apart (the val split is blocked);
  "previous" means previous in the list.

**Development verdict** (same draws, systems and 60 cells; Δ = `union_ttbnc` − `union`):
* **C-H:** no cell has Δ ≤ −0.0014 with its interval below zero.
* **C-G:** at least one cell has Δ ≥ +0.0060 with its interval above zero.
* N4c passes iff C-H and C-G.
* Descriptive: `union_ttbnc` − `union_ttbn` per cell (the price of causality), and gate rates.

A pohang04 test of N4c would be a separate registration, written only after its development verdict.

No constant of either variant is fitted on pohang04, and N4's own verdicts are unchanged.

## Amendment 1 (7:14 PM, before any N4c2 detection on draws 942–944): N4c failed by construction; N4c2 registered

**N4c's verdict stands: FAIL** (`docs/eval/ttbn/ttbn_variants_dev_2026-10-11.md`). Seen before writing:
every number in that report, N4g's verdict included.

**Diagnosis (post hoc).** As registered, N4c's statistics are those of each layer's input measured during
block b−1's own pass. That pass normalised the earlier layers with block b−2's statistics, not as N4
does. So each layer receives statistics measured on activations that were normalised differently from
the ones it now normalises. N4c's own records show the effect (low light, draw 942, system 0):
* the first block of a run, at source statistics, detects like plain inference;
* the next block, the first to use carried statistics, collapses: max confidence 0.11, 0.04, 0.18 and
  0.24 on frames 8–11, against N4's 0.77, 0.58, 0.83 and 0.84;
* by block 25 (frames 200–203) it returns no box at all, where N4 returns 20–26;
* over the 1,200 day frames, mean max confidence is 0.212, against N4's 0.750.

Such a recursion can need about one block per BatchNorm layer (130) to settle, which is longer than any
run. N4c therefore measures this construction, not the cost of causality.

**N4c2: N4, one block late.**
* Block b of a run is normalised with BN-mix of the source statistics and the per-layer batch statistics
  that **N4** measured on block b−1.
  * Block b−1 is also run through N4's closed-loop BN-mix, the same pass N4 runs, and its statistics are
    stored. The output of that recording pass is discarded.
* A run's first block uses source statistics. No frame's normalisation reads its own block or a later
  one.
* Grid, blocks run, N = 16, n = 8, gate (N4's) and fusion (E4 `union`) are N4c's.
* Precision: fp16 autocast, channels_last and fused statistics, as in N4's pohang04 build, which was
  priced on development at ≤ 0.0003 for N4.
* Code: `scripts/ttbn_causal2.py`.

**Mechanics check before registering** (draw 941, the screen draw; system 0; low light; first five
blocks of pohang00; CPU; no label read): N4c2's mean max confidence per block is 0.622, 0.734, 0.755, 0.609
and 0.615, against N4's 0.670, 0.728, 0.750, 0.596 and 0.636.

**Development verdict** (draws 942–944, five systems, the 60 cells):
* C-H and C-G as for N4c, on Δ = `union_ttbnc2` − `union`.
* Descriptive: `union_ttbnc2` − `union_ttbn` (N4's fp32 development scores), the price of causality.

**Second use, disclosed.** Draws 942–944 already carry N4's and N4c's verdicts, so a pass here is weaker
evidence than a first test. A pohang04 test of N4c2 would be a separate registration, written only
after this verdict.

## Amendment 2 (9:45 PM, before any N4c2 detection on pohang04): N4c2's pohang04 test (P-C2)

**N4c2 passed development at 9:41 PM** (`docs/eval/ttbn/ttbn_causal2_dev_2026-10-11.md`). No cell is worse.
Four cells are better, the largest low-light/clean/day at +0.0668 [+0.0559, +0.0777]. The price of
causality against N4 is −0.0038 [−0.0045, −0.0031] there.

**Seen before writing:**
* every number in that report;
* N4's pohang04 verdict, P-T REPLICATES (9:11 PM): noise_s2 +0.1823, fog +0.0370, low light +0.0186 over
  `union`, none worse;
* all gate-free pohang04 numbers.

No N4c2 detection on pohang04 exists. N4ae's development scoring was running; none of its numbers had
been read.

**Method on pohang04:** N4c2 exactly as in Amendment 1.
* Grid: N4c's per-run grid over pohang04's VIS frame list.
* Block b of a run is normalised with N4's statistics of block b−1. A run's first block uses source
  statistics.
* fp16 autocast, channels_last, fused statistics; N = 16, n = 8.
* Stream: N4c2's records where N4's gate fires in the cell, plain records elsewhere. Fusion: E4 `union`.
* N4's statistics of block b−1 come from N4's closed-loop pass over b−1's own frames. They do not depend
  on any earlier block. So only these passes are run:
  * the output pass of each block holding a gated frame (gated under any IR condition of the registered
    cells, as `ttbn.needed`);
  * the recording pass of each such block's predecessor in the grid.
  This gives the records the full grid would. It is checked on development before the build: sparse
  against full-grid records, draw 942, low light, system 0.

**Test (P-C2):** P-T's cells, statistic and replicates (`docs/prereg-ttbn-2026-10-11.md`).
* Δ = `union_ttbnc2` − `union`, paired.
* **REPLICATES** iff no cell has Δ ≤ −0.0014 with its interval below zero AND at least one cell has
  Δ ≥ +0.0060 with its interval above zero. Otherwise DOES NOT REPLICATE.
* Descriptive: Δ against `union_ttbn` (N4's pohang04 records), the price of causality on pohang04.

Code: `scripts/ttbn_causal2.py` (`build-p04`, `score-p04`, `report-p04`). N4's verdicts and P-T are unchanged.

**The equivalence check as run (9:56 PM, before any pohang04 detection).**
* The first version compared the sparse records with build-dev's stored records. It failed: 39 of 880
  frames differed in box count, and the largest confidence difference was 0.046.
* Diagnosis, same kept blocks: a full-grid re-run in a new process differs from the stored records just
  as much (34 of 880 frames, 0.064). In one process, sparse and full grid agree bit for bit on all 880
  frames, and a second sparse run matches the first. fp16 inference with cuDNN autotuning is not
  bit-reproducible across processes; the sparse plan adds nothing.
* The check now requires bit-exact agreement between sparse and full grid run in one process. The
  comparison with the stored records is reported descriptively.
* The cross-process variation also touches N4's and N4c2's own fp16 records. It is not measured on its own.
  N4's fp16 check (development, one fp16 build against fp32) moved `union_ttbn` by at most 0.0003 (tolerance
  0.0010).
