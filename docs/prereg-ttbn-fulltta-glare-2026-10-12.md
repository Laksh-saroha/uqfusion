# Pre-registration: N4F with N4g's glare term (N4Fg), development and pohang04 glare/clean (P-FG), 2026-10-12

Written at 11:44 PM on 10-12, before any N4Fg number exists on development or pohang04, and before any `full_tta`
or N4F detection exists on pohang04's glare/clean cell. Logged in `docs/exposure-ledger-2026-09-09.md` at the same
time. Code: `scripts/ttbn_fulltta_glare.py`.

Base methods:
* N4F: `docs/prereg-ttbn-fulltta-2026-10-11.md` (development PASS on draws 945–947; P-FT REPLICATES).
* N4g: `docs/prereg-ttbn-variants-2026-10-11.md` (development PASS on draws 942–944) and
  `docs/prereg-ttbn-glare-p04-2026-10-12.md` (P-G REPLICATES on the new glare/clean cell).

**Why.** P-FT registered N4F before pohang04 had a glare cell. P-G added glare/clean but scored only N4g against
`union`; its registration says `full_tta` and the ensemble arms were not built there. So the paper's Table 3 and
Fig. 2 carry "not scored" for N4F on glare. N4F uses N4's gate, which fires on 1.9% of glare/clean frames, so the
arm that answers the question is N4F with N4g's gate (N4Fg). Laksh asked for it on 10-12. This is a new exposure of
pohang04 under the standing rule of 10-11 (scored whenever a result needs held-out validation).

## What was seen before writing

* Every development and pohang04 number of N4, N4g, N4c, N4c2, N4ae and N4F, and of the gate-free systems:
  `docs/eval/ttbn/*` and `docs/eval/holdout_p04_gatefree_2026-10-11.md`. In particular:
  * N4F development (draws 945–947): glare cells Δ = −0.0000 to +0.0000 (N4's gate fires on ≤ 0.2% of glare day
    frames); `full_tta` glare/clean/day 0.2012.
  * N4g development (draws 942–944): glare/clean/day +0.0240 [+0.0178, +0.0302] over `union`.
  * P-FT: N4F above `full_tta` on noise, fog, low light and rain; none worse.
  * P-G: glare/clean `union` 0.1417, N4g 0.1726 (+0.0309 [+0.0290, +0.0327]); N4g's gate 81.0%, N4's 1.9%;
    `ttbng_lex` 0.1762.
* **Label-free, today** (frame statistics and the record files' block lists; no detection scored, no label read):
  * Development draws 945–947: N4g's gate differs from N4's only on VIS glare, under IR clean, fog_s1, fog_s2 and
    fog_s3 (under IR noise the IR health term closes both gates). N4g gates 1,199–1,200 glare frames per draw in
    151 blocks; N4F's build ran 0, 4 and 3 of them (draws 945, 946, 947), so 446 blocks are new.
  * pohang04: N4g needs flipped adapted blocks that N4F's build did not run in glare (1,361–1,362 blocks per
    draw; N4F never ran glare), clean VIS (block 145) and VIS fog on draws 942 and 943 (block 144). Raw adapted
    records for every N4g block already exist (N4's records, then P-G's new blocks).

## Method (`full_tta_ttbng`)

`full_tta` exactly (five checkpoints, raw + flipped VIS views, IR NLM where the IR self-check flags, plain WBF at
IoU 0.55, `union`, `full_tta`'s development capability prior), except on frames where N4g's gate passes:

(`q_vis < 1` OR (VIS mean > 117.92 AND VIS std > 62.31)) AND `ir_p05 ≤ 41.5` AND `ir_d2 ≤ 222.7`.

There each of the ten VIS views is replaced by its BN-mix-adapted record (N4's BN-mix, N = 16, non-overlapping
8-frame blocks on N4's grid; the flipped view is the same BN-mix applied to the mirrored block, mapped back), as
N4F does on N4's gate. IR is unchanged. Nothing is fitted or tuned.

**Records.** A block's adapted output depends only on its own frames. So every block N4F's development build, N4F's
pohang04 build, N4's pohang04 build or P-G's build already ran is reused; only missing blocks are built. Where N4g's
gate equals N4's, N4Fg is N4F by construction, and the gates are checked equal from both the frame statistics and the
loaded context.

**Precision.** The new adapted views are fp16 autocast, channels_last, fused statistics (N4's pohang04 precision, as
N4F). The plain flipped glare views on pohang04 are fp16 (P-F's Amendment 1 precision for every flipped view).

## 1. Development fp16 check (before any development verdict)

* Draw 945 (N4F ran no glare block there, so the fp16 side is all new). Build N4g's glare blocks in fp32 too.
* Score `full_tta_ttbng` glare/clean/day with each. It passes if ship AP moves by at most 0.0010.
* If it fails, the new blocks (development and pohang04) are rebuilt in fp32 before anything else is scored.
* The outcome is written to the ledger.

## 2. Development test (draws VIS 945–947 / IR 955–957, N4F's 60 cells)

These draws carry N4F's verdict and the no-retrain verdict; this is their third use, disclosed. N4Fg was not tuned
on them, but a pass here is weaker evidence than a first test; pohang04 decides.

* **Statistic:** ship AP; Δ = `full_tta_ttbng` − `full_tta`, mean over the three draws.
* **Interval:** N4F's development interval: 2.5th and 97.5th percentiles of 1,000 moving-block replicates (L = 20,
  seed 1, blocks within runs, over the slice's frames); one resample scores both arms and all three draws. N4F's
  stored replicates are the baseline's; N4Fg's presorts get the same weights.
* **Cells where the gates are equal on every draw:** N4Fg is N4F; Δ is N4F's, reported as such.
* **Checks (nothing is reported until a failure is explained):**
  * `full_tta` recomputed here equals its per-draw no-retrain values (|difference| < 5e-5), its mean equals
    `docs/eval/no_retrain/ensembles_fresh.json` to the fourth decimal, and it equals N4F's development `full_tta`
    exactly, in all 60 cells.
  * Where the gates differ, N4F recomputed here equals N4F's stored score exactly.
* **PASS iff both:**
  * (a) no cell has Δ ≤ −0.0014 with its interval below zero;
  * (b) every cell where the glare term adds gated frames has N4Fg − N4F ≥ −0.0014.
* Descriptive: N4Fg − N4F with its interval in those cells; gate rates.
* If development fails, the work stops and is reported; pohang04 is not built or scored.

## 3. Held-out test P-FG (only if development passes)

* **Cell:** glare/clean (VIS v2 glare severity 2, corruption code `86ce9080b2b6`, seed = draw, draws 941–944; IR clean).
* **Build:** the plain flipped VIS glare views for the five checkpoints (`p04_heldout.py build-unit` arithmetic),
  then N4Fg's missing flipped adapted blocks (glare; clean block 145; fog block 144 on draws 942–943).
* **Assembly:** as P-FT (`ttbn_fulltta._p04_job`): the look's context with development calibration
  (`holdout_p04_look.system_context`, P-G's substrate for the glare stream), `full_tta`'s development capability
  prior (`runs/holdout_p04/gatefree/devcal.json`).
* **Statistic:** ship AP (class 0, AP50:95) against the VIS ground truth, mean over the four draws (the ensemble arms
  are identical for every system).
* **Δ** = `full_tta_ttbng` − `full_tta`, paired, on `runs/holdout_p04/gatefree/boot_weights.npy` (L = 20, 1,000
  replicates, seed 1); 95% interval = 2.5th and 97.5th percentiles.
* **P-FG REPLICATES iff both:**
  * (a) glare/clean has Δ ≥ +0.0060 with its interval above zero;
  * (b) gate-equality check against P-FT, as in P-G: in the six cells where N4g's gate equals N4's on every draw,
    N4Fg is N4F (checked, not scored); in clean/blur_s2, clean/fog_s2 and fog/clean, N4Fg is built and scored, and
    no cell has N4Fg − N4F ≤ −0.0014 with its interval below zero.
* Otherwise **P-FG DOES NOT REPLICATE**, reported in the same places a replication would be.

**Descriptive, no verdict:** `full_tta` and N4F (`full_tta_ttbn`, N4's gate) on glare/clean, with N4F − `full_tta`;
the gate rates of N4g and N4; N4Fg − N4g (P-G's `union_ttbng`, one pass per stream) with its interval.

N4's, N4g's, N4F's and every earlier verdict are unchanged. No pohang04 glare value of any ensemble arm is displayed
before all four draws are scored and bootstrapped.

## Order

1. Ledger entry for this registration.
2. Development: build the 446 missing blocks (fp16, raw + flipped); fp16 check on draw 945 (ledger); score;
   bootstrap; report (`docs/eval/ttbn/ttbn_fulltta_glare_dev_2026-10-12.md`). Stop here on a failure.
3. pohang04: plain flipped glare views; N4Fg's missing flipped adapted blocks; score (`UQF_FAST_*`); bootstrap
   (`p04_heldout_score.py boot`); report (`docs/eval/ttbn/ttbn_fulltta_glare_p04_2026-10-12.md`); ledger.

## Amendment 1 (11:47 PM on 10-12): pohang04 runs whatever development's verdict

Laksh asked at 11:46 PM for the whole pipeline to be queued unattended and run regardless of the result. Written
before any N4Fg number exists: the development build had finished draw 945's blocks; no check, score or report had
run.

* The pohang04 build and P-FG run after the development report whatever its verdict. They still stop if an
  integrity check fails: the fp16 check (step 1), the reproduction checks of step 2, or any gate-equality or
  frame-identity check in scoring.
* P-FG's rule is unchanged. If development FAILS, the pohang04 result is reported with that verdict beside it and
  is not adopted as a replication of a development pass.
* Each step is logged in the ledger with its time by the queue script
  (`runs/holdout_p04/ttbn_fulltta_glare/queue.sh`).
