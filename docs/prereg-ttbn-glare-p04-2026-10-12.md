# Pre-registration: N4g on the held-out run through a new glare cell (P-G), 2026-10-12

Written at 9:36 AM on 10-12. At that time no glare detection and no glare frame statistic exist on pohang04.
Logged in `docs/exposure-ledger-2026-09-09.md` at the same time. Base method: N4g,
`docs/prereg-ttbn-variants-2026-10-11.md` (passed development: glare/clean/day +0.0240 [+0.0178, +0.0302]).
Code: `scripts/ttbn_glare_p04.py`.

**Why.** N4g's registration had no pohang04 test: none of the registered pohang04 cells corrupts VIS with glare.
Laksh asked for the test on 10-12 (handoff `docs/handoff-2026-10-12.md`, §4 item 4). This adds one cell
to pohang04 and scores N4g on it and on the cells already registered. It is a new exposure of pohang04 under
the standing rule of 10-11: pohang04 is scored whenever a result needs held-out validation.

## What was seen before writing

* Every development number in `docs/eval/ttbn/ttbn_variants_dev_2026-10-11.md`.
* Every pohang04 verdict and table to date: P-U, P-F, P-T, P-A, P-FT, P-C2 and the night test.
* **Label-free, today:** where N4g's gate differs from N4's in the nine registered pohang04 cells. The
  frame-statistics files were already on disk; no detection and no label was read.
  * The glare term fires on 8–20 frames of 12,482 per stream: 12 on clean VIS (0.10%), 17–20 on VIS fog.
  * Where those frames also pass the gate's IR terms, they become gated frames N4 does not gate. These are
    the (cell, draw) pairs, with frame indices:

    | cell (VIS/IR) | 941 | 942 | 943 | 944 |
    |---|---|---|---|---|
    | clean/blur_s2 | 348, 350 | 349, 1166 | — | 348, 1166 |
    | clean/fog_s2 | 1164 | — | 1166 | 1166 |
    | fog/clean | 1157 | 1154, 1156 | 1155 | 1156 |

    In the other six cells N4g's gate equals N4's on every draw.

**Correction to the N4g registration.** That registration says the glare term "adds no gated frame in any
registered pohang04 cell except one frame of VIS fog, draw 941". This is incomplete. That frame is
fog/clean on draw 941 (frame 1157). The table above lists all of them, from the same statistics. N4g's
development verdict is unaffected: no pohang04 test had been registered. P-G covers all three cells.

## The new cell: glare/clean

* **VIS:** corruption v2 `glare` at severity 2, as on development (`ttbn.DEV_CONDS["glare"]`). Code revision
  `86ce9080b2b6`; corruption seed = the VIS draw, 941–944.
* **IR:** clean. pohang04 has no night frame, so every glare frame is a daytime sun glare.
* **Plain caches:** the five Phase 3 VIS checkpoints, built by `build_cache_multi.py --batch 4` at fp32 with
  the gate's frame statistics in the same pass. These are the settings the 2026-10-10 re-score used for
  every corrupted pohang04 cell. Stored beside them in `runs/holdout_p04/v2/caches/seed{k}/draw{v}_{v+10}/`
  as `gauss_vis_paired_glare.pkl`, and `runs/holdout_p04/v2/derived/{brightness,structure}/draw{v}_{v+10}/`.
* **Assembly:** the look's (`holdout_p04_look.system_context`, development calibration injected), as every
  pohang04 scoring. Nothing is fitted on pohang04.

## N4g on pohang04

* **Gate:** (`q_vis < 1` OR (VIS `mean > 117.92` AND VIS `std > 62.31`)) AND IR reads day AND `ir_d2 ≤ 222.7`.
  The thresholds and gate are exactly as registered.
* **Adaptation:** N4's BN-mix with N = 16 on N4's block grid: non-overlapping blocks of 8 consecutive frames
  of pohang04's VIS list. A block is run if it holds a frame gated under any IR condition registered for
  that VIS stream. A block's statistics depend only on its own frames.
* **Records:**
  * N4's pohang04 records (fp16) are reused for every block N4 already ran.
  * New records are built only for blocks N4 did not run: every glare block, plus blocks 144–145 where the
    table above needs them.
  * So where N4g's gate equals N4's, N4g's stream is N4's stream, bit for bit.
* **Precision:** fp16 autocast, channels_last and fused statistics, as N4's pohang04 build. Condition: a
  development check passes first.
  * The check builds N4g's glare records on draw 942 in fp16 and scores them against the stored fp32
    records. Cell: glare/clean/day, IR clean, five systems.
  * It passes if `union_ttbng` ship AP moves by at most 0.0010.
  * N4's fp16 check covered fog, fog_s1 and low light only; this extends it to glare.
  * If the check fails, the new pohang04 blocks are built in fp32. The outcome is written to the ledger
    before any pohang04 glare detection is computed.
* **Fusion:** E4 `union` (`crossmodal26m`, `veto=None`) of the gated VIS stream with the plain IR stream.

## Test P-G

* **Cells:** the nine cells of P-T, plus glare/clean.
* **Gate check:** in every (cell, draw, system), N4g's gate from the frame-statistics files must equal the
  gate computed from the loaded context. Otherwise nothing is reported until the difference is explained.
* **Six cells:** the gates are equal on every draw. There `union_ttbng` is `union_ttbn` exactly, Δ is P-T's
  value, and it is reported as such.
* **Four cells built and scored:** clean/blur_s2, clean/fog_s2, fog/clean, glare/clean.
* **Statistic:** ship AP (class 0, AP50:95) against the VIS ground truth, the mean over the five systems and
  the four draws.
* **Δ:** `union_ttbng` − `union`, paired.
* **Interval:** the 2.5th and 97.5th percentiles of the gate-free scoring's 1,000 moving-block replicates
  (`runs/holdout_p04/gatefree/boot_weights.npy`, L = 20, seed 1).
* **P-G REPLICATES** iff both hold:
  * (a) no cell has Δ ≤ −0.0014 with its interval below zero;
  * (b) glare/clean has Δ ≥ +0.0060 with its interval above zero. This is G-G's condition, and glare/clean
    is pohang04's only glare cell.
* Otherwise **P-G DOES NOT REPLICATE**.
* **If (b) fails while N4g's gate fires on fewer than 10% of glare/clean frames,** the report says the term
  did not act on pohang04's glare. It does not then say that adaptation failed.

**Descriptive, no verdict:**
* gate rates of N4g and of N4 on glare/clean;
* `union_ttbn` − `union` on glare/clean (N4 alone);
* `union_ttbng` − `union_ttbn` in every cell (what the glare term adds);
* Δ against `shipped`;
* the claim, fused − max(VIS, IR), with VIS the gated stream;
* `ttbng_lex` − `lex`;
* on glare/clean, the values of `shipped`, `union`, `lex` and the single streams.

`full_tta` and the other ensemble arms are not built for glare/clean. N4's, N4g's and every earlier verdict
are unchanged.

## Order

1. Run the development fp16 check, and record its outcome in the ledger.
2. Build the plain glare caches and statistics on pohang04: four draws, one at a time.
3. Build the new N4g records (glare blocks and the missing blocks 144–145).
4. Score the presorts: the gate-free arms on glare/clean, then N4's and N4g's arms.
5. Bootstrap, then write the report (`docs/eval/ttbn/ttbn_glare_p04_2026-10-12.md`).

No pohang04 glare value is displayed before all four built cells are scored.
