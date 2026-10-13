# R-D1 re-scored on the pre-registered metric: ship AP

**Measured 2026-10-08.** Both pre-registrations name `gated_fusion` **ship AP** as the metric (`prereg-uq-mechanism-ablation.md` §4; `prereg-uq-mechanism-ablation-26m.md` §3). Both recorded runs scored the ship+buoy macro (`out["gated_fusion"]["map50_95"]`). This file re-scores R-D1 on ship AP and changes nothing else. **No pohang04 frame was scored**: the paired val manifest holds pohang00–03 only.

## Verdict

**On the pre-registered metric, the score path is POSITIVE under both presets. The coordinate path stays NULL.** Real σ beats shuffled σ by ≥ 0.0060 with a block-bootstrap CI excluding zero on clean, fog and glare. That is 3 of 4 conditions, which meets the rule (lowlight +0.0040 / +0.0034 falls short). The macro hid this in two ways. On fog and lowlight, buoy AP is 0 in every arm (at most 0.0062 on `crossmodal26m` fog), so the macro delta is half the ship delta, or close to it (crossmodal fog +0.0079 → +0.0040). On clean and glare, real σ re-ranks buoys *worse* than shuffled σ (buoy Δ ≈ −0.0157 clean, −0.0054 glare), which turns the clean macro delta negative and cuts glare's to +0.0005 / +0.0007. The verdict depends on the floor: POSITIVE at 0.0014, 0.0024, 0.0031, 0.0047 and 0.0060; NULL at 0.0100 (0/4 crossmodal, 1/4 26m). The third passing cell clears 0.0060 by only +0.0004 (crossmodal glare) and +0.0008 (26m glare). Against the measured ship floor of 0.0047, the verdict is unchanged. **What POSITIVE does not mean:** on ship AP, every score-path arm is below the system with no σ (S0) on 7 of 8 cells, from point estimates (table 3). Real σ in the score at the untuned α = 1.0 loses less AP than shuffled σ. It is not better than leaving σ out. The one exception is crossmodal fog, S5 − S0 = +0.0032. Under `crossmodal`, VIS is vetoed on every fog frame, so this is IR σ re-ranking IR boxes, not fusion. The pre-registered consequence of POSITIVE is that the shipped preset "should be re-examined for adopting it". Re-examination against S0 already says not to adopt at α = 1.0. **The headline "predicted uncertainty does not improve the fusion (pre-registered null)" is not supported on the registered metric for the score path.**

## Method

`py -3.13 scripts/ablate_uq_mechanism.py --cls 0` (new flag; it scores `per_class[0]["ap50_95"]` in the arm table and passes `cls=0` to `block_bootstrap_delta`; without the flag the script's output is unchanged). Everything else is the same: arms S0–S7, conditions clean/fog/lowlight/glare, caches `runs/cache_m` for **both** presets (the recorded identity blocks say `runs/cache_m`, not `runs/cache`), L = 20, n_boot 1000, seed 0, α = 1.0. Runs at `00e67dc` + the `--cls` diff. Raw output is in `runs/eval/rd1_{repro_macro,ship}_{crossmodal,26m}_2026-10-08.{md,json}`.

**Reproduction check.** The unchanged macro path was re-run first, at the same HEAD. It reproduces both recorded tables **exactly**: 0.000000 difference on all 32 arm APs and all 24 delta/CI cells per preset, and the same verdicts (NULL/NULL). The crossmodal record `runs/eval/uq_mechanism_ablation.md` is char-mangled by the 2026-09-10 `write_md` bug. Decoded, it equals `uq_mechanism_ablation_v2.md`. The permutation audits of the ship runs equal those of the macro runs. The ship bootstrap's observed A equals the arm-table AP to 0. There were no undefined draws.

**Rule, verbatim (prereg §5 + Amendment 1; 26m §3).** "UQ is judged to improve the fusion only if, on at least three of the four conditions: 1. `AP(S1) − AP(S3) ≥` [0.0060 as amended], **and** 2. the paired moving-block bootstrap CI for that delta excludes zero at n_boot 1000 … The same rule is applied independently to the score path (S5 vs S7)." Outcomes are POSITIVE (rule met), NULL, or NEGATIVE (S1 worse than S3 beyond the floor).

## 1. Primary comparisons: macro (recorded) beside ship

| preset | path | condition | macro Δ [95% CI] (recorded) | **ship Δ [95% CI]** | ship CI excl. 0 | ship Δ / 0.0047 |
|---|---|---|---|---|---|---:|
| `crossmodal` | coordinate (S1 − S3) | clean | −0.000566 [−0.000893, −0.000041] | **−0.000103** [−0.000428, +0.000036] | no | −0.02 |
| | | fog | +0.000000 [0, 0] | **+0.000000** [0, 0] | no | 0.00 |
| | | lowlight | −0.000001 [−0.000004, +0.000008] | **−0.000003** [−0.000009, +0.000017] | no | 0.00 |
| | | glare | +0.000406 [−0.000123, +0.000764] | **−0.000003** [−0.000139, +0.000097] | no | 0.00 |
| | score (S5 − S7) | clean | −0.004420 [−0.011030, +0.005829] | **+0.006875** [+0.002661, +0.011151] | **yes** | +1.46 |
| | | fog | +0.003960 [+0.002691, +0.005183] | **+0.007919** [+0.005383, +0.010365] | **yes** | +1.68 |
| | | lowlight | +0.001995 [+0.000944, +0.002684] | **+0.003990** [+0.001887, +0.005368] | **yes** | +0.85 |
| | | glare | +0.000514 [−0.004569, +0.004948] | **+0.006407** [+0.002266, +0.009536] | **yes** | +1.36 |
| `crossmodal26m` | coordinate (S1 − S3) | clean | −0.000578 [−0.000879, −0.000036] | **−0.000126** [−0.000285, +0.000052] | no | −0.03 |
| | | fog | −0.000009 [−0.000039, +0.000090] | **−0.000018** [−0.000078, +0.000179] | no | 0.00 |
| | | lowlight | −0.000001 [−0.000005, +0.000010] | **−0.000002** [−0.000009, +0.000019] | no | 0.00 |
| | | glare | +0.000397 [−0.000124, +0.000761] | **−0.000020** [−0.000139, +0.000097] | no | 0.00 |
| | score (S5 − S7) | clean | −0.004232 [−0.010648, +0.005676] | **+0.007252** [+0.003290, +0.012387] | **yes** | +1.54 |
| | | fog | +0.006156 [+0.003934, +0.007107] | **+0.011957** [+0.009406, +0.014171] | **yes** | +2.54 |
| | | lowlight | +0.001722 [+0.000592, +0.002397] | **+0.003444** [+0.001185, +0.004794] | **yes** | +0.73 |
| | | glare | +0.000724 [−0.004205, +0.005273] | **+0.006825** [+0.003113, +0.011398] | **yes** | +1.45 |

## 2. Counts and verdicts (Δ ≥ F and CI excluding zero; no cell fails in the negative direction at any F)

| preset | path | macro verdict (recorded) | **ship verdict at 0.0060** | ship passing at 0.0014 / 0.0024 / 0.0031 / 0.0047 / **0.0060** / 0.0100 |
|---|---|---|---|---|
| `crossmodal` | coordinate | NULL (0/4) | **NULL** | 0 / 0 / 0 / 0 / **0** / 0 |
| `crossmodal` | score | NULL (0/4) | **POSITIVE** | 4 / 4 / 4 / 3 / **3** / 0 |
| `crossmodal26m` | coordinate | NULL (0/4) | **NULL** | 0 / 0 / 0 / 0 / **0** / 0 |
| `crossmodal26m` | score | NULL (1/4) | **POSITIVE** | 4 / 4 / 4 / 3 / **3** / 1 |

0.0024 and 0.0047 are the ship floor and ship floor × 1.95 (`delta_noise_floor_ship_2026-10-08.md`). The floor's per-cell values exist only for clean (clean/clean 0.0009) and fog (fog/clean 0.0010). Every score-path ship delta is ≥ 3.4× those. The floor is day-only, and R-D1 includes night frames.

## 3. Ship AP by arm, and the score path against no σ (S0)

| preset | arm | clean | fog | lowlight | glare |
|---|---|---:|---:|---:|---:|
| `crossmodal` | S0 shipped | 0.235091 | 0.043589 | 0.058091 | 0.197616 |
| | S5 real, score | 0.215158 | 0.046777 | 0.053934 | 0.182619 |
| | S6 const, score | 0.211417 | 0.040618 | 0.051489 | 0.178180 |
| | S7 shuffled, score | 0.208283 | 0.038858 | 0.049944 | 0.176212 |
| | **S5 − S0** (point) | **−0.019933** | **+0.003188** | **−0.004157** | **−0.014997** |
| `crossmodal26m` | S0 shipped | 0.252393 | 0.087111 | 0.062957 | 0.213847 |
| | S5 real, score | 0.219531 | 0.084250 | 0.054693 | 0.187229 |
| | S6 const, score | 0.217758 | 0.078022 | 0.053546 | 0.184374 |
| | S7 shuffled, score | 0.212279 | 0.072294 | 0.051249 | 0.180404 |
| | **S5 − S0** (point) | **−0.032862** | **−0.002861** | **−0.008264** | **−0.026618** |

Coordinate arms S1–S4 are within 0.0001 of S0 on every cell (full table in the raw `.md`). Buoy Δ(S5 − S7) = 2·macro − ship: crossmodal −0.0157 / 0.0000 / 0.0000 / −0.0054; 26m −0.0157 / +0.0004 / 0.0000 / −0.0054 (clean / fog / lowlight / glare). Block-bootstrap CIs for S5/S6/S7 − S0, ship and macro, come from `scripts/diag_rd1_ship_vs_s0.py` → `runs/eval/rd1_vs_s0_{crossmodal,26m}_2026-10-08.json`. Its arms are bit-identical to the ones above. **Ship S5 − S0 [95% CI, L = 20]:**

| preset | clean | fog | lowlight | glare |
|---|---|---|---|---|
| `crossmodal` | −0.0199 [−0.0275, −0.0126] | **+0.0032 [+0.0012, +0.0046]** | −0.0042 [−0.0051, +0.0014] | −0.0150 [−0.0214, −0.0105] |
| `crossmodal26m` | −0.0329 [−0.0411, −0.0234] | −0.0029 [−0.0056, −0.0010] | −0.0083 [−0.0116, −0.0050] | −0.0266 [−0.0339, −0.0190] |

S5 is below S0 with a CI excluding zero on 6 of 8 cells. Crossmodal lowlight spans zero. Crossmodal fog is above S0 with a CI excluding zero, but VIS is vetoed on every fog frame there, so that cell is IR-only re-ranking. S6 − S0 and S7 − S0 exclude zero below S0 on all 8 cells.

Secondary (ship): S5 − S6 (real vs constant) is positive on all 8 cells and excludes zero on fog and lowlight (crossmodal) and on fog (26m). Under the macro, clean was −0.0132/−0.0141 with the CI excluding zero. That was the paper's "real σ worse than a constant" finding, and on ship it does not hold. S1 − S0 is within 0.0001 on every cell, as for the macro.

## 4. Where the score-path gain comes from (ship AP, both presets)

Descriptive, after the verdict; it cannot change it. `py -3.13 scripts/diag_rd1_fog_score_path.py --cls 0 --conds clean fog glare lowlight [--preset crossmodal]` (new flags; without them the output is unchanged). Same arms, seeds and substitution as table 1. `vis_only` empties every IR record, so whatever S5 − S7 remains there is re-ranking inside the VIS stream. Raw output: `runs/eval/rd1_ship_decomp_{26m,crossmodal}_2026-10-08.txt`. The `all` rows reproduce table 1 exactly for both presets. VIS is vetoed on 1.000 of night frames in every cell, and on 0.000 of day frames except `crossmodal` fog (1.000).

| preset | condition | full, day (1,200) | VIS alone, day | cross-stream remainder, day | full, night (1,032) |
|---|---|---|---|---:|---|
| `crossmodal` | clean | +0.015193 [+0.009164, +0.026658] | +0.015954 [+0.009949, +0.027332] | −0.000761 | +0.008292 [+0.004402, +0.011633] |
| | fog | +0.006261 [+0.003830, +0.008726] | 0 (VIS vetoed) | (IR only) | +0.009434 [+0.005014, +0.013005] |
| | glare | +0.014267 [+0.008571, +0.022976] | +0.015253 [+0.009421, +0.024233] | −0.000985 | +0.009829 [+0.005656, +0.013531] |
| | lowlight | −0.000222 [−0.001454, +0.000240] | +0.000599 [−0.000215, +0.000807] | −0.000821 | +0.010446 [+0.005676, +0.013878] |
| `crossmodal26m` | clean | +0.019430 [+0.010128, +0.029787] | +0.015954 [+0.009949, +0.027332] | +0.003476 | +0.008292 [+0.004402, +0.011633] |
| | fog | +0.012693 [+0.008981, +0.015897] | +0.016660 [+0.010603, +0.020245] | −0.003967 | +0.009434 [+0.005014, +0.013005] |
| | glare | +0.017202 [+0.009173, +0.025286] | +0.015253 [+0.009421, +0.024233] | +0.001949 | +0.009829 [+0.005656, +0.013531] |
| | lowlight | −0.000352 [−0.001570, +0.000061] | +0.000599 [−0.000215, +0.000807] | −0.000951 | +0.010446 [+0.005676, +0.013878] |

* **The gain is within-stream re-ranking, under both presets.** By day, VIS re-ranking alone gives +0.0160 (clean) and +0.0153 (glare) under both presets, and +0.0167 on fog under `crossmodal26m`. Adding the IR stream moves these by −0.0040 to +0.0035, with no consistent sign. The remainder is a difference of two deltas and has no interval. Under `crossmodal`, fog's day part is IR re-ranking, since VIS is vetoed there. At night VIS is vetoed, so the night part is IR re-ranking IR boxes. The night system is identical in every cell (IR is uncorrupted, S5 night = 0.087702 throughout), and its deltas differ only because the IR shuffle draw follows the VIS draw.
* **Lowlight fails by day.** There, real σ does nothing to VIS ranking (+0.0006, CI spans zero). Lowlight's pooled +0.0040 / +0.0034 is the night IR re-ranking, diluted.
* AP does not decompose over slices. The pooled delta (table 1) is not a weighted mean of the day and night deltas, and on clean and glare it is smaller than either.

## 5. What this cannot settle

The pre-registrations' own caveats still apply: development data, one recording, the pre-restore checkpoints, α fixed at 1.0 and never tuned.
