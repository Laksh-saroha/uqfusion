# Pre-registration — the weak fallback, the clause V2 left out of scope (V3)

**Written 2026-09-04, before any V3 arm has been scored.** V2 returned
INCONCLUSIVE and the veto stands; nothing below reopens that verdict.
`docs/prereg-night-veto-v2.md` and its two stages are closed.

**This is the last registration on this axis.** A stop rule is fixed in §6 so
that "iterate until a clause passes" cannot happen by increments.

## 1. What V2 failed on, and why it is not the health gate

`runs/eval/night_veto_v2_stage1.md`: clause (a) passed at **+0.1785** against a
floor of 0.0054, clause (b) failed on **5 of 11** night cells, day guard clean.
On the four `clean` VIS / degraded IR cells the veto still fired on ~100% of
night frames and V2's delta was **exactly +0.0000** — the rule changed nothing
there, while VIS alone scored 0.2233 against a fused 0.0001–0.1511.

The health instrument was not at fault. Firing decomposition on clean-VIS night
frames, `runs/cache_nv_draw1`, measured **before this document was written**:

| IR condition | `ir_night` | `~ir_ok` | `concentrated` | **fallback fires** | `q < 1` (unhealthy) |
|---|---:|---:|---:|---:|---:|
| clean | 100.0% | 0.0% | 100.0% | **0.0%** | 0.0% |
| glare_s2 | 53.6% | 46.4% | 100.0% | **46.4%** | 0.0% |
| blur_s2 | 0.0% | 100.0% | 100.0% | **100.0%** | 0.0% |
| noise_s2 | 0.0% | 100.0% | 100.0% | **99.4%** | 0.0% |
| fog_s2 | 0.0% | 100.0% | 100.0% | **100.0%** | 0.0% |

`q < 1` is **0.0% everywhere** — the instrument correctly calls a clean night VIS
healthy in every IR condition. What vetoes it is `night_weak_fallback`:

    ir_night_raw AND (NOT ir_ok) AND (concentrated OR (dark AND veil))

A degraded IR fails its authority bound, so `ir_night` collapses and the main arm
— the clause V2 replaced — stops firing. The fallback then fires instead, on
`concentrated`, a VIS-side texture statistic that is **100% true on night frames
in every IR condition** because IR corruption cannot move it.

**V2 replaced the front door and the veto came through the back one.** That
scoping decision was deliberate and stated in V2 ("the registration replaces the
night arm and nothing else"), and it was the right call there — widening a
registration mid-flight is worse than losing a verdict to it. It is named here as
the reason V2 failed, not as a defect discovered afterwards.

## 2. The fallback rests on the same retracted premise as V1

The fallback exists, per its own source comment, because "on fogged night frames
with a blurred IR the arm never fires, **blind VIS** stays in the fusion, and the
cell loses -0.0296". That is a claim about a detector trained on empty night
labels. `docs/experiment-log-2026-09-02.md` §7.2 retired it: night VIS goes
0.0000 → 0.2520 once the 94,553 boxes are restored, and this run measures
VIS-only night at **0.2233**.

So the fallback is a second instance of the V1 error, one clause down. V1 asked
"is it dark?"; the fallback asks "does it look like real night?" Both are proxies
for "is VIS blind", and both stopped tracking it.

## 3. The rule under test

    veto_vis  =  ( ir_night  OR  (ir_night_raw AND NOT ir_ok) )  AND  (q < thr)

Health becomes the **single authority** over whether VIS leaves the fusion; the
two night signals decide only *when the question is asked*. Everything else is
untouched: preset `crossmodal26m`, `iou_thr` 0.85, same IR stream, same VIS
checkpoint (`runs/full_scale/gauss_vis_nightfull/weights/best.pt`), 4 paired
draws (VIS 1/901/902/903, IR 7/911/912/913), n_boot 2000.

**The instrument and threshold are INHERITED, not re-selected.** `q_refit` at the
Youden threshold **1.000000**, exactly as Stage 0 chose and Stage 1 used. There is
no Stage 0 in V3 and no new screening; re-running selection after seeing V2's
cells would be the selection this two-stage structure exists to prevent.

## 4. The bands

Baseline for every delta is **V1 as shipped** (veto ON), the same baseline V2 used,
so V2 and V3 are directly comparable.

`floor = max(2 * se, 0.002)`, `se` the mean of the per-draw paired bootstrap se.

**ADOPT requires ALL THREE:**

* **(a)** draw-averaged night delta `>= +floor` on the clean-VIS / clean-IR cell.
* **(b)** **no night cell below `max(VIS_only, IR_only) - floor`.** Unchanged from
  V2, and the clause this registration exists to satisfy.
* **(c) day false-veto stays at zero.** Across every IR corruption arm, the veto
  must fire on **0.0%** of day frames whose VIS is clean — the rate measured today
  for both V1 and V2. See §5 for why this is a hard requirement in V3 and was not
  in V2.

**REJECT** if (a) fails by `<= -floor`. **INCONCLUSIVE** otherwise, and
INCONCLUSIVE **leaves the veto standing**.

**Day guard.** Every day cell within `floor` of its V1 value; beyond that the run
is VOID rather than interpreted.

## 5. The weakness V3 introduces, stated plainly

**V3 removes a vote.** The shipped system's safety story
(`runs/eval/ir_night_robustness.md`) is that a fogged IR misreads 95% of clear
days as night, and what stops that becoming a false veto of a working VIS is
*two independent votes* plus the authority bound. V3 collapses the VIS side of
that to one instrument. If `q_refit` is wrong on a frame, nothing catches it.

Today's measurement says the margin is there: on day frames with a corrupted IR,
`~ir_ok` runs at **82.1–100.0%** — the authority bound is disarming IR on nearly
every day frame — and the fallback still fires at 0.0% only because
`ir_night_raw` is at most 0.6% there. So under V3 the day protection rests on the
night signal, with health as the second gate rather than the only one. Clause (c)
is what verifies that claim rather than assuming it, and it is a hard ADOPT
requirement precisely because V3 is the version that could break it.

**And the instrument is saturated.** `q = clip(bound/d2, 0, 1)` pins to exactly
1.0 for 100% of clean-night frames and 1.59% of degraded-night ones. It is a
binary novelty-bound test, not a graded score, so it has no margin to degrade
gracefully — it is right or it is wrong, per frame.

## 6. Stop rule

**If V3 does not ADOPT, this axis is closed.** The result is written up as *no
health-gated night rule clears the both-degraded bar on this detector*, and the
veto stands. No V4, no widening of scope to a fourth clause, no re-selection of
the instrument, and no adjustment of the floor or of clause (b).

This is registered because the failure mode of a sequence like V1 → V2 → V3 is
that each round removes exactly the obstacle the last one hit, until something
passes. Two rounds have already gone that way. The difference between that and
science is a stop rule written before the third result is known.

## 7. What an ADOPT verdict does NOT license

**Deploying anything.** Under the shipped `gauss_vis_seed0`, VIS scores 0.0000 at
night and every version of this rule reaches the same answer by a longer route.
V3 is meaningful only with the detector re-baseline priced in
`docs/rebaseline-proposal-2026-09-04.md`, whose night-capable VIS ensemble does
not exist and costs ~40 h — the four `ens_vis_seed{1..4}_ft` runs on dgxanode01
all started 8–10 days before the labels were restored and are void for this
purpose (verified 2026-09-04).

Out of scope and not reopened: V1's and V2's verdicts, the cross-modal gate
mechanism, `cap_ir_scale`, `iou_thr`, the veil repair, and the IR authority bound
itself.

## 8. Weaknesses carried forward

1. **Stage 0 selected `q_refit` on the same 1,032 night frames V3 scores.**
   `pohang01` is the only night run and no split can create a second one. This
   paragraph travels with any ADOPT.
2. **The instrument's negative class was four synthetic corruptions.** It may not
   separate real degradation.
3. **One seed, early-stopped.** `gauss_vis_nightfull` is seed 0, stopped at epoch
   11 of 100 with the LR still annealing.
4. **Fusion at 0.85 is ~99.9% concatenation** — 0.143% of VIS boxes had an IR
   partner on the clean night cell. Any gain is union recall, not consensus, and
   the report must quote the partner rate rather than imply otherwise.
5. **Night val contains zero buoys.** All 16,179 night GT boxes are class 0, so
   the endpoint is ship AP.
