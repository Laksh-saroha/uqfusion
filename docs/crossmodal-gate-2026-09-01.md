# The cross-modal gate: every cell at or above the better sensor (2026-09-01)

Raw outputs, all under gitignored `runs/`, which is why the tables are reproduced
here: `runs/eval/final_system_crossmodal.md`, `final_system_adopted_regress.md`,
`architecture_v3.md`, `gate_lab_twosided.md`, `gate_lab_scalefree.md`,
`probe_structure_full.md`, `probe_detector_evidence.md`,
`runs/eval/structure_constants.json`.

**This document does not modify `docs/gated-fusion-handoff.md`** beyond a pointer
added at its head, following the convention set by
`docs/veil-veto-and-rule-sweep-2026-09-01.md`. The handoff is the other chat's
starting point; this file is the record of what replaced it.

---

## 0. The result in one table

Ship AP. Preset `crossmodal` against the adopted system, same caches, same
homography, same WBF, same `iou_thr` 0.85. Paired frame-level bootstrap, n=1000,
seed 0. `bar` = max(VIS, IR): the score the fusion has to beat to have earned its
place.

| cell | VIS | IR | bar | adopted | **crossmodal** | gap to bar | Δ vs adopted | 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| clean/day | 0.3683 | 0.0177 | 0.3683 | 0.3715 | **0.3717** | **+0.0034** | +0.0003 | [+0.0002, +0.0003] |
| clean/night | 0.0000 | 0.0810 | 0.0810 | 0.0813 | **0.0810** | **+0.0000** | −0.0003 | spans 0 |
| fog/day | 0.0020 | 0.0177 | 0.0177 | 0.0166 | **0.0177** | **+0.0000** | +0.0011 | [+0.0004, +0.0018] |
| fog/night | 0.0000 | 0.0810 | 0.0810 | 0.0813 | **0.0810** | **+0.0000** | −0.0003 | spans 0 |
| lowlight/day | 0.0346 | 0.0177 | 0.0346 | 0.0166 | **0.0381** | **+0.0035** | **+0.0215** | [+0.0184, +0.0246] |
| lowlight/night | 0.0000 | 0.0810 | 0.0810 | 0.0813 | **0.0810** | **+0.0000** | −0.0003 | spans 0 |
| glare/day | 0.2892 | 0.0177 | 0.2892 | 0.2928 | **0.2960** | **+0.0068** | +0.0032 | [+0.0024, +0.0041] |
| glare/night | 0.0000 | 0.0810 | 0.0810 | 0.0813 | **0.0810** | **+0.0000** | −0.0003 | spans 0 |

**Worst cell: +0.0000.** This is the first configuration in the project's record
with no cell below `max(VIS, IR)`. The adopted system's worst cell is −0.0180.

Read as two claims, because the two halves are different in kind:

* **Day (4 cells): fusion beats the better single sensor, every time, CI excluding
  zero.** Against `visible_only`: clean +0.0034 [+0.0017, +0.0056], fog +0.0157
  [+0.0127, +0.0188], lowlight +0.0035 [+0.0019, +0.0049], glare +0.0068
  [+0.0049, +0.0082].
* **Night (4 cells): the system is EXACTLY `ir_only`.** Not "+0.0003, spans zero"
  — bit-identical, delta 0.0000, CI [0.0000, 0.0000]. `single_passthrough` means
  a fully vetoed frame returns the surviving stream rather than passing it through
  single-list WBF, so a night cell reduces to IR by construction. "The gate hands
  over to IR at zero cost" is now a statement with no residual to explain.

On **fog/day the gated system IS `ir_only`** (VIS vetoed 100%). Fusion adds
nothing there; it correctly declines to use VIS. That is the honest reading, and
it is the whole of the +0.0157 against `visible_only`.

The adopted preset was re-run after every code change in this session and is
**bit-identical on all eight cells** to its pre-change values
(`final_system_adopted_regress.md` vs `final_system_veil.md`). Nothing here moved
the old system.

---

## 1. What the system is now

```
preset "crossmodal"                             preset "adopted" (unchanged, still default)
─────────────────────────────────────────       ────────────────────────────────────────────
weights   capability prior alone                 capability x r_frame(Mahalanobis) x r_box
veto      grad_gini < 0.4826                     dilate15(p05 < 10.5)
          OR ir_p05 > 34.0                       OR majority15(lap_var < 508.7)
filters   none on either axis                    dilate-15 and majority-15
fusion    single_passthrough=True                single-list WBF on vetoed frames
```

Both are reachable from `load_context(preset=...)`; `"adopted"` remains the
default so every existing script reproduces its published numbers untouched.

Constants, all fitted as hard novelty bounds over CLEAN frames of the fit runs
pohang00/02/03, with pohang01 and all four corrupted conditions held out
(`scripts/fit_structure_gate.py` → `runs/eval/structure_constants.json`):

| constant | value | fires | role |
|---|---:|---|---|
| `grad_gini` | 0.4826 | below | veil / fog |
| `ir_p05` | 34.0 | above | cross-modal night |
| `lap_over_var` | 4.278 | above | real-night, VIS side (fitted, not adopted) |

Each axis is clean enough that no temporal filter is needed — and that is a
measurement, not a simplification:

| axis | clean/day | clean/night | fog/day | fog/night | lowlight/day | lowlight/night | glare/day | glare/night |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `grad_gini` | 0.0% | 0.0% | **100%** | **100%** | 0.0% | 0.0% | 0.0% | 0.0% |
| `ir_p05` | 0.0% | **100%** | 0.0% | **100%** | 0.0% | **100%** | 0.0% | **100%** |
| OR | 0.0% | 100% | 100% | 100% | **0.0%** | 100% | 0.0% | 100% |

The hysteresis the adopted gate needed existed to repair a *flickering* switch —
fog lifts `p05` above `mu_b` on 71% of night frames. That flickering was a
property of reading darkness off the **degraded** sensor. Read off the intact one,
there is nothing to repair.

---

## 2. The three findings, and each one's evidence

### 2.1 The Mahalanobis term is the MECHANISM of the lowlight/day loss

Not "inert", not "within noise" — load-bearing in the wrong direction. Measured
mean `w_vis` per cell (`gate_lab_twosided.md` §0):

| cell | w_vis, full soft gate | w_vis, capability only | VIS ship | IR ship |
|---|---:|---:|---:|---:|
| clean/day | 0.971 | 0.973 | 0.3683 | 0.0177 |
| **lowlight/day** | **0.320** | 0.973 | **0.0346** | 0.0177 |
| **fog/night** | **0.029** | 0.973 | 0.0000 | 0.0810 |
| glare/night | 0.939 | 0.973 | 0.0000 | 0.0810 |

lowlight drives VIS's Mahalanobis D to ~248 against a `mu_d` of 71, so
`r_frame_vis` collapses to ~0.009 while clean IR keeps ~0.82. After the capability
prior that is `w_vis` 0.320 against IR's 0.680 — **the gate hands the frame to the
weaker sensor**, on the one cell where VIS is alive and degraded rather than dead.
Note the glare/night row too: D reads that cell as almost pristine (`w_vis` 0.939)
on frames where VIS scores exactly 0.0000. D answers "is this frame unusual?", and
that question is uncorrelated with "can this sensor see?" in both directions.

The handoff's §6 ablation saw `no_maha` as "within ±0.0033, positive on day"
because it was measured **with the veto still on**, and the veto had already
thrown lowlight/day away — both arms were scoring the same IR-only output. Turn
the veto off on that cell and the term costs **−0.0228** [−0.0257, −0.0194]
(`final_system_crossmodal.md` §3, `with_maha`).

### 2.2 VIS brightness cannot tell a dark world from a dark sensor

Handoff §7.2 asks for a frame statistic that keeps lowlight/day while vetoing
clean/night, and argues none exists because lowlight/day sits *between* two cells
that must both be vetoed. The premise is refutable — `probe_structure_full.md` §2
finds five statistics that separate that pair — but the deeper problem is simpler
and fatal:

| cell | VIS p05 | VIS works? |
|---|---:|---|
| lowlight/day | **0** | yes (0.0346) |
| clean/night | 2.5 | no (0.0000) |
| glare/night | 3.5 | no (0.0000) |

**lowlight/day is photometrically darker than the real night footage, on frames
where the detector still works.** So any statistic monotone in VIS brightness must
rank digitally-dimmed daylight below real night — the exact opposite of the truth.
The axis is not badly thresholded; it is answering the wrong question.

The other sensor answers it. On lowlight/day the IR frame is an ordinary daytime
frame, because the corruption was applied to VIS alone. IR `p05` separates day from
night **completely**: fit-run day 2..34, held-out night 49..113, a margin of +15
grey levels across a bound that no night frame informed. If one sensor reports
darkness and the other reports daylight, the sensor is the anomaly, not the scene.
That is an ordinary cross-modal consistency check, and it is what a two-sensor
system is *for*.

Swapping only the veto rule and leaving everything else alone is worth **+0.0204**
[+0.0175, +0.0233] on lowlight/day and **exactly 0.0000 on all seven other cells**
(`final_system_crossmodal.md` §3, `photometric_veto`).

### 2.3 `grad_gini` is a veil detector that needs no filter

The Gini coefficient of gradient magnitude is scale-free — multiplying every pixel
by k leaves it unchanged — so it reads how CONCENTRATED edge energy is rather than
how much there is. At its novelty bound it fires on 100% of fog/day, 100% of
fog/night, and **0.0% of every other cell**, with no temporal filtering. `lap_var`
reaches the same place only after a majority-15, and the handoff's §4.2 argument
for why the two axes must be filtered *differently* dissolves once neither needs
filtering at all.

---

## 3. What was tried and did NOT work

These cost real compute and are the reason the adopted arm is what it is.

**Detector-evidence VETO — exactly ties, never beats.** Windowed `sum_conf` or
`n_c25` at a novelty bound reproduces the photometric+veil gate on **all eight
cells to four decimals** (`gate_lab_scalefree.md`). One axis, computed from the
detector's own output, no image statistics at all — an appealing simplification,
and worth nothing on this benchmark. Wider windows are strictly worse.

**Per-frame evidence SCALING actively hurts.** Multiplying VIS scores by
`max_conf / thr` costs lowlight/day 0.0381 → 0.0135. The reason is diagnostic:
lowlight/day's AP comes from a broad tail of *low-confidence but correct*
detections, not from the 2.8% of frames with confident ones — vetoing all but
those 2.8% also lands at 0.0166. Any confidence-keyed rule destroys exactly the
detections that cell depends on, and no confidence-keyed rule can separate them
from the junk VIS emits at night.

**Two-sided veto (drop IR) works, and is dominated.** The record's #1 open item.
Vetoing IR reaches lowlight/day 0.0353, just over the 0.0346 bar — so the
hypothesis is confirmed. But fixing the weights reaches **0.0381** on the same
cell, and dropping IR costs clean/day −0.0124 and glare/day −0.0127
(`gate_lab_twosided.md`). Closed: right diagnosis, dominated remedy.

**Pure sensor selection loses.** Forcing exactly one stream per frame gives
clean/day 0.3593 and glare/day 0.2833 against 0.3717 / 0.2960 for fusion. Fusion
genuinely beats selection on the day cells; the system is not a switch.

**`lap_over_var` separates but cannot be used.** It is the one VIS-side statistic
that cleanly splits lowlight/day (0.018..13.52) from clean/night (14.22..24.85),
and it is fitted and stored. It cannot carry the veto because it misses
glare/night (12.4%), and dilating it far enough to catch glare/night takes
lowlight/day down with it (dilate-61: glare/night 0.0806 but lowlight/day 0.0169).

**The `single_passthrough` artifact, quantified.** Single-list WBF clips boxes to
the canvas, merges above `iou_thr`, and re-scores merged clusters. On a vetoed day
cell that is worth −0.0011 (IR 0.0177 → 0.0166); at night it is worth +0.0003.
Same post-process, both signs. Bypassing it is what turns the night cells from
"+0.0003, spans zero" into "exactly `ir_only`".

---

## 4. Limitations — read these before quoting anything above

1. **IR is uncorrupted in all eight cells.** The cross-modal night test is
   therefore graded on the easiest version of its job: it can never be wrong,
   because the sensor it consults is never damaged. A deployment where both
   sensors degrade needs the check run in **both** directions, with an abstain
   when they disagree — `R_sys` already exists as that signal and is not wired to
   this. **This is the single largest threat to the result** and the top open
   item. Nothing in the current cache set can test it: there are no corrupted IR
   paired caches.
2. **IR `p05` is not a temperature.** Pohang IR is 8-bit via per-frame min–max
   normalisation (OQ-3), so `p05` measures where a frame sits in its own thermal
   range; at night the sea/sky contrast collapses and that normalised floor rises.
   It is a real scene statistic, but it is not "the world is cold".
3. **The margin rests on three day runs.** +15 grey levels between a bound fitted
   on pohang00/02/03 and the held-out pohang01. One more daylight run with unusual
   thermal contrast could close it.
4. **lowlight is a physically impossible input.** `RandomBrightnessContrast
   (brightness_limit=(-0.9,-0.7))` is digitally dimmed daylight — night-level
   brightness with daylight structure, which no real sensor emits. The handoff is
   right that the gate was being marked down on an impossible input. The fix is
   not tuned to it: the veto rule contains no lowlight-specific term, the IR axis
   was fitted on daylight fit runs alone, and swapping it moves *only* that cell.
5. **Night is still single-sensor by construction.** VIS is exactly 0.0000 on all
   1032 night frames, so the four night cells measure hand-over, not fusion. They
   are now exactly `ir_only`, which is the best a hand-over can do, and says
   nothing about fusion quality.
6. **yolo26s caches.** As before: this freezes the architecture, not the numbers.
   The full-scale retrain replaces every value in these tables.
7. **The "oracle" figures in the earlier sweep are a lower bound**, not a ceiling
   — coordinate ascent, and it lands below an achievable rule on two night cells.

---

## 5. Reproducing

```bash
python scripts/frame_structure.py --cache runs/cache/gauss_vis_paired_clean.pkl --modality vis
```

```bash
python scripts/fit_structure_gate.py
```

```bash
python scripts/smoke_crossmodal_gate.py
```

```bash
python scripts/eval_final_system.py --preset crossmodal --n-boot 1000 --out runs/eval/NEW_NAME.md
```

`smoke_apmetrics.py` still gates every confidence interval. `smoke_crossmodal_gate.py`
adds eight checks aimed at the failure mode recorded in
`veil-veto-and-rule-sweep-2026-09-01.md` §5 — a feature that silently fails to load
its data and reproduces the other configuration's numbers. Checks B/C/H are
assembly checks for exactly that: they assert what `load_context` actually built,
and that the adopted preset did not move. `load_context` also now prints which
preset is live in its first two seconds.

`smoke_fusion_options.py` pins `score_scale` and `single_passthrough` as inert
when off. It records one incidental finding: `ensemble_boxes.get_weighted_box`
accumulates the score-weighted coordinate average in **float32**, so rescaling all
scores perturbs fused coordinates at ~1e-7 relative. Homogeneity is exact at
powers of two and holds to ~1e-6 otherwise.

### New files

| what | where |
|---|---|
| scale-free frame statistics | `scripts/frame_structure.py` → `runs/derived/structure/` |
| gate constants | `scripts/fit_structure_gate.py` → `runs/eval/structure_constants.json` |
| preset wiring | `src/uqfusion/eval/ctx.py` (`load_context(preset=...)`) |
| soft trust + passthrough | `src/uqfusion/uq/fusion.py`, `eval/fusion_eval.py` |
| smoke | `scripts/smoke_crossmodal_gate.py`, `scripts/smoke_fusion_options.py` |
| rule lab (3 outcomes x 3 weight arms, cached) | `scripts/gate_lab.py` |
| arm comparison | `scripts/eval_architecture_v2.py` |
| probes | `scripts/probe_structure_separation.py`, `scripts/probe_detector_evidence.py` |

---

## 6. Open items, ranked

1. **Corrupt IR and re-run.** Limitation 1 is the result's main exposure. Build
   paired IR caches under blur/noise (the two IR ladder conditions that exist) and
   measure what the cross-modal night test does when the sensor it trusts is the
   broken one. Add the reverse check and an `R_sys` abstain.
2. **Both-degraded cells.** The eight-cell grid never degrades both sensors at
   once, which is the case scope §7.4 says the whole `R_sys` design exists for.
3. **`sigma_weighted` is still off**, so the Gaussian head's sigma still does not
   reach the fused coordinates. Unmeasured under the new preset.
4. **The learned gate** is still not in any headline table.
5. **fog/day headroom.** The gated system is exactly `ir_only` there; the earlier
   ascent found 0.0204 against the 0.0177 bar, so a rule that keeps *part* of VIS
   on fog may still pay ~+0.003.
