# Gated VIS–IR fusion — architecture, constants, and measured numbers

> **SUPERSEDED IN PART, 2026-09-01 — read this first.** Two later records measure
> things this document states as settled, and disagree with it:
>
> * `docs/veil-veto-and-rule-sweep-2026-09-01.md` — §7.1's veil term is now
>   measured (fog/day +0.0040, fog/night +0.0004, guards unmoved), and the
>   29-rule sweep is reported.
> * `docs/crossmodal-gate-2026-09-01.md` — **§7.2 is wrong, and §6's reading of
>   the Mahalanobis ablation is wrong.** lowlight/day is fixable: it now scores
>   0.0381 against VIS-alone's 0.0346 (+0.0215 [+0.0184, +0.0246] over the
>   system below). The Mahalanobis soft term is not "essentially nothing" — it is
>   the mechanism of that cell's loss, costing −0.0228 once the veto stops hiding
>   it. A `crossmodal` preset lands every one of the eight cells at or above
>   max(VIS, IR).
>
> Everything below still describes the `adopted` preset, which is still the
> default and still reproduces these numbers bit-for-bit. Nothing here has been
> edited or deleted.


**Status 2026-09-01.** Architecture frozen 2026-08-20 (decisions D27, D28/A-1). The
photometric numbers below are measured and stable. A second veto axis (the "veil
term") was added 2026-09-01 and its effect on the tables is **being measured as this
is written** — §7.1 says exactly which numbers are provisional.

---

## 0. GROUND RULES FOR WHOEVER PICKS THIS UP

**Do not delete or overwrite anything.** Not run directories, not caches, not
checkpoints, not `runs/derived/`, not archived or `_broken`/`_bak` files. Several
things here look like junk and are not:

- `runs/mc_dropout/*_broken-20260823/` — a genuinely broken MC arm, but the ONLY
  IR run whose deployed head loaded whole. It is the reference that proves the
  2026-08-31 repair. Keep it.
- `runs/queue_*/state.json`, `control.json` — queue machinery. Deleting one loses
  the resume point for a multi-day run.
- `archive/server_pull_2026-08-31/` — the only local copy of the server results.
- `runs/eval/*_photometric-only.*` — the pre-veil baseline, needed to diff against.
- `Test_1/` — untracked, 177 MB, provenance unclear. **Leave it alone**, do not
  "clean it up".

`runs/` is gitignored, so nothing in it is recoverable from git. Treat every file
under it as irreplaceable. If something genuinely needs to go, say so and let a
human do it.

**Also:** do not resume `runs/queue_ir_benchmark_stride4` (abandoned on purpose, see
`docs/ir-benchmark-closed-2026-09-01.md`) or `queue_vis_benchmark_stride4` on the
server (paused to free the GPU; 85 runs / roughly 51 days remaining).

**Write new results to new filenames.** Re-running an eval over its own output
destroys the comparison you need.

---

## 1. What the system does

Two sensors observe the same scene: **VIS** (visible camera) and **IR** (thermal).
Each has its own YOLO26 detector (`yolo26m` for VIS, `yolo26m-p2feat` for IR). Per
frame, the two detection sets are merged by **Weighted Boxes Fusion** in the VIS
image plane, with per-frame weights supplied by an uncertainty gate.

The claim under test: *a gate that reads per-frame uncertainty beats fixed-weight
fusion and beats either single stream.*

Dataset: Pohang Canal, 2232 paired val frames — **1200 day / 1032 night**. The night
frames are all `pohang01`; VIS mAP there is exactly **0.0000** (max detector
confidence 0.0043 over 1032 frames), so night is really a single-sensor regime.

---

## 2. Data flow

```
VIS frame ──► VIS detector ──► boxes, conf, sigma_ltrb ──┐
     │              │                                     │
     │              └─► pooled neck features ──► D_vis ──► r_frame_vis
     │                                                     │
     └─► frame_brightness.py ──► p05, lap_var ─────────────┼──► VETO (§4.2)
                                                           │
IR frame ───► IR detector ───► boxes, conf, sigma_ltrb ────┤
     │              └─► pooled neck features ──► D_ir ───► r_frame_ir
     │                                                     │
     └─► per-frame homography H(IR→VIS) ───────────────────┤
                                                           ▼
                                          fusion_weights ──► WBF (iou_thr 0.85)
                                                           ▼
                                                     fused detections
```

Both streams are cached to disk first (`runs/cache/gauss_*_paired_*.pkl`), so the
whole gate/fusion sweep is CPU-only replay — no GPU needed to reproduce §6.

---

## 3. Fitted constants

All pre-registered on clean validation data, never on the corrupted test conditions.
Sources: `runs/eval/reliability_constants.json`, `runs/eval/brightness_constants.json`.

| constant | VIS | IR | meaning |
|---|---:|---:|---|
| `lam` | 1.02392 | 0.71113 | box-uncertainty decay; `r_box = exp(-lam*U_box)` |
| `mu_d` | 71.0672 | 72.5245 | Mahalanobis D at 50% clean-mAP retention |
| `tau` | 37.2094 | 25.7364 | logistic scale of that retention curve |
| `map_clean` | 0.26640 | 0.06755 | clean-val mAP |
| `mu_b` | 10.5 | — | brightness veto midpoint (statistic: `p05`) |
| `tau_b` | 2.625 | — | logistic scale of the brightness term |
| `tau_lap` | 508.7 | — | veil veto threshold (statistic: `lap_var`), added 2026-09-01 |

**Capability prior actually used in the tables: VIS 0.3352, IR 0.0092 (ratio 36.2x)** —
refit on the `fit` frame subset, not the `map_clean` row above.

IR carries no photometric and no veil term **by design**: dark IR is cold water, not a
blind sensor. IR can never be vetoed under the default configuration.

`mu_d`/`tau` come from a *retention-curve* fit, not the original D5/B5 rule (95th
percentile / IQR, which gave mu_d 35.5 VIS / 43.6 IR). Both live in the JSON, the
older one under `previous_rule`.

---

## 4. The gate

### 4.1 Soft weights

```
r_frame = 1 - sigmoid(-(D - mu_d)/tau)      # "is this frame unusual?"
r_box   = exp(-lam * U_box)                  # conf-weighted box uncertainty
R       = r_frame * r_box                    # combination rule: multiplicative (D5/B5-3)
w_m     = R_m * capability_m, normalized     # capability restores absolute skill
R_sys   = max(R_vis, R_ir)                   # NOT normalized — the abstain signal
```

`alpha = 1.0`, i.e. temporal EMA smoothing is **off** (D14).

The capability factor matters. R is a *retained fraction*, so a clean IR frame and a
clean VIS frame both score about 0.9 even though VIS detects 36x better. Without it
the weights sit near 0.5/0.5 and IR dilutes a far stronger stream.

### 4.2 The vetoes — this is where the work is

A vetoed modality is **removed from the WBF input list**, not down-weighted. That is
forced, not a preference: WBF rescales each cluster's score by the number of input
lists, so a blind stream's mere *presence* halves the surviving stream's scores.
Down-weighting cannot fix a blind stream — measured, `w_vis` stays at 0.199 on the
night run purely from the capability prior.

Two axes, OR-ed, **each filtered with a different filter**:

```
dark = p05 < mu_b                 → filter: DILATE,   window 15
veil = lap_var < tau_lap          → filter: MAJORITY, window 15
veto_vis = dilate15(dark) OR majority15(veil)
```

**Why two filters and not one — do not "simplify" this.** Dilation only ever *adds*
vetoes. It is right for brightness because brightness UNDER-fires inside a true dark
stretch: fog lifts p05 above `mu_b` on 71% of night frames, so the switch flickers off
where it should hold on. The veil signal has the opposite failure mode — it never
flickers (100% inside both fog cells) but occasionally trips on a single texture-poor
frame. Dilating the OR-ed switch turns glare/day's 4 flagged frames into 58, a 4.8%
veto rate on a guard cell where VIS scores 0.2892 against IR's 0.0177: about −0.013 AP,
more than double what the fog fix is worth. `scripts/smoke_veil_gate.py` check G
asserts this and fails if the filters are merged.

**Why brightness holds the veto and Mahalanobis does not.** D answers "is this frame
unusual?", which is not "can this sensor see?". Glare on daylight fit frames pushes D
past `mu_d` on about 47% of them while VIS still scores 0.2626 against IR's 0.0092, and
vetoing on D costs glare/day 0.2532 → 0.1435. A signal that is not monotone in
capability may down-weight; it must not veto.

### 4.3 Fusion

WBF at `iou_thr 0.85`, IR boxes warped into the VIS plane by a per-frame homography.
`sigma_weighted` (inverse-variance coordinate averaging, so the Gaussian head's sigma
reaches the fused coordinates) exists and is **off** in the headline configuration.

---

## 5. Read per-class, not macro

`map50_95` macro-averages ship and buoy. IR is **nc=1 ship-only** (D28/A-1 — IR buoy AP
measured 0.00019 against 29,131 buoy detections for 596 GT boxes), so buoys can only
come from VIS and fusion acts on the ship half alone. A macro delta is
`(delta_ship + delta_buoy)/2`, a mixture that answers no single question.

This is not academic. Gated vs `visible_only` on clean/day:

| metric | delta | 95% CI | verdict |
|---|---:|---|---|
| **ship AP** | **+0.0031** | [+0.0014, +0.0054] | significant |
| macro mAP | +0.0006 | [−0.0013, +0.0030] | spans zero |

Same frames, opposite conclusion. **Always read the ship column.**

---

## 6. Measured results

Photometric-only gate, n_boot=1000, paired frame-level bootstrap, seed 0.
Full tables: `docs/eval/final_system_2026-09-01.md`.

Ship AP, gated vs `ir_only`:

| cell | VIS ship | IR ship | gated ship | delta | 95% CI | VIS veto |
|---|---:|---:|---:|---:|---|---:|
| clean/day | 0.3683 | 0.0177 | 0.3715 | **+0.3537** | [+0.3445, +0.3630] | 0% |
| clean/night | 0.0000 | 0.0810 | 0.0813 | +0.0003 | spans 0 | 100% |
| **fog/day** | 0.0020 | 0.0177 | 0.0126 | **−0.0051** | [−0.0062, −0.0030] | **0%** |
| fog/night | 0.0000 | 0.0810 | 0.0809 | −0.0001 | spans 0 | 71% |
| **lowlight/day** | 0.0346 | 0.0177 | 0.0166 | −0.0011 | [−0.0018, −0.0004] | **100%** |
| lowlight/night | 0.0000 | 0.0810 | 0.0813 | +0.0003 | spans 0 | 100% |
| glare/day | 0.2892 | 0.0177 | 0.2928 | **+0.2751** | [+0.2643, +0.2867] | 0% |
| glare/night | 0.0000 | 0.0810 | 0.0813 | +0.0003 | spans 0 | 100% |

Ablation — the veto is the only component that earns its place, and only at night:

| variant | cell | ship delta vs adopted |
|---|---|---:|
| `no_veto` | glare/night | **−0.0173** |
| `no_veto` | lowlight/night | −0.0038 |
| `no_veto` | fog/night | −0.0037 |
| `no_veto` | clean/night | −0.0051 |
| `no_maha` (r_frame ≡ 1) | all | within ±0.0033, **positive on day** |
| `cap_only` (r_box ≡ 1 too) | all | within ±0.0032, **positive on day** |

**The Mahalanobis soft term contributes essentially nothing beyond the capability
prior**, and is slightly *harmful* on day cells. That is what "veto-only" (D27) means,
and this ablation re-confirms it independently.

---

## 7. Known problems

### 7.1 fog/day — fix built, effect being measured right now

The photometric gate cannot see fog. fog/day has the **highest p05 of all eight cells**
(58, against clean/day's 35) while VIS ship AP collapses from 0.3683 to 0.0020. Fog is
a veil: it raises brightness and destroys edges at the same time, so a histogram-based
statistic reads it as pristine.

`lap_var` (variance of the Laplacian) separates it cleanly — **fog/day 18, fog/night
38, against ≥1334 for every non-fog cell**. The threshold is a hard novelty bound on
clean fit-run frames only (minimum over pohang00/02/03 clean = 508.7); no corrupted
frame informs it. Switch-level effect, measured:

| cell | veto% before | veto% after |
|---|---:|---:|
| fog/day | 0.0 | **100.0** |
| fog/night | 70.7 | **100.0** |
| clean/day (guard) | 0.0 | **0.0** |
| glare/day (guard) | 0.0 | **0.0** |

**Predicted** AP effect: fog/day recovers about +0.0051, back to `ir_only`. **This is a
prediction, not a result.** The eval producing `runs/eval/final_system_veil.md` was
still running when this was written. Check whether that file exists and diff it against
`runs/eval/final_system_2026-09-01_photometric-only.md` before quoting anything here.

A quantile threshold was tried and rejected: p01 of the clean fit frames (611.9) fires
on 7.0% of the clean/day guard cell, not because those frames are degraded but because
pohang02's own clean footage is less textured than pohang00's (per-run clean minima:
pohang00 2802.8, pohang02 529.9, pohang03 1406.3). A quantile pooled across runs reads
the least-textured clean run as a corruption.

### 7.2 lowlight/day — not fixable, and probably should not be

The veto fires on 100% of frames where VIS (0.0346) beats IR (0.0177); it costs −0.0180
against `visible_only`. No rule fixes it, because on every structure metric tried,
lowlight/day sits BETWEEN two cells that must both be vetoed:

| cell | lap_var | VIS works? | must veto? |
|---|---:|---|---|
| clean/night | 1334 | no (0.0000) | yes |
| **lowlight/day** | **152** | **yes (0.0346)** | **no** |
| lowlight/night | 23 | no (0.0000) | yes |

Real night scenes are harbour lamps on black: large intensity jumps, high `lap_var`, no
visible ship shapes. The metric is fooled and no monotone threshold unfools it.

The corruption is `RandomBrightnessContrast(brightness_limit=(-0.9,-0.7))` — digitally
dimmed *daylight*, i.e. night-level brightness with daylight structure, which no real
sensor emits. Dimming is multiplicative: it scales pixels down but leaves shapes intact
at low amplitude, which is why the detector still works. Real darkness is empty, not
faint. So the gate is being marked down on a physically impossible input.

Current position: document as a synthetic-condition artifact rather than bend the gate
around it. **Argue with this if you can do better** — but any proposal must keep
clean/day and glare/day at exactly 0% veto, and must keep clean/night vetoed.

### 7.3 Other open items

- The learned gate (`src/uqfusion/eval/learned_gate.py`) exists as a baseline and is
  not in the headline table. Whether it beats the hand-built gate is unmeasured at
  scale.
- `sigma_weighted` is off, so the Gaussian head's sigma does not reach the fused
  coordinates in the headline system.
- Night is single-sensor by construction (VIS 0.0000 on all 1032 night frames). Every
  night cell really measures "does the veto hand over to IR correctly", not fusion
  quality.

---

## 8. Reproducing

```bash
python scripts/eval_final_system.py --n-boot 1000 --out runs/eval/NEW_NAME.md
```

```bash
python scripts/fit_veil_gate.py --report-only
```

```bash
python scripts/smoke_veil_gate.py
```

```bash
python scripts/smoke_apmetrics.py
```

`smoke_apmetrics.py` pins the AP implementation and MUST pass before any confidence
interval is quoted. `smoke_veil_gate.py` has 7 checks; A and B prove the veil term
reduces exactly to the old photometric gate when `lap_var` is absent or `tau_lap` is
unset, so the finalized D27 result cannot have moved.

Use `--out` with a **new** filename. Do not overwrite `final_system.md` or the
`_photometric-only` baseline.

### File map

| what | where |
|---|---|
| gate / reliability | `src/uqfusion/uq/reliability.py` |
| WBF + veto | `src/uqfusion/uq/fusion.py` |
| veto hysteresis, both filters | `src/uqfusion/eval/hysteresis.py` |
| eval context, veto composition | `src/uqfusion/eval/ctx.py` |
| AP / matching | `src/uqfusion/eval/matching.py`, `src/uqfusion/eval/apmetrics.py` |
| per-frame photometric stats | `scripts/frame_brightness.py` |
| constants | `runs/eval/reliability_constants.json`, `runs/eval/brightness_constants.json` |
| results | `docs/eval/final_system_2026-09-01.md` |
| decisions | `docs/phase1-experimental-record.md`, `docs/D31-checkpoint-selection-2026-09-01.md` |

---

## 9. Where a fresh pair of eyes would help most

1. **§7.2** — is there a frame statistic that keeps lowlight/day without touching the
   guard cells? The constraint: clean/night (`lap_var` 1334, VIS dead) must stay
   vetoed while lowlight/day (152, VIS alive) must not. I claim no such monotone rule
   exists on histogram or focus statistics.
2. **§6 ablation** — `no_maha` and `cap_only` are within noise of the adopted system
   and *better* on day cells. Is there any cell where the Mahalanobis term earns its
   place? If not, the honest paper drops it and describes a brightness/veil gate plus
   a capability prior, which is a much simpler and more defensible system.
3. **§5** — should the paper report ship-only throughout, given IR is nc=1? The macro
   column currently exists only for continuity with older numbers.
4. **Night cells** are all "spans zero" at ±0.0003. Is there anything to say there
   beyond "the veto costs nothing"?
5. **§4.3** — `sigma_weighted` is off. Does turning it on help, and if not, what does
   that say about the Gaussian head's value to the fusion story?
