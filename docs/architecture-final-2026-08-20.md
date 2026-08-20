# Architecture finalization for full-scale training — 2026-08-20

**What this is.** The yolo26s screening rounds are done
([`fusion-gate-experiment-record.md`](fusion-gate-experiment-record.md),
[`followup-analysis-2026-08-20.md`](followup-analysis-2026-08-20.md)). This
document freezes what full-scale training will train and what the final system
will be, records the pre-registered decision rules for the few items that still
depend on a measurement, and states the publication framing the frozen
architecture supports. The matching code changes are listed in §7 and are in
this commit.

**The one-paragraph verdict.** The fusion layer's defensible core is a
**hard photometric veto with temporal hysteresis** — sensor selection first,
weighting second. The soft photometric gate term is measurably redundant
(followup §3) and the night-superiority claim did not survive its error bars
(followup §1–§2). The finalized-system readout (`runs/eval/final_system.md`,
§2 below) settles the rest: the veto is load-bearing on every night cell, the
Mahalanobis soft weight survives as a residual corrector (it recovers the
fog/night frames the switch misses, at a small disclosed cost on glare/day),
and glare/day emerges as the one cell where gated fusion beats *both* single
streams with a CI excluding zero. The uncertainty machinery's main publication
weight still moves to Table 2 (calibration of the single-pass σ head against
sampling baselines) and the methodological findings; the fusion table becomes
an honest analysis of when selection, weighting, and neither help.

---

## 0. The change, in plain language

The system is **two detectors, one merge rule, one kill switch**: per-modality
yolo26s σ-head detectors (IR = p2feat @640); WBF in the VIS frame with each
sensor weighted by *clear-day capability × how normal the frame looks to it*;
and a hard veto that removes VIS from the merge whenever its content
brightness says no photons arrived. What this finalization changed against the
2026-08-19 system:

| | 2026-08-19 system | Finalized (this doc) | Why |
|---|---|---|---|
| Photometric term | in the soft weight (`min`) AND the veto | **veto only** | the soft copy is redundant — `veto_only` equals gate+veto in every cell (followup §3) |
| Veto timing | per-frame, instantaneous | **sticky: dilate-15 in capture order** | darkness doesn't flicker; fog made the switch flicker and lose fog/night — now closed to parity (followup §4) |
| Capability prior | computed over all paired frames (incl. the held-out night run) | **daytime/fit runs only** | removes the leak; also turned clean/day and glare/day from apparent losses vs VIS into a tie and a win (followup §7; `final_system.md`) |
| IR detector | stock yolo26s neck | **p2feat neck** | +34% IR night, the one screening lever that mattered (followup §2) |
| Headline claim | "gated fusion beats IR-only at night" | **sensor selection: the veto recovers the working sensor; night is parity; glare/day beats both single streams** | the night delta spans zero and inverts under the better IR detector (followup §1–§2) |
| Mahalanobis D | soft weight | soft weight, **kept by measurement** | removal costs fog/night −0.0028 (the frames the switch misses); still never holds the switch |
| GPU launching | `sys.executable` | **`resolve_gpu_python()`, CUDA-probed** | the 19.8× CPU-torch incident (followup §9) |

The governing principle that survived everything: **a signal may hold the
switch only if it is monotone in sensor health.** Brightness is (no photons →
no detections, always). Mahalanobis strangeness is not (glare looks strange
while the detector still works), so it may only nudge weights — giving it the
switch costs glare/day −43%. And weights alone can never rescue a blind
stream: they renormalize, WBF rescales scores instead of dropping boxes, and
mAP is rank-based, so exclusion — not down-weighting — is the correct
mechanism.

## 1. The finalized decision layer

| Component | Frozen setting | Status vs 2026-08-19 | Evidence |
|---|---|---|---|
| Coordinate frame | fuse + evaluate in VIS frame, per-run homography | unchanged | D7 |
| Box merge | WBF, `iou_thr` 0.85, `skip_box_thr` 0 | unchanged | record §5.1 |
| σ-weighted WBF | implemented, available, reported with its diagnosed null | unchanged | record §5.2–§5.4 |
| Capability prior | clean mAP per modality on this evaluation, **fit runs only** (run-disjoint from the held-out run) | **changed** — the all-frames prior included pohang01 | followup §7 |
| Photometric statistic | `p05` of content rows | unchanged | record §3.2 |
| Threshold | margin rule, `mu_b`=10.500, `tau_b`=2.625 | unchanged | record §3.3 |
| Photometric soft term | **deleted from the soft weight** (`bright_soft=False`); `mu_b` now serves the veto only | **changed** — `veto_only` equals gate+veto in every cell | followup §3 |
| Hard veto | VIS excluded when `r_bright < 0.5` (⟺ `b < mu_b`); IR never vetoed | unchanged | record §4 |
| Veto hysteresis | **dilate, k=15, capture order, per run** | **new** — closes fog/night (+0.0020, CI [+0.0007, +0.0028]); guard cell unmoved at every window tried | followup §4 |
| Soft weight | `R = r_maha × r_box`, × capability prior, normalized | unchanged — **kept by measurement** (§2b: removal costs fog/night −0.0028, CI excludes zero) | record §0; `final_system.md` §3 |
| Mahalanobis D | soft weight only, **never a switch** | unchanged (the principle survived everything) | record §4.3 |
| Temporal smoothing | α = 1 (off) | unchanged | D14 |
| Abstain signal | `R_sys` on unscaled R | unchanged | D3 |

**Retired claims and abandoned branches** (do not resurrect without new
evidence): the gated-beats-`ir_only`-on-night claim (spans zero at 11.4% sign
flips; inverts under the p2feat IR detector); the photometric term inside the
soft weight; cross-modal score calibration (fitted, works, nets ~zero);
top-k truncation (null at every k that doesn't hurt); the A2 per-run
registration constant (within-run drift of up to 10 px exceeds the between-run
swing that killed the global constant — a time-varying correction is future
work, and §5.3's "fusion is concatenation at iou 0.85" stands as a limitation).

## 2. What the veto-vs-soft-weight ablation decided (measured)

`scripts/eval_final_system.py` (new) measured the finalized system and three
ablations with paired bootstraps (1000 resamples, seed 0); full tables in
`runs/eval/final_system.md`. Three outcomes:

**a. The veto is the load-bearing component — now with error bars.** `no_veto`
costs every night cell: clean −0.0051, fog −0.0037, lowlight −0.0038, glare
−0.0173, all CIs excluding zero. Day cells identical or within noise.

**b. The Mahalanobis soft weight STAYS — the pre-registered drop condition was
not met.** The rule was: drop it if no soft term helps any cell. Measured:
removing it costs **fog/night −0.0028 [−0.0036, −0.0021]** — exactly the 29%
of fog/night frames the (dilated) veto still misses, where D catches the
synthetic fog and down-weights VIS. It is not free elsewhere: removal *gains*
glare/day +0.0023 [+0.0014, +0.0029] and clean/day +0.0002 (D fires on some
healthy daylight frames). Net across cells ≈ −0.0005 for removal, with
offsetting signs — so the frozen system keeps `R = r_maha × r_box`, and the
paper reports the per-cell trade honestly: *the soft weight is a residual
corrector for what the switch misses, and it pays a small tax where the
sensor is actually fine.* `cap_only` behaves the same as `no_maha` (r_box is
inert here, consistent with the record's r_box findings).

**c. Open item 1 resolved: glare/day is a genuine fusion win; clean/day is a
tie.** Gated vs `visible_only` on day frames: glare/day **+0.0022 [+0.0004,
+0.0042], 1.5% flips** — the one cell where gated fusion beats BOTH single
streams (vs `ir_only` it is +0.2556). clean/day +0.0006 spans zero (18.6%
flips): a tie, not a loss — the record's point-estimate worry that VIS beats
gated on clean/day does not survive the run-disjoint prior. fog/day +0.0055
over VIS but still −0.0026 under `ir_only`; lowlight/day −0.0086 under VIS is
the known synthetic-lowlight veto cost (record §4.5), disclosed. Night cells
sit at parity with `ir_only` (all span zero) with fog/night now −0.0001 —
the hysteresis closed it to statistical parity.

So the defensible fusion sentence is: *selection recovers the working sensor
everywhere; weighting adds a small, real margin on glare/day and costs a
small, disclosed tax on synthetic lowlight; nowhere does the system lose to
the better single stream by more than that disclosed case.*

## 3. The finalized training recipe (full scale)

### 3.1 Frozen now

| Item | Setting | Evidence |
|---|---|---|
| Backbone (both modalities) | `yolo26s` + Gaussian σ head (D26 end2end port) | all paired-fusion evidence is 26s; see §3.3 note on D25 |
| IR neck | **p2feat** variant | `ir_only` night 0.0810 → 0.1087 (+34%, CI [+0.0255, +0.0296]) — the single biggest lever found |
| IR imgsz | **640** (native 640×512) | 960 is +0.0005 last-5 for 2.3× time — noise |
| IR preprocessing | per-frame min–max 8-bit map, **no CLAHE** | CLAHE +0.0006, null |
| rect training | **rejected for IR**; not adoptable anywhere until re-run with shuffling controlled | −0.0060, but confounded: `rect=True` forces `shuffle=False` in ultralytics 8.4.90 |
| VIS imgsz | 640 on the prepared tree now; full-res/1280 stays OQ-11, **out of the full-scale matrix** | OQ-11 cost compounds ×14 runs (§3.4) |
| Epoch budget | 100 epochs, patience 20, seeds per role (D15/D22) | config.yaml |
| IR train stride | 1 (full 23,279 frames — the set is small) | stride-2 was a screening economy |
| VIS train stride | 2 (as approved A2-6) | budget |
| Anti-leakage | pohang01 held out of every fit; constants refit at full scale by the same pre-registered rules (D-6 ladder, margin rule); tune/report corruption seeds split (D23) | record §7 |

### 3.2 Pending, with pre-registered decision rules

1. **IR ship-only (nc=1).** `s4_ir_shiponly` last-5 0.1272 *is* ship AP; the
   comparison it needs is stage-3's per-class ship AP (unmeasured — GPU busy
   with `s4_vis_rect`). Implied stage-3 ship AP ≈ 2×0.0607 − 0.0002 ≈ 0.121, so
   the expected delta is ≈ +0.006. **Rule: adopt nc=1 for IR iff measured ship
   AP (last-5 mean) improves by > +0.003** (the CLAHE-null magnitude is the
   noise yardstick); otherwise keep nc=2. If adopted, the IR class index must
   be remapped so `ship` matches the VIS label space before fusion — IR simply
   stops emitting buoys, which it effectively never did (buoy AP 0.00022).
2. **`s4_vis_rect` (running, 20/25 epochs).** It carries the same
   shuffle-off confound as the IR rect arm and has no matched non-rect control
   at 896. **Rule: treat as exploratory; adopt nothing from it without a
   shuffle-controlled control run.** It does not gate the full-scale launch.
3. **Host / ensemble budget (OQ-12).** ~14 training runs (§3.4) do not fit the
   laptop's wall clock; DGX MIG is the host. M=5 vs M=3 stays Laksh's call —
   the recipe is M-agnostic.

### 3.3 A note on D25 (yolo26m, full-res)

D25 selected `yolo26m` on the full-resolution twin for the *main backbone*.
The full-scale **UQ/fusion matrix** deliberately stays on `yolo26s` at 640:
every paired-fusion number, every screening arm, and the p2feat variant were
measured there, and the matrix multiplies any per-run cost by ~14. Recommended
disposition: one `yolo26m` (or full-res) Gaussian arm as a scale check *after*
the matrix, not under it. This narrows D25's scope and needs Laksh's sign-off.

### 3.4 The run matrix

Per modality: Gaussian σ (1) + MC-Dropout (1, D19) + ensemble members (M=5,
D22 — seed replicates) = 7 runs; ×2 modalities = **14 runs**, plus cache
builds (clean + 6 corruptions × tune/report seeds on val; paired caches for
fusion) and constants refits. IR at ~170 s/epoch × 2 (stride 1) ≈ 9–10 h/run
laptop-scale; VIS stride-2 at 640 is the long pole and prices the DGX slice.

## 4. Baselines (MC-Dropout, Deep Ensembles) — what they inherit

**Yes to every detector-side change; the fusion layer is shared, not
per-method.** Concretely:

- **Same architecture and data**: p2feat neck, imgsz, class set, stride,
  preprocessing — identical to the Gaussian arm. Table 2's claim is
  "single-pass σ is competitive with sampling UQ"; any architecture delta
  confounds it, and a reviewer will find it. MC-Dropout is the D19 protocol on
  this same architecture; the ensemble is seed replicates of it.
- **The veto and hysteresis are UQ-agnostic system components**, not part of
  any UQ method: they read pixel statistics. Every Table 3 row (gated-σ,
  gated-MC, gated-ensemble) runs behind the same veto. Do not "port" the veto
  into the baselines as if it were method surgery — hold the fusion rule fixed
  and swap only the uncertainty source.
- **Cheap and worth it**: once paired caches exist for MC/ensemble, one fusion
  row per baseline is CPU-only and demonstrates the fusion conclusion is
  invariant to the uncertainty source — which is now a *supporting* result for
  the sensor-selection framing, not a competition the σ head must win.

## 5. Publication framing

The "uncertainty-gated fusion beats single modality" story is dead; what
survived is arguably more publishable, because each pillar is mechanistic
rather than a leaderboard delta:

1. **Sensor selection, with a principle.** A hard veto on a *monotone* pixel
   statistic recovers the working sensor with no supervision; the uncertainty
   scorer must not hold the switch because it is not monotone in capability
   (glare/day −43% when it does). Why soft weighting alone cannot rescue a
   blind stream is proved mechanically: weights normalize, WBF rescales
   instead of dropping, and mAP is rank-based. *"A non-monotone signal may
   down-weight; it must never switch"* is the quotable rule — and §2b now
   gives it a constructive half: the soft weight earns its keep exactly on the
   residue the switch misses (fog/night), while fusion proper contributes one
   small CI-clean win (glare/day, beats both single streams).
2. **Corruption ladders cannot find real-shift failures.** Synthetic darkening
   vs real night: 21× different Mahalanobis response, because real night was
   inside the reference set. No ladder, at any severity count, could surface
   it. This is a general warning for the robustness-benchmark literature and
   the strongest single finding.
3. **Single-pass σ vs sampling UQ (Table 2).** The σ head's case is made on
   calibration/compute, not on fusion mAP. Needs the full-scale baselines and
   the σ-calibration check (§6.3) — currently the least-supported pillar.
4. **Disciplined negatives.** Paired bootstraps retired the headline (+0.0003,
   CI spans zero, 11.4% flips); a better IR detector *inverted* it; σ-weighted
   WBF is null because 99.9% of boxes have no cross-modal partner at the tuned
   IoU. Reviewers reward this; it also inoculates against "why is your delta
   0.0003" reviews.

Honest title shape: *"When does uncertainty help multimodal marine detection?
A photometric veto, a blind OOD gate, and calibrated single-pass σ"* — not
"uncertainty-aware fusion".

## 6. Experiments still required (and what each gates)

### 6.1 Before/with the full-scale launch

| # | Test | Cost | Gates |
|---|---|---|---|
| 1 | ~~`eval_final_system.py` full readout~~ **DONE** (`runs/eval/final_system.md`) | — | resolved in §2: Mahalanobis kept, glare/day is a real fusion win, open item 1 closed |
| 2 | Stage-3 per-class ship AP (`perclass_ap.py`) once `s4_vis_rect` frees the GPU | ~min | the nc=1 rule (§3.2-1) |
| 3 | Day-only Mahalanobis reference refit + re-measure D on pohang01 | ~20 min CPU | whether the story is "mis-composed reference set" (falsification test 1); needed for the limitations section either way |
| 4 | Leave-one-fit-run-out on `mu_b` | ~1 h CPU | fragility of the margin rule (one run defines half of it) |

### 6.2 During full-scale (free or scheduled)

5. MC-Dropout + ensemble training per §4 (the matrix itself).
6. σ-calibration check — predicted σ vs realised localisation error on matched
   detections. Precondition for reporting σ-weighted WBF as anything but
   implemented-and-blocked.
7. Per-baseline fusion rows from caches (CPU).

### 6.3 Worth testing for performance, not blocking

8. IR empty-frame ratio (pohang01 is 93.1% empty after the night filter —
   19.5% of VIS train): ablate empty-frame retention on the IR/VIS full runs.
9. Time-varying registration correction (the resurrected A2): only if someone
   wants σ-weighted WBF to matter; a per-run constant is measured insufficient.
10. `yolo26m` scale-check arm (§3.3).
11. Second held-out run: impossible for fusion (pohang04 has no IR), possible
    for VIS-side claims — report as a scoped limitation, not a fix.

## 7. Code changes in this finalization

| File | Change |
|---|---|
| `src/uqfusion/uq/reliability.py` | `bright_soft` flag on `ReliabilityConstants` (default True = record-faithful; finalized system sets False); exponent clipping |
| `src/uqfusion/eval/hysteresis.py` | **new** — `temporal_order`, `filter_veto`, `raw_veto_flags`, `ADOPTED_VETO_FILTER = ("dilate", 15)` |
| `src/uqfusion/eval/ctx.py` | finalized defaults: `capability_sel="fit"`, `bright_soft=False`, `veto_filter=("dilate",15)`; hysteresis applied in `run_systems` via `veto_override`; legacy config documented |
| `scripts/run_fusion_eval.py` | `--capability-runs {fit,all}`, `--bright-soft`, `--veto-dilate` (default 15); 08-19 table reproducible via `--bright-soft --capability-runs all --veto-dilate 1` |
| `scripts/eval_final_system.py` | **new** — §2's measurements |
| `scripts/eval_fusion_ci.py`, `eval_topk_truncation.py`, `eval_score_calibration.py`, `eval_ir_upgrade_fusion.py`, `eval_capability_refit.py`, `eval_veto_hysteresis.py` | pinned to the 2026-08-19 configuration so their recorded outputs stay reproducible |
| `src/uqfusion/config.py` + `config.yaml` | `gpu_python` + `resolve_gpu_python()` (CUDA-probed) — the §9 CPU-interpreter trap closed |
| `scripts/chain_stage4.py` | launches through `resolve_gpu_python()` instead of `sys.executable` |

## 8. Sign-off needed from Laksh

1. §3.3 — the matrix stays on `yolo26s`; `26m`/full-res becomes one follow-up
   arm (narrows D25).
2. §3.2-3 — M=5 vs M=3 for the ensemble, and confirming DGX as host.
3. §5 — the reframing away from "uncertainty-gated fusion wins" (this is a
   thesis-level pivot; the evidence compels it, but it is an owner decision).
