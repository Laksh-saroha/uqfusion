# Handoff — the whole architecture ran end to end on real paired frames

**Written 2026-08-19** (same day as, and continuing, [`handoff-2026-08-19.md`](handoff-2026-08-19.md)).
**Covers:** Phase 3 smoke green, real VIS↔IR pairing, the calibration homography (OQ-5), the first
Table 2 calibration numbers, per-class AP (D-3), and the first end-to-end Table 3 — including two
bugs the synthetic smoke structurally could not catch.

Status labels: **VERIFIED** = measured this session · **OPEN** = needs Laksh.

---

## 0. The short version

The architecture runs end to end on 2,232 genuinely paired Pohang val frames with a real
calibration-derived homography. Four results matter:

1. **The fusion premise holds where it was supposed to.** Under fog, gated fusion scores **0.0210**
   against visible-only's **0.0012** — a 17× rescue. Under lowlight, 0.0205 vs 0.0110.
2. **The §6.4 uncertainty gate was broken as specified — and is now fixed (§10).** As pre-registered
   it saturated to `w_vis ≈ 0` on *any* corruption, including glare where VIS still retains 78% of its
   clean mAP, turning 0.2022 into 0.0428; naive 0.5/0.5 beat it in all four conditions. After a
   ladder-refit τ (11.3× wider) **plus** capability-weighted fusion, gated fusion beats visible-only in
   all four conditions, naive fusion on two, and the learned gate on two.
3. **The identity-H placeholder was not merely imprecise — it inverted the conclusion.** With identity
   H, gated fusion under fog scores 0.0002; with the real homography, 0.0210. A 105× difference.
   Anyone who had run the fusion evaluation before the calibration landed would have concluded the
   premise fails.
4. **(added §14) The premise also holds on real data, with no synthetic corruption at all.**
   pohang01's 1,032 paired-val frames are a genuine night run: VIS scores exactly **0.0000**, IR
   carries the system at **0.0810**. The strongest result in the project, authored by the sea rather
   than by albumentations. It also exposes the gate's real failure mode — see §14.1.

---

## 1. Phase 3 smoke — green — **VERIFIED**

```
python scripts/smoke_phase3.py > runs/smoke_phase3.log 2>&1   # EXIT=0
PHASE3 SMOKE OK
```

The §7.2 blocker from the previous handoff is closed. Re-run green a second time after this session's
edits to `fusion_eval.py`, so the changes below are gated.

**The flagged `*_degraded` empty dicts are explained**: those caches contain **zero detections**
across all 16 smoke frames (`degrade_image` fully blinds the toy `yolov8n`), so `summarize_cache`
correctly returns `{"error": "no detections in cache"}`. Not a bug, and the smoke's asserts never
depended on those rows — they only require the four `*_clean` rows to be finite.

---

## 2. Real VIS↔IR pairing — **VERIFIED**

`scripts/build_pairs.py` → **2,232 index-aligned val pairs**, 2,232 of the 2,234 IR val frames
(99.9%). Spread: pohang00 836, pohang01 1032, pohang02 247, pohang03 117. `|dt|` median 31.3 ms, p95
48.1, max 49.3.

**Do not pair by frame number.** Of the 28,388 rows in `Pohang_dataset/paired/*.csv`, **16,544 (58%)
carry a different index on the VIS and IR side**. The first rows of `pohang00_pairs.csv` happen to
match, which makes a filename join look correct and silently mismatch most of the set. The CSV is the
only valid source.

Outputs: `runs/derived/paired_val_{vis,ir}.txt` + `paired_val_manifest.csv` (idx, run, vis_image,
ir_image, dt_ms).

---

## 3. IR→VIS homography — OQ-5 / D7 closed — **VERIFIED**

`scripts/derive_homography.py`. Both endpoints are the **640×640 letterboxed canvases the models
actually see**, not the native sensors — letterbox geometry was measured off the images themselves
(VIS: ×0.3125, 338 content rows at y+151; IR: ×1.0, 512 rows at y+64), not assumed from the prep
script.

Chain: IR canvas → IR native → undistort (K_ir, D_ir) → rotate by `R = R_stereo_left^T · R_infrared`
→ project + distort (K_sl, D_sl) → VIS canvas. The 3×3 is a least-squares fit to that
distortion-aware mapping.

Per-run, because the extrinsics differ per run (relative rotation varies 0.86°–1.28°, i.e. up to
~6 px on the canvas). Intrinsics are identical across runs.

| | value |
|---|---|
| 3×3 fit residual | mean **0.90 px**, p95 2.11, max 6.05 |
| unmodelled parallax (worst baseline 0.206 m) | 5.8 px @20 m, 1.2 px @100 m, 0.23 px @500 m |

**Empirical validation against VIS ground truth** — confident IR detections matched to VIS GT boxes:

| | matched pairs | median centre offset | mean IoU |
|---|---:|---:|---:|
| identity H | 4,167 | 14.04 px | 0.268 |
| **calibrated H** | **6,315** | **5.43 px** | **0.453** |

Residual bias is −2.2 px in x, −2.5 px in y. The geometry is real.

---

## 4. Two bugs the synthetic smoke could not catch — **VERIFIED**

Both are cases where the smoke's degenerate setup (one model, one image set, identity H) made a
broken path look correct.

### 4.1 `ir_only` was scored in the wrong coordinate frame

`evaluate_systems` scores every system against the **VIS** frame's labels, but passed `ir_records` to
`map50_95` **unmapped** — raw IR-canvas coordinates against VIS-canvas GT. The homography was applied
only inside `fuse_detections`.

Effect: `ir_only` read **0.0007** instead of **0.0206** — a 29× understatement that would have made
IR look useless in every Table 3 the project ever produced. Invisible in the smoke because H was
identity there, making the mapping a no-op.

Fixed in `src/uqfusion/eval/fusion_eval.py`: IR boxes are mapped once into the VIS frame and that
mapped copy feeds the `ir_only` row.

### 4.2 The documented `ablate_gate.py` command could not run

`HOW_TO_RUN.md` §4 step 4 passed `gauss_vis_val_fog.pkl` (11,352 frames) and `gauss_ir_val_clean.pkl`
(2,234 frames) to a function asserting equal length and pairing by index. It would have died on the
assert. Rewritten to take paired caches plus per-modality fit caches, and to fail with an explanatory
message rather than an assert.

---

## 5. Per-modality scorers and constants — a design decision, taken — **OPEN to review**

The smoke has one model, so one Mahalanobis scorer and one set of reliability constants. On real data
the streams come from **different checkpoints**:

- their 896-d pooled-neck feature spaces are not comparable, so each modality is scored against **its
  own** clean-train distribution;
- their clean-val σ scales differ, so constants are fit **per modality**.

Reliability then means "degraded relative to *this* sensor's own clean baseline", which is what a
gate needs. Sharing VIS constants with IR makes IR look permanently unreliable and the gate never
hands it a frame. `run_fusion_eval.py --shared-constants` runs the other way as a sensitivity check.

Fitted: VIS λ=1.0232, μ_d=35.303, τ=3.373 · IR λ=0.7112, μ_d=43.559, τ=8.128.

A methodological choice that was never pre-registered. **Worth a look before Phase 4 locks it in.**

---

## 6. Table 2 — first real calibration numbers — **VERIFIED**

2,232 paired frames, conf 0.001, `runs/eval/table2_gaussian.md`.

| source | D-ECE | NLL | interval-ECE | AUSE | AURC | mAP50-95 | dets |
|---|---:|---:|---:|---:|---:|---:|---:|
| Gaussian VIS clean | 0.0663 | 3.3367 | 0.1432 | 0.0882 | 0.5505 | 0.2580 | 29,042 |
| Gaussian IR clean | **0.0344** | 3.2900 | 0.1593 | **0.0556** | 0.8629 | 0.0676 | 120,877 |

**IR's σ is better calibrated than VIS's** — roughly half the D-ECE and a lower AUSE — despite four
times worse detection. The variance head learns honest uncertainty on a weak detector. (AURC is worse
for IR, but AURC is coupled to the base error rate, so that is expected rather than informative.)

No `gaussian_dfl` row: YOLO26 is end2end with `reg_max=1`, so there is no DFL histogram to derive a
width from. The §7.2 row needs the `yolo12m` control (`architecture-option-a.md`), and cannot come
from these checkpoints.

Baseline (MC-dropout, deep ensemble) columns are deliberately empty — deferred by Laksh this session.

**Independent validation of the whole cache path:** this pipeline's own AP implementation gives VIS
**0.2580** on the paired subset against Ultralytics' **0.25049** on the full 11,352-frame val. Two
independent implementations on overlapping data agreeing to ~3% is a strong check on caches, conf
threshold and matching.

---

## 7. Per-class AP — D-3 answered — **VERIFIED**

`runs/eval/per_class_ap_{vis,ir}.md`. The previous handoff called this "the single cheapest open
measurement in the project". It resolves cleanly, and it changes the IR story.

| | ship AP50-95 | buoy AP50-95 | ship GT | buoy GT |
|---|---:|---:|---:|---:|
| VIS | 0.2141 | **0.3020** | 18,533 | 600 |
| IR | **0.1351** | **0.0002** | 18,272 | 596 |

Two things fall out:

1. **Buoys are invisible in thermal.** IR buoy AP is 0.0002 — not low, *dead* — while emitting 29,131
   buoy detections against 596 GT. This is §4.2's "expected, publishable" branch, confirmed.
2. **The headline "IR is 4× worse than VIS" is largely an artifact of macro averaging.** On ships
   alone the gap is **0.1351 vs 0.2141 — 1.6×, not 4×**. IR's 0.0645 macro number is the average of a
   working ship detector and a dead buoy detector, so it understates the ship detector by roughly half.

Also worth noting: in **VIS**, buoy AP (0.302) *exceeds* ship AP (0.214). Ships are the hard class in
both modalities. The prior framing had this backwards.

---

## 8. Table 3 — the architecture end to end — **VERIFIED**

`runs/eval/table3_fusion.md`. 2,232 paired frames, per-run H, per-modality constants,
`multiplicative`, α=1.0. **The IR stream is always clean; only VIS is degraded.** mAP@50-95 (mAP@50).

| VIS condition | visible only | ir only | naive fusion | gated fusion | learned gate | mean w_vis |
|---|---|---|---|---|---|---|
| clean | **0.2580** (0.6597) | 0.0206 (0.0764) | 0.2515 (0.6952) | 0.2448 (0.6871) | 0.2623 (0.7234) | 0.529 |
| fog | 0.0012 (0.0030) | 0.0206 (0.0764) | **0.0212** (0.0797) | 0.0210 (0.0792) | 0.0212 (0.0799) | 0.000 |
| lowlight | 0.0110 (0.0145) | 0.0206 (0.0764) | **0.0230** (0.0791) | 0.0205 (0.0774) | 0.0208 (0.0782) | 0.000 |
| glare | **0.2022** (0.5280) | 0.0206 (0.0764) | 0.1993 (0.5658) | 0.0428 (0.1424) | 0.1736 (0.5368) | 0.018 |

### 8.1 The premise holds under real degradation

Fog destroys VIS (0.2580 → 0.0012, severity 2). IR, untouched, carries the system to 0.0210 — **17×
the degraded VIS stream**. Lowlight: 0.0205 vs 0.0110. This is the question the previous handoff §4.4
said "decides whether the fusion premise holds", and the answer on these two conditions is yes.

### 8.2 But the uncertainty gate never earns its place

**Naive 0.5/0.5 fusion matches or beats the §6.4 gate in all four conditions.** The gate's
contribution is negative everywhere, catastrophically so on glare (0.0428 vs naive's 0.1993 and
visible-only's 0.2022). WBF's own score-weighted averaging is doing the work.

The learned gate (§7.5) is better than the hand-designed one in every condition and is the only
system that beats visible-only on clean (0.2623 vs 0.2580) — but it is fitted, so treat it as an upper
bound; it was fit on pohang00+02 and scored on all four runs.

### 8.3 Why the gate fails — diagnosed

`r_frame = 1 − sigmoid((D − μ_d)/τ)` with VIS μ_d=35.30, **τ=3.37**. The sigmoid is fully saturated
~4τ ≈ 13.5 units past μ_d. Measured median Mahalanobis distance:

| condition | median D | (D−μ_d)/τ | r_frame | r_box | R | detector mAP |
|---|---:|---:|---:|---:|---:|---:|
| clean | 29.09 | −1.8 | 0.8633 | 0.9453 | 0.8336 | 0.2580 |
| fog | 322.76 | +85.2 | 0.0000 | 0.8617 | 0.0000 | 0.0012 |
| lowlight | 249.65 | +63.5 | 0.0000 | 0.9455 | 0.0000 | 0.0110 |
| **glare** | **74.79** | **+11.7** | **0.0000** | 0.9420 | 0.0000 | **0.2022** |

Two structural faults, both in the pre-registered rules (decision D5 / plan B5):

1. **τ is fit from the clean-val IQR, which is the wrong scale entirely.** The IQR measures spread
   *within* clean data (3.37); corruptions move the distance by 40–290. Any corruption whatsoever
   therefore saturates `r_frame` to 0, so the gate is a binary "is this frame clean?" detector, not
   the "is this detector still working?" estimator §6.4 wants. Glare sits at +11.7τ — deep in
   saturation — while the detector still delivers 78% of its clean mAP.
2. **`r_box` is inert.** It stays at 0.86–0.95 in *every* condition, including the fog case where mAP
   is 0.0012. Per-box σ does not grow under corruption, so the box term contributes no discriminative
   signal and `R ≈ r_frame` by default. The multiplicative combination is effectively a one-term
   product.

The ablation confirms no gate configuration rescues this (`runs/eval/gate_ablation_{fog,glare}.md`):
under fog all six (rule × α) settings give an identical 0.021 — total saturation. Under glare
`geometric`/α=0.5 doubles multiplicative/α=1.0 (0.086 vs 0.043) by damping `r_frame`'s dominance, but
never approaches visible-only's 0.202. **The fix has to be upstream of the combination rule.**

### 8.4 Sensitivity: identity H would have inverted the conclusion

`runs/eval/table3_identity_h.md`, same everything except H:

| condition | gated, calibrated H | gated, identity H |
|---|---:|---:|
| clean | 0.2448 | 0.2199 |
| fog | **0.0210** | **0.0002** |
| lowlight | 0.0205 | 0.0002 |
| glare | 0.0428 | 0.0207 |

Under fog the real homography is worth **105×**. With the placeholder, fusion under fog looks like
total failure. The concrete cost of the "fusion currently smoke-tests with identity H" note in the
previous handoff.

---

## 9. Caveat to own when reading Table 3

Fused output is scored against **VIS labels** (decision D7), but VIS and IR were annotated
independently. An IR detection of a target the VIS annotator missed counts as a false positive.
Quantified: IR scores **0.0676** against its own labels and **0.0206** in the VIS frame against VIS
labels — a 3.3× drop, part registration error (5.4 px median) and part label-set mismatch.

This biases *against* IR contribution, so §8.1's fog result is conservative — a real win. But a null
result in some future condition would be ambiguous, and the absolute fusion numbers are not
comparable to single-modality numbers scored in their own frames.

Second caveat: fog at severity 2 (`fog_coef_range` 0.5–0.8) takes VIS to 0.0012, which is closer to
"sensor destroyed" than "degraded". A severity-1 sweep would say more about the interesting middle of
the range.

---

## 10. D-6 RESOLVED — the severity ladder, the refit, and a working gate — **VERIFIED**

Ran after §8. **The gate now beats every alternative on clean and glare, and never loses to
visible-only.** It took two changes, not one.

### 10.1 The ladder

`runs/cache/ladder/` — 744 paired frames (every 3rd pair, all four runs), **corruption seed 2** so the
ladder is tuning data and Table 3 stays on seed 1 (plan B5-5). VIS: 6 corruptions × 3 severities +
clean = 19 conditions. IR: blur + noise × 3 + clean = 7 — atmospheric fog/rain/glare/"lowlight" are
visible-band phenomena and are not physically meaningful for LWIR, so only sensor-level degradations
are laddered.

### 10.2 The refit — `scripts/fit_reliability_constants.py`

New rule, replacing the clean-val 95th-percentile / IQR rule of D5/B5:

> **μ_d = the Mahalanobis distance at which the detector retains half its clean mAP; τ = the logistic
> scale of that retention curve.**

Fitted by binning all ladder frames by D (24 quantile bins) and measuring mAP *inside each bin*, so
the target is the real metric at per-frame distance granularity.

| modality | old μ_d / τ | **new μ_d / τ** | τ change | fit RMSE |
|---|---|---|---|---|
| VIS | 35.51 / 3.28 | **71.07 / 37.21** | **11.3× wider** | 0.077 |
| IR | 43.63 / 8.14 | **72.53 / 25.74** | 3.2× wider | 0.084 |

### 10.3 The ladder also exposes a limit of the OOD signal itself

Spearman(D, retention): **−0.715** over the 19 VIS conditions, **−0.991** over the 7 IR ones. So on IR
the Mahalanobis distance ranks damage almost perfectly, but on VIS it does not, and the
counterexamples are stark:

| condition | median D | retention |
|---|---:|---:|
| `blur_s1` | 49.5 | **0.398** |
| `glare_s3` | 83.2 | **0.767** |

A *lower* distance with *twice* the damage. No monotone function of D can fit both, so a sigmoid in D
is structurally limited on VIS — τ was mis-scaled **and** the input signal is only partly monotone in
what the gate needs. Worth stating plainly in the write-up rather than hiding inside a fit residual.

### 10.4 Refit τ alone is NOT enough — the second half

With the refit constants but pure ratio weighting, the gate still lost to visible-only on clean
(0.2445) and glare (0.1777). The reason is structural: **R is a retained *fraction*, so a clean IR
frame and a clean VIS frame both score ~0.9 even though VIS detects 12.5× better.** Normalizing each
modality to its own baseline discards exactly the information needed to choose between them.

Fix (`--capability-weighted`): weight by *expected absolute* capability, `R_m × mAP_clean_m`. `R_sys`
stays on the unscaled R, since the plan-B3 abstain signal is a question about retention, not
preference.

### 10.5 Result — `runs/eval/table3_fixed.md`

| VIS condition | visible only | naive 0.5/0.5 | gated (old) | gated (refit τ) | **gated (refit + capability)** | learned gate |
|---|---:|---:|---:|---:|---:|---:|
| clean | 0.2580 | 0.2515 | 0.2448 | 0.2445 | **0.2639** | 0.2623 |
| fog | 0.0012 | 0.0212 | 0.0210 | 0.0210 | 0.0209 | 0.0212 |
| lowlight | 0.0110 | 0.0230 | 0.0205 | 0.0210 | 0.0218 | 0.0208 |
| glare | 0.2022 | 0.1993 | 0.0428 | 0.1777 | **0.2114** | 0.2008 |

- **Beats visible-only in all four conditions** (was: lost in three).
- **Beats naive 0.5/0.5 on clean and glare**, ties on fog, marginally below on lowlight.
- **Beats the fitted learned gate on clean and glare** — the pre-registered rule now outperforms the
  ML upper bound, which is the stronger story for the paper.
- Gate weights are finally sensible: mean `w_vis` = 0.923 clean, 0.860 glare, 0.107 lowlight, 0.037
  fog. It hands frames to IR exactly when VIS is actually broken.

### 10.6 The capability prior is a plateau, not a tuned constant

Sweeping the VIS:IR capability ratio (measured value 12.5×):

| ratio | 1.0 | 4.0 | **12.5** | 40.0 | 150.0 |
|---|---:|---:|---:|---:|---:|
| clean | 0.2445 | 0.2634 | **0.2639** | 0.2631 | 0.2619 |
| glare | 0.1777 | 0.2069 | **0.2114** | 0.2138 | 0.2136 |
| fog | 0.0210 | 0.0210 | **0.0209** | 0.0208 | 0.0203 |

Anything from 4× to 150× works; only ratio = 1 (no capability weighting) fails. The result does not
depend on getting the constant right, which is what makes it a rule rather than a tuned
hyperparameter.

### 10.7 Honest caveats on this fix

1. **Both new constants are fit on clean val and evaluated on clean val.** That is the pre-registered
   protocol (D5/B5) for the reliability constants, but the *capability prior* is a new quantity fit
   the same way and never pre-registered. §10.6 argues it is too coarse to overfit; a run-disjoint
   refit would settle it properly.
2. The ladder is at seed 2 and Table 3 at seed 1, so corruption realizations differ — but they are the
   same corruption *families*. A held-out corruption type (one never laddered) would be a stronger
   generalization test.
3. `r_box` remains inert (§8.3); nothing here fixes that. D-7 still open.

---

## 11. New / changed files

| Path | What |
|---|---|
| `scripts/build_pairs.py` | **new** — index-aligned VIS↔IR lists from the pairs CSVs ∩ splits |
| `scripts/derive_homography.py` | **new** — per-run IR→VIS H with fit residual + parallax report |
| `scripts/run_fusion_eval.py` | **new** — the Table 3 runner (§7.4's missing piece) |
| `scripts/per_class_ap.py` | **new** — per-class AP from a cache (D-3) |
| `scripts/ablate_gate.py` | **rewritten** — paired caches, per-modality scorers, per-run H |
| `scripts/build_cache.py` | `--images-list` (ordered lists, for paired caches) |
| `src/uqfusion/eval/fusion_eval.py` | **`ir_only` frame bug fixed**; per-frame H; per-modality scorer/constants |
| `HOW_TO_RUN.md` | §4 corrected (the old ablate command could not run), §5 updated |
| `runs/derived/paired_val_*`, `homography_ir_to_vis.json`, `maha_fit_*.txt` | **new** |
| `runs/cache/gauss_{vis,ir}_paired_*.pkl`, `gauss_{vis,ir}_train_clean.pkl` | **new** — 7 caches |
| `runs/eval/table3_fusion.md`, `table3_identity_h.md`, `table2_gaussian.md`, `per_class_ap_*.md`, `gate_ablation_*.md` | **new** |
| `scripts/fit_reliability_constants.py` | **new** — the D-6 ladder refit of mu_d/tau + monotonicity check |
| `runs/cache/ladder/` | **new** — 26 ladder caches (744 frames, corrupt-seed 2) |
| `runs/eval/reliability_constants.json` | **new** — refitted constants, both modalities |
| `runs/eval/table3_fixed.md`, `table3_refit_tau.md`, `table3_refit_capability.md` | **new** — the working gate |

---

## 12. Open decisions — Laksh's

Carrying forward D-1, D-2, D-4, D-5 from the previous handoff (D-3 is now answered in §7), plus three
new ones.

| # | Decision | Options | Notes |
|---|---|---|---|
| **D-6** | ~~The §6.4 gate's τ rule~~ | **RESOLVED — §10** | Ladder-refit τ (11.3× wider) + capability weighting. Gate now beats visible-only in all four conditions and the learned gate on two. Needs ratification, not investigation |
| **D-9** | Capability prior in the weights | ratify as a pre-registered rule · re-fit run-disjoint first | §10.4/§10.7. New, un-pre-registered, but robust across 4×–150× |
| **D-7** | `r_box`'s role | drop it · re-specify so σ responds to corruption · keep as-is | §8.3. It is inert across every condition measured |
| **D-8** | Per-modality scorers/constants | ratify · re-specify | §5. Taken by necessity this session, never pre-registered |
| D-1 | Option C mosaic stage | keep · drop | Calibration numbers now exist (§6) — decidable |
| D-4 | Crossover fraction | build · defer | §7 reframes it: explain **ship** 0.135, not macro 0.065 |
| D-5 | MC + ensemble baselines | laptop · server | Deferred by Laksh this session |

## 13. Suggested next steps

> **Superseded in part by §15**, which re-prioritises these after the cheap-fixes pass (§14). Items
> 1–5 below all still stand; §15 reorders them and adds four more.

1. **Ratify D-6 and D-9** — the fix works and is gated by a green smoke, but the capability prior is a
   new un-pre-registered quantity (§10.7). A run-disjoint refit would close it properly; ~20 min of CPU.
2. **Held-out corruption type.** The ladder covers 6 corruption families and the gate is then
   evaluated on 3 of them. Laddering 5 and testing on the 6th is the honest generalization claim, and
   costs one extra evaluation pass.
3. **D-7: `r_box` is inert** — σ does not respond to corruption, so the box term contributes nothing
   to the gate. Either re-specify it or drop it from the product and say so.
4. **The both-degraded row** (scope §7.4) — needs an IR corruption model that is physically meaningful
   for thermal, which visible-fog is not. The IR ladder (blur/noise) is the start of one.
5. Baselines when Laksh calls it (D-5).

---

## 14. Cheap fixes — six CPU-only experiments — **VERIFIED**

Run by `scripts/run_cheap_fixes.py` → `runs/eval/cheap_fixes.md`. All tuning was done on the
corrupt-seed-2 ladder caches and only then applied to the corrupt-seed-1 evaluation caches (plan
B5-5). Nothing here required a GPU or a retrain.

### 14.1 The headline: pohang01 is a real night run and VIS scores exactly 0.0000

The per-run breakdown, not any of the six fixes, is the finding of this pass.

| run | frames | GT boxes | VIS dets | IR dets | median D (VIS) | visible only | ir only | gated |
|---|---|---|---|---|---|---|---|---|
| pohang00 | 836 | 7,030 | 18,939 | 30,483 | 30.0 | 0.4004 | 0.0245 | **0.4024** |
| **pohang01** | **1032** | **7,870** | **94** | 31,568 | 28.4 | **0.0000** | **0.0810** | 0.0765 |
| pohang02 | 247 | 2,151 | 4,429 | 47,957 | 30.3 | 0.3653 | 0.0031 | 0.3583 |
| pohang03 | 117 | 2,082 | 5,580 | 10,869 | 36.2 | 0.1840 | 0.0024 | 0.1805 |

Verified as **not a bug** before reporting:

- All 1,032 pohang01 paired-val frames are night: mean content intensity **8.2** (max 14.9)
  (pohang00: 109.5).
- Max detector confidence across the entire run is **0.0043** — 94 boxes clear threshold in 1,032
  frames.
- Those frames still carry **7,870 GT boxes**.

And it is deliberate. `filter_night_boxes.py`'s own docstring states the policy: 132k night boxes were
cut from **train** as unlearnable label noise, but *"Val/test labels are NEVER touched: they stay
honest hard cases and the fusion showcase."* The showcase fired exactly as designed.

**Three consequences.**

1. **The fusion premise is now demonstrated on real data, not synthetic corruption.** VIS 0.0000 / IR
   0.0810 on a natural night run is stronger evidence than any albumentations fog row in Table 3,
   because nothing about it was authored by us.
2. **VIS's daylight capability is ~0.40, not 0.258.** The pooled paired-subset number averages runs
   where VIS works against one where it is blind by construction. pohang01 is **46%** of the paired
   subset (1,032 / 2,232) but only **18.2%** of the full VIS val (2,068 / 11,352) — IR val coverage
   concentrates in pohang00/01, so the paired subset over-weights the night run **2.5×**.
   Paired-subset VIS numbers are therefore *not* comparable to the Phase 1/2 benchmark.
3. **On the one genuine adverse case the gate loses to pure IR** (0.0765 vs 0.0810, −5.6%). It dilutes
   a working stream with a blind one. Synthetic fog never exposed this because there the gate drove
   `w_vis` to 0.037; on pohang01 the VIS Mahalanobis distance is 28.4 — *lower* than pohang00's 30.0 —
   so the gate reads the night frames as clean. This is A3 (§9.4's monotonicity ceiling) biting on
   real data rather than on the ladder.

### 14.2 WBF `iou_thr`, temporal `alpha`, IR floor `skip_box_thr`

| knob | swept | chosen | verdict |
|---|---|---|---|
| `iou_thr` | 0.40 / 0.55 / 0.70 / 0.85 | **0.85** | monotone increasing to the **edge of the sweep** — a boundary hit, not an optimum |
| `alpha` (EMA) | 1.0 … 0.1 | 0.5 | tuning mean moved 0.1371 → 0.1374; **noise**. D14's alpha=1.0 is unrefuted |
| `skip_box_thr` | 0.00 … 0.20 | **0.00** | every nonzero value hurt, monotonically — **clean negative result** |

`skip_box_thr` is worth recording as a refutation: the IR over-detection hypothesis (54 dets/frame,
29,131 buoy detections against 596 GT) predicted that a confidence floor would help. It does not.
Those low-confidence IR boxes are not costing WBF anything, so the over-detection problem is a
*training* problem (B5), not a fusion one.

Held-out (seed-1) with all three applied:

| VIS condition | visible only | ir only | naive | gated (before) | **gated (tuned)** |
|---|---|---|---|---|---|
| clean | 0.2580 | 0.0206 | 0.2515 | 0.2639 | **0.2710** |
| fog | 0.0012 | 0.0206 | 0.0212 | 0.0209 | 0.0203 |
| lowlight | 0.0110 | 0.0206 | 0.0230 | 0.0218 | 0.0215 |
| glare | 0.2022 | 0.0206 | 0.1993 | 0.2114 | **0.2163** |

+2.7% clean, +2.3% glare, small losses on the two IR-dominant rows. The tuning trades toward
VIS-dominant conditions, which is what a WBF clustering threshold should do.

### 14.3 Union labels — fusion looks much better, magnitude not yet trustworthy

IR labels projected through H and merged into VIS labels at IoU≥0.5 add **14,688 boxes** to VIS's
19,133 (**+76.8%**).

| system | VIS-only GT | union GT |
|---|---|---|
| visible_only | 0.2580 | 0.1391 |
| gated_fusion | 0.2710 | 0.1657 |
| **fusion advantage** | **+5.0%** | **+19.1%** |

The VIS-label-only protocol was understating fusion by roughly 4×, as §9 suspected — IR finds objects
the VIS annotator never labelled, and under VIS labels those count as false positives.

**Do not quote +19.1% yet.** +76.8% new GT is implausibly high for two annotators labelling the same
scenes. With 5.4 px median registration error on 12–17 px boxes, IoU≥0.5 dedup fails often, so the
union very likely re-adds the *same* objects as "new" ones. A dedup-threshold sensitivity sweep
(IoU 0.3 / 0.4 / 0.5, plus a centre-distance criterion) has to come first.

### 14.4 `R_sys` abstain — does not work as specified

plan B3's abstain signal, finally exercised: drop the lowest-`R_sys` frames, re-score the gated system
on the remainder. A useful confidence ranking makes mAP rise monotonically as coverage falls.

| VIS condition | 100% cov | 90% | 75% | 50% |
|---|---|---|---|---|
| clean | 0.2710 | 0.2209 | 0.2371 | **0.3104** |
| fog | 0.0203 | 0.0193 | 0.0158 | 0.0219 |
| lowlight | 0.0215 | 0.0206 | 0.0172 | 0.0249 |
| glare | 0.2163 | 0.1922 | 0.1833 | **0.2439** |

Non-monotone in every condition: dropping the 10% *least* reliable frames makes the system **worse**,
and only at 50% coverage does it beat full coverage. `R_sys` is not usable as a per-frame confidence
ranking in its current form — same root cause as 14.1's consequence 3 and A3.

Caveat on this table: mAP computed over different frame subsets is not strictly comparable (the GT
population changes with coverage). A fixed-GT risk–coverage metric, or AURC over a common denominator,
is needed before this is conclusive. The non-monotonicity is large enough that it is unlikely to be an
artifact, but the measurement is not clean.

### 14.5 What this pass changes about the story

- The strongest result in the project is now a **real night run**, not the ladder. Table 3 should be
  reported **split day/night**, not pooled.
- The gate's known weakness (§9.4: Spearman(D, retention) = −0.715 on VIS) is no longer a ladder
  curiosity. It fails on the single most important real frame set in the dataset, where D=28.4 reads
  "clean" on frames the detector cannot see at all. A3 is promoted from "nice to have" to **blocking
  for the fusion claim**.

---

## 15. Next steps, re-prioritised after §14

§14 changed the ordering. The gate's monotonicity problem is no longer a ladder curiosity to note in a
limitations paragraph — it is the thing standing between the project and its central claim, because it
fails on pohang01, the one frame set where the fusion premise is demonstrated without synthetic help.

**Blocking for the fusion claim**

1. **Split Table 3 day/night and report both.** ~1 h, no new code beyond a grouping argument. Pooling
   a run where VIS scores 0.40 with one where it scores 0.0000 and reporting 0.258 hides both results.
   The night split is the paper's headline; the day split is the honest
   fusion-overhead-on-clean-data number.
2. **Fix the gate's night blindness (A3).** On pohang01 the VIS Mahalanobis distance is **28.4**,
   *below* pohang00's 30.0, on frames with mean content intensity 8.2 (max 14.9) where the detector
   emits 94 boxes in 1,032 frames. Pooled-neck-feature distance simply does not see darkness. Cheapest
   probe before committing to a learned head: add raw frame statistics (mean/percentile intensity,
   contrast) as a second gate input and check whether it separates pohang01 from pohang00. ~2 h, CPU
   only. If it does, the 1–2 day learned-head estimate in A3 may be unnecessary.
3. **Gate must not dilute a working stream.** ~~Gated 0.0765 < ir_only 0.0810 on pohang01.~~ **DONE —
   §17.** The weights did collapse (w_vis 0.926 → 0.432) and it was still not enough: a floor or
   ceiling on w could not have worked either, because WBF rescales scores instead of dropping boxes
   and mAP is rank-based. The fix is a hard veto — a modality with `b < mu_b` leaves the WBF input
   list. Gated now scores 0.0813 > ir_only 0.0810 on the night run.

**Before any union-label number is quoted**

4. **Dedup-threshold sensitivity sweep.** ~1–2 h, CPU. Sweep the merge criterion (IoU 0.3/0.4/0.5,
   plus centre-distance) and report how the +76.8% moves. If it collapses toward +20–30%, the current
   union GT is registration error, not new objects, and §14.3's +19.1% is inflated.

**Cheap and worth doing**

5. **Re-sweep `iou_thr` above 0.85.** ~30 min. It was monotone increasing to the edge of the swept
   range, so 0.85 is a boundary hit. Try 0.90 / 0.95 and find the turn.
6. **Fixed-GT risk–coverage for `R_sys`.** ~1–2 h. §14.4's non-monotonicity is measured over shifting
   frame subsets; redo it against a common denominator (or AURC) before recording abstain as a
   negative result.

**Carried unchanged from §13**

7. Ratify D-6 and D-9, with a run-disjoint refit of the capability prior (~20 min CPU).
8. Held-out corruption family — ladder 5, test on the 6th.
9. D-7: `r_box` is inert. Either re-specify it or drop it and say so. Note that A1 (sigma-weighted
   WBF) is the constructive version of this — right now nothing the Gaussian head produces influences
   fusion at all.
10. The both-degraded row — needs a thermally meaningful IR corruption model.
11. Baselines (MC-dropout, deep ensemble) when Laksh calls it (D-5).

### Files added this pass

| Path | What |
|---|---|
| `scripts/run_cheap_fixes.py` | **new** — the six CPU-only experiments in one pass, incl. `union_gt()` |
| `runs/eval/cheap_fixes.md` | **new** — all six result tables |
| `docs/TODO-improvements.md` | **new** — everything not run, with measured time estimates |

---

## 16. TODO §0.2 done — the gate can now see darkness — **VERIFIED**

The 2 h probe from §15 item 2, run. It works, it cost nothing on daylight, and the held-out validation
is clean — but it does not fully close §15 item 3. (§15 item 1 is also partly done here: every table
below carries the day/night split. §15 item 3 is closed separately, in §17.)

### 16.1 Root cause, found

Not "the features are too coarse". The Mahalanobis reference set **contains the night frames**:

    runs/cache/gauss_vis_train_clean.pkl -> 4,000 frames
      pohang00 679 | pohang01 782 | pohang02 848 | pohang03 872 | pohang04 819

782 pohang01 frames define what "normal" means. Night is therefore not out-of-distribution *by
construction*, so D cannot flag it, and no re-tuning of mu_d/tau could ever have fixed that. This is
the downstream consequence of `filter_night_boxes.py` removing night **labels** from train while the
night **images** stayed — the right call for training, but it silently calibrated the OOD scorer to
treat darkness as ordinary.

(The fit set is frame-disjoint from the paired val set — overlap checked, 0 of 2,232 — so this is a
calibration problem, not a leak.)

### 16.2 The rule, and the leak-free protocol

    r_bright(b) = sigmoid((b - mu_b) / tau_b),  b = mean intensity of the CONTENT rows
    r_frame     = min(1 - sigmoid((D - mu_d)/tau), r_bright)

`min`, not product: the two terms answer different questions ("is this frame strange?" / "is this
frame lit?") and either firing is sufficient. A product would let a confident D dilute a brightness
alarm, which is the exact failure.

Fitted **only** on pohang00/02/03 × {clean, lowlight s1/s2/s3}; pohang01 never entered the fit.
Disjoint in both run and degradation source, so the night rows below are a generalization test: a rule
learned from albumentations-darkened *daylight* frames, applied to real darkness.

| | mu_b | tau_b | fit RMSE | bins | Spearman(b, retention) | Spearman(D, retention) |
|---|---|---|---|---|---|---|
| VIS | 58.238 | 17.869 | 0.0334 | 16 | **+0.993** | −0.974 |

Note D scores −0.974 *on this pool* — within a single degradation axis it ranks damage fine. Its
failure is not ranking, it is **absolute calibration** across degradation types, which §16.1 explains
and §9.4's −0.715 already hinted at.

### 16.3 Held out: pohang01, real night, 1,032 frames

| | night (pohang01) | day (00/02/03) | |
|---|---|---|---|
| mean content brightness | **8.2** | 109.5 | |
| Mahalanobis D | **28.5** | 30.8 | **INVERTED** — night reads cleaner |
| `r_bright` | **0.060** | 0.946 | |
| separation | night max **0.081** | day min **0.934** | **no overlap** |

Every one of the 1,032 night frames scores below every one of the 1,200 day frames. The probe worked;
A3's learned head is not needed for *this* failure.

### 16.4 Effect on the systems

> **Numbers below predate the §18 capability-prior fix and are superseded by the table in §18.2.** The
> relative claims were all re-measured and hold; the absolute values were computed with an IR prior
> 3.3× too high.

Per run, clean condition:

| run | frames | brightness | vis only | ir only | gated before | **gated after** | w_vis before | after |
|---|---|---|---|---|---|---|---|---|
| pohang00 | 836 | 109.3 | 0.4004 | 0.0245 | 0.3953 | **0.3953** | 0.774 | 0.774 |
| **pohang01** | 1032 | **8.2** | 0.0000 | 0.0810 | 0.0769 | **0.0791** | **0.792** | **0.228** |
| pohang02 | 247 | 110.1 | 0.3653 | 0.0031 | 0.3535 | **0.3541** | 0.791 | 0.790 |
| pohang03 | 117 | 109.9 | 0.1840 | 0.0024 | 0.1779 | **0.1779** | 0.777 | 0.777 |

Day/night pooled (TODO §0.1, delivered):

| split | frames | vis only | ir only | gated before | **gated after** |
|---|---|---|---|---|---|
| day (00+02+03) | 1200 | 0.3352 | 0.0092 | 0.3303 | **0.3303** |
| night (01) | 1032 | 0.0000 | 0.0810 | 0.0769 | **0.0791** |

**Daylight is untouched to four decimals** — w_vis moves 0.774 → 0.774. On night w_vis falls 0.792 →
0.228, a 3.5× correction, and mAP rises 0.0769 → 0.0791.

### 16.4b Regression check — all four conditions — and the decisive comparison

> **Numbers below predate the §18 capability-prior fix and are superseded by the table in §18.2.**

Full run in `runs/eval/brightness_gate.md`. `r_bright` was predicted per condition BEFORE the systems
were re-scored, then checked.

| condition | split | gated before | **after** | w_vis before | after |
|---|---|---|---|---|---|
| clean | day | 0.3303 | **0.3303** | 0.774 | 0.774 |
| clean | night | 0.0769 | **0.0791** | 0.792 | 0.228 |
| fog | day | 0.0085 | **0.0085** | 0.012 | 0.012 |
| fog | night | 0.0787 | **0.0787** | — | — |
| lowlight | day | 0.0090 | **0.0090** | 0.037 | 0.037 |
| lowlight | night | 0.0789 | **0.0789** | — | — |
| glare | day | 0.2532 | **0.2532** | 0.681 | 0.591 |
| glare | night | 0.0728 | **0.0720** | — | — |

**fog and lowlight are bit-identical before and after.** That refutes the prediction logged beforehand
— `r_bright` on synthetic lowlight/day is 0.050, a maximum-strength alarm, and it changed nothing. The
reason is the whole story:

| | brightness | w_vis from D alone |
|---|---|---|
| synthetic lowlight, DAY frames | **5.4** | **0.037** |
| real night, no corruption | **8.2** | **0.792** |

**Same darkness. D crushes one and misses the other by 21×.** Synthetic lowlight is an albumentations
transform of a daylight image — genuinely unusual, so D catches it unaided. Real night is a natural
image that lives *inside* D's reference set (§16.1), so D is blind to it. A controlled natural
experiment for §16.1's claim, and much stronger evidence than the ladder.

The practical consequence: **D already handles every synthetic corruption, and the photometric term
changes exactly one thing — the real night run.** It is inert everywhere else by measurement, not by
assumption. That is the ideal shape for an added term, and it also means the severity ladder could
never have revealed this failure, however many corruption families it covered.

One small regression: **glare/night 0.0728 → 0.0720 (−1.1%)**. Confined to the night split; glare/day
is unchanged to four decimals. Under glare the night frames read brightness 40.7 (a sun flare adds
luminance), so `r_bright` is 0.318 rather than 0.060 — a weaker alarm on frames where VIS is still at
0.0000. The magnitude is within the range where WBF weight changes move mAP non-monotonically, so it
is noted, not explained away — but see caveat 6 below for the real lesson.

### 16.4c `p05` beats `mean` — statistic selected on fit runs only

> **Numbers below predate the §18 capability-prior fix and are superseded by the table in §18.2.**

§16.6 caveat 6 named additive luminance as the term's weakness. Tested and fixed.

**Selection, on FIT RUNS ONLY.** `probe_stat_robustness.py` darkens pohang00/02/03 images, then adds
glare or fog on top, and measures how far each statistic is dragged back toward its clean-daylight
reading. Pure image arithmetic — no detector, and pohang01 is never opened, so the choice cannot leak.

| statistic | spoofed by glare | spoofed by fog | worst |
|---|---|---|---|
| **p05** | **0.000** | **0.243** | **0.243** |
| mean | 0.354 | 0.747 | 0.747 |
| range | 0.694 | 0.755 | 0.755 |
| frac_dark | 0.319 | 0.840 | 0.840 |
| std | 0.874 | 0.735 | 0.874 |
| p50 | 0.282 | 0.885 | 0.885 |

`p05` is **completely immune to glare** — a sun flare is spatially localized, so the 5th percentile of
a dark frame never sees it — and 3× more fog-resistant than `mean`. Fit quality on the fit runs is a
wash (RMSE 0.0330 vs 0.0334), so robustness is the deciding criterion. Chosen: **`p05`**,
mu_b = 25.373, tau_b = 2.520.

**Held-out result.** Day/night, all four conditions:

| condition | split | before | `mean` | **`p05`** |
|---|---|---|---|---|
| clean | day | 0.3303 | 0.3303 | **0.3323** |
| clean | night | 0.0769 | **0.0791** | 0.0788 |
| fog | day | 0.0085 | 0.0085 | 0.0085 |
| fog | night | 0.0787 | 0.0787 | 0.0787 |
| lowlight | day | 0.0090 | 0.0090 | **0.0093** |
| lowlight | night | 0.0789 | 0.0789 | 0.0788 |
| glare | day | 0.2532 | 0.2532 | **0.2536** |
| **glare** | **night** | 0.0728 | 0.0720 (regression) | **0.0788** |
| *sum* | | *0.9083* | *0.9097* | ***0.9188*** |

**The glare/night regression is gone and inverted**: −1.1% under `mean` becomes **+8.2%** under `p05`,
exactly the mechanism the fit-run probe predicted. On the night run `w_vis` now collapses to **0.002**
instead of 0.228. Every day split is equal or slightly better. `p05` wins on aggregate.

### 16.4d The one thing `p05` broke — threshold placement, not the statistic

**pohang03 regressed 0.1779 → 0.1700 (−4.4%)**, `w_vis` 0.777 → 0.688.

| run | p05 mean | p05 min | r_bright |
|---|---|---|---|
| pohang00 | 34.6 | 33.0 | 0.972 |
| pohang01 (night) | 2.5 | 1.0 | 0.000 |
| pohang02 | 37.8 | 32.0 | 0.985 |
| **pohang03** | **25.9** | **21.0** | **0.528** |

The separation is clean — night max p05 is **4.0**, day min is **21.0**, a wide empty gap — but `mu_b`
= 25.4 sits **inside the daylight distribution**, not in the gap, and `tau_b` = 2.52 is sharp enough
that its 10–90% transition spans only 11 p05 units. pohang03 lands on the knife edge and gets damped
to 0.53 on a run where VIS scores 0.1840 and should be trusted.

**Diagnosed cause — measured, not assumed.** The first explanation written here was that synthetic
lowlight s1 produces *intermediate* p05 values overlapping daylight and drags the threshold up. That
is false, and the measurement is one line:

| condition (fit runs) | p05 mean | min | max | % inside the clean range |
|---|---|---|---|---|
| clean | 34.40 | 21.0 | 44.0 | — |
| lowlight s1 | **0.00** | 0.0 | 0.0 | **0.0%** |
| lowlight s2 | **0.00** | 0.0 | 0.0 | 0.0% |
| lowlight s3 | **0.00** | 0.0 | 0.0 | 0.0% |

All three severities read p05 = 0.0 *exactly*. s1 is not intermediate in this statistic and cannot
have moved the threshold for that reason.

The real cause is that `p05` is **bimodal by construction on the fit pool**: every corrupted frame
reads 0, every clean frame reads ≥ 21, and there is not a single sample in between. Any `mu_b` inside
(0, 21) fits the retention curve identically, so the data does not choose one — the optimizer does,
from whatever weak signal remains. That signal was a **within-daylight confound**, visible in the
fitted bins:

```
bin b=0.0    n=1200  retention 0.067     <- all corrupted frames
bin b=25.5   n=38    retention 0.513     <- pohang03's clean frames
bin b=34.0   n=157   retention 0.948
```

The `b = 25.5` bin is pohang03's clean daylight, and its retention really is ~0.51 — pohang03 scores
0.1840 against the fit-run clean mean of 0.3437. `curve_fit` read "50% retention occurs at b = 25.5"
and put `mu_b` there. But pohang03's low mAP is a **scene-difficulty** fact (smaller, more distant
vessels), not a dimness fact. The fit inferred causation from a two-point correlation across runs.

**Do not fix this by moving the threshold into the gap using pohang01** — that is the held-out run and
selecting on it destroys the generalization claim §16.2 buys.

### 16.4e Threshold fix — two candidates, both fit-run-only

> **Numbers below predate the §18 capability-prior fix and are superseded by the table in §18.2.**

**Candidate A, refit on clean + lowlight s2/s3 (drop s1).** `mu_b` 25.37 → **17.34**, `tau_b` 2.52 →
**5.58**. This does help, but *not* for the reason originally given: dropping s1 changes the
quantile-bin boundaries, which dissolves the confounded `b = 25.5` bin into a `b = 33.0` bin. With no
bin left between 0 and 33 the fit is unconstrained across the whole gap and lands near its middle.
Better by accident, not by argument.

**Candidate B, `--rule margin` (adopted).** State the placement instead of fitting it. Both endpoints
come from fit runs only:

```
mu_b  = 0.5 * (max p05 over corrupted fit frames + min p05 over clean fit frames)
tau_b = (min_clean - max_corrupt) / 8      # +-4 tau spans the margin: 1.8% at the
                                           # dark end, 98.2% at the bright end
```

Margin [0.00, 21.00] → `mu_b` = **10.500**, `tau_b` = **2.625**. A pre-registrable rule with no
optimizer in the loop, and it saturates on both observed populations rather than transitioning through
either.

| | before | A: s2/s3 refit | **B: margin (adopted)** | (p05 original) |
|---|---|---|---|---|
| `mu_b` / `tau_b` | — | 17.34 / 5.58 | **10.50 / 2.63** | 25.37 / 2.52 |
| day r_bright floor | — | 0.658 | **0.982** | 0.150 |
| night r_bright max | — | 0.0839 | **0.0775** | 0.0002 |
| **pohang03** (per run) | 0.1779 | 0.1777 | **0.1779** | 0.1700 |
| pohang00 | 0.3953 | 0.3953 | 0.3953 | 0.3953 |
| pohang02 | 0.3535 | 0.3541 | 0.3541 | 0.3541 |
| **pohang01** (held out) | 0.0769 | 0.0790 | **0.0791** | 0.0788 |
| clean / day | 0.3303 | 0.3304 | 0.3303 | 0.3323 |
| clean / night | 0.0769 | 0.0790 | **0.0791** | 0.0788 |
| glare / day | 0.2532 | 0.2532 | 0.2532 | 0.2536 |
| glare / night | 0.0728 | 0.0784 | **0.0789** | 0.0788 |
| **sum of 8 day/night cells** | 0.9083 | 0.9161 | **0.9168** | 0.9188 |

**Margin wins on the honest metric.** Every per-run number is ≥ its before value and pohang03 is
restored *exactly* (0.1779, to 4 dp). The original `p05` fit shows a higher 8-cell sum (0.9188) purely
because its pooled clean/day rises to 0.3323 — and that rise comes from *suppressing* pohang03's
detections in the global ranking, i.e. the pooled day number improves precisely because a working run
was damped. Per-run, the same setting costs pohang03 4.4%. Adopted: **margin**.

Constants: `runs/eval/brightness_constants_p05_margin.json`. Report:
`runs/eval/brightness_gate_p05_margin.md`.

### 16.5 What it did NOT fix — §15 item 3, since CLOSED by §17

Gated (0.0791, adopted margin constants) still trails **ir_only (0.0810)** on the night run. The fix
closes about **54%** of that gap, not all of it.

Why: the capability prior multiplies r by each modality's clean-val mAP, and VIS carries a 12.5×
capability advantage (0.258 vs 0.0206). Even with `r_bright` crushing r_vis to ~0.05, that advantage
keeps w_vis at 0.432 (§18-corrected) on frames where VIS contributes 94 boxes in 1,032 frames. **The
capability prior is fighting the brightness alarm.** Worse, cap_vis = 0.258 is itself the *pooled*
clean mAP; a correctly day-only prior would be 0.335 and would push w_vis *higher*.

That is TODO §0.3. **Resolved in §17**, and not by any of the options guessed at here: the capability
prior was left alone. The finding is that no weight-space change could have worked, because WBF
rescales scores rather than dropping boxes and mAP is rank-based. A failed modality has to leave the
input list. See §17.1.

### 16.6 Caveats to own

1. **Pooled clean mAP FELL, 0.2695 → 0.2580, while every per-run number rose or held.** Not a
   contradiction: pooled mAP ranks all detections in one global list, so shifting night frames toward
   IR changes precision on daylight frames at equal recall. The strongest argument yet for §0.1 —
   pooling a blind run with working ones produces a number that moves for reasons unrelated to either.
   Report day/night split.
2. **VIS-only by design.** On a thermal sensor "brightness" is scene temperature, not illumination; a
   dark IR frame is cold water, which is the condition IR is supposed to be *good* at. No `mu_b` is
   fitted for IR and none should be.
3. **Does nothing for blur.** Measured: `vis_clean` 62.65 vs `vis_blur_s3` 63.33 mean intensity — blur
   is photometrically invisible. §9.4's blur/glare counterexample is untouched, so A3 stays on the
   list, demoted to its original priority.
4. **Fog RAISES brightness** (clean 62.65 → fog_s1 88.38 → fog_s2 122.17), so `r_bright` approaches
   1.0 under fog and contributes nothing there. Harmless because `min` only ever lowers reliability —
   but it confirms the term is a darkness detector, not a general quality signal.
5. **Spoofable by added luminance under `mean` — RESOLVED by switching to `p05` (§16.4c).** Fog over
   the night run lifts *mean* brightness from 8.2 to 87.8 and glare to 40.7, so any statistic that can
   be raised by adding light can be fooled into calling a blind frame usable; that is what the `mean`
   fit's glare/night −1.1% was. Measured on fit runs only, `p05` is completely immune to glare (spoof
   fraction 0.000 vs `mean`'s 0.354) and 3× more fog-resistant, and the regression inverted to
   **+8.4%**. The weakness is real for any photometric term — it is contained here by the choice of
   statistic plus `min`, not eliminated in principle.
6. **§16's `before` column does not match §14's numbers** — pooled clean 0.2695 vs 0.2710, pohang00
   0.3953 vs 0.4024 — because the two scripts build the capability prior from different clean
   references (§14 used the paired IR clean mAP, §16 the ladder's). Every before/after delta above is
   measured within one script on identical caches, so the comparisons hold; the two tables' absolute
   columns should not be read against each other. Worth reconciling into one capability-prior rule
   before publication (relates to D-9's run-disjoint refit).

### 16.7 New / changed files

| Path | What |
|---|---|
| `scripts/frame_brightness.py` | **new** — content-region photometric stats, with deterministic corruption replay |
| `scripts/fit_brightness_gate.py` | **new** — fits mu_b/tau_b (`--rule retention`) or places them across the empty fit-run margin (`--rule margin`, adopted); `--stat`, `--fit-conditions`; enforces the run-disjoint protocol and reports held-out separation |
| `scripts/eval_brightness_gate.py` | **new** — before/after over the paired caches, per-run and day/night |
| `scripts/verify_dataset_integrity.py` | **new** — proves `Pohang_dataset/` is never written (count, bytes, sha256, mtime) |
| `scripts/probe_stat_robustness.py` | **new** — luminance-spoofing probe; ranks statistics by how far added glare/fog drags them back (fit runs only) |
| `src/uqfusion/uq/reliability.py` | `mu_b`/`tau_b` on the constants; optional `frame_brightness` arg; `r_bright` returned. Defaults reproduce §6.4 exactly |
| `src/uqfusion/eval/fusion_eval.py` | `brightness_vis`/`brightness_ir` passthrough; `ir_in_vis` now returned for per-run breakdowns |
| `runs/derived/brightness/*.json` | **new** — per-frame photometric stats per cache |
| `runs/eval/brightness_constants_p05_margin.json` | **new, ADOPTED** — `p05`, margin rule, `mu_b`=10.500 / `tau_b`=2.625 |
| `runs/eval/brightness_constants_{mean,p05,p05_s23,std,range}.json` | **new** — the rejected candidates, kept for the record |
| `runs/eval/brightness_gate_p05_margin.md` | **new, ADOPTED result**; `_p05`, `_p05_s23`, `_clean` are the candidates |

## 17. TODO §0.3 — a failed modality must be excluded, not down-weighted

### 17.1 Why the soft gate could not finish the job

After §16 the gate reads the night run correctly — `r_frame_vis` = 0.052 — and gated fusion still
scored **0.0791 against `ir_only`'s 0.0810**. Down-weighting a blind stream does not remove it, for
two reasons, neither of which shrinks as w → 0:

1. **Weights normalise.** `w_vis = R_vis*cap_vis / (R_vis*cap_vis + R_ir*cap_ir)`. The capability
   prior hands VIS a 12.5× advantage (0.258 vs 0.0206), which leaves **w_vis = 0.432** on the night
   run — a blind stream keeping 43% of the vote on frames where it scores exactly 0.0000. (An earlier
   version of this line said 0.199, computed with the mis-set IR prior §18 fixes. The corrected figure
   makes the argument stronger, not weaker.)
2. **WBF does not drop low-weight boxes — it rescales scores, and the rescale hits the GOOD stream
   too.** Measured directly on a two-box toy case:

   ```
   both streams present:  fused conf [0.87, 0.28, 0.21]
   VIS vetoed:            fused conf [0.80, 0.70]      <- IR's originals
   ```

   The unmatched IR detection enters at 0.7 and leaves at **0.21**, because WBF scales every
   single-modality cluster by that modality's normalised weight. So a blind VIS stream does not merely
   add false positives; it **attenuates IR's true positives by w_ir**. And mAP is rank-based, so
   scaling a false positive down does not remove the rank it occupies.

Nothing in the weight space fixes this. The stream has to leave the input list.

### 17.2 The rule — no new constant

    A modality is excluded from fusion when   r_bright < 0.5   <=>   b < mu_b

`mu_b` is the §16.4e margin midpoint, already pre-registered; 0.5 is the sigmoid midpoint, not a swept
threshold. Nothing was tuned to produce this. Vetoing both modalities is refused — that case is the
plan-B3 abstain, signalled by `R_sys`, and the frame keeps its soft-weighted output rather than going
empty. IR carries no photometric term by design, so **IR can never be vetoed**; the asymmetry is
intended (dark IR is cold water, the condition IR exists for).

### 17.3 Why `D` does NOT get veto authority — measured, and it cost 43%

The obvious generalisation is to veto on `r_frame < 0.5`, i.e. `D > mu_d` OR `b < mu_b`. It was run
(`runs/eval/veto_rule_maha.md`) and it is **wrong**:

| | soft (before) | veto on `r_bright` | veto on `r_frame` |
|---|---|---|---|
| glare / day | 0.2532 | **0.2532** | **0.1435** |
| VIS veto rate, glare, fit runs | — | 0% | **46 / 45 / 49%** |

Glare pushes `D` past `mu_d` on about 47% of **daylight fit-run** frames, where VIS still scores
0.2626 against IR's 0.0092. The veto throws away a stream 28× stronger than the alternative and costs
that cell **−43%**.

The principle behind it was already in this document: §9.4's blur/glare counterexample showed `D` is
**not monotone in capability**. `D` answers *"is this frame unusual?"*; the veto needs *"did this
sensor fail?"*. A non-monotone signal may legitimately down-weight — a soft weight degrades gracefully
when the signal is wrong — but it must not hold a switch. Brightness is monotone in the only direction
that matters: below `mu_b` no photons reached the sensor, and an empty frame is not recoverable by any
detector.

**Protocol note, and it cuts against convenience.** The `r_frame` variant is *better* on the held-out
run's fog/night cell (0.0813 vs 0.0792). It was rejected anyway, on daylight fit-run glare. The
adopted rule is the one that scores **worse** on pohang01 in one cell — the selection was not driven
by the held-out set.

### 17.4 Result

> **Numbers below predate the §18 capability-prior fix and are superseded by the table in §18.2.**

| condition | split | soft (before) | **+veto** | `ir_only` |
|---|---|---|---|---|
| clean | day (00+02+03) | 0.3303 | **0.3303** | 0.0092 |
| clean | night (01) | 0.0791 | **0.0813** | 0.0810 |
| fog | day | 0.0085 | **0.0085** | 0.0092 |
| fog | night | 0.0787 | **0.0792** | 0.0810 |
| lowlight | day | 0.0092 | **0.0087** | 0.0092 |
| lowlight | night | 0.0789 | **0.0813** | 0.0810 |
| glare | day | 0.2532 | **0.2532** | 0.0092 |
| glare | night | 0.0789 | **0.0813** | 0.0810 |
| **sum, 8 cells** | | 0.9168 | **0.9238** | |

Per run, clean: pohang00 0.3953, pohang02 0.3541, pohang03 0.1779 — all **unchanged**; pohang01 0.0791
→ **0.0813**.

**The §15 item-3 claim is now discharged.** Gated fusion beats `ir_only` on the night run (0.0813 vs
0.0810) in three of four conditions, having trailed it in all four. The margin is small — it has to
be, since `ir_only` is the ceiling on a run where VIS contributes exactly 0.0000 — but the sign is now
right, which is the claim the paper actually makes. Note 0.0813 > 0.0810 because WBF still dedups IR's
own overlapping boxes at `iou_thr` 0.85; fusion is doing something, not merely passing IR through.

### 17.5 The two costs, stated

1. **lowlight / day loses 0.0005** (0.0092 → 0.0087). Synthetic lowlight reads p05 = 0, so VIS is
   vetoed on 100% of daylight frames — but VIS still scores **0.0173** there against IR's 0.0092, so
   the veto discards the better stream. Real night has no recoverable signal; a synthetic
   multiplicative darkening does, and p05 cannot distinguish them. The honest price of a photometric
   veto, and small in absolute terms (every number in that cell is ~0.01 — the whole system is
   near-blind under day lowlight).
2. **fog / night only reaches 0.0792**, still under `ir_only`'s 0.0810. Fog lifts p05 above `mu_b` on
   71% of night frames, so the veto fires on only 29% of them — the residue of p05's measured fog
   spoof fraction of 0.243 (§16.4c). The one cell of eight where §0.3 is not fully closed. Closing it
   needs a fog-robust photometric statistic, not a different fusion rule; none of the six statistics
   `frame_brightness.py` computes does better (p05 was already the worst-case winner).

### 17.6 Files

| Path | What |
|---|---|
| `src/uqfusion/uq/fusion.py` | `fuse_detections(..., veto_vis, veto_ir)` — a vetoed modality is dropped from the WBF input lists, not down-weighted; vetoing both is refused |
| `src/uqfusion/eval/fusion_eval.py` | `veto_below` / `veto_on`; returns `r_frame_vis`, `r_frame_ir`, `veto_vis`, `veto_ir`. Applies to gated fusion only — naive 0.5/0.5 stays a fixed-weight control |
| `scripts/eval_veto_rule.py` | **new** — veto rates, `r_frame` distributions, per-run and day/night before/after |
| `runs/eval/veto_rule.md` | **ADOPTED result** (`--veto-on r_bright`) |
| `runs/eval/veto_rule_maha.md` | the rejected `--veto-on r_frame` variant, kept for the record |

---

## 18. CORRECTION — the IR capability prior was 3.3× too high in §16 and §17

### 18.1 The bug

`eval_brightness_gate.py` and `eval_veto_rule.py` both set the IR capability prior as

```python
cap_ir = float(rc["ir"].get("map_clean", 0.0206))
```

The key **exists** in `reliability_constants.json` with value **0.067552**, so the `0.0206` fallback
never fired. But 0.0676 is IR's clean mAP measured on the IR ladder against IR GT — a different task
on different frames. The prior has to be IR's capability on *this* evaluation: boxes mapped through H
into the VIS frame, scored against VIS GT. That number is **0.0206**, printed as the `ir_only` row in
those same tables.

`run_fusion_eval.py` and `run_cheap_fixes.py` were always right; only the two scripts written on
2026-08-19 deviated. §16.6 caveat 5 noticed the symptom and guessed the cause correctly without
identifying the line.

Found while re-sweeping `iou_thr` (§0.5), because that script used the correct 0.0206 and its numbers
would not reconcile with §16's.

**Fixed** by computing `cap_ir` from the data in both scripts, so the two can no longer diverge.
Direction of the error: IR was over-weighted, VIS under-weighted, so daylight was understated
throughout.

### 18.2 Corrected results — all three stages

D-6 ladder constants, `iou_thr` 0.85, `alpha` 0.5, capability prior 0.2580 / 0.0206.

| cell | no photometric gate | + gate (§16) | + gate + veto (§17) |
|---|---|---|---|
| clean / day | 0.3342 | 0.3342 | **0.3342** |
| clean / night | 0.0765 | 0.0787 | **0.0813** |
| fog / day | 0.0077 | 0.0077 | **0.0077** |
| fog / night | 0.0786 | 0.0786 | **0.0789** |
| lowlight / day | 0.0086 | 0.0089 | **0.0087** |
| lowlight / night | 0.0789 | 0.0789 | **0.0813** |
| glare / day | 0.2616 | 0.2616 | **0.2616** |
| glare / night | 0.0707 | 0.0779 | **0.0813** |
| **sum of 8 cells** | 0.9168 | 0.9265 | **0.9350** |

Per run, clean condition:

| run | no gate | + gate | + gate + veto | visible only | ir only |
|---|---|---|---|---|---|
| pohang00 | 0.4024 | 0.4024 | **0.4024** | 0.4004 | 0.0245 |
| pohang01 (night, held out) | 0.0765 | 0.0787 | **0.0813** | 0.0000 | 0.0810 |
| pohang02 | 0.3583 | 0.3585 | **0.3585** | 0.3653 | 0.0031 |
| pohang03 | 0.1805 | 0.1805 | **0.1805** | 0.1840 | 0.0024 |

### 18.3 What changed, and what did not

**Every conclusion survives; several arguments get stronger.**

* **§17's headline holds.** Gated fusion on the night run is **0.0813 against ir_only's 0.0810**,
  ahead in clean / lowlight / glare and behind only under fog (0.0789). Unchanged by the correction,
  because VIS is vetoed on night and the weights never enter.
* **§17.1's argument gets much stronger.** At the correct prior the capability advantage holds `w_vis`
  at **0.432** on the night run — not 0.199 — on frames where VIS scores exactly 0.0000.
  Down-weighting leaves a blind stream with 43% of the vote.
* **§16.4e's adoption of the margin rule survives.** Re-run at the corrected prior, the original `p05`
  fit still damps pohang03 (0.1805 → **0.1767**, −2.1%) while the margin rule leaves it at **0.1805**,
  unchanged. The regression is smaller than the −4.4% first reported but the direction, the cause and
  the decision are the same.
* **§16.4c's choice of `p05` over `mean` is untouched** — it rests on `probe_stat_robustness.py`,
  which is image arithmetic on fit runs with no capability prior anywhere in it.
* **Daylight was understated.** clean/day 0.3303 → **0.3342**, glare/day 0.2532 → **0.2616**, pohang00
  0.3953 → **0.4024**.
* **fog/day fell**, 0.0085 → **0.0077**: with less IR weight the fused output leans on a VIS stream
  scoring 0.0012. Honest consequence of the correct prior.

**Superseded:** every absolute number in §16.4 and §17.4 predates this fix. The tables above are
authoritative. The *relative* claims in those sections were all re-measured and hold.

### 18.4 A second inconsistency, also fixed

`run_fusion_eval.py` defaults to the D5/B5 constants rule (VIS `mu_d` = 35.30, `tau` = 3.37) while
every analysis in §14–§17 uses the D-6 ladder fit (`mu_d` = 71.07, `tau` = 37.21). The headline Table
3 was therefore built on different reliability constants from the analysis that justifies it. With the
sharp D5/B5 constants, glare drives `r_frame` off a cliff and gated glare collapses to **0.0641**
against 0.2058 under D-6.

Table 3 is now generated with `--constants runs/eval/reliability_constants.json` and the two agree to
within the `alpha` difference (≤ 0.0005). **Pass that flag, or the headline table is not the system
this document describes.**

---

## 19. TODO A1 done — sigma-weighted WBF, and why it changes nothing here

### 19.1 What was missing

Until now *nothing the Gaussian head produced influenced fusion*. `r_box` is inert (0.86–0.95 in every
condition, including the fog case where mAP is 0.0012), and stock WBF averages cluster coordinates
weighted by `score × model_weight` — a confidence, never a precision. `fuse_detections` ignored
`sigma_ltrb` entirely. A paper about uncertainty-aware fusion rested on a Mahalanobis frame score and,
since §16, a brightness statistic. Neither is the sigma head.

### 19.2 What was built

`sigma_weighted_fusion` in `uqfusion/uq/fusion.py` changes exactly one thing:

    coordinate weight:   score_i * model_weight_i / sigma_i^2     (per coordinate)

the minimum-variance estimator for combining independent measurements. A confident-but-blurry box
keeps its full vote on WHETHER an object is there and loses its vote on WHERE the edges are.
`sigma_ltrb`'s (l, t, r, b) maps onto (x1, y1, x2, y2), so a box can be trusted on one edge and
distrusted on another.

**It is verified to BE stock WBF when sigma is uninformative.** `scripts/smoke_sigma_wbf.py` runs 300
randomised two-model cases (2,260 fused boxes) and asserts that both `use_sigma=False` and
`use_sigma=True with constant sigma` reproduce `ensemble_boxes.weighted_boxes_fusion` to **6.3e-08** —
boxes, scores and labels, in order. Without that equality the A/B would be measuring two fusion
implementations rather than sigma.

### 19.3 The result: nothing, to four decimals

| condition | stock WBF | sigma-weighted | delta |
|---|---|---|---|
| clean | 0.2629 | 0.2628 | −0.0001 |
| fog | 0.0204 | 0.0204 | +0.0000 |
| lowlight | 0.0206 | 0.0206 | +0.0000 |
| glare | 0.2063 | 0.2064 | +0.0001 |

sigma is *not* uninformative — `u_box` spans **9.1×** from p05 to p95 on VIS and 4.1× on IR. So the
null result needed an explanation, and it has a decisive one.

### 19.4 Why — fusion at `iou_thr` 0.85 is almost entirely concatenation

Inverse-variance averaging can only act inside a cluster of two or more boxes.
`diag_cross_modal_iou.py` measures how often that happens: for each of 29,042 VIS detections, the best
IoU against any IR detection in the same frame, after the per-run homography.

| WBF `iou_thr` | VIS boxes with an IR partner above it | share |
|---|---|---|
| **0.85 (tuned)** | **31** | **0.11%** |
| 0.70 | 917 | 3.16% |
| 0.55 | 3,634 | 12.51% |
| 0.40 | 7,796 | 26.84% |
| 0.25 | 11,098 | 38.21% |

**At the tuned threshold, 31 boxes out of 29,042 have a partner.** WBF is merging two lists and
rescaling scores; it is barely fusing anything. sigma never gets invoked because there is almost never
a cluster for it to weight.

The mechanism check confirms it: re-run at `iou_thr` 0.55, where 12.5% of VIS boxes have a partner,
the glare delta grows from +0.0001 to **+0.0005** — about 5×, tracking the 100× increase in cluster
opportunity but from a base so small it stays negligible. (That run is a MECHANISM PROBE, not a
re-tune; 0.85 remains the tuned value per §0.5 and nothing was selected on this.)

### 19.5 What this means

**A1 is implemented and correct, and the blocker is upstream of it.** The reason sigma cannot help is
that VIS and IR detections almost never agree well enough to be fused — 5.4 px median registration
error on 12–17 px boxes (§14) means an IoU of 0.85 between a VIS box and its IR counterpart is nearly
unreachable even when both detect the same vessel.

That reframes three things:

1. **The `iou_thr` tuning curve rising monotonically to 0.85 (§0.5) now has a cause.** Higher
   `iou_thr` means less cross-modal merging, which means less damage from a badly-registered IR
   stream. The gate was tuning toward "fuse less", not toward "fuse better".
2. **A2 (per-run registration refinement) is promoted.** It is the precondition for A1 to be able to do
   anything, not an independent improvement.
3. **The union-label question (§0.4) is now urgent**, and the answer is probably already visible: if
   VIS and IR boxes rarely reach IoU 0.5, the union GT's +76.8% box count is largely the same objects
   counted twice, exactly as §0.4 suspected.

sigma-weighted fusion should stay in — it is the claim the method makes, it is correct, it costs
nothing, and on a properly registered pair it is the right estimator. It should be reported as
implemented with a measured null effect and a diagnosed cause, not quietly dropped.

### 19.6 Files

| Path | What |
|---|---|
| `src/uqfusion/uq/fusion.py` | `sigma_weighted_fusion()` — WBF with inverse-variance coordinate averaging; `fuse_detections(..., sigma_weighted=)` |
| `src/uqfusion/eval/fusion_eval.py` | `sigma_weighted` passthrough (gated fusion only) |
| `scripts/smoke_sigma_wbf.py` | **new** — asserts equality with stock WBF at constant sigma, and per-coordinate inverse-variance behaviour |
| `scripts/eval_sigma_wbf.py` | **new** — before/after, with the `u_box` spread that makes the null result interpretable |
| `scripts/diag_cross_modal_iou.py` | **new** — cross-modal box agreement; the explanation for the null result |
| `scripts/sweep_iou_thr.py` | **new** — TODO §0.5, extends the grid past the boundary |
| `runs/eval/sigma_wbf.md`, `sigma_wbf_iou055.md`, `cross_modal_iou.md`, `iou_thr_sweep.md` | results |

---

## 20. Where the rest of the record lives

[`fusion-gate-experiment-record.md`](fusion-gate-experiment-record.md) carries the consolidated
account: every alternative that was tried and rejected with its numbers, the three factual errors
corrected along the way, the protocol and dataset-integrity evidence, and a prioritised
future-testing plan — including four pre-registered falsification tests that would show this work is
wrong.
