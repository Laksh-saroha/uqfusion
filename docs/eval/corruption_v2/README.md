# Corruption v2: what was wrong with v1, what replaces it, how it was checked (2026-10-09/10)

Follows `docs/handoff-2026-10-09.md` §2. v1 is unchanged and stays the library default
(`make_corruption(..., version="v1")`), so every existing cache still reproduces. v2 is opt-in
(`version="v2"`), writes to separate trees, and is what the paper's corrupted cells now report
(§8; pohang04 in §8.10). **This file describes revision 3** (code revision `86ce9080b2b6`, stamped into every v2 cache
as `corrupt_code`). Revisions 1 and 2 (2026-10-09/10) were never scored; their defects are listed in §3.

## 1. The problem with v1 (confirmed in the albumentations 2.0.8 source)

| v1 condition | Root cause in the library | Effect |
|---|---|---|
| fog | `add_fog` ends with `cv2.GaussianBlur(min(h, w)//30 = 21 px)` at every severity; the veil discs are packed around the image **centre**, not by distance. O(particles × pixels): 234 ms/frame. | Fine detail falls to 1.1–1.5% of clean at all severities. The median VIS target is 12 px tall, so the fog cells measure a 21-px blur. Night fog becomes a bright veil (mean 7.9 → 88). |
| lowlight | `RandomBrightnessContrast(brightness_by_max=True)` is **additive** (`x − β·255`, then clip); the docstring says `x·(1+β)`. | s2 leaves 82% of day pixels black (real night: 32%); night frames become 99.9% black. |
| glare | `flare_roi` defaults to the top half of the 640×640 canvas; the top pad is rows 0–150. Severity changes only the core radius. | Source in the pad on 45% of frames; a sun is painted onto night frames. |
| all | Applied to the letterboxed canvas. | 70–100% of pad pixels changed. |
| IR fog/glare/noise | Visible-light filters on thermal frames; `GaussNoise` is per-channel. | Coloured flare and false-colour noise on grey thermal frames; IR fog raises the mean instead of compressing the range the IR stretch would restore. |

Also found: `uqfusion/eval/identity.py` decoded `git diff` as cp1252, so any UTF-8 diff killed the
reader thread and **every cache built from a dirty tree was stamped `git_dirty: False`** (fixed).

## 2. The rule v2 follows

A synthetic corruption must produce what the real camera plus the dataset's processing chain would
have produced, in physical order:

```
atmosphere (fog) -> optics (glare, glint, motion blur) -> sensor noise
    -> exposure control / ISP / the IR per-frame stretch -> quantisation
```

The frames on disk are at the end of that chain. VIS: the camera's 8-bit output, resized 2048×1080 →
640×338 and letterboxed (rows 151–488 content, verified on 45 frames). IR: 16-bit raw, per-frame
min-max stretched to 8 bit, letterboxed without resize (rows 64–575; every content block spans exactly
0–255, verified on 56 frames). Each condition is undone to the stage it acts on, applied there, and the
later stages are re-applied. Every condition touches the content rows only and is seeded per frame
exactly as v1 (`seed·100003 + index`).

## 3. What revisions 2 and 3 fixed

Revision 2: raised by a review of revision 1 (side session, 2026-10-10) and by the audit in §6.
Revision 3: found by reading the shipped gate's own statistics on revision-2 frames against real
night frames (§6.1), before any v2 detection was scored (exposure ledger, 2026-10-10).

| Defect | Measured | Fix |
|---|---|---|
| **IR fog skipped the dataset's min-max re-stretch** | Fogged IR content kept a median 38.4% of the clean range (p10 20.8%); the real pipeline stretches it back to 0–255, so damage was overstated ~2.6× in contrast. | Min-max is affine, so fog on the 8-bit content followed by a fresh per-frame min-max equals fog on raw. Every IR condition (fog, noise, blur) now ends with the dataset's rule `(x − lo)/(hi − lo)·255 + 0.5`. With β = 0 the output is the input bit-for-bit (15/15 frames). |
| Fog multiplied the clean frame's **sensor noise** by t | IR noise σ 1–2 DN at 640 px; VIS day σ 0.25–0.70 DN (TRAIN NLF) | The removed share is restored after the atmosphere: IR with the frame's own σ (Immerkaer residual, robust median over flat pixels), VIS with the measured TRAIN noise level function at the new level: var = f·(σ(y_new)² − k²σ(y₀)²), k the transform's local slope. Sub-quantisation on VIS day; not on IR. |
| **Motion blur smoothed the frame's sensor noise** (blur acts before readout) | Blur kernels have Σk² = 0.07–0.14 | The kernel's white-noise factor ρ is measured exactly per frame by running the same seeded transform on a **uint8** noise probe (albumentations draws a *different* kernel for float32 input: a float32 probe gave ρ 0.034 where the uint8 impulse kernel has Σk² 0.098), and (1 − ρ) of the noise is restored; IR then re-stretched. |
| **No exposure response to glare** | — | An average-metering AE model (`_ae_gain`: the gain that brings mean(clip(g·J)) to the set point, never > 1) with strength AE_STRENGTH **fitted** to data (§4): 0. |
| **Glare and lamp intensities uncalibrated**; revision-1 night lamps (peak ≤ 1.5) never clipped at all | real TRAIN night lights clip blobs of 100 / 927 / 1,748 px (q90 / q99 / max) | SUN s2 fitted to the one real sun-in-frame event in TRAIN; LAMP fitted to the real night-light blob sizes (§4). |
| **Rev 2: the glare spread's Lorentzian (r⁻²) tail lifted the dark floor of the whole frame** | At matched clipped size (~930 px) the revision-2 lamp put the 5th-percentile grey at 16 (real night lights of that size: 7) and above the veto's `dark` threshold (10.5) on 100% of night frames (real: 0% of 101 lights). Its far-field floor (5th percentile of linear radiance 240–800 px from the light) was 6–9× the real floor. That would have switched the glare/night veto off for a reason absent from real night frames. | The spread is I0 (1 + r²/r0²)^−β with β fitted (§4): **1.5 by night and, independently, 1.5 by day**, one lens tail falling as r⁻³. The glint path is held at its revision-2 intensity (`LAMP_GLINT`, `SUN_GLINT`) so that a steeper core, which needs a higher peak for the same clipped area, does not brighten the reflection. |
| Rev 2: AE strength 0, fitted with a Lorentzian sun | With the tail free the same event prefers AE 0.375 (error 0.056 vs 0.097 at AE 0, 0.217 at 0.75; full AE rejected) | AE_STRENGTH 0.375, applied to glare and fog (one camera); sensitivity rows AE 0 and AE 1 |
| 2.24 cross-channel noise factor **assumed** (independent channels) | measured 2.25 day / 1.83 night | measured values used (demosaicing and the resize correlate the channels at night). |
| Caches carried no code revision | — | `corrupt_code` (sha of `corruptions_v2.py`) in every cache and statistics file; builders refuse, and `corruption_from_meta` refuses to replay, a cache from another revision. |

## 4. Calibration (TRAIN split only; `docs/eval/corruption_v2_calibration/`)

`scripts/calibrate_corruptions_v2.py` (`calibration.json`) and `scripts/calibrate_glare_ae.py`
(`glare_ae.json`, `glare_ae_frames.json`). No validation or test frame tunes a corruption; GT boxes are
read only to measure target range for the fog bands.

* **AE set point.** Median linear content mean of TRAIN day frames: **0.2161** (IQR 0.204–0.225).
* **AE metering.** Over 5,777 TRAIN day frames, sd of log of candidate metering statistics: linear mean
  **0.083**, centre-weighted 0.173, trimmed-90% 0.178, unclipped mean 0.186, log-mean 0.478, median 0.517.
  The camera holds the linear frame mean across scenes: an average-metering AE.
* **AE response to an in-frame source** (AE_STRENGTH), **sun intensity and spread**: the one real sun-in-frame event
  in TRAIN, pohang00 L 18884–19000 (the sun behind thin cloud entering view; the most-clipped frames of the
  run, inspected visually). Clipped area 0 → 12.9%. Radiance around the glow centre (362, 117), after/before:
  ×1.33, 1.53, 1.62, 1.71, 1.38, **1.00**, 2.01 in rings 0–40 … 350–800 px — the far field did not darken.
  The model was fitted on that profile with the source pinned at the glow centre. Revision 2 fitted a
  Lorentzian (1,008 (I0, r0, halo, AE) combinations; best AE 0, I0 20, r0 10 px, no wide halo, error
  0.18, its 250–350 px ring ×1.28 against the real ×1.00). Revision 3 frees the tail exponent β
  (`scripts/calibrate_sun_tail.py`, coarse grid β 1–3 × I0 5–10,240 × r0 5–40 × AE 0/0.5/1, then a fine
  grid around the optimum): **β 1.5, I0 14, r0 40 px, AE_STRENGTH 0.375, error 0.056**; model ring
  ratios 1.33, 1.54, 1.65, 1.47, 1.42, 1.01, 1.84 against the real 1.33, 1.52, 1.62, 1.71, 1.37, 1.00,
  2.01. Best error by β: 1 → 0.177, 1.25 → 0.061, 1.5 → 0.056, 1.75 → 0.127, 2 → 0.324, 3 → 0.572.
  By AE strength: 0 → 0.097, 0.25 → 0.065, 0.375 → 0.056, 0.5 → 0.074, 0.625 → 0.086, 0.75 → 0.217,
  1 → 1.50. One event: AE 0.25–0.5 fit within 0.02 of each other, and the rows in §8.5 bracket it.
  *A first fit on aggregate statistics (clipped fraction, frame mean, unclipped mean) preferred AE 0.875
  with a far-field veil, i.e. a far field 40× darker: not identifying, and contradicted by the profile.*
  The 22 high-saturation events elsewhere in TRAIN are not usable: their saturation is the consequence of a
  brighter exposure (under-bridge frames exposing for the shadow), not of a source.
* **Night lamp** (`scripts/calibrate_lamp_tail.py`, `lamp_tail.json`). Largest clipped blob in TRAIN
  night frames: q50 6, q90 100, q99 927, max 1,748 px. Reference: 9,413 TRAIN pohang01 frames (stride 2);
  the 101 whose largest light is 600–1,300 px are the s2 reference. Around the light's centroid: ring
  means of linear grey (15–240 px, the glow) and ring floors (5th percentile, 30–800 px, which no veil
  can exceed). Model on 96 night frames without a light, through the real `make_corruption` call; at
  every (β, r0) on a 6 × 6 grid I0 is re-solved for the q99 blob area. **Best β 1.5, r0 12 px, I0 7.17**
  (RMS log error 0.83; best Lorentzian 1.59). Ring floors 240–400 / 400–800 px, linear: real 0.00026 /
  0.00037, β 1.5 0.00018 / 0.00011, revision-2 Lorentzian 0.0023 / 0.0014. Frame 5th-percentile grey:
  real 7.0, β 1.5 5.0, Lorentzian 15.0; above 10.5 on 0% / 3.1% / 100% of frames. s1 / s3 keep β and
  the r0 ladder (×2/3, ×4/3) and re-solve I0 for 100 / 1,748 px (fitted 100 / 1,752). What the smooth
  model cannot reproduce: real lights are structured (lit façades, several lamps), so within 60 px the
  real floor is far darker than any radial glow (real 0.045 at 15–30 px, model 0.32), and real frames
  with big lights are lit scenes whose ring means beyond 240 px are 10–20× the model's.
* **Noise.** VIS day NLF σ 0.25–0.70 DN, night 0.47–0.88 DN (grey, 640 px, flat pixels); cross-channel
  factor 2.25 day / 1.83 night; IR σ 0.4–2.2 DN. **Night/day decision** (content mean < 50): 0 errors on
  744 paired-val and 4,161 pohang04 frames sampled.

## 5. v2 conditions and their constants (`corruptions_v2.DEFAULTS`)

| Condition | Model | Constants (status) |
|---|---|---|
| fog | Koschmieder in linear radiance on per-frame metric depth (Depth Anything V2-Small disparity fitted to the AHRS flat-sea range; §7); A = the frame's sky just above the horizon; patchiness field; then noise restoration, AE (VIS, strength 0.375), quantisation / IR stretch | visibility bands s1 600–1000, s2 250–500, s3 100–200 m (definition: fog < 1 km, dense < 200 m); patchiness 0.20 (ASSUMED); **IR_BETA_RATIO 0.5 (ASSUMED; literature 0.3–1; sensitivity rows)**; AE 0 / 1 sensitivity rows. No forward-scattering glow around lights at night (not modelled). |
| lowlight | exposure scaling (multiplication in the stored domain for a power-law response), then the measured night NLF | target means s1 16–32, s2 4–12, s3 1.5–3 (MEASURED: s2 = TRAIN pohang01 core, median 6.8); NLF (MEASURED); never brightens. VIS only. |
| glare | sun 1–6° above the computed horizon by day, lamp on the shore line at night; veil I0 (1 + r²/r0²)^−1.5 + glint path on the water, linear radiance; AE; noise restoration | SUN s2 (14, 40 px), SUN_BETA 1.5, LAMP (2.59, 8) / (7.17, 12) / (7.69, 16), LAMP_BETA 1.5, AE_STRENGTH 0.375 (FITTED); s1/s3 sun = s2 × (0.4, 2.67) intensity, × (0.7, 1.43) radius, lamp r0 ladder (ASSUMED); glint scale = revision-2 intensities (ASSUMED). Sensitivity rows: AE 0, AE 1, intensity ×0.5, ×2, revision-2 Lorentzian. VIS only. |
| noise | VIS: ImageNet-C shot noise (Poisson, 60/25/12 photons). IR: grey temporal + column fixed-pattern noise, then the stretch. | **ImageNet-C-style severities (ASSUMED)**: not this camera's noise, whose σ is < 1 DN |
| blur | albumentations MotionBlur (random direction), content rows, noise restored; IR then stretched | **ImageNet-C-style (ASSUMED)**: kernel 9–15 / 15–25 px; the AHRS rates could set direction and length, not done |
| rain | albumentations RandomRain, content rows. VIS only. | **ImageNet-C-style (ASSUMED)**: no attenuation, no drops on the lens |

Refused on IR: lowlight, glare, rain (LWIR does not depend on scene illumination, does not produce
coloured flare, and rain is not a visible streak). Table 6's and Table 7's IR-glare rows have no v2
counterpart.

**Measured, v1 vs v2 (revision 3)**, same 150 day + 80 night paired frames as the 10-09 inspection
(`stats.json`, `vis_sheet_*.png`, `ir_sheet.png`). *Detail* = Laplacian variance relative to clean;
*target contrast* = RMS contrast inside VIS GT boxes relative to clean (labels read for this
measurement only); *range* = (p99 − p01) relative to clean.

| cell | mean | black | clipped | detail | target contrast | range | pad changed |
|---|---:|---:|---:|---:|---:|---:|---:|
| day clean | 109.8 | 0.000 | 0.001 | 1.000 | 1.000 | 1.00 | 0 |
| day fog s1 v1 / **v2** | 128.0 / **138.4** | 0 / 0 | 0 / 0 | 0.015 / **0.184** | 0.452 / **0.242** | 0.96 / **0.58** | 0.82 / **0** |
| day fog s2 v1 / **v2** | 152.4 / **145.8** | 0 / 0 | 0 / 0 | 0.013 / **0.101** | 0.396 / **0.115** | 0.97 / **0.47** | 0.90 / **0** |
| day fog s3 v1 / **v2** | 177.7 / **154.6** | 0 / 0 | 0 / 0 | 0.011 / **0.058** | 0.332 / **0.039** | 0.95 / **0.33** | 0.94 / **0** |
| day lowlight s2 v1 / **v2** | 5.5 / **7.3** | 0.818 / **0.115** | 0 / 0 | 0.077 / **0.040** | 0.144 / **0.073** | 0.27 / 0.07 | 1.00 / **0** |
| day glare s1 / s2 / s3 v1 | 123 / 124 / 128 | 0 | 0.000 / 0.000 / 0.001 | ≈1.0 | 0.90 / 0.90 / 0.88 | ≈1.0 | 0.24–0.32 |
| day glare s1 / s2 / s3 **v2** | **124 / 141 / 174** | 0 | **0.043 / 0.120 / 0.298** | 3.9 / 1.9 / 0.41 | **0.80 / 0.49 / 0.22** | 1.07 / 1.03 / 0.93 | **0** |
| day blur s3 v1 / **v2** | 110.0 / 109.8 | 0 | 0 | 0.096 / 0.125 | 0.607 / 0.608 | 0.97 | 0.03 / **0** |
| day noise s2 v1 / **v2** | 110.2 / 109.0 | | | 63.5 / 50.7 | | | 0.98 / **0** |
| night clean | 7.9 | 0.320 | 0 | 1.000 | | 1.00 | 0 |
| night fog s2 v1 / **v2** | 88.4 / **10.3** | 0.015 / 0.172 | 0 | 0.058 / 0.343 | | 10.2 / 0.49 | 0.90 / **0** |
| night lowlight s2 v1 / **v2** | 0.0 / **4.7** | 0.999 / **0.372** | 0 | 0.046 / 0.737 | | 0.00 / 0.78 | 1.00 / **0** |
| night glare s1 / s2 / s3 v1 / **v2** | 42.6 / 45.1 / 54.2 → **20.1 / 33.5 / 41.1** | 0.25 → **0.06 / 0.00 / 0.00** | 0.000 → 0.001 / 0.004 / 0.007 | 3.1 → 5.4 / 5.2 / 4.8 | | | 0.23–0.33 / **0** |
| night noise s2 v1 / **v2** | 19.9 / **7.8** | | | | | | 0.98 / **0** |
| IR day clean | 63.2 | | | 1.000 | | 1.00 | 0 |
| IR day fog s1 / s2 / s3 v1 | 86.7 / 118.3 / 151.8 | | | 0.07 / 0.06 / 0.06 | | 1.06 / 1.31 / 1.50 | 0.71–0.89 |
| IR day fog s1 / s2 / s3 **v2** | **60.5 / 77.5 / 121.1** | | | 3.1 / 7.1 / 21.8 | | **0.83 / 0.90 / 1.15** | **0** |
| IR day noise s2 v1 / **v2** (colour B−R 38.4 / **0**) | 65.7 / 78.1 | 0.017 / 0 | | 267 / 27.7 | | 1.22 / 0.96 | 0.98 / **0** |

What this says. VIS fog strength grows with severity through contrast, not blur; the AE (strength 0.375) pulls the brighter fogged frame down by a gain of 0.83 / 0.80 / 0.76. Low light s2 looks like the real night. Glare now clips a real fraction of the frame (12% at s2 by day, the real event's 12.9%) and costs target contrast with severity (0.80 / 0.49 / 0.22), with the AE gain 0.85 / 0.66 / 0.40 and the rest of the frame 1.03 / 1.10 / 1.41× its clean brightness. At night the lamp leaves the far field near the real floor (5th-percentile grey 4–8 on the eight-frame check, real 7). IR fog keeps the IR range (the stretch restores it) and instead amplifies the sensor noise, which is what a min-max-stretched thermal camera in fog shows (IR "detail" rises because the Laplacian variance of IR frames is noise-dominated). Nothing touches the pad.

## 6. Audit of the whole chain (requested 2026-10-10: "does evaluation see what deployment would see")

| Item | Finding | Action |
|---|---|---|
| a. Post-capture steps skipped or re-applied | IR min-max: was skipped by revision-1 fog → fixed for fog, noise, blur. VIS gamma/ISP: approximated by a 2.2 power law (ASSUMED; no calibration target). 2048→640 resize: every NLF was measured on the 640-px frames it is applied to; fog/glare/lowlight are smooth fields; blur/rain/VIS-noise kernels are ImageNet-C-style, not physical. | fixed / disclosed |
| b. Sensor noise attenuated or smoothed by a corruption | fog (IR material, VIS sub-quantisation), motion blur (both), glare/AE (VIS, via k). Lowlight scales the clean noise by its gain a ≤ 0.1 at s2 before adding the night NLF (a²σ² ≈ 0.003 DN²; left). Rain's internal blur smooths noise (VIS, < 1 DN; left). | fixed / quantified |
| c. Every constant measured or assumed | labelled in `DEFAULTS`; sensitivity rows for IR_BETA_RATIO (0.3, 1.0), AE_STRENGTH (0 and 1, on glare and on fog), source intensity (×0.5, ×2) and the revision-2 Lorentzian glare in §8.5. ImageNet-C-style severities (SHOT_C, BLUR_LIMIT, rain) have no physical anchor; their severity ladder is their sensitivity and the paper labels them so. | done |
| d. Gate statistics on exactly the frames the detector sees | computed in the build pass on the corrupted content rows by the GPU interpreter. The project's rule (paper §10) was to compute statistics under `.venv`; py-3.13 vs `.venv` differ by ≤ 6.3e-7 relative on float statistics and are identical on percentile statistics (measured on all clean paired frames). Replaying the corruption under `.venv` would not reproduce the exact frames the detector saw, so the in-pass statistics are the right ones. A decision can flip between interpreters only if a statistic sits within that distance of its threshold: `scripts/gate_threshold_proximity.py` counts the v2 frame statistics within 1e-4 relative of a gate threshold (`gate_threshold_proximity.txt`). | disclosed, bounded |
| e. Beyond corruptions | clean IR conversion: reproduced bit-exactly from raw by step 1 (60 frames). Letterbox: pad 114 outside rows 151–488 (VIS) / 64–575 (IR), verified. Pairing: |dt| median 31 ms, max 49 ms (paired val); 26 / 49 ms (pohang04). Night split: 0 day/night errors at the corruption's threshold. Label filters, Mahalanobis references: unchanged from their audits (§3.6, project memory). | checked |
| f. Rain, blur, VIS noise | ImageNet-C-style, not upgraded. | labelled in the paper |
| Depth | paired val: 0 fallbacks (R² median 0.998). pohang04 VIS: **276 fallbacks (2.2%)**, all canal/bridge frames with little open water (frames 327–606, 5800–5999): there the sea-plane range is used below the horizon and everything above it is capped at 300 m, so a near bridge gets too much fog; targets on the water keep the sea-plane range. pohang04 IR: 18 (0.14%). | disclosed |
| g. Does a corruption look real **to the gate**? | The shipped veto reads `dark` (p05), `veil` (grad_gini) and `concentrated` (lap_over_var). Comparing those statistics on corrupted frames with real frames of the same kind found the revision-2 lamp defect (§3): a real night light of ~930 px never lifts p05 above 10.5, the revision-2 lamp always did. The same comparison shows that v1's fog was what made `veil` "100% correct on fog": `veil` detects v1's 21-px blur, not contrast loss (§8.1). | fixed (lamp) / reported (axes) |
| Builder memory | albumentations imports torch (~2 GB commit per worker); 28 workers plus the depth model hit the 58.9 GB commit limit and killed two jobs. | blur/rain units capped at 3 workers |

## 7. Depth maps (fog)

Depth Anything V2-Small (Apache-2.0, `transformers` 4.57.6 in the isolated `.venv_depth`, revision
`5426e4f0`), disparity fitted per frame to the flat-sea range from the frame's own AHRS attitude and
calibration (`frame_geometry.py`), robust (40 binned medians, water to 400 m), clipped to [3, 2000] m.
Residual bias on 3,760 GT boxes: in-box range 1.24× the waterline's sea-plane range below the horizon,
1.56× for boxes crossing it (flat-sea map alone 1.51×). Tried and rejected: 1036-px input (1.50×),
piecewise-linear map (1.97×), forcing the sky to 2 km (1.60×). The horizon was checked by drawing it on
12 frames (VIS and IR, day and night).

## 8. Downstream results

### 8.1 The gate's own statistics, v1 against v2 (`gate_axes.md`, `scripts/gate_axes_v1_v2.py`)

Firing rate of each VIS axis, development draws 941–944 (`crossmodal26m` vetoes at night on
`dark OR veil`; by day, with clean IR, it never vetoes):

| condition / slice | dark v1 → v2 | veil v1 → v2 | concentrated v1 → v2 | median p05 v1 → v2 | median grad_gini v1 → v2 |
|---|---|---|---|---|---|
| fog s2 / day | 0 → 0 | **1.000 → 0.091** | 0 → 0 | 58 → 83 | 0.410 → 0.539 |
| fog s2 / night | 0.280 → 0.864 | **1.000 → 0.377** | 0 → 1.000 | 15 → 5.5 | 0.379 → 0.541 |
| fog s1 / day | 0 → 0 | **0.986 → 0.007** | 0 → 0 | 46 → 66 | 0.431 → 0.609 |
| fog s1 / night | 0.804 → 1.000 | 1.000 → 0.023 | 0 → 1.000 | 8 → 4 | 0.417 → 0.616 |
| lowlight / day | 1 → 1 | **0 → 1.000** | 0.035 → 0.228 | 0 → 2 | 0.914 → 0.380 |
| lowlight / night | 1 → 1 | 0 → 0 | 1 → 1 | 0 → 1 | 0.999 → 0.575 |
| glare / day | 0 → 0 | 0 → 0 | 0 → 0 | 36 → 47 | 0.621 → 0.666 |
| glare / night | 0.998 → 1.000 | 0 → 0 | 0.125 → 0.001 | 4 → 6 | 0.765 → 0.728 |

`veil` was introduced as a fog detector ("100% on fog, 0% on the other cells") and it was, on v1 fog,
whose 21-px blur flattens the gradient distribution. Physical fog lowers contrast with distance but
leaves near edges and the frame's noise, so the Gini of gradient magnitude barely moves: `veil` fires on
9% of v2 fog days and 38% of v2 fog nights. At night the veto still fires on 86% of fogged frames,
through `dark`, because v2 night fog (airlight = the dark night sky) keeps the frame dark, where v1's
white veil had lifted it. v2 low light trips `veil` on every day frame, because sensor noise dominates
an underexposed frame's gradients; this is harmless by day with clean IR, and it is what opens the
weak-IR fallback in §8.2. Glare/night `dark` is unchanged (100%) once the lamp's tail is fitted;
revision 2's Lorentzian lamp had it at 0%.

### 8.2 Table 6 under v2: the IR night switch with corrupted IR (`ir_hazards_v2.md`)

All 2,232 paired frames, corruption seed 7 as the v1 probe. False night on the 1,200 day frames:

| IR arm | raw, s1 / s2 / s3 | v1 raw | hardened |
|---|---|---|---|
| blur | 0 / 0 / 0% | 0 / 0 / 0% | 0% |
| fog | 13.4 / 4.8 / 61.4% | 43.8 / 86.7 / 94.8% | 0% |
| noise | 0 / 16.7 / 96.3% | 0 / 0 / 0% | 0% |

The health score flags 93–100% of fogged and noisy day frames, so the hardened vote holds everywhere
(in-sample, as before). The replayed `crossmodal26m` veto never fires on a day frame whose VIS is clean,
glared or fogged. With **low-light VIS** it fires through the weak-IR fallback on 13.4 / 4.8 / 61.4% of day
frames with fogged IR and 0 / 16.7 / 96.3% with noisy IR: the fallback confirms a disarmed IR's night call
with VIS `dark AND veil` or `concentrated`, and v2 low light trips `dark` and `veil` on every day frame
(sensor noise dominates an underexposed frame's gradients) where v1 low light never tripped `veil`. The v1
both-degraded worst case (1.3%) does not survive v2.

### 8.3 Table 3b under v2 (`docs/eval/p3_corrupt_cells_v2_2026-10-10.md`, `..._fog_s1_v2_...`)

Five Phase 3 systems, `crossmodal26m`, VIS draws 941–944, ship AP; same fail criterion as the
2026-09-27 v1 entry (≤ −0.0060 with the between-seed interval below zero).

| cell | VIS | IR | fused | fused − VIS | veto | claim | v1: VIS / fused, claim |
|---|---:|---:|---:|---|---:|---|---|
| fog / day | 0.0106 | 0.0218 | 0.0239 | +0.0133 [+0.0065, +0.0200] | 0% | holds | 0.0601 / 0.0659, holds |
| lowlight / day | 0.1719 | 0.0218 | 0.1790 | +0.0070 [+0.0047, +0.0094] | 0% | holds | 0.0531 / 0.0607, holds |
| glare / day | 0.1581 | 0.0218 | 0.1701 | +0.0120 [+0.0084, +0.0156] | 0% | holds | 0.2833 / 0.2940, holds |
| fog / night | 0.0439 | 0.0687 | 0.0687 | +0.0248 [+0.0172, +0.0324] | 86.4% | holds | 0.0003 / 0.0687, holds |
| lowlight / night | 0.2390 | 0.0687 | 0.0687 | −0.1703 [−0.1862, −0.1543] | 100% | **fails** | 0.0005 / 0.0687, holds |
| glare / night | 0.0833 | 0.0687 | 0.0687 | −0.0146 [−0.0268, −0.0024] | 100% | **fails** | 0.1559 / 0.0687, fails |
| fog s1 / day | 0.0270 | 0.0218 | 0.0360 | +0.0090 [+0.0036, +0.0144] | 0% | holds | 0.0834 / 0.0895, holds |
| fog s1 / night | 0.1066 | 0.0687 | 0.0687 | −0.0378 [−0.0516, −0.0241] | 100% | **fails** | 0.0004 / 0.0687, holds |

With the clean cells (day holds +0.0081; night fails −0.1847) the eight-cell count goes from 6/8
(v1) to **5/8** (v2). What moved: v1 fog and low light blinded VIS at night (0.0003, 0.0005), so the
night veto looked right there; v2 low light barely changes a night frame (it is already at the night
exposure level) and v2 fog s1 leaves VIS at 0.1066, so the veto is now wrong on every night cell except
fog s2. By day v2 fog is harder than v1 fog for VIS (0.0106 vs 0.0601: contrast loss, not a blur the
detector partly survives), v2 low light far easier (0.1719 vs 0.0531), v2 glare harder (0.1581 vs
0.2833: a sun of the real event's size covers part of the scene).

### 8.4 R-D1 under v2 (`runs/eval/rd1_ship_v2_{crossmodal,26m}_2026-10-10.md`)

`runs/cache_m_v2`: the pre-restore full-scale checkpoints, VIS fog / lowlight / glare s2 at seed 1
under v2. Ship AP, block bootstrap L = 20. Coordinate path (S1 − S3): NULL, 0/4 at every floor,
both presets, as in v1. Score path (S5 − S7):

| preset | clean | fog | lowlight | glare | pass at 0.0060 | v1 |
|---|---|---|---|---|---|---|
| `crossmodal` | +0.0069 [+0.0027, +0.0112] | +0.0042 [+0.0027, +0.0065] | +0.0081 [+0.0052, +0.0105] | +0.0037 [+0.0004, +0.0059] | 2/4 → NULL | 3/4 POSITIVE |
| `crossmodal26m` | +0.0073 [+0.0033, +0.0124] | +0.0041 [+0.0026, +0.0063] | +0.0012 [−0.0027, +0.0039] | +0.0029 [+0.0011, +0.0052] | 1/4 → NULL | 3/4 POSITIVE |

At 0.0014: 4/4 and 3/4; at 0.0031: 4/4 and 2/4. Every delta stays positive; on corrupted frames
it shrinks below the registered floor. The registered verdict is the v1 one; this is a disclosed
robustness check that says the POSITIVE depends on the corruption model.

### 8.5 Sensitivity rows (`sensitivity.md`, `scripts/v2_sensitivity.py`)

One constant moved, draw 941, five systems, Δ = row − shipped v2 cell (ship AP, 95% t-interval):

| row | day: Δ fused (Δ VIS) | night: Δ fused (Δ VIS) | night veto row / base |
|---|---|---|---|
| glare, AE 0 (no reaction) | −0.0069 [−0.0084, −0.0055] (−0.0071) | 0 (0) | 100 / 100% |
| glare, AE 1 (converged) | +0.0035 [+0.0020, +0.0050] (+0.0053) | 0 (0) | 100 / 100% |
| glare, intensity ×0.5 | +0.0360 [+0.0332, +0.0388] (+0.0363) | 0 (+0.0213) | 100 / 100% |
| glare, intensity ×2 | −0.0362 [−0.0380, −0.0345] (−0.0359) | −0.0001 (−0.0222) | 91.6 / 100% |
| glare, revision-2 Lorentzian | +0.0480 [+0.0453, +0.0506] (+0.0499) | +0.0057 [+0.0007, +0.0107] (−0.0677) | **0** / 100% |
| fog, AE 0 | −0.0002 | 0 | 87 / 87% |
| fog, AE 1 | +0.0008 | 0 | 87 / 87% |
| IR fog s2, β ratio 0.3 | +0.0070 (IR +0.0091) | +0.0139 (IR +0.0207) | 99.8 / 97.9% |
| IR fog s2, β ratio 1.0 | −0.0086 (IR −0.0079) | −0.0261 (IR −0.0271) | 98.2 / 97.9% |

What this says. The glare intensity is the constant that matters: ±0.036 on glare/day, and at ×2 the
glare/night VIS falls to 0.060 (below IR 0.0687), so the glare/night failure (−0.0146) holds at the
fitted intensity but not at twice it. The AE strength is a second-order effect (−0.0069 to +0.0035 on
glare, < 0.001 on fog). Revision 2's Lorentzian lamp would have switched the night veto off on every
glared night frame (veto 0%) while cutting VIS to 0.016, which is the defect §3 describes, measured.
IR_BETA_RATIO moves the IR-fog cell by −0.026 to +0.014 at night.

### 8.6 What the weak-IR fallback costs (`fallback_cost.md`, `scripts/v2_fallback_cost.py`)

§8.2 counts the fallback's vetoes; this prices them. Draw 941 (IR 951), five systems, ship AP;
claim = fused − max(VIS, IR) with the between-seed interval, as in Table 3b. Day frames:

| VIS | IR | VIS | IR | fused | veto | fused − max [95% CI] |
|---|---|---:|---:|---:|---:|---|
| low light | clean | 0.1712 | 0.0218 | 0.1783 | 0% | +0.0071 [+0.0048, +0.0094] |
| low light | fog s1 | 0.1712 | 0.0254 | 0.1612 | 14.2% | −0.0100 [−0.0167, −0.0032] |
| low light | fog s2 | 0.1712 | 0.0126 | 0.1644 | 4.2% | −0.0067 [−0.0100, −0.0035] |
| low light | fog s3 | 0.1712 | 0.0033 | 0.0577 | 61.7% | **−0.1135** [−0.1264, −0.1005] |
| low light | noise s2 | 0.1712 | 0.0298 | 0.1586 | 16.6% | −0.0125 [−0.0213, −0.0038] |
| low light | noise s3 | 0.1712 | 0.0007 | 0.0069 | 96.1% | **−0.1643** [−0.1879, −0.1407] |
| clean | fog s2 | 0.3485 | 0.0126 | 0.3460 | 0% | −0.0025 [−0.0053, +0.0003] |
| clean | fog s3 | 0.3485 | 0.0033 | 0.3389 | 0% | −0.0096 [−0.0128, −0.0063] |
| clean | noise s2 | 0.3485 | 0.0298 | 0.3654 | 0% | +0.0169 [+0.0144, +0.0193] |

What this says. By day, every low-light cell with degraded IR fails the claim, and the loss grows with
the veto rate: at IR noise s3 the fallback drops a VIS stream scoring 0.17 on 96% of day frames and
leaves IR's 0.0007. The fallback is doing what it was built to do (confirm a disarmed IR's night call
with VIS darkness) on frames where darkness is the corruption, not the time of day. A second,
independent loss: with clean VIS and IR fog s3 nothing is vetoed, yet the union sits 0.0096 below VIS
alone, because a near-dead IR stream's boxes still enter the merge. `crossmodal26m` keeps them on
purpose (the IR merge veto was measured net-harmful on v1); §8.7 finds where they act. At night every cell is the §6.6
defect, amplified: IR noise takes IR to 0.0000 at night, the veto still drops VIS (0.24–0.25), and
fused is 0.0000.

### 8.7 Where a dead IR stream hurts the union (`ir_merge_veto.md`, `scripts/v2_ir_merge_veto.py`)

Same cells, two one-switch arms against the shipped preset, paired by system (day frames; night cells
move by ≤ 0.0007):

| VIS / IR | shipped fused | IR merge veto on: Δ | IR dropped | support bonus off: Δ |
|---|---:|---|---:|---|
| clean / clean | 0.3566 | 0 | 0% | −0.0068 [−0.0086, −0.0049] |
| clean / fog s1 | 0.3553 | −0.0033 [−0.0045, −0.0021] | 13.8% | −0.0033 [−0.0066, +0.0001] |
| clean / fog s2 | 0.3460 | −0.0144 [−0.0186, −0.0102] | 60.9% | **+0.0050** [+0.0029, +0.0072] |
| clean / fog s3 | 0.3389 | −0.0023 [−0.0049, +0.0004] | 92.1% | **+0.0113** [+0.0083, +0.0143] |
| clean / noise s2 | 0.3654 | −0.0169 [−0.0193, −0.0144] | 100% | −0.0157 [−0.0201, −0.0112] |
| clean / noise s3 | 0.3500 | −0.0015 [−0.0058, +0.0028] | 100% | −0.0003 [−0.0008, +0.0002] |
| low light / any IR | | 0 (VIS unhealthy, so the veto never fires) | 0% | −0.0039 to +0.0024 |

What this says. The v1 decision on the IR merge veto stands under v2: wherever it fires it costs AP
(−0.0033 to −0.0169) and it never gains. The loss under a fogged IR goes through the cross-modal
support bonus (`crossmodal26m`: a VIS box overlapped by any IR box at IoU ≥ 0.30 has its score ×1.5).
With clean IR the bonus is worth +0.0068; with IR fog s2/s3 it costs 0.0050/0.0113, because the junk
boxes of a near-dead IR vouch for VIS false positives. Without it clean VIS + IR fog s3 is 0.3502,
above VIS alone (0.3485). The bonus is a claim about the IR detector, like the vetoes; a bonus gated on
IR health would be the obvious repair, and a new rule.

### 8.8 Stage 1 under v2 (`runs/eval/stage1_crossing_v2_2026-10-10.md`)

The registered crossing (`scripts/stage1_crossing.py`, now with `--cache-dir --bright-dir
--structure-dir`) on `runs/cache_m_v2`, only the corrupted VIS caches changed. The clean cell
reproduces the registered run exactly (A 0.3894, A − D +0.0151 [+0.0095, +0.0210]). A − D on TUNE
(non-inferior if the upper bound < 0.0060): fog +0.0001 [−0.0007, +0.0007], low light +0.0032
[+0.0016, +0.0053], glare +0.0002 [−0.0013, +0.0018]. Applied to these caches, the rule counts **3/4**,
where the registered v1 run counted 1/4 (S1-NULL).

What this says. Not a rescue. On v2's corrupted frames relaxing correspondence costs almost nothing
(C − A 0.0000 / −0.0031 / −0.0003 against −0.0148 on clean), so D matches A whatever σ does; the
interaction (D − C) − (B − A) is within ±0.0005 on every cell, every interval spanning zero. It is the
mechanism behind v1's lone low-light pass, now on three cells. The registered verdict is the v1 one,
and the substance does not change: σ arbitrating relaxed correspondences recovers nothing.

### 8.9 The veil repair under v2 (`veil_reprice.md`, `scripts/v2_veil_reprice.py`)

Five Phase 3 systems, VIS draws 941–944, clean IR, one switch per arm against the shipped
`night AND (dark OR veil)`:

| cell | shipped fused (claim) | veil unconditional: Δ | veil axis removed: Δ |
|---|---|---|---|
| fog / day | 0.0239 (+0.0021, holds) | −0.0071 [−0.0099, −0.0042] (veto 9.1%) | 0 |
| low light / day | 0.1790 (+0.0070, holds) | **−0.1572** [−0.1789, −0.1354] (veto 100%) | 0 |
| fog s1 / day | 0.0360 (+0.0068, holds) | −0.0000 (veto 0.7%) | 0 |
| every other cell (clean, glare, all night) | | 0 | 0 |

Cells holding the claim, of ten: shipped 6, unconditional 5, removed 6.

What this says. With clean IR the veil axis carries nothing on v2: removing it changes no cell, because
at night `dark` already vetoes wherever `veil` would (fog/night veto 86.4% either way) and by day the
shipped rule never asks it. The repair's direction survives: the pre-repair rule would veto VIS on every
v2 low-light day frame and cost 0.157. On v1 the same repair was worth 0.0416 on fog/clean (macro); its
whole value came from v1 fog's blur. On v2 the axis acts only inside the weak-IR fallback
(`dark AND veil`), which is the hole of §8.6.

### 8.9b A candidate repair of the fallback (`fallback_fix.md`, `scripts/v2_fallback_fix.py`)

Since `veil` is inert with clean IR (§8.9), its only role is the fallback's `dark AND veil`. One arm, the
axis removed, against the shipped preset, VIS clean / low light / fog × IR clean, fog s1–s3, noise s2–s3,
day and night, draw 941. Cells that change (all low-light day; Δ fused, claim before → after):

| IR | veto shipped → no veil | Δ fused [95% CI] | claim (fused − max) shipped → no veil |
|---|---|---|---|
| fog s1 | 14.2% → 3.8% | +0.0163 [+0.0099, +0.0227] | −0.0100 fails → +0.0063 holds |
| fog s2 | 4.2% → 1.1% | +0.0074 [+0.0069, +0.0079] | −0.0067 fails → +0.0007 holds |
| noise s2 | 16.6% → 3.7% | +0.0182 [+0.0109, +0.0255] | −0.0125 fails → +0.0057 holds |
| fog s3 | 61.7% → 14.2% | +0.0942 [+0.0826, +0.1057] | −0.1135 fails → −0.0193 fails |
| noise s3 | 96.1% → 23.0% | +0.1431 [+0.1234, +0.1628] | −0.1643 fails → −0.0212 fails |

Every other cell of 36, including fogged VIS at night with fogged or noisy IR (what the fallback was
built for), moves by exactly 0.0000. Cells holding the claim: 14 shipped → 17. What remains: `concentrated`
still confirms the false night on 14–23% of low-light days at severity 3, and every night failure is the
§6.6 night-veto defect, which this does not touch. Descriptive, one draw, development frames: a candidate
for a registered fix, not a change.

### 8.10 pohang04 under v2: a disclosed second exposure (`docs/eval/holdout_p04_v2_rescore_2026-10-10.md`)

Pre-registered (`docs/prereg-p04-v2-rescore-2026-10-10.md`, amended for revision 3 and for the GPU
bootstrap). Look's systems, preset, calibration, draws, statistic, bootstrap; 12,482 day pairs.

| cell (VIS / IR) | v2 AP [95% CI] | v1 (look) |
|---|---|---:|
| clean / clean (verdict, not re-scored) | 0.2682 [0.2576, 0.2793] | 0.2682 |
| clean / blur_s2 | 0.2673 [0.2570, 0.2779] | 0.2671 |
| clean / noise_s2 | 0.2635 [0.2542, 0.2738] | 0.2692 |
| clean / fog_s2 | 0.2666 [0.2565, 0.2771] | 0.2691 |
| rain_s2 / clean | 0.2209 [0.2107, 0.2324] | 0.1850 |
| lowlight / clean (new) | 0.2134 [0.2040, 0.2237] | — (v1 lowlight / glare_s2 0.0121) |
| fog / clean | 0.0445 [0.0393, 0.0505] | 0.0580 |
| blur_s3 / clean | 0.0419 [0.0389, 0.0448] | 0.0418 |
| noise_s2 / clean | 0.0123 [0.0097, 0.0142] | 0.0114 |
| clean / glare_s2, blur_s3 / glare_s2, lowlight / glare_s2 | not modelled in v2 | 0.2668, 0.0407, 0.0121 |

What this says. IR-side corruption moves the fused output by at most 0.0047 (v1 ±0.0014); VIS-side by
up to 0.2559: it tracks VIS, as in development. What changed with the corruption model: low light
(0.0121 → 0.2134; v1 blacked the frame out), rain (+0.0359; v1 also streaked the pad) and fog (−0.0136;
contrast loss with distance is harder than v1's blur). The CPU bootstrap (24 processes) was replaced
mid-run by the bit-identical GPU path (§9); the CPU run was stopped before it wrote, which is why
`runs/v2_p04_score.log` ends in "score FAILED". The report and JSON are the GPU run's
(`runs/holdout_p04/v2/score_gpu.log`).

## 9. Speed and parallelism (16 physical / 32 logical cores, RTX 4080 Laptop)

| step | before | now |
|---|---|---|
| fog corruption, per frame | 234 ms (v1, one core) | ~150–400 ms (v2 rev 2, under load; depth read, noise restoration), in a process pool |
| corrupted cache, 5 checkpoints | serial corruption + 1 checkpoint per pass | one corruption pass feeds all 5 checkpoints; `--jobs` units in parallel |
| inference | batch 1 | `--batch 4` for pohang04 (1.6× per checkpoint; |Δ ship AP| ~1e-5, `batched_inference.json`); development cells stay at batch 1 |
| gate statistics | a second full replay | computed in the same pass |
| scoring | serial over (system, draw) | process pool over (system, draw); parallel scorer reproduces the recorded v1 Table 3b exactly (max |diff| 0) |
| pohang04 bootstrap | serial, all classes (~18.6 h for the look) | class-0 presort memory-mapped; on the CPU (24 processes) ~8 h, so moved to the GPU (`--boot-device cuda --reuse-pre`): float64 scans along the contiguous axis, interpolation in numpy on the bracketing points, ~27 ms per AP, ~75 min for 8 cells × 1,000 replicates × 20 (system, draw); checked `==` against the CPU `_score` on 96 + 15 replicates before use |

Checks: pooled builder bit-identical to serial (v1, 5 × 200 frames); in-pass statistics identical to a
replay (0 mismatches); `predict_batch` at B = 1 bit-identical to `__call__`.

Measured wall time, revision 3 (2026-10-10, `runs/v2_pipeline.log`): development cells 65 min, sensitivity
rows 41 min, R-D1 mirror run alongside. pohang04 (12,482 pairs × 5 checkpoints, 32 units, 4 at once,
6 corruption workers each, batch 4): ~3.3 frames/s per unit, ~13 frames/s in all, ~63 min per round of
four units, ~8.5 h for the build; GPU and CPU both at 100% during it, so more parallelism buys nothing on
this machine. The pohang04 scoring stage (20 system-draw contexts, then presorts) took 1.8 h; re-verifying the 160 caches (pickles plus 12,482 label files each) takes ~40 min under load. Calibration fits run in a process pool (`--workers`); the lamp tail fit (36 grid points,
13-step I0 bisection each) and the fine sun fit (432 points) each finish in under an hour.

## 10. Files

* New: `src/uqfusion/eval/corruptions_v2.py`, `frame_geometry.py`, `parallel_frames.py`;
  `scripts/build_depth_maps.py` (`.venv_depth`), `calibrate_corruptions_v2.py`, `calibrate_glare_ae.py`,
  `inspect_corruptions_v2.py`, `build_corruption_v2_dev.py`, `v2_sensitivity.py`, `ir_hazards_v2.py`,
  `holdout_p04_v2_rescore.py`, `check_batched_inference.py`; revision 3 added `calibrate_lamp_tail.py`
  (night spread tail, real TRAIN lights), `calibrate_sun_tail.py` (`--fine`; day tail and AE strength),
  `gate_axes_v1_v2.py` (gate axis firing rates, v1 against v2) and `gate_threshold_proximity.py` (how close
  any cached gate statistic sits to its threshold). The degraded-IR and re-pricing checks of §8.6–8.9b added
  `v2_fallback_cost.py`, `v2_ir_merge_veto.py`, `v2_veil_reprice.py` and `v2_fallback_fix.py`.
* Changed: `corruptions.py` (`version=`, `params=`, `corruption_from_meta` with the revision check),
  `build_cache_multi.py` (`--workers --corrupt-version --modality --stats-out --batch --params`),
  `uq/infer.py` (`predict_batch`), `frame_brightness.py`, `frame_structure.py`, `verify_cache_m.py`
  (version-aware replay), `p3_corrupt_cells.py` (`--cache-root --stats-root --workers`), the three R-D1
  scripts (`--bright-dir --structure-dir`, fog diag `--cache-dir`), `fig_detections.py`
  (`--corrupt-version`, default v2), `identity.py` (UTF-8), `paper_figures.py` (Figure 7 reads the v2
  Table 3b JSON), `ir_hazards_v2.py` output carries `corrupt_code`, `stage1_crossing.py` (`--cache-dir
  --bright-dir --structure-dir`, defaults reproduce the registered run), `build_corruption_v2_dev.py` (IR fog
  s1/s3 and noise s2/s3 at draw 941), `holdout_p04_v2_rescore.py` (`--boot-device cuda --reuse-pre`).
* Data (untracked): `runs/derived/depth_v2/`, `runs/cache_p3dev_v2/` + `runs/derived_p3dev_v2/`,
  `runs/cache_m_v2/` + `runs/derived_m_v2/`, `runs/holdout_p04/v2/`.

```
.venv_depth/Scripts/python.exe scripts/build_depth_maps.py --list runs/derived/paired_val_vis.txt --modality vis
py -3.13 scripts/calibrate_corruptions_v2.py --workers 24
py -3.13 scripts/calibrate_glare_ae.py --workers 16 --stride 2
py -3.13 scripts/calibrate_lamp_tail.py --workers 24
py -3.13 scripts/calibrate_sun_tail.py --workers 20 && py -3.13 scripts/calibrate_sun_tail.py --fine --out docs/eval/corruption_v2_calibration/sun_tail_fine.json
py -3.13 scripts/inspect_corruptions_v2.py --workers 16
bash runs/v2_pipeline.sh            # dev cells -> sensitivity rows -> pohang04 caches
bash runs/v2_side.sh                # inspection sheets -> R-D1 mirror caches -> R-D1, S5-S0, Table 6
bash runs/v2_score.sh               # Table 3b, gate axes, threshold proximity, sensitivity rows
bash runs/v2_p04_score.sh           # pohang04 score once the build is done (CPU bootstrap, ~8 h), or:
py -3.13 scripts/holdout_p04_v2_rescore.py --score --reuse-pre --boot-device cuda --out docs/eval/holdout_p04_v2_rescore_2026-10-10.md
py -3.13 scripts/build_corruption_v2_dev.py --draws 941 --conds ir_fog_s1 ir_fog_s3 ir_noise_s2 ir_noise_s3
py -3.13 scripts/v2_fallback_cost.py && py -3.13 scripts/v2_ir_merge_veto.py && py -3.13 scripts/v2_fallback_fix.py
py -3.13 scripts/v2_veil_reprice.py
py -3.13 scripts/stage1_crossing.py --cache-dir runs/cache_m_v2 --bright-dir runs/derived_m_v2/brightness --structure-dir runs/derived_m_v2/structure --out runs/eval/stage1_crossing_v2_2026-10-10.md
```
