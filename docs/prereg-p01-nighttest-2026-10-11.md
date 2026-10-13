# Pre-registration: a night test on pohang01's test split, 2026-10-11

Written before any detection is computed on these frames, before their frame statistics are measured,
and before any of their labels is read. Logged in `docs/exposure-ledger-2026-09-09.md` at the same time.
Code: `scripts/p01_nighttest.py`.

**Why.** Every night number in the paper comes from development. pohang01, the only night run, supplies
1,032 frames of the development set (`runs/derived/paired_val_vis.txt`, all from the VIS **val** split).
pohang04, the held-out run, is all daylight. So the night verdicts have never been tested on frames that
no fusion decision has seen. pohang01's **test** split is such a set:
* 1,629 VIS frames;
* 1,109 of them pair (`Pohang_dataset/paired/pohang01_pairs.csv`) with an IR frame in the IR test split.

## Exposure, checked before writing

* **Fusion.** Every directory under `runs/` holding a detection pickle was checked (263 directories; one
  pickle each, IR caches mapped to their VIS partners; `scratchpad/night_test_exposure.py`). None holds a
  frame of pohang01's test split, VIS or IR. These frames have never been fusion-scored.
* **Detectors.** The Phase 3 detectors trained on pohang01's train split and validated on its val split
  (`split: val`). No script evaluates on the test split.
* The test split's 1,629 frames sit in 520 contiguous blocks interleaved with training blocks of the same
  run.
* **What this test therefore is:** the fusion rules on night frames they have never seen. It is not a new
  night, a new harbour or new weather.

## Systems

All five Phase 3 systems, assembled as the look assembles them (`holdout_p04_look.system_context`):
`crossmodal26m`, with the development calibration injected. Nothing is fitted on these frames.

| arm | definition (identical to development) | role |
|---|---|---|
| `shipped` | `crossmodal26m` as is | reference |
| `union` | E4: veto off (`veto=None`) | **primary** |
| `lex` | E9: every VIS ship box above every IR ship box | descriptive |
| `ens5_union` | five checkpoints per stream, plain WBF at IoU 0.55, IR NMS 0.70, no veto, development ensemble prior (`runs/holdout_p04/gatefree/devcal.json`) | descriptive |

Every arm's VIS-only and IR-only streams are scored too.

**Inputs, built like pohang04's.**
* Caches: `build_cache.py --source gaussian`, imgsz 640, conf 0.001, fp32, Phase 3 weights.
* Clean frames only.
* Frame statistics: `holdout_p04_frame_stats.measure` under the `.venv` interpreter, after its bit-exact
  check against development files passes.

## Statistic, interval

* **Statistic:** ship AP (class 0, AP50:95, local linear interpolation) against the VIS labels (night
  boxes restored, 2026-09-02). Mean over the five systems.
* **Δ:** arm − `shipped`, paired.
* **Interval:** 2.5th and 97.5th percentiles of 1,000 moving-block replicates (L = 20, seed 1,
  `holdout_p04_v2_rescore.resample_weights`), over the 1,109 frames in capture order.
* One resample scores every arm and system on the same frames.

## Predictions and decision rules

Development (clean/clean/night, fresh draws 945–947, `docs/eval/no_retrain/ensembles_fresh.md`):

| arm | ship AP | Δ vs `shipped` | claim |
|---|---:|---:|---:|
| `shipped` | 0.0687 | — | −0.19 (FAIL) |
| `union` | 0.2895 | +0.2208 | +0.0360 |
| `ens5_union` | 0.3124 | +0.2437 | +0.0382 |

* **P-N1, `union` beats `shipped` at night** (primary): REPLICATES iff Δ ≥ +0.0060 with its interval
  above zero.
* **P-N2, `union` keeps the claim at night:** PASS iff fused − max(VIS, IR) is not ≤ −0.0060 with its
  interval below zero.
* **Descriptive, no verdict:**
  * `shipped`'s claim;
  * `lex` and `ens5_union` (Δ and claim);
  * per-system values;
  * the share of frames on which IR reads night, and on which `shipped` vetoes VIS.

`crossmodal26m`'s night axis stays closed ("No V4"): this test changes no shipped constant.

## Amendment 1 (6:21 PM, before any flipped or restored view of these frames is built): `full_tta` and `full_tta_lex`, descriptive

Two arms are added. Neither carries a verdict.

* **`full_tta`**, exactly as P-F defines it for pohang04 (`docs/prereg-p04-heldout-gatefree-2026-10-11.md`):
  * five checkpoints per stream;
  * each stream merges ten views, five unflipped and five horizontally flipped with boxes mapped back,
    frame by frame with plain WBF at IoU 0.55;
  * IR views get the shipped NMS at 0.70, and the unflipped IR view is NLM-restored where the IR health
    check fails (`ir_d2 > ir_bound`);
  * no veto; development prior `devcal["full_tta"]` (`runs/holdout_p04/gatefree/devcal.json`).
* **`full_tta_lex`**: E9 on the same two streams (pohang04 Amendment 2).

**Inputs.**
* Unflipped views: the caches this test already built (fp32).
* Flipped and restored views: `p01_nighttest.py build-tta`, built as `p04_heldout.py build-unit` builds
  pohang04's (fp16 autocast, channels_last, batch 8).
* The IR health flag: from this test's own IR structure statistics and the frozen health model
  (`p04_heldout.ir_flags`), checked against every system context.

**Reported.** Ship AP; Δ against `shipped` and against `union` (paired, the same 1,000 replicates); the
claim; values per test stretch.

**Development** (clean/clean/night, draws 945–947, `docs/eval/no_retrain/ensembles_fresh.md`):

| arm | ship AP | Δ vs `shipped` | Δ vs `union` | claim |
|---|---:|---:|---:|---:|
| `union` | 0.2895 | +0.2208 | — | +0.0360 |
| `full_tta` | 0.3192 | +0.2504 | +0.0297 | +0.0342 |
| `full_tta_lex` | 0.3010 | +0.2323 | +0.0115 | +0.0161 |

**Why no verdict.** Every result in `docs/eval/night_p01test_2026-10-11.md` and
`docs/eval/night_episodes_2026-10-11.md` was seen before this amendment. That includes `ens5_union`,
the same ensemble without flips or restoration: 0.0490 against `union`'s 0.0506, below `union` on test
stretch A and above it on B. A prediction written now would be written knowing that.
