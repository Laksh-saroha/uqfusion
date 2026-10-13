# pohang04 corrupted cells under corruption v2 — a disclosed second exposure

**Written 2026-10-10, before any v2 pohang04 number exists.** Script:
`scripts/holdout_p04_v2_rescore.py`. Output: `docs/eval/holdout_p04_v2_rescore_2026-10-10.{md,json}`.

## Why

The single look of 2026-09-20 (`docs/eval/holdout_p04_look.md`) scored ten descriptive
corrupted cells with the v1 corruptions. On 2026-10-09 those were found broken at the source
(`docs/eval/corruption_v2/README.md`): v1 fog is a fixed 21-px blur over a white disc layer,
v1 lowlight an additive clip, v1 glare lands in the letterbox pad on 45% of frames, every v1
filter rewrites the pad, and IR received visible-light filters with per-channel colour noise.
Table 7's corrupted rows therefore measure the filters as much as the system. Laksh decided on
2026-10-10 to re-score those cells under v2 and disclose it.

## What this records

**pohang04 is no longer held out.** It was spent by the single look, and this re-score is a
second exposure of it. The exposure ledger says so in its 2026-10-10 entry. Nothing scored
here, and nothing scored on pohang04 afterwards, may be called a held-out result.

## What is fixed before the number

* **Untouched:** the `clean/clean` verdict (NO-GAP, D = +0.0216, CI [−0.0120, +0.0502]).
  It involves no corruption and is not re-scored. `runs/holdout_p04/LOOK_TAKEN.json` and
  `docs/eval/holdout_p04_LOOK_TAKEN.json` are read, never written; the script refuses
  if either changes during the run.
* **Same as the look:** the five Phase 3 systems (VIS seed *k* + IR seed *k*), preset
  `crossmodal26m`, development calibration injected (`holdout_p04_look.system_context`),
  the 12,482 day pairs, fused ship AP against VIS ground truth, VIS draws 941–944 with
  IR = VIS + 10, seed mean of draw means, moving-block bootstrap L = 20, n_boot = 1000,
  seed 1, the look's clean caches and clean frame statistics.
* **Changed:** the corruption (v2, content rows only) and the inference batch of the
  corrupted caches (4; measured |Δ ship AP| ~1e-5 against batch 1).
* **Cells (8):** `clean/blur_s2`, `clean/noise_s2`, `clean/fog_s2`, `blur_s3/clean`,
  `noise_s2/clean`, `rain_s2/clean`, `fog/clean`, `lowlight/clean`. v2 does not model IR
  glare, so `clean/glare_s2`, `lowlight/glare_s2` and `blur_s3/glare_s2` are reported as
  not modelled with their v1 values kept for the record. `lowlight/clean` is new and stands in
  for `lowlight/glare_s2`.
* **Descriptive only.** Value and interval per cell. No verdict, no pass/fail language, no
  outcome label, nothing adopted, no constant tuned. No VIS-only or IR-only arm: a per-stream
  comparison on pohang04 is a new question and was not authorised.
* **Reporting.** Table 7 shows the v2 values as primary for the corrupted rows and keeps the
  v1 values beside them. Both the deviation and the second exposure are stated in §7 and §9.

## Amendment, 2026-10-10, before any v2 pohang04 cache was built

Corruption v2 moved to revision 3 (exposure ledger, 2026-10-10): the glare spread's tail and
the AE strength were refitted on TRAIN frames (AE_STRENGTH 0 → 0.375, which also acts on fog).
The re-score uses whatever revision the caches are stamped with (`corrupt_code`), and the
script refuses caches from another revision. Nothing else above changes.

## Implementation note, 2026-10-10, during the run

The CPU bootstrap (`_boot_chunk`, 24 processes) proved to need ~8 h on this machine. The same
replicates were computed on the GPU instead (`--boot-device cuda --reuse-pre`): the presorts and the
1,000 block-resample weight vectors that the CPU run had already written are reused unchanged, and
`_ap_gpu` reproduces `_score` exactly (float64 cumulative sums of 0/1 flags are exact integers; division
and max are correctly rounded; the 101-point interpolation stays in numpy). The run checks `==` against
the CPU path on every cell before using any replicate and refuses on one mismatch. Nothing about what is
scored, on which frames, or how the interval is formed changes; the CPU run was stopped before it wrote.
