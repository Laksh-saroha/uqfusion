# Shipped constants re-priced under corruption v2 (development, 2026-10-10)

`scripts/v2_constants_audit.py`. Draws 942, 943, 944, five Phase 3 systems, 60 cells, ship AP. Δ = arm − shipped fused, per system the draw mean, 95% t-interval over systems (df 4). better / worse = |Δ| ≥ floor with the interval excluding zero. Claim = fused − max(VIS, IR) failing at ≤ −0.0060 with the interval below zero. Descriptive: no arm replaces a shipped value without a registered test on unseen draws.

| arm | better ≥0.0014 | worse ≥0.0014 | better ≥0.0060 | worse ≥0.0060 | worst Δ (cell) | best Δ (cell) | claim fails |
|---|---:|---:|---:|---:|---|---|---:|
| shipped | 0 | 0 | 0 | 0 | +0.0000 (clean/clean/day) | +0.0000 (clean/clean/day) | 28 |
| cap_ir_scale 1 | 0 | 8 | 0 | 0 | -0.0054 (lowlight/clean/day) | +0.0012 (fog/noise_s2/day) | 28 |
| cap_ir_scale 2 | 0 | 3 | 0 | 0 | -0.0021 (lowlight/clean/day) | +0.0006 (fog/noise_s2/day) | 28 |
| cap_ir_scale 8 | 0 | 0 | 0 | 0 | -0.0010 (glare/fog_s1/night) | +0.0013 (lowlight/clean/day) | 28 |
| cap_ir_scale 16 | 1 | 1 | 0 | 0 | -0.0023 (glare/fog_s1/night) | +0.0020 (lowlight/clean/day) | 28 |
| iou_thr 0.55 | 2 | 11 | 0 | 8 | -0.0278 (clean/fog_s1/day) | +0.0040 (clean/fog_s3/day) | 31 |
| iou_thr 0.70 | 1 | 8 | 0 | 2 | -0.0109 (clean/fog_s1/day) | +0.0017 (glare/fog_s3/day) | 29 |
| iou_thr 0.95 | 0 | 10 | 0 | 0 | -0.0022 (glare/noise_s2/day) | +0.0001 (clean/fog_s3/night) | 28 |
| support off | 5 | 9 | 1 | 2 | -0.0158 (clean/noise_s2/day) | +0.0123 (clean/fog_s3/day) | 26 |
| support gamma 0.25 | 1 | 0 | 0 | 0 | -0.0009 (lowlight/clean/day) | +0.0020 (clean/fog_s3/day) | 28 |
| support gamma 1.0 | 0 | 2 | 0 | 0 | -0.0015 (clean/fog_s3/day) | +0.0011 (clean/noise_s3/day) | 28 |
| support iou 0.20 | 0 | 0 | 0 | 0 | -0.0012 (clean/fog_s3/day) | +0.0007 (clean/clean/day) | 28 |
| support iou 0.50 | 2 | 1 | 0 | 0 | -0.0029 (clean/clean/day) | +0.0033 (clean/fog_s3/day) | 28 |
| weak-IR fallback off | 25 | 0 | 25 | 0 | +0.0000 (clean/clean/day) | +0.2552 (clean/noise_s3/night) | 5 |
| IR merge veto on | 0 | 11 | 0 | 6 | -0.0176 (fog/noise_s2/day) | +0.0000 (clean/clean/day) | 30 |
| veil not night-conditional | 0 | 11 | 0 | 8 | -0.1574 (lowlight/clean/day) | +0.0001 (fog/noise_s3/day) | 30 |
| veto off (union) | 30 | 0 | 30 | 0 | +0.0000 (clean/clean/day) | +0.2552 (clean/noise_s3/night) | 1 |
