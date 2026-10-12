# N4 on top of `full_tta` (N4F) on pohang04

`docs/prereg-ttbn-fulltta-2026-10-11.md`. Ship AP against the VIS ground truth; both arms are ensembles of all five systems, mean over the four draws of a corrupted cell. Δ = `full_tta_ttbn` − `full_tta`, paired; 95% interval from the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1).

**P-FT: REPLICATES.**

| cell (VIS/IR) | gated frames | full_tta | full_tta_ttbn | Δ [95%] |
|---|---:|---:|---:|---:|
| clean/clean | 0 | 0.2925 | = full_tta | 0 (identical) |
| clean/blur_s2 | 0.000 | 0.2936 | 0.2936 | +0.0000 [+0.0000, +0.0000] |
| clean/noise_s2 | 0 | 0.2928 | = full_tta | 0 (identical) |
| clean/fog_s2 | 0 | 0.2923 | = full_tta | 0 (identical) |
| blur_s3/clean | 0.092 | 0.0530 | 0.0531 | +0.0000 [+0.0000, +0.0001] |
| noise_s2/clean | 0.852 | 0.0206 | 0.2380 | +0.2174 [+0.2079, +0.2265] |
| rain_s2/clean | 0.186 | 0.2534 | 0.2567 | +0.0033 [+0.0022, +0.0047] |
| fog/clean | 0.678 | 0.0586 | 0.1031 | +0.0445 [+0.0401, +0.0488] |
| lowlight/clean | 0.852 | 0.2458 | 0.2648 | +0.0190 [+0.0152, +0.0232] |
