# N4Fg (N4F with N4g's glare term) on development

`docs/prereg-ttbn-fulltta-glare-2026-10-12.md`. Draws VIS 945–947 / IR 955–957 (N4F's; a third use). Ship AP; Δ = `full_tta_ttbng` − `full_tta`, mean over the three draws; 95% interval from N4F's 1,000 moving-block replicates (L = 20, seed 1) over the slice's frames. `full_tta` recomputed here equals its per-draw no-retrain values (|difference| < 5e-5), `docs/eval/no_retrain/ensembles_fresh.json` to the fourth decimal, and N4F's development `full_tta` exactly, in all 60 cells. Where the gates differ, N4F recomputed here equals N4F's stored scores exactly.

Precision: fp16 (check on glare/clean/day, draw 945: fp32 0.21928, fp16 0.21922, Δ -0.00006, tolerance 0.001; PASS).

**N4Fg PASSES development.** No cell worse than `full_tta`: PASS. Cells where the glare term adds gated frames at or above −0.0014 against N4F: PASS.

| cell (VIS/IR/slice) | N4 gate | N4g gate | full_tta | full_tta_ttbng | Δ vs full_tta [95%] | N4Fg − N4F [95%] |
|---|---:|---:|---:|---:|---:|---:|
| clean/clean/day | 0.000 | 0.000 | 0.3801 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/clean/night | 0.000 | 0.000 | 0.3192 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/fog_s1/day | 0.000 | 0.000 | 0.3793 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/fog_s1/night | 0.000 | 0.000 | 0.3229 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/fog_s2/day | 0.000 | 0.000 | 0.3749 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/fog_s2/night | 0.000 | 0.000 | 0.3178 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/fog_s3/day | 0.000 | 0.000 | 0.3656 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/fog_s3/night | 0.000 | 0.000 | 0.2889 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/noise_s2/day | 0.000 | 0.000 | 0.3793 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/noise_s2/night | 0.000 | 0.000 | 0.3221 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/noise_s3/day | 0.000 | 0.000 | 0.3807 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| clean/noise_s3/night | 0.000 | 0.000 | 0.2994 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog/clean/day | 0.995 | 0.995 | 0.0288 | = N4F | +0.0130 [+0.0079, +0.0176] (N4F's) | 0 (gates equal) |
| fog/clean/night | 0.000 | 0.000 | 0.0982 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog/fog_s1/day | 0.428 | 0.428 | 0.0396 | = N4F | +0.0037 [+0.0004, +0.0060] (N4F's) | 0 (gates equal) |
| fog/fog_s1/night | 0.000 | 0.000 | 0.0984 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog/fog_s2/day | 0.152 | 0.152 | 0.0281 | = N4F | +0.0048 [+0.0018, +0.0071] (N4F's) | 0 (gates equal) |
| fog/fog_s2/night | 0.000 | 0.000 | 0.0924 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog/fog_s3/day | 0.050 | 0.050 | 0.0174 | = N4F | +0.0027 [+0.0007, +0.0041] (N4F's) | 0 (gates equal) |
| fog/fog_s3/night | 0.000 | 0.000 | 0.0610 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog/noise_s2/day | 0.000 | 0.000 | 0.0411 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog/noise_s2/night | 0.000 | 0.000 | 0.0927 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog/noise_s3/day | 0.000 | 0.000 | 0.0339 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog/noise_s3/night | 0.000 | 0.000 | 0.0626 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog_s1/clean/day | 0.623 | 0.623 | 0.0458 | = N4F | +0.0096 [+0.0045, +0.0155] (N4F's) | 0 (gates equal) |
| fog_s1/clean/night | 0.000 | 0.000 | 0.1666 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog_s1/fog_s1/day | 0.186 | 0.186 | 0.0547 | = N4F | +0.0004 [-0.0019, +0.0017] (N4F's) | 0 (gates equal) |
| fog_s1/fog_s1/night | 0.000 | 0.000 | 0.1697 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog_s1/fog_s2/day | 0.043 | 0.043 | 0.0452 | = N4F | +0.0007 [-0.0004, +0.0013] (N4F's) | 0 (gates equal) |
| fog_s1/fog_s2/night | 0.000 | 0.000 | 0.1666 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog_s1/fog_s3/day | 0.005 | 0.005 | 0.0382 | = N4F | +0.0005 [+0.0001, +0.0009] (N4F's) | 0 (gates equal) |
| fog_s1/fog_s3/night | 0.000 | 0.000 | 0.1345 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog_s1/noise_s2/day | 0.000 | 0.000 | 0.0555 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog_s1/noise_s2/night | 0.000 | 0.000 | 0.1684 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog_s1/noise_s3/day | 0.000 | 0.000 | 0.0499 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| fog_s1/noise_s3/night | 0.000 | 0.000 | 0.1384 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| lowlight/clean/day | 1.000 | 1.000 | 0.2141 | = N4F | +0.0696 [+0.0620, +0.0757] (N4F's) | 0 (gates equal) |
| lowlight/clean/night | 0.000 | 0.000 | 0.3013 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| lowlight/fog_s1/day | 0.428 | 0.428 | 0.2138 | = N4F | +0.0385 [+0.0270, +0.0480] (N4F's) | 0 (gates equal) |
| lowlight/fog_s1/night | 0.000 | 0.000 | 0.3046 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| lowlight/fog_s2/day | 0.152 | 0.152 | 0.2102 | = N4F | +0.0071 [+0.0020, +0.0103] (N4F's) | 0 (gates equal) |
| lowlight/fog_s2/night | 0.000 | 0.000 | 0.2998 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| lowlight/fog_s3/day | 0.050 | 0.050 | 0.2039 | = N4F | +0.0012 [-0.0015, +0.0017] (N4F's) | 0 (gates equal) |
| lowlight/fog_s3/night | 0.000 | 0.000 | 0.2731 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| lowlight/noise_s2/day | 0.000 | 0.000 | 0.2136 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| lowlight/noise_s2/night | 0.000 | 0.000 | 0.3026 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| lowlight/noise_s3/day | 0.000 | 0.000 | 0.2119 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| lowlight/noise_s3/night | 0.000 | 0.000 | 0.2825 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| glare/clean/day | 0.002 | 1.000 | 0.2012 | 0.2191 | +0.0179 [+0.0141, +0.0207] | +0.0179 [+0.0141, +0.0208] |
| glare/clean/night | 0.000 | 0.000 | 0.1598 | 0.1598 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| glare/fog_s1/day | 0.001 | 0.428 | 0.2057 | 0.2146 | +0.0089 [+0.0052, +0.0121] | +0.0089 [+0.0052, +0.0121] |
| glare/fog_s1/night | 0.000 | 0.000 | 0.1617 | 0.1617 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| glare/fog_s2/day | 0.001 | 0.152 | 0.1986 | 0.2000 | +0.0014 [-0.0007, +0.0028] | +0.0014 [-0.0007, +0.0029] |
| glare/fog_s2/night | 0.000 | 0.000 | 0.1557 | 0.1557 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| glare/fog_s3/day | 0.001 | 0.050 | 0.1895 | 0.1890 | -0.0006 [-0.0018, +0.0001] | -0.0005 [-0.0018, +0.0001] |
| glare/fog_s3/night | 0.000 | 0.000 | 0.1204 | 0.1204 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| glare/noise_s2/day | 0.000 | 0.000 | 0.2042 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| glare/noise_s2/night | 0.000 | 0.000 | 0.1610 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| glare/noise_s3/day | 0.000 | 0.000 | 0.2024 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
| glare/noise_s3/night | 0.000 | 0.000 | 0.1229 | = N4F | +0.0000 [+0.0000, +0.0000] (N4F's) | 0 (gates equal) |
