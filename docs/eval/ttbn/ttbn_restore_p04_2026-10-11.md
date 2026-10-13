# N4ae on pohang04 (P-A)

`docs/prereg-ttbn-restore-2026-10-11.md`. P-T's cells, statistic and replicates: ship AP against the VIS ground truth, mean over the five systems and the four draws of a corrupted cell; Δ paired; 95% interval from the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1).

**P-A: REPLICATES.**

| cell (VIS/IR) | gate | dark gated frames | union | N4ae Δ vs union [95%] | N4ae − N4 [95%] |
|---|---:|---:|---:|---:|---:|
| clean/clean | 0 | 0 | 0.2682 | 0 (identical) | |
| clean/blur_s2 | 0.000 | 0 | 0.2673 | +0.0000 [+0.0000, +0.0000] | +0.0000 [+0.0000, +0.0000] |
| clean/noise_s2 | 0 | 0 | 0.2635 | 0 (identical) | |
| clean/fog_s2 | 0 | 0 | 0.2666 | 0 (identical) | |
| blur_s3/clean | 0.092 | 87 | 0.0419 | +0.0001 [+0.0000, +0.0001] | -0.0000 [-0.0000, +0.0000] |
| noise_s2/clean | 0.852 | 126 | 0.0169 | +0.1820 [+0.1732, +0.1913] | -0.0003 [-0.0007, -0.0000] |
| rain_s2/clean | 0.186 | 9291 | 0.2209 | +0.0000 [-0.0000, +0.0001] | -0.0036 [-0.0048, -0.0024] |
| fog/clean | 0.678 | 224 | 0.0445 | +0.0368 [+0.0327, +0.0405] | -0.0002 [-0.0005, +0.0000] |
| lowlight/clean | 0.852 | 42536 | 0.2152 | +0.0259 [+0.0231, +0.0285] | +0.0073 [+0.0046, +0.0099] |
