# Corruption v2 sensitivity rows

One constant moved per row; shipped v2 cell as the baseline; five Phase 3 systems; VIS draw 941 (IR 951); paired val (day = pohang00/02/03, night = pohang01). Ship AP. Δ = row − baseline, mean over the five systems, 95% t-interval (df 4).

| row | what moved | slice | baseline fused | Δ fused [95% CI] | Δ VIS-only | Δ IR-only | veto rate (row / base) |
|---|---|---|---:|---|---:|---:|---|
| glare_ae0 | AE_STRENGTH 0.375 -> 0 (no AE reaction) | day | 0.1703 | -0.0069 [-0.0084, -0.0055] | -0.0071 | +0.0000 | 0.000 / 0.000 |
| glare_ae0 | AE_STRENGTH 0.375 -> 0 (no AE reaction) | night | 0.0687 | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 | 1.000 / 1.000 |
| glare_ae1 | AE_STRENGTH 0.375 -> 1 (fully mean-holding AE) | day | 0.1703 | +0.0035 [+0.0020, +0.0050] | +0.0053 | +0.0000 | 0.000 / 0.000 |
| glare_ae1 | AE_STRENGTH 0.375 -> 1 (fully mean-holding AE) | night | 0.0687 | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 | 1.000 / 1.000 |
| glare_i0half | source intensity x0.5 | day | 0.1703 | +0.0360 [+0.0332, +0.0388] | +0.0363 | +0.0000 | 0.000 / 0.000 |
| glare_i0half | source intensity x0.5 | night | 0.0687 | +0.0000 [+0.0000, +0.0000] | +0.0213 | +0.0000 | 1.000 / 1.000 |
| glare_i0x2 | source intensity x2 | day | 0.1703 | -0.0362 [-0.0380, -0.0345] | -0.0359 | +0.0000 | 0.000 / 0.000 |
| glare_i0x2 | source intensity x2 | night | 0.0687 | -0.0001 [-0.0006, +0.0003] | -0.0222 | +0.0000 | 0.916 / 1.000 |
| glare_rev2 | revision-2 glare: Lorentzian spread (beta 1), its fitted intensities, no AE | day | 0.1703 | +0.0480 [+0.0453, +0.0506] | +0.0499 | +0.0000 | 0.000 / 0.000 |
| glare_rev2 | revision-2 glare: Lorentzian spread (beta 1), its fitted intensities, no AE | night | 0.0687 | +0.0057 [+0.0007, +0.0107] | -0.0677 | +0.0000 | 0.000 / 1.000 |
| fog_ae0 | AE_STRENGTH 0.375 -> 0 on fog | day | 0.0240 | -0.0002 [-0.0009, +0.0006] | +0.0001 | +0.0000 | 0.000 / 0.000 |
| fog_ae0 | AE_STRENGTH 0.375 -> 0 on fog | night | 0.0688 | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 | 0.874 / 0.874 |
| fog_ae1 | AE_STRENGTH 0.375 -> 1 on fog (a converged AE in persistent fog) | day | 0.0240 | +0.0008 [-0.0009, +0.0024] | +0.0000 | +0.0000 | 0.000 / 0.000 |
| fog_ae1 | AE_STRENGTH 0.375 -> 1 on fog (a converged AE in persistent fog) | night | 0.0688 | +0.0000 [+0.0000, +0.0000] | +0.0000 | +0.0000 | 0.874 / 0.874 |
| ir_fog_b03 | IR_BETA_RATIO 0.5 -> 0.3 | day | 0.3460 | +0.0070 [+0.0039, +0.0101] | +0.0000 | +0.0091 | 0.000 / 0.000 |
| ir_fog_b03 | IR_BETA_RATIO 0.5 -> 0.3 | night | 0.0373 | +0.0139 [+0.0090, +0.0188] | +0.0000 | +0.0207 | 0.998 / 0.979 |
| ir_fog_b10 | IR_BETA_RATIO 0.5 -> 1.0 | day | 0.3460 | -0.0086 [-0.0120, -0.0051] | +0.0000 | -0.0079 | 0.000 / 0.000 |
| ir_fog_b10 | IR_BETA_RATIO 0.5 -> 1.0 | night | 0.0373 | -0.0261 [-0.0366, -0.0155] | +0.0000 | -0.0271 | 0.982 / 0.979 |
