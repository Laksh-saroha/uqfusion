# N4Fg (N4F with N4g's glare term) on pohang04's glare cell

Development verdict: **N4Fg PASSES development** (`ttbn_fulltta_glare_dev_2026-10-12.md`). Under Amendment 1 of the registration pohang04 was scored whatever that verdict; development passed, so the original order held.

`docs/prereg-ttbn-fulltta-glare-2026-10-12.md`. Ship AP against the VIS ground truth; every arm here is an ensemble of all five systems, mean over the four draws (VIS v2 glare severity 2, draws 941–944, IR clean). Δ = `full_tta_ttbng` − `full_tta`, paired; 95% interval from the gate-free scoring's 1,000 moving-block replicates (L = 20, seed 1).

**P-FG: REPLICATES.**

| glare/clean | ship AP | Δ vs full_tta [95%] |
|---|---:|---:|
| `full_tta` | 0.1656 | |
| N4F (`full_tta_ttbn`, N4's gate 1.9%) | 0.1656 | +0.0000 [-0.0000, +0.0000] |
| **N4Fg** (`full_tta_ttbng`, N4g's gate 81.0%) | **0.1967** | **+0.0311 [+0.0286, +0.0332]** |

Descriptive (one pass per stream, P-G):
* `union` 0.1417, N4g (`union_ttbng`) 0.1726; N4Fg − N4g: +0.0240 [+0.0216, +0.0261].

## The other pohang04 cells (gate-equality check against P-FT)

Where N4g's gate equals N4's on every draw, N4Fg is N4F (P-FT's values). Where it adds frames, N4Fg is built and scored; a cell fails if N4Fg − N4F ≤ −0.0014 with its interval below zero.

| cell (VIS/IR) | N4g gate | N4 gate | N4Fg | N4Fg − N4F [95%] |
|---|---:|---:|---:|---:|
| clean/clean | 0.000 | 0.000 | = N4F (gates equal on every draw) | 0 |
| clean/blur_s2 | 0.000 | 0.000 | 0.2936 (N4F 0.2936) | +0.0000 [-0.0000, +0.0000] |
| clean/noise_s2 | 0.000 | 0.000 | = N4F (gates equal on every draw) | 0 |
| clean/fog_s2 | 0.000 | 0.000 | 0.2923 (N4F 0.2923) | +0.0000 [-0.0000, +0.0000] |
| blur_s3/clean | 0.092 | 0.092 | = N4F (gates equal on every draw) | 0 |
| noise_s2/clean | 0.852 | 0.852 | = N4F (gates equal on every draw) | 0 |
| rain_s2/clean | 0.186 | 0.186 | = N4F (gates equal on every draw) | 0 |
| fog/clean | 0.678 | 0.678 | 0.1031 (N4F 0.1031) | +0.0000 [+0.0000, +0.0000] |
| lowlight/clean | 0.852 | 0.852 | = N4F (gates equal on every draw) | 0 |
