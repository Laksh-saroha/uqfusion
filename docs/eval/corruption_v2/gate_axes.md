# The shipped gate's VIS axes under v1 and v2 corruptions

Thresholds: dark = p05 < 10.5, veil = grad_gini < 0.4826, concentrated = lap_over_var > 4.278 (fitted on clean frames). Firing rate per slice; dev rows average draws 941-944. `crossmodal26m` vetoes VIS at night on `dark OR veil`; by day it never vetoes with clean IR.

| condition | slice | set | dark | veil | concentrated | dark OR veil | median p05 | median grad_gini |
|---|---|---|---:|---:|---:|---:|---:|---:|
| clean | day | dev v1 (4) | 0.000 | 0.000 | 0.000 | 0.000 | 35.0 | 0.609 |
| clean | day | dev v2 (4) | 0.000 | 0.000 | 0.000 | 0.000 | 35.0 | 0.609 |
| clean | day | R-D1 v1 (1) | 0.000 | 0.000 | 0.000 | 0.000 | 35.0 | 0.609 |
| clean | day | R-D1 v2 (1) | 0.000 | 0.000 | 0.000 | 0.000 | 35.0 | 0.609 |
| clean | night | dev v1 (4) | 1.000 | 0.000 | 1.000 | 1.000 | 2.5 | 0.683 |
| clean | night | dev v2 (4) | 1.000 | 0.000 | 1.000 | 1.000 | 2.5 | 0.683 |
| clean | night | R-D1 v1 (1) | 1.000 | 0.000 | 1.000 | 1.000 | 2.5 | 0.683 |
| clean | night | R-D1 v2 (1) | 1.000 | 0.000 | 1.000 | 1.000 | 2.5 | 0.683 |
| fog | day | dev v1 (4) | 0.000 | 1.000 | 0.000 | 1.000 | 58.0 | 0.410 |
| fog | day | dev v2 (4) | 0.000 | 0.091 | 0.000 | 0.091 | 83.0 | 0.539 |
| fog | day | R-D1 v1 (1) | 0.000 | 1.000 | 0.000 | 1.000 | 58.0 | 0.411 |
| fog | day | R-D1 v2 (1) | 0.000 | 0.090 | 0.000 | 0.090 | 83.0 | 0.540 |
| fog | night | dev v1 (4) | 0.280 | 1.000 | 0.000 | 1.000 | 15.0 | 0.379 |
| fog | night | dev v2 (4) | 0.864 | 0.377 | 1.000 | 0.864 | 5.5 | 0.541 |
| fog | night | R-D1 v1 (1) | 0.295 | 1.000 | 0.000 | 1.000 | 15.0 | 0.379 |
| fog | night | R-D1 v2 (1) | 0.845 | 0.367 | 1.000 | 0.845 | 5.0 | 0.543 |
| lowlight | day | dev v1 (4) | 1.000 | 0.000 | 0.035 | 1.000 | 0.0 | 0.914 |
| lowlight | day | dev v2 (4) | 1.000 | 1.000 | 0.228 | 1.000 | 2.0 | 0.380 |
| lowlight | day | R-D1 v1 (1) | 1.000 | 0.000 | 0.033 | 1.000 | 0.0 | 0.916 |
| lowlight | day | R-D1 v2 (1) | 1.000 | 1.000 | 0.229 | 1.000 | 2.0 | 0.381 |
| lowlight | night | dev v1 (4) | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 | 0.999 |
| lowlight | night | dev v2 (4) | 1.000 | 0.000 | 1.000 | 1.000 | 1.0 | 0.575 |
| lowlight | night | R-D1 v1 (1) | 1.000 | 0.000 | 1.000 | 1.000 | 0.0 | 0.999 |
| lowlight | night | R-D1 v2 (1) | 1.000 | 0.000 | 1.000 | 1.000 | 1.0 | 0.578 |
| glare | day | dev v1 (4) | 0.000 | 0.000 | 0.000 | 0.000 | 36.0 | 0.621 |
| glare | day | dev v2 (4) | 0.000 | 0.000 | 0.000 | 0.000 | 47.0 | 0.666 |
| glare | day | R-D1 v1 (1) | 0.000 | 0.000 | 0.000 | 0.000 | 36.0 | 0.620 |
| glare | day | R-D1 v2 (1) | 0.000 | 0.000 | 0.000 | 0.000 | 47.0 | 0.666 |
| glare | night | dev v1 (4) | 0.998 | 0.000 | 0.125 | 0.998 | 4.0 | 0.765 |
| glare | night | dev v2 (4) | 1.000 | 0.000 | 0.001 | 1.000 | 6.0 | 0.728 |
| glare | night | R-D1 v1 (1) | 1.000 | 0.000 | 0.124 | 1.000 | 3.5 | 0.765 |
| glare | night | R-D1 v2 (1) | 1.000 | 0.000 | 0.000 | 1.000 | 6.0 | 0.728 |
| fog_s1 | day | dev v1 (4) | 0.000 | 0.986 | 0.000 | 0.986 | 46.0 | 0.431 |
| fog_s1 | day | dev v2 (4) | 0.000 | 0.007 | 0.000 | 0.007 | 66.0 | 0.609 |
| fog_s1 | night | dev v1 (4) | 0.804 | 1.000 | 0.000 | 1.000 | 8.0 | 0.417 |
| fog_s1 | night | dev v2 (4) | 1.000 | 0.023 | 1.000 | 1.000 | 4.0 | 0.616 |
