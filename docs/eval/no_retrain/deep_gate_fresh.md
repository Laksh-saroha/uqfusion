# Deep-feature trust (E7), `fresh` (draws 945, 946, 947)

`scripts/deep_gate.py`, registered in `docs/prereg-no-retrain-2026-10-10.md` Amendment 2. Ship AP, five systems, draw mean per system, 95% t-interval (df 4). P1 = claim (fused − max(VIS, IR)) not failing on any cell; P2 = no cell worse than shipped at ≤ −0.0060 with the interval below zero, and at least one better.

| arm | P1 holds on | P2 worse cells | P2 better cells | worse at 0.0014 | mean fused | verdict |
|---|---|---:|---:|---:|---|---|
| shipped | 32/60 | 0 | 0 | 0 | 0.0894 | — |
| deep | 60/60 | 5 | 28 | 17 | 0.1485 | FAIL |
| deep_nb | 60/60 | 5 | 28 | 15 | 0.1505 | FAIL |

| VIS | IR | slice | VIS | IR | shipped | deep | deep_nb |
|---|---|---|---:|---:|---:|---:|---:|
| clean | clean | day | 0.3485 | 0.0218 | 0.3566 | 0.3397 | 0.3397 |
| clean | clean | night | 0.2535 | 0.0687 | 0.0687 | 0.2763 | 0.2861 |
| fog | clean | day | 0.0105 | 0.0218 | 0.0239 | 0.0214 | 0.0214 |
| fog | clean | night | 0.0432 | 0.0687 | 0.0687 | 0.0797 | 0.0818 |
| fog_s1 | clean | day | 0.0267 | 0.0218 | 0.0358 | 0.0327 | 0.0327 |
| fog_s1 | clean | night | 0.1059 | 0.0687 | 0.0687 | 0.1379 | 0.1395 |
| glare | clean | day | 0.1550 | 0.0218 | 0.1672 | 0.1597 | 0.1597 |
| glare | clean | night | 0.0815 | 0.0687 | 0.0687 | 0.1230 | 0.1256 |
| lowlight | clean | day | 0.1705 | 0.0218 | 0.1775 | 0.1663 | 0.1663 |
| lowlight | clean | night | 0.2383 | 0.0687 | 0.0687 | 0.2599 | 0.2688 |
| clean | fog_s1 | day | 0.3485 | 0.0257 | 0.3552 | 0.3440 | 0.3440 |
| clean | fog_s1 | night | 0.2535 | 0.0577 | 0.0577 | 0.2702 | 0.2816 |
| fog | fog_s1 | day | 0.0105 | 0.0257 | 0.0299 | 0.0270 | 0.0270 |
| fog | fog_s1 | night | 0.0432 | 0.0577 | 0.0577 | 0.0720 | 0.0742 |
| fog_s1 | fog_s1 | day | 0.0267 | 0.0257 | 0.0427 | 0.0388 | 0.0388 |
| fog_s1 | fog_s1 | night | 0.1059 | 0.0577 | 0.0577 | 0.1312 | 0.1336 |
| glare | fog_s1 | day | 0.1550 | 0.0257 | 0.1717 | 0.1657 | 0.1657 |
| glare | fog_s1 | night | 0.0815 | 0.0577 | 0.1194 | 0.1159 | 0.1192 |
| lowlight | fog_s1 | day | 0.1705 | 0.0257 | 0.1609 | 0.1739 | 0.1739 |
| lowlight | fog_s1 | night | 0.2383 | 0.0577 | 0.0577 | 0.2536 | 0.2644 |
| clean | fog_s2 | day | 0.3485 | 0.0130 | 0.3459 | 0.3409 | 0.3409 |
| clean | fog_s2 | night | 0.2535 | 0.0315 | 0.0351 | 0.2584 | 0.2694 |
| fog | fog_s2 | day | 0.0105 | 0.0130 | 0.0194 | 0.0173 | 0.0173 |
| fog | fog_s2 | night | 0.0432 | 0.0315 | 0.0316 | 0.0576 | 0.0577 |
| fog_s1 | fog_s2 | day | 0.0267 | 0.0130 | 0.0341 | 0.0308 | 0.0308 |
| fog_s1 | fog_s2 | night | 0.1059 | 0.0315 | 0.0325 | 0.1189 | 0.1200 |
| glare | fog_s2 | day | 0.1550 | 0.0130 | 0.1625 | 0.1576 | 0.1576 |
| glare | fog_s2 | night | 0.0815 | 0.0315 | 0.1034 | 0.0996 | 0.1017 |
| lowlight | fog_s2 | day | 0.1705 | 0.0130 | 0.1642 | 0.1687 | 0.1687 |
| lowlight | fog_s2 | night | 0.2383 | 0.0315 | 0.0346 | 0.2428 | 0.2519 |
| clean | fog_s3 | day | 0.3485 | 0.0034 | 0.3386 | 0.3414 | 0.3414 |
| clean | fog_s3 | night | 0.2535 | 0.0019 | 0.0090 | 0.2464 | 0.2541 |
| fog | fog_s3 | day | 0.0105 | 0.0034 | 0.0123 | 0.0114 | 0.0114 |
| fog | fog_s3 | night | 0.0432 | 0.0019 | 0.0057 | 0.0426 | 0.0403 |
| fog_s1 | fog_s3 | day | 0.0267 | 0.0034 | 0.0285 | 0.0264 | 0.0264 |
| fog_s1 | fog_s3 | night | 0.1059 | 0.0019 | 0.0075 | 0.1046 | 0.1041 |
| glare | fog_s3 | day | 0.1550 | 0.0034 | 0.1552 | 0.1523 | 0.1523 |
| glare | fog_s3 | night | 0.0815 | 0.0019 | 0.0830 | 0.0805 | 0.0816 |
| lowlight | fog_s3 | day | 0.1705 | 0.0034 | 0.0577 | 0.1679 | 0.1679 |
| lowlight | fog_s3 | night | 0.2383 | 0.0019 | 0.0091 | 0.2314 | 0.2375 |
| clean | noise_s2 | day | 0.3485 | 0.0299 | 0.3654 | 0.3419 | 0.3419 |
| clean | noise_s2 | night | 0.2535 | 0.0000 | 0.0000 | 0.2461 | 0.2540 |
| fog | noise_s2 | day | 0.0105 | 0.0299 | 0.0282 | 0.0249 | 0.0249 |
| fog | noise_s2 | night | 0.0432 | 0.0000 | 0.0000 | 0.0421 | 0.0398 |
| fog_s1 | noise_s2 | day | 0.0267 | 0.0299 | 0.0395 | 0.0354 | 0.0354 |
| fog_s1 | noise_s2 | night | 0.1059 | 0.0000 | 0.0000 | 0.1040 | 0.1038 |
| glare | noise_s2 | day | 0.1550 | 0.0299 | 0.1693 | 0.1588 | 0.1588 |
| glare | noise_s2 | night | 0.0815 | 0.0000 | 0.0828 | 0.0802 | 0.0815 |
| lowlight | noise_s2 | day | 0.1705 | 0.0299 | 0.1580 | 0.1676 | 0.1676 |
| lowlight | noise_s2 | night | 0.2383 | 0.0000 | 0.0000 | 0.2311 | 0.2374 |
| clean | noise_s3 | day | 0.3485 | 0.0002 | 0.3499 | 0.3381 | 0.3381 |
| clean | noise_s3 | night | 0.2535 | 0.0000 | 0.0000 | 0.2462 | 0.2539 |
| fog | noise_s3 | day | 0.0105 | 0.0002 | 0.0106 | 0.0098 | 0.0098 |
| fog | noise_s3 | night | 0.0432 | 0.0000 | 0.0000 | 0.0423 | 0.0399 |
| fog_s1 | noise_s3 | day | 0.0267 | 0.0002 | 0.0269 | 0.0248 | 0.0248 |
| fog_s1 | noise_s3 | night | 0.1059 | 0.0000 | 0.0000 | 0.1044 | 0.1040 |
| glare | noise_s3 | day | 0.1550 | 0.0002 | 0.1582 | 0.1500 | 0.1500 |
| glare | noise_s3 | night | 0.0815 | 0.0000 | 0.0828 | 0.0801 | 0.0814 |
| lowlight | noise_s3 | day | 0.1705 | 0.0002 | 0.0066 | 0.1667 | 0.1667 |
| lowlight | noise_s3 | night | 0.2383 | 0.0000 | 0.0000 | 0.2310 | 0.2375 |
