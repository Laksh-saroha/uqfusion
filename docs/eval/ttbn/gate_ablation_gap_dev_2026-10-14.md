# Gate ablation, Amendment 2: Yoo et al.'s feature-gap gate, development

`docs/prereg-gate-ablation-2026-10-14.md`, Amendment 2. Draws VIS 942–944 / IR 952–954, five systems, 60 cells; same adapted records, fusion, statistics and rule as the other arms (`gate_ablation_dev_2026-10-14.md`). Δ paired against `union`, IG-BN and the clock gate (`q_vis` ∧ solar).

| arm | better | worse | verdict vs `union` | vs IG-BN (better / worse) | vs clock gate (better / worse) |
|---|---:|---:|---|---|---|
| gap (Yoo's rule) | 23 | 18 | FAIL | 20 / 18 | 12 / 18 |
| gap (calibrated bound) | 23 | 18 | FAIL | 20 / 19 | 12 / 24 |
| q_vis ∧ gap (Yoo) | 17 | 12 | FAIL | 13 / 12 | 0 / 12 |
| q_vis ∧ gap (calibrated) | 17 | 12 | FAIL | 13 / 13 | 0 / 18 |

**GA-6: a model-internal feature gap CANNOT stand in for the day term on this data** (`q_vis` ∧ gap fails development).

**gap (Yoo's rule).** Worse than `union`: clean/clean/day, clean/clean/night, clean/fog_s1/day, clean/fog_s1/night, clean/fog_s2/day, clean/fog_s2/night, clean/fog_s3/day, clean/fog_s3/night, clean/noise_s2/day, clean/noise_s2/night, clean/noise_s3/day, clean/noise_s3/night, lowlight/clean/night, lowlight/fog_s1/night, lowlight/fog_s2/night, lowlight/fog_s3/night, lowlight/noise_s2/night, lowlight/noise_s3/night. Better: fog/clean/day, fog/fog_s1/day, fog/fog_s2/day, fog/fog_s3/day, fog/noise_s3/day, fog_s1/clean/day, fog_s1/fog_s1/day, fog_s1/fog_s2/day, fog_s1/fog_s3/day, fog_s1/noise_s2/day, fog_s1/noise_s3/day, lowlight/clean/day, lowlight/fog_s1/day, lowlight/fog_s2/day, lowlight/fog_s3/day, lowlight/noise_s2/day, lowlight/noise_s3/day, glare/clean/day, glare/fog_s1/day, glare/fog_s2/day, glare/fog_s3/day, glare/noise_s2/day, glare/noise_s3/day. Worse than the clock gate: clean/clean/day, clean/clean/night, clean/fog_s1/day, clean/fog_s1/night, clean/fog_s2/day, clean/fog_s2/night, clean/fog_s3/day, clean/fog_s3/night, clean/noise_s2/day, clean/noise_s2/night, clean/noise_s3/day, clean/noise_s3/night, lowlight/clean/night, lowlight/fog_s1/night, lowlight/fog_s2/night, lowlight/fog_s3/night, lowlight/noise_s2/night, lowlight/noise_s3/night. Better than the clock gate: fog_s1/clean/day, fog_s1/fog_s1/day, fog_s1/fog_s2/day, fog_s1/fog_s3/day, fog_s1/noise_s2/day, fog_s1/noise_s3/day, glare/clean/day, glare/fog_s1/day, glare/fog_s2/day, glare/fog_s3/day, glare/noise_s2/day, glare/noise_s3/day.

**gap (calibrated bound).** Worse than `union`: clean/clean/day, clean/clean/night, clean/fog_s1/day, clean/fog_s1/night, clean/fog_s2/day, clean/fog_s2/night, clean/fog_s3/day, clean/fog_s3/night, clean/noise_s2/day, clean/noise_s2/night, clean/noise_s3/day, clean/noise_s3/night, lowlight/clean/night, lowlight/fog_s1/night, lowlight/fog_s2/night, lowlight/fog_s3/night, lowlight/noise_s2/night, lowlight/noise_s3/night. Better: fog/clean/day, fog/fog_s1/day, fog/fog_s2/day, fog/fog_s3/day, fog/noise_s3/day, fog_s1/clean/day, fog_s1/fog_s1/day, fog_s1/fog_s2/day, fog_s1/fog_s3/day, fog_s1/noise_s2/day, fog_s1/noise_s3/day, lowlight/clean/day, lowlight/fog_s1/day, lowlight/fog_s2/day, lowlight/fog_s3/day, lowlight/noise_s2/day, lowlight/noise_s3/day, glare/clean/day, glare/fog_s1/day, glare/fog_s2/day, glare/fog_s3/day, glare/noise_s2/day, glare/noise_s3/day. Worse than the clock gate: clean/clean/day, clean/clean/night, clean/fog_s1/day, clean/fog_s1/night, clean/fog_s2/day, clean/fog_s2/night, clean/fog_s3/day, clean/fog_s3/night, clean/noise_s2/day, clean/noise_s2/night, clean/noise_s3/day, clean/noise_s3/night, lowlight/clean/day, lowlight/clean/night, lowlight/fog_s1/day, lowlight/fog_s1/night, lowlight/fog_s2/day, lowlight/fog_s2/night, lowlight/fog_s3/day, lowlight/fog_s3/night, lowlight/noise_s2/day, lowlight/noise_s2/night, lowlight/noise_s3/day, lowlight/noise_s3/night. Better than the clock gate: fog_s1/clean/day, fog_s1/fog_s1/day, fog_s1/fog_s2/day, fog_s1/fog_s3/day, fog_s1/noise_s2/day, fog_s1/noise_s3/day, glare/clean/day, glare/fog_s1/day, glare/fog_s2/day, glare/fog_s3/day, glare/noise_s2/day, glare/noise_s3/day.

**q_vis ∧ gap (Yoo).** Worse than `union`: clean/clean/night, clean/fog_s1/night, clean/fog_s2/night, clean/fog_s3/night, clean/noise_s2/night, clean/noise_s3/night, lowlight/clean/night, lowlight/fog_s1/night, lowlight/fog_s2/night, lowlight/fog_s3/night, lowlight/noise_s2/night, lowlight/noise_s3/night. Better: fog/clean/day, fog/fog_s1/day, fog/fog_s2/day, fog/fog_s3/day, fog/noise_s3/day, fog_s1/clean/day, fog_s1/fog_s1/day, fog_s1/fog_s2/day, fog_s1/fog_s3/day, fog_s1/noise_s2/day, fog_s1/noise_s3/day, lowlight/clean/day, lowlight/fog_s1/day, lowlight/fog_s2/day, lowlight/fog_s3/day, lowlight/noise_s2/day, lowlight/noise_s3/day. Worse than the clock gate: clean/clean/night, clean/fog_s1/night, clean/fog_s2/night, clean/fog_s3/night, clean/noise_s2/night, clean/noise_s3/night, lowlight/clean/night, lowlight/fog_s1/night, lowlight/fog_s2/night, lowlight/fog_s3/night, lowlight/noise_s2/night, lowlight/noise_s3/night. Better than the clock gate: none.

**q_vis ∧ gap (calibrated).** Worse than `union`: clean/clean/night, clean/fog_s1/night, clean/fog_s2/night, clean/fog_s3/night, clean/noise_s2/night, clean/noise_s3/night, lowlight/clean/night, lowlight/fog_s1/night, lowlight/fog_s2/night, lowlight/fog_s3/night, lowlight/noise_s2/night, lowlight/noise_s3/night. Better: fog/clean/day, fog/fog_s1/day, fog/fog_s2/day, fog/fog_s3/day, fog/noise_s3/day, fog_s1/clean/day, fog_s1/fog_s1/day, fog_s1/fog_s2/day, fog_s1/fog_s3/day, fog_s1/noise_s2/day, fog_s1/noise_s3/day, lowlight/clean/day, lowlight/fog_s1/day, lowlight/fog_s2/day, lowlight/fog_s3/day, lowlight/noise_s2/day, lowlight/noise_s3/day. Worse than the clock gate: clean/clean/night, clean/fog_s1/night, clean/fog_s2/night, clean/fog_s3/night, clean/noise_s2/night, clean/noise_s3/night, lowlight/clean/day, lowlight/clean/night, lowlight/fog_s1/day, lowlight/fog_s1/night, lowlight/fog_s2/day, lowlight/fog_s2/night, lowlight/fog_s3/day, lowlight/fog_s3/night, lowlight/noise_s2/day, lowlight/noise_s2/night, lowlight/noise_s3/day, lowlight/noise_s3/night. Better than the clock gate: none.

## Per cell: gate rate and Δ vs `union` [95%]

| cell | union | gap (Yoo's rule) | gap (calibrated bound) | q_vis ∧ gap (Yoo) | q_vis ∧ gap (calibrated) |
|---|---:|---|---|---|---|
| clean/clean/day | 0.3566 | 0.99: -0.0100 [-0.0144, -0.0057] | 0.61: -0.0060 [-0.0112, -0.0008] | 0.00: +0.0000 [+0.0000, +0.0000] | 0.00: +0.0000 [+0.0000, +0.0000] |
| clean/clean/night | 0.2895 | 1.00: -0.0195 [-0.0288, -0.0101] | 0.89: -0.0180 [-0.0264, -0.0096] | 1.00: -0.0195 [-0.0288, -0.0101] | 0.89: -0.0180 [-0.0264, -0.0096] |
| clean/fog_s1/day | 0.3553 | 0.99: -0.0112 [-0.0158, -0.0067] | 0.61: -0.0064 [-0.0118, -0.0011] | 0.00: +0.0000 [+0.0000, +0.0000] | 0.00: +0.0000 [+0.0000, +0.0000] |
| clean/fog_s1/night | 0.2846 | 1.00: -0.0193 [-0.0281, -0.0106] | 0.89: -0.0180 [-0.0258, -0.0102] | 1.00: -0.0193 [-0.0281, -0.0106] | 0.89: -0.0180 [-0.0258, -0.0102] |
| clean/fog_s2/day | 0.3457 | 0.99: -0.0096 [-0.0143, -0.0050] | 0.61: -0.0056 [-0.0098, -0.0015] | 0.00: +0.0000 [+0.0000, +0.0000] | 0.00: +0.0000 [+0.0000, +0.0000] |
| clean/fog_s2/night | 0.2697 | 1.00: -0.0183 [-0.0255, -0.0112] | 0.89: -0.0168 [-0.0229, -0.0108] | 1.00: -0.0183 [-0.0255, -0.0112] | 0.89: -0.0168 [-0.0229, -0.0108] |
| clean/fog_s3/day | 0.3380 | 0.99: -0.0097 [-0.0144, -0.0049] | 0.61: -0.0064 [-0.0108, -0.0020] | 0.00: +0.0000 [+0.0000, +0.0000] | 0.00: +0.0000 [+0.0000, +0.0000] |
| clean/fog_s3/night | 0.2556 | 1.00: -0.0190 [-0.0269, -0.0111] | 0.89: -0.0176 [-0.0245, -0.0106] | 1.00: -0.0190 [-0.0269, -0.0111] | 0.89: -0.0176 [-0.0245, -0.0106] |
| clean/noise_s2/day | 0.3655 | 0.99: -0.0094 [-0.0125, -0.0063] | 0.61: -0.0051 [-0.0096, -0.0006] | 0.00: +0.0000 [+0.0000, +0.0000] | 0.00: +0.0000 [+0.0000, +0.0000] |
| clean/noise_s2/night | 0.2552 | 1.00: -0.0192 [-0.0274, -0.0109] | 0.89: -0.0175 [-0.0248, -0.0103] | 1.00: -0.0192 [-0.0274, -0.0109] | 0.89: -0.0175 [-0.0248, -0.0103] |
| clean/noise_s3/day | 0.3498 | 0.99: -0.0096 [-0.0151, -0.0041] | 0.61: -0.0062 [-0.0113, -0.0011] | 0.00: +0.0000 [+0.0000, +0.0000] | 0.00: +0.0000 [+0.0000, +0.0000] |
| clean/noise_s3/night | 0.2552 | 1.00: -0.0192 [-0.0274, -0.0109] | 0.89: -0.0176 [-0.0248, -0.0103] | 1.00: -0.0192 [-0.0274, -0.0109] | 0.89: -0.0176 [-0.0248, -0.0103] |
| fog/clean/day | 0.0239 | 0.99: +0.0093 [+0.0046, +0.0139] | 0.86: +0.0094 [+0.0048, +0.0139] | 0.99: +0.0091 [+0.0045, +0.0137] | 0.85: +0.0092 [+0.0047, +0.0137] |
| fog/clean/night | 0.0827 | 1.00: -0.0028 [-0.0100, +0.0043] | 1.00: -0.0028 [-0.0100, +0.0043] | 1.00: -0.0028 [-0.0100, +0.0043] | 1.00: -0.0028 [-0.0100, +0.0043] |
| fog/fog_s1/day | 0.0298 | 0.99: +0.0083 [+0.0021, +0.0145] | 0.86: +0.0086 [+0.0026, +0.0145] | 0.99: +0.0082 [+0.0020, +0.0143] | 0.85: +0.0084 [+0.0025, +0.0143] |
| fog/fog_s1/night | 0.0745 | 1.00: -0.0016 [-0.0084, +0.0052] | 1.00: -0.0016 [-0.0084, +0.0052] | 1.00: -0.0016 [-0.0084, +0.0052] | 1.00: -0.0016 [-0.0084, +0.0052] |
| fog/fog_s2/day | 0.0192 | 0.99: +0.0113 [+0.0053, +0.0173] | 0.86: +0.0114 [+0.0055, +0.0173] | 0.99: +0.0111 [+0.0052, +0.0170] | 0.85: +0.0113 [+0.0054, +0.0171] |
| fog/fog_s2/night | 0.0598 | 1.00: -0.0008 [-0.0074, +0.0058] | 1.00: -0.0008 [-0.0074, +0.0058] | 1.00: -0.0008 [-0.0074, +0.0058] | 1.00: -0.0008 [-0.0074, +0.0058] |
| fog/fog_s3/day | 0.0121 | 0.99: +0.0140 [+0.0075, +0.0204] | 0.86: +0.0141 [+0.0076, +0.0205] | 0.99: +0.0138 [+0.0074, +0.0202] | 0.85: +0.0139 [+0.0075, +0.0203] |
| fog/fog_s3/night | 0.0436 | 1.00: -0.0006 [-0.0064, +0.0052] | 1.00: -0.0006 [-0.0064, +0.0052] | 1.00: -0.0006 [-0.0064, +0.0052] | 1.00: -0.0006 [-0.0064, +0.0052] |
| fog/noise_s2/day | 0.0285 | 0.99: +0.0052 [-0.0001, +0.0105] | 0.86: +0.0055 [+0.0003, +0.0107] | 0.99: +0.0051 [-0.0002, +0.0103] | 0.85: +0.0053 [+0.0002, +0.0104] |
| fog/noise_s2/night | 0.0436 | 1.00: -0.0008 [-0.0066, +0.0049] | 1.00: -0.0008 [-0.0066, +0.0049] | 1.00: -0.0008 [-0.0066, +0.0049] | 1.00: -0.0008 [-0.0066, +0.0049] |
| fog/noise_s3/day | 0.0107 | 0.99: +0.0137 [+0.0075, +0.0200] | 0.86: +0.0138 [+0.0076, +0.0200] | 0.99: +0.0135 [+0.0074, +0.0196] | 0.85: +0.0136 [+0.0075, +0.0197] |
| fog/noise_s3/night | 0.0436 | 1.00: -0.0008 [-0.0065, +0.0049] | 1.00: -0.0008 [-0.0065, +0.0049] | 1.00: -0.0008 [-0.0065, +0.0049] | 1.00: -0.0008 [-0.0065, +0.0049] |
| fog_s1/clean/day | 0.0360 | 0.99: +0.0368 [+0.0261, +0.0474] | 0.82: +0.0368 [+0.0261, +0.0475] | 0.62: +0.0094 [+0.0047, +0.0142] | 0.50: +0.0095 [+0.0049, +0.0141] |
| fog_s1/clean/night | 0.1416 | 1.00: -0.0077 [-0.0235, +0.0080] | 1.00: -0.0077 [-0.0235, +0.0080] | 1.00: -0.0077 [-0.0235, +0.0080] | 1.00: -0.0077 [-0.0235, +0.0080] |
| fog_s1/fog_s1/day | 0.0428 | 0.99: +0.0347 [+0.0252, +0.0442] | 0.82: +0.0349 [+0.0255, +0.0444] | 0.62: +0.0079 [+0.0033, +0.0126] | 0.50: +0.0082 [+0.0038, +0.0126] |
| fog_s1/fog_s1/night | 0.1368 | 1.00: -0.0071 [-0.0226, +0.0084] | 1.00: -0.0071 [-0.0226, +0.0084] | 1.00: -0.0071 [-0.0226, +0.0084] | 1.00: -0.0071 [-0.0226, +0.0084] |
| fog_s1/fog_s2/day | 0.0342 | 0.99: +0.0365 [+0.0260, +0.0471] | 0.82: +0.0367 [+0.0261, +0.0473] | 0.62: +0.0093 [+0.0044, +0.0142] | 0.50: +0.0095 [+0.0047, +0.0143] |
| fog_s1/fog_s2/night | 0.1236 | 1.00: -0.0068 [-0.0229, +0.0092] | 1.00: -0.0068 [-0.0229, +0.0092] | 1.00: -0.0068 [-0.0229, +0.0092] | 1.00: -0.0068 [-0.0229, +0.0092] |
| fog_s1/fog_s3/day | 0.0288 | 0.99: +0.0380 [+0.0250, +0.0510] | 0.82: +0.0381 [+0.0251, +0.0511] | 0.62: +0.0102 [+0.0042, +0.0162] | 0.50: +0.0103 [+0.0044, +0.0162] |
| fog_s1/fog_s3/night | 0.1071 | 1.00: -0.0060 [-0.0211, +0.0091] | 1.00: -0.0060 [-0.0211, +0.0091] | 1.00: -0.0060 [-0.0211, +0.0091] | 1.00: -0.0060 [-0.0211, +0.0091] |
| fog_s1/noise_s2/day | 0.0394 | 0.99: +0.0334 [+0.0208, +0.0460] | 0.82: +0.0337 [+0.0210, +0.0463] | 0.62: +0.0062 [+0.0011, +0.0113] | 0.50: +0.0064 [+0.0013, +0.0115] |
| fog_s1/noise_s2/night | 0.1071 | 1.00: -0.0062 [-0.0212, +0.0088] | 1.00: -0.0062 [-0.0212, +0.0088] | 1.00: -0.0062 [-0.0212, +0.0088] | 1.00: -0.0062 [-0.0212, +0.0088] |
| fog_s1/noise_s3/day | 0.0270 | 0.99: +0.0394 [+0.0272, +0.0516] | 0.82: +0.0395 [+0.0273, +0.0517] | 0.62: +0.0104 [+0.0058, +0.0151] | 0.50: +0.0106 [+0.0059, +0.0152] |
| fog_s1/noise_s3/night | 0.1070 | 1.00: -0.0062 [-0.0212, +0.0087] | 1.00: -0.0062 [-0.0212, +0.0087] | 1.00: -0.0062 [-0.0212, +0.0087] | 1.00: -0.0062 [-0.0212, +0.0087] |
| lowlight/clean/day | 0.1792 | 0.99: +0.0703 [+0.0591, +0.0816] | 0.72: +0.0498 [+0.0410, +0.0585] | 0.99: +0.0703 [+0.0591, +0.0816] | 0.72: +0.0498 [+0.0410, +0.0585] |
| lowlight/clean/night | 0.2726 | 1.00: -0.0215 [-0.0304, -0.0126] | 1.00: -0.0215 [-0.0304, -0.0126] | 1.00: -0.0215 [-0.0304, -0.0126] | 1.00: -0.0215 [-0.0304, -0.0126] |
| lowlight/fog_s1/day | 0.1818 | 0.99: +0.0686 [+0.0563, +0.0808] | 0.72: +0.0487 [+0.0397, +0.0577] | 0.99: +0.0686 [+0.0563, +0.0808] | 0.72: +0.0487 [+0.0397, +0.0577] |
| lowlight/fog_s1/night | 0.2678 | 1.00: -0.0216 [-0.0306, -0.0126] | 1.00: -0.0216 [-0.0306, -0.0126] | 1.00: -0.0216 [-0.0306, -0.0126] | 1.00: -0.0216 [-0.0306, -0.0126] |
| lowlight/fog_s2/day | 0.1745 | 0.99: +0.0671 [+0.0550, +0.0793] | 0.72: +0.0467 [+0.0368, +0.0565] | 0.99: +0.0671 [+0.0550, +0.0793] | 0.72: +0.0467 [+0.0368, +0.0565] |
| lowlight/fog_s2/night | 0.2536 | 1.00: -0.0214 [-0.0296, -0.0132] | 1.00: -0.0214 [-0.0296, -0.0132] | 1.00: -0.0214 [-0.0296, -0.0132] | 1.00: -0.0214 [-0.0296, -0.0132] |
| lowlight/fog_s3/day | 0.1688 | 0.99: +0.0670 [+0.0548, +0.0792] | 0.72: +0.0470 [+0.0374, +0.0566] | 0.99: +0.0670 [+0.0548, +0.0792] | 0.72: +0.0470 [+0.0374, +0.0566] |
| lowlight/fog_s3/night | 0.2413 | 1.00: -0.0229 [-0.0303, -0.0156] | 1.00: -0.0229 [-0.0303, -0.0156] | 1.00: -0.0229 [-0.0303, -0.0156] | 1.00: -0.0229 [-0.0303, -0.0156] |
| lowlight/noise_s2/day | 0.1799 | 0.99: +0.0706 [+0.0590, +0.0822] | 0.72: +0.0498 [+0.0420, +0.0576] | 0.99: +0.0706 [+0.0590, +0.0822] | 0.72: +0.0498 [+0.0420, +0.0576] |
| lowlight/noise_s2/night | 0.2413 | 1.00: -0.0233 [-0.0306, -0.0160] | 1.00: -0.0233 [-0.0306, -0.0160] | 1.00: -0.0233 [-0.0306, -0.0160] | 1.00: -0.0233 [-0.0306, -0.0160] |
| lowlight/noise_s3/day | 0.1749 | 0.99: +0.0703 [+0.0581, +0.0824] | 0.72: +0.0501 [+0.0413, +0.0589] | 0.99: +0.0703 [+0.0581, +0.0824] | 0.72: +0.0501 [+0.0413, +0.0589] |
| lowlight/noise_s3/night | 0.2413 | 1.00: -0.0233 [-0.0307, -0.0160] | 1.00: -0.0233 [-0.0307, -0.0160] | 1.00: -0.0233 [-0.0307, -0.0160] | 1.00: -0.0233 [-0.0307, -0.0160] |
| glare/clean/day | 0.1700 | 0.99: +0.0237 [+0.0175, +0.0299] | 0.69: +0.0125 [+0.0086, +0.0164] | 0.00: +0.0000 [-0.0000, +0.0000] | 0.00: +0.0000 [-0.0000, +0.0000] |
| glare/clean/night | 0.1293 | 1.00: +0.0028 [-0.0100, +0.0156] | 1.00: +0.0028 [-0.0100, +0.0156] | 1.00: +0.0028 [-0.0100, +0.0156] | 1.00: +0.0028 [-0.0100, +0.0156] |
| glare/fog_s1/day | 0.1745 | 0.99: +0.0221 [+0.0164, +0.0278] | 0.69: +0.0118 [+0.0079, +0.0157] | 0.00: +0.0000 [-0.0000, +0.0000] | 0.00: +0.0000 [-0.0000, +0.0000] |
| glare/fog_s1/night | 0.1220 | 1.00: +0.0037 [-0.0090, +0.0164] | 1.00: +0.0037 [-0.0090, +0.0164] | 1.00: +0.0037 [-0.0090, +0.0164] | 1.00: +0.0037 [-0.0090, +0.0164] |
| glare/fog_s2/day | 0.1652 | 0.99: +0.0222 [+0.0166, +0.0278] | 0.69: +0.0111 [+0.0078, +0.0145] | 0.00: -0.0000 [-0.0000, +0.0000] | 0.00: -0.0000 [-0.0000, +0.0000] |
| glare/fog_s2/night | 0.1058 | 1.00: +0.0059 [-0.0076, +0.0195] | 1.00: +0.0059 [-0.0076, +0.0195] | 1.00: +0.0059 [-0.0076, +0.0195] | 1.00: +0.0059 [-0.0076, +0.0195] |
| glare/fog_s3/day | 0.1582 | 0.99: +0.0217 [+0.0157, +0.0277] | 0.69: +0.0110 [+0.0083, +0.0136] | 0.00: -0.0000 [-0.0001, +0.0000] | 0.00: -0.0000 [-0.0001, +0.0000] |
| glare/fog_s3/night | 0.0854 | 1.00: +0.0094 [-0.0040, +0.0228] | 1.00: +0.0094 [-0.0040, +0.0228] | 1.00: +0.0094 [-0.0040, +0.0228] | 1.00: +0.0094 [-0.0040, +0.0228] |
| glare/noise_s2/day | 0.1721 | 0.99: +0.0240 [+0.0174, +0.0305] | 0.69: +0.0127 [+0.0084, +0.0170] | 0.00: +0.0000 [-0.0000, +0.0001] | 0.00: +0.0000 [-0.0000, +0.0001] |
| glare/noise_s2/night | 0.0851 | 1.00: +0.0094 [-0.0039, +0.0228] | 1.00: +0.0094 [-0.0039, +0.0228] | 1.00: +0.0094 [-0.0039, +0.0228] | 1.00: +0.0094 [-0.0039, +0.0228] |
| glare/noise_s3/day | 0.1610 | 0.99: +0.0235 [+0.0175, +0.0295] | 0.69: +0.0128 [+0.0098, +0.0157] | 0.00: +0.0000 [-0.0001, +0.0001] | 0.00: +0.0000 [-0.0001, +0.0001] |
| glare/noise_s3/night | 0.0851 | 1.00: +0.0094 [-0.0039, +0.0227] | 1.00: +0.0094 [-0.0039, +0.0227] | 1.00: +0.0094 [-0.0039, +0.0227] | 1.00: +0.0094 [-0.0039, +0.0227] |
