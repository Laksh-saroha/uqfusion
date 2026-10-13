# IR night switch under IR corruption, corruption v2 (Table 6)

All 2232 paired IR frames (1200 day / 1032 night, pohang01), stride 1, corruption seed 7 (as the v1 probes), v2 IR corruptions on content rows. Shipped constants: `ir_p05` > 41.5; IR health authority bound 64.9, merge bound 222.7; VIS `mu_b` 10.5; `grad_gini` < 0.4826; `lap_over_var` > 4.278. v2 does not model IR glare, lowlight, rain (a thermal sensor does not see a visible-light flare, an exposure cut or RGB rain streaks), so those v1 rows have no v2 counterpart.

## 1. Raw rule and hardened vote

| IR arm | day p05 med | night p05 med | **false night, raw** | v1 raw | missed night, raw | separable after refit | **false night, hardened** | missed night, hardened | health flag day / night |
|---|---:|---:|---:|---:|---:|---|---:|---:|---|
| clean | 18.0 | 75.0 | **0.0%** | — | 0.0% | yes | **0.0%** | 0.0% | 1.9% / 0.0% |
| blur s1 | 19.0 | 55.5 | **0.0%** | 0.0% | 0.0% | yes | **0.0%** | 16.9% | 14.1% / 16.9% |
| blur s2 | 19.5 | 53.0 | **0.0%** | 0.0% | 6.5% | no | **0.0%** | 57.8% | 19.2% / 57.8% |
| blur s3 | 20.0 | 51.0 | **0.0%** | 0.0% | 24.6% | no | **0.0%** | 83.3% | 35.1% / 83.3% |
| fog s1 | 20.0 | 110.0 | **13.4%** | 43.8% | 0.0% | no | **0.0%** | 99.8% | 93.2% / 99.8% |
| fog s2 | 25.0 | 94.0 | **4.8%** | 86.7% | 2.1% | no | **0.0%** | 100.0% | 96.9% / 100.0% |
| fog s3 | 46.0 | 58.0 | **61.4%** | 94.8% | 1.2% | no | **0.0%** | 100.0% | 99.7% / 100.0% |
| noise s1 | 27.0 | 78.0 | **0.0%** | 0.0% | 0.0% | yes | **0.0%** | 100.0% | 100.0% / 100.0% |
| noise s2 | 38.0 | 80.0 | **16.7%** | 0.0% | 0.0% | yes | **0.0%** | 100.0% | 100.0% / 100.0% |
| noise s3 | 53.0 | 82.0 | **96.3%** | 0.0% | 0.0% | no | **0.0%** | 100.0% | 100.0% / 100.0% |

Worst raw false night: **96.3%** (noise s3). Worst hardened false night: **0.0%** (blur s1).

## 2. The shipped veto, replayed (day frames; VIS the better stream except fog)

`crossmodal26m`: veto = IR night (hardened) AND (VIS dark OR veil), OR the weak-IR fallback (raw IR night, IR disarmed, VIS concentrated-highlight or dark-and-veiled). A day veto on clean, lowlight or glare VIS throws away the better stream.

| IR arm | VIS clean | VIS lowlight | VIS glare | VIS fog (veto correct) |
|---|---:|---:|---:|---:|
| clean | 0.0% | 0.0% | 0.0% | 0.0% |
| blur s1 | 0.0% | 0.0% | 0.0% | 0.0% |
| blur s2 | 0.0% | 0.0% | 0.0% | 0.0% |
| blur s3 | 0.0% | 0.0% | 0.0% | 0.0% |
| fog s1 | 0.0% | 13.4% | 0.0% | 0.0% |
| fog s2 | 0.0% | 4.8% | 0.0% | 0.0% |
| fog s3 | 0.0% | 61.4% | 0.0% | 0.0% |
| noise s1 | 0.0% | 0.0% | 0.0% | 0.0% |
| noise s2 | 0.0% | 16.7% | 0.0% | 0.0% |
| noise s3 | 0.0% | 96.3% | 0.0% | 0.0% |

Worst harmful day veto over the corrupted IR arms: **96.3%** (IR noise s3 + VIS lowlight).

The fallback and both IR bounds were fitted with data that includes pohang01, the only night run (§4.2, §9), so the night columns are in-sample.
