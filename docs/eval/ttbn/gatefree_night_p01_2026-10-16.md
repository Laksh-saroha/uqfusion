# Shallow BN-mix and ungated BN-mix on the pohang01 night test

`docs/prereg-gatefree-night-p01-2026-10-16.md`. The night test's 1109 frame pairs, VIS and IR clean; ship AP against the VIS labels, mean over the five Phase 3 systems; Δ paired against the stored `union`; 95% interval from the night test's 1,000 moving-block replicates (L = 20, seed 1). Worse: Δ ≤ −0.0014 with interval below zero; better: Δ ≥ +0.0060 with interval above zero. IG-BN and the clock gate equal `union` here.

**N1 (shallow BN-mix not worse than `union`): PASS.**  
**N2 (ungated BN-mix worse than `union`): FAILS.**  
**N3 (shallow BN-mix within ±0.005 of `union`, stated expectation): HOLDS.**

`union`: 0.0506 (VIS stream 0.0516).

| arm | `union` + arm | Δ vs `union` [95%] | rule | VIS stream | VIS Δ vs plain [95%] | per system |
|---|---:|---:|---|---:|---:|---|
| gf31 | 0.0512 | +0.0006 [-0.0020, +0.0031] | neither | 0.0514 | -0.0002 [-0.0023, +0.0022] | 0.0577, 0.0429, 0.0528, 0.0471, 0.0554 |
| mix16 | 0.0612 | +0.0106 [+0.0043, +0.0163] | better | 0.0598 | +0.0082 [+0.0008, +0.0149] | 0.0720, 0.0530, 0.0631, 0.0537, 0.0642 |

## Descriptive: per test stretch (not registered)

Ship AP, mean over the five systems, on each of the test split's two contiguous stretches (`p01_nighttest.STRETCHES`).

| stretch | `union` | `union` + GF31 | `union` + ungated | VIS plain | VIS ungated |
|---|---:|---:|---:|---:|---:|
| test A (frames 11,059–11,575) | 0.0406 | 0.0406 | 0.0584 | 0.0391 | 0.0551 |
| test B (frames 26,161–26,752) | 0.1095 | 0.1110 | 0.0996 | 0.0956 | 0.0886 |

Ungated BN-mix's gain comes from stretch A (+0.0178); on stretch B it is below `union` (−0.0099). Its development
loss at night (−0.0195 on clean night, Table 21) was measured on pohang01's validation split; these test frames come
from the same run. `mix16` reproduces the development screen's ungated records exactly (`gatefree_screen_2026-10-15.md`),
so the difference is in the frames, not the arithmetic.
