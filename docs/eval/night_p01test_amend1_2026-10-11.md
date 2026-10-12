# Night test on pohang01's test split

`docs/prereg-p01-nighttest-2026-10-11.md`. 1109 night frame pairs never seen by any fusion system. Ship AP against the VIS labels, mean over the five Phase 3 systems; Δ = arm − `shipped`, paired; 95% interval from 1,000 moving-block replicates (L = 20, seed 1). Claim = fused − max(VIS, IR) of the arm's own streams.

**P-N1 (`union` beats `shipped` at night): REPLICATES.**  
**P-N2 (`union` keeps the claim at night): PASS.**

| arm | ship AP | Δ vs shipped [95%] | VIS | IR | claim [95%] | per system |
|---|---:|---:|---:|---:|---:|---|
| shipped | 0.0233 | — | 0.0516 | 0.0112 | -0.0283 [-0.0393, -0.0170] | 0.0174, 0.0226, 0.0243, 0.0211, 0.0311 |
| union | 0.0506 | +0.0273 [+0.0154, +0.0400] | 0.0516 | 0.0112 | -0.0010 [-0.0049, +0.0031] | 0.0540, 0.0463, 0.0564, 0.0439, 0.0523 |
| lex | 0.0547 | +0.0314 [+0.0198, +0.0424] | 0.0516 | 0.0112 | +0.0031 [+0.0020, +0.0043] | 0.0598, 0.0475, 0.0606, 0.0479, 0.0574 |
| ens5_union | 0.0490 | +0.0257 [+0.0128, +0.0364] | 0.0473 | 0.0117 | +0.0017 [-0.0003, +0.0047] | 0.0490 |
| full_tta | 0.0520 | +0.0287 [+0.0156, +0.0393] | 0.0506 | 0.0113 | +0.0013 [-0.0022, +0.0040] | 0.0520 |
| full_tta_lex | 0.0535 | +0.0302 [+0.0172, +0.0409] | 0.0506 | 0.0113 | +0.0028 [+0.0017, +0.0042] | 0.0535 |

IR reads night on 27.8% of frames (mean over systems); `shipped` vetoes VIS on 90.3%.

## Amendment 1: `full_tta` and `full_tta_lex` (descriptive, no verdict)

Δ against `union`, paired, on the same replicates.

| arm | ship AP | Δ vs `union` [95%] | claim [95%] |
|---|---:|---:|---:|
| full_tta | 0.0520 | +0.0014 [-0.0065, +0.0068] | +0.0013 [-0.0022, +0.0040] |
| full_tta_lex | 0.0535 | +0.0029 [-0.0053, +0.0082] | +0.0028 [+0.0017, +0.0042] |

Per test stretch (ship AP; mean over systems where the arm is per system):

| stretch | shipped | union | ens5_union | full_tta | full_tta_lex | VIS | VIS, full | IR | IR, full |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| test A (11,059–11,575) | 0.0136 | 0.0406 | 0.0327 | 0.0355 | 0.0374 | 0.0391 | 0.0356 | 0.0136 | 0.0126 |
| test B (26,161–26,752) | 0.0506 | 0.1095 | 0.1411 | 0.1509 | 0.1445 | 0.0956 | 0.1372 | 0.0140 | 0.0206 |
