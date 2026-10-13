# Pre-registration — R-D1 under the shipped preset `crossmodal26m`

Written 2026-09-27, **before the run**. A follow-up to
[`prereg-uq-mechanism-ablation.md`](prereg-uq-mechanism-ablation.md) (R-D1, committed `a8f087c`,
amendment 1 `f713961`). R-D1's §6 says "any follow-up requires a new pre-registration naming what
changed and why". This is that document.

## 1. What changed

**One thing: the preset.** R-D1 ran under `preset="crossmodal"` (§7, §8). The system the paper
describes and the pohang04 look scored is `crossmodal26m`. The two differ in exactly the terms
R-D1's argument leaned on:

| term | `crossmodal` (R-D1) | `crossmodal26m` (shipped) |
|---|---|---|
| VIS veto | `grad_gini < thr OR (ir_p05 > thr AND VIS dark)` | `ir_p05 > thr AND (VIS dark OR grad_gini < thr)` |
| VIS veto on day fog | **100%** | 0% (veil requires night) |
| weak night fallback (`ir_night_raw AND NOT ir_ok ...`) | off | on |
| IR merge veto | on | off |
| cross-modal support term | off | IoU 0.30, γ 0.5 |
| `cap_ir_scale`, IR NMS | 4, 0.70 | 4, 0.70 (unchanged) |

## 2. Why

R-D1's fog result has a structural explanation that is true only of `crossmodal`: with VIS vetoed
on every fog frame there is never a second stream for sigma to arbitrate, so no coordinate-path
effect is possible there. Under `crossmodal26m` the day-fog frames keep both streams, so sigma
has frames to act on that R-D1 never offered it. The paper may not state the null for the shipped
system until it has been measured on it.

## 3. What is held fixed

Everything else in R-D1, transcribed and not re-chosen: arms S0–S7 (§3), caches
`runs/cache_m/gauss_{vis,ir}_paired_*.pkl` unchanged, conditions `clean`, `fog`, `lowlight`,
`glare` (§4), metric `gated_fusion` ship AP, `ALPHA = 1.0`, paired moving-block bootstrap
L = 20, n_boot = 1000, seed 0, and the decision rule of §5 **as amended**: adopted floor
**0.0060**, on at least 3 of 4 conditions, CI excluding zero, applied separately to the
coordinate path (S1 vs S3) and the score path (S5 vs S7). Counts reported at 0.0014 / 0.0031 /
0.0060 / 0.0100.

Outcomes are R-D1's three: POSITIVE, NULL, NEGATIVE, with R-D1's consequences.

## 4. Scope

* **Development data**, like R-D1. Logged in `exposure-ledger-2026-09-09.md` §7 before the run.
* **The old checkpoints** (`runs/cache_m`, the full-scale `yolo26m` detectors R-D1 used), so the
  preset is the only thing that moves between R-D1 and this run. It is **not** a Phase 3 result;
  the Phase 3 corrupted caches do not exist yet, and a Phase 3 version would need its own document.
* **No fifth run.** If this returns NULL, the claim is "NULL under both presets" and the axis
  stays closed.

## 5. Provenance

`python scripts/ablate_uq_mechanism.py --preset crossmodal26m --cache-dir runs/cache_m
--out docs/eval/uq_mechanism_ablation_26m_2026-09-27.md`. The script is unchanged since R-D1;
`--preset` is an existing argument.
