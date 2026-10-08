# The paired-delta noise floor on ship AP — and whether any ship verdict depended on it

**Measured 2026-10-08.** `PAPER_DRAFT2.md` §5.3 measured its noise floor on the ship+buoy macro, but the Phase 3 cells, the Stage 1 crossing and the pohang04 look decide on ship AP (class 0). §5.3 disclosed that the macro floor "understates the noise in a ship-AP delta by up to a factor of two". This file measures the ship floor directly and re-reads every ship-AP verdict that used a floor. **No pohang04 frame was scored**; its recorded numbers are only re-read.

Run: `py -3.13 scripts/probe_delta_noise_floor_ship.py` → `runs/eval/delta_noise_floor_ship.md` (+ `.json`), 4,004 s, at `c8197e7`. The tree was dirty only from other sessions' edits to `PAPER_DRAFT2.md` and `PAPER_WRITING_INSTRUCTIONS.md`, which no evaluation code reads. Everything else is the same as `runs/eval/delta_noise_floor.md`: preset `crossmodal26m`, caches `runs/cache_m` + `cache_m_draw901..903`, the same 11 cells, the same probe arm (VIS soft-NMS σ 0.5), the 1,200 day frames, n_boot 2000, and bootstrap seeds 0 / 1 / 2. Ship AP is `ap_weighted(...)["per_class"][0]` (the unweighted call equals `ap_from_parts(...)["per_class"][0]`, as in `p3_night_check.py`). The paired bootstrap is `bootstrap_delta(..., cls=0)`. The macro is recomputed in the same run on the same resample sequence.

## Verdict

**No recorded ship-AP verdict flips.** The ship floor is lower than the macro floor, not higher. Its maximum over cells is **0.0024** against the macro's 0.0031, so the Phase 3 construction (max × 1.95) gives **0.0047** against 0.0060. Every recorded verdict is the same at 0.0047, 0.0060 and the macro range 0.0014–0.0031, and each sits a long way from its flip point (§3). The only descriptive change is fog/day (+0.0059). It is below 0.0060 but above 0.0047, so the paper's "resolved in sign but not beyond the floor" would read "beyond the measured ship floor" if the ship floor were used. The pre-registered 0.0060 stays the floor of record, and for ship deltas it is now known to be conservative. §5.3's "understates by up to 2×" sentence is wrong for paired deltas and should be replaced (§4).

## 1. Ship floor beside the macro floor

`2×total` = 2·hypot(sd draw, sd paired bootstrap), the construction of `runs/eval/delta_noise_floor.md` §2. sd paired = (CI width)/(2·1.96) of `bootstrap_delta`, as before.

| cell (vis/ir) | Δ macro | Δ ship | sd draw macro / ship | sd paired macro / ship | **2×total macro** | **2×total ship** | ship / macro |
|---|---:|---:|---|---|---:|---:|---:|
| clean/clean | +0.0012 | +0.0018 | 0.0000 / 0.0000 | 0.0008 / 0.0004 | 0.0015 | **0.0009** | 0.58× |
| clean/glare_s2 | +0.0012 | +0.0017 | 0.0001 / 0.0002 | 0.0007 / 0.0004 | 0.0015 | **0.0009** | 0.60× |
| clean/blur_s2 | +0.0012 | +0.0017 | 0.0000 / 0.0001 | 0.0007 / 0.0004 | 0.0015 | **0.0009** | 0.60× |
| clean/noise_s2 | +0.0026 | +0.0046 | 0.0001 / 0.0002 | 0.0008 / 0.0008 | 0.0017 | **0.0016** | 0.97× |
| clean/fog_s2 | +0.0024 | +0.0041 | 0.0002 / 0.0003 | 0.0008 / 0.0007 | 0.0017 | **0.0016** | 0.95× |
| blur_s3/clean | +0.0002 | +0.0022 | 0.0011 / 0.0006 | 0.0008 / 0.0008 | 0.0027 | **0.0020** | 0.73× |
| noise_s2/clean | +0.0000 | +0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0000 | 0.0000 | **0.0000** | — |
| rain_s2/clean | +0.0015 | +0.0019 | 0.0003 / 0.0002 | 0.0003 / 0.0003 | 0.0008 | **0.0008** | 1.00× |
| fog/clean | +0.0008 | +0.0014 | 0.0008 / 0.0003 | 0.0004 / 0.0004 | 0.0017 | **0.0010** | 0.57× |
| lowlight/glare_s2 | +0.0000 | +0.0000 | 0.0000 / 0.0000 | 0.0000 / 0.0001 | 0.0001 | **0.0002** | 2.00× |
| blur_s3/glare_s2 | −0.0004 | +0.0010 | 0.0013 / 0.0007 | 0.0008 / 0.0010 | **0.0031** | **0.0024** | 0.79× |

| | macro | ship |
|---|---:|---:|
| range, nine informative cells | 0.0008–0.0031 | **0.0008–0.0024** |
| full range | 0.0000–0.0031 | **0.0000–0.0024** |
| max (the cell that sets the Phase 3 floor) | 0.0031 (blur_s3/glare_s2) | **0.0024** (blur_s3/glare_s2) |
| max × 1.95 (block-bootstrap inflation, §5.4) | 0.0060 | **0.0047** |
| pairing gain, sd unpaired / sd paired, informative cells | 3–16× | 3–21× |

**Reproduction check.** The macro column reproduces `runs/eval/delta_noise_floor.md` (2026-09-02, `573ad5d`). Every delta, draw sd, paired sd and `2×total` is within 0.00011 of the recorded value, and the maximum is unchanged at 0.0031. Two things account for the residue. First, the old script computed `total` from its own 4-dp-rounded paired sd (`psd = float(rows[i][2])`), which this run does not do. Second, `apmetrics` changed after the recorded run: the missing-class policy (`a4cf208`, 09-09) and the declared AP convention (`99a6bc9`, 09-10).

## 2. Why the 2× level ratio did not carry over

`metric_noise_floor.md` §2 compared the **level** sd of each metric under the frame bootstrap. A gate reads a **paired delta**, and the macro delta is (Δship + Δbuoy)/2.

* **Where buoy AP is constant** (noise_s2/clean, lowlight/glare_s2, both buoy AP 0.0000), Δbuoy = 0. The macro delta and its noise are then both exactly half the ship delta's. The ratio is 2.00× on lowlight/glare_s2, as §5.3 predicted. But the absolute floor there is 0.0002, two orders below any floor used, so it cannot bind.
* **Where buoy AP moves**, the probe arm moves buoys too, and the 600-box buoy delta is far noisier than the 10,663-box ship delta. The buoy term adds noise to the macro faster than halving removes it. The macro floor therefore *exceeds* the ship floor: 0.57–0.79× on six cells, about 1× on three.

The two effects run in opposite directions. The one that wins is the one on the cells with real floors, so the maximum, which is what the Phase 3 floor was built from, falls.

## 3. Every ship-AP verdict that used a floor

`F` is the floor. Flip point = the value of F at which the recorded verdict would change.

### Phase 3 cells — `p3_night_check_2026-09-27.md`, `p3_corrupt_cells_2026-09-27.md`, `p3_fog_s1_2026-09-27.md`

Rule: **fails** iff seed-mean delta ≤ −F and the between-seed 95% t-interval (df 4) lies entirely below zero. A lower F can only create fails, and a higher F can only remove them.

| cell | delta (fused − VIS) | seed CI | recorded | at 0.0047 | flip point |
|---|---:|---|---|---|---|
| clean/day | +0.0081 | [+0.0047, +0.0115] | holds | holds | none (positive) |
| clean/night | −0.1847 | [−0.2039, −0.1655] | **FAILS** | **FAILS** | F > 0.1847 |
| fog/day | +0.0059 | [+0.0036, +0.0082] | holds | holds | none (positive) |
| fog/night | +0.0685 | [+0.0618, +0.0751] | holds | holds | none |
| lowlight/day | +0.0076 | [+0.0039, +0.0112] | holds | holds | none |
| lowlight/night | +0.0682 | [+0.0610, +0.0755] | holds | holds | none |
| glare/day | +0.0107 | [+0.0079, +0.0135] | holds | holds | none |
| glare/night | −0.0872 | [−0.1047, −0.0697] | **FAILS** | **FAILS** | F > 0.0872 |
| fog_s1/day | +0.0061 | [+0.0041, +0.0081] | holds | holds | none |
| fog_s1/night | +0.0684 | [+0.0612, +0.0755] | holds | holds | none |

Fused − IR is ≥ −0.0000 in every cell, so no fail is possible on that side. No delta lies in (−0.0060, −0.0047], the band a lower floor could newly fail.

*Descriptive only:* the "beyond the floor" reading of the day gains (`PAPER_DRAFT2.md` §6.3) is 3/4 at 0.0060 and **4/4** at 0.0047. fog/day's +0.0059 clears 0.0047.

### Stage 1 — `stage1-s1-null-2026-09-14.md`, `runs/eval/stage1_crossing_2026-09-14.md` (`runs/cache_m`, the generation this floor was measured on)

Rule: D non-inferior iff the upper 95% block-bootstrap CI of AP(A) − AP(D) < F, on ≥ 3 of 4 TUNE conditions. Here a **higher** F makes passing easier.

| condition (TUNE, decides) | A − D | upper CI | at 0.0060 | at 0.0047 |
|---|---:|---:|---|---|
| clean | +0.0151 | +0.0210 | no | no |
| fog | +0.0160 | +0.0221 | no | no |
| lowlight | +0.0014 | +0.0040 | yes | yes |
| glare | +0.0115 | +0.0157 | no | no |
| **count** | | | **1/4 → S1-NULL** | **1/4 → S1-NULL** |

Flip point: S1-NULL becomes a pass only at F > 0.0210, 4.5× the ship floor. TEST (report-only) stays at 2/4 at both floors (glare/test upper +0.0062, clean/test +0.0097).

### pohang04 look — `holdout_p04_look.md` (re-read, not re-scored)

Rule: HOLDOUT-GAP iff D = AP_ref − AP_p04 ≥ F **and** the 95% interval lies entirely above zero. D = +0.0216, interval [−0.0120, +0.0502]. The interval spans zero, so the verdict is **NO-GAP at every F**. The point estimate clears any F ≤ 0.0216, including 0.0047. **Cannot flip** on the floor.

### Out of scope, checked because it uses 0.0060

R-D1 (§6.4, `uq_mechanism_ablation_26m_2026-09-27.md`) decides on the **macro**: `ablate_uq_mechanism.py` reads `gated_fusion["map50_95"]`. Table 4's caption says so. The macro floor is therefore the correct one for it, and this file does not bear on it. Its pre-registration (`prereg-uq-mechanism-ablation.md` §4) names "`gated_fusion` ship AP". That mismatch predates this file and is not resolved here.

## 4. Caveats that apply to both floors equally

* **One probe arm.** The floor is the spread of *this* arm's delta (VIS soft-NMS 0.5). An arm that changes more boxes, such as a veto, can have a wider paired sd. That limitation was already present in the macro floor and is unchanged here.
* **Day only.** At night the veto drops VIS, so a VIS-only probe has a delta of exactly zero. No night floor exists for either metric. The two night fails are 18× and 39× the ship floor × 1.95.
* **Frame-iid bootstrap.** The floor is iid. The ×1.95 block inflation is a lower bound (§5.4).
* **Checkpoint generation.** The floor is measured on `runs/cache_m`, the pre-restore checkpoints, as the macro floor was. It is used as a proxy for the Phase 3 retrain cells.

## 5. Proposed wording for `PAPER_DRAFT2.md` §5.3 (not applied)

Replace the sentences from "These floors were measured on the macro over ship and buoy." through "…by up to a factor of two (`runs/eval/metric_noise_floor.md` §2)." with:

> These floors were measured on the macro over ship and buoy, while the Phase 3, Stage 1 and held-out verdicts are on ship AP. We therefore re-measured the floor on ship AP, with the same arm, cells, draws and resamples (`docs/eval/delta_noise_floor_ship_2026-10-08.md`). The ship floor is 0.0008–0.0024 on the nine informative cells (full range 0.0000–0.0024). Its maximum, which is what sets the magnitude floor, is 0.0024 against the macro's 0.0031, and 0.0024 × 1.95 = 0.0047. The level standard deviation of ship AP is up to twice the macro's (`runs/eval/metric_noise_floor.md` §2), but that does not carry over to a paired delta. Where buoy AP is constant, the macro delta and its noise are both exactly half the ship delta's. Where buoy AP moves, the noisy buoy delta (600 day boxes) raises the macro floor above the ship floor. The 0.0060 floor is therefore conservative for ship-AP deltas, and no recorded verdict changes at 0.0047 (§6.3, §6.5, §7).

Optional, same source:

* §6.3, after "On fog/day it is just below the 0.0060 magnitude floor, so it is resolved in sign but not beyond the floor there.", add: "It clears the measured ship-AP floor, 0.0047 (§5.3)."
* Abstract, "a paired noise floor of 0.0014–0.0031 AP" → "a paired noise floor of 0.0014–0.0031 AP (0.0008–0.0024 on ship AP)".
