# Capability prior: leave-one-run-out refit

Preset `crossmodal26m`, `runs/cache_m`, conditions clean, fog, lowlight, glare (VIS stream; IR clean). The shipped prior is fitted with `capability_sel="fit"` = the 1,200 clean day paired frames, the same frames every day cell below is scored on. Each LORO fold refits it on the other two day runs and keeps everything else, including the ÷4 `cap_ir_scale`, fixed. Deltas are **LORO − in-sample**, paired, with a moving-block bootstrap within run (L = 20, 1000 resamples, seed 0). Night (pohang01) is not re-scored: its prior was already fitted without it.

## 1. Fitted priors

`IR` is after the ÷4. `IR wt vs shipped` is the fold's IR/VIS weight ratio divided by the shipped one, i.e. the multiplier on IR's relative weight -- the axis `runs/eval/reprice_constants.md` priced at ×2 and ×0.5. Per-class columns are the raw clean AP50-95 behind each prior (before the ÷4); IR's buoy AP is 0 because the IR detector is single-class, which is why IR's macro prior is half its ship AP.

| prior | fit runs | frames | VIS | IR | VIS/IR | IR wt vs shipped | w_vis | VIS ship | VIS buoy | IR ship | IR buoy |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| in-sample (shipped) | pohang00+02+03 | 1200 | 0.3233 | 0.002396 | 134.9x | 1.00x | 0.992643 | 0.3686 | 0.2780 | 0.0192 | 0.0000 |
| LORO, held out pohang00 | pohang02+03 | 364 | 0.3151 | 0.000861 | 365.8x | 0.37x | 0.997274 | 0.3518 | 0.2783 | 0.0069 | 0.0000 |
| LORO, held out pohang02 | pohang00+03 | 953 | 0.2095 | 0.002661 | 78.7x | 1.71x | 0.987454 | 0.3613 | 0.0576 | 0.0213 | 0.0000 |
| LORO, held out pohang03 | pohang00+02 | 1083 | 0.3526 | 0.002659 | 132.6x | 1.02x | 0.992517 | 0.3868 | 0.3184 | 0.0213 | 0.0000 |

Ground truth on the paired day frames, per run. A run with no buoy boxes scores macro == ship (the missing class is dropped, not scored 0), and a fold that trains on few buoys gets a noisy VIS buoy term in its prior:

| run | frames | ship boxes | buoy boxes |
|---|---:|---:|---:|
| pohang00 | 836 | 7030 | 0 |
| pohang02 | 247 | 1619 | 532 |
| pohang03 | 117 | 2014 | 68 |

Verification: the fold that holds out pohang00 trains on exactly `TEST_RUNS`, so it was rebuilt with a genuine `load_context(capability_sel="test")`: prior equal to 0.0e+00, fused AP equal to 0.0e+00 on all four conditions. The `dataclasses.replace` path is the real system. The swap is not a no-op: it moves the all-frame clean AP by 3.0e-04.

## How to read the floor column

`|delta| vs floor`: `below` = under 0.0014 (the paper's convention reports this as *not resolved*), `within` = inside the 0.0014–0.0031 band, `ABOVE` = over 0.0031. Ship deltas are judged against the macro floor. The ship-AP floor, measured separately (`docs/eval/delta_noise_floor_ship_2026-10-08.md`), is 0.0008–0.0024 on the informative cells; pass `--ship-floor` to judge against it. The block CI is this arm's own scene-resampling interval on one corruption draw (`runs/cache_m`). It omits draw variance, which the floor includes (draw sd 0.0008 on fog/clean), so a CI that excludes 0 says the sign is stable over scenes in these recordings, not that the effect is larger than the floor.

## 2. Pooled: the LORO system against the shipped one

Each day frame scored under the prior that never saw its run, stitched into one prediction set, then pooled. This is the number to put next to a published day cell.

### Macro (ship + buoy)

| scored on | cell | in-sample | LORO | delta | 95% block CI | se | CI spans 0 | |delta| vs floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all day (1,200) | clean/day | 0.3286 | 0.3290 | +0.0004 | [+0.0001, +0.0006] | 0.0001 | **no** | below |
| all day (1,200) | fog/day | 0.0482 | 0.0482 | +0.0001 | [+0.0000, +0.0001] | 0.0000 | **no** | below |
| all day (1,200) | lowlight/day | 0.0244 | 0.0247 | +0.0003 | [-0.0002, +0.0009] | 0.0003 | yes | below |
| all day (1,200) | glare/day | 0.2734 | 0.2737 | +0.0003 | [+0.0000, +0.0005] | 0.0001 | **no** | below |
| TEST pohang02+03 | clean/day | 0.3173 | 0.3168 | -0.0005 | [-0.0007, -0.0003] | 0.0001 | **no** | below |
| TEST pohang02+03 | fog/day | 0.0229 | 0.0229 | -0.0000 | [-0.0001, +0.0001] | 0.0000 | yes | below |
| TEST pohang02+03 | lowlight/day | 0.0032 | 0.0029 | -0.0003 | [-0.0009, -0.0001] | 0.0002 | **no** | below |
| TEST pohang02+03 | glare/day | 0.2602 | 0.2596 | -0.0006 | [-0.0008, -0.0004] | 0.0001 | **no** | below |

### Ship (class 0)

| scored on | cell | in-sample | LORO | delta | 95% block CI | se | CI spans 0 | |delta| vs floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| all day (1,200) | clean/day | 0.3792 | 0.3801 | +0.0009 | [+0.0003, +0.0013] | 0.0003 | **no** | below |
| all day (1,200) | fog/day | 0.0908 | 0.0909 | +0.0001 | [+0.0001, +0.0002] | 0.0000 | **no** | below |
| all day (1,200) | lowlight/day | 0.0487 | 0.0494 | +0.0007 | [-0.0003, +0.0017] | 0.0005 | yes | below |
| all day (1,200) | glare/day | 0.3144 | 0.3152 | +0.0008 | [+0.0002, +0.0011] | 0.0002 | **no** | below |
| TEST pohang02+03 | clean/day | 0.3565 | 0.3556 | -0.0009 | [-0.0014, -0.0006] | 0.0002 | **no** | below |
| TEST pohang02+03 | fog/day | 0.0381 | 0.0381 | -0.0000 | [-0.0001, +0.0001] | 0.0001 | yes | below |
| TEST pohang02+03 | lowlight/day | 0.0064 | 0.0058 | -0.0006 | [-0.0017, -0.0001] | 0.0004 | **no** | below |
| TEST pohang02+03 | glare/day | 0.2867 | 0.2856 | -0.0011 | [-0.0015, -0.0007] | 0.0002 | **no** | below |

## 3. Per fold: held-out run only

Smaller samples (836 / 247 / 117 frames), so wider intervals; a held-out run is where an in-sample fit would show if it were doing work.

### Macro (ship + buoy)

| scored on | cell | in-sample | LORO | delta | 95% block CI | se | CI spans 0 | |delta| vs floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pohang00 (836) | clean/day | 0.3894 | 0.3902 | +0.0008 | [+0.0004, +0.0013] | 0.0002 | **no** | below |
| pohang00 (836) | fog/day | 0.1167 | 0.1168 | +0.0001 | [+0.0001, +0.0002] | 0.0000 | **no** | below |
| pohang00 (836) | lowlight/day | 0.0679 | 0.0691 | +0.0012 | [+0.0002, +0.0016] | 0.0004 | **no** | below |
| pohang00 (836) | glare/day | 0.3280 | 0.3290 | +0.0010 | [+0.0006, +0.0015] | 0.0002 | **no** | below |
| pohang02 (247) | clean/day | 0.3639 | 0.3629 | -0.0009 | [-0.0014, -0.0005] | 0.0002 | **no** | below |
| pohang02 (247) | fog/day | 0.0252 | 0.0252 | +0.0000 | [-0.0001, +0.0001] | 0.0001 | yes | below |
| pohang02 (247) | lowlight/day | 0.0021 | 0.0021 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | yes | below |
| pohang02 (247) | glare/day | 0.3012 | 0.3003 | -0.0009 | [-0.0012, -0.0006] | 0.0002 | **no** | below |
| pohang03 (117) | clean/day | 0.1878 | 0.1878 | -0.0000 | [-0.0000, -0.0000] | 0.0000 | **no** | below |
| pohang03 (117) | fog/day | 0.0183 | 0.0183 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | **no** | below |
| pohang03 (117) | lowlight/day | 0.0042 | 0.0042 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | yes | below |
| pohang03 (117) | glare/day | 0.1526 | 0.1526 | -0.0000 | [-0.0000, -0.0000] | 0.0000 | **no** | below |

### Ship (class 0)

| scored on | cell | in-sample | LORO | delta | 95% block CI | se | CI spans 0 | |delta| vs floor |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| pohang00 (836) | clean/day | 0.3894 | 0.3902 | +0.0008 | [+0.0004, +0.0013] | 0.0002 | **no** | below |
| pohang00 (836) | fog/day | 0.1167 | 0.1168 | +0.0001 | [+0.0001, +0.0002] | 0.0000 | **no** | below |
| pohang00 (836) | lowlight/day | 0.0679 | 0.0691 | +0.0012 | [+0.0002, +0.0016] | 0.0004 | **no** | below |
| pohang00 (836) | glare/day | 0.3280 | 0.3290 | +0.0010 | [+0.0006, +0.0015] | 0.0002 | **no** | below |
| pohang02 (247) | clean/day | 0.4101 | 0.4083 | -0.0018 | [-0.0028, -0.0010] | 0.0004 | **no** | within |
| pohang02 (247) | fog/day | 0.0394 | 0.0395 | +0.0001 | [-0.0002, +0.0003] | 0.0001 | yes | below |
| pohang02 (247) | lowlight/day | 0.0043 | 0.0043 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | yes | below |
| pohang02 (247) | glare/day | 0.3392 | 0.3373 | -0.0018 | [-0.0025, -0.0013] | 0.0003 | **no** | within |
| pohang03 (117) | clean/day | 0.3201 | 0.3200 | -0.0000 | [-0.0000, -0.0000] | 0.0000 | **no** | below |
| pohang03 (117) | fog/day | 0.0365 | 0.0365 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | **no** | below |
| pohang03 (117) | lowlight/day | 0.0085 | 0.0085 | +0.0000 | [+0.0000, +0.0000] | 0.0000 | yes | below |
| pohang03 (117) | glare/day | 0.2475 | 0.2475 | -0.0000 | [-0.0000, -0.0000] | 0.0000 | **no** | below |

## 4. Observed w_vis on the scored frames

Mean `w_vis_gated` from `run_systems`, held-out run only. Constant within a fold, because the weights are the prior alone.

| held-out run | condition | in-sample | LORO |
|---|---:|---:|---:|
| pohang00 | clean | 0.992643 | 0.997274 |
| pohang00 | fog | 0.992643 | 0.997274 |
| pohang00 | lowlight | 0.992643 | 0.997274 |
| pohang00 | glare | 0.992643 | 0.997274 |
| pohang02 | clean | 0.992643 | 0.987454 |
| pohang02 | fog | 0.992643 | 0.987454 |
| pohang02 | lowlight | 0.992643 | 0.987454 |
| pohang02 | glare | 0.992643 | 0.987454 |
| pohang03 | clean | 0.992643 | 0.992517 |
| pohang03 | fog | 0.992643 | 0.992517 |
| pohang03 | lowlight | 0.992643 | 0.992517 |
| pohang03 | glare | 0.992643 | 0.992517 |

## 5. Reading

* Largest pooled day |delta|: macro 0.0004, ship 0.0009. Largest single-fold |delta|: macro 0.0012, ship 0.0018. Macro floor (PAPER_DRAFT2 §5.3, paired 2σ on informative cells): 0.0014–0.0031.
* Pooled cells whose block CI excludes 0: macro day clean, test clean, day fog, test lowlight, day glare, test glare; ship day clean, test clean, day fog, test lowlight, day glare, test glare.
* Fold cells whose block CI excludes 0: macro pohang00 clean, pohang02 clean, pohang03 clean, pohang00 fog, pohang03 fog, pohang00 lowlight, pohang00 glare, pohang02 glare, pohang03 glare; ship pohang00 clean, pohang02 clean, pohang03 clean, pohang00 fog, pohang03 fog, pohang00 lowlight, pohang00 glare, pohang02 glare, pohang03 glare.
* Cells at `within` or `ABOVE`: pohang02 clean ship -0.0018, pohang02 glare ship -0.0018.

## 6. Interpretation (written by hand after the run)

**The in-sample prior does not flatter the day cells.** The pooled LORO system is
slightly *better* on all four day cells (macro +0.0001 to +0.0004, ship +0.0001 to
+0.0009) and slightly worse on TEST (macro up to −0.0006, ship up to −0.0011). Every
pooled delta is below the 0.0014 floor. The one band-level shift is a single run:
ship AP on pohang02 moves −0.0018 on clean and glare, with 247 frames.

**The sign follows the IR weight, not the fit.** The fold that raises IR's relative
weight (×1.71, held-out pohang02) loses, the one that lowers it (×0.37, held-out
pohang00) gains, and the one that leaves it at ×1.02 (pohang03) moves by less than
0.0001. That is the `cap_ir_scale` response already measured in
`runs/eval/reprice_constants.md`: day clean −0.0006 at ×2 and +0.0004 at ×0.5. An
optimistic in-sample fit would show up as a loss under every fold. It does not,
because a clean-mAP prior is not an AP-maximising weight: it measures sensor
quality, not the ranking optimum.

**The priors vary more than the results.** VIS's prior ranges 0.2095–0.3526. Most of
that range is the buoy term: pohang00 has no buoy boxes, so a fold's VIS buoy AP
comes only from the runs it trains on (0.0576 from pohang03's 68 buoys when
pohang02 is held out). IR's prior ranges 0.000861–0.002661 after the ÷4, because
IR's clean ship AP is about 3× lower on pohang02+03 (0.0069) than in any fit set
that includes pohang00 (0.0213). With w_vis at 0.9875–0.9973 either way,
the merge is still VIS-dominated concatenation.

**Not covered.** One corruption draw, so fog/lowlight/glare intervals omit draw
variance. Night is not re-scored, because its prior never saw pohang01. Ship deltas
are judged against the macro floor. The ship-AP floor measured afterwards
(`docs/eval/delta_noise_floor_ship_2026-10-08.md`) is 0.0008–0.0024 on the informative
cells, lower than the macro's on most of them; against it the pooled ship deltas (at
most 0.0011) and the pohang02 single-run shift (0.0018) sit inside its range, not below
it.

---

_Generated by `scripts/eval_capability_loro.py` in 1120s._

## Provenance

Written automatically (R-E1/F14). A result that records a preset *name* cannot say which system produced it -- `preset="crossmodal"` named three different systems on 2026-09-01. These are values.

| field | value |
|---|---|
| `written_utc` | `2026-10-08T14:58:19+00:00` |
| `git_head` | `c8197e75a7ac4f54d1c1fe19fcb661ca4a2ed255` |
| `git_dirty` | `True` |
| `dirty_sha256` | `a5516537acb9e515` |
| `git_branch` | `fusion-uq-phase3` |
| `ap_convention` | `local-linear-interp` |
| `missing_class_policy` | `drop` |
| `sort_kind` | `stable` |
| `preset` | `crossmodal26m` |
| `preset_resolved` | `crossmodal` |
| `role` | `develop` |
| `ir_nms` | `0.7` |
| `cap_ir_scale` | `4.0` |
| `cap_vis` | `0.3232903212523053` |
| `cap_ir` | `0.0023961514901125443` |
| `iou_thr` | `0.85` |
| `veto` | `0.5` |
| `veto_rule` | `gini+ir_night` |
| `veto_filter` | `['dilate', 15]` |
| `veto_health_mode` | `night_arm` |
| `single_passthrough` | `True` |
| `cap_note` | `capability prior over fit frames` |
| `load_context_inputs` | `{'conditions': ['clean', 'fog', 'lowlight', 'glare'], 'cache_dir': 'A:\\Uncertain\\runs\\cache_m', 'manifest': 'runs/derived/paired_val_manifest.csv', 'homography': 'runs/derived/homography_ir_to_vis.json', 'constants': 'runs/eval/reliability_constants.json', 'brightness_constants': 'runs/eval/brightness_constants.json', 'bright_dir': 'runs/derived/brightness', 'iou_thr': 0.85, 'veto': 0.5, 'capability_sel': 'fit', 'role': 'develop', 'bright_soft': False, 'veto_filter': ['dilate', 15], 'veil_filter': ['majority', 15], 'tau_lap': 508.6742858886719, 'preset': 'crossmodal', 'ir_condition': None, 'ir_nms': 0.7, 'vis_soft_nms': None, 'cap_ir_scale': 4.0, 'structure_dir': 'runs/derived/structure', 'structure_constants': 'runs/eval/structure_constants.json', 'ir_bright': None, 'veto_keep_cls': [], 'config': None}` |
| `block_len` | `20` |
| `n_boot` | `1000` |
| `seed` | `0` |
| `cache` | `runs/cache_m` |

**The working tree was dirty when this ran.** `git_head` alone does not identify the source that produced these numbers; `dirty_sha256` is a hash of `git diff HEAD` and is the part that does.
