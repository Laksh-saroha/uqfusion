# The finalized system (2026-08-20), measured

Configuration: capability prior over fit runs (VIS 0.3352 / IR 0.0092), photometric term veto-only, hard veto at r_bright<0.5 with ('dilate', 15) hysteresis, WBF iou_thr 0.85. Paired frame-level bootstrap, n=1000, seed 0. Caches: the record's yolo26s checkpoints (the full-scale retrain replaces them; this table freezes the architecture, not the numbers).

## 1. Eight cells, gated vs `ir_only` — SHIP AP (the headline)

Ship is the only class both streams can produce: IR is nc=1 ship-only (D28/A-1 — IR buoy AP measured 0.00019). Read this table, not the macro one below. The macro delta is `(delta_ship + delta_buoy) / 2`, mixing the class fusion acts on with one only VIS can supply, so it answers no single question: it dilutes a large ship gain (where delta_buoy < delta_ship), inflates a small one (where VIS buoy AP is high and the ship gain is not), and in a vetoed cell carries a buoy zero the system was never able to avoid. None of those apply here.

| cell | VIS ship | IR ship | gated ship | delta vs ir | 95% CI | flips | VIS veto |
|---|---:|---:|---:|---:|---|---:|---:|
| clean/day | 0.3683 | 0.0177 | 0.3715 | +0.3537 | [+0.3445, +0.3630] | 0.0% | 0% |
| clean/night | 0.0000 | 0.0810 | 0.0813 | +0.0003 | [-0.0002, +0.0008] (spans 0) | 11.4% | 100% |
| fog/day | 0.0020 | 0.0177 | 0.0126 | -0.0051 | [-0.0062, -0.0030] | 0.0% | 0% |
| fog/night | 0.0000 | 0.0810 | 0.0809 | -0.0001 | [-0.0006, +0.0004] (spans 0) | 36.2% | 71% |
| lowlight/day | 0.0346 | 0.0177 | 0.0166 | -0.0011 | [-0.0018, -0.0004] | 0.1% | 100% |
| lowlight/night | 0.0000 | 0.0810 | 0.0813 | +0.0003 | [-0.0002, +0.0008] (spans 0) | 11.4% | 100% |
| glare/day | 0.2892 | 0.0177 | 0.2928 | +0.2751 | [+0.2643, +0.2867] | 0.0% | 0% |
| glare/night | 0.0000 | 0.0810 | 0.0813 | +0.0003 | [-0.0002, +0.0008] (spans 0) | 11.4% | 100% |

### 1b. The same cells, macro mAP over both classes (continuity with the record)

`buoy gated` is VIS-only by construction; where it drops to ~0 the veto has removed VIS from the merge, and the macro column below is carrying that zero. This table exists so earlier macro-quoted numbers stay comparable — it is not the result.

| cell | visible_only | ir_only | gated | delta vs ir | 95% CI | buoy gated | buoy VIS |
|---|---:|---:|---:|---:|---|---:|---:|
| clean/day | 0.3352 | 0.0092 | 0.3357 | +0.3265 | [+0.3182, +0.3377] | 0.3000 | 0.3020 |
| clean/night | 0.0000 | 0.0810 | 0.0813 | +0.0003 | [-0.0002, +0.0008] (spans 0) | 0.0000 | 0.0000 |
| fog/day | 0.0012 | 0.0092 | 0.0067 | -0.0026 | [-0.0031, -0.0015] | 0.0008 | 0.0004 |
| fog/night | 0.0000 | 0.0810 | 0.0809 | -0.0001 | [-0.0006, +0.0004] (spans 0) | 0.0000 | 0.0000 |
| lowlight/day | 0.0173 | 0.0092 | 0.0087 | -0.0006 | [-0.0009, -0.0002] | 0.0008 | 0.0000 |
| lowlight/night | 0.0000 | 0.0810 | 0.0813 | +0.0003 | [-0.0002, +0.0008] (spans 0) | 0.0000 | 0.0000 |
| glare/day | 0.2626 | 0.0092 | 0.2649 | +0.2556 | [+0.2453, +0.2678] | 0.2369 | 0.2360 |
| glare/night | 0.0000 | 0.0810 | 0.0813 | +0.0003 | [-0.0002, +0.0008] (spans 0) | 0.0000 | 0.0000 |

## 2. Gated vs `visible_only`, day frames (the record's open item 1)

Ship AP is the headline column here too; macro follows for continuity.

| condition | VIS ship | gated ship | delta | 95% CI | flips |
|---|---:|---:|---:|---|---:|
| clean/day | 0.3683 | 0.3715 | +0.0031 | [+0.0014, +0.0054] | 0.0% |
| fog/day | 0.0020 | 0.0126 | +0.0106 | [+0.0080, +0.0141] | 0.0% |
| lowlight/day | 0.0346 | 0.0166 | -0.0180 | [-0.0216, -0.0153] | 0.0% |
| glare/day | 0.2892 | 0.2928 | +0.0036 | [+0.0015, +0.0052] | 0.0% |

| condition | visible_only | gated | delta (macro) | 95% CI | flips |
|---|---:|---:|---:|---|---:|
| clean/day | 0.3352 | 0.3357 | +0.0006 | [-0.0013, +0.0030] (spans 0) | 18.6% |
| fog/day | 0.0012 | 0.0067 | +0.0055 | [+0.0042, +0.0073] | 0.0% |
| lowlight/day | 0.0173 | 0.0087 | -0.0086 | [-0.0104, -0.0073] | 0.0% |
| glare/day | 0.2626 | 0.2649 | +0.0022 | [+0.0004, +0.0042] | 1.5% |

## 3. Soft-weight ablation (veto kept; deltas vs adopted)

`no_maha`: r_frame forced to 1. `cap_only`: additionally r_box forced to 1, so weights are the pure capability prior. `no_veto`: the soft system alone.

| variant | cell | ship AP | ship delta | ship 95% CI | mAP | delta (macro) | 95% CI |
|---|---|---:|---:|---|---:|---:|---|
| no_maha | clean/day | 0.3718 | +0.0003 | [+0.0002, +0.0004] | 0.3359 | +0.0002 | [+0.0001, +0.0003] |
| no_maha | clean/night | 0.0813 | +0.0000 | identical | 0.0813 | +0.0000 | identical |
| no_maha | fog/day | 0.0119 | -0.0007 | [-0.0028, +0.0017] (spans 0) | 0.0065 | -0.0002 | [-0.0012, +0.0011] (spans 0) |
| no_maha | fog/night | 0.0781 | -0.0028 | [-0.0036, -0.0021] | 0.0781 | -0.0028 | [-0.0036, -0.0021] |
| no_maha | lowlight/day | 0.0166 | +0.0000 | identical | 0.0087 | +0.0000 | identical |
| no_maha | lowlight/night | 0.0813 | +0.0000 | identical | 0.0813 | +0.0000 | identical |
| no_maha | glare/day | 0.2961 | +0.0033 | [+0.0024, +0.0041] | 0.2672 | +0.0023 | [+0.0014, +0.0029] |
| no_maha | glare/night | 0.0813 | +0.0000 | identical | 0.0813 | +0.0000 | identical |
| cap_only | clean/day | 0.3717 | +0.0003 | [+0.0002, +0.0003] | 0.3359 | +0.0002 | [+0.0000, +0.0003] |
| cap_only | clean/night | 0.0813 | +0.0000 | identical | 0.0813 | +0.0000 | identical |
| cap_only | fog/day | 0.0131 | +0.0005 | [-0.0014, +0.0028] (spans 0) | 0.0072 | +0.0005 | [-0.0006, +0.0017] (spans 0) |
| cap_only | fog/night | 0.0781 | -0.0028 | [-0.0036, -0.0021] | 0.0781 | -0.0028 | [-0.0036, -0.0021] |
| cap_only | lowlight/day | 0.0166 | +0.0000 | identical | 0.0087 | +0.0000 | identical |
| cap_only | lowlight/night | 0.0813 | +0.0000 | identical | 0.0813 | +0.0000 | identical |
| cap_only | glare/day | 0.2960 | +0.0032 | [+0.0024, +0.0041] | 0.2671 | +0.0023 | [+0.0014, +0.0029] |
| cap_only | glare/night | 0.0813 | +0.0000 | identical | 0.0813 | +0.0000 | identical |
| no_veto | clean/day | 0.3715 | +0.0000 | identical | 0.3357 | +0.0000 | identical |
| no_veto | clean/night | 0.0763 | -0.0051 | [-0.0060, -0.0036] | 0.0763 | -0.0051 | [-0.0060, -0.0036] |
| no_veto | fog/day | 0.0126 | +0.0000 | identical | 0.0067 | +0.0000 | identical |
| no_veto | fog/night | 0.0772 | -0.0037 | [-0.0046, -0.0022] | 0.0772 | -0.0037 | [-0.0046, -0.0022] |
| no_veto | lowlight/day | 0.0153 | -0.0013 | [-0.0023, +0.0013] (spans 0) | 0.0080 | -0.0007 | [-0.0012, +0.0006] (spans 0) |
| no_veto | lowlight/night | 0.0775 | -0.0038 | [-0.0047, -0.0024] | 0.0775 | -0.0038 | [-0.0047, -0.0024] |
| no_veto | glare/day | 0.2928 | +0.0000 | identical | 0.2649 | +0.0000 | identical |
| no_veto | glare/night | 0.0640 | -0.0173 | [-0.0186, -0.0149] | 0.0640 | -0.0173 | [-0.0186, -0.0149] |
