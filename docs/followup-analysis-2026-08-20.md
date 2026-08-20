# Follow-up round — 2026-08-20

Seven tests queued against the open questions in
`docs/fusion-gate-experiment-record.md` §9, plus the stage-3/stage-4 training arms.
Everything below is measured, not projected. All fusion numbers use the paired
frame-level bootstrap (the same resample scores both systems, so frame-composition
noise cancels in the difference), 2232 paired frames, 1200 day / 1032 night.

**Bottom line: the fusion contribution as framed in the record does not survive its
own error bars, and a 25-epoch detector improvement inverts it. The one durable
positive is that the photometric soft term is redundant — the hard veto does all
the work.**

---

## 1. The night claim was never significant

`runs/eval/x_fusion_ci.md` §1 — adopted system (iou_thr 0.85, veto 0.5,
capability VIS 0.2580 / IR 0.0206), 1000 resamples, seed 0.

| cell | gated | ir_only | delta | 95% CI | sign flips | verdict |
|---|---:|---:|---:|---|---:|---|
| clean/day | 0.3341 | 0.0092 | +0.3249 | [+0.3165, +0.3360] | 0.0% | gated wins |
| **clean/night** | 0.0813 | 0.0810 | **+0.0003** | **[-0.0002, +0.0008]** | **11.4%** | **spans zero** |
| fog/day | 0.0076 | 0.0092 | -0.0016 | [-0.0022, -0.0007] | 0.0% | ir_only wins |
| fog/night | 0.0789 | 0.0810 | -0.0021 | [-0.0029, -0.0009] | 0.0% | ir_only wins |
| lowlight/day | 0.0087 | 0.0092 | -0.0006 | [-0.0009, -0.0002] | 0.1% | ir_only wins |
| lowlight/night | 0.0813 | 0.0810 | +0.0003 | [-0.0002, +0.0008] | 11.4% | spans zero |
| glare/day | 0.2602 | 0.0092 | +0.2509 | [+0.2401, +0.2634] | 0.0% | gated wins |
| glare/night | 0.0813 | 0.0810 | +0.0003 | [-0.0002, +0.0008] | 11.4% | spans zero |

The night result the record leans on is **+0.0003 with an interval straddling
zero and an 11.4% sign-flip rate**. It was never a win. Three of the four night
cells are the same number because VIS is vetoed on all of them and the system
reduces to IR.

Gated fusion also trails single-modality VIS on the two cells it does win
(`x_fusion_ci.md` §4): clean/day `visible_only` 0.3352 vs gated 0.3341;
glare/day 0.2626 vs 0.2602. **These two are point estimates — the
gated-vs-`visible_only` contrast was not bootstrapped and should be.** If it
holds, no cell in the table shows fusion beating the better single stream.

### 1.1 Evidence base

`x_fusion_ci.md` §5. Six of the eight cells are synthetic corruptions:
**75% of the cell count, 55% of the summed score.** Real evidence is two cells
(clean/day, clean/night). §2.4 of the record already established that the
Mahalanobis score responds 21x differently to synthetic darkening than to real
night, so the corruption cells test a transform the gate can see rather than
real-world failure.

---

## 2. A better IR detector inverts the result

`runs/eval/x_ir_upgrade_p2feat.md`. IR caches rebuilt from
`runs/screen3/s3_p2feat_640_b10/weights/best.pt`; VIS caches are hardlinks to the
originals, so the VIS stream is byte-identical and all movement is the IR
detector. Image lists, imgsz and conf were read off the original caches' metadata
rather than assumed.

`ir_only` on night: **0.0810 -> 0.1087 (+0.0277, CI [+0.0255, +0.0296], +34%).**

Gated fusion followed it up to 0.1077 — but from below:

| cell | gated | ir_only | delta | 95% CI | verdict |
|---|---:|---:|---:|---|---|
| clean/day | 0.3330 | 0.0111 | +0.3220 | [+0.3142, +0.3332] | gated wins |
| clean/night | 0.1077 | 0.1087 | -0.0010 | [-0.0016, -0.0004] | **ir_only wins** |
| fog/day | 0.0082 | 0.0111 | -0.0028 | [-0.0031, -0.0022] | ir_only wins |
| fog/night | 0.1062 | 0.1087 | -0.0025 | [-0.0032, -0.0015] | ir_only wins |
| lowlight/day | 0.0094 | 0.0111 | -0.0017 | [-0.0021, -0.0013] | ir_only wins |
| lowlight/night | 0.1077 | 0.1087 | -0.0010 | [-0.0016, -0.0004] | ir_only wins |
| glare/day | 0.2584 | 0.0111 | +0.2474 | [+0.2369, +0.2598] | gated wins |
| glare/night | 0.1077 | 0.1087 | -0.0010 | [-0.0016, -0.0004] | ir_only wins |

**Six of eight cells now lose with intervals excluding zero.** The two survivors
are day cells where IR contributes nothing and gated fusion is effectively VIS
alone. §9.3's premise — that only raising `ir_only` can move night — was correct;
what it did not anticipate is that raising `ir_only` removes the claim rather
than strengthening it.

The capability prior for IR moved 0.0206 -> 0.0267 automatically with the better
detector.

---

## 3. The photometric gate is redundant (the one clean positive)

`x_fusion_ci.md` §3. The record reports the gate and veto as a chain, so the
interaction had never been read off. `veto_only` keeps the switch and removes the
photometric term from the soft weight.

| cell | nogate | gate | veto_only | gate+veto | interaction |
|---|---:|---:|---:|---:|---:|
| clean/day | 0.3341 | 0.3341 | 0.3341 | 0.3341 | +0.0000 |
| clean/night | 0.0764 | 0.0787 | **0.0813** | 0.0813 | -0.0023 |
| fog/day | 0.0076 | 0.0076 | 0.0076 | 0.0076 | +0.0000 |
| fog/night | 0.0784 | 0.0784 | 0.0789 | 0.0789 | +0.0000 |
| lowlight/day | 0.0086 | 0.0087 | 0.0087 | 0.0087 | -0.0001 |
| lowlight/night | 0.0786 | 0.0786 | **0.0813** | 0.0813 | +0.0000 |
| glare/day | 0.2602 | 0.2602 | 0.2602 | 0.2602 | +0.0000 |
| glare/night | 0.0664 | 0.0778 | **0.0813** | 0.0813 | -0.0114 |

`veto_only` equals the full system in **every** cell. The gate's +0.0114 on
glare/night is exactly cancelled by a -0.0114 interaction. **The photometric term
inside the soft weight can be deleted with zero loss** — the hard switch subsumes
it. This simplifies the method and removes a fitted component (`mu_b`, `tau_b`,
`p05`) from the description.

---

## 4. Veto hysteresis — the one cell that closes

`runs/eval/x_veto_hysteresis.md`. Filters applied to the veto decision in capture
order; `k=1` reproduces the adopted per-frame rule exactly (asserted). Target is
fog/night, the cell §4.5 leaves open; guard is clean/day.

| mode | k | fog/night | delta vs adopted | 95% CI | clean/day |
|---|---:|---:|---:|---|---:|
| majority | 1 | 0.0789 | +0.0000 | — | 0.3341 |
| majority | 31 | 0.0805 | +0.0015 | [+0.0003, +0.0024] | 0.3341 |
| majority | 61 | 0.0808 | +0.0019 | [+0.0007, +0.0028] | 0.3341 |
| dilate | 9 | 0.0805 | +0.0016 | [+0.0003, +0.0025] | 0.3341 |
| **dilate 15** | 15 | **0.0809** | **+0.0020** | **[+0.0007, +0.0028]** | **0.3341** |
| dilate | 31 | 0.0809 | +0.0020 | [+0.0007, +0.0028] | 0.3341 |
| dilate | 61 | 0.0809 | +0.0020 | [+0.0007, +0.0029] | 0.3341 |

The guard cell does not move by a single digit at any setting. The mechanism is
visible in the veto rate: fog/night goes from **29% -> 71%** VIS-vetoed under
`dilate 15` *(this line originally said 89%, a transcription error against
`x_veto_hysteresis.json`'s 0.7074; caught 2026-08-20 when
`eval_final_system.py` independently reproduced 71% and mAP 0.0809 exactly)*. The per-frame rule was simply under-firing on that run, and the right
answer was to veto almost always.

`dilate 15` brings fog/night to 0.0809 against `ir_only`'s 0.0810 — **equal, not
above.** It closes the gap; it does not produce a win.

---

## 5. Registration drift — the A2 per-run plan does not hold

`runs/eval/x_registration_drift.md`. IR **GT** boxes mapped through the per-run
homography, matched to VIS **GT** at IoU >= 0.1, same class: 13,506 pairs. This is
geometry, not the detector.

Within-run dx swing across 10 equal-count time bins:

| run | pairs | median abs residual | dx swing | dy swing | per-run correction leaves |
|---|---:|---:|---:|---:|---:|
| pohang00 | 5144 | 5.74 | **10.01 px** | 1.75 | +1.46 px |
| pohang01 | 6103 | 4.11 | 9.47 px | 2.24 | +0.68 px |
| pohang02 | 721 | 3.14 | 4.93 px | 0.84 | +0.40 px |
| pohang03 | 1538 | 5.82 | 6.59 px | 1.80 | +0.06 px |

The **-4.18 px between-run swing killed the global correction** (§9.2 item 6). The
within-run swing is **larger than that** on two of four runs. A per-run constant
is therefore the same mistake one level down, and A2 as specified needs a
time-varying term or should be dropped. The drift is x-only; dy is stationary
(swing 0.84-2.24 px).

---

## 6. Two nulls

**Top-k truncation** (`runs/eval/x_topk_truncation.md`). VIS emits 29,042
detections and IR 120,877 on the same 2232 frames (4.16x, 13.0 vs 54.2 per frame)
at conf 0.001. A rank cutoff, unlike the score cutoff rejected in §8, does not
require the two score scales to be comparable — but it is null. Every k >= 50 is
within 0.0001 or exactly zero on every cell; k=25 hurts (clean/day -0.0018,
fog/night -0.0008, both with intervals excluding zero). **Reject.**

**Score calibration** (`runs/eval/x_score_calibration.md`). Isotonic
`conf -> P(TP@0.5)` fitted per modality on pohang00/02/03 and applied to both
streams; pohang01 held out, so the night rows are a generalisation test.
`visible_only` and `ir_only` are invariant under a monotone map — asserted for
every condition — so all movement is cross-modal interleaving.

The two scales are genuinely incomparable:

| raw conf | VIS -> P(TP) | IR -> P(TP) | ratio |
|---:|---:|---:|---:|
| 0.10 | 0.2323 | 0.0523 | 4.4x |
| 0.50 | 0.6704 | 0.1159 | 5.8x |
| 0.75 | 0.8426 | 0.1306 | 6.5x |

But fixing it nets ~zero: day cells gain (+0.0009 to +0.0031), night cells lose
(-0.0013 to -0.0033). **Not a lever.**

---

## 7. Capability prior refit

`runs/eval/x_capability_refit.md`. The adopted prior is fitted on all 2232 clean
frames including the held-out pohang01 (`scripts/run_fusion_eval.py:163-175`).
Refitted run-disjoint on pohang00/02/03:

| prior | VIS | IR | ratio | frames |
|---|---:|---:|---:|---:|
| adopted | 0.2580 | 0.0206 | 12.50x | 2232 |
| run-disjoint | 0.3352 | 0.0092 | 36.25x | 1200 |

**The leak does not touch the headline**: clean/night moves +0.0000, because VIS
is vetoed there and the weights never enter. Largest movement anywhere is
glare/day at +0.0047.

It does move the number §4.1's argument quotes. Mean `w_vis` on the night run goes
**0.430 -> 0.672**. That argument needs rewriting — though it cuts in favour of
the veto being necessary, not against it.

---

## 8. Training arms

### 8.1 Stage 3 — p2feat replication (`runs/screen3/`)

Ranking statistic is the **mean of the last 5 epochs**, not peak.

| run | imgsz | seed | last-5 mAP50-95 | peak |
|---|---:|---:|---:|---:|
| s3_p2feat_640_b10 | 640 | 0 | 0.06066 | 0.06591 |
| s3_p2feat_960_seed1 | 960 | 1 | 0.06116 | 0.06700 |

The 960 arm is +0.0005 last-5 over 640 — inside the noise floor, and it costs
~2.3x the time. Consistent with §3 of `docs/screen-small-object-2026-08-19.md`:
native IR is 640x512, so imgsz 960 is pure upsampling. **640 stands.**

These are 2-class mAP. **Per-class ship AP for the stage-3 checkpoints has not
been measured** — `perclass_ap.py` needs a GPU val pass and the card is busy.

### 8.2 Stage 4 (`runs/screen4/`) — 3 of 4 complete

All arms: p2feat, batch 10, 25 epochs, `sigma_width` 32, seed 0, same 11,640
stride-2 frames, so each differs from `s3_p2feat_640_b10` in exactly one thing.

| run | change | last-5 | peak | vs s3 baseline (0.06066 last-5) |
|---|---|---:|---:|---|
| s4_ir_shiponly | buoy class dropped, nc=1 | 0.12722 | 0.13592 | **not comparable** — see below |
| s4_ir_clahe | CLAHE before the 8-bit map | 0.06126 | 0.06347 | +0.0006, inside noise |
| s4_ir_rect | rect=True, no letterbox padding | 0.05471 | 0.05809 | **-0.0060, worse** |
| s4_vis_rect | rect=True, VIS 896, batch 8 | 0.23768 | 0.24884 | 18/25 epochs, still running |

- **B5 ship-only**: nc=1, so its mAP **is** ship AP and must not be read against a
  2-class mAP. The comparison it needs is stage-3's ship AP, which is the
  unmeasured number in §8.1. Deferred, not concluded.
- **B3 CLAHE**: +0.0006 last-5. Local contrast is not the answer to the IR
  small-object problem. **Null.**
- **B1 rect (IR)**: **worse by 0.0060.** Removing the 128 grey padding rows was
  predicted to be worth ~1.12x linear resolution; instead it cost. The likely
  cause is that Ultralytics forces `shuffle=False` when `rect=True`
  (`.venv/Lib/site-packages/ultralytics/models/yolo/detect/train.py:96`), so the
  arm confounds aspect-ratio with loss of shuffling. **The rect result is not
  clean and should not be read as an aspect-ratio finding.**

---

## 9. Incident — stage 4 first launched on the CPU torch build

`chain_stage4.py` launches the training queue with `sys.executable`, and the chain
itself was started under `.venv\Scripts\python.exe`, which is **torch+cpu**.
`s4_ir_shiponly` ran 4 epochs at **3,585 s/epoch** before this was caught, against
**181 s/epoch** on the correct interpreter — a 19.8x penalty, and ~100 h for the
four arms instead of ~5 h. Caught from `nvidia-smi` showing 12% utilisation and
1,093 MiB in use, which is not a batch-10 job.

The CPU epochs are preserved at `runs/screen4/s4_ir_shiponly_cpu_aborted/`; the run
was restarted fresh rather than resumed, to avoid carrying CPU-side optimizer state.

**Fix not yet applied:** `chain_stage4.py` should pin the GPU interpreter
explicitly rather than inherit `sys.executable`. Any future chain script has the
same trap.

---

## 10. What this implies

The fusion contribution as currently framed does not survive. The defensible
reframing is narrower and still true:

> A hard photometric veto recovers the working sensor without being told which one
> failed, and it needs no soft weight, no Mahalanobis term, and no photometric
> gate term.

That is a **sensor-selection** result, not a fusion result. Supporting it rests on
§3 (the gate is redundant) and §4 (hysteresis closes the last cell), and requires
retiring the night-superiority claim in §1 and §2.

### Open items

1. Bootstrap gated vs `visible_only` on clean/day and glare/day — the only
   remaining point estimates in §1.
2. Per-class ship AP for the stage-3 checkpoints, so B5 can be read (§8.1, §8.2).
3. Re-run B1 rect with shuffling controlled, or drop the arm (§8.2).
4. Pin the interpreter in `chain_stage4.py` (§9).

---

## 11. File index

| file | section |
|---|---|
| `runs/eval/x_fusion_ci.md` / `.json` | §1, §3 |
| `runs/eval/x_ir_upgrade_p2feat.md` / `.json` | §2 |
| `runs/eval/x_veto_hysteresis.md` / `.json` | §4 |
| `runs/eval/x_registration_drift.md` / `.json` | §5 |
| `runs/eval/x_topk_truncation.md` / `.json` | §6 |
| `runs/eval/x_score_calibration.md` / `.json` | §6 |
| `runs/eval/x_capability_refit.md` / `.json` | §7 |
| `runs/screen3/`, `runs/screen4/` | §8 |
| `runs/queue_analysis/queue.log` | run log for §1-§7 |
| `runs/queue_screen4/chain.log` | run log for §8.2, §9 |

Scripts: `scripts/eval_fusion_ci.py`, `eval_capability_refit.py`,
`diag_registration_drift.py`, `eval_score_calibration.py`,
`eval_veto_hysteresis.py`, `eval_topk_truncation.py`,
`eval_ir_upgrade_fusion.py`, `run_analysis_queue.py`, `chain_stage4.py`.
Shared: `src/uqfusion/eval/ctx.py`, `src/uqfusion/eval/apmetrics.py`
(gated by `scripts/smoke_apmetrics.py`, 4 checks).

**Note:** `runs/` is gitignored, so none of the result files above are under
version control.

---

## 12. Disposition

The consequences of this round are frozen in
[`architecture-final-2026-08-20.md`](architecture-final-2026-08-20.md)
(decisions D27–D30): veto-only photometric term, dilate-15 hysteresis,
run-disjoint capability prior, p2feat IR recipe, and the sensor-selection
reframing. The four open items in §10: item 1 and the soft-weight ablation are
measured by `scripts/eval_final_system.py` (`runs/eval/final_system.md`); item
2 waits on the GPU; item 3 is dropped-unless-controlled; item 4 is fixed
(`resolve_gpu_python`).
