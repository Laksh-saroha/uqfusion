# §5.1 — the largest-common-fit batch on this laptop

**Measured 2026-09-10** by `scripts/probe_batch_ceiling.py`, under Amendment 3's decision
that Phase 3 trains on the laptop (RTX 4080 Laptop, 11.99 GiB usable). Satisfies §5.1's
"largest-common-fit batch is measured once and recorded in the manifest, and it is identical
across every arm."

---

## Verdict

**Batch 12, both arms.** IR's ceiling is 14 and VIS's is 12; §5.1 takes the common value, so
12. It costs IR ~2.3% against its own optimum and costs VIS nothing — 12 *is* VIS's optimum.

---

## Why this could not be an OOM probe

Windows WDDM does not raise OOM when a run exceeds VRAM. It pages into host RAM and the run
**succeeds**, slowly: `phase1-experimental-record.md:688` measured `yolo26x` at batch 16
peaking 15.31 GB on a 12.28 GB card and running at 2.4 img/s, about **17× slower**. A probe
that asks "did it crash" would call that a pass.

So the probe measures **ms/img**, and applies two criteria: throughput within 15% of the
best seen, and driver-level headroom of at least 1.5 GiB. `memory_reserved` excludes CUDA
context, cuDNN workspaces and driver overhead; §16 of the Phase 1 record measured a 9.70 GiB
reserved band sitting at ~10.7 GiB driver-level, so overhead is ~1.0 GiB and 1.5 GiB is the
stated safe margin.

Each batch runs in a fresh subprocess — allocator fragmentation survives within a process,
so one large batch would poison the next.

## IR — `yolo26m-p2feat`, nc=1, full epoch over 11,640 frames

| batch | ms/img | reserved | driver ~ | headroom | |
|---:|---:|---:|---:|---:|---|
| 10 | 39.7 | 6.37 | 7.37 | 4.62 | ok |
| 12 | 39.6 | 7.56 | 8.56 | 3.43 | ok |
| 14 | 38.7 | 8.68 | 9.68 | 2.31 | **ceiling** |
| 16 | 35.9 | 10.12 | 11.12 | **0.87** | tight |

Batch 16 is the fastest cell measured but leaves 0.87 GiB against the 1.5 GiB margin. **The
memory criterion binds here, not throughput.**

## VIS — `yolo26m`, nc=2, 11,488 frames

Run twice, ascending and descending, because the first result looked like an artifact.

| batch | ascending | descending | reserved | headroom | |
|---:|---:|---:|---:|---:|---|
| 12 | 44.6 (1st) | 43.0 (3rd) | 6.83 | 4.16 | **ceiling** |
| 14 | 55.1 | 55.2 | 7.84 | 3.15 | slow |
| 16 | 54.3 (3rd) | 55.3 (1st) | 8.96 | 2.03 | slow |

**The throughput criterion binds here, not memory** — every cell has 2 GiB or more of
headroom.

### A hypothesis that was wrong, recorded because it was acted on

The ascending run showed 12 fast and 14/16 slow and roughly equal to each other, which is
the signature of **thermal throttling across sequential runs on a laptop**. That was the
stated reason for re-running descending.

**It is not thermal.** Batch 12 is fast whether it runs first (44.6) or last (43.0), and
16 is slow whether it runs first (55.3) or last (54.3). Peak memory reproduces to 0.01 GiB
across both orders. The effect is a real and highly repeatable property of the batch size:
`yolo26m` at 640 runs **~28% slower** at 14 and 16 than at 12 on this card, with GiB of
memory to spare.

The mechanism is not identified here — plausibly cuDNN algorithm selection at those shapes —
and identifying it is not necessary to fix the constant. What matters is that it reproduces
under order reversal, so it is a property of the configuration and not of the measurement.

`fraction` was ruled out as an explanation separately: ultralytics takes a deterministic
prefix (`im_files[: round(len * fraction)]`, `data/base.py:183`), so every VIS probe ran the
identical 11,488 frames.

## Carried into the manifest

* batch **12**, both arms, `workers=8` (16 measured worse; §16 of the Phase 1 record)
* `imgsz` 640, and the accumulation Ultralytics derives from it:
  `accumulate = max(round(64 / 12), 1)` = 5, effective batch 60 against `nbs` 64
* raw rows in `runs/eval/batch_ceiling_{vis,ir}.json`

## What this does not settle

* **It is one card in one thermal envelope.** The numbers do not transfer to `dgxanode01`,
  which Amendment 3 removed from Phase 3 anyway.
* **The VIS probe used a 30% prefix, IR a full epoch.** IR's full-epoch peaks came in
  slightly *below* its 15%-subset peaks (10.12 vs 10.36 at batch 16), so the subset was not
  under-reading — but VIS's ceiling rests on a prefix and a full-epoch peak could differ by
  the ±0.35 GiB the allocator sawtooth spans. It has 4.16 GiB of headroom, so this does not
  change the verdict.
* **Nothing here says batch 12 is optimal for accuracy**, only for throughput within a
  memory bound. §5.1 requires one batch across every arm precisely so that batch is not a
  free variable, and it is not re-opened per arm.

---

# Addendum — 2026-09-11: production contradicts the VIS verdict

Measured on `p3_vis_seed0` (Stage 2, running) against `gauss_vis_seed0_nightfull`, the
closest prior run on record. **The probe's VIS ordering is inverted in production.**

## The two runs are comparable

| | `gauss_vis_seed0_nightfull` | `p3_vis_seed0` |
|---|---|---|
| model | `yolo26m.pt` | `yolo26m.pt` |
| head | gaussian + sigma (`nll_loss` columns present) | gaussian + sigma |
| workers | 8 | 8 |
| imgsz | 640 | 640 |
| **batch** | **16** | **12** |
| train / val frames | 48,136 / 11,352 | 38,295 / 9,009 |
| steady epoch | **1,083.2 s** (median, ep4+, n=36) | 990.3 s (ep3; still settling) |

Only the batch and the frame counts differ. The frame counts differ because G5 holds
pohang04 out — 20.4% of the VIS stride-2 list — which is why the epoch is shorter in wall
clock while being *slower per image*.

## Per-image rates

The live counter reads **3.39–3.42 it/s at batch 12 = ~41 img/s**, directly measured, train
only. For the batch-16 run the rate has to be derived: the current run's epoch is 990.3 s
against 933 s of training, leaving ~57 s for 9,009 val frames (~158 img/s); applying that
val rate to 11,352 frames gives ~72 s, so its training was ~1,011 s for 48,136 frames =
**~47.6 img/s**.

**Batch 16 is therefore ~16% faster per image than batch 12 in production. The probe
measured batch 12 as 28% faster.** The sign is reversed, not the magnitude.

## What the probe cannot be blamed for

It is **not** setup dominance. The VIS probe ran `fraction=0.30` — 11,488 frames, not a
token slice — and a constant per-run setup cost adds the same ms/img to every batch size,
compressing ratios rather than inverting them. The 43.0 vs 55.3 ms/img gap is 141 s of real
work at that frame count; no fixed overhead produces it.

It is **not** thermal, and it is **not** `fraction` non-determinism. Both were tested at the
time: the ordering reproduced under full reversal, and ultralytics takes a deterministic
prefix.

## Three ways the probe's conditions differ from production

Recorded as candidates, none of them confirmed as the mechanism:

1. **The 30% prefix is not a 30% sample.** The list is ordered by run, so `fraction=0.30`
   trained on **pohang00 (8,193) + pohang01 (3,295) and no pohang02 or pohang03 at all**.
   pohang01 is the all-night run and is box-sparse. Assigner and loss cost scales with boxes
   per batch, and that cost scales with batch size, so a box-sparse prefix is a plausible
   place for batch-size scaling to behave differently than it does on the full list.
2. **`val=False` in the probe**, `val=True` in production.
3. **One draw per batch size.** Two orders agreed with each other, which establishes
   repeatability *within the probe's conditions* and says nothing about transfer out of them.

The honest reading is that the probe measured a real property of the configuration it ran,
and that configuration was not the training configuration.

## The verdict stands anyway, and this is a cost decision not a correctness one

**Batch stays 12 for all of Stage 2.** §5.1 requires one batch "identical across every arm",
four IR seeds are already trained at 12, and switching VIS mid-stage would introduce a
throughput-motivated confound into the one stage the pre-registration exists to protect.

The price is ~129 s/epoch on this run, about **1.6 h per VIS seed at the observed ~45-epoch
early-stop depth, so ~8 h across the five**. That is the cost of the wrong constant, paid
knowingly, and it is cheaper than the confound.

## What this changes for the next measurement

* **A throughput probe must run the production data, or its ordering does not transfer.**
  A prefix of a run-ordered list is a different distribution, not a smaller one. Either use
  `fraction=1.0` (as the IR probe did — and IR's ordering has not been contradicted) or
  shuffle before taking the prefix.
* **Epoch 1 is not a rate.** Both runs show ~22% above steady on the first epoch
  (1,351 → 1,083 and 1,205 → ~990) from dataset scan, AMP check and worker spin-up. Reading
  a rate off epoch 1, in either direction, is reading setup.
* The headroom criterion added on 2026-09-10 was the right addition and is untouched by
  this: batch 16 peaked at 8.96 GiB reserved, ~2.03 GiB of headroom, and would have passed
  it. It was the **throughput** criterion that chose 12, and the throughput criterion is the
  one production contradicts.
