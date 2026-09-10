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
