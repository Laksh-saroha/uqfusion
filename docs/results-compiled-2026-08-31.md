# Compiled results — 2026-08-31, on return from the four-day unattended window

**Written 2026-08-31 ~17:00 UTC / 22:30 IST.** Compiles everything both boxes
produced since [`vis-benchmark-stride4-2026-08-27.md`](vis-benchmark-stride4-2026-08-27.md)
was written. Server files pulled through the JupyterLab contents API
(`http://<server>:<port>`); raw pull banked at `archive/server_pull_2026-08-31/`.

Compiled tables: `phase1_benchmark/compiled/`.

| file | contents |
|---|---|
| `ir_benchmark_stride4_variants.csv` | laptop IR sweep, seeds folded per variant |
| `ir_benchmark_stride4_runs.csv` | laptop IR sweep, per run |
| `vis_benchmark_stride4_variants.csv` | server VIS sweep, seeds folded (2 variants so far) |
| `dgxanode01_runs_corrected.csv` | every non-pending server run, with the §1 correction applied |

---

## 0. State in one paragraph

Nothing needed rescuing. Both boxes ran the whole four days unattended: the
server finished its 14-run UQ-arm queue and is 6/93 into the VIS 2-class
benchmark with `vis_bench_yolo26x_seed0` live at epoch 29; the laptop is 43/93
into the IR benchmark with `ir_bench_yolov10s_seed1` live at epoch 56. Both
trackers were refreshing normally. Three things came out of the compile that are
not operational: a **one-epoch reporting defect** on six server rows (§1), the
**IR sweep is now formally unable to rank architectures** (§2), and
**`mc_vis_seed0_ft` is a genuine failure** that survives the correction (§4).

---

## 1. Defect: `best_mAP50-95` off by one epoch on six server rows

`scripts/run_queue.py` had an off-by-one in `on_fit_epoch_end` — `stopper.best_epoch`
is 1-based, `trainer.epoch` is 0-based, and comparing them directly matched one
epoch late, so `best_map50_95` recorded the metric of the epoch **after** the one
saved as `best.pt`. The comment at `scripts/run_queue.py:686-689` documents the
fix, dated 2026-08-24.

Every run started before that fix reached the server carries the bug. Checked by
re-reading each run's own `results.csv` at its recorded `best_epoch`:

| scope | rows checked | off by one |
|---|---|---|
| laptop IR benchmark (`status_laptop.csv`) | 43 | **0** |
| server VIS benchmark (`queue_vis_benchmark_stride4`) | 6 | **0** |
| server UQ arms (`queue_vis_server`) | 14 | **6** |

The six are exactly the first-batch runs — `gauss_vis_seed0`, `mc_vis_seed0`,
`ens_vis_seed0` and their three `_ft` companions. `ens_vis_seed1` onward are clean,
which dates the server's code sync to between `ens_vis_seed0` and `ens_vis_seed1`.

| run | as tracked | corrected | delta |
|---|---|---|---|
| `gauss_vis_seed0` | 0.25552 | 0.25591 | +0.00039 |
| `gauss_vis_seed0_ft` | 0.22642 | **0.24750** | **+0.02108** |
| `mc_vis_seed0` | 0.24654 | 0.24667 | +0.00013 |
| `mc_vis_seed0_ft` | 0.15071 | 0.15123 | +0.00052 |
| `ens_vis_seed0` | 0.24402 | **0.25086** | **+0.00684** |
| `ens_vis_seed0_ft` | 0.23617 | **0.24711** | **+0.01094** |

**Consequences.** Two readings that the raw tracker supports are artifacts and
must not be carried forward:

- "the mosaic-off `_ft` stage badly hurts the σ-head arm" — `gauss_vis_seed0_ft`
  loses 0.008 from its parent, not 0.029.
- the deep-ensemble members' seed spread. Corrected, the five `_ft` members are
  **mean 0.24706, sd 0.00212, spread 0.00533** — against sd 0.00531 / spread
  0.01361 as tracked. The ensemble is considerably tighter than it looked, and
  `ens_vis_seed0` is no longer an outlier.

`runs/status_dgxanode01.xlsx` and `.csv` are the server's own artifacts and still
carry the uncorrected values; `dgxanode01_runs_corrected.csv` adds
`best_mAP50-95_corrected` and `correction_applied` beside them rather than
overwriting, so both are auditable. **Anything that reaches a table should read
the corrected column.**

Not yet done: the server's `run_queue.py` still holds the buggy revision, so any
run it starts before the code is synced will reproduce this. The six affected
runs do not need retraining — the checkpoints on disk are correct, only the
reported number was wrong, and it is recoverable from `results.csv` as done here.

---

## 2. IR benchmark (laptop) — the sweep cannot rank architectures

43 of 93 done, 220.7 GPU-h, 13 of 16 variants with all three seeds.
Full table: `phase1_benchmark/compiled/ir_benchmark_stride4_variants.csv`.

| variant | GFLOPs | mean mAP50-95 | sd | spread |
|---|---|---|---|---|
| yolov8x | 258.5 | 0.06688 | 0.00176 | 0.00335 |
| yolov9e | 193.0 | 0.06502 | 0.00215 | 0.00391 |
| yolov8l | 165.7 | 0.06865 | 0.00470 | 0.00907 |
| yolo26l | 93.8 | 0.07316 | 0.00235 | 0.00467 |
| yolov8m | 79.3 | 0.07170 | 0.00325 | 0.00591 |
| yolov9m | 77.9 | 0.06942 | 0.00412 | 0.00808 |
| yolo26m | 75.4 | 0.06745 | 0.00274 | 0.00525 |
| yolov8s | 28.8 | 0.06784 | 0.00579 | 0.01121 |
| yolov9s | 27.6 | 0.06692 | 0.00581 | 0.01097 |
| yolo26s | 22.8 | 0.06950 | 0.00613 | 0.01160 |
| yolov8n | 8.9 | 0.06613 | 0.00662 | 0.01291 |
| yolov10n | 8.7 | 0.06124 | 0.00857 | 0.01684 |
| yolov9t | 8.5 | 0.06908 | 0.00654 | 0.01308 |

The §5 concern in [`handoff-2026-08-26.md`](handoff-2026-08-26.md) — raised on 15
runs — is now settled on 43. One-way ANOVA over the 13 complete variants:

```
between-variant MS 2.675e-05    within-variant (seed) MS 2.580e-05
F(12,26) = 1.037                [F_crit ~2.0 at p=.05]
variance explained by variant: 32.4%   (chance level for 13 groups of 3 is ~32%)
```

F is 1.04. Variant identity explains **nothing** beyond seed noise. The size
effect is likewise absent: regressing mean mAP on log GFLOPs gives slope
+0.0017/decade (r = +0.300), so the full 30x compute range from `yolo26n` (6.1
GFLOPs) to `yolov8x` (258.5) predicts **+0.0028 mAP** — smaller than the 0.0047
median seed sd *within a single variant*. The best and worst variant means differ
by 0.0119; the worst variant's own three seeds differ by 0.0168.

**This is a result, not a failure**, and it is the same shape as Phase 1's
headline: at this IR operating point (mAP50-95 ~0.06–0.07, i.e. near the floor)
architecture choice is not identifiable. Two things to decide, neither urgent:

1. **Whether to spend the remaining ~50 runs.** They cannot change the ranking
   conclusion — they can only tighten error bars around a null. The 220 GPU-h
   already spent bought the answer.
2. **Whether the operating point itself is the finding.** Every model sits near
   0.07 mAP50-95 with recall ~0.16. That is a data/task ceiling, not a model
   ceiling, and it is the more interesting sentence for the paper.

Also on the laptop: `ir_bench_yolov9c_seed0` and `_seed1` are marked `diverged`
(epochs 7 and 21). Given the `yolov8l` false positive documented in
[`ir-benchmark-divergence-falsepos-2026-08-26.md`](ir-benchmark-divergence-falsepos-2026-08-26.md),
these two are worth reading before being accepted as real divergences.
`ir_bench_yolo26x_seed0` is `paused` at epoch 1.

---

## 3. VIS 2-class benchmark (server) — 6/93, too early to read

Full table: `phase1_benchmark/compiled/vis_benchmark_stride4_variants.csv`.

| variant | GFLOPs | seed0 | seed1 | seed2 | mean | sd |
|---|---|---|---|---|---|---|
| yolov8x | 258.5 | 0.23344 | 0.22469 | 0.23153 | 0.22989 | 0.00460 |
| yolo12x | 200.3 | 0.21775 | 0.22084 | 0.23246 | 0.22368 | 0.00776 |

The two means differ by 0.0062 against seed sds of 0.005–0.008 — not separable,
same as Phase 1. But `vis_bench_yolo26x_seed0` is **live at 0.26407 (epoch 29 of
100, still climbing)**, which is +0.034 over `yolov8x`'s mean — roughly 7x the
seed sd. If it holds through its remaining epochs and its two siblings agree,
this sweep will separate `yolo26x` from the rest, which the ship-only Phase 1
grid never managed. Worth watching; not yet a claim.

Unlike the IR sweep, seed spread here is *not* swamping the model effect, so the
§5.5 "trim it" decision from the setup doc does **not** apply to this queue.

### The ETA in the setup doc is wrong

[`vis-benchmark-stride4-2026-08-27.md`](vis-benchmark-stride4-2026-08-27.md) §4
projected ~320 GPU-h / 13–14 days, by halving Phase 1's 647 GPU-h for halving the
frames per epoch. Measured against Phase 1's own per-variant h/epoch, the
realised ratio is **0.78x, not 0.5x** — and it is not stable across variants
(`yolov8x` 0.98x, `yolo12x` 0.58x). Re-projecting per variant off Phase 1
timings at 46 mean epochs gives **~680 GPU-h total, ~25 more days**, and with
only two variants measured the honest range is **500–900 GPU-h / 20–30 days**.

Do not use the tracker's own fitted ETA yet: its `h/epoch ≈ a + b × GFLOPs` fit
currently returns a **negative** slope, because the only two variants observed are
`yolov8x` (258 GFLOPs, 0.205 h/epoch) and `yolo12x` (200 GFLOPs, 0.371 h/epoch) —
the smaller model is the slower one, so GFLOPs is inverted over the sampled range.
The fit becomes meaningful once the m/s/n end lands, exactly as §3 of that doc
warned.

---

## 4. Phase 3 VIS UQ arms — complete, and one arm is broken

All 14 runs done, 163 GPU-h, `yolo26m` @ stride 2, imgsz 640. Corrected values:

| arm | pre-`_ft` | `_ft` | delta |
|---|---|---|---|
| σ-head (`gauss_vis_seed0`) | 0.25591 | 0.24750 | −0.0084 |
| MC-Dropout (`mc_vis_seed0`) | 0.24667 | **0.15123** | **−0.0954** |
| ensemble seed0 | 0.25086 | 0.24711 | −0.0038 |
| ensemble seed1 | 0.24948 | 0.24560 | −0.0039 |
| ensemble seed2 | 0.25098 | 0.24978 | −0.0012 |
| ensemble seed3 | 0.24530 | 0.24835 | +0.0031 |
| ensemble seed4 | 0.24944 | 0.24445 | −0.0050 |

Deep-ensemble members (5x `_ft`): **mean 0.24706, sd 0.00212, spread 0.00533**.

**`mc_vis_seed0_ft` is a real failure and survives the §1 correction.** Its own
epoch curve is unambiguous — it never had a good epoch:

| epoch | train/box | train/cls | mAP50 | mAP50-95 |
|---|---|---|---|---|
| 1 | 1.973 | 1.201 | 0.3845 | 0.1482 |
| 2 | 1.375 | 0.519 | 0.3779 | **0.1512** (best) |
| 5 | 1.152 | 0.469 | 0.3626 | 0.1474 |
| 10 | 1.069 | 0.444 | 0.3515 | 0.1434 |

Compare epoch-1 losses against the other two arms' fine-tunes: `gauss_vis_seed0_ft`
starts at box 0.629 / cls 0.229, `ens_vis_seed0_ft` at 0.752 / 0.284. The MC
fine-tune starts **~3x higher on both** and never gets below box 1.069 — where the
others finish near 0.51 and 0.59. Its parent `mc_vis_seed0` reached 0.24667, so
the head dropout is not inherently this costly.

**Root cause — found and reproduced on CPU, 2026-08-31.** `insert_head_dropout`
(`src/uqfusion/uq/mc_dropout.py:73`) rebuilds each head branch as
`nn.Sequential(*children[:-1], nn.Dropout2d(p), children[-1])`. That **renumbers
the final conv**: `one2one_cv2.0.2.weight` becomes `one2one_cv2.0.3.weight`.

The fine-tune reloads through the *training* path — `MCDropoutTrainer.get_model`
calls `super().get_model(cfg, weights)`, which builds a plain model from yaml and
loads the checkpoint **by key name**, and only then inserts dropout. The shifted
keys do not match, and Ultralytics' loader intersects silently:

```
checkpoint tensors : 768
transferred        : 756
SILENTLY DROPPED   :  12   <- one2one_cv2/.cv3 final convs, all 3 scales
parameters lost    : 62,460  (0.28% of the model)
```

0.28% of the parameters, but **100% of the output layer**: every box coordinate
and every class logit comes from those six 1x1 convs. They arrive
randomly initialised. That is precisely the observed curve — backbone and neck
intact (so it reaches 0.15 rather than 0.00), output heads from scratch, and 10
epochs at lr 1.67e-4 decaying to 1.67e-5 is nowhere near enough to relearn them.

**The IR twin is hit too, and the archive proves the mechanism.** `mc_ir_seed0_ft`
(post-rebuild) starts at box 2.246 / cls 1.863 and ends at 0.11894, *below* its
parent's 0.12973. The pre-rebuild `mc_ir_seed0_ft_broken-20260823` starts at box
1.269 / cls 0.562 and reaches 0.12871 — level with its parent. Before the
2026-08-23 rebuild dropout went into `cv2`/`cv3`, the *discarded* one2many branch,
so the deployed `one2one` output convs were never renumbered and loaded fine. The
rebuild fixed the zero-variance defect and exposed this one. Same code, opposite
failure.

**Blast radius is the training-time reload only.**

| path | mechanism | status |
|---|---|---|
| eval / inference (`MCDropoutPredictor` → `PlainPredictor` → `load_model` → `YOLO()`) | restores the **pickled module**, dropout and indices intact — no key matching | **safe**, and `enable_mc_dropout` raises if dropout is missing |
| parent runs `mc_vis_seed0`, `mc_ir_seed0` | started from plain `yolo26m.pt`, so insert-after-load is correct | **sound** |
| `_ft` / any warm start from an MC checkpoint | `model.load()` by key name | **broken** |

So `mc_vis_seed0` itself is fine — 0.24667, in family with σ-head 0.25591 and the
ensemble's 0.24948 mean — and its cached predictions are trustworthy. It is the
two `_ft` checkpoints that are invalid.

**Why the gate missed it.** `scripts/smoke_mc_e2e.py` checks placement, execution
and stochasticity on a **freshly built** model. It never saves a checkpoint and
reloads it. The file's own lesson — "a layer that exists, and even a layer that is
invoked, is not a layer that does something" — needs one more clause: *and a model
that trains is not a model that reloads.*

### 4b. Can `mc_vis_seed0_ft` be re-run on the laptop?

**Yes, provided it starts from the server's parent checkpoint — not a laptop
retrain of the parent.** The repo already contains the paired measurement that
settles this: `gauss_vis_seed0` was run on *both* boxes at an identical config
(seed 0, batch 16, imgsz 640, `data_vis_stride2.yaml`, lr0 0.01, `yolo26m`).

| | laptop | server | delta |
|---|---|---|---|
| `gauss_vis_seed0` | 0.24959 (45 ep, best 25) | 0.25591 (73 ep, best 52) | **−0.00632** |
| `gauss_vis_seed0_ft` | 0.24113 | 0.24750 | **−0.00637** |

Two readings, and the second is the useful one:

- **A full parent retrain on the laptop is not comparable.** The 0.0063 gap is
  larger than the deep ensemble's entire 5-seed spread (0.00533) and ~3x its seed
  sd (0.00212) — the size of effect the Phase 3 table is trying to resolve. The
  trajectories diverge outright (45 vs 73 epochs) and early stopping amplifies it,
  so it is not a fixed offset that could be corrected away.
- **The fine-tune stage itself contributes almost nothing.** Parent gap −0.00632,
  `_ft` gap −0.00637: the `_ft` added **−0.00005**. The whole gap is inherited from
  the parent. A 10-epoch, mosaic-off, lr 1.67e-4 fine-tune is evidently
  reproducible across these two GPUs.

So: download `mc_vis_seed0/weights/best.pt` (42 MB) from the server and fine-tune
*that* on the laptop. VRAM is not a concern — `gauss_vis_seed0_ft` already ran
locally at batch 16 / 640 with the heavier σ head. Budget ~2.7 h (dgxanode01's MIG
slice is only 1.23x the laptop).

**One caveat worth 2.7 h.** The −0.00005 figure is n=1. Since the whole point is a
number that survives review, re-run one *already-known* server `_ft` on the laptop
from the server's own checkpoint — `ens_vis_seed0_ft`, server value **0.24711** —
and compare. That converts an inference into a measured control: land within
~0.001 and the MC number is defensible as-is; land further out and you have the
correction in hand. It also costs no server time, which is committed to the
benchmark for ~25 days.

---

## 5. What to do next

1. **Sync `scripts/run_queue.py` to the server** so the VIS benchmark's remaining
   87 runs cannot reacquire the §1 defect. The six affected runs need no retrain.
2. ~~**Fix `insert_head_dropout` so it does not renumber**~~ — **DONE 2026-08-31.**
   The final conv of each deployed branch is now an `MCDropoutConv2d`
   (`nn.Conv2d` subclass applying dropout in its own `forward`, armed for eval via
   `mc_armed`), so the module tree — and every state_dict key — is identical to a
   stock model. Verified: keys match a plain model exactly, the training-path
   reload transfers **768/768** tensors (was 756/768), two armed passes differ,
   and disarmed eval is bit-identical. `enable_mc_dropout` still arms the legacy
   inserted-`Dropout2d` form, so `mc_vis_seed0` / `mc_ir_seed0` keep working
   unretrained. `scripts/smoke_mc_e2e.py` gained **check F**, a save→training-reload
   round trip, and was confirmed to *fail* on the old implementation (catching all
   12 tensors). All six gates green.
3. **Re-run the two fine-tunes.** `mc_ir_seed0_ft` (~1 h) is unambiguous — the whole
   IR UQ arm set is laptop-native, so a laptop rerun is exactly matched.
   `mc_vis_seed0_ft` needs care: see §4b.
4. **Verify `mc_vis_seed0/best.pt` gives non-zero epistemic variance.** The
   2026-08-23 rebuild was verified on *a* trained checkpoint, not this final one.
   CPU, minutes.
5. **Decide the IR sweep's remaining ~50 runs** (§2) — they cannot change the
   conclusion.
6. **Leave the VIS benchmark running** (§3) and re-read after `yolo26x`'s three
   seeds land. Budget 20–30 days, not 13–14.
7. Read the two `yolov9c` divergences against the known false-positive pattern.
