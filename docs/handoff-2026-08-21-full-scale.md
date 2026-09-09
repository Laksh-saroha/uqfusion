# Handoff — C-1 full-scale matrix launched and running

**Written 2026-08-21.** Covers the C-1 full-scale training matrix going from plan to live queue, the
first two arms' real timing data, a publication-scope class-mismatch finding (VIS vs Phase 1), and
the state of uncommitted code.

Companion docs: [`TODO-2026-08-20-full-scale.md`](TODO-2026-08-20-full-scale.md) §C-1 (the matrix
spec, A-1/A-2/A-3 decisions), and the plan this session executed (`bright-discovering-umbrella.md`,
under the user's local `.claude/plans/`, not in-repo).

Status labels: **VERIFIED** = measured this session · **RECORDED** = from an existing doc ·
**OPEN** = undecided, needs Laksh.

---

## 1. The matrix — split across two machines 2026-08-22 — **VERIFIED**

The original single 28-row queue at `runs/queue_full_scale/{queue.json,state.json,control.json}` is
**paused and now historical**. On 2026-08-22 the remaining work was split by modality onto two
machines. The server is **not** meaningfully faster than the laptop (measured 4.5% at steady state,
§2.1) — the win is parallelism, not speed:

| | machine | queue dir | rows | runner | dashboard |
|---|---|---|---:|---|---|
| VIS | server `dgxanode01` (A100 MIG 3g.40gb) | `/workspace/uqfusion/runs/queue_vis_server` | 14 | **running** | — (drive via JupyterLab terminal) |
| IR | this laptop (RTX 4080 12GB) | `runs/queue_ir_laptop` | 12 | **running**, PID 10344 | PID 14832, port 8771 |
| — | *historical, paused* | `runs/queue_full_scale` | 28 | stopped | killed 2026-08-22 (was PID 9400 / 8770) |

Results land in `runs/full_scale_server/` on the server and `runs/full_scale/` (+ `runs/mc_dropout/`,
`runs/ensemble/`) on the laptop — see §1.2 and D-8.

The split is safe because **VIS vs IR was never an apples-to-apples comparison** — different data,
`nc=2` vs `nc=1`, `yolo26m` vs `yolo26m-p2feat`. The comparison the paper rests on (gaussian vs
mc_dropout vs ensemble) lives *within* each modality, so each modality must be machine-homogeneous.
Hence two decisions:

- **`gauss_vis_seed0` is retrained on the server**, so all 14 VIS arms share one machine. The
  laptop's copy is kept for a full-length cross-machine comparison of the same arm — stronger
  evidence than any short parity probe.
- **`mc_vis_seed0` restarts from scratch on the server**, not resumed from its laptop epoch-6
  checkpoint; resuming would make one arm half-laptop, half-server and uninterpretable.

Environment is matched where it counts — torch 2.7.1+cu118 / cuDNN 90100, ultralytics 8.4.90, batch
16, imgsz 640, seed 0, `deterministic: True`, and a byte-verified identical dataset. It is **not**
matched on `workers` (server 6 vs laptop 8, forced by the 1 GB `/dev/shm`), Python (3.10.12 vs
3.13.0), numpy (2.2.6 vs 2.1.3), or the GPU. The `workers` difference reshards the augmentation RNG,
so cross-machine runs are not bitwise reproducible. §2.2 measures what that costs.

| id | kind | variant | data | status | epochs | best ep | mAP50-95 |
|---|---|---|---|---|---:|---:|---:|
| `gauss_vis_seed0` | gaussian | `yolo26m` | `data_vis_stride2.yaml` | done | 45 | 25 | 0.24961 |
| `gauss_vis_seed0_ft` | gaussian | `yolo26m` | `data_vis_stride2.yaml` | done | 10 | 1 | 0.24111 |
| `gauss_ir_seed0` | gaussian | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | done | 27 | 7 | 0.11981 |
| `gauss_ir_seed0_ft` | gaussian | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | done | 10 | 8 | **0.14214** |
| `mc_vis_seed0` *(laptop)* | mc_dropout | `yolo26m` | `data_vis_stride2.yaml` | **abandoned** at ep 6, **directory deleted 2026-08-23** (§1.2) | 6 | — | — |
| ~~`mc_ir_seed0`~~ *(1st attempt)* | mc_dropout | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | **INVALID** — trained cleanly but dropout was off the inference path (§1.5, D-14). Dir now `mc_ir_seed0_broken-20260823`; **re-run queued** behind `ens_ir_seed3_ft` | 41 | 21 | ~~0.13948~~ |
| ~~`mc_ir_seed0_ft`~~ *(1st attempt)* | mc_dropout | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | **INVALID** — same defect (§1.5, D-14). Dir now `mc_ir_seed0_ft_broken-20260823`; **re-run queued** | 10 | 9 | ~~0.12871~~ |
| `ens_ir_seed0` | ensemble | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | done (laptop) | 45 | 25 | 0.12986 |
| `ens_ir_seed0_ft` | ensemble | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | done (laptop) | 10 | 6 | 0.12171 |
| `ens_ir_seed1` | ensemble | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | done (laptop) | 52 | 32 | **0.14902** |
| `ens_ir_seed1_ft` | ensemble | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | done (laptop) | 10 | 1 | 0.12516 |
| `ens_ir_seed2` | ensemble | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | done (laptop) | 46 | 26 | 0.12798 |
| `ens_ir_seed2_ft` | ensemble | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | done (laptop) | 10 | 1 | 0.11385 |
| `ens_ir_seed3` | ensemble | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | done (laptop), early-stopped | 31 | 11 | 0.13370 |
| `ens_ir_seed3_ft` | ensemble | `yolo26m-p2feat` | `data_ir_shiponly.yaml` | running (laptop) — queue restarted 2026-08-23 17:26 after the MC smoke, ~95 min | 10 | — | — |
| `gauss_vis_seed0` *(server retrain)* | gaussian | `yolo26m` | `data_vis_stride2.yaml` | done (server), early-stopped | 72 | 52 | 0.25589 |
| `gauss_vis_seed0_ft` *(server)* | gaussian | `yolo26m` | `data_vis_stride2.yaml` | done (server) | 10 | 3 | 0.24759 |
| ~~`mc_vis_seed0`~~ *(server, 1st attempt)* | mc_dropout | `yolo26m` | `data_vis_stride2.yaml` | **failed — diverged at ep 22** (§1.4), stopped manually 04:52. Dir now `mc_vis_seed0_broken-20260823`, state key likewise | 33 | 15 | ~~0.2400~~ |
| ~~`mc_vis_seed0_ft`~~ *(server, 1st attempt)* | mc_dropout | `yolo26m` | `data_vis_stride2.yaml` | **skipped** — parent failed | — | — | — |
| ~~`mc_vis_seed1`~~ *(server)* | mc_dropout | `yolo26m` | `data_vis_stride2.yaml` | **failed — diverged at ep 18**, the seed replication that proved it structural (§1.4). Deliberately **not** re-run: the matrix needs seed 0, and seed 1 has already answered its question | 22 | 15 | ~~0.24604~~ |
| ~~`mc_vis_seed1_ft`~~ *(server)* | mc_dropout | `yolo26m` | `data_vis_stride2.yaml` | **skipped** — parent failed | — | — | — |
| **`mc_vis_seed0`** *(server, rebuilt)* | mc_dropout | `yolo26m` | `data_vis_stride2.yaml` | **running (server) from 2026-08-23 12:26 UTC, PID 18025** — first run on the fixed one2one placement (§1.5). Decision point epoch ~18 | 1+ | — | — |
| **`mc_vis_seed0_ft`** *(server, rebuilt)* | mc_dropout | `yolo26m` | `data_vis_stride2.yaml` | pending, follows the above automatically | — | — | — |

**Epoch counts corrected 2026-08-23.** Every `epochs` value above was previously one too high — read
from `state.json`'s `epochs_done`, which is systematically +1 (§1.6). All six laptop runs and both
completed server runs were re-counted from `results.csv`; `best ep` was already correct everywhere.
The `mAP50-95` column is the final re-validation of `best.pt` (`state.json`'s `map50_95`), the right
field, differing from the best CSV row only in the 5th decimal.

Notable: `gauss_ir_seed0`'s fine-tune stage *improved* on its base (0.11981 → 0.14214) — opposite of
every VIS/IR pair in the Phase 2 laptop test (`handoff-2026-08-19.md` §5, 5-for-5 against).
**`mc_ir_seed0_ft` has now landed and went the other way** (0.13942 → 0.12869), back in line with
Phase 2. So the gaussian-IR result looks like the outlier, not the new pattern. Re-check once the
`ens_ir` fine-tune stages land.

**Every IR arm — gaussian, mc_dropout, and all five ensemble seeds — trains on `yolo26m-p2feat`.**
Confirmed by reading `queue.json` directly, not inferred: `mc_ir_seed0`, `ens_ir_seed{0..4}` all
carry `variant: yolo26m-p2feat`. VIS arms are uniformly plain `yolo26m`. This satisfies D29
(MC-Dropout/Ensemble must inherit every detector-side architecture change).

### 1.1 One unplanned pause this session

Between `gauss_ir_seed0_ft` main training and now, the queue was paused via the dashboard for ~3.4h
(12:15→17:00 08-21) with no explanation recorded beyond `control.json`'s `"source": "dashboard"`. It
was cleared and the runner restarted this session (`run_queue.py --queue-dir runs/queue_full_scale
resume`, then `run`). If this recurs, check whether it's the laptop sleeping or someone hitting Pause
deliberately — the dashboard doesn't currently log *who* paused it.

### 1.2 Where results actually land — **VERIFIED 2026-08-22**

`out_subdir` in `queue.json` (`"full_scale"`) is **not** honoured matrix-wide.
`scripts/run_queue.py:589` passes it only on the `gaussian` branch; the `else` branch calls
`trainer(cfg, **common)` without it, and neither `train_mc_dropout` nor `train_ensemble_member`
accepts the argument — each hardcodes its own root:

| kind | output root | rows |
|---|---|---:|
| gaussian | `runs/full_scale/<id>/` (server: `runs/full_scale_server/<id>/`) | 4 |
| mc_dropout | `runs/mc_dropout/<id>/` | 4 |
| ensemble | `runs/ensemble/<id>/` — **fixed 2026-08-22**, was `runs/benchmark/runs/<id>/` | 20 |

So only **4 of the 28 rows** ever land under `runs/full_scale/` — §1's table lists run *ids*, not
paths. Confirmed on disk: the laptop's `runs/mc_dropout/` holds `mc_ir_seed0`, `mc_ir_seed0_ft`, and
`smoke_mc` alongside `runs/full_scale/gauss_*`; the server's holds `mc_vis_seed0`.

**One name collided, now resolved (2026-08-23).** Because both machines write mc rows to the same
*relative* path, the server's live `mc_dropout/mc_vis_seed0` had the same path as the laptop's
abandoned 6-epoch VIS attempt (2026-08-21 21:32–23:29). A sync back to the laptop would have merged
two different runs under one directory — plausibly `results.csv` from one beside `weights/` from the
other, with no error. The laptop copy was **deleted 2026-08-23** (252 MB; already recorded as
abandoned in §1's table and superseded by the server run). The ensemble rows are safe by luck only:
the server writes `ens_vis_seed*` and the laptop `ens_ir_seed*`, so no name is shared. Re-check this
before any future cross-machine split — see D-6.

### 1.3 The ensemble output dir — **BUG, FIXED 2026-08-22**

**What was wrong.** `ensemble.py`'s `out_root = outputs_root / "ensemble"` was used only for the
member CSVs; the *training run* went to `outputs_root/benchmark/runs/<name>`, because
`train_ensemble_member` reuses `run_grid` and `run_grid` hardcoded its own project dir. All 20
ensemble arms were landing in the **Phase 1 benchmark tree**, mixed with unrelated benchmark runs —
worse than a missing directory, since "collect the matrix" and "collect the Phase 1 grid" became the
same glob.

**Fix, part 1 — the path.** `run_grid` gained an optional `runs_root` (default `None` →
`outputs_root/benchmark/runs`, so **Phase 1 behaviour is unchanged**; `run_benchmark.py` and
`bench/smoke.py` do not pass it), and `ensemble.py` passes `runs_root=out_root`. Members now land in
`runs/ensemble/<name>/`, matching `mc_dropout/`'s layout. Member CSVs stay at
`runs/ensemble/csv/<name>.csv`.

**Fix, part 2 — resuming a run whose directory moved.** Moving the dirs and restarting was not
enough: `model.train(resume=True)` reloads project/name/**save_dir** from the checkpoint's own
`train_args` and ignores anything passed at the call site, so the resumed run re-created its old
directory and trained into it. Caught on the first restart — `args.yaml` and an empty `weights/`
reappeared under `benchmark/runs/ens_ir_seed0/` — and stopped before it checkpointed there.
`run_grid` now calls `_retarget_checkpoint()` before `YOLO(last_ckpt)`, which rewrites `train_args`'s
project/name/save_dir on disk when they disagree with the run's actual location (no-op otherwise). A
general fix: any moved run dir now resumes in place.

**Migration done on the laptop.** Queue paused at an epoch boundary, runner killed, five `ens_*` dirs
moved from `runs/benchmark/runs/` to `runs/ensemble/`, recorded paths in `ensemble_members.csv`
rewritten, runner restarted. `ens_ir_seed0` resumed from epoch 7 at the new path — **no work lost**,
and `runs/benchmark/runs/` now holds zero `ens_*` entries.

**Both machines carry identical source** — LF-normalised sha256 `grid.py` `024ee90e060cd25e`,
`ensemble.py` `5a710db27acde443`.

**Server runner restarted 2026-08-22 05:17 UTC — both machines now run the fix.** `run_queue.py run`
trains in-process and does not fork per row, so an edit on disk is inert until the process restarts;
the server had to be cycled before it reached `ens_vis_seed0` or its five VIS members would have
landed in the benchmark tree regardless of what the source said. Sequence: pause requested 04:54 →
callback fired at the epoch-29 boundary 05:05 (checkpoints written 05:05) → old runner PID 8712 and
its 18 dataloader children killed → `__pycache__` purged → `resume` → relaunched under
`setsid nohup` as PID 12891. The log confirms `RESUME from
.../runs/full_scale_server/gauss_vis_seed0/weights/last.pt` with
`save_dir=/workspace/uqfusion/runs/full_scale_server/gauss_vis_seed0` — **path unchanged, epoch 29
preserved, nothing lost**. Source hashes re-verified on the server after the restart:
`grid.py 024ee90e060cd25e`, `ensemble.py 5a710db27acde443`, matching the laptop. Cost: one partial
epoch, ~17 min. `resume=True` also resets `warmup_bias_lr` 0.1 → 0.0, which is ultralytics' normal
resume behaviour and inert here (warmup ends at epoch 3).

**Nothing downstream breaks.** Fine-tune rows resolve their parent through `state.json`'s recorded
`best_weights` (`run_queue.py:546`), not by path convention, so every `_ft` chain is unaffected;
`status` and the dashboard read state rather than scanning directories. The only cost is that the
matrix's results are split across four trees instead of one (five, counting the server's
`full_scale_server/`) — anything that globs `runs/full_scale/*` to collect the matrix will silently
see the Gaussian arms only.

**Deliberately not fixed mid-matrix.** `gauss_ir_seed0` is already in `full_scale/` and the mc rows in
`mc_dropout/`; changing the dispatch now would move `mc_ir_seed0_ft` out of its own parent's
directory and make the layout inconsistent row-to-row instead of kind-to-kind. One predictable rule
beats a half-migrated tree — see D-6.

### 1.4 `mc_vis_seed0` diverged at epoch 22 — **ROOT CAUSE FOUND, see §1.5**

> **Read §1.5 first.** The symptom is recorded here; the cause is an architecture incompatibility
> between the MC-Dropout placement and YOLO26's end-to-end head. The "optimisation collapse" reading
> below was the working hypothesis on 2026-08-23 morning and is **superseded** — it described the end
> state correctly and the cause incorrectly.

Caught 2026-08-23 04:41 UTC at epoch 33. The classification head blew up over epochs 16–21 and the
model then settled into a degenerate all-background predictor: **eleven consecutive epochs of exactly
zero P, R, mAP50 and mAP50-95**, against a val set that certainly has labels (11,352 images /
112,969 instances).

| epoch | P | R | mAP50 | mAP50-95 | val/cls_loss | val/dfl_loss |
|---:|---:|---:|---:|---:|---:|---:|
| 15 | 0.762 | 0.668 | 0.6440 | **0.2400** | 2.17 | 0.005 |
| 16 | 0.731 | 0.622 | 0.5962 | 0.2211 | 5.03 | 0.005 |
| 18 | 0.505 | 0.565 | 0.5069 | 0.1784 | 14.74 | 0.005 |
| 19 | 0.367 | 0.648 | 0.3579 | 0.1169 | 59.25 | 0.005 |
| 20 | 0.396 | 0.509 | 0.2726 | 0.0924 | 71.85 | 0.005 |
| 22 | 0.000 | 0.000 | 0.0000 | 0.0000 | 3.28 | 0.033 |
| 32 | 0.000 | 0.000 | 0.0000 | 0.0000 | 2.23 | 0.037 |

**It failed silently.** Train losses stayed normal, `val/box_loss` actually *fell* (2.53 → 1.87), and
`runner.log` contains no NaN, no warning, no traceback anywhere in the MC section. Only
`val/cls_loss` and the metric columns show it. A monitor watching loss curves would not have caught
this; a monitor alerting on `mAP50 == 0` would have caught it 4.5 hours earlier.

#### What the model actually did — plain language

The detector fell into the **"predict nothing" solution**, a real and stable local optimum in object
detection, not a crash.

In any detection image, the overwhelming majority of candidate positions are background. A model that
outputs near-zero confidence *everywhere* is therefore **correct about ~99.9% of the positions it is
scored on**, so its classification loss is low — lower, in fact, than the honest model's. That is the
trap: the degenerate answer looks good to the loss function. It scores zero on mAP because mAP only
counts detections, and there are none.

Once in that state it cannot climb back out. It makes no confident positive predictions, so the
assigner has no good candidates to match to ground-truth boxes, so there is no gradient pulling it
back toward detecting things. That is why the eleven epochs are **exactly** `0.0000` rather than a
slow recovery, and why `val/box_loss` and `val/dfl_loss` change character at the same moment (2.53 →
1.87 and 0.005 → 0.04): after the collapse those two losses are measured over a degenerate set of
assignments, so their values are no longer comparable to the healthy epochs at all.

#### What pushed it in — established vs inferred

**Established by the data:**

- An optimisation collapse, **not** a numerical failure. No NaN, no inf, no overflow anywhere in the
  log; every loss stayed finite.
- The run-up is specifically in the **classification** loss: `val/cls_loss` goes 2.17 → 5.03 → 14.74
  → 59.25 → 71.85 across epochs 15–20, then the model lands in the background basin at epoch 22. Box
  and DFL losses show no such run-up beforehand.
- **The recipe is not broken in general.** `mc_ir_seed0` ran 41 epochs with the same dropout insertion
  and never collapsed.
- **The data and LR schedule are not sufficient to cause it.** `gauss_vis_seed0` trained 72 clean
  epochs on the same VIS data with the same schedule.

**Inferred, and explicitly not proven:** the MC-Dropout modification is the prime suspect, the only
structural difference from the Gaussian arm that survived. `insert_head_dropout` (`uq/mc_dropout.py`)
puts a **`Dropout2d(0.15)` immediately before the final 1x1 conv of each head branch**. `Dropout2d`
zeroes *entire feature-map channels*, not individual activations, and that final conv maps its input
channels down to just **two** class logits — so removing 15% of the channels can swing those logits
hard. At epochs 16–21 the LR is still near its peak (~0.025 of a 0.029 maximum), so a handful of
unlucky steps is enough to throw the head somewhere it cannot recover from.

**What does *not* cleanly explain it.** The obvious VIS-vs-IR difference is that VIS has two classes
and IR has one, so only VIS has a class-imbalance axis for the cls loss to blow up along. But the
imbalance is **13:1 (ship 93.0% / 105,098 instances, buoy 7.0% / 7,871, measured on the val label
cache)** — moderate, not the extreme rarity that usually triggers this failure. So class imbalance is
a contributing suspect, not a smoking gun. The other live differences (batch 16 vs 10, `yolo26m` vs
`yolo26m-p2feat`) are untested.

**The experiment that settles it** is `mc_vis_seed1`: same recipe, data, LR, only the seed changed. If
it trains cleanly, epoch 22 was bad luck and the arm is simply re-run. If it collapses too, the
instability is structural, and the fix (lower `p`, lower `lr0`, or gradient clipping) must be applied
to **both** modalities so the matrix stays internally comparable. First meaningful checkpoint ~epoch
16.

#### What was done about it, 2026-08-23 04:52–04:57 UTC

`best.pt` (epoch 15) was never at risk — only `last.pt` carried the dead weights. Rather than let
patience run the remaining two epochs to 35:

1. Runner PID 12891 killed (`kill -TERM -12891`, whole process group; 18 dataloader children went
   with it). The kill landed mid-epoch-34, so `results.csv` holds 33 completed epochs.
2. **`mc_vis_seed0` marked `failed`, deliberately not `done`** — `failed` is terminal so the runner
   skips it, *and* anything that collects on `status == "done"` now excludes the collapsed model
   automatically instead of silently tabling a 15-epoch detector. The `error` field carries the
   reason; `best.pt` and `run_dir` are retained for forensics.
3. `mc_vis_seed0_ft` was **not** run. The runner's own parent guard skipped it (`parent
   'mc_vis_seed0' did not produce weights`) — no `control.json` edit needed. The MC-Dropout VIS arm
   therefore currently has **no fine-tune stage from either seed**.
4. `mc_vis_seed1` + `mc_vis_seed1_ft` inserted into `queue.json` after `mc_vis_seed0_ft` (backup:
   `queue.json.bak-20260823`), cloned from the seed-0 specs with `seed: 1` and nothing else changed.
   Verified first that `seed` actually reaches ultralytics (`uq/mc_dropout.py:132`) — otherwise a
   "different seed" run would have been bit-identical and would have diverged identically.
5. Runner relaunched under `setsid nohup` as **PID 14668**. Log confirms `queue: 16 run(s)`, the
   three skips, and `save_dir=.../runs/mc_dropout/mc_vis_seed1`, `seed=1`, `resume=False`.

**Note on restart mechanics:** `scripts/run_queue.py:512` builds its spec list **once** at startup,
so editing `queue.json` while the runner is live has no effect. Any queue change requires a runner
restart — see D-10 for the zero-loss pause-at-epoch-boundary variant when there is no reason to
discard work.

**Why it matters for the paper.** A 15-epoch collapsed model cannot sit in a table beside a Gaussian
arm that got 52 epochs. See D-11.

### 1.5 ROOT CAUSE: MC-Dropout was never on the inference path — **CONFIRMED 2026-08-23**

`insert_head_dropout` inserts `Dropout2d(0.15)` before the final 1x1 conv of `cv2` and `cv3`.
**YOLO26's `Detect` head is end-to-end** (`head.end2end is True`, `reg_max=1`) and carries two
parallel head sets. The inserted dropout landed entirely on the branch discarded at inference:

```
branch cv2            dropout2d=3     <- one2many: trains the loss, discarded at inference
branch cv3            dropout2d=3     <- one2many
branch one2one_cv2    dropout2d=0     <- what val() and predict() actually execute
branch one2one_cv3    dropout2d=0
dfl                   dropout2d=0
```

**Proof, not inference.** With `nn.Dropout2d.forward` patched at class level and a call counter
attached, a full 11,352-image validation reported **`forward_calls: 0`** — the dropout was never
executed once in 710 batches.

#### Three consequences

1. **The MC-Dropout arm cannot produce uncertainty at all.** `MCDropoutPredictor` runs `T` stochastic
   passes, but inference decodes from `preds["one2one"]`, which has no dropout. Every pass is
   identical and epistemic variance is **exactly zero**. Worse, `enable_mc_dropout` counts the 6
   layers, returns happily and raises nothing — its "no dropout layers found" guard gives false
   comfort precisely when the layers are on the wrong branch.
2. **`mc_ir_seed0` and `mc_ir_seed0_ft` are invalid too.** They trained cleanly for 41 and 10 epochs,
   but they are the same no-op: a mildly regularised deterministic detector, not MC-Dropout. Their
   apparent success only means the one2many noise did not happen to destabilise that model. **Both
   must be re-run**, and neither may be reported as an MC-Dropout row.
3. **It explains the VIS divergence.** Dropout perturbs one2many's gradients, which shape the shared
   backbone/neck. `one2one` sits on that trunk *without* the dropout, and (ultralytics feeds it
   detached features) without pushing gradients back into the trunk. So the branch that is scored
   degrades while the training loss — dominated by the branch that has the dropout and is adapted to
   it — stays flat. Exactly the observed signature: `train/cls_loss` flat at 0.42 → 0.40 while
   `val/cls_loss` goes 2.17 → 71.85, at the same epoch under two independent seeds.

#### Does this affect the Gaussian or Ensemble arms? **No, and here is why**

| arm | head surgery? | end2end-aware? | verdict |
|---|---|---|---|
| gaussian | yes — adds `cv4` σ² branch | **yes, explicitly** | unaffected |
| ensemble | **none** | n/a | unaffected |
| mc_dropout | yes — inserts `Dropout2d` | **no** | broken |

- **Gaussian is correct by construction.** `uq/gaussian.py:_sigma_box_head` returns
  `self.one2one_cv2 if self.end2end else self.cv2`, and the module docstring spells out all three
  end2end differences (σ rides one2one, `postprocess` overridden for the top-k gather, μ from the raw
  box output at `reg_max=1`). It even warns about this exact trap in advance: *"σ attached to
  one2many would be calibrated against a box predictor that never reaches the output —
  plausible-looking and wrong."* There is a dedicated gate, `scripts/smoke_gaussian_e2e.py`, which
  asserts `head.end2end` and that `cv4` shadows one2one. `mc_dropout.py` simply never got that
  treatment — its docstring predates the end2end knowledge.
- **Ensemble does no head surgery at all.** `uq/ensemble.py` contains no reference to
  `cv2`/`cv3`/`cv4`/`one2one` — members are stock YOLO26m runs, and the uncertainty comes from
  variation *across* independently seeded models, not stochasticity inside one.
- `uq/variants.py` also handles the one2one twins for weight transfer
  (`^((?:one2one_)?cv[23])\.(\d+)\.`), so the variant seeding path is fine.

So the blast radius is the MC-Dropout arm only: **4 of the 28 matrix rows** (`mc_vis_seed0(+_ft)`,
`mc_ir_seed0(+_ft)`). The other 24 stand.

#### The IR arms carry the identical defect — **confirmed from the checkpoints 2026-08-23**

The IR runs finished cleanly and never diverged, which made them look fine. They are not. Loading
both IR checkpoints and counting `Dropout2d` per branch:

```
runs/mc_dropout/mc_ir_seed0/weights/best.pt      (and mc_ir_seed0_ft)
  head: Detect  end2end = True  nl = 3  nc = 1  reg_max = 1
    cv2            dropout2d = 3     <- discarded one2many branch
    cv3            dropout2d = 3     <- discarded one2many branch
    one2one_cv2    dropout2d = 0     <- what inference actually decodes
    one2one_cv3    dropout2d = 0
```

`yolo26m-p2feat` changes the neck, not the head — its own header says *"Detect stays 3-level
(P3,P4,P5) -> anchor count and head widths unchanged"* — so it inherits a stock end2end `Detect` and
the identical misplacement. Both IR MC arms therefore produce **exactly zero epistemic variance**,
same as VIS. D-14 stands: they must be re-run, and until then they are INVALID rather than merely
untidy.

#### Why IR did not *diverge* — LR ruled out, `nc` is the leading suspect

Two different questions, and only the second is open. IR **was** hit by the placement bug (above).
What IR escaped was the training collapse.

**Ruled out — the learning-rate schedule.** The obvious confound was that the IR runs sat at a
different point on the schedule during the critical window. They do not; the curves are identical,
not merely similar:

| epoch | 18 | 19 | 20 |
|---|---|---|---|
| VIS `lr/pg0` | 0.02495 | 0.02465 | 0.02436 |
| IR `lr/pg0` | 0.02495 | 0.02465 | 0.02436 |

Same `optimizer: auto`, `lr0: 0.01`, `lrf: 0.01`, `momentum: 0.937`, `nbs: 64`, `warmup_epochs: 3.0`,
`cos_lr: false`. Batch differs (10 vs 16) but `nbs` accumulation absorbs most of it — effective batch
~60 vs 64.

**Ruled out — a milder version of the same failure.** Against a no-dropout ensemble control on the
same modality, IR's dropout run is indistinguishable from its own control; it is not a sub-threshold
case of the VIS blow-up:

| run | max/min `val/cls_loss` |
|---|---|
| `mc_ir_seed0` (dropout) | **1.44** |
| `ens_ir_seed0` (no dropout, control) | 1.37 |
| `mc_vis_seed0` (dropout) | **~33** |

**Leading explanation, inferred and not proven: `nc=1` vs `nc=2`.** The failure was specifically in
the *classification* loss. With a single class the cls head has no inter-class decision boundary to
lose — its output degenerates to an objectness score, and trunk noise has nothing to destabilise. VIS
must hold a ship-vs-buoy boundary, and that is what exploded.

This also reconciles the per-class detail that looked contradictory in §1.4. Mid-collapse the
*majority* class fell hardest (ship 0.292 → 0.084, buoy 0.200 → 0.163). If a two-class boundary
collapses and prediction mass drifts onto one label, the class with the most to lose is the 93%
majority — so ship cratering while buoy holds is what a boundary collapse looks like, not evidence
against one.

Untested alternatives, in rough order of plausibility: batch 10 vs 16, the p2feat neck, dataset
content. **This does not need settling before the re-runs.** Under option A the dropout moves onto
`one2one`, which Ultralytics feeds *detached* features, so it contributes no gradient to the shared
trunk and the destabilisation mechanism is gone for both modalities regardless of class count. The
`nc=2` hypothesis only becomes actionable if VIS destabilises *again* after the move — in which case
it argues for a lower `p` on VIS, or option D. Watch epoch ~18 of the VIS re-run.

#### Placement options for the redesign

All four require a **retrain** — dropout must be present during training; applying it only at
inference is not MC-Dropout (the module docstring already rejects that, and it would just degrade a
model that never saw it).

| # | placement | pros | cons |
|---|---|---|---|
| **A** *(recommended)* | `Dropout2d` before the final conv of **`one2one_cv2` / `one2one_cv3`** | Mirrors the Gaussian fix exactly; smallest deviation from B6-3 (same rule, right branch); one2one reads **detached** features so the noise **cannot destabilise the trunk** — structurally immune to the failure we just hit; gives genuine MC variance at inference | one2one is thinner and `reg_max=1`, so channel-dropout may bite harder per unit; `p` may need lowering |
| B | dropout on **both** one2many and one2one | Closest to a literal "every head branch" reading of B6-3 | Reintroduces the exact trunk destabilisation that killed two runs. **Do not.** |
| C | dropout in the shared neck (P3/P4/P5 feature maps before `Detect`) | Standard MC-Dropout practice for detectors; automatically on the inference path (detach blocks gradients, not forward values) | Changes what the uncertainty *means* (feature-level, not head-level); larger deviation from B6-3; still perturbs the one2many gradient path |
| D | element-wise `nn.Dropout` instead of `Dropout2d`, on one2one | Gentler than zeroing whole channels immediately before a 2-logit conv; closer to Gal & Ghahramani's original | Further from B6-3's literal wording; combine with A rather than treat as separate |

**CHOSEN: A** (2026-08-23), with D as the fallback if A at `p=0.15` proves unstable. Sequence before
spending 30+ hours: (i) write the gate first (D-15), (ii) a 10-epoch smoke on both modalities, (iii)
only then commit the full runs.

**Cost:** ~20 h VIS + ~12 h IR = **~32 h**, queued behind ~4.5 days of ensembles unless deliberately
prioritised. A pre-registered-protocol deviation (plan B6-3 specified `cv2`/`cv3`) and must be written
up as one: B6-3 was authored for a conventional detect head and does not transfer to YOLO26's
dual-head design.

#### Implemented 2026-08-23 — gate first, then the fix

**`scripts/smoke_mc_e2e.py`** (new, D-15) — the missing counterpart to `smoke_gaussian_e2e.py`. CPU,
~5 s, no dataset, no GPU. Five checks:

| | check | asserts |
|---|---|---|
| A | placement | dropout on `one2one_cv2`/`one2one_cv3`, and **zero** on `cv2`/`cv3` |
| B | execution | each layer invoked, **in training mode**, and actually perturbing its input |
| C | stochastic | two armed passes of the whole model produce different boxes |
| D | determinism | *without* arming, eval is bit-identical, so the deterministic row stays reproducible |
| E | guard | `enable_mc_dropout` **raises** on a model with dropout only on the discarded branch |

**Red before green, deliberately.** Run against the unfixed module the gate failed on check A
(`expected 6 Dropout2d on one2one_cv2/one2one_cv3, found 0`). A gate that cannot fail on the bug it
was written for is decoration.

**`uq/mc_dropout.py`** (fixed) — new `deployed_head_branches(head)` returns
`("one2one_cv2", "one2one_cv3")` when `head.end2end` else `("cv2", "cv3")`; `insert_head_dropout`
iterates it (still idempotent, so the resume path is unchanged); `enable_mc_dropout` now raises when
no armed layer sits on the deployed branch, naming the branch in the message. Only `smoke_mc_e2e.py`
references these functions outside the module, so nothing else needed touching.

**Two traps found while building the gate — both would have made it pass on broken code:**

1. **A yaml-built model makes check C pass vacuously.** Random-init YOLO26n has head activations of
   order **1e-6**, so zeroing whole channels moves the decoded boxes by less than float32 resolution:
   measured max |Δ| between two armed passes was **0.0**, identical to the broken case. With
   pretrained `yolo26n.pt` the same check gives **272**. The gate now requires pretrained weights and
   says why, so nobody later "optimises away" the download.
2. **A call counter proves invocation, not activity.** An eval-mode `Dropout2d` *is* invoked and
   returns its input unchanged. Check B originally counted calls only; it now also asserts each
   layer's `training` flag and a non-zero in/out delta. Caught by check C failing while check B
   passed — the checks covering each other is the only reason it surfaced.

#### 10-epoch smoke on real data, both modalities — 2026-08-23 (D-13 step a)

`scripts/smoke_mc_train.py` (new). The module gate proves plumbing on an untrained model in seconds;
it cannot prove that a checkpoint produced by the *training loop* yields non-zero variance through
`MCDropoutPredictor`. This does. Per modality it carves an evenly-strided 2000-train / 400-val subset
out of the production lists (real images, real labels, production model and batch — only image and
epoch counts reduced), trains 10 epochs through `train_mc_dropout`, then loads `best.pt` through
`MCDropoutPredictor` and asserts the T passes disagree.

| modality | epochs | final mAP50 | final mAP50-95 | `val/cls_loss` | armed | MC spread (`sigma_ltrb`) |
|---|---|---|---|---|---|---|
| VIS (`yolo26m`, b16) | 10/10 | 0.5219 | 0.1844 | 2.77 → 2.29 | 6 | mean **2.52** px, max 27.6 (98 dets) |
| IR (`yolo26m-p2feat`, b10) | 10/10 | 0.1688 | 0.0552 | 3.82 → 2.36 | 6 | mean **4.81** px, max 37.9 (19 dets) |
| IR — crashed first attempt † | 9/10 | 0.164 | 0.0553 | — | 6 | mean 18.33 px, max 34.1 (7 dets) |

Both non-zero, on real trained checkpoints, in both modalities — the property the whole redesign
exists to restore. `val/cls_loss` falls rather than rises, though 10 epochs says nothing about
epoch-18 stability (the script docstring notes this checks *plumbing and variance*, not long-run
stability; the first real evidence on that is epoch ~20 of the full re-run).

**† The first IR attempt crashed at epoch 9 — environment, not code.** Ultralytics' `save_metrics`
failed with `PermissionError: [Errno 13]` opening its own `results.csv` for append; the file was not
locked a minute later, and the same code had written eight epochs before it. A transient Windows lock
(indexer / real-time scan), not a defect in the fix. Kept at
`runs/mc_dropout/smoke_mc_ir_crashed_ep9/` and listed above because it is a second, independent
trained checkpoint that also gave non-zero spread — the row that matters is the clean 10/10 re-run.
**Worth noting beyond this smoke:** the identical failure at epoch 60 of a production run would abort
it. `train_mc_dropout` resumes from `last.pt` automatically, so re-queueing the spec recovers the work
— but the queue must not mark such a run permanently `failed`. See D-16.

**Two bugs in the smoke script itself, both found by running it:**

1. **`ModuleNotFoundError: No module named 'uqfusion'`.** The repo is not pip-installed in the GPU
   interpreter (`gpu_python`), only in `.venv` — which carries CPU torch and so cannot run this. Fixed
   by copying `run_queue.py`'s `sys.path` bootstrap.
2. **Every val image reported "corrupt image/label".** The production lists mix path conventions: the
   train lists are absolute, but `Pohang_dataset/visible/val.txt` is `./images/...` relative — and
   Ultralytics resolves relative entries against *the list file's own parent*, so copying them
   verbatim into a new subset directory silently repoints every one of them. `make_subset` now
   resolves to absolute and asserts the first 20 exist. A trap for any future script that subsets
   these lists.

#### Server sync and the MC re-runs — 2026-08-23 (D-13 steps b and c)

**The fix is on both machines and byte-identical.** `mc_dropout.py`, `smoke_mc_e2e.py` and
`smoke_mc_train.py` were tarred, uploaded through the JupyterLab Contents API and extracted into
`/workspace/uqfusion`. md5 parity:

```
59d0b8ecafcab900dd6cbc667140d9e1  src/uqfusion/uq/mc_dropout.py
3388858bf0c6fe916e1ea94b6db45ea8  scripts/smoke_mc_e2e.py
35ae69830fae315d48824238f3e3bed7  scripts/smoke_mc_train.py
```

The old server copy is kept at `src/uqfusion/uq/mc_dropout.py.bak-20260823` (md5
`3b298372e35c74e2937593e56b7bd206`). The gate then ran **on the server** and passed all five checks
with numbers matching the laptop (stochasticity max |Δ| 271.09 against output scale 114.68), so this
is verified there, not assumed from a matching hash.

**Path note.** On `dgxanode01` the JupyterLab contents root is `/workspace` and the repo is
`/workspace/uqfusion`; the venv is `/opt/venv_match`. The `/workspace/Saroha_Work` path in the older
notes belongs to the other server.

**Three traps that make a naive re-run silently do nothing — or worse.**

1. **`specs` and `state` are both read once, at runner startup** (`run_queue.py:505-512`), and
   `save_state` rewrites the *whole* dict. So editing `state.json` while a runner is live gets
   clobbered at its next save, and queue edits do not reach a running process at all. Any change
   needs a runner restart.
2. **A live runner holds the OLD module in memory.** Overwriting `mc_dropout.py` on disk does not
   affect the process that already imported it — so a runner started before the sync would still
   misplace dropout on any MC arm it reached, with no error. The new guard cannot save this case
   either: it lives in `enable_mc_dropout`, which training never calls. **The restart is the fix, not
   the sync.**
3. **`train_mc_dropout` resumes from `last.pt` before anything else.** Re-queueing a spec whose old
   output dir still exists silently *continues the broken run* rather than starting over. The old dirs
   must be moved aside first. Done 2026-08-23, kept as evidence for §1.5:
   `runs/mc_dropout/mc_vis_seed{0,1}_broken-20260823` (server) and
   `runs/mc_dropout/mc_ir_seed0{,_ft}_broken-20260823` (laptop).

**No `queue.json` edit was needed on either machine.** The MC specs already exist; they were merely
`failed` (VIS) or `done` (IR). Two mechanisms cover the re-run:

| machine | mechanism | why |
|---|---|---|
| server | reset `mc_vis_seed0`/`_ft` to `pending` in `state.json`, restart the runner normally | the MC specs already sit *ahead of* the ensembles in queue order, so the runner does MC first and then flows on into `ens_vis_seed0` (resuming it from `last.pt`) with no second launch |
| laptop | `run --only mc_ir_seed0 mc_ir_seed0_ft --redo` | the IR queue has nothing else left, so a filtered one-shot is simpler than state surgery. `--redo` overrides the `done` status; the parent guard passes because `mc_ir_seed0` completes earlier in the same invocation |

`mc_vis_seed1`/`_ft` are deliberately **left `failed`**. Seed 1 was only ever the replication test for
the divergence; the matrix needs seed 0. Re-running it would cost another ~20 h to re-answer a
question already answered.

**Order, decided 2026-08-23:** MC ahead of the remaining ensembles. The one open risk in the redesign
is whether option A is *stable*, and that shows at epoch ~18 — about 4 h in. Learning it now rather
than in ~5 days is worth pushing `ens_vis_seed1-4` back by roughly a day; the cost is that D-9's
seed-noise estimate slips with them. `ens_vis_seed0` was paused at the **epoch 4/100** checkpoint and
resumes from `last.pt`, so nothing is lost beyond the wait.

**Placement verified on the production model, not just the gate's stand-in.** The gate runs
`yolo26n`; the arm trains `yolo26m`. Building the real one on the server and counting per branch:

```
yolo26m + insert_head_dropout(p=0.15):  end2end True  nl 3
  {'cv2': 0, 'cv3': 0, 'one2one_cv2': 3, 'one2one_cv3': 3}
```

Exactly inverted from the broken checkpoints. **Still to confirm at the first checkpoint:** load
`mc_vis_seed0/weights/last.pt` and re-count — an in-memory build is not proof that what the trainer
*saved* carries the same placement, which is precisely the class of assumption that produced this
defect.

#### Method note: three attempts to arm dropout, two silently wrong

Worth recording, because each failure mode is a trap for the next person:

| attempt | method | control result | why it failed |
|---|---|---|---|
| 1 | patch `m.forward` on each instance | armed == unarmed, **bit-identical** | `val()` never used the patched objects |
| 2 | swap in an `nn.Dropout2d` subclass | `swapped=6`, **`alive_after_val=0`** | `val()` rebuilds the model from the checkpoint **path** via `AutoBackend` and discards in-memory edits |
| 3 | patch `nn.Dropout2d.forward` on the **class** + call counter | **`forward_calls=0`** | worked correctly — and revealed the layers are never executed at all |

Attempts 1 and 2 would each have been reported as "dropout makes no difference" if the controls had
been omitted. **Any future eval-time model surgery in this repo must carry a did-it-actually-run
counter**, not just a "did I find the layers" count.

#### Harness validation

Every diagnostic number reproduced a known `results.csv` value, which is why the above is
trustworthy:

| checkpoint | diagnostic mAP50-95 | `results.csv` says |
|---|---:|---:|
| `mc_vis_seed0/best.pt` | 0.24095 | 0.2400 (ep 15) |
| `mc_vis_seed0/last.pt` | 0.00000 | 0.0000 (ep 33) |
| `mc_vis_seed1/best.pt` | 0.24604 | 0.2446 (ep 13) |
| `mc_vis_seed1/last.pt` | 0.12357 | 0.1240 (ep 20) |

**Also: the raw training weights are unrecoverable.** Ultralytics 8.4.90 saves `model: None` in every
checkpoint and keeps only the EMA — by design ("resume and final checkpoints derive from EMA"). There
is no non-EMA copy of these runs to inspect, so any hypothesis requiring the raw weights is untestable
after the fact.

### 1.6 `state.json`'s `best_map50_95` is off by one — **do not read it**

Verified against `results.csv` on all three runs that have reached a best epoch. The queue's summary
field lands on the row *after* the best one, and `epochs_done` is one higher than the number of CSV
rows:

| run | `results.csv` truth | `state.json` `best_map50_95` | `epochs_done` vs rows |
|---|---|---|---|
| `gauss_vis_seed0` | ep52 = 0.25591 (max), ep53 = 0.25552 | 0.25552 = **ep53** | 73 vs 72 |
| `gauss_vis_seed0_ft` | ep3 = 0.2475 (max), ep4 = 0.22642 | 0.22642 = **ep4** | 11 vs 10 |
| `mc_vis_seed0` | ep15 = 0.2400 (max), ep16 = 0.2211 | 0.22106 = **ep16** | — (running) |

`best_epoch` and `best_fitness` are **correct**. `map50_95` is also correct and is the field to report
— it is ultralytics' final re-validation of `best.pt` after early stopping (0.25589 for
`gauss_vis_seed0`, which is ep52 revalidated, not a CSV row). Only `best_map50_95` is wrong. D-9's
ensemble spread must come from `results.csv` or from `map50_95`, never from `best_map50_95`. See D-12.

---

## 2. Timing — measured, not projected, for the first VIS and IR arms — **VERIFIED**

| arm | main | ft | total compute | epochs (main+ft) |
|---|---:|---:|---:|---:|
| `gauss_vis_seed0` | 13.71h | 2.75h | 16.46h | 57 |
| `gauss_ir_seed0` | 4.30h* | 1.48h | 5.78h | 39 |

\* excludes the 3.4h pause gap in §1.1 — pure compute time, back-calculated from epoch throughput
either side of the pause.

IR ran faster than the VIS-proxy projection made earlier this session (5.78h actual vs ~8.9h
projected) — both because its main stage stopped earlier under patience=20 (28 epochs vs VIS's 46)
and because per-epoch cost is genuinely ~52% of VIS's (matches the ~48% dataset-size ratio: 23,279 IR
train images vs 48,136 VIS).

**Updated remaining-time estimate**, using these two real arms as the proxy (VIS arm ≈ 16.5h, IR arm
≈ 5.8h) for the 12 arms not yet complete (`mc_vis` in progress, `mc_ir`, `ens_vis_seed0-4`,
`ens_ir_seed0-4`):

- 6 remaining VIS-family arms × 16.5h ≈ 99h
- 6 remaining IR-family arms × 5.8h ≈ 35h
- **≈ 134h ≈ 5.6 days of continuous compute from now (2026-08-21 ~23:20)**, landing at roughly
  **2026-08-27**, if nothing pauses and every arm's epoch-to-convergence keeps tracking gauss's
  pattern.

### 2.1 Revised for the two-machine split — **VERIFIED 2026-08-22**

The ~134h figure assumed one machine running all 12 remaining arms in series. With the split, the two
families run concurrently.

**Re-derived 2026-08-22 from steady-state epoch times, not probes** (D-4 closed). Epoch 1 is ~30–40%
slower than steady state on both machines and must be excluded; the laptop's IR epoch 1 was further
inflated to 796s by the VIS archive build competing for the A: drive.

| | steady-state s/epoch | epochs to early-stop | per arm (main + `_ft`) |
|---|---:|---:|---:|
| laptop, IR (`yolo26m-p2feat`, batch 10, workers 8) | **550** | 42 + 11 (`mc_ir_seed0`, observed) | ~8.1h |
| server, VIS (`yolo26m`, batch 16, workers 6) | **1013** | ~45 + ~11 (from the laptop `gauss_vis` run) | ~15.8h |

| | arms remaining | total |
|---|---:|---:|
| laptop, IR | 5 (`ens_ir_seed0-4`) | **~40h** |
| server, VIS | 7 (`gauss_vis` retrain, `mc_vis`, `ens_vis_seed0-4`) | **~110h** |

**The server is the critical path at ~110h (~4.6 days)**, and the laptop finishes IR well inside that.

**Correction to the earlier speed claims — both previous numbers were wrong.** The 1.23× figure came
from a stripped probe (no σ² head, no mosaic) and was optimistic. The "server is ~7% slower" figure
that replaced it was read off epoch 1 and was pessimistic. At steady state on the real job the server
does **1013 s/epoch vs the laptop's 1058–1065 s — about 4.5% faster.** The split is worth keeping;
there is no case for swapping modalities between machines.

Same caveat as before on the epoch budget: `mc_ir_seed0` early-stopped at 42 epochs, close enough to
Gaussian's 45 that the "similar budget" assumption holds for MC-Dropout on IR. Ensemble members remain
unconfirmed, and this still assumes no further multi-hour pauses like §1.1.

### 2.2 D-7 result: the same arm on both machines — **ANSWERED 2026-08-22, with a caveat**

`gauss_vis_seed0` has now run on both machines against a byte-verified identical dataset. Laptop:
`runs/full_scale/gauss_vis_seed0`, 45 epochs, plateau ~0.2470. Server:
`runs/full_scale_server/gauss_vis_seed0`, 26 epochs so far, 0.25345 and still climbing.

Server minus laptop, mAP50-95 at matched epochs:

| epochs | deltas |
|---|---|
| 1-8 | -0.012, +0.008, **-0.036**, +0.018, +0.011, +0.021, +0.018, +0.001 |
| 9-16 | -0.016, -0.020, +0.000, -0.007, +0.010, +0.002, -0.002, -0.000 |
| 17-26 | +0.004, +0.004, +0.006, +0.006, +0.006, +0.006, +0.005, +0.004, +0.004, +0.004 |

Early epochs scatter in both directions, exactly as the "judge drift, not magnitude" rule
anticipated. But the **last ten epochs are all positive, mean +0.0049, sd 0.0009** — roughly +2%
relative. One-directional drift, which is the signal D-7 was written to catch. The server run is
genuinely sitting above the laptop run, not wobbling around it.

**Environment parity, checked directly on both interpreters:**

| | laptop | server |
|---|---|---|
| torch / CUDA / cuDNN | 2.7.1+cu118 / 11.8 / 90100 | **identical** |
| ultralytics | 8.4.90 | **identical** |
| dataset | image sha + normalised label sha | **identical** (CRLF-only diff, benign) |
| python | 3.13.0 | 3.10.12 |
| numpy | 2.1.3 | 2.2.6 |
| `workers` | 8 | **6** (forced by the 1 GB `/dev/shm`) |
| mp sharing strategy | `file_descriptor` | `file_system` (`.pth` shim) |
| GPU | RTX 4080 Laptop | A100 MIG 3g.40gb |

`workers 6` vs `8` reshards the augmentation RNG (`base_seed + worker_id`, draws taken inside
`__getitem__`), so the two runs are **not bitwise reproducible** — they are independent draws from the
same recipe. Different SM count and AMP kernel selection compound that. The sharing strategy affects
IPC transport only, not numerics.

**Why this is survivable:** every comparison the paper makes is same-machine. All 14 VIS arms run on
the server, all IR arms on the laptop, and VIS-vs-IR was never apples-to-apples anyway (`nc=2` vs
`nc=1`, different backbone, different imagery). The one cross-machine artifact is the laptop's
`runs/full_scale/gauss_vis_seed0`, which the server retrain exists to supersede. **Do not mix the two
in any table.**

**What is still unknown:** whether +0.005 sits inside seed noise. There are no seed replicates
anywhere in `runs/` at full scale, so the offset cannot be bounded today. The five VIS ensemble seeds
queued on the server will supply exactly that denominator, measured on the machine the offset appeared
on — see D-9.

---

## 3. Publication-scope finding: VIS class mismatch vs Phase 1 — **VERIFIED, OPEN mitigation not yet checked**

Raised mid-session: "we benchmarked ships only, now we're training ships + buoys — does that break the
publication?"

**The mismatch is real but asymmetric:**

- Phase 1's entire benchmark suite (69-run pilot grid + the 27-row `main` fingerprint `682dbe9f0f05`)
  ran ship-only, single-class (`--classes 0` via `run_benchmark.py`) — confirmed in
  `phase1-experimental-record.md`, `phase1_benchmark/README.md`.
- **IR is scope-matched.** A-1 already resolved IR to `nc=1` ship-only (`data_ir_shiponly.yaml`), by a
  pre-registered rule (`TODO-2026-08-20-full-scale.md` lines 57–94).
- **VIS is not scope-matched.** `data_vis_stride2.yaml` declares `names: {0: ship, 1: buoy}` — genuine
  2-class detection — and `scripts/run_queue.py` has no class-filtering mechanism anywhere (checked
  via grep, zero matches), unlike `run_benchmark.py`'s `--classes` flag.

**Mitigation already built into the codebase, not yet exercised downstream:**
`src/uqfusion/eval/apmetrics.py` computes a `per_class` dict in both its reference and
fast/bootstrap AP paths (its own docstring: "mAP here is macro-averaged over ship and buoy, so a class
[problem] needs to be read per class before it is believed"), and `TODO-2026-08-20-full-scale.md` line
262 already commits to "report per-class AP everywhere." So the fix is a discipline requirement, not a
retrain: any comparison of the finished VIS checkpoint against Phase 1 numbers must use the model's
**ship-class AP** specifically, sliced via `per_class`, never its pooled 2-class mAP.

**Not yet verified — the actual next step:** whether the downstream comparison scripts that will run
against these checkpoints (corruption ladder, Mahalanobis refit, calibration, risk-coverage) already
do this class-slicing, or need it added. Candidates worth checking first, going by filename: the
untracked `scripts/eval_risk_coverage_fixed_gt.py` and `scripts/refit_maha_dayonly.py` (see §5) —
unread this session, purpose inferred from name only.

**Also flagged, unmeasured:** buoy AP has been ~0.0002 everywhere measured on IR. Worth checking VIS
buoy AP once `gauss_vis_seed0` is fully evaluable — if similarly near-zero, headline 2-class mAP would
understate ship performance, another reason to always lead with ship AP.

---

## 4. How to monitor / control the queues — **updated 2026-08-22**

Two live-ish queues; both scripts default to `runs/queue/` (the Phase 2 dir), so **`--queue-dir` is
always required** or they will operate on the wrong (or nonexistent) state.

**Laptop, IR family — running now** (runner PID 10344, dashboard PID 14832):

```bash
python scripts/run_queue.py --queue-dir runs/queue_ir_laptop status
python scripts/run_queue.py --queue-dir runs/queue_ir_laptop pause
python scripts/run_queue.py --queue-dir runs/queue_ir_laptop resume
python scripts/run_queue.py --queue-dir runs/queue_ir_laptop run     # resumes from last.pt
python scripts/dashboard.py --queue-dir runs/queue_ir_laptop --port 8771 --open
```

**Server, VIS family** — same commands with the server's queue dir, driven over the JupyterLab
terminal websocket (no SSH; see `project-server-dgxanode01` in memory).

**Historical, paused** — `runs/queue_full_scale` holds the completed Gaussian arms and the abandoned
`mc_vis_seed0`. Its dashboard (PID 9400, port 8770) was killed 2026-08-22 to avoid confusing it with
the live IR one on 8771. Leave the queue paused; it is the record of what ran before the split. Do not
`resume` it — it would restart `mc_vis_seed0` on the laptop, exactly the hybrid arm the split exists
to avoid.

The `--queue-dir` above is where the queue *bookkeeping* lives; it is not where training output goes.
For that see §1.2 — the results tree is split by kind (`full_scale/`, `mc_dropout/`,
`benchmark/runs/`, plus `full_scale_server/` on the server), not gathered under one subdir.

The existing `queue.cmd`/`dashboard.cmd` launchers do **not** pass `--queue-dir` — don't use them
unmodified for any of these; they'll operate on `runs/queue/` instead.

Per standing instruction: once confirmed running, go quiet — no unprompted status narration; check
only when asked.

---

## 5. Uncommitted code — nothing committed this session, review before it piles up — **RECORDED**

All of this predates this session's work (implemented per the launch plan in an earlier, now-summarized
session) and is still sitting uncommitted:

| Path | State | Note |
|---|---|---|
| `src/uqfusion/uq/variants.py` | modified, +12 | adds `yolo26m-p2feat` to `VARIANT_SPECS` |
| `src/uqfusion/bench/grid.py` | modified, +61/-… | `train_ensemble` gains resume/weights/callbacks branch |
| `src/uqfusion/uq/ensemble.py` | modified, +26 | `train_ensemble_member` single-seed wrapper |
| `src/uqfusion/uq/mc_dropout.py` | modified, +62/-… | same resume/weights/callbacks branch as grid.py |
| `scripts/run_queue.py` | modified, +173/-… | `full_scale_queue`, `--matrix` flag, kind dispatch; `out_subdir` reaches the gaussian branch only (§1.2) |
| `scripts/dashboard.py` | modified, +19 | minor, likely the `kind`-awareness note from the plan |
| `configs/models/yolo26m-p2feat.yaml` | new | m-scale copy of the p2feat neck config |
| `docs/TODO-2026-08-20-full-scale.md` | modified | A-1 decision write-up |
| `docs/followup-analysis-2026-08-20.md` | modified | — |
| `scripts/smoke_queue_kinds.py` | new, untracked | likely the plan's Validation step 2 (mc/ensemble dry-run) — unread this session |
| `scripts/eval_risk_coverage_fixed_gt.py` | new, untracked | unread this session |
| `scripts/loro_mu_b.py` | new, untracked | unread this session |
| `scripts/refit_maha_dayonly.py` | new, untracked | unread this session |
| `scripts/sweep_dedup_iou.py` | new, untracked | unread this session |

The five untracked scripts beyond `smoke_queue_kinds.py` weren't touched or read this session — their
purpose is inferred from filename only. Worth a pass to confirm what's ready to commit vs still
scratch, especially since `eval_risk_coverage_fixed_gt.py` and `refit_maha_dayonly.py` look relevant
to §3's open mitigation-verification step.

---

## 6. Open decisions

| # | Decision | Notes |
|---|---|---|
| D-1 | Verify downstream comparison scripts slice to ship-class AP before comparing C-1 VIS checkpoints against Phase 1 | §3. Not started — check `eval_risk_coverage_fixed_gt.py` / `refit_maha_dayonly.py` first |
| ~~D-2~~ | ~~Check VIS buoy AP~~ | **ANSWERED 2026-08-23, §1.5.** Measured while validating `mc_vis_seed0/best.pt`: **ship 0.284, buoy 0.198** mAP50-95. Buoy is weaker but not degenerate. The class-imbalance theory for the divergence is **dead**: mid-collapse (seed 1, ep 20) ship falls 0.292 → 0.084 while buoy only 0.200 → 0.163 — the *majority* class breaks first |
| D-3 | Commit or clean up the 6 uncommitted files | §5. Nothing blocks this except review time |
| ~~D-4~~ | ~~Re-check remaining-time estimate once `mc_ir_seed0` finishes~~ | **CLOSED 2026-08-22.** `mc_ir_seed0` early-stopped at 42 epochs (best ep 21, 0.13942), `_ft` at 11 (best ep 9, 0.12869). §2.1 re-derived from steady-state epoch times |
| D-5 | Investigate the unexplained ~3.4h pause if it recurs | §1.1 |
| D-6 | Decide whether `out_subdir` should apply to all kinds, and whether to consolidate the three results trees once the matrix lands | §1.2. Needs a signature change on `train_mc_dropout`/`train_ensemble_member`; do it between matrices, not during one |
| D-7 | ~~Confirm the server VIS retrain tracks the laptop `gauss_vis_seed0` curve~~ | **ANSWERED 2026-08-22, §2.2.** It does not track cleanly: the last 10 matched epochs are all positive, mean **+0.0049, sd 0.0009** (~+2% relative). One-directional drift, the failure signal. Env parity verified (identical torch/cuDNN/ultralytics/dataset); the live differences are `workers` 6 vs 8, GPU, and python/numpy minor versions. Superseded by D-9 |
| D-8 | Anything that collects "the matrix" must glob **every** results tree, not `runs/full_scale/*` | §1.2, §1.3. Silent-undercount risk: globbing one tree returns the 4 Gaussian arms and no error. Laptop: `full_scale/`, `mc_dropout/`, `ensemble/`. Server: `full_scale_server/`, `mc_dropout/`, `ensemble/`. The ensemble tree is `runs/ensemble/` as of the 2026-08-22 fix — it was `benchmark/runs/` for the first five laptop members, which have been migrated. Collect by matching run *id*, not "everything in the directory" |
| ~~D-10~~ | ~~Restart the server runner before it reaches `ens_vis_seed0`~~ | **CLOSED 2026-08-22 05:17 UTC.** §1.3. Paused at the epoch-29 boundary, PID 8712 killed, relaunched as PID 12891; resumed from `last.pt` at the unchanged path with no work lost. Both machines now execute the fixed module |
| D-9 | **Bound the machine offset against seed noise.** *Early read 2026-08-23: the four completed **IR** ensemble seeds give mean 0.13514, **sd 0.00955**, range 0.02104 — **1.95×** the +0.0049 offset. On D-9's own rule (sd ≥ ~0.005 → noise) that points to "noise", but it is IR/`yolo26m-p2feat`, not the VIS seeds D-9 specifies, so it is an indication and not the answer.* Compute sd of final mAP50-95 across `ens_vis_seed0-4` (all on the server) and compare to the +0.0049 laptop-vs-server offset | §2.2. If seed sd ≥ ~0.005 the offset is noise and the laptop `gauss_vis_seed0` can stay as a secondary reference. If it lands near 0.001 the offset is a real machine effect and that run must be **dropped from every table**, not merely deprioritised. Free — the seeds are already queued; arithmetic once they land |
| ~~D-11~~ | ~~Decide what to do about the diverged `mc_vis_seed0`~~ | **CLOSED 2026-08-23, §1.5 — structural, and the cause is now known.** `mc_vis_seed1` reproduced the divergence at the *same epoch 18* with the same magnitude (`val/cls_loss` 17.5 / 57.4 / 113.8 vs seed 0's 14.7 / 59.3 / 71.9), so it was never seed luck. Superseded by D-13/D-14/D-15. *Original entry:* **ACTIONED 2026-08-23 04:57 UTC.** §1.4. `mc_vis_seed0` stopped and marked `failed`; `mc_vis_seed0_ft` skipped by the parent guard; `mc_vis_seed1` (same recipe, `seed: 1`) started as PID 14668. **The open half:** if seed 1 also collapses, the instability is structural and a pre-registered change (lower `p`, lower `lr0`, or gradient clipping) must be applied to **both** modalities — never to this one arm alone, which would break comparability exactly as badly as the divergence does. Decision point ~epoch 16, roughly 4.5 h after 04:57. Note the VIS MC arm currently has **no** fine-tune stage from either seed |
| D-13 | **MC-Dropout placement: code done and gated; runs LAUNCHED 2026-08-23** | §1.5. **Option A implemented and gated 2026-08-23** (`deployed_head_branches`; `enable_mc_dropout` now refuses off-path dropout). Remaining: ~~(a) 10-epoch smoke on both modalities~~ **done 2026-08-23, both modalities give non-zero MC spread on real trained checkpoints (§1.5)**, ~~(b) sync `uq/mc_dropout.py` to the server~~ **done — byte-identical on both machines, gate re-run and green on the server**, (c) the runs — **`mc_vis_seed0` started on the server 12:26 UTC (PID 18025); `mc_ir_seed0(+_ft)` chained on the laptop behind `ens_ir_seed3_ft`.** Fallback if p=0.15 is unstable on the thinner one2one head: element-wise `nn.Dropout` (option D) |
| D-14 | **Re-run `mc_ir_seed0` and `mc_ir_seed0_ft`** | §1.5. **Confirmed from the checkpoints 2026-08-23**, not merely inferred from shared code: both IR `best.pt` files carry `end2end=True` with 6 `Dropout2d` on `cv2`/`cv3` and **0** on `one2one_cv2`/`one2one_cv3`. They completed cleanly and never diverged, which is exactly what made them look valid; they are the same no-op — zero epistemic variance. **Note for whoever queues this:** `runs/queue_ir_laptop/state.json` still records both as `done`, so the runner skips them (`[queue] skip mc_ir_seed0 — already done`). Their state entries must be reset, and their output dirs moved aside, or the re-run silently does nothing. Currently sitting in `runs/mc_dropout/` looking like valid results; mark them invalid *before* anyone builds a table from that directory. ~12 h. **Queued 2026-08-23**: dirs moved to `*_broken-20260823`, and a watcher launches `run --only mc_ir_seed0 mc_ir_seed0_ft --redo` the moment the `ens_ir_seed3_ft` runner exits |
| ~~D-15~~ | ~~Add `scripts/smoke_mc_e2e.py`~~ | **CLOSED 2026-08-23, §1.5.** Written, verified red on the unfixed module (failed check A) then green on the fix. Five checks: placement / execution / stochasticity / determinism-when-disarmed / guard. CPU, ~5 s. A call counter turned out to be insufficient on its own — the check also asserts each layer's `training` flag and a non-zero in/out delta |
| D-16 | Make the queue survive a transient mid-run crash | §1.5. The IR smoke died at epoch 9 on a `PermissionError` writing `results.csv` — a Windows lock, gone a minute later. At epoch 60 of a real run that costs the run unless someone re-queues it. `train_mc_dropout` (and `train_gaussian`) already resume from `last.pt`, so the fix is in `run_queue.py`: on a non-zero exit where `last.pt` exists and the epoch count advanced, retry once instead of going straight to `failed`. Cheap, and it pairs naturally with D-12 |
| D-12 | Add an `mAP50 == 0` alarm to the queue heartbeat | §1.4. The divergence ran 4.5 h unnoticed because every loss curve looked normal. One comparison in `run_queue.py`'s epoch callback would have flagged it at epoch 22. Cheap; do it before the ensemble arms start |

---

## 7. Suggested order of work — **updated 2026-08-22**

| # | Action | Cost | Blocks on |
|---|---|---|---|
| ~~1~~ | ~~Upload, extract, launch the VIS queue~~ | done 2026-08-22 | — |
| ~~2~~ | ~~Watch the server `gauss_vis_seed0` retrain against the laptop curve (D-7)~~ | done, §2.2 | — |
| ~~7~~ | ~~Re-derive timing estimate after `mc_ir_seed0` (D-4)~~ | done, §2.1 | — |
| 3 | Let both queues run; laptop dashboard on port 8771 (PID 14832) | — | — |
| 4 | Read `eval_risk_coverage_fixed_gt.py` / `refit_maha_dayonly.py`, confirm ship-class slicing (D-1) and **six**-tree globbing (D-8) | ~15 min read | — |
| ~~5~~ | ~~VIS buoy AP check (D-2)~~ | done 2026-08-23, §1.5 — ship 0.284 / buoy 0.198 | — |
| 6 | Review + commit the uncommitted files (D-3) | ~20 min | — |
| 8 | Compute seed sd across `ens_vis_seed0-4`, settle the machine offset (D-9) | ~5 min | server reaching the ensemble arms (~4 days) |
| ~~9~~ | ~~Watch `mc_vis_seed1`; decide structural-vs-luck (D-11)~~ | done 2026-08-23, §1.5 | — |
| ~~11~~ | ~~Write the MC-Dropout gate `smoke_mc_e2e.py` (D-15)~~ | done 2026-08-23, red→green | — |
| ~~11b~~ | ~~Re-implement the placement on one2one (D-13)~~ | done 2026-08-23, gate green | — |
| 12 | 10-epoch MC smoke on both modalities, then sync `mc_dropout.py` to the server (D-13) | ~2 h GPU | — |
| 13 | Full re-run of both MC arms (D-13, D-14) | ~32 h GPU | D-13 smoke passing; competes with the ensemble queue |
| 10 | Add the `mAP50 == 0` alarm to the epoch callback (D-12) | ~15 min | before `ens_vis_seed0` starts |

Steps 4–6 are laptop-sized and don't need the GPU; they can happen while both queues run. Note the
laptop GPU is busy with IR, so step 5 will contend with training — it is one minute, but schedule it
between arms if the timing of an arm matters.
