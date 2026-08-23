# Handoff — MC-Dropout rebuilt on the deployed branch, both machines re-running

**Written 2026-08-23 ~13:00 UTC / 18:30 IST.** Supersedes the operational parts of
[`handoff-2026-08-21-full-scale.md`](handoff-2026-08-21-full-scale.md); that
document keeps the full forensic account and stays the reference for §1.5 (root
cause), §2 (timing), §3 (VIS class mismatch) and the D-list.

---

## 0. State in one paragraph

The MC-Dropout arm was never producing epistemic uncertainty. `insert_head_dropout`
inserted `Dropout2d` into `cv2`/`cv3`, but YOLO26's `Detect` is end-to-end: it runs
the head twice and inference decodes from `preds["one2one"]`, discarding the
one2many outputs. All six dropout layers trained and none ever executed at
inference — T stochastic passes came out bit-identical, variance was exactly zero,
and every aggregate check passed. It also diverged both VIS runs. The placement is
now on `one2one_cv2`/`one2one_cv3`, gated by a new CPU smoke test, verified on a
real trained checkpoint, synced byte-identical to the server, and **re-running on
both machines as of 12:26 UTC**. Four of the 28 matrix rows were affected; the
other 24 stand.

---

## 1. What is running right now

| machine | run | state | notes |
|---|---|---|---|
| server `dgxanode01`, PID **18025** | `mc_vis_seed0` | epoch 4/100, **1008 s/epoch** (not 16 min) | first run on the fixed placement. **First anomaly point is epoch 16, ≈17:05 UTC** — see §4 |
| server, PID **23574** | `watch_divergence.py` | watching `mc_vis_seed0` | added 13:53 UTC. Alarms on `val/cls_loss`, pauses the queue. §4 |
| server, queued | `mc_vis_seed0_ft` | pending | 10 epochs, AdamW `lr0=1.67e-4`, follows automatically |
| server, queued | `ens_vis_seed0` | **paused at epoch 4/100** | resumes at epoch 5 from `last.pt` once MC finishes; then `ens_vis_seed1-4` |
| laptop, PID **4696** | `ens_ir_seed4` | epoch 2/100, 3.6 it/s, ~640 s/epoch | started 18:54 IST |
| laptop, queued | `ens_ir_seed4_ft` | pending | ~1.5 h, follows automatically |
| laptop, chained | `mc_ir_seed0` + `_ft` | waiting | watcher PID 9564 fires `run --only mc_ir_seed0 mc_ir_seed0_ft --redo` when PID 4696 exits |

**Correction to an earlier reading of this table.** `ens_ir_seed3_ft` was not
"the last IR ensemble arm" — there are five seeds (0–4). It finished 18:54 IST
and the runner moved straight on to `ens_ir_seed4`. At seed 4's historical stop
epoch plus its `_ft`, PID 4696 exits around **05:00 IST**, so the IR MC re-run
(D-14) is roughly ten hours out, not imminent. Nothing is broken; the ETA was
simply short by a full arm.

`mc_vis_seed0` epoch 1: `box 2.245  cls 1.552  mAP50 0.586  mAP50-95 0.22476`,
`val/cls_loss 2.285`. For scale, the no-dropout ensemble control read
mAP50-95 0.210 at its own epoch 1.

### Watching it

The runner log is the single source of truth. Progress bars are carriage-return
redraws, so convert them before filtering or a `grep` returns every tick:

```bash
tail -F /workspace/uqfusion/runs/queue_vis_server/runner-20260823b.log | stdbuf -oL tr '\r' '\n'
```

Epoch summaries only (the Ultralytics block, live):

```bash
tail -F /workspace/uqfusion/runs/queue_vis_server/runner-20260823b.log | stdbuf -oL tr '\r' '\n' | stdbuf -oL grep -aE '^ +(Epoch +GPU_mem|[0-9]+/[0-9]+ .*100%|Class +Images.*100%|all +[0-9])'
```

Drop `-F` for history instead of follow. `stdbuf -oL` on both stages matters —
`tr` and `grep` block-buffer into pipes and output would arrive minutes late.

Two helpers are already running on the server and can be killed freely:
`runs/diag/progress.log` gets a status line every 30 s, and a live tail sits on
**Jupyter terminal 8** (left sidebar → *Running Terminals and Kernels*).

The watcher's own record is `runs/mc_dropout/mc_vis_seed0/divergence-watch.log`
(one line per epoch, `ok` or `ALARM`) with stdout at
`runs/diag/divergence-watch.out`. If it ever fires it also drops
`runs/mc_dropout/mc_vis_seed0/DIVERGENCE-ALARM.txt`, which is visible in the
JupyterLab file browser without opening a terminal at all.

**Reading server state without a terminal.** JupyterLab's Contents API serves
any file under `/workspace` over plain HTTP with the browser's existing session,
so `results.csv` can be pulled and diffed against another run with no shell, no
SSH, and zero contact with the training process — useful when the question is
"how is it doing" and the answer must not risk the run:

```js
await (await fetch('/api/contents/uqfusion/runs/mc_dropout/mc_vis_seed0/' +
  'results.csv?content=1&type=file&format=text', {credentials:'same-origin'})).json()
```

`PUT` to the same path uploads (needs the `_xsrf` cookie as an `X-XSRFToken`
header); `POST /api/terminals` plus a websocket to
`/terminals/websocket/<name>` drives a shell when one is genuinely needed. That
is how `watch_divergence.py` got there and was started.

---

## 2. The fix

`uq/mc_dropout.py` gained `deployed_head_branches(head)`, which returns
`("one2one_cv2", "one2one_cv3")` when `head.end2end` and `("cv2", "cv3")`
otherwise. `insert_head_dropout` iterates it (still idempotent, so resume is
unchanged), and `enable_mc_dropout` now **raises** when no armed layer sits on the
deployed branch — the old guard only checked that *some* dropout existed anywhere,
which is exactly why the defect survived.

This is the same criterion `gaussian._sigma_box_head` already applied, and the
reason the Gaussian arm was never affected. It is a deviation from plan B6-3,
which named `cv2`/`cv3`; B6-3 was written for a conventional detect head and does
not transfer to YOLO26's dual-head design. Write it up as a pre-registered
protocol deviation, not a silent change.

---

## 3. Why we believe it works — five independent links

Counting layers is not evidence they execute; that assumption is what cost 30 GPU
hours. Each link below tests something the previous one cannot.

| # | check | result |
|---|---|---|
| 1 | `scripts/smoke_mc_e2e.py`, five assertions, CPU, ~5 s | run against the **unfixed** module it fails check A; against the fix, all five pass. A gate that cannot fail on its own bug is decoration |
| 2 | placement on the production `yolo26m` head, in memory | `{'cv2': 0, 'cv3': 0, 'one2one_cv2': 3, 'one2one_cv3': 3}` |
| 3 | placement on the **saved** `last.pt` after epoch 1 | same counts, `p=[0.15]`, `nc 2`, read from the `ema` entry — the only one Ultralytics keeps |
| 4 | 10-epoch training smoke, real data, **both** modalities | non-zero MC spread from a trained checkpoint: VIS mean 2.52 px, IR mean 4.81 px |
| 5 | same gate re-run **on the server** after the sync | all five pass, numbers matching the laptop |

Link 3 is the one that closes the loop: an in-memory build proves nothing about
what the trainer wrote to disk.

The gate itself needed two corrections, both of which would have made it pass on
broken code:

- **A call counter proves invocation, not activity.** An eval-mode `Dropout2d` is
  invoked and is the identity. The check now also asserts each layer's `training`
  flag and a non-zero in/out delta.
- **A yaml-built model makes the stochasticity check vacuous.** Random-init
  YOLO26n has head activations of order 1e-6, so two armed passes differ by
  `0.0` — identical to broken. Pretrained weights give 271. The gate requires
  them and says why.

---

## 4. What to watch, and the fallback

**Epoch 16, not 18, is where to look.** Reading the broken run's own
`results.csv` end to end gives a sharper timeline than "~18":

| epoch | val/cls_loss | mAP50 | mAP50-95 | |
|---|---|---|---|---|
| 15 | 2.17 | 0.644 | 0.240 | last clean epoch |
| **16** | **5.02** | 0.596 | 0.221 | **first anomaly** — 2.02× its trailing-5 median |
| 17 | 3.10 | 0.611 | 0.226 | partial relapse, still elevated |
| 18 | 14.74 | 0.507 | 0.178 | unrecoverable from here |
| 19–20 | 59.2 → 71.8 | 0.358 → 0.273 | 0.117 → 0.092 | |
| **22** | 3.28 | **0** | **0** | mAP first reads exactly 0 |

`train/cls_loss` stayed flat (0.42 → 0.41) throughout. Seed 1 reproduced the
same epoch, so it was structural, not seed luck.

**This is why the D-12 spec was wrong.** An `mAP50 == 0` alarm first fires at
epoch 22 — six epochs and ~100 minutes of A100 time after the run is gone. The
detector has to be `val/cls_loss`, which is what this section already said to
watch but D-12 never encoded.

**The armed rule** (`scripts/watch_divergence.py`, running as PID 23574):
`val/cls_loss` > 1.5× its trailing-5 median, with `mAP50-95 < 0.6 × best` as a
backstop and `mAP50 == 0` as a floor. The 1.5 is not taste — over epochs 1–15
of the broken run that ratio never exceeded **1.10**, and at epoch 16 it was
**2.02**, so the threshold sits between them with ~35% margin either way.
Replayed over the broken curve it first fires at epoch 16; over `ens_vis_seed0`
and the live run it stays silent. On alarm it requests a queue *pause*, not a
kill — `control.json` is read live every batch and the runner stops at
`on_model_save`, right after `last.pt`, so resuming loses nothing.

**Clock for the live run** (started 12:28 UTC, 1008 s/epoch): epoch 16 ≈
**17:05 UTC / 22:35 IST**, epoch 18 ≈ 17:38 UTC, epoch 22 ≈ 18:45 UTC.

### Epochs 1–4 so far — nothing conclusive, one mild positive

| | ep1 | ep2 | ep3 | ep4 |
|---|---|---|---|---|
| val/cls_loss — new / broken / control | 2.28 / 2.25 / 2.45 | 2.58 / 2.48 / 2.46 | 2.55 / 2.52 / 2.56 | 2.40 / 2.30 / 2.39 |
| mAP50-95 — new / broken / control | 0.225 / 0.212 / 0.210 | 0.193 / 0.222 / 0.233 | 0.177 / 0.215 / 0.187 | 0.207 / 0.224 / 0.224 |
| train/cls_loss — new / broken / control | 1.552 / 1.480 / 1.453 | 0.872 / 0.846 / 0.810 | 0.790 / 0.779 / 0.741 | — |

The mAP dip at epochs 2–3 is **not** a warning sign: the no-dropout control does
the same thing (0.233 → 0.187 → 0.224) because the LR warmup peaks at epoch 3
(`lr/pg0` 0.0294). Both recover at epoch 4. Without the control alongside it that
dip reads as alarming, and it isn't.

The one mild positive: `train/cls_loss` in the new run sits consistently *above*
both the broken run and the control at every epoch. That is what an actually-
active regulariser looks like, and it is the first evidence from training rather
than from the placement gate. It is weak — but the broken run was indistinguishable
from healthy through epoch 15, so nothing before 16 can be strong.

The mechanism should now be gone: dropout on `one2one` sits behind features
Ultralytics feeds **detached**, so it contributes no gradient to the shared trunk.
The previous failure was noise perturbing the trunk through one2many gradients
while the scored branch got neither the regularisation nor a gradient path back.

**If it diverges again at ~18**, the mechanism was not what we think, and the
fallback is option D — element-wise `nn.Dropout` instead of `Dropout2d`, which
drops individual activations rather than whole channels and is far gentler on a
head this thin. Do not lower `p` for VIS alone without applying the same change to
IR; an arm-specific hyperparameter breaks comparability as badly as the divergence
does.

**Watch `val/cls_loss`, not mAP.** In both failures the loss blew up over epochs
16–21 while mAP still looked plausible; by the time mAP hit zero the run was long
gone. There is no automatic alarm yet — that is D-12.

---

## 5. Deliberately not being re-run

- **`mc_vis_seed1` / `_ft`** stay `failed`. Seed 1 existed only to test whether the
  divergence was seed luck. It answered that. The matrix needs seed 0, and
  re-running seed 1 would cost ~20 h to re-answer a settled question.
- **The IR MC arms are INVALID, not merely untidy.** `mc_ir_seed0` and `_ft`
  trained cleanly and never diverged, which is exactly what made them look fine.
  Their checkpoints carry the identical defect (`end2end=True`, 6 on `cv2`/`cv3`,
  0 on `one2one`), so they produce zero epistemic variance. They are queued to
  re-run.

**Why IR never diverged, when it had the same defect.** The LR schedule is
identical, not merely similar (VIS and IR both read 0.02495 / 0.02465 / 0.02436 at
epochs 18/19/20), so that confound is ruled out. IR also shows no sub-threshold
version: max/min `val/cls_loss` is 1.44 against its own no-dropout control's 1.37,
where VIS was ~33. The leading explanation — inferred, not proven — is **`nc=1` vs
`nc=2`**: the failure was specifically in the classification loss, and with one
class there is no inter-class boundary to lose. It also reconciles the per-class
detail, since a collapsing two-class boundary hurts the 93% majority most (ship
0.292 → 0.084, buoy 0.200 → 0.163). Untested: batch 10 vs 16, the p2feat neck.
This does not need settling before the re-runs, but it becomes actionable if VIS
destabilises again.

---

## 6. Operational traps — read before touching a queue

1. **`specs` and `state` are both read once, at runner startup**
   (`run_queue.py:505-512`), and `save_state` rewrites the whole dict. Editing
   `state.json` under a live runner gets clobbered at its next save; queue edits
   never reach a running process. Any change needs a restart.
2. **A live runner holds the old module in memory.** Syncing a `.py` does nothing
   for a process that already imported it. The new guard cannot save this case
   either — it lives in `enable_mc_dropout`, which training never calls. **The
   restart is the fix, not the sync.**
3. **`train_mc_dropout` resumes from `last.pt` before anything else.** Re-queueing
   a spec whose old output dir survives silently continues the broken run. Move it
   aside first. Done today: `runs/mc_dropout/mc_vis_seed{0,1}_broken-20260823`
   (server), `mc_ir_seed0{,_ft}_broken-20260823` (laptop).
4. **`run_grid` skips a run outright if a row exists in its CSV**
   (`grid.py:211`), independently of `state.json`.
5. **`resume=True` reloads `save_dir` from the checkpoint**, so a run whose
   directory was moved resumes into the old path and re-creates it
   (`grid.py:114-118`). Checked for `ens_vis_seed0`: its checkpoint points at its
   real location, so its epoch-4 resume is safe.
6. **`state.json`'s `best_map50_95` is off by one** — do not read it. Use
   `results.csv`.
7. A **fresh Jupyter terminal opens in `/workspace`**, not `/workspace/uqfusion`.
   Jupyter's contents root is `/workspace`; the repo is `/workspace/uqfusion`; the
   venv is `/opt/venv_match`. (`/workspace/Saroha_Work` belongs to the *other*
   server.)

### Re-running an MC arm, end to end

```bash
mv runs/mc_dropout/<id> runs/mc_dropout/<id>_broken-$(date +%Y%m%d)
# then either reset its state.json entry to {"status": "pending"} and restart the
# runner, or run it filtered:
python scripts/run_queue.py --queue-dir <dir> run --only <id> <id>_ft --redo
```

`--redo` overrides a terminal status; the `_ft` parent guard passes because the
base arm completes earlier in the same invocation. No `queue.json` edit is needed
— the specs already exist.

---

## 7. Open items

| id | item |
|---|---|
| **D-12** | **Armed 2026-08-23 13:53 UTC — but not as specified.** Both halves of the original spec were wrong: `mAP50 == 0` fires at epoch 22, six epochs late (§4), and an in-process epoch callback cannot reach a runner that is already going — trap 2 in §6 applies to the alarm itself. `scripts/watch_divergence.py` polls `results.csv` from outside instead; server PID 23574. **Still open:** fold the same rule into `on_fit_epoch_end` so future runs are covered from epoch 1 without a sidecar process. That half lands naturally on the next runner restart (the IR MC re-run, ≈05:00 IST). |
| **D-16** | Make the queue survive a transient mid-run crash. The IR smoke died at epoch 9 on a `PermissionError` writing `results.csv` — a Windows lock, gone a minute later. At epoch 60 that costs a run. Both trainers already resume from `last.pt`, so: on non-zero exit where `last.pt` exists and the epoch count advanced, retry once instead of going straight to `failed`. |
| **D-9** | Seed-noise bound. Slips by ~1 day because MC was put ahead of `ens_vis_seed1-4`. Early read from four IR ensemble seeds: mean 0.13514, **sd 0.00955** — 1.95× the +0.0049 machine offset, pointing to "noise", but that is IR/`p2feat`, not the VIS seeds D-9 specifies. |
| **D-14** | Re-run the IR MC arms — queued, see §1. |
| **D-3** | **Partly cleared 2026-08-23.** Committed: `2a6d30e` the MC fix + both smoke gates (gate re-run green on the laptop first), `c5dcdc6` the divergence watcher. **Still uncommitted and unreviewed:** `run_queue.py` (+173), `bench/grid.py` (+100), `dashboard.py`, `eval/apmetrics.py`, `uq/ensemble.py`, `uq/variants.py` (the `yolo26m-p2feat` spec) and its `configs/models/yolo26m-p2feat.yaml`, both handoff docs, and the five loose analysis scripts. These were left deliberately — they are ~550 lines nobody has read this session, and the MC fix does not depend on them (`is_variant`/`build_variant` are already in HEAD; the new `train_mc_dropout` kwargs are optional). |
| D-1, D-6, D-8 | Carried forward unchanged from the 08-21 handoff. |

---

## 8. Files touched today

| path | what |
|---|---|
| `src/uqfusion/uq/mc_dropout.py` | the fix — `deployed_head_branches`, branch-aware guard, B6-3 deviation recorded in the docstring |
| `scripts/smoke_mc_e2e.py` | new, the §18-2 gate. CPU, ~5 s, no dataset. Run before any MC GPU time |
| `scripts/smoke_mc_train.py` | new, 10-epoch training smoke on real data, both modalities |
| `scripts/watch_divergence.py` | new, the D-12 alarm. Polls `results.csv` from outside the trainer; alarms on `val/cls_loss` > 1.5× trailing-5 median; optionally pauses the queue. Thresholds calibrated on the broken run, replay-tested against it |
| `docs/handoff-2026-08-21-full-scale.md` | §1.5 grown with root cause, IR analysis, smoke results, server sync; D-13/D-14/D-15 updated; D-16 added |
| server `runs/diag/check_placement_ckpt.py` | per-branch `Dropout2d` count on a saved checkpoint |
| server `src/uqfusion/uq/mc_dropout.py.bak-20260823` | the pre-fix copy (`3b298372e35c74e2937593e56b7bd206`) |

md5 parity, laptop ↔ server:

```
59d0b8ecafcab900dd6cbc667140d9e1  src/uqfusion/uq/mc_dropout.py
3388858bf0c6fe916e1ea94b6db45ea8  scripts/smoke_mc_e2e.py
35ae69830fae315d48824238f3e3bed7  scripts/smoke_mc_train.py
```
