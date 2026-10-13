# IR benchmark: divergence false positive, and the two make-up runs — 2026-08-26

Queue `runs/queue_ir_benchmark_stride4` (31 variants × 3 seeds = 93 runs, IR @ train stride 4,
2-class). Two rows were lost in the first 14. Both are back in the queue; one needed a code fix
first.

---

## 1. `ir_bench_yolov8l_seed1` — divergence alarm was a **false positive**

Killed at **epoch 4 of 100**, 11 minutes in:

```
val/cls_loss 4.389 is 1.58x its trailing-3 median 2.774 (limit 1.50x)
```

**It was not diverging.** Its own `results.csv`, all four epochs:

| epoch | train/box | train/cls | train/dfl | val/cls | mAP50-95 | lr/pg0 |
|---|---|---|---|---|---|---|
| 1 | 2.654 | 2.215 | 1.681 | 3.626 | 0.0052 | 0.00055 |
| 2 | 2.345 | 1.743 | 1.538 | 2.774 | 0.0047 | 0.00110 |
| 3 | 2.151 | 1.525 | 1.445 | 2.226 | 0.0220 | 0.00163 |
| 4 | 2.000 | 1.371 | 1.369 | **4.389** | 0.0197 | 0.00162 |

All three training losses fall monotonically through epoch 4, and mAP50-95 is still ~4× its
epoch-1 value. One noisy validation pass, not the multi-epoch monotonic blow-up
`mc_vis_seed0_broken` showed (val/cls 2.17 → 5.02 → 14.74 → 71.85 over epochs 15–20).

### Root cause — an off-by-one in the guard argument, not in the threshold

`args.yaml` has `warmup_epochs: 3.0`, confirmed by the LR column: pg0 climbs through epochs 1–3
and turns over at epoch 4. **Epoch 4 is the first epoch at peak LR.**

The loss rule's window was `rows[max(0, i - window):i]`, armed as soon as
`len(history) >= min_history`. With `min_history=3` that is first satisfied at row 3 — epoch 4 —
and the baseline is then rows 0–2, i.e. **the warmup ramp alone**: a steeply-descending baseline
compared against the LR-peak epoch.

`watch_divergence.py`'s docstring had explicitly argued this rule needed no warmup guard, "its
`min_history` window does not fill until the warmup is over." Off by exactly one epoch, in the
worst direction: the window fills *precisely at* the LR peak. Both mAP rules already excluded
warmup from their baseline (`rows[warmup:i]`); the loss rule did not.

### Fix

`scripts/watch_divergence.py`, one line plus the corrected docstring:

```python
history = [r["val_cls"] for r in rows[max(args.warmup, i - args.window):i]]
```

Warmup rows are now excluded from the baseline **and** the test, matching the mAP rules. First
possible fire moves from epoch 4 to epoch 7.

### Verified, not assumed

For `i >= warmup + window` (= 8, epoch 9) the slice is **byte-identical** to the old one, so
nothing past epoch 8 changes by construction — and the real divergence's first hit was epoch 16.

Replayed old vs new rule over **20 runs** (14 IR-benchmark rows, the archived
`mc_ir_seed0_broken-20260823`, and the four `full_scale` Gaussian runs). **Exactly one verdict
changes: this false positive.** No completed run would have been newly condemned, and no alarm
that fired is lost.

### Campaign consistency

Rows 1–12 trained under the old rule, the rest under the new one. Since the two rules agree on
all 12, the campaign is judged consistently end to end. The one disagreeing row is being
retrained under the new rule. Run dir preserved as
`runs/ir_benchmark_stride4/ir_bench_yolov8l_seed1_falsepos-20260826/` (with its
`DIVERGENCE-ALARM.txt`) so the re-run does not overwrite the evidence.

---

## 2. `ir_bench_yolov8x_seed0` — transient download failure

```
ConnectionError: Download failure for .../v8.4.0/yolov8x.pt.
Retry limit reached. Curl return value 35
```

Curl 35 is an SSL connect error — a network blip at 17:17, not a defect. It cost the row 42
seconds; the queue moved on and `yolov8x_seed1` (17:17→20:43) and `seed2` trained from the same
weights without trouble, because the file landed locally on a later attempt.
`A:\Uncertain\yolov8x.pt` is present (136,890,692 bytes, loads, `nc=80`), so the re-run resolves
offline. No code change needed.

---

## 3. Why a plain restart would NOT have fixed either

`run_queue.py:69` — `TERMINAL = {"done", "failed", "skipped", "diverged"}` — and `cmd_run` skips
any row already terminal unless `--redo` is passed. Restarting as-is would have logged
`skip ... already failed` / `skip ... already diverged` and walked past both.

Both rows were reset to `pending` in `state.json`, the previous attempt preserved under a
`previous_attempts` list (status, metrics, error, and why it was requeued). Backup:
`state.json.bak-20260826-restart`.

`--only <id> --redo` is the documented alternative, but it runs *only* those ids and exits,
needing a second restart to continue. Resetting the status lets the normal pass pick them up in
queue order and carry on.

---

## 4. What the restart did

The queue had been paused from the dashboard at 22:41. The pause landed cleanly at
`on_model_save` — end of epoch 32 of `ir_bench_yolov8x_seed2`, `last.pt` written, status
`paused` — so the runner was parked with nothing on the GPU: the safe moment to cycle it.

1. Archived the false-positive run dir.
2. Killed runner PID 6680 and its 47 dataloader workers (`taskkill /T /F`). Killing **before**
   editing `state.json` is load-bearing: the live process holds `state` in memory and
   `wait_while_paused` calls `save_state` on resume, which would have clobbered the resets.
3. Reset the two rows to `pending`, unpaused `control.json`.
4. Relaunched detached via `ir_benchmark_run_detached.cmd` (new PID 10872).

**Resulting order:** `yolov8l_seed1` from epoch 0, then `yolov8x_seed0` from epoch 0, then
`yolov8x_seed2` **resumes from its epoch-32 `last.pt`** (`paused` is not terminal). So seed2 sits
parked ~7 h while the two make-up runs go first. Total GPU time unchanged — only ordering
differs.

---

## 5. Carried forward

- The 12 completed rows are unaffected and need no re-run (§1, verified).
- `scripts/watch_divergence.py` and `scripts/run_queue.py` both still carry **uncommitted**
  changes (the `map_confirm` backstop work from earlier today, plus this fix). They should land
  in one commit describing both guards.
- Still open from the same session: the IR benchmark is **2-class** while the IR fusion path is
  `nc=1` ship-only — see `TODO-2026-08-26-phase1-classset.md` §5. Decide before the sweep
  finishes.

---

## 6. Confirmed in production, and synced to the server — 2026-08-26 ~23:15 IST

### The re-run cleared the window, and proves the diagnosis

`ir_bench_yolov8l_seed1` restarted from epoch 0 and reproduced epochs 1–4 **bit-identically** —
train/box_loss 2.65368 / 2.3449 / 2.15096 / 1.99969, and the same epoch-4 `val/cls_loss` spike —
then kept going:

| epoch | 5 | 6 | 7 | 8 |
|---|---|---|---|---|
| mAP50-95 | 0.02326 | 0.03512 | 0.04607 | 0.03868 |

Same training, same spike, only the rule differed — the cleanest possible confirmation that the
run was never diverging and epoch 4 was a warmup artifact.

### Server sync (`dgxanode01`)

The server runs `watch_divergence.py` as a **standalone sidecar** (PID 23650), so this fix
matters there independently of `run_queue.py`. Both changed files were byte-verified as
unmodified copies of `ac86cd6` before being overwritten, then rebuilt in place and verified by
read-back:

| file | before | after | verified |
|---|---|---|---|
| `scripts/watch_divergence.py` | `ac86cd6`, LF | `630cbd5` | 21,961 B, fnv1a64 `856ded74f9b98911` = local HEAD |
| `src/uqfusion/uq/train_gaussian.py` | `ac86cd6`, CRLF | `630cbd5`, LF | 15,001 B, fnv1a64 `e92072ffb465285b` = local HEAD |

Method: JupyterLab Contents API `PUT` from the browser pane (no SSH, no git remote). Rather than
shipping whole files, each was rebuilt from the server's own copy by exact string substitution —
every anchor asserted to match exactly once, and the result hashed against the local blob before
the write. A mismatch aborts without writing.

**`scripts/run_queue.py` was deliberately NOT synced.** The server's copy has no
`check_divergence` at all — the fork §6 of `handoff-2026-08-26.md` warns about ("Server
`run_queue.py` deliberately has no in-process divergence alarm; the sidecar covers it. Don't
'fix' that."). Overwriting it would install an in-process alarm the server is designed not to
have. That fork needs a decision, not a sync.

### Still to do on the server

**The sidecar (PID 23650) is still running the old code** — Python loaded it at start, so the new
file does nothing until the watcher is restarted. It is read-only (polls `results.csv`, can
request a queue pause), so restarting costs and risks nothing, but its exact launch arguments are
not recoverable without `ps`: there is no `divergence-watch.log` in `runs/ensemble/ens_vis_seed3/`,
so it was started with an explicit `--log` elsewhere.

**No live exposure right now.** `ens_vis_seed3` is at epoch 7 with `val/cls_loss` flat at
2.37–2.54 (peak ratio ~1.06× against its own trailing median, far under the 1.5× limit) and
mAP50-95 climbing 0.212 → 0.238. The ensemble stages start from warm weights, so they never show
the cold-start warmup crater that produced the false positive. The remaining exposure is
`ens_vis_seed4`, the last cold start in the queue.
