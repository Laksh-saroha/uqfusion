# Handoff — 2026-09-11, Phase 3 Stage 0 closed and Stage 2 half-trained

Written to be picked up cold. Covers the Phase 3 pre-registration
(`prereg-phase3-retrain-2026-09-10.md`) from Stage 0 gates through the first half of
Stage 2 training: five amendments appended to the prereg, two gate bugs found and fixed,
one contaminated shipped component found by audit, one IR seed lost to a real
divergence, and a queue that died mid-run and was restarted.

Everything below was measured on this laptop. Amendment 3 pins Phase 3 training here;
`dgxanode01` is not involved in Phase 3 and nothing on it changed this session.

---

## 1. What is running right now

| | status |
|---|---|
| **Laptop GPU (RTX 4080)** | `p3_vis_seed0` training, epoch 4/100, ~36 img/s, 9.0 GB / 12.28 GB, 62 °C. Runner pid **10244**, dashboard pid **4596** at <http://127.0.0.1:8772>. |
| **Queue** | `runs/queue_phase3_stage2/` — 10 runs. IR finished, 4 of 5 (§5). VIS seeds 0–4 remain, ~12 h/seed at the observed early-stop depth, so **~2.5 days**. |
| **Repo** | `fusion-uq-phase3`, 14 unpushed commits, HEAD `6c0af91`. One uncommitted file: `scripts/dashboard.py` (§8). Nothing pushed to origin. |

Check status without touching anything:
```bash
python scripts/run_queue.py --queue-dir runs/queue_phase3_stage2 status
```

---

## 2. What Phase 3 is, in one paragraph

The project has never had a run-disjoint evaluation set — `FIT_RUNS` excludes only
pohang01, which is entirely night (`project-benchmark-holdout` memory). Phase 3 holds
**pohang04** out of training entirely, retrains both arms without it, and then looks at
it **once**, under §7.2's single-look policy. The value of the holdout is destroyed by
looking early: Stop rule 5 says a premature look voids it with no remedy. That is why
most of this session went into gates rather than into training.

---

## 3. Stage 0 — the gates, and the two that were wrong

### G1 — the two machines' label trees, reconciled

G1 as written compared a **tree-scope** hash against a **train-scope** hash. Those are
different algorithms over different file sets, not a subset relation, so the comparison
could never have passed and its failure would have meant nothing. The comparable pair is
`8ed69b5974ed` (laptop) vs `287b11c50b5a` (server).

They differ, and `scripts/reconstruct_prerestore_hash.py` proves *how* rather than
inferring it: rebuilding the server's tree in memory from `visfilter_manifest.json` (the
17,502 files `cut_dark` emptied) reproduces `287b11c50b5a` exactly, at 616,891 boxes. The
difference is the **+94,553-box night restore** (`project-visfilter-night-cut` memory)
and nothing else.

Amendment 3 then made this moot for Phase 3: one machine trains, so G1 became **G1′** — a
host assertion that proves *which* machine, rather than a cross-machine hash match.

### G5 — holdout exclusion, and the bug that made it worthless

`scripts/make_holdout_free_splits.py` writes `_p04out` copies of the VIS lists. It
**filters, never regenerates** — regenerating stride-2 over a smaller pool would pick a
different subset of pohang00–03 than every measurement on record was taken on, so the
held-out run would stop being the only difference.

**The bug.** `Pohang_dataset/visible/val.txt` stores *relative* paths, which a loader
resolves against the list file's own directory. Copying those lines verbatim into
`runs/derived/` repointed 18,063 rows at `runs/derived/images/...`, which does not exist.
The lists had the right counts, contained no pohang04, and **loaded zero images**. G5's
gate passed them. It surfaced only when the VIS batch probe tried to train on one.

Fixed in both halves, which is the point:

- `absolutise()` in the writer — resolve each line against the list it came from;
- a **50-row resolvability sample** in the gate — a list that resolves to nothing now fails.

> The lesson, which recurred three times this session: **a gate that verifies the wrong
> property, or that can never fire, is worse than no gate** — it converts an unchecked
> assumption into a documented one.

### G5 is now a per-invocation precondition

`queue.json` carries `"assert_holdout": true`, and `run_queue.py` calls
`assert_training_host()` + `assert_clean_yaml()` **at the dispatch point of every run**,
not once at queue load. A yaml edited or regenerated mid-queue therefore cannot slip a
held-out frame into a later run, and the refusal names the run that would have consumed
it. Verified working: the 18:55 restart passed the gate on `data_vis_stride2_p04out.yaml`.

---

## 4. The audit that mattered most — pohang04 was inside a shipped component

Full record: `docs/holdout-contamination-audit-2026-09-10.md`.

"No pohang04 in the training lists" was never the whole question. §7.2 does not
distinguish a detector that saw the held-out run from a **threshold or reference
distribution fitted on it**. Auditing every list that feeds a fitted constant found one:

`runs/derived/maha_fit_vis.txt` holds **819 pohang04 frames of 4,000** (20.5%). It is the
source of every `gauss_vis_train_clean.pkl` on disk (18 caches), and `ctx.fit_scorer`
(`src/uqfusion/eval/ctx.py:496`) fits the **shipped Mahalanobis reference distribution**
on exactly that cache. As shipped, the OOD scorer would score pohang04 as
in-distribution because it literally fitted on 819 of its frames.

The benchmark itself is clean — every §6 arm reads caches built from `paired_val_*`,
which carry zero pohang04 — but that is a fact about how those caches happen to be built,
not a safeguard anyone put there.

**The fix is half done, and the remaining half is a GPU job.**

- ✅ The list: `runs/derived/maha_fit_vis_p04out.txt`, 3,181 frames. Filtered rather than
  resampled back to 4,000 — resampling would change *which* pohang00–03 frames sit in the
  reference set, and that composition is the thing not to disturb. The covariance is
  estimated from ~20% fewer frames, recorded rather than corrected.
- ❌ **The cache.** A list does not retro-fit a cache. `gauss_vis_train_clean.pkl` must be
  **rebuilt** from the clean list **before §7's single look**, or the shipped scorer keeps
  its contaminated reference whatever the list says.

> A cache records the list it was built from and then **outlives it**. Same lesson as the
> `_p04out` path bug, one level further out.

---

## 5. Stage 2 — IR is done, and one seed is genuinely gone

IR trained first by design: cheaper failure (~4.5 h/seed vs ~12 h), and it is the arm
never before trained as the thing deployed (nc=1 ship-only).

| seed | status | epochs | best mAP50-95 | best epoch |
|---|---|---:|---:|---:|
| 0 | done | 46 | 0.14775 | 25 |
| 1 | done | 48 | 0.13431 | 27 |
| 2 | done | 50 | 0.14131 | 29 |
| 3 | **diverged @ 6** | 7 | 0.09597 | 4 |
| 4 | done | 30 | 0.13184 | 9 |

Four seeds span 0.1318–0.1478, a 0.0160 spread. **Do not read that spread as a result
yet.** Amendment 4 cut the "retrained vs deployed" comparison entirely under §9
BUDGET-CUT: no artifact may claim the retrain is better, worse, or equivalent.

### `p3_ir_seed3` — this one is real, unlike 2026-08-26

There is a documented IR **false positive** for this same alarm
(`docs/ir-benchmark-divergence-falsepos-2026-08-26.md`), so the trajectory was checked
against it rather than taken at face value:

| epoch | train/box | train/cls | val/box | val/cls | mAP50-95 | vs best |
|---|---|---|---|---|---:|---:|
| 4 | 2.074 | 1.399 | 2.342 | 2.410 | **0.09597** | — |
| 5 | 1.930 | 1.243 | 2.561 | 2.924 | 0.03910 | 0.41× |
| 6 | 1.837 | 1.140 | 3.242 | 3.042 | 0.00501 | **0.052×** |

The 2026-08-26 case was *one* noisy val pass with every loss falling and mAP still ~4× its
epoch-1 value. This is a different shape: **val** losses rise for two consecutive epochs
while train losses keep falling — the model fitting train and losing val — and mAP falls
19× over those two epochs. The rule requires two consecutive epochs below 0.60× of the
**post-warmup** best, and the tightest healthy margin across 27 real runs is 0.63×, so the
threshold sits on measured ground. The kill was correct.

**Consequence:** §5.2 registered **5 seeds per arm and you have 4.** The checkpoint cannot
be continued — Ultralytics strips epoch and optimizer state on `trainer.stop` — so
recovery means a full rerun from epoch 0, about 4.5 h:

```bash
python scripts/run_queue.py --queue-dir runs/queue_phase3_stage2 run --only p3_ir_seed3 --redo
```

**Decide this before looking at any Stage 2 numbers, and write the decision down.**
Rerunning after seeing the 4-seed result, or dropping the seed after seeing the 5-seed
one, is selection on the outcome. Either choice is defensible in advance; neither is
defensible retroactively.

---

## 6. The queue died, and how it was restarted

At **13:44** a pause was requested from the dashboard. `live.json` recorded
`pause_pending: true` at epoch 4 of `p3_vis_seed0`, and the runner (pid 40132) then
**exited** — no python processes at all, GPU idle at 1116 MiB / 2%, while `state.json`
still read `queue_status: running` with `p3_vis_seed0` `running`. A dead runner leaves no
trace in `state.json`: **the only reliable liveness check is the process table plus
`nvidia-smi`.**

Restarted cleanly:

```bash
python scripts/run_queue.py --queue-dir runs/queue_phase3_stage2 resume
```

then relaunching the runner detached, redirected to a **new timestamped log**. It skipped
all five IR runs as terminal, passed the holdout gate, and resumed `p3_vis_seed0` from
`last.pt` at **epoch 4** — `results.csv` held three complete epochs and the checkpoint was
written at 13:30, so at most one epoch of work was lost. `run_queue.py` restores the
early-stopper to the recorded `best_epoch` on resume, so patience counts from the true
best rather than from the resume point.

Logs are **not** appended on restart: each launch writes `runner-<timestamp>.log`, so the
original `runner.log` covering the IR runs stays intact.

---

## 7. The five amendments, one line each

Appended, never edited in place (§11).

1. **A1** — G1's two hashes were incomparable; G5 covers 15 lists, not 5; `all.zip` is VIS labels already on disk.
2. **A2** — pohang04 thermal imagery is **already extracted** (22,235 frames on `D:`), so §7.1 step 1 becomes a transfer and the 16→8-bit conversion becomes part of holdout provenance. **pohang04 has no IR labels anywhere** — the bundle's `tir` directory is absent, not empty.
3. **A3** — Phase 3 trains on the laptop only; G1 → G1′. The server is ~4.5% faster, which does not buy a second machine's worth of reconciliation risk.
4. **A4** — G4 measured; 5 seeds stand; the "retrained vs deployed" comparison is **CUT** (§9 BUDGET-CUT). The design contains no independently-trained-arm comparison at all — §6's S0–S7 re-score the same checkpoints.
5. **A5** — §7's single look covers the **sigma-head system only**. The VIS ensemble and MC-dropout arms were trained on the contaminated `data_vis_stride2.yaml`; the IR ones are clean by construction. Extending the look to them would mean retraining them too.

---

## 8. Uncommitted, and why

`scripts/dashboard.py` — its `TERMINAL` set omitted `"diverged"`, which `run_queue.py:72`
includes. The dashboard therefore counted `p3_ir_seed3` as still remaining and reported 6
runs left instead of 5. One-line fix with a mirrored comment. Safe to commit.

Two rewrite backups also still exist, deletable whenever you like:
`backup-before-trailer-strip` and `refs/original/`, left from stripping `Co-Authored-By`
trailers across the 14 unpushed commits. Content was verified byte-identical before and
after (`git diff backup-before-trailer-strip HEAD` empty, same tree hash).

---

## 9. What is pending, in the order it has to happen

1. **Stage 2 VIS** — 5 seeds, ~2.5 days. Running now.
2. **Decide `p3_ir_seed3`** — rerun, or report 4 seeds. **Before looking at numbers** (§5).
3. **Rebuild `gauss_vis_train_clean.pkl`** from `maha_fit_vis_p04out.txt`. GPU job over
   3,181 frames. **Blocking for §7** — without it the shipped Mahalanobis scorer keeps its
   contaminated reference (§4).
4. **Stage 1, cell D** — the CPU-only correspondence × mechanism crossing. Never started;
   needs no GPU, so it can run alongside VIS training.
5. **Stage 4** — the pohang04 thermal annotation pass (§7.1), still required in full.
6. **Then, and only then, the single look** (§7.2). Once.

Left deliberately untreated: `day_val_vis_stride1.txt` is still contaminated (2,343
pohang04 frames of 9,284). It feeds the rerank and oracle-headroom substrate, which is
exploratory and ships nothing. **Any constant adopted from that work needs the same
treatment before it could appear in a preset.**

---

## 10. Measured constants worth not re-deriving

- **Batch 12, workers 8**, both arms (`docs/batch-ceiling-2026-09-10.md`). `config.yaml`'s
  32 is H100-sized; the MIG figures do not apply either.
- **The probe measures seconds per image, not whether it crashed.** On Windows WDDM the
  driver does not raise OOM when a run exceeds VRAM — it pages into host RAM and the run
  *succeeds*, about **17× slower**. A crash-based probe would have handed Stage 2 a silent
  17× slowdown across every seed.
- Two wrong calls recorded in that doc because they were acted on: thermal throttling did
  **not** explain VIS's 14/16 slowdown (it reproduced exactly in reverse order), and
  subsets did **not** under-read memory peaks (10.36 GiB subset vs 10.12 full).
