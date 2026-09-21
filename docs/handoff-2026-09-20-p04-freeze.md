# Handoff — 2026-09-20, pohang04 inputs frozen; the look is armed and unfired

Written to be picked up cold. The Phase 3 §7.2 holdout inputs finished building overnight and
are now frozen in a commit. **The single look has not been taken.** Everything it needs is on
disk and hashed; the only thing left is to run it, deliberately, once.

Prereg: `docs/prereg-phase3-retrain-2026-09-10.md`, Amendments 8 and 9.
Procedure: `docs/holdout-p04-freeze-2026-09-14.md`.

---

## 1. State right now

| | status |
|---|---|
| **Repo** | `fusion-uq-phase3`, HEAD **`85a07c1`** `FREEZE: pohang04 single look inputs (190 caches, 76 statistics, 10 checkpoints)`. Tree clean except untracked `local.md`. Nothing pushed to origin. |
| **Builds** | Finished. 190/190 caches, 76/76 frame statistics, **0 failures**. |
| **Look** | **Not taken.** `runs/holdout_p04/LOOK_TAKEN.json` does not exist. |
| **GPU** | Idle. All builder processes exited. |

**This handoff is deliberately left uncommitted.** Committing it would move HEAD off the
FREEZE commit, and refusal 1 below requires HEAD's subject to start with `FREEZE`. Untracked
files are ignored by the check, so the file is safe sitting here unstaged. Commit it *after*
the look is taken, not before — and do not amend it into `85a07c1`; rewriting a freeze commit
destroys the audit trail it exists to provide.

The commit before the freeze is `fd95a55`, an unrelated `.gitignore` entry for the
`server_dgxanode01/` snapshot. It had to be committed first because the look refuses on any
modified tracked file, and leaving it dirty would have blocked the look later.

---

## 2. What was frozen

`scripts/holdout_p04_freeze.py` wrote `docs/eval/holdout_p04_freeze_manifest.json`:
**316 files, 0 missing, `ready: true`, `not_ready_because: []`, `draft: false`.**

| group | files |
|---|---:|
| checkpoints | 10 |
| Mahalanobis references | 10 |
| development caches | 10 |
| pohang04 caches | 190 |
| pohang04 frame statistics | 76 |
| calibration and geometry | 5 |
| development substrate | 8 |
| pohang04 substrate | 5 |
| committed reference | 2 |

The readiness gate passed on its own terms: all 190 caches re-passed the builder's meta check
(weights, image list, frame count, corruption kind / severity / seed, conf), all 76 statistic
files matched their stream, the pohang04 label hash is `c06611a684f4`, and
`runs/holdout_p04/look_selftest.log` ends in `EXACT`.

Recorded alongside the hashes: preset `crossmodal26m`, systems VIS seed k + IR seed k for
k = 0..4, VIS draws 941–944 with IR = VIS + 10, verdict cell `clean/clean`, decision floor
**0.0060** with 0.0014 / 0.0031 / 0.0060 / 0.0100 all reported, moving-block bootstrap L = 20
and n_boot = 1000 (reference seed 0, pohang04 seed 1), and
**AP_ref = 0.28978364906888066** on `pohang02+pohang03`.

---

## 3. The build, for the record

Relaunched 2026-09-19 12:43 after a restart, three shards, resuming over what already existed.

| shard | finished | jobs |
|---|---|---|
| shard2 | 2026-09-19 21:59:33 | 63/63 verified |
| shard0 | 2026-09-19 23:08:23 | 64/64 verified |
| shard1 | 2026-09-20 02:18:31 | 63/63 verified |

Frame statistics finished earlier the same day: `[done] all streams verified`, 76 files.

Costs worth knowing if this is ever rebuilt: fog dominates at ~3,550–5,400 s per job and is
GPU-bound, not core-starved — shard1 alone on the machine ran fog in 3,550 s versus 4,255 s
with three shards competing. Everything else (blur, noise, rain, lowlight, glare, clean) is
370–1,900 s. The earlier `FAILED: build exited 1073807364` line in `shard0.out` is a Windows
shutdown kill code from a user restart, not a real failure; that job was rebuilt and verified.

The laptop's modern standby cost ~6.6 h on the night of 2026-09-18. The signature is processes
alive, logs stopped, CPU time still creeping — diagnose by comparing lifetime average cores
against a live sample and confirm with `Kernel-Power` event IDs 42 / 107 / 566.

---

## 4. Taking the look

One command, and it is the whole point of Phase 3. `--out` is required — the procedure doc's
line omits it:

```bash
PYTHONPATH=A:/Uncertain/src python scripts/holdout_p04_look.py --out docs/eval/holdout_p04_look.md
```

It writes that Markdown plus a sibling `.json`, and stamps `runs/holdout_p04/LOOK_TAKEN.json`.

**It cannot be run twice.** The marker is created with `O_EXCL` *before* scoring, so a crash
mid-look still counts as the look. §7.2 allows exactly one.

Refusals, in the order they fire:

1. HEAD is not a `FREEZE` commit, or any tracked file is modified (untracked files are ignored);
2. `runs/holdout_p04/LOOK_TAKEN.json` already exists;
3. the committed manifest is missing, is a draft, or any of the 316 hashes differs;
4. the pohang04 label hash differs from `c06611a684f4`;
5. `check_inputs` fails on the caches or statistics;
6. the development reference AP does not reproduce exactly, per seed.

Budget 45–90 min. The gate alone re-hashes 16.2 GB and unpickles the 190 caches
(`load_cache` runs at ~35 MB/s, sha256 at ~1 GB/s), then scoring walks the caches again for
5 systems × 11 cells, and 11 bootstraps of 1,000 replicates follow. The freeze's own pass over
the same files took 3.5 min with the files warm, so the hashing half is cheap; the scoring and
bootstrap half is the unmeasured part.

Sanity check it will pass, without touching the look:

```bash
git log -1 --format='%H %s' && git status --porcelain --untracked-files=no
```

HEAD must start `FREEZE` and the status must be empty.

---

## 5. What is deliberately not done

- **The look.** Left unfired on purpose. It was not run unattended because an ambiguous
  instruction is not the explicit instruction §7.2 wants, and the action is unrecoverable.
- **Nothing pushed.** `origin` has none of this.
- **Stage 1** closed as S1-NULL (`docs/` record); correspondence is closed forever and fusion
  is union aggregation.
