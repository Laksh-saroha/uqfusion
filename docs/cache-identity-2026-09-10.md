# Cache identity — making a cache able to refuse

**R-E1 slice 2, review finding F14.** Written 2026-09-10. Slice 1 (`f3364c9`)
recorded what produced a result. Its own docstring listed what it did not do:
*"no ordered pair IDs ... and nothing here is validated on cache load"*. This is
that part — the half that can **refuse** a bad run rather than describe one
afterwards.

Four defects were reproduced on the real caches before any code moved. Two are
now closed, two are characterised and deferred with reasons.

---

## 1. What was reproduced

### D1 — `load_cache` validated nothing

| payload handed to `load_cache` | old behaviour |
|---|---|
| `meta` claims `n_frames` 99999, holds 10 records | **loaded, no error** |
| records stripped to `{"image_path": ...}` — no `conf`, `cls`, `boxes` | **loaded, no error** |

### D2 — pairing checked lengths, not frames

`evaluate_systems` opened with `assert len(vis_records) == len(ir_records)`. Equal
lengths are not equal frames. Measured on the clean cell, `preset="crossmodal"`:

| IR cache handed in | gated fusion | Δ | naive fusion Δ |
|---|---:|---:|---:|
| correct | 0.271103 | — | — |
| fully reversed | 0.245473 | **−0.025630** | −0.025185 |
| shifted by 100 | 0.243266 | **−0.027837** | −0.022853 |
| shifted by **1** | 0.270135 | **−0.000968** | −0.000634 |

The last row is the dangerous one. A one-frame misalignment moves the result by
**less than the 0.0014–0.0031 paired noise floor**, so no bootstrap, seed sweep or
significance test could ever have found it. Only identity can.

### D3 — `split_fingerprint` is label-blind

Reproduced on the real function against a synthetic split (the dataset was not
touched):

| state | fingerprint |
|---|---|
| original labels | `c3354ed2f2b1` |
| one box **deleted** | `c3354ed2f2b1` — unchanged |
| a class id **changed** | `c3354ed2f2b1` — unchanged |

It hashes sorted `run/filename` ids, so any edit to label *content* is invisible.
That is the exact edit this project made: the night cut removed 132,688 boxes and
the restoration put 94,553 back.

### D4 — the grid's completed-run lookup omits the recipe

The skip key is `(variant, seed)` plus `split_fingerprint` and `classes`.
`epochs_cfg` is written into the row but **never consulted**, and `imgsz`, `batch`,
initial `weights` and `overrides` are not recorded at all. Re-running the same
variant and seed with a different budget is silently skipped as "already done".

Latent rather than live: across the nine benchmark CSVs, `epochs_cfg` takes three
values (100 ×53, 50 ×10, 15 ×2), but each budget went to its own file **by
convention**, not by enforcement. The only duplicate `(file, variant, seed)` key in
the corpus is the already-known Excel-mangled row.

## 2. A wrong turn, recorded because it is the interesting part

The first version of the pairing check used `run/ordinal` equality — the frame
number, which reduces `pohang00_L_006767` and `pohang00_006767` to the same
`pohang00/6767`. It refused the **legitimate** caches on **1,396 of 2,232 frames
(62.5%)**, first mismatch at index 836: `pohang01/7724` vs `pohang01/7725`.

That was not a discovered misalignment. It was a documented fact, rediscovered and
briefly mistaken for a defect. `scripts/build_pairs.py` has said so since it was
written:

> Do NOT pair by frame number: 16,544 of the 28,388 pair rows have a DIFFERENT
> index on the VIS and IR side, so a filename join silently mismatches 58% of the
> set.

The source of truth is `Pohang_dataset/paired/pohang*_pairs.csv`, timestamp-matched
by the dataset authors with a `dt_ms` column. Measured per-run IR−VIS index offsets:

| run | offsets present |
|---|---|
| `pohang00` | 0, 1 |
| `pohang01` | 0, 1 |
| `pohang02` | 0, 1, 2, 3, 4 |
| `pohang03` | **−155 … +1** (16 distinct values) |

So it is not even a fixed per-run offset. Any check built on frame arithmetic would
be wrong in a different way on every run. The pairing check now reads the authors'
table, which makes it stronger than the version that was intended: it validates
against timestamps rather than against a naming convention.

## 3. What changed

**`src/uqfusion/eval/identity.py`** gains the frame-identity half:

* `pair_id(path)` — ordered identity of one frame **within one modality**
  (`run/ordinal`), with the docstring stating plainly that this is *not* the
  pairing key and why;
* `frames_sha256(records)` — hash of the **ordered** pair-id sequence, so a
  reordered cache is a different cache (a set hash would call the reversal above
  identical);
* `labels_sha256(records)` — hash of the GT label **bytes**, which is what
  `split_fingerprint` cannot see;
* `_pair_table()` — `stereo_L_file → ir_file` from the dataset CSVs;
* `paired_id_report()` / `assert_paired()` — compare and refuse.

**`src/uqfusion/eval/cache.py`**: `build_cache` stamps `pair_ids`,
`frames_sha256`, `labels_sha256` and a `system_identity()` block. `load_cache`
validates the payload shape, the `n_frames` claim, every record's `image_path` and
per-detection array consistency, and — when stamped — re-derives both hashes and
refuses on a mismatch. `validate=False` is the explicit escape hatch, a parameter
rather than a silent fallback so that skipping the check is visible at the call
site.

**Three consumers now refuse a mis-paired pair before doing any work:**
`fusion_eval.evaluate_systems`, `iralign.aligned_homographies` (a residual measured
between two different instants is not a registration quantity), and
`learned_gate.LearnedGate.fit` (its label is "which stream was better on this
frame" — mis-pairing does not make it noisy, it makes it wrong).

`evaluate_systems` also returns its `pairing` report, including
`n_unidentifiable`, so a caller can tell a real pass from a vacuous one.

**`scripts/smoke_cache_identity.py`** — new, always-on, 19 cases.

## 4. Verification

* **Nothing moves.** `crossmodal` clean gated fusion is 0.271103 before and after,
  and all eight preset × condition cells reproduce.
* **The legitimate path passes non-vacuously**: 2,232 frames, 0 mismatched,
  **0 unidentifiable** against a 28,388-row pair table.
* **All three mispairings are refused**, including the sub-noise-floor one-frame
  shift.
* **All 258 prediction caches on disk still load.** The five refusals under
  `runs/**/*.pkl` are the gate/rerank model pickles, which are not caches and were
  never loaded through `load_cache`.
* Existing caches stamp neither hash, so the hash checks apply to new work only.
  A validator does not get to retroactively invalidate recorded results.
* **Every cache pair on disk that anything actually pairs is aligned.** Sweeping
  all 58 vis/ir cache pairs under `runs/cache*/`: **42 aligned, 0 mismatched**. The
  16 that report mismatches are all `gauss_*_train_clean.pkl`, which are the
  Mahalanobis fit caches — `load_context` hands them to `fit_scorer` **separately**,
  one per modality, and nothing pairs them. Their lists were built independently
  (`maha_fit_vis.txt` / `maha_fit_ir.txt`), so 3,124 of 4,000 frames are not in the
  pair table at all. Pairing them would now be refused, which is the right answer.
* smoke_apmetrics / smoke_cocoparity / smoke_metric_contracts / smoke_sigma_wbf /
  smoke_crossmodal_gate / smoke_phase3 / smoke_cache_identity all PASS.

## 5. What is still open

* **D3 and D4 are reproduced, not fixed.** Both need a new column in
  `RESULT_FIELDS` (a `label_fingerprint`, and the recipe in the skip key), and
  `_append_row` deliberately refuses to append to a CSV written under an older
  schema. That is correct behaviour, and it means the change forces a fresh CSV —
  worth doing deliberately in its own slice rather than as a side effect of this
  one. `run_queue.py`, which drives the server work, does not use `run_grid`, so
  nothing in flight is affected either way.
* **Nothing is validated on *resume*.** R-E1 asks for both; this slice covers cache
  load only.
* **No checkpoint content hash.** A cache records its `weights` path, not what was
  in the file.
* **The stamped hashes are only as good as the first stamp.** `build_cache` records
  what it saw at write time; nothing yet compares a cache's `labels_sha256` against
  the hash recorded at *training* time, which is R-E1's actual acceptance criterion
  and needs `scripts/label_hash_ledger.py` wired to both ends.
* **~73 JSON writers outside `write_md` still stamp nothing** (carried from slice 1).
