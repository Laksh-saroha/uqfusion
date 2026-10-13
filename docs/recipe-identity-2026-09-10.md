# Label state and recipe are part of a run's identity

**R-E1 slice 3, review finding F14.** Written 2026-09-10. Slice 2 made a prediction
cache able to refuse and left two defects reproduced but unfixed, because both needed
a new `RESULT_FIELDS` column. This closes them, and turns up a third on the way.

---

## 1. D3 — `split_fingerprint` cannot see a label edit

It hashes sorted `run/filename` ids, so it answers *"which frames?"* and nothing about
what is written in them. Reproduced on the real function against a synthetic split
(the dataset was not touched):

| state | `split_fingerprint` | label content hash |
|---|---|---|
| original | `0358ccc7de88` | `9c62da55016c` |
| a box **deleted** | `0358ccc7de88` | `0d5ff2fc413a` |
| a class id **changed** | `0358ccc7de88` | `6690f6d5b7c0` |
| a label **file removed** | `0358ccc7de88` | `33d2479c4d94` |

That is exactly the edit this project made twice — the night cut removed 132,688
boxes and the restoration put 94,553 back — and a results CSV could not tell the two
states apart.

## 2. D4 — the completed-run lookup omits the recipe

The skip key was `(variant, seed)` plus `split_fingerprint` and `classes`.
`epochs_cfg` was written into the row but **never consulted**, and `imgsz`, `batch`,
initial `weights` and `train_overrides` were not recorded at all. Reproduced at the
level where the defect lives — one CSV row for `(yolo26m, 0)` at `epochs_cfg` 25:

```
re-run with epochs= 25 -> skip as 'already done'? True
re-run with epochs= 50 -> skip as 'already done'? True
re-run with epochs=100 -> skip as 'already done'? True
```

The CSV goes on reporting the 25-epoch number as the answer to a 50-epoch question.

Latent rather than live: across the nine benchmark CSVs, `epochs_cfg` takes three
values (100 ×53, 50 ×10, 15 ×2), but each budget went to its own file **by
convention**, not by enforcement.

## 3. A third defect, found while fixing the first two

The project held **three independent implementations** of the `images -> labels`
path swap and **two byte-identical copies** of the label content hash:

| implementation | rule |
|---|---|
| `eval/matching.label_path_for` | replaces **every** component named `images` |
| `filter_night_boxes._label_path` | rightmost `images` **substring** (`rfind`) |
| `verify_dataset_state.label_path` | last `images` **separator group** |

These are not the same function. The substring version rewrites the wrong component
whenever a directory *name* ends in `images` — `d/images/run_images/x.png` becomes
`d/images/run_labels/x.txt` under it and `d/labels/run_images/x.txt` under the other
two.

**Measured before changing anything: 0 disagreements between any pair, over all
133,140 real train+val image paths across both modalities.** A latent trap, not a
live defect — which is the argument for collapsing it while it is still latent. They
are now one definition in `src/uqfusion/data/labels.py`, and the shared
`label_content_hash` **reproduces the ledger's recorded `8ed69b5974ed`
bit-for-bit** on the train scope, so `verify_dataset_state.py --expect-label-hash`
and `runs/label_hash_ledger.csv` stay comparable.

## 4. What changed

**Three new `RESULT_FIELDS` columns:** `label_fingerprint_trainval`,
`recipe_fingerprint`, `recipe`.

**`label_fingerprint(data_yaml)`** — the ledger's algorithm over train+val. **The
scope is in the column name on purpose.** The same algorithm over train alone
(`8ed69b5974ed`), over train+val, and over the whole tree gives three different
numbers that mean nothing against each other; the ledger doc already records what
happened the last time two such numbers were compared as if they were one quantity
(`df0cb307adc9` read against `b92739202127`).

Measured now:

| dataset | `split_fingerprint` | `label_fingerprint_trainval` |
|---|---|---|
| vis | `fd7850b36213` | `ae7fa57efb2b` |
| ir | `26c23d66f5ed` | `5fd58f37c799` |

**`recipe_identity(**knobs)`** — fingerprint plus canonical JSON over `epochs`,
`imgsz`, `batch`, `mosaic`, `close_mosaic`, `optimizer`, `patience`, `amp`,
`deterministic`, `weights`, `weights_sha256`, `train_overrides`.

* `weights_sha256` is the **content** hash of the starting checkpoint. R-E1 asks for
  a checkpoint content hash and a path is not one: `best.pt` is overwritten by every
  run that produces it.
* `workers` and `device` are **excluded on purpose**. They can perturb
  nondeterminism, but they are properties of the host, and including them would make
  a fingerprint that never matches across machines — which would break resume, the
  one thing this lookup exists to do.

**`plan_grid(done, dataset, recipe_fp, variants, seeds)`** — the gating decision,
extracted so it is testable without a GPU. Two outputs, deliberately different in
kind:

* **stale** — a different split fingerprint, label state or class filter refuses the
  whole CSV. Two datasets must never share one results file.
* **conflict** — a row for a `(variant, seed)` *this grid will run*, on the same
  dataset but a different recipe. **Not** part of the stale check: one CSV may
  legitimately hold rows trained under different budgets. It **refuses rather than
  re-runs**, because the run directory and the CSV row are both keyed by `name`, so
  re-running would overwrite a different experiment's outputs — the very failure this
  item is about.

**`recover_row.py` now reads the run's own `args.yaml`** for the recipe, and refuses
if it is missing. Reconstructing the recipe from today's config would stamp a
fingerprint for a recipe that never ran — the exact defect R-E1 is about, committed
by the fix for it. `epochs_cfg` comes from the same place.

**`scripts/smoke_recipe_identity.py`** — new, always-on, 17 cases.

## 5. What this costs

* **~15–24 s warm, 383 s cold** to hash the 107,627 VIS train+val label files on this
  machine. Paid once per grid launch (`lru_cache`) against runs measured in hours.
  The cold number is the honest one to plan around on a fresh boot.
* **Old CSVs become read-only.** Every existing row lacks
  `label_fingerprint_trainval`, so it reads as stale and refuses a continuation. That
  is the intended reading — those rows were produced against a label tree that has
  since changed twice, and the CSV cannot say which state each one saw. It is also
  barely a change in practice: `_append_row` already refused to append to a CSV
  written under an older `RESULT_FIELDS`.
* **Nothing in flight is affected.** `scripts/run_queue.py`, which drives the server
  work, does not use `run_grid`.
* Phase 1 is complete, so no live grid depends on the old skip behaviour.

## 6. Verification

* `smoke_recipe_identity` (17 cases), `smoke_cache_identity`, `smoke_apmetrics`,
  `smoke_metric_contracts`, `smoke_benchmark` (which runs `run_grid` end to end on
  synthetic data) all PASS.
* The shared `label_content_hash` reproduces `8ed69b5974ed` — the ledger's recorded
  train-scope hash on the restored tree.
* Every recipe knob produces a distinct fingerprint (10 knobs, 10 distinct values);
  `workers`/`device` produce none.
* No published number moves: nothing here touches a metric.

## 7. What is still open

* **`verify_dataset_state.py` has its own `split_fingerprint`** taking a dict of
  images-by-split, unrelated to `bench.grid.split_fingerprint(data_yaml)`. Same name,
  different signature, different computation — the F14 pattern again, noticed here
  and not fixed.
* **Nothing is validated on resume.** R-E1 asks for cache load *and* resume; slices 2
  and 3 cover load and launch.
* **The training-time label hash is still not compared to the caching-time one.**
  That is R-E1's actual acceptance criterion ("label hashes at training start/end and
  at caching/evaluation agree"). The pieces now exist on both sides —
  `runs/label_hash_ledger.csv` and `label_fingerprint_trainval` — but nothing joins
  them.
* **`train_overrides` is hashed by its repr through `json.dumps(default=str)`.** A
  callable or object in there fingerprints by its `repr`, which can carry a memory
  address. No current caller passes one.
