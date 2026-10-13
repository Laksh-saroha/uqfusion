# G5 — the pohang04-free VIS splits, and what removing a daylight run does to val

**Built 2026-09-10.** Gate G5 of
[`prereg-phase3-retrain-2026-09-10.md`](prereg-phase3-retrain-2026-09-10.md) §3, and the
"regenerated **without** pohang04" clause of §5.1. Written by
`scripts/make_holdout_free_splits.py`; gated by `scripts/assert_holdout_excluded.py`.

---

## 1. Filtered, not regenerated

The pre-registration says "regenerated". It was **filtered**, and the difference is
deliberate.

Rebuilding stride-2 from the master list would re-run the stride selection over a smaller
pool and pick a **different subset of pohang00–03 frames** than every measurement in the
record was taken on. Dropping pohang04's rows from the existing list leaves every surviving
frame exactly where it was, so the only difference between the old training set and the new
one is the held-out run — which is the only difference Stage 2 is entitled to introduce
here. Regenerating would confound the holdout with a fresh frame draw.

**Nothing was overwritten.** Outputs carry a `_p04out` suffix and sit beside the originals,
which stay valid for reproducing pre-Phase-3 results. Each input keeps its own path style —
`visible/val.txt` stores relative paths, the derived lists absolute — so lines were dropped,
never rewritten.

| output | in | dropped | kept |
|---|---:|---:|---:|
| `data_vis_train_stride2_p04out.txt` | 48,136 | 9,841 | **38,295** |
| `vis_val_p04out.txt` | 11,352 | 2,343 | **9,009** |
| `vis_test_p04out.txt` | 11,445 | 2,391 | **9,054** |

`runs/derived/data_vis_stride2_p04out.yaml` points at all three.

**The IR side needed no equivalent.** `ir_shiponly/{train_stride2,val,test}.txt` carry zero
pohang04 rows and will until Stage 4 extracts thermal frames — the run has no IR labels at
all (Amendment 2). `data_ir_shiponly_stride2.yaml` passes G5 unmodified.

---

## 2. The composition shift, which is not neutral

pohang04 is daylight throughout, so removing it does not shrink val evenly. Every night
frame in val comes from pohang01 and none of them are touched:

| | day | night | night share |
|---|---:|---:|---:|
| val before | 9,284 | 2,068 | 18.2% |
| **val after** | **6,941** | **2,068** | **23.0%** |

**Val is now 23.0% night, and all of that night is one recording.** §8 already forbids
pooling day and night into a headline for exactly this reason; this makes the reason
sharper, because the pooled val number moves toward pohang01 without any model changing.
Any comparison against a pre-Phase-3 pooled val figure is comparing two different
populations and must not be made.

Per-run composition after filtering:

| list | pohang00 | pohang01 | pohang02 | pohang03 |
|---|---:|---:|---:|---:|
| train | 8,193 | 9,414 | 10,212 | 10,476 |
| val | 1,672 | 2,068 | 2,690 | 2,579 |
| test | 1,912 | 1,629 | 2,864 | 2,649 |

**This is an input to G4, not a footnote.** The MDE for every Stage 2 comparison is computed
over the val/test sets above, which are ~21% smaller than the ones earlier power estimates
would have assumed, and differently composed. G4 runs against these lists or it is measuring
the wrong design.

---

## 3. The gate was wrong first, and how

`assert_holdout_excluded.py` originally asserted zero held-out rows across **every list on
disk**. That gate can never pass: the originals are deliberately kept so pre-Phase-3 results
stay reproducible, so the survey is expected to report hits forever — and a gate that cannot
pass is a gate everyone learns to skip.

It now has two modes, and the narrow one is the precondition:

* **`--yaml PATH`** — gate on the train/val/test lists that config actually trains on. This
  is what a training invocation calls. Verified: passes on
  `data_vis_stride2_p04out.yaml` and `data_ir_shiponly_stride2.yaml`, fails on
  `data_vis_stride2.yaml` naming all three contaminated lists.
* **no argument** — survey every list under the split roots. Still reports 15 lists and
  58,144 rows, and is expected to. Informational.

The yaml is parsed with a three-key reader rather than a yaml library, because the
precondition runs under the GPU interpreter — the system Python, not the project venv — and
must not fail on a missing optional dependency.

`--check-host` enforces G1′ (Amendment 3) in the same call.

---

## 4. What this does not settle

* **The other 12 contaminated lists are untouched**, including `visible/train.txt` and the
  `maha_fit_vis.txt` selection set. They are not Stage 2 inputs, but anything that fits a
  constant on them still sees pohang04. Before a pohang04 number is looked at, every such
  fit must be re-checked against the `_p04out` lists or declared out of scope — §7.2's
  single look does not distinguish between a detector that saw the run and a threshold that
  did.
* **This is a frame-level exclusion, not an exposure repair.** pohang04 frames left the
  training lists today; they were in them before, and any checkpoint on disk that trained
  under the old yaml saw them. Only checkpoints trained from here forward are clean.
