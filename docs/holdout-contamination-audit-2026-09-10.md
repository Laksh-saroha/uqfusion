# Where pohang04 still reaches the shipped system — the audit G5 did not cover

**Measured 2026-09-10**, while Stage 2's first IR seed trains. Closes the item left open by
[`g5-holdout-free-splits-2026-09-10.md`](g5-holdout-free-splits-2026-09-10.md) §4: *"anything
that fits a constant on them still sees pohang04."*

§7.2 does not distinguish a detector that saw the held-out run from a threshold that did. So
"no pohang04 in the training lists" was never the whole question.

---

## 1. Verdict

**One list reaches a shipped component, and it reaches 18 caches.**

`runs/derived/maha_fit_vis.txt` holds **819 pohang04 frames of 4,000** (20.5%). It is the
source of every `gauss_vis_train_clean.pkl` on disk, and `ctx.fit_scorer` (`ctx.py:496`)
fits the **shipped Mahalanobis reference distribution** on exactly that cache. As it stands
the OOD scorer would score pohang04 as in-distribution because it fitted on 819 of its
frames.

Everything else is clean, and two of the clean results are clean by construction rather than
by anyone's care — which is worth knowing, because construction can change.

---

## 2. What was checked, and what each answer rests on

| source | pohang04 | reaches | verdict |
|---|---:|---|---|
| `maha_fit_vis.txt` | **819 / 4,000** | 18 caches → shipped Mahalanobis scorer | **CONTAMINATED** |
| `derived_day/day_val_vis_stride1.txt` | **2,343 / 9,284** | 1 cache → `fit_rerank.py`, `probe_oracle_headroom.py` | contaminated, **not shipped** |
| `maha_fit_ir.txt` | 0 / 4,000 | 16 caches | clean |
| `paired_val_vis.txt` / `paired_val_ir.txt` | 0 / 2,232 | 194 caches — every benchmark cell | clean |
| `ladder_vis.txt` / `ladder_ir.txt` | 0 / 744 | 26 caches | clean |
| `ctx.FIT_RUNS` / `TUNE_RUNS` / `TEST_RUNS` | — | every constant fitted through `ctx.py` | clean — selects by **run name**, and pohang04 is in none of them |
| `visible/train.txt`, `stride5/10/19`, `fullres_*`, `phase1val_*`, `uqfusion_bench/train.txt` | 19,687 / 3,943 / 3,937 / 1,970 / 1,041 / 2,343 / 2,391 / 409 | **nothing** — no cache, no fit, no live consumer | contaminated, inert |

The benchmark is clean because the caches are built from `paired_val_*`, and the gate, veto,
capability prior and every §6 arm read those. **That is the reason the headline numbers are
not affected**, and it is a fact about how the caches happen to be built, not a safeguard.

---

## 3. The fix, which is two things and only one of them is done

**Done — the list.** `runs/derived/maha_fit_vis_p04out.txt`, 3,181 frames, written by
`scripts/make_holdout_free_splits.py`. Filtered rather than resampled back to 4,000:
resampling would change *which* pohang00–03 frames sit in the reference set, and the
reference set's composition is the thing not to disturb. The covariance is therefore
estimated from ~20% fewer frames, which is recorded rather than corrected.

**Not done — the cache.** A list does not retro-fit a cache. `gauss_vis_train_clean.pkl`
must be **rebuilt** from the clean list before §7's single look, or the shipped scorer keeps
its contaminated reference whatever the list says. That is a GPU job over 3,181 frames, and
the GPU is training Stage 2. **It is queued after Stage 2 and before the single look.**

Which caches need it depends on scope: `runs/cache_m/` is what the shipped preset reads, and
the 17 draw/variant directories matter only if a report is regenerated from them.

---

## 4. The gate that would have caught this

`scripts/assert_holdout_excluded.py --caches` walks every cache, reads its stamped
`meta.images_list`, and fails if that list carries a held-out frame. It reports 19 today.

The lesson is the same one the `_p04out` path-bug taught, one level further out. A cache
records the list it was built from and then **outlives it**. Cleaning a list does not clean
the caches already built from it, and a constant fitted on such a cache inherits whatever
that list held on the day it was built. So a holdout can be absent from every training list
and still sit inside a shipped component — which is exactly what was true here between the
G5 fix and this audit.

---

## 5. What this does not settle

* **It audits the caches that exist.** A cache built later from a contaminated list would
  pass unnoticed until the gate is run again; the gate is not yet wired into cache building
  the way G5 is wired into training.
* **`day_val_vis_stride1.txt` is left contaminated.** It feeds the rerank and oracle-headroom
  substrate, which is exploratory. **Any constant adopted from that work would need the same
  treatment before it could appear in a shipped preset**, and nothing currently does.
* **It says nothing about exposure through reading.** These are mechanical dependencies.
  Constants chosen by a human who looked at pohang04 numbers would not appear here — and
  §7.2's protection is against exactly that too. No pohang04 score has been computed at any
  point, which is the reason to think this list is complete.
