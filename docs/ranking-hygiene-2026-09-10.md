# Historical ranking hygiene — R-F3 / F15

**R-F3.** Written 2026-09-10, after [R-F1](positioning-2026-09-10.md) and
[R-F2](bibliography-2026-09-10.md).

Two re-runnable audits, one real statistical correction, and one thing the review asked
for that turned out to already exist.

| | command | artifact |
|---|---|---|
| acceptance check on the 93 rows | `python scripts/audit_phase1_manifests.py` | [`eval/phase1_manifest_audit_2026-09-10.json`](eval/phase1_manifest_audit_2026-09-10.json) |
| IR equivalence interval | `python scripts/ir_equivalence_interval.py` | [`eval/ir_equivalence_2026-09-10.md`](eval/ir_equivalence_2026-09-10.md) |

---

## 1. The acceptance check, answered rather than asserted

The review states it exactly: *"Each table row resolves to one manifest and a compatible
metric. A comparison can be reproduced without recovering undocumented pilot data."*

Measured over all 93 rows of `phase1_benchmark/results.csv`:

| check | result |
|---|---|
| unique rows; no variant double-counted | **PASS** — no repeated `(grid, variant, seed)`, no repeated `(variant, seed)`, **no variant in both campaigns** |
| one metric **scale** for reported numbers | **PASS** — every row is scored on 8.4.90 |
| one **training** software version | **FAIL** — `main_yolo26m_seed0` trained under 8.4.7 (re-scored on 8.4.90; its in-training curve is on the old scale) |
| every row resolves to one manifest | **FAIL** — see below |
| reproducible without undocumented pilot data | **FAIL** — the pilot machine was wiped |

The identity PASS is worth stating plainly: the record's earlier decision to retire the
pilot's `yolo12s` rows **worked**. Each variant now sits in exactly one campaign. The
pooled ranking's defect is mixed conditions, not double counting.

**Why the manifest check fails, and the interesting part is not the missing fields:**

* `git_commit` names nothing recoverable on **83 of 93 rows** (all 66 pilot, 17 of 27 main).
* `trained_on` is `server_wiped` on all 66 pilot rows.
* There is **no fingerprint column at all** — `split_fingerprint`, `label_fingerprint_trainval`
  and `recipe_fingerprint` all postdate this campaign (R-E1 slice 3).
* **The sharp one:** `data_yaml` reads `runs/derived/data_vis_stride2.yaml` on **all 93
  rows, identically, across both campaigns** — while §15 of the experimental record proves
  the two campaigns had *different splits*, because the 2026-07-14 resplit dropped 3,673
  frames and regenerated the file at that same path.

That last item is the **F14 pattern again**: a name that does not pin a computation. The
CSV asserts an identical training manifest for two campaigns that provably did not share
one, and it does so in the one column a reader would trust to check. No prose caveat can
fix a column that says the wrong thing; the audit exists so the contradiction is a
machine-readable finding rather than a paragraph in §15 that a table reader will not
reach.

**A trap worth naming: there are two different 93s.** The VIS Phase-1 table is 93 *rows* (66 pilot + 27 main, pooled to 31 variants). The **IR** ladder is 31 variants × 3 seeds = 93 *planned runs*, and it stopped at **44/93** — see [`ir-benchmark-divergence-falsepos-2026-08-26.md`](ir-benchmark-divergence-falsepos-2026-08-26.md) and the closure. A reader who sees “93” and “complete” in the same sentence can easily carry the VIS count onto the IR campaign, which is the exact substitution the review warns against. `docs/handoff-2026-08-17.md` now carries a dated note saying so; its historical text is unchanged.

**Ladder coverage.** `main` covers **9 of 31 variants**; `pilot` covers the other 22. So
"31 variants" is true **only of the pool**, and the review's warning applies directly: the
completeness of the ladder is inherited from the campaign whose conditions are the
unrecoverable ones.

## 2. The pooled ranking is exploratory, and the record now says so

The experimental record already labels each pooled row with its campaign, already reports
that pooling *widened* the inseparable group from four variants to eight, and already
carries a `CORRECTION` retiring the "26-series beats the 12-series" claim. That is good
practice and it is kept.

What was missing is the word. A table that mixes an unrecoverable split, a wiped machine,
and 83 rows without a commit is **exploratory** — it generates hypotheses about
architecture, it does not test them. §12 now says so above the pooled block, and points
here.

The one conclusion that survives at full strength is the capacity floor: every `n`/`t`-scale
model lands at 0.2486–0.2567, roughly 0.05 below the leaders, with no exceptions in any
family. A gap of 0.05 is not something mixed batch sizes and a 17-frame split offset
manufacture.

## 3. The statistical correction — and it is the substantive result of R-F3

[`ir-benchmark-closed-2026-09-01.md`](ir-benchmark-closed-2026-09-01.md) justified stopping
the IR queue at 44/93 with, as its reason 2:

> **2. It measures nothing.** … one-way ANOVA: **F(12, 26) = 1.037, p = 0.45**. … Thirteen
> architectures spanning 2.6–68 M params and 8.7–258 GFLOPs are statistically
> indistinguishable.

**That inference is invalid, and the data says so loudly.** `scripts/ir_equivalence_interval.py`
reproduces the ANOVA from the compiled per-seed table (F(12, 26) = 1.037, p = 0.447 — the
published number is the number in the file) and then asks the question the ANOVA cannot
answer.

| quantity | value |
|---|---:|
| complete variants (3 seeds) | 13 (`yolov9c`, `yolov10s`, `yolo26n` excluded — incomplete) |
| pooled within-variant sd | 0.00508 |
| observed spread of variant means | 0.01193 (`yolo26l` 0.07316 − `yolov10n` 0.06124) |
| **minimum detectable spread** (α=0.05, power=0.80) | **0.02067** |
| observed spread as a fraction of that | **0.58×** |
| **Tukey HSD 95% simultaneous CI on the largest gap** | **[−0.00314, +0.02700]** |
| same pair, no multiplicity correction | [+0.00340, +0.02045] |

**The design was powered to detect only a spread ~1.7× larger than anything the ladder
produced.** A non-significant result was close to the expected outcome regardless of
whether architecture matters, so p = 0.45 carries almost no evidence either way.

Equivalence at a declared delta, which is what the review asked for:

| tolerable delta | supported? |
|---:|---|
| 0.0014 (paired 2σ floor, low) | **no** |
| 0.0031 (paired 2σ floor, high) | **no** |
| 0.0050 | **no** |
| 0.0100 | **no** |
| 0.0150 | **no** |
| 0.0200 | **no** |

The CI upper bound of **0.0270 is larger than the entire observed range of variant means
(0.0119)** and is roughly 37% of the best variant's absolute score. **No equivalence claim
at any delta a person would care about is supported.** The correct statement is *"this
ladder could not resolve architecture differences at the scale IR operates on"*, not *"the
architectures are indistinguishable"*.

There is a second, uncomfortable reading in the same table: **without the multiplicity
correction the `yolo26l` − `yolov10n` interval is [+0.0034, +0.0205] and excludes zero.**
The correction is the right call — that pair is the maximum of 78 comparisons — but it
means the data is not the featureless null the closure describes. It is an underpowered
design that saw a suggestive gap and could not adjudicate it.

**What this does not do: it does not reopen the decision.** The closure gave three
reasons, and reasons 1 and 3 are untouched and sufficient on their own —

1. IR architecture was frozen on 2026-08-20; the queue was created 2026-08-26, six days
   *after* the decision it was meant to inform.
3. The ladder trains 2-class `data_ir_stride4.yaml` while production IR is nc=1 ship-only,
   and **the deployed `yolo26m-p2feat` is not in the ladder at all**.

Stopping was right. **The justification for it was wrong, and it is the kind of wrong that
gets repeated** — "p > 0.05, therefore no effect" is the single most common statistical
error in applied ML papers, and it was about to go into one from here. The closure keeps
its decision and loses reason 2's claim; an amendment is appended to that document rather
than rewriting the reasoning that was recorded on the day.

The same objection applies to the VIS side's *"within one seed SD"* language, and the
record already handles it correctly by refusing the positive claim ("Any claim that one of
these eight beats another is unsupported"). No change needed there — refusing to separate
is not the same error as asserting equivalence.

## 4. The accuracy/latency tradeoff already existed

The review asks the project to *"state the practical accuracy/latency tradeoff used to
select YOLO26m."* It is stated — in [`handoff-2026-08-17.md`](handoff-2026-08-17.md),
which records D25 as an FPS tie-break under D8/D24 with the soft spot disclosed. It simply
never reached the experimental record, where the ranking lives.

Measured across all three seeds (`phase1_benchmark/fps.csv`, batch-1, pinned clocks):

| variant | mAP50-95 | fp32 FPS | fp32 ms/img | two-stream FPS |
|---|---:|---:|---:|---:|
| `yolo26x` | 0.3049 | 30.7 | 32.6 | 15.3 |
| **`yolo26m`** | **0.3016** | **57.0** | **17.5** | **28.5** |
| `yolo12x` | 0.3007 | 23.8 | 42.1 | 11.9 |
| `yolo26l` | 0.2998 | 44.9 | 22.3 | 22.4 |
| `yolo26s` | 0.2813 | 59.9 | 16.7 | 30.0 |
| `yolo26n` | 0.2540 | 58.6 | 17.1 | 29.3 |

**`26m` costs −0.0033 mAP against `26x` and buys 1.86× the throughput** (15.1 ms/img
saved). The "two-stream" column is the one that matters for this project and appears
nowhere in the original rationale: fusion runs a detector **per modality**, so `26x` lands
at ~15 FPS on paired input while `26m` clears ~28. Given that the top eight variants are
statistically inseparable, spending 1.86× the compute to move within the noise is not a
defensible trade. `26s` is faster still but gives up 0.020, which is well outside the
inseparable group.

*(A curiosity, recorded not explained: for `26m`/`26s`/`26n`, fp16 measures slightly
**slower** than fp32 — 53.9 vs 57.0 FPS for `26m`. Consistent across seeds. Not chased.)*

## 5. What R-F3 does not do

* **No confirmatory run.** The review suggests "a small confirmatory comparison of the
  selected model and the strongest relevant controls on the actual deployment recipe."
  That is GPU work on the deployment recipe, not a documentation repair, and it needs its
  own pre-registration with the tolerable delta fixed **in advance** — writing one after
  seeing §3's intervals would be exactly the practice this backlog exists to correct. It
  is listed as the open follow-up.
* **No historical row is repaired.** They cannot be: the pilot machine was wiped and the
  yaml at that path was overwritten. The audit records the gap; it does not close it.
* **The IR closure decision stands.** Only its reason 2 is amended.
* **No number moves.** Every mAP in this document is read from existing artifacts.
