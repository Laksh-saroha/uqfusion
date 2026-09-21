# Handoff — 2026-09-21, the look is taken; Phase 3 has its number

Supersedes `handoff-2026-09-20-p04-freeze.md`, which described the armed-and-unfired state.
The pohang04 single look ran on 2026-09-20 and **pohang04 is now spent**. Written to be
picked up cold.

Prereg: `prereg-phase3-retrain-2026-09-10.md`, §7.2 and Amendments 8–9.
Result: `docs/eval/holdout_p04_look.md` and `.json`.
Ledger: `docs/exposure-ledger-2026-09-09.md` §1 and §7.

---

## 1. The result

**NO-GAP**, verdict cell `clean/clean`, rule A9.3.

| | AP |
|---|---:|
| pohang00 (development, high group) | 0.3955 |
| pohang02+03 (development, low group → `AP_ref`) | 0.2898 |
| **pohang04 (held out)** | **0.2682** |
| **D = `AP_ref` − `AP_p04`** | **+0.0216** |
| D 95% CI, unpaired moving-block bootstrap (L = 20, n_boot = 1000, seed 1) | **[−0.0120, +0.0502]** |

12,482 pohang04 day pairs; fused ship AP against VIS ground truth only (A8); five Phase 3
systems (VIS seed *k* + IR seed *k*, *k* = 0..4); preset `crossmodal26m`; freeze commit
`85a07c1`. Ran 18.6 h.

### The one thing not to get wrong

**NO-GAP is not a finding of no gap.** A9.3 requires two things and only the second failed:

```python
"outcome": "HOLDOUT-GAP" if (d_obs >= FLOOR and lo > 0) else "NO-GAP",
```

`d_obs = 0.0216` clears **every** reported floor including 0.0100. `lo = −0.0120` is what
fails. That single conjunct is why all four `gap_by_floor` entries read `false`.

The defensible sentence is: *"a gap of +0.0216 was observed; the data exclude neither zero
nor a gap as large as +0.05."* Writing "pohang04 shows no generalization gap" would repeat
R-F3 (`project-ir-ladder-underpowered`) — where p = 0.45 on an underpowered ladder was read
as absence of effect, and the Tukey upper bound later ruled out the very zero it had been
taken to support. That error is now one edit away from the paper.

Why the interval is wide: it is **unpaired**, so it carries run-level variance, and this
data has a great deal of it. The two development groups differ from each other by **0.1057**
— 4.9× the held-out gap. pohang04 sits just below the bottom of the development range, not
outside it. `above_development` is `false` (U = −0.1272 against pohang00).

### The corruption cells

Sane, and they confirm the system is VIS-dominant.

| | |
|---|---|
| IR-side corruption | 0.2668–0.2692 against 0.2682 clean — **no movement** |
| VIS-side corruption | noise_s2 0.0114, blur_s3 0.0418, fog 0.0580, rain_s2 0.1850 |

Only `clean/clean` carries a verdict (A9.2). The other ten are descriptive.

---

## 2. State of the repo

HEAD is the commit that recorded all of this. The freeze commit `85a07c1` is its parent and
**was not amended** — rewriting a freeze commit destroys the audit trail it exists to provide.
Nothing has been pushed to `origin`.

Committed in this pass: the two look outputs, the marker, the ledger update, the superseded
freeze handoff, this handoff, and `prereg-p2feat-vis-2026-09-20.md`.

---

## 3. What is now impossible

**pohang04 is spent.** `runs/holdout_p04/LOOK_TAKEN.json` exists with
`numbers_written: true`. The script refuses on an existing marker, and §7.2 allows exactly
one look. Scoring pohang04 again — a different architecture, a new cell, even a re-run of the
identical command — is a **second look** and requires a new pre-registration that states
plainly that the set is no longer held out.

This also closes §5 option 3 of the exposure ledger. There is no second untouched release.

---

## 4. Next, in order

1. **Paper §7.** Delete the pre-written outcome paragraph for the outcome that did not occur,
   insert the §1 numbers **with the interval**, and state the caution above in the text rather
   than a footnote. Then lift the `PAPER_WRITING_INSTRUCTIONS.md` §0 rule 6 embargo.
2. **Figures are still placeholders throughout `PAPER_DRAFT.md`.** Unrelated to the look, but
   it is the largest remaining block of work.
3. **The p2feat VIS screen** is unblocked. `docs/prereg-p2feat-vis-2026-09-20.md` is committed
   and governs it. Before launching: verify batch 12 holds for the stride-4 branch against the
   1.5 GiB driver margin of `batch-ceiling-2026-09-10.md` — **stop rule 2 halts the grid if
   batch must drop**, because batch is a recipe field and dropping it un-matches the arms.
   Three treatment seeds, ~4–5 h each, writing to `runs/p2feat_vis/`. `runs/phase3_stage2/` is
   hash-frozen and read-only.
4. **`docs/TODO-improvements.md` §D** pre-publication integrity items remain open.

---

## 5. Cost, for whoever plans the next long run

The look took **18.6 h**, not the 45–90 min the previous handoff budgeted. Both halves of
that miss are worth recording:

* **Scoring**: 205 cells at ~152 s each ≈ 8.7 h. The previous estimate priced the hashing
  gate carefully (507 s, accurate) and called the scoring half "the unmeasured part."
* **Bootstrapping**: 11 cells at **3,209–4,719 s each** ≈ 9.9 h — *slower than scoring*.
  This was the real surprise. Resampling 1,000 times over 12,482 pairs swamps the cache walk
  and WBF that a bootstrap replicate skips.

It ran at **0.997 cores** start to finish. The work is embarrassingly parallel across cells
and the script does not parallelize it; the GPU was idle throughout because the look reads
cached detections and runs no inference. Three or four workers would plausibly have turned
18.6 h into 5–6 h. If a second look is ever pre-registered, parallelize and time the scoring
path on development data first — the selftest already exercises the same code and would have
caught this.

No sleep stall this time: CPU time tracked wall clock to within 0.3% for the whole run.
