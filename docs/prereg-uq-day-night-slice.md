# Pre-registration — the day/night slice of the UQ table (U1)

**Written 2026-09-03, before any UQ cache is built or any metric is recomputed.**
Committed ahead of the run so the rule is verifiable in git history.

## The claim under test

The argument that the UQ arms need no retraining after the night label restore
(`docs/experiment-log-2026-09-02.md` §7.2) is borrowed from Phase 1: every arm was
trained on the same emptied night labels, so the handicap is uniform and cancels out
of a comparison.

For Phase 1 that argument was **measured and it held** — night AP was 0.0000 on all
27 checkpoints, spread 0.0000, and the day ranking matched the pooled ranking in all
9 positions (`runs/eval/phase1_day_night_slice.md`).

**It has not been measured here, and the two cases are not alike.** Phase 1 ranked on
mAP, where a constant handicap cancels. Per D31 the UQ arms are explicitly *not*
ranked on mAP — `docs/uq-arms-compiled-2026-09-01.md` §4: "The arm comparison is
carried by the calibration metrics (d-ECE, NLL, AUSE, AURC)." Calibration metrics do
not merely fall when labels are wrong; they measure whether a model's confidence
matched reality, scored against a reality the labels misstate. AUSE and AURC are the
sharpest case: they reward an arm for being uncertain where the errors are, and night
is now a block of manufactured errors.

The three arms generate uncertainty by three different mechanisms — T stochastic
passes (MC), seed spread over n=5 (ensemble), a learned sigma head. There is no
reason their distortion over a mislabelled region should be identical, which is
precisely what the borrowed argument assumes.

## Why the pooled table cannot be the endpoint

`runs/eval/table2_gaussian.json` is computed over `runs/derived/paired_val_vis.txt`
(2,232 frames, verified):

| run | frames | |
|---|---:|---|
| pohang00 | 836 | day |
| pohang02 | 247 | day |
| pohang03 | 117 | day |
| **pohang01** | **1,032** | **night** |
| total | 2,232 | |

**The UQ table is 46.2% night** — not the 18.2% of full val. Nearly half of every
pooled calibration number is scored on frames whose VIS training labels were emptied.
A pooled number cannot be decomposed after the fact, so the slice has to be measured
rather than argued.

## The arms

All three VIS arms are 10-epoch fine-tunes on `runs/derived/data_vis_stride2.yaml`,
patience 10, verified from `args.yaml`:

| arm | checkpoint | local? |
|---|---|---|
| sigma-head | `runs/full_scale/gauss_vis_seed0_ft/` | yes |
| MC-Dropout | `runs/mc_dropout/mc_vis_seed0_ft_refit/` | yes |
| deep ensemble (n=5) | `ens_vis_seed{0..4}_ft` | **seed 0 only** |

Ensemble seeds 1–4 weights are on `dgxanode01` and are **not** in
`archive/server_pull_2026-08-31/ensemble/` — those directories carry `results.csv`
but their `weights/` are empty. Verified.

## The rule

Fixed before the run, not to be edited afterwards.

1. **Caches.** One inference pass per arm over the *full* paired list, into **new**
   files under `runs/cache_uqslice/`. Nothing under `runs/cache/`, `runs/cache_m/`,
   `runs/derived/`, `runs/eval/` or `archive/` is written.
   `build_cache.py --split val --images-list runs/derived/paired_val_vis.txt
   --imgsz 640 --conf 0.001`, matching the settings recorded in the existing cache
   meta. Before scoring, assert from each cache's `meta` that `images_list`, `imgsz`,
   `conf` and `corrupt` agree across arms, and that the `image_path` sequences are
   identical element-by-element.
2. **Slicing is post-hoc, not a second inference.** Day and night are partitions of
   one prediction set (`pohang01` in `image_path` → night). Both halves therefore
   carry bit-identical predictions and the split introduces no inference variance.
3. **Uncertainty source.** `sigma_ltrb` for all three arms. No `dfl_sigma_ltrb` row
   enters the decision.
4. **Decision metrics** — the five calibration metrics: `d_ece`, `nll`,
   `interval_ece`, `ause`, `aurc`. `map50_95` and `map50` are **reported and are not
   decision inputs**, per D31.
5. **The statistic.** For each decision metric `M` and arm `a`:
   * `D(a) = M_pooled(a) - M_day(a)` — the night pull on that arm.
   * `spread(M) = max_a D(a) - min_a D(a)` — how *differentially* night moves the arms.
   * `sep(M) = max_a M_day(a) - min_a M_day(a)` — the arm separation on the clean half.
   * **`r(M) = spread(M) / sep(M)`.**

   If night were a uniform handicap — the borrowed Phase 1 condition — `spread` is
   zero and `r` is zero. `r` is the whole question, expressed as a fraction of the
   separation the table is trying to report.
6. **Bootstrap.** 2,000 frame-level resamples, seed 0, **paired across arms** — the
   same frame indices for every arm in every draw, so level noise cancels (pairing
   buys 3–16x on this project's metrics). Yields `se` for `sep` and for `spread`.
7. **Magnitude floor, so `r` cannot degenerate into a ratio of noise.** A metric
   enters the verdict only if

   `sep(M) >= max(2 * se_sep(M), 0.02 * mean_a |M_pooled(a)|)`.

   A metric failing this is reported **NO-SIGNAL** — it carries no arm-ranking
   information at all — and is excluded from the verdict rather than counted as a
   pass. The relative form of the floor is used because NLL is unbounded and one
   fixed absolute floor would not transfer across all five metrics.
8. **Bands on `r`,** evaluated only on metrics clearing the floor:
   * **CLEAN** — `r < 0.25`. Night is a near-uniform offset; it cannot reorder the arms.
   * **SUSPECT** — `0.25 <= r < 1.0`. Night's differential effect is a material
     fraction of the separation being reported.
   * **CONTAMINATED** — `r >= 1.0`. Night's differential effect equals or exceeds the
     entire arm separation; the pooled number is not an uncertainty result.
9. **Hard trigger.** If the arm *ordering* on any decision metric differs between day
   and pooled, the verdict is **CONTAMINATED** whatever `r` says. Ordering respects
   direction (lower is better for all five).
10. **Aggregation.** The run's verdict is the **worst band** over the metrics that
    clear the floor. Taking the best, or a mean, would let one clean metric launder
    four dirty ones.
11. **TEST is not consulted.** Nothing is being selected; this is a single
    pre-specified measurement.

## Consequences, fixed in advance

* **CLEAN** → the pooled UQ table stands as an uncertainty result. Publish the day
  column beside it. **No retrain.**
* **SUSPECT** → day-only becomes the primary reporting basis; pooled is retained as a
  secondary with its 46.2% night share disclosed in the caption. **No retrain.**
* **CONTAMINATED** → the pooled table is withdrawn as an uncertainty result and
  day-only becomes the reported basis.

**No outcome of this document licenses a retrain.** Retraining the three VIS arms on
restored labels would need its own pre-registration, and day-only reporting is the
cheaper remedy that should be priced first. This mirrors the night-restore prereg,
where an ALIVE verdict licensed a new pre-registration and nothing else.

## The negative control, and what it can overturn

The same procedure is run on the three **IR** arms (`gauss_ir_seed0_ft`,
`mc_ir_seed0_ft_refit`, `ens_ir_seed{0..4}_ft` — all five IR seeds are local with
`best.pt`, verified). IR labels were never filtered: 0 `.pre_visfilter` backups under
`infrared/`, verified.

Pre-registered reading, so the story cannot be fitted afterwards:

* IR **CLEAN** while VIS is SUSPECT or CONTAMINATED → the distortion tracks the
  labels. The diagnosis in this document holds.
* IR in the **same band** as VIS → the distortion is *not* label-driven; night is
  intrinsically harder to calibrate on, for both sensors. That is a real finding but a
  different one, and it would **not** be repaired by retraining anything. Recording
  this branch now is the point of the control.

## Staging, and an asymmetry that must not be exploited

* **Stage A (local, no server):** sigma-head and MC on VIS, plus the complete IR control.
* **Stage B (needs a server pull):** the VIS ensemble arm, once seeds 1–4 weights land.

Stage A runs over two VIS arms, so `sep` and `spread` are two-arm quantities there.

**Stage A can return CONTAMINATED but it cannot return CLEAN.** Two arms showing
differential night distortion is sufficient to condemn the pooled table; two arms
agreeing says nothing about the third. A CLEAN verdict requires all three. This is
written down because the temptation to upgrade a partial pass is exactly what a
pre-registration exists to block.

## Out of scope

D31 checkpoint selection (unchanged), any re-ranking of the arms published from this
run, every fusion preset, and the shipped `crossmodal26m` benchmark. No weight is
trained and no label is touched.

## Weaknesses, stated in advance

1. **Day-only costs power.** `n` falls 2,232 → 1,200 and every interval widens. A
   CLEAN verdict is bought partly with a smaller sample; the widened `se` is reported
   next to it rather than buried.
2. **This is the most night-heavy view in the project.** At 46.2% night versus full
   val's 18.2%, a CONTAMINATED verdict here does not transfer unchanged to a full-val
   table, and must not be quoted as though it did.
3. **Three arms give the ordering test only six states,** so an ordering match is weak
   evidence by itself. That is why `r` is primary and ordering is a hard-fail trigger
   only.
4. **The arms have structurally different variance floors** — seed spread over n=5, T
   stochastic passes, a learned sigma. `D` can differ between them for reasons that
   have nothing to do with labels. The IR control is the only thing separating those
   two explanations, which is why a VIS-only result is not reportable.
