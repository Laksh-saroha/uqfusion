# U2 — the day/night UQ slice at Stage B, on three night-capable VIS arms

**Written 2026-09-09, before any new cache is built and before any new number is
computed.** Same pre-commit discipline as [`prereg-uq-day-night-slice.md`](prereg-uq-day-night-slice.md)
(`a4f9364`) and its nightfull amendment (`1080f65`): committed ahead of the run so the
rule is verifiable in git history and cannot be edited afterwards.

This is a **new pre-registration, not a third amendment.** The two prior runs were
Stage A — two VIS arms, and by U1's own staging rule a Stage A run *can return
CONTAMINATED but cannot return CLEAN*. Stage B has been unreachable since U1 was
written because no night-capable VIS ensemble existed. It exists now. That changes the
arm set, the denominator of the statistic, and the set of verdicts the run can reach,
which is more than an amendment can carry.

---

## 1. What became true on 2026-09-09

`runs/queue_vis_uq_retrain` completed all seven runs. Every VIS arm now has a
checkpoint cold-started on the **restored** night labels (`b92739202127`), trained with
identical sizing (`yolo26m`, `runs/derived/data_vis_stride2.yaml`, imgsz 640, batch 16,
epochs 100, patience 20):

| arm | checkpoint | epochs | best ep | mAP50-95 |
|---|---|---:|---:|---:|
| sigma-head | `runs/full_scale/gauss_vis_seed0_nightfull` | 37 | 16 | 0.26983 |
| MC-Dropout | `runs/mc_dropout/mc_vis_nightfull` | 97 | 76 | 0.25858 |
| ensemble(n=5) | `runs/ensemble/ens_vis_nightfull_seed{0..4}` | 35–61 | — | 0.27894 / 0.26982 / 0.26808 / 0.26607 / 0.27109 |

All seven `best.pt` files are local and verified present. **This is the first time the
VIS ensemble arm has existed at all** — U1 §Staging recorded that seeds 1–4 lived on
`dgxanode01` and that the archived directories carried `results.csv` with empty
`weights/`. Those server seeds are additionally void: they predate the label restore
(`rebaseline-proposal-2026-09-04.md` §2.3). The five members scored here are new local
cold starts, not a server pull.

## 2. What changes from U1

* **VIS sigma-head:** `gauss_vis_seed0_ft` → `gauss_vis_seed0_nightfull`.
  New cache `runs/cache_uqslice/sigma_vis_seed0_nightfull.pkl`.
* **VIS MC-Dropout:** `mc_vis_seed0_ft_refit` → `mc_vis_nightfull`.
  New cache `runs/cache_uqslice/mc_vis_nightfull.pkl`. This arm was explicitly left
  unchanged by the nightfull amendment and is the reason that run stayed at Stage A.
* **VIS ensemble(n=5):** newly present. New cache
  `runs/cache_uqslice/ens_vis_nightfull.pkl`.
* **Consequence for the statistic:** `sep` and `spread` become **three-arm** quantities
  on the VIS side, matching the IR control for the first time. The ordering test gains
  states (three arms, six orderings, versus two arms and two).
* **Consequence for the verdict space:** this is **Stage B**, so **CLEAN is reachable**.
  It has never been reachable before on VIS.

## 3. What does not change, and is not to be touched

The **IR control** (`gauss_ir_seed0_ft`, `mc_ir_seed0_ft_refit`,
`ens_ir_seed{0..4}_ft`) and its three existing caches, the five decision metrics
(`d_ece`, `nll`, `interval_ece`, `ause`, `aurc`), `sigma_ltrb` as the uncertainty
source, the 2,000-draw paired frame bootstrap at seed 0, the magnitude floor
`sep >= max(2*se_sep, 0.02*scale)`, the bands (CLEAN < 0.25 <= SUSPECT < 1.0 <=
CONTAMINATED), rule 10 worst-band aggregation, and rule 11 (TEST is not consulted).

**Rule 9 keeps its dual reporting.** The floored form and the as-registered unfloored
form are both computed and both printed, and the as-registered verdict is never
dropped. That carry-forward is not renegotiated here.

`map50_95` and `map50` remain **reported and not decision inputs**, per D31.

**Cache build command is U1's rule 1 verbatim**, so `verify()`'s cross-arm settings
assertion still binds:
`build_cache.py --split val --images-list runs/derived/paired_val_vis.txt --imgsz 640 --conf 0.001`.
Nothing outside `runs/cache_uqslice/` and `runs/eval/uq_day_night_slice_u2.md` is
written.

## 4. A diagnostic that is not a decision input

Before scoring, the night detection count per new checkpoint over the 1,032 `pohang01`
frames is recorded in the report. It exists to confirm the retrain did the thing it was
run for — `mc_vis_seed0_ft_refit` emitted **28** detections there, `gauss_vis_seed0_ft`
**2**, `gauss_vis_nightfull` **8,391**. **It enters no band and moves no verdict.** It
is stated here in advance so it cannot later be presented as though it had.

## 5. Disclosed weaknesses, fixed in advance

1. **The VIS arms are single-stage; the IR control arms are two-stage `_ft`.** Under U1
   both sides were 10-epoch mosaic-off fine-tunes. All three VIS arms are now cold
   starts with no `_ft` continuation — a decision recorded in
   `runs/queue_vis_uq_retrain/queue.json` and reversed once during queue construction.
   This makes the VIS side *more* internally symmetric than U1 (all three arms share a
   recipe) and the VIS-vs-IR comparison *less* so. The IR control's job is to answer
   "does the uncorrupted modality show the same distortion", and a training-stage
   difference is a live alternative explanation for a VIS/IR divergence. **A divergence
   this run finds cannot be attributed to labels without confronting that.**
2. **The three VIS arms early-stopped at very different epochs** — 37, 97, 35–61. U1's
   three arms were matched at 10 epochs. U1 weakness 4 already warned that the arms have
   structurally different variance floors; unequal training length compounds it. `D(a)`
   can differ between arms for reasons that have nothing to do with labels.
3. **One seed per arm on sigma-head and MC.** The ensemble arm's five members are five
   seeds of one arm, not five replicates of the comparison. No interval here covers
   seed-to-seed variation on the other two arms.
4. **Frame-level bootstrap on a 10 Hz recording overstates independence**
   (`TODO-2026-09-09-architecture-review.md` R-A3, open). Every `se` here inherits that.
   It is not repaired in this run, and the intervals should be read as narrower than the
   truth.
5. **This is still the most night-heavy view in the project** (46.2% night vs full val's
   18.2%). A verdict here does not transfer unchanged to a full-val table.
6. **Night has one run.** `pohang01` is the entire night side, so the between-run night
   component cannot be estimated at all.

## 6. Why the open evaluator findings do not gate this run

`TODO-2026-09-09-architecture-review.md` workstream A (F03 AP interpolation convention,
F04 bootstrap fast/reference divergence) is P1 and argues it precedes everything. It
does not bind the decision here, for a checkable reason: **none of the five decision
metrics routes through the AP path.** `d_ece`, `interval_ece` and `gaussian_nll` are
computed directly from per-detection arrays in `src/uqfusion/eval/metrics.py`, and
`ause`/`aurc` come from `sparsification` over per-detection uncertainty and risk.
`np.interp` on the precision envelope (`apmetrics.py:74`, `matching.py:169`) and the
`presort`/`ap_from_parts` class-set divergence live in `map50_95`, which supplies only
the **reported, non-decision** mAP column.

**Therefore:** the verdict stands independently of workstream A; the `map50_95` column
in this report carries the F03/F04 bias and is labelled as such. If workstream A later
changes an AP convention, this report's bands do not move and its mAP column is
rescored.

## 7. Consequences, fixed in advance

* **CLEAN** — reachable for the first time. The pooled UQ table stands as an uncertainty
  result; publish the day column beside it.
* **SUSPECT** — day-only becomes the primary reporting basis; pooled retained as
  secondary with its 46.2% night share disclosed in the caption.
* **CONTAMINATED** — the pooled table is withdrawn as an uncertainty result; day-only
  becomes the reported basis.

**No outcome licenses a retrain, and no outcome licenses a detector swap in the shipped
fusion pipeline.** The `crossmodal26m` preset, the night-veto axis (closed,
`night-veto-axis-closed-2026-09-04.md`), and the re-baseline priced in
`rebaseline-proposal-2026-09-04.md` §4 are **out of scope here** and are decided on
their own terms with their own registration. A friendly verdict in this run is not an
argument for spending that budget.

## 8. The negative control's reading, carried forward unchanged

IR labels were never filtered (0 `.pre_visfilter` backups under `infrared/`, verified).
The three registered branches stand as amended on 2026-09-03:

* VIS dirty, IR clean → the distortion tracks the labels.
* Same band → not label-driven; night is intrinsically harder to calibrate on, and
  retraining repairs nothing.
* **IR dirtier than VIS** → cannot be label contamination, since IR labels were never
  filtered. A finding about night, not about labels.
