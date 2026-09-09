# TODO 2026-08-26 — Phase 1 class-set mismatch: can ship-only weights be re-trained into a 2-class benchmark?

**Status: analysis complete; DECIDED 2026-08-27 — §4.2 taken, and widened.** Laksh's call:
cold restart from COCO on **all 31 variants × 3 seeds**, VIS, both classes, at train stride 4,
queued on dgxanode01 behind the ensemble queue. Setup, sizing, chain and tracker in
[`vis-benchmark-stride4-2026-08-27.md`](vis-benchmark-stride4-2026-08-27.md). The warm-start
route of §2 stays rejected. The zero-GPU mitigation of §4.1 (per-class AP everywhere) is
**not** superseded — it is still what makes the ship-only Phase 1 table honest, and its open
sub-task (do the comparison scripts slice by class?) is still open.

Raised by Laksh 2026-08-26:

> "If I were to use the Phase 1 VIS benchmark weights and train them on both classes (the
> benchmark was trained only on ships), can I treat it as a new benchmark for the paper?"

**Recommendation: no — not as a replacement for Table 1.** The cheap mitigation (per-class AP,
ship-vs-ship comparison) is already on record and costs no GPU time. A genuinely 2-class
Table 1 must be a **cold restart from COCO**, reported as a *second* table, not a replacement.

This is the source record for that decision. It supersedes nothing; it extends
`handoff-2026-08-21-full-scale.md` §3 (the original scope finding) and `TODO-improvements.md`
§D-3 (the pre-registration rule).

---

## 1. What was verified

All measured against the working tree, not inferred.

| Claim | Verified how |
|---|---|
| Phase 1 is ship-only, all 93 rows | `results.csv` `classes` column: `Counter({'0': 93})` |
| All 93 checkpoints survive locally | `find phase1_benchmark/runs -name best.pt` → 93 |
| The checkpoints carry a **2-class head** | `torch.load('main_yolo26s_seed0/weights/best.pt')` → `nc = 2`, `names = {0: 'ship', 1: 'buoy'}`, `model.yaml['nc'] = 2` |
| Pilot machine is gone | `results.csv` `trained_on`: `server_wiped` × 66 |
| Phase 1 total cost | 647.1 GPU-h across 93 rows (main 266.8 h / 27 rows, pilot 380.3 h / 66 rows) |
| Early stop varied per variant | epochs_trained 24–61, mean 35.5 (main) |

The head finding is load-bearing: `--classes 0` in `src/uqfusion/bench/grid.py:284` filters
the **labels** in the dataloader, but the model is still built from the data yaml, which
declares `names: {0: ship, 1: buoy}`. So `nc=2` throughout.

---

## 2. Why the warm-start route is not a benchmark

### 2.1 Unequal warm starts, correlated with the thing being compared

Phase 1 early-stopped per run (`patience=20`), so stage-1 exposure differs by more than 2×
across variants:

```
yolo12l   epochs [34, 50, 34]      yolo26l   epochs [28, 28, 26]
yolo12m   epochs [46, 61, 45]      yolo26m   epochs [29, 26, 33]
yolo12s   epochs [31, 38, 32]      yolo26n   epochs [36, 43, 36]
yolo12x   epochs [29, 35, 24]      yolo26s   epochs [31, 31, 42]
                                   yolo26x   epochs [28, 32, 50]
```

`yolo12m` enters stage 2 from 45–61 ship-only epochs; `yolo26l` from 26–28 — a nuisance
variable correlated with architecture, the exact variable Table 1 exists to isolate. Per-variant
seed sd in Phase 1 is **0.005–0.011**, so the asymmetry need not be large to be the size of
the measured effect.

### 2.2 The buoy channel is suppressed, not naive

Because `nc=2` and buoy labels were filtered out, the class-1 logit was trained to fire low
**on buoy pixels** for 24–61 epochs. Stage 2 must un-learn that, and the amount of un-learning
scales with stage-1 length — so §2.1's asymmetry lands directly on the class being added. A
COCO cold start has a freshly initialised head and no such handicap.

### 2.3 Disqualifier: 66 of 93 rows would launder a known contamination

The `pilot` campaign ran on partition `f0220e716277`, unrecoverable.
`docs/phase1-pilot-grid.md` already records *"Possible train/val contamination,
unquantifiable"* — pilot val begins at `pohang00_L_006750`, main val at `006767`, 17 frames
later, both cutting val as contiguous blocks.

Fine-tuning those checkpoints and scoring on the current val (`682dbe9f0f05`) would **launder**
that: the table would present as one clean split end-to-end while 22 of 31 variants started
from weights that may have seen current-val frames. Today the contamination is disclosed and
confined to rows carrying a `pilot_` prefix.

**Consequence:** the warm-start route can only honestly cover the **9 `main` variants** — a far
smaller table than the 31-variant Table 1.

### 2.4 Compute does not argue for it

Warm-starting only saves time if the stage-2 schedule is shortened — and shortening it is
precisely what stops it being a benchmark. Run stage 2 at the full 100/patience-20 budget and
it costs cold-start price with a confound attached.

---

## 3. The larger risk is already written down

`TODO-improvements.md` §D-3 pre-commits:

> **Class-set choice must precede the results.** Dropping buoys moves VIS 0.2580 → 0.2141 and
> IR 0.0676 → 0.1351. Picking the framing after seeing that is not defensible; report
> per-class AP always.

Deciding to build a new benchmark **now**, with those numbers already visible, is what that
rule exists to prevent. No amount of retraining repairs it; only reporting both class sets does.

---

## 4. Recommended path

### 4.1 Do this (zero GPU cost) — **the actual mitigation**

Already specified in `handoff-2026-08-21-full-scale.md` §3: compare **ship AP to ship AP**, and
report per-class AP everywhere. `src/uqfusion/eval/apmetrics.py` already computes a `per_class`
dict in both its reference and fast/bootstrap paths. VIS buoy is **not** degenerate —
`mc_vis_seed0/best.pt` measured ship 0.284 / buoy 0.198 — so the 2-class number is honest,
merely macro-averaged over a class Phase 1 never had.

**Open sub-task, unchanged from 08-21 §3 and still unverified:** confirm the downstream
comparison scripts actually slice by class. Candidates by filename, still unread:
`scripts/eval_risk_coverage_fixed_gt.py`, `scripts/refit_maha_dayonly.py`. Est. ~1 day, no GPU.

### 4.2 If a 2-class Table 1 is genuinely wanted

Cold restart from COCO weights. Not a fine-tune.

- 9 `main` variants × 3 seeds, split fingerprint `682dbe9f0f05`, imgsz 640, 100 epochs /
  patience 20, `classes` **unset**.
- Est. ~270 GPU-h ≈ 11 days on dgxanode01 (A100 MIG 3g.40gb, ~1.23× the laptop, no concurrent
  runs — see `project-server-dgxanode01`).
- Publish as a **second** table beside the ship-only one, with both class sets reported, so
  §3's rule is satisfied.
- The 22 `pilot` variants cannot join at any price: their split is gone.

### 4.3 Do not do

- Fine-tune any `pilot_*` checkpoint and score it on the current val (§2.3).
- Present a warm-started table as "we trained these detectors on Pohang".
- Shorten the stage-2 schedule to make the warm-start route affordable (§2.4).

---

## 5. Adjacent inconsistency spotted while checking — **open, unresolved**

The IR benchmark launched 2026-08-26 (`runs/queue_ir_benchmark_stride4`) trains on
`runs/derived/data_ir_stride4.yaml`, which declares `names: {0: ship, 1: buoy}` —
**2-class**. But A-1 resolved the IR *fusion* path to `nc=1` ship-only
(`data_ir_shiponly.yaml`) by a pre-registered rule (`TODO-2026-08-20-full-scale.md` lines
57–94), on the grounds that IR buoy AP is 0.00022 against 29,131 emitted buoy detections for
596 GT boxes.

So the IR benchmark and the IR fusion model will not share a class set, and the IR benchmark's
headline mAP is macro-averaged over a class the sensor cannot see — the same understatement §3
warns about, roughly halving the number (IR 0.0676 pooled vs 0.1351 ship-only).

**Not a defect in the running queue** — its own note calls it "a ranking sweep, not a final
recipe", and a ranking may well be class-set-insensitive. But if any row reaches a paper table,
the class set has to be stated and per-class AP reported. Decide before the sweep finishes.
