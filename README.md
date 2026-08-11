# uqfusion — Uncertainty-Aware VIS–IR Fusion for Maritime Object Detection

Maritime object detector that estimates, in a single forward pass, how much its own predictions can be trusted — and uses that per-frame reliability to adaptively fuse visible (RGB) and thermal-infrared imagery, down-weighting whichever sensor is degraded by fog, glare, darkness, or thermal crossover.

UG Research Fellowship project, Thapar Institute of Engineering and Technology (Laksh Saroha; mentor Dr. Sandeep Mandia).

**Start here for context:**
- [`scope.md`](scope.md) — source of truth: motivation, architecture, datasets, evaluation plan.
- [`docs/plan-2026-07-07-kickoff.md`](docs/plan-2026-07-07-kickoff.md) — approved architecture review + backbone selection plan.
- [`progress.md`](progress.md) — current phase, decision log, open questions, run log.
- [`HOW_TO_RUN.md`](HOW_TO_RUN.md) — **per-phase runbook**: exact commands for every phase, dev machine and GPU server.

## Repository layout

```
config.yaml            <- the ONE file to edit when moving machines
requirements.txt       <- pinned dependencies (see header for lock-file procedure)
src/uqfusion/          <- project package (installed editable)
docs/                  <- approved plans / design memos
data/                  <- datasets (gitignored; layout below)
runs/                  <- training & eval outputs (gitignored)
```

## Setup — identical on every machine

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt -e .
python -m uqfusion.config          # smoke test: config loads & paths resolve
```

On the GPU training server, additionally freeze the exact environment once:

```bash
pip freeze > requirements.lock.txt   # commit it (scope.md §11.1: which commit produced which result)
```

## Portability rules (enforced project-wide)

1. **No absolute paths anywhere in code.** Every machine-specific path and every hyperparameter goes through `config.yaml`; relative paths resolve against that file's location.
2. Upload repo + datasets to a new machine → edit the `paths:` block (and `device:`) in `config.yaml` → run. Nothing else changes.
3. Every trainable component has a tiny-subset smoke path (`smoke:` block in the config) that runs on CPU in minutes — used before committing any GPU time.

## Expected dataset layout (scope.md §5)

```
data/
├── pohang/                      # Pohang Canal + PoLaRIS annotations (primary)
│   ├── data_vis.yaml            # Ultralytics dataset yaml — visible stream
│   ├── data_ir.yaml             # Ultralytics dataset yaml — infrared stream
│   └── <per-modality YOLO-format images/ and labels/ trees, as referenced
│        by the two yamls; runs pohang00–pohang04>
└── mit_marine/                  # MIT Sea Grant Marine Perception (onboarded Phase 2+)
    └── <YOLO-format tree after annotation conversion; yamls added then>
```

The contract the code relies on: `data_vis.yaml` and `data_ir.yaml` exist, are valid Ultralytics dataset files, and point at their own modality's images/labels within `data/pohang/`. The exact inner tree follows the PoLaRIS release and is confirmed against the actual download (progress.md, open question A1-2). Labels are YOLO `.txt` (`class cx cy w h`, normalized); classes: `ship`, `buoy`.

## Dataset preparation — Pohang (local edits)

The Pohang VIS/IR release needed three edits before it was trainable. Full detail
(hashes, manifests, exact counts, reversal commands) lives in
[`docs/dataset-changes-2026-07.md`](docs/dataset-changes-2026-07.md); summary here.
All edits below were run on the **local** copy first and verified (`audit_split.py`
+ balance/coverage gates) before being reproduced on the server — see that doc's
"Server status" for what's applied where.

**1. IR downscaling/letterbox.** IR ships at 640×512; VIS is native 640×640. IR was
**letterboxed** (not stretched) to 640×640 so both modalities share one `imgsz` —
stretching would distort small-object aspect ratios. Labels were transformed to
match, out-of-bounds boxes clipped, and full-resolution IR originals archived to
`infrared_orig/` (kept, not deleted — `imgsz` is the small-object recall lever, so
the unscaled source stays available; the archive now lives on `D:\Datasets\Pohang labels\`).
Script: `scripts/prepare_pohang.py`.

Both modalities pad with **114** (gray). Worth knowing downstream: `filter_night_boxes.py`
detects "padding" with `PAD_LEVEL = 4`, i.e. near-black, so it does **not** exclude
these gray bars — see the caveat under the night filter below.

**2. Splits — two rounds, both list-only (no images/labels moved).**
- *Round 1* (`prepare_pohang.py`, 2026-07-10): the delivered split was
  frame-interleaved (train N / val N+1) — with 10 Hz video that leaks
  near-duplicate frames across train/val/test. Re-split into **contiguous per-run
  ordinal blocks** (fixed positional windows, ~80/90% cut points), same ordinals
  used for both modalities so VIS/IR pairs and L/R stereo pairs never split across
  sets, plus a per-run guard band at each boundary. `audit_split.py` (leakage:
  cross-split duplicates + temporal buffer) went from FAIL to PASS.
- *Round 2* (`scripts/resplit_balanced.py`, 2026-07-14): Round 1's split passed
  leakage but **failed balance** on the server — val came out 77% buoy vs 5%
  global class share, 1.35 vs 8.10 boxes/frame, because fixed positional windows
  sample one mission phase. A backbone selected on that val set would be selected
  on the wrong signal, so it was replaced with an **interleaved K-block split**:
  cut each run's shared VIS+IR timeline into K equal blocks, assign in a repeating
  cycle of 10 (block%10==4 → val, ==9 → test, else train → 80/10/10 by
  construction), guard bands dropped at every train/eval boundary per modality, K
  raised until leakage + balance + coverage gates all pass. This is the split
  currently used for the grid (verified server-side by a uniform
  `split_fingerprint` across the 2026-07-31 run). Round 1's lists are preserved as
  `*.txt.pre_resplit` (and identically as `*.txt.orig` from the `./`-prefix fix below).
- Housekeeping fix in between (`scripts/fix_split_lists.py`, 2026-07-12): Ultralytics
  only resolves a list line relative to the txt file's folder if it starts with
  `./`; bare lines like `images/pohang04/x.png` resolved against cwd instead and
  failed to load. Rewrote train/val/test.txt per modality with the `./` prefix —
  no semantic change to the split itself, backup `*.txt.orig`.

**3. Night-box visibility filter** (`scripts/filter_night_boxes.py --cut-dark
pohang01:100`, 2026-07-15): many `pohang01` VIS frames are night footage where
ship/buoy boxes are annotated (geometrically correct) but **not visible in the
pixels** — below the sensor's usable light level. Training on them teaches the
detector to hallucinate objects in dark water. The filter drops **all TRAIN-only
VIS** boxes in any `pohang01` frame whose content-median luminance is < 100
(0–255); **val/test are never touched** — eval labels are the "VIS misses, IR
catches" question this whole benchmark measures, and editing them would be
self-grading. Frames left with zero boxes are kept as background images (helps
precision). Effect: 17,502 of 96,275 scanned train label files emptied, 132,688
boxes dropped (126,948 ship / 5,740 buoy). Reversible: every edited label backed
up once to `*.pre_visfilter`, `--restore` undoes, manifest at
`runs/visfilter/visfilter_manifest.json` records the cut spec + before/after
content hashes.

> **Caveat on what "content-median" measured.** `PAD_LEVEL = 4` treats only near-black
> pixels as padding, but the VIS letterbox pads with 114, so 47% of every 640 VIS frame
> sits at exactly 114 and dominates the median. Every day-run frame therefore reports a
> median of exactly 114.0 (pohang00/02/03/04: min = max = 114 across 77,449 frames);
> only pohang01 reports real values. The rule that actually ran was *"pohang01 frames
> whose content ~95th-percentile luminance < 100"*. The **outcome is sound** — the
> threshold was eyeball-calibrated against these very numbers, and 17,502 frames were
> confirmed by spot-check — but the statistic is not what its name suggests, and the
> **threshold does not transfer to unpadded images**. See
> [`docs/dataset-changes-2026-07.md`](docs/dataset-changes-2026-07.md) for the
> measurement and what it means for the full-resolution tree.

**4. Full-resolution twin** (`scripts/prepare_pohang_fullres.py`, 2026-08-01):
`D:\Datasets\Pohang_dataset_full\` holds the same dataset at **native** resolution
(VIS 2048×1080, IR 640×512, labels in native coordinates) carrying the **same split
and the same night filter**, so a higher-`imgsz` run is directly comparable against
the 640 grid. Images are hardlinked from the archives, so it costs 0 new image bytes
and the sources stay intact. The split lists are copied verbatim and the filter's
file list is **ported, not recomputed** (see the caveat above — the threshold does not
survive the loss of padding). Its own detailed doc travels with the data at
`D:\Datasets\Pohang_dataset_full\PREPARATION.md`.

Every benchmark CSV row is stamped with a `split_fingerprint` (hash of the
train+val frame ids) and a `classes` tag (`grid.py`, commit `6e24167`) so a
re-split or a filter change can never be silently averaged against older rows —
resume logic refuses a mismatch instead of skipping it.

## Compute model

This repo is developed and smoke-tested on CPU; **all real training runs on a GPU Jupyter server** (H100). Phase 1's first GPU action is a 1-epoch timing dry-run to calibrate the benchmark budget before committing the full grid (plan §C5).

## Phase 1 — backbone benchmark

Data contract and the exact server run order: [`dataset_requirement.md`](dataset_requirement.md). In short: `smoke_env` → `smoke_benchmark` → **`audit_split` (must PASS)** → `make_stride_subset` → timing dry-run → full grid → `measure_fps` → `make_table1`. The grid is resume-safe: re-running skips (variant, seed) pairs already in the results CSV.

## Phase 2 — single-pass UQ (core contribution)

`src/uqfusion/uq/`:

- `gaussian.py` — σ² branch (`cv4`) added **alongside** the untouched DFL head by in-place conversion of a loaded model (no Ultralytics fork), + β-NLL loss with warm-up/ramp. Gradient policy: detached features + detached μ by default, so the deterministic detector trains bit-identically to baseline (`gaussian:` block in config to relax). Also emits DFL-derived uncertainty (scope §7.2 option (a)) from the same model at inference.
- `train_gaussian.py` — stock Ultralytics training loop via a thin custom trainer (adds the 4th loss item + warm-up epoch sync).
- `infer.py` — `UQPredictor`: one forward pass → boxes + per-box σ (rides through NMS as extra channels) + pooled neck features; the producer for cached-prediction ablations.
- `mahalanobis.py` — frame-level OOD scorer (Ledoit-Wolf), fit on clean features only.
- `reliability.py` — scope §6.4 verbatim: size-normalized `u_i`, confidence-weighted `U_box`, calibrated `O`, combination rules, empty-frame fallback, temporal smoothing, fusion weights + absolute `R_sys`.
- `fusion.py` — IR→VIS homography + reliability-weighted WBF.

Smoke gates (CPU, synthetic heteroscedastic data — run before any GPU time):

```bash
python scripts/smoke_gaussian.py      # §18-2 gate: trains, warm-up engages, σ non-degenerate & tracks noise
python scripts/smoke_uq_pipeline.py   # OOD separation -> reliability gate -> WBF fusion, end-to-end
```

## Phase 3 — baselines + evaluation harness

`src/uqfusion/uq/`: `mc_dropout.py` (pre-fixed dropout insertion + T-pass predictor), `ensemble.py` (M seed replicates via the grid runner), `clustering.py` (the one shared cross-pass matching protocol). `src/uqfusion/eval/`: `cache.py` (prediction caches — every downstream number reads from these), `corruptions.py` (6 albumentations conditions, seed-stamped), `metrics.py` (pre-registered: D-ECE, interval-ECE/coverage, NLL, AUSE/AURC, OOD AUROC, local COCO mAP), `learned_gate.py` (§7.5 upper bound), `fusion_eval.py` (system comparison + §6.4 gate ablations). CLIs: `train_mc_dropout` / `train_ensemble` / `build_cache` / `evaluate_uq` / `ablate_gate`.

```bash
python scripts/smoke_phase3.py        # gate: baselines, caches, corruptions, metrics, learned gate, fusion eval
```

## License note

Built on Ultralytics (AGPL-3.0) — the fork/extension and this repo inherit AGPL-3.0 obligations for public distribution (confirmation pending, progress.md question A3-10).
