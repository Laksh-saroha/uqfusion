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
