# uqfusion — Uncertainty-Aware VIS–IR Fusion for Maritime Object Detection

Maritime object detector that estimates, in a single forward pass, how much its own predictions can be trusted — and uses that per-frame reliability to adaptively fuse visible (RGB) and thermal-infrared imagery, down-weighting whichever sensor is degraded by fog, glare, darkness, or thermal crossover.

UG Research Fellowship project, Thapar Institute of Engineering and Technology (Laksh Saroha; mentor Dr. Sandeep Mandia).

**Start here for context:**
- [`scope.md`](scope.md) — source of truth: motivation, architecture, datasets, evaluation plan.
- [`docs/plan-2026-07-07-kickoff.md`](docs/plan-2026-07-07-kickoff.md) — approved architecture review + backbone selection plan.
- [`progress.md`](progress.md) — current phase, decision log, open questions, run log.

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

This repo is developed and smoke-tested on CPU; **all real training runs on a GPU Jupyter server**. Phase 1's first GPU action is a 1-epoch timing dry-run to calibrate the benchmark budget before committing the full grid (plan §C5).

## License note

Built on Ultralytics (AGPL-3.0) — the fork/extension and this repo inherit AGPL-3.0 obligations for public distribution (confirmation pending, progress.md question A3-10).
