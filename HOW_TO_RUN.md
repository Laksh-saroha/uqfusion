# HOW TO RUN — uqfusion, phase by phase

Two machines, one rule:

| Machine | Purpose | What differs |
|---|---|---|
| Dev machine (no GPU) | scaffolding, correctness, CPU smoke tests | nothing to edit |
| **GPU server (H100)** | all real training and evaluation | edit the `paths:` block (and optionally `device:`) in [`config.yaml`](config.yaml) — **nothing else** |

Every command below runs from the repo root. Everything lands under `runs/` (gitignored). Datasets must match [`dataset_requirement.md`](dataset_requirement.md).

---

## What to upload to the server (per phase)

The **repo** (git clone/pull) carries all code, `config.yaml`, scripts, and `yolov8n.pt` (smoke). The **dataset is separate** (gitignored) — upload the prepared Pohang trees as each phase needs them. The dataset was prepared locally by `scripts/prepare_pohang.py` (IR letterboxed to 640×640; both modalities re-split into leakage-free per-run blocks; audit PASSES). COCO pretrained weights for the 6 variants auto-download on first run (needs internet on the server); if the server is offline, pre-stage them.

**Placement / paths (resolves OQ-9):** put the prepared dataset at `<data_root>/pohang/` so that `<data_root>/pohang/data_vis.yaml` and `.../data_ir.yaml` exist (that's what `config.yaml` `datasets.pohang.*` points to). Then set the single `path:` line in each yaml to the server location of `pohang/visible` and `pohang/infrared` (currently an absolute dev-machine path). Nothing else in the yamls changes — the `train/val/test.txt` entries are relative to `path:`.

| Phase | New data to upload | Size | Cumulative on server |
|---|---|---|---|
| Setup + smoke (§0–1) | nothing beyond the repo | — | repo only |
| **Phase 1 — VIS grid** (§2 steps 1–5) | `pohang/visible/` — `images/<run>/`, `labels/<run>/`, `train\|val\|test.txt`, `data_vis.yaml` | ~33 GB | repo + visible |
| **Phase 1 — IR confirm** (§2 step 6) | `pohang/infrared/` — same structure + `data_ir.yaml` | ~4.5 GB | + infrared |
| **Phase 2** (§3) | — (reuses visible + infrared) | — | same |
| **Phase 3** (§4) | — (corruptions are generated on the fly from the images) | — | same |
| **Phase 4** (§5) | `pohang/meta/<run>/calibration/` (per-run intrinsics+extrinsics, for the IR→VIS homography) + `pohang/paired/*.csv` (VIS↔IR pairs); MIT dataset when ready | ~0.3 GB + MIT | + calibration + pairs + MIT |

**Do NOT upload** (keep as local backup only): `infrared_orig/` (~4.5 GB, the pre-letterbox 640×512 IR originals — needed only to re-export IR) and the full-res VIS originals (the `imgsz`>640 small-object lever, archived off the dev machine; upload only if you later train at 960+ from originals). See [`Pohang_dataset/IR_PREPROCESSING.md`](Pohang_dataset/IR_PREPROCESSING.md) for the IR conversion record.

---

## 0. One-time setup (per machine)

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt -e .
python -m uqfusion.config            # config loads, paths resolve
python scripts/smoke_env.py          # imports OK; CUDA must be True on the server;
                                     # all 6 benchmark variants build
```

On the **server only**, freeze the exact environment once and commit it:

```bash
pip freeze > requirements.lock.txt
```

## 1. Smoke suite — run before ANY GPU time

Order matters (later smokes reuse earlier artifacts). All CPU, minutes each.

```bash
python scripts/smoke_benchmark.py    # Phase 1 harness: grid -> val -> CSV -> FPS -> Table 1
python scripts/smoke_gaussian.py     # Phase 2 §18-2 gate: σ head trains, warm-up engages, σ tracks noise
python scripts/smoke_uq_pipeline.py  # Phase 2: OOD -> reliability -> gated WBF end-to-end
python scripts/smoke_phase3.py       # Phase 3: baselines, caches, corruptions, metrics, gate, ablation
```

Expected final lines: `SMOKE OK`, `GAUSSIAN SMOKE OK`, `UQ PIPELINE SMOKE OK`, `PHASE3 SMOKE OK`. A red smoke = stop; nothing downstream is trustworthy.

## 2. Phase 1 — backbone benchmark → Table 1 (server)

```bash
# 1. Verify the split is leakage-free (MUST exit 0; report lands in runs/audit/).
#    The split was already re-built + audited GREEN on the prep machine, and the
#    audit is deterministic (filename-based), so expect an instant PASS. Its job
#    here is a path/upload sanity gate: a FAIL means config paths or the upload
#    are wrong, not the split. It reads filenames only (does not stat every
#    image), so also confirm counts, e.g. `wc -l pohang/visible/train.txt` vs
#    images on disk, to catch a partial upload.
python scripts/audit_split.py

# 2. Stride-subsample the training frames (approved A2-6); prints the derived yaml path
python scripts/make_stride_subset.py --data vis

# 3. (optional) 1-epoch timing dry-run to sanity-check wall time / batch
python scripts/run_benchmark.py --data runs/derived/data_vis_stride2.yaml --variants yolov8s --seeds 0 --epochs 1

# 4. Full grid: 6 variants x 3 seeds (resume-safe — rerun continues after a crash)
python scripts/run_benchmark.py --data runs/derived/data_vis_stride2.yaml

# 5. Consolidate every run of both campaigns into phase1_benchmark/ with one CSV.
#    Idempotent; --plan prints what it would move without touching anything.
#    The CSV carries best_epoch / last_epoch / patience_gap / stop_reason and an
#    `admissible` flag — filter on that, not by hand. Rows are keyed by
#    <grid>_<variant>_seed<n>; `grid` separates the 2026-08 campaign from the
#    2026-07 pilot, which ran on a different split (docs/phase1-pilot-grid.md).
python scripts/consolidate_phase1.py --plan
python scripts/consolidate_phase1.py --execute

# 6. FPS protocol (batch=1, fp32+fp16, on this GPU) and Table 1.
#    MUST run on a CUDA build of torch — the script refuses a CPU device, because a
#    CPU timing is not the number Table 1 wants. The repo .venv on the laptop is
#    torch+cpu, so use the interpreter that has the GPU build and put the package on
#    the path instead of installing into it:
PYTHONPATH=src python scripts/measure_fps.py       # -> phase1_benchmark/fps.csv
python scripts/make_table1.py                      # -> runs/benchmark/table1.md

# 7. IR confirmation grid for the top-2 variants (plan C4/D8)
python scripts/make_stride_subset.py --data ir
python scripts/run_benchmark.py --data runs/derived/data_ir_stride2.yaml \
    --variants <top1> <top2> --out-csv runs/benchmark/ir_results.csv --run-prefix ir
python scripts/make_table1.py --results-csv runs/benchmark/ir_results.csv --out runs/benchmark/table1_ir.md
```

**Send back:** `runs/benchmark/table1.md`, `table1_ir.md`, and the audit report → selection memo + §7.2 resolution get written, phase gate closes.

> **The VIS grid is finished.** Its consolidated record is
> `phase1_benchmark/results.csv` — 93 rows, 31 variants, two campaigns. Start from that
> folder's `README.md` for the layout and column dictionary,
> `docs/phase1-experimental-record.md` for provenance and known defects, and
> `docs/phase1-pilot-grid.md` before any comparison that crosses the `grid` column.
> Steps 1–4 above describe how it was produced; do not re-run them against the existing
> record.

## 3. Phase 2 — Gaussian σ² model on real data (server)

Replace `<winner>` with the Phase 1 selection (grid/loss knobs: `gaussian:` block in config).

```bash
# Gaussian-head training (σ branch rides on the untouched detector)
python scripts/train_gaussian_model.py --data runs/derived/data_vis_stride2.yaml --variant <winner> --seed 0
python scripts/train_gaussian_model.py --data runs/derived/data_ir_stride2.yaml  --variant <winner> --seed 0 --name gauss_ir_seed0

# §12.1 parity check (A4-11's condition): the SAME variant WITHOUT σ at identical settings
# — reuse the Phase 1 grid row if it exists; otherwise:
python scripts/run_benchmark.py --data runs/derived/data_vis_stride2.yaml --variants <winner> --seeds 0
```

**Check:** `runs/gaussian/<name>/results.csv` — `train/nll_loss` is 0 for `warmup_epochs`, then activates; val mAP within seed noise of the parity row (guaranteed-by-construction with the default detached-gradient config, but verify anyway).

## 4. Phase 3 — baselines, caches, calibration, gate (server)

```bash
# 1. Baselines (deterministic rows come free from their results.csv)
python scripts/train_mc_dropout.py --data runs/derived/data_vis_stride2.yaml --variant <winner> --seed 0
python scripts/train_ensemble.py   --data runs/derived/data_vis_stride2.yaml --variant <winner>   # M=5 seeds from config

# 2. Prediction caches — every downstream number comes from these (plan B5-2).
#    One clean cache per source + corrupted caches for calibration-under-shift.
#    ANTI-LEAKAGE (plan B5-5): tuning caches and final-test caches must use
#    different --corrupt-seed values; the seed is stamped into the cache meta.
G=runs/gaussian/gauss_<winner>_seed0/weights/best.pt
M=runs/mc_dropout/mc_<winner>_seed0/weights/best.pt
E="runs/benchmark/runs/ens_<winner>_seed0/weights/best.pt runs/benchmark/runs/ens_<winner>_seed1/weights/best.pt \
   runs/benchmark/runs/ens_<winner>_seed2/weights/best.pt runs/benchmark/runs/ens_<winner>_seed3/weights/best.pt \
   runs/benchmark/runs/ens_<winner>_seed4/weights/best.pt"

python scripts/build_cache.py --source gaussian --weights $G --data vis --split train --out runs/cache/gauss_vis_train_clean.pkl
python scripts/build_cache.py --source gaussian --weights $G --data vis --split val   --out runs/cache/gauss_vis_val_clean.pkl
python scripts/build_cache.py --source gaussian --weights $G --data vis --split val   --corrupt fog      --corrupt-seed 1 --out runs/cache/gauss_vis_val_fog.pkl
python scripts/build_cache.py --source gaussian --weights $G --data vis --split val   --corrupt lowlight --corrupt-seed 1 --out runs/cache/gauss_vis_val_lowlight.pkl
python scripts/build_cache.py --source mc       --weights $M --data vis --split val   --out runs/cache/mc_vis_val_clean.pkl
python scripts/build_cache.py --source mc       --weights $M --data vis --split val   --corrupt fog --corrupt-seed 1 --out runs/cache/mc_vis_val_fog.pkl
python scripts/build_cache.py --source ensemble --weights $E --data vis --split val   --out runs/cache/ens_vis_val_clean.pkl
python scripts/build_cache.py --source ensemble --weights $E --data vis --split val   --corrupt fog --corrupt-seed 1 --out runs/cache/ens_vis_val_fog.pkl
# ... repeat for --data ir with the IR-trained weights

# 3. Table 2 — pre-registered calibration metrics (incl. the DFL-derived §7.2 row:
#    same Gaussian cache, sigma key switched)
python scripts/evaluate_uq.py \
    --cache gaussian_clean=runs/cache/gauss_vis_val_clean.pkl \
    --cache gaussian_dfl_clean=runs/cache/gauss_vis_val_clean.pkl:dfl_sigma_ltrb \
    --cache gaussian_fog=runs/cache/gauss_vis_val_fog.pkl \
    --cache mc_clean=runs/cache/mc_vis_val_clean.pkl \
    --cache mc_fog=runs/cache/mc_vis_val_fog.pkl \
    --cache ensemble_clean=runs/cache/ens_vis_val_clean.pkl \
    --cache ensemble_fog=runs/cache/ens_vis_val_fog.pkl \
    --out runs/eval/table2_vis.md

# 4. Gate-level ablation (combination rule x α) — pure CPU over the caches
python scripts/ablate_gate.py \
    --vis-cache runs/cache/gauss_vis_val_fog.pkl \
    --ir-cache  runs/cache/gauss_ir_val_clean.pkl \
    --clean-cache runs/cache/gauss_vis_val_clean.pkl \
    --fit-cache runs/cache/gauss_vis_train_clean.pkl \
    --out runs/eval/gate_ablation.md
```

The learned-gate upper bound (§7.5) and full fusion-system comparison run inside the Python API (`uqfusion.eval.learned_gate`, `uqfusion.eval.fusion_eval.evaluate_systems`) and are exercised by `smoke_phase3.py`; their manuscript-facing CLI arrives with the Phase 4 Table 3 runner, where the VIS/IR pairing of real frames is defined.

**Inference-cost column note (plan B6):** report measured wall-clock from the cache-build logs (Gaussian 1 pass vs MC T=10 vs ensemble M=5), not just the nominal multiplier.

## 5. Phase 4 — publication run (not yet coded)

Multi-seed final runs on both datasets, Table 3 (adverse-condition robustness incl. the both-degraded row + `R_sys` histogram), MIT dataset in, DETR benchmark row decision, manuscript artifacts. The Table 3 runner + real VIS↔IR frame pairing + IR→VIS homography (needs the calibration files, OQ-5) are the main new code. Phase gate: Laksh's go-ahead after Phase 3 results.

---

## Where outputs land

```
runs/
├── audit/                    split-leakage reports
├── derived/                  stride-subsampled train lists + yamls
├── benchmark/                Table 1 CSVs, fps.csv, table1*.md; runs/ per-training dirs (+ ensemble members ens_*)
├── gaussian/                 Gaussian-head training runs
├── mc_dropout/               MC-Dropout training runs
├── cache/                    prediction caches (*.pkl) — the input to ALL evaluation
├── eval/                     table2*.md/json, gate_ablation.md
└── smoke*/                   smoke-test artifacts (safe to delete)
```

## Troubleshooting

- **`CUDA available: False` on the server** — wrong venv or driver mismatch; fix before anything else (`python scripts/smoke_env.py`).
- **Grid/ensemble crashed mid-way** — just rerun the same command; completed (variant, seed) rows are skipped via the results CSV.
- **`WeightsUnpickler error` loading Gaussian checkpoints** — load through project code (any `uqfusion.uq` import registers the safe classes); plain `torch.load` outside the package will refuse them.
- **Dataloader hangs on Windows** — set `benchmark.workers: 0` in config (Linux server: keep 8).
- **OOM on the full grid** — lower `benchmark.batch` (largest-common-fit rule pins one value for ALL variants).
- **W&B instead of TensorBoard** — `tracking.backend: wandb` in config + `pip install wandb` + `wandb login` (optional; TensorBoard is the no-account default).
- **Val labels missing warning during audit/eval** — the labels tree must mirror `images/` (Ultralytics convention); see dataset_requirement.md §5.
