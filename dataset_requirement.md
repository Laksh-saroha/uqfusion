# Dataset Requirements — what the code expects on the training server

> Requested at Phase 1 kickoff (answer A4-13 follow-up). This is the contract between your data preparation and every script in this repo. If any point can't be met, say which — most parsers are one function to adjust (`src/uqfusion/data/lists.py`), but I need to know rather than discover it mid-run.

## 1. Directory layout (matches README)

```
data/
├── pohang/
│   ├── data_vis.yaml            # visible stream   (Phase 1 benchmark uses this)
│   ├── data_ir.yaml             # infrared stream  (Phase 1 IR confirmation)
│   ├── calibration/             # <- REQUESTED, see §6 (Phase 2, open question A2-8)
│   └── <images/ and labels/ trees as referenced by the yamls>
└── mit_marine/                  # Phase 2+ (open question A2-7); same contract when ready
```

Everything below `data/` may live anywhere on the server — `config.yaml`'s `paths.data_root` is the only place that path is written.

## 2. The dataset yamls

`data_vis.yaml` / `data_ir.yaml` are standard Ultralytics dataset files with:

- `train`, `val`, `test` entries — **txt file lists** (one image path per line; absolute, or relative to the yaml's `path:` root) or directories. Txt lists preferred: they make the split explicit and auditable.
- `names`: `{0: ship, 1: buoy}` (classes confirmed, answer A2-7).

**Pohang-only for Phase 1.** You described the data as a *combined* Pohang+MIT set (answer A1-2). The approved plan (C4/D8) ranks backbones on **Pohang only** — MIT enters Phase 2+. If your current yamls mix MIT frames in, additionally provide Pohang-only yamls (e.g. `data_vis_pohang.yaml`) and point `config.yaml` → `datasets.pohang.*` at those. If the yamls are already Pohang-only, nothing to do.

## 3. The split — three ways, not two

Your custom split (answer A2-4) must expose **train / val / test**:

- **val** — drives Table 1 ranking, early stopping (patience), and later the §6.4 gate-constant fitting.
- **test** — untouched until Phase 4's final evaluation. Val and test must be disjoint (anti-leakage rule, plan B5-5).

If the split is currently only train/test, tell me — I'll add a tool that carves val out of train by temporal blocks rather than have you re-split by hand.

**Split-quality rules (decision D6-rev, replacing D6):**
- pohang01 (night) **is allowed in training** (your call, answer A2-5).
- Within any recording run, train frames must keep a temporal buffer (≈10 s; `data_audit.min_gap_frames` in config) from val/test frames. Contiguous-block splits satisfy this automatically; random frame-level splits will fail the audit.
- No image (by path or by per-run stem) in more than one split.
- **Please include night (pohang01) blocks in *test*** as well as train — that preserves a real-night evaluation row for Table 3 now that pohang01 isn't fully held out.

**Verification is mechanical, not on trust:** run `python scripts/audit_split.py` on the server. Exit 0 = split usable; the report lands in `runs/audit/`. This is the confirmation you asked for in A2-4 — I can't inspect the split from the dev machine.

## 4. Filenames

Frame filenames must sort temporally within a run: an epoch timestamp (`1544674707.213211.png`) or a zero-padded frame index (`000123.png`, `pohang00_000123.png`) both work. The audit and stride tools derive frame order from the stem's number; frames whose stems contain no number are reported and skipped by the temporal check (weakening the audit — avoid). The run identity is read from a `pohangNN` fragment anywhere in the path, else the parent directory name.

## 5. Images and labels

**Image size (Laksh, 2026-07-08): all images will be provided at 640×640.** Constraints on that:
- **Letterbox (scale + pad, aspect preserved) — do not stretch to square.** Stretching distorts VIS (1.9:1 native) and IR (1.25:1) by *different* factors, which hurts cross-modal box overlap for fusion, makes σ anisotropic in world terms, and complicates the homography. If the 640×640 copies were produced any other way, say exactly how (the transform must be invertible for the homography composition).
- Labels must be normalized to the stored 640×640 image (padding included).
- **Keep the full-resolution originals archived.** `imgsz` is the one allowed lever for small-object (buoy) recall (scope §4 rules out architecture fixes); training at 960+ from originals stays possible only if they survive.
- Note this does NOT replace the calibration files (§6) — same-size ≠ registered.

- Labels: YOLO txt (`class cx cy w h`, normalized), same stem as the image, in the `labels/` tree mirroring `images/` (standard Ultralytics convention).
- **IR bit depth:** training images must be 8-bit (1- or 3-channel as saved). Pohang thermal is natively 16-bit — if your IR trees were converted to 8-bit, **document which normalization was used** (global min-max? per-frame percentile?) in a short note (e.g. `data/pohang/IR_PREPROCESSING.md`). This matters twice later: per-frame auto-scaling can visually mask thermal crossover (scope §10), and the Mahalanobis OOD features (O3) see whatever the normalization leaves. If the IR trees are still 16-bit, flag it — we'll pick a normalization together before the IR runs.
- No modality-copied annotations in violation of scope §10.2 (don't copy RGB boxes onto IR frames where the vessel has no thermal contrast, etc.). If the current labels predate that rule, say so — the §10.3 visibility-score filter is scheduled for Phase 2 and can flag candidates automatically.

## 6. Calibration files (open question A2-8 — Phase 2, not blocking Phase 1)

Place the Pohang sensor calibration (camera intrinsics + VIS↔IR extrinsics, any format the original dataset ships) under `data/pohang/calibration/`. Needed for the IR→VIS homography (decision D7) before fusion work in Phase 2.

## 7. MIT Marine Perception (open question A2-7 — Phase 2+)

Same contract as above under `data/mit_marine/` when the annotation set is ready. Still needed from you: how many images are annotated, and their QA status.

---

### Server run order for Phase 1 (also in README)

```bash
pip install -r requirements.txt -e .        # once
python scripts/smoke_env.py                 # imports + CUDA + variant names
python scripts/smoke_benchmark.py           # harness end-to-end on synthetic data
python scripts/audit_split.py               # MUST exit 0 before training
python scripts/make_stride_subset.py --data vis          # -> derived yaml path
python scripts/run_benchmark.py --data <derived-yaml> --variants yolov8s --seeds 0 --epochs 1   # timing dry-run
# report the dry-run wall time back -> epochs/batch get pinned -> full grid:
python scripts/run_benchmark.py --data <derived-yaml>
python scripts/measure_fps.py --data vis
python scripts/make_table1.py
```
