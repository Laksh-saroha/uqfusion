# Dataset changes — Pohang VIS, July 2026

Record of every modification made to the Pohang dataset during Phase 1 prep. Two kinds of change:

1. **A source-label edit** (night-box visibility filter) — rewrites label files on disk.
   Reversible.
2. **Derived training subsets** (stride subsampling) — never touch source data; they emit new
   train lists + dataset yamls under `runs/derived/`.

A third mechanism, **class filtering**, is *not* a dataset change and is documented at the end to
prevent confusion.

Everything below was executed on the **local** dataset at `A:\Uncertain\Pohang_dataset\`. The
**server** dataset (`{data_root}/pohang/`) was **not** modified — see "Server status".

---

## 1. Night-box visibility filter (source-label edit)

**Script:** `scripts/filter_night_boxes.py` · **Commit:** `0be9718` · **Executed:** 2026-07-15
10:37, local dataset · **Manifest:** `runs/visfilter/visfilter_manifest.json`

### Problem

Many `pohang01` VIS frames are night footage. Ships/buoys are annotated (labels are geometrically
correct) but **not visible** in the pixels — below the sensor's usable light level. Training on
boxes with no learnable signal teaches the detector to hallucinate objects in dark water, hurting
precision. These are VIS-invisible targets IR is meant to catch — exactly the fusion case the
benchmark measures — so they must leave the VIS learning signal without leaving evaluation.

### Scope rule (critical)

**Only VIS TRAIN labels are edited. val/test are never touched.** Train = the learning signal, so
removing unlearnable boxes is legitimate. val/test = the question being asked; editing them is
self-grading — it would erase the "VIS misses / IR catches" result and break comparability with
every other row and modality. Frozen.

### Method (why `--cut-dark`, not photometric thresholds)

The initial design scored each box (mean intensity + Sobel gradient + box-vs-ring contrast) and
dropped those below threshold. Eyeball calibration against generated crops showed the thresholds
were too conservative *and* that in `pohang01` **nothing** is learnable past the bright dusk head
— every photometric band spot-checked showed no visible target. So a deterministic per-frame
luminance cut replaced the per-box score.

`--cut-dark pohang01:100` = drop **all** train boxes in any `pohang01` frame whose **content-median
luminance < 100** (0–255). This is **camera-fair**: an earlier ordinal-based cut (`--cut RUN:N`,
drop past frame ordinal N) was rejected because the L/R stereo cameras share one ordinal timeline
unevenly — it would have wrongly dropped ~454 genuinely-bright R-camera frames. Luminance is judged
per frame regardless of camera.

#### What "content-median" actually measured (correction, 2026-08-01)

`content_bbox()` uses `PAD_LEVEL = 4` — only near-black pixels count as padding. The VIS letterbox
pads with **114**, so the gray bars were never excluded. They are 47% of every 640 VIS frame and
dominate the median. Measured over all 96,275 cached values in `frame_medians.csv`:

| run | n | min | median | max | below 100 |
|---|---:|---:|---:|---:|---:|
| pohang00 | 16,376 | 114 | 114 | 114 | 0 |
| pohang01 | 18,826 | 3 | 12 | 114 | 17,515 |
| pohang02 | 20,424 | 114 | 114 | 114 | 0 |
| pohang03 | 20,962 | 114 | 114 | 114 | 0 |
| pohang04 | 19,687 | 114 | 114 | 114 | 0 |

Every day-run frame reports exactly the pad value; the statistic is saturated for four runs of
five. Because the padding occupies 47% and the content median sits below 114, the 50th percentile
of the padded frame lands at roughly the **content's 95th percentile**. So the rule that ran was:

> drop all `pohang01` train boxes in frames whose content **~95th-percentile** luminance < 100 —
> i.e. frames with essentially no bright pixels anywhere.

**The outcome is sound.** The threshold was eyeball-calibrated against these exact numbers against
generated crops, and the day runs were never candidates anyway (the cut is scoped to `pohang01`).
Nothing about the 640 dataset needs redoing.

**But the threshold does not transfer to unpadded images.** On native-resolution frames there are
no bars: a bright day frame measures 33 and a night frame 8 — both below 100. Re-running
`--cut-dark pohang01:100` against native images would empty essentially *all* pohang01 train labels
rather than 17,515 / 18,826. This is why the full-resolution tree (§4) **ports the file list**
instead of recomputing it.

Frames left with zero boxes are **kept as background images** (empty label file). Ultralytics
trains on these normally; they help precision.

### Exact effect (from manifest)

| Field | Value |
|---|---|
| Mode | `cut_dark`, `pohang01: 100.0` |
| Train frames scanned | 96,275 |
| Frames affected (below cut) | 17,515 |
| Label files rewritten (emptied) | 17,502 |
| Frames now background-only | 17,502 |
| Boxes dropped | 132,688 (ship 126,948 / buoy 5,740) |
| Train-label content hash before | `fd60c0834fdd` |
| Train-label content hash after | `287b11c50b5a` |

(17,515 affected vs 17,502 rewritten: the difference is frames already label-empty / not present as
label files — nothing to rewrite.)

### Reversibility

- Every edited label backed up **once** to `*.pre_visfilter` beside the label.
- `python scripts/filter_night_boxes.py --restore` restores from backups.
- Manifest records the cut spec, rationale, per-file dropped line indices, and before/after hashes.

### Provenance interaction

The grid's `split_fingerprint` hashes **image ids** (`run/filename`), not label bytes — so
filtering labels does **not** change the fingerprint. Label content is provenance-tracked by the
manifest's train-label hash (`fd60c0834fdd` → `287b11c50b5a`). **Filtered labels = a new
experiment: use a fresh `--out-csv`.** Any pre-filter benchmark rows are invalid against filtered
labels.

### Post-edit gotcha

Ultralytics caches labels in `*.cache` files next to the label dirs. These go **stale** after the
edit and must be **deleted** before training, or the trainer reads the old boxes.

---

## 2. Derived training subsets (stride subsampling)

**Script:** `scripts/make_stride_subset.py` (module `uqfusion.data.subset`). **Source data: never
modified.** Outputs land in `runs/derived/`.

### Why

Pohang is 10 Hz video: consecutive frames are near-duplicates. Striding thins temporal redundancy
for faster iteration. The stride is over **unique per-run frame ordinals** (time steps), not raw
frame lists — L/R stereo frames share an ordinal and are kept/dropped together, so a stride keeps
whole stereo pairs and thins *time* instead of alternating cameras.

### Subsets created this session

| Yaml | Stride | Train frames kept | Purpose |
|---|---|---|---|
| `runs/derived/data_vis_stride2.yaml`  | 2  | 48,136 / 96,275 | full-scale local runs |
| `runs/derived/data_vis_stride10.yaml` | 10 | 9,629 / 96,275  | ~10k mosaic ablation (yolov8s) |
| `runs/derived/data_vis_stride19.yaml` | 19 | 5,077 / 96,275  | ~5k mosaic ablation (yolo26s) |

Each derived yaml keeps the **same val/test/names** as the source and only swaps the train list
(`runs/derived/data_vis_train_strideN.txt`, absolute paths). Machine-local artifacts — regenerate
per machine; not portable.

Because the subsets were built **after** the night-box filter, their train frames carry the
filtered (emptied) labels for the affected `pohang01` frames.

---

## 3. Class filtering — NOT a dataset change

`--classes 0` (ship-only) passed to `run_benchmark.py` filters classes at Ultralytics
**train+val load time**. It edits **no** labels and touches **no** files. Buoy boxes (class 1)
simply aren't loaded for that run. Recorded in the results CSV `classes` column and guarded so
ship-only and all-class rows can't be averaged together — but the dataset on disk is unchanged.
Listed here only so it isn't mistaken for a data edit.

---

## 4. Full-resolution twin (`D:\Datasets\Pohang_dataset_full\`, 2026-08-01)

**Script:** `scripts/prepare_pohang_fullres.py` · **Full narrative:**
`D:\Datasets\Pohang_dataset_full\PREPARATION.md` (travels with the data)

`imgsz` is the small-object lever (OQ-7), but the trainable tree is letterboxed down to 640×640.
This builds a second tree at **native resolution** — VIS 2048×1080, IR 640×512 — carrying the
**same split and the same night filter**, so a higher-`imgsz` run is directly comparable against
the 31-variant grid.

### Sources (all on D:, verified 1:1 against the 640 tree by filename stem)

| Stream | Path | Format | Frames |
|---|---|---|---:|
| VIS native | `Pohang_YOLO\visible_old\{images,labels}\{train,val,test}\` | 2048×1080 RGB PNG | 127,309 |
| IR native | `Pohang labels\infrared_orig\{images,labels}\{train,val,test}\` | 640×512 8-bit L PNG | 31,010 |

Zero stems missing in either direction, and all 147,103 entries of the 640 tree's split lists
resolve. `Pohang labels\{vis,ir}\labels\` are byte-identical duplicates of the two label trees
(verified: matching name sets, 0 symmetric difference, 1,200/1,200 sampled hashes equal) and are
left untouched as an independent backup.

### What was done

1. **Layout** — rebuilt from split-foldered into the canonical per-run layout
   (`{images,labels}/pohang{NN}/` + `{train,val,test}.txt`), matching the 640 tree so future
   re-splits stay list-only. Images are **hardlinked** (`os.link`): 309 GB of VIS costs 0 new bytes
   and `visible_old/` stays intact. Labels are **real copies**, because step 3 rewrites them and an
   in-place edit through a hardlink would corrupt the source too.
2. **Split** — the balanced K-block lists copied **verbatim**. They are already in
   `./images/pohang{NN}/<stem>.png` form, which resolves unchanged in the new tree, so a verbatim
   copy proves identical membership *and* ordering. No leakage/balance/coverage gate needs
   re-running.
3. **Night filter** — the 17,502 emptied files **ported from the 640 manifest, not recomputed**,
   for the threshold-transfer reason in §1. Independent confirmation the port is exact: summing
   boxes in the native labels of those 17,502 frames gives **132,688** — the identical count the
   640 run dropped.

### What was deliberately NOT done

- **No resize.** Native pixels, and labels keep their native normalized coordinates (no letterbox
  transform). Ultralytics letterboxes at load time to whatever `imgsz` you pass, so one tree serves
  640/1024/1280.
- No val/test label edits, no class filtering, no stride subsetting.

### Reverting

Images are hardlinks, so deleting the tree frees the label copies only and cannot destroy image
data — `prepare_pohang_fullres.py --revert` verifies that every image is still inode-shared with a
live source *before* it will delete anything, and refuses otherwise. Label-only undo: restore each
`*.pre_visfilter`.

### Caution

`pohang00`–`pohang03` `stereo.zip` were deleted after the original build, so their 2048×1080 VIS
frames now exist **only** in `visible_old\images` (and, as hardlinks, in this tree). Regenerating
means a ~440 GB S3 re-download. `pohang04` still has its `stereo.zip`.

---

## Server status

The server dataset (`{data_root}/pohang/` on the H100 Jupyter box) has **not** been filtered or
subset by any of the above. To reproduce there, after the current Phase 1 grid finishes:

1. `python scripts/filter_night_boxes.py --cut-dark pohang01:100 --execute`
2. Delete the stale `*.cache` label caches.
3. Regenerate the derived stride yaml on the server (absolute paths differ).
4. Use a **fresh** `--out-csv` — filtered labels are a new experiment.

## File map

- `scripts/filter_night_boxes.py` — the filter (+ `--restore`).
- `runs/visfilter/visfilter_manifest.json` — what was cut, hashes, rationale.
- `*.pre_visfilter` — per-label backups (undo source).
- `scripts/make_stride_subset.py` — stride subsets.
- `runs/derived/data_vis_stride{2,10,19}.yaml` — derived train yamls.
