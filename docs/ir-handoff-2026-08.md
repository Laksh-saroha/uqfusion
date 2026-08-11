# IR handoff — state, findings, and pending work

**Written:** 2026-08-01
**Scope:** everything infrared. What exists, how it got that way, what is known to be
wrong with it, and what still has to be done. A cold read of this file should be enough
to pick the IR thread up without re-deriving anything.

Companion docs: [`dataset-changes-2026-07.md`](dataset-changes-2026-07.md) (all dataset
edits, both modalities), `Pohang_dataset/IR_PREPROCESSING.md` (the original conversion
record), [`../progress.md`](../progress.md) (decision log, open questions).

Status labels used below: **VERIFIED** = measured or checked this session ·
**RECORDED** = taken from an existing project doc · **PROPOSED** = not built, not agreed.

---

## 1. What exists on disk right now

| # | Artifact | Location | Format | Frames | Size | Status |
|---|---|---|---|---:|---:|---|
| 1 | Raw thermal | `D:\Datasets\Pohang\pohang{NN}\infrared\images\` | 640×512 **16-bit** PNG (`I;16`) | 118,578 (all, incl. unlabeled) | ~31 GB | **VERIFIED** |
| 2 | 8-bit native | `D:\Datasets\Pohang labels\infrared_orig\` | 640×512 8-bit `L` | 31,010 (labeled only) | 4.6 GB | **VERIFIED** |
| 3 | 8-bit letterboxed | `A:\Uncertain\Pohang_dataset\infrared\` | **640×640** 8-bit `L` | 31,010 | 4.7 GB | **VERIFIED** |
| 4 | Full-res twin | `D:\Datasets\Pohang_dataset_full\infrared\` | 640×512 8-bit `L`, hardlinked from #2 | 31,010 | 0 new bytes | **VERIFIED** |
| 5 | Duplicate IR labels | `D:\Datasets\Pohang labels\ir\labels\` | byte-identical backup of #2's labels | 31,010 | 53 MB | **VERIFIED** |

**#3 is the only one anything trains on.** #1 is the archive of record. #2 is the
re-export source (the runbook already designates it as *"needed only to re-export IR"*,
HOW_TO_RUN §1). #4 exists for high-`imgsz` experiments.

Per-run counts (**VERIFIED**, and they reproduce the original build's own table):

| run | IR frames | note |
|---|---:|---|
| pohang00 | 10,918 | dense |
| pohang01 | 11,995 | dense — **the night run**, where IR matters most |
| pohang02 | 6,175 | moderately sparse |
| pohang03 | 1,922 | very sparse |
| pohang04 | **0** | PoLaRIS released no IR labels for this run |

Split (identical membership in #3 and #4, **VERIFIED** by `audit_split.py` PASS on both):
train 23,279 / val 2,234 / test 2,518 = 28,031 listed, plus 2,979 guard-band frames on
disk but in no list.

---

## 2. How the current IR images were produced

Four steps. Traced end-to-end and reproduced byte-exactly this session (**VERIFIED**).

1. **Extract** (`01_extract_light.py`) — `infrared.zip` → 640×512 16-bit PNG, 10 Hz, all frames.
2. **Build YOLO tree** (`04_stream_and_write.py`) — byte-copy of the *labeled* frames only,
   renamed `pohang{NN}_{frame}.png`, hash-based 80/10/10 split. Labels clipped to [0,1];
   degenerate boxes dropped where `w·h < 1e-6` (48 boxes, all pohang03 IR, ~0.3%).
3. **16→8 bit** (`convert_ir_to_8bit.py`) — **per-frame min–max**, in place:
   ```python
   lo, hi = arr.min(), arr.max()
   arr8 = ((arr.astype(np.float32) - lo) / (hi - lo) * 255 + 0.5).astype(np.uint8)
   ```
   Re-running this formula on the raw source reproduces the stored 8-bit file with
   **max abs diff 0**. Zero-variance frames written all-black.
4. **Letterbox** (`prepare_pohang.py`) — 640×512 → 640×640. **No downscale**: width is
   already 640 and height 512 < 640, so scale = 1.0 and it is pure padding — 64 rows of
   **114** top and bottom. Labels `cy' = 0.8·cy + 0.1`, `h' = 0.8·h`, clipped [0,1].
   Verified: rows 0–63 and 576–639 are all 114, centre 512 rows byte-identical to #2.

Then: re-split twice (list-only, same per-run ordinal plan as VIS, per-modality guard
bands — IR's are wider because pohang02/03 are sparse).

**The night-box filter never touched IR.** It is VIS-train-only by design.

---

## 3. Finding: the bit depth is fine, the normalization is not

Measured over 300 raw frames sampled across all five runs (**VERIFIED**).

### 3.1 The 16 bits are ~99% empty

| run | median span (raw counts) | effective bits |
|---|---:|---:|
| pohang00 | 1,008 | 7.84 |
| pohang01 | 519 | 7.26 |
| pohang02 | 487 | 6.56 |
| pohang03 | 512 | 7.43 |
| pohang04 | 913 | 8.16 |
| **all (n=300)** | **702** | **7.72** |

A 16-bit container holds 65,536 levels; a typical frame occupies ~702. "Effective bits"
is `log2(span / noise floor)` — the scene carries **~7.7 bits**, which fits in 8.

### 3.2 The 8-bit step sits at the sensor noise floor

Median 8-bit step = **2.75 raw counts**. Noise floor measured two independent ways:
**2.1 counts** temporally (consecutive-frame difference, MAD-based) and 4.19 spatially
(inflated by fixed-pattern/column noise, so the temporal figure is the truer one).

Going to 16-bit would resolve differences finer than the sensor can measure. Real gain
≈ 0.3 bits.

### 3.3 The actual damage: outlier-driven range inflation

Per-frame min–max keys the mapping to the single hottest and coldest pixel, so one hot
spot (exhaust, sun glint off metal) compresses the whole scene. Inflation = full span ÷
p0.5–p99.5 span:

- **51%** of frames > 1.5× · **27%** > 2× · **10%** > 3× · **3.3%** > 5×
- In **28%** of frames the scene bulk is squeezed into < 128 of the 255 available levels

Worked example, `pohang00/013000.png` (inflation 1.80×):

```
min–max     : std 23.22, uses 202 of 256 levels, IQR 24
percentile  : std 41.06, uses 256 of 256 levels, IQR 43
```

A **1.77× contrast loss** caused by a handful of hot pixels elsewhere in the frame.
16-bit does not fix this — it preserves the same bad mapping at higher precision.

### 3.4 The trap, if anyone tries raw 16-bit directly

Ultralytics loads via `cv2.imread(..., IMREAD_COLOR)`:

```
raw uint16                  : 7204 – 7943
cv2.imread(IMREAD_COLOR)    : 28 – 31      <- 4 distinct levels, near-black
PIL .convert("L")           : 255 – 255    <- fully saturated
```

Pointing YOLO at tree #1 produces garbage. `load_ir.py` already carries the PIL warning;
the cv2 behaviour is the one that would actually bite, because it fails *quietly*.

---

## 4. Recommended change (PROPOSED — not built, not agreed)

Re-export IR from the raw 16-bit using **per-frame percentile clipping** instead of
min–max. Chain: raw 16-bit → percentile 8-bit 640×512 → letterbox 640×640, i.e. the same
pipeline with step 3 swapped.

**Expected effect:** median ~1.5× more usable contrast, up to 1.8×+ on the worst quarter
of frames. Gains should concentrate in the hard tail — distant ships, buoys, targets near
the sea's own temperature — which is where the weak `buoy` class lives.

**Honest limit:** contrast was measured, mAP was not. The direction is well supported;
the magnitude is unknown.

**Secondary effect worth stating:** the σ² head and the reliability gate learn what a hard
frame looks like. Frames whose contrast was randomly crushed by an outlier are spurious
hard frames. Cleaner normalization should make learned uncertainty track real difficulty.

**Cost:** minutes of local CPU on 31,010 frames, no GPU, no re-download. Raw source
untouched, current tree retained.

**Timing argument:** no IR training has ever run (see §5.1), so this invalidates nothing
today. Later it would mean discarding IR results.

### 4.1 Open decisions — these are Laksh's, not mine

| # | Decision | Options | Notes |
|---|---|---|---|
| IR-D1 | Percentile bounds | 0.5–99.5 (what §3 was measured against) · 1–99 (more aggressive) · 0.1–99.9 (conservative) | tighter helps the worst frames more, saturates more real signal in clean ones |
| IR-D2 | Per-frame vs global scale | per-frame percentile · global/dataset percentile · both trees | per-frame is better for detection; global preserves cross-frame radiometric comparability, which O3 (Mahalanobis) and the scope §10 crossover argument need. My lean: per-frame now, revisit global when O3 hits real data — it can be re-derived from tree #1 later without redoing detection training |
| IR-D3 | Keep both variants or replace | keep both (+4.7 GB) · replace in place | keeping both is what makes the choice defensible to a reviewer |

---

## 5. Pending IR work

### 5.1 The IR confirmation grid has never run — **VERIFIED**
Phase 1 step 6 (HOW_TO_RUN §2) trains the **top-2 VIS variants** on IR and writes
`ir_results.csv` → `table1_ir.md`. It has not happened: the 2026-07-31 server run was
VIS ship-only (`data_vis_stride2.yaml`, `classes=0`), and no `runs/derived/data_ir_stride*.yaml`
exists — only the three VIS stride yamls. So `make_stride_subset.py --data ir` is also
still pending.

Consequence: **there are zero IR results to invalidate.** This is the cheapest possible
moment to change the IR representation.

### 5.2 Thermal-crossover filter on IR train labels — **RECORDED, not built**
scope §10 asks for the IR-side analogue of the VIS night-box filter:

> "If a vessel annotation in IR has near-zero contrast, flag it as a thermal crossover
> candidate — remove it from the IR training stream or assign a loss weight close to zero.
> These flagged frames automatically become the calibration test set for O5."

This is a real, unbuilt dataset operation. Four detection methods are already named in
scope §10.3: local Weber/Michelson contrast, frame histogram entropy collapse, Sobel edge
response at the box perimeter, and the O3 Mahalanobis score (scope calls the last the most
principled). `filter_night_boxes.py` already implements the box-scoring machinery
(intensity / Sobel / box-vs-ring contrast) for VIS and is the obvious starting point.

**Two warnings inherited from the VIS filter, which must not be repeated:**
1. **Train-only.** Removing crossover boxes from val/test would erase the very result the
   fusion benchmark measures.
2. **Do not port the VIS threshold, and do not compute "content median" the VIS way.**
   `PAD_LEVEL = 4` does not exclude the 114 letterbox padding — see
   [`dataset-changes-2026-07.md`](dataset-changes-2026-07.md) §1. IR letterboxing pads 64
   rows top and bottom (20% of the frame), so any IR frame statistic must exclude rows
   0–63 and 576–639 explicitly, or work from tree #2 (unpadded) instead.

### 5.3 Thermal bloom / IR box extent — **RECORDED, unverified**
scope §10 Scenario 4 notes IR thermal extent runs 10–30% larger than the hull, and warns
against copying RGB boxes to IR. PoLaRIS annotated IR natively rather than by copying, so
this may already be handled — but **nobody has checked**. Worth one measurement: compare
IR vs VIS box areas for timestamp-matched pairs (`paired/pohang{NN}_pairs.csv` has 28,388
pairs) before assuming either way.

### 5.4 IR→VIS homography — **RECORDED** (D7, OQ-5, partially resolved)
Per-run intrinsics + extrinsics are present at `Pohang_dataset/meta/pohang{NN}/calibration/`
with keys including `infrared` and `stereo_left/right`. Calibration is **per-run**, so it
is one homography per run, not one globally. Fusion currently smoke-tests with identity H.
Phase 2/4 work.

### 5.5 Full-resolution percentile variant — **PROPOSED**
If §4 is adopted, the same re-export should produce a percentile variant of tree #4 so the
high-`imgsz` path doesn't silently keep the old normalization. +4.6 GB on D:.

### 5.6 Cross-dataset normalization convention — **open, unowned**
MIT AUVLab / MassMIND IR is a different sensor delivered as 8-bit. No decision exists on
whether Pohang and MIT IR should share a normalization convention. This surfaces at Phase 4
/ MIT onboarding, and it interacts with IR-D2: a per-frame scheme makes the two datasets
harder to compare radiometrically than a global one would.

---

## 6. If two IR variants coexist — arrangement

Normalization changes **pixels only**. Labels, split lists, frame counts, `meta/`,
`paired/` are identical across variants, so only images are genuinely duplicated.

```
Pohang_dataset/
├── data_ir.yaml              -> infrared/       (min–max, current)
├── data_ir_pct.yaml          -> infrared_pct/   (percentile)
├── infrared/      images/pohang{NN}/  labels/pohang{NN}/  {train,val,test}.txt
├── infrared_pct/  images/pohang{NN}/  labels/pohang{NN}/  {train,val,test}.txt
└── meta/  paired/            <- shared
```

- **Sibling trees are forced, not chosen.** Ultralytics derives the label path by swapping
  the last `/images/` for `/labels/`, so each image tree needs its own labels dir. It also
  gives each variant its own `labels/*.cache` — they cannot poison each other's cached scan.
- **Copy the labels, don't hardlink them** (53 MB). A copy means a future label edit — e.g.
  the §5.2 crossover filter — cannot silently hit both trees.
- **No config change needed.** `resolve_data_yaml` passes any non-alias value through as an
  explicit path: `--data "Pohang_dataset/data_ir_pct.yaml"`.
- **Don't rename mid-project.** Add `data_ir_pct.yaml`, leave `data_ir.yaml` alone, flip the
  default once a winner is chosen.

### 6.1 Provenance trap — must be handled before any two-variant run — **VERIFIED**

`split_fingerprint` hashes `run/filename` ids only:

```python
ids = sorted(f"{run_key(p)}/{p.name}" for p in split_image_list(data, split))
```

Two IR trees with the same frames and the same split produce an **identical fingerprint**.
`run_grid` then hits `done.get((variant, seed)) == fingerprint` and **silently skips** every
percentile run as "already done." No error, and the mix-refusal does not fire — nothing
looks mismatched. This is the same blind spot the visfilter docs already flag ("filtering
labels does not change the fingerprint"); pixel content is the same category of change.

**Interim:** a fresh `--out-csv` and `--run-prefix` per variant — the precedent set by
`benchmark_results_ship_visfilter.csv`. Works, but fails silently if forgotten.

**Proper fix (~3 lines):** `_completed()` already blanks rows whose `classes` differ from
the current run. Add the `data_yaml` **basename** to that key the same way. Different
basenames → different keys → no false skip, and mixing them into one CSV trips the existing
refusal. Stays machine-independent (basename, not path), preserving what `split_fingerprint`
was designed for, and the `data_yaml` column **already exists** in the schema — so no column
bump and no incompatibility with existing CSVs.

---

## 7. Permanent gaps to disclose in the manuscript

- **pohang04 has zero IR labels.** Every per-run IR table shows N/A there.
- **pohang03 IR is very sparse** (1,922 vs ~27k VIS), pohang02 moderately so (6,175).
  Real distribution skew — do not resample to hide it.
- **Radiometry is not recoverable from trees #2–#4.** Per-frame min/max were never stored.
  It *is* recoverable by re-exporting from tree #1, which is intact — the claim in
  `IR_PREPROCESSING.md` that it is "not recoverable" is true of the 8-bit copies only.
- **Per-frame normalization of any kind breaks cross-frame radiometric comparability**,
  which is exactly what O3 (Mahalanobis) and the scope §10 crossover argument lean on.
  Percentile clipping improves contrast but does not solve this — only a global scale does
  (see IR-D2).
- **Thermal crossover can be visually masked** by any per-frame stretch: a near-isothermal
  vessel still gets expanded to full range, so 8-bit crops look more separable than the
  radiance was.

---

## 8. Suggested order of work

| # | Action | Cost | Blocks on |
|---|---|---|---|
| 1 | Decide IR-D1 / IR-D2 / IR-D3 (§4.1) | a conversation | — |
| 2 | Apply the §6.1 resume-key fix | ~3 lines | — |
| 3 | Re-export percentile IR (trees #3 and #5, both resolutions) | minutes CPU, local | 1 |
| 4 | `make_stride_subset.py --data ir` | minutes | 3 |
| 5 | Phase 1 IR confirm grid, top-2 variants, fresh `--out-csv` | GPU | 3, 4 |
| 6 | Build the §5.2 crossover filter (train-only) | build + eyeball calibration | 3 |
| 7 | Measure §5.3 bloom on paired frames | CPU, ~1h | — |
| 8 | IR→VIS homography (D7) from per-run calibration | Phase 2/4 | — |
| 9 | Settle §5.6 with MIT onboarding | Phase 4 | — |

Steps 1–4 and 6–7 are all local CPU. **Only step 5 needs the server.**

---

## 9. File map

| Path | What |
|---|---|
| `D:\Datasets\Pohang\pohang{NN}\infrared\images\` | raw 16-bit archive of record (118,578 frames) |
| `D:\Datasets\Pohang\_scripts\convert_ir_to_8bit.py` | the min–max conversion that produced the current pixels |
| `D:\Datasets\Pohang\_scripts\load_ir.py` | correct 16-bit loader + the PIL warning |
| `D:\Datasets\Pohang labels\infrared_orig\` | 8-bit 640×512, the re-export source |
| `A:\Uncertain\Pohang_dataset\infrared\` | the tree everything trains on |
| `A:\Uncertain\Pohang_dataset\IR_PREPROCESSING.md` | original conversion record (see §7 caveat) |
| `D:\Datasets\Pohang_dataset_full\` | full-res twin + its own `PREPARATION.md` |
| `A:\Uncertain\scripts\prepare_pohang.py` | IR letterbox + first re-split |
| `A:\Uncertain\scripts\filter_night_boxes.py` | VIS filter; the box-scoring base for §5.2 |
| `A:\Uncertain\src\uqfusion\bench\grid.py` | `split_fingerprint`, `_completed` — see §6.1 |
