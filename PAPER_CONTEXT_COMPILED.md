# uqfusion — Compiled Project Context for Paper Writing

> Auto-compiled 2026-09-17 from every tracked `.md` file in the repository (`progress.md`, `scope.md`, `README.md`, `HOW_TO_RUN.md`, `dataset_requirement.md`, all of `docs/`, `docs/eval/`, `docs/readable/`, `runs/eval/`, `runs/queue_ideas/`, `archive/*/README.md`, `Pohang_dataset/*.md`, `phase1_benchmark/README.md`, and smoke/audit reports). This is a reference document, not a draft — it preserves exact numbers, hashes, dates, verdicts and file names from the source so nothing has to be re-derived when writing the actual manuscript. Where sources disagree or a number was later corrected, both are kept with the correction noted (the project's own discipline: nothing is silently overwritten).

---

## PART 0 — WHAT THIS PROJECT IS, IN ONE PAGE

**Title (working):** Uncertainty-Aware Fusion of Visible and Infrared Imagery for Reliable Maritime Object Detection.
**Author:** Laksh Saroha (Roll No. 1024060068), ECED, Thapar Institute of Engineering and Technology, Patiala. Mentor: Dr. Sandeep Mandia. UG Research Fellowship project. No venue/deadline pinned (A3-9); no release/licensing plan yet (A3-10; built on Ultralytics AGPL-3.0, obligations pending confirmation).

**What the system is, as measured (2026-09-14, current):** two independent per-modality YOLO26 detectors (VIS, IR), each with a single-pass Gaussian σ² head bolted onto the detection head; a frame-level Mahalanobis OOD scorer; and a decision layer that (a) computes an **image-statistic sensor-selection veto** — ask the IR stream whether it is night, and if so check whether VIS is dark or veiled; a vetoed stream is dropped entirely from the merge — and (b) **concatenates** the surviving streams' detections via Weighted Boxes Fusion at IoU threshold 0.85, with weights coming from a **constant, run-disjoint capability prior**, not from live per-frame uncertainty. Sigma-squared and Mahalanobis distance do **not** reach the fusion weight in the shipped preset (`crossmodal26m`): `mu_d=1e9, lam=0` makes both terms mathematically inert, and the measured fusion weight `w_vis` is a **single constant, 0.9930**, across all 2,232 paired evaluation frames × 4 corruption conditions (R-D1, 2026-09-10). Describe the shipped method as **"image-statistic sensor selection with union aggregation of detections,"** not "uncertainty-gated fusion."

**What the system was designed to be (scope.md, 2026-07-07, largely superseded by measurement):** a per-frame reliability score `R_m` per modality, combining size-normalized box-level aleatoric uncertainty (from the σ² head) and frame-level distributional/OOD uncertainty (Mahalanobis), fed as live weights into Weighted Boxes Fusion — "the more reliable modality dominates instant-by-instant." This design is preserved in `scope.md` §6 (architecture) and §6.4 (reliability formulas) with an explicit editorial note that reality has diverged from it and every divergence is measured, not assumed.

**The project's core epistemic stance, stated in scope.md and enforced throughout:** null results are results. The project has run at least 9 formal pre-registrations (with append-only amendments), tracked an "exposure ledger" of every time a held-out number was looked at, computed real (not assumed) noise floors, and closed multiple research axes permanently via pre-committed stop rules rather than re-trying until something looks good. The headline finding of the whole project, as of 2026-09-10, is a **null result**: predicted uncertainty does not improve the fusion at any tested floor (R-D1). The gain that is real and reproducible comes from **image-statistic sensor selection** (a hard veto driven by IR-darkness + VIS-darkness/veil statistics), which sits at or above `max(VIS, IR)` on all 8 benchmark cells once the veto rule was hardened (`crossmodal` preset, 2026-09-01).

**Datasets:** Pohang Canal + PoLaRIS is the **only** dataset used. MIT Marine Perception was planned (decision D10, 2026-07-07) but **never onboarded** — no data on disk, no code references, and a claim in an earlier scope.md draft that "we have manually annotated a subset of images from this dataset ourselves" was **measured false** and retired 2026-09-10. SMD and MassMIND are cited as literature only, unused.

**Where the project stands, 2026-09-17 (from `progress.md`, last updated 2026-09-14):** Phase 3 retrain Stages 0–2 closed. Stage 1 returned **S1-NULL** (correspondence-relaxation question closed permanently). Stage 2 trained 5 VIS + 5 IR seeds on yolo26m/yolo26m-p2feat without pohang04 in any split. Stage 4 — the single pre-registered look at pohang04 (true held-out data, used for nothing until now) — has its inputs built (190 caches, 76 frame-statistic files) but **is not frozen and not scored**. The freeze commit and the single look itself are the only remaining steps, and the look runs only on the author's explicit instruction.

---

## PART 1 — SCOPE, MOTIVATION, ARCHITECTURE DESIGN (from `scope.md`, `README.md`)

### 1.1 Motivation and research gap
EO sensors (visible + infrared) fail silently under specific conditions: visible degrades in fog/haze/glare/rain/darkness; infrared sees through glare/darkness but fails during thermal crossover (vessel and water reach the same temperature). The safety risk is *confident* error with no signal the sensor has become unreliable.

Two literature gaps claimed (scope.md §2, revised 2026-09-10 after positioning review):
1. Most maritime detectors (SID-YOLOv5, EG-YOLO, RDSC-YOLOv4, YOLOv7-sea, feature-fusion nets) output no uncertainty at all. Single-pass localization variance is established outside maritime (Gaussian YOLOv3, R1) and uncertainty-aware cross-modal fusion is established outside maritime too (R24 UA-CMDet 2022, R25 DICTA 2024) — so neither the head nor the idea of conditioning fusion on uncertainty is claimed as new. **What is thin is the maritime evidence**: whether these uncertainties are calibrated on paired VIS+LWIR maritime video, measured against a stated noise floor.
2. Calibrated evaluation of uncertainty-conditioned fusion is scarce; the open question is whether the uncertainty doing the conditioning is itself trustworthy, measured with dependence-aware intervals, with null results reported.

**Retired claim (kept for audit):** "Existing visible–infrared fusion is static — fixed rules or learned-but-static attention, never conditioned on a live per-frame reliability estimate." — factually wrong (see Part 9, Positioning).

### 1.2 Aim and objectives (O1–O5)
- **O1 — Baseline:** deterministic detection baseline via YOLO backbone selection (→ Phase 1).
- **O2 — Aleatoric UQ (core contribution):** single-pass probabilistic head predicting predictive variance per box coordinate, NLL-trained.
- **O3 — Distributional/epistemic UQ:** frame-level Mahalanobis distance to training distribution, catches total-sensor-degradation that per-box uncertainty misses.
- **O4 — Uncertainty-gated fusion:** per-modality reliability score from O2+O3 driving adaptive WBF.
- **O5 — Validation:** calibration + a pre-registered fusion test, reporting the outcome either way. **Half met, half NO**: gated fusion is at/above max(VIS,IR) on all 8 cells, but vs uncertainty-blind fusion the pre-registered R-D1 test is NULL at every floor, 0/4 conditions.

### 1.3 Scope boundaries
**In scope:** Pohang + synthetic augmentation; single YOLO backbone with only a variance-output branch added; Gaussian aleatoric head; Mahalanobis distributional UQ; uncertainty-gated soft fusion (as designed); MC-Dropout and Deep Ensemble comparisons; calibration analysis; optional conformal extension.
**Out of scope, explicitly:** accuracy-oriented architecture changes (no P2/small-object head, no SPD-Conv, no attention modules, no backbone redesign — detector held fixed so effects attribute to the UQ method); embedded deployment; new data collection; non-EO sensors; 3D detection/tracking/re-ID.

### 1.4 Architecture — pipeline design (scope §6.1, the "diagram that was tested, not the one that ships")
Each modality runs its own YOLO backbone (separate weights — different visual domains). From each backbone: Gaussian σ² head (per-box aleatoric) + Mahalanobis frame-level OOD score → per-modality reliability `R_m` (multiplicative — either failure mode condemns a stream) → weights WBF; fog/glare drives `R_vis` down (lean on IR), thermal crossover does the reverse. Single-pass throughout (real-time preserved).

### 1.5 Gaussian head and NLL loss (scope §6.2)
Head predicts μ, σ² per coordinate `t∈{x,y,w,h}`. Base loss:
`L_loc = Σ_t [ (t_gt-μ_t)²/(2σ_t²) + ½log(σ_t²) ]`
Pre-empted failure mode: vanilla NLL "explains away" hard examples (inflates σ² instead of improving μ), degrading localization and calibration [Seitzer 2022 β-NLL; Skafte 2019 warm-up]. Fix: β-NLL (stop-gradient weight `σ^{2β}`, β∈[0,1]) + warm-up (μ trains under plain box loss first, σ² unfrozen later).

### 1.6 Frame-level OOD score (scope §6.3)
Backbone features via forward hook → Mahalanobis distance to training-feature distribution; large distance = OOD = unreliable frame. Catches a fully-degraded frame (blank fog → few/no detections, which naively looks like *low* uncertainty) that box-level aleatoric misses.

### 1.7 Reliability score and fusion — the formal design (scope §6.4)
Per frame, per modality m:
- `u_i = ¼ Σ_t σ_{i,t}/s_{i,t}` (size-normalized per-box uncertainty, s=box w/h)
- `U_box,m = Σ_i c_i u_i / Σ_i c_i` (confidence-weighted)
- `O_m = sigmoid((d_m - μ_d)/τ)` (Mahalanobis distance calibrated to [0,1] on clean val)
- `r_box,m = exp(-λ·U_box,m)`, `r_frame,m = 1-O_m`, `R_m = r_frame,m · r_box,m`
- Empty-frame fallback: `R_m = r_frame,m` (distinguishes clear-empty-sea from fog-blind)
- Temporal smoothing: `R̄_m(t) = α R_m(t) + (1-α) R̄_m(t-1)`
- Fusion weight: `w_m = R̄_m / (R̄_vis + R̄_ir)`, fed into WBF.
- **Critical caveat in the original doc:** every constant (λ, sigmoid μ_d/τ, α, the multiplicative form) is a tunable design choice "to be validated empirically... not a settled formula."

### 1.8 Status vs design, as measured 2026-09-14 — "the fusion is union aggregation, not consensus"
Three pinned facts replace the formal design:
- **Weights:** capability prior alone; `w_vis` is one constant across all frames (R-D1). Mahalanobis and box-uncertainty terms never reach the weight.
- **Selection:** a stream is dropped per frame by an image-statistic veto (IR says night AND VIS is dark or veiled). A frame with one surviving stream returns that stream untouched.
- **Merging:** at iou_thr 0.85, 0.05% of VIS boxes have an IR partner — surviving streams are **concatenated**, with a cross-modal score bonus at IoU 0.30 that never moves a coordinate. Sensor *agreement* is not what the fusion measures.
- Stage 1 (Phase 3) closed the obvious repair: relaxing correspondence to 0.55 while letting σ arbitrate the resulting merges is non-inferior on only 1/4 conditions (need 3). **S1-NULL; correspondence question closed permanently.**

### 1.9 Architecture review — five refinements (scope §7, from the 2026-07-07 kickoff)
- **7.1** β-NLL / warm-up against heteroscedastic-NLL failure (highest priority, addressed).
- **7.2** DFL-derived vs explicit Gaussian σ²: YOLO26 (selected backbone) has no DFL (`reg_max=1`), so option (a) is undefined on the selected backbone. Never fully resolved (OQ-10 still open in decision log — `yolo12m`+cv4 built as a compute-matched control but the ablation itself is not reported as a headline Table 2 row in the digests reviewed here).
- **7.3** Deep Evidential Regression cited, not benchmarked (D2).
- **7.4** Independence assumption fails under correlated degradation (both sensors degrade together) — "both-degraded" row added to Table 3 design; measured later (`runs/eval/both_degraded.md`, see Part 8) that releasing the veto when both are flagged **prevents 0 bad vetoes and loses 2,095 correct ones** — R_sys abstain is a flag, not an actionable override.
- **7.5** Learned gate as upper-bound comparison, not replacement (implemented, `eval/learned_gate.py`).

### 1.10 Detector choice rationale (scope §8)
YOLO family: (1) real-time single-stage; (2) mature Ultralytics tooling for multi-seed benchmarking and custom-head forks; (3) some variants have DFL heads pairing naturally with variance modeling. UQ method is detector-agnostic in principle. One alternative flagged as worth a benchmark row: NMS-free DETR variants (RT-DETR CVPR 2024; D-FINE, **ICLR 2025** — corrected from an earlier wrong venue/year in bibliography review) — recommended as optional extra row, not primary; deferred to Phase 4 (D9), never run.

### 1.11 Experimental design — two separate comparisons (scope §9)
- **Comparison 1 (O1):** detector benchmark, no uncertainty, ≥3 seeds mean±std → Table 1.
- **Comparison 2 (core UQ result):** Gaussian head vs MC-Dropout vs Deep Ensemble as *alternative ways to produce the uncertainty signal on the same backbone* — not rival fusion pipelines. Claim: Gaussian head matches ensemble/MC-Dropout calibration at a fraction of compute (2 models vs 10 for M=5 ensemble × 2 modalities).
- **Fusion baselines (§9.3):** visible-only, IR-only, naive (uncertainty-blind) fusion.
- **Table skeletons** (§9.5): Table 1 (backbone benchmark), Table 2 (per-modality UQ calibration: ECE/NLL/sparsification/AURC/inference-cost/#models), Table 3 (fusion robustness: Clean/Fog/Glare/Low-light/Thermal-crossover/Both-degraded × 4 systems).

### 1.12 Annotation strategy for multimodal mismatch (scope §10) — design only, largely unused in practice
Four modality-visibility scenarios defined (clear/both; thermal-crossover/RGB-only; fog-darkness-glare/IR-only; thermal-bloom/both-different-size) with the principle "frames where one modality fails are not annotation problems to clean away — they are the primary calibration and fusion training signal." A visibility-score formula and thermal-crossover detection approaches (local contrast, histogram entropy, edge response, Mahalanobis) are specified. **Not evidenced as implemented in the measured pipeline** — the actual "no modality-copied annotations" rule shows up operationally as the night-box visibility filter (Part 3) rather than this formal per-scenario scheme.

### 1.13 Technology stack (scope §11)
Python 3.10+/PyTorch 2.x/CUDA; Ultralytics YOLO (fork via in-place head conversion, not a vendored fork — D17); MC-Dropout + Deep Ensembles via Ultralytics mechanics, Torch-Uncertainty for metrics; pytorch-ood + scikit-learn (LedoitWolf) for Mahalanobis; ensemble-boxes (WBF) + scipy Hungarian for matching; netcal/Torch-Uncertainty/torchmetrics for calibration; Albumentations for adverse-condition generation (RandomFog, RandomSunFlare, RandomRain, MotionBlur, GaussNoise, ISONoise); TorchCP/MAPIE for optional conformal extension (caveat: video violates exchangeability, acknowledged not assumed away).

### 1.14 Evaluation plan (scope §12)
Four parts: (1) detection accuracy (P/R/mAP50/mAP50-95, confirms UQ head doesn't degrade detection); (2) calibration — central: reliability diagrams, ECE, predicted-variance-vs-realized-IoU correlation, sparsification/accuracy-rejection curves; (3) OOD separation histograms; (4) downstream robustness under synthetic degradation including both-degraded row.
`ECE = Σ_m (|B_m|/n)·|accuracy(B_m) − confidence(B_m)|`.

### 1.15 Integration risks (scope §13, table)
Key entries: Ultralytics head/loss modification non-trivial (mitigated by β-NLL/warm-up, tiny-subset validation); Torch-Uncertainty not detection-native (use Ultralytics for mechanics, Torch-Uncertainty for metrics); Pohang IR coverage varies per run (pohang04 zero IR, pohang03 sparse — disclosed not hidden); small datasets → unstable variances (augmentation, seed spread reporting); **modality registration imperfect — "mitigation FAILED as stated"**: original mitigation was "fuse at decision level, which tolerates misalignment"; measured reality is decision-level fusion avoids the residual by almost never merging across modalities (0.05% partner rate at 0.85), and σ-arbitrated relaxed correspondence does not recover it (S1-NULL); correlated degradation of both sensors (both-degraded test row added, independence caveat).

### 1.16 Assumptions and limitations (scope §17, current text)
- Data pairing by nearest timestamp; decision-level fusion does NOT absorb the registration residual; the fusion aggregates rather than combines-agreeing streams at this registration quality; S1-NULL closed the alternative. Any claim about cross-modal *agreement* is out of reach of this system.
- Pohang IR gaps disclosed (pohang04 zero IR labels, pohang03 sparse).
- MIT dataset annotation status: not onboarded (corrected from an earlier claim of manual annotation).
- Adverse conditions are simulated (Albumentations), not field-collected.
- Conformal guarantees would assume exchangeability, violated by video — acknowledged.
- Real-time claim: single-pass by design, inference cost reported, no embedded deployment performed.

---

## PART 2 — REPOSITORY / RUNBOOK FACTS (from `README.md`, `HOW_TO_RUN.md`, `dataset_requirement.md`)

### 2.1 Repository layout
```
config.yaml            <- the ONE file to edit when moving machines
requirements.txt       <- pinned dependencies
src/uqfusion/          <- project package (installed editable)
scripts/               <- runnable tools (grid, consolidation, tables, audits)
docs/                  <- approved plans / design memos
data/                  <- datasets (gitignored)
runs/                  <- training & eval outputs (gitignored — all runs/eval/*.md and runs/queue_ideas/*.md are therefore NOT under version control, a recurring housekeeping flag in the docs)
phase1_benchmark/      <- the finished Phase 1 backbone benchmark: 93 runs, one CSV (CSVs/*.md tracked, run trees not)
```

### 2.2 Portability rules
No absolute paths in code — everything machine-specific goes through `config.yaml`. New machine = edit `paths:`/`device:` block, nothing else. Every trainable component has a CPU tiny-subset smoke path.

### 2.3 Compute model — two machines
- **Laptop, RTX 4080 12 GB** — all of Phase 3 trains here (Amendment 3 to Phase 3 prereg: dgxanode01 trains **nothing** in Phase 3, since it's only ~4.5% faster at steady state, not the ~1.23× or "7% slower" claimed at various earlier points — both retracted as measurement artifacts). GPU interpreter is system Python 3.13 set as `gpu_python` (the repo `.venv` is CPU-only torch). Frame-statistics scripts must still run under `.venv` because its numpy/opencv reproduce the development statistic files bit-exactly (GPU interpreter float32 sums differ by up to 4.4e-7).
- **dgxanode01, A100 MIG 3g.40gb** — Jupyter-only (no SSH), ran the Phase 1 grid; not used in Phase 3. Paths there: `/workspace/uqfusion`, `/workspace/pohang/visible`, `/workspace/derived` (NOT `Saroha_Work`/`Pohang_dataset/`/`runs/derived/` as some older docs said).
- Both machines have no GPU-hour cap (A1-1, H100 originally quoted but actual Phase 1 server hardware measured as MIG `3g.40gb` slice — 40,320 MiB / 60 SMs, NOT the full 80GB card, driver 535.161.08 / CUDA 12.5).

### 2.4 Dataset directory contract (`dataset_requirement.md`)
```
data/
├── pohang/
│   ├── data_vis.yaml / data_ir.yaml   # Ultralytics dataset yamls
│   ├── calibration/                    # per-run intrinsics+extrinsics (Phase 2, homography)
│   └── <images/, labels/ trees>
└── mit_marine/                         # planned, NOT present
```
- yamls: `train/val/test` as txt lists (preferred over dirs — auditable) or dirs; `names: {0: ship, 1: buoy}`.
- **Split rules (D6-rev):** pohang01 (night) allowed in training (owner's call); temporal buffer ≈10s within any run between train and val/test; no image in more than one split; verified mechanically by `scripts/audit_split.py` — must exit 0 before training. Night blocks requested in **test** too (real-night Table 3 row).
- Filenames must sort temporally within a run (epoch timestamp or zero-padded index); run identity read from a `pohangNN` fragment.
- **Images:** 640×640, **letterboxed not stretched** (aspect-preserving) — stretching would distort VIS (1.9:1 native) and IR (1.25:1) by different factors, hurting cross-modal overlap. Full-resolution originals archived (the one allowed lever for buoy/small-object recall, since architecture changes are out of scope). Labels normalized to the 640×640 stored image including padding.
- **IR bit depth:** natively 16-bit; must document normalization if converted to 8-bit (matters for masking thermal crossover and for what the Mahalanobis features see).
- No modality-copied annotations violating scope §10.2 (flagged as a Phase-2-scheduled filter, `§10.3 visibility-score filter` — in practice implemented differently, see Part 3).
- MIT multi-camera layout spec (never executed): pool all cameras for detection training; fix ONE canonical pair (left-VIS↔left-IR) for fusion; `pairs.csv` with nearest-timestamp matching ≤42ms; `scripts/make_pairs.py` + homography estimator "planned," never built.

### 2.5 Per-phase runbook highlights (`HOW_TO_RUN.md`)
- **Smoke suite order** (CPU, run before any GPU time): `smoke_benchmark.py` → `smoke_gaussian.py` (DFL path) → `smoke_gaussian_e2e.py` (end2end/YOLO26 path — **both gates must be green**, they catch different failure modes since σ rides different branches on the two head families) → `smoke_uq_pipeline.py` → `smoke_phase3.py`.
- **Dataset gate** (`verify_dataset_state.py`, run first on any new machine): re-implements split-fingerprint/leakage/night-filter-hash/balance-gate logic **independently** of the package (deliberately imports nothing from `uqfusion`) so it cross-checks rather than trusts the code. 13 checks; `--hash-labels` compares train-label content hash across machines (this is what proved the two Phase 1 machines trained on identical labels, hash `287b11c50b5a`, before the night-restore diverged them).
- **Phase 1 server order:** `audit_split.py` (must PASS) → `make_stride_subset.py` → optional 1-epoch timing dry-run → full grid (`run_benchmark.py`, resume-safe) → `consolidate_phase1.py --plan/--execute` → `measure_fps.py` (must run on CUDA torch build; refuses CPU device) → `make_table1.py` → IR top-2 confirmation grid. **Already done** — do not re-run against the existing 93-row record.
- **Phase 2 laptop queue** (`run_queue.py`, `dashboard.py` at `http://127.0.0.1:8770/8771`): pause is graceful (raises out of the loop at `on_model_save`, re-enters Ultralytics' own resume path with optimizer/EMA/scaler/epoch intact) — NOT a process kill (a real trap: pausing does not stop the OS process, confirmed via `Get-CimInstance Win32_Process` catching two live PIDs against one queue-dir simultaneously, 2026-09-04 incident). Two things had to be fixed for pause/resume correctness: `GaussianTrainer.get_model` now converts before loading weights (stock order intersected checkpoint against a σ-less model, dropping σ keys silently on every resume); early stopping now restored from the run's own `results.csv` (Ultralytics rebuilds `EarlyStopping` on resume without restoring it).
- **Batch sizing (RTX 4080 12.0GB), measured via `tune_batch.py`:** for `yolo26s`+σ@640, batch 32 is fastest (78.2 img/s) but peaks 11.42/12.0 GB (any concurrent GPU use pages); **batch 24 gives 75.8 img/s at 9.26 GB**, shipped default. Workers 4/8/12/16 span only 76.0–79.6 img/s (inside probe noise) — `workers: 8` retained, proven across nine multi-hour laptop runs.
- **Phase 3 server steps:** baselines (MC-Dropout, Ensemble) → prediction caches (anti-leakage: tuning and final-test caches must use different `--corrupt-seed`) → Table 2 (`evaluate_uq.py`, incl. DFL-derived §7.2 row via sigma-key switch on the same cache) → paired VIS↔IR frames + per-run homography (`build_pairs.py`, `derive_homography.py` — pairing from `Pohang_dataset/paired/*.csv`, **not** frame numbers, which disagree on 58% of pairs) → paired caches → gate-level ablation (`ablate_gate.py`, CPU-only over caches) → Table 3 (`run_fusion_eval.py`, the finalized D27–D30 system) → `eval_final_system.py` (bootstrap CIs + soft-weight ablation) → per-class AP (mAP is macro-averaged, hides a dead class — this is how the IR buoy-AP≈0 problem was first caught).
- **Historical bug on record:** the pre-2026-08-19 Table-3 invocation fed an 11,352-frame VIS val cache against a 2,234-frame IR val cache to an index-pairing function and died on an assert — fusion requires genuinely paired caches (built via `build_pairs.py`/`build_cache.py --images-list`).

---

## PART 3 — DATASET: PREPARATION, EDITS, DEFECTS, VERIFIED COUNTS

### 3.1 Pohang Canal + PoLaRIS — what it is
Multimodal maritime dataset, 7.5km route, Pohang canal/inner-outer port/near-coastal, Chung et al. IJRR 2023 (arXiv 2303.05555), KAIST MORIN lab. Stereo visible (2048×1080, 10Hz) + thermal IR (640×512, 16-bit, 10Hz). **PoLaRIS** is a *separate, later* annotation release (2024 preprint arXiv 2412.06192; **ICRA 2025**; code github.com/sparolab/PoLaRIS) providing YOLO-format boxes, 2 classes (ship, buoy). License CC BY-NC 4.0. 5 runs: pohang00 (day, dense both modalities), pohang01 (night), pohang02–04 (varying IR coverage). **pohang04 has zero IR labels** (VIS-only). Not spatially co-registered — pairing by nearest timestamp; residual measured (not assumed) at **3–6px median**, within-run swings up to ~10px.

### 3.2 Verified counts (2026-09-10, `scripts/verify_dataset_claims.py`) — supersede all earlier estimates
**158,319 images** (VIS 127,309 / IR 31,010); **1,183,736 boxes** (VIS 962,960 / IR 220,776); **28,388 paired VIS↔IR rows** (pohang00 10,786 / 01 11,990 / 02 3,739 / 03 1,873 / 04 0). Per-run image counts: pohang00 21,768 VIS/10,918 IR; pohang01 24,473/11,995; pohang02 27,795/6,175; pohang03 27,085/1,922; pohang04 26,188/0.
- **Corrections this reconciled:** earlier "~1.22M boxes" overstated by ~36k; earlier "pohang03 ~13k VIS" was wrong by 2× (actual 27,085).
- **Pairing is not by frame ordinal**: 16,544 of 28,388 pair rows have different VIS/IR indices (pohang03 offset ranges −155…+1, 16 distinct values) — true pairing key is the dataset's own timestamp CSV, not filename arithmetic.

### 3.3 IR preprocessing (`Pohang_dataset/IR_PREPROCESSING.md`, closes OQ-3)
Native 16-bit → delivered **8-bit grayscale PNG via per-frame min-max normalization** (each frame's own min→0, max→255 — NOT global or percentile). Consequences: thermal crossover can be visually masked (near-isothermal vessel gets stretched to full local contrast); no cross-frame radiometric comparability; raw 16-bit is not recoverable from the 8-bit product (per-frame min/max not stored). Letterbox 640×512→640×640: scale 1.0, pad top/bottom 64px each (value 114), label transform `cy'=0.8·cy+0.1, h'=0.8·h`, cx/w unchanged, clipped to [0,1]. Originals archived at `infrared_orig/`.
- **16-bit dynamic-range measurement**: median span only 702 raw counts, effective bits ~7.72 (fits comfortably in 8 bits); noise floor 2.1–4.19 counts. 51% of frames >1.5× range-inflated by outlier hot pixels, 27% >2×, 10% >3×, 3.3% >5× — a single hot pixel compresses whole-scene contrast under min-max normalization. A percentile-clip re-export was **built and tested and rejected**: +1.75% mAP50-95 but −1.6pt recall, net negative (handoff-2026-08-19).
- **cv2.imread(IMREAD_COLOR)** on raw 16-bit silently produces near-black garbage (4 distinct levels 28–31) — a documented trap for anyone touching raw IR.

### 3.4 VIS resize walkthrough (`Pohang_dataset/resize_walkthrough.md`)
2048×1080 → 640×640: scale factor 0.3125, scaled size 640×338, pad top/bottom 151px each (color 114,114,114), `cv2.INTER_AREA`. **127,309 VIS images** resized (train 101,919 / val 12,604 / test 12,786, images=labels exact match). **1,001,095 total boxes**, 0 out-of-[0,1] across all splits. Verified 640×640 on 50 samples/split. Consequence flagged repeatedly downstream: 47% of every stored 640×640 VIS frame is exactly the pad value 114 (real content is 640×338) — this is what broke the original "content-median luminance" night filter (§3.5).

### 3.5 Night-box visibility filter — the central dataset event of the project
**Original filter** (`scripts/filter_night_boxes.py --cut-dark pohang01:100`, commit `0be9718`, executed 2026-07-15 10:37, train-only, val/test never touched by design):
- Rule as intended: drop boxes in any pohang01 frame whose *content*-median luminance < 100 (0–255).
- **Bug discovered later**: `PAD_LEVEL=4` treats only near-black pixels as padding, but VIS letterbox pads with 114 — so the VIS "content median" is dominated by the 47%-gray-pad pixels, and the statistic that actually ran was closer to *"pohang01 frames whose content ~95th-percentile luminance < 100."* Outcome deemed sound for the 640 tree (17,502 frames confirmed by spot-check) but the **threshold does not transfer to unpadded/native-resolution images**.
- **Effect**: 17,502 of 96,275 scanned train label files emptied, **132,688 boxes dropped** (126,948 ship / 5,740 buoy). Hash before `fd60c0834fdd` → after `287b11c50b5a`. Reversible (`*.pre_visfilter` backups, `--restore` flag, manifest at `runs/visfilter/visfilter_manifest.json`).
- **Per-box audit** (`box_scores.csv`, thresholds int=45/grad=8/contrast=10, dark-median=40): of 120,829 scored boxes only **38,135 flagged** (fail all three tests — legitimately bad); **82,694 boxes were scored, would NOT individually have failed, but were deleted anyway** because their whole frame was cut; plus 11,859 boxes never scored (frame median 40–100). Flagged-vs-kept populations **overlap** on 2 of 3 axes — the frame-level cut was not a clean separation.

**The restore (`docs/prereg-night-label-restore.md`, executed 2026-09-02, commit `030244e`):**
- Change: restore all 132,688 boxes, then re-drop only the 38,135 individually-flagged → net **+94,553 boxes**.
- Endpoint: night-only VIS mAP@50-95 (bands DEAD<0.005, WEAK 0.005–0.02, **ALIVE≥0.02**); day-guard floor −max(2×sd_paired, 0.002).
- Fine-tuned from `gauss_vis_seed0/best.pt`, 25 epochs/patience 10, early-stopped epoch 20, best epoch 10, 5.82h.
- **Result: night val mAP@50-95 0.0000 → 0.2520** [CI 0.2473,0.2567], se 0.0024 — **12.6× the ALIVE threshold**. mAP50 0.0000→0.4957. Day guard PASSED (delta +0.0162 vs floor −0.0045; caveat: day gain driven by buoy AP not ship AP, likely just extra training budget, not the restore itself). **Verdict: ALIVE.**
- **Interpretation, the project's own words:** "The shipped VIS detector was not blind at night. It was UNTRAINED at night." The 0.0000 that stood in every earlier night table was manufactured by the filter, not a sensor limitation.
- **What did NOT change**: the veto still fires 100% on the *shipped* checkpoint (verified genuinely 0.0000 there); the fused `crossmodal26m` benchmark was not re-run under this event (deliberate — night veto still active). This finding **licensed** a new pre-registration about the veto itself (→ Part 6, the V1/V2/V3 saga), it did not by itself change any shipped number.
- **Provenance/hash correction (Amendment 1 to the Phase 3 prereg, `docs/g1-label-reconciliation-2026-09-10.md`):** the widely-quoted post-restore hash `b92739202127` (also recorded in user memory) is actually a **tree-scope** hash (127,309 files / 962,960 boxes), not the **train-scope** hash. The correct train-scope hash is **`8ed69b5974ed`** (96,275 files / 711,444 boxes). Box arithmetic reconciles exactly: pre-filter 749,579 (`fd60c0834fdd`) → post-cut-dark 616,891 (`287b11c50b5a`) → post-restore 749,579 → post-re-drop of 38,135 = **711,444** (`8ed69b5974ed`); 711,444−616,891 = 94,553, matching the restore exactly. The server (dgxanode01) remained on `287b11c50b5a` (pre-restore) — reconciliation is a data transfer, not an investigation, once this was understood.
- **Still-open incident (OQ-13):** on 2026-09-03 21:19, 7,591 pohang01 train label files were silently rewritten back to pre-filter content (+18,772 boxes) with no project script running, an idle terminal, and no live Python process. `A:` is a local NTFS fixed disk with VSS/File History Stopped/Manual — no scheduled snapshot service could explain it; a one-off manual revert is not excluded. `scripts/label_hash_ledger.py` now keeps an append-only hash/count/mtime timeline so a recurrence has a bounded window; it separately found **8,237 orphan label files belonging to no split list**, invisible to every existing gate.
- **Phase 1 ranking robustness check**: sliced the 27 archived Phase-1 checkpoints (all pre-restore) by day/night without retraining — night AP is 0.0000 on all 27 (uniform, not selective), and day ranking vs pooled ranking agree on all 9 positions. **The published Phase 1 backbone selection is unaffected** by the night-label defect.

### 3.6 Splits — two rounds, both list-only
- **Round 1** (`prepare_pohang.py`, 2026-07-10): delivered split was frame-interleaved (train N / val N+1) — with 10Hz video this leaks near-duplicate frames across splits. Re-split into contiguous per-run ordinal blocks (fixed positional windows, ~80/90% cut points), same ordinals for both modalities (VIS/IR pairs and stereo L/R never split across sets), guard band at each boundary. `audit_split.py` went FAIL→PASS.
- **Round 2** (`scripts/resplit_balanced.py`, 2026-07-14): Round 1 passed leakage but **failed balance** on the server — val came out 77% buoy vs 5% global share (1.35 vs 8.10 boxes/frame) because fixed positional windows sample one mission phase. Replaced with an **interleaved K-block split**: cut each run's shared timeline into K equal blocks, cycle of 10 (block%10==4→val, ==9→test, else train, 80/10/10 by construction), guard bands at every boundary, K raised until leakage+balance+coverage gates all pass. This is the split used for the grid, verified via a uniform `split_fingerprint` on the 2026-07-31 run.
- Housekeeping fix (`fix_split_lists.py`, 2026-07-12): Ultralytics only resolves a list line relative to the txt file's folder if it starts with `./` — bare lines resolved against cwd and failed to load; rewrote lists with the prefix (no semantic split change).

### 3.7 Full-resolution twin (2026-08-01)
`D:\Datasets\Pohang_dataset_full\`: VIS 2048×1080 native, IR 640×512 native, same split + same night filter (**ported, not recomputed** — the padding-dependent threshold does not transfer, but the box-count match to the 640 tree was verified exact: 132,688). Images hardlinked (0 new VIS bytes for 309GB); labels copied not hardlinked. Every benchmark CSV row stamped with `split_fingerprint` + `classes` tag so a re-split/filter change can never be silently averaged against older rows.

### 3.8 Derived stride subsets and split fingerprints
`data_vis_stride2.yaml`: 48,136/96,275 train frames kept, fingerprint `682dbe9f0f05`. Other strides exist (`stride10`≈10k, `stride19`≈5k). VIS train stride 2 is production; IR train stride 1 (full 23,279 frames). Class filtering (`--classes 0`, ship-only) is a load-time Ultralytics filter, not a dataset edit — it writes no files.

### 3.9 g5 — holdout-free splits for Phase 3 (2026-09-10)
Built `_p04out`-suffixed lists by **filtering** (not regenerating) existing lists — drops pohang04 rows only, preserves every surviving frame's stride-selection identity. Counts: VIS train stride2 48,136→**38,295** (9,841 dropped); val 11,352→**9,009** (2,343 dropped); test 11,445→**9,054** (2,391 dropped). IR needed no filtering (0 pohang04 rows in any IR list — no thermal labels for pohang04 exist at all). **Composition shift, not neutral**: val night share rose 18.2%→**23.0%** once pohang04 (all-day) is removed — any comparison against pre-Phase-3 pooled val numbers now compares different populations. Gate bug found and fixed: `assert_holdout_excluded.py` originally checked *all* lists on disk (impossible to pass, since originals are deliberately retained) — fixed with `--yaml PATH` (narrow, training-consumed lists) vs a no-arg survey mode (informational, 15 lists/58,144 rows).

### 3.10 Contamination audit (`docs/holdout-contamination-audit-2026-09-10.md`) — the finding that forced Stage 0
`runs/derived/maha_fit_vis.txt` (the Mahalanobis reference-set list) holds **819 of 4,000 frames (20.5%) from pohang04**. This list feeds every `gauss_vis_train_clean.pkl` cache and fits the **shipped** Mahalanobis reference distribution — meaning the OOD scorer, as shipped, would score pohang04 as partly in-distribution because it literally trained on 819 of its frames. Full audit table across every list on disk: `day_val_vis_stride1.txt` also contaminated (2,343/9,284, but exploratory only, not shipped); `maha_fit_ir.txt`, `paired_val_vis/ir.txt` (0/2,232, feeds **194 caches**, every benchmark cell), `ladder_vis/ir.txt` (0/744), and `ctx.FIT_RUNS/TUNE_RUNS/TEST_RUNS` (select by run name — pohang04 in none of them) are all **clean**. Fix status **half done**: a clean replacement list `maha_fit_vis_p04out.txt` (3,181 frames, filtered not resampled) exists, but the cache itself (`gauss_vis_train_clean.pkl`) has **not yet been rebuilt from it** — flagged as "a list does not retro-fit a cache," queued as a GPU job before the single look.

### 3.11 Dataset audits — both PASS (`runs/audit/`)
- **VIS**: splits 96,275/11,352/11,445 (pohang00 16,376/1,672/1,912; pohang01 18,826/2,068/1,629; pohang02 20,424/2,690/2,864; pohang03 20,962/2,579/2,649; pohang04 19,687/2,343/2,391). 0 cross-split duplicates, 0 temporal-proximity violations. **PASS**.
- **IR**: splits 23,279/2,234/2,518 (pohang00 8,229/836/950; pohang01 8,800/1,034/1,111; pohang02 4,844/247/330; pohang03 1,406/117/127). 0/0. **PASS**.

### 3.12 Calibration/registration facts (`docs/calibration_and_registration.md`)
Homography `H_∞ = K_vis·R·K_ir⁻¹` (3×3, IR→VIS) computed from Pohang's shipped per-run calibration (Route A) or RANSAC on box correspondences (Route B, for any future non-calibrated dataset like MIT). Storage convention: `data/<dataset>/calibration/H_<ircam>_to_<viscam>.json`. Validation targets: median reprojection ≲5–8px, visual overlay, box-IoU test. Real measured residual (`runs/eval/x_registration_drift.md`): median |d| 3–6px per run, with **within-run drift up to a 10px swing** (pohang00, across capture-order bins) — comparable in magnitude to a −4.18px *between-run* swing that already defeated an earlier attempt at a single global correction; a per-run constant still leaves 0.4–1.5px on the table vs. a full time-varying model, which was never built.

### 3.13 MIT Marine Perception — confirmed absent (2026-09-10)
No `data/` directory for it exists; `datasets.mit_marine.vis_yaml`/`.ir_yaml` are both `null` in `config.yaml`; zero `.py`/`.yaml`/`.json` references anywhere in the repo. Deferred by decision D10 (2026-07-07) to "Phase 2+," never onboarded. The claim "we have manually annotated a subset of images from this dataset ourselves" that appeared in an earlier scope.md draft is **retired as false** (bibliography review, 2026-09-10) — no such annotation exists anywhere in the repository.

---

## PART 4 — PHASE 1: BACKBONE BENCHMARK (COMPLETE)

### 4.1 Design
Ship-only (`--classes 0`), `data_vis_stride2.yaml` (fingerprint `682dbe9f0f05`), 48,136 train / 11,352 val images, 105,098 val instances, 100 epochs/patience 20 (no run reached 100; observed range 24–61), imgsz 640. Original 6-variant×3-seed shortlist (D8: yolov8s/9s/10s/11s/12s/26s) was **amended (D24, 2026-07-11)** to the **full size ladder per family** — 31 variants (v8 n/s/m/l/x, v9 t/s/m/c/e, v10 n/s/m/b/l/x, v11/v12/v26 n/s/m/l/x) at **single seed [0]**, later some variants got 3 seeds. Selection rule: top mAP50-95, tie-break DFL-present > simplest fork > FPS.

### 4.2 Two campaigns, merged into 93 rows (`phase1_benchmark/results.csv`)
- **`main`**: 27 rows, 9 variants (yolo12{s,m,l,x}, yolo26{n,s,m,l,x}) × 3 seeds; split across DGX server (18 rows) and RTX 4080 laptop (9 rows). Split fingerprint `682dbe9f0f05`.
- **`pilot`**: 66 rows, 23 variants × 3 seeds, one machine, split fingerprint `f0220e716277` — **server later wiped, split is irrecoverable**; possible train/val contamination between the two campaigns' splits cannot be quantified (pilot val starts 17 frames earlier than main val, both cut contiguous blocks from the same underlying sequence). `docs/phase1-pilot-grid.md` documents this caveat; **any comparison crossing the `grid` column needs it read first**.
- Only `yolo12s` ran in both campaigns (retired to `extra/`, not the headline tables): main 0.2783 (n=3) vs pilot 0.2810 (n=2), gap +0.0027, inside main's own seed sd (0.0109).

### 4.3 `phase1_benchmark/README.md` — record structure
39 columns in 5 groups: identity/config (`run_id, grid, variant, seed, trained_on, scored_on, batch, imgsz, epochs_cfg, patience_cfg`); metrics (`precision, recall, map50, map50_95` from explicit best.pt val); cost (`params_m, gflops, train_time_s`); training dynamics (`best_epoch, best_map50_95_curve, last_epoch, epochs_trained, patience_gap, patience_fires_epoch, stopper_ref_epoch, stop_reason, resume_segments, dup_epoch_rows, epoch_regressions`); status/provenance (`admissible, exclude_reason, run_path, train_dir_orig, val_dir_orig`, library/torch/commit, `classes, note`). Fitness = mAP50-95 alone (ultralytics 8.4.90 — corrected from an earlier belief it was a 0.1/0.9 blended fitness). 92/93 admissible; sole exception `main_yolo12x_seed1` (patience_gap=14, lower-bound-only reading). `stop_reason` breakdown: early_stop 87, early_stop_after_resume 2, killed_past_patience 2, shared_dir_past/before_patience 1 each.

### 4.4 Results — seed-means (mAP50-95, `main` campaign, admissible)
```
yolo26x  n=3  0.3049 ± 0.0020
yolov9e  n=?  0.3033 ± 0.0066   (pooled/pilot figure, per one digest pass)
yolov9c  n=?  0.3019 ± 0.0057
yolo26m  n=3  0.3016 ± 0.0050
yolo12x  n=2  0.3007 ± 0.0049
yolov8x  n=?  0.2999 ± 0.0010
yolo26l  n=3  0.2998 ± 0.0026
yolo11l  n=?  0.2994 ± 0.0041
yolo12m  n=3  0.2906 ± 0.0046
yolo12l  n=3  0.2870 ± 0.0096
yolo26s  n=3  0.2813 ± 0.0045
yolo12s  n=3  0.2783 ± 0.0109
yolo26n  n=3  0.2540 ± 0.0058
```
**Headline (negative result, stated as such throughout):** top-8 span only **0.0055 mAP50-95** vs seed sds of 0.0010–0.0066 — **not separable**; `yolo26x`'s 0.0016–0.0033 lead over the next-best sits below its own seed sd. Order is not even stable across metric-reading conventions (Table-1 vs in-training curve-peak orderings disagree). The one **robust** finding is the capacity floor: every n/t-scale model across 5 families lands **0.2486–0.2567**.
- Robustness cross-check (Table-1 vs curve-peak): mean offset +0.00110±0.00092, split by family not machine (YOLO26 family +0.0003, YOLO12 family +0.0021 — plausibly because YOLO26 is NMS-free so both validators share more of a code path).

### 4.5 Backbone selected: `yolo26m` (D25, 2026-08-17) — why not the nominal leader `yolo26x`
Rule-compliant under D8/D24: `26m` sits inside the leading group's pooled seed sd; since no YOLO26 variant has DFL, the tie-break falls to FPS, which `26m` wins decisively for the fusion use case — **two detectors run per frame**, so throughput matters twice. Measured: 26m 0.3016 mAP / 57.0 FPS single-stream / **28.5 FPS two-stream** vs 26x 0.3049 mAP / 30.7 FPS / **15.3 FPS two-stream** — 26m costs −0.0033 mAP (a gap that exceeds 26x's own sd of 0.0020, disclosed as the "soft spot" of the decision) for **1.86× two-stream throughput**. Main-backbone training moved to the full-resolution twin (native 2048×1080) because the 640 tree stores VIS pre-letterboxed at 640×338 (47% padding) — native frames are the actual `imgsz`/small-object lever.

### 4.6 Defects found and resolved during Phase 1 (full account: `docs/phase1-experimental-record.md`)
1. **Ultralytics version defect (critical, resolved):** 8.4.7 reports mAP50-95 **~0.034 higher** than 8.4.90 for *identical weights and data* (precision/recall shift only ~0.01% — localizes to the AP-integration code, likely `compute_ap`). One machine upgraded 8.4.7→8.4.90 mid-project (2026-08-11); `yolo26m` seed0 re-scored 0.3452→0.3061 and is now tagged `ultralytics_version: 8.4.7+val8.4.90` in the record — **this row must never be compared bare against other rows.**
2. **Shared-directory collision (resolved):** `yolo12x` seed0/seed1 concurrently wrote to one run directory. An earlier withdrawn claim of "0.0072 noise at identical settings" was actually batch 16 (killed epoch 16) vs batch 8 (natural stop epoch 29) — not a seed-noise measurement at all. Resolution: seed0 kept as an artifact-backed 0.30414 row; seed1 marked inadmissible (only 14 epochs past its own peak vs required patience 20) → `yolo12x` reports at n=2.
3. **Heterogeneous batch size** (32/24/16/8 across runs, machine-dependent) → effective batch/weight-decay interaction differs ~12.5%; bounded effect estimated +0.0017, below the smallest observed seed sd (0.0020) — disclosed as a limitation, not corrected.
4. **Early-stopping resume bug:** `EarlyStopping` rebuilt with `best_fitness=0` on resume (not restored from history) → a resumed run trains past where an uninterrupted run would have stopped. Admissibility rule: a run is admissible iff it trained ≥patience epochs past its *own* peak epoch. 26/27 `main` rows pass.
5. Git commit unrecorded for 17/18 server rows (provenance gap, disclosed).
6. `train_time_s` under-reports resumed runs (segment-only, not cumulative) — the reported 266.8h sum is a lower bound.
7. Two coexisting opencv distributions — incidentally useful, since it helped isolate the ultralytics-version defect above.

### 4.7 FPS measurement (`fps.csv`, 184 measurements)
GPU clock pinned at 1500MHz (0.0% spread) for the valid protocol; an earlier unpinned run (1200–2400MHz swing) is flagged invalid and must never be quoted. fp16 helps compute-bound large models (yolov8x +69%, yolo26x +35%, yolo12x +36%) but not nano models.

### 4.8 IR equivalence ladder re-analysis — a statistics correction, not a new experiment (R-F3, 2026-09-10)
The original stopping justification for the 44/93-complete IR architecture ladder ("ANOVA F(12,26)=1.037, p=0.447 ⇒ architectures are indistinguishable") was **wrong statistics**. Minimum detectable spread at α=0.05/power=80% is **0.02067**; the observed spread of variant means is only **0.01193** (0.58× — the design was underpowered, not evidence of equivalence). Tukey HSD 95% CI on the largest gap (yolo26l−yolov10n): **[−0.00314, +0.02700]** — the upper bound alone exceeds the entire observed range, so no equivalence claim at any δ≤0.027 is supported; without multiplicity correction that same interval [+0.0034,+0.0205] would exclude zero. **Correct statement: "the ladder could not resolve architecture differences at this scale," not "architectures are indistinguishable."** The decision to stop the ladder still stands, but on the *other* two reasons only: the architecture was frozen 6 days before the queue was created, and the ladder trains 2-class while the deployed IR config is nc=1 with a p2feat neck that was never in the ladder at all.

### 4.9 Day/night slice of the Phase 1 grid
Splits the 27-model grid by day (9,284 frames) vs night (2,068 frames, **0.0000 AP everywhere** — pre-restore labels). Day AP runs uniformly ~0.05 higher than pooled published numbers, but day-only ranking and pooled ranking **agree on all top-3 positions** (yolo26x/26m/26l) — the uniform night handicap does not change model selection; the published winner stands.

---

## PART 5 — PHASE 2: GAUSSIAN σ² UQ HEAD

### 5.1 Head integration design (D17, 2026-07-07; ported D26, 2026-08-17)
**In-place conversion, no Ultralytics fork.** A loaded `DetectionModel` gets a fresh `cv4` log-variance branch bolted onto its live `Detect` head + class swap (`GaussianDetect`/`GaussianDetectionModel`); trainer is a thin `DetectionTrainer` subclass (4th loss term + warm-up epoch-sync callback). σ parameterized as log σ² over LTRB distances in stride units (same targets as DFL), converted to pixels at inference; rides through NMS as extra channels. **Gradient policy default (A4-11 conservatism):** σ branch reads detached features, NLL sees detached μ → deterministic detector trains bit-identically to baseline **by construction** (all relaxable via a `gaussian:` config block). Warm-up = NLL weight 0 (no param-group surgery needed).

**Porting to YOLO26's end2end head (D26, three forced differences):**
1. **σ rides the one2one branch only** — inference decodes from one2one; the two branches (topk 10 vs 7/topk2 1) have different assignments, so σ on one2many would describe a predictor that never reaches the output.
2. **`postprocess` overridden** to gather σ with the boxes' top-k index — stock code splits `[4, nc]` exactly, so trailing σ columns would otherwise be misread as class logits.
3. **μ = raw box output, NLL target unclamped** — at `reg_max=1`, `bbox2dist(..., reg_max-1)` collapses every target to the constant −0.01, so clamping must be bypassed.
- NLL added **outside** the o2m/o2o loss schedule (which moves the one2one weight 0.2→0.9 across a training run) so `nll_gain`/warm-up keep a stable meaning and the detector's three loss terms stay bit-identical.
- **Known, disclosed limitation:** `sigma_detach_features=False` cannot be honored on end2end heads (Ultralytics detaches one2one upstream regardless) — forced True with a warning; that specific ablation needs a plain-`Detect` (non-end2end) backbone instead.
- Gated by `smoke_gaussian.py` (DFL family) and `smoke_gaussian_e2e.py` (end2end family) — **both must be green**, since each catches failure modes the other structurally cannot see (five structural checks, no training needed, run in seconds).

### 5.2 DFL-derived uncertainty (D18, §7.2 option a)
Per-coordinate std of the DFL bin distribution, extracted at inference from the *same trained model*, zero extra training. **Undefined on the selected YOLO26 backbone** (`reg_max=1`, no DFL) — this is OQ-10, and the digests reviewed here do not show it as ever having been resolved into a headline Table 2 row; a compute-matched `yolo12m`+cv4 control (21.9M params/75.4 GFLOPs/0.3016 mAP-equivalent scale) was built as a dead-end diagnostic branch, never fused, appearing only as two extra Table-2 rows in early planning.

### 5.3 Laptop architecture test (2026-08-18)
640 dataset, `yolo26s`, VIS stride5 (19,256 frames)/IR stride2 (11,640), 100ep/patience20, batch24/workers8.
| run | epochs | best ep | mAP50-95 |
|---|---|---|---|
| gauss_vis_seed0 | 54 | 34 | 0.25049 |
| gauss_vis_seed0_ft (mosaic-off) | 10 | 8 | 0.25053 |
| parity_vis_seed0 (no σ) | 68 | 48 | 0.26124 |
σ head confirmed working: NLL=0 during 5-epoch warmup, monotone active from epoch 6 (0→−0.169 by epoch 10); box_loss unaffected by activation.

### 5.4 Four operational defects found and fixed on the laptop queue
1. σ weights **silently discarded on every resume** — `BaseModel.load` intersects the checkpoint against a model that has no `cv4` yet if converted after loading; fixed by converting before loading.
2. Early-stopping forgot its best-epoch state across a pause/resume.
3. `os.replace` onto `live.json` could crash the trainer (WinError 5, file-locking race) — fixed with retry + in-place write.
4. `cv4` initialization advanced the global RNG, offsetting all subsequent random draws — fixed via `torch.random.fork_rng()`.

### 5.5 The unresolved parity gap (§12.1 check) — OPEN
D17 claimed **bit-identical** baseline training with σ attached. Measured max|Δ| = **3.9e-2** in early losses, even after fixing an RNG divergence (`reseed_at_train_start`) that made the two runs' data streams byte-identical. Ruled out as causes: nondeterminism, data/augmentation, config, assigner top-k, gradient-norm-clipping *leakage into the sigma path itself* (cv4 grad confirmed exactly 0 during warmup), AMP step-skip differences. **Prime remaining suspect:** floating-point non-associativity in cuDNN under autocast, given that gradient-clipping is active on nearly every step (observed grad norms 786/284 against `max_norm=10`). `smoke_parity.py` is deliberately kept RED pending resolution.
- **Formalized as R-C1 in the 2026-09-09 external review (F06):** shared gradient-norm clipping across *all* parameters (norm bound 10) mathematically **couples** the detector and σ-branch gradients even though σ's own gradient contribution is detached upstream. Analytical counterexample given in the review: detector-only grad 6 + an independent, undetached σ grad of 100 → the clipped detector gradient becomes 0.598923 instead of 6 — i.e., detaching σ's *contribution to the loss* does not guarantee the clipping *norm* is unaffected by σ's existence. **Open decision Q4** (in the TODO backlog, Part 10): adopt a strict bit-parity contract, or relax to a pre-registered non-inferiority margin.

### 5.6 Mosaic augmentation — never fully closed
Training runs mosaicked throughout, scored on clean (unmosaicked) validation, across all 93 Phase-1 rows plus every Phase-2 UQ run. A second-stage "mosaic-off" continuation (Option C) was measured **neutral** (+0.00004 mAP) but is structurally unable to answer the real question, because continuing a run resets its EMA state. The methodologically correct construction — a single run with a patience-keyed mosaic-off callback (Option B) — was **never built**.

### 5.7 Checkpoint-selection instability (D31, 2026-09-01) — a genuine methodological finding
On the **best-epoch** convention, MC-Dropout appears to lead the UQ arms by +4.53 sd over the ensemble; on the **epoch-mean** convention, the *same* MC-Dropout run is the **worst** arm by −3.00 sd — **the ranking inverts** depending purely on which summary statistic is used to read a noisy training curve. Root cause: `best.pt` is a `max()` over ~10 noisy validation epochs, which rewards the *noisiest* run, not the best one — the MC-Dropout arm's epoch-4 peak stood ~2σ above its own neighboring epochs (a single-epoch spike) and then declined monotonically at roughly 10× the control run's slope; the defensible reading is "MC-Dropout costs accuracy here," the opposite of what best-fitness alone reports. Between-seed sd of best-fitness (0.00212) was found to be **2–5× smaller** than the within-run epoch-to-epoch noise (0.0035–0.0106) — i.e., "the selection noise baked into each headline number is larger than the between-seed spread we'd otherwise quote as its uncertainty." **Decision:** keep best-epoch as the reported checkpoint (historical convention; earlier runs have no other checkpoint on disk) but **report the 10-epoch mean alongside**, and — more importantly — **do not rank UQ arms on mAP at all**; the comparison is carried by calibration metrics (D-ECE, NLL, AUSE, AURC), where separations between arms are large relative to this noise.

### 5.8 Cross-machine reproducibility instability (`docs/mc-server-control-2026-09-01.md`)
Same seed/recipe/data run on laptop vs. server (dgxanode01) produces an **opposite winner depending on which estimator is used**: best-epoch favors the laptop (+0.00884), epoch-mean favors the server (+0.00497). Laptop epoch-scatter is 2.4× the server's (sd 0.01056 vs 0.00440), and the laptop's "selection premium" (best−mean) is 2.6× larger. No stable machine offset was found (mean +0.00497, sd 0.00789, range [−0.00884,+0.01421], positive on 7/10 comparisons). **Consequence: best-epoch numbers are not comparable across machines**, which is part of why Phase 3's Amendment 3 moved *all* Phase 3 training onto the laptop alone.

### 5.9 MC/ensemble estimand mismatch — R-C2, open (2026-09-10 review)
`clustering.py` measures disagreement *among surviving detections* (confidence-weighted mean box, unweighted coordinate SD) — this omits the within-member variance term of the correct decomposition `Var(Y|x) = E_m[Var(Y|x,m)] + Var_m(E[Y|x,m])`. Consequence: two identical-but-wrong detections produce σ=0 → NaN NLL, zero coverage; and M=5 in the ensemble is one replicate of the M-member statistic, not five independent measurements of it. **Open decision Q3** (blocks the UQ table): should the MC/ensemble estimand be redefined as disagreement-ranking (cheap, matches what's currently computed) or predictive-likelihood (correct in the textbook sense, more code)? Not yet decided as of the latest reviewed docs.

### 5.10 Two distinct MC-Dropout bugs, later correctly disentangled and NOT to be conflated
1. **2026-08-21/23 — wrong branch:** dropout inserted on the **discarded one2many branch** of YOLO26's end2end head, so it never executed at inference (proven with a call counter; two earlier "verification" attempts had been silently wrong). This caused a real VIS training divergence at epoch ~18–22 (classification loss exploded, model collapsed to predicting nothing), independently reproduced across seeds. Fixed by moving dropout to `one2one_cv2`/`cv3`, detached from trunk gradients; verified via 5 independent checks (module gate, in-memory placement check, saved-checkpoint placement check, 10-epoch real-data smoke with confirmed non-zero variance, server-side re-verification); a new divergence watcher (val/cls_loss > 1.5× trailing-5-epoch median, calibrated against the broken run's own curve) fires correctly at epoch 16.
2. **2026-08-31 — key-renumbering on checkpoint reload:** `insert_head_dropout` renumbers the final conv layer (`one2one_cv2.0.2`→`.0.3`), which causes Ultralytics' key-based checkpoint loader to silently drop 12 tensors (62,460 params, 0.28% of the model but **100% of the output heads**) whenever an `_ft` (fine-tune) stage reloads from an MC checkpoint by key. The eval/inference path was always safe (it restores the pickled module object, not by key). This directly explains **`mc_vis_seed0_ft`'s genuine failure** (best epoch 2, mAP50-95 0.1512, monotonically worse after) recorded in `results-compiled-2026-08-31.md`. Fix verified: 768/768 tensors now transfer correctly (was 756/768); the IR twin had the *opposite*-direction version of this bug pre-2026-08-23 (dropout hit the discarded branch, masking a different zero-variance defect there).

### 5.11 Off-by-one fitness bug and its downstream correction (`docs/results-compiled-2026-08-31.md` §1)
`run_queue.py`'s `on_fit_epoch_end` recorded `stopper.best_epoch` (1-based) against `trainer.epoch` (0-based), so `best_map50_95` was actually read from the epoch *after* `best.pt`. Affected only the first 6 server UQ-arm runs (pre-code-sync); zero effect on the 43 laptop runs checked or later server VIS runs. Corrections applied: `gauss_vis_seed0` 0.25552→0.25591 (+0.00039); **`gauss_vis_seed0_ft` 0.22642→0.24750 (+0.02108)**; `mc_vis_seed0` +0.00013; `mc_vis_seed0_ft` +0.00052; **`ens_vis_seed0` 0.24402→0.25086 (+0.00684)**; **`ens_vis_seed0_ft` 0.23617→0.24711 (+0.01094)**. Consequence: an earlier claim that "mosaic-off hurts the σ-head badly" was a pure measurement artifact — the real loss was 0.008, not 0.029. Corrected ensemble spread narrows to 0.00533 (sd 0.00212) from an uncorrected 0.01361 (sd 0.00531) — much tighter than it first looked.
- Also documented there: cross-machine paired control `gauss_vis_seed0` laptop 0.24959 vs server 0.25591 — delta **−0.00632, larger than the entire 5-seed ensemble spread** — confirming (again) that a laptop retrain of a server-trained parent is not comparable; a fine-tune stage must start from the server's own downloaded checkpoint.

### 5.12 Batch-ceiling measurement (2026-09-10/11, `docs/batch-ceiling-2026-09-10.md`)
Windows WDDM pages rather than OOMing on an over-large batch (17× slowdown, silently) — must probe by measured ms/img, not crash/no-crash. IR (`yolo26m-p2feat`): ceiling batch 14. VIS (`yolo26m`): ceiling batch 12 **by throughput** (batch14/16 measured ~28% *slower* despite ≥2GiB headroom, reproduced under order-reversal, not a thermal artifact). Common batch chosen: **12**.
- **Addendum, 2026-09-11 — production contradicted the VIS probe.** The actual `p3_vis_seed0` production run measured batch 16 as **16% faster per image** than batch 12 — the exact opposite sign of the probe's finding. Root cause unconfirmed; leading candidates are that the probe's "30%-fraction" sample wasn't actually random (it took only pohang00+01, missing 02/03, and pohang01 is box-sparse) and/or a `val=False` (probe) vs `val=True` (production) discrepancy. **Decision: batch stays 12 anyway** for Stage 2 consistency — pre-registration protection was judged to outweigh the ~8h aggregate cost of training at a suboptimal batch size.

---

## PART 6 — FUSION / GATING MECHANISM: FULL EVOLUTION

This is the project's central engineering narrative — six successive rewrites of the decision layer, each triggered by a specific measured failure of the previous one.

### 6.1 (a) 2026-08-19 — first working gate (`docs/fusion-gate-experiment-record.md`)
- Photometric term via **`p05`** (5th-percentile luminance, content-region only) chosen because it is immune to glare/fog spoofing where a simple mean is not (spoof fractions measured: p05 under glare 0.000 / fog 0.243 vs mean under the same conditions 0.354 / 0.747).
- Threshold: **margin rule** (no optimizer) — `mu_b=10.500, tau_b=2.625`, placed in the empty gap between the corrupted-fit maximum and the clean-fit minimum p05.
- **Hard veto**: a failed modality is *excluded from the WBF input list*, not merely down-weighted (down-weighting alone left `w_vis=0.432` on a genuinely blind stream, because WBF renormalizes weights across whatever is handed to it).
- **Root cause of the original night-blindness identified here**: the Mahalanobis reference set itself contained 782 of 4,000 pohang01 (night) frames — night was *in-distribution* by construction, so `D_night(28.4) < D_day(30.0)`. The same raw darkness produced a 21× different fusion response depending on whether it was synthetic lowlight (`w_vis=0.037`, correctly distrusted) or real night (`w_vis=0.792`, wrongly trusted) — "no corruption ladder, however many severities, could have surfaced this," since synthetic lowlight and real night are photometrically similar but statistically opposite relative to the fit set.
- Result at this stage: gated fusion 0.0813 vs `ir_only` 0.0810 on clean/night — first time ahead, in 3 of 4 tested conditions.
- σ-weighted WBF implemented and verified numerically correct (6.3e-08 agreement with stock WBF under constant σ) but measured as a **null** (±0.0001) — because only 0.11% (31/29,042) of VIS boxes have an IR partner above iou_thr=0.85; there is almost nothing for inverse-variance weighting to act on.

### 6.2 (b) 2026-08-20 — the claim didn't survive bootstrapping (`docs/followup-analysis-2026-08-20.md`)
Paired bootstrap (1000 resamples) on the night-fusion "win": delta +0.0003, CI **[−0.0002,+0.0008]**, 11.4% sign flips — **spans zero, never a real win.** Swapping in a better IR detector (p2feat) later **inverts** the sign entirely: `ir_only` at night jumps 0.0810→0.1087, and gated fusion then trails it on 6 of 8 cells. The photometric *soft* term was found **fully redundant** with the hard veto — `veto_only` equals the full gate+veto configuration in every measured cell (the interaction term cancels exactly), leading to `bright_soft=False` being adopted. Veto hysteresis (dilate-15, applied in capture order) partially closes the fog/night gap: 0.0789→0.0809 (still tied with, not beating, `ir_only`'s 0.0810). Registration drift measured directly: within-run x-swing 4.9–10px exceeds the −4.18px between-run swing that had already defeated an earlier global-correction attempt — confirming a per-run correction doesn't hold up either. Two further nulls established here: top-k truncation of detections, and cross-modal score calibration.

### 6.3 (c) 2026-09-01 — the fog fix (`docs/veil-veto-and-rule-sweep-2026-09-01.md`)
`p05` (a histogram statistic) is blind to fog — fog raises the black floor without darkening the frame overall. A second veto axis, **`lap_var`** (Laplacian variance, texture/edge sharpness), was added with its own hard novelty bound `tau_lap=508.674` (the minimum over clean fit frames), filtered by a **majority-15** window — the *opposite* filter direction from brightness's dilate-15, since the two axes have opposite failure modes (confirmed by a dedicated smoke check that they cannot share one filter). Result: fog/day 0.0126→0.0166 (a small residual −0.0011 gap attributed to single-list-WBF post-processing artifact, not the gate itself). A 29-rule sweep found 22 of 29 candidate veto formulations tie at a worst-cell gap of −0.0180, all sharing the *same* worst cell: **lowlight/day**. Key discovery: lowlight/day is **not fixable by any VIS-side statistic** — the VIS-alone bar there is 0.0346, and no VIS-veto-only rule can reach it, because the fix that's actually needed is an **IR veto** (removing IR from the merge on that cell), not a better VIS veto — flagged at the time as the highest-value unmeasured open item.

### 6.4 (d) 2026-09-01 — the "crossmodal" rewrite (`docs/crossmodal-gate-2026-09-01.md`)
This is the biggest single redesign. New **"crossmodal" preset**:
- **Night-arm veto asks IR "is it night?"**, not VIS brightness — because VIS brightness alone cannot distinguish "dark world" from "dark sensor": synthetic lowlight/day has `p05=0`, literally *darker* than real night (`p05=2.5–3.5`), yet VIS still works fine on lowlight/day. Brightness alone is the wrong axis for this decision.
- **`grad_gini`** (a scale-free Gini coefficient of gradient magnitude) introduced as a fog/veil detector that needs **no temporal filter at all** (100% correct on fog, 0% false-positive on everything else — a clean single-frame statistic, unlike brightness/lap_var which both needed asymmetric temporal filtering).
- The Mahalanobis term is re-identified as the **actual mechanism** behind the lowlight/day loss (previously believed "inert") — `w_vis` collapses to 0.320 there once the veto is turned off, correctly handing the frame to the weaker sensor via soft weighting rather than a hard cut.
- **Result: every one of the 8 benchmark cells lands at or above `max(VIS,IR)`** for the first time — worst-cell gap **+0.0000** (the prior adopted system's worst cell was −0.0180). Night cells become **exactly** `ir_only` (bit-identical, via a new `single_passthrough` code path for the single-surviving-stream case).
- **§3a hardening** (safety, not accuracy): an IR self-check (`lap_over_var`, stable across clean day/night) plus a second vote (does VIS also agree it's dark) — this took the false-night rate on 19 IR-corruption arms from 94.8% down to **0%**, at zero measured benchmark cost (all 8 cells bit-identical before/after).
- **§3b hardening**: a multivariate (11-statistic) Mahalanobis health score catches IR glare at 64–75% detection vs a single-axis check's 12–34%; combined with the authority bound (set at the p99 of clean statistics), the both-degraded worst-case false-veto rate falls **24% → 8.7% → 1.3%**.
- An abstain mechanism (R_sys-based) was implemented and measured, then demoted to an advisory flag only — it prevented 0 bad vetoes while losing 2,095 correct ones in a direct test (`runs/eval/both_degraded.md`).

### 6.5 (e) 2026-09-01 — detector swap re-prices everything (`docs/levers-and-the-26m-swap-2026-09-01.md`)
Swapping the deployed detector from `yolo26s` to `yolo26m` (with p2feat IR) **breaks the old veil veto**, which becomes a **−0.0632 regression** on the new detector (VIS's own fog performance jumped from 0.0020 to 0.0824 — a 41× improvement that removes the justification for vetoing VIS on fog at all). The project's own framing: **"a veto encodes a claim about the detector, not about the image."** Fix: reorder the boolean logic from `veil OR (night AND dark)` to **`night AND (dark OR veil)`** — this alone takes the worst cell from −0.0632 back to +0.0000.
- Cross-modal box agreement measured directly and precisely for the first time: only **0.05%** of VIS boxes have an IR partner at iou_thr 0.85, even after a per-frame registration refinement that raises raw agreement 80× (to 4.02%) — and even then, held-out AP still *falls* (−0.0025 to −0.0026), because merging drags a good VIS box toward a worse-localized IR one.
- **Cross-modal support** (a *score multiplier*, not a coordinate merge — confirm-but-don't-move) at IoU 0.30/γ=0.5 is the one thing from this whole family of ideas that survives held-out testing: TEST delta **+0.0033**.
- **First genuine tune/TEST split introduced** here: TUNE = pohang00 (836 frames), TEST = pohang02+03 (364 frames). This immediately caught a real overfitting trap: `support_iou=0.55` *wins* on TUNE but is *negative* on all six held-out variants tried; `support_iou=0.30` is positive on all six. "Under the old single-set discipline both would have read as wins, and the wrong one would have been adopted."
- Final 10-cell grid (`final_26m_grid_v2`) at this stage: worst-cell gap day −0.0004, night −0.0081 — improved but not yet perfect.
- Lift screen introduced (used repeatedly downstream, see Part 8): confidence-above-median lifts detections 4.80×, sigma-below-median 3.00× (3.39× at IoU 0.75), cross-modal IoU-0.30 agreement 2.08×, and **temporal support is a null (1.00× lift)** — "persistent false positives are the most stable things in a fixed scene... the value of a redundancy axis is its independence, not its abundance." (This finding is cross-referenced by the `project-redundancy-independence` memory entry.)

### 6.6 (f) 2026-09-10 — R-D1, the mechanism ablation that names the final headline finding
See Part 7 in full — this is the null result that the project's abstract now leads with.

### 6.7 Final open items as of the crossmodal-gate document
Corrupt-IR-detector re-runs still needed at the time of writing (later covered by the IR-night-robustness ladder, Part 8); the last 1.3% both-degraded residual is a missed-detection problem, not a switch-logic problem; `sigma_weighted` remains off in the headline configuration; the learned gate (§7.5) never made it into the headline table; no benchmark cell directly prices the capability ratio itself.

---

## PART 7 — THE R-D1 NULL RESULT (UQ MECHANISM ABLATION) — HEADLINE FINDING

### 7.1 The measured baseline that motivated the test
Under the shipped `crossmodal`/`crossmodal26m` preset: `mu_d=1e9` and `lam=0.0` for **both** modalities → the Mahalanobis soft weight is mathematically inert; `sigma_weighted=False` → box σ never moves fused coordinates; `sigma_score_alpha=0.0` → box σ never moves the fused score. `r_frame_vis`/`r_frame_ir` are exactly 1.0000 on every frame and condition. **`w_vis` is a single constant, `0.9930`, across all 2,232 frames × 4 corruption conditions** (one distinct value in the entire dataset). Contrast: an older `preset="adopted"` had genuinely live `mu_d`≈71–72, `lam`≈0.7–1.0, and a `w_vis` that actually varied (0.0031–0.5041 under fog).

### 7.2 Pre-registration (`docs/prereg-uq-mechanism-ablation.md`, 2026-09-10, commit `a8f087c`, Amendment 1 `f713961`)
**The real question:** does *real* predicted uncertainty beat *shuffled* uncertainty (same numeric values, attached to the wrong boxes) — not "does uncertainty beat a constant," since inverse-variance weighting with equal σ reduces analytically to plain WBF and would be a nearly vacuous control.
**Eight arms (S0–S7):** S0 shipped; S1 real/coordinate-path; S2 constant/coordinate; S3 shuffled-within-frame/coordinate; S4 shuffled-across-cache/coordinate; S5 real/score-path; S6 constant/score; S7 shuffled-within-frame/score. α*=1.0 fixed (not tuned).
**Decision rule (Amendment 1 correction):** the magnitude floor was corrected from 0.0031 to **0.0060** (0.0031 × 1.95, applying the R-A3 finding that iid bootstrap intervals on this 10Hz video are ~1.95× too narrow — see Part 9). Block bootstrap L=20, n_boot=1000. **Needs ≥3 of 4 conditions to pass for a POSITIVE verdict.**

### 7.3 Result (`docs/uq-mechanism-2026-09-10.md`; full numbers in `runs/eval/uq_mechanism_ablation_v2.md`)
**Verdict: NULL on both primary comparisons.**

Coordinate path (S1−S3, real vs shuffled):
| condition | delta | 95% CI | excludes zero? |
|---|---:|---|---|
| clean | **−0.000566** | [−0.000893, −0.000041] | yes (negative) |
| fog | +0.000000 | [0, 0] | no |
| lowlight | −0.000001 | [−0.000004, +0.000008] | no |
| glare | +0.000406 | [−0.000123, +0.000764] | no |
**0 of 4 conditions pass at ANY tested floor** (0.0014 / 0.0031 / 0.0060 / 0.0100).

Score path (S5−S7): fog +0.003960 and lowlight +0.001995 both exclude zero, but clean −0.004420 and glare +0.000514 span zero — **also 0 of 4 at the 0.0060 floor** (2/4 clear the older, now-superseded 0.0014 floor, but not the corrected one).
S1−S0 (real vs shipped, no sigma at all): clean −0.000089, fog +0.000000, lowlight −0.000000, glare −0.000073 — "turning on the mechanism this project is named for changes essentially nothing."

**Why the fog/lowlight "positive" score-path numbers are illusory, not evidence of fusion working:** VIS is vetoed on **100.0%** of fog frames — there is no fusion happening on those frames at all. The coordinate path can only act where two streams actually cluster (never on fog); the score path acts even on single-stream frames, so what it's measuring there is **within-stream re-ranking**, not cross-modal fusion. The project's own summary: "**where uncertainty could influence the fusion it does nothing; where it shows a signal it is not fusing anything.**"

Two results reported "pointing the wrong way," per the project's transparency norm (reported rather than discarded): S1−S3 clean = −0.000566 (shuffled beats real, statistically but not practically — 10× below the floor); S5−S6 clean = **−0.013157** [−0.022630, −0.000678] (real-sigma score re-ranking is *worse* than a constant, costing 0.0167 absolute AP).

**Self-identified flaw in the rule, disclosed rather than hidden:** the "3 of 4 conditions" bar should really have been "3 of 3 *informative* conditions," since fog is structurally incapable of ever showing a coordinate-path effect (VIS never survives to be merged there). Recorded as a bias *against* finding a positive effect — i.e., it does not invalidate the NULL, it if anything makes the NULL conservative.

### 7.4 What follows from R-D1
The thesis claim narrows explicitly to **"image-statistic sensor selection"** — the mechanism that is measured and does work — with the Gaussian σ² head now positioned as a *localization-error-scale estimate*, evaluated on its own calibration terms (Table 2), rather than as a fusion input. No fourth arm/run was authorized past this. A separate future registration (not run within the reviewed material) is flagged for within-stream score re-ranking via the score path specifically, since that is the one place a real, if small, signal appeared.

---

## PART 8 — FULL EXPERIMENT CATALOG (`runs/eval/*.md`, `docs/eval/*.md`) — DENSE REFERENCE

This part indexes the ~150 individual experiment reports in `runs/eval/` plus the 11 dated reports in `docs/eval/`, grouped by topic, each with its headline number(s) and verdict. Use this as a lookup table when writing the Results section — go to the named file for the full table if a figure needs a citation-grade source.

### 8.1 Headline fusion tables (Table 2 / Table 3 lineage)
- **`table2_gaussian.md`** — UQ calibration snapshot, 2,232 paired frames, no fusion: VIS σ-head d_ece 0.0663, NLL 3.34, AUSE 0.088, mAP50-95 0.2580; IR σ-head d_ece 0.0344, NLL 3.29, AUSE 0.056, mAP50-95 0.0676.
- **`table3_clean_only.md`** (earliest, no capability weighting) — naive fusion 0.2515, gated fusion 0.2448 (*worse* than naive at this stage), learned gate 0.2593 (best); gate favors IR on 26.1% of frames.
- **`table3_fixed.md`** (capability weights added, VIS 0.2580/IR 0.0206, all 4 conditions, no day/night split) — clean gated 0.2639 (beats naive 0.2515); glare gated 0.2114.
- **`table3_fusion.md`** (adds the day/night split — the critical correction) — clean/day gated 0.3341 vs clean/night 0.0813; **VIS scores exactly 0.0000 at night everywhere**; pooling day+night flagged as actively misleading. Adds photometric gate (`p05`, `mu_b=10.5`) + hard veto `r_bright<0.5`.
- **`table3_identity_h.md`** (sensitivity: IDENTITY homography, no real registration) — clean fusion collapses 0.2515→0.2207; fog/lowlight/glare naive fusion falls to near-zero (IR AP itself collapses to 0.0007) — confirms registration quality matters a great deal.
- **`table3_refit_capability.md` / `table3_refit_tau.md`** — intermediate refit checkpoints, numerically identical to `table3_fixed.md`, superseded by `table3_fusion.md`.
- **`final_system*.md` family** (11 filenames: `final_system.md` plus dated/scoped variants `2026-08-20_macro-only`, `2026-09-01_photometric-only`, `adopted_regress`/`regress2`, `crossmodal`/`crossmodal_hardened`/`crossmodal_irnms`/`crossmodal_rsys`/`crossmodal_v2`, `perclass_smoke`, `preflight`, `veil`, `veil_INERT-BUG`) — successive regression snapshots of the same 8-cell ship-AP table as the system evolved from "adopted" (photometric-OR-veil) to "crossmodal" (gini+ir_night, single_passthrough). Consistent findings across all snapshots: crossmodal preset settles at clean/day gated ≈0.371–0.374 (vs VIS-alone ≈0.368) and glare/day ≈0.296–0.298 (vs VIS-alone ≈0.289), both clear of zero; night cells always exactly `ir_only` (CI spans zero by construction); `no_veto` ablation always costs ~−0.002 to −0.017 on night/fog/glare; `with_maha` (re-adding Mahalanobis weighting) costs −0.02 to −0.03 specifically on lowlight/day, reconfirming that Mahalanobis weighting can actively hurt in some conditions even though R-D1 shows it's inert in the shipped preset. `final_system_veil_INERT-BUG.md` is explicitly flagged invalid/superseded by its own filename.
- **`docs/eval/final_system_2026-09-01.md`** — dated snapshot: capability prior VIS 0.3352/IR 0.0092; gated-vs-ir_only ship AP clean/day +0.3537, glare/day +0.2751 (real gains); all night cells span zero (~+0.0003, VIS 100% vetoed); gated-vs-visible_only day: clean +0.0031, fog +0.0106, **lowlight −0.0180** (the persistent worst cell), glare +0.0036.

### 8.2 Registration / cross-modal alignment attempts
- **`align_26m.md`** — per-frame homography correction via median-offset translation: best is score-only "support i0.3 g0.5," TEST +0.0033 [+0.0010,+0.0067]; geometric alignment underperforms support-only.
- **`align_nomerge_26m.md`** — separates "create agreement" from "let it merge": best TUNE arm (align+iou0.95+support) gives +0.0121 on TUNE but TEST delta **−0.0010, spans zero** — alignment's tune-set gains don't generalize.
- **`cross_modal_iou.md`** — at adopted iou_thr=0.85, only **0.11%** of VIS boxes have an IR partner (day); at 0.55, 12.5%. Fusion is near-total concatenation at the shipped operating point.
- **`x_registration_drift.md`** — median |d| 3–6px per run, within-run drift up to 10px swing (pohang00); comparable to the −4.18px between-run swing that already killed a global-correction attempt.

### 8.3 Brightness/photometric gate tuning sweeps
- **`brightness_gate.md`** — baseline `mean`-statistic gate (mu_b=58.238, tau_b=17.869): real night w_vis collapses 0.792→0.228 when fitted only on synthetic day corruptions; clean pooled AP 0.2695→0.2610 (over-suppresses VIS on real night).
- **`brightness_gate_p05.md`** — `p05` statistic (mu_b=25.373, tau_b=2.520): harsher — night w_vis 0.923→0.495, lowlight collapses to w_vis≈0.001.
- **`brightness_gate_p05_margin.md`** — gentler p05 (mu_b=10.500, tau_b=2.625): night w_vis 0.926→0.432, glare/night improves 0.707→0.788. This is the version that became the adopted margin rule.
- **`brightness_gate_p05_s23.md`** — intermediate seed/margin (mu_b=17.341, tau_b=5.581): middle-ground result.
- **`gate_ablation_fog.md`** — under fog, `mean(w_vis)=0.000` in every combination-rule variant tested (multiplicative/min/geometric × α) — the combination rule is moot once VIS is fully distrusted.
- **`gate_ablation_glare.md`** — under glare (partial trust, w_vis 0.017–0.040): geometric-mean combination + α=0.5 best (0.086/0.278 mAP50-95/50); multiplicative α=1.0 worst (0.043/0.142).
- **`gate_lab_rules.md` / `_scalefree.md` / `_twosided.md` / `_variants.md`** — large exhaustive veto-rule sweeps (dozens of variants: photometric-only, veil-only, evidence-based, dilation 15/31/61, two-sided VIS+IR vetoes). Adopted rule sits at worst-cell gap −0.0180 (lowlight/day). Two variants ("gini-veil only [cap_only]" and "no veto [cap_only]") reach worst-cell −0.0054 but **only by also removing the Mahalanobis soft weight** — i.e., the improvement there comes from turning off Mahalanobis, not from the veto rule. Two-sided vetoes (vetoing IR when VIS is kept) crater on IR-corrupted night cells (worst gap −0.0810) — vetoing both sensors is unsafe.

### 8.4 Crossmodal parameter re-tuning
- **`crossmodal_tuning_cap_ratio.md`/`crossmodal_tuning2_cap_ratio.md`** — best x256/x64 gives +0.0025 over adopted on fit-clean; flagged as "not a win" at this scale.
- **`crossmodal_tuning_iou_thr.md`** — 0.85 remains best; 0.95 loses −0.0011; 0.55–0.75 loses substantially (−0.0024 to −0.0138).
- **`crossmodal_tuning_sigma_weighted.md`** — statistically identical to adopted (Δ≈0), confirming inertness independently of R-D1.
- **`crossmodal_tuning_vis_scale.md`** — matches adopted on fit-clean but loses up to −0.0191 on corrupted night cells — rejected.
- **`crossmodal_tuning2_ir_nms.md`** — best "ir_nms 0.70 + cap_ratio x16" +0.0024 vs adopted, marginal.
- **`levers_26m.md`** — comprehensive lever sweep on tune/TEST: only **`support i0.3 g1`** passes "wins tune AND no cell loses" (TUNE +0.0084, TEST +0.0031, 9/0 cells better/worse) — flagged **adopt candidate**. Reveals the fog/clean cell has a baked-in −0.0632 gap vs bar in "adopted" itself (the pre-detector-swap regression, see §6.5).
- **`merge_support_split.md`** (I9) — separately varies WBF merge iou_thr vs support iou/gamma: best combo (0.85/0.30/1) ≈ shipped (0.85/0.30/0.5). No merge-iou setting meaningfully beats 0.95 (effectively fully disabled) — conclusion: the architecture should be documented as `merge_iou='off'` explicitly, not implied by 0.85.
- **`final_26m_grid.md`/`_v2.md`** — isotonic score calibration ("calibrated") actively *hurts* (TUNE/TEST −0.0044/−0.0034) unless paired with support, where it partially recovers but still underperforms plain support.
- **`ir_dedup.md`** — best IR-stream dedup: `nms @ 0.70` (+0.0003 tune, +0.0019 night); WBF-based dedup variants lose at high IoU thresholds.

### 8.5 Extended robustness grids under corrupted IR
- **`extended_grid.md`/`_v2.md`** — v1 (26s) worst day cell gap −0.0015; v2 (26m) improves to worst-cell **+0.0000** across 10 day cells — the cap_ratio-scaled gate improvements hold under IR corruption, not just clean.
- **`extended_tuning.md`** — re-prices cap_ratio and veto_ir@authority against this grid's "no cell may lose" rule: all cap_ratio alternatives (x4/x16/x64) lose on ≥1 cell despite winning clean — rejected by that rule.
- **`cap_ir_gated.md`** (I7) — a gate-conditional cap_ir_scale never actually fires under these corruptions (IR-vetoed share 0.000 on every tested cell) — inconclusive by construction, needs a different conditioning signal.
- **`both_degraded.md`** — releasing the veto when both sensors are flagged prevents **0** bad vetoes and loses **2,095** correct ones over 76 both-flagged pairs — confirms R_sys abstain is a flag, not an actionable override.

### 8.6 IR night-robustness ladder — is the IR-based night switch safe under corrupted IR?
Rule under test: `ir_p05 > 41.5` (fitted once on clean IR, frozen). Six severities×hazards tested; "false night" = dangerous (vetoes VIS on a clear day):
| hazard | false-night rate | separable? |
|---|---|---|
| blur | 0% at all 3 severities | fully robust |
| **fog** | **43.8%→86.7%→94.8%** | **NOT separable at s2/s3 — worst finding** |
| **glare** | 19.2%→20.3%→26.8% | **NOT separable at any severity** |
| lowlight | 0% false-night but day/night indistinguishable (100% missed-night) | NOT separable |
| noise | 0%, missed-night creeps to 16.1% at s3 | still separable |
| rain | 0% at all severities | fully robust |
**Verdict:** the IR-night rule is safe under blur/noise/rain but **unsafe under fog and glare** — this is the empirical basis for the two-vote + multivariate-health + tighter-authority-bound fix (§6.4/§6.5), which took the both-degraded worst case from 24%→1.3%.
A related earlier probe, **`_ir_night_robustness_s8.md`** (underscore-prefixed but a genuine, not smoke, result): fog severity 2–3 and glare produce 30–96% false-night rate — the first surfacing of this fragility, later formalized into the full ladder above.

### 8.7 Night veto pre-registered decisions (V1/V2/V3) — see Part 6-adjacent detail already in Part 6; raw files are:
`night_restore_audit.md` (I5, pre-retrain audit), `night_restore_verdict.md` (ALIVE, +0.2520), `night_veto_verdict.md`/`night_veto_secondary.md` (V1, INCONCLUSIVE), `night_veto_v2_stage0.md` (instrument screening, q_refit AUROC 0.9921), `night_veto_v2_stage1.md` (V2, INCONCLUSIVE), `night_veto_v3.md` (V3, VOID, axis closed), `night_weak_fallback2_26m.md` (a related but separate fix: adding a VIS-independent confirmation roughly halves the worst night gap under the veil-repair mechanism, from −0.0296 to −0.0141, without touching day performance).

### 8.8 Veto rule sweeps and axis re-pricing (post-detector-swap)
- **`veto_rule.md`** — hard veto `r_bright<0.5` improves night modestly (0.0787→0.0813) without hurting day.
- **`veto_rule_maha.md`** — broader trigger `r_frame<0.5` fires far more aggressively (glare day veto rate 45–100% vs 0%), **regresses glare** (pooled 0.2004→0.1242) — confirms `r_bright` is the correct trigger.
- **`veto_rule_sweep.md`** — 29-rule sweep, lowlight/day is the binding constraint across nearly all variants (see §6.3).
- **`veto_axes_26m.md`** — re-prices every veto axis on yolo26m: **veil axis is mispriced on fog/clean** (−0.0632, deletes the better stream — the detector-swap regression); `ir_p05>thr night raw` mispriced (−0.377 on clean/fog_s2); photometric VIS `p05<mu_b` mispriced on lowlight (−0.024 to −0.028). IR-health axes (mute/merge) hold correctly everywhere they fire.
- **`veto_ir_decision_26m.md`** — should `veto_ir` (the IR merge bound) be removed? Genuine tension: keeping it wins the "is this stream worse" claim test (+0.32–0.37 on fires) but dropping it improves 4/9 ablation cells (summed day +0.0128, TEST +0.0047) with only 1 cell worse (CI spans zero).
- **`veil_night_exposure_26m.md`** — the veil-behind-night repair fixes fog/clean: bad-veto rate 100%→0% on 4/5 fog cells, worst day gap −0.0632→+0.0004.
- **`veil_veto_repair_26m.md`** — full before/after: adopted "veil AND night" fixes fog/day (0.0192→0.0908) while preserving night; stacking support+class_veto reaches fog/day 0.0908, glare/day 0.3144.
- **`reprice_constants.md`** — see §9.1 below (Part 9), same finding restated.
- **`x_veto_hysteresis.md`** — temporal smoothing (dilate k up to 61) targeting the residual fog/night gap (0.0789 vs ir_only 0.0810): dilate k≥15 nearly closes it (+0.0020, CI[+0.0007,+0.0028]) with zero effect on the guard cell — a viable small fix, not shown as adopted in the reviewed material.

### 8.9 IR upgrade / calibration / registration-adjacent probes
- **`x_ir_upgrade_p2feat.md`** — swapping IR to p2feat: `ir_only` improves everywhere (night AP 0.081→0.109, +34%), but gated fusion's night advantage over `ir_only` **flips negative** (−0.0010 to −0.0025) — a better IR detector *closes* the case for gating at night, while day cells (clean/glare) still strongly favor gated fusion (+0.32, +0.25).
- **`x_maha_dayonly_refit.md`** — falsification test: excluding night frames from the Mahalanobis reference (day-only refit) **fixes** the day/night score inversion (contaminated: D_night 28.6 < D_day 30.8 "night looks cleaner"; day-only refit: D_night 89.0 ≫ D_day 30.9, correctly separated) — confirms the OOD scorer's original night-blindness was a reference-set contamination artifact, correctable by refit.
- **`x_score_calibration.md`** — isotonic conf→P(TP) calibration per modality: VIS and IR raw-confidence-to-TP-probability scales differ by 2.8–6.5× at matched raw confidence (the two streams' scores were never directly comparable pre-calibration); net fusion effect small and mixed.
- **`x_capability_refit.md`** — refitting the capability prior run-disjoint (excluding pohang01) shifts the VIS/IR ratio from 12.5× to 36.25×; table-level effect small (+0.0016 to +0.0047) but non-trivially changes the quoted w_vis in the veto argument.
- **`x_dedup_iou_sweep.md`** — union-label GT dedup threshold sweep: only 8.7% of IR GT boxes are unambiguously novel objects; mAP under union-GT drops substantially (−0.08 to −0.12) at every threshold — a labeling-methodology caveat, not a fusion result.
- **`x_loro_mu_b.md`** — leave-one-fit-run-out sensitivity of `mu_b`: dropping pohang03 moves mu_b by +5.5 points — the 3-run min/max margin rule is fragile and would benefit from a percentile-based floor instead.
- **`x_risk_coverage_fixed_gt.md`** — corrects an AURC computation bug (shifting-denominator non-monotone on 4/4 conditions); even fixed, **R_sys does not beat random abstention ordering on 0 of 4 conditions** — confirms the abstain signal is null under the corrected metric too.
- **`x_topk_truncation.md`** — top-k truncation of per-modality detections has negligible effect (<0.0002) for k≥50; only k=25 shows small real losses.

### 8.10 Oracle headroom / re-ranking
- **`oracle_headroom.md`** (paired, 2,232 frames) — actual mAP50-95 0.3233 vs oracle re-ranking ceiling 0.4293 (**+0.1060 headroom**); buoy headroom (+0.1152) exceeds ship (+0.0968) despite far fewer GT boxes (600 vs 10,663) — macro-averaging over-weights buoy. Cross-modal union recall adds only +0.01–0.02 over VIS alone.
- **`oracle_headroom_day.md`** (VIS-only day, 9,284 frames) — actual 0.2771, oracle 0.4059 (**+0.1288**), consistent per-run headroom ~0.09–0.12.
- **`per_class_levers.md`** — buoy's oracle-only re-rank gain (+0.0576) beats ship's (+0.0484); re-ranking should be fit per-class, not pooled. `support_gamma` is flat on buoy (IR is nc=1, can't support buoy at all) — a ship-only lever mislabeled as global.
- **`rerank_loro.md`** (paired substrate, leave-one-run-out) — best OOF arm (4 features, monotone, λ=0.30): **+0.0037** held-out gain, much smaller than the in-sample fit gain (+0.0149) — restates the overfitting lesson; 8/18-feature arms actively hurt OOF.
- **`rerank_loro_day.md`** (larger 9,284-frame day-only substrate) — best OOF arm is **λ=0.00, i.e. no re-ranking at all** (delta 0.0000) — re-ranking doesn't generalize at this larger scale, a *stronger* null than the smaller substrate. (The corresponding coarse-grid **preflight** version of this experiment, `runs/queue_ideas/preflight/rerank_loro.md`, only tested λ∈{0,0.5} and concluded "nothing works" — this is the one documented case in the whole project where a preflight pilot's coarse grid *missed* a real signal that the finer full-scale sweep later found.)

### 8.11 Sigma-as-signal probes (feeding into R-D1)
- **`nll_floor.md`** — diagnoses a pure metric artifact: NLL blows up to ~1e13–1e17 when a 2-member cluster's σ=0 exactly; fixing the epsilon clip constant moves every MC/ensemble NLL by six orders of magnitude with zero weight changes.
- **`sigma_residual.md`/`_day.md`** — tests whether σ predicts the *signed* residual direction (not just magnitude) via leave-one-run-out ridge: only 2/4 box edges (y1,y2) show live OOF R² (0.017–0.098); applying the correction end-to-end **hurts** mAP substantially (−0.02 to −0.11) — σ is informative about error *magnitude* but using it as a direct coordinate correction backfires.
- **`probe_detector_evidence.md`**, **`probe_structure.md`/`_full.md`** — no single image-statistic (or combination) achieves the full night/fog/glare/lowlight separation target across all 8 benchmark cells — confirms no single evidence axis can replace the compound veto rule.
- **`tta_o2m.md`** — test-time augmentation view-agreement lift is ≈1.0× once confidence-matched (not genuinely informative beyond confidence); WBF-merging TTA views gives marginal, mostly CI-spanning-zero gains — "the cleanest test of whether merging helps at all" (views are pixel-exact registered), and it still barely moves — reinforces that merging as a family (cross-modal, within-modal, multi-view) is close to exhausted as a lever, leaving score-only support as the one surviving fusion mechanism.
- **`sigma_wbf.md`/`_iou055.md`** — sigma-weighted WBF vs stock: deltas ≤0.0005 everywhere — inert.
- **`sigma_score_26m.md`**/**`_perstream_26m.md`** — sigma-below-median gives 3.00× TP lift (second-strongest signal after confidence's 4.80×) but adding it to the fusion score barely moves the TEST metric (+0.0010, CI spans zero).
- **`signal_lift_26m.md`** — the underlying lift screen: conf-above-median 4.80×, sigma-below-median 3.00×, cross-modal IoU-0.30 2.08×, temporal support ~1.00× (dead) — establishes temporal support carries no independent signal.
- **`within_modality.md`** — WBF merge (dedup within one stream) always *hurts* mAP50-95 (−0.003 to −0.005) despite raising mAP50; soft-NMS σ=0.5 is the best single-stream dedup arm (+0.0025).

### 8.12 Soft-NMS adoption saga — a second small pre-registered rejection
- **`vis_soft_nms_adoption.md`** (σ=0.7) — every cell ≥ shipped, looked adoptable.
- **`vis_soft_nms_adoption_v2.md`** (σ=0.5, the actually-considered width) — worst cell **day −0.0004** on blur_s3/glare_s2 — fails the every-cell bar.
- **`snms_cell_redraw.md`** — redrawing the corruption seed on the failing cell across 6 draws: mean delta +0.0010, negative on 2/6 draws — the −0.0004 failure is draw noise.
- **`snms_gate_draw_avg.md`** — the actual draw-averaged pre-registered gate: day passes on every cell; **night worst cell −0.0000 on blur_s3/glare_s2, negative on 4 of 4 draws** — fails the pre-registered "every cell night ≥0" bar even after averaging, despite being 20× smaller than the AP-convention parity gap. **Verdict: DO NOT ADOPT.** `vis_soft_nms` stays off; `crossmodal26m` remains shipped.
- Underscore-prefixed smoke versions (`_smoke_snms.md`, `_snms_redraw_smoke.md`) reach the same qualitative conclusions on fewer draws, superseded by the full versions above.

### 8.13 Statistical-validity/noise-floor experiments (see also Part 9)
- **`interval_block_sensitivity.md`/`_v2`/`_v3`** — establishes frame-level iid bootstrap on 10Hz video underestimates SE; final/authoritative reading: inflation **at least 1.9×**, worst pair **1.99×**, measured up to block length L=20 (2s) before the shortest run (117 frames) makes longer blocks unreliable. Night (single run, pohang01) has no block length that makes a between-night-run interval estimable at all.
- **`metric_noise_floor.md`** — buoy carries **74–75% of macro-metric variance** on most cells despite being only 5.3% of day GT boxes; 2 of 11 cells carry ~zero buoy-variance signal; combined 2σ noise floor 0.0011–0.0120 depending on cell.
- **`delta_noise_floor.md`** — the corrected **paired**-delta noise floor (pairing buys 11–52× tighter SD than unpaired resampling): combined draw+bootstrap 2σ floor **0.0000–0.0031** across cells — this is the number quoted in memory as "real 2σ paired floor 0.0014–0.0031."
- **`change_impact.md`/`_v2`/`_v3`/`_v4`** — applying both corrections (AP-convention 0.00029, interval-widening 1.95×) to every previously-"significant" fusion decision: final tally **20 of 74** significant findings become **INDETERMINATE**; large effects (no_veto on night/fog/glare, veil-repair +0.0716) survive; the constants-reprice and soft-NMS-rejection decisions are explicitly NOT re-adjudicated by this table (they used sign-count rules, not confidence intervals, so the correction doesn't apply the same way).
- **`ap_convention_parity.md`** (R-A1) — local AP reads systematically low vs COCO convention; worst gap in an absolute number −0.00044 (day); cross-arm delta disagreement max **0.000285**, 5× below the 0.0014–0.0031 noise floor — convention choice cannot flip a sound decision, but absolute numbers must state their convention.
- **`ap_by_size.md`** — small objects carry 89.8% of GT mass but the lowest AP50-95 and headroom — resolution is the lever, not re-ranking.

### 8.14 Ensembling / cheap fixes
- **`cheap_fixes.md`** — six CPU-only tuning experiments confirmed on held-out seed-1 caches: iou_thr=0.85, α=0.5, skip_box_thr=0.00 all confirmed; union-label eval shows VIS-only-GT metric overstates VIS's real coverage (clean VIS-only AP 0.258 drops to 0.139 under union GT including IR-only boxes); R_sys-abstain coverage curve is genuinely informative (mAP rises as low-R_sys frames are dropped, e.g. 0.2710→0.3104 at 50% coverage) even though it doesn't beat random ordering in the corrected `x_risk_coverage_fixed_gt.md` test — the two findings coexist because "informative under coverage-dropping" and "beats random re-ordering" are different claims.
- **`checkpoint_ensemble.md`** (I8) — two-checkpoint VIS ensembling: agreement lift ≈1.00× once confidence-binned (mirrors the cross-modal "redundancy=independence" finding); all coordinate-merging ensembling arms underperform the single checkpoint by −0.002 to −0.095; only score-only "support" arms are neutral.

### 8.15 UQ day/night contamination slices (U1/U2 family)
- **`uq_day_night_slice.md`** (Stage A) — VIS **CLEAN**; IR **SUSPECT**.
- **`uq_day_night_slice_nightfull.md`** (amendment) — VIS flips to **CONTAMINATED** (d_ece r=1.03) but confounded by the MC-Dropout arm still being night-blind (28 vs 8,391 detections) — the "contamination" signal is arm-population mismatch, not real label-driven distortion.
- **`uq_day_night_slice_u2*.md`** (Stage B, all VIS arms finally night-capable) — **VIS SUSPECT, IR SUSPECT** — same band → night miscalibration is **not label-driven**; retraining would not fix it; day-only becomes the primary reporting basis, pooled retained as secondary with its 46.2%-night share disclosed in the caption.
- **`uq_mechanism_ablation.md`/`_v2.md`** — the R-D1 numbers themselves, see Part 7.
- **`stage1_crossing_2026-09-14.md`** — the S1-NULL numbers, see Part 6.1/6.8 and Part 4-adjacent Phase-3 section (Part 11).

### 8.16 Phase 1 / detector-selection adjacent
- **`phase1_day_night_slice.md`** — day AP uniformly ~0.05 higher than pooled; day and pooled rankings agree on top-3 (see §4.9).
- **`per_class_ap_ir.md`/`_vis.md`** — IR macro AP@50-95 0.068 (ship 0.135, buoy 0.0002 — IR essentially cannot see buoys); VIS macro AP@50-95 0.258 (ship 0.214, buoy 0.302).

### 8.17 docs/eval/ dated finalized reports (11 files) — headline numbers
- **`ap_convention_parity_2026-09-09.md`** — see §8.13 above (this is the source doc; `runs/eval/ap_convention_parity.md` is the underlying report).
- **`change_impact_2026-09-09.md`/`_v4.md`** — see §8.13 (53→74 findings reviewed across versions; 16→20 become INDETERMINATE).
- **`final_system_2026-09-01.md`** — see §8.1.
- **`g4_mde_2026-09-10.md`** — Minimum Detectable Effect, computed before Stage 2 spending: VIS between-seed σ=0.00638→5-seed MDE 0.01291 (does not resolve the 0.006 floor; would need 19 seeds); IR σ=0.00499→5-seed MDE 0.01010 (would need 12 seeds). **5 seeds stand anyway** — bought for stability, not for power on a retrained-vs-deployed comparison, which is explicitly cut and may never be reported under this pre-registration (BUDGET-CUT, Amendment 4).
- **`holdout_p04_devref_2026-09-14.md`** — the development reference the pohang04 look will be judged against: pohang02+03 pooled AP **0.2898** [0.2580,0.3175], pohang00 group AP 0.3955 [0.3352,0.4827] — spread between dev groups 0.1057 (larger than the pre-Phase-3 expectation of 0.033).
- **`interval_block_sensitivity_2026-09-09.md`** — the authoritative 1.9–1.99× inflation-factor measurement (see §8.13/Part 9).
- **`ir_equivalence_2026-09-10.md`** — the corrected IR-ladder statistics (see §4.8).
- **`uq_day_night_slice_u2_2026-09-09.md`/`_nanpolicy_2026-09-09.md`** — VIS SUSPECT, IR SUSPECT, not label-driven (see §8.15); the two files differ only in NaN-handling presentation for the NLL row, not in verdict.
- **`uq_mechanism_ablation_v2.md`** — R-D1 confirmatory pass (see Part 7); also confirms via permutation audit that the shuffle step genuinely moved rows (~52k/54k IR rows, ~18.6k/19.9k VIS rows moved in arms S3/S7).

### 8.18 Empty/uninformative or explicitly superseded files
`final_system_veil_INERT-BUG.md` (superseded by `final_system_veil.md`, filename self-flags an inert bug); most underscore-prefixed files in `runs/eval/` (`_ir_night_robustness_s8.md` is the one genuine exception — a real early result, not a smoke test) are smoke-scale precursors to a full-scale file of the same topic and are superseded by it once the full version exists.

---

## PART 9 — STATISTICAL METHODOLOGY: NOISE FLOORS, METRIC CONTRACTS, AP CONVENTIONS

This part matters for the manuscript's Methods/Statistics section — it is the project's answer to "how do you know a measured delta is real."

### 9.1 The chain of statistical self-correction (chronological, each motivated by the previous finding)
1. **2026-09-02 — magnitude-floor problem discovered.** The soft-NMS adoption gate initially had no absolute magnitude floor — a night regression of **−1.03e-5** failed the bar identically to a 1e-2 regression would have. The project explicitly refused to retroactively invent an equivalence margin to reverse this specific verdict ("a margin invented after seeing that it flips this verdict is a rationalisation with a formula attached") and instead measured a real noise floor.
2. **2026-09-02 — real (paired) noise floor measured** (`metric_noise_floor.md`, `delta_noise_floor.md`): the correct comparison uses **paired** deltas, not independent resampling — pairing buys an **11–52×** tighter SD. Combined draw+bootstrap 2σ floor: **0.0014–0.0031** for most cells (memory-recorded headline number), full range across all cells **0.0000–0.0031**. Buoy carries 74–75% of macro variance despite being 5.3% of GT mass; 2 of 11 cells carry essentially zero information.
3. **2026-09-02 — inherited constants re-priced against this real floor** (`docs/prereg-reprice-inherited-constants.md`): `cap_ir_scale`, `iou_thr`, and the veil-repair flag were all re-tested with a fresh, pre-registered margin (`2×hypot(sd_draw, sd_paired)`). **All three STAND**: `cap_ir_scale=4.0` is not dominated (nearest challenger x8.0 is a near-miss — wins 5 cells, loses 2); `iou_thr=0.85` stands (0.70 clearly worse, 0.95 unresolvable against 0.85, both effectively confirming merging does very little either way); **veil-repair-off costs −0.0416 on fog/clean**, 7.7× the margin — "not a close call." Self-critique: 8 of 39 verdicts in this very analysis rested on |delta|<1e-4 against a margin that also rounds to 0.0000 — the same flaw as #1 above, recounted at multiple floors and found robust.
4. **2026-09-09 — the external architecture review (F03/F04) forces a deeper statistical audit.** Two things questioned: is local AP even comparable to COCO AP, and are the bootstrap CIs themselves computed correctly given 10Hz video autocorrelation?
5. **R-A1 — AP convention parity** (`docs/ap-convention-rule-2026-09-10.md`): measured, not assumed. Delta-level disagreement between local and COCO convention: worst case **0.00028501**, 5× below the noise floor — **safe for deltas**. Absolute-number disagreement: −0.0033/−0.0050 in hand-built cases — **not safe for absolutes**, so every absolute AP number must state its convention. The Ultralytics-inter-version gap (~0.034) is **two orders of magnitude larger** and must never be placed beside custom-fusion AP in one table. Kept `local-linear-interp` as the working convention (322 call sites/70 files/12 docs already use it; switching to COCO wouldn't even be "real" COCO since `max_dets` differs from COCO's 100).
6. **R-A3 — dependence-aware intervals** (the single most consequential correction): plain iid bootstrap on 10Hz video is too narrow. Measured inflation factor: **1.9–1.99×** (varies by comparison pair and block length, ceiling at block length L=20/2s due to the shortest run, pohang03, having only 117 frames). **This 1.95× factor is applied project-wide** from this point forward — it is why the Phase 3 mechanism-ablation floor moved from 0.0031 to **0.0060**, and it is the single largest driver of the 20/74 "significant→indeterminate" reclassification in `change_impact_v4.md`.
7. **R-A4 — metric contracts** (`docs/metric-contracts-2026-09-10.md`): six specific claims about the metric implementations tested and found to reproduce correctly (D-ECE conditions on confidence only; AUSE/AURC are ranking-only, verified with a rank-*reversing* control that correctly moves AUSE 0.0630→0.3569; NLL/interval-ECE are TP-only, now published alongside `tp_share`/`recall_tp_over_gt` so comparisons carry their denominators; AURC is a grid-mean not a trapezoidal integral — gap to the true integral is 0.0215, 7–15× the noise floor, both quantities now published side by side without changing the headline `aurc` key). One genuine, disclosed-but-unfixed defect found: WBF's fused confidence scores can exceed 1.0 (max 1.753204, 0.0641% of detections) due to `k=min(n_models,n_cluster)` treating two same-stream overlapping boxes as "confirmation" — traced to WBF mechanics, NOT to cross-modal support (which is inert in both shipped presets) as an earlier reviewer guess assumed. Left as a disclosed counter (`conf_out_of_unit_range`), not repaired, because fixing it is a system change requiring a new pre-registration and re-score.

### 9.2 Exposure ledger and the "no real test set" finding (R-B2, `docs/exposure-ledger-2026-09-09.md`)
Central claim: **"This repository contains no untouched test set."** pohang00/01 are development runs held out of gate-*fitting* only, but scored on repeatedly; pohang02/03 were declared TEST_RUNS **after the fact**, already fail the model-selection-bias test (a candidate — support IoU 0.55 — was rejected specifically because it lost on this "test" set), and `sel("fit")==sel("day")` is literally `True` in code — every "fit-run day" constant was reported on the exact frames it was tuned on. This motivated `load_context(role=...)`: `role="develop"` (default, backward compatible) permits the historical fit-on-reported-frames behavior; `role="final"` structurally refuses any selector that spans scoring frames — this is the code-level mechanism protecting the pohang04 single look (Part 11).

### 9.3 Naming-is-not-provenance defect (F14, cascades through several docs)
The name `preset="crossmodal"` referred to **three different systems within a single day** (2026-09-01) — three different `cap_ir` values across 5 saved config files, with `ir_nms`/`cap_ir_scale` present in none of the recorded config blocks despite affecting results. All three reproduce exactly from their own saved configs (no drift), but the *name* alone cannot distinguish which ran. `crossmodal26m`/`crossmodal26m_snms` (introduced after this was found) are not ambiguous — they always inherit both repairs. The largest per-cell effect of the naming ambiguity was measured at +0.0017/+0.0019 — inside the noise floor, so no verdict in the record actually turns on it, but it is disclosed as a reproducibility gap. The same pattern (`data_yaml` column reading identical across two campaigns despite the underlying split having changed) recurs in the Phase 1 record (`docs/ranking-hygiene-2026-09-10.md`).

### 9.4 Holdout/contamination gates — see Part 3.9/3.10 for full detail; summarized statistically here
The Mahalanobis fit list (`maha_fit_vis.txt`) was 20.5% pohang04 — meaning the shipped OOD scorer's reference distribution genuinely included holdout data before the fix. `assert_holdout_excluded.py` (mechanical gate) initially had a silent path-resolution bug that made a "clean" filtered list load **zero** images while still reporting a PASS — recorded as recurring 3× in one session, with the lesson "a gate that verifies the wrong property is worse than no gate."

---

## PART 10 — POSITIONING VS PRIOR LITERATURE, RETIRED CLAIMS, BIBLIOGRAPHY CORRECTIONS

### 10.1 Two independent failures in the original novelty claim (`docs/positioning-2026-09-10.md`, R-F1)
**(a) The literature claim was false.** scope.md originally claimed "existing VIS-IR fusion is static... unlike most prior maritime fusion work, this system blends based on live per-frame uncertainty." This is contradicted by:
- **UA-CMDet** (Sun et al., 2022, github.com/SunYM2020/UA-CMDet) — drone-based RGB-IR cross-modality vehicle detection with uncertainty-aware learning **and illumination-aware NMS at inference** (genuinely per-frame adaptive).
- **DICTA 2024** ("Uncertainty-Aware Cross-Modality Fusion for Visible-Infrared Object Detection," doi 10.1109/DICTA63115.2024.00029).
- **Gaussian YOLOv3** (Choi et al., ICCV 2019) — already cited as R1, "the pattern" this project follows for the σ head; it cannot simultaneously be cited as prior art AND claimed as the project's own novelty.
- Logical point recorded: learned attention isn't "static" merely because its parameters are frozen at inference time — attention *values* still depend on the input.

**(b) The system claim was false, and this is the more serious failure because it's measured, not a literature-search miss.** R-D1 (Part 7) shows `w_vis` is a constant 0.9930, not "live per-frame uncertainty." The project's own note: "a reviewer who ran the code would find this in an afternoon."

**Replacement comparison axes** (domain, sensors, uncertainty target, inference-time adaptation, calibration evaluated, registration assumptions, compute) drawn up against UA-CMDet, DICTA 2024, Gaussian YOLOv3, and maritime-YOLO variants with no uncertainty at all (SID-YOLOv5, EG-YOLO, RDSC-YOLOv4, YOLOv7-sea). This project's row in that table: "hard veto on image statistics; fusion weight **constant 0.9930 (measured)**"; calibration column: "Yes — D-ECE, interval-ECE, NLL, AUSE/AURC, declared metric contracts, measured noise floor." Cells for the other papers' calibration protocols/registration assumptions/compute are deliberately left blank rather than filled from memory, pending an actual read of those papers.

**What is defensible, stated explicitly (3 things):** (1) a controlled maritime uncertainty study that publishes negative results, with a measured (not assumed) noise floor and declared metric contracts; (2) a lightweight interpretable sensor-selection baseline (`grad_gini` + `ir_p05`) that beats both single streams on all 8 cells, with its failure cases documented (the veil-veto −0.0632 regression; 100% VIS-veto on fog and all night); (3) an honest account of when uncertainty-driven fusion does *not* pay off (R-D1 itself).

### 10.2 Bibliography corrections (`docs/bibliography-2026-09-10.md`, R-F2)
1. RT-DETR (CVPR 2024) — already correct.
2. D-FINE — corrected from a wrong "arXiv 2024/2025" to **ICLR 2025**.
3. **Pohang and PoLaRIS were wrongly conflated as one release** — corrected: Pohang Canal Dataset (Chung et al., IJRR 2023, arXiv 2303.05555) is the *sensor* release; **PoLaRIS is a separate, later annotation release** (2024 preprint arXiv 2412.06192, **ICRA 2025**, github.com/sparolab/PoLaRIS). Author names for PoLaRIS deliberately left uncited pending a direct read.
4. MassMIND's 7 categories are *segmentation* classes, not this project's ship/buoy taxonomy — previously understated, now flagged with an explicit "needs a documented instance-to-class mapping before use" caveat, and no such mapping exists in the repo.
5. **MIT dataset annotation claim retired as false** (the most serious bibliography finding — see Part 3.13). Consistent with decision D10 deferring MIT to Phase 2+, but §5.1 of scope.md had never been updated to reflect that MIT was never actually touched. R21's status tracker corrected from "Primary" to "Deferred, not onboarded."
6. SMD (Singapore Maritime Dataset) — pairing assumption overstated (separate EO/IR material presented as though pre-paired); corrected to note it is **not** a fusion testbed as-is.

Two additional defects found by this project's own re-verification, not flagged by the external reviewer: (a) scope.md §5.1 self-contradicted on registration ("co-registered" vs "No spatial registration" eight lines later) — corrected to the measured 3–6px median residual, 50ms nearest-timestamp pairing; (b) local dataset counts didn't reproduce from the claimed numbers — corrected per Part 3.2.

### 10.3 Ranking hygiene / statistical rigor corrections (`docs/ranking-hygiene-2026-09-10.md`, R-F3)
See Part 4.8 and Part 9.1 item 5–6 for the substantive statistics. Additional Phase-1 acceptance-check finding: of five reproducibility criteria (unique rows, single metric scale, single training-software version, manifest resolvability, reproducibility without undocumented pilot data), only the first two **pass**; the other three **fail** (one row on ultralytics 8.4.7 vs 8.4.90 elsewhere; `data_yaml` manifest column doesn't actually pin the computation that ran; the pilot campaign's machine was wiped, making its 66 rows formally unreproducible even though they're kept in the record with the caveat attached).

### 10.4 External architecture review (`docs/architecture-review-2026-09-09.md`) — full finding list, F01–F20
Conducted against repo revision `1886258`, mostly P1 severity. Central verdicts, several of which seeded the entire late-project statistical-audit workstream: (F01) night-held-out data participates in fitting the cross-modal gate; (F02) the "development set" has effectively become the test set (→ exposure ledger, Part 9.2); (F03/F04) local AP ≠ COCO AP and the project's own bootstrap can disagree with itself under a class-drop policy (→ R-A1/R-A2); (F05) the night-label deletion caused much of the failure the veto was built to fix, and its later restoration invalidates the original "no signal at night" premise (→ the whole night-restore/veto saga, Parts 3.5 and 6/8.7); (F06) gradient-clipping couples the detector and σ-branch gradients even when σ's *loss contribution* is detached (→ R-C1, Part 5.5); (F07) MC/ensemble box disagreement is not the same estimand as the Gaussian head's predictive distribution (→ R-C2, Part 5.9); (F09) the shipped `crossmodal26m` preset disables the learned Mahalanobis/box-σ weights entirely (`mu_d=1e9, lam=0`), so it does not actually test the original UQ-gated-fusion claim (→ directly motivates R-D1, Part 7); (F14/F18) cache/label/queue identity checks were conventions, not enforced interfaces (→ R-E1, Part 12.4). A full decision-by-decision (D1–D31) assessment table accompanies the review; this spawned the `TODO-2026-09-09-architecture-review.md` repair backlog (Part 13), most of which closed by 2026-09-10.

---

## PART 11 — PHASE 3 RETRAIN: FULL PRE-REGISTRATION, AMENDMENTS 1–9, STAGE VERDICTS

### 11.1 Why the retrain was forced (three reasons, `docs/prereg-phase3-retrain-2026-09-10.md` §2)
(a) There was **no held-out data and never had been** — `FIT_RUNS` excluded only pohang01 (all night); pohang04 (26,188 VIS images/labels, 0 IR) had **never been used for anything** at all before this pre-registration. (b) The uncertainty mechanism is a measured null (R-D1), but two prior negative correspondence results (per-frame homography raising the partner rate from 0.05%→4.02% but *lowering* AP; `iou_thr` 0.95→0.85→0.55 costing −0.0138 at 0.55) were each measured with σ *inert* — the interaction of relaxed correspondence with a *live* σ-arbitration mechanism had never been crossed. That crossing is Stage 1. (c) The deployed configuration (IR nc=1, yolo26m-p2feat) had never actually been trained inside the architecture-selection ladder.

### 11.2 Stage 0 — gates, closed 2026-09-10
G1 (machine label agreement — later corrected for hash-scope error, Part 3.5), G2 (estimand choice: **disagreement ranking**, not predictive likelihood — a scope decision, not a measured one), G3 (parity contract: **drop strict bit-identity, declare a non-inferiority margin across matched seeds instead**, following F06/R-C1), G4 (compute MDE for every planned comparison *before* running it, cutting underpowered arms in advance — see Part 8.17/g4), G5 (pohang04 mechanically excluded from every split, see Part 3.9).

### 11.3 Stage 1 — correspondence × mechanism crossing, CLOSED with verdict S1-NULL
Design: a 2×2 crossing of `iou_thr∈{0.85 shipped, 0.55 relaxed}` × `sigma_weighted∈{off, live}` — cell A (shipped) and cell C (relaxed, sigma off) were already measured historically (C−A = **−0.0138**, a known cost); cell B (shipped, sigma on) was measured bit-identical to A; **cell D (relaxed + sigma live) was the actual experiment.** Decision rule: D must be non-inferior to A (|A−D| < 0.0060 with block-bootstrap CI) on ≥3 of 4 conditions, measured on TUNE (pohang00), reported on TEST (pohang02+03).

**Result** (`docs/stage1-s1-null-2026-09-14.md`, full numbers in `runs/eval/stage1_crossing_2026-09-14.md`):
| condition | A−D | 95% CI | non-inferior? |
|---|---:|---|---|
| clean | +0.0151 | [+0.0095,+0.0210] | no |
| fog | +0.0160 | [+0.0104,+0.0221] | no |
| lowlight | +0.0014 | [−0.0001,+0.0040] | **yes** |
| glare | +0.0115 | [+0.0070,+0.0157] | no |
**1 of 4 conditions non-inferior — verdict S1-NULL.** Robust across every tested floor (0.0014/0.0031/0.0060/0.0100 all give 0-1 of 4). The lone pass (lowlight) is not a rescue — it passes only because relaxing correspondence barely costs anything there in the first place (C−A=−0.0013), not because live σ recovered anything. `sigma_weighted` is confirmed to be live in the plumbing during this test (it changes the fused output on 753–836 of 836 clean frames) yet the B−A / D−C interaction terms are confined to [−0.0002,+0.0004] everywhere, every CI spanning zero. TEST-set numbers (reported, not acted on per the pre-registered rule) show 2/4 non-inferior with one condition (fog) actually favoring the *rejected* relaxed setting (+0.0028) — explicitly not used to override the TUNE-based verdict, exactly as pre-declared.

**Consequences, triggered exactly as pre-declared:** the mechanism test (Stage 3, adding an S8 "image-statistic selector, no predicted uncertainty at all" baseline) does **not** run as originally designed; the correspondence question is **closed permanently** — no threshold below 0.55 is to be tried, and no re-opening without an entirely new pre-registration; the fusion is now to be documented in scope.md as **union aggregation, not consensus**; Phase 3 narrows to Stage 2 (done) plus Stage 4 (the single pohang04 look, still pending as of the latest reviewed material).

### 11.4 Stage 2 — retrain, done
5 VIS seeds + 5 IR seeds, `runs/phase3_stage2/`, trained entirely on the laptop (Amendment 3), batch 12 (Part 5.12), architecture frozen (yolo26m nc=2 VIS / yolo26m-p2feat nc=1 IR — the deployed configuration, trained inside a proper multi-seed protocol for the first time). `p3_ir_seed3` diverged genuinely at epoch 6 (distinguished from a documented 2026-08-26 false-positive divergence by requiring two *consecutive* val-loss-rising epochs while train loss is still falling) and was correctly killed and rerun from epoch 0 (~4.5h), early-stopping at epoch 38 with best epoch 18, mAP50-95 0.13907. A bookkeeping bug initially mislabeled the healthy rerun as still-diverged (`run_queue.py --redo` was inheriting the dead run's alarm state — fixed); a related concurrency bug (two runners starting 13 seconds apart after a host restart) was found and fixed with a `runner.lock` file.

### 11.5 All nine amendments, in order (append-only, `docs/prereg-phase3-retrain-2026-09-10.md`)
1. **A1** — corrected the G1 hash-scope comparison error (tree vs train hash, Part 3.5); corrected the pohang04-list count from 5 to **15 lists/58,144 rows**; confirmed the pohang04 label archive was already-extracted VIS labels, not new data.
2. **A2** — pohang04 thermal imagery is already extracted (22,235 frames on disk), correcting an earlier belief it needed extraction; confirmed no thermal *labels* exist anywhere for pohang04 (the source `tir` bundle directory is absent entirely, not merely empty).
3. **A3** — moved **all** Phase 3 training to the laptop; dgxanode01 trains nothing in Phase 3 (server is only ~4.5% faster at steady state — retracting earlier claims of 1.23× or "7% slower," both measurement artifacts). G1 restated as G1′: a hostname assertion, not a cross-machine reconciliation.
4. **A4** — measured MDE (Part 8.17); confirmed the design contains **no** independently-trained-arm comparison that the 5-seed count could power anyway; formally cut and recorded as **BUDGET-CUT**: no "retrained vs deployed" comparison may ever be reported under this pre-registration (would need ~19 VIS seeds / ~12 IR seeds, not being spent).
5. **A5** — restricted the eventual single look to the sigma-head system *only*; the VIS ensemble/MC-Dropout arms were trained on a contaminated split (9,841 pohang04 frames) — no 3-arm UQ comparison may ever be reported on pohang04 as a result. IR ensemble/MC arms are clean by construction (no IR labels for pohang04 exist to be contaminated by).
6. **A6** — logged the Stage 2 seed-3 divergence/rerun and the two bookkeeping/concurrency bugs found while closing Stage 2 (Part 11.4).
7. **A7** — recorded that Stage 1 returned S1-NULL; Stage 3 as originally designed does not run.
8. **A8** — narrowed the single look's endpoint: **fused score only**, scored against the existing upstream **VIS** labels (hash `c06611a684f4`, 26,188 label files, 156,652 boxes, 286 empty) — **no thermal annotation is drawn for pohang04 at all**, since it has none and none will be created.
9. **A9** — resolved four remaining open items: (A9.1) seed aggregation — every seed scored, headline = mean over 5 seeds, VIS seed *k* paired with IR seed *k*, no re-tuning of any constant after seeing results; (A9.2) all 11 benchmark cells scored but **only clean/clean carries the verdict**, the other 10 are descriptive only, fresh corruption-draw seeds 941–944 (VIS)/951–954 (IR), budget ~16 GPU-hours for 190 caches; (A9.3) **HOLDOUT-GAP** is declared iff `AP_ref − AP_p04 ≥ 0.0060` **and** the 95% CI is entirely above zero, where `AP_ref` is deliberately the **weaker** of the two development reference groups (pohang00 at 0.3955 vs pohang02+03 pooled at 0.2898 — the weaker one, 0.2898, is `AP_ref`), so the test asks "worse than run-to-run variation *already observed* in development," not "worse than the best development number"; (A9.4) day/night is classified by solar elevation from GPS timestamp (elevation > 0° = day), not by the IR-derived night flag.

### 11.6 Stage 4 — the pohang04 single look, status as of the latest reviewed material: PREPARED, NOT TAKEN
`docs/holdout-p04-freeze-2026-09-14.md`: `holdout_p04_freeze.py` inventories **316 files** (sha256+size) into a manifest — 10 checkpoints, 10 Mahalanobis reference caches, 10 development caches, **190 pohang04 caches**, 76 pohang04 frame-statistics files, 5 calibration/geometry files, 8 development-substrate files, 5 pohang04-substrate files, 2 committed-reference files — plus the pohang04 label hash `c06611a684f4`, the git HEAD, and package versions. A draft readiness run on 2026-09-14 17:15 found 259/316 files still missing (build in progress at that time). Readiness gate before scoring requires: all 190 caches + 76 stats present and hash-verified; pohang04 labels hash to `c06611a684f4`; the selftest reproduces Stage 1's clean/clean number exactly (**0.3894193201201913**, confirmed reproduced "EXACT" per progress.md); no missing files; no manifest already written.
`holdout_p04_look.py` (the script that performs the actual single scoring) refuses to run unless: HEAD is a clean `FREEZE` commit; every file hash in the manifest matches; labels and inputs verify; the development reference reproduces exactly; and no `LOOK_TAKEN.json` marker already exists — **the exclusive marker is created BEFORE scoring begins**, so a crash mid-look still counts as the look having been taken (irreversibility is enforced mechanically, not by discipline alone). Frame statistics for this stage must run under `.venv` specifically, because the GPU interpreter's float32 sums differ from `.venv`'s by up to 4.4e-7 relative to the values the freeze manifest was hashed against.

**As of `progress.md`'s last update (2026-09-14), the immediate next actions were, in order:** (1) let the 3 parallel cache-build shards and the frame-statistics pass finish; (2) run `holdout_p04_freeze.py --draft` to see what's still missing, then the real run once complete; (3) make the `FREEZE` commit exactly as specified in the freeze doc; (4) run `holdout_p04_look.py` **once, only when the author explicitly says so**, then log the exposure in the ledger and write the verdict record. **This step had not been taken as of the material reviewed for this compilation.**

---

## PART 12 — INFRASTRUCTURE, IDENTITY CHECKS, AND OPERATIONAL INCIDENTS

These are the kind of details that belong in a "Reproducibility" or "Implementation Details" section, or in an appendix documenting engineering rigor.

### 12.1 Machines
- **Laptop** — RTX 4080, 12,282 MiB VRAM (11.99 GiB usable), Windows/WDDM. Does all Phase 3 training (Amendment 3). GPU interpreter is system Python 3.13 (`gpu_python` in config.yaml, `resolve_gpu_python()` CUDA-probed before every launch — chain scripts must never inherit `sys.executable`, a rule created after Stage 4 once ran 4 epochs at a **19.8× penalty** on the CPU-torch `.venv` build). VRAM does not leak across epochs; workers=16 measurably worse than workers=8; WDDM pages silently on an over-large batch (17× slowdown with no error).
- **dgxanode01** — A100, MIG `3g.40gb` slice (**not** the full 80GB card), 60 SMs, driver 535.161.08/CUDA 12.5. Jupyter-only access, no SSH. Ran the Phase 1 grid; used for nothing in Phase 3 (only ~4.5% faster than the laptop at steady state — repeatedly mis-measured earlier at both 1.23× faster and 7% slower, both retracted). `/opt/venv_match` provides a matched torch build; no concurrent runs permitted on the shared MIG slice; `/dev/shm` caps `workers` at 2 there.

### 12.2 Queue/runner engineering incidents (all found and fixed, listed as a record of what breaks in long-running unattended GPU queues)
- **Pause ≠ process exit** (2026-09-04): pausing the queue blocks it in a polling loop rather than exiting the OS process; starting a second `run` process and calling `resume` woke both simultaneously — two live PIDs against one queue directory. Confirmed via `Get-CimInstance Win32_Process`; fixed by killing the stale PID.
- **Zombie CUDA contexts** (2026-09-06): processes with growing CPU time were silently holding orphaned CUDA contexts, causing repeated OOM despite `nvidia-smi` showing zero live processes — 31 runs died this way. Diagnosed via a `ps aux` anomaly (zombies shouldn't accumulate CPU time) and fixed with `kill -9`. A separate, completely independent stale queue (paused 3 days, unknown to anyone in the session at the time) was also discovered and killed in the same incident.
- **Dead runner leaves no trace** (2026-09-11): the dashboard's pause state left no evidence in `state.json` of a dead runner — only the OS process table and `nvidia-smi` are reliable liveness checks; this recurred as a lesson multiple times across the project.
- **False-positive divergence kill** (2026-08-26): `watch_divergence.py`'s loss-guard window filled exactly at the LR warmup peak (an off-by-one against the warmup boundary), killing a healthy run (`ir_bench_yolov8l_seed1`) on a false alarm. Fixed by excluding warmup rows from both the baseline and the test window; verified byte-identical replay across 20 historical runs changed exactly one verdict (this false positive) with no real divergence missed.
- **Genuine divergence, correctly caught** (2026-09-11, Stage 2 seed 3): distinguished from the above false-positive pattern by requiring two *consecutive* epochs of rising val loss while train loss is still falling — the kill was correct this time.
- **Shared temp-filename race** (R-E2, still open at time of writing): `write_json` used a shared deterministic temp filename with an in-place-overwrite fallback — a concurrency risk under two runners, cited as the likely cost driver behind the 2026-09-06 incident.
- **Excel corruption trap**: opening the Phase-1 results CSV in Excel truncated floats to 9 significant figures and mangled a version-string column into a date serial — never open results CSVs in a spreadsheet application; use pandas/a text editor.

### 12.3 Batch and throughput measurements — see Part 5.12 for the full account (RTX 4080 ceilings, the probe-vs-production sign flip, and the decision to keep batch 12 anyway for pre-registration consistency).

### 12.4 Cache/recipe/label identity checks (R-E1, `docs/cache-identity-2026-09-10.md` / `docs/recipe-identity-2026-09-10.md`)
Four defects found and closed, each demonstrating that a *name* or *count* match is not the same as content identity:
- **D1** — `load_cache` didn't validate payload shape/n_frames/image_path/detection-array consistency; now does.
- **D2** — pairing was checked only by *length*, not by content. Measured cost of getting this wrong: a **1-frame shift** moves gated fusion by **−0.000968** — below the 0.0014–0.0031 noise floor, i.e. genuinely undetectable by any statistical test and catchable *only* by an identity check. A reversed IR cache moves it by −0.025630; a 100-frame shift by −0.027837.
- **D3** — `split_fingerprint` is **label-blind** (hashes filenames only): deleting a box, changing a class ID, or removing an entire label file all leave the fingerprint unchanged (reproduced directly). A new `label_fingerprint_trainval` column was added (measured values: VIS `ae7fa57efb2b`, IR `5fd58f37c799`).
- **D4** — the "completed run" lookup ignored epochs/imgsz/batch/weights entirely — the same CSV row would satisfy a re-request at epochs=25, 50, *or* 100. A new `recipe_fingerprint` (10 knobs including a content hash of the weights file, since `best.pt`'s *path* is overwritten every run; `workers`/`device` deliberately excluded so cross-machine resume still works) was added.
- A third, independent defect found in passing: three separate implementations of the images↔labels path-swap existed in the codebase (two were byte-identical copies) — collapsed into one (`src/uqfusion/data/labels.py`); 0 disagreements found across all 133,140 real image paths on disk (a latent, not live, bug).
- Cost of the new hashing: 15–24s warm / 383s cold per grid launch, to hash 107,627 VIS label files. Old CSV rows become read-only by convention (deliberate) once these identity checks exist.

### 12.5 File/script pointers for a Methods section
`src/uqfusion/uq/gaussian.py` (σ head), `src/uqfusion/uq/fusion.py` (WBF + veto), `src/uqfusion/eval/hysteresis.py` (veto temporal filters), `src/uqfusion/eval/ctx.py` (preset/context assembly, `FusionContext.inputs` auto-capture), `src/uqfusion/eval/apmetrics.py` + `matching.py` (AP/bootstrap), `src/uqfusion/eval/cocoparity.py` (COCO-parity harness), `src/uqfusion/eval/identity.py` + `cache.py` (R-E1 identity checks), `src/uqfusion/data/labels.py` (unified label-path resolution), `src/uqfusion/bench/grid.py` (Phase 1 grid driver, `split_fingerprint`), `scripts/eval_final_system.py`, `scripts/fit_structure_gate.py`, `scripts/probe_signal_lift.py`, `scripts/holdout_p04_freeze.py` / `holdout_p04_look.py` (Stage 4 gates), `scripts/verify_dataset_claims.py` (reports, doesn't gate), `scripts/label_hash_ledger.py` (append-only provenance timeline), `scripts/assert_holdout_excluded.py` (mechanical contamination gate, `--yaml` and `--caches` modes).

### 12.6 Archive/backup facts
- `archive/phase1/` — two campaigns' raw run trees (`pilot_2026-07-31`, fingerprint `f0220e716277`, unrecoverable split; `main_2026-08-10`, fingerprint `682dbe9f0f05`) merged into `phase1_benchmark/` on 2026-08-17. Original tarballs (2.6GB + 5.7GB) deleted only after 100%-file-level verification (696/696 and 2,281/2,281 files matched; 60 CRC32 + 80 SHA-256 spot checks byte-identical) — directory went 8.5GB→~108MB.
- `archive/server_2026-08-25/` — a live mid-training backup pulled from dgxanode01 (`ens_vis_seed1` at epoch 51/100 when captured), 1.16 GiB, sha256-verified, 18 checkpoints; a tar exit-code-1 was confirmed to be a benign "file changed while reading" warning (gzip -t clean, file-list diff empty), not truncation — though `ens_vis_seed1`'s own checkpoint may be torn mid-write as a direct consequence of capturing it live.

---

## PART 13 — OPEN TODOS, DECISIONS PENDING, AND LOOSE ENDS (latest snapshot: `TODO-2026-09-09-architecture-review.md`, cross-checked against `progress.md`'s open-questions table)

### 13.1 Workstream status as of the latest reviewed material (2026-09-10)
- **A (evaluator/statistics)** — R-A1 through R-A5 all **DONE** (see Part 9.1). R-A5's first pass reclassified 16/53 findings as INDETERMINATE; the `_v4` update later found 20/74 — both counts appear across different digest passes, the larger number reflecting a wider net cast as more decision families were folded in.
- **B (data/split integrity)** — R-B2 (dev-set-is-test-set) label+guard **DONE** (`role="final"`), but a genuinely **open** gap remains: nested leave-one-run-out validation cannot be run at all, because no uncontaminated held-out data existed until pohang04 (and that's reserved for exactly one look). R-B1 (gate fit on night data) blocked on a partition decision. R-B3 (annotation-release documentation) open. R-B5 (strict GT loading) **DONE** — 0 missing labels found across 133,140 images, but a malformed-line silent-drop bug was found and fixed along the way.
- **C (training/UQ semantics)** — R-C1 (parity/gradient-clipping coupling, Part 5.5) **open**, blocked on decision **Q4** (strict bit-parity vs a pre-registered non-inferiority margin — Stage 0's G3 tentatively answered this for Phase 3 specifically by choosing non-inferiority, but the general-purpose decision for the rest of the codebase is still open). R-C2 (MC/ensemble estimand, Part 5.9) **open**, blocked on decision **Q3** (disagreement-ranking vs predictive-likelihood) and directly blocks any UQ-baseline comparison table. R-C3/C4 (naming/disclosure conventions) open.
- **D (fusion/geometry)** — R-D1 (mechanism ablation) **DONE**, NULL (Part 7) — this is the finding that narrows the whole thesis. R-D2/D3 (covariance transform under homography, signed-vs-unsigned homography convention) open, lower priority (P2). R-D4 (registration/correspondence) closed by Stage 1's S1-NULL (Part 11.3) as far as the specific relaxed-correspondence question goes, but the general registration-quality question (Part 3.12, the never-built time-varying homography) remains open. R-D5 (abstention contract, i.e. what R_sys-below-threshold should formally mean) open.
- **E (system identity)** — R-E1 cache/recipe/label identity, slices D1–D4 **DONE** (Part 12.4); resume-time validation (does a resumed run's recipe still match its original launch recipe) still open. R-E6 (a single authoritative architecture document, since the story is currently spread across `architecture-final-2026-08-20.md`, `crossmodal-gate-2026-09-01.md`, and `levers-and-the-26m-swap-2026-09-01.md`) open — **this compiled document is, in effect, a stopgap for R-E6** for paper-writing purposes, but the repo itself still lacks one canonical file.
- **F (claims/publication)** — R-F1 (novelty claim) **CLOSED** (Part 10.1). R-F2 (bibliography) **CLOSED** (Part 10.2), including the MIT-annotation-claim retraction. R-F3 (ranking hygiene) documentation **closed**, but the **confirmatory GPU re-run implied by the corrected statistics is still open** and would need its own pre-registration if ever run (i.e., nobody has actually gone back and run a properly-powered IR architecture comparison — the corrected conclusion is "we can't tell," not "we now know"). R-F4/R-F5 open (not detailed in the reviewed material beyond being listed).

### 13.2 Explicit open decisions for the project owner (Laksh), as listed in the TODO backlog
- **Q1** — data partition: accept that no genuine holdout ever existed pre-pohang04 (the current working assumption), or pursue new recording to build one properly.
- **Q2** — re-score scope: answered — re-adjudicate only the indeterminate cells surfaced by R-A5, not a full project re-score.
- **Q3** — UQ estimand for MC/ensemble (disagreement-ranking vs predictive-likelihood) — blocks R-C2 and any UQ-baseline comparison table.
- **Q4** — parity contract (strict bit-identity vs a declared non-inferiority margin) — Stage 0 answered this narrowly for the Phase 3 retrain (non-inferiority), general codebase policy still open.

### 13.3 Older, now-superseded-but-informative TODOs
- `TODO-improvements.md` (2026-08-19) — items 0.1–0.7 all closed; A1 (σ-weighted WBF) done-and-null; A2 (registration correction) was promoted then later shown not to hold under the detector swap (Part 6.5); A3 (a better frame-quality signal than brightness alone) was demoted, later effectively answered by `grad_gini` (Part 6.4).
- `TODO-2026-08-20-full-scale.md` — A-1 (IR nc=1 adoption rule) closed (Part 6, IR section); A-2 sign-offs (yolo26m backbone, M=5 ensemble, laptop-not-DGX host) all closed; batch sizes measured and pinned (IR batch10 at the time, VIS@640 batch16, VIS@1280 batch4 — since superseded by the later batch-12 unification, Part 5.12).
- `TODO-2026-08-26-phase1-classset.md` — decided in favor of a cold-restart-from-COCO route for a genuine second 2-class table (31 variants×3 seeds, stride4, ~270 GPU-h estimated); a warm-start route was rejected because it would give unequal warm starts across variants and would launder the pilot campaign's contamination into a new table.

### 13.4 `progress.md`'s own open-questions table (OQ-1 through OQ-14) — status snapshot
Resolved: OQ-1 (val set — resolved via re-split), OQ-2 (Pohang-only yamls — confirmed), OQ-3 (IR bit depth/normalization — documented), OQ-5 (calibration files — partially resolved, per-run intrinsics/extrinsics present), OQ-6 (dry-run timing — demoted, standard defaults used), OQ-7 (letterbox vs stretch — letterbox confirmed), OQ-14 (repo location — resolved, github.com/Laksh-saroha/uqfusion, public).
**Still open:** OQ-4 (MIT annotation count/QA — moot now that MIT is confirmed never onboarded), OQ-8 (MIT fusion camera pair — moot, same reason), OQ-9 (server data-placement path reconciliation — largely resolved in practice by Amendment 3 moving all Phase 3 training to the laptop, but not formally closed for any future server use), OQ-10 (§7.2 DFL-vs-explicit-σ resolution — never reached a headline Table 2 row, per Part 5.2), OQ-11 (`imgsz` for full-res runs — the full-resolution twin exists but a systematic imgsz sweep with per-class AP was never completed per the reviewed material), OQ-12 (full-res host and ensemble budget — the M=5 ensemble question was effectively narrowed by Phase 3's move to laptop-only single-machine training, but the original full-res/1280 question is separate and not shown as closed), **OQ-13 (the unexplained 2026-09-03 21:19 label rewrite — still genuinely unexplained)**.

### 13.5 Phase 4 — not started
Multi-seed final runs on both modalities, the both-degraded Table-3 row (partially measured already via `both_degraded.md`, Part 8.5, but not as a formal Phase-4 deliverable), MIT dataset onboarding (blocked indefinitely — no annotation pipeline exists and none is scheduled), DETR benchmark-row decision (deferred, D9), manuscript artifacts. Phase gate: the author's go-ahead after Phase 3 results — which, per Part 11.6, are not yet complete (the pohang04 look has not been taken).

---

## PART 14 — QUICK-REFERENCE NUMBER SHEET

For fast lookup while drafting — every number here is sourced to a specific file elsewhere in this document; do not requote without checking the fuller context above, since several early numbers were later corrected.

| Quantity | Value | Source / caveat |
|---|---|---|
| Total images (verified) | 158,319 (VIS 127,309 / IR 31,010) | Part 3.2, 2026-09-10 |
| Total boxes (verified) | 1,183,736 (VIS 962,960 / IR 220,776) | Part 3.2 |
| Paired VIS↔IR frames | 28,388 | Part 3.2 |
| Night-filter boxes dropped (original) | 132,688 | Part 3.5 |
| Night-filter boxes restored (net) | +94,553 | Part 3.5 |
| Night VIS mAP50-95, pre-restore | 0.0000 | Part 3.5 |
| Night VIS mAP50-95, post-restore | 0.2520 [0.2473,0.2567] | Part 3.5 |
| Phase 1 top-8 span | 0.0055 mAP50-95 | Part 4.4 — not separable |
| Backbone selected | yolo26m (0.3016±0.0050) | Part 4.5 |
| yolo26x (nominal leader) | 0.3049±0.0020 | Part 4.4 |
| Two-stream FPS, 26m vs 26x | 28.5 vs 15.3 (1.86×) | Part 4.5 |
| Ultralytics version AP gap | ~0.034 | Part 4.6, never compare across |
| VIS/IR cross-modal partner rate @ iou 0.85 | 0.05–0.14% | Part 6, various cells |
| VIS/IR partner rate @ iou 0.55 | ~11–12.5% | Part 6.5, 8.2 |
| Shipped fusion weight w_vis | constant 0.9930 | Part 7.1 — R-D1 |
| R-D1 verdict | NULL, 0/4 conditions at 0.0060 floor | Part 7.3 |
| Real 2σ paired noise floor | 0.0014–0.0031 (full range 0.0000–0.0031) | Part 9.1 |
| Bootstrap interval under-estimate factor | 1.9–1.99× | Part 9.1 |
| AP-convention (local vs COCO) delta gap | 0.00028501 (safe, below floor) | Part 9.1 |
| Soft-NMS night verdict | DO NOT ADOPT (fails 4/4 draws) | Part 8.12 |
| Stage 1 (S1-NULL) pass rate | 1/4 conditions non-inferior (need 3) | Part 11.3 |
| Development reference for pohang04 look | AP_ref = 0.2898 [0.2580,0.3175] (pohang02+03) | Part 11.5 (A9.3), Part 8.17 |
| HOLDOUT-GAP trigger | ΔAP ≥ 0.0060 AND CI > 0 | Part 11.5 |
| Both-degraded veto release test | prevented 0 bad vetoes, lost 2,095 correct | Part 8.5 |
| False-night rate under fog corruption | 43.8%→94.8% (s1→s3) | Part 8.6 |
| Crossmodal gate worst-cell gap (final) | +0.0000 (was −0.0180, was −0.0632 mid-repair) | Part 6.4/6.5 |
| Oracle re-ranking headroom (paired) | +0.1060 mAP50-95 | Part 8.10 |
| IR ladder MDE vs observed spread | MDE 0.02067 vs observed 0.01193 (underpowered) | Part 4.8 |

---

*End of compiled context. This document draws only from files already tracked or generated in this repository as of 2026-09-17; it introduces no new analysis or claims beyond what those files already state. Where a finding was later corrected (e.g. the g1 hash-scope error, the fitness off-by-one bug, the IR-ladder ANOVA misinterpretation), both the original and the corrected version are recorded because the project's own convention is to preserve the correction trail, not overwrite it.*
