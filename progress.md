# Progress — uqfusion

> Purpose: a cold read of this file (plus `scope.md` and `docs/plan-2026-07-07-kickoff.md`) must reconstruct where the project stands without re-deriving anything. Updated at every phase boundary and every non-trivial decision.

**Last updated:** 2026-07-07 (session 1; Phase 2 core built and smoke-verified)
**Current phase:** Phase 2 — proof of concept — **CPU-verifiable portion COMPLETE, both smoke gates GREEN.** The full O2→O3→O4 chain (σ² head + β-NLL/warm-up → Mahalanobis → reliability → gated WBF) is implemented and passes its synthetic-data gates on CPU (see Run log). Remaining for Phase 2 close-out (needs server/data): small-scale run on real Pohang frames, the §12.1 detection-parity check (A4-11's condition), and the real IR→VIS homography once calibration files arrive (OQ-5).
**Phase 1 — code COMPLETE** (commit `37bac1e`), **GPU portion deferred**: Laksh runs the server sequence (audit → stride → grid → FPS → Table 1) whenever he gets server time; epochs/batch set to standard values (D15). Phase 1 closes when Table 1 + selection memo exist.
**Phase 0 — COMPLETE** (commit `ffbe87f`): repo, pinned env (install-verified), portable config + loader, docs, env smoke script.

---

## Cold-start pointers

| File | Role |
|---|---|
| `scope.md` | Source of truth for the science (architecture, datasets, eval plan) |
| `docs/plan-2026-07-07-kickoff.md` | Approved (2026-07-07, unedited) architecture review + backbone selection plan; all decision rationale lives there |
| `progress.md` (this file) | Current state, decision log, open questions, run log |
| `config.yaml` | The one file edited when moving machines; all paths + hyperparameters |

## Working agreement (from kickoff, 2026-07-07)

- Collaboration mode: senior-engineer peer — push back with evidence, don't rubber-stamp.
- Plan → Laksh reviews/pushes back/approves → implement, **one phase at a time**; Laksh gives an **explicit go-ahead after each completed phase** before the next starts (restated 2026-07-07).
- **No GPU in the dev environment.** CPU smoke tests only here; all real training on Laksh's GPU Jupyter server.
- Portability: no absolute paths anywhere; machine specifics live in `config.yaml` only.
- Every trainable component ships a tiny-subset smoke path before any GPU time (scope §18-2 discipline, applied everywhere).
- Git commit at every phase boundary (scope §11.1: which commit produced which result).

## Decision log

| # | Date | Decision | One-line rationale | Source |
|---|---|---|---|---|
| D1 | 2026-07-07 | §7.2: Gaussian σ² branch **added alongside** DFL (option c); DFL-derived uncertainty = free same-model ablation row; full DFL replacement = fallback only. Final call after Phase 1 winner is known. **Amended per A4-11 (approved conditionally):** branch retained iff it (i) doesn't degrade detection (§12.1 check) and (ii) yields calibrated, useful σ² — else fall back to option (b). Interpretation on record: "drop σ² entirely" is not an option since the branch *is* O2, the core contribution — Laksh to flag if he meant something stricter. | Single-model ablation, preserves deterministic baseline and §4 attribution | plan B1; scope §7.2; answer A4-11 |
| D2 | 2026-07-07 | §7.3: evidential regression cited, **not benchmarked**; head/loss built as pluggable interface | Doesn't test Comparison 2's claim; second custom head ≈ weeks off critical path | plan B2; scope §7.3 |
| D3 | 2026-07-07 | §7.4: keep caveat + both-degraded Table 3 row; **add absolute `R_sys` abstain signal** | Failure is weight normalization discarding magnitude, not missing signals | plan B3; scope §7.4 |
| D4 | 2026-07-07 | §7.5: learned gate included as upper bound, Phase 3, tiny logistic/MLP on existing features | Cheap ceiling comparison; supervision target TBD in Phase 3 | plan B4; scope §7.5 |
| D5 | 2026-07-07 | §6.4 constants: every constant = pre-registered statistic of clean validation data; **all gate ablations on cached predictions** (CPU post-processing) | Reproducible, no hand-tuning, ablation table nearly free | plan B5; scope §6.4 |
| D6-rev | 2026-07-07 | **Revised per A2-4/A2-5, replacing D6's defaults:** Laksh's existing custom split is used (not the D6-default block-builder, not an official PoLaRIS split); pohang01 (night) **is allowed in training** (reverses D6's full hold-out). Leakage-freedom is not assumed — it is verified mechanically by `scripts/audit_split.py` (cross-split duplicates + within-run temporal buffer, ~10 s), which **must exit 0 before any training**. Consequences: Table 3's low-light column is primarily synthetic darkness; test split should include pohang01 blocks to preserve a real-night eval row (requested in dataset_requirement.md §3). | 10 Hz video → random splits leak near-duplicates into every table; audit replaces trust | answers A2-4/A2-5; plan B6-2, C5 |
| D7 | 2026-07-07 | Fusion coordinate frame: static IR→VIS homography from dataset calibration; fuse + evaluate in VIS frame | WBF needs a common image plane; scope §5.1's "no registration needed" is too strong | plan B6-9 — pending A2-8 |
| D8 | 2026-07-07 | Table 1 shortlist: YOLOv8s/v9s/v10s/11s/12s/26s, s-scale, seeds {0,1,2}, Pohang VIS + winner/runner-up IR confirm; selection rule pre-declared (within ~1 std of best mAP@50–95 → DFL > simplest fork > FPS). **Confirmed by A4-12** (no expansion to a full ×2-modality grid; IR rows can be backfilled later with a config flip if the manuscript wants them). | All Ultralytics-native = one config/API; rule prevents post-hoc choice | plan C2/C4/C5; scope §8, §9.1; answer A4-12 |
| D9 | 2026-07-07 | DETR row deferred to Phase 4 | Gates nothing; identical-config unfair to DETRs; that SOTA churning (RF-DETR/DEIMv2/RT-DETRv4) | plan C3; scope §8 |
| D10 | 2026-07-07 | MIT Marine Perception deferred to Phase 2+ parallel onboarding; hard requirement by Phase 4 | Annotation pipeline is a named risk (§13); keep it off the Phase 1 critical path | plan C4; scope §5.1, §13 |
| D11 | 2026-07-07 | Phases: added Phase 0 scaffold; benchmark harness built in Phase 1 (not §18 step 5). **Phase 1/2 decoupled (Laksh, same day):** "Just move on to phase 2. I will run the server test later" — Phase 2 CPU-side work proceeds now; Phase 1's GPU portion (audit → grid → Table 1) runs whenever Laksh gets server time. Backbone dependency handled as planned: the Gaussian head targets the shared v8/11 `Detect` codepath and ports to whichever DFL variant Table 1 selects (plan B1 conditionals cover a non-DFL winner). | Wall-clock win; head work is the §18 highest-risk item | plan D; scope §18; Laksh go-ahead 2026-07-07 |
| D12 | 2026-07-07 | Tracking default: TensorBoard (local); W&B = opt-in toggle in config | No account dependency; scope §11.2 allows either | plan C5; scope §11.2 |
| D13 | 2026-07-07 | Stack pinned (see `requirements.txt`): torch 2.12.1, ultralytics 8.4.90 (YOLO26 support), + UQ/eval libs; full lock frozen on server at Phase 1 start | Reproducibility per §11.1; versions verified against PyPI 2026-07-07 | scope §11 |
| D14 | 2026-07-07 | Table 3 evaluated **per-frame (α = 1.0, smoothing off)** as the headline protocol; temporal smoothing (α < 1) becomes a separate sequence-level ablation | Cleaner attribution — the headline fusion result shouldn't owe anything to temporal state; matches the scope's "instant-by-instant" claim | answer A4-13 (delegated); plan B5-4; scope §6.4 |
| D15 | 2026-07-07 | Compute: H100, no GPU-hour cap (A1-1) → full 6-variant × 3-seed grid at s-scale stands; stride-2 training subsampling approved (A2-6). **Update (Laksh, same day):** server tests deferred to later; epochs/batch set to standard values now instead of dry-run-pinned — `epochs: 100`, `patience: 20`, `batch: 32` — "we can tweak this later". Dry-run timing is now optional tuning, not a gate. | Budget constraint gone; standard defaults over premature optimization | answers A1-1, A2-6; Laksh follow-up 2026-07-07; plan C5 |
| D16 | 2026-07-07 | `dataset_requirement.md` added as the binding data contract (layout, yamls, 3-way split, filename ordinals, IR bit-depth note, calibration files). **Phase 1 grid must train on Pohang-only yamls** — if the current combined Pohang+MIT yamls (A1-2) mix MIT frames, Pohang-only yamls are required first. | Benchmark decision is Pohang-based per plan C4; a combined training set would silently change what Table 1 measures | answer A1-2; plan C4 |
| D17 | 2026-07-07 | **Gaussian head integration = in-place conversion, no Ultralytics fork.** A loaded `DetectionModel` gets a fresh `cv4` log-variance branch added to its live `Detect` head + a class swap (`GaussianDetect`/`GaussianDetectionModel`); trainer is a thin `DetectionTrainer` subclass (4th loss item, warm-up epoch sync callback). σ parameterized as log σ² over LTRB distances in stride units (same targets as DFL), converted to pixels at inference; σ rides through NMS as extra channels. **Gradient policy defaults (A4-11 conservatism): σ branch reads detached features, NLL sees detached μ → deterministic detector trains bit-identically to baseline by construction.** Warm-up = NLL weight 0 (σ's only gradient source) — no param-group surgery. All relaxable via the `gaussian:` config block for the Phase 2 ablation. | Survives ultralytics upgrades better than a vendored fork; pretrained weights untouched; parity guaranteed, not hoped for | plan B1; scope §6.2, §11.3 B.1; verified against pinned ultralytics 8.4.90 internals |
| D18 | 2026-07-07 | **DFL-derived uncertainty (§7.2 option (a)) emitted at inference from the same trained model** — per-coordinate std of the DFL bin distribution, appended after σ. Table 2's "DFL-derived vs explicit Gaussian" ablation therefore needs zero extra training. | The single-model ablation D1 promised, made concrete | plan B1; scope §7.2; R2 |

## Answers received (Laksh, 2026-07-07) — §A questions

| Q | Answer | Consequence |
|---|---|---|
| A1-1 | H100, no GPU-hour limit | D15: full grid stands; batch default 32 pending dry-run |
| A1-2 | Combined Pohang+MIT data already in README layout | D16: Pohang-only yamls required for the Phase 1 grid — see dataset_requirement.md §2 |
| A1-3 | Colab/Kaggle not needed | — |
| A2-4 | Custom train/test split already built; wants leakage confirmation | D6-rev: `scripts/audit_split.py` is the confirmation; must PASS before training |
| A2-5 | pohang01 (night) goes **into training** | D6-rev; night test blocks requested to keep a real-night Table 3 row |
| A2-6 | Stride-subsampling approved | `benchmark.train_stride: 2`; `scripts/make_stride_subset.py` |
| A2-7 | Classes = ship, buoy. MIT annotation count/QA **still open** | Class harmonization ✓; MIT onboarding schedule still unknowable |
| A2-8 | Labels complete; calibration files **still open** | D7 (homography) blocked until answered — Phase 2, not urgent |
| A3-9 | No venue/deadline | Optional items scheduled by value, not deadline |
| A3-10 | No release planned — licensing deferred | — |
| A4-11 | σ² branch approved conditionally (must earn its keep empirically) | D1 amended; interpretation on record there |
| A4-12 | No preference → D8 default confirmed | VIS grid + IR top-2 confirmation |
| A4-13 | Claude's call on Table 3 smoothing | D14: per-frame (α=1) headline, smoothing as ablation |

Also: one-phase-at-a-time with explicit per-phase go-ahead restated (D11 overlap suspended); `dataset_requirement.md` requested and written.

## State — implemented / stubbed / untested

| Component | Status |
|---|---|
| Repo scaffold (git, layout, .gitignore) | **Implemented** (this session) |
| `config.yaml` + loader (`src/uqfusion/config.py`) | **Implemented + smoke-tested**: `python -m uqfusion.config` |
| `requirements.txt` (pinned) | **Implemented + install-verified** (clean resolve, py3.13). Note: `lightning`/`rich` pinned explicitly because torch-uncertainty 0.12.1 needs them at import but doesn't declare them |
| `scripts/smoke_env.py` (env + variant-name check) | **Implemented + green**: all imports OK; all 6 Table 1 variants build in ultralytics 8.4.90 (params from-yaml: v8s 11.2M, v9s 7.3M, v10s 8.1M, 11s 9.5M, 12s 9.3M, 26s 10.0M — run on server post-install too) |
| Data-list utilities (`uqfusion/data/lists.py`: yaml/txt/dir resolution, run keys, frame ordinals) | **Implemented + exercised by smoke** |
| Split audit (`uqfusion/data/audit.py` + `scripts/audit_split.py`) | **Implemented**; logic exercised on synthetic lists only — real verdict happens on the server against Laksh's split |
| Stride subsetting (`uqfusion/data/subset.py` + `scripts/make_stride_subset.py`) | **Implemented**; untested against real Pohang lists (needs server data) |
| Benchmark grid (`uqfusion/bench/grid.py` + `scripts/run_benchmark.py`, resume-safe CSV) | **Implemented + CPU smoke-tested end-to-end** (`scripts/smoke_benchmark.py` → SMOKE OK) |
| FPS harness (`uqfusion/bench/fps.py` + `scripts/measure_fps.py`) | **Implemented + smoke-tested** (fp16 path untestable on CPU — first real test on server) |
| Table 1 generator (`uqfusion/bench/table.py` + `scripts/make_table1.py`) | **Implemented + smoke-tested** (params/GFLOPs populate; env stamp embedded) |
| Val-split carving tool (train/test → train/val/test) | **Not built** — only needed if Laksh's split lacks val (open question OQ-1) |
| Gaussian σ² head + β-NLL/warm-up (`uq/gaussian.py`, `uq/train_gaussian.py`) | **Implemented + §18-2 gate GREEN** (`scripts/smoke_gaussian.py`): warm-up engages (nll=0 epochs 1–2), σ non-degenerate (CV 0.42), **σ tracks injected noise** (noisy-cue u 0.0639 > clean-cue 0.0538), detection unharmed (toy mAP50 0.97) |
| DFL-derived uncertainty extractor (§7.2 option (a), D18) | **Implemented + smoke-verified** — emitted from the same pass; toy stats: 9.39px, CV 0.81 vs trained σ 4.47px, CV 0.42 |
| UQ inference (`uq/infer.py` UQPredictor: σ through NMS + pooled features + safe-globals ckpt loading) | **Implemented + smoke-tested** |
| Mahalanobis OOD scorer (`uq/mahalanobis.py`) | **Implemented + smoke-tested** — toy separation: clean d̄ 33.9 vs degraded 59.3, 100% beyond clean 95th pct |
| `compute_reliability()` + constants fitting (`uq/reliability.py`, §6.4 verbatim incl. empty-frame fallback, R_sys) | **Implemented + smoke-tested** — gate favored clean stream 16/16 frames (R̄ 0.81 vs 0.16) |
| Reliability-weighted WBF fusion + homography transform (`uq/fusion.py`) | **Implemented + smoke-tested with identity H** — real IR→VIS homography blocked on calibration files (OQ-5); per-fused-box σ propagation deferred to Phase 3 |
| Bulk prediction-cache runner (all val/test frames → npz for B5-2 ablations) | **Not built** — Phase 3 harness work; the per-frame producer (UQPredictor) exists |
| Gaussian σ² head + β-NLL (scope §6.2) | Not started (Phase 2; CPU-side may overlap Phase 1) |
| `compute_reliability()` (scope §6.4) | Not started (Phase 2) |
| IR→VIS homography + WBF fusion | Not started (Phase 2; blocked on A2-8 for calibration files) |
| Mahalanobis OOD scorer (scope §6.3) | Not started (Phase 2) |
| MC-Dropout / Deep Ensembles / calibration suite | Not started (Phase 3) |
| MIT dataset onboarding | Not started (Phase 2+, parallel) |

## Open questions — awaiting Laksh

| # | Question | Blocks |
|---|---|---|
| OQ-1 | **Does the custom split include a `val` set, or only train/test?** Val drives Table 1 ranking + early stopping + later gate fitting; test stays untouched until Phase 4 (dataset_requirement.md §3). If only train/test, I'll build the val-carving tool. | Grid launch |
| OQ-2 | **Are the Phase 1 yamls Pohang-only,** or do the combined (A1-2) yamls mix MIT frames? If mixed → Pohang-only yamls needed (D16; dataset_requirement.md §2). | Grid launch |
| OQ-3 | **IR bit depth/normalization:** are the IR training images 8-bit already, and if converted from 16-bit, which normalization was used? Please add `data/pohang/IR_PREPROCESSING.md` (dataset_requirement.md §5). | IR confirmation grid; matters again for crossover realism + Mahalanobis in Phases 2–4 |
| OQ-4 | MIT annotation count + QA status (carried from A2-7) | MIT onboarding schedule (Phase 2+) |
| OQ-5 | Pohang sensor calibration files → `data/pohang/calibration/` (carried from A2-8) | Homography D7 — Phase 2 |
| OQ-6 | ~~Dry-run timing report~~ **Demoted per Laksh:** epochs/batch set to standard values (100/32, patience 20); dry-run timing is optional tuning whenever the server runs happen | Nothing — tweak-later |

## Next actions

**Laksh, on the H100 server, whenever server time happens** (full command sequence: dataset_requirement.md, bottom):
1. Clone repo, edit `config.yaml` `paths:`, `pip install -r requirements.txt -e .`, `pip freeze > requirements.lock.txt`.
2. `python scripts/smoke_env.py` (expect CUDA True), `python scripts/smoke_benchmark.py`, `python scripts/smoke_gaussian.py`, `python scripts/smoke_uq_pipeline.py` — all four should print their OK lines on the server too.
3. **Phase 1 GPU portion:** `audit_split` (must PASS) → `make_stride_subset` → full grid (`run_benchmark.py --data <derived-yaml>`) → `measure_fps` → `make_table1` → IR top-2 confirmation grid. Epochs/batch are set to standard values (D15); a 1-epoch dry-run first is optional tuning.
4. **Phase 2 real-data validation (after the Phase 1 winner is known, or on yolov8s meanwhile):** train one Gaussian model on a small real Pohang subset via `uqfusion.uq.train_gaussian.train_gaussian`, plus the same variant WITHOUT the σ branch at identical settings → the §12.1 parity check (A4-11's condition: mAP within seed noise). I'll wrap this pair as a script when the split questions (OQ-1/OQ-2) are answered.
5. Answer OQ-1/OQ-2/OQ-3 (val split, Pohang-only yamls, IR bit depth); drop calibration files at `data/pohang/calibration/` (OQ-5).

**Then (me):** Table 1 → selection memo + §7.2 resolution (D1/D18 make it a one-model comparison) → port the Gaussian conversion to the winner if it isn't v8/11-family → **phase gate: Laksh's go-ahead for Phase 3** (baselines: MC-Dropout insertion + deep ensembles; eval harness with pre-registered metrics; cache runner; learned gate).

## Run log

| Date | Phase | Run | Config / commit | Result | Notes |
|---|---|---|---|---|---|
| 2026-07-07 | 2 | `smoke_gaussian` (CPU, synthetic heteroscedastic, yolov8n@320, 10 ep, warmup 2 + ramp 2) | post-`37bac1e` working tree | **GREEN** — nll/epoch [0, 0, 0.06, −0.11, −0.15, −0.17, −0.18, −0.14, −0.15, −0.11]; σ: 4.47px CV 0.42; noise-direction PASS (0.0639 > 0.0538); toy mAP50 0.97 | Negative NLL is correct (log σ² < 0 when σ < 1 stride unit). Deterministic: identical trajectory across re-runs |
| 2026-07-07 | 2 | `smoke_uq_pipeline` (CPU, same weights) | ditto | **GREEN** — O3 separation clean 33.9 / degraded 59.3 (100% > clean p95); λ=2.00, μ_d=48.8, τ=4.68; gate 16/16 frames; empty-frame fallback exercised | Blank-frame R=0 is correct here (blank IS OOD for the toy model) — the real "clear empty sea = reliable" semantics can only be tested on real data with empty-sea frames in the clean fit set |
