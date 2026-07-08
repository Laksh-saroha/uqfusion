# Progress — uqfusion

> Purpose: a cold read of this file (plus `scope.md` and `docs/plan-2026-07-07-kickoff.md`) must reconstruct where the project stands without re-deriving anything. Updated at every phase boundary and every non-trivial decision.

**Last updated:** 2026-07-07 (session 1; Phase 3 code complete and smoke-verified; HOW_TO_RUN.md written)
**Current phase:** Phase 3 — baselines + eval harness — **CPU-verifiable portion COMPLETE, smoke gate GREEN** (`scripts/smoke_phase3.py`; see Run log). MC-Dropout, Deep Ensembles, prediction caches, corruption generation, the pre-registered calibration suite, learned gate, and the gate-ablation sweep are all implemented; the three eval CLIs are verified. **`HOW_TO_RUN.md` is the canonical per-phase runbook.** Remaining for Phases 1–3 close-out: the GPU/server runs (Laksh, whenever server time happens) + the OQ answers below.
**Phase 2 — CPU portion COMPLETE** (commit `e75c110`): O2→O3→O4 chain green on both gates; server-side remainder = real-data run + §12.1 parity check + real homography (OQ-5).
**Phase 1 — code COMPLETE** (commit `37bac1e`), GPU portion deferred; epochs/batch at standard values (D15). Closes with Table 1 + selection memo.
**Phase 0 — COMPLETE** (commit `ffbe87f`): repo, pinned env, portable config, docs.
**Phase 4 — NOT STARTED** (needs go-ahead + Phase 1–3 server results): Table 3 runner, real VIS↔IR pairing, both-degraded row + R_sys histogram, MIT in, DETR row decision, multi-seed publication runs.

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
| D19 | 2026-07-07 | **MC-Dropout protocol pre-fixed:** one `Dropout2d` inserted before the final 1×1 conv of each head branch (cv2 + cv3, every level); p=0.15, T=10 (config `baselines.mc_dropout`); retrained from pretrained weights; its own training results.csv IS the required deterministic row (val runs dropout-off) | Dropout isn't free on YOLO — a different deterministic model, disclosed | plan B6-3; scope §9.2, R5 |
| D20 | 2026-07-07 | **One cross-pass/member matching protocol** (`uq/clustering.py`, shared by MC-Dropout and ensembles): greedy by descending conf; per other source, best-IoU unused same-class detection at IoU≥0.55 joins; cluster conf = mean(conf)·support/n_sources; σ = per-coordinate member std; **singletons get σ = box size** (max size-normalized uncertainty ≈ 1) — a disclosed convention, not NaN | Calibration metrics are sensitive to matching; fixed as method, not knob | plan B6-4 |
| D21 | 2026-07-07 | **Metric definitions pre-registered in code** (`eval/metrics.py` module docstring): D-ECE (conf-vs-precision bins @IoU≥0.5), per-edge z-score coverage + interval-ECE, per-edge Gaussian NLL, AUSE + AURC (risk = 1−IoU, FP=1), OOD AUROC; mAP = local COCO-style 101-point (no pycocotools dependency). Calibration-under-shift = same metrics on corrupted caches | Kills post-hoc metric shopping (B6-5) and makes B6-6 a cache re-read | plan B6-5/B6-6; scope §12 |
| D22 | 2026-07-07 | **Ensemble = M seed replicates trained via the Phase 1 grid runner** (resume-safe; per-member deterministic rows free in the CSV); M=5 seeds {0..4} in config; frame features for OOD taken from member 0 only (one consistent feature space) | plan B6-7 seed accounting, executed | plan B6-7; scope §9.4, R4 |
| D23 | 2026-07-07 | **Corruptions** (fog/rain/lowlight/glare/blur/noise) via albumentations, severity 1–3, deterministic per frame from (seed, index); the seed is stamped into cache meta. **Anti-leakage rule operationalized: tuning caches and final-test caches must use different `--corrupt-seed`** | plan B5-5's train/test degradation separation, enforced by convention + traceable metadata | plan B5-5; scope §5.2, §11.3 B.6 |

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
| Prediction caches (`eval/cache.py` + `scripts/build_cache.py`, gaussian/mc/ensemble sources, optional corruption, meta-stamped) | **Implemented + smoke-tested** (8 caches round-trip) + CLI verified |
| MC-Dropout (`uq/mc_dropout.py` + `scripts/train_mc_dropout.py`; D19 protocol) | **Implemented + smoke-tested**: 6 dropout layers armed, cluster σ CV 1.62, toy deterministic mAP50-95 0.60 |
| Deep Ensemble (`uq/ensemble.py` + `scripts/train_ensemble.py`; D22, trains via grid runner) | **Implemented + smoke-tested** (M=2 toy): member-disagreement σ CV 0.96 |
| Shared clustering protocol (`uq/clustering.py`; D20) | **Implemented + smoke-tested** via both baselines |
| Corruption generation (`eval/corruptions.py`, 6 conditions, albumentations 2.0.8; D23) | **Implemented + smoke-tested** (all 6 valid + changed) |
| Pre-registered metrics (`eval/metrics.py`: D-ECE, interval-ECE/coverage, NLL, AUSE, AURC, OOD AUROC, local COCO mAP; D21) + `scripts/evaluate_uq.py` | **Implemented + smoke-tested** — finite for all 3 sources incl. the DFL §7.2 row; observed toy-scale MC/ens NLL blow-up (4003/627 vs Gaussian 2.8) = the known agreeing-passes-overconfidence effect, informative not a bug |
| Learned gate (`eval/learned_gate.py`; D4/B4 design) | **Implemented + smoke-tested**: favors clean stream 100% (mean w_vis 0.101) |
| Fusion-system eval + gate ablation (`eval/fusion_eval.py` + `scripts/ablate_gate.py`) | **Implemented + smoke-tested**: gated 0.739 vs blind-VIS 0.000 vs IR 0.778 mAP50-95; 6-row ablation sweep |
| `HOW_TO_RUN.md` (per-phase runbook) | **Written; all documented CLIs verified on this machine** |
| Phase 4 (Table 3 runner, real VIS↔IR pairing, MIT onboarding, DETR row) | **Not started** — awaiting phase gate |
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
| OQ-7 | **640×640 delivery (Laksh 2026-07-08): letterboxed or stretched?** Letterbox required (dataset_requirement.md §5); stretching distorts each modality differently and degrades fusion overlap. Also confirm full-res originals stay archived (imgsz is the small-object lever). Same-size does NOT remove the homography need (OQ-5). | Data prep correctness; buoy-recall headroom |
| OQ-4 | MIT annotation count + QA status (carried from A2-7) | MIT onboarding schedule (Phase 2+) |
| OQ-5 | Pohang sensor calibration files → `data/pohang/calibration/` (carried from A2-8) | Homography D7 — Phase 2 |
| OQ-6 | ~~Dry-run timing report~~ **Demoted per Laksh:** epochs/batch set to standard values (100/32, patience 20); dry-run timing is optional tuning whenever the server runs happen | Nothing — tweak-later |

## Next actions

**The canonical runbook is now [`HOW_TO_RUN.md`](HOW_TO_RUN.md)** — per-phase commands for the server, verified locally.

**Laksh, on the H100 server, whenever server time happens:**
1. Setup + full smoke suite (HOW_TO_RUN §0–§1; expect CUDA True and all five OK lines).
2. **Phase 1:** audit (must PASS) → stride → full grid → FPS → Table 1 → IR top-2 confirm (HOW_TO_RUN §2). Send back `table1.md`, `table1_ir.md`, audit report.
3. **Phase 2:** Gaussian training on the winner + the §12.1 parity row (HOW_TO_RUN §3).
4. **Phase 3:** baselines → caches → `evaluate_uq` (Table 2) → `ablate_gate` (HOW_TO_RUN §4).
5. Answer the OQ table above (OQ-1/2/3 gate real-data training; OQ-5 calibration files gate the homography).

**Then (me):** selection memo + §7.2 resolution from Table 1 → port Gaussian conversion if the winner isn't v8/11-family → analyze Table 2/ablation outputs → **phase gate: Laksh's go-ahead for Phase 4** (Table 3 runner with real VIS↔IR pairing + homography, both-degraded row + R_sys histogram, MIT onboarding, DETR row decision, multi-seed publication runs, landscape re-check per plan C1).

## Run log

| Date | Phase | Run | Config / commit | Result | Notes |
|---|---|---|---|---|---|
| 2026-07-07 | 2 | `smoke_gaussian` (CPU, synthetic heteroscedastic, yolov8n@320, 10 ep, warmup 2 + ramp 2) | post-`37bac1e` working tree | **GREEN** — nll/epoch [0, 0, 0.06, −0.11, −0.15, −0.17, −0.18, −0.14, −0.15, −0.11]; σ: 4.47px CV 0.42; noise-direction PASS (0.0639 > 0.0538); toy mAP50 0.97 | Negative NLL is correct (log σ² < 0 when σ < 1 stride unit). Deterministic: identical trajectory across re-runs |
| 2026-07-07 | 2 | `smoke_uq_pipeline` (CPU, same weights) | ditto | **GREEN** — O3 separation clean 33.9 / degraded 59.3 (100% > clean p95); λ=2.00, μ_d=48.8, τ=4.68; gate 16/16 frames; empty-frame fallback exercised | Blank-frame R=0 is correct here (blank IS OOD for the toy model) — the real "clear empty sea = reliable" semantics can only be tested on real data with empty-sea frames in the clean fit set |
| 2026-07-07 | 3 | `smoke_phase3` (CPU: MC p=0.15/T=5 + M=2 ensemble @10 ep; 8 caches; full metric/gate/fusion chain) | post-`e75c110` working tree | **GREEN** — all 6 corruptions valid; MC σ CV 1.62, ens σ CV 0.96; metrics finite for all sources incl. DFL row; learned gate 100% clean-preference; systems mAP50-95: vis(blind) 0.000 / ir 0.778 / naive 0.738 / gated 0.739 / learned 0.740; 6-row ablation | Toy-scale notes: (1) MC/ens NLL blow-up (4003/627 vs Gaussian 2.8) = agreeing-passes overconfidence — the contrast Table 2 exists to measure, exaggerated at T=5/tiny models; (2) fully-blind degraded stream → all combination rules converge in the ablation (expected: R≈0 condemns the stream under every rule); differences emerge under partial degradation at real scale. Two smoke-design fixes en route: gate training needs both outcomes represented (interleaved degradation), ensemble smoke CSV must live under the smoke root (resume-skip reused stale members) |
