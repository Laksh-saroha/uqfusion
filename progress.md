# Progress — uqfusion

> Purpose: a cold read of this file (plus `scope.md` and `docs/plan-2026-07-07-kickoff.md`) must reconstruct where the project stands without re-deriving anything. Updated at every phase boundary and every non-trivial decision.

**Last updated:** 2026-07-07 (session 1)
**Current phase:** Phase 0 — scaffold — **COMPLETE**. Repo, pinned env (install-verified, `SMOKE OK` on Python 3.13/CPU), portable config + loader, docs, smoke script all in and committed.
**Next phase:** Phase 1 — backbone benchmark harness. Can be *coded* now; its config values (epochs/batch/stride/scale) stay PENDING until the open questions below are answered and the on-server timing dry-run runs.

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
- Plan → Laksh reviews/pushes back/approves → implement, **one phase at a time**.
- **No GPU in the dev environment.** CPU smoke tests only here; all real training on Laksh's GPU Jupyter server.
- Portability: no absolute paths anywhere; machine specifics live in `config.yaml` only.
- Every trainable component ships a tiny-subset smoke path before any GPU time (scope §18-2 discipline, applied everywhere).
- Git commit at every phase boundary (scope §11.1: which commit produced which result).

## Decision log

| # | Date | Decision | One-line rationale | Source |
|---|---|---|---|---|
| D1 | 2026-07-07 | §7.2: Gaussian σ² branch **added alongside** DFL (option c); DFL-derived uncertainty = free same-model ablation row; full DFL replacement = fallback only. Final call after Phase 1 winner is known. | Single-model ablation, preserves deterministic baseline and §4 attribution | plan B1; scope §7.2 — pending A4-11 confirmation |
| D2 | 2026-07-07 | §7.3: evidential regression cited, **not benchmarked**; head/loss built as pluggable interface | Doesn't test Comparison 2's claim; second custom head ≈ weeks off critical path | plan B2; scope §7.3 |
| D3 | 2026-07-07 | §7.4: keep caveat + both-degraded Table 3 row; **add absolute `R_sys` abstain signal** | Failure is weight normalization discarding magnitude, not missing signals | plan B3; scope §7.4 |
| D4 | 2026-07-07 | §7.5: learned gate included as upper bound, Phase 3, tiny logistic/MLP on existing features | Cheap ceiling comparison; supervision target TBD in Phase 3 | plan B4; scope §7.5 |
| D5 | 2026-07-07 | §6.4 constants: every constant = pre-registered statistic of clean validation data; **all gate ablations on cached predictions** (CPU post-processing) | Reproducible, no hand-tuning, ablation table nearly free | plan B5; scope §6.4 |
| D6 | 2026-07-07 | Splits: by run + contiguous temporal blocks with buffer gaps; **never random frames**; pohang01 (night) held out entirely | 10 Hz video → random splits leak near-duplicates into every table | plan B6-2, C5 — pending A2-4/5 |
| D7 | 2026-07-07 | Fusion coordinate frame: static IR→VIS homography from dataset calibration; fuse + evaluate in VIS frame | WBF needs a common image plane; scope §5.1's "no registration needed" is too strong | plan B6-9 — pending A2-8 |
| D8 | 2026-07-07 | Table 1 shortlist: YOLOv8s/v9s/v10s/11s/12s/26s, s-scale, seeds {0,1,2}, Pohang VIS + winner/runner-up IR confirm; selection rule pre-declared (within ~1 std of best mAP@50–95 → DFL > simplest fork > FPS) | All Ultralytics-native = one config/API; rule prevents post-hoc choice | plan C2/C4/C5; scope §8, §9.1 |
| D9 | 2026-07-07 | DETR row deferred to Phase 4 | Gates nothing; identical-config unfair to DETRs; that SOTA churning (RF-DETR/DEIMv2/RT-DETRv4) | plan C3; scope §8 |
| D10 | 2026-07-07 | MIT Marine Perception deferred to Phase 2+ parallel onboarding; hard requirement by Phase 4 | Annotation pipeline is a named risk (§13); keep it off the Phase 1 critical path | plan C4; scope §5.1, §13 |
| D11 | 2026-07-07 | Phases: added Phase 0 scaffold; benchmark harness built in Phase 1 (not §18 step 5); Phase 2 CPU-side head work overlaps Phase 1 GPU runs | §18's ordering internally inconsistent (Table 1 needs the harness); wall-clock win | plan D; scope §18 |
| D12 | 2026-07-07 | Tracking default: TensorBoard (local); W&B = opt-in toggle in config | No account dependency; scope §11.2 allows either | plan C5; scope §11.2 |
| D13 | 2026-07-07 | Stack pinned (see `requirements.txt`): torch 2.12.1, ultralytics 8.4.90 (YOLO26 support), + UQ/eval libs; full lock frozen on server at Phase 1 start | Reproducibility per §11.1; versions verified against PyPI 2026-07-07 | scope §11 |

## State — implemented / stubbed / untested

| Component | Status |
|---|---|
| Repo scaffold (git, layout, .gitignore) | **Implemented** (this session) |
| `config.yaml` + loader (`src/uqfusion/config.py`) | **Implemented + smoke-tested**: `python -m uqfusion.config` |
| `requirements.txt` (pinned) | **Implemented + install-verified** (clean resolve, py3.13). Note: `lightning`/`rich` pinned explicitly because torch-uncertainty 0.12.1 needs them at import but doesn't declare them |
| `scripts/smoke_env.py` (env + variant-name check) | **Implemented + green**: all imports OK; all 6 Table 1 variants build in ultralytics 8.4.90 (params from-yaml: v8s 11.2M, v9s 7.3M, v10s 8.1M, 11s 9.5M, 12s 9.3M, 26s 10.0M — run on server post-install too) |
| Phase 1 benchmark harness (`benchmark/`) | Not started (next) |
| Split-builder (by-run temporal blocks) | Not started (Phase 1; blocked on A2-4/5) |
| Gaussian σ² head + β-NLL (scope §6.2) | Not started (Phase 2; CPU-side may overlap Phase 1) |
| `compute_reliability()` (scope §6.4) | Not started (Phase 2) |
| IR→VIS homography + WBF fusion | Not started (Phase 2; blocked on A2-8 for calibration files) |
| Mahalanobis OOD scorer (scope §6.3) | Not started (Phase 2) |
| MC-Dropout / Deep Ensembles / calibration suite | Not started (Phase 3) |
| MIT dataset onboarding | Not started (Phase 2+, parallel) |

## Open questions — awaiting Laksh (full text: plan §A)

Answered: none yet (plan approved without edits 2026-07-07; answers expected with the next review pass).

| Q | Topic | Blocks |
|---|---|---|
| A1-1 | GPU model/VRAM + Phase 1 GPU-hour budget | Final epochs/batch/scale; whether 6 rows or 4 |
| A1-2 | Pohang+PoLaRIS downloaded? layout matches §5.1? disk? | Dataset-prep scope of Phase 1; README layout confirmation |
| A1-3 | Colab/Kaggle overflow still available? | Contingency planning only |
| A2-4 | Official PoLaRIS split? | Split-builder implementation (D6 default stands otherwise) |
| A2-5 | pohang01 (night) fully held out? | Split-builder |
| A2-6 | Stride-subsampling of training frames OK? | Benchmark cost; `benchmark.train_stride` |
| A2-7 | MIT annotation status (count/classes/QA) | MIT onboarding schedule (D10) |
| A2-8 | Pohang sensor calibration files in hand? | Homography (D7) — Phase 2, not urgent yet |
| A3-9 | Target venue / deadline | Sizing optional items (DETR row, conformal) |
| A3-10 | AGPL-3.0 acceptable? | Nothing immediate; affects release packaging |
| A4-11 | Confirm §4 permits σ² branch alongside DFL | D1 — needed before Phase 2 head work |
| A4-12 | Table 1 = VIS ranking + IR confirm? | Benchmark grid size |
| A4-13 | Table 3 per-frame (α=1) or smoothed sequences? | α ablation design — Phase 3, not urgent yet |

## Run log

*(empty — begins with the Phase 1 timing dry-run on the GPU server)*

| Date | Phase | Run | Config / commit | Result | Notes |
|---|---|---|---|---|---|

## Next actions

1. Close Phase 0: verify `pip install -r requirements.txt -e .` + config smoke test in the local venv; initial commit.
2. Laksh: answer §A questions (13 items above) — A1-1/A1-2/A2-4/A2-5/A2-6/A4-12 gate Phase 1 config; the rest can trail.
3. Phase 1 coding (after answers, or with PENDING defaults if instructed): split-builder + `benchmark/run_benchmark.py` (multi-seed loop) + FPS measurement + Table 1 generator, each with a CPU smoke path.
4. On the GPU server: 1-epoch timing dry-run → finalize epochs/batch/stride → launch the grid.
