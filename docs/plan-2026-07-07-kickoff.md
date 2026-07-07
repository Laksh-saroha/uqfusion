# Kickoff — Architecture Review (Task 1) + Backbone Selection Plan (Task 2)

> Approved by Laksh on 2026-07-07 (unedited). Copied into the repo so a cold clone is self-contained; the working copy of open questions lives in `progress.md`.

## Context

Pre-code review deliverable for the maritime UQ-fusion project. `scope.md` is the source of truth; every recommendation below cites the section it responds to. Nothing is coded yet; per the agreed workflow, no implementation until this document is reviewed, questions answered, and signed off. One scope-doc inconsistency and one genuine design gap are flagged below (B6-2, B6-9, D-note) — neither changes the architecture's spine, which I agree is sound and publishable as scoped.

---

## A. Clarifying questions (batched by topic)

**A1 — Compute & logistics**
1. What GPU is on the Jupyter server (model, VRAM, count), and is there a wall-clock/queue limit? Rough GPU-hour budget for Phase 1? The benchmark protocol (C5) is parameterized by this.
2. Is Pohang Canal + PoLaRIS already downloaded and organized in the per-modality YOLO layout of §5.1 (`data_vis.yaml` / `data_ir.yaml`), or is that prep part of Phase 1? How much disk is available on the server?
3. Are Colab/Kaggle still in play as overflow compute (§11.1), or is the Jupyter server the only trainer?

**A2 — Data & splits**
4. Does the PoLaRIS release define an official detection train/val/test split (check [github.com/sparolab/PoLaRIS](https://github.com/sparolab/PoLaRIS))? If not, my default is split-by-run + contiguous temporal blocks, never random frames (B6-2). OK?
5. Related: I propose holding out pohang01 (night) entirely from training, so the Table 3 "low-light" column tests genuine distribution shift. Agree, or do you want night data in training?
6. May I stride-subsample training frames (every 2nd–3rd) for the Phase 1 benchmark? 10 Hz video is hugely redundant; this cuts GPU cost 2–3× and won't change the ranking. Val/test never subsampled.
7. MIT Marine Perception: how many images are annotated so far, what classes, QA status? (Determines when it can realistically enter — I propose Phase 2+, off the critical path; C4.)
8. Does the Pohang dataset's sensor calibration (intrinsics/extrinsics between VIS and IR) come with your download? Needed for the fusion coordinate-frame issue in B6-9.

**A3 — Manuscript & constraints**
9. Target venue / fellowship deadline? This sizes the optional items (DETR row, evidential, conformal).
10. Ultralytics is AGPL-3.0. Fine for a public research fork + paper? Any institutional constraint?

**A4 — Method confirmations**
11. §4 guardrail reading: I read "single YOLO backbone used as-is except for the variance output branch" as permitting a σ² branch **added alongside** the existing DFL head, DFL box path untouched (my 7.2 recommendation depends on this). Confirm?
12. Table 1 scope: benchmark on VIS for ranking, confirm winner + runner-up on IR — rather than the full variant grid on both modalities (halves cost; C4). OK?
13. Is Table 3 evaluated per-frame (α = 1) or on sequences with temporal smoothing? Affects how α is ablated and disclosed (B5-4).

---

## B. Task 1 — Open decisions

### B1. §7.2 — DFL-derived vs explicit Gaussian σ² (conditional on backbone)

**Recommendation: option (c) — add a Gaussian σ² branch on top, keep the DFL box path intact — with option (a) as a free ablation row.** Conditional per candidate:

- **Backbone with DFL (v8/v9/v10/11/12):** add the σ branch alongside DFL. The trained model then yields *both* uncertainty sources — explicit σ² and DFL-distribution statistics (spread/entropy per GFLv2) — so the §7.2 ablation "DFL-derived vs explicit Gaussian" becomes a **single-model comparison at ~zero extra GPU cost**, and it can't be confounded by "those are two different trainings." This is the strongest version of the ablation §7.2 asks for.
- **YOLO26 (DFL removed — see C1):** only the explicit Gaussian branch is possible; the DFL-derived row is unavailable. This is a real strike against 26 as the selected backbone unless it clearly wins Table 1.
- **Option (b), replace DFL entirely:** documented fallback only, if joint training destabilizes. It also technically *changes* the detector's regression pathway, which cuts against §4's attribution logic — (c) honors the guardrail better.

Why (c) over (a)-alone: NLL-trained σ² is a proper scoring rule giving a real predictive distribution — which Table 2's calibration metrics (NLL, coverage) need. DFL's distribution is trained to be sharp around the target, not to be calibrated; its spread is a useful *ranking* signal (GFLv2) but not a calibrated variance. Expected outcome: explicit σ² wins on calibration, DFL-derived is a decent free proxy — either way it's a paper paragraph, per §7.2's own framing.

Implementation decisions to record now (details in Phase 2): σ branch predicts in the same parameterization as the box head (LTRB in stride units for the v8 family), converted to x/y/w/h σ for §6.4's `u_i`; warm-up = μ path trains normally first (§6.2); β-NLL default β = 0.5; σ-branch gradients into the trunk stopped during warm-up to protect baseline mAP, then verified against §12.1's "UQ head doesn't degrade detection" check.

Per §18, final call is made after Phase 1 — the above pre-commits the decision *rule*, not the answer.

### B2. §7.3 — Evidential regression

**Recommendation: cite (R6, R7), do not benchmark in the core grid.** Reasons: (1) Comparison 2's claim is "single-pass matches sampling-based UQ at 1× cost" — evidential is *another* single-pass method, so benchmarking it doesn't test the claim, it duplicates our seat at the table; (2) a NIG head + evidential loss is a second custom-head engineering effort with its own known instabilities (regularizer sensitivity), realistically weeks, on a UG-fellowship timeline where the Gaussian head is already the hardest item (§11.3 B.1); (3) the epistemic story is already covered — Deep Ensembles give per-box epistemic (variance of member means) once built for Comparison 2, and Mahalanobis covers frame-level (§6.3).

Mitigation for reviewers: build the head/loss as a **pluggable interface** (Gaussian head = one implementation) so an evidential head is a bounded revision-time addition, and say so in the manuscript. Revisit only if a target venue demands it.

### B3. §7.4 — Independence under correlated degradation

**Agree with the caveat + "both-degraded" Table 3 row; add one cheap mechanism.** Precise diagnosis first: under correlated degradation the per-modality signals *do* fire (each stream's Mahalanobis distance and σ² both degrade) — what fails is the **normalization** `w_m = R̄_m / ΣR̄`, which discards absolute magnitude, so two near-zero reliabilities still produce confident-looking weights summing to 1.

**Fix: expose absolute system reliability `R_sys = max(R̄_vis, R̄_ir)`** (or noisy-OR `1−(1−R_vis)(1−R_ir)`) alongside the fused output; below a validation-calibrated threshold the system flags "no reliable modality" instead of silently fusing. Zero architecture change — one extra scalar output plus one extra OOD-separation histogram in §12.3 (does R_sys separate both-degraded frames?). This converts the acknowledged weakness into a small positive result: the same signals detect total-system degradation.

Both-degraded test conditions: heavy rain (physically degrades VIS *and* LWIR) and fog + synthetic crossover combined, generated per §10.5.

### B4. §7.5 — Learned gate upper bound

**Include, scheduled in Phase 3** (needs the eval harness and cached predictions; must not gate Phase 2). Keep it tiny: logistic regression or 2-layer MLP on per-frame features already computed — `[U_box, O, #detections, mean confidence] × 2 modalities` → fusion weight. Training is minutes on CPU.

The one real design question is the supervision target: propose per-frame soft label = relative detection quality of each modality vs GT (per-modality F1 or mean-IoU ratio) on degraded validation frames. Flagging now; finalize in Phase 3.

Bonus framing: fitting §6.4's constants on validation (B5) is the *constrained* version of this gate, so the ablation reads "interpretable form + fitted constants vs unconstrained learned ceiling" — exactly the story §7.5 wants.

### B5. §6.4 — The tunable constants (λ, μ_d, τ, α, multiplicative form)

1. **No hand-picked numbers.** Define every constant as a statistic of the clean validation set: μ_d and τ from the clean-val Mahalanobis distribution (e.g., μ_d = 95th percentile, τ ∝ IQR → O ≈ 0 on clean data *by construction*); λ from the clean-val U_box distribution (e.g., λ such that r_box = 0.9 at clean-val median). Pre-register the rule; grid-search only what remains.
2. **All gate-level tuning and ablation runs on cached predictions.** Run each detector once over the eval data, cache boxes/σ²/features; the entire §6.4 design space (λ rule, combination form, α) is then CPU post-processing. This makes the ablation table nearly free — a major experimental-design win worth exploiting deliberately.
3. **Combination-form ablation (pre-registered):** multiplicative (default — encodes "either failure condemns a stream") vs `min(r_frame, r_box)` vs weighted geometric mean `r_frame^γ · r_box^(1−γ)`.
4. **α:** sweep on validation sequences including α = 1 (off); disclose whichever Table 3 uses (question A4-13); note the interaction with the 10 Hz frame rate.
5. **Anti-leakage rule:** constants fitted on a val split disjoint from test; synthetic-degradation severities/seeds used for tuning ≠ those used for test (§11.3 B.6 discipline, extended to gate tuning).

### B6. §9 sanity check — two-comparison design

The separation itself (backbone benchmark vs uncertainty-source comparison) is correct and reviewer-proof; keep it. Nine flags:

1. **Pre-declare the Table 1 selection rule** (C5) — otherwise the backbone choice looks post-hoc no matter how honest it was.
2. **Split hygiene is the biggest validity threat in the whole design.** 10 Hz video → random frame-level splits leak near-duplicate frames across train/val and inflate *every* number in *every* table. Split by run + contiguous temporal blocks with buffer gaps. (scope.md doesn't specify splits; this must be fixed before any training.)
3. **MC-Dropout isn't free on YOLO.** Ultralytics detect heads have no dropout; inserting it and retraining = a *different deterministic model*. Pre-fix placement (head convs, p ≈ 0.1–0.25) and T (e.g., 10), and report that model's deterministic row too, so the calibration comparison is fair.
4. **Cross-pass/member box matching** (needed to get box variance from MC-Dropout/ensembles) must be one fixed protocol (WBF clustering or Hungarian on IoU) shared by all methods — calibration metrics are sensitive to it. Document as method, not knob.
5. **Define detection-calibration metrics now.** §12's ECE formula is classification-ECE; detection needs precise definitions: confidence calibration = D-ECE (objectness-vs-precision binning at IoU ≥ 0.5); regression calibration = z-score coverage / interval-ECE per coordinate (is |gt−μ| ≤ kσ at rate Φ(k)?); ranking quality = sparsification error + AURC on σ-ranked detections vs realized IoU. Pre-registering these kills any metric-shopping suspicion.
6. **Add calibration-under-shift to Table 2.** Same metrics on fog/low-light/crossover val sets, not just clean — the fusion story depends on σ² staying informative exactly where the gate needs it. Nearly free via cached predictions (B5-2).
7. **Seed/budget accounting.** Ensemble members ARE the seed replicates: one M=5 ensemble per modality (not 3 × 5). Gaussian head and MC-Dropout get 3 seeds each. State this; the naive reading of "≥3 seeds × M=5 × 2 modalities" is 30 trainings.
8. **Decouple Table 3 from Comparison 2.** Every Table 3 row uses the Gaussian head only, so the headline fusion result lands in Phase 2 — before MC-Dropout/ensembles exist. The phase plan exploits this.
9. **Fusion needs a common coordinate frame — a real gap.** §5.1 says "no spatial registration; decision-level fusion (WBF) tolerates this," but WBF only merges boxes that overlap *in the same image plane*. VIS (2048×1080) and IR (640×512) have different FOVs; with no mapping at all, the same vessel's boxes may not overlap → WBF degenerates to a weighted union, and there's no single reference frame for evaluating fused output against GT. Fix is cheap and standard: derive a static homography IR→VIS from the dataset's sensor calibration (the Pohang IJRR release ships calibration; question A2-8), evaluate fusion in the VIS frame. Decision-level fusion tolerates *imperfect* registration, not *no* registration — the scope wording should be amended.

Also: report Comparison 2's "inference cost ×" column as measured wall-clock, not just nominal ×T/×M (batching effects make the nominal number misleading). And a notation nit: §6.2 says per-detection uncertainty = mean predicted *variance* while §6.4's `u_i` uses σ/s (std, size-normalized) — σ/s is the right dimensionless choice; fix §6.2's wording.

### B7. §4 guardrail — position

**Agree with the freeze; no push-back.** It's what makes the attribution claim clean, and §12 is explicit that the project is judged on calibration, not mAP. Two disclosures (not design changes): (1) absolute small-object (buoy) numbers will be modest without P2/SPD — identical handicap across every compared system, and imgsz is a config lever, not architecture, if needed; (2) B1's recommendation (add branch, keep DFL) is chosen partly *because* replacing DFL would technically alter the detector and weaken attribution.

---

## C. Task 2 — Backbone selection plan (O1)

### C1. Landscape update — newer than §19's reference list

- **YOLO26** (Ultralytics, released 2026-01-14; [arXiv 2606.03748](https://arxiv.org/abs/2606.03748), [docs](https://docs.ultralytics.com/models/yolo26)): **DFL removed**, natively NMS-free end-to-end, ProgLoss, STAL (small-target-aware label assignment — relevant to buoys), MuSGD optimizer. In-family NMS-free candidate, but no DFL → kills the free §7.2 ablation on it.
- **YOLO12** (Feb 2025, attention-centric): in Ultralytics, DFL present — belongs in the shortlist. **YOLOv13** (third-party, hypergraph): skip — outside Ultralytics tooling, violating §11.2's identical-config machinery.
- DETR side has moved past R17/R18: **RT-DETRv4** ([arXiv 2510.25257](https://arxiv.org/abs/2510.25257)), **DEIMv2** (DINOv3-based), **RF-DETR** (ICLR 2026, [arXiv 2511.09554](https://arxiv.org/abs/2511.09554); DINOv2 + NAS; first real-time detector past 60 AP COCO). §19 should gain these citations regardless of whether one is benchmarked.
- **PoLaRIS** has a citable paper ([arXiv 2412.06192](https://arxiv.org/abs/2412.06192), [repo](https://github.com/sparolab/PoLaRIS)) — update R20; check the repo for an official split (A2-4).
- Re-check the landscape at Phase 4 (space turns over roughly quarterly).

### C2. Shortlist — six rows, one scale

| Table 1 row | DFL | NMS-free | Head/loss fork difficulty | Notes |
|---|---|---|---|---|
| YOLOv8s | ✓ | ✗ | **Low** — canonical `Detect`/`v8DetectionLoss`; most-forked codepath in the ecosystem | Reference point |
| YOLOv9s | ✓ | ✗ | Low–med — v8-style head; PGI/aux specifics | GELAN backbone |
| YOLOv10s | ✓ | ✓ (one-to-one head) | Med — dual assignment → loss touched in two places | In-family NMS-free data point |
| YOLO11s | ✓ | ✗ | **Low** — same `Detect` codepath as v8 | Ultralytics mainline pre-26 |
| YOLO12s | ✓ | ✗ | Low–med — standard head; attention backbone; known training-speed quirks | |
| YOLO26s | ✗ | ✓ (native) | Med–high — new head, no DFL, MuSGD; newest code = churn risk | STAL may help buoys; no DFL ablation |

All Ultralytics-native → one config, one API (§11.2). **Fix one scale — s** (params/FPS sweet spot; every row clears the 10 Hz = 100 ms sensor budget, so FPS is a tiebreaker, not a gate). Drop to n-scale or 4 rows (v8/v10/11/26) only if the GPU budget forces it (A1-1).

Pre-declared prior, so the benchmark confirms or surprises rather than rationalizes: v8s/11s likely land within noise of the best mAP and win on fork simplicity.

### C3. DETR row — defer to Phase 4

(1) §9.1's sole purpose is selecting the YOLO backbone; the DETR row evidences the "detector-agnostic" claim (§8) and gates nothing downstream. (2) Identical-config fairness doesn't transfer anyway — the smallest Ultralytics RT-DETR (`rtdetr-l`) isn't size-matched to s-scale YOLOs and DETRs want different optimizer/epoch regimes, so the row needs its own footnoted config regardless of when it runs. (3) The DETR SOTA is churning (C1); choosing the row at manuscript time (RF-DETR / DEIMv2 / RT-DETR-for-the-citable-anchor) keeps it current instead of stale. (4) **YOLOv10 in the shortlist already provides an NMS-free point *inside* the identical-config comparison** — defusing most of the "NMS distorts uncertainty semantics" question at zero extra cost. If the Phase 1 GPU sits idle, `rtdetr-l` via Ultralytics is the cheap opportunistic version.

### C4. Benchmark dataset — Pohang only

Table 1 = **Pohang VIS** for ranking (127k images, densest labels), then **winner + runner-up confirmed on Pohang IR** (guards against a modality-specific surprise at ~1/5 the grid cost). MIT Marine Perception deferred to Phase 2+: its annotation pipeline is a named risk (§13) and would put the slowest data work on the critical path for the *least* decision-relevant experiment; its §5.1 role is fusion evaluation/generalization, not backbone ranking. Bring MIT online in parallel with Phase 2 coding; hard requirement by Phase 4. SMD/MassMIND stay supplementary per §5.3.

### C5. Protocol (exact)

- **Splits:** by run + contiguous temporal blocks with buffer gaps (B6-2). Default: pohang00/02/03 (+04 VIS-only) → train/val via within-run blocks; **pohang01 (night) fully held out** for Phase 4 adverse eval, so Table 1 ranking reflects day-domain fit, not night shift (A2-5). Adopt the official PoLaRIS split instead if one exists (A2-4).
- **Fixed across variants:** imgsz 640; batch = largest fitting *all* variants, pinned; epochs ≈ 60 with patience 15 (finalized after the dry-run below); optimizer = per-model tooling default under one uniform rule, resolved values disclosed (forcing a single optimizer on all rows would unfairly tank variants tuned around another, e.g., YOLO26/MuSGD); default Ultralytics augmentation; AMP; `deterministic=True`; seeds {0, 1, 2}; same GPU; pinned `ultralytics`/`torch` versions; W&B (or TensorBoard) logging.
- **Subsampling:** stride 2–3 on training frames if budget requires (A2-6); val/test never subsampled.
- **Metrics:** P, R, mAP@50, mAP@50–95 on the common val split — mean ± std over 3 seeds. FPS measured once per variant (seed-invariant): batch = 1, fp16 and fp32 both reported, 50-frame warmup, ≥500 frames, ms/frame alongside, on the actual training GPU. Params + GFLOPs from the profiler.
- **Selection rule (pre-declared, written to progress.md before the first run):** among variants within ~1 pooled std of best mAP@50–95, prefer DFL-present > simplest fork path > higher FPS.
- **First GPU action:** 1-epoch timing dry-run on one variant to calibrate the cost table below and finalize epochs/stride before committing the grid.
- **Cost estimate (to be calibrated):** 6 variants × 3 seeds = 18 runs (+ ~4 IR confirmation runs). At stride-2 (~60k train images), 60 epochs, s/640: order 2–5 h/run on A100-class ≈ 40–90 GPU-h total; ~3–4× that on T4-class. Tight budget → n-scale or 4 rows.
- **Output:** `benchmark/` harness → per-run CSV → auto-generated Table 1 (markdown, mean ± std) + a one-page selection memo that also resolves §7.2 per B1. Commit tagged `phase1-table1`.

---

## D. Phase plan (user's structure, three amendments)

| Phase | Content | Amendment |
|---|---|---|
| **0 — Scaffold** (immediately post-approval) | git init; `progress.md`; pinned `requirements.txt`; config file skeleton (all paths/hypers); README with §5 dataset layout | Named explicitly — it's where the portability constraints get locked in |
| **1 — Backbone selection** | C5 protocol → Table 1 + selection memo + §7.2 resolution | **Multi-seed benchmark harness is built here**, not at §18 step 5 — Table 1 is impossible without it; §18's own ordering is internally inconsistent on this |
| **2 — Proof of concept** | Gaussian head + β-NLL/warm-up on tiny subset (σ² non-degenerate per §18-2), then `compute_reliability()` + homography + WBF fusion + Mahalanobis, end-to-end small-scale | **Start Phase 2's CPU-side work during Phase 1's GPU runs** — the head is the highest-risk item (§18) and is CPU-verifiable on tiny subsets; fork targets the shared v8/11 `Detect` codepath so it ports to whichever DFL variant wins. Pure wall-clock win, no dependency violation |
| **3 — Baselines + eval harness** | MC-Dropout (B6-3), Deep Ensembles, full calibration suite w/ pre-registered metrics (B6-5, B6-6), learned gate (B4), cached-prediction gate ablations (B5) | MIT dataset onboarding runs in parallel here |
| **4 — Publication run** | Multi-seed runs both datasets; Table 3 incl. both-degraded row + R_sys histogram (B3); DETR row decision (C3); landscape re-check (C1); manuscript outputs | |

**Verification gates:** Phase 0 = repo runs `pip install -r requirements.txt` + config loads on a path-free clone. Phase 1 = Table 1 regenerates from CSVs by one script; selection memo approved. Phase 2 = tiny-subset harness green (loss ↓, σ² non-degenerate, baseline mAP within noise) *before* any GPU commitment; end-to-end fusion demo on a handful of paired frames. Phase 3 = calibration suite reproduces a known sanity case; baselines' deterministic rows reported. Phase 4 = every table regenerated from committed artifacts + tagged commits (§11.1).

**This turn's gate:** user reviews this document, answers §A, pushes back on §B/§C. No code until sign-off.
