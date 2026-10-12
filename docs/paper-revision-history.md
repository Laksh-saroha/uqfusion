# PAPER_DRAFT2.md: revision history

Moved out of the paper's header on 2026-10-12, when the draft was finalized. Newest last.

* **2026-09-27, Draft 2.** Claims revised after the 2026-09-27 claims audit and the Phase 3 night check
  (`docs/eval/p3_night_check_2026-09-27.md`).
* **2026-10-08/09, figures.** Figures 1–8 are drawn by `scripts/paper_figures.py` from the files their captions cite
  (Figure 2 added 2026-10-09). Figures 9–11 are drawn by `scripts/fig_detections.py` from the Phase 3 development
  caches (2026-10-09; frames re-chosen the same day by visual review of ground-truth-only renders).
* **2026-10-09, throughput.** The numbers of §6.1 and Table 1b were found to be fp32, not fp16 as labelled, and
  yolo26l to have been timed in the same sweep; both corrected from `phase1_benchmark/fps.csv`.
* **AP convention.** Every number carries its AP convention (local linear-interpolation AP unless stated).
* **2026-09-20, the pohang04 look.** Taken once; §7 reports it.
* **2026-10-09/10, corruption v2.** The synthetic corruptions were found broken at the source and replaced by a
  camera-chain model (v2, §5.1). Table 3b, Table 6, Figures 7, 9 and 10 and the corrupted rows of Table 7 report v2;
  pohang04's corrupted cells were re-scored under it as a disclosed second exposure; R-D1 and Stage 1 were re-run on it
  as robustness checks.
* **R-D1's metric.** R-D1 under `crossmodal26m` returned NULL on 2026-09-27 on the ship-and-buoy macro
  (`docs/eval/uq_mechanism_ablation_26m_2026-09-27.md`). Re-scored on 2026-10-08 on its registered metric, ship AP,
  the score path is POSITIVE under both presets and the coordinate path NULL
  (`docs/eval/uq_mechanism_ablation_ship_2026-10-08.md`); §6.4 reports both and the deviation.
* **2026-09-27, Phase 3 corrupted cells** (`docs/eval/p3_corrupt_cells_2026-09-27.md`). Table 2 is the
  pre-registered three-arm calibration slice, with day as the primary basis.
* **2026-10-09, the σ re-ranker.** A σ-free ablation of the out-of-fold VIS re-ranker showed σ adds +0.0050
  [+0.0038, +0.0064] ship AP (`docs/eval/class_set_audit_2026-10-08.md` §8, not pre-registered); the title and
  abstract were reframed to "useful within a detector". Its pre-registered replication on the five Phase 3 detectors
  (`docs/prereg-sigma-rerank-replication-2026-10-09.md`) returned FAILS the same day, so the title, abstract, §1,
  contribution 1, §6.4 and §8 reverted to the 2026-10-08 framing, as registered
  (`docs/eval/sigma_rerank_replication_2026-10-09.md`).
* **2026-10-10/11, no-retraining changes.** Replacements for the gate, and changes that retrain no detector, were
  registered and scored on fresh corruption draws (§6.9, §6.10, Table 8); the abstract, contribution 2, §8 and §9
  item 2 report them. All are development results.
* **2026-10-11/12, test-time adaptation and held-out scorings.** §6.11 (N4 and its variants, development), §7.3
  (six later registered pohang04 scorings: P-U, P-F, P-T, P-A, P-FT, P-C2, P-G) and §7.4 (the pohang01 test-split
  night test). §2.5 (related work on test-time adaptation) added 10-12 after the novelty check
  (`docs/novelty-ttbn-2026-10-12.md`).
* **2026-10-12, finalized.** Decisions by Laksh: the paper recommends N4 (N4c2 for a live stream), and N4F where ten
  passes per stream are affordable; IR-gated test-time adaptation becomes contribution 4; the title is widened to
  cover it. Changed: title, abstract, §1, §2.4 Table R (this work's row), §2.5 (three more papers read in full:
  DASP, SuMi, ABPEM; the "each variant" sentence corrected), new §4.3 (N4 as a method), §7.3 (Figure 12, the
  `ttbn_lex` observation), §8 (two new paragraphs: the thermal camera as a judge; what we recommend), §9 (item 26),
  references (three added). The header's revision note moved here.

* **2026-10-12, one paper.** The four venue drafts were collapsed into one TMLR paper, `PAPER.md`, at Laksh's decision. It is led by the IR-gated test-time adaptation (N4): new title, abstract, introduction, related work, method (§4, with a new method figure, `fig_n4_gate`), results (§5, Tables 1–2 and Figure 2 rebuilt from the `docs/eval/ttbn/` reports) and a short account of what did not work (§6). The uncertainty, veto and gate-replacement results moved to Appendices D–F, drawn from Draft 2 with tables, figures and cross-references renumbered. Dropped from the paper (kept in Draft 2): Table R, Table 1b, and Figures 2, 4, 5, 6, 10 and 11. Draft 2 stays as the evidence record. The MaCVi, PBVS, JOE and old TMLR drafts and their build script moved to `venues/archive/`; `venues/build_tmlr_latex.py` became `paper/build_latex.py`. The compiled PDF is 50 pages; the main body ends on page 11 and the references start on page 12.
