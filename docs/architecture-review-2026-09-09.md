# Architecture review: problems, evidence, and suggested changes

Review date: 9 September 2026. Repository: `D:\\project`. Source revision inspected: `1886258`. Review started on 8 September. No agents were launched. No training, dataset edits, production-code fixes, or remote queue changes were performed.

## Assessment

The project has a useful research scaffold, but its strongest architectural claims are not yet supported by its evaluation. The most urgent work is to repair the experiment and data contracts before expanding the model search.

The central problems are:

1. The nominally held-out night set participates directly in fitting the cross-modal gate, and the reported day set has repeatedly participated in model selection.
2. The local AP implementation differs from COCO AP; its optimized bootstrap also disagrees with its own reference when a resample loses a class.
3. Night-label deletion created much of the failure that the night veto was designed to fix. Later restoration experiments invalidate the earlier claim that night VIS contains no useful signal.
4. Detaching the Gaussian branch does not guarantee identical detector training. Shared gradient clipping still couples the updates.
5. MC/ensemble box disagreement is being compared with a learned residual variance as if both were the same predictive distribution.
6. The later `crossmodal26m` preset disables the learned Mahalanobis and box-uncertainty soft weights. Its reported improvements therefore do not establish the original uncertainty-gated-fusion claim.
7. Cache identity, label integrity, and queue ownership are largely conventions rather than enforced interfaces.

I would retain the separate modality detectors, the inexpensive Gaussian branch as a hypothesis, cached evaluation, the corrected end-to-end-head integration, and the practice of recording negative results. I would revise the evaluation protocol, narrow the scientific claims, and consolidate the implementation around an explicit versioned system specification.

### How to read this review

* **P1:** resolve before using affected results as confirmatory evidence or starting a campaign that depends on them.
* **P2:** material correctness, interpretation, maintainability, or deployment issue; some affect experimental options or historical configurations only.
* **Reproduced:** a small local counterexample was executed against project functions.
* **Source-confirmed:** the code establishes the behavior; its full-dataset effect was not rerun.
* **Documented:** the repository reports the measurement; I have not independently repeated that training/evaluation.
* **Inference:** a consequence or architectural judgment, with its assumptions stated.

Historical mistakes are not automatically current bugs. Several findings below were already recognized in later handoffs; the remaining issue is that code comments, defaults, reports, and public-facing claims still carry earlier assumptions.

## 1\. Priority findings

### F01 — P1: held-out night data is used to fit the gate

**Evidence: source-confirmed.** [fit\_structure\_gate.py:132](D:/project/scripts/fit_structure_gate.py:132) computes:

```python
ir\_thr = (ir\_v\[ir\_fit].max() + ir\_v\[ir\_night].min()) / 2
```

The night observations are from the same paired reference used for the night evaluation. The resulting threshold is explicitly described as a midpoint involving held-out night. This contradicts the earlier claim in the file that no night frame informs a threshold. Calling a sample held out in metadata does not make it independent of fitting.

The IR health fit also estimates its mean, standard deviations, covariance, and maximum-distance bound from **all clean paired IR frames, including night**. See [fit\_structure\_gate.py:224](D:/project/scripts/fit_structure_gate.py:224). Its reported 0% clean outlier rate under the maximum bound is an in-sample construction, not validation of a low false-alarm rate. The separate p99 authority bound has a different in-sample interpretation and should not be conflated with that maximum.

**Consequence.** Night results remain descriptive measurements on this recording. They do not establish an unseen-night generalization result for the fitted gate. This does not imply every numerical gain disappears after repair.

**Suggestion.** Provide a genuine calibration set containing healthy day and night examples, then freeze the gate before scoring a different test block/run. Store calibration frame IDs with each fitted parameter. If only one night run is available, use time-separated calibration/test blocks with guard intervals and explicitly limit the claim to that recording; obtain another night recording for a cross-run claim. Refit all affected image-statistic bounds and recompute the affected tables.

**Acceptance check.** A test manifest must have zero intersection with every fitting manifest; changing any test image or statistic must leave fitted constants byte-identical. The clean false-alarm rate must be measured on data outside the bound fit.

### F02 — P1: the development set has become the test set

**Evidence: source-confirmed and documented.** [ctx.py:55](D:/project/src/uqfusion/eval/ctx.py:55) defines the 1,200 day frames through `FIT\_RUNS`. Its comments acknowledge that these were used for both fitting and reporting. A later partition defines `TUNE\_RUNS=(pohang00,)` and `TEST\_RUNS=(pohang02,pohang03)`, but [load\_context:251](D:/project/src/uqfusion/eval/ctx.py:251) still defaults to `capability\_sel="fit"`, so the capability prior uses the larger set unless explicitly overridden.

The [1 September experiment log:147](D:/project/docs/experiment-log-2026-09-01.md:147) discusses keeping support IoU 0.30 after 0.55 performs poorly on TEST. A test that rejects candidates is participating in selection. It need not choose the winning number explicitly to cause adaptive evaluation. The [2 September log:943](D:/project/docs/experiment-log-2026-09-02.md:943) also documents an actual held-out selection mistake and a cache-alignment error.

Different corruption seeds on the same scenes test robustness to the random transform draw; they do not supply new scene-level test data. New preregistrations after looking at these runs improve transparency but cannot erase previous exposure. This is the model-selection bias discussed by [Cawley and Talbot](https://jmlr.org/papers/v11/cawley10a.html).

**Suggestion.** Treat all repeatedly inspected paired/day substrates as development data. Use nested leave-one-run-out evaluation there for tuning and diagnostics, with every fitted component inside the fold. Reserve the original untouched test split, if it is still untouched, for one frozen end-to-end evaluation. If it has already been inspected, label it accordingly and reserve new data. Fix the corruption families, seeds, comparisons, and stopping rule before that evaluation.

**Acceptance check.** The final scoring command cannot fit constants or select checkpoints. Its manifest names a pre-existing frozen system and an independent test release. Night held out from the **gate** must not be described as night held out from detector training: D6-rev explicitly allows night training frames.

### F03 — P1: local AP is not standard COCO AP

**Evidence: reproduced.** [matching.py:169](D:/project/src/uqfusion/eval/matching.py:169) and [apmetrics.py:74](D:/project/src/uqfusion/eval/apmetrics.py:74) linearly interpolate the precision envelope with `np.interp`. [Official COCO evaluation](https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocotools/cocoeval.py) samples the envelope at the first attained recall at or above each threshold using `searchsorted`. COCO also specifies stable score sorting, ignored objects, area ranges, and maximum detections.

Two tiny ranked-detection examples, without crowd/ignore complications, reproduce a difference:

|Ranked outcomes|Project AP|COCO recall lookup|Difference|
|-|-:|-:|-:|
|TP, FP, TP; 2 GT|0.83168317|0.83498350|-0.00330033|
|TP, FP, FP, TP; 2 GT|0.74752475|0.75247525|-0.00495050|

These differences are large relative to several reported improvements. They do **not** prove that a real result changes sign; that requires rescoring. The installed Ultralytics evaluator uses its own interpolation/integration convention, so matching Ultralytics and matching COCO must also be distinguished. Comparing two local functions that share the same formula is not an independent metric check.

**Suggestion.** Use one authoritative evaluator for final tables, preferably COCO evaluation with an explicit task configuration. Keep the fast cached implementation only after parity testing against that evaluator. If retaining the current formula for historical continuity, name it a custom AP variant and provide an official-AP companion table.

**Acceptance check.** Cover imperfect recall, duplicate recall values, tied scores, missing classes, no predictions, wrong classes, per-image detection caps, and image resampling. Recompute effect sizes and intervals from the same metric version; do not compare Phase 1 Ultralytics numbers directly with the custom fusion AP.

### F04 — P1: the bootstrap has both a numerical defect and an independence problem

**Evidence: reproduced and source-confirmed.** The optimized `presort` / `ap\_weighted` path retains classes from the original dataset even when a resample contains no GT for one of them. The literal `ap\_from\_parts` path derives the class set from the resample. For two perfect one-object frames of different classes, resampling the first frame twice produces **1.0** through the reference path and **0.5** through the optimized path. See [apmetrics.py](D:/project/src/uqfusion/eval/apmetrics.py).

This counterexample concerns macro AP and sparse-class resamples; it is not evidence that a large ship-only bootstrap suffers the same class-drop error. Default unstable `argsort` calls also make exact repeat/tie semantics worth specifying.

Separately, many headline intervals resample individual frames from 10 Hz recordings. Paired resampling correctly preserves the comparison between systems on each frame, but does not remove dependence between neighboring frames. The effective number of independent observations is not automatically 1,032 because a run contains 1,032 frames. A frame bootstrap is principally an interval conditional on this observed recording and resampling assumption; it cannot estimate variation between unseen night runs.

**Suggestion.** First repair fast/reference equivalence under a declared missing-class policy. Then use paired contiguous-block resampling within runs, show sensitivity to block length, and report run-level effects. Estimate training-seed variability separately from scene-sampling variability. With one night run, state that between-run night uncertainty cannot be estimated. Avoid treating bootstrap sign-flip fractions as calibrated p-values or probabilities that the architectural hypothesis is true.

**Acceptance check.** Reference/fast equivalence on randomized small fixtures, then a sensitivity table comparing frame and block intervals. Every reported interval must identify its resampling unit and what randomness it covers.

### F05 — P1: night-label filtering changed the problem being solved

**Evidence: documented, with the filtering mechanism in source.** The [2 September log:702](D:/project/docs/experiment-log-2026-09-02.md:702) records the frame-level night deletion and restoration. The original operation removed 132,688 boxes from 17,502 training label files; the later per-box procedure restored 94,553 and retained 38,135 exclusions. The newer tree is therefore not simply the untouched original annotation release.

The [rebaseline proposal:15](D:/project/docs/rebaseline-proposal-2026-09-04.md:15) reports clean-night fused AP increasing from 0.0850 with the veto to 0.2635 without it when using the night-capable VIS detector. That is strong evidence against the earlier blanket premise that night VIS has no learnable information. It is not an independently reproduced result of this review, and it is not proof that every night veto should be removed: other degraded cells in the same table benefit from rejecting VIS.

Deleting a visible object's annotation while leaving its image in training teaches an ordinary detector to treat that object as background. A visibility-score heuristic is not a ground-truth determination of learnability. This matters for both the old frame filter and any proposed IR thermal-crossover label filter.

**Suggestion.** Publish separate immutable annotation releases: original, frame-filtered historical, and reviewed restoration. Audit the remaining per-box exclusions with human inspection stratified by visibility, size, and run. Represent genuinely unassessable annotations as ignore/censored regions when the training/evaluation pipeline supports them, instead of silently converting them into negatives. Retrain comparison arms on the same frozen release, as the later retraining handoff already proposes. Re-evaluate the veto as a sensor-health decision on that release.

**Acceptance check.** Labels cannot change during training; a starting and ending content hash must match. Every checkpoint names its label release. Historical night-cut and restored-label results appear in different tables. A preregistered INCONCLUSIVE result must not be presented as evidence that the old physical premise is correct.

### F06 — P1: detached gradients do not guarantee detector training parity

**Evidence: source-confirmed; coupling reproduced analytically.** [gaussian.py:194](D:/project/src/uqfusion/uq/gaussian.py:194) detaches the mean in the NLL; its branch reads detached features. These correctly remove direct gradient paths into the detector. However, the trainer is inherited, and [installed Ultralytics trainer.py:785](D:/project/.venv/Lib/site-packages/ultralytics/engine/trainer.py:785) clips **all model parameters together** at norm 10 before stepping the optimizer.

With a detector gradient of 6 and an independent sigma-branch gradient of 100, the detector's clipped gradient becomes approximately **0.598923**, although it would remain 6 without the sigma branch. Shared AMP overflow/step skipping is another possible optimizer-level coupling. This is sufficient to refute bit identity “by construction”; it does not identify the magnitude of this effect in the recorded GPU runs.

The [18 August parity investigation:47](D:/project/docs/phase2-laptop-2026-08-18.md:47) already records failed end-to-end parity. Its exclusion of sigma clipping concerns **zero-NLL warm-up**, when sigma gradients are zero. That exclusion does not establish parity once NLL is active. Conversely, active-NLL clipping does not explain the earlier warm-up discrepancy.

**Suggestion.** Choose the intended contract. For strict detector parity, isolate sigma optimization, including clipping and any skip decision that could affect detector updates. For a statistical non-degradation claim, remove the bit-identity promise and predefine a meaningful non-inferiority margin across matched seeds. In both cases test actual optimizer steps with active NLL, not just unscaled raw gradients.

**Acceptance check.** Compare detector parameters after full steps in warm-up, ramp, and active NLL, with AMP on/off and resume. For a statistical claim, compare independently evaluated checkpoints under the same data and training protocol. A zero NLL weight is also not a literal freeze of every sigma-branch state: BatchNorm buffers and optimizer weight decay deserve separate treatment.

### F07 — P1: the uncertainty baselines do not represent comparable predictive variance

**Evidence: source-confirmed and reproduced.** [clustering.py](D:/project/src/uqfusion/uq/clustering.py) uses a confidence-weighted mean box, unweighted coordinate standard deviation among matched predictions, and a support-penalized confidence. Singletons receive box-size sigma. Two identical but wrong detections give sigma zero; the current NLL evaluator returns NaN, while interval coverage correctly exposes severe undercoverage in this example.

For a Gaussian mixture, total predictive variance contains both the average member variance and variation between member means:

```text
Var(Y | x) = E\_m\[Var(Y | x, m)] + Var\_m(E\[Y | x, m])
```

The project MC/ensemble sigma is principally disagreement among detected boxes. It omits a learned residual/noise term and is not automatically comparable to the Gaussian branch's residual-scale prediction. The original [Deep Ensembles paper](https://arxiv.org/pdf/1612.01474) explicitly uses probabilistic regression members and combines within-member variance with between-member variation. The project is entitled to use deterministic detection ensembles, but must describe and evaluate that different estimator accurately.

IoU clustering further conditions uncertainty on detection survival, source agreement, and a 0.55 association threshold. Support-penalized scores, arbitrary singleton variance, and confidence-weighted locations change accuracy and ranking along with uncertainty. M=5 members supply one ensemble, not five independently trained ensemble replicates.

**Suggestion.** State whether the comparison targets disagreement ranking or a full predictive distribution. For ranking, evaluate disagreement honestly alongside recall and calibration on matched detection subsets. For predictive likelihood, use probabilistic members with total-variance aggregation or a declared mixture likelihood, or fit a residual component on an independent calibration set for every relevant arm. Show sensitivity to clustering and singleton handling. Distinguish MC deterministic dropout-off accuracy, stochastic MC accuracy, member accuracy, and ensemble accuracy.

**Acceptance check.** Identical wrong members cannot be interpreted as reliable predictions. Report degenerate-sigma frequency, association support, TP counts, recall, and common-TP comparisons beside NLL. The code already records nonpositive-sigma share; preserve and surface it. Do not replace undefined/degenerate likelihoods with a convenient winning scalar or label a failed predictive model merely “no signal.”

### F08 — P2: the Gaussian branch estimates residual scale, not identified pure aleatoric uncertainty

**Evidence: mathematical interpretation of the implemented objective.** With a fixed mean predictor, the population optimum of ordinary Gaussian NLL is approximately:

```text
sigma²(x) = E\[(Y - mu\_fixed(x))² | x]
          = Var(Y | x) + (E\[Y | x] - mu\_fixed(x))²
```

Thus systematic detector bias enters the fitted scale. Assignment changes, truncation, incomplete labels, and dataset shift complicate the interpretation further. The diagonal four-edge representation also omits dependence between edges and does not guarantee joint box coverage. Per-edge coverage is a valid limited measurement, but cannot certify a joint 4D confidence region.

Beta-NLL is a defensible training option, not a guarantee of calibrated variance. Its stop-gradient sample weighting changes the optimization; the detached-mean architecture already removes a major mean/variance interaction that motivates the method. See [Seitzer et al.](https://arxiv.org/abs/2203.09168). Likewise, the DFL bin spread is a localization-quality signal; [GFLv2](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.html) does not establish that its raw variance is a calibrated posterior over physical box edges.

**Suggestion.** Call the output a learned conditional localization-error scale unless a stronger decomposition is tested. Compare beta=0 and beta=0.5 on a development protocol; consider post-hoc scale calibration on held-out calibration frames. Report coverage by edge, object size, modality, day/night, and uncertainty quantile. If joint coverage is claimed, evaluate a joint region explicitly. Keep DFL-vs-Gaussian comparisons within a DFL-capable model; a YOLO12-versus-YOLO26 comparison changes much more than the variance source.

### F09 — P1: the later fusion architecture no longer tests its stated UQ mechanism

**Evidence: source-confirmed.** [ctx.py:557](D:/project/src/uqfusion/eval/ctx.py:557) sets `mu\_d=1e9` and `lam=0` in the cross-modal path. This makes the learned Mahalanobis soft weight and box-sigma soft weight inert for ordinary values. `sigma\_weighted` and sigma-in-score are also off by default. The [rebaseline proposal:58](D:/project/docs/rebaseline-proposal-2026-09-04.md:58) explicitly acknowledges this.

The later `crossmodal26m` behavior is primarily image-statistic sensor rejection plus capability-weighted box aggregation and cross-stream score support. Gaussian-head training may influence its detector checkpoints, but that is different from using predicted uncertainty to make the fusion decision. Image-statistic health scores are not the learned O2/O3 mechanism described in the original scope and progress report.

**Suggestion.** Separate two claims: (a) localization uncertainty quality and its compute cost; (b) image-statistic-guided sensor selection/support. Describe each active path exactly. To claim UQ improves fusion, run an ablation with identical predictions and identical fusion options, changing only the supplied uncertainty values, including a constant and a shuffled-uncertainty control. Remove dead reference-cache/scorer requirements from an inference path that does not use them, once compatibility is explicitly handled.

**Acceptance check.** A system manifest enumerates every active component. The final artifact cannot label a constant-UQ system as evidence for learned uncertainty weighting. A genuine mechanism ablation must show a reproducible difference attributable to uncertainty on independent evaluation data.

### F10 — P2: registration and uncertainty live in inconsistent coordinate systems

**Evidence: source-confirmed.** [fuse\_detections](D:/project/src/uqfusion/uq/fusion.py:278) transforms IR box corners into the VIS plane but later reads the original IR `sigma\_ltrb` and divides it by the VIS canvas scale. The comment says sigma is in VIS pixels; there is no matching covariance transformation in this path. A scale-2 transform must map a 1-pixel standard deviation to 2 pixels. Rotations, projective mapping, and corner-envelope selection require more than that scalar correction.

The inverse-variance path is experimental/off in the later default preset, so this defect does **not** by itself invalidate default `crossmodal26m` AP. It does invalidate the claim that the sigma-weighted option has been fully verified. The path also returns boxes, scores, and classes, without fused sigma, despite the original [scope:114](D:/project/scope.md:114) promising per-box uncertainty at the final output.

There is a separate reproduced geometry defect: [apply\_homography:34](D:/project/src/uqfusion/uq/fusion.py:34) clips the projective denominator to a positive epsilon. Homographies H and -H represent the same projective mapping, yet the function produces normal coordinates for one and enormous negative coordinates for the other. This is a latent issue for arbitrary valid homographies, not evidence that the stored normalized calibration currently has negative denominators.

**Suggestion.** Make coordinate frame and covariance representation explicit in prediction records. Propagate covariance with an appropriate Jacobian or sample-based transformation, including registration uncertainty where material. Define fused uncertainty under a stated dependence model. Validate signed projective division and reject points near the projective horizon instead of silently clipping their denominator.

**Acceptance check.** Identity, translation, scaling, sign-equivalent H, rotation, and horizon cases; uncertainty-transform checks against Monte Carlo samples; and an output schema that either contains tested fused uncertainty or explicitly declares it unavailable.

### F11 — P2: a common plane does not make boxes or labels correspond

**Evidence: documented.** [followup-analysis:141](D:/project/docs/followup-analysis-2026-08-20.md:141) reports registration residuals of several pixels and within-run horizontal swings approaching 10 pixels. The same record and later overlap probes report very few cross-modal partners at merge IoU 0.85. These observations justify caution about coordinate averaging; they do not establish a universal impossibility of useful registration.

A single homography is exact for a plane or pure camera rotation under the relevant camera model; depth variation with separated cameras introduces parallax. Distant-scene use is an approximation whose error must be measured. [OpenCV's homography explanation](https://docs.opencv.org/4.10.0/d9/dab/tutorial_homography.html) supports those conditions. Thermal and visible box corners need not refer to the same physical points, so RANSAC on box corners is not automatically a sound calibration method. The median-offset estimator's assertion that it survives a **majority** of incorrect correspondences is also false in general; see [iralign.py:17](D:/project/src/uqfusion/eval/iralign.py:17).

The current VIS-plane evaluation measures agreement with **VIS annotations**. IR-only physical objects absent from those annotations may count as false positives. That is a legitimate explicitly defined task, but differs from “detect every maritime obstacle.” A naive union of independently annotated boxes can double-count objects; varying its deduplication IoU changes the target definition.

**Suggestion.** Keep calibration-derived geometry as a baseline. Measure residuals by range proxy/size, location, run, and timestamp difference. Preserve a one-to-one timestamped pair manifest and enforce a declared maximum time offset. Build a small manually reconciled cross-modal test subset with object identity and visibility annotations before using union-label AP as a main result. Keep native-plane detector AP and VIS-plane system AP clearly separate.

**Acceptance check.** Pair identities and order are validated on loading, mapped geometry is inspected on small objects, and any jointly annotated benchmark has a documented correspondence/ignore policy. A low-overlap null is described as conditional on the tested detector/geometry combination.

### F12 — P2: the metric and abstention contracts do not match several claims

**Evidence: source-confirmed.** [metrics.py](D:/project/src/uqfusion/eval/metrics.py) computes confidence-binned TP-at-IoU-0.5 ECE, TP-only per-edge coverage/NLL, and detection-list sparsification. These are useful measurements with narrower meanings than the surrounding prose sometimes gives them:

* Confidence-only detection ECE is not a multidimensional calibration assessment conditional on box location and scale.
* Error/uncertainty correlation measures ranking, not calibrated uncertainty magnitude. Multiplying every sigma by 100 can preserve correlation and destroy coverage.
* TP-only likelihood compares different selected subsets when different detectors have different recall. Frames with no detections contribute no localization-risk observations.
* The reported detection AURC is a mean over a 20-point removal grid ending at 95% removal, with integer rounding. It is a particular approximation, not an exact full-coverage integral.
* [eval\_risk\_coverage\_fixed\_gt.py:14](D:/project/scripts/eval_risk_coverage_fixed_gt.py:14) calls its fixed-full-GT denominator the standard selective-risk definition. Standard selective risk conditions on accepted examples; see [SelectiveNet](https://proceedings.mlr.press/v97/geifman19a/geifman19a.pdf). Counting abstained GT as misses is a useful full-system metric, but answers a different question. Report both with their actual definitions.

Under the later gate, [ctx.py:846](D:/project/src/uqfusion/eval/ctx.py:846) builds `R\_sys\_gate` from hand-shaped novelty ratios and exports an abstain flag. It does not actually abstain from producing detections. The docstring acknowledges that choice. Ratios bounded to \[0,1] are not automatically probabilities of sensor correctness, and a threshold of 0.5 is not inherently a calibrated failure boundary. The [V2 preregistration:29](D:/project/docs/prereg-night-veto-v2.md:29) shows the old day-fitted VIS-health score flagging all clean night frames as unhealthy.

**Suggestion.** Define an output contract separating ranking score, calibrated probability, localization scale, novelty score, and an advisory health flag. Use “abstention” for an actual consumer decision, or name the current output an advisory reject flag. Pair conditional accepted-set risk with full-system missed-object cost and a random-rejection control. Report ship AP, buoy AP, and macro AP: macro AP still answers a valid two-class system question even when IR only supplies ships.

**Acceptance check.** A consumer can distinguish advisory flags from suppressed output. Calibration plots use probabilities within their declared domain. Cross-stream support can produce scores above 1 (a local example gives 1.47015), so those scores must not be presented as probabilities without recalibration. Every UQ table includes its denominator, TP count, and detection recall.

### F13 — P1: the split audit can pass an incomplete or insufficiently separated split

**Evidence: reproduced and source-confirmed.** [data/audit.py](D:/project/src/uqfusion/data/audit.py) checks proximity between train and the combined val/test category. It does not check val against test. A fixture with adjacent validation/test frames passes. A fixture with no test split also passes: the missing split is printed but does not change `ok`.

Runs containing no parseable ordinals never enter `by\_run`, and partial ordinal coverage is only a warning. The check uses a median filename-derived interval, which must not be assumed to equal the acquisition period after subsampling or missing frames. Exact path/stem checks also do not detect duplicated image content under different names.

These are audit defects, **not proof that the current full dataset actually leaks**. The configured dataset tree is absent from this checkout, so its present split could not be independently certified here.

**Suggestion.** Make the required split policy explicit per operation. A final three-way experiment should fail on a missing/empty required split, missing timestamps, and proximity across every relevant split pair, including val/test. Use actual timestamps or an explicitly supplied acquisition period. Add content/near-duplicate checks on a practical sampled or hashed basis, and audit paired modalities jointly.

**Acceptance check.** Adversarial fixtures fail: missing test, no ordinals, adjacent val/test blocks, renamed duplicate images, and cross-modal pairs split inconsistently. Emit an audit artifact bound to the exact data release and training recipe.

### F14 — P1: caches and resume keys do not identify the experiment completely

**Evidence: source-confirmed and partly reproduced.** [cache.py](D:/project/src/uqfusion/eval/cache.py) stores a pickle of records and caller-supplied metadata plus frame count and git HEAD. The main [build\_cache script](D:/project/scripts/build_cache.py) records useful paths and options, but does not establish a complete content identity of weights, labels, ordered pairs, preprocessing, and all inference options. [load\_cache](D:/project/src/uqfusion/eval/cache.py:55) performs no compatibility validation. [fusion\_eval.py:185](D:/project/src/uqfusion/eval/fusion_eval.py:185) checks equal lengths, not matching pair identities. Equal-length reordered or cross-substrate caches can therefore be accepted.

[split\_fingerprint:40](D:/project/src/uqfusion/bench/grid.py:40) hashes train/val frame names. A local label modification leaves the fingerprint unchanged. The [IR handoff:251](D:/project/docs/ir-handoff-2026-08.md:251) already identifies the corresponding normalization-tree false-skip problem. Adding a YAML basename, as proposed there, would only partly fix it: editing labels or pixels under the same basename still evades the key.

The grid's completed-run lookup does not include the complete recipe. Reusing a results CSV after changing epochs, image size, initial weights, or training overrides can skip a different requested experiment. Git HEAD alone also misses uncommitted source changes.

Finally, [load\_gt](D:/project/src/uqfusion/eval/matching.py:29) silently treats a missing label as empty and skips short malformed lines. Missing-path evaluation can therefore look like a valid background-only dataset. Explicit empty-label annotations are legitimate; missing files should not be silently equated with them.

**Suggestion.** Create an immutable experiment manifest and hash it. Include ordered pair IDs, data/annotation release hashes, checkpoint content hash, preprocessing definition, dimensions, class map, inference branch, thresholds, MC seed/T/p, ensemble member hashes, clustering policy, corruption parameters, software versions, and source revision/dirty hash. Validate it on every cache load and resume. Use a declared allow-empty label manifest with strict parsing. Keep pickle restricted to trusted local artifacts; a typed record schema is still needed even if the storage format remains pickle.

**Acceptance check.** Changing any scientifically relevant input changes the experiment ID. Reordered pairs, stale constants, missing labels, mismatched classes, and a different checkpoint fail before metrics run. Label hashes at training start/end and caching/evaluation must agree.

### F15 — P1: historical backbone rankings mix experiments and overstate equivalence

**Evidence: documented.** [phase1-experimental-record:589](D:/project/docs/phase1-experimental-record.md:589) records 66 pilot rows merged with 27 main rows despite an unrecoverable different pilot split. Other sections document mixed batch sizes, software-version metric changes, missing provenance, interrupted runs, and a resolved directory collision. Those corrections are valuable, but a decision to pool the campaigns cannot make their conditions identical.

The later two-class stride-4 benchmark is a different experiment from the older class-filtered benchmark, and the IR ladder differs from the deployed ship-only P2-feature recipe. Ranking architectures under a common short training budget is reasonable; it estimates performance under **that budget and recipe**, not the best attainable performance of each architecture. Training stride and schedule can change rankings. Matching family scale letters or parameter counts does not isolate an architectural mechanism.

The [IR benchmark closure](D:/project/docs/ir-benchmark-closed-2026-09-01.md) makes a reasonable compute-priority decision. However, a non-significant ANOVA (reported p about 0.45) is not an equivalence test and does not prove that architecture cannot improve IR. Likewise, being within one observed seed SD is a selection heuristic, not statistical indistinguishability. The later documented P2-feature gains themselves caution against a universal “architecture is not a lever” conclusion.

**Suggestion.** Separate pilot and main campaigns, class sets, annotation releases, metric versions, and schedules in all tables. Treat the old pooled ranking as exploratory. Use a small confirmatory comparison of the selected model and the strongest relevant controls on the actual deployment recipe; do not automatically finish every historical ladder. State the practical accuracy/latency tradeoff used to select YOLO26m. If equivalence matters, define a tolerable delta and estimate an interval for it.

**Acceptance check.** Each table row resolves to one manifest and a compatible metric. A comparison can be reproduced without recovering undocumented pilot data. Report incomplete/failed cells and exclusions explicitly; do not describe the latest 93-cell campaign as completed based on an older campaign's count.

### F16 — P2: local null experiments are repeatedly promoted into universal limits

**Evidence: documented interpretations.** The idea logs contain useful screens, but some conclusions exceed the tested hypothesis:

|Inference in the record|What the experiment actually permits|
|-|-|
|A logistic/MLP gate is an upper bound on the features|It is one learned baseline with a chosen target, model class, regularization, and training protocol. It is not an upper bound.|
|A GT-informed coordinate-ascent oracle establishes the remaining ceiling|A local optimum under a restricted action space is not a global bound. It must at least include the baseline as an available solution. AP's global ranking also couples per-frame decisions.|
|Temporal support has lift near 1, so tracking cannot help|That feature, threshold, and dataset yielded little marginal signal. Motion, identity, conditional interactions, and other association rules were not all tested.|
|A fixed detection union's recall bounds every future improvement|It bounds re-ranking of those fixed candidate boxes under that matching rule. Better detection, localization, or registration changes the candidate set.|
|A single batch or resolution comparison isolates the corresponding cause|Different seeds, head initialization, shuffling, schedules, or transfer fractions can confound it. The P2-vs-P2-feature screen explicitly documents some of these differences.|
|An extra fine-tune checkpoint cannot hurt because it is optional|It enlarges the selection space; choosing it on repeatedly inspected validation data changes selection bias.|
|A predetermined AP floor is above single-seed noise|That requires measured variability for the relevant comparison. Preregistration makes a rule auditable, not statistically calibrated.|

The draw-averaged adoption rules also mix draw SD, bootstrap uncertainty, and fixed floors. A quadrature combination is not automatically a confidence interval: the quantities must target compatible variance components, and independence/overlap must be justified. A four-draw SD and the uncertainty in the four-draw mean are different quantities. Cellwise zero-tolerance guards can reject negligible changes while offering no formal non-inferiority guarantee.

**Suggestion.** Rename learned/oracle rows precisely and preserve the negative results as conditional findings. Specify the estimand first—mean change over scenes, corruption draws, and/or trained models—then choose its uncertainty estimator and practical margin. Use a separate development screen and a limited confirmatory test, rather than adding increasingly elaborate decision bands to reused data.

### F17 — P1 for publication: the broad novelty claim is false

**Evidence: externally verified.** The [URF progress report](D:/project/docs/reports/URF%20Progress%20Report%20-%20Laksh%20Saroha.docx) says existing visible/infrared fusion is static and never conditioned on live reliability. That claim is contradicted by [UA-CMDet's official repository and paper](https://github.com/SunYM2020/UA-CMDet): the 2022 work combines uncertainty-aware cross-modal learning with illumination-aware NMS at inference. There is also [Uncertainty-Aware Cross-Modality Fusion for Visible-Infrared Object Detection, DICTA 2024](https://doi.org/10.1109/DICTA63115.2024.00029).

Learned attention is not “static” merely because its parameters are fixed at inference; the attention values can depend on the current input. A single-pass Gaussian localization head is also established prior art, including [Gaussian YOLOv3](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.pdf). These facts do not establish that the project's precise maritime protocol or implementation has already been published.

**Suggestion.** Replace categorical novelty language with a comparison table: domain, sensor types, uncertainty target, inference-time adaptation, calibration evaluation, registration assumptions, and compute. The defensible contribution may be a carefully controlled maritime uncertainty study and a lightweight, interpretable sensor-selection baseline, including its failure cases. A stronger new-method claim needs an active mechanism and independent evidence beyond combining known components.

### F18 — P2: queue safety depends on operator discipline

**Evidence: source-confirmed.** [cmd\_run:783](D:/project/scripts/run_queue.py:783) loads state and writes its own PID without acquiring exclusive queue ownership. Two runners can both start on one directory. Different queue directories can also target the same GPU without a common device lease. The [6 September handoff](D:/project/docs/handoff-2026-09-06-redo31.md) records the operational cost of concurrent ownership.

[write\_json:105](D:/project/scripts/run_queue.py:105) uses a shared deterministic temporary filename and falls back to overwriting the destination in place. Tolerant readers make sense for cosmetic live telemetry; silently tolerating a torn **durable queue state** or failed checkpoint-selection write is a different risk. Multiple writers can lose updates. Failed/skipped runs are terminal by design, so a queue restart is not a retry policy for transient failures.

The latest resume fix carries best-epoch information from queue state while using checkpoint best fitness. This is an improvement, but also makes consistency between those two artifacts important; stale state should not be silently combined with a different checkpoint.

**Suggestion.** Acquire an exclusive OS-level queue lock before loading or changing durable state. Add a GPU/device lease if multiple queues share a device. Separate best-effort telemetry from durable run events/state; use transactional storage such as SQLite or atomic, uniquely named writes with recovery. Bind state to run/checkpoint IDs. Make retryable infrastructure errors distinct from scientific divergence and deliberate cancellation.

**Acceptance check.** A second runner fails before training; interrupted writes recover the last valid state; resumed best-epoch/fitness/checkpoint identities agree; failed downloads do not become permanent missing benchmark cells without an explicit status.

### F19 — P2: real-time performance and environment reproducibility are not yet established for the full system

**Evidence: source-confirmed and locally observed.** [bench/fps.py](D:/project/src/uqfusion/bench/fps.py) measures single-model `predict()` throughput. That is useful for backbone selection but does not establish two-stream system latency with Gaussian output, image statistics, feature extraction, homography, fusion, or MC/ensemble aggregation. “One pass” means one pass per detector; it does not imply that the complete system satisfies a target frame deadline. Mean FPS alone also hides tail latency and queueing.

The old `adopted` filter is explicitly noncausal: [hysteresis.py:97](D:/project/src/uqfusion/eval/hysteresis.py:97) centers a 15-observation window. Changing future observation 7 changes output at observation 0 in a reproduced fixture. At continuous 10 Hz that implies up to 0.7 s lookahead; on sparse paired frames the duration can differ. **The later cross-modal rule does not use this filter**, so this finding applies to the historical/default-adopted protocol, not every preset.

This checkout's `.venv` executable points to a missing former-user Python installation, and the configured `gpu\_python` path is also from that machine. The configured `Pohang\_dataset` and `data` directories are absent. These are concrete local reproducibility limitations, not evidence that the remote training environment is broken.

The pinned torch/torchvision pair is real and compatible according to [official PyTorch installation instructions](https://pytorch.org/get-started/previous-versions/). However, the operating system alone is not a sufficient wheel-build specification: CPU/CUDA indexes and runtime/driver versions should be recorded. The requested full `requirements.lock.txt` is not present in the normal repository inventory. Several UQ libraries in `requirements.txt` appear only in environment-smoke imports; core metrics/scorers are locally implemented.

**Suggestion.** Benchmark synchronized end-to-end pairs on the declared target device, with mean/p50/p95/p99 latency, peak memory, warm-up, batch size, input dimensions, and active options. Report detector-only and complete-system timings separately. Use a causal hold/release filter if streaming is required, or disclose buffering delay for centered filters. Create per-platform tested environment locks; keep optional unused research dependencies out of the minimal evaluation environment. Preserve the current tested model version before any upgrade.

**Acceptance check.** A fresh machine can install, load a versioned artifact, run a small real-data inference/evaluation fixture, and reproduce the declared metric. The streaming output for frame t cannot depend on frames after t unless the declared latency includes that buffering.

### F20 — P2: there is no single authoritative current architecture

**Evidence: source-confirmed and documented.** The original scope describes DFL plus Gaussian sigma, feature Mahalanobis weighting, and final per-box uncertainty. The August “finalized” architecture uses YOLO26s and centered veto filtering. The September architecture uses larger detectors, image-statistic gates, support scoring, and disabled learned soft UQ. The later night-restoration work changes the detector and challenges the veto again.

Meanwhile, [load\_context](D:/project/src/uqfusion/eval/ctx.py:251) defaults to the older `adopted` preset. The `crossmodal26m\_snms` comment calls soft-NMS adopted, while the [2 September experiment log:1012](D:/project/docs/experiment-log-2026-09-02.md:1012) says not to adopt it after the draw-averaged gate. [docs/eval/final\_system\_2026-09-01.md](D:/project/docs/eval/final_system_2026-09-01.md) actually freezes the August small-model system, as its text explains. Its filename alone is not enough to select the current result.

This is more than documentation clutter: callers can unknowingly evaluate different systems through defaults, inherited constants, and historical cache paths. Extensive in-place class conversion also relies on private Ultralytics internals and checkpoint serialization contracts; avoiding a vendored fork does not make that integration version-independent. The record's previous dropped-sigma weights, wrong MC inference branch, and layer-key migration failures demonstrate the need for behavioral compatibility checks.

**Suggestion.** Add one current architecture document generated or checked against an immutable system manifest. Give historical systems explicit dated IDs and require callers to choose one. Record active mechanisms, checkpoints, label release, class map, calibration release, preprocessing, corruption recipe, and evaluator version. Keep historical documents intact with clear supersession links. Move reusable evaluation logic out of one-off scripts into typed modules and exercise the existing meaningful smoke checks automatically on supported environments.

**Acceptance check.** Two commands claiming to evaluate the same system print the same manifest hash. Incompatible presets/checkpoints/constants fail early. Tests verify output branch execution, sigma-index alignment, save/load, resume, and disabled-option identity under the exact pinned Ultralytics version.

## 2\. Decision-by-decision assessment

This covers the explicit decision register in [progress.md:147](D:/project/progress.md:147), including amendments, and the later component choices. “Keep” means the choice is defensible for this project, not that its performance claim has been proved.

|Decision|Assessment|Suggested treatment|
|-|-|-|
|D1: add Gaussian branch alongside detector regression|Keep as an experiment; non-degradation and calibrated usefulness remain conditions, not established properties|Resolve F06–F08. On YOLO26, describe it as an added variance branch alongside DFL-free regression.|
|D2: cite evidential regression without implementing another baseline|Keep scope control|Cite both the original and its critique; do not assert a clean epistemic decomposition from a single evidential head.|
|D3: absolute system reliability and both-degraded case|Keep the question, revise the contract|An advisory novelty flag is not calibrated abstention. Define consumer behavior and costs (F12).|
|D4: small learned gate as upper bound|Revise|Call it a learned baseline. Match target and evaluation objective, with nested fitting (F16).|
|D5: preregistered constants and cached gate ablations|Keep caching; repair fitting discipline|A formula can still use test data, and repeated development selection remains selection (F01–F02, F14).|
|D6-rev: custom split, night permitted in training, mandatory audit|Keep conditional on stronger audit|Test all split boundaries, actual timestamps, and cross-modal consistency; state the narrow generalization target (F13).|
|D7: fuse in VIS coordinates using calibration|Keep the coordinate convention|Propagate uncertainty and measure residuals; separate VIS-reference AP from physical-object coverage (F10–F11).|
|D8: common API shortlist, seed replication, tie-breaking|Reasonable experiment design, heuristic tie rule|Preserve the original scope as historical; do not call one-SD ties equivalent (F15).|
|D9: defer DETR|Keep for bounded scope|No requirement to train every detector family. Narrow “detector-agnostic” to the implementations actually demonstrated.|
|D10: defer MIT onboarding|Acceptable for development, unresolved for external validation|No cross-dataset generalization claim until a documented independent annotation/evaluation release exists.|
|D11: decouple GPU benchmark from CPU engineering|Keep|Synthetic gates must not be treated as real-data validation; require a small real-data contract test.|
|D12: TensorBoard default, W\&B optional|Keep|Tracking is a view over immutable run artifacts, not their only source of truth.|
|D13: pinned software and promised full lock|Keep pinning, complete reproducibility|Lock transitive dependencies/platform builds; an unreadable moved venv is not a portable environment (F19).|
|D14: per-frame headline, smoothing ablated separately|Keep|Centered filters violate that headline convention. The later cross-modal path should be documented separately (F19).|
|D15: standard 100-epoch/patience-20 recipe, stride-2 training|Defensible budgeted protocol|Early stopping before mosaic closure changes the effective recipe. Validate the actual schedule; do not claim full annealing merely from a 100-epoch maximum.|
|D16: explicit data contract and Pohang-only Phase 1|Keep|Enforce the contract in code and bind results to content hashes, not only YAML paths (F13–F14).|
|D17: in-place Ultralytics conversion, detached features/mean|Keep with a narrow compatibility layer|Private API coupling remains. Remove bit-identity guarantee until full-step verification (F06, F20).|
|D18: zero-extra-training DFL uncertainty|Keep within DFL-capable models|It is unavailable on YOLO26's `reg\_max=1` path. Raw DFL spread still needs calibration evidence.|
|D19: retrained head-dropout, p=0.15, T=10|Reasonable fixed baseline|Keep deployed-branch execution checks; show small T-convergence and p-sensitivity studies on development data. Head-only dropout is a restricted approximation, not exact Bayesian uncertainty.|
|D20: common greedy MC/ensemble association and singleton rule|Useful shared interface, weak distribution model|Fix/justify mean and variance weighting, support semantics, and arbitrary singleton scale; report association sensitivity (F07).|
|D21: predeclared metric suite|Keep the range of questions, repair implementation and names|Official AP parity, accepted-set versus full-system risk, and TP-selection disclosure are required (F03–F04, F12).|
|D22: five ensemble seeds, member-0 OOD features|Acceptable compute choice, limited interpretation|One ensemble is one ensemble replicate. Member-0 features keep a common space but are not ensemble-level epistemic evidence.|
|D23: seeded synthetic corruptions, different final seed|Keep deterministic stress testing|New seeds alone do not establish scene/family generalization. Hash transforms and hold out scenes/families separately (F02).|
|D24: full size ladder with seed 0|Exploratory model selection only|Winner-only replication does not estimate uncertainty in the full ranking. Confirm a small relevant finalist set.|
|D25: YOLO26m and native-resolution follow-up|Sensible accuracy/compute candidate|Its selection is conditional on the historical benchmark limitations. Upsampling prepared 640 images cannot recover native VIS detail.|
|D26: Gaussian on one-to-one branch; explicit sigma gathering; unclamped targets|Keep; these are necessary compatibility corrections|Preserve branch/assignment/index tests. Verify the actual deployed branch for each pinned version rather than assuming all future YOLO26 defaults match it.|
|D27: high-IoU WBF, hard veto, dilate-15, soft-term choices|Historical baseline, superseded in part|Preserve its results with dated IDs. Noncausal filtering, low merge frequency, and later night-label changes constrain its claims.|
|D28: IR P2 features, ship-only, 640, no CLAHE/rect; prepared VIS|Reasonable tested recipe, not a universal optimum|P2 versus P2-feature initialization differs; IR class removal narrows coverage. Keep buoy and native-resolution limitations visible.|
|D29: share detector changes and gate across UQ arms|Keep; essential for attribution|Also share annotation release, training stage, deployment branch, scoring policy, and comparable predictive semantics (F07).|
|D30: explicit GPU interpreter with CUDA probe|Keep|Move machine paths into local profiles and validate them before launching; do not rely on a former user's installation (F19).|
|D31: score the saved best checkpoint|Keep as the deployment unit|Select on validation, score once on test. Best validation AP is not an unbiased performance estimate; epoch SD is not uncertainty over trained models.|

Additional choices not fully captured by D1–D31:

|Choice|Assessment and suggestion|
|-|-|
|Separate VIS and IR weights|Keep: the modalities have different input statistics, labels, and observability. Shared weights or early feature fusion are alternatives, not prerequisites for this review.|
|Three-channel expansion of grayscale IR|A practical way to reuse pretrained RGB models; it does not create three independent measurements. Test a different stem only if a concrete accuracy/compute question justifies it.|
|Min–max 16-to-8-bit IR conversion|Retain as a historical pixel contract; do not infer temperature or cross-frame radiometric consistency from it. Store raw values and mapping parameters for future comparisons.|
|Percentile mapping or CLAHE|Valid candidate preprocessing, with local negative results. Those experiments reject the tested recipe, not every contrast-preserving or radiometric alternative.|
|Prepared 640 VIS versus full-resolution VIS|The prepared image irreversibly removes small-object detail. A larger model input on that image is not a native-resolution experiment. Preserve both provenance and geometry.|
|Rectangular training|Its shuffle/augmentation changes confound a pure “remove padding” claim. Keep it off if it loses empirically, but describe the whole recipe change.|
|P2 detection level versus P2 feature injection|The record documents different head widths and transferred weights. Interpret the comparison as complete recipes; a causal P2-head conclusion needs matched initialization/capacity controls.|
|IR ship-only head|Reasonable given the reported near-zero buoy performance; not evidence that thermal buoy detection is physically impossible. The full system still needs buoy coverage accounting.|
|Gaussian log-variance clamps, width, and warm-up/ramp|Numerically practical choices. Report saturation frequency and sensitivity; stability/non-degenerate sigma alone is not calibration.|
|Pooled-feature Mahalanobis with Ledoit–Wolf covariance|Keep as a cheap novelty baseline. Shrinkage stabilizes estimation; it cannot make training-distribution membership equivalent to sensor health.|
|Multiplicative/min/geometric reliability combination|All are heuristic aggregations unless calibrated to a target. Fit-and-ablate fairly; none supplies a probabilistic interpretation by its range alone.|
|Capability prior from clean AP, IR downscale|Treat as an empirical ranking prior, not a probability or sensor characteristic. It depends on class mix, reference plane, annotations, and detector version. Fit within development folds.|
|Merge IoU 0.85 and support IoU 0.30/gamma 0.5|A defensible attempt to use agreement without moving poorly aligned boxes. It is mostly selection/re-ranking when merges are rare. Freeze only after independent validation.|
|IR deduplication/NMS around IoU 0.7|Valid postprocessing option; distinguish within-stream suppression from cross-modal fusion. Preserve a raw-IR control and disclose that the nominal NMS-free detector now has added NMS.|
|Per-class veto exceptions|Sensible to evaluate because only VIS supplies buoys. If rejected on actual evidence, keep the rejected result; still report the cost of no buoy-producing stream in vetoed cells.|
|Soft-NMS|Keep as an experimental option with an explicit historical status. Latest documented draw-averaged result says not adopted; code comments should agree.|
|Temporal support, TTA, one-to-many output, checkpoint averaging|Reasonable bounded screens. TTA/heads/checkpoints are correlated; do not price them as independent sensors or generalize a local null to every temporal method.|
|Day/night-sliced UQ and restored-label retraining|Keep; these expose important confounds. Compare all arms under one frozen recipe and label release before interpreting differences.|
|“Cold scratch” initialization|The documented new runs start from COCO weights. Call them cold starts from COCO; “scratch” conventionally suggests random initialization.|
|Six synthetic corruption families/severity ladders|Useful controlled perturbations. Do not equate their equal-weight mean or cell count with real maritime operating frequencies or physical sensor failure coverage.|
|Conformal prediction as a stretch goal|Defer until calibration/test isolation is fixed. Ordinary exchangeability-based coverage guarantees do not automatically survive correlated video and arbitrary domain shift.|
|More model families or an end-to-end fusion rewrite|Not presently the highest-value work. First establish a trustworthy comparison; architecture complexity cannot repair a reused test set or mislabeled target.|

## 3\. Specific factual claims to correct or qualify

These are distinct from whether an engineering choice is useful.

|Claim|Fact check|Suggested wording/action|
|-|-|-|
|“A product lets a confident OOD score dilute a brightness alarm” in [reliability.py](D:/project/src/uqfusion/uq/reliability.py)|False for reliabilities in \[0,1]: `a\*b <= min(a,b)`. Example: 0.2×0.8=0.16, below 0.2.|Min may avoid additional suppression from two imperfect signals; justify it empirically, without the reversed algebra.|
|A failed stream's mere presence always halves the survivor's WBF scores|False for arbitrary model weights. The exact rescaling depends on the weights, member counts, and implementation.|Cite the actual formula and tested configuration.|
|Every retained low-score false positive necessarily hurts AP|False. After all GT have been recalled, an FP tail can leave AP unchanged; reproduced at AP=1.|Discuss effects on interleaved ranking, recall, operational thresholds, and precision, rather than asserting inevitability.|
|`score \* weight / sigma²` is automatically the minimum-variance estimator|Overstated. Inverse covariance weighting needs unbiased common-target measurements and a dependence model; extra confidence factors need their own justification.|Call this a heuristic precision-weighted fusion unless its statistical model is supplied.|
|Median offsets tolerate a majority of wrong pairs|False in general; a majority can move the median arbitrarily within the permitted search region.|Require sufficient inlier consensus and validate adversarial correspondences.|
|A small feature Mahalanobis distance means the sensor is healthy|False as a general implication; in-distribution sensor failure can be common in training data.|“Feature-space novelty relative to the fitted reference.” The [original method](https://arxiv.org/abs/1807.03888) is not an epistemic posterior certificate.|
|“IR sees through glare,” or a reflected target has no thermal signal|Too broad. Thermal imaging includes emitted and reflected radiation; emissivity, viewing geometry, atmospheric transmission, and sensor behavior matter.|Limit claims to measured conditions. [FLIR's measurement guidance](https://www.flir.com/discover/professional-tools/how-does-emissivity-affect-thermal-imaging/) explicitly includes reflected radiation.|
|Thermal crossover is simply equal physical temperature|Incomplete: detected contrast concerns apparent radiance, emissivity, reflected/background radiation, atmosphere, and sensor processing.|Use radiometric evidence or an explicitly defined image-contrast failure proxy. Do not derive physical temperatures from per-frame normalized values.|
|Roughly 7.7 “effective bits” proves 8-bit conversion loses essentially nothing|Overstated. The IR handoff defines this as `log2(span / noise floor)`, not image entropy or a direct task-preservation measurement. It supports investigating an 8-bit representation, but frame extrema, low-contrast targets, and noise variation still matter. The raw measurement was not rerun here.|Report quantization error and target contrast/SNR against the raw source across conditions. Preserve the stated definition; do not relabel it entropy or a measured ADC resolution.|
|Synthetic lowlight is a physically impossible sensor output|Too strong. It is an incomplete image-space model of low light, not proof that no camera/postprocessing pipeline can yield such values.|Disclose absent photon/read-noise/exposure modeling and validate against real low-light data.|
|Gini/structure statistics are illumination invariant|Only under the assumptions of the exact transform/statistic; gain invariance need not survive clipping, quantization, offsets, or noise.|State the invariance mathematically and test practical perturbations.|
|Healthy/failed sensors are identified at threshold 0.5|A midpoint of a hand-shaped score is not a calibrated operating point.|Fit thresholds to a declared error/cost target on independent calibration data.|
|A true zombie has zero accumulated CPU time and can be killed to release CUDA|Incorrect as a general Linux account. A zombie has terminated but retains status/resource accounting; it cannot execute or be killed again. Live threads/children, namespaces, and driver state complicate diagnosis.|Correct the [6 September handoff](D:/project/docs/handoff-2026-09-06-redo31.md). Use process/thread and parent-reaping evidence; see [Linux wait documentation](https://man7.org/linux/man-pages/man2/waitpid.2.html). This review does not determine the remote incident's exact cause.|
|A single same-seed cross-machine pair identifies a hardware AP offset|Unsupported. Hardware, library versions, data order, and numerical paths may differ; one pair cannot estimate their separate effects or variance.|Report the observed paired difference with its confounds. Do not subtract it from unrelated runs as a universal correction.|
|All Gaussian/MC/ensemble gates are green, therefore all components work|Historical and overly broad. Later documents record failures that earlier aggregate smokes missed.|Name each tested contract, software revision, and result; list unsupported real-data behavior separately.|

### External references and dataset premises

The main cited methods are real; the larger problem is how their guarantees are being transferred to this implementation. Selected checks against primary sources:

|Reference/premise|Verified result and limit|
|-|-|
|Gaussian YOLOv3, ICCV 2019|Existing Gaussian localization-uncertainty detector; supports the design precedent, not novelty of adding variance. [Paper](https://openaccess.thecvf.com/content_ICCV_2019/papers/Choi_Gaussian_YOLOv3_An_Accurate_and_Fast_Object_Detector_Using_Localization_ICCV_2019_paper.pdf).|
|GFLv2, CVPR 2021|Uses distribution information for localization-quality estimation; raw DFL variance still needs its own calibration evaluation. [Paper](https://openaccess.thecvf.com/content/CVPR2021/html/Li_Generalized_Focal_Loss_V2_Learning_Reliable_Localization_Quality_Estimation_for_CVPR_2021_paper.html).|
|BayesOD|Directly relevant prior art on probabilistic detection and information loss in NMS. [Author paper](https://arxiv.org/abs/1903.03838).|
|MC Dropout, ICML 2016|Approximate Bayesian interpretation; not an exact posterior guarantee for this chosen head-only dropout configuration. [Paper](https://proceedings.mlr.press/v48/gal16.html).|
|Deep Ensembles, NeurIPS 2017|Probabilistic regression aggregation includes within-member uncertainty. [Paper](https://arxiv.org/pdf/1612.01474).|
|Deep Evidential Regression / critique|Original NeurIPS 2020 method and AAAI 2023 critique exist; citing rather than benchmarking is reasonable scope control. [Original](https://proceedings.neurips.cc/paper/2020/hash/aab085461de182608ee9f607f3f7d18f-Abstract.html), [critique](https://ojs.aaai.org/index.php/AAAI/article/view/26096).|
|Packed-Ensembles|ICLR 2023 attribution checks out. It is an efficient-ensemble alternative, not implemented by this project's five independent detectors. [Proceedings paper](https://openreview.net/pdf?id=XXTyv1zD9zD).|
|Variance-network training / beta-NLL|NeurIPS 2019 and ICLR 2022 references support investigating optimization/calibration failure modes, not automatic calibration. [Variance networks](https://proceedings.neurips.cc/paper_files/paper/2019/hash/07211688a0869d995947a8fb11b215d6-Abstract.html), [beta-NLL](https://arxiv.org/abs/2203.09168).|
|Mahalanobis / relative Mahalanobis|Relevant OOD methods; neither proves an arbitrary pooled-feature score detects all sensor failures. [Original](https://arxiv.org/abs/1807.03888), [relative distance](https://arxiv.org/abs/2106.09022).|
|WBF, 2021|Author implementation and publication exist; its confidence-weighted box aggregation is not a general probabilistic sensor-fusion theorem. [Author repository](https://github.com/zfturbo/weighted-boxes-fusion).|
|Confidence calibration, ICML 2017|Calibration concerns predicted probabilities versus observed correctness; correlation alone is insufficient. [Guo et al.](https://proceedings.mlr.press/v70/guo17a).|
|Conformal introduction, 2021|Correct reference; use its assumptions and an independent calibration split before claiming coverage guarantees. [Author paper](https://arxiv.org/abs/2107.07511).|
|RT-DETR / D-FINE|RT-DETR is CVPR 2024. Update the D-FINE tracker from an arXiv-only label to ICLR 2025. [RT-DETR](https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html), [D-FINE](https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html).|
|YOLO26|DFL-free regression and optional one-to-one NMS-free inference are supported. Current upstream docs distinguish inference branches; rely on the pinned local implementation for historical behavior. [Official docs](https://docs.ultralytics.com/models/yolo26/).|
|Pohang / PoLaRIS|The 2023 sensor dataset and later annotation dataset are distinct releases. PoLaRIS has a 2024 preprint and an ICRA 2025 official repository; update the undated tracker. [Pohang](https://arxiv.org/abs/2303.05555), [PoLaRIS](https://arxiv.org/abs/2412.06192), [author code](https://github.com/sparolab/PoLaRIS). Local filtered counts are not verified merely by citing the source dataset.|
|MIT Marine Perception|The source confirms visible 12 fps, IR 30 fps, and CC BY-NC-SA 4.0 for the described collection. Different rates require explicit pairing. This does not verify the project's claimed manual annotation subset or a ready-to-use aligned detection benchmark. [MIT source](https://seagrant.mit.edu/auvlab-datasets-marine-perception-2-3/).|
|SMD|The source establishes a maritime dataset with separate visible/IR material. Do not assume the mere presence of both modalities provides synchronized, calibrated pairs for this fusion pipeline. [Author dataset page](https://sites.google.com/site/dilipprasad/home/singapore-maritime-dataset).|
|MassMIND|Approximately 2,900 LWIR segmented images and seven categories are supported. Those categories are not the same ship/buoy taxonomy; deriving boxes requires a documented instance/class mapping. [Dataset paper](https://journals.sagepub.com/doi/10.1177/02783649231153020).|

The original progress report also describes a 23-variant/68-run state and a DFL-based shortlist. It should remain an explicitly dated historical report. It cannot serve as the current method/results description after the 31-model campaigns, YOLO26 migration, MC repairs, and night-label restoration.

## 4\. Suggested order of work

The following is a proposed repair sequence; this review did not execute it.

|Order|Work|Evidence needed to finish|
|-|-|-|
|1|Freeze existing source, labels, checkpoints, pair lists, caches, and metric version into manifests; identify every dataset already used for selection|A readable registry of historical versus current artifacts; no ambiguously named “final” result|
|2|Repair official-AP parity, bootstrap equivalence, strict GT loading, and split/pair validation|Passing adversarial fixtures and a small real-data parity run|
|3|Re-score existing cached predictions with the corrected evaluator and dependence-aware intervals|A change-impact table: which previously claimed gains, rankings, and adoption decisions survive|
|4|Choose independent calibration and final evaluation data; refit gate constants only on calibration|No test contribution to any parameter or checkpoint selection; a measured out-of-sample clean false-alarm rate|
|5|Finish annotation QA and the comparable restored-label UQ experiment already described in the latest handoff|Identical label/recipe manifests across arms; full-step parity or a declared non-inferiority test; comparable UQ definitions|
|6|Test the scientific mechanism with fixed predictions, including constant/shuffled uncertainty controls and simple sensor-selection baselines|Independent evidence that the claimed uncertainty actually changes and improves the intended decision|
|7|Add full-system timing and an independent recording/dataset evaluation|A latency/accuracy/calibration tradeoff with measured limits, not a “single-pass” proxy|
|8|Rewrite the current architecture and manuscript-facing claims from those artifacts|One accurate method description, current bibliography, and explicit historical limitations|

Minimal useful baselines for the repaired fusion comparison are VIS-only, mapped IR-only, a clearly defined uncertainty-blind aggregate, a simple image-statistic selector, the current support-scoring system, and any proposed active-UQ variant. Hold detector predictions and class/geometry policies fixed wherever the goal is to isolate the decision layer. Adding more backbone families is secondary to making these comparisons trustworthy.

## 5\. Verification performed and limits

I inventoried the 53 pre-existing files under [D:\\project\\docs](D:/project/docs), including both the Word and PDF progress reports, and extracted their text. The review used the architecture documents, decision register, code, preregistrations, correction histories, and decision-bearing sections of the larger experiment/operational logs. The inventory records each document's size and SHA-256 hash. Historical tables were treated as reported evidence, with later corrections taking precedence; their underlying GPU experiments were not all repeated.

I traced the principal data, training, inference, uncertainty, registration, fusion, evaluation, caching, and queue paths. The broad decision matrices above assess the choices identified in that material; this is not a claim of formal verification of every possible behavior of every script or third-party library.

The numerical counterexamples were executed with the bundled Python/NumPy runtime against project functions. Pure-Python YAML was loaded from the local package tree where needed. The original environment could not run as installed, and the configured dataset is absent, so full-data AP changes, GPU training parity, current remote queue health, and all historical dataset statistics remain unverified in this review.

|Local check|Observed result|
|-|-|
|AP interpolation versus official COCO recall lookup|Differences of -0.00330033 and -0.00495050 on two small imperfect rankings|
|Fast versus literal resample, a class absent|0.5 versus 1.0 macro AP|
|Future-only veto change|Future observation 7 changes output 0 under centered dilate-15|
|Identical wrong ensemble members|Zero sigma, NaN NLL, zero coverage at all tested nominal intervals|
|Homography H versus -H|Different output coordinates despite equivalent projective maps|
|Global gradient clipping calculation|Detector gradient 6 becomes 0.598923 when an independent sigma gradient of 100 shares norm-10 clipping|
|Validation/test temporal separation|Adjacent val/test fixture passes the existing audit|
|Missing required test split|Existing audit returns PASS with `missing\_splits=\[test]`|
|Label content changes|Split fingerprint remains `896841440b53`|
|Missing ground-truth path|Loader returns empty GT|
|False positives after complete recall|Official recall-lookup AP remains 1.0|
|Support score domain|Valid input scores can yield an output score of 1.47015|

Reproduction artifacts, stored under the ignored review scratch directory:

* [Numerical counterexample script](D:/project/runs/architecture_review_2026-09-08/reproduce_findings.py)
* [Machine-readable results](D:/project/runs/architecture_review_2026-09-08/evidence.json)
* [Document inventory and hashes](D:/project/runs/architecture_review_2026-09-08/document_inventory.json)
* [Document extraction script](D:/project/runs/architecture_review_2026-09-08/inspect_documents.py)

On this machine the counterexamples can be repeated with:

```powershell
\& 'C:/Users/shiva/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe' `
  'D:/project/runs/architecture\_review\_2026-09-08/reproduce\_findings.py'
```

The assertions in that script intentionally demonstrate the current defects; they are evidence fixtures, not acceptance tests for the repaired implementation. The clipping example is an analytical optimizer counterexample, not a Torch training trace. The sigma-transform example states the necessary scale transformation and is supported by source inspection; it does not execute full covariance propagation. No remote experiment results were fabricated or assumed current.

