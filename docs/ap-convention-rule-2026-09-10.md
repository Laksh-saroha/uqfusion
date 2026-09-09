# Standing rule: AP numbers carry their convention, and never cross it

**Written 2026-09-10 for R-A1 (review finding F03).** This is a rule, not a result. It
exists because "mAP@50-95" names at least three different computations in this
repository's history and the differences are not all small.

---

## 1. The rule

**Never compare an absolute AP produced under one convention against an absolute AP
produced under another.** Every claim this project makes must be a *paired delta*
scored under a single convention, or an absolute number that names its convention in
the same sentence.

Concretely, the three conventions in play:

| convention | integration rule | where |
|---|---|---|
| `local-linear-interp` | `np.interp` of the precision envelope onto a 101-point recall grid | `matching.local_ap50_95` (alias `map50_95`) — **everything this project reports** |
| COCO / pycocotools | `searchsorted` — the envelope at the *first attained* recall | `eval.cocoparity.coco_ap`, for comparison only |
| Ultralytics | its own rule again, and version-dependent | Phase 1 tables only |

## 2. Why the rule is worth more than picking a winner

**The gap that matters is small, and it was measured, not assumed.** On this project's
own caches the worst local-vs-COCO disagreement in a **delta** is **0.00028501** —
5× below the 0.0014–0.0031 paired 2σ noise floor
(`docs/eval/ap_convention_parity_2026-09-09.md`). Both arms of every comparison are
scored the same way, so a systematic offset subtracts out. Paired deltas are safe.

**The gap in an absolute number is not small.** The hand-built cases in
`scripts/smoke_cocoparity.py` differ by −0.0033 and −0.0050, above the noise floor. An
absolute AP is convention-bound; a delta is much less so. That asymmetry *is* the rule.

**The Ultralytics gap is two orders larger.** `docs/phase1-experimental-record.md`
records ~0.034 mAP between ultralytics *versions* on identical weights. Phase 1
ultralytics numbers must never be set beside custom fusion AP in a table, under any
convention, for any purpose. This is the one that can actually produce a wrong
conclusion.

## 3. Why the local convention was kept

Three reasons, and the third is the one that decided it.

1. `map50_95` has **322 call sites across 70 files** plus 12 documents. Replacing it
   moves every recorded number in the project for a 0.00029 delta effect.
2. A "COCO companion column" on every table would agree to 5× below the noise floor
   every time, which trains readers to skip it.
3. **It would not actually be COCO.** `cocoparity.TASK_CONFIG` pins
   `max_dets: None` — max-per-image, not COCO's 100 — because a truncating cap would
   silently change what is being measured. A column labelled "COCO" would therefore be
   a **mislabel of exactly the kind this review item is about**: see
   `docs/exposure-ledger-2026-09-09.md` §6, where `preset="crossmodal"` named three
   different systems. Fixing a naming defect by adding a second naming defect is not a
   fix.

So the convention is **declared and recorded** instead of replaced.

## 4. What changed in code

* `apmetrics.AP_CONVENTION = "local-linear-interp"` — the third declared policy
  constant, beside `MISSING_CLASS_POLICY` and `SORT_KIND`.
* `apmetrics.declared_policies()` returns all three for stamping into a result's
  config block. Wired into `scripts/eval_final_system.py`; **not yet wired into the
  other result writers** — see §5.
* `matching.local_ap50_95` is the honest name. `matching.map50_95` remains as a
  working alias and the returned dict keys are unchanged, so no call site and no
  recorded number moves. Its docstring no longer claims to be "COCO-style", which it
  had said since the function was written.
* `cocoparity.PARITY_BOUND = 0.00028501` and `PARITY_TOLERANCE = 1.5` pin the measured
  gap. `scripts/ap_convention_parity.py` now **fails** if the real-cache gap exceeds
  0.00042751, and `smoke_cocoparity.py` case 9 asserts on every run that the trip point
  stays under the noise floor and that `max_dets` is still `None`.
* Separately (R-E1, same root cause): `FusionContext` now records `ir_nms` and
  `cap_ir_scale`, which were local variables applied and discarded, and
  `eval_final_system.py` stamps them into the config block. Verified that
  `preset="crossmodal"` still reproduces `cap_ir = 0.0023539426113108287` bit-for-bit.

## 5. What this does not fix

**The other result writers still stamp nothing.** `declared_policies()` is wired into
one script. Roughly forty others write result JSON with a hand-built config block, and
each is a place a future reader will have to infer the convention from whatever code
was checked out. That is the general form of R-E1 and it is not done.

**The bound is pinned to specific caches.** `PARITY_BOUND` was measured on the three
VIS UQ arms. A new arm or a re-trained detector can legitimately move it. The trip
point has 1.5× headroom for that; if it fires, **re-measure and re-argue** that the gap
still sits below the noise floor, in a commit that says so. Raising the bound to make
the assertion pass would defeat the entire point.

**Decisions below the noise floor are still unsound.** soft-NMS was rejected at
−1.03e-5 (`project-snms-not-adopted`); the convention gap alone is ~20× that margin.
This rule adds a second independent reason those decisions carry no information. It
does not rescue any of them.
