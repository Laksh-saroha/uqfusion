# Re-baselining onto a night-capable VIS detector — a costed proposal

**Written 2026-09-04.** This is the document `docs/prereg-night-veto.md` says an ADOPT verdict
would license, and it is deliberately *only* a proposal: it prices the change and names what
breaks. It authorises nothing and nothing here has been executed. Written before the veto
verdict could be argued about, because the verdict turned out not to be the thing that matters.

---

## 1. What the veto run actually found

`runs/eval/night_veto_verdict.md`, primary arm, cold-started `gauss_vis_nightfull`, 4 paired
draws, preset `crossmodal26m`:

| night cell | veto ON | veto OFF | delta | band |
|---|---:|---:|---:|---|
| clean VIS, clean IR | 0.0850 | 0.2635 | **+0.1785** | ADOPT |
| clean VIS, IR glare | 0.0585 | 0.2522 | +0.1937 | ADOPT |
| clean VIS, IR blur | 0.0328 | 0.2374 | +0.2046 | ADOPT |
| clean VIS, IR noise | 0.0055 | 0.2255 | +0.2200 | ADOPT |
| clean VIS, IR fog | 0.0001 | 0.2256 | +0.2255 | ADOPT |
| VIS blur, clean IR | 0.0850 | 0.0534 | −0.0316 | REJECT |
| VIS rain, clean IR | 0.0850 | 0.0554 | −0.0296 | REJECT |
| VIS fog, clean IR | 0.0850 | 0.0112 | **−0.0737** | REJECT |
| VIS lowlight, IR glare | 0.0585 | 0.0227 | −0.0358 | REJECT |
| VIS blur, IR glare | 0.0529 | 0.0426 | −0.0102 | REJECT |

The registered verdict is **INCONCLUSIVE** — guard 2 fired, two IR-corrupted night cells fall
below their own `ir_only`, and the prereg caps the verdict there. The veto stands. Correct
under the rule, and not being reopened.

**But the table says something the bands cannot express.** Every cell where VIS is *healthy*
says ADOPT by 33× the floor. Every cell where VIS is *degraded* says REJECT. The night veto is
neither wrong nor right — it is asking the wrong question. It vetoes on **darkness**, and what
actually predicts whether VIS should be dropped is **VIS health**. Under the crippled detector
those were the same variable, because a detector trained on empty night labels is degraded
exactly when it is dark. They are now different variables, and the rule has not caught up.

That is the finding, and the reason this document exists: the +0.1785 is not a fusion result,
it is a *detector* result wearing fusion's clothes.

### 1.1 The gain is concatenation, and the report says so

Partner rate at `iou_thr` 0.85 on the clean night cell: **0.143%** of VIS boxes have an IR
partner. With the veto on, night fused *is* `ir_only` (0.0850 both columns, exactly). With it
off, fused ≈ VIS. Nothing merged; a second box list was appended. Any write-up presenting
+0.1785 as evidence for decision-level fusion would be misreading its own number.

---

## 2. What a re-baseline would cost

Measured on this machine this week, not estimated.

### 2.1 Cheaper than the record implies — most of the gate is detector-free

The record has said the swap moves "all 17 VIS caches and the 4,000-frame Mahalanobis
reference, whose features move with the model". Checked directly:

* `fit_structure_gate.py` reads **only** `runs/derived/structure/*.json` and
  `runs/derived/brightness/*.json`. Every axis it fits — `grad_gini`, `ir_p05`,
  `lap_over_var`, `vis_health` — is an image statistic. `structure_constants.json` **does not
  move under a detector swap.**
* Under `crossmodal26m`, `load_context` sets `mu_d=1e9, lam=0`. The Mahalanobis soft term is
  **inert**, so the 4,000-frame reference is rebuilt for consistency (117 s) but enters no
  shipped decision.
* The **capability prior is recomputed from the caches** at every `load_context`, so it follows
  the new checkpoint with no refit step.

What genuinely does move: `mu_b`/`tau_b`, because the two-of-two night vote reads
`dark = brightness < c_vis.mu_b`, and `fit_brightness_gate.py` fits it against a VIS corruption
ladder (19 caches × 744 frames).

### 2.2 The bill

| item | measured basis | cost |
|---|---|---|
| VIS paired grid, one cache dir (8 caches) | 2 uncorrupted 173 s + 5 corrupted 916 s | **~18 min** |
| VIS corruption ladder, 19 caches × 744 frames | scaled from the same rates | **~16 min** |
| 4 draw dirs for any draw-averaged decision | measured end to end this session | **64 min** |
| σ-head UQ arm re-cache | one paired build | **~5 min** |
| MC-dropout VIS arm re-cache | T passes over the paired list | **~30 min** |
| refit `mu_b`/`tau_b` | CPU | **minutes** |
| **subtotal, all caching and refitting** | | **~2.5 h on the 4080** |
| **VIS ensemble, 4 seeds, cold start on restored labels** | 9.80 h measured for one cold start | **~40 h** |

**The whole re-baseline is one line item.** Everything except the ensemble is an afternoon. The
ensemble is 40 hours and the only thing worth arguing about.

### 2.3 The ensemble seeds that just finished ARE void — verified 2026-09-04

Checked on dgxanode01 rather than assumed. Two facts settle it:

* **The server's labels are restored and match this laptop exactly.**
  `/workspace/_label_stage/verify_night_labels.py` on
  `/workspace/pohang/visible/labels/pohang01`: **18,826 files, 96,584 boxes, hash
  `43ee6078395c` — PASS.** Directory mtimes put the restore at **2026-09-03 17:00 UTC**.
* **All four ensemble seeds started before that.** `args.yaml` timestamps: `seed1` 2026-08-25
  22:58, `seed2` 2026-08-26 13:12, `seed3` 2026-08-27 05:35, `seed4` 2026-08-28 00:10 — eight
  to ten days ahead of the restore.

So `ens_vis_seed{1..4}_ft` are trained on the **night-cut** labels. They encode "night is
empty", precisely the contamination `docs/prereg-uq-day-night-slice.md` Stage B exists to test
for. **They cannot be used as the Stage B arm.** `seed4` is the worst case, not the best: it
was still running when the labels changed underneath it and only finished 2026-09-04, so it may
straddle both trees within one run.

**The 40 hours in §2.2 is therefore real, not contingent** — there is no night-capable VIS
ensemble anywhere, and the one that exists was spent on labels now known to be wrong.

**One thing this clears:** the ep25 benchmark now running started at **2026-09-03 18:06 UTC**,
an hour after the restore, so all 93 runs are on the correct labels. That campaign does not
need restarting.

### 2.4 A lead on OQ-13, from looking at the right machine

`/workspace` on dgxanode01 keeps automatic filesystem snapshots (`/workspace/.snapshot/daily.*`,
`hourly.*`). That is the server, not the laptop, so it does not explain the 2026-09-03 21:19
rewrite — but it prompted the question nobody had asked: is `A:` snapshotted too?

Checked: `A:` is a **local NTFS fixed disk** (DriveType 3, 488 GB), not a network or synced
volume, and both **VSS and File History are Stopped/Manual**. So no scheduled snapshot service
could have rolled the directory back on its own. This does not fully exclude a one-off manual
`vssadmin`-style revert — confirming that needs admin queries — but it removes the most likely
mechanical explanation. OQ-13 stays open, one hypothesis narrower.

---

## 3. What a re-baseline would break, honestly

1. **Every headline number reproduces from `runs/full_scale/gauss_vis_seed0/`.** Swapping the
   checkpoint re-baselines the fusion tables, the UQ table, the gate ablations and the constants
   re-pricing. None are wrong today; they become answers to a question about a different
   detector.
2. **The Phase 1 ranking is not affected.** It ranks *architectures* at stride 4 and does not
   depend on which yolo26m checkpoint is called "the VIS detector". The 93-run campaign running
   on dgxanode01 does not need restarting.
3. **The night veto would have to be re-registered, not merely removed.** §1 shows the correct
   rule is conditional on VIS health, and `vis_health` is already fitted, already
   detector-independent, and already loaded by `crossmodal26m` as `q_vis_by_cond`. The V2
   question is "veto when IR says night AND VIS is unhealthy" — a *new* pre-registration with
   its own bands, not an amendment to V1 and not a re-run of it.
4. **One seed.** `gauss_vis_nightfull` is seed 0, stopped at epoch 11 of a 100-epoch schedule,
   so it is not a fully annealed model. Re-baselining onto a single early-stopped cold start
   would inherit that.

---

## 4. Recommendation

**Do not swap the detector yet, and do not spend the 40 hours yet.** In order:

1. ~~Verify the server's label tree~~ **done — §2.3.** Server labels PASS; the ensemble seeds
   predate the restore and are void; the ep25 benchmark is clean.
2. **Write the V2 veto pre-registration** — veto conditional on VIS health rather than
   darkness. The instrument exists and costs nothing new to evaluate; the registration is the
   work. Highest-value next document.
3. **Re-cache the σ-head and MC arms** onto `gauss_vis_nightfull` (~35 min) and re-run the
   day/night slice. Stage A currently returns VIS CLEAN for a reason that is itself an artefact
   — the VIS arms emit 2 detections across 1,032 night frames — and that reason disappears
   under a night-capable detector. **Cheapest way to find out whether the UQ table was ever
   contaminated.**
4. **The ensemble decision is now a clean question:** there is no night-capable VIS ensemble and
   building one costs ~40 h. Worth it only if the ensemble arm is load-bearing for a claim in
   the paper. If it is only a comparison row, say so and drop it rather than spending the
   compute.

The thing to resist is treating +0.1785 as a fusion win. It measures how much was lost when
94,553 boxes were deleted, recovered through the only channel left open once the veto was
removed.
