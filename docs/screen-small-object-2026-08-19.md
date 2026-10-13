# Small-object screen — what actually improves IR detection on `yolo26s`

**Written 2026-08-19.** A two-stage screening campaign, 12 runs, 19.6 h of laptop GPU, asking one
question: which cheap lever raises IR mAP on the stride-2 tree?

Companion docs: [`handoff-2026-08-19.md`](handoff-2026-08-19.md) (the anchor — this doc **confirms
its §4.5 claim that imgsz is a null IR lever** and supplies the mechanism),
[`ir-handoff-2026-08.md`](ir-handoff-2026-08.md) (the IR trees and resolutions),
[`phase2-laptop-2026-08-18.md`](phase2-laptop-2026-08-18.md) (the Phase 2 baseline).

Status labels: **VERIFIED** = measured this session · **OPEN** = not measured.
Run dirs: `runs/screen1/<id>/`, `runs/screen2/<id>/`. Queue state:
`runs/queue_screen{,2}/{queue.json,state.json,control.json}`.

---

## 1. Verdict — **VERIFIED**

**Adopt `p2feat` at 25 epochs. Do not extend past 25 epochs. `imgsz 960` is unresolved and
probably unnecessary — one 75-minute run would settle it (§8 item 1).**

`p2feat` is a low-level feature-injection variant (`configs/models/yolo26s-p2feat.yaml`): the
backbone's stride-4 P2 map is downsampled once and concatenated into the P3 neck, so the existing
head sees finer detail. It adds **no detection level and no anchors** — only richer features at the
levels already there.

At 25 epochs it is worth **+9% ship AP**, measured twice in two different corners of the grid.
Everything else in the screen is noise, a confound, or a loss.

**The qualifier, and it is real:** at 640 the gain is *convergence speed, not a ceiling*.
`p2feat` @ 640 @ 25 ep reaches 0.14243 ship AP; the stock model @ 640 @ 50 ep reaches 0.14232 —
the same number for twice the compute. Push `p2feat` to 50 epochs and it decays to 0.13424,
**below** the stock model on the same schedule (§4.2).

The single configuration clearing every baseline at every schedule tested is **`p2feat` @ 960 @
batch 10 @ 25 ep, 0.15440 ship AP** — +8.5% over the best baseline anywhere. Whether the 960 is
load-bearing is **OPEN**, and §4.2 gives a concrete reason to think it is not.

## 2. The grid — **VERIFIED**

Per-class AP from a fresh val pass on each `best.pt` (`scripts/perclass_ap.py`), on
`runs/derived/data_ir_stride2.yaml`, seed 0 throughout, `lr0` 0.01, `close_mosaic` 10:

| model | batch | imgsz | epochs | ship AP | buoy AP | mAP50-95 |
|---|---:|---:|---:|---:|---:|---:|
| base | 16 | 640 | 25 | 0.13022 | 0.00514 | 0.06768 |
| base | 16 | 640 | 50 | 0.14232 | 0.00047 | 0.07139 |
| base | 10 | 640 | 25 | 0.13938 | 0.00044 | 0.06991 |
| base | 10 | 960 | 25 | 0.14167 | 0.00013 | 0.07090 |
| base | 10 | 960 | 50 | 0.13745 | 0.00080 | 0.06913 |
| p2feat | 16 | 640 | 25 | 0.14243 | 0.00042 | 0.07142 |
| p2feat | 16 | 640 | 50 | 0.13424 | 0.00025 | 0.06725 |
| **p2feat** | **10** | **960** | **25** | **0.15440** | 0.00430 | **0.07935** |
| *p2feat* | *10* | *640* | *25* | **not run** | — | — |

Three effects fall out, all at 25 epochs:

| lever | comparison | effect on ship AP |
|---|---|---:|
| batch 16 → 10 | base/640 pair, nothing else differs | **+7.0%** |
| imgsz 640 → 960 | base/b10 pair | +1.6% (inside noise) |
| p2feat | at b16/640 | **+9.4%** |
| p2feat | at b10/960 | **+9.0%** |

The two `p2feat` measurements agree to within half a point across a different batch size *and* a
different input size — the reason to trust it on a campaign with no seed replicates (§7). But it
holds only at 25 epochs (§4.2).

**The missing cell is not a nicety.** `p2feat` @ 640 is measured at batch 16; `p2feat` @ 960 at
batch 10. Batch alone is worth +7.0% on the base, so `p2feat` @ 640 @ **batch 10** projects to
≈0.152 — within noise of the 0.15440 currently credited to 960. If that holds, the recipe is
*`p2feat` + batch 10* and the 960 is pure cost. One 75-minute run decides it.

### 2.1 Ranking statistic

Ranking is on the **mean of the last five epochs**, not peak (`scripts/report_screen.py`). Peak is
a single-epoch max over a jittery curve: the control alone swings 0.0579 → 0.0676 inside five
epochs, larger than the entire seed-to-seed gap. Peak is reported alongside for continuity with the
Phase 2 table, which quotes `best.pt`.

| arm | peak | @ep | last-5 mean | vs matched control |
|---|---:|---:|---:|---:|
| `s1_base_seed0` (b16/640) | 0.06762 | 13 | 0.06018 | control |
| `s1_base_seed1` | 0.06597 | 20 | 0.05925 | −1.5% → **noise floor** |
| `s2_base_b10` (b10/640) | 0.06990 | 17 | 0.06360 | +5.7% vs b16 |
| `s1_imgsz960` (b10/960) | 0.07105 | 21 | 0.06983 | +9.8% vs b10 |
| `s1_p2feat` (b16/640) | 0.07142 | 20 | 0.06663 | +10.7% vs b16 |
| `s2_p2feat_960` (b10/960) | **0.07938** | 20 | **0.07385** | **+16.1% vs b10/640** |

Noise floor: **0.00092 (1.5%)** on the last-5 mean, from the one seed pair. One pair is an order of
magnitude for run-to-run spread, not a confidence interval.

## 3. What `imgsz 960` was, and why it is null for IR — **VERIFIED**

**What the knob does.** Ultralytics resizes every input frame to a square of side `imgsz`
(letterboxed, aspect preserved) before it enters the network. Default 640; the arm set 960. Nothing
else changes — same weights, data, labels. Because a 12 GB card cannot hold 960 at batch 16, the
arm was forced to batch 10, which is where the confound entered.

**Why it cannot add information here.** The IR frames on disk are already at native resolution:

- Native PoLaRIS IR is **640×512, 8-bit** (`ir-handoff-2026-08.md` §2, tree #2).
- The training tree stores them as **640×640** — the native 640×512 letterboxed with 64 grey rows
  (value 114) top and bottom. Verified directly: 128 constant rows per frame.

So at `imgsz 640` the model already sees every pixel the sensor ever produced. Setting 960
**upsamples 640 → 960**: it invents no detail, only interpolates.

What it does change is *sampling density*. The P3 grid goes from 80×80 to 120×120, so a median
21×13 px ship becomes ~31×20 px and lands on more anchor points. A real mechanism — worth **+1.6%**,
inside the noise floor, for **2.2× the epoch time** (276 s vs 134 s). Not a trade worth making.

**This is where stage 1 misled.** Stage 1 ranked `imgsz960` first at +16.0% and it was read as the
top lever. It was not: batch 16 → 10 accounts for +5.7% on its own, and the rest is a curve-shape
artefact — `imgsz960` holds its endpoint steadier (last-5 0.06983 against a peak of 0.07105) while
the control decays (0.06360 from a peak of 0.06990). On peak and on ship AP the two agree at +1.6%.
It stabilises the tail; it does not detect more. `handoff-2026-08-19.md` §4.5 called imgsz a null
IR lever and was right.

**The asymmetry worth remembering:** this argument is specific to IR. VIS is natively 2048×1080 and
*is* downsampled to 640, so for VIS a larger `imgsz` would recover detail that genuinely exists on
disk. **OPEN** — untested, outside this screen's scope.

**Aside, unmeasured:** 20% of every IR input is grey padding at any `imgsz`. Rectangular training
(e.g. 640×512) would spend that fifth of the compute on pixels instead. **OPEN**.

## 4. Fifty epochs buys nothing — **VERIFIED**

An LR schedule depends on total epochs (`lf(x) = (1−x/epochs)(1−lrf) + lrf`), so a 50-epoch run is
not a superset of a 25-epoch run; each needed its own control.

| arm | 25-ep last-5 | 50-ep last-5 | change |
|---|---:|---:|---:|
| base | 0.06018 | 0.06293 | +4.6% |
| p2feat | 0.06663 | 0.06518 | −2.2% |
| imgsz960 | 0.06983 | 0.06421 | −8.0% |

Doubling the budget mildly helps the plain baseline and **hurts both winners**; all three converge
to ~0.063–0.065 and the ranking collapses. `s2_base_50ep` and `s2_imgsz960_50ep` both peak at epoch
25 and decay after. **25 epochs is already past the useful point.** Caveat: the combination arm at
50 epochs (`p2feat` @ 960) was never run. **OPEN.**

### 4.2 What the 50-epoch runs say about `p2feat` — **VERIFIED**

On ship AP the 50-epoch schedule inverts the ranking:

| model @ 640 / b16 | 25 ep | 50 ep |
|---|---:|---:|
| base | 0.13022 | **0.14232** |
| p2feat | **0.14243** | 0.13424 |

Two readings, the second uncomfortable:

1. **The baseline improves with the longer schedule (+9.3%); `p2feat` decays (−5.7%)** and ends
   *below* the stock model. Whatever `p2feat` adds, it also overfits sooner.
2. **`p2feat` @ 25 ep (0.14243) and base @ 50 ep (0.14232) are the same number.** At 640, `p2feat`
   reaches in half the epochs what the stock model reaches eventually — **faster convergence, not a
   higher ceiling**, a materially weaker claim than "+9%".

The one result resisting this reading is `p2feat` @ 960 @ 25 ep at **0.15440** — above every base
configuration at every schedule tested (best base anywhere: 0.14232). If real, the combination does
raise the ceiling. Two candidate explanations, not separable from what is on disk:

- **960 interacts with `p2feat` specifically.** Plausible: `p2feat`'s injected P2 branch exists to
  resolve fine detail, and upsampling gives it more grid to resolve on. The base model has no such
  branch to feed, which would explain why 960 is null for it (§3).
- **It is n=1**, and §2's batch-corrected projection says `p2feat` @ 640 @ b10 lands in the same
  place without the 960.

§8 item 1 is the direct test. Until it runs, the defensible claim is: **at the 25-epoch schedule
you would actually use, `p2feat` wins by ~9%.**

## 5. Buoy AP is zero in every run, and it is a data problem — **VERIFIED**

| arm | buoy AP |
|---|---:|
| base b16/640 | 0.00514 |
| base b10/640 | 0.00044 |
| base b10/960 | 0.00013 |
| p2feat b16/640 | 0.00042 |
| p2feat b10/960 | 0.00430 |

No architecture or resolution lever moved it. The labels explain why:

```
train  ship 78754 boxes / 79.8% of frames   median 25.0 x 19.0 px
train  buoy  3869 boxes / 24.9% of frames   median 10.0 x 23.0 px
val    ship 18281 boxes / 94.7% of frames   median 21.0 x 13.3 px
val    buoy   596 boxes /  6.8% of frames   median  3.7 x 11.4 px
```

A **20:1 class imbalance** in training, and val buoys are **3.7 px wide** — under half the width of
the train buoys they were learned from. A train/val distribution shift on top of the imbalance, and
no feature extractor fixes it.

**Consequence for every number here, and for the handoff's IR numbers:** mAP50-95 is macro-averaged
over two classes, one scoring ~0, so **the headline is approximately ship AP ÷ 2**. Every delta in
this screen is a ship delta wearing a disguise. This answers handoff decision **D-3**.

## 6. Rejected arms — **VERIFIED**

| arm | last-5 vs control | verdict |
|---|---:|---|
| `s1_p2` — a real P2/4 **detection level** | **−13.5%** | reject |
| `s1_scale025` — scale aug 0.5 → 0.25 | +0.5% | inside noise |
| `s1_closemosaic` — 5 clean epochs instead of 10 | +2.3% | marginal |

`s1_p2` is the informative failure. The hypothesis was strong — 54.8% of IR boxes get fewer
candidate anchors than `topk=7` under `TaskAlignedAssigner`, so a stride-4 level should help. It
was the worst arm in the screen. Two confounds, neither cheap to remove: adding a stride-4 level
makes Ultralytics size every Detect branch from `ch[0]`, which drops 128 → 64, so the arm started
from **92.5%** weight transfer with an entirely fresh 568k-param Detect head, against `p2feat`'s
**98.3%** with the head intact. Whether P2/4 fails on its merits or on its initialisation is
**OPEN** — but at −13.5% it does not earn a rerun.

`s1_closemosaic` was mislabeled when queued. Ultralytics' default `close_mosaic: 10` means the
*control* also closes mosaic, at CSV epoch 16 of 25; the arm closed later, at 21. It is "5 clean
epochs vs the control's 10", not "closed vs never".

## 7. What this screen does not establish

- **No seed replicates in stage 2.** Every stage-2 delta is n=1 against a noise floor estimated
  from a single stage-1 pair. `p2feat` survives only because it was measured twice at different
  settings; nothing else in stage 2 does.
- **The `p2feat` @ 640 @ b10 cell is missing**, and it is the load-bearing gap. The best config
  pairs `p2feat` with 960; §3 argues 960 adds no information to IR, and §2's batch correction
  projects the missing cell to within noise of it. ~75 min.
- **Whether `p2feat` raises the ceiling or only reaches it sooner is unresolved** (§4.2). At 640 the
  evidence favours "sooner". Only the 960 combination argues otherwise.
- **Everything here is IR, `yolo26s`, stride-2 tree, σ head on.** Nothing transfers to VIS or to
  fusion without being re-measured.
- **All deltas are ship-AP deltas** (§5).

## 8. Recommended next steps

| # | action | cost | why |
|---|---|---|---|
| 1 | Run `p2feat` @ 640 @ batch 10, 25 ep | 75 min | **the one that matters.** Closes the missing cell; if it lands near 0.152 as §2 projects, drop 960 and save 2.2× epoch time |
| 2 | Second seed on `p2feat` @ 960 | 130 min | the +8.5% ceiling claim in §4.2 rests on a single run |
| 3 | Treat buoy as a data task, not a model task | — | oversampling, or drop it from the headline metric and report ship AP |
| 4 | Re-measure `imgsz` on **VIS**, where the pixels are real | — | §3's asymmetry |

Not recommended: more epochs (§4), the P2/4 detection level (§6), `imgsz 960` on IR (§3).

## 9. Reproduction

```
python scripts/run_queue.py --queue-dir runs/queue_screen  run     # stage 1, 8.4 h
python scripts/run_queue.py --queue-dir runs/queue_screen2 run     # stage 2, 11.2 h
python scripts/report_screen.py --dir runs/screen1 --control s1_base_seed0
python scripts/report_screen.py --dir runs/screen2 --control s2_base_b10 --queue runs/queue_screen2/queue.json
python scripts/perclass_ap.py runs/screen2/s2_p2feat_960 runs/screen2/s2_base_b10
```

Dashboards (pause/resume mid-run, resuming from `last.pt` with optimizer and EMA intact):
`screen_dashboard.cmd` (port 8770), `screen2_dashboard.cmd` (port 8772).
