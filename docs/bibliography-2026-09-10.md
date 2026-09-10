# Bibliography and dataset premises — R-F2 / F17 citation table

**R-F2.** Written 2026-09-10, after [R-F1](positioning-2026-09-10.md).

The backlog scopes this as six citation corrections. Two of them were already right, one
is a genuine venue fix, and **three are not citation problems at all** — they are claims
about *this project* that the external review could only flag from the outside because it
could not see the disk. From the inside they are measurable, and two of them are false.

Local numbers here are reproducible: `python scripts/verify_dataset_claims.py`.
Raw artifacts (`runs/` is gitignored, so the tracked copies live under `docs/eval/`):
[`eval/dataset_claims_2026-09-10.json`](eval/dataset_claims_2026-09-10.json), [`eval/dataset_claims_2026-09-10.txt`](eval/dataset_claims_2026-09-10.txt).

---

## 1. The six items, one line each

| # | Review's correction | Status here |
|---|---|---|
| 1 | RT-DETR is CVPR 2024 | **Already correct** — `scope.md` R17 says CVPR 2024. No change. |
| 2 | D-FINE is ICLR 2025, not arXiv-only | **Was wrong** — R18 said "arXiv 2024/2025". Fixed. |
| 3 | Pohang and PoLaRIS are distinct releases; tracker is undated | **Was wrong** — R20 had `—` for venue. Fixed. |
| 4 | MassMIND's 7 categories are not our ship/buoy taxonomy | **Understated here** — §5.3 named the 7 classes but called the conversion routine. Fixed. |
| 5 | MIT: 12 fps VIS / 30 fps IR, CC BY-NC-SA 4.0; source does not verify our annotation subset | **Partly right, and worse than stated** — the fps and licence in §5.1 were already correct. The annotation claim is **false on this machine**. See §3. |
| 6 | SMD has separate visible/IR material, not synchronized pairs | **Was overstated** — §5.3's "Potential Role" assumed pairs. Fixed. |

Items 1–3 and 6 are recorded from the review's own primary-source checks
([`architecture-review-2026-09-09.md`](architecture-review-2026-09-09.md), *External
references and dataset premises*), which link the sources. I did not re-open those
sources; §5 says what that means.

## 2. The venue fixes

* **D-FINE** — R18's venue becomes **ICLR 2025**
  (<https://proceedings.iclr.cc/paper_files/paper/2025/hash/6cf58a87e3097e7d1f9be3e8693a93de-Abstract-Conference.html>).
* **RT-DETR** — R17 already read CVPR 2024
  (<https://openaccess.thecvf.com/content/CVPR2024/html/Zhao_DETRs_Beat_YOLOs_on_Real-time_Object_Detection_CVPR_2024_paper.html>).
  Left alone. Recording that it was already right matters: a repair backlog that silently
  "fixes" correct things stops being evidence of anything.
* **Pohang / PoLaRIS** — R19 and R20 are **two releases, not one**. Pohang Canal Dataset,
  Chung et al., IJRR 2023 (<https://arxiv.org/abs/2303.05555>) is the sensor release.
  PoLaRIS is the later annotation release: 2024 preprint
  (<https://arxiv.org/abs/2412.06192>), ICRA 2025, author code
  <https://github.com/sparolab/PoLaRIS>. R20's venue column was blank; it is now dated,
  and the two rows say explicitly that our labels come from PoLaRIS while our imagery
  comes from Pohang. **R20 names no authors on purpose** — I do not have them from a
  source I have read, and a plausible-looking author list is exactly the kind of error
  this pass exists to remove. The row says so, and points at the preprint.

## 3. The part the review could not see: MIT Marine Perception is not here

`scope.md` §5.1 listed MIT Sea Grant Marine Perception as one of "the two primary datasets
**used for training and evaluating** the uncertainty-gated fusion system", and stated:

> We have manually annotated a subset of images from this dataset ourselves.

**Measured on this machine, 2026-09-10:**

| check | result |
|---|---|
| `data/` directory (`data_root` in `config.yaml`) | **does not exist** |
| `datasets.mit_marine.vis_yaml` / `.ir_yaml` in `config.yaml` | **both `null`** — comment: "onboarded Phase 2+; yamls filled then" |
| MIT / MassMIND / SMD referenced in any `.py`, `.yaml`, or `.json` | **none** (prose docs only) |
| dataset trees on disk | `Pohang_dataset/` only |

So there is **no MIT imagery, no MIT annotation, and no code path that could load one**.
Every trained model and every evaluated number in this project is Pohang-only. This is
consistent with the project's own record — decision **D10, 2026-07-07**, deferred MIT to
Phase 2+ — but §5.1 was never updated, so a scope document read in Phase 3 asserts
completed annotation work that does not exist.

**This is the most serious item in R-F2 and the review classified it as a citation
footnote.** A venue year is an error a reader forgives; a claim to have annotated data you
have not annotated is not. It is now corrected in three places: §5.1's lead sentence, the
MIT row's description, and R21's status in the §19.7 tracker (**Primary → Deferred, not
onboarded**).

## 4. Two more defects found while checking, which the review did not raise

**(a) §5.1 contradicted itself about registration.** Its lead sentence said both datasets
"provide **co-registered** visible and infrared imagery" — eight lines below, the same
section says "**No spatial registration** between VIS and IR". Measurement agrees with the
second: **3–6 px median residual** (`runs/eval/x_registration_drift.md`), within-run horizontal swings approaching 10 px, and at `iou_thr` 0.85 only ~0.05% of VIS boxes have an IR
partner (`project-fusion-mechanism`). The word is removed; the pairing is by nearest
timestamp within 50 ms, which is what the tree actually contains.

**(b) Two of §5.1's local counts do not reproduce.**

| claim in §5.1 | measured 2026-09-10 | verdict |
|---|---:|---|
| ~158k images | **158,319** | holds |
| **~1.22M boxes** | **1,183,736** | **off by ~36k** |
| ~28k paired VIS↔IR | **28,388** | holds |
| VIS 127k / IR 31k images | **127,309 / 31,010** | holds |
| pohang04 zero IR labels | **0** | holds |
| pohang03 IR ~1,922 | **1,922** | holds |
| **pohang03 "~13k VIS"** | **27,085** | **wrong by 2×** |

Per-run, VIS then IR: pohang00 21,768/10,918 · pohang01 24,473/11,995 · pohang02
27,795/6,175 · pohang03 27,085/1,922 · pohang04 26,188/0. Boxes: VIS 962,960, IR 220,776.
Pairs: 10,786 / 11,990 / 3,739 / 1,873 / 0.

The VIS side cross-checks against `scripts/label_hash_ledger.py`, which reports tree hash
**`b92739202127`, 127,309 files, 962,960 boxes** — the same numbers by an independent
walk. I did not chase where "~1.22M" and "~13k" came from; both predate the 2026-09-02
night-box restore and the re-split, and reconstructing a stale number's provenance is not
worth the hours. They are replaced by measured values **with the date and the script that
produced them**, which is the review's actual point: *local filtered counts are not
verified merely by citing the source dataset.*

## 5. What is still unverified, and stays that way

* **I did not re-open the primary sources for items 1–4 and 6.** The venue years, the fps
  figures, the licence, and MassMIND's category count are taken from the external review's
  checks, which link the sources. `scope.md` now attributes them there rather than
  presenting them as independently confirmed. Anyone preparing a manuscript should open
  the links.
* **MassMIND's instance/class mapping is named as required, not designed.** §5.3 now says
  the seven segmentation categories are not the ship/buoy taxonomy and that a documented
  mapping is a prerequisite. Writing that mapping is not R-F2's job and no such mapping
  exists in the repo.
* **SMD and MassMIND remain unused.** §5.3 already framed them as optional supplements;
  that framing is correct and is kept. Only the "Potential Role" wording that presumed
  synchronized SMD pairs is changed.
* **R-F1's blank comparison cells are still blank.** R24/R25 calibration protocol,
  registration assumptions and compute are unaffected by this pass.

## 6. What this does not do

It changes documentation only. No number in `runs/` moves, no model is retrained, and
`scripts/verify_dataset_claims.py` reports rather than gates — it exits 0 on a mismatch,
because its job is to make the scope document checkable, not to block work.
