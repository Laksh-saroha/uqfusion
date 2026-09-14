# pohang04 freeze procedure (Phase 3 §7.2, Amendment 9 §A9.5)

Status: **prepared, not frozen.** Written 2026-09-14 while the 190 caches and 76 statistic files build.

§7.2 allows one look at pohang04, after everything that decides its score is frozen and committed.
Most of those inputs live under the git-ignored `runs/` tree, so a commit of code alone freezes
nothing. The freeze is therefore a commit that contains a **hash manifest of every file the look
reads**, and the look script refuses to score if any byte differs from it.

## What is frozen

`scripts/holdout_p04_freeze.py` inventories 316 files into
`docs/eval/holdout_p04_freeze_manifest.json` (sha256 + size):

| Group | Files | Where |
|---|---:|---|
| checkpoints | 10 | `runs/phase3_stage2/p3_{vis,ir}_seed{0-4}/weights/best.pt` |
| Mahalanobis references | 10 | `runs/cache_p3/seed{k}/gauss_{vis,ir}_train_clean.pkl` |
| development caches | 10 | `runs/cache_p3/seed{k}/gauss_{vis,ir}_paired_clean.pkl` |
| pohang04 caches | 190 | `runs/holdout_p04/caches/seed{k}/{clean,draw{v}_{v+10}}/` |
| pohang04 frame statistics | 76 | `runs/holdout_p04/derived/{brightness,structure}/` |
| calibration and geometry | 5 | reliability / brightness / structure constants, IR→VIS homography, `config.yaml` |
| development substrate | 8 | paired-val lists and manifest, dev clean statistics, Stage 1 JSON (selftest target) |
| pohang04 substrate | 5 | pair lists, pair manifest, step 1 manifest, `pohang04_pairs.csv` |
| committed reference | 2 | dev reference (AP_ref) and step 1 manifest copies in `docs/eval/` |

Recorded alongside: the label hash `c06611a684f4`, the converted-IR hash, the rules of A9.1–A9.4
(preset `crossmodal26m`, systems VIS k + IR k, draws, verdict cell clean/clean, floor 0.0060 with
the other floors reported, bootstrap L=20 / 1000 / seeds 0 and 1, AP_ref = 0.2898), the git HEAD
before the freeze, and the package versions of both interpreters.

## Readiness gate

The non-draft run refuses unless:

1. all 190 caches pass the builder's meta check and all 76 statistic files match their stream
   and were written under numpy 2.4.6 / opencv 5.0.0 (`holdout_p04_look.check_inputs`);
2. the pohang04 labels hash to `c06611a684f4`;
3. `runs/holdout_p04/look_selftest.log` ends in `EXACT` (the scoring path reproduces
   Stage 1 `clean/A/tune` on development data);
4. no inventoried file is missing;
5. the manifest does not already exist (written once).

`--draft --out <scratch>` writes the same manifest with missing entries as `null` and never
refuses. Draft on 2026-09-14 17:15: 316 inventoried, 259 missing (183 caches, 76 statistics).

## Procedure

```
# 1. builds finish; the 3 shard logs end in "done", frame_stats.log ends in "[done]"
# 2. write the manifest (GPU interpreter, PYTHONPATH=src)
python scripts/holdout_p04_freeze.py
# 3. commit it with every script and doc the look depends on; the subject must start "FREEZE"
git add docs/eval/holdout_p04_freeze_manifest.json docs/holdout-p04-freeze-2026-09-14.md scripts/holdout_p04_*.py
git commit -m "FREEZE: pohang04 single look inputs (190 caches, 76 statistics, 10 checkpoints)"
# 4. the look -- once, only on explicit instruction
python scripts/holdout_p04_look.py
```

## What the look checks before scoring

`holdout_p04_look.py` refuses, in order, if: HEAD is not a `FREEZE` commit or any tracked file is
modified; `runs/holdout_p04/LOOK_TAKEN.json` exists; the committed manifest is missing, a draft,
or any listed file's hash differs; the labels hash differs; the inputs check fails; the dev
reference per-seed AP does not reproduce exactly. It then creates the marker (exclusive create)
before scoring, so a crash mid-look still counts as the look.

## What is not frozen by this, and why that is acceptable

- **Source code under `src/` and `scripts/`** is frozen by the commit itself (clean tree required).
- **Images** (`Pohang_dataset/.../pohang04`) are not re-hashed: the look never reads them — every
  detection is in the caches, every statistic in the frame files. The converted IR set's hash is
  recorded from step 1.
- **The GPU interpreter's packages** are recorded, not enforced; scoring is numpy/pycocotools
  arithmetic over cached detections, and the selftest must reproduce Stage 1 exactly under it.
