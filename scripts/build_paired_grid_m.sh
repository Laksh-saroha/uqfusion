#!/usr/bin/env bash
# Rebuild the whole paired fusion grid on the FULL-SCALE yolo26m detectors.
#
# Every number in docs/crossmodal-gate-2026-09-01.md rests on runs/phase2/* --
# yolo26s, VIS trained on stride5, and an IR model still carrying nc=2 despite
# D28/A-1. The deployed architecture (handoff SS1) is yolo26m for VIS and
# yolo26m-p2feat / nc=1 for IR. Those checkpoints have sat in runs/full_scale/
# since 2026-08-21 with no fusion number ever computed on them.
#
# STAGE, per modality, by D31's rule (highest validation mAP50-95), decided from
# numbers already in runs/queue_full_scale/state.json and NOT from the paired
# fusion metric:
#     VIS  gauss_vis_seed0      base  0.24961   (ft 0.24111 -- worse)
#     IR   gauss_ir_seed0_ft    ft    0.14214   (base 0.11981)
# The IR fine-tune improving on its base is the documented outlier (handoff
# SS1); the rule is applied per modality precisely so that does not have to be
# argued either way. All four clean caches are built regardless, into
# runs/cache_m_stageprobe/, so the choice is reported and not merely asserted.
#
# Writes to runs/cache_m/ -- a NEW directory. runs/cache/ is untouched, so every
# published 26s number stays reproducible and the two are diffable.
#
# STEMS ARE IDENTICAL to runs/cache/ on purpose: the per-frame image statistics
# in runs/derived/{brightness,structure}/ are a function of the image list and
# the corruption (kind, severity, seed) alone, never of the detector, so equal
# stems reuse those files exactly. verify_cache_m.py checks that claim on the
# pixels rather than assuming it.
#
# Corruption kind/severity/seed are copied verbatim from each 26s cache's meta,
# so the 26m detector sees the pixels the 26s one saw.
#
# Usage:  bash scripts/build_paired_grid_m.sh
set -uo pipefail
cd "$(dirname "$0")/.."

GP="C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe"
export PYTHONPATH="$PWD/src"
COMMON="--source gaussian --imgsz 640 --conf 0.001"
mkdir -p runs/cache_m runs/logs_cache_m

VIS_W="runs/full_scale/gauss_vis_seed0/weights/best.pt"
IR_W="runs/full_scale/gauss_ir_seed0_ft/weights/best.pt"

build () {  # modality out_stem list corrupt severity seed
  local mod=$1 stem=$2 list=$3 kind=$4 sev=$5 seed=$6
  local w out extra=""
  if [ "$mod" = ir ]; then w=$IR_W; else w=$VIS_W; fi
  out="runs/cache_m/${stem}.pkl"
  if [ -f "$out" ]; then echo "[skip] $out"; return 0; fi
  if [ "$kind" != "none" ]; then extra="--corrupt $kind --severity $sev --corrupt-seed $seed"; fi
  echo "=== $out ==="
  "$GP" -u scripts/build_cache.py $COMMON --weights "$w" --images-list "$list" \
      $extra --out "$out" > "runs/logs_cache_m/${stem}.log" 2>&1 \
      && echo "[ok] $out" || { echo "[FAIL] $out"; tail -5 "runs/logs_cache_m/${stem}.log"; }
}

PV=runs/derived/paired_val_vis.txt
PI=runs/derived/paired_val_ir.txt

# The scorer-fit caches come first: feat dimensionality changes with the
# backbone, so nothing else can be scored until these exist.
build vis gauss_vis_train_clean runs/derived/maha_fit_vis.txt none 0 0
build ir  gauss_ir_train_clean  runs/derived/maha_fit_ir.txt  none 0 0

# The paired val grid. VIS seed 1, IR seed 7 -- as in runs/cache/.
build vis gauss_vis_paired_clean    "$PV" none     0 0
build ir  gauss_ir_paired_clean     "$PI" none     0 0
build vis gauss_vis_paired_fog      "$PV" fog      2 1
build vis gauss_vis_paired_lowlight "$PV" lowlight 2 1
build vis gauss_vis_paired_glare    "$PV" glare    2 1
build vis gauss_vis_paired_blur_s3  "$PV" blur     3 1
build vis gauss_vis_paired_noise_s2 "$PV" noise    2 1
build vis gauss_vis_paired_rain_s2  "$PV" rain     2 1
build ir  gauss_ir_paired_fog_s2    "$PI" fog      2 7
build ir  gauss_ir_paired_glare_s2  "$PI" glare    2 7
build ir  gauss_ir_paired_blur_s2   "$PI" blur     2 7
build ir  gauss_ir_paired_noise_s2  "$PI" noise    2 7

echo "ALL 26m CACHES BUILT"
