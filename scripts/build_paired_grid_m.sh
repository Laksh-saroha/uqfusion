#!/usr/bin/env bash
# Rebuild the whole paired fusion grid on the FULL-SCALE yolo26m detectors.
#
# Everything measured in docs/crossmodal-gate-2026-09-01.md rests on
# `runs/phase2/*` -- yolo26s, VIS trained on stride5, and an IR model that is
# still nc=2 despite D28/A-1. The deployed architecture (handoff SS1) is
# yolo26m for VIS and yolo26m-p2feat / nc=1 for IR, and those checkpoints have
# existed in runs/full_scale/ since 2026-08-21 without any fusion number ever
# being computed on them.
#
# Writes to runs/cache_m/ -- a NEW directory. runs/cache/ is untouched, so every
# published 26s number stays reproducible and the two can be diffed.
#
# STEM NAMES ARE DEEPLY IDENTICAL to runs/cache/ on purpose: the per-frame image
# statistics in runs/derived/{brightness,structure}/ are a function of the image
# list and the corruption (kind, severity, seed) alone, never of the detector, so
# reusing the stems reuses those files exactly. build_paired_grid_m_verify.py
# checks that claim rather than assuming it.
#
# Corruption kind/severity/seed are copied verbatim from each 26s cache's own
# meta, so the pixels the 26m detector sees are the pixels the 26s one saw.
#
# Usage:  bash scripts/build_paired_grid_m.sh
set -uo pipefail
cd "$(dirname "$0")/.."

GP="C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe"
export PYTHONPATH="$PWD/src"
COMMON="--source gaussian --imgsz 640 --conf 0.001"
mkdir -p runs/cache_m runs/logs_cache_m

STAGE="${1:-}"          # "" = base best.pt (matches the 26s convention), "_ft" = fine-tuned
VIS_W="runs/full_scale/gauss_vis_seed0${STAGE}/weights/best.pt"
IR_W="runs/full_scale/gauss_ir_seed0${STAGE}/weights/best.pt"
SUF="${STAGE}"

build () {  # modality out_stem list corrupt severity seed
  local mod=$1 stem=$2 list=$3 kind=$4 sev=$5 seed=$6
  local w out extra=""
  if [ "$mod" = ir ]; then w=$IR_W; else w=$VIS_W; fi
  out="runs/cache_m/${stem}${SUF}.pkl"
  if [ -f "$out" ]; then echo "[skip] $out"; return 0; fi
  if [ "$kind" != "none" ]; then extra="--corrupt $kind --severity $sev --corrupt-seed $seed"; fi
  echo "=== $out ==="
  "$GP" -u scripts/build_cache.py $COMMON --weights "$w" --images-list "$list" \
      $extra --out "$out" > "runs/logs_cache_m/${stem}${SUF}.log" 2>&1 \
      || { echo "[FAIL] $out"; tail -5 "runs/logs_cache_m/${stem}${SUF}.log"; return 1; }
  echo "[ok] $out"
}

PV=runs/derived/paired_val_vis.txt
PI=runs/derived/paired_val_ir.txt

# The two the scorers are fitted on -- needed first, because feat dimensionality
# changes with the backbone and nothing else can be scored without them.
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

echo "ALL 26m CACHES BUILT (stage='${STAGE:-base}')"
