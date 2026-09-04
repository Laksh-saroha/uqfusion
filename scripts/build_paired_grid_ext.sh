#!/usr/bin/env bash
# Extend the paired benchmark past the VIS-only, four-condition grid.
#
# Two gaps close here, and they need different caches:
#
#  (a) CORRUPTED IR. Every cell of the original grid corrupts VIS and leaves IR
#      pristine, so the cross-modal night test is graded on a benchmark where the
#      sensor it consults can never be wrong, and no eight-cell number can exercise
#      its failure mode. `probe_ir_night_robustness.py` measured that at the
#      SWITCH level from image statistics alone; these caches measure it at the
#      FUSION level, with a degraded IR detector.
#
#  (b) A CELL WHERE VIS IS UNVETOED AND IR IS THE BETTER STREAM. Without one, the
#      capability ratio cannot be tuned honestly: it only matters where VIS is
#      unvetoed, and on every existing such cell VIS wins, so a larger ratio is
#      free. blur/noise/rain on VIS are the candidates -- on the ladder they take
#      VIS below IR's 0.0181 -- and they are also three corruption families the
#      veil axis has never been tested against, so the run doubles as a
#      generalisation check on a gate fitted without ever seeing them.
#
# Seeds follow plan B5-5 (different draws for tuning vs testing). The existing VIS
# paired caches used seed 1, so the new VIS conditions use 1 to stay comparable
# within the VIS grid; the IR conditions use 7, matching the seed the IR image
# statistic probes already ran at, so the statistics and the detector outputs
# describe the same pixels.
#
# Writes only NEW filenames. Nothing existing is touched.
#
# Usage:  bash scripts/build_paired_grid_ext.sh

set -euo pipefail
cd "$(dirname "$0")/.."

GP="C:/Users/lasa2/AppData/Local/Programs/Python/Python313/python.exe"   # config.gpu_python
export PYTHONPATH="$PWD/src"
IR_W=runs/phase2/gauss_ir_seed0/weights/best.pt
VIS_W=runs/phase2/gauss_vis_seed0/weights/best.pt
COMMON="--source gaussian --imgsz 640 --conf 0.001"

build () {  # modality kind severity seed
  local mod=$1 kind=$2 sev=$3 seed=$4
  local w list out
  if [ "$mod" = ir ]; then w=$IR_W; list=runs/derived/paired_val_ir.txt
  else w=$VIS_W; list=runs/derived/paired_val_vis.txt; fi
  out="runs/cache/gauss_${mod}_paired_${kind}_s${sev}.pkl"
  if [ -f "$out" ]; then echo "[skip] $out exists"; return; fi
  echo "=== $out ==="
  "$GP" scripts/build_cache.py $COMMON --weights "$w" --images-list "$list" \
      --corrupt "$kind" --severity "$sev" --corrupt-seed "$seed" --out "$out"
}

# (a) corrupted IR -- fog and glare are the two that broke the night switch at the
#     image-statistic level; blur and noise are the IR ladder's own conditions.
for k in fog glare blur noise; do build ir "$k" 2 7; done

# (b) VIS conditions where IR should be the better stream, and which the veil axis
#     has never been tested against.
build vis blur  3 1
build vis noise 2 1
build vis rain  2 1

echo "ALL CACHES BUILT"
