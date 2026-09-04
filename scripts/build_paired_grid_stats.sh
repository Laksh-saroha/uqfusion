#!/usr/bin/env bash
# Per-frame statistics for the extended paired grid, so the gate can run on it.
#
# The gate reads two things off each frame: the photometric block
# (`frame_brightness.py` -> p05, used by the night arm and the VIS-dark vote) and
# the scale-free structure block (`frame_structure.py` -> grad_gini for the veil
# axis, plus the 11-statistic vector the IR/VIS health models score). A new cache
# without both is a condition the gate cannot be evaluated on.
#
# Both scripts replay the corruption recorded in the cache meta, so the pixels
# measured are the pixels the detector saw. Both are read-only on the dataset.
#
# CPU-only and embarrassingly parallel across caches; runs after
# build_paired_grid_ext.sh.
#
# Usage:  bash scripts/build_paired_grid_stats.sh

set -uo pipefail
cd "$(dirname "$0")/.."
PY=.venv/Scripts/python.exe

pids=()
for pkl in runs/cache/gauss_ir_paired_*_s[0-9].pkl runs/cache/gauss_vis_paired_*_s[0-9].pkl; do
  [ -f "$pkl" ] || continue
  stem=$(basename "$pkl" .pkl)
  case "$stem" in gauss_ir_*) mod=ir ;; *) mod=vis ;; esac
  for kind in brightness structure; do
    out="runs/derived/$kind/$stem.json"
    [ -f "$out" ] && { echo "[skip] $out"; continue; }
    echo "[run] $kind $stem"
    $PY -u "scripts/frame_${kind/brightness/brightness}.py" \
        --cache "$pkl" --modality "$mod" --out "runs/derived/$kind" \
        > "runs/derived/${kind}_${stem}.log" 2>&1 &
    pids+=($!)
  done
done
for p in "${pids[@]:-}"; do [ -n "$p" ] && wait "$p"; done
echo "ALL STATS BUILT"
