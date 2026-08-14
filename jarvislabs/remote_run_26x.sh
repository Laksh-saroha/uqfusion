#!/usr/bin/env bash
# yolo26x x seeds 0,1,2 on the rented L4. Mirrors run_tail_laptop.cmd, same two
# safety layers: grid.py resumes an interrupted run from its own weights/last.pt
# (epoch-level, so an interruption costs minutes), and this loop restarts the grid
# if the process dies for any other reason. Finished (variant, seed) rows are read
# back from the CSV and skipped, so re-running this script is always safe.
#
# It is also what the watchdog re-executes after a spot pause/resume, hence the
# flock: two trainers sharing one GPU would OOM each other and corrupt both runs.
#
# --data / --classes / --out-csv must stay identical across invocations or
# grid.py's split-fingerprint and class-mix guards abort the run.
set -uo pipefail

ROOT="${ROOT:-/home/uqfusion}"
BATCH="${BATCH:-16}"
WORKERS="${WORKERS:-8}"
SEEDS="${SEEDS:-0 1 2}"
MAX_TRIES="${MAX_TRIES:-40}"
LOG="$ROOT/runs/grid_l4.log"
CSV="$ROOT/runs/benchmark/benchmark_results_l4_26x.csv"

mkdir -p "$ROOT/runs/benchmark"
cd "$ROOT"
export PYTHONPATH="$ROOT/src"
# Long runs fragment the allocator; expandable segments cost nothing numerically
# and remove the class of OOM that only shows up after 20 epochs.
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export YOLO_VERBOSE=true

exec 9>"$ROOT/.grid.lock"
if ! flock -n 9; then
  echo "[wrapper] another grid holds $ROOT/.grid.lock — refusing to start a second trainer" | tee -a "$LOG"
  exit 3
fi

rm -f "$ROOT/runs/GRID_DONE" "$ROOT/runs/GRID_FAILED"
echo "[wrapper] launch $(date -Is) batch=$BATCH workers=$WORKERS seeds='$SEEDS'" >> "$LOG"

# A pause/resume hands back a FRESH CONTAINER: only /home survives. site-packages
# lives in /root/miniconda3, so ultralytics 8.4.90 is gone after every reclaim and
# the grid dies with ModuleNotFoundError on a box whose dataset and checkpoints are
# all perfectly intact. Rebuild the env before training, every time — it is ~90 s,
# and it is the difference between a resume that works unattended and one that
# crash-loops until the retry budget runs out.
if ! python -c "import ultralytics, sys; sys.exit(0 if ultralytics.__version__ == '8.4.90' else 1)" 2>/dev/null; then
  echo "[wrapper] ultralytics 8.4.90 not importable — rebuilding the environment" >> "$LOG"
  if ! bash "$ROOT/jarvislabs/remote_setup.sh" "$ROOT" >> "$LOG" 2>&1; then
    echo "[wrapper] environment rebuild FAILED — see above" >> "$LOG"
    date -Is > "$ROOT/runs/GRID_FAILED"
    exit 2
  fi
  echo "[wrapper] environment rebuilt" >> "$LOG"
fi

for ((i = 1; i <= MAX_TRIES; i++)); do
  echo "" >> "$LOG"
  echo "[wrapper] attempt $i at $(date -Is)" >> "$LOG"
  python scripts/run_benchmark.py \
      --data runs/derived/data_vis_stride2.yaml \
      --classes 0 \
      --seeds $SEEDS \
      --variants yolo26x \
      --batch "$BATCH" \
      --workers "$WORKERS" \
      --out-csv "$CSV" \
      --run-prefix ship >> "$LOG" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then
    echo "[wrapper] grid complete at $(date -Is)" >> "$LOG"
    date -Is > "$ROOT/runs/GRID_DONE"
    exit 0
  fi
  if tail -c 200000 "$LOG" | grep -q "CUDA out of memory"; then
    # Ultralytics reloads batch from the checkpoint on resume, so a smaller batch
    # needs the run dir cleared — a decision with a real cost, left to a human.
    echo "[wrapper] OOM at batch $BATCH — see runbook, batch cannot change on resume" >> "$LOG"
  fi
  echo "[wrapper] exit $rc on attempt $i, retrying in 60s" >> "$LOG"
  sleep 60
done

echo "[wrapper] $MAX_TRIES consecutive failures — giving up at $(date -Is)" >> "$LOG"
date -Is > "$ROOT/runs/GRID_FAILED"
exit 1
