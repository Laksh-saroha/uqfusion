#!/bin/bash
# Keep one run_queue.py queue working, unattended, for as long as it takes.
#
# Why this exists: `run_queue.py run` is sequential and resume-safe, but it is
# one process. If it dies -- a CUDA fault, a torn dataloader, an OOM, a killed
# terminal -- the queue simply stops, and on a box nobody is watching that costs
# every hour until someone notices. This restarts it, from the same checkpoint
# it stopped at, and gets out of the way otherwise. It never kills anything.
#
#   ./scripts/supervise_queue.sh --queue-dir runs/queue_vis_benchmark_stride4 \
#       --wait-pid 26701            # start only once THAT runner has exited
#
# Options:
#   --queue-dir DIR    the queue to work (required)
#   --wait-pid PID     wait for this pid to exit before starting (one GPU, one job)
#   --wait-queue DIR   also wait until that queue has no unfinished runs
#   --python PATH      interpreter (default: python)
#   --log FILE         log file (default: <queue-dir>/supervisor.log)
#   --poll SECONDS     wait-loop poll interval (default 60)
#   --max-restarts N   give up after N restarts (default 200; 0 = unlimited)
#
# Backoff: a runner that exits within 5 minutes of starting is treated as
# crash-looping and the wait doubles (60s -> 30 min cap) instead of hammering the
# GPU. A runner that ran longer resets the backoff, because that is a normal
# exit-and-continue, not a loop.

set -u

QUEUE_DIR=""; WAIT_PID=""; WAIT_QUEUE=""; PY="python"; LOG=""; POLL=60; MAX_RESTARTS=200
while [ $# -gt 0 ]; do
  case "$1" in
    --queue-dir) QUEUE_DIR="$2"; shift 2;;
    --wait-pid) WAIT_PID="$2"; shift 2;;
    --wait-queue) WAIT_QUEUE="$2"; shift 2;;
    --python) PY="$2"; shift 2;;
    --log) LOG="$2"; shift 2;;
    --poll) POLL="$2"; shift 2;;
    --max-restarts) MAX_RESTARTS="$2"; shift 2;;
    *) echo "unknown option: $1" >&2; exit 2;;
  esac
done
[ -n "$QUEUE_DIR" ] || { echo "--queue-dir is required" >&2; exit 2; }

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO" || exit 2
[ -n "$LOG" ] || LOG="$QUEUE_DIR/supervisor.log"
mkdir -p "$(dirname "$LOG")"

say() { echo "$(date -Is)  $*" | tee -a "$LOG"; }

# A heartbeat file the xlsx report reads. Without it, a queue that is waiting its
# turn on the GPU is indistinguishable from a queue nobody is working -- and the
# report would cry IDLE at someone who is four days away and cannot check.
# `updated` is refreshed every poll, so a dead supervisor shows up as a stale
# heartbeat rather than a lie.
status() {
  printf '{"state": "%s", "detail": "%s", "pid": %s, "wait_pid": "%s", "restarts": %s, "updated": "%s"}\n' \
    "$1" "$2" "$$" "${WAIT_PID:-}" "${RESTARTS:-0}" "$(date -Is)" > "$QUEUE_DIR/supervisor.json" 2>/dev/null
}

say "supervisor starting: repo=$REPO queue=$QUEUE_DIR python=$PY pid=$$"
status starting "just launched"

# --- 1. wait for the GPU ------------------------------------------------------
# Two trainings on one MIG slice measure ~8% SLOWER than running them one after
# the other, so this waits rather than overlapping.
if [ -n "$WAIT_PID" ]; then
  if kill -0 "$WAIT_PID" 2>/dev/null; then
    say "waiting for pid $WAIT_PID to exit before starting (polling every ${POLL}s)"
    while kill -0 "$WAIT_PID" 2>/dev/null; do
      status waiting_pid "holding the GPU for pid $WAIT_PID (the queue ahead of this one)"
      sleep "$POLL"
    done
    say "pid $WAIT_PID has exited"
    sleep 30   # let its CUDA context and dataloader workers actually go away
  else
    say "pid $WAIT_PID is already gone"
  fi
fi

if [ -n "$WAIT_QUEUE" ]; then
  say "waiting for queue $WAIT_QUEUE to have no unfinished runs"
  while true; do
    status waiting_queue "holding until $WAIT_QUEUE has no unfinished runs"
    "$PY" scripts/queue_remaining.py --queue-dir "$WAIT_QUEUE" >>"$LOG" 2>&1 && break
    sleep "$POLL"
  done
  say "queue $WAIT_QUEUE is complete"
fi

# --- 2. work the queue, restarting it if it dies ------------------------------
RESTARTS=0
BACKOFF=60
while true; do
  if "$PY" scripts/queue_remaining.py --queue-dir "$QUEUE_DIR" >>"$LOG" 2>&1; then
    say "queue complete — supervisor exiting"
    status complete "every run is terminal"
    break
  fi
  STARTED=$(date +%s)
  say "starting runner (restart #$RESTARTS)"
  status running "runner working the queue (restart #$RESTARTS)"
  "$PY" scripts/run_queue.py --queue-dir "$QUEUE_DIR" run >>"$QUEUE_DIR/runner_stdout.log" 2>&1
  RC=$?
  RAN=$(( $(date +%s) - STARTED ))
  say "runner exited rc=$RC after ${RAN}s"

  if "$PY" scripts/queue_remaining.py --queue-dir "$QUEUE_DIR" >>"$LOG" 2>&1; then
    say "queue complete — supervisor exiting"
    status complete "every run is terminal"
    break
  fi

  RESTARTS=$((RESTARTS + 1))
  if [ "$MAX_RESTARTS" -gt 0 ] && [ "$RESTARTS" -ge "$MAX_RESTARTS" ]; then
    say "hit --max-restarts $MAX_RESTARTS — stopping. Something needs a human."
    status gave_up "hit --max-restarts $MAX_RESTARTS after rc=$RC; needs a human"
    exit 1
  fi
  if [ "$RAN" -lt 300 ]; then
    BACKOFF=$(( BACKOFF * 2 )); [ "$BACKOFF" -gt 1800 ] && BACKOFF=1800
    say "that exit was fast (<5 min) — backing off ${BACKOFF}s before retrying"
  else
    BACKOFF=60
  fi
  status backoff "runner exited rc=$RC after ${RAN}s; retrying in ${BACKOFF}s"
  sleep "$BACKOFF"
done
