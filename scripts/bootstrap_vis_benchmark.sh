#!/bin/bash
# One-shot server-side setup for the 93-run VIS 2-class benchmark (dgxanode01).
#
# Run it from the repo root:
#     cd /workspace/uqfusion && bash scripts/bootstrap_vis_benchmark.sh
#
# It is idempotent -- rerunning it after a container restart is the intended
# recovery path, and it will not start a second copy of anything it finds
# already running.
#
# What it does, in order:
#   1. find the VIS ensemble runner and the interpreter IT is using, so the
#      benchmark runs on the same torch build and waits for that GPU
#   2. build runs/derived/data_vis_stride4.yaml (train stride 4; val/test whole)
#   3. pre-fetch all 31 COCO checkpoints + yolo11n.pt (the AMP probe) so a TLS
#      failure four days from now cannot cost a run, as it already did once
#   4. start the supervisor: it waits for the ensemble runner to exit, then
#      works the benchmark queue and restarts it if it dies
#   5. start the xlsx tracker over BOTH queues, refreshing every 5 minutes
#
# It never pauses, signals or otherwise touches the ensemble run.

set -u
cd "$(dirname "$0")/.." || exit 2
REPO="$(pwd)"
QDIR="runs/queue_vis_benchmark_stride4"
XLSX="runs/status_dgxanode01.xlsx"
ENS_QUEUE="runs/queue_vis_server"
# The queue.json shipped in the payload carries the laptop's relative data path;
# on this box the dataset yamls are absolute, exactly as queue_vis_server's are.
DATA_YAML="/workspace/derived/data_vis_stride4.yaml"

say() { echo "[bootstrap] $*"; }
fail() { echo "[bootstrap] ERROR: $*" >&2; exit 1; }

say "repo = $REPO"
[ -f scripts/run_queue.py ] || fail "not in the uqfusion repo (no scripts/run_queue.py)"
[ -f "$QDIR/queue.json" ] || fail "$QDIR/queue.json is missing — unzip the bootstrap payload first"
chmod +x scripts/supervise_queue.sh 2>/dev/null

# Point every run at this machine's dataset yaml. Idempotent: a queue already
# rewritten reports 0 and is left byte-identical, so state.json stays valid.
python3 - "$QDIR/queue.json" "$DATA_YAML" <<'PY' || fail "could not rewrite the queue's data paths"
import json, sys
path, data = sys.argv[1], sys.argv[2]
q = json.loads(open(path).read())
n = sum(1 for r in q["runs"] if r.get("data") != data)
if n:
    for r in q["runs"]:
        r["data"] = data
    open(path, "w").write(json.dumps(q, indent=2))
print(f"[bootstrap] queue: {len(q['runs'])} runs, {n} data paths rewritten to {data}")
PY

# --- 1. the ensemble runner: its pid, and its interpreter ---------------------
ENS_PID="$(pgrep -f "run_queue.py .*${ENS_QUEUE}.* run" | head -1)"
if [ -n "$ENS_PID" ]; then
  PY="$(tr '\0' '\n' < "/proc/$ENS_PID/cmdline" 2>/dev/null | head -1)"
  say "ensemble runner: pid $ENS_PID"
else
  say "no ensemble runner found — the benchmark will start immediately"
fi
[ -n "${PY:-}" ] && [ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
[ -n "$PY" ] || fail "no python interpreter found"
say "interpreter = $PY"
"$PY" -c "import torch; print('[bootstrap] torch', torch.__version__, '| cuda', torch.cuda.is_available())" \
  || fail "that interpreter cannot import torch — do not start training with it"

# --- 2. the stride-4 VIS dataset yaml ----------------------------------------
# On dgxanode01 the derived lists live at /workspace/derived, NOT in the repo's
# runs/derived, and there is no Pohang_dataset/data_vis.yaml for `--data vis` to
# resolve -- only the split lists under /workspace/pohang/visible. So the source
# yaml is synthesised here and make_stride_subset is called directly. Striding is
# per-run over unique frame ordinals (stereo pairs stay together), which is why
# this cannot be faked with `sed -n '1~2p'` on the stride-2 list.
DERIVED=/workspace/derived
if [ -f "$DERIVED/data_vis_stride4.yaml" ]; then
  say "$DERIVED/data_vis_stride4.yaml already exists — keeping it"
else
  say "building $DERIVED/data_vis_stride4.yaml"
  [ -f "$DERIVED/data_vis.yaml" ] || printf \
    'train: /workspace/pohang/visible/train.txt\nval: /workspace/pohang/visible/val.txt\ntest: /workspace/pohang/visible/test.txt\nnames:\n  0: ship\n  1: buoy\n' \
    > "$DERIVED/data_vis.yaml"
  "$PY" - <<PY || fail "make_stride_subset failed"
import sys
sys.path.insert(0, 'src')
from uqfusion.data.lists import load_data_yaml
from uqfusion.data.subset import make_stride_subset
print(make_stride_subset(load_data_yaml('$DERIVED/data_vis.yaml'), stride=4, out_dir='$DERIVED'))
PY
fi
cat "$DERIVED/data_vis_stride4.yaml"
TRAIN_LIST="$(awk '/^train:/{print $2}' "$DERIVED/data_vis_stride4.yaml")"
[ -f "$TRAIN_LIST" ] && say "train frames: $(wc -l < "$TRAIN_LIST")  (expect 24070)"

# --- 3. pre-fetch every checkpoint the sweep needs ---------------------------
say "pre-fetching COCO checkpoints (31 variants + the yolo11n AMP probe)"
"$PY" - <<'PY'
from ultralytics.utils.downloads import attempt_download_asset
from pathlib import Path

names = ["yolov8n", "yolov8s", "yolov8m", "yolov8l", "yolov8x",
         "yolov9t", "yolov9s", "yolov9m", "yolov9c", "yolov9e",
         "yolov10n", "yolov10s", "yolov10m", "yolov10b", "yolov10l", "yolov10x",
         "yolo11n", "yolo11s", "yolo11m", "yolo11l", "yolo11x",
         "yolo12n", "yolo12s", "yolo12m", "yolo12l", "yolo12x",
         "yolo26n", "yolo26s", "yolo26m", "yolo26l", "yolo26x"]
missing = []
for name in names:
    f = Path(f"{name}.pt")
    if f.is_file() and f.stat().st_size > 1_000_000:
        print(f"  have  {f.name} ({f.stat().st_size/2**20:.1f} MB)")
        continue
    for attempt in (1, 2, 3):
        try:
            attempt_download_asset(f"{name}.pt")
            print(f"  got   {name}.pt")
            break
        except Exception as exc:
            print(f"  retry {name}.pt ({attempt}/3): {type(exc).__name__}: {exc}")
    else:
        missing.append(name)
print("MISSING:", missing if missing else "none")
PY

# --- 4. the supervisor -------------------------------------------------------
if pgrep -f "supervise_queue.sh .*$QDIR" >/dev/null; then
  say "supervisor already running (pid $(pgrep -f "supervise_queue.sh .*$QDIR" | head -1)) — leaving it alone"
else
  WAIT_ARG=""
  [ -n "$ENS_PID" ] && WAIT_ARG="--wait-pid $ENS_PID"
  say "starting supervisor $WAIT_ARG"
  nohup ./scripts/supervise_queue.sh --queue-dir "$QDIR" --python "$PY" $WAIT_ARG \
      >> "$QDIR/supervisor_stdout.log" 2>&1 &
  sleep 2
  say "supervisor pid $(pgrep -f "supervise_queue.sh .*$QDIR" | head -1)"
fi

# --- 5. the tracker ----------------------------------------------------------
if pgrep -f "queue_xlsx_report.py" >/dev/null; then
  say "tracker already running (pid $(pgrep -f queue_xlsx_report.py | head -1)) — leaving it alone"
else
  TRACK_ARGS="--queue-dir $QDIR"
  [ -f "$ENS_QUEUE/queue.json" ] && TRACK_ARGS="$TRACK_ARGS --queue-dir $ENS_QUEUE"
  say "starting tracker -> $XLSX (every 300 s)"
  # shellcheck disable=SC2086
  nohup "$PY" scripts/queue_xlsx_report.py $TRACK_ARGS --out "$XLSX" --interval 300 \
      >> "$QDIR/tracker_stdout.log" 2>&1 &
  sleep 8
  say "tracker pid $(pgrep -f queue_xlsx_report.py | head -1)"
fi

# --- 6. verification ---------------------------------------------------------
echo
echo "================ VERIFY (paste this block back) ================"
echo "date            : $(date -Is)"
echo "ensemble pid    : ${ENS_PID:-none} $([ -n "${ENS_PID:-}" ] && kill -0 "$ENS_PID" 2>/dev/null && echo '(alive)')"
echo "supervisor pid  : $(pgrep -f "supervise_queue.sh .*$QDIR" | head -1)"
echo "tracker pid     : $(pgrep -f queue_xlsx_report.py | head -1)"
echo "queue           : $("$PY" scripts/queue_remaining.py --queue-dir "$QDIR")"
echo "ensemble queue  : $("$PY" scripts/queue_remaining.py --queue-dir "$ENS_QUEUE" 2>/dev/null)"
echo "workbook        : $(ls -l "$XLSX" 2>/dev/null || echo 'not written yet')"
echo "/dev/shm        : $(df -h /dev/shm | tail -1)"
echo "disk            : $(df -h /workspace | tail -1)"
echo "gpu             : $(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>/dev/null)"
echo "--- supervisor log ---"
tail -5 "$QDIR/supervisor.log" 2>/dev/null
echo "--- tracker log ---"
tail -3 "$QDIR/tracker_stdout.log" 2>/dev/null
echo "================================================================"
