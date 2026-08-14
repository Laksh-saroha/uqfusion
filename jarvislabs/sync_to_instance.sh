#!/usr/bin/env bash
# Push code + the 16.4 GB VIS payload to a JarvisLabs instance, resumably.
#
#   bash jarvislabs/sync_to_instance.sh <machine_id>
#
# Why not `jl upload`: it is scp, and scp of 59k files over a ~2 MB/s uplink is
# both slow (per-file round trips) and all-or-nothing. This streams tar over ssh in
# ~1 GiB groups and drops a marker per finished group, so an interruption replays
# one group (~8 min), not two hours. Re-running is always safe and always cheap.
#
# Every remote path is derived from RROOT, which must match the roots baked into
# the uploaded split lists (jarvislabs/build_payload.py --remote-root).
set -euo pipefail

MID="${1:?usage: sync_to_instance.sh <machine_id>}"
REPO="${REPO:-/a/Uncertain}"
RROOT="${RROOT:-/home/uqfusion}"
JL="${JL:-jl}"

PAYLOAD="$REPO/jarvislabs/payload"
MANIFEST="$PAYLOAD/manifest"
LOCAL_VIS="$REPO/Pohang_dataset/visible"
RVIS="$RROOT/data/pohang/visible"

[ -d "$MANIFEST" ] || { echo "no manifest — run: python jarvislabs/build_payload.py"; exit 1; }

echo "== resolving ssh for machine $MID"
# SSH_LINE can be passed in (the CLI's rich renderer dies on some non-tty pipes).
SSH_LINE="${SSH_LINE:-$("$JL" ssh "$MID" --print-command | tr -d '\r' | grep -m1 '^ssh ')}"
echo "   $SSH_LINE"
read -r -a SSH <<< "$SSH_LINE"
# Insert non-interactive hardening before the target host (last token).
HOST="${SSH[-1]}"
unset 'SSH[-1]'
SSH+=(-o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15
      -o ServerAliveCountMax=4 -o StrictHostKeyChecking=no
      -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR "$HOST")

rsh() { "${SSH[@]}" "$@"; }

send_tree() {  # send_tree <local dir> <remote dir> [tar args...]
  local src="$1" dst="$2"; shift 2
  rsh "mkdir -p '$dst'"
  tar -C "$src" --exclude='__pycache__' --exclude='*.pyc' --exclude='*.cache' \
      -cf - "$@" | rsh "tar -xf - -C '$dst'"
}

echo "== 1/5 code (src, scripts, config)"
send_tree "$REPO" "$RROOT" config.yaml src scripts

echo "== 2/5 dataset yaml + split lists"
send_tree "$PAYLOAD/repo" "$RROOT" .

echo "== 3/6 remote scripts + upload manifest"
send_tree "$REPO/jarvislabs" "$RROOT/jarvislabs" remote_setup.sh remote_run_26x.sh remote_verify.py
tar -C "$MANIFEST" -cf - expect.tsv | rsh "tar -xf - -C '$RROOT/jarvislabs'"
rsh "chmod +x '$RROOT'/jarvislabs/*.sh"

# Before the two-hour transfer, not after: this is what catches a box with no
# CUDA, a too-small disk, or a pip that wants to swap the torch build.
echo "== 4/6 environment setup + preflight"
rsh "bash '$RROOT/jarvislabs/remote_setup.sh' '$RROOT'"

echo "== 5/6 pretrained weights (yolo26x.pt, 119 MB)"
if rsh "test -s '$RROOT/yolo26x.pt' && [ \$(stat -c%s '$RROOT/yolo26x.pt') -eq $(stat -c%s "$REPO/yolo26x.pt") ]"; then
  echo "   already present, skipping"
else
  tar -C "$REPO" -cf - yolo26x.pt | rsh "tar -xf - -C '$RROOT'"
fi

echo "== 6/6 images + labels (16.4 GB)"
rsh "mkdir -p '$RVIS' '$RROOT/.upload'"
DONE_LIST="$(rsh "ls '$RROOT/.upload' 2>/dev/null" | tr -d '\r')"
groups=("$MANIFEST"/group_*.txt)
t_start=$(date +%s)
for f in "${groups[@]}"; do
  g="$(basename "$f" .txt)"
  n="$(wc -l < "$f")"
  if grep -qx "$g.done" <<< "$DONE_LIST"; then
    echo "   $g  skip (already uploaded)"
    continue
  fi
  for attempt in 1 2 3; do
    echo "   $g  $n files, attempt $attempt ..."
    if tar --totals -C "$LOCAL_VIS" -cf - -T "$f" \
         | rsh "tar -xf - -C '$RVIS' && touch '$RROOT/.upload/$g.done'"; then
      break
    fi
    echo "   $g  FAILED attempt $attempt"
    [ "$attempt" = 3 ] && { echo "giving up on $g"; exit 1; }
    sleep 20
  done
done
echo "== upload wall time: $(( ($(date +%s) - t_start) / 60 )) min"

echo "== verifying on the instance"
rsh "cd '$RROOT' && PYTHONPATH='$RROOT/src' python jarvislabs/remote_verify.py --root '$RROOT'"
