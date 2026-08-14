#!/usr/bin/env bash
# One-time environment setup on the JarvisLabs instance, idempotent.
#
# The single load-bearing pin is ultralytics==8.4.90: 8.4.7 reports mAP ~0.034
# higher for identical weights on identical data (docs/phase1-handoff-2026-08-11.md
# §1), so a row measured under any other version cannot join this CSV.
#
# torch is NOT pinned — it comes from the instance template. Two machines already
# in the CSV disagree on torch (2.12.1+cu126 server, 2.7.1+cu118 laptop) and agree
# on the metric, so the template's build is fine as long as CUDA works. What must
# NOT happen is pip silently swapping it: a constraints file freezes whatever
# torch/torchvision the image ships, because a plain PyPI torch wheel is built for
# a newer CUDA than the driver and every run then dies with "driver too old".
set -euo pipefail

ROOT="${1:-/home/uqfusion}"
cd "$ROOT"

echo "== [setup] host"
nvidia-smi || { echo "[setup] FATAL no nvidia-smi"; exit 1; }
echo "vCPU: $(nproc)"; free -g | head -2; df -h "$ROOT" /dev/shm | sed 's/^/  /'

echo "== [setup] system libs (ultralytics + albumentations need libGL in a headless container)"
if ! ldconfig -p | grep -q libGL.so.1; then
  apt-get update -qq
  DEBIAN_FRONTEND=noninteractive apt-get install -y -qq libgl1 libglib2.0-0
fi

echo "== [setup] python env"
python -m pip --version
TORCH_BEFORE="$(python -c 'import torch; print(torch.__version__)')"
python - <<'PY' > /tmp/uq_constraints.txt
import torch, torchvision
# Pin the installed versions with the local build tag STRIPPED. A constraint of
# "torch==2.11.0+cu130" makes pip hunt the index for that exact build, find
# nothing (local versions are never published) and give up with
# ResolutionImpossible. "torch==2.11.0" matches the installed 2.11.0+cu130 under
# PEP 440, so the requirement resolves as already-satisfied and the CUDA build
# on this box is left alone — which is the whole point of the constraint.
print(f"torch=={torch.__version__.split('+')[0]}")
print(f"torchvision=={torchvision.__version__.split('+')[0]}")
PY
cat /tmp/uq_constraints.txt
python -m pip install -q -c /tmp/uq_constraints.txt "ultralytics==8.4.90" "pyyaml>=6"

TORCH_AFTER="$(python -c 'import torch; print(torch.__version__)')"
if [ "$TORCH_BEFORE" != "$TORCH_AFTER" ]; then
  echo "[setup] FATAL pip changed torch $TORCH_BEFORE -> $TORCH_AFTER"; exit 1
fi

echo "== [setup] versions (these land in the CSV)"
python - <<'PY'
import torch, ultralytics, cv2, sys
print("python     ", sys.version.split()[0])
print("torch      ", torch.__version__, "cuda_available", torch.cuda.is_available())
print("device     ", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "NONE")
print("ultralytics", ultralytics.__version__)
print("cv2        ", cv2.__version__, cv2.__file__)
assert ultralytics.__version__ == "8.4.90", "ultralytics must be exactly 8.4.90"
assert torch.cuda.is_available(), "no CUDA device"
PY

# The dataset is verified separately (jarvislabs/remote_verify.py), at the end of
# the upload — this script runs before it, so that a box with no CUDA or no disk
# fails in two minutes instead of after a two-hour transfer.
echo "== [setup] OK"
