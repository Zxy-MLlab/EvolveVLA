#!/bin/bash
set -eo pipefail

LOG="/root/autodl-tmp/EvolveVLA/results/robotwin_phaseB_smoke_rerun.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== PHASE B SMOKE RE-RUN (robotwin_cu128, after curobo yml fix) ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

source /etc/network_turbo 2>/dev/null || true
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TMPDIR=/root/autodl-tmp/tmp

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/robotwin_cu128

echo "=== Pre-flight: env + CUDA cap ==="
python -c "
import torch
print(f'TORCH={torch.__version__} CUDA={torch.version.cuda}')
cap = torch.cuda.get_device_capability(0)
print(f'COMPUTE_CAP=sm_{cap[0]*10+cap[1]}')
"

echo "=== Pre-flight: curobo yml presence (report only, do not create) ==="
ls -la /root/autodl-tmp/EvolveVLA/robotwin_repo/assets/embodiments/aloha-agilex/ | grep -i curobo || echo "NO_CUROBO_YML_FOUND"

echo "=== Phase B smoke — ONE Aloha-AgileX joint task ==="
cd /root/autodl-tmp/EvolveVLA/robotwin_repo
python scripts/robotwin_phaseB_smoke.py 2>&1

echo ""
echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "PHASE_B_SMOKE_RERUN_EXIT=0"
