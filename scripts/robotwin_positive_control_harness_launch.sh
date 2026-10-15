#!/bin/bash
set -eo pipefail
LOG="/root/autodl-tmp/EvolveVLA/results/robotwin_positive_control_harness.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== ROBOTWIN SUCCESS-SIGNAL EXTENDED POSITIVE CONTROL HARNESS ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
source /etc/network_turbo 2>/dev/null || true
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TMPDIR=/root/autodl-tmp/tmp
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/robotwin_cu128
echo "=== Preflight: CUDA cap ==="
python -c "import torch; cap=torch.cuda.get_device_capability(0); print(f'COMPUTE_CAP=sm_{cap[0]*10+cap[1]}')"
echo "=== Run extended positive control harness ==="
cd /root/autodl-tmp/EvolveVLA/robotwin_repo
python /root/autodl-tmp/EvolveVLA/scripts/robotwin_positive_control_harness.py 2>&1
echo ""
echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "POSITIVE_CONTROL_HARNESS_EXIT=0"
