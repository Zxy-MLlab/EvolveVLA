#!/bin/bash
set -euo pipefail

LOG="/root/autodl-tmp/EvolveVLA/results/robotwin_stage1_dump.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== ROBOTWIN STAGE1 ORBAX DUMP ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

source /etc/network_turbo 2>/dev/null || true
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TMPDIR=/root/autodl-tmp/tmp

mkdir -p /root/autodl-tmp/EvolveVLA/checkpoints/convert_stageA

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/scratch_orbax

export JAX_PLATFORMS=cpu
export CUDA_VISIBLE_DEVICES=""

python /root/autodl-tmp/EvolveVLA/scripts/robotwin_stage1_dump.py

echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
