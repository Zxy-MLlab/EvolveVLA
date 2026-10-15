#!/bin/bash
set -eo pipefail

LOG="/root/autodl-tmp/EvolveVLA/results/gate3_single_forward.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== GATE 3 SINGLE-FORWARD PRE-GATE ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

source /etc/network_turbo 2>/dev/null || true
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TMPDIR=/root/autodl-tmp/tmp
export XDG_CACHE_HOME=/root/autodl-tmp/.cache
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/huggingface/datasets
export TRANSFORMERS_CACHE=/root/autodl-tmp/.cache/huggingface/transformers

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/evolvevla_pi312

set +e
python /root/autodl-tmp/EvolveVLA/scripts/gate3_single_forward.py
STATUS=$?
set -e

echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "GATE3_SINGLE_FORWARD_WRAPPER_EXIT=$STATUS"
exit "$STATUS"
