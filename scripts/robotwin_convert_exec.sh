#!/usr/bin/env bash
set -euo pipefail

LOG="/root/autodl-tmp/EvolveVLA/results/robotwin_convert_exec.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== ROBOTWIN STAGE2 PI05 CONVERSION ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

source /etc/network_turbo 2>/dev/null || true
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TMPDIR=/root/autodl-tmp/tmp
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export TRANSFORMERS_CACHE=/root/autodl-tmp/.cache/huggingface
export HF_DATASETS_CACHE=/root/autodl-tmp/.cache/huggingface/datasets

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/evolvevla_pi312

set +e
/root/autodl-tmp/conda_envs/evolvevla_pi312/bin/python \
  /root/autodl-tmp/EvolveVLA/scripts/robotwin_convert_exec.py "$@"
RC=$?
set -e

echo "=== CONVERSION FINAL LINES ==="
grep -E '^(CONVERT_OUTPUT_PATH|CONVERT_OUTPUT_BYTES|CONVERT_OUTPUT_SHA256|PARITY_OK|LEROBOT_LOAD|CONVERT_EXIT)=' "$LOG" | tail -n 6 || true
echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
exit "$RC"
