#!/bin/bash
set -eo pipefail

LOG="/root/autodl-tmp/EvolveVLA/results/sidneyxie_load_test.log"
SCRIPT="/root/autodl-tmp/EvolveVLA/scripts/sidneyxie_load_test.py"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== SIDNEYXIE PI05 COMMUNITY CHECKPOINT LOAD TEST ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

source /etc/network_turbo 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=""
export XDG_CACHE_HOME="/root/autodl-tmp/.cache"
export HF_HOME="/root/autodl-tmp/.cache/huggingface"
export TRANSFORMERS_CACHE="/root/autodl-tmp/.cache/huggingface"
export TORCH_HOME="/root/autodl-tmp/.cache/torch"
export PIP_CACHE_DIR="/root/autodl-tmp/.cache/pip"
export CONDA_PKGS_DIRS="/root/autodl-tmp/.conda/pkgs"
export TMPDIR="/root/autodl-tmp/tmp"
export HF_HUB_DISABLE_XET="1"
unset HF_TOKEN HUGGING_FACE_HUB_TOKEN HUGGINGFACEHUB_API_TOKEN

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/evolvevla_pi312

set +e
python "$SCRIPT"
status=$?
set -e

final_line="$(grep '^SIDNEYXIE_FULLY_LOADABLE=' "$LOG" | tail -n 1 || true)"
if [ -n "$final_line" ]; then
    printf '%s\n' "$final_line"
else
    printf '%s\n' "SIDNEYXIE_FULLY_LOADABLE=False"
fi

echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
exit "$status"
