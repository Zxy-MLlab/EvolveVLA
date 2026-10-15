#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/EvolveVLA
LOG="$ROOT/results/sidneyxie_sim_eval.log"
RESULT="$ROOT/results/sidneyxie_sim_eval.json"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"
exec >> "$LOG" 2>&1

echo "=== SIDNEYXIE ROBOTWIN SIM-EVAL ==="
echo "LABEL=community checkpoint (SidneyXie) - NOT official baseline"
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

source /etc/network_turbo 2>/dev/null || true

export CUDA_VISIBLE_DEVICES=0
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TMPDIR=/root/autodl-tmp/tmp
export XDG_CACHE_HOME=/root/autodl-tmp/.cache
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HUGGINGFACE_HUB_CACHE=/root/autodl-tmp/.cache/huggingface/hub
export TRANSFORMERS_CACHE=/root/autodl-tmp/.cache/huggingface/transformers
export TORCH_HOME=/root/autodl-tmp/.cache/torch
export MPLCONFIGDIR=/root/autodl-tmp/.cache/matplotlib
export TOKENIZERS_PARALLELISM=false
export SAPIEN_LOG_LEVEL=off

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/evolvevla_pi312
cd "$ROOT"

if python scripts/sidneyxie_sim_eval.py \
  --model-path /root/autodl-tmp/EvolveVLA/checkpoints/sidneyxie_pi05_robotwin \
  --task beat_block_hammer \
  --episodes 5 \
  --episode-steps 1200 \
  --sim-python /root/autodl-tmp/conda_envs/robotwin_cu128/bin/python \
  --robotwin-root /root/autodl-tmp/EvolveVLA/robotwin_repo \
  --tokenizer-path /root/autodl-tmp/paligemma_tokenizer \
  --task-config demo_clean \
  --output "$RESULT"
then
  code=0
else
  code=$?
fi

echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
if grep -q "Traceback" "$LOG"; then
  code=1
fi
exit "$code"
