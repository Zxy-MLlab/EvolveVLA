#!/bin/bash
set -eo pipefail

LOG="/root/autodl-tmp/EvolveVLA/results/robotwin_gate3.log"
CHECKPOINT_ROOT="/root/autodl-tmp/EvolveVLA/checkpoints"
EXTRACTED_DIR="${CHECKPOINT_ROOT}/pi05_robotwin"
TOKENIZER="/root/autodl-tmp/paligemma_tokenizer"
HARNESS="/root/autodl-tmp/EvolveVLA/scripts/robotwin_eval_harness.py"
RESULT_JSON="/root/autodl-tmp/EvolveVLA/results/gate3_beat_block_hammer.json"

mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== GATE 3 ROBOTWIN LAUNCH ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

source /etc/network_turbo 2>/dev/null || true
export PIP_CACHE_DIR="/root/autodl-tmp/.cache/pip"
export TMPDIR="/root/autodl-tmp/tmp"
export HF_HOME="/root/autodl-tmp/.cache/huggingface"
export HF_HUB_DISABLE_XET=1

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/robotwin_cu128

if [[ ! -d "$EXTRACTED_DIR" ]]; then
    echo "GATE3_PRECONDITION_NOT_MET: extracted checkpoint directory is missing: $EXTRACTED_DIR"
    exit 2
fi

EXTRACTED_SIZE="$(du -sh "$EXTRACTED_DIR")"
EXTRACTED_BYTES="$(du -sb "$EXTRACTED_DIR" | cut -f1)"
echo "GATE3_EXTRACTED_SIZE=$EXTRACTED_SIZE"
if [[ -z "$EXTRACTED_BYTES" || "$EXTRACTED_BYTES" -le 0 ]]; then
    echo "GATE3_PRECONDITION_NOT_MET: extracted checkpoint directory is empty: $EXTRACTED_DIR"
    exit 2
fi

MODEL_CONFIG="$(find "$EXTRACTED_DIR" -type f -name 'config.json' -print -quit)"
if [[ -z "$MODEL_CONFIG" ]]; then
    echo "GATE3_PRECONDITION_NOT_MET: no LeRobot config.json under $EXTRACTED_DIR"
    exit 2
fi
MODEL_PATH="$(dirname "$MODEL_CONFIG")"

if [[ ! -d "$TOKENIZER" ]] || [[ -z "$(find "$TOKENIZER" -type f -print -quit)" ]]; then
    echo "GATE3_PRECONDITION_NOT_MET: tokenizer path is missing or empty: $TOKENIZER"
    exit 2
fi

if [[ ! -f "$HARNESS" ]]; then
    echo "GATE3_PRECONDITION_NOT_MET: harness is missing: $HARNESS"
    exit 2
fi

echo "GATE3_MODEL_PATH=$MODEL_PATH"
echo "GATE3_TOKENIZER_PATH=$TOKENIZER"
echo "GATE3_TASK=beat_block_hammer"
echo "GATE3_EPISODES=5"
echo "GATE3_EPISODE_STEPS=1200"

set +e
CUDA_VISIBLE_DEVICES=0 \
/root/autodl-tmp/conda_envs/evolvevla_pi312/bin/python \
  "$HARNESS" \
  --model-path "$MODEL_PATH" \
  --tokenizer-path "$TOKENIZER" \
  --sim-python /root/autodl-tmp/conda_envs/robotwin_cu128/bin/python \
  --robotwin-root /root/autodl-tmp/EvolveVLA/robotwin_repo \
  --task beat_block_hammer \
  --episodes 5 \
  --episode-steps 1200 \
  --output "$RESULT_JSON"
HARNESS_RC=$?
set -e

if [[ "$HARNESS_RC" -ne 0 ]]; then
    echo "GATE3_HARNESS_FAILED: exit_code=$HARNESS_RC"
    exit "$HARNESS_RC"
fi

VERDICT_LINE="$(grep -E '^GATE 3 SUCCESS_RATE=[0-9]+/[0-9]+ \([0-9]+([.][0-9]+)?%\)$' "$LOG" | tail -n 1 || true)"
if [[ -z "$VERDICT_LINE" ]]; then
    echo "GATE3_VERDICT_MISSING: harness completed without the required literal verdict"
    exit 5
fi

echo "$VERDICT_LINE"
echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
