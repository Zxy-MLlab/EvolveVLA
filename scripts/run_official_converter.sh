#!/bin/bash
set -eo pipefail
LOG="/root/autodl-tmp/EvolveVLA/results/official_converter.log"
CHECKPOINT="/root/autodl-tmp/EvolveVLA/checkpoints/pi05_robotwin/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999"
OUTPUT="/root/autodl-tmp/EvolveVLA/checkpoints/convert_official"
OPENPI="/root/autodl-tmp/openpi_feasibility_src"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "=== OFFICIAL OPENPI CONVERTER ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
source /etc/network_turbo 2>/dev/null || true
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip TMPDIR=/root/autodl-tmp/tmp
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR"
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/scratch_orbax
python -c 'import torch' >/dev/null 2>&1 || \
  pip install torch --index-url https://download.pytorch.org/whl/cpu -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir
python -c 'import openpi' >/dev/null 2>&1 || \
  pip install -e "$OPENPI" --no-cache-dir -i https://pypi.tuna.tsinghua.edu.cn/simple
set +e
python "$OPENPI/examples/convert_jax_model_to_pytorch.py" \
  --checkpoint-dir "$CHECKPOINT" \
  --config-name pi05_aloha \
  --output-path "$OUTPUT" \
  --precision bfloat16
CONVERT_EXIT=$?
set -e
echo "OFFICIAL_CONVERT_EXIT=$CONVERT_EXIT"
if [ -d "$OUTPUT" ]; then
  ls -la "$OUTPUT"
  [ ! -f "$OUTPUT/model.safetensors" ] || sha256sum "$OUTPUT/model.safetensors"
fi
exit "$CONVERT_EXIT"
