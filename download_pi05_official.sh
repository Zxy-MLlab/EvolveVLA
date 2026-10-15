#!/bin/bash
# Selective download of the official RoboCasa365 π0.5 checkpoint.
# Only the INFERENCE set (params + assets + metadata) = 12.4GB.
# Excludes train_state/ (32.3GB optimizer state — not needed for eval).
set -e
exec > /root/autodl-tmp/EvolveVLA/download_pi05_official.log 2>&1

source /etc/network_turbo 2>/dev/null
source /root/miniconda3/bin/activate
conda activate evolvevla
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

echo "$(date) Starting SELECTIVE π0.5 checkpoint download (12.4GB inference set)"

mkdir -p /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365

BASE='pi05_pretrain_human300/multitask_learning/75000'

echo "=== Downloading params/ + assets/ + metadata (excluding train_state) ==="
hf download robocasa/robocasa365_checkpoints \
  --include "$BASE/params/**" "$BASE/assets/**" "$BASE/_CHECKPOINT_METADATA" \
  --local-dir /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365

echo "=== Verifying ==="
du -sh /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365/$BASE/ 2>/dev/null
echo "--- params dir ---"
du -sh /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365/$BASE/params/ 2>/dev/null
echo "--- confirm train_state NOT present ---"
ls /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365/$BASE/ 2>/dev/null
echo "--- incomplete files ---"
find /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365/ -name "*.incomplete" 2>/dev/null | wc -l

echo "=== Disk ==="
df -h /
df -h /root/autodl-tmp/
echo "PI05 SELECTIVE DOWNLOAD COMPLETE $(date)"
