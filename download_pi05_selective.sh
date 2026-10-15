#!/bin/bash
# Selective download of the official RoboCasa365 π0.5 checkpoint.
# Only inference files (params/ + assets/ + metadata) = ~12.4GB.
# Excludes train_state/ (32.3GB optimizer state, unnecessary for evaluation).
set -e
exec > /root/autodl-tmp/EvolveVLA/download_pi05_selective.log 2>&1

source /etc/network_turbo 2>/dev/null
source /root/miniconda3/bin/activate
conda activate evolvevla
export HF_HOME=/root/autodl-tmp/.cache/huggingface
export HF_HUB_DISABLE_XET=1
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

echo "$(date) Starting SELECTIVE pi05 download (params + assets, ~12.4GB)"

mkdir -p /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365

# Download only the inference files — exclude train_state/
hf download robocasa/robocasa365_checkpoints \
  --include 'pi05_pretrain_human300/multitask_learning/75000/params/**' \
            'pi05_pretrain_human300/multitask_learning/75000/assets/**' \
            'pi05_pretrain_human300/multitask_learning/75000/_CHECKPOINT_METADATA' \
  --local-dir /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365

echo "=== Verifying download size ==="
du -sh /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365/pi05_pretrain_human300/ 2>/dev/null
echo "=== Checking no train_state was downloaded ==="
test -d /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000/train_state && echo "WARNING: train_state present" || echo "GOOD: no train_state"
echo "=== Key files present ==="
ls -la /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365/pi05_pretrain_human300/multitask_learning/75000/ 2>/dev/null
find /root/autodl-tmp/EvolveVLA/checkpoints/robocasa365/pi05_pretrain_human300/ -name "*.incomplete" 2>/dev/null | wc -l

echo "=== Disk ==="
df -h /root/autodl-tmp/
echo "PI05 SELECTIVE DOWNLOAD COMPLETE $(date)"
