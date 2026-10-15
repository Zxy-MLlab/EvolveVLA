#!/bin/bash
set -e
exec > /root/autodl-tmp/EvolveVLA/install_final.log 2>&1

source /etc/network_turbo 2>/dev/null
source /root/miniconda3/bin/activate
conda activate evolvevla
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_HOME=/root/autodl-tmp/.cache/huggingface

echo "$(date) Starting final package install"

echo "=== Step 1: Downgrade transformers ==="
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ 'transformers==4.53.2'
python -c "import transformers; print(f'transformers {transformers.__version__}')"
echo "STEP 1 DONE"

echo "=== Step 2: Install robocasa ==="
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ robocasa
python -c "import robocasa; print('robocasa OK')"
echo "STEP 2 DONE"

echo "=== Step 3: Install lerobot ==="
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ lerobot
python -c "import lerobot; print('lerobot OK')"
echo "STEP 3 DONE"

echo "=== Step 4: Verify all ==="
python -c "
import torch, mujoco, robosuite, robocasa, lerobot, peft, transformers
print('ALL IMPORTS PASS')
print(f'torch={torch.__version__}')
print(f'transformers={transformers.__version__}')
print(f'robosuite={robosuite.__version__}')
"
echo "STEP 4 DONE"

echo "=== Step 5: Disk ==="
df -h /
pip cache purge
df -h /
echo "ALL STEPS COMPLETE $(date)"
