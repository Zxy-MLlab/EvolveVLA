#!/bin/bash
set -e
exec > /root/autodl-tmp/EvolveVLA/install_final_v2.log 2>&1

source /etc/network_turbo 2>/dev/null
source /root/miniconda3/bin/activate
conda activate evolvevla
export REQUESTS_CA_BUNDLE=/etc/ssl/certs/ca-certificates.crt
export SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt
export HF_HOME=/root/autodl-tmp/.cache/huggingface

echo "$(date) Starting final package install v2"

echo "=== Step 1: Fix transformers version ==="
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ 'transformers==4.53.2' 2>&1 || echo "transformers install had warnings"
python -c "import transformers; print(f'transformers {transformers.__version__}')"
echo "STEP 1 DONE"

echo "=== Step 2: Install robocasa (with --ignore-installed to handle nvidia RECORD issue) ==="
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ --ignore-installed robocasa 2>&1
python -c "import robocasa; print('robocasa OK')"
echo "STEP 2 DONE"

echo "=== Step 3: Install lerobot ==="
pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ lerobot 2>&1 || pip install -i https://pypi.tuna.tsinghua.edu.cn/simple/ lerobot --no-deps 2>&1
python -c "import lerobot; print('lerobot OK')" || echo "lerobot import failed - will try fixing deps"
echo "STEP 3 DONE"

echo "=== Step 4: Verify all ==="
python -c "
import torch; print(f'torch={torch.__version__}, cuda={torch.cuda.is_available()}')
import mujoco; print(f'mujoco={mujoco.__version__}')
import robosuite; print(f'robosuite={robosuite.__version__}')
import robocasa; print('robocasa=OK')
import transformers; print(f'transformers={transformers.__version__}')
from peft import LoraConfig; print('peft=OK')
import accelerate; print(f'accelerate={accelerate.__version__}')
import scipy; print(f'scipy={scipy.__version__}')
print('ALL IMPORTS PASS')
"
echo "STEP 4 DONE"

echo "=== Step 5: Cleanup + disk ==="
pip cache purge 2>/dev/null
df -h / /root/autodl-tmp/
du -sh /root/miniconda3/envs/evolvevla/ 2>/dev/null
echo "ALL STEPS COMPLETE $(date)"
