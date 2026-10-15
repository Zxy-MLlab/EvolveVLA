#!/bin/bash
set -e

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/robotwin_cu128
source /etc/network_turbo

export PIP_INDEX_URL=https://pypi.tuna.tsinghua.edu.cn/simple
export PIP_TRUSTED_HOST=pypi.tuna.tsinghua.edu.cn

echo "=== PHASE B v3: CuRobo installation and gate test ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

echo "=== Install numpy-quaternion ==="
python -m pip install numpy-quaternion
python -c "import quaternion; print('NUMPY_QUATERNION_IMPORT=PASS')"

echo "=== Install CuRobo ==="
cd /root/autodl-tmp/EvolveVLA/robotwin_repo
python -m pip install -e envs/curobo
python -c "import curobo; print('CUROBO_IMPORT=PASS')"

echo "=== CuRobo CUDA Pose kernel test ==="
python -c "
import torch
from curobo.types.math import Pose

capability = torch.cuda.get_device_capability(0)
print(f'CUDA_DEVICE_CAPABILITY=sm_{capability[0]}{capability[1]}')
position = torch.tensor([[0.0, 0.0, 0.0]], device='cuda:0')
quaternion = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device='cuda:0')
pose = Pose(position=position, quaternion=quaternion)
print(f'CUROBO_POSE_DEVICE={pose.position.device}')
print('CUROBO_CUDA_POSE_TEST=PASS')
"

echo "=== Warp verification ==="
python -c "
import warp as wp
wp.init()
print(f'WARP_VERSION={wp.config.version}')
print('WARP_VERIFICATION=PASS')
"

echo "=== SAPIEN verification ==="
python -c "
import sapien
print(f'SAPIEN_VERSION={sapien.__version__}')
print('SAPIEN_VERIFICATION=PASS')
"

echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "PHASE_B_V3_GATE_VERDICT=PASS"
