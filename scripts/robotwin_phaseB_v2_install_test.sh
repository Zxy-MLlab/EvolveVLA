#!/bin/bash
set -eo pipefail

# Logging
LOG="/root/autodl-tmp/EvolveVLA/results/robotwin_phaseB_v2.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== PHASE B v2: FRESH ENV (robotwin_cu128) + smoke ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

# Proxy + C006 env vars
source /etc/network_turbo 2>/dev/null || true
export CUDA_VISIBLE_DEVICES=0
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TMPDIR=/root/autodl-tmp/tmp
export CONDA_PKGS_DIRS=/root/autodl-tmp/.conda/pkgs
mkdir -p "$PIP_CACHE_DIR" "$TMPDIR" "$CONDA_PKGS_DIRS"

source /root/miniconda3/etc/profile.d/conda.sh

# ====== STEP 1: Create fresh env ======
echo "=== STEP 1: Create robotwin_cu128 env (py3.10) ==="
ENV_PATH="/root/autodl-tmp/conda_envs/robotwin_cu128"
if [ -d "$ENV_PATH" ]; then
    echo "Env already exists, removing for clean build"
    conda env remove -p "$ENV_PATH" -y 2>/dev/null || rm -rf "$ENV_PATH"
fi
conda create -p "$ENV_PATH" python=3.10 -y
conda activate "$ENV_PATH"

# ====== STEP 2: Install torch cu128 ======
echo "=== STEP 2: Install torch cu128 ==="
echo "TARGET_TORCH=2.11.0+cu128 (available for Python 3.10)"
# Try torch 2.11.0 first; if unavailable for cp310, fall back
pip install torch==2.11.0 torchvision --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir 2>&1 || {
    echo "torch 2.11.0 not available for cp310, trying latest cu128..."
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128 --no-cache-dir 2>&1
}

python -c "
import torch
print(f'TORCH_VERSION={torch.__version__}')
print(f'CUDA_VERSION={torch.version.cuda}')
cap = torch.cuda.get_device_capability(0)
print(f'COMPUTE_CAP=sm_{cap[0]*10+cap[1]}')
x = torch.randn(4, device='cuda:0')
y = torch.sign(x)
print('TORCH_SIGN_CUDA=OK')
"

# ====== STEP 3: Install SAPIEN + RoboTwin deps ======
echo "=== STEP 3: Install SAPIEN + RoboTwin runtime deps ==="
pip install sapien==3.0.3 -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir 2>&1
pip install \
    numpy==1.26.4 \
    warp-lang \
    mplib \
    toppra \
    gymnasium==0.29.1 \
    pyyaml \
    scipy \
    transforms3d \
    opencv-python==4.10.0.84 \
    opencv-python-headless==4.10.0.84 \
    trimesh==4.4.3 \
    open3d==0.18.0 \
    imageio==2.34.2 \
    'pydantic>=2.5' \
    zarr \
    'h5py>=3.8' \
    'websockets>=13.0' \
    'msgpack>=1.0.8' \
    msgpack-numpy \
    'rich>=13.7' \
    'pyglet<2' \
    matplotlib \
    -i https://pypi.tuna.tsinghua.edu.cn/simple \
    --no-cache-dir 2>&1

python -c "
import sapien; print(f'SAPIEN={sapien.__version__}')
import warp; print(f'WARP={warp.config.version}')
print('SAPIEN_DEPS=OK')
"

# ====== STEP 4: Install CuRobo ======
echo "=== STEP 4: Install CuRobo ==="
echo "CUROBO_TORCH_REQUIREMENT=torch (no pinned version in pyproject.toml)"
pip install 'setuptools>=45' 'setuptools_scm>=6.2' wheel ninja pybind11 \
    -i https://pypi.tuna.tsinghua.edu.cn/simple --no-cache-dir 2>&1
cd /root/autodl-tmp/EvolveVLA/robotwin_repo
pip install -e envs/curobo --no-build-isolation --no-cache-dir 2>&1
echo "CUROBO_INSTALL=DONE"

# ====== STEP 5: Import verification (the exact chain that failed) ======
echo "=== STEP 5: Import verification ==="
cd /root/autodl-tmp/EvolveVLA/robotwin_repo
python -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('SAPIEN_LOG_LEVEL', 'off')

import torch
cap = torch.cuda.get_device_capability(0)
print(f'VERIFY_compute_cap=({cap[0]},{cap[1]})')

x = torch.randn(4, device='cuda:0')
y = torch.sign(x)
print('VERIFY_torch_sign=OK')

from curobo.types.math import *
print('VERIFY_curobo_math=OK')

from curobo.wrap.reacher.motion_gen import MotionGen
print('VERIFY_MotionGen=OK')

from envs.robot.planner import CuroboPlanner
print('VERIFY_CuroboPlanner=OK')

print('ALL_IMPORTS=PASS')
"

# ====== STEP 6: Phase B smoke ======
echo "=== STEP 6: Phase B smoke — ONE task ==="
cd /root/autodl-tmp/EvolveVLA/robotwin_repo
python scripts/robotwin_phaseB_smoke.py 2>&1

echo ""
echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "EXIT_CODE=0"
