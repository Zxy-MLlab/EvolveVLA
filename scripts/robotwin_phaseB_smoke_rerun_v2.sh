#!/bin/bash
set -eo pipefail

LOG="/root/autodl-tmp/EvolveVLA/results/robotwin_phaseB_smoke_rerun_v2.log"
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1

echo "=== PHASE B SMOKE RE-RUN v2 (pin warp-lang==1.12.0) ==="
echo "START_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"

source /etc/network_turbo 2>/dev/null || true
export PIP_CACHE_DIR=/root/autodl-tmp/.cache/pip
export TMPDIR=/root/autodl-tmp/tmp
MIRROR="https://pypi.tuna.tsinghua.edu.cn/simple"

source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/autodl-tmp/conda_envs/robotwin_cu128

echo "=== STEP 1: Pin warp-lang==1.12.0 ==="
pip install "warp-lang==1.12.0" -i "$MIRROR" --no-cache-dir 2>&1
python -c "
import warp
print(f'WARP_VERSION={warp.config.version}')
import warp.torch
print('WARP_TORCH_SUBMODULE=OK')
"

echo "=== STEP 2: Confirm CUDA cap ==="
python -c "
import torch
cap = torch.cuda.get_device_capability(0)
print(f'COMPUTE_CAP=sm_{cap[0]*10+cap[1]} torch={torch.__version__}')
"

echo "=== STEP 3: Verify the curobo warp.torch path imports ==="
cd /root/autodl-tmp/EvolveVLA/robotwin_repo
python -c "
import sys, os
sys.path.insert(0, '.')
os.environ.setdefault('SAPIEN_LOG_LEVEL', 'off')
import warp, warp.torch
from curobo.geom.sdf.world_mesh import WorldMeshCollision
print('WORLD_MESH_COLLISION_IMPORT=OK')
from envs.robot.planner import CuroboPlanner
print('CUROBOPLANNER_IMPORT=OK')
"

echo "=== STEP 4: Phase B smoke — ONE Aloha-AgileX joint task ==="
cd /root/autodl-tmp/EvolveVLA/robotwin_repo
python scripts/robotwin_phaseB_smoke.py 2>&1

echo ""
echo "END_TIME=$(date -u '+%Y-%m-%dT%H:%M:%SZ')"
echo "PHASE_B_SMOKE_RERUN_V2_EXIT=0"
