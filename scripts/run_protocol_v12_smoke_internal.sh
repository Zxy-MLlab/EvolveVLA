#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/EvolveVLA
ROBOTWIN_ROOT=$ROOT/robotwin_repo_ba0fd21
PY=/root/autodl-tmp/conda_envs/scratch_orbax/bin/python
CKPT=$ROOT/checkpoints/pi05_robotwin/RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999
BASE=$ROOT/results/pi05_official_orbax_baseline/protocol_v12_smoke_10_oidn_internal

export CUDA_VISIBLE_DEVICES=0
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache
export PYTHONUNBUFFERED=1
export PYTHONPATH=$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_05/openpi/src
export ROBOTWIN_OPENPI_SRC=$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_05/openpi/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export ROBOTWIN_RENDER_BACKEND=rt
export ROBOTWIN_RT_DENOISER=oidn
export ROBOTWIN_OIDN_RUNTIME=$ROOT/third_party/oidn-2.5.1/lib

for setting in demo_clean demo_randomized; do
  for task in adjust_bottle beat_block_hammer click_bell; do
    "$PY" "$ROOT/scripts/robotwin_public_openpi_harness.py" \
      --checkpoint "$CKPT" --task "$task" --task-config "$setting" \
      --episodes 10 --seed 100000 --prompt-prefix none --normalization quantile \
      --policy-seed 0 --noise-source internal --episode-steps 2000 --resume \
      --robotwin-root "$ROBOTWIN_ROOT" \
      --output "$BASE/${setting#demo_}/$task/result.json"
  done
done
