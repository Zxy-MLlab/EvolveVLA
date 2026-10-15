#!/usr/bin/env bash
set -euo pipefail

# Public Hugging Face PyTorch artifact (motus-robotics/pi0.5_robotwin2).
# Keep this result tree separate from the 29999 Orbax reproduction.
ROOT=/root/autodl-tmp/EvolveVLA
ROBOTWIN_ROOT=$ROOT/robotwin_repo_ba0fd21
PY=/root/autodl-tmp/conda_envs/scratch_orbax/bin/python
CKPT=$ROOT/checkpoints/pi05_robotwin2_public
RESULTS=$ROOT/results/pi05_official_orbax_baseline

export CUDA_VISIBLE_DEVICES=0
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache
export PYTHONUNBUFFERED=1
export PYTHONPATH=$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_05/openpi/src
export ROBOTWIN_OPENPI_SRC=$ROBOTWIN_ROOT/XPolicyLab/policy/Pi_05/openpi/src
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export ROBOTWIN_RENDER_BACKEND=rt
export ROBOTWIN_RT_DENOISER=oidn
export ROBOTWIN_OIDN_RUNTIME=$ROOT/third_party/oidn-2.5.1/lib

run_setting() {
  local task_config=$1
  local output_root=$2
  "$PY" "$ROOT/scripts/run_public_openpi_parallel.py" \
    --checkpoint "$CKPT" \
    --task-config "$task_config" \
    --episodes 10 \
    --seed 100000 \
    --workers 1 \
    --prompt-prefix none \
    --normalization quantile \
    --policy-seed 0 \
    --noise-source internal \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --output-root "$output_root" \
    --resume
}

run_setting demo_clean "$RESULTS/protocol_public_hf_clean_10_oidn_internal"
run_setting demo_randomized "$RESULTS/protocol_public_hf_randomized_10_oidn_internal"
