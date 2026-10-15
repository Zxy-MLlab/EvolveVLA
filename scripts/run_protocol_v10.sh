#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/EvolveVLA
PY=/root/autodl-tmp/conda_envs/openpi_public/bin/python
CKPT=$ROOT/checkpoints/pi05_robotwin2_public
RESULTS=$ROOT/results/pi05_public_baseline

export CUDA_VISIBLE_DEVICES=0
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache
export PYTHONUNBUFFERED=1
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
    --noise-source torch \
    --output-root "$output_root" \
    --resume
}

run_setting demo_clean "$RESULTS/protocol_v10_full_clean_10_oidn_torch0" || \
  echo "Clean sweep stopped incomplete; rerun this script to resume."
run_setting demo_randomized "$RESULTS/protocol_v10_full_randomized_10_oidn_torch0" || \
  echo "Randomized sweep stopped incomplete; rerun this script to resume."
