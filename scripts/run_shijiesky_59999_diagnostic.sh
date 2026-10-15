#!/usr/bin/env bash
set -euo pipefail

# Pre-registered diagnostic only.  This is a third-party 60k-step re-train,
# not the unpublished RoboTwin leaderboard checkpoint and not Generation-0.

ROOT=/root/autodl-tmp/EvolveVLA
ROBOTWIN_ROOT=$ROOT/robotwin_repo_ba0fd21
PY=/root/autodl-tmp/conda_envs/scratch_orbax/bin/python
CKPT=$ROOT/checkpoints/third_party_shijiesky_pi05_robotwin/checkpoints/59999
RESULTS=$ROOT/results/pi05_official_orbax_baseline
TASKS=lift_pot,place_dual_shoes,place_mouse_pad,place_cans_plasticbox,click_bell

if [[ ! -f "$CKPT/params/manifest.ocdbt" ]] || [[ ! -d "$CKPT/params/ocdbt.process_0/d" ]]; then
  echo "Incomplete diagnostic checkpoint: $CKPT" >&2
  exit 1
fi
if find "$ROOT/checkpoints/third_party_shijiesky_pi05_robotwin/.cache" \
    -type f -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
  echo "Diagnostic checkpoint still has incomplete Hugging Face downloads" >&2
  exit 1
fi

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
    --tasks "$TASKS" \
    --prompt-prefix none \
    --normalization quantile \
    --policy-seed 0 \
    --noise-source jax \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --output-root "$output_root" \
    --resume
}

run_setting demo_clean "$RESULTS/shijiesky_59999_diagnostic_clean_10"
run_setting demo_randomized "$RESULTS/shijiesky_59999_diagnostic_randomized_10"

"$PY" "$ROOT/scripts/summarize_pi05_diagnostic.py" \
  --candidate-clean "$RESULTS/shijiesky_59999_diagnostic_clean_10" \
  --candidate-randomized "$RESULTS/shijiesky_59999_diagnostic_randomized_10" \
  --candidate-label shijiesky_seed42_59999 \
  --candidate-warning "ShijieSKY/pi05-robotwin is an independently trained seed-42 checkpoint; it is not the RoboTwin Team leaderboard artifact." \
  --baseline-clean "$RESULTS/protocol_v11_ba0fd21_full_clean_10_oidn_jax0" \
  --baseline-randomized "$RESULTS/protocol_v11_ba0fd21_full_randomized_10_oidn_jax0" \
  --output "$RESULTS/shijiesky_59999_diagnostic_summary.json"
