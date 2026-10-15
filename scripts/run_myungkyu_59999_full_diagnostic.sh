#!/usr/bin/env bash
set -euo pipefail

# The final report reads the official leaderboard JSON.  Use the host's
# configured accelerator in the same way as the checkpoint/evaluation jobs so
# report generation is not dependent on an interactive shell environment.
source /etc/network_turbo 2>/dev/null || true

# Full 50-task follow-up gated by the pre-registered five-task diagnostic.
# This checkpoint is an independent seed-0/60k reproduction, not the
# unpublished RoboTwin leaderboard artifact.

ROOT=/root/autodl-tmp/EvolveVLA
ROBOTWIN_ROOT=$ROOT/robotwin_repo_ba0fd21
PY=/root/autodl-tmp/conda_envs/scratch_orbax/bin/python
CKPT=$ROOT/checkpoints/myungkyu_pi05_jax_robotwin_vanilla_b128_60k
RESULTS=$ROOT/results/pi05_official_orbax_baseline

if [[ ! -f "$CKPT/params/manifest.ocdbt" ]] || [[ ! -d "$CKPT/params/ocdbt.process_0/d" ]]; then
  echo "Incomplete diagnostic checkpoint: $CKPT" >&2
  exit 1
fi
if find "$CKPT/.cache" -type f -name '*.incomplete' -print -quit 2>/dev/null | grep -q .; then
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
    --prompt-prefix none \
    --normalization quantile \
    --policy-seed 0 \
    --noise-source jax \
    --robotwin-root "$ROBOTWIN_ROOT" \
    --output-root "$output_root" \
    --resume
}

run_setting demo_clean "$RESULTS/myungkyu_59999_full_clean_10"
run_setting demo_randomized "$RESULTS/myungkyu_59999_full_randomized_10"

"$PY" "$ROOT/scripts/aggregate_pi05_candidate_eval.py" \
  --clean-root "$RESULTS/myungkyu_59999_full_clean_10" \
  --randomized-root "$RESULTS/myungkyu_59999_full_randomized_10" \
  --checkpoint "$CKPT" \
  --output-dir "$RESULTS/myungkyu_59999_full_report" \
  --artifact-dir "$ROOT/artifacts/gen0_myungkyu_59999" \
  --candidate-name "Myungkyu seed-0 59999" \
  --candidate-source "Myungkyu/pi0_5_jax_robotwin_vanilla_b128_60k" \
  --candidate-revision "37e10d7f4f49149e2ce5e08a77b07f228a84322f" \
  --robotwin-root "$ROBOTWIN_ROOT"
