#!/usr/bin/env bash
# Fresh fixed-budget pilot: validated data -> train -> CPU reload -> Clean-only rollout.
set -euo pipefail
ROOT=/root/autodl-tmp/EvolveVLA
PY=/root/autodl-tmp/conda_envs/gen1_train_ready/bin/python
CKPT=$ROOT/checkpoints/gen1_pilot_2000_20261010
PROV=$ROOT/outputs/gen1_pilot_2000_20261010
OUT=$ROOT/results/gen1_pilot_2000_clean_two_task_20261010
LOG=$ROOT/logs/gen1_pilot_2000_training_20261010.log
EXIT_FILE=$ROOT/logs/gen1_pilot_2000_training_20261010.exit
[[ ! -e "$CKPT" && ! -L "$CKPT" && ! -e "$PROV" && ! -e "$OUT" && ! -e "$LOG" && ! -e "$EXIT_FILE" ]]
# Do not share GPU with unrelated training or simulator processes.
GPU_PROCESSES=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits)
[[ -z "$GPU_PROCESSES" ]] || { printf 'GPU occupied; preserving active processes: %s\n' "$GPU_PROCESSES"; exit 1; }
COMMON=(--repo-id gen1_pilot_train_lossless_20261010 --val-repo-id gen1_pilot_val_20261010
  --lerobot-home "$ROOT/data"
  --split-manifest "$ROOT/data/gen1_pilot_two_task_20261010/split_manifest.json"
  --train-audit "$ROOT/data/gen1_pilot_train_lossless_20261010/conversion_audit.json"
  --val-audit "$ROOT/data/gen1_pilot_val_20261010/conversion_audit.json"
  --checkpoint-dir "$CKPT" --provenance-dir "$PROV")
"$PY" -I -u -B "$ROOT/scripts/gen1_train_pilot.py" --mode loader "${COMMON[@]}"
printf '%s\n' 'STAGE loader_passed; starting fixed 2000-step training'
set +e
(set -o noclobber; "$PY" -I -u -B "$ROOT/scripts/gen1_train_pilot.py" --mode train --coordinator-approved-gpu "${COMMON[@]}" > "$LOG" 2>&1)
rc=$?
set -e
(set -o noclobber; printf '%s\n' "$rc" > "$EXIT_FILE")
[[ "$rc" == 0 ]] || { printf 'STAGE training_failed exit=%s log=%s\n' "$rc" "$LOG"; exit "$rc"; }
printf '%s\n' 'STAGE training_saved; independent CPU full-state reload'
"$PY" -I -u -B "$ROOT/scripts/gen1_verify_pilot.py" --checkpoint-dir "$CKPT" \
  --training-log "$LOG" --exit-code-file "$EXIT_FILE" --repo-id gen1_pilot_train_lossless_20261010 \
  --provenance-file "$PROV/run_config.json"
printf '%s\n' 'STAGE reload_passed; starting two-task Clean evaluation'
[[ ! -e "$OUT" && -f "$CKPT/1999/params/manifest.ocdbt" && -d "$CKPT/1999/assets/aloha_agilex_sim" ]]
export CUDA_VISIBLE_DEVICES=0
export OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache
export PYTHONUNBUFFERED=1
export PYTHONPATH=$ROOT/robotwin_repo_ba0fd21/XPolicyLab/policy/Pi_05/openpi/src
export ROBOTWIN_OPENPI_SRC=$PYTHONPATH
unset XLA_CLIENT_MEM_FRACTION JAX_PLATFORMS
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.7
export ROBOTWIN_RENDER_BACKEND=rt
export ROBOTWIN_RT_DENOISER=oidn
export ROBOTWIN_OIDN_RUNTIME=$ROOT/third_party/oidn-2.5.1/lib
unset ROBOTWIN_SKIP_EXPERT ROBOTWIN_OBS_HEIGHT ROBOTWIN_OBS_WIDTH ROBOTWIN_TRAJECTORY_JSONL ROBOTWIN_CAM_DUMP_DIR
/root/autodl-tmp/conda_envs/scratch_orbax/bin/python "$ROOT/scripts/run_public_openpi_parallel.py" \
  --checkpoint "$CKPT/1999" --task-config demo_clean --tasks place_mouse_pad,hanging_mug \
  --episodes 10 --seed 100000 --episode-steps 2000 --workers 1 --prompt-prefix none \
  --normalization quantile --policy-seed 0 --noise-source jax \
  --robotwin-root "$ROOT/robotwin_repo_ba0fd21" --output-root "$OUT"
printf '%s\n' 'STAGE Clean_evaluation_finished; inspect exact results and accepted seeds'
