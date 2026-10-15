#!/usr/bin/env bash
set -u -o pipefail

if [ "$#" -ne 5 ]; then
  printf 'usage: %s <dataset_root> <dataset_repo_id> <output_dir> <log_path> <steps>\n' "$0" >&2
  exit 2
fi

DATASET_ROOT=$1
DATASET_REPO_ID=$2
OUTPUT_DIR=$3
LOG=$4
STEPS=$5

ENV=/root/autodl-tmp/conda_envs/evolvevla_pi312
ROOT=/root/autodl-tmp/EvolveVLA
RUN_DIR=$(dirname "$LOG")
WATCHDOG_LOG="${LOG}.watchdog"

mkdir -p "$RUN_DIR" "$OUTPUT_DIR"
: > "$LOG"
: > "$WATCHDOG_LOG"

export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1

"$ENV/bin/lerobot-train" \
  --policy.path "$ROOT/checkpoints/sidneyxie_pi05_robotwin_localpre" \
  --policy.push_to_hub false \
  --use_policy_training_preset true \
  --dataset.root "$DATASET_ROOT" \
  --dataset.repo_id "$DATASET_REPO_ID" \
  --batch_size 2 \
  --steps "$STEPS" \
  --num_workers 2 \
  --save_checkpoint true \
  --save_freq "$((STEPS / 2))" \
  --output_dir "$OUTPUT_DIR" \
  --wandb.enable false \
  > "$LOG" 2>&1 &
JOB=$!

cpu_ticks() {
  awk '{print $14 + $15}' "/proc/$1/stat" 2>/dev/null || printf '0'
}

(
  previous_cpu=-1
  while kill -0 "$JOB" 2>/dev/null; do
    now=$(date +%s)
    mtime=$(stat -c %Y "$LOG" 2>/dev/null || printf '%s' "$now")
    age=$((now - mtime))
    current_cpu=$(cpu_ticks "$JOB")
    if [ "$previous_cpu" -lt 0 ]; then
      delta=0
    else
      delta=$((current_cpu - previous_cpu))
    fi
    previous_cpu=$current_cpu
    ps_state=$(ps -o etime=,time=,stat= -p "$JOB" 2>/dev/null | tr '\n' ' ')
    printf 'WATCHDOG pid=%s log_age=%s cpu_ticks=%s cpu_delta=%s ps=%s\n' \
      "$JOB" "$age" "$current_cpu" "$delta" "$ps_state" >> "$WATCHDOG_LOG"
    if [ "$age" -gt 120 ]; then
      printf 'WATCHDOG_LOG_STALE age=%s pid=%s\n' "$age" "$JOB" >> "$LOG"
      kill -TERM "$JOB" 2>/dev/null || true
      exit 124
    fi
    sleep 30
  done
) &
WATCHDOG=$!

wait "$JOB"
CODE=$?
kill "$WATCHDOG" 2>/dev/null || true
printf 'TRAIN_EXIT=%s\n' "$CODE" >> "$LOG"
exit "$CODE"
