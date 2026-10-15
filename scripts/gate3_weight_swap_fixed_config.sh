#!/usr/bin/env bash
set -u

ROOT=/root/autodl-tmp/EvolveVLA
ENV=/root/autodl-tmp/conda_envs/evolvevla_pi312
RUN_DIR=/root/runs/zhouxueyangapexinai
LOG=$RUN_DIR/gate3_weight_swap_fixed_config.log
WATCHDOG_LOG=$RUN_DIR/gate3_weight_swap_fixed_config.watchdog.log

mkdir -p "$RUN_DIR" "$ROOT/results"
: > "$LOG"
: > "$WATCHDOG_LOG"
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
export HEARTBEAT_LOG="$LOG"
printf 'LAUNCH_TIME=%s\n' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$LOG"

"$ENV/bin/python" "$ROOT/scripts/gate3_weight_swap_fixed_config.py" \
  >> "$LOG" 2>&1 &
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
printf 'PY_EXIT=%s\n' "$CODE" >> "$LOG"
exit "$CODE"
