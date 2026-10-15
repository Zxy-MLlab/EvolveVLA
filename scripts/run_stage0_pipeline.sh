#!/usr/bin/env bash
set -u

# Long-running Stage-0 continuation.  The first diagnostic process is started
# separately so this watcher can preserve one-GPU ordering without requiring an
# interactive shell to remain open.
ROOT=/root/autodl-tmp/EvolveVLA
PY=/root/autodl-tmp/conda_envs/openpi_public/bin/python
CKPT=$ROOT/checkpoints/pi05_robotwin2_public
export ROBOTWIN_RENDER_BACKEND=${ROBOTWIN_RENDER_BACKEND:-rt}
export ROBOTWIN_RT_DENOISER=${ROBOTWIN_RT_DENOISER:-oidn}
export ROBOTWIN_OIDN_RUNTIME=${ROBOTWIN_OIDN_RUNTIME:-$ROOT/third_party/oidn-2.5.1/lib}
PID_TO_WAIT=${1:?coverage PID required}
LOG=$ROOT/results/pi05_public_baseline/stage0_pipeline.log
mkdir -p "$(dirname "$LOG")"
exec > >(tee -a "$LOG") 2>&1
echo "waiting for diagnostic coverage pid=$PID_TO_WAIT"
while kill -0 "$PID_TO_WAIT" 2>/dev/null; do sleep 30; done
echo "diagnostic coverage exited"

if [ "${SKIP_COVERAGE:-0}" = "1" ]; then
  echo "skipping auxiliary randomized coverage (official sweeps take priority)"
else
  echo "starting randomized one-episode coverage"
  CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache PYTHONUNBUFFERED=1 \
    "$PY" "$ROOT/scripts/run_public_openpi_parallel.py" \
    --checkpoint "$CKPT" --task-config demo_randomized --episodes 1 --seed 100000 --workers 1 \
    --output-root "$ROOT/results/pi05_public_baseline/coverage_randomized" --resume || echo "randomized coverage returned nonzero; continuing"
fi

echo "starting clean diagnostic: 50 tasks x 10 episodes (non-official quick setting)"
CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache PYTHONUNBUFFERED=1 \
  "$PY" "$ROOT/scripts/run_public_openpi_parallel.py" \
  --checkpoint "$CKPT" --task-config demo_clean --episodes 10 --seed 100000 --workers 1 \
  --output-root "$ROOT/results/pi05_public_baseline/full_clean_10" --resume || echo "clean 10-episode diagnostic returned nonzero; inspect manifest and resume"

echo "starting randomized diagnostic: 50 tasks x 10 episodes (non-official quick setting)"
CUDA_VISIBLE_DEVICES=0 OPENPI_DATA_HOME=/root/autodl-tmp/openpi_cache PYTHONUNBUFFERED=1 \
  "$PY" "$ROOT/scripts/run_public_openpi_parallel.py" \
  --checkpoint "$CKPT" --task-config demo_randomized --episodes 10 --seed 100000 --workers 1 \
  --output-root "$ROOT/results/pi05_public_baseline/full_randomized_10" --resume || echo "randomized 10-episode diagnostic returned nonzero; inspect manifest and resume"

echo "validating and freezing Generation-0 (only if both sweeps are complete)"
"$PY" "$ROOT/scripts/finalize_stage0.py" || echo "Generation-0 not frozen; full evaluation remains incomplete"
echo "stage0 pipeline evaluation runs finished"
