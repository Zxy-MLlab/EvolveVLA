#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/EvolveVLA
PYTHON=/root/autodl-tmp/conda_envs/scratch_orbax/bin/python
OUTPUT="$ROOT/outputs/gen1_all_tasks_clean10_20261018"
CHECKPOINT="$ROOT/checkpoints/gen1_expanded_lora_step24000_export_20261015"
SEED=1800000

QUEUE_META="$ROOT/outputs/gen1_all_tasks_clean10_20261018.queue_meta.txt"
date -u +%FT%TZ > "$QUEUE_META"
echo "Waiting for the active rollout to release the GPU..."
while true; do
  apps="$(nvidia-smi --query-compute-apps=pid --format=csv,noheader,nounits 2>/dev/null || true)"
  if [[ -z "${apps//[[:space:]]/}" ]]; then
    break
  fi
  sleep 60
done

exec "$PYTHON" -u "$ROOT/scripts/gen1_multitask_eval.py" run \
  --output "$OUTPUT" \
  --checkpoint "$CHECKPOINT" \
  --seed "$SEED"
