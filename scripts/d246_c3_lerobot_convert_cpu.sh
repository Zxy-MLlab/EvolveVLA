#!/usr/bin/env bash
set -euo pipefail

ROOT=/root/autodl-tmp/EvolveVLA
ENV=/root/autodl-tmp/conda_envs/evolvevla_pi312
SOURCE="$ROOT/robotwin_repo/data/c3_from_c1/beat_block_hammer/aloha_agilex/data"
OUTPUT="$ROOT/data/d246_c3_targeted_lerobot"
LOG="$ROOT/logs/d246_c3_targeted_lerobot_convert.log"

mkdir -p "$(dirname "$LOG")"
: > "$LOG"

export CUDA_VISIBLE_DEVICES=
export HF_HUB_OFFLINE=1
export HF_HOME=/root/autodl-tmp/hfhub
export TRANSFORMERS_CACHE=/root/autodl-tmp/hfhub
export HF_DATASETS_CACHE=/root/autodl-tmp/hfhub/datasets
export PYTHONUNBUFFERED=1

cd "$ROOT"
set +e
"$ENV/bin/python" scripts/hdf5_to_lerobot_v3.py \
  --source "$SOURCE" \
  --output "$OUTPUT" \
  --fps 15 \
  --task "beat_block_hammer scripted expert" \
  --overwrite \
  2>&1 | tee "$LOG"
rc=${PIPESTATUS[0]}
set -e

echo "CONVERT_EXIT $rc"
grep -E "Traceback|ValueError|AttributeError" "$LOG" || true
exit "$rc"
