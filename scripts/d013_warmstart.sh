#!/usr/bin/env bash
set -euo pipefail

ENV_ROOT="/root/autodl-tmp/conda_envs/evolvevla_pi312"
TRAIN="${ENV_ROOT}/bin/lerobot-train"
BASE="/root/autodl-tmp/.cache/huggingface/hub/models--lerobot--pi05_base"
DATASET_ROOT="${D013_DATASET_ROOT:-/root/autodl-tmp/EvolveVLA/data/robocasa_lerobot}"
OUTPUT_DIR="/root/autodl-tmp/EvolveVLA/checkpoints/d013_warmstart"
STEPS="${D013_STEPS:-3000}"
BATCH_SIZE="${D013_BATCH_SIZE:-1}"
SEED="${D013_SEED:-20261001}"

# Include the pi05 action/state projections so the adapter can specialize the
# cached 32-D policy to the RoboCasa feature layout after LeRobot padding.
TARGET_MODULES='(.*\.gemma_expert\..*\.self_attn\.(q|v)_proj|model\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))'

if [[ ! -x "${TRAIN}" ]]; then
    echo "D013 blocked: missing LeRobot train entrypoint: ${TRAIN}" >&2
    exit 2
fi

if [[ ! -f "${BASE}/config.json" ]]; then
    echo "D013 blocked: missing cached pi05 base config: ${BASE}/config.json" >&2
    exit 2
fi

if [[ ! -f "${DATASET_ROOT}/meta/info.json" ]]; then
    echo "D013 blocked: no local LeRobot RoboCasa dataset at ${DATASET_ROOT}" >&2
    echo "Expected a dataset root containing meta/info.json and episode metadata." >&2
    exit 2
fi

CMD=(
    "${TRAIN}"
    "--dataset.repo_id=robocasa/d013_local"
    "--dataset.root=${DATASET_ROOT}"
    "--policy.path=${BASE}"
    "--policy.type=pi05"
    "--policy.device=cuda:0"
    "--policy.push_to_hub=false"
    "--output_dir=${OUTPUT_DIR}"
    "--job_name=d013_pi05_neutral_warmstart"
    "--seed=${SEED}"
    "--num_workers=0"
    "--batch_size=${BATCH_SIZE}"
    "--persistent_workers=false"
    "--steps=${STEPS}"
    "--save_freq=1000"
    "--wandb.enable=false"
    "--save_checkpoint_to_hub=false"
    "--peft.target_modules=${TARGET_MODULES}"
    "--peft.full_training_modules=[]"
    "--peft.method_type=LORA"
    "--peft.r=16"
    "--peft.lora_alpha=32"
)

printf 'FULL_TRAINING_COMMAND='
printf '%q ' "${CMD[@]}"
printf '\n'
exec "${CMD[@]}"
