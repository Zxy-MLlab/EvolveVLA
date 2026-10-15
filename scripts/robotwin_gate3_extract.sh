#!/bin/bash
set -eo pipefail

CHECKPOINT_ROOT="/root/autodl-tmp/EvolveVLA/checkpoints"
ARCHIVE_NAME="RoboTwin-lerobot_v30-aloha_agilex-joint-0.tar.gz"
EXTRACTED_DIR="${CHECKPOINT_ROOT}/pi05_robotwin"

source /etc/network_turbo 2>/dev/null || true
export PIP_CACHE_DIR="/root/autodl-tmp/.cache/pip"
export TMPDIR="/root/autodl-tmp/tmp"
mkdir -p "$CHECKPOINT_ROOT" "$EXTRACTED_DIR" "$TMPDIR"

ARCHIVE="${CHECKPOINT_ROOT}/robotwin_pi05/${ARCHIVE_NAME}"
if [[ ! -f "$ARCHIVE" ]]; then
    ARCHIVE="$(find "$CHECKPOINT_ROOT" -type f -name "$ARCHIVE_NAME" -print -quit)"
fi

if [[ -z "$ARCHIVE" || ! -f "$ARCHIVE" ]]; then
    echo "GATE3_EXTRACT_FAILED: checkpoint archive not found: ${ARCHIVE_NAME}" >&2
    exit 2
fi

ARCHIVE_BYTES="$(stat -c '%s' "$ARCHIVE")"
if [[ -z "$ARCHIVE_BYTES" || "$ARCHIVE_BYTES" -le 0 ]]; then
    echo "GATE3_EXTRACT_FAILED: checkpoint archive is empty: $ARCHIVE" >&2
    exit 2
fi

echo "GATE3_ARCHIVE=$ARCHIVE"
echo "GATE3_ARCHIVE_BYTES=$ARCHIVE_BYTES"
tar -xzf "$ARCHIVE" -C "$EXTRACTED_DIR"

DU_LINE="$(du -sh "$EXTRACTED_DIR")"
echo "GATE3_EXTRACTED_SIZE=$DU_LINE"
EXTRACTED_BYTES="$(du -sb "$EXTRACTED_DIR" | cut -f1)"
if [[ -z "$EXTRACTED_BYTES" || "$EXTRACTED_BYTES" -le 0 ]]; then
    echo "GATE3_EXTRACT_FAILED: extracted directory is empty: $EXTRACTED_DIR" >&2
    exit 3
fi

CONFIG_PATH="$(find "$EXTRACTED_DIR" -type f -name 'config.json' -print -quit)"
OPENPI_METADATA_PATH="$(find "$EXTRACTED_DIR" -type f -name '_CHECKPOINT_METADATA' -print -quit)"
OPENPI_PARAMS_PATH="$(find "$EXTRACTED_DIR" -type d -name 'params' -print -quit)"
CHECKPOINT_ROOT_PATH=""
CHECKPOINT_FORMAT=""

if [[ -n "$CONFIG_PATH" ]]; then
    CHECKPOINT_ROOT_PATH="$(dirname "$CONFIG_PATH")"
    CHECKPOINT_FORMAT="lerobot"
elif [[ -n "$OPENPI_METADATA_PATH" || -n "$OPENPI_PARAMS_PATH" ]]; then
    if [[ -n "$OPENPI_METADATA_PATH" ]]; then
        CHECKPOINT_ROOT_PATH="$(dirname "$OPENPI_METADATA_PATH")"
    else
        CHECKPOINT_ROOT_PATH="$(dirname "$OPENPI_PARAMS_PATH")"
    fi
    if [[ ! -d "$CHECKPOINT_ROOT_PATH/params" || ! -d "$CHECKPOINT_ROOT_PATH/assets" ]]; then
        echo "GATE3_EXTRACT_FAILED: OpenPI checkpoint marker lacks params/assets: $CHECKPOINT_ROOT_PATH" >&2
        exit 4
    fi
    CHECKPOINT_FORMAT="openpi_orbax"
else
    echo "GATE3_EXTRACT_FAILED: no LeRobot config.json or OpenPI _CHECKPOINT_METADATA/params under $EXTRACTED_DIR" >&2
    exit 4
fi

WEIGHTS_PATH="$(find "$CHECKPOINT_ROOT_PATH" -type f \( -name '*.safetensors' -o -name '*.bin' -o -name '*.pt' \) -print -quit)"
PROCESSOR_PATH="$(find "$CHECKPOINT_ROOT_PATH" -type f \( -name 'processor_config.json' -o -name 'preprocessor.json' -o -name 'postprocessor.json' -o -name 'policy_preprocessor.json' -o -name 'policy_postprocessor.json' \) -print -quit)"
STATS_PATH="$(find "$CHECKPOINT_ROOT_PATH" -type f \( -name 'stats.json' -o -name 'normalization.json' -o -name 'norm_stats.json' -o -name '*normalizer*.safetensors' \) -print -quit)"

if [[ "$CHECKPOINT_FORMAT" == "openpi_orbax" ]]; then
    WEIGHTS_PATH="$CHECKPOINT_ROOT_PATH/params"
    PROCESSOR_PATH="$CHECKPOINT_ROOT_PATH/assets"
fi

if [[ "$CHECKPOINT_FORMAT" == "lerobot" && -z "$WEIGHTS_PATH" ]]; then
    echo "GATE3_EXTRACT_FAILED: no LeRobot weight file under $CHECKPOINT_ROOT_PATH" >&2
    exit 4
fi
if [[ "$CHECKPOINT_FORMAT" == "lerobot" && -z "$PROCESSOR_PATH" ]]; then
    echo "GATE3_EXTRACT_FAILED: no processor asset under $CHECKPOINT_ROOT_PATH" >&2
    exit 4
fi
if [[ -z "$STATS_PATH" ]]; then
    echo "GATE3_EXTRACT_FAILED: no normalization statistics under $CHECKPOINT_ROOT_PATH" >&2
    exit 4
fi

echo "GATE3_CHECKPOINT_FORMAT=$CHECKPOINT_FORMAT"
echo "GATE3_CHECKPOINT_ROOT=$CHECKPOINT_ROOT_PATH"
echo "GATE3_CONFIG=${CONFIG_PATH:-UNAVAILABLE_FOR_OPENPI_ORBAX}"
echo "GATE3_WEIGHTS=$WEIGHTS_PATH"
echo "GATE3_PROCESSOR=$PROCESSOR_PATH"
echo "GATE3_STATS=$STATS_PATH"

if [[ "$CHECKPOINT_FORMAT" == "openpi_orbax" ]]; then
    echo "GATE3_EXTRACT_OK_OPENPI_ONLY"
    exit 4
fi

echo "GATE3_EXTRACT_OK_LEROBOT"
