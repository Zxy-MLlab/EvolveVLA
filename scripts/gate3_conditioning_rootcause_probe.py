#!/usr/bin/env python3
"""Bounded Gate-3 PI0.5 select_action conditioning probe."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch


REMOTE_ROOT = Path("/root/autodl-tmp/EvolveVLA")
HARNESS_PATH = REMOTE_ROOT / "scripts/robotwin_eval_harness.py"
DATA_ROOT = REMOTE_ROOT / "data/gate2_robotwin_cache/clean"
CHECKPOINT = REMOTE_ROOT / "checkpoints/pi05_robotwin_lerobot"
TOKENIZER = Path("/root/autodl-tmp/paligemma_tokenizer")
OUTPUT = REMOTE_ROOT / "results/gate3_conditioning_rootcause.md"
EPISODE = 50
POSITION = 31
CAMERA_FILES = {
    "head_camera": DATA_ROOT
    / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
    "left_camera": DATA_ROOT
    / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
    "right_camera": DATA_ROOT
    / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
}
STATE_PERMUTATION = np.array(
    [6, 12, 0, 8, 3, 10, 1, 13, 5, 9, 2, 11, 7, 4], dtype=np.int64
)
CONFIG_PATH = CHECKPOINT / "config.json"
PREPROCESSOR_PATH = CHECKPOINT / "policy_preprocessor.json"
NORM_STATS_PATH = CHECKPOINT / "norm_stats.json"


def load_harness():
    scripts_dir = str(HARNESS_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "robotwin_eval_harness", HARNESS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import harness from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode_frame(video_path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_rgb().to_ndarray()
    raise RuntimeError(f"{video_path}: missing frame {frame_index}")


def fmt(values: np.ndarray | torch.Tensor | list[float], precision: int = 6) -> str:
    if torch.is_tensor(values):
        arr = values.detach().float().cpu().numpy().reshape(-1)
    else:
        arr = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(v): .{precision}f}" for v in arr) + "]"


def stats(values: np.ndarray | torch.Tensor) -> dict[str, object]:
    if torch.is_tensor(values):
        arr = values.detach().float().cpu()
        return {
            "shape": list(arr.shape),
            "dtype": str(values.dtype),
            "device": str(values.device),
            "min": float(arr.amin()),
            "max": float(arr.amax()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }
    arr = np.asarray(values)
    return {
        "shape": list(arr.shape),
        "dtype": str(arr.dtype),
        "min": float(arr.min()),
        "max": float(arr.max()),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
    }


def get_normalizer(pipeline):
    for step in pipeline.steps:
        if step.__class__.__name__ == "NormalizerProcessorStep":
            return step
    raise RuntimeError("NormalizerProcessorStep not found")


def make_observation(harness, decoded: dict[str, np.ndarray], state: np.ndarray):
    raw = {
        "observation": {
            camera: {"rgb": decoded[camera]}
            for camera in ("head_camera", "left_camera", "right_camera")
        },
        "joint_action": {"vector": state.tolist()},
    }
    return harness.simulator_observation(raw)


def make_batch(
    harness,
    observation,
    instruction: str,
    env_preprocessor,
    preprocessor,
    policy_config,
):
    return harness.make_policy_batch(
        observation,
        instruction,
        env_preprocessor,
        preprocessor,
        policy_config,
    )


def postprocess_action(postprocessor, env_postprocessor, action):
    from lerobot.utils.constants import ACTION

    action = postprocessor(action)
    return env_postprocessor({ACTION: action})[ACTION]


def run_select_action(
    policy,
    batch,
    postprocessor,
    env_postprocessor,
) -> tuple[torch.Tensor, torch.Tensor]:
    policy.reset()
    with torch.inference_mode():
        raw_action = policy.select_action(batch)
    final_action = postprocess_action(postprocessor, env_postprocessor, raw_action)
    return raw_action, final_action


def clone_batch(batch: dict[str, object]) -> dict[str, object]:
    return {
        key: value.clone() if torch.is_tensor(value) else value
        for key, value in batch.items()
    }


def rescale_image_batch(batch: dict[str, object]) -> dict[str, object]:
    scaled = clone_batch(batch)
    for key, value in scaled.items():
        if "observation.images." in str(key) and torch.is_tensor(value):
            scaled[key] = value * 2.0 - 1.0
    return scaled


def replace_language_tokens(
    batch: dict[str, object],
    prompt: str,
    tokenizer_path: Path,
    max_length: int,
) -> dict[str, object]:
    from transformers import AutoTokenizer
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    tokenized = AutoTokenizer.from_pretrained(
        str(tokenizer_path),
        local_files_only=True,
        padding_side="right",
    )(
        [prompt],
        max_length=max_length,
        padding="max_length",
        truncation=True,
        return_tensors="pt",
    )
    updated = clone_batch(batch)
    device = updated[OBS_LANGUAGE_TOKENS].device
    updated[OBS_LANGUAGE_TOKENS] = tokenized["input_ids"].to(device)
    updated[OBS_LANGUAGE_ATTENTION_MASK] = tokenized["attention_mask"].to(
        device=device, dtype=torch.bool
    )
    return updated


def declared_image_preprocessing() -> dict[str, object]:
    config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    policy_preprocessor = json.loads(
        PREPROCESSOR_PATH.read_text(encoding="utf-8")
    )
    norm_stats = json.loads(NORM_STATS_PATH.read_text(encoding="utf-8"))
    config_map = config.get("normalization_mapping", {})
    preprocessor_map = (
        policy_preprocessor["steps"][3]["config"].get("norm_map", {})
    )
    image_keys = [
        key for key in norm_stats if key.startswith("observation.images.")
    ]
    return {
        "config_visual": config_map.get("VISUAL"),
        "policy_preprocessor_visual": preprocessor_map.get("VISUAL"),
        "image_norm_stats_keys": image_keys,
        "declared_mean_std": None,
        "source_lines": {
            "config": f"{CONFIG_PATH}:80-81",
            "policy_preprocessor": f"{PREPROCESSOR_PATH}:66-67",
            "norm_stats": f"{NORM_STATS_PATH}: no observation.images.* entries",
        },
    }


def config_value(policy, name: str):
    for owner in (policy.config, policy, getattr(policy, "model", None)):
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name)
    return None


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
    harness = load_harness()

    data_path = DATA_ROOT / "data/chunk-000/file-000.parquet"
    data = pq.read_table(
        data_path,
        columns=[
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    ).to_pydict()
    tasks = pq.read_table(DATA_ROOT / "meta/tasks.parquet").to_pydict()
    task_text = dict(zip(tasks["task_index"], tasks["task"]))
    rows = [
        {key: data[key][row] for key in data}
        for row in range(len(data["episode_index"]))
        if data["episode_index"][row] == EPISODE
    ]
    if not rows:
        raise RuntimeError(f"episode {EPISODE} not found")
    rows.sort(key=lambda row: row["frame_index"])
    row = rows[POSITION]
    video_index = int(row["index"])
    instruction = str(task_text[int(row["task_index"])])
    state = np.asarray(row["observation.state"], dtype=np.float32).reshape(14)
    recorded_action = np.asarray(row["action"], dtype=np.float32).reshape(14)

    decoded = {
        camera: decode_frame(path, video_index)
        for camera, path in CAMERA_FILES.items()
    }
    raw_camera_stats = {
        camera: {
            "shape": list(frame.shape),
            "dtype": str(frame.dtype),
            "min": int(frame.min()),
            "max": int(frame.max()),
            "normalized_min": float(frame.min() / 255.0),
            "normalized_max": float(frame.max() / 255.0),
        }
        for camera, frame in decoded.items()
    }

    args = Namespace(
        model_path=str(CHECKPOINT),
        task="beat_block_hammer",
        episode_steps=1200,
        tokenizer_path=str(TOKENIZER),
        task_config="demo_clean",
    )
    # Use the production harness loader, including its exact processor path.
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        harness.load_policy(args)
    )
    normalizer = get_normalizer(preprocessor)

    observation_real = make_observation(harness, decoded, state)
    observation_zero = make_observation(
        harness, decoded, np.zeros(14, dtype=np.float32)
    )
    observation_shuffled = make_observation(
        harness, decoded, state[STATE_PERMUTATION]
    )
    batch_real = make_batch(
        harness,
        observation_real,
        instruction,
        env_preprocessor,
        preprocessor,
        policy.config,
    )
    batch_zero = make_batch(
        harness,
        observation_zero,
        instruction,
        env_preprocessor,
        preprocessor,
        policy.config,
    )
    batch_shuffled = make_batch(
        harness,
        observation_shuffled,
        instruction,
        env_preprocessor,
        preprocessor,
        policy.config,
    )

    from lerobot.utils.constants import ACTION

    normalized_transition = preprocessor(
        {
            "observation.state": torch.from_numpy(state),
            "task": [instruction],
            "action": torch.from_numpy(recorded_action),
            **{
                harness.POLICY_CAMERA_KEYS[key]: value
                for key, value in observation_real["pixels"].items()
            },
        }
    )
    normalized_action = normalized_transition[ACTION]
    roundtrip_action = postprocess_action(
        postprocessor,
        env_postprocessor,
        normalized_action,
    )
    roundtrip = roundtrip_action.detach().float().cpu().numpy().reshape(14)
    roundtrip_error = np.abs(roundtrip - recorded_action)

    raw_real, final_real = run_select_action(
        policy, batch_real, postprocessor, env_postprocessor
    )
    raw_zero, final_zero = run_select_action(
        policy, batch_zero, postprocessor, env_postprocessor
    )
    raw_shuffled, final_shuffled = run_select_action(
        policy, batch_shuffled, postprocessor, env_postprocessor
    )
    batch_image_neg1 = rescale_image_batch(batch_real)
    raw_image_neg1, final_image_neg1 = run_select_action(
        policy, batch_image_neg1, postprocessor, env_postprocessor
    )
    tokenizer_max_length = int(config_value(policy, "tokenizer_max_length") or 200)
    batch_prompt_bare = replace_language_tokens(
        batch_real,
        instruction,
        TOKENIZER,
        tokenizer_max_length,
    )
    raw_prompt_bare, final_prompt_bare = run_select_action(
        policy, batch_prompt_bare, postprocessor, env_postprocessor
    )

    final_arrays = {
        "real": final_real.detach().float().cpu().numpy().reshape(14),
        "zeroed": final_zero.detach().float().cpu().numpy().reshape(14),
        "shuffled": final_shuffled.detach().float().cpu().numpy().reshape(14),
    }
    raw_arrays = {
        "real": raw_real.detach().float().cpu().numpy().reshape(14),
        "zeroed": raw_zero.detach().float().cpu().numpy().reshape(14),
        "shuffled": raw_shuffled.detach().float().cpu().numpy().reshape(14),
        "image_neg1": raw_image_neg1.detach().float().cpu().numpy().reshape(14),
        "prompt_bare": raw_prompt_bare.detach().float().cpu().numpy().reshape(14),
    }
    final_arrays["image_neg1"] = (
        final_image_neg1.detach().float().cpu().numpy().reshape(14)
    )
    final_arrays["prompt_bare"] = (
        final_prompt_bare.detach().float().cpu().numpy().reshape(14)
    )
    arm_norms = {
        name: float(np.linalg.norm(values[:6]))
        for name, values in final_arrays.items()
    }
    arm_pairwise_delta = {
        f"{left}-{right}": float(
            np.linalg.norm(final_arrays[left][:6] - final_arrays[right][:6])
        )
        for left, right in (("real", "zeroed"), ("real", "shuffled"), ("zeroed", "shuffled"))
    }
    arm_response = max(arm_pairwise_delta.values())
    arm_eps = 1e-3
    arm_dead = (
        max(arm_norms.values()) < arm_eps and arm_response < arm_eps
    )
    raw_action_stack = np.stack(list(raw_arrays.values()))
    action_head_non_degenerate = bool(
        float(raw_action_stack.std()) > 1e-5
        and float(np.ptp(raw_action_stack, axis=1).max()) > 1e-5
    )

    batch_camera_stats = {}
    for camera in ("head_camera", "left_camera", "right_camera"):
        key = harness.POLICY_CAMERA_KEYS[camera]
        batch_camera_stats[camera] = stats(batch_real[key])
        batch_camera_stats[camera]["raw_normalized_min"] = raw_camera_stats[camera][
            "normalized_min"
        ]
        batch_camera_stats[camera]["raw_normalized_max"] = raw_camera_stats[camera][
            "normalized_max"
        ]
        batch_camera_stats[camera]["min_delta_vs_raw_normalized"] = (
            batch_camera_stats[camera]["min"]
            - raw_camera_stats[camera]["normalized_min"]
        )
        batch_camera_stats[camera]["max_delta_vs_raw_normalized"] = (
            batch_camera_stats[camera]["max"]
            - raw_camera_stats[camera]["normalized_max"]
        )

    wrist_corrupt = any(
        abs(
            batch_camera_stats[camera]["min_delta_vs_raw_normalized"]
        )
        > 0.05
        or abs(
            batch_camera_stats[camera]["max_delta_vs_raw_normalized"]
        )
        > 0.05
        for camera in ("left_camera", "right_camera")
    )
    if arm_dead:
        verdict = "DEAD ARM PATH: arm output is near zero and state-insensitive."
    elif wrist_corrupt:
        verdict = (
            "WRIST IMAGE CORRUPTION: post-preprocessor wrist range is materially "
            "different from the raw frame range."
        )
    else:
        verdict = (
            "ARM PATH IS STATE-SENSITIVE: the zero-arm symptom is not explained "
            "by a dead action head or state-independent arm path; inspect image/"
            "prompt/other conditioning and oracle mismatch."
        )

    config_steps = {
        "n_action_steps": config_value(policy, "n_action_steps"),
        "num_denoising_steps": config_value(policy, "num_inference_steps"),
    }
    declared_preprocessing = declared_image_preprocessing()
    recorded_arm_norm = float(np.linalg.norm(recorded_action[:6]))
    arm_ablation_norms = {
        "image_[0,1]_canonical": arm_norms["real"],
        "image_[-1,1]_canonical": arm_norms["image_neg1"],
        "image_[0,1]_prompt_bare": arm_norms["prompt_bare"],
        "recorded": recorded_arm_norm,
    }
    arm_ablation_errors = {
        name: abs(value - recorded_arm_norm)
        for name, value in arm_ablation_norms.items()
        if name != "recorded"
    }
    best_ablation = min(arm_ablation_errors, key=arm_ablation_errors.get)
    baseline_error = arm_ablation_errors["image_[0,1]_canonical"]
    moved_toward = {
        name: error < baseline_error
        for name, error in arm_ablation_errors.items()
        if name != "image_[0,1]_canonical"
    }
    normalized_state_for_prompt = (
        normalized_transition["observation.state"]
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(-1)
    )
    discretized_state = (
        np.digitize(
            normalized_state_for_prompt,
            bins=np.linspace(-1, 1, 257)[:-1],
        )
        - 1
    )
    cleaned_instruction = instruction.strip().replace("_", " ").replace("\n", " ")
    canonical_instruction = (
        f"Task: {cleaned_instruction}, "
        f"State: {' '.join(map(str, discretized_state))};\nAction: "
    )
    meta = json.loads((DATA_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    dataset_state_order = meta["features"]["observation.state"]["names"][0]
    dataset_action_order = meta["features"]["action"]["names"][0]
    norm_map = {str(k): str(v) for k, v in normalizer.norm_map.items()}

    lines = [
        "# Gate-3 Conditioning Select-Action Probe",
        "",
        f"- Checkpoint: `{CHECKPOINT}`",
        f"- Harness: `{HARNESS_PATH}`",
        f"- Harness sha256: `{hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest()}`",
        f"- Dataset: `{DATA_ROOT}`",
        f"- Episode/position: `{EPISODE}/{POSITION}`; video index `{video_index}`",
        f"- Instruction: `{instruction}`",
        f"- Canonical π0.5 prompt: `{canonical_instruction}`",
        f"- CUDA device: `{torch.cuda.current_device()}` ({torch.cuda.get_device_name(0)})",
        f"- Processor normalization map: `{norm_map}`",
        f"- Declared image preprocessing: `{declared_preprocessing}`",
        f"- State/action order: `{dataset_state_order}` / `{dataset_action_order}`",
        "",
        "## Raw before/after postprocess",
        "",
        f"- REAL raw model output: `{fmt(raw_arrays['real'])}`",
        f"- REAL after postprocess: `{fmt(final_arrays['real'])}`",
        f"- ZEROED raw model output: `{fmt(raw_arrays['zeroed'])}`",
        f"- ZEROED after postprocess: `{fmt(final_arrays['zeroed'])}`",
        f"- SHUFFLED raw model output: `{fmt(raw_arrays['shuffled'])}`",
        f"- SHUFFLED after postprocess: `{fmt(final_arrays['shuffled'])}`",
        f"- IMAGE [-1,1] after postprocess: `{fmt(final_arrays['image_neg1'])}`",
        f"- PROMPT bare after postprocess: `{fmt(final_arrays['prompt_bare'])}`",
        "",
        "## State sensitivity",
        "",
        f"- Shuffled permutation: `{STATE_PERMUTATION.tolist()}`",
        f"- REAL arm dims 0:6: `{fmt(final_arrays['real'][:6])}`",
        f"- ZEROED arm dims 0:6: `{fmt(final_arrays['zeroed'][:6])}`",
        f"- SHUFFLED arm dims 0:6: `{fmt(final_arrays['shuffled'][:6])}`",
        f"- Arm L2 norms: `{arm_norms}`",
        f"- Pairwise arm L2 deltas: `{arm_pairwise_delta}`",
        f"- Sensitivity verdict: `{'DEAD/STATE-INSENSITIVE' if arm_dead else 'RESPONSIVE'}`",
        "",
        "## Image ranges",
        "",
        "Raw values are decoded uint8; raw_normalized is raw/255; batch is the "
        "post-preprocessor tensor passed to `select_action`.",
        "",
        "```json",
        json.dumps(
            {"raw": raw_camera_stats, "batch": batch_camera_stats},
            indent=2,
            sort_keys=True,
        ),
        "```",
        f"- Image-range verdict: `{'WRIST RANGE CORRUPTION' if wrist_corrupt else 'NO MATERIAL WRIST RANGE TRUNCATION DETECTED'}`",
        "",
        "## Image-scale and prompt A/B",
        "",
        f"- Arm L2 magnitude, recorded: `{recorded_arm_norm:.8f}`",
        f"- Arm L2 magnitudes: `{arm_ablation_norms}`",
        f"- Distance to recorded arm magnitude: `{arm_ablation_errors}`",
        f"- Declared mean/std variant: `{declared_preprocessing['declared_mean_std']}`",
        f"- Moves toward recorded magnitude: `{moved_toward}`",
        f"- Closest tested convention: `{best_ablation}`",
        "- `[0,1]` is the existing canonical-prompt path; `[-1,1]` changes only image tensors.",
        "- Bare prompt tokenization changes only language tokens; the canonical prompt is generated by the existing π0.5 processor.",
        "",
        "## Flow-matching and action head",
        "",
        f"- `n_action_steps`: `{config_steps['n_action_steps']}`",
        f"- `num_denoising_steps`: `{config_steps['num_denoising_steps']}`",
        f"- Raw action stats per condition: `{ {name: stats(value) for name, value in raw_arrays.items()} }`",
        f"- Action-head degeneracy: `{'NON-DEGENERATE' if action_head_non_degenerate else 'DEGENERATE'}`",
        "",
        "## Sanity",
        "",
        f"- Recorded action: `{fmt(recorded_action)}`",
        f"- Round-trip action: `{fmt(roundtrip)}`",
        f"- Round-trip max abs error: `{float(roundtrip_error.max()):.8f}`",
        f"- Round-trip: `{'PASS' if float(roundtrip_error.max()) < 1e-4 else 'FAIL'}`",
        "",
        "## Verdict",
        "",
        (
            f"{verdict} Image/prompt A/B: `{best_ablation}` is closest to the "
            f"recorded arm magnitude; image scale moved toward target="
            f"{moved_toward['image_[-1,1]_canonical']}, prompt format moved "
            f"toward target={moved_toward['image_[0,1]_prompt_bare']}."
        ),
        "",
    ]
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines), encoding="utf-8")

    print(f"OUTPUT={OUTPUT}")
    print(f"RESULT_SHA256={hashlib.sha256(OUTPUT.read_bytes()).hexdigest()}")
    print(f"REAL_RAW={fmt(raw_arrays['real'])}")
    print(f"REAL_AFTER={fmt(final_arrays['real'])}")
    print(f"ZEROED_ARM={fmt(final_arrays['zeroed'][:6])}")
    print(f"SHUFFLED_ARM={fmt(final_arrays['shuffled'][:6])}")
    print(f"REAL_ARM={fmt(final_arrays['real'][:6])}")
    print(f"IMAGE_NEG1_ARM={fmt(final_arrays['image_neg1'][:6])}")
    print(f"PROMPT_BARE_ARM={fmt(final_arrays['prompt_bare'][:6])}")
    print(f"RECORDED_ARM_L2={recorded_arm_norm:.8f}")
    print(f"ARM_ABLATION_NORMS={json.dumps(arm_ablation_norms, sort_keys=True)}")
    print(f"ARM_ABLATION_ERRORS={json.dumps(arm_ablation_errors, sort_keys=True)}")
    print(f"DECLARED_IMAGE_PREPROCESSING={json.dumps(declared_preprocessing, sort_keys=True)}")
    print(f"CANONICAL_PROMPT={canonical_instruction!r}")
    print(f"MOVED_TOWARD={json.dumps(moved_toward, sort_keys=True)}")
    print(f"BEST_ABLATION={best_ablation}")
    print(f"ARM_PAIRWISE_DELTA={arm_pairwise_delta}")
    print(f"RAW_CAMERA_STATS={json.dumps(raw_camera_stats, sort_keys=True)}")
    print(f"BATCH_CAMERA_STATS={json.dumps(batch_camera_stats, sort_keys=True)}")
    print(f"FLOW_STEPS={json.dumps(config_steps, sort_keys=True)}")
    print(f"ACTION_HEAD_NON_DEGENERATE={action_head_non_degenerate}")
    print(f"VERDICT={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
