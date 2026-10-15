#!/usr/bin/env python
"""pi0.5/RoboCasa state-convention ablation.

The mode order is intentional:
  convention: primary, checkpoint-layout correction.
  sanitized: stopgap, floor/clamp only the checkpoint's zero-std slots.
  asis: current baseline.

This script is for managed evaluation runs. It does not modify checkpoint files.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.cache/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np
import torch
from lerobot.envs.configs import RoboCasaEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import ACTION
from safetensors.torch import load_file


MODE_ORDER = ("convention", "sanitized", "asis")
TASK = "CloseBlenderLid"
SPLIT = "target"
OBS_SIZE = 256
DEFAULT_ROLLOUTS = 5
DEFAULT_STEPS = 300
DEFAULT_SEED = 1000
NORMALIZER_EPS = 1e-8
OFFICIAL_TOKENIZER_PATH = "/root/autodl-tmp/paligemma_tokenizer"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/.cache/huggingface/hub/models--lerobot--pi05_robocasa"

# Current LeRobot RoboCasa wrapper order:
#   base_pos[0:3], base_quat[0:4], ee_pos_rel[0:3],
#   ee_quat_rel[0:4], gripper_qpos[0:2].
CURRENT_STATE_LAYOUT = (
    "base_pos[0:3],base_quat[0:4],ee_pos_rel[0:3],"
    "ee_quat_rel[0:4],gripper_qpos[0:2]"
)

# Best-supported checkpoint order from the normalizer statistics:
#   ee_pos_rel[0:3], ee_quat_rel[0:4], base_pos[0:3],
#   base_quat[0:4], gripper_qpos[0:2].
CHECKPOINT_STATE_LAYOUT = (
    "ee_pos_rel[0:3],ee_quat_rel[0:4],base_pos[0:3],"
    "base_quat[0:4],gripper_qpos[0:2]"
)
CONVENTION_REORDER = np.asarray(
    [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6, 14, 15],
    dtype=np.int64,
)


def _normalizer_path(model_path: str | Path) -> Path:
    return Path(model_path) / "policy_preprocessor_step_3_normalizer_processor.safetensors"


def _load_state_stats(model_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    stats = load_file(str(_normalizer_path(model_path)))
    mean = stats["observation.state.mean"].detach().cpu().numpy().astype(np.float64)
    std = stats["observation.state.std"].detach().cpu().numpy().astype(np.float64)
    if mean.shape != (16,) or std.shape != (16,):
        raise ValueError("expected 16-D observation.state statistics")
    return mean, std


def _replace_state_type(original: Any, state: np.ndarray) -> Any:
    if torch.is_tensor(original):
        return torch.as_tensor(state, dtype=original.dtype, device=original.device)
    return state.astype(np.asarray(original).dtype, copy=False)


def apply_state_mode(observation: dict[str, Any], mode: str, mean: np.ndarray, std: np.ndarray) -> dict[str, Any]:
    if mode == "asis":
        return observation
    if "observation.state" not in observation:
        raise KeyError("env preprocessor did not produce observation.state")

    original = observation["observation.state"]
    raw = np.asarray(original.detach().cpu() if torch.is_tensor(original) else original)
    if raw.shape[-1] != 16:
        raise ValueError(f"expected state width 16, got {raw.shape}")

    if mode == "convention":
        transformed = raw[..., CONVENTION_REORDER]
    elif mode == "sanitized":
        transformed = raw.copy()
        zero_std = std == 0.0
        if np.any(zero_std):
            floor_std = np.maximum(std[zero_std], 1e-2)
            z = (raw[..., zero_std] - mean[zero_std]) / floor_std
            z = np.clip(z, -5.0, 5.0)
            # The stock normalizer still runs after this function. Encode the
            # stopgap z value so its eps denominator reproduces the clamp.
            transformed[..., zero_std] = mean[zero_std] + z * NORMALIZER_EPS
    else:
        raise ValueError(f"unknown state mode: {mode}")

    output = dict(observation)
    output["observation.state"] = _replace_state_type(original, transformed)
    return output


def _tensor_stats(value: Any) -> dict[str, Any]:
    tensor = value if torch.is_tensor(value) else torch.as_tensor(value)
    array = tensor.detach().float().cpu().numpy()
    return {
        "shape": list(array.shape),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "std": float(np.std(array)),
        "finite": bool(np.isfinite(array).all()),
    }


def _success_from_info(info: Any) -> bool:
    if not isinstance(info, dict):
        return False
    if "final_info" in info:
        final_info = info["final_info"]
        if isinstance(final_info, dict):
            return bool(np.asarray(final_info.get("is_success", False)).reshape(-1)[0])
        if isinstance(final_info, (list, tuple)):
            return any(
                isinstance(item, dict)
                and bool(np.asarray(item.get("is_success", False)).reshape(-1)[0])
                for item in final_info
            )
    if "is_success" in info:
        return bool(np.asarray(info["is_success"]).reshape(-1)[0])
    return False


def build_policy(model_path: str, env_cfg: RoboCasaEnv):
    policy_cfg = PI05Config.from_pretrained(model_path, local_files_only=True)
    policy_cfg.device = "cuda:0"
    policy_cfg.pretrained_path = Path(model_path)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_path,
        preprocessor_overrides={
            "device_processor": {"device": "cuda:0"},
            "tokenizer_processor": {"tokenizer_name": OFFICIAL_TOKENIZER_PATH},
        },
    )
    return policy, preprocessor, postprocessor


def rollout(
    env: Any,
    policy: Any,
    env_preprocessor: Any,
    env_postprocessor: Any,
    preprocessor: Any,
    postprocessor: Any,
    mode: str,
    mean: np.ndarray,
    std: np.ndarray,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    policy.reset()
    observation, _ = env.reset(seed=[seed])
    success = False
    last_reward = 0.0
    action_batches: list[dict[str, Any]] = []
    step_count = 0

    for step_index in range(steps):
        observation = preprocess_observation(observation)
        observation["task"] = list(env.call("task_description"))
        observation = env_preprocessor(observation)
        observation = apply_state_mode(observation, mode, mean, std)
        batch = preprocessor(observation)
        with torch.inference_mode():
            action = policy.select_action(batch)
        action_batches.append(_tensor_stats(action))

        if not action_batches[-1]["finite"]:
            raise FloatingPointError(f"non-finite policy action at step {step_index}")

        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        observation, reward, terminated, truncated, info = env.step(
            action.detach().cpu().numpy()
        )
        last_reward = float(np.asarray(reward).reshape(-1)[0])
        success = _success_from_info(info) or success
        step_count = step_index + 1
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(
            np.asarray(truncated).reshape(-1)[0]
        ):
            break

    return {
        "seed": seed,
        "success": success,
        "steps": step_count,
        "last_reward": last_reward,
        "action_batches": action_batches,
    }


def evaluate(
    model_path: str,
    mode: str,
    task: str,
    split: str,
    rollouts: int,
    steps: int,
    seed: int,
) -> dict[str, Any]:
    mean, std = _load_state_stats(model_path)
    env_cfg = RoboCasaEnv(
        task=task,
        split=split,
        observation_height=OBS_SIZE,
        observation_width=OBS_SIZE,
    )
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[task][0]
    policy, preprocessor, postprocessor = build_policy(model_path, env_cfg)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy.config
    )

    reports = []
    try:
        for rollout_index in range(rollouts):
            reports.append(
                rollout(
                    env,
                    policy,
                    env_preprocessor,
                    env_postprocessor,
                    preprocessor,
                    postprocessor,
                    mode,
                    mean,
                    std,
                    steps,
                    seed + rollout_index,
                )
            )
    finally:
        env.close()

    successes = sum(int(report["success"]) for report in reports)
    return {
        "mode": mode,
        "task": task,
        "split": split,
        "rollouts": rollouts,
        "steps": steps,
        "seed": seed,
        "state_layout_current": CURRENT_STATE_LAYOUT,
        "state_layout_used": (
            CHECKPOINT_STATE_LAYOUT if mode == "convention" else CURRENT_STATE_LAYOUT
        ),
        "zero_std_indices": np.flatnonzero(std == 0.0).tolist(),
        "successes": successes,
        "success_rate": successes / rollouts if rollouts else 0.0,
        "reports": reports,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="pi0.5/RoboCasa state convention ablation; no evaluation is run by --help."
    )
    parser.add_argument("--mode", choices=MODE_ORDER, required=True)
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", default=TASK)
    parser.add_argument("--split", default=SPLIT)
    parser.add_argument("--rollouts", type=int, default=DEFAULT_ROLLOUTS)
    parser.add_argument("--steps", type=int, default=DEFAULT_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")

    report = evaluate(
        args.model_path,
        args.mode,
        args.task,
        args.split,
        args.rollouts,
        args.steps,
        args.seed,
    )
    output = args.output or Path(
        f"/root/autodl-tmp/EvolveVLA/results/state_ablation_{args.mode}.json"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(report, indent=2)
    print(text)
    output.write_text(text + "\n")
    print(f"RESULT_JSON={output}")


if __name__ == "__main__":
    main()
