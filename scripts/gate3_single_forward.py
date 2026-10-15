#!/usr/bin/env python3
"""Single real-policy GATE 3 forward pre-gate."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import traceback
from pathlib import Path
from typing import Any

import torch

from robotwin_eval_harness import (
    DEFAULT_ROBOTWIN_ROOT,
    DEFAULT_SEED,
    DEFAULT_SIM_PYTHON,
    DEFAULT_TASK,
    DEFAULT_TOKENIZER_PATH,
    derive_camera_rename_map,
    make_policy_batch,
    receive_message,
    send_message,
)


DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/EvolveVLA/checkpoints/sidneyxie_pi05_robotwin"
)
HARNESS_PATH = Path(__file__).with_name("robotwin_eval_harness.py")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run exactly one SidneyXie PI05 policy forward."
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--robotwin-root", default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--task-config", default="demo_clean")
    return parser.parse_args()


def load_sidneyxie_policy(
    args: argparse.Namespace,
) -> tuple[Any, Any, Any, Any]:
    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    policy_cfg = PI05Config.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    policy_cfg.device = device
    policy_cfg.pretrained_path = Path(args.model_path)
    rename_map = derive_camera_rename_map(policy_cfg)

    policy = PI05Policy.from_pretrained(
        args.model_path,
        config=policy_cfg,
        local_files_only=True,
    )
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.model_path,
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {
                "tokenizer_name": args.tokenizer_path,
            },
        },
    )
    env_cfg = RoboTwinEnvConfig(
        task=args.task,
        episode_length=1,
        obs_type="pixels_agent_pos",
        camera_names="head_camera,left_camera,right_camera",
        observation_height=240,
        observation_width=320,
        action_mode="joint",
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg, policy.config)
    return policy, preprocessor, env_preprocessor, policy_cfg


def receive_real_observation(args: argparse.Namespace):
    child_env = os.environ.copy()
    child_env.setdefault("CUDA_VISIBLE_DEVICES", "0")
    child_env["PYTHONUNBUFFERED"] = "1"
    child = subprocess.Popen(
        [
            args.sim_python,
            str(HARNESS_PATH),
            "--role",
            "sim",
            "--task",
            args.task,
            "--episodes",
            "1",
            "--episode-steps",
            "1",
            "--seed",
            str(args.seed),
            "--robotwin-root",
            args.robotwin_root,
            "--task-config",
            args.task_config,
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
        env=child_env,
        cwd=args.robotwin_root,
    )
    assert child.stdin is not None
    assert child.stdout is not None
    try:
        message = receive_message(child.stdout)
        if message.get("type") == "error":
            raise RuntimeError(message["message"])
        if message.get("type") != "episode_start":
            raise RuntimeError(
                f"unexpected simulator message: {message.get('type')}"
            )
        return child, message
    except Exception:
        returncode = child.poll()
        if returncode is None:
            child.kill()
            returncode = child.wait()
        raise RuntimeError(
            f"simulator IPC receive failed; child_returncode={returncode}"
        ) from None


def main() -> int:
    args = parse_args()
    forward_ok = False
    action_shape: tuple[int, ...] | None = None
    action_finite = False
    child = None
    try:
        policy, preprocessor, env_preprocessor, policy_cfg = (
            load_sidneyxie_policy(args)
        )
        child, message = receive_real_observation(args)
        observation = message["observation"]
        instruction = str(message["instruction"])
        batch = make_policy_batch(
            observation,
            instruction,
            env_preprocessor,
            preprocessor,
            policy_cfg,
        )
        policy.reset()
        with torch.inference_mode():
            action = policy.select_action(batch)
        action_shape = tuple(action.shape)
        action_finite = bool(torch.isfinite(action).all().item())
        forward_ok = True
    except Exception:
        traceback.print_exc()
    finally:
        if child is not None:
            try:
                assert child.stdin is not None
                send_message(child.stdin, {"cmd": "stop"})
                child.stdin.close()
            except (BrokenPipeError, OSError):
                pass
            child.wait()

    exit_code = 0 if forward_ok and action_finite else 1
    print("IMAGE_INPUT_LAYOUT=CHW_FLOAT32")
    print(f"FORWARD_OK={forward_ok}")
    print(f"ACTION_SHAPE={action_shape}")
    print(f"ACTION_FINITE={action_finite}")
    print(f"SINGLE_FORWARD_EXIT={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
