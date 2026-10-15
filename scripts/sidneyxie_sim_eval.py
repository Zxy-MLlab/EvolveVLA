#!/usr/bin/env python3
"""One-task RoboTwin SAPIEN evaluation for the SidneyXie community checkpoint.

Deployment contract:
  GPU: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120), cuda:0 only.
  Driver/runtime: NVIDIA 590.44.01 / CUDA 13.1.
  Batch: 1 policy observation.
  Precision: bfloat16 policy inference.
  Task: beat_block_hammer by default, 5 episodes, 1200-step cap.
  LABEL: community checkpoint (SidneyXie) — NOT official baseline.

The policy runs in the Python 3.12 LeRobot environment. The RoboTwin SAPIEN
simulator runs in the Python 3.10 environment. The parent and child exchange
length-prefixed pickle messages over stdin/stdout.
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import pickle
import struct
import subprocess
import sys
from pathlib import Path
from typing import Any, BinaryIO

from robotwin_eval_harness import (
    ACTION_DIM,
    DEFAULT_EPISODE_STEPS,
    DEFAULT_ROBOTWIN_ROOT,
    DEFAULT_SIM_PYTHON,
    DEFAULT_TASK,
    DEFAULT_TOKENIZER_PATH,
    OBS_HEIGHT,
    OBS_WIDTH,
    STATE_DIM,
    decimate_frame_sequence,
    load_policy,
    make_policy_batch,
    receive_message,
    safe_close_task,
    send_message,
    simulator_observation,
)


LABEL = "community checkpoint (SidneyXie) — NOT official baseline"
DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/EvolveVLA/checkpoints/sidneyxie_pi05_robotwin"
)
DEFAULT_EPISODES = 5
DEFAULT_SEED = 100000
SOURCE_FPS = 50
TARGET_FPS = 25


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate the SidneyXie community RoboTwin checkpoint."
    )
    parser.add_argument("--role", choices=("parent", "sim"), default="parent")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument(
        "--episode-steps", type=int, default=DEFAULT_EPISODE_STEPS
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--robotwin-root", default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_parent(args: argparse.Namespace) -> int:
    import numpy as np
    import torch
    from lerobot.utils.constants import ACTION

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episode_steps <= 0 or args.episode_steps > DEFAULT_EPISODE_STEPS:
        raise ValueError(
            f"--episode-steps must be in [1, {DEFAULT_EPISODE_STEPS}]"
        )

    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        load_policy(args)
    )

    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = "0"
    child_env["PYTHONUNBUFFERED"] = "1"
    child = subprocess.Popen(
        [
            args.sim_python,
            str(Path(__file__).resolve()),
            "--role",
            "sim",
            "--task",
            args.task,
            "--episodes",
            str(args.episodes),
            "--episode-steps",
            str(args.episode_steps),
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
    )
    assert child.stdin is not None
    assert child.stdout is not None

    reports: list[dict[str, Any]] = []
    reset_eval_success: list[bool] = []

    try:
        while True:
            message = receive_message(child.stdout)
            message_type = message.get("type")
            if message_type == "error":
                raise RuntimeError(message["message"])
            if message_type == "complete":
                reset_eval_success = list(
                    message.get("episode_reset_eval_success") or []
                )
                break
            if message_type != "episode_start":
                raise RuntimeError(
                    f"unexpected simulator message: {message_type}"
                )

            policy.reset()
            instruction = str(message["instruction"])
            observation = message["observation"]
            episode_steps = 0
            while True:
                batch = make_policy_batch(
                    observation,
                    instruction,
                    env_preprocessor,
                    preprocessor,
                    policy.config,
                )
                with torch.inference_mode():
                    action = policy.select_action(batch)
                action = postprocessor(action)
                action = env_postprocessor({ACTION: action})[ACTION]
                action_array = action.detach().float().cpu().numpy().reshape(-1)
                assert action_array.shape == (ACTION_DIM,), (
                    f"policy action must be shape (14,), got {action_array.shape}"
                )
                if not np.isfinite(action_array).all():
                    raise RuntimeError("policy returned a non-finite action")

                send_message(
                    child.stdin,
                    {"cmd": "action", "action": action_array},
                )
                response = receive_message(child.stdout)
                if response.get("type") == "error":
                    raise RuntimeError(response["message"])
                if response.get("type") != "step":
                    raise RuntimeError(
                        "unexpected simulator step message: "
                        f"{response.get('type')}"
                    )

                episode_steps += 1
                if response["terminated"] or response["truncated"]:
                    report = {
                        "episode": int(message["episode"]) + 1,
                        "seed": int(message["seed"]),
                        "success": bool(response["success"]),
                        "steps": episode_steps,
                        "eval_success": bool(response["eval_success"]),
                        "check_success": bool(response["check_success"]),
                    }
                    reports.append(report)
                    print(
                        f"EPISODE={report['episode']} "
                        f"SUCCESS={report['success']} "
                        f"STEPS={report['steps']} "
                        f"EVAL_SUCCESS={report['eval_success']} "
                        f"CHECK_SUCCESS={report['check_success']}"
                    )
                    break
                observation = response["observation"]
    finally:
        try:
            send_message(child.stdin, {"cmd": "stop"})
        except (BrokenPipeError, OSError):
            pass
        child.stdin.close()
        return_code = child.wait()
        if return_code != 0:
            raise RuntimeError(f"simulator child exited with code {return_code}")

    successes = sum(bool(report["success"]) for report in reports)
    assert len(reports) == args.episodes, (
        f"completed episodes {len(reports)} != requested {args.episodes}"
    )
    assert successes <= args.episodes, "success count cannot exceed episodes"
    assert all(
        int(report["steps"]) <= DEFAULT_EPISODE_STEPS for report in reports
    ), "an episode exceeded the 1200-step cap"
    assert STATE_DIM == ACTION_DIM == 14

    reset_audit = len(reset_eval_success) == args.episodes and all(
        not value for value in reset_eval_success
    )
    signal_agreement_audit = all(
        bool(report["eval_success"]) == bool(report["check_success"])
        for report in reports
    )

    print(f"SIDNEYXIE_SIM_SUCCESS_RATE={successes}/{len(reports)}")
    print(f"AUDIT_EVAL_SUCCESS_FALSE_AT_RESET_ALL={reset_audit}")
    print(
        "AUDIT_EVAL_SUCCESS_CHECK_SUCCESS_AGREE="
        f"{signal_agreement_audit}"
    )

    payload = {
        "label": LABEL,
        "task": args.task,
        "checkpoint": args.model_path,
        "episodes_requested": args.episodes,
        "episodes_completed": len(reports),
        "episode_length": args.episode_steps,
        "successes": successes,
        "success_rate": successes / len(reports),
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "camera_output_shape": [OBS_HEIGHT, OBS_WIDTH, 3],
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "episode_reset_eval_success": reset_eval_success,
        "audit_eval_success_false_at_reset_all": reset_audit,
        "audit_eval_success_check_success_agree": signal_agreement_audit,
        "reports": reports,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
    return 0


def run_simulator(args: argparse.Namespace) -> int:
    import traceback

    import numpy as np

    fps_probe = decimate_frame_sequence(
        np.zeros((2, 1, 1, 3), dtype=np.uint8),
        source_fps=SOURCE_FPS,
        target_fps=TARGET_FPS,
    )
    assert fps_probe.shape[0] == 1

    robotwin_root = Path(args.robotwin_root).resolve()
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(robotwin_root / "scripts"))
    eval_module = importlib.import_module("eval_policy_xpolicylab")

    protocol_out = sys.stdout.buffer
    sys.stdout = sys.stderr
    task_args, _ = eval_module.load_task_args(
        {
            "task_name": args.task,
            "task_config": args.task_config,
            "policy_name": "sidneyxie_sim_eval",
            "ckpt_setting": "sidneyxie_pi05_robotwin",
        }
    )
    task_args.update(
        {
            "eval_mode": True,
            "need_plan": False,
            "render_freq": 0,
            "policy_name": "sidneyxie_sim_eval",
            "save_data": False,
            "collect_data": False,
        }
    )

    task_env = eval_module.class_decorator(args.task)
    accepted = 0
    seed = args.seed
    episode_reset_eval_success: list[bool] = []

    try:
        while accepted < args.episodes:
            try:
                task_env.setup_demo(
                    now_ep_num=accepted,
                    seed=seed,
                    is_test=True,
                    **task_args,
                )
                reset_success = bool(
                    getattr(task_env, "eval_success", False)
                )
                episode_reset_eval_success.append(reset_success)

                instruction = args.task
                task_env.set_instruction(instruction=instruction)
                send_message(
                    protocol_out,
                    {
                        "type": "episode_start",
                        "episode": accepted,
                        "seed": seed,
                        "instruction": instruction,
                        "observation": simulator_observation(
                            task_env.get_obs()
                        ),
                    },
                )

                step = 0
                while step < args.episode_steps:
                    command = receive_message(sys.stdin.buffer)
                    if command.get("cmd") == "stop":
                        return 0
                    if command.get("cmd") != "action":
                        raise RuntimeError(
                            f"unexpected parent command: {command}"
                        )

                    action = np.asarray(
                        command["action"], dtype=np.float32
                    ).reshape(-1)
                    assert action.shape == (ACTION_DIM,), (
                        f"parent action must be 14-D, got {action.shape}"
                    )
                    task_env.take_action(action, action_type="qpos")
                    step += 1

                    eval_success = bool(
                        getattr(task_env, "eval_success", False)
                    )
                    if eval_success:
                        success = True
                        # Audit the fallback signal even when the primary
                        # wrapper signal already reports success.
                        check_success = bool(task_env.check_success())
                    else:
                        check_success = bool(task_env.check_success())
                        success = check_success
                    terminated = success
                    truncated = step >= args.episode_steps
                    response = {
                        "type": "step",
                        "observation": None
                        if terminated or truncated
                        else simulator_observation(task_env.get_obs()),
                        "terminated": terminated,
                        "truncated": truncated,
                        "success": success,
                        "eval_success": eval_success,
                        "check_success": check_success,
                    }
                    send_message(protocol_out, response)
                    if terminated or truncated:
                        break

                safe_close_task(task_env)
                accepted += 1
                seed += 1
            except Exception as exc:
                safe_close_task(task_env)
                send_message(
                    protocol_out,
                    {
                        "type": "error",
                        "message": (
                            f"{type(exc).__name__}: {exc}\n"
                            f"{traceback.format_exc()}"
                        ),
                    },
                )
                return 2

        send_message(
            protocol_out,
            {
                "type": "complete",
                "episode_reset_eval_success": episode_reset_eval_success,
            },
        )
        return 0
    finally:
        safe_close_task(task_env)


def main() -> int:
    args = parse_args()
    if args.role == "sim":
        return run_simulator(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
