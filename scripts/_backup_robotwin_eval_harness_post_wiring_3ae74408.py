#!/usr/bin/env python3
"""Single-task RoboTwin GATE 3 evaluation harness.

Deployment specification:
  GPU model: NVIDIA RTX PRO 6000 Blackwell Server Edition (sm_120), cuda:0 only.
  Driver: NVIDIA 590.44.01; CUDA runtime reported as 13.1.
  Batch: 1 policy observation.
  Precision: bfloat16 policy inference; verify against extracted checkpoint config.
  Action mode: joint, 14 float32 values, one dual-arm action.
  Simulator: Python 3.10 child process; policy: Python 3.12 parent process.

The parent never imports SAPIEN or RoboTwin task modules. It loads the official
LeRobot PI05 checkpoint and communicates with a simulator child over a
length-prefixed pickle stream on stdin/stdout. The child follows the official
RoboTwin setup_demo/play_once/check_success protocol.
"""

from __future__ import annotations

import argparse
import importlib
import os
import pickle
import struct
import subprocess
import sys

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)
from robotwin_instruction_source import seen_instruction  # noqa: E402

from pathlib import Path
from typing import Any, BinaryIO

from robotwin_obs_bridge import (
    CAMERA_KEY_RENAME,
    DEFAULT_OUTPUT_HEIGHT,
    DEFAULT_OUTPUT_WIDTH,
    bridge_observation,
    decimate_frame_sequence,
)


DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/EvolveVLA/checkpoints/pi05_robotwin"
)
DEFAULT_SIM_PYTHON = "/root/autodl-tmp/conda_envs/robotwin_cu128/bin/python"
DEFAULT_TOKENIZER_PATH = "/root/autodl-tmp/paligemma_tokenizer"
DEFAULT_ROBOTWIN_ROOT = "/root/autodl-tmp/EvolveVLA/robotwin_repo"
DEFAULT_TASK = "beat_block_hammer"
DEFAULT_EPISODES = 5
DEFAULT_EPISODE_STEPS = 1200
DEFAULT_SEED = 100000
STATE_DIM = 14
ACTION_DIM = 14
OBS_HEIGHT = DEFAULT_OUTPUT_HEIGHT
OBS_WIDTH = DEFAULT_OUTPUT_WIDTH
SOURCE_FPS = 50
TARGET_FPS = 25

CAMERA_RENAME_MAP = {target: source for source, target in CAMERA_KEY_RENAME.items()}
RAW_CAMERAS = ("head_camera", "left_camera", "right_camera")
POLICY_CAMERA_KEYS = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


def send_message(stream: BinaryIO, payload: Any) -> None:
    encoded = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    stream.write(struct.pack("!Q", len(encoded)))
    stream.write(encoded)
    stream.flush()


def _read_exact(stream: BinaryIO, n: int) -> bytes:
    # Popen(bufsize=0) makes this a raw FileIO whose read(n) may return short;
    # loop until n bytes or a true EOF.
    chunks: list[bytes] = []
    remaining = n
    while remaining:
        chunk = stream.read(remaining)
        if not chunk:
            raise EOFError(f"IPC stream closed with {remaining} bytes outstanding")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def receive_message(stream: BinaryIO) -> Any:
    header = _read_exact(stream, 8)
    (size,) = struct.unpack("!Q", header)
    payload = _read_exact(stream, size)
    return pickle.loads(payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate official RoboTwin pi0.5 on one task."
    )
    parser.add_argument("--role", choices=("parent", "sim"), default="parent")
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--episode-steps", type=int, default=DEFAULT_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--robotwin-root", default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--task-config", default="demo_clean")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def derive_camera_rename_map(policy_cfg: Any) -> dict[str, str]:
    declared = {
        str(key)
        for key in (getattr(policy_cfg, "input_features", {}) or {})
        if "observation.images." in str(key)
    }
    if not declared:
        raise RuntimeError("checkpoint declares no observation image features")

    rename_map: dict[str, str] = {}
    for source, target in CAMERA_RENAME_MAP.items():
        if target in declared:
            rename_map[source] = target
        elif source not in declared:
            raise RuntimeError(
                "checkpoint camera contract is not the expected RoboTwin contract: "
                f"missing {target!r}; declared={sorted(declared)}"
            )

    resolved = set(rename_map.values()) | {
        source for source in CAMERA_RENAME_MAP if source in declared
    }
    if resolved != declared:
        raise RuntimeError(
            "checkpoint declares unsupported or incomplete camera keys: "
            f"declared={sorted(declared)} resolved={sorted(resolved)}"
        )
    return rename_map


def load_policy(args: argparse.Namespace) -> tuple[Any, Any, Any, Any, Any]:
    import torch
    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; GATE 3 requires cuda:0")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected exactly one visible GPU, got {torch.cuda.device_count()}"
        )
    if not Path(args.model_path).is_dir():
        raise FileNotFoundError(
            f"checkpoint directory does not exist: {args.model_path}"
        )
    if not Path(args.tokenizer_path).is_dir():
        raise FileNotFoundError(
            "official gated tokenizer directory is missing: "
            f"{args.tokenizer_path}"
        )

    env_cfg = RoboTwinEnvConfig(
        task=args.task,
        episode_length=args.episode_steps,
        obs_type="pixels_agent_pos",
        camera_names="cam_high,cam_left_wrist,cam_right_wrist",
        observation_height=OBS_HEIGHT,
        observation_width=OBS_WIDTH,
        action_mode="joint",
    )
    policy_cfg = PI05Config.from_pretrained(
        args.model_path, local_files_only=True
    )
    policy_cfg.device = "cuda:0"
    policy_cfg.pretrained_path = Path(args.model_path)
    rename_map = derive_camera_rename_map(policy_cfg)

    from lerobot.policies.pi05.modeling_pi05 import PI05Policy
    policy = PI05Policy.from_pretrained(
        args.model_path, config=policy_cfg, local_files_only=True
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=args.model_path,
        preprocessor_overrides={
            "device_processor": {"device": "cuda:0"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": args.tokenizer_path},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy.config
    )

    expected_state = getattr(policy.config, "input_features", {}).get(
        "observation.state"
    )
    if expected_state is not None and tuple(expected_state.shape) != (STATE_DIM,):
        raise RuntimeError(
            "checkpoint state contract is not the required 14-D RoboTwin joint "
            f"state: {expected_state.shape}"
        )
    expected_action = getattr(policy.config, "output_features", {}).get("action")
    if expected_action is not None and tuple(expected_action.shape) != (ACTION_DIM,):
        raise RuntimeError(
            "checkpoint action contract is not the required 14-D joint action: "
            f"{expected_action.shape}"
        )

    print(f"CHECKPOINT={args.model_path}")
    print(f"CAMERA_RENAME_MAP={rename_map}")
    print(f"POLICY_DTYPE={getattr(policy.config, 'dtype', 'checkpoint-configured')}")
    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def make_policy_batch(
    observation: dict[str, Any],
    instruction: str,
    env_preprocessor: Any,
    preprocessor: Any,
    policy_config: Any,
) -> dict[str, Any]:
    import numpy as np
    import torch

    observation = dict(observation)
    state = observation.get("agent_pos")
    if state is None:
        raise RuntimeError("simulator observation has no agent_pos state")
    state = np.asarray(state, dtype=np.float32).reshape(-1)
    assert state.shape == (STATE_DIM,), (
        f"simulator state must be 14-D, got {state.shape}"
    )
    pixels = observation.get("pixels")
    if not isinstance(pixels, dict):
        raise RuntimeError("simulator observation has no pixels mapping")
    obs: dict[str, Any] = {
        "observation.state": torch.from_numpy(state.astype(np.float32)),
        "task": [instruction],
    }
    for cam_name, frame in pixels.items():
        obs[POLICY_CAMERA_KEYS[cam_name]] = frame
    batch = preprocessor(obs)
    got_image_keys = {
        key for key in batch if "observation.images." in str(key)
    }
    expected_image_keys = {
        key
        for key in (getattr(policy_config, "input_features", {}) or {})
        if "observation.images." in str(key)
    }
    if got_image_keys != expected_image_keys:
        raise RuntimeError(
            f"image batch key mismatch: expected={sorted(expected_image_keys)} "
            f"got={sorted(got_image_keys)}"
        )
    return batch

def run_parent(args: argparse.Namespace) -> int:
    import json
    import numpy as np
    import torch
    from lerobot.utils.constants import ACTION

    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    if args.episode_steps <= 0:
        raise ValueError("--episode-steps must be positive")
    if args.episode_steps > DEFAULT_EPISODE_STEPS:
        raise ValueError(
            f"--episode-steps must be <= {DEFAULT_EPISODE_STEPS} for GATE 3"
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
        cwd=args.robotwin_root,
    )
    assert child.stdin is not None
    assert child.stdout is not None

    reports: list[dict[str, Any]] = []
    try:
        while True:
            message = receive_message(child.stdout)
            message_type = message.get("type")
            if message_type == "error":
                raise RuntimeError(message["message"])
            if message_type == "complete":
                break
            if message_type != "episode_start":
                raise RuntimeError(f"unexpected simulator message: {message_type}")

            policy.reset()
            instruction = str(message["instruction"])
            observation = message["observation"]
            episode_steps = 0
            success = False
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
                action_array = (
                    action.detach().float().cpu().numpy().reshape(-1)
                )
                assert action_array.shape == (ACTION_DIM,), (
                    f"policy action must be shape (14,), got {action_array.shape}"
                )
                if not np.isfinite(action_array).all():
                    raise RuntimeError("policy returned a non-finite action")
                action_list = np.asarray(action_array, dtype=np.float32).reshape(-1).tolist()
                send_message(child.stdin, {"cmd": "action", "action": action_list})
                response = receive_message(child.stdout)
                if response.get("type") == "error":
                    raise RuntimeError(response["message"])
                if response.get("type") != "step":
                    raise RuntimeError(
                        f"unexpected simulator step message: {response.get('type')}"
                    )
                episode_steps += 1
                success = bool(response["info"].get("is_success", False)) or success
                if response["terminated"] or response["truncated"]:
                    break
                observation = response["observation"]

            result = {
                "episode": int(message["episode"]),
                "seed": int(message["seed"]),
                "success": success,
                "steps": episode_steps,
            }
            reports.append(result)
            print(
                f"EPISODE {result['episode'] + 1}/{args.episodes}: "
                f"success={success} steps={episode_steps}"
            )
    finally:
        try:
            send_message(child.stdin, {"cmd": "stop"})
        except (BrokenPipeError, OSError):
            pass
        child.stdin.close()
        return_code = child.wait()
        if return_code != 0 and not reports:
            raise RuntimeError(f"simulator child exited with code {return_code}")

    successes = sum(bool(item["success"]) for item in reports)
    rollouts = len(reports)
    assert successes <= rollouts, "success count cannot exceed completed rollouts"
    assert all(
        int(item["steps"]) <= DEFAULT_EPISODE_STEPS for item in reports
    ), "a rollout exceeded the 1200-step audit cap"
    assert STATE_DIM == ACTION_DIM == 14
    percent = 100.0 * successes / rollouts if rollouts else 0.0
    verdict = f"GATE 3 SUCCESS_RATE={successes}/{rollouts} ({percent:.1f}%)"
    print(verdict)
    payload = {
        "task": args.task,
        "checkpoint": args.model_path,
        "episodes_requested": args.episodes,
        "episodes_completed": rollouts,
        "successes": successes,
        "success_rate": successes / rollouts if rollouts else 0.0,
        "episode_length": args.episode_steps,
        "action_mode": "joint",
        "state_dim": STATE_DIM,
        "action_dim": ACTION_DIM,
        "camera_rename_map": CAMERA_RENAME_MAP,
        "camera_output_shape": [OBS_HEIGHT, OBS_WIDTH, 3],
        "source_fps": SOURCE_FPS,
        "target_fps": TARGET_FPS,
        "reports": reports,
        "verdict": verdict,
    }
    text = json.dumps(payload, indent=2, ensure_ascii=True)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    return 0


def simulator_observation(raw: dict[str, Any]) -> dict[str, Any]:
    import numpy as np

    cameras = raw.get("observation", {})
    dataset_images: dict[str, Any] = {}
    for camera_name in RAW_CAMERAS:
        camera = cameras.get(camera_name) or {}
        image = camera.get("rgb")
        if image is None:
            raise RuntimeError(f"missing RoboTwin camera frame: {camera_name}")
        image = np.asarray(image, dtype=np.uint8)
        if image.ndim == 2:
            image = np.repeat(image[..., None], 3, axis=-1)
        if image.ndim != 3 or image.shape[-1] != 3:
            raise RuntimeError(
                f"camera {camera_name} must be HxWx3 uint8, got {image.shape}"
            )
        dataset_images[POLICY_CAMERA_KEYS[camera_name]] = image

    # The bridge owns dataset-to-wrapper camera renaming and the 480x640 ->
    # 240x320 spatial adapter. This harness receives one frame per policy
    # action, so 50 -> 25 FPS decimation remains a sequence-boundary concern.
    bridged = bridge_observation(
        dataset_images,
        output_height=OBS_HEIGHT,
        output_width=OBS_WIDTH,
    )
    import torch

    # The bridge and simulator stay HWC uint8. At the PI05 policy boundary,
    # modeling_pi05.py:985-1001 expects CHW and batch_processor.py:116-119
    # adds the batch dimension. PI05's image path expects float32 pixels in
    # [0, 1], so convert only after the HWC bridge output is assembled.
    pixels = {
        camera_name: (
            torch.from_numpy(
                bridged[f"observation.images.{camera_name}"]
            )
            .permute(2, 0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            .div(255.0)
        )
        for camera_name in RAW_CAMERAS
    }

    joint_action = raw.get("joint_action") or {}
    vector = joint_action.get("vector")
    if vector is None:
        vector = (
            list(joint_action.get("left_arm", []))
            + [joint_action.get("left_gripper", 0.0)]
            + list(joint_action.get("right_arm", []))
            + [joint_action.get("right_gripper", 0.0)]
        )
    state = np.asarray(vector, dtype=np.float32).reshape(-1)
    assert state.shape == (STATE_DIM,), (
        f"RoboTwin joint state must be 14-D, got {state.shape}"
    )
    return {"pixels": pixels, "agent_pos": state}


def safe_close_task(task_env: Any) -> None:
    close_env = getattr(task_env, "close_env", None)
    if callable(close_env):
        close_env()


def run_simulator(args: argparse.Namespace) -> int:
    import traceback

    import numpy as np

    fps_probe = decimate_frame_sequence(
        np.zeros((2, 1, 1, 3), dtype=np.uint8),
        source_fps=SOURCE_FPS,
        target_fps=TARGET_FPS,
    )
    assert fps_probe.shape[0] == 1, "50 -> 25 FPS bridge contract failed"

    robotwin_root = Path(args.robotwin_root).resolve()
    sys.path.insert(0, str(robotwin_root))
    sys.path.insert(0, str(robotwin_root / "scripts"))
    # RoboTwin modules open bundle-relative paths (e.g. ./assets/objects/...)
    # at import time, so the CWD must be robotwin_root before importing them.
    os.chdir(robotwin_root)
    # Native libs (SAPIEN / svulkan2 / OIDN / Warp) write straight to fd 1 and
    # would corrupt the length-prefixed IPC stream. Move all of fd 1 to stderr
    # and keep a private duplicate for the pickled protocol.
    protocol_fd = os.dup(1)
    os.dup2(2, 1)
    protocol_out = os.fdopen(protocol_fd, "wb")
    eval_module = importlib.import_module("eval_policy_xpolicylab")
    class_decorator = eval_module.class_decorator
    load_task_args = eval_module.load_task_args
    build_instruction = eval_module.build_instruction
    ROBOTWIN_SKIP_EXPERT = os.environ.get("ROBOTWIN_SKIP_EXPERT", "")
    SKIP_EXPERT = ROBOTWIN_SKIP_EXPERT == "1"

    config = {
        "task_name": args.task,
        "task_config": args.task_config,
        "policy_name": "robotwin_eval_harness",
        "ckpt_setting": "official_pi05",
    }
    task_args, _ = load_task_args(config)
    task_args["eval_mode"] = True
    task_args["render_freq"] = 0
    task_args["policy_name"] = "robotwin_eval_harness"
    task_env = class_decorator(args.task)
    accepted = 0
    seed = args.seed

    try:
        while accepted < args.episodes:
            episode_info = {"info": {}}
            try:
                if not SKIP_EXPERT:
                    task_env.setup_demo(
                        now_ep_num=accepted,
                        seed=seed,
                        is_test=True,
                        **task_args,
                    )
                    episode_info = task_env.play_once()
                    expert_ok = bool(
                        getattr(task_env, "plan_success", True)
                        and task_env.check_success()
                    )
                    safe_close_task(task_env)
                    if not expert_ok:
                        seed += 1
                        continue

                task_env.setup_demo(
                    now_ep_num=accepted,
                    seed=seed,
                    is_test=True,
                    **task_args,
                )
                instruction, instruction_source = seen_instruction(
                    args.task, arm_tag=None, seed=seed
                )
                if instruction_source == "task_name_fallback":
                    # Helper could not build a seen instruction -> official path.
                    instruction = build_instruction(
                        task_args,
                        episode_info,
                        task_args.get("eval_instruction"),
                        args.episodes,
                        seed,
                    )
                    instruction_source = "task_name_fallback"
                task_env.set_instruction(instruction=instruction)
                raw_observation = task_env.get_obs()
                print(f"INSTRUCTION={instruction!r}")
                print("EXPERT_PREFILTER=" + ("OFF" if SKIP_EXPERT else "ON"))
                try:
                    _src = instruction_source
                except Exception:
                    _src = "unknown"
                print(f"INSTRUCTION_SOURCE={_src}")
                send_message(
                    protocol_out,
                    {
                        "type": "episode_start",
                        "episode": accepted,
                        "seed": seed,
                        "instruction": instruction,
                        "observation": simulator_observation(raw_observation),
                    },
                )

                step = 0
                while step < args.episode_steps:
                    command = receive_message(sys.stdin.buffer)
                    if command.get("cmd") == "stop":
                        return 0
                    if command.get("cmd") != "action":
                        raise RuntimeError(f"unexpected parent command: {command}")
                    action = np.asarray(command["action"], dtype=np.float32).reshape(-1)
                    assert action.shape == (ACTION_DIM,), (
                        f"parent action must be 14-D, got {action.shape}"
                    )
                    task_env.take_action(action, action_type="qpos")
                    step += 1
                    is_success = bool(getattr(task_env, "eval_success", False))
                    if not is_success:
                        is_success = bool(task_env.check_success())
                    terminated = is_success
                    truncated = step >= args.episode_steps
                    info = {
                        "is_success": is_success,
                        "task": args.task,
                        "step": step,
                    }
                    if terminated or truncated:
                        send_message(
                            protocol_out,
                            {
                                "type": "step",
                                "observation": None,
                                "terminated": terminated,
                                "truncated": truncated,
                                "info": info,
                            },
                        )
                        break
                    send_message(
                        protocol_out,
                        {
                            "type": "step",
                            "observation": simulator_observation(task_env.get_obs()),
                            "terminated": False,
                            "truncated": False,
                            "info": info,
                        },
                    )

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
        send_message(protocol_out, {"type": "complete"})
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
