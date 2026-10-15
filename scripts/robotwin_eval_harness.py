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
import datetime
import faulthandler
import importlib
import os
import pickle
import struct
import signal
import subprocess
import sys
import threading
import time

_SCRIPTS_DIR = os.path.dirname(os.path.abspath(__file__))
if _SCRIPTS_DIR not in sys.path:
    sys.path.insert(0, _SCRIPTS_DIR)

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
# Safety ceiling only. The simulator stops at each task's official step_lim;
# the largest current official limit is 1700.
DEFAULT_EPISODE_STEPS = 2000
DEFAULT_SEED = 100000
STATE_DIM = 14
ACTION_DIM = 14
OBS_HEIGHT = int(os.environ.get("ROBOTWIN_OBS_HEIGHT", DEFAULT_OUTPUT_HEIGHT))
OBS_WIDTH = int(os.environ.get("ROBOTWIN_OBS_WIDTH", DEFAULT_OUTPUT_WIDTH))
SOURCE_FPS = 50
TARGET_FPS = 25
OFFICIAL_INSTRUCTION_CANDIDATES = 100

CAMERA_RENAME_MAP = {target: source for source, target in CAMERA_KEY_RENAME.items()}
RAW_CAMERAS = ("head_camera", "left_camera", "right_camera")
POLICY_CAMERA_KEYS = {
    "head_camera": "observation.images.cam_high",
    "left_camera": "observation.images.cam_left_wrist",
    "right_camera": "observation.images.cam_right_wrist",
}


def _proc_fd_count(pid: int | str) -> int:
    try:
        return len(os.listdir(f"/proc/{pid}/fd"))
    except Exception:
        return -1


def _proc_thread_count(pid: int | str) -> int:
    try:
        with open(f"/proc/{pid}/status", "r", encoding="utf-8") as handle:
            for line in handle:
                if line.startswith("Threads:"):
                    return int(line.split()[1])
    except Exception:
        return -1
    return -1


def _stall_snapshot(
    snapshot_dir: str,
    log_path: str,
    child_pid: int,
    last_step: object,
    reason: str,
) -> None:
    try:
        os.makedirs(snapshot_dir, exist_ok=True)
    except Exception:
        pass
    try:
        ts = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    except Exception:
        ts = "unknown-time"
    snapshot_path = os.path.join(snapshot_dir, f"WATCHDOG_SNAPSHOT_{ts}.txt")
    lines: list[str] = []
    try:
        now = time.time()
        lines.append(f"timestamp={datetime.datetime.now(datetime.timezone.utc).isoformat()}")
        lines.append(f"reason={reason}")
        lines.append(f"log_path={log_path}")
        try:
            stat_result = os.stat(log_path)
            lines.append(f"log_mtime={stat_result.st_mtime}")
            lines.append(f"log_age={now - stat_result.st_mtime:.1f}")
        except Exception as exc:
            lines.append(f"log_mtime=ERROR {type(exc).__name__}: {exc}")
            lines.append("log_age=unknown")
        try:
            lines.append(f"self_fds={len(os.listdir('/proc/self/fd'))}")
        except Exception as exc:
            lines.append(f"self_fds=ERROR {type(exc).__name__}: {exc}")
        try:
            child_fd_dir = f"/proc/{child_pid}/fd"
            if os.path.isdir(child_fd_dir):
                lines.append(f"child_fds={len(os.listdir(child_fd_dir))}")
            else:
                lines.append("child_fds=missing")
        except Exception as exc:
            lines.append(f"child_fds=ERROR {type(exc).__name__}: {exc}")
        try:
            lines.append(f"self_threads={threading.active_count()}")
        except Exception as exc:
            lines.append(f"self_threads=ERROR {type(exc).__name__}: {exc}")
        try:
            with open(f"/proc/{child_pid}/wchan", "r", encoding="utf-8") as handle:
                lines.append(f"child_wchan={handle.read().strip()}")
        except Exception as exc:
            lines.append(f"child_wchan=ERROR {type(exc).__name__}: {exc}")
        try:
            with open("/proc/self/wchan", "r", encoding="utf-8") as handle:
                lines.append(f"self_wchan={handle.read().strip()}")
        except Exception as exc:
            lines.append(f"self_wchan=ERROR {type(exc).__name__}: {exc}")
        lines.append("NOTE=killpg may not reap an uninterruptible (D-state/do_poll) child; its value is fail-fast + forensics, not recovery")
        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-compute-apps=pid,used_memory",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
            lines.append("nvidia_smi_stdout=<<<")
            lines.append(smi.stdout.rstrip())
            lines.append(">>>")
            lines.append("nvidia_smi_stderr=<<<")
            lines.append(smi.stderr.rstrip())
            lines.append(">>>")
            lines.append(f"nvidia_smi_returncode={smi.returncode}")
        except Exception as exc:
            lines.append(f"nvidia_smi=ERROR {type(exc).__name__}: {exc}")
        lines.append(f"last_step={last_step}")
        try:
            with open(snapshot_path, "w", encoding="utf-8") as handle:
                handle.write("\n".join(lines) + "\n")
        except Exception:
            pass
        try:
            with open(snapshot_path, "a", encoding="utf-8") as handle:
                handle.write(chr(10) + "parent_faulthandler=<<<" + chr(10))
                faulthandler.dump_traceback(file=handle, all_threads=True)
                handle.write(">>>" + chr(10))
                try:
                    with open(f"/proc/{child_pid}/cmdline", "rb") as cmdline_handle:
                        cmdline = (
                            cmdline_handle.read()
                            .replace(bytes([0]), b" ")
                            .decode("utf-8", errors="replace")
                            .strip()
                        )
                    handle.write(f"child_cmdline={cmdline}" + chr(10))
                except Exception as exc:
                    handle.write(
                        f"child_cmdline=ERROR {type(exc).__name__}: {exc}" + chr(10)
                    )
                try:
                    child_pyspy = subprocess.run(
                        ["py-spy", "dump", "--pid", str(child_pid)],
                        capture_output=True,
                        text=True,
                        timeout=15,
                    )
                    handle.write("child_pyspy=<<<" + chr(10))
                    handle.write("stdout:" + chr(10))
                    handle.write(child_pyspy.stdout)
                    if child_pyspy.stdout and not child_pyspy.stdout.endswith(chr(10)):
                        handle.write(chr(10))
                    handle.write("stderr:" + chr(10))
                    handle.write(child_pyspy.stderr)
                    if child_pyspy.stderr and not child_pyspy.stderr.endswith(chr(10)):
                        handle.write(chr(10))
                    handle.write(f"returncode={child_pyspy.returncode}" + chr(10))
                    handle.write(">>>" + chr(10))
                except FileNotFoundError:
                    handle.write("child_pyspy=UNAVAILABLE" + chr(10))
                except Exception as exc:
                    handle.write(
                        f"child_pyspy=ERROR {type(exc).__name__}: {exc}" + chr(10)
                    )
        except Exception:
            pass
    except Exception:
        pass


def _heartbeat_watchdog(
    log_path: str,
    child_pid: int,
    state: dict[str, object],
    stall_seconds: int = 120,
    poll: int = 10,
) -> None:
    snapshot_dir = os.path.join(os.getcwd(), "results")
    last_progress_time = time.time()
    last_seen_step = state.get("last_step", "unknown")
    while True:
        time.sleep(poll)
        self_fds = _proc_fd_count("self")
        child_fds = _proc_fd_count(child_pid)
        child_threads = _proc_thread_count(child_pid)
        cur_step = state.get("last_step", "unknown")
        print(
            f"STEP_METRICS child_fds={child_fds} child_threads={child_threads} "
            f"self_fds={self_fds} el={cur_step}",
            file=sys.stderr,
            flush=True,
        )
        now = time.time()
        if cur_step != last_seen_step:
            last_seen_step = cur_step
            last_progress_time = now
            continue
        age = now - last_progress_time
        if age > stall_seconds:
            print(
                f"WATCHDOG_TIMEOUT no_progress step={cur_step} stall={age:.1f}s",
                file=sys.stderr,
                flush=True,
            )
            _stall_snapshot(
                snapshot_dir,
                log_path,
                child_pid,
                cur_step,
                f"no step progress for {age:.1f}s",
            )
            try:
                import signal as _signal
                try:
                    os.kill(child_pid, _signal.SIGUSR1)
                    print(
                        "FAULTHANDLER_CHILD_DUMP_REQUESTED",
                        file=sys.stderr,
                        flush=True,
                    )
                    time.sleep(5)
                except Exception as exc:
                    print(
                        f"FAULTHANDLER_CHILD_DUMP_FAILED {exc!r}",
                        file=sys.stderr,
                        flush=True,
                    )
                os.killpg(os.getpgid(child_pid), _signal.SIGKILL)
            except Exception:
                pass
            os._exit(124)


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
    parser.add_argument(
        "--video-dir",
        type=Path,
        help="Optional directory for official RoboTwin-style head-camera MP4 output.",
    )
    parser.add_argument("--controlled-case", action="store_true",
                        help="run exactly one scheduled seed; reject without substitution")
    parser.add_argument("--first-instruction")
    parser.add_argument("--first-instruction-seed", type=int)
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
    def _check_lm_head_tie(branch):
        try:
            if branch == "paligemma":
                head = policy.model.paligemma_with_expert.paligemma.lm_head
                embed = (
                    policy.model.paligemma_with_expert.paligemma.model
                    .language_model.embed_tokens
                )
            elif branch == "gemma_expert":
                head = policy.model.paligemma_with_expert.gemma_expert.lm_head
                embed = policy.model.paligemma_with_expert.gemma_expert.model.embed_tokens
            else:
                raise ValueError(f"unknown lm_head tie branch: {branch}")
            if head is None:
                raise AttributeError("lm_head is None")
            if embed is None:
                raise AttributeError("embed_tokens is None")
            return bool(torch.equal(head.weight, embed.weight))
        except Exception as exc:
            print(
                f"LM_HEAD_TIED {branch}=UNVERIFIED reason={exc!r}",
                flush=True,
            )
            return None

    _pg_tied = _check_lm_head_tie("paligemma")
    _expert_tied = _check_lm_head_tie("gemma_expert")
    print(f"LM_HEAD_TIED paligemma={_pg_tied} gemma_expert={_expert_tied} (language heads; informational)", flush=True)
    _ao = getattr(policy.model, "action_out_proj", None)
    _ai = getattr(policy.model, "action_in_proj", None)
    if _ao is None or _ai is None or getattr(_ao, "weight", None) is None or getattr(_ai, "weight", None) is None:
        print("ACTION_PROJ=MISSING_INIT (action_in_proj/action_out_proj absent -> unusable)", flush=True)
        os._exit(3)
    print(f"ACTION_PROJ=LOADED action_out_proj={tuple(_ao.weight.shape)} action_in_proj={tuple(_ai.weight.shape)}", flush=True)

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
    if not getattr(make_policy_batch, "_printed_img_range", False):
        _ranges = {}
        for _k, _v in batch.items():
            if "observation.images." in str(_k) and torch.is_tensor(_v):
                _ranges[str(_k)] = (float(_v.detach().amin().cpu()), float(_v.detach().amax().cpu()))
        print(f"IMG_RANGE={_ranges}", file=sys.stderr, flush=True)
        make_policy_batch._printed_img_range = True
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

    heartbeat_log = os.environ.get("ROBOTWIN_HEARTBEAT_LOG")
    if not heartbeat_log or not os.path.isabs(heartbeat_log):
        raise RuntimeError(
            "ROBOTWIN_HEARTBEAT_LOG must be an absolute path to the run stdout log"
        )

    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        load_policy(args)
    )
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = "0"
    child_env["PYTHONUNBUFFERED"] = "1"
    # os.setpgrp: child must stay in the parent's process group so an external timeout group-kill reaps both; start_new_session orphaned the sim child
    child_command = [
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
        ]
    if args.video_dir is not None:
        child_command.extend(["--video-dir", str(args.video_dir)])
    child = subprocess.Popen(
        child_command,
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=None,
        bufsize=0,
        env=child_env,
        cwd=args.robotwin_root,
        preexec_fn=os.setpgrp,
    )
    assert child.stdin is not None
    assert child.stdout is not None
    import faulthandler as _fh
    _fh_fd = None
    try:
        _fh_path = os.path.join(os.path.dirname(_SCRIPTS_DIR), "results", "FAULTHANDLER_DUMP.txt")
        _fh_fd = open(_fh_path, "a")
        _fh.enable(file=_fh_fd, all_threads=True)
        _fh.dump_traceback_later(180, repeat=True, exit=False, file=_fh_fd)
    except Exception as _exc:
        print(f"FAULTHANDLER_SETUP_FAILED {_exc!r}", flush=True)
    try:
        last_mtime = os.stat(heartbeat_log).st_mtime
    except FileNotFoundError:
        last_mtime = time.time()
    watchdog_state: dict[str, object] = {
        "last_mtime": last_mtime,
        "last_step": "startup",
    }
    threading.Thread(
        target=_heartbeat_watchdog,
        args=(heartbeat_log, child.pid, watchdog_state),
        name="heartbeat_watchdog",
        daemon=True,
    ).start()

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
            action_episode_min: float | None = None
            action_episode_max: float | None = None
            watchdog_state["last_step"] = (
                f"episode={int(message['episode'])} step={episode_steps}"
            )
            success = False
            while True:
                batch = make_policy_batch(
                    observation,
                    instruction,
                    env_preprocessor,
                    preprocessor,
                    policy.config,
                )
                print("POLICY_IMG_SHAPE", {k: getattr(v, "shape", None) for k, v in batch.items() if "observation.images." in str(k)}, flush=True)
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
                action_min = float(np.min(action_array))
                action_max = float(np.max(action_array))
                action_episode_min = (
                    action_min
                    if action_episode_min is None
                    else min(action_episode_min, action_min)
                )
                action_episode_max = (
                    action_max
                    if action_episode_max is None
                    else max(action_episode_max, action_max)
                )
                action_list = np.asarray(action_array, dtype=np.float32).reshape(-1).tolist()
                watchdog_state["last_step"] = (
                    f"episode={int(message['episode'])} step={episode_steps}"
                )
                send_message(child.stdin, {"cmd": "action", "action": action_list})
                response = receive_message(child.stdout)
                if response.get("type") == "error":
                    raise RuntimeError(response["message"])
                if response.get("type") != "step":
                    raise RuntimeError(
                        f"unexpected simulator step message: {response.get('type')}"
                    )
                episode_steps += 1
                watchdog_state["last_step"] = (
                    f"episode={int(message['episode'])} step={episode_steps}"
                )
                success = bool(response["info"].get("is_success", False)) or success
                if response["terminated"] or response["truncated"]:
                    break
                observation = response["observation"]

            result = {
                "episode": int(message["episode"]),
                "seed": int(message["seed"]),
                "success": success,
                "steps": episode_steps,
                "block_pose_x": message.get("block_pose_x"),
                "arm_tag": message.get("arm_tag"),
                "arm_binding_source": message.get("arm_binding_source"),
                "instruction_source": message.get("instruction_source"),
                "action_postprocessed_min": action_episode_min,
                "action_postprocessed_max": action_episode_max,
            }
            reports.append(result)
            print(
                f"EPISODE {result['episode'] + 1}/{args.episodes}: "
                f"success={success} steps={episode_steps}"
            , flush=True)
            print(
                "ACTION_POSTPROCESSED_MINMAX "
                f"seed={result['seed']} "
                f"min={action_episode_min} max={action_episode_max}",
                flush=True,
            )
    finally:
        try:
            send_message(child.stdin, {"cmd": "stop"})
        except (BrokenPipeError, OSError):
            pass
        child.stdin.close()
        try:
            return_code = child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            try:
                os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            except Exception:
                pass
            try:
                return_code = child.wait(timeout=10)
            except subprocess.TimeoutExpired:
                return_code = child.returncode
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
        print("RAW_CAM_SHAPE", camera_name, image.shape, flush=True)
        dataset_images[POLICY_CAMERA_KEYS[camera_name]] = image

    # The bridge owns dataset-to-wrapper camera renaming and the observation resize hook.
    # D435 sim cameras natively output 240x320, so the default 240x320 target is an
    # IDENTITY (not a 480x640 -> 240x320 downsample); ROBOTWIN_OBS_HEIGHT/WIDTH can override the target.
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
    import inspect
    import traceback

    try:
        import faulthandler as _fh_sim
        _sim_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _fh_sim_path = (os.path.join(os.environ['ROBOTWIN_CONTROLLED_OUTPUT'], 'faulthandler_sim.txt')
                        if getattr(args, 'controlled_case', False)
                        else os.path.join(_sim_root, "results", "FAULTHANDLER_DUMP_SIM.txt"))
        _fh_sim_fd = open(_fh_sim_path, "a")
        _fh_sim.enable(file=_fh_sim_fd, all_threads=True)
        _fh_sim.register(
            signal.SIGUSR1,
            file=_fh_sim_fd,
            all_threads=True,
            chain=False,
        )
        _fh_sim.dump_traceback_later(180, repeat=True, exit=False, file=_fh_sim_fd)
        print(f"FAULTHANDLER_SIM enabled {_fh_sim_path}", file=sys.stderr, flush=True)
    except Exception as _exc:
        print(f"FAULTHANDLER_SIM_SETUP_FAILED {_exc!r}", file=sys.stderr, flush=True)

    import numpy as np
    import json

    def _jsonable(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): _jsonable(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [_jsonable(item) for item in value]
        if hasattr(value, "tolist"):
            return _jsonable(value.tolist())
        return value

    def _normalize_arm_tag(value: Any) -> str | None:
        text = str(value).strip().lower()
        if text in {"left", "right"}:
            return text
        return None

    def _arm_from_episode_info(episode_info: dict[str, Any]) -> tuple[str | None, str]:
        info = episode_info.get("info") if isinstance(episode_info, dict) else None
        if isinstance(info, dict):
            arm_tag = _normalize_arm_tag(info.get("{a}"))
            if arm_tag is not None:
                return arm_tag, 'episode_info["info"]["{a}"]'
        return None, "episode_info_missing"

    def _reference_pose_x(task_env: Any) -> float | None:
        """Return a stable task-object x coordinate when one is available.

        Different official RoboTwin tasks expose different actor attributes
        (`block`, `bottle`, `cup`, ...).  This value is diagnostic only; arm
        selection is sourced from official ``play_once`` info whenever it is
        present and never depends on a guessed actor name.
        """
        for name in ("block", "bottle", "cup", "pot", "plate", "object", "container", "box"):
            actor = getattr(task_env, name, None)
            if actor is None or not hasattr(actor, "get_functional_point"):
                continue
            try:
                pose = actor.get_functional_point(0, "pose").p
                return float(np.asarray(pose)[0])
            except Exception:
                try:
                    pose = actor.get_functional_point(0)
                    return float(np.asarray(pose)[0])
                except Exception:
                    continue
        return None

    def _arm_from_reference_pose(reference_pose_x: float | None) -> str | None:
        if reference_pose_x is None:
            return None
        return "left" if reference_pose_x < 0 else "right"

    def _dump_cam_high_png(
        raw: dict[str, Any],
        episode: int,
        seed: int,
        step: int,
    ) -> None:
        try:
            frame = raw["observation"]["head_camera"]["rgb"]
            image = np.asarray(frame, dtype=np.uint8)
            if image.ndim == 2:
                image = np.repeat(image[..., None], 3, axis=-1)
            if image.ndim != 3 or image.shape[-1] != 3:
                raise RuntimeError(f"cam_high frame must be HxWx3, got {image.shape}")
            dump_dir_value = os.environ.get("ROBOTWIN_CAM_DUMP_DIR")
            if not dump_dir_value:
                return
            dump_dir = Path(dump_dir_value)
            dump_dir.mkdir(parents=True, exist_ok=True)
            path = dump_dir / f"episode{episode}_seed{seed}_step{step}_cam_high.png"
            from PIL import Image

            Image.fromarray(image).save(path)
            print(f"CAM_HIGH_PNG step={step} path={path}", flush=True)
        except Exception as exc:
            print(
                f"CAM_HIGH_PNG_ERROR step={step} error={type(exc).__name__}: {exc}",
                flush=True,
            )

    def _start_eval_video(task_env: Any, raw: dict[str, Any], episode: int):
        if args.video_dir is None:
            return None
        frame = np.asarray(raw["observation"]["head_camera"]["rgb"], dtype=np.uint8)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise RuntimeError(f"video frame must be HxWx3, got {frame.shape}")
        video_dir = Path(args.video_dir)
        video_dir.mkdir(parents=True, exist_ok=True)
        task_env.eval_video_path = str(video_dir)
        output = video_dir / f"episode{episode}_seed{seed}.mp4"
        try:
            ffmpeg_bin = "ffmpeg"
            import shutil

            if shutil.which(ffmpeg_bin) is None:
                import imageio_ffmpeg

                ffmpeg_bin = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception as exc:
            raise RuntimeError("video output requires ffmpeg or imageio-ffmpeg") from exc
        return subprocess.Popen(
            [
                ffmpeg_bin, "-y", "-loglevel", "error",
                "-f", "rawvideo", "-pixel_format", "rgb24",
                "-video_size", f"{frame.shape[1]}x{frame.shape[0]}",
                "-framerate", str(TARGET_FPS), "-i", "-",
                "-pix_fmt", "yuv420p", "-vcodec", "libx264", "-crf", "23",
                str(output),
            ],
            stdin=subprocess.PIPE,
        )

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
    build_instruction_accepts_seed = len(inspect.signature(build_instruction).parameters) >= 5
    controlled_case = bool(getattr(args, "controlled_case", False))
    if controlled_case and (args.episodes != 1 or args.first_instruction is None
                            or args.first_instruction_seed != args.seed):
        raise ValueError("controlled case requires one scheduled seed and fixed instruction")
    if controlled_case and os.environ.get("ROBOTWIN_SKIP_EXPERT"):
        raise ValueError("controlled case forbids ROBOTWIN_SKIP_EXPERT")
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
    if args.video_dir is not None:
        task_args["eval_video_save_dir"] = str(args.video_dir)
    task_env = class_decorator(args.task)
    trajectory_stream = None
    trajectory_path = os.environ.get("ROBOTWIN_TRAJECTORY_JSONL")
    if trajectory_path:
        Path(trajectory_path).parent.mkdir(parents=True, exist_ok=True)
        trajectory_stream = open(
            trajectory_path, "w", encoding="utf-8", buffering=1
        )
    accepted = 0
    seed = args.seed
    consecutive_pre_action_errors = 0

    try:
        while accepted < args.episodes:
            episode_info = {"info": {}}
            video_process = None
            episode_started = False
            try:
                if not SKIP_EXPERT:
                    task_env.setup_demo(
                        now_ep_num=accepted,
                        seed=seed,
                        is_test=True,
                        **task_args,
                    )
                    episode_info = task_env.play_once()
                    plan_success = bool(getattr(task_env, "plan_success", True))
                    expert_check_success = bool(task_env.check_success()) if plan_success else None
                    expert_ok = bool(plan_success and expert_check_success)
                    print(f"EXPERT_SEED={seed} EXPERT_OK={expert_ok}", flush=True)
                    safe_close_task(task_env)
                    if not expert_ok:
                        if controlled_case:
                            send_message(protocol_out, {"type": "case_status", "seed": seed,
                                                        "status": "expert_rejected",
                                                        "plan_success": plan_success,
                                                        "check_success": expert_check_success})
                            return 0
                        seed += 1
                        continue

                task_env.setup_demo(
                    now_ep_num=accepted,
                    seed=seed,
                    is_test=True,
                    **task_args,
                )
                block_pose_x = _reference_pose_x(task_env)
                arm_tag, arm_binding_source = _arm_from_episode_info(episode_info)
                if arm_tag is None:
                    arm_tag = _arm_from_reference_pose(block_pose_x)
                    if arm_tag is not None:
                        arm_binding_source = "reference_actor_pose.x"
                if arm_tag is None:
                    arm_tag = "left"
                    arm_binding_source = "deterministic_left_fallback"
                instruction_type = task_args.get("eval_instruction")
                instruction_args = (
                    task_args,
                    episode_info,
                    instruction_type,
                    OFFICIAL_INSTRUCTION_CANDIDATES,
                )
                if (
                    accepted == 0
                    and args.first_instruction is not None
                    and args.first_instruction_seed == seed
                ):
                    instruction = args.first_instruction
                else:
                    instruction = (
                        build_instruction(*instruction_args, seed)
                        if build_instruction_accepts_seed
                        else build_instruction(*instruction_args)
                    )
                instruction_source = f"official_{instruction_type}"
                task_env.set_instruction(instruction=instruction)
                raw_observation = task_env.get_obs()
                video_process = _start_eval_video(task_env, raw_observation, accepted)
                if video_process is not None:
                    task_env._set_eval_video_ffmpeg(video_process)
                _dump_cam_high_png(raw_observation, accepted, seed, 0)
                print(f"INSTRUCTION={instruction!r}")
                print("EXPERT_PREFILTER=" + ("OFF" if SKIP_EXPERT else "ON"))
                try:
                    _src = instruction_source
                except Exception:
                    _src = "unknown"
                print(f"INSTRUCTION_SOURCE={_src}")
                print(
                    "SEED_ARM "
                    f"seed={seed} reference_pose.x={block_pose_x} "
                    f"arm={arm_tag} arm_binding_source={arm_binding_source}",
                    flush=True,
                )
                print(f"ARM_TAG={arm_tag}")
                print(f"ARM_BINDING_SOURCE={arm_binding_source}")
                send_message(
                    protocol_out,
                    {
                        "type": "episode_start",
                        "episode": accepted,
                        "seed": seed,
                        "instruction": instruction,
                        "instruction_source": _src,
                        "instruction_candidate_count": OFFICIAL_INSTRUCTION_CANDIDATES,
                        "block_pose_x": block_pose_x,
                        "arm_tag": arm_tag,
                        "arm_binding_source": arm_binding_source,
                        "step_limit": int(task_env.step_lim),
                        "observation": simulator_observation(raw_observation),
                    },
                )
                episode_started = True

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
                    raw_after_action = task_env.get_obs()
                    _dump_cam_high_png(raw_after_action, accepted, seed, step)
                    if trajectory_stream is not None:
                        endpose = raw_after_action.get("endpose") or {}
                        joint_action = raw_after_action.get("joint_action") or {}
                        hammer_target_pose = task_env.hammer.get_functional_point(
                            0, "pose"
                        ).p
                        block_pose = task_env.block.get_functional_point(
                            1, "pose"
                        ).p
                        eps = np.array([0.02, 0.02])
                        xy_error = np.abs(
                            np.asarray(hammer_target_pose)[:2]
                            - np.asarray(block_pose)[:2]
                        )
                        actors_contact = bool(
                            task_env.check_actors_contact(
                                task_env.hammer.get_name(),
                                task_env.block.get_name(),
                            )
                        )
                        trajectory_stream.write(
                            json.dumps(
                                {
                                    "step": step,
                                    "endpose_l": {
                                        "pose": endpose.get("left_endpose"),
                                        "gripper": endpose.get("left_gripper"),
                                    },
                                    "endpose_r": {
                                        "pose": endpose.get("right_endpose"),
                                        "gripper": endpose.get("right_gripper"),
                                    },
                                    "qpos": joint_action.get("vector"),
                                    "action": action,
                                    "success_predicate_intermediates": {
                                        "hammer_target_xy": np.asarray(
                                            hammer_target_pose
                                        )[:2],
                                        "block_target_xy": np.asarray(block_pose)[:2],
                                        "xy_abs_error": xy_error,
                                        "xy_eps": eps,
                                        "xy_within_eps": bool(np.all(xy_error < eps)),
                                        "actors_contact": actors_contact,
                                        "is_success": bool(is_success),
                                    },
                                },
                                ensure_ascii=True,
                                default=_jsonable,
                            )
                            + "\n"
                        )
                    terminated = is_success
                    truncated = (
                        task_env.take_action_cnt >= task_env.step_lim
                        or step >= args.episode_steps
                    )
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
                            "observation": simulator_observation(raw_after_action),
                            "terminated": False,
                            "truncated": False,
                            "info": info,
                        },
                    )

                if video_process is not None:
                    final_frame = np.asarray(
                        task_env.get_obs()["observation"]["head_camera"]["rgb"],
                        dtype=np.uint8,
                    )
                    if video_process.stdin is not None:
                        video_process.stdin.write(final_frame.tobytes())
                        video_process.stdin.close()
                    video_process.wait(timeout=30)
                    video_process = None
                safe_close_task(task_env)
                accepted += 1
                seed += 1
                consecutive_pre_action_errors = 0
            except Exception as exc:
                if controlled_case:
                    safe_close_task(task_env)
                    send_message(protocol_out, {"type": "case_status", "seed": seed,
                        "status": "simulator_error" if episode_started else "setup_error",
                        "message": f"{type(exc).__name__}: {exc}"})
                    return 2
                # Official RoboTwin generation can reject an unstable random
                # object before an episode is accepted.  Skip that seed and
                # continue sampling, matching the upstream evaluation loop;
                # this is not a policy failure or a completed trial.
                if type(exc).__name__ == "UnStableError" or "UnStableError" in str(exc):
                    print(f"SKIP_UNSTABLE seed={seed} error={exc}", flush=True)
                    safe_close_task(task_env)
                    seed += 1
                    continue
                # Match the official evaluator's expert/setup handling: a
                # scene-generation or play_once failure before the first policy
                # action is not an accepted trial.  In particular, RoboTwin
                # tasks can raise ordinary TypeError/ValueError for an invalid
                # object pose, not only UnStableError.
                if not episode_started:
                    consecutive_pre_action_errors += 1
                    print(
                        f"SKIP_PRE_ACTION_ERROR seed={seed} "
                        f"consecutive={consecutive_pre_action_errors} "
                        f"error={type(exc).__name__}: {exc}",
                        flush=True,
                    )
                    safe_close_task(task_env)
                    seed += 1
                    if consecutive_pre_action_errors >= 5:
                        send_message(
                            protocol_out,
                            {
                                "type": "error",
                                "message": (
                                    "five consecutive pre-action simulator errors; "
                                    f"last={type(exc).__name__}: {exc}\n"
                                    f"{traceback.format_exc()}"
                                ),
                            },
                        )
                        return 2
                    continue
                if video_process is not None:
                    try:
                        if video_process.stdin is not None:
                            video_process.stdin.close()
                        video_process.wait(timeout=10)
                    except Exception:
                        video_process.kill()
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
        if trajectory_stream is not None:
            trajectory_stream.close()


def main() -> int:
    args = parse_args()
    if args.role == "sim":
        return run_simulator(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
