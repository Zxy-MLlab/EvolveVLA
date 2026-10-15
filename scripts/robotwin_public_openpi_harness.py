#!/usr/bin/env python3
"""RoboTwin runner for the public motus-robotics/pi0.5_robotwin2 checkpoint.

The simulator side is the existing official-protocol bridge.  The parent side
loads OpenPI's public policy and executes each returned action chunk exactly
as the upstream Pi_05 deployment does: actions are sent sequentially, then a
fresh observation is used for the next chunk.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
_OPENPI_SRC = Path(
    os.environ.get(
        "ROBOTWIN_OPENPI_SRC",
        str(
            _HERE.parent
            / "robotwin_repo"
            / "XPolicyLab"
            / "policy"
            / "Pi_05"
            / "openpi"
            / "src"
        ),
    )
).expanduser().resolve()
if str(_OPENPI_SRC) not in sys.path:
    sys.path.insert(0, str(_OPENPI_SRC))

from pi05_openpi_adapter import PublicPi05Adapter, checkpoint_summary
from robotwin_eval_harness import (
    receive_message,
    run_simulator,
    send_message,
)


DEFAULT_CHECKPOINT = str(
    Path(__file__).resolve().parents[1] / "checkpoints" / "pi05_robotwin2_public"
)
DEFAULT_SIM_PYTHON = "/root/autodl-tmp/conda_envs/robotwin_cu128/bin/python"
DEFAULT_ROBOTWIN_ROOT = "/root/autodl-tmp/EvolveVLA/robotwin_repo"
DEFAULT_OIDN_RUNTIME = _HERE.parent / "third_party" / "oidn-2.5.1" / "lib"


def _renderer_metadata() -> dict[str, str]:
    denoiser = os.environ.get("ROBOTWIN_RT_DENOISER", "oidn")
    metadata = {
        "render_backend": os.environ.get("ROBOTWIN_RENDER_BACKEND", "rt"),
        "rt_denoiser": denoiser,
    }
    if denoiser == "oidn":
        metadata["oidn_runtime"] = os.environ.get(
            "ROBOTWIN_OIDN_RUNTIME",
            str(DEFAULT_OIDN_RUNTIME),
        )
    return metadata


def _configure_sim_renderer_env(child_env: dict[str, str]) -> None:
    renderer = _renderer_metadata()
    child_env["ROBOTWIN_RENDER_BACKEND"] = renderer["render_backend"]
    child_env["ROBOTWIN_RT_DENOISER"] = renderer["rt_denoiser"]
    if renderer["rt_denoiser"] != "oidn":
        return
    runtime = Path(renderer["oidn_runtime"])
    frontend = runtime / "libOpenImageDenoise.so.2"
    if not frontend.is_file():
        raise FileNotFoundError(
            f"OIDN runtime is missing {frontend}; set ROBOTWIN_OIDN_RUNTIME"
        )
    previous_library_path = child_env.get("LD_LIBRARY_PATH", "")
    child_env["LD_LIBRARY_PATH"] = (
        str(runtime)
        if not previous_library_path
        else f"{runtime}{os.pathsep}{previous_library_path}"
    )
    previous_preload = child_env.get("LD_PRELOAD", "")
    child_env["LD_PRELOAD"] = (
        str(frontend)
        if not previous_preload
        else f"{frontend} {previous_preload}"
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--role", choices=("parent", "sim"), default="parent")
    p.add_argument("--checkpoint", default=DEFAULT_CHECKPOINT)
    p.add_argument("--task", default="beat_block_hammer")
    p.add_argument("--episodes", type=int, default=1)
    p.add_argument("--episode-steps", type=int, default=2000)
    p.add_argument("--seed", type=int, default=100000)
    p.add_argument("--task-config", default="demo_clean")
    p.add_argument("--prompt-prefix", choices=("none", "scene"), default="none")
    p.add_argument("--normalization", choices=("zscore", "quantile"), default="quantile")
    p.add_argument("--policy-seed", type=int, default=0)
    p.add_argument("--noise-source", choices=("jax", "torch", "internal"), default="torch")
    p.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    p.add_argument("--robotwin-root", default=DEFAULT_ROBOTWIN_ROOT)
    p.add_argument("--video-dir", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--resume", action="store_true", help="continue a partially written output using the next seed")
    p.add_argument("--first-instruction")
    p.add_argument("--first-instruction-seed", type=int)
    return p.parse_args()


def _absolute_optional_path(value: Path | None) -> Path | None:
    return None if value is None else value.expanduser().resolve()


def _to_hwc_uint8(value: object) -> np.ndarray:
    """Convert the simulator bridge's CHW float tensor/array to HWC uint8."""
    x = value.detach().cpu().numpy() if hasattr(value, "detach") else np.asarray(value)
    if x.ndim == 3 and x.shape[0] in (1, 3) and x.shape[-1] not in (1, 3):
        x = np.transpose(x, (1, 2, 0))
    if x.ndim != 3 or x.shape[-1] != 3:
        raise RuntimeError(f"expected image HWC/CHW with 3 channels, got {x.shape}")
    if np.issubdtype(x.dtype, np.floating):
        if float(np.nanmax(x)) <= 1.01:
            x = x * 255.0
        x = np.clip(x, 0, 255)
    return np.asarray(x, dtype=np.uint8)


def _observation_for_openpi(observation: dict[str, object]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    pixels = observation.get("pixels")
    if not isinstance(pixels, dict):
        raise RuntimeError("simulator observation has no pixels mapping")
    images = {
        "cam_high": _to_hwc_uint8(pixels["head_camera"]),
        "cam_left_wrist": _to_hwc_uint8(pixels["left_camera"]),
        "cam_right_wrist": _to_hwc_uint8(pixels["right_camera"]),
    }
    state = np.asarray(observation.get("agent_pos"), dtype=np.float32).reshape(-1)
    if state.shape != (14,):
        raise RuntimeError(f"simulator state must be 14-D, got {state.shape}")
    return images, state


def run_parent(args: argparse.Namespace) -> int:
    if args.episodes <= 0 or args.episode_steps <= 0:
        raise ValueError("episodes and episode-steps must be positive")
    heartbeat = os.environ.get("ROBOTWIN_HEARTBEAT_LOG")
    if not heartbeat:
        heartbeat = str(Path(args.output or "results") .with_suffix(".heartbeat.log"))
    checkpoint = Path(args.checkpoint).resolve()
    args.video_dir = _absolute_optional_path(args.video_dir)
    args.output = _absolute_optional_path(args.output)
    pending_path = (
        args.output.with_suffix(args.output.suffix + ".pending.json")
        if args.output is not None
        else None
    )
    reports: list[dict[str, object]] = []
    if args.resume and args.output and args.output.is_file():
        try:
            previous = json.loads(args.output.read_text(encoding="utf-8"))
            compatible = (
                previous.get("task") == args.task
                and previous.get("task_config") == args.task_config
                and previous.get("checkpoint") == str(checkpoint)
                and previous.get("episode_steps") in (None, args.episode_steps)
                and previous.get("prompt_prefix", "scene") == args.prompt_prefix
                and previous.get("render_backend", "default")
                == _renderer_metadata()["render_backend"]
                and previous.get("rt_denoiser", "none")
                == _renderer_metadata()["rt_denoiser"]
                and previous.get("normalization", "quantile") == args.normalization
                and previous.get("policy_seed") == args.policy_seed
                and previous.get("noise_source") == args.noise_source
            )
            if compatible:
                reports = list(previous.get("reports", []))
                if len(reports) > args.episodes:
                    raise RuntimeError("existing output has more reports than requested episodes")
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot resume output {args.output}: {exc}") from exc
    completed_before = len(reports)
    remaining_episodes = args.episodes - completed_before
    resume_seed = args.seed
    if reports:
        recorded_seeds = [int(report["seed"]) for report in reports]
        if len(recorded_seeds) != len(set(recorded_seeds)):
            raise RuntimeError("existing output contains duplicate accepted seeds")
        resume_seed = max(recorded_seeds) + 1
    pending: dict[str, object] | None = None
    if args.resume and pending_path is not None and pending_path.is_file():
        try:
            candidate = json.loads(pending_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise RuntimeError(f"cannot resume pending episode {pending_path}: {exc}") from exc
        pending_compatible = (
            candidate.get("task") == args.task
            and candidate.get("task_config") == args.task_config
            and candidate.get("checkpoint") == str(checkpoint)
            and candidate.get("episode") == completed_before
            and isinstance(candidate.get("seed"), int)
            and isinstance(candidate.get("instruction"), str)
        )
        if pending_compatible:
            pending = candidate
            resume_seed = int(candidate["seed"])
    if remaining_episodes == 0:
        print(json.dumps({"task": args.task, "task_config": args.task_config,
                          "episodes_requested": args.episodes,
                          "episodes_completed": len(reports),
                          "successes": sum(bool(x["success"]) for x in reports)}, indent=2), flush=True)
        return 0
    print("PUBLIC_CHECKPOINT_SUMMARY=" + json.dumps(checkpoint_summary(checkpoint), sort_keys=True), flush=True)
    child_env = os.environ.copy()
    child_env["CUDA_VISIBLE_DEVICES"] = "0"
    child_env["PYTHONUNBUFFERED"] = "1"
    _configure_sim_renderer_env(child_env)
    command = [
        args.sim_python, str(Path(__file__).resolve()), "--role", "sim",
        "--task", args.task, "--episodes", str(remaining_episodes),
        "--episode-steps", str(args.episode_steps), "--seed", str(resume_seed),
        "--task-config", args.task_config, "--robotwin-root", args.robotwin_root,
    ]
    if pending is not None:
        command += [
            "--first-instruction", str(pending["instruction"]),
            "--first-instruction-seed", str(pending["seed"]),
        ]
    if args.video_dir is not None:
        command += ["--video-dir", str(args.video_dir)]
    child = subprocess.Popen(
        command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=None,
        bufsize=0, env=child_env, cwd=args.robotwin_root, preexec_fn=os.setpgrp,
    )
    assert child.stdin is not None and child.stdout is not None
    try:
        # Start the simulator before initializing JAX. Forking after JAX has
        # created worker threads is unsupported and can deadlock.
        policy = PublicPi05Adapter(
            checkpoint,
            device="cuda:0",
            prompt_prefix=args.prompt_prefix,
            normalization=args.normalization,
            policy_seed=args.policy_seed,
            noise_source=args.noise_source,
        )
        completed_inferences = sum(
            (int(report["steps"]) + policy.action_horizon - 1)
            // policy.action_horizon
            for report in reports
        )
        policy.advance_rng(completed_inferences)
        if completed_inferences:
            print(
                f"RESUME_RNG_ADVANCE inferences={completed_inferences} "
                f"episodes={completed_before}",
                flush=True,
            )
        while True:
            message = receive_message(child.stdout)
            kind = message.get("type")
            if kind == "complete":
                break
            if kind == "error":
                raise RuntimeError(message.get("message"))
            if kind != "episode_start":
                raise RuntimeError(f"unexpected simulator message: {kind}")
            episode = completed_before + int(message["episode"])
            seed = int(message["seed"])
            step_limit = int(message["step_limit"])
            instruction = str(message["instruction"])
            instruction_candidate_count = int(message["instruction_candidate_count"])
            observation = message["observation"]
            if pending_path is not None:
                pending_record = {
                    "task": args.task,
                    "task_config": args.task_config,
                    "checkpoint": str(checkpoint),
                    "episode": episode,
                    "seed": seed,
                    "instruction": instruction,
                }
                pending_path.parent.mkdir(parents=True, exist_ok=True)
                pending_path.write_text(
                    json.dumps(pending_record, indent=2) + "\n",
                    encoding="utf-8",
                )
            done = False
            success = False
            steps = 0
            chunk_index = 0
            min_action = None
            max_action = None
            while not done:
                images, state = _observation_for_openpi(observation)
                chunk = policy.infer_chunk(images=images, state=state, prompt=instruction)
                image_shapes = {key: list(value.shape) for key, value in images.items()}
                print(
                    f"EPISODE={episode} STEP={steps} PROMPT={instruction!r} "
                    f"IMAGE_SHAPES={image_shapes} "
                    f"STATE_SHAPE={state.shape} RAW_CHUNK_SHAPE={chunk.shape} "
                    f"RAW_CHUNK_FIRST={chunk[0].tolist()}", flush=True,
                )
                for action_index, action in enumerate(chunk):
                    if steps >= args.episode_steps:
                        done = True
                        break
                    action = np.asarray(action, dtype=np.float32).reshape(14)
                    min_action = float(np.min(action)) if min_action is None else min(min_action, float(np.min(action)))
                    max_action = float(np.max(action)) if max_action is None else max(max_action, float(np.max(action)))
                    if not np.isfinite(action).all():
                        raise RuntimeError("public policy returned a non-finite action")
                    send_message(child.stdin, {"cmd": "action", "action": action.tolist()})
                    response = receive_message(child.stdout)
                    if response.get("type") == "error":
                        raise RuntimeError(response.get("message"))
                    if response.get("type") != "step":
                        raise RuntimeError(f"unexpected simulator response: {response.get('type')}")
                    steps += 1
                    success = success or bool(response.get("info", {}).get("is_success", False))
                    done = bool(response.get("terminated") or response.get("truncated"))
                    print(
                        f"ACTION_SENT episode={episode} step={steps} chunk={chunk_index} "
                        f"chunk_index={action_index} action={action.tolist()} success={success} done={done}",
                        flush=True,
                    )
                    if done:
                        break
                    observation = response["observation"]
                chunk_index += 1
            reports.append({
                "episode": episode,
                "seed": seed,
                "success": success,
                "steps": steps,
                "step_limit": step_limit,
                "action_min": min_action,
                "action_max": max_action,
            })
            print(f"EPISODE_DONE episode={episode} success={success} steps={steps}", flush=True)
            # Make completed episodes durable so a preemption only repeats the
            # current episode on the next --resume invocation.
            if args.output:
                partial = {
                    "task": args.task, "task_config": args.task_config, "checkpoint": str(checkpoint),
                    "episodes_requested": args.episodes, "episodes_completed": len(reports),
                    "successes": sum(bool(x["success"]) for x in reports), "reports": reports,
                    "action_horizon": policy.action_horizon, "action_dim": 14,
                    "episode_steps": args.episode_steps,
                    "prompt_prefix": args.prompt_prefix,
                    "normalization": args.normalization,
                    "policy_seed": args.policy_seed,
                    "noise_source": args.noise_source,
                    **_renderer_metadata(),
                    "instruction_candidate_count": instruction_candidate_count,
                    "checkpoint_format": policy.checkpoint_format,
                    "protocol": (
                        "official RoboTwin setup_demo/play_once/check_success; "
                        f"OpenPI {policy.action_horizon}-step chunks"
                    ),
                }
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(json.dumps(partial, indent=2) + "\n", encoding="utf-8")
                if pending_path is not None:
                    pending_path.unlink(missing_ok=True)
    finally:
        try:
            send_message(child.stdin, {"cmd": "stop"})
            child.stdin.close()
        except Exception:
            pass
        try:
            child.wait(timeout=60)
        except subprocess.TimeoutExpired:
            os.killpg(os.getpgid(child.pid), signal.SIGKILL)
            child.wait(timeout=10)
    result = {
        "task": args.task, "task_config": args.task_config, "checkpoint": str(checkpoint),
        "episodes_requested": args.episodes, "episodes_completed": len(reports),
        "successes": sum(bool(x["success"]) for x in reports),
        "reports": reports, "action_horizon": policy.action_horizon, "action_dim": 14,
        "episode_steps": args.episode_steps,
        "prompt_prefix": args.prompt_prefix,
        "normalization": args.normalization,
        "policy_seed": args.policy_seed,
        "noise_source": args.noise_source,
        **_renderer_metadata(),
        "instruction_candidate_count": 100,
        "checkpoint_format": policy.checkpoint_format,
        "protocol": (
            "official RoboTwin setup_demo/play_once/check_success; "
            f"OpenPI {policy.action_horizon}-step chunks"
        ),
    }
    text = json.dumps(result, indent=2) + "\n"
    print(text, flush=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text, encoding="utf-8")
    return 0


def main() -> int:
    args = parse_args()
    if args.role == "sim":
        from robotwin_eval_harness import run_simulator

        return run_simulator(args)
    return run_parent(args)


if __name__ == "__main__":
    raise SystemExit(main())
