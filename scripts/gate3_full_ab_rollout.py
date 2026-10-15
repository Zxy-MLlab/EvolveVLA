#!/usr/bin/env python3
"""Run the protocol-comparable RoboTwin A/B and extract rollout forensics.

The pinned harness already emits one JSONL record per executed simulator step
when ROBOTWIN_TRAJECTORY_JSONL is set. This runner only orchestrates the
existing harness and parses those records; it does not modify harness/OpenPI
source.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time
from typing import Any, Iterable


PROJECT_ROOT = Path("/root/autodl-tmp/EvolveVLA")
HARNESS = PROJECT_ROOT / "scripts/robotwin_eval_harness_dispatch.py"
POLICY_PYTHON = Path("/root/autodl-tmp/conda_envs/evolvevla_pi312/bin/python")
SIM_PYTHON = Path("/root/autodl-tmp/conda_envs/robotwin_cu128/bin/python")
ROBOTWIN_ROOT = PROJECT_ROOT / "robotwin_repo"
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
RUN_ROOT = Path("/root/runs/zhouxueyangapexinai")
RESULTS_DIR = PROJECT_ROOT / "results"

MODEL_PATHS = {
    "official": PROJECT_ROOT / "checkpoints/pi05_robotwin_lerobot",
    "sidneyxie": PROJECT_ROOT / "checkpoints/sidneyxie_pi05_robotwin",
}

DEFAULT_SEEDS = (42, 43, 44)
EPISODES = 5
EPISODE_STEPS = 400
TASK = "beat_block_hammer"
TASK_CONFIG = "demo_clean"
GRIPPER_CLOSE_THRESHOLD = 0.2
WATCHDOG_SECONDS = 120
DEFAULT_OBS_HEIGHT = 240
DEFAULT_OBS_WIDTH = 320
DEFAULT_INSTRUCTION_SOURCE = "ours"
DEFAULT_CAM_PROFILE = "D435"
CAMERA_PROFILES = {
    "D435": {"h": 240, "w": 320},
    "Large_D435": {"h": 480, "w": 640},
}
ARM_INDICES = tuple(range(6)) + tuple(range(7, 13))
GRIPPER_INDICES = (6, 13)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run full-protocol official-vs-SidneyXie RoboTwin A/B."
    )
    parser.add_argument(
        "--model",
        choices=tuple(MODEL_PATHS),
        required=True,
        help="checkpoint arm to evaluate",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="base seeds; each seed runs the full 5-episode protocol",
    )
    parser.add_argument(
        "--watchdog-seconds",
        type=int,
        default=WATCHDOG_SECONDS,
        help="fail a live child when its log mtime is older than this many seconds",
    )
    parser.add_argument(
        "--obs-height",
        type=int,
        default=None,
        help="override observation height; defaults to the selected camera profile",
    )
    parser.add_argument(
        "--obs-width",
        type=int,
        default=None,
        help="override observation width; defaults to the selected camera profile",
    )
    parser.add_argument(
        "--cam-profile",
        default=os.environ.get("ROBOTWIN_CAM_PROFILE", DEFAULT_CAM_PROFILE),
        help="native RoboTwin camera profile passed as ROBOTWIN_CAM_PROFILE",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="print resolved env and harness command without starting a rollout",
    )
    camera_group = parser.add_mutually_exclusive_group()
    camera_group.add_argument(
        "--cam-dump-dir",
        type=Path,
        help="explicitly enable camera dumps in this directory",
    )
    camera_group.add_argument(
        "--no-cam-dump",
        action="store_true",
        help="force ROBOTWIN_CAM_DUMP_DIR to be unset",
    )
    parser.add_argument(
        "--instruction-source",
        choices=("ours", "official"),
        default=DEFAULT_INSTRUCTION_SOURCE,
        help="instruction selector used by the evaluation harness",
    )
    return parser.parse_args()


def require_path(path: Path, label: str) -> None:
    if not path.exists():
        raise RuntimeError(f"AB_PRECONDITION_NOT_MET missing {label}: {path}")


def resolve_camera_profile(name: str) -> dict[str, int]:
    profile = str(name).strip()
    try:
        config = CAMERA_PROFILES[profile]
    except KeyError as exc:
        supported = ", ".join(CAMERA_PROFILES)
        raise ValueError(
            f"unsupported camera profile {profile!r}; expected one of {supported}"
        ) from exc
    return dict(config)


def resolve_observation_shape(
    args: argparse.Namespace,
    camera_config: dict[str, int],
) -> tuple[int, int]:
    obs_height = (
        int(args.obs_height)
        if args.obs_height is not None
        else int(camera_config["h"])
    )
    obs_width = (
        int(args.obs_width)
        if args.obs_width is not None
        else int(camera_config["w"])
    )
    return obs_height, obs_width


def resolved_cam_dump_dir(args: argparse.Namespace) -> str | None:
    if args.no_cam_dump:
        return None
    if args.cam_dump_dir is not None:
        return str(args.cam_dump_dir.resolve())
    return os.environ.get("ROBOTWIN_CAM_DUMP_DIR")


def run_paths(
    model: str,
    seed: int,
    instruction_source: str = DEFAULT_INSTRUCTION_SOURCE,
) -> tuple[Path, Path, Path, Path]:
    stem = f"gate3_full_ab_{model}_{instruction_source}_s{seed}"
    return (
        RUN_ROOT / f"{stem}.log",
        RUN_ROOT / f"{stem}.jsonl",
        RESULTS_DIR / f"{stem}.harness.json",
        RESULTS_DIR / f"{stem}.json",
    )


def harness_command(model: str, seed: int, output_path: Path) -> list[str]:
    checkpoint = MODEL_PATHS[model]
    return [
        str(POLICY_PYTHON),
        str(HARNESS),
        "--role",
        "parent",
        "--task",
        TASK,
        "--episodes",
        str(EPISODES),
        "--episode-steps",
        str(EPISODE_STEPS),
        "--seed",
        str(seed),
        "--model-path",
        str(checkpoint),
        "--tokenizer-path",
        str(TOKENIZER_PATH),
        "--sim-python",
        str(SIM_PYTHON),
        "--robotwin-root",
        str(ROBOTWIN_ROOT),
        "--task-config",
        TASK_CONFIG,
        "--output",
        str(output_path),
    ]


def dry_run(
    args: argparse.Namespace,
    cam_profile: str,
    obs_height: int,
    obs_width: int,
    cam_dump_dir: str | None,
) -> int:
    cam_state = cam_dump_dir if cam_dump_dir is not None else "UNSET"
    for seed in args.seeds:
        log_path, trajectory_path, output_path, _ = run_paths(
            args.model,
            seed,
            args.instruction_source,
        )
        command = harness_command(args.model, seed, output_path)
        resolved_env = [
            ("CUDA_VISIBLE_DEVICES", "0"),
            ("PYTHONUNBUFFERED", "1"),
            ("ROBOTWIN_SKIP_EXPERT", "0"),
            ("ROBOTWIN_INSTRUCTION_SOURCE", args.instruction_source),
            ("ROBOTWIN_CAM_PROFILE", cam_profile),
            ("VK_ICD_FILENAMES", "/etc/vulkan/icd.d/nvidia_icd.json"),
            ("ROBOTWIN_HEARTBEAT_LOG", str(log_path)),
            ("ROBOTWIN_TRAJECTORY_JSONL", str(trajectory_path)),
            ("ROBOTWIN_OBS_HEIGHT", str(obs_height)),
            ("ROBOTWIN_OBS_WIDTH", str(obs_width)),
        ]
        if cam_dump_dir is not None:
            resolved_env.append(("ROBOTWIN_CAM_DUMP_DIR", cam_dump_dir))
        env_text = " ".join(
            f"{key}={shlex.quote(value)}" for key, value in resolved_env
        )
        print(
            f"AB_DRY_RUN model={args.model} seed={seed} "
            f"cam_profile={cam_profile} native={obs_height}x{obs_width} "
            f"obs={obs_height}x{obs_width} "
            f"instruction_source={args.instruction_source} cam_dump={cam_state}"
        )
        print(f"AB_DRY_RUN_ENV {env_text}")
        print(f"AB_DRY_RUN_COMMAND {shlex.join(command)}")
    return 0


def l2(values: Iterable[Any], indices: Iterable[int]) -> float:
    values_list = list(values)
    return math.sqrt(
        sum(float(values_list[index]) ** 2 for index in indices)
    )


def _float_xy(value: Any) -> tuple[float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    try:
        return float(value[0]), float(value[1])
    except (TypeError, ValueError):
        return None


def _first_close(action: list[Any]) -> tuple[int, float] | None:
    crossings = [
        (index, float(action[index]))
        for index in GRIPPER_INDICES
        if index < len(action) and float(action[index]) < GRIPPER_CLOSE_THRESHOLD
    ]
    return min(crossings, key=lambda item: item[0]) if crossings else None


def _episode_key(record: dict[str, Any], fallback_episode: int) -> Any:
    if "episode" in record:
        return record["episode"]
    if "seed" in record:
        return record["seed"]
    return fallback_episode


def parse_trajectory(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise RuntimeError(f"AB_TRAJECTORY_MISSING path={path}")

    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as stream:
        for line_number, raw_line in enumerate(stream, start=1):
            if not raw_line.strip():
                continue
            try:
                record = json.loads(raw_line)
            except json.JSONDecodeError as exc:
                raise RuntimeError(
                    f"AB_TRAJECTORY_INVALID_JSON line={line_number}: {exc}"
                ) from exc
            if not isinstance(record, dict):
                raise RuntimeError(
                    f"AB_TRAJECTORY_INVALID_RECORD line={line_number}"
                )
            records.append(record)

    if not records:
        raise RuntimeError(f"AB_TRAJECTORY_EMPTY path={path}")

    step_metrics: list[dict[str, Any]] = []
    episodes: dict[str, dict[str, Any]] = {}
    previous_qpos_by_episode: dict[str, list[Any]] = {}
    fallback_episode = 0
    previous_step_by_episode: dict[str, int] = {}

    for global_index, record in enumerate(records, start=1):
        action = list(record.get("action", []))
        qpos = list(record.get("qpos", []))
        if len(action) != 14 or len(qpos) != 14:
            raise RuntimeError(
                "AB_TRAJECTORY_BAD_DIM "
                f"global_step={global_index} action={len(action)} qpos={len(qpos)}"
            )

        explicit_episode = _episode_key(record, fallback_episode)
        episode_key = str(explicit_episode)
        local_step = int(record.get("step", global_index))
        previous_step = previous_step_by_episode.get(episode_key)
        if previous_step is not None and local_step <= previous_step:
            fallback_episode += 1
            episode_key = str(fallback_episode)
        previous_step_by_episode[episode_key] = local_step

        previous_qpos = previous_qpos_by_episode.get(episode_key)
        arm_pos_delta = (
            0.0
            if previous_qpos is None
            else math.sqrt(
                sum(
                    (float(qpos[index]) - float(previous_qpos[index])) ** 2
                    for index in ARM_INDICES
                )
            )
        )
        previous_qpos_by_episode[episode_key] = qpos

        predicates = record.get("success_predicate_intermediates") or {}
        if not isinstance(predicates, dict):
            predicates = {}
        xy_within_eps = bool(predicates.get("xy_within_eps", False))
        xy_eps = predicates.get("xy_eps", [0.02, 0.02])
        grasp_eps = float(xy_eps[0]) if isinstance(xy_eps, list) else 0.02
        hammer_xy = _float_xy(predicates.get("hammer_target_xy"))
        block_xy = _float_xy(predicates.get("block_target_xy"))

        ee_distances: dict[str, dict[str, float | None]] = {}
        for side in ("l", "r"):
            endpose = record.get(f"endpose_{side}") or {}
            pose = endpose.get("pose") if isinstance(endpose, dict) else None
            ee_xy = _float_xy(pose)
            ee_distances[side] = {
                "ee_xy_to_hammer": (
                    math.dist(ee_xy, hammer_xy)
                    if ee_xy is not None and hammer_xy is not None
                    else None
                ),
                "ee_xy_to_block": (
                    math.dist(ee_xy, block_xy)
                    if ee_xy is not None and block_xy is not None
                    else None
                ),
            }
        hammer_distances = [
            value["ee_xy_to_hammer"]
            for value in ee_distances.values()
            if value["ee_xy_to_hammer"] is not None
        ]
        block_distances = [
            value["ee_xy_to_block"]
            for value in ee_distances.values()
            if value["ee_xy_to_block"] is not None
        ]
        ee_to_hammer = min(hammer_distances) if hammer_distances else None
        ee_to_block = min(block_distances) if block_distances else None
        close = _first_close(action)
        success = bool(predicates.get("is_success", False))

        metric = {
            "global_step": global_index,
            "episode": record.get("episode", episode_key),
            "seed": record.get("seed"),
            "step": local_step,
            "RAW_ACTION_ARM_L2": l2(action, ARM_INDICES),
            "ARM_POS_DELTA": arm_pos_delta,
            "action": action,
            "qpos": qpos,
            "gripper_close_command": (
                {"dim": close[0], "value": close[1]} if close else None
            ),
            "xy_within_eps": xy_within_eps,
            "is_success": success,
            "ee_xy_within_grasp_eps": (
                (ee_to_block if ee_to_block is not None else ee_to_hammer)
                is not None
                and (ee_to_block if ee_to_block is not None else ee_to_hammer)
                <= grasp_eps
            ),
            "ee_xy_to_hammer": ee_to_hammer,
            "ee_xy_to_block": ee_to_block,
            "ee_xy_to_hammer_left": ee_distances["l"]["ee_xy_to_hammer"],
            "ee_xy_to_hammer_right": ee_distances["r"]["ee_xy_to_hammer"],
            "ee_xy_to_block_left": ee_distances["l"]["ee_xy_to_block"],
            "ee_xy_to_block_right": ee_distances["r"]["ee_xy_to_block"],
        }
        step_metrics.append(metric)

        episode = episodes.setdefault(
            episode_key,
            {
                "episode": record.get("episode", episode_key),
                "seed": record.get("seed"),
                "steps": 0,
                "success": False,
                "first_gripper_close_command_step": None,
                "first_gripper_close_command_dim": None,
                "xy_within_eps_any": False,
                "ee_xy_within_grasp_eps_any": False,
                "min_ee_xy_to_hammer": None,
                "min_ee_xy_to_block": None,
                "step_metrics": [],
            },
        )
        episode["steps"] += 1
        episode["success"] = bool(episode["success"] or success)
        if close and episode["first_gripper_close_command_step"] is None:
            episode["first_gripper_close_command_step"] = local_step
            episode["first_gripper_close_command_dim"] = close[0]
        episode["xy_within_eps_any"] = bool(
            episode["xy_within_eps_any"] or xy_within_eps
        )
        ee_within = bool(metric["ee_xy_within_grasp_eps"])
        episode["ee_xy_within_grasp_eps_any"] = bool(
            episode["ee_xy_within_grasp_eps_any"] or ee_within
        )
        for field, value in (
            ("min_ee_xy_to_hammer", ee_to_hammer),
            ("min_ee_xy_to_block", ee_to_block),
        ):
            if value is not None and (
                episode[field] is None or value < episode[field]
            ):
                episode[field] = value
        episode["step_metrics"].append(metric)

    episode_list = list(episodes.values())
    return {
        "trajectory_records": len(records),
        "episodes": episode_list,
        "step_metrics": step_metrics,
        "successes_from_trajectory": sum(
            int(bool(episode["success"])) for episode in episode_list
        ),
        "episodes_from_trajectory": len(episode_list),
    }


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _watchdog_kill(process: subprocess.Popen[Any], reason: str) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
    print(reason, flush=True)


def run_one(
    model: str,
    seed: int,
    watchdog_seconds: int,
    obs_height: int,
    obs_width: int,
    cam_dump_dir: str | None,
    instruction_source: str,
    cam_profile: str,
) -> dict[str, Any]:
    checkpoint = MODEL_PATHS[model]
    log_path, trajectory_path, output_path, result_path = run_paths(
        model,
        seed,
        instruction_source,
    )

    for path in (log_path, trajectory_path, output_path, result_path):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    log_path.parent.mkdir(parents=True, exist_ok=True)
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    if cam_dump_dir is not None:
        Path(cam_dump_dir).mkdir(parents=True, exist_ok=True)

    command = harness_command(model, seed, output_path)
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "ROBOTWIN_SKIP_EXPERT": "0",
            "ROBOTWIN_INSTRUCTION_SOURCE": instruction_source,
            "ROBOTWIN_CAM_PROFILE": cam_profile,
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "ROBOTWIN_HEARTBEAT_LOG": str(log_path),
            "ROBOTWIN_TRAJECTORY_JSONL": str(trajectory_path),
            "ROBOTWIN_OBS_HEIGHT": str(obs_height),
            "ROBOTWIN_OBS_WIDTH": str(obs_width),
        }
    )
    if cam_dump_dir is None:
        env.pop("ROBOTWIN_CAM_DUMP_DIR", None)
    else:
        env["ROBOTWIN_CAM_DUMP_DIR"] = cam_dump_dir

    with log_path.open("w", encoding="utf-8") as log_stream:
        log_stream.write(
            "AB_RUN_START "
            f"model={model} seed={seed} watchdog_seconds={watchdog_seconds}\n"
        )
        log_stream.flush()
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        watchdog_reason: str | None = None
        while process.poll() is None:
            time.sleep(10)
            try:
                age = time.time() - log_path.stat().st_mtime
            except FileNotFoundError:
                age = float("inf")
            if age > watchdog_seconds:
                watchdog_reason = (
                    "AB_WATCHDOG_FAIL "
                    f"model={model} seed={seed} log_age_seconds={age:.1f}"
                )
                _watchdog_kill(process, watchdog_reason)
                break
        return_code = process.wait()

    payload: dict[str, Any] = {
        "model": model,
        "seed": seed,
        "checkpoint": str(checkpoint),
        "task": TASK,
        "task_config": TASK_CONFIG,
        "episodes_requested": EPISODES,
        "episode_steps": EPISODE_STEPS,
        "expert_prefilter": True,
        "arm_binding": "pinned harness path",
        "camera_profile": f"{TASK_CONFIG} ({cam_profile})",
        "shader": "default raster",
        "cuda_visible_devices": "0",
        "harness_command": command,
        "harness_returncode": return_code,
        "log_path": str(log_path),
        "trajectory_path": str(trajectory_path),
        "harness_output_path": str(output_path),
        "watchdog_seconds": watchdog_seconds,
        "watchdog_failure": watchdog_reason,
        "observation_height": obs_height,
        "observation_width": obs_width,
        "camera_dump_dir": cam_dump_dir,
        "instruction_source": instruction_source,
    }
    if output_path.exists():
        try:
            payload["harness_result"] = json.loads(
                output_path.read_text(encoding="utf-8")
            )
        except json.JSONDecodeError as exc:
            payload["harness_result_parse_error"] = str(exc)

    if trajectory_path.exists():
        try:
            parsed = parse_trajectory(trajectory_path)
            payload.update(parsed)
            payload["successes"] = parsed["successes_from_trajectory"]
            payload["episodes"] = parsed["episodes_from_trajectory"]
            payload["success_rate"] = (
                payload["successes"] / payload["episodes"]
                if payload["episodes"]
                else 0.0
            )
        except Exception as exc:
            payload["trajectory_parse_error"] = f"{type(exc).__name__}: {exc}"
    else:
        payload["trajectory_records"] = 0
        payload["episodes"] = 0
        payload["successes"] = 0
        payload["success_rate"] = 0.0

    _write_json(result_path, payload)
    print(
        f"AB_MODEL={model} SEED={seed} "
        f"SUCCESS={payload['successes']}/{payload['episodes']}",
        flush=True,
    )
    if return_code != 0:
        print(
            f"AB_SEED_FAIL model={model} seed={seed} "
            f"returncode={return_code} log={log_path}",
            flush=True,
        )
    return payload


def main() -> int:
    args = parse_args()
    camera_config = resolve_camera_profile(args.cam_profile)
    obs_height, obs_width = resolve_observation_shape(args, camera_config)
    cam_dump_dir = resolved_cam_dump_dir(args)
    if args.dry_run:
        return dry_run(
            args,
            args.cam_profile,
            obs_height,
            obs_width,
            cam_dump_dir,
        )
    checkpoint = MODEL_PATHS[args.model]
    for path, label in (
        (HARNESS, "harness"),
        (POLICY_PYTHON, "policy Python"),
        (SIM_PYTHON, "simulator Python"),
        (ROBOTWIN_ROOT, "RoboTwin root"),
        (TOKENIZER_PATH, "official tokenizer"),
        (checkpoint, f"{args.model} checkpoint"),
    ):
        require_path(path, label)

    summaries: list[dict[str, Any]] = []
    for seed in args.seeds:
        summaries.append(
            run_one(
                args.model,
                seed,
                args.watchdog_seconds,
                obs_height,
                obs_width,
                cam_dump_dir,
                args.instruction_source,
                args.cam_profile,
            )
        )

    counts = [
        f"{summary['successes']}/{summary['episodes']}" for summary in summaries
    ]
    print(
        f"AB_DONE official={','.join(counts) if args.model == 'official' else 'not-run'} "
        f"sidneyxie={','.join(counts) if args.model == 'sidneyxie' else 'not-run'}",
        flush=True,
    )
    return 0 if all(
        summary["harness_returncode"] == 0 for summary in summaries
    ) else 1


if __name__ == "__main__":
    raise SystemExit(main())
