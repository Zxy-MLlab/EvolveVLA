#!/usr/bin/env python3
"""D127 paired RoboTwin instruction-selection test.

This file is intentionally additive.  It reuses the pinned evaluation
harness for policy loading, observation packing, camera mapping, checker,
resolution, and rollout IPC.  The simulator role only changes which
instruction selector is exposed to that harness:

* ours: the current ``seen_instruction`` helper;
* official: force the harness to its existing ``build_instruction`` path,
  whose selector is exactly ``random.Random(f"{seed}:select").choice(...)``.

The paired role runs the same seed list serially for both instruction modes
and verifies that the completed episode ``(episode, seed)`` lists match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time
from typing import Any


SCRIPT_PATH = Path(__file__).resolve()
PROJECT_ROOT = SCRIPT_PATH.parents[1]
HARNESS_PATH = PROJECT_ROOT / "scripts" / "robotwin_eval_harness.py"
EXPECTED_HARNESS_SHA256 = (
    "ab2846cf5cd5fd99ae1749227c6d2f1b6c1747d092a5ee7c4d7549d96eb5e4f4"
)

DEFAULT_MODEL_PATH = (
    "/root/autodl-tmp/EvolveVLA/checkpoints/pi05_robotwin_lerobot"
)
DEFAULT_SIM_PYTHON = "/root/autodl-tmp/conda_envs/robotwin_cu128/bin/python"
DEFAULT_TOKENIZER_PATH = "/root/autodl-tmp/paligemma_tokenizer"
DEFAULT_ROBOTWIN_ROOT = "/root/autodl-tmp/EvolveVLA/robotwin_repo"
DEFAULT_POLICY_PYTHON = "/root/autodl-tmp/conda_envs/evolvevla_pi312/bin/python"
DEFAULT_RUN_ROOT = "/root/runs/zhouxueyangapexinai"
DEFAULT_TASK = "beat_block_hammer"
DEFAULT_TASK_CONFIG = "demo_clean"
DEFAULT_EPISODES = 5
DEFAULT_EPISODE_STEPS = 400
DEFAULT_SEEDS = (42, 43, 44)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "D127 paired official-vs-current RoboTwin instruction selection "
            "test; use --dry-run-instructions for a CPU-only build check."
        )
    )
    parser.add_argument("--role", choices=("parent", "sim"), default="parent")
    parser.add_argument(
        "--instruction-mode",
        choices=("official", "ours"),
        default=os.environ.get("ROBOTWIN_INSTRUCTION_MODE", "ours"),
        help="instruction selector for one arm; paired mode runs both arms",
    )
    parser.add_argument(
        "--paired",
        action="store_true",
        help="run the same seed list serially in official and ours modes",
    )
    parser.add_argument(
        "--dry-run-instructions",
        action="store_true",
        help="CPU-only instruction construction; no simulator or policy step",
    )
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--task", default=DEFAULT_TASK)
    parser.add_argument("--episodes", type=int, default=DEFAULT_EPISODES)
    parser.add_argument("--episode-steps", type=int, default=DEFAULT_EPISODE_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEEDS[0])
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="base seeds for --paired or --dry-run-instructions",
    )
    parser.add_argument("--sim-python", default=DEFAULT_SIM_PYTHON)
    parser.add_argument("--robotwin-root", default=DEFAULT_ROBOTWIN_ROOT)
    parser.add_argument("--tokenizer-path", default=DEFAULT_TOKENIZER_PATH)
    parser.add_argument("--task-config", default=DEFAULT_TASK_CONFIG)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--arm-tag", choices=("left", "right"), default="left")
    parser.add_argument("--policy-python", default=DEFAULT_POLICY_PYTHON)
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument(
        "--watchdog-seconds",
        type=int,
        default=120,
        help="passed through to the paired process watchdog policy",
    )
    return parser.parse_args()


def verify_harness() -> str:
    digest = hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest()
    print(f"HARNESS_SHA256={digest}", flush=True)
    if digest != EXPECTED_HARNESS_SHA256:
        raise RuntimeError(
            "pinned harness sha256 mismatch: "
            f"expected={EXPECTED_HARNESS_SHA256} got={digest}"
        )
    print("HARNESS_SHA256_VERIFIED=True", flush=True)
    return digest


def import_harness():
    scripts_dir = str(HARNESS_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import robotwin_eval_harness as harness

    return harness


def import_instruction_source():
    scripts_dir = str(HARNESS_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    import robotwin_instruction_source as source

    return source


def _instruction_candidates(
    *,
    task: str,
    seed: int,
    episodes: int,
    robotwin_root: str,
    arm_tag: str,
) -> list[str]:
    root = str(Path(robotwin_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)
    from description.utils.generate_episode_instructions import (
        generate_episode_descriptions,
    )

    episode_params = {
        "{A}": "020_hammer/base0",
        "{a}": arm_tag,
    }
    generated = generate_episode_descriptions(
        task,
        [episode_params],
        max_descriptions=episodes,
        seed=seed,
    )
    if not generated or not generated[0].get("seen"):
        raise RuntimeError(
            f"no seen instruction candidates for task={task!r} seed={seed}"
        )
    return [str(item) for item in generated[0]["seen"]]


def instruction_for_mode(
    *,
    mode: str,
    task: str,
    seed: int,
    episodes: int,
    robotwin_root: str,
    arm_tag: str,
) -> str:
    if mode == "ours":
        source_module = import_instruction_source()
        instruction, source = source_module.seen_instruction(
            task,
            arm_tag=arm_tag,
            seed=seed,
        )
        if source == "task_name_fallback":
            raise RuntimeError(
                f"ours selector fell back for task={task!r} seed={seed}"
            )
        return instruction

    candidates = _instruction_candidates(
        task=task,
        seed=seed,
        episodes=episodes,
        robotwin_root=robotwin_root,
        arm_tag=arm_tag,
    )
    return random.Random(f"{seed}:select").choice(candidates)


def run_cpu_instruction_dry_run(args: argparse.Namespace) -> int:
    if args.episodes <= 0:
        raise ValueError("--episodes must be positive")
    print("DRY_RUN_NO_SIM_STEP=True")
    print("DRY_RUN_NO_POLICY_LOAD=True")
    print(f"TASK={args.task}")
    print(f"EPISODES_PARAMETER={args.episodes}")
    for seed in args.seeds:
        official = instruction_for_mode(
            mode="official",
            task=args.task,
            seed=seed,
            episodes=args.episodes,
            robotwin_root=args.robotwin_root,
            arm_tag=args.arm_tag,
        )
        ours = instruction_for_mode(
            mode="ours",
            task=args.task,
            seed=seed,
            episodes=args.episodes,
            robotwin_root=args.robotwin_root,
            arm_tag=args.arm_tag,
        )
        print(f"SEED={seed} MODE=official INSTRUCTION={official!r}")
        print(f"SEED={seed} MODE=ours INSTRUCTION={ours!r}")
    return 0


def run_one_arm(args: argparse.Namespace) -> int:
    verify_harness()
    harness = import_harness()
    # The pinned harness spawns ``Path(__file__)`` for its simulator child.
    # Point that lookup at this additive file and pass the selector through the
    # inherited environment; no shared harness file is edited.
    harness.__file__ = str(SCRIPT_PATH)
    os.environ["ROBOTWIN_INSTRUCTION_MODE"] = args.instruction_mode
    return int(harness.run_parent(args))


def _episode_keys(payload: dict[str, Any]) -> list[tuple[int, int]]:
    reports = payload.get("reports")
    if not isinstance(reports, list):
        raise RuntimeError("harness output has no reports list")
    keys: list[tuple[int, int]] = []
    for item in reports:
        if not isinstance(item, dict):
            raise RuntimeError("harness report entry is not an object")
        keys.append((int(item["episode"]), int(item["seed"])))
    return keys


def _run_paths(args: argparse.Namespace, mode: str, seed: int) -> dict[str, Path]:
    root = Path(args.run_root).resolve()
    stem = f"gate3_instruction_paired_test_{mode}_s{seed}"
    result_dir = PROJECT_ROOT / "results"
    return {
        "log": root / f"{stem}.log",
        "camera": root / f"{stem}_cam",
        "trajectory": root / f"{stem}.jsonl",
        "output": result_dir / f"{stem}.harness.json",
        "summary": result_dir / f"{stem}.json",
    }


def _cpu_ticks(pid: int) -> int | None:
    try:
        fields = Path(f"/proc/{pid}/stat").read_text(encoding="utf-8")
        fields = fields[fields.rfind(")") + 2 :].split()
        return int(fields[11]) + int(fields[12])
    except Exception:
        return None


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    deadline = time.monotonic() + 30.0
    while process.poll() is None and time.monotonic() < deadline:
        time.sleep(0.25)
    if process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


def _run_arm_subprocess(args: argparse.Namespace, mode: str, seed: int) -> dict[str, Any]:
    paths = _run_paths(args, mode, seed)
    for path in paths.values():
        if path.suffix in {".log", ".jsonl", ".json"}:
            path.unlink(missing_ok=True)
    paths["log"].parent.mkdir(parents=True, exist_ok=True)
    paths["camera"].mkdir(parents=True, exist_ok=True)
    paths["output"].parent.mkdir(parents=True, exist_ok=True)

    command = [
        str(args.policy_python),
        str(SCRIPT_PATH),
        "--role",
        "parent",
        "--instruction-mode",
        mode,
        "--model-path",
        str(args.model_path),
        "--task",
        args.task,
        "--episodes",
        str(args.episodes),
        "--episode-steps",
        str(args.episode_steps),
        "--seed",
        str(seed),
        "--sim-python",
        str(args.sim_python),
        "--robotwin-root",
        str(args.robotwin_root),
        "--tokenizer-path",
        str(args.tokenizer_path),
        "--task-config",
        args.task_config,
        "--output",
        str(paths["output"]),
    ]
    env = os.environ.copy()
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": "0",
            "PYTHONUNBUFFERED": "1",
            "ROBOTWIN_SKIP_EXPERT": "0",
            "ROBOTWIN_HEARTBEAT_LOG": str(paths["log"]),
            "ROBOTWIN_CAM_DUMP_DIR": str(paths["camera"]),
            "ROBOTWIN_TRAJECTORY_JSONL": str(paths["trajectory"]),
            "VK_ICD_FILENAMES": "/etc/vulkan/icd.d/nvidia_icd.json",
            "ROBOTWIN_INSTRUCTION_MODE": mode,
        }
    )
    with paths["log"].open("w", encoding="utf-8") as stream:
        stream.write(
            f"D127_RUN_START mode={mode} seed={seed} "
            f"watchdog_seconds={args.watchdog_seconds}\n"
        )
        stream.flush()
        process = subprocess.Popen(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=stream,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )
        watchdog_failure: str | None = None
        previous_ticks = _cpu_ticks(process.pid)
        while process.poll() is None:
            time.sleep(10.0)
            if process.poll() is not None:
                break
            try:
                log_age = time.time() - paths["log"].stat().st_mtime
            except FileNotFoundError:
                log_age = float("inf")
            current_ticks = _cpu_ticks(process.pid)
            tick_delta = (
                None
                if current_ticks is None or previous_ticks is None
                else current_ticks - previous_ticks
            )
            if log_age > args.watchdog_seconds:
                watchdog_failure = (
                    "D127_WATCHDOG_FAIL "
                    f"mode={mode} seed={seed} log_age_seconds={log_age:.1f} "
                    f"cpu_ticks_delta={tick_delta}"
                )
                stream.write(watchdog_failure + "\n")
                stream.flush()
                _terminate_process_group(process)
                break
            previous_ticks = current_ticks
        returncode = process.wait()

    payload: dict[str, Any] = {
        "mode": mode,
        "seed": seed,
        "command": command,
        "returncode": returncode,
        "log_path": str(paths["log"]),
        "camera_dump_dir": str(paths["camera"]),
        "trajectory_path": str(paths["trajectory"]),
        "harness_output_path": str(paths["output"]),
        "watchdog_seconds": args.watchdog_seconds,
        "watchdog_failure": watchdog_failure,
    }
    if paths["output"].exists():
        payload["harness"] = json.loads(
            paths["output"].read_text(encoding="utf-8")
        )
        payload["episode_keys"] = _episode_keys(payload["harness"])
    paths["summary"].write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(
        f"D127_ARM mode={mode} seed={seed} returncode={returncode} "
        f"episodes={payload.get('episode_keys', [])}",
        flush=True,
    )
    if returncode != 0:
        raise RuntimeError(
            f"{mode} arm failed for seed {seed}; see {paths['log']}"
        )
    return payload


def run_paired(args: argparse.Namespace) -> int:
    verify_harness()
    if args.episodes <= 0 or args.episode_steps <= 0:
        raise ValueError("--episodes and --episode-steps must be positive")

    comparisons: list[dict[str, Any]] = []
    for seed in args.seeds:
        official = _run_arm_subprocess(args, "official", seed)
        ours = _run_arm_subprocess(args, "ours", seed)
        official_keys = official["episode_keys"]
        ours_keys = ours["episode_keys"]
        if official_keys != ours_keys:
            raise RuntimeError(
                "D127_PAIRED_SEED_MISMATCH "
                f"seed={seed} official={official_keys} ours={ours_keys}"
            )
        official_harness = official["harness"]
        ours_harness = ours["harness"]
        comparisons.append(
            {
                "base_seed": seed,
                "episode_keys": official_keys,
                "official_successes": official_harness.get("successes"),
                "ours_successes": ours_harness.get("successes"),
                "official_success_rate": official_harness.get("success_rate"),
                "ours_success_rate": ours_harness.get("success_rate"),
            }
        )

    result = {
        "experiment": "D127_instruction_selection_paired_test",
        "task": args.task,
        "task_config": args.task_config,
        "checkpoint": str(args.model_path),
        "episodes_per_base_seed": args.episodes,
        "episode_steps": args.episode_steps,
        "instruction_modes": ["official", "ours"],
        "paired_seed_comparisons": comparisons,
        "harness_sha256": EXPECTED_HARNESS_SHA256,
        "cuda_visible_devices": "0",
        "serialized": True,
    }
    output = PROJECT_ROOT / "results" / "gate3_instruction_paired_test.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(f"D127_PAIRED_RESULT={output}", flush=True)
    print("D127_PAIRED_SEED_MATCH=True", flush=True)
    return 0


def main() -> int:
    args = parse_args()
    if args.dry_run_instructions:
        return run_cpu_instruction_dry_run(args)
    if args.paired:
        return run_paired(args)
    if args.role == "sim":
        harness = import_harness()
        mode = args.instruction_mode
        if mode == "official":
            # Returning the fallback source makes the pinned harness call its
            # own build_instruction(), including the exact official selector.
            def official_fallback(
                task_name: str,
                arm_tag: str | None = None,
                seed: int = 0,
            ) -> tuple[str, str]:
                return task_name, "task_name_fallback"

            harness.seen_instruction = official_fallback
        return int(harness.run_simulator(args))
    return run_one_arm(args)


if __name__ == "__main__":
    raise SystemExit(main())
