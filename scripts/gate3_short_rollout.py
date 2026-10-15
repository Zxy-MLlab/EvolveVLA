#!/usr/bin/env python3
"""Run a short official RoboTwin pi0.5 rollout and summarize arm motion."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys


PROJECT_ROOT = Path("/root/autodl-tmp/EvolveVLA")
HARNESS = PROJECT_ROOT / "scripts/robotwin_eval_harness.py"
POLICY_PYTHON = Path("/root/autodl-tmp/conda_envs/evolvevla_pi312/bin/python")
SIM_PYTHON = Path("/root/autodl-tmp/conda_envs/robotwin_cu128/bin/python")
ROBOTWIN_ROOT = PROJECT_ROOT / "robotwin_repo"
MODEL_PATH = PROJECT_ROOT / "checkpoints/pi05_robotwin_lerobot"
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
RESULTS_DIR = PROJECT_ROOT / "results"
RUN_LOG = RESULTS_DIR / "gate3_short_rollout.log"
TRAJECTORY = RESULTS_DIR / "gate3_short_rollout_trajectory.jsonl"
RESULT_JSON = RESULTS_DIR / "gate3_short_rollout.json"

EPISODES = 2
EPISODE_STEPS = 30
ARM_INDICES = tuple(range(6)) + tuple(range(7, 13))


def require_path(path: Path, kind: str) -> None:
    if not path.exists():
        raise RuntimeError(f"SHORT_ROLLOUT_PRECONDITION_NOT_MET: missing {kind}: {path}")


def l2(values: list[object], indices: tuple[int, ...]) -> float:
    return math.sqrt(
        sum(float(values[index]) ** 2 for index in indices)
    )


def emit_summary() -> int:
    if not TRAJECTORY.exists():
        print(f"SHORT_ROLLOUT_TRAJECTORY_MISSING path={TRAJECTORY}")
        return 3

    previous_qpos: list[object] | None = None
    previous_step = 0
    episode = 0
    global_step = 0
    action_magnitudes: list[float] = []
    motion_deltas: list[float] = []
    successes = 0

    with TRAJECTORY.open("r", encoding="utf-8") as stream:
        for raw_line in stream:
            if not raw_line.strip():
                continue
            record = json.loads(raw_line)
            local_step = int(record["step"])
            if previous_step and local_step <= previous_step:
                episode += 1
                previous_qpos = None
            previous_step = local_step

            action = list(record["action"])
            qpos = list(record["qpos"])
            raw_action_arm_l2 = l2(action, ARM_INDICES)
            if previous_qpos is None:
                arm_pos_delta = 0.0
            else:
                arm_pos_delta = math.sqrt(
                    sum(
                        (float(qpos[index]) - float(previous_qpos[index])) ** 2
                        for index in ARM_INDICES
                    )
                )

            success = bool(
                record.get("success_predicate_intermediates", {}).get(
                    "is_success", False
                )
            )
            global_step += 1
            action_magnitudes.append(raw_action_arm_l2)
            motion_deltas.append(arm_pos_delta)
            successes += int(success)
            print(
                f"STEP={global_step} "
                f"RAW_ACTION_ARM_L2={raw_action_arm_l2:.9f} "
                f"ARM_POS_DELTA={arm_pos_delta:.9f} "
                f"SUCCESS={str(success).lower()} "
                f"EPISODE={episode} LOCAL_STEP={local_step}",
                flush=True,
            )
            previous_qpos = qpos

    if not action_magnitudes:
        print("SHORT_ROLLOUT_NO_STEPS")
        return 4

    mean_action = sum(action_magnitudes) / len(action_magnitudes)
    total_motion = sum(motion_deltas)
    print(
        f"SHORT_ROLLOUT_DONE n_steps={len(action_magnitudes)} "
        f"mean_action_arm_l2={mean_action:.9f} "
        f"total_arm_motion={total_motion:.9f} successes={successes}",
        flush=True,
    )
    return 0


def main() -> int:
    for path, kind in (
        (HARNESS, "harness"),
        (POLICY_PYTHON, "policy Python"),
        (SIM_PYTHON, "simulator Python"),
        (ROBOTWIN_ROOT, "RoboTwin root"),
        (MODEL_PATH, "official cotrain checkpoint"),
        (TOKENIZER_PATH, "official tokenizer"),
    ):
        require_path(path, kind)

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    for path in (RUN_LOG, TRAJECTORY, RESULT_JSON):
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = "0"
    env["PYTHONUNBUFFERED"] = "1"
    env["ROBOTWIN_HEARTBEAT_LOG"] = str(RUN_LOG)
    env["ROBOTWIN_TRAJECTORY_JSONL"] = str(TRAJECTORY)

    command = [
        str(POLICY_PYTHON),
        str(HARNESS),
        "--model-path",
        str(MODEL_PATH),
        "--tokenizer-path",
        str(TOKENIZER_PATH),
        "--sim-python",
        str(SIM_PYTHON),
        "--robotwin-root",
        str(ROBOTWIN_ROOT),
        "--task",
        "beat_block_hammer",
        "--episodes",
        str(EPISODES),
        "--episode-steps",
        str(EPISODE_STEPS),
        "--task-config",
        "demo_clean",
        "--output",
        str(RESULT_JSON),
    ]

    print(
        "SHORT_ROLLOUT_COMMAND="
        + " ".join(command),
        flush=True,
    )
    print(f"SHORT_ROLLOUT_LOG={RUN_LOG}", flush=True)
    print(f"SHORT_ROLLOUT_TRAJECTORY={TRAJECTORY}", flush=True)

    with RUN_LOG.open("w", encoding="utf-8") as log_stream:
        completed = subprocess.run(
            command,
            cwd=str(PROJECT_ROOT),
            env=env,
            stdout=log_stream,
            stderr=subprocess.STDOUT,
            check=False,
        )

    print(f"SHORT_ROLLOUT_HARNESS_RC={completed.returncode}", flush=True)
    summary_rc = emit_summary()
    if completed.returncode != 0:
        print("SHORT_ROLLOUT_HARNESS_TAIL_BEGIN")
        try:
            lines = RUN_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
            for line in lines[-60:]:
                print(line)
        except Exception as exc:
            print(f"SHORT_ROLLOUT_LOG_READ_FAILED={exc!r}")
        print("SHORT_ROLLOUT_HARNESS_TAIL_END")
        return completed.returncode
    return summary_rc


if __name__ == "__main__":
    raise SystemExit(main())
