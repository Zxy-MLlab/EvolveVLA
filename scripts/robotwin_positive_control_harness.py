#!/usr/bin/env python3
"""Positive control for the full RoboTwin harness success-read path."""

from __future__ import annotations

import importlib
import os
import sys
from pathlib import Path

import numpy as np


ROOT = Path("/root/autodl-tmp/EvolveVLA/robotwin_repo")


def build_args(task_name: str) -> dict:
    import yaml

    task_config_dir = ROOT / "env_cfg" / "task_config"
    with (task_config_dir / "demo_clean.yml").open("r", encoding="utf-8") as handle:
        args = yaml.safe_load(handle)
    with (task_config_dir / "_embodiment_config.yml").open(
        "r", encoding="utf-8"
    ) as handle:
        embodiments = yaml.safe_load(handle)

    embodiment_name = "aloha-agilex"
    embodiment_dir = ROOT / embodiments[embodiment_name]["file_path"]
    with (embodiment_dir / "config.yml").open("r", encoding="utf-8") as handle:
        embodiment_config = yaml.safe_load(handle)

    args.update(
        {
            "task_name": task_name,
            "task_config": "demo_clean",
            "left_robot_file": str(embodiment_dir),
            "right_robot_file": str(embodiment_dir),
            "left_embodiment_config": embodiment_config,
            "right_embodiment_config": embodiment_config,
            "dual_arm_embodied": True,
            "eval_mode": True,
            "need_plan": False,
            "render_freq": 0,
            "save_data": False,
        }
    )
    return args


def harness_is_success(task) -> bool:
    # Keep this expression identical to robotwin_eval_harness.py:552-554.
    is_success = bool(getattr(task, "eval_success", False))
    if not is_success and hasattr(task, "check_success"):
        is_success = bool(task.check_success())
    return is_success


def main() -> int:
    os.chdir(ROOT)
    sys.path.insert(0, str(ROOT))

    task_name = "turn_switch"
    task_module = importlib.import_module(f"envs.{task_name}")
    task = getattr(task_module, task_name)()
    args = build_args(task_name)

    try:
        task.setup_demo(now_ep_num=0, seed=0, is_test=True, **args)

        eval_success_at_reset = task.eval_success
        check_success_at_reset = task.check_success()
        harness_is_success_at_reset = harness_is_success(task)
        print(f"EVAL_SUCCESS_AT_RESET={eval_success_at_reset!r}")
        print(f"CHECK_SUCCESS_AT_RESET={check_success_at_reset!r}")
        print(f"HARNESS_IS_SUCCESS_AT_RESET={harness_is_success_at_reset!r}")

        limits = np.asarray(task.switch.get_qlimits())[0]
        task.switch.set_qpos([float(limits[1])])
        task.scene.step()

        # First verify the check_success() fallthrough while eval_success is false.
        fallthrough_eval_success = task.eval_success
        fallthrough_check_success = task.check_success()
        fallthrough_harness_is_success = harness_is_success(task)
        print(f"EVAL_SUCCESS_AFTER_FALLTHROUGH={fallthrough_eval_success!r}")
        print(f"CHECK_SUCCESS_AFTER_FALLTHROUGH={fallthrough_check_success!r}")
        print(f"HARNESS_IS_SUCCESS_AFTER_FALLTHROUGH={fallthrough_harness_is_success!r}")

        # The harness reads this primary flag before consulting check_success().
        task.eval_success = True
        eval_success_after_force = task.eval_success
        check_success_after_force = task.check_success()
        harness_is_success_after_force = harness_is_success(task)
        print(f"EVAL_SUCCESS_AFTER_FORCE={eval_success_after_force!r}")
        print(f"CHECK_SUCCESS_AFTER_FORCE={check_success_after_force!r}")
        print(f"HARNESS_IS_SUCCESS_AFTER_FORCE={harness_is_success_after_force!r}")
        positive_control = (
            harness_is_success_at_reset is False
            and fallthrough_harness_is_success is True
            and harness_is_success_after_force is True
        )
        print(f"HARNESS_POSITIVE_CONTROL_LIT_TRUE={positive_control}")

        # Match the harness episode boundary: close the old task and setup a new one.
        task.close_env(clear_cache=True)
        task.setup_demo(now_ep_num=1, seed=1, is_test=True, **args)
        harness_is_success_after_reset = harness_is_success(task)
        print(
            "HARNESS_IS_SUCCESS_AFTER_EPISODE_RESET="
            f"{harness_is_success_after_reset!r}"
        )

        return 0 if positive_control and harness_is_success_after_reset is False else 1
    finally:
        if hasattr(task, "close_env"):
            task.close_env(clear_cache=True)


if __name__ == "__main__":
    raise SystemExit(main())
