#!/usr/bin/env python3
"""Positive control for the RoboTwin turn_switch success contract."""

from __future__ import annotations

import sys
import os
from pathlib import Path

import numpy as np
import yaml


REPO = Path("/root/autodl-tmp/EvolveVLA/robotwin_repo")


def build_args(task_name: str) -> dict:
    task_config_dir = REPO / "env_cfg" / "task_config"
    args = yaml.safe_load((task_config_dir / "demo_clean.yml").read_text())
    embodiments = yaml.safe_load(
        (task_config_dir / "_embodiment_config.yml").read_text()
    )

    embodiment_name = "aloha-agilex"
    embodiment_dir = REPO / embodiments[embodiment_name]["file_path"]
    embodiment_config = yaml.safe_load(
        (embodiment_dir / "config.yml").read_text()
    )

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


def main() -> int:
    os.chdir(REPO)
    sys.path.insert(0, str(REPO))

    from envs.turn_switch import turn_switch

    task_name = "turn_switch"
    task = turn_switch()
    try:
        print(f"TASK={task_name}")
        print("EMBODIMENT=aloha-agilex")
        print("ACTION_MODE=joint")
        print("SETUP_DEMO_BEGIN")
        task.setup_demo(
            now_ep_num=0,
            seed=0,
            is_test=True,
            **build_args(task_name),
        )
        print("SETUP_DEMO_OK")

        at_reset = task.check_success()
        print(f"SUCCESS_TYPE_AT_RESET={type(at_reset)}")
        print(f"SUCCESS_VALUE_AT_RESET={at_reset!r}")

        limits = np.asarray(task.switch.get_qlimits())[0]
        task.switch.set_qpos([float(limits[1])])
        task.scene.step()

        after_force = task.check_success()
        print(f"SUCCESS_TYPE_AFTER_FORCE={type(after_force)}")
        print(f"SUCCESS_VALUE_AFTER_FORCE={after_force!r}")
        print(f"POSITIVE_CONTROL_LIT_TRUE={bool(after_force)}")
        return 0 if bool(after_force) is True else 1
    finally:
        task.close_env(clear_cache=True)


if __name__ == "__main__":
    raise SystemExit(main())
