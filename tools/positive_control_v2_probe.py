#!/usr/bin/env python3
"""Probe the RoboCasa positive-control path and policy-input confounds."""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.cache/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("TMPDIR", "/root/autodl-tmp/tmp")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")
os.environ.setdefault("REQUESTS_CA_BUNDLE", "/etc/ssl/certs/ca-certificates.crt")
os.environ.setdefault("SSL_CERT_FILE", "/etc/ssl/certs/ca-certificates.crt")

import numpy as np

from lerobot.envs.robocasa import RoboCasaEnv


TASK = "CloseBlenderLid"
OUT = Path("/root/autodl-tmp/EvolveVLA/results/positive_control_v2")
IMG_OUT = Path("/root/autodl-tmp/EvolveVLA/results/ctrl_imgs")


def print_json(label: str, value) -> None:
    print(label + "=" + json.dumps(value, sort_keys=True, default=str))


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    IMG_OUT.mkdir(parents=True, exist_ok=True)

    env = RoboCasaEnv(
        task=TASK,
        split="target",
        observation_height=256,
        observation_width=256,
    )
    observation, reset_info = env.reset(seed=1000)
    raw_wrapper = env._env
    raw_task = raw_wrapper.env

    print("ENV_CLASS", type(env))
    print("INNER_WRAPPER_CLASS", type(raw_wrapper))
    print("RAW_TASK_CLASS", type(raw_task))
    print_json("RESET_INFO", reset_info)
    print_json("RESET_OBS_SHAPES", {
        "state": list(np.asarray(observation["agent_pos"]).shape),
        "images": {k: list(np.asarray(v).shape) for k, v in observation["pixels"].items()},
    })
    print("RAW_CHECK_SUCCESS_RESET", bool(raw_task._check_success()))
    print("RAW_REWARD_RESET", float(1.0 if raw_task._check_success() else 0.0))

    blender = raw_task.blender
    lid = blender.blender_lid
    print_json("BLENDER_STATE_RESET", blender.get_state())
    print_json("LID_OBJECT", {
        "name": lid.name,
        "joints": getattr(lid, "joints", None),
        "joint_names": list(getattr(lid, "joint_names", []) or []),
        "closed_pos": np.asarray(blender.get_lid_closed_pos(raw_task)).tolist(),
        "current_pos": np.asarray(blender.get_curr_lid_pos(raw_task)).tolist(),
    })
    model = raw_task.sim.model
    joint_names = [model.joint_id2name(i) for i in range(model.njnt)]
    print_json("BLENDER_LID_JOINT_NAMES", [n for n in joint_names if n and "blender_lid" in n])
    print_json("BLENDER_LID_BODY_NAMES", [
        model.body_id2name(i) for i in range(model.nbody)
        if model.body_id2name(i) and "blender_lid" in model.body_id2name(i)
    ])

    # Dump the raw camera arrays from the identical wrapper before any action.
    from PIL import Image

    for name, image in observation["pixels"].items():
        array = np.asarray(image)
        Image.fromarray(array.astype(np.uint8)).save(IMG_OUT / f"{name}.png")
        print_json("CAMERA_STATS_" + name, {
            "shape": list(array.shape),
            "dtype": str(array.dtype),
            "min": float(array.min()),
            "max": float(array.max()),
            "mean": float(array.mean()),
            "std": float(array.std()),
        })

    # A zero action provides a controlled, repeatable T2 rollout.
    state_rows = []
    current_observation = observation
    for step in range(30):
        state = np.asarray(current_observation["agent_pos"], dtype=np.float64)
        raw_success = bool(raw_task._check_success())
        action = np.zeros(12, dtype=np.float32)
        next_observation, reward, terminated, truncated, info = env.step(action)
        state_rows.append({
            "step": step,
            "raw_check_success": raw_success,
            "reward": float(reward),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "state": state.tolist(),
            "info": info,
        })
        current_observation = next_observation
        if terminated or truncated:
            break
    state_array = np.asarray([row["state"] for row in state_rows])
    print_json("T2_SUMMARY", {
        "steps": len(state_rows),
        "state_first": state_array[0].tolist() if len(state_array) else None,
        "state_last": state_array[-1].tolist() if len(state_array) else None,
        "state_max_abs_delta": (
            float(np.max(np.abs(np.diff(state_array, axis=0)))) if len(state_array) > 1 else 0.0
        ),
        "raw_success_any": any(row["raw_check_success"] for row in state_rows),
        "reward_any": any(row["reward"] > 0 for row in state_rows),
    })
    for row in state_rows:
        print_json("T2_STEP", row)

    # Oracle fallback: move the free lid body to the task's closed pose and
    # set an upright quaternion, then let the task update its state from MuJoCo.
    env.reset(seed=1000)
    raw_task = env._env.env
    blender = raw_task.blender
    lid = blender.blender_lid
    model = raw_task.sim.model
    data = raw_task.sim.data
    lid_joint = str(lid.joints[0])
    joint_id = model.joint_name2id(lid_joint)
    qpos_addr = model.jnt_qposadr[joint_id]
    closed_pos = np.asarray(blender.get_lid_closed_pos(raw_task), dtype=np.float64)
    data.qpos[qpos_addr : qpos_addr + 3] = closed_pos
    data.qpos[qpos_addr + 3 : qpos_addr + 7] = np.asarray([1.0, 0.0, 0.0, 0.0])
    data.qvel[:] = 0
    raw_task.sim.forward()
    raw_task.update_state()
    oracle_raw = bool(raw_task._check_success())
    oracle_obs, oracle_reward, oracle_term, oracle_trunc, oracle_info = env.step(np.zeros(12, dtype=np.float32))
    oracle_final = bool(oracle_info.get("is_success", False))
    print_json("ORACLE_AFTER_STATE_SET", {
        "joint": lid_joint,
        "qpos_addr": int(qpos_addr),
        "closed_pos": closed_pos.tolist(),
        "current_pos": np.asarray(blender.get_curr_lid_pos(raw_task)).tolist(),
        "blender_state": blender.get_state(),
        "raw_check_success_before_step": oracle_raw,
        "wrapper_reward_after_step": float(oracle_reward),
        "wrapper_terminated_after_step": bool(oracle_term),
        "wrapper_info_after_step": oracle_info,
        "wrapper_is_success_after_step": oracle_final,
        "observation_state_after_step": np.asarray(oracle_obs["agent_pos"]).tolist(),
    })

    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
