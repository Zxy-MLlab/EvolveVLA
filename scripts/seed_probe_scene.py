#!/usr/bin/env python3
"""Probe RoboTwin scene initialization without executing the task policy."""

import sys

sys.path.append("./")

import sapien.core as sapien  # noqa: F401
from sapien.render import clear_cache  # noqa: F401
from collections import OrderedDict  # noqa: F401
import pdb  # noqa: F401
from envs import *  # noqa: F401,F403
import yaml
import importlib
import json  # noqa: F401
import re
import traceback  # noqa: F401
import os
import time  # noqa: F401
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path

import numpy as np
from PIL import Image


TASK_NAME = "beat_block_hammer"
TASK_CONFIG = "demo_clean"
REPO_ROOT = Path(__file__).resolve().parents[1] / "robotwin_repo"
RESULTS_ROOT = Path(__file__).resolve().parents[1] / "results"


def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args


def build_collector_args():
    config_path = os.path.join(CONFIGS_PATH, f"{TASK_CONFIG}.yml")
    with open(config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    args["task_name"] = TASK_NAME
    embodiment_type = args.get("embodiment")
    embodiment_config_path = os.path.join(CONFIGS_PATH, "_embodiment_config.yml")
    with open(embodiment_config_path, "r", encoding="utf-8") as f:
        embodiment_types = yaml.load(f.read(), Loader=yaml.FullLoader)

    def get_embodiment_file(embodiment):
        robot_file = embodiment_types[embodiment]["file_path"]
        if robot_file is None:
            raise RuntimeError("missing embodiment files")
        return robot_file

    if len(embodiment_type) == 1:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["dual_arm_embodied"] = True
    elif len(embodiment_type) == 3:
        args["left_robot_file"] = get_embodiment_file(embodiment_type[0])
        args["right_robot_file"] = get_embodiment_file(embodiment_type[1])
        args["embodiment_dis"] = embodiment_type[2]
        args["dual_arm_embodied"] = False
    else:
        raise ValueError("number of embodiment config parameters should be 1 or 3")

    args["left_embodiment_config"] = get_embodiment_config(args["left_robot_file"])
    args["right_embodiment_config"] = get_embodiment_config(args["right_robot_file"])

    if len(embodiment_type) == 1:
        embodiment_name = str(embodiment_type[0])
    else:
        embodiment_name = str(embodiment_type[0]) + "+" + str(embodiment_type[1])

    args["embodiment_name"] = embodiment_name
    args["task_config"] = TASK_CONFIG
    embodiment_dir = re.sub(r"[^A-Za-z0-9_]+", "_", embodiment_name).lower()
    args["save_path"] = os.path.join(
        args["save_path"],
        args["task_config"],
        str(args["task_name"]),
        embodiment_dir,
    )
    return args


def class_decorator(task_name):
    envs_module = importlib.import_module(f"envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        return env_class()
    except Exception as exc:
        raise SystemExit("No such task") from exc


def write_initial_head_rgb(task_env, seed):
    raw_observation = task_env.get_obs()
    frame = np.asarray(
        raw_observation["observation"]["head_camera"]["rgb"], dtype=np.uint8
    )
    if frame.ndim == 2:
        frame = np.repeat(frame[..., None], 3, axis=-1)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise RuntimeError(f"head camera RGB must be HxWx3, got {frame.shape}")

    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    png_path = RESULTS_ROOT / f"seed_probe_scene_seed{seed}_head_rgb.png"
    Image.fromarray(frame).save(png_path)
    return sha256(png_path.read_bytes()).hexdigest()


def probe_seed(task_env, args, episode_idx, seed):
    # This call intentionally matches collect_data.py: setup_demo(..., **args).
    task_env.setup_demo(now_ep_num=episode_idx, seed=seed, **args)
    block_pose = task_env.block.get_functional_point(0, "pose")
    position = np.asarray(block_pose.p, dtype=float).reshape(-1)[:3]
    rotation = np.asarray(block_pose.q, dtype=float).reshape(-1)
    rgb_sha = write_initial_head_rgb(task_env, seed)
    print(
        "SEED={} BLOCK_POSE=({:.9g},{:.9g},{:.9g}) ROT=({}) RGB_SHA={}".format(
            seed,
            position[0],
            position[1],
            position[2],
            ",".join(f"{value:.9g}" for value in rotation),
            rgb_sha,
        ),
        flush=True,
    )
    task_env.close_env()


def main():
    parser = ArgumentParser()
    parser.add_argument("seeds", nargs="+", type=int)
    parsed = parser.parse_args()

    os.chdir(REPO_ROOT)
    args = build_collector_args()
    task_env = class_decorator(TASK_NAME)
    try:
        for episode_idx, seed in enumerate(parsed.seeds):
            probe_seed(task_env, args, episode_idx, seed)
    finally:
        close_env = getattr(task_env, "close_env", None)
        if callable(close_env):
            close_env()


if __name__ == "__main__":
    main()
