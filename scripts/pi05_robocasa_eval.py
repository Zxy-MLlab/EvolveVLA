#!/usr/bin/env python
"""Evaluate a cached pi0.5 policy on RoboCasa through LeRobot.

Serves BOTH lerobot/pi05_robocasa (declares RAW robot0_* camera keys ->
identity) and lerobot/pi05_base (declares base_0_rgb-style keys ->
name-matched rename). The env->policy camera map is DERIVED from the
checkpoint's own input_features, never hardcoded for one convention.

Split: target (framework encoding of the official RoboCasa365 Atomic-Seen
protocol). Not leaderboard-comparable (published 39.6% is pretrain).
"""
from __future__ import annotations
import argparse
import json
import os
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("HF_HOME", "/root/autodl-tmp/.cache/huggingface")
os.environ.setdefault("HF_HUB_DISABLE_XET", "1")
os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
os.environ.setdefault("MUJOCO_EGL_DEVICE_ID", "0")

import numpy as np
import torch
from lerobot.envs.configs import RoboCasaEnv
from lerobot.envs.factory import make_env, make_env_pre_post_processors
from lerobot.envs.utils import preprocess_observation
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.utils.constants import ACTION

DEFAULT_MODEL_PATH = "/root/autodl-tmp/.cache/huggingface/hub/models--lerobot--pi05_base"
OFFICIAL_TOKENIZER_PATH = "/root/autodl-tmp/paligemma_tokenizer"
DEFAULT_TASKS = ["CloseBlenderLid", "CloseFridge", "CloseToasterOvenDoor"]

# Single obs size used by BOTH smoke and evaluate_task (matches declared [3,256,256])
OBS_SIZE = 256

# Camera semantics (do NOT transpose):
#   robot0_agentview_left  = external view
#   robot0_eye_in_hand     = wrist
#   robot0_agentview_right = second external view
_RPBOCASA_TO_BASE = {
    "robot0_agentview_left": "base_0_rgb",
    "robot0_eye_in_hand": "left_wrist_0_rgb",
    "robot0_agentview_right": "right_wrist_0_rgb",
}


def derive_rename_map(policy_cfg):
    feats = getattr(policy_cfg, "input_features", {}) or {}
    declared = [str(k) for k in feats if "observation.images." in str(k)]
    if declared and all(k.rsplit(".", 1)[-1].startswith("robot0_") for k in declared):
        return {}
    out = {}
    for env_short, base_short in _RPBOCASA_TO_BASE.items():
        src = "observation.images." + env_short
        dst = "observation.images." + base_short
        if dst in declared:
            out[src] = dst
    return out


def build_policy(model_path, env_cfg):
    policy_cfg = PI05Config.from_pretrained(model_path, local_files_only=True)
    policy_cfg.device = "cuda:0"
    policy_cfg.pretrained_path = Path(model_path)
    rename_map = derive_rename_map(policy_cfg)
    print("RESOLVED CAMERA MAP:", rename_map or "identity (raw robot0_* keys)")
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=model_path,
        preprocessor_overrides={
            "device_processor": {"device": "cuda:0"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": OFFICIAL_TOKENIZER_PATH},
        },
    )
    return policy, preprocessor, postprocessor


def _assert_batch(batch, policy_cfg):
    feats = getattr(policy_cfg, "input_features", {}) or {}
    expected_img = {str(k) for k in feats if "observation.images." in str(k)}
    got_img = {k for k in batch if "observation.images." in k}
    assert got_img == expected_img, (
        "image key contract mismatch\n  expected=%s\n  got=%s" % (sorted(expected_img), sorted(got_img))
    )
    for k in sorted(got_img):
        print("BOUNDARY IMAGE %s: shape=%s dtype=%s" % (k, tuple(batch[k].shape), batch[k].dtype))


def rollout(env, policy, env_preprocessor, env_postprocessor, preprocessor, postprocessor, steps, seed):
    policy.reset()
    observation, info = env.reset(seed=[seed])
    success = False
    last_reward = 0.0
    prepared_summary = None
    first_action = None
    for step in range(steps):
        observation = preprocess_observation(observation)
        observation["task"] = list(env.call("task_description"))
        observation = env_preprocessor(observation)
        batch = preprocessor(observation)
        if step == 0:
            _assert_batch(batch, policy.config)
            prepared_summary = {
                key: {
                    "shape": list(value.shape) if hasattr(value, "shape") else None,
                    "dtype": str(value.dtype) if hasattr(value, "dtype") else type(value).__name__,
                }
                for key, value in batch.items()
            }
        with torch.inference_mode():
            action = policy.select_action(batch)
        if step == 0:
            a0 = action.detach().float().cpu()
            assert torch.isfinite(a0).all(), "non-finite action: %s" % a0
            expected_action_dim = policy.config.output_features[ACTION].shape[0]
            assert a0.shape[-1] == expected_action_dim, (
                "action dim mismatch: got=%s expected=%s" % (a0.shape[-1], expected_action_dim)
            )
            first_action = a0.numpy().tolist()
            print("FIRST ACTION finite: min=%.4f max=%.4f (informational raw range)" % (float(a0.min()), float(a0.max())))
        action = postprocessor(action)
        action = env_postprocessor({ACTION: action})[ACTION]
        observation, reward, terminated, truncated, info = env.step(action.detach().cpu().numpy())
        last_reward = float(np.asarray(reward).reshape(-1)[0])
        if "final_info" in info:
            fi = info["final_info"]
            if isinstance(fi, dict):
                success = bool(np.asarray(fi.get("is_success", False)).reshape(-1)[0])
            else:
                for item in fi:
                    if isinstance(item, dict):
                        success = success or bool(item.get("is_success", False))
        elif "is_success" in info:
            success = bool(np.asarray(info["is_success"]).reshape(-1)[0]) or success
        if bool(np.asarray(terminated).reshape(-1)[0]) or bool(np.asarray(truncated).reshape(-1)[0]):
            break
    return {"success": success, "steps": step + 1, "last_reward": last_reward,
            "first_action": first_action, "prepared_inputs": prepared_summary}


def evaluate_task(model_path, task, rollouts, steps, seed):
    env_cfg = RoboCasaEnv(task=task, split="target",
                          observation_height=OBS_SIZE, observation_width=OBS_SIZE)
    envs = make_env(env_cfg, n_envs=1, use_async_envs=False)
    env = envs[task][0]
    policy, preprocessor, postprocessor = build_policy(model_path, env_cfg)
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(env_cfg, policy.config)
    results = []
    try:
        for index in range(rollouts):
            results.append(rollout(env, policy, env_preprocessor, env_postprocessor,
                                   preprocessor, postprocessor, steps, seed + index))
    finally:
        env.close()
    successes = sum(r["success"] for r in results)
    return {"task": task, "rollouts": rollouts, "successes": successes,
            "success_rate": successes / rollouts if rollouts else 0.0, "results": results}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS)
    parser.add_argument("--rollouts", type=int, default=1)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError("Expected one visible GPU, got %d" % torch.cuda.device_count())
    reports = []
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
    def write_payload():
        payload = {"torch": torch.__version__, "cuda": torch.version.cuda,
                   "device": torch.cuda.get_device_name(0), "obs_size": OBS_SIZE,
                   "split": "target", "reports": reports}
        text = json.dumps(payload, indent=2)
        if args.output:
            args.output.write_text(text + "\n")
        return text
    for i, task in enumerate(args.tasks):
        try:
            report = evaluate_task(args.model_path, task, args.rollouts, args.steps, args.seed + i * 100)
        except Exception as exc:
            report = {"task": task, "status": "error", "error": str(exc),
                      "traceback": traceback.format_exc()}
        reports.append(report)
        text = write_payload()
    print(text)


if __name__ == "__main__":
    main()
