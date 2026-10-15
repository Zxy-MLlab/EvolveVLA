#!/usr/bin/env python3
"""D149 official-checkpoint serving-path adjudication.

The real path calls robotwin_eval_harness.load_policy(), which constructs the
official PI05Policy through PI05Policy.from_pretrained and returns the exact
harness policy and environment processors. No simulator or rollout is used.

--dry-run constructs the config and processor path on CPU, imports the pinned
harness and fixed-observation loader, and exercises metric formatting with
synthetic actions. It never instantiates PI05Policy.
"""

from __future__ import annotations

import argparse
import ast
import importlib.util
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path("/root/autodl-tmp/EvolveVLA")
SCRIPTS = ROOT / "scripts"
HARNESS_PATH = SCRIPTS / "robotwin_eval_harness.py"
PROBE_PATH = SCRIPTS / "gate3_camera_permutation_repro.py"
CHECKPOINT = ROOT / "checkpoints/pi05_robotwin_lerobot"
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
TASK = "beat_block_hammer"
EPISODE = 50
POSITION = 31
VIDEO_INDEX = 7219
EPISODE_STEPS = 1200
CHUNK_SIZE = 50
ACTION_DIM = 14
ARM_INDICES = np.asarray(
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    dtype=np.int64,
)
CURRENT_CAMERA_MAP = {
    "cam_high": "cam_high",
    "cam_left_wrist": "cam_left_wrist",
    "cam_right_wrist": "cam_right_wrist",
}


def load_module(path: Path, name: str) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def load_harness() -> Any:
    return load_module(HARNESS_PATH, "d149_robotwin_eval_harness")


def load_probe() -> Any:
    return load_module(PROBE_PATH, "d149_camera_permutation_probe")


def validate_sample(sample: dict[str, Any]) -> None:
    if sample["episode"] != EPISODE or sample["position"] != POSITION:
        raise RuntimeError(
            f"unexpected sample coordinates: {sample['episode']}/{sample['position']}"
        )
    if sample["video_index"] != VIDEO_INDEX:
        raise RuntimeError(
            f"unexpected video index: {sample['video_index']} != {VIDEO_INDEX}"
        )
    if sample["recorded"].shape != (CHUNK_SIZE, ACTION_DIM):
        raise RuntimeError(f"unexpected recorded chunk shape: {sample['recorded'].shape}")


def harness_args(model_path: Path, tokenizer_path: Path) -> argparse.Namespace:
    return argparse.Namespace(
        model_path=str(model_path),
        tokenizer_path=str(tokenizer_path),
        task=TASK,
        episode_steps=EPISODE_STEPS,
    )


def make_observation_and_batch(
    harness: Any,
    probe: Any,
    sample: dict[str, Any],
    env_preprocessor: Any,
    preprocessor: Any,
    policy_config: Any,
) -> dict[str, Any]:
    observation = probe.make_observation(
        harness,
        sample["frames"],
        CURRENT_CAMERA_MAP,
        sample["state"],
    )
    return harness.make_policy_batch(
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy_config,
    )


def heartbeat(label: str) -> None:
    path_text = os.environ.get("ROBOTWIN_HEARTBEAT_LOG")
    if not path_text:
        return
    path = Path(path_text)
    if not path.is_absolute():
        raise RuntimeError("ROBOTWIN_HEARTBEAT_LOG must be absolute")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    print(f"HEARTBEAT={label}", flush=True)


def start_watchdog() -> tuple[threading.Event, threading.Thread]:
    path_text = os.environ.get("ROBOTWIN_HEARTBEAT_LOG")
    if not path_text or not Path(path_text).is_absolute():
        raise RuntimeError(
            "ROBOTWIN_HEARTBEAT_LOG must be an absolute path for the GPU run"
        )
    path = Path(path_text)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    stop = threading.Event()

    def monitor() -> None:
        while not stop.wait(30.0):
            try:
                age = time.time() - path.stat().st_mtime
            except OSError as exc:
                print(f"WATCHDOG_ERROR={type(exc).__name__}: {exc}", flush=True)
                os._exit(124)
            if age > 120.0:
                print(f"WATCHDOG_STALE_LOG age={age:.1f}s path={path}", flush=True)
                os._exit(124)

    thread = threading.Thread(
        target=monitor,
        name="d149-log-watchdog",
        daemon=True,
    )
    thread.start()
    return stop, thread


def emit_metrics(
    predicted: np.ndarray,
    recorded: np.ndarray,
    *,
    include_steps: bool,
    synthetic: bool,
) -> None:
    predicted_arm = predicted[:, ARM_INDICES]
    recorded_arm = recorded[:, ARM_INDICES]
    predicted_arm_l2 = np.linalg.norm(predicted_arm, axis=1)
    recorded_arm_l2 = np.linalg.norm(recorded_arm, axis=1)
    error_arm_l2 = np.linalg.norm(predicted_arm - recorded_arm, axis=1)
    if synthetic:
        print("DRY_RUN_SYNTHETIC_METRICS=1")
    if include_steps:
        for index, value in enumerate(predicted_arm_l2):
            print(f"PREDICTED_ARM_L2[{index}]={float(value):.8f}")
    else:
        print(f"PREDICTED_ARM_L2[0]={float(predicted_arm_l2[0]):.8f}")
    print(
        "PREDICTED_ARM_L2_CHUNK_MEAN="
        f"{float(np.mean(predicted_arm_l2)):.8f}"
    )
    print(
        "ERROR_TO_RECORDED_ARM_L2_MEAN="
        f"{float(np.mean(error_arm_l2)):.8f}"
    )
    print(f"RECORDED_ARM_L2_CHUNK_MEAN={float(np.mean(recorded_arm_l2)):.8f}")
    print(f"PREDICTED_ACTION_MIN={float(predicted.min()):.8f}")
    print(f"PREDICTED_ACTION_MAX={float(predicted.max()):.8f}")


def dry_run(model_path: Path, tokenizer_path: Path) -> int:
    for path in (Path(__file__), HARNESS_PATH, PROBE_PATH):
        ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    print("DRY_RUN_AST_PARSE=PASS")

    harness = load_harness()
    probe = load_probe()
    if not callable(getattr(harness, "load_policy", None)):
        raise RuntimeError("harness.load_policy is unavailable")
    if not callable(getattr(harness, "make_policy_batch", None)):
        raise RuntimeError("harness.make_policy_batch is unavailable")
    print("DRY_RUN_IMPORT_SMOKE=PASS")

    import torch
    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    env_cfg = RoboTwinEnvConfig(
        task=TASK,
        episode_length=EPISODE_STEPS,
        obs_type="pixels_agent_pos",
        camera_names="cam_high,cam_left_wrist,cam_right_wrist",
        observation_height=harness.OBS_HEIGHT,
        observation_width=harness.OBS_WIDTH,
        action_mode="joint",
    )
    policy_cfg = PI05Config.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    policy_cfg.device = "cpu"
    policy_cfg.pretrained_path = model_path
    rename_map = harness.derive_camera_rename_map(policy_cfg)
    preprocessor, _postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(model_path),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
        },
    )
    env_preprocessor, _env_postprocessor = make_env_pre_post_processors(
        env_cfg,
        policy_cfg,
    )
    sample = probe.load_episode()
    validate_sample(sample)
    batch = make_observation_and_batch(
        harness,
        probe,
        sample,
        env_preprocessor,
        preprocessor,
        policy_cfg,
    )
    expected_image_keys = {
        key
        for key in (getattr(policy_cfg, "input_features", {}) or {})
        if "observation.images." in str(key)
    }
    got_image_keys = {
        key for key in batch if "observation.images." in str(key)
    }
    if got_image_keys != expected_image_keys:
        raise RuntimeError(
            f"dry-run image keys mismatch: expected={sorted(expected_image_keys)} "
            f"got={sorted(got_image_keys)}"
        )
    print(f"DRY_RUN_SAMPLE=episode={EPISODE} position={POSITION} video_index={VIDEO_INDEX}")
    print(f"DRY_RUN_CAMERA_RENAME_MAP={rename_map}")
    print(f"DRY_RUN_BATCH_IMAGE_KEYS={sorted(got_image_keys)}")
    print("DRY_RUN_MODEL_LOAD=SKIPPED")
    print("DRY_RUN_ROLLOUT=SKIPPED")

    synthetic_predicted = np.zeros((CHUNK_SIZE, ACTION_DIM), dtype=np.float32)
    emit_metrics(
        synthetic_predicted,
        np.asarray(sample["recorded"], dtype=np.float32),
        include_steps=False,
        synthetic=True,
    )
    print(
        "DRY_RUN_PREDICTED_MAGNITUDE_PRINT=PASS "
        "(PREDICTED_ARM_L2 and PREDICTED_ARM_L2_CHUNK_MEAN emitted)"
    )
    del torch
    return 0


def run_adjudication(model_path: Path, tokenizer_path: Path) -> int:
    import torch
    from lerobot.utils.constants import ACTION

    watchdog_stop, watchdog_thread = start_watchdog()
    try:
        return _run_adjudication_with_watchdog(
            model_path,
            tokenizer_path,
            torch,
            ACTION,
        )
    finally:
        watchdog_stop.set()
        watchdog_thread.join(timeout=1.0)


def _run_adjudication_with_watchdog(
    model_path: Path,
    tokenizer_path: Path,
    torch: Any,
    action_key: str,
) -> int:
    torch.manual_seed(0)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(0)
    heartbeat("before_sample_load")
    harness = load_harness()
    probe = load_probe()
    sample = probe.load_episode()
    validate_sample(sample)
    print(f"EPISODE={EPISODE} POSITION={POSITION} VIDEO_INDEX={VIDEO_INDEX}", flush=True)
    print(f"INSTRUCTION={sample['instruction']}", flush=True)
    heartbeat("before_harness_load_policy")
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        harness.load_policy(harness_args(model_path, tokenizer_path))
    )
    policy.eval()
    heartbeat("after_harness_load_policy")

    batch = make_observation_and_batch(
        harness,
        probe,
        sample,
        env_preprocessor,
        preprocessor,
        policy.config,
    )
    policy.reset()
    n_steps = int(
        getattr(
            policy.config,
            "n_action_steps",
            getattr(policy.config, "chunk_size", CHUNK_SIZE),
        )
    )
    if n_steps != CHUNK_SIZE:
        raise RuntimeError(
            f"official policy action horizon is {n_steps}, expected {CHUNK_SIZE}"
        )

    predicted: list[np.ndarray] = []
    with torch.inference_mode():
        for index in range(n_steps):
            action = policy.select_action(batch)
            action = postprocessor(action)
            action = env_postprocessor({action_key: action})[action_key]
            array = action.detach().float().cpu().numpy().reshape(-1)
            if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
                raise RuntimeError(f"invalid predicted action at step {index}: {array.shape}")
            predicted.append(array)
            if index == 0 or (index + 1) % 10 == 0:
                heartbeat(f"after_select_action_{index + 1}")

    predicted_array = np.stack(predicted, axis=0)
    print(
        "SERVING_PATH=robotwin_eval_harness.load_policy->"
        "PI05Policy.from_pretrained->harness_pre_post_processors",
        flush=True,
    )
    emit_metrics(
        predicted_array,
        np.asarray(sample["recorded"], dtype=np.float32),
        include_steps=True,
        synthetic=False,
    )
    heartbeat("complete")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model-path", default=str(CHECKPOINT))
    parser.add_argument("--tokenizer-path", default=str(TOKENIZER_PATH))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_path = Path(args.model_path)
    tokenizer_path = Path(args.tokenizer_path)
    if args.dry_run:
        return dry_run(model_path, tokenizer_path)
    return run_adjudication(model_path, tokenizer_path)


if __name__ == "__main__":
    raise SystemExit(main())
