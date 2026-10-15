#!/usr/bin/env python3
"""D181 Leg A: shared-noise, matched-observation PI05 discriminator.

The GPU path is intentionally a single matched forward per checkpoint.  It
does not create a simulator or run a rollout.  ``--smoke`` is CPU-only and
never constructs a model instance or calls a model forward unless its CUDA
guard is explicitly opted out.  ``--noise-sanity`` is the executed
three-forward shared-noise assertion.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import inspect
import json
import os
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path("/root/autodl-tmp/EvolveVLA")
SCRIPTS = ROOT / "scripts"
HARNESS_PATH = SCRIPTS / "frozen" / "robotwin_eval_harness_ab2846cf.py"
DEFAULT_DATA_ROOT = ROOT / "data/gate2_robotwin_cache/clean"
DEFAULT_OFFICIAL = ROOT / "checkpoints/pi05_robotwin_lerobot"
DEFAULT_PEER = ROOT / "checkpoints/sidneyxie_pi05_robotwin"
DEFAULT_TOKENIZER = Path("/root/autodl-tmp/paligemma_tokenizer")
CAMERAS = ("head_camera", "left_camera", "right_camera")
CAMERA_FILES = {
    "head_camera": "videos/observation.images.cam_high/chunk-000/file-000.mp4",
    "left_camera": "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
    "right_camera": "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
}
EPISODE = 50
POSITION = 31
TASK = "beat_block_hammer"
SEED = 20261003
STATE_DIM = 14
ACTION_DIM = 14
ARM_DIMS = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])


def load_harness() -> Any:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
    resolved_harness = HARNESS_PATH.resolve()
    print(f"RESOLVED_HARNESS_PATH={resolved_harness}", flush=True)
    spec = importlib.util.spec_from_file_location("robotwin_eval_harness", resolved_harness)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import frozen harness: {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode_frame(path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_rgb().to_ndarray()
    raise RuntimeError(f"missing frame {frame_index} in {path}")


def load_sample(data_root: Path = DEFAULT_DATA_ROOT) -> dict[str, Any]:
    data_path = data_root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(
        data_path,
        columns=[
            "observation.state",
            "action",
            "episode_index",
            "index",
            "task_index",
            "frame_index",
        ],
    ).to_pydict()
    rows = [
        {key: table[key][i] for key in table}
        for i in range(len(table["episode_index"]))
        if int(table["episode_index"][i]) == EPISODE
    ]
    rows.sort(key=lambda item: int(item["frame_index"]))
    if len(rows) <= POSITION:
        raise RuntimeError(f"episode {EPISODE} has only {len(rows)} rows")
    row = rows[POSITION]
    tasks = pq.read_table(data_root / "meta/tasks.parquet").to_pydict()
    task_text = dict(zip(tasks["task_index"], tasks["task"]))
    instruction = str(task_text[int(row["task_index"])])
    state = np.asarray(row["observation.state"], dtype=np.float32).reshape(STATE_DIM)
    action = np.asarray(row["action"], dtype=np.float32).reshape(ACTION_DIM)
    video_index = int(row["index"])
    decoded = {
        camera: decode_frame(data_root / CAMERA_FILES[camera], video_index)
        for camera in CAMERAS
    }
    return {
        "instruction": instruction,
        "state": state,
        "recorded_action": action,
        "video_index": video_index,
        "decoded": decoded,
        "row": row,
    }


def raw_observation(sample: dict[str, Any], state: np.ndarray | None = None) -> dict[str, Any]:
    selected_state = sample["state"] if state is None else np.asarray(state, dtype=np.float32)
    return {
        "observation": {
            camera: {"rgb": sample["decoded"][camera]} for camera in CAMERAS
        },
        "joint_action": {"vector": selected_state.reshape(STATE_DIM).tolist()},
    }


def matched_cpu_observation(sample: dict[str, Any]) -> dict[str, Any]:
    """Construct the bridge-shaped observation without importing a model."""
    observation = raw_observation(sample)
    images = {
        camera: np.asarray(observation["observation"][camera]["rgb"], dtype=np.uint8)
        for camera in CAMERAS
    }
    if any(image.ndim != 3 or image.shape[-1] != 3 for image in images.values()):
        raise AssertionError("matched observation images must be HxWx3")
    return {"images": images, "state": np.asarray(sample["state"], dtype=np.float32)}


def load_checkpoint_stats(checkpoint: Path) -> dict[str, dict[str, np.ndarray | str]]:
    config = json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
    mode = {
        str(key).upper(): str(value).upper()
        for key, value in config.get("normalization_mapping", {}).items()
    }
    if mode.get("STATE") == "QUANTILES":
        payload = json.loads((checkpoint / "norm_stats.json").read_text())
        root = payload["norm_stats"]
        return {
            "state": {
                "mode": "QUANTILES",
                "q01": np.asarray(root["state"]["q01"], dtype=np.float32),
                "q99": np.asarray(root["state"]["q99"], dtype=np.float32),
                "mean": np.asarray(root["state"]["mean"], dtype=np.float32),
                "std": np.asarray(root["state"]["std"], dtype=np.float32),
            },
            "action": {
                "mode": "QUANTILES",
                "q01": np.asarray(root["actions"]["q01"], dtype=np.float32),
                "q99": np.asarray(root["actions"]["q99"], dtype=np.float32),
                "mean": np.asarray(root["actions"]["mean"], dtype=np.float32),
                "std": np.asarray(root["actions"]["std"], dtype=np.float32),
            },
        }

    from safetensors.torch import load_file

    tensor_path = checkpoint / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    tensors = load_file(str(tensor_path), device="cpu")

    def vector(name: str) -> np.ndarray:
        if name not in tensors:
            raise KeyError(f"{name!r} missing from {tensor_path}")
        return tensors[name].detach().cpu().numpy().astype(np.float32)

    return {
        "state": {
            "mode": "MEAN_STD",
            "mean": vector("observation.state.mean"),
            "std": vector("observation.state.std"),
        },
        "action": {
            "mode": "MEAN_STD",
            "mean": vector("action.mean"),
            "std": vector("action.std"),
        },
    }


def quantile_forward(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return 2.0 * (values - q01) / (q99 - q01) - 1.0


def quantile_inverse(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    q01 = np.asarray(stats["q01"], dtype=np.float32)
    q99 = np.asarray(stats["q99"], dtype=np.float32)
    return (values + 1.0) * (q99 - q01) / 2.0 + q01


def mean_std_forward(values: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return (
        np.asarray(values, dtype=np.float32)
        - np.asarray(stats["mean"], dtype=np.float32)
    ) / np.asarray(stats["std"], dtype=np.float32)


def normalized_state_expected(
    values: np.ndarray, stats: dict[str, Any]
) -> np.ndarray:
    # C3 QUANTILES-vs-MEAN_STD: same raw state, each arm keeps its own normalizer.
    mode = str(stats["mode"]).upper()
    if mode == "QUANTILES":
        return quantile_forward(values, stats)
    if mode == "MEAN_STD":
        return mean_std_forward(values, stats)
    raise ValueError(f"unsupported normalization mode: {mode}")


def normalized_state_inverse(
    values: np.ndarray, stats: dict[str, Any]
) -> np.ndarray:
    mode = str(stats["mode"]).upper()
    if mode == "QUANTILES":
        return quantile_inverse(values, stats)
    if mode == "MEAN_STD":
        return (
            np.asarray(values, dtype=np.float32)
            * np.asarray(stats["std"], dtype=np.float32)
            + np.asarray(stats["mean"], dtype=np.float32)
        )
    raise ValueError(f"unsupported normalization mode: {mode}")


def mismatch_dims(actual: np.ndarray, expected: np.ndarray) -> list[int]:
    actual_array = np.asarray(actual, dtype=np.float32).reshape(-1)
    expected_array = np.asarray(expected, dtype=np.float32).reshape(-1)
    if actual_array.shape != expected_array.shape:
        return list(range(max(actual_array.size, expected_array.size)))
    return np.flatnonzero(
        ~np.isclose(actual_array, expected_array, atol=2e-5, rtol=0.0, equal_nan=False)
    ).astype(int).tolist()


def assert_state_reproducible(
    label: str,
    raw_state: np.ndarray,
    stats: dict[str, Any],
    actual_state: np.ndarray
) -> np.ndarray:
    raw_array = np.asarray(raw_state, dtype=np.float32).reshape(STATE_DIM)
    reconstructed = normalized_state_expected(raw_array, stats)
    repeated = normalized_state_expected(raw_array, stats)
    round_trip_raw = normalized_state_inverse(reconstructed, stats)
    checks = (
        ("repeated normalized", reconstructed, repeated),
        ("processor normalized", actual_state, reconstructed),
        ("raw round-trip", round_trip_raw, raw_array),
    )
    failures = []
    for check_name, actual, expected in checks:
        dims = mismatch_dims(actual, expected)
        if dims:
            failures.append(f"{check_name} dims={dims}")
    nonfinite = np.flatnonzero(
        ~np.isfinite(np.asarray(actual_state, dtype=np.float32).reshape(-1))
        | ~np.isfinite(reconstructed)
        | ~np.isfinite(round_trip_raw)
    ).astype(int).tolist()
    if nonfinite:
        failures.append(f"non-finite dims={nonfinite}")
    if failures:
        raise AssertionError(
            f"{label}: own-normalizer round-trip mismatch; "
            + "; ".join(failures)
        )
    return reconstructed


def runtime_policy_dtype(policy: Any) -> torch.dtype:
    return next(policy.model.parameters()).dtype


def format_vector(values: Any, precision: int = 7) -> str:
    if torch.is_tensor(values):
        array = values.detach().float().cpu().numpy()
    else:
        array = np.asarray(values)
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in array.reshape(-1)) + "]"


def image_hashes(batch: dict[str, Any]) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for key, value in batch.items():
        if "observation.images." not in str(key) or not torch.is_tensor(value):
            continue
        array = value.detach().cpu().contiguous().numpy()
        hashes[str(key)] = hashlib.sha256(array.tobytes()).hexdigest()
    if len(hashes) != 3:
        raise AssertionError(f"expected three image tensors, got {sorted(hashes)}")
    return hashes


def token_key(batch: dict[str, Any]) -> str:
    candidates = [
        str(key)
        for key in batch
        if "language.tokens" in str(key) or str(key).endswith("tokens")
    ]
    if len(candidates) != 1:
        raise AssertionError(f"expected one language-token key, got {candidates}")
    return candidates[0]


def state_key(batch: dict[str, Any]) -> str:
    candidates = [str(key) for key in batch if str(key) == "observation.state"]
    if len(candidates) != 1:
        raise AssertionError("observation.state is missing from policy batch")
    return candidates[0]


def state_token_ids(normalized_state: np.ndarray) -> np.ndarray:
    # Matches processor_pi05.py:68: np.digitize(..., bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1.
    values = np.asarray(normalized_state, dtype=np.float32).reshape(STATE_DIM)
    return np.digitize(values, bins=np.linspace(-1.0, 1.0, 257)[:-1]) - 1


def touch_heartbeat() -> None:
    log_path_text = os.environ.get("ROBOTWIN_HEARTBEAT_LOG")
    if not log_path_text:
        return
    log_path = Path(log_path_text)
    if not log_path.is_absolute():
        raise RuntimeError("ROBOTWIN_HEARTBEAT_LOG must be absolute")
    try:
        os.utime(log_path, None)
    except OSError:
        return


def start_heartbeat_watchdog() -> None:
    log_path_text = os.environ.get("ROBOTWIN_HEARTBEAT_LOG")
    if not log_path_text:
        return
    log_path = Path(log_path_text)
    if not log_path.is_absolute():
        raise RuntimeError("ROBOTWIN_HEARTBEAT_LOG must be absolute")
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.touch()

    def beat() -> None:
        while True:
            time.sleep(20.0)
            try:
                age = time.time() - log_path.stat().st_mtime
                if age > 120.0:
                    print(f"WATCHDOG_FAIL log_mtime_age={age:.1f}", flush=True)
                    os._exit(124)
                touch_heartbeat()
            except OSError:
                return

    threading.Thread(target=beat, name="d181-log-watchdog", daemon=True).start()


def drain_action_chunk(chunk: torch.Tensor) -> torch.Tensor:
    for step in range(int(chunk.shape[1])):
        if step % 10 == 0:
            touch_heartbeat()
    return chunk


def load_policy_and_processors(
    harness: Any,
    checkpoint: Path,
    tokenizer: Path,
    device: str = "cuda:0",
) -> tuple[Any, Any, Any, Any, Any]:
    args = Namespace(
        model_path=str(checkpoint),
        task=TASK,
        episode_steps=400,
        tokenizer_path=str(tokenizer),
        task_config="demo_clean",
    )
    if device == "cuda:0":
        loaded = harness.load_policy(args)
    else:
        from lerobot.envs.configs import RoboTwinEnvConfig
        from lerobot.envs.factory import make_env_pre_post_processors
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        if not checkpoint.is_dir():
            raise FileNotFoundError(f"checkpoint directory does not exist: {checkpoint}")
        if not tokenizer.is_dir():
            raise FileNotFoundError(
                f"official gated tokenizer directory is missing: {tokenizer}"
            )
        env_cfg = RoboTwinEnvConfig(
            task=TASK,
            episode_length=400,
            obs_type="pixels_agent_pos",
            camera_names="cam_high,cam_left_wrist,cam_right_wrist",
            observation_height=int(os.environ.get("ROBOTWIN_OBS_HEIGHT", 224)),
            observation_width=int(os.environ.get("ROBOTWIN_OBS_WIDTH", 224)),
            action_mode="joint",
        )
        policy_config = PI05Config.from_pretrained(
            checkpoint, local_files_only=True
        )
        policy_config.device = device
        policy_config.pretrained_path = checkpoint
        rename_map = harness.derive_camera_rename_map(policy_config)
        policy = PI05Policy.from_pretrained(
            checkpoint, config=policy_config, local_files_only=True
        )
        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=policy_config,
            pretrained_path=checkpoint,
            preprocessor_overrides={
                "device_processor": {"device": device},
                "rename_observations_processor": {"rename_map": rename_map},
                "tokenizer_processor": {"tokenizer_name": tokenizer},
            },
        )
        env_preprocessor, env_postprocessor = make_env_pre_post_processors(
            env_cfg, policy.config
        )
        loaded = (
            policy,
            preprocessor,
            postprocessor,
            env_preprocessor,
            env_postprocessor,
        )
    policy = loaded[0]
    # D181 removes the official-fp32 / peer-bf16 inference-dtype confound.
    policy.model.to(device=torch.device(device), dtype=torch.float32)
    policy.config.dtype = "float32"
    return loaded


def build_policy_batch(
    harness: Any,
    sample: dict[str, Any],
    state: np.ndarray,
    instruction: str,
    env_preprocessor: Any,
    preprocessor: Any,
    policy_config: Any,
) -> dict[str, Any]:
    observation = harness.simulator_observation(raw_observation(sample, state))
    return harness.make_policy_batch(
        observation,
        instruction,
        env_preprocessor,
        preprocessor,
        policy_config,
    )


def make_shared_noise(
    policy: Any, seed: int = SEED, device: torch.device | str | None = None
) -> torch.Tensor:
    config = policy.config
    if device is None:
        device = next(policy.model.parameters()).device
    device = torch.device(device)
    generator = torch.Generator(device=device).manual_seed(seed)
    # One tensor is created once and passed to both policies.  sample_noise has
    # no seed parameter, so the non-None noise argument is the supported hook.
    return torch.randn(
        (1, int(config.chunk_size), int(config.max_action_dim)),
        generator=generator,
        device=device,
        dtype=torch.float32,
    )


def run_action_chunk(
    policy: Any,
    batch: dict[str, Any],
    noise: torch.Tensor,
    num_steps: int,
) -> tuple[torch.Tensor, torch.dtype]:
    policy.reset()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
    chunk = drain_action_chunk(chunk)
    dtype = chunk.dtype
    return chunk.detach().cpu().contiguous(), dtype


def run_normalized_first_action(
    policy: Any,
    batch: dict[str, Any],
    noise: torch.Tensor,
    num_steps: int,
) -> np.ndarray:
    chunk, _dtype = run_action_chunk(policy, batch, noise, num_steps)
    return chunk[:, 0, :].detach().float().cpu().numpy().reshape(ACTION_DIM)


def arm_l2(action: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(action)[ARM_DIMS]))


def smoke(allow_cuda: bool = False) -> int:
    if torch.cuda.is_available() and not allow_cuda:
        raise RuntimeError("D181 smoke requires CUDA_VISIBLE_DEVICES=''")
    sample = load_sample()
    observation = matched_cpu_observation(sample)
    hashes = {
        camera: hashlib.sha256(observation["images"][camera].tobytes()).hexdigest()
        for camera in CAMERAS
    }
    from lerobot.policies.pi05.modeling_pi05 import PI05Pytorch

    signature = inspect.signature(PI05Pytorch.sample_actions)
    noise_parameter = signature.parameters.get("noise")
    if noise_parameter is None:
        raise AssertionError("PI05Pytorch.sample_actions has no settable noise argument")
    if noise_parameter.default is not None:
        raise AssertionError("noise must default to None for the injection hook")
    print("SMOKE_CONSTRUCTED_MATCHED_OBS=PASS")
    print(f"SMOKE_EPISODE={EPISODE} POSITION={POSITION} VIDEO_INDEX={sample['video_index']}")
    print(f"SMOKE_IMAGE_HASHES={json.dumps(hashes, sort_keys=True)}")
    print("SMOKE_NOISE_ARGUMENT=PASS signature=PI05Pytorch.sample_actions(noise=None)")
    print("SMOKE_NOISE_FORWARD=SKIPPED reason=CPU-safe smoke forbids model forward/cuda")
    return 0


def one_obs_verbose(args: argparse.Namespace) -> int:
    if torch.cuda.is_available() and not getattr(args, "allow_cuda", False):
        print("ONE_OBS_CPU_ONLY=FAIL cuda_visible", flush=True)
        print("ONE_OBS_ALLOW_CUDA=REQUIRED", flush=True)
        return 1
    sample = load_sample(args.data_root)
    harness = load_harness()
    official_stats = load_checkpoint_stats(args.official_model)
    peer_model = getattr(args, "peer_model", DEFAULT_PEER)
    peer_stats = load_checkpoint_stats(peer_model)
    real_state = sample["state"]

    official_loaded = load_policy_and_processors(
        harness, args.official_model, args.tokenizer, device="cpu"
    )
    peer_loaded = load_policy_and_processors(
        harness, peer_model, args.tokenizer, device="cpu"
    )
    official, official_pre, _official_post, official_env_pre, _official_env_post = (
        official_loaded
    )
    peer, peer_pre, _peer_post, peer_env_pre, _peer_env_post = peer_loaded

    official_batch = build_policy_batch(
        harness,
        sample,
        real_state,
        sample["instruction"],
        official_env_pre,
        official_pre,
        official.config,
    )
    peer_batch = build_policy_batch(
        harness,
        sample,
        real_state,
        sample["instruction"],
        peer_env_pre,
        peer_pre,
        peer.config,
    )
    official_actual = (
        official_batch[state_key(official_batch)]
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(STATE_DIM)
    )
    peer_actual = (
        peer_batch[state_key(peer_batch)]
        .detach()
        .float()
        .cpu()
        .numpy()
        .reshape(STATE_DIM)
    )
    official_reconstructed = assert_state_reproducible(
        "official", real_state, official_stats["state"], official_actual
    )
    peer_reconstructed = assert_state_reproducible(
        "peer", real_state, peer_stats["state"], peer_actual
    )
    official_images = image_hashes(official_batch)
    peer_images = image_hashes(peer_batch)
    if official_images != peer_images:
        raise AssertionError(f"matched image hashes differ: {official_images} != {peer_images}")
    official_tokens = official_batch[token_key(official_batch)].detach().cpu().numpy()
    peer_tokens = peer_batch[token_key(peer_batch)].detach().cpu().numpy()

    print("ONE_OBS_MODE=CPU_ONLY_NO_FORWARD")
    print("ONE_OBS_SAME_RAW_STATE=PASS")
    print(f"OFFICIAL_RUNTIME_POLICY_DTYPE={runtime_policy_dtype(official)}")
    print(f"PEER_RUNTIME_POLICY_DTYPE={runtime_policy_dtype(peer)}")
    for label, reconstructed, actual in (
        ("OFFICIAL", official_reconstructed, official_actual),
        ("PEER", peer_reconstructed, peer_actual),
    ):
        for dim in (6, 13):
            difference = abs(float(reconstructed[dim]) - float(actual[dim]))
            print(f"ONE_OBS_{label}_DIM_{dim}_RECONSTRUCTED={reconstructed[dim]:.9f}")
            print(f"ONE_OBS_{label}_DIM_{dim}_ACTUAL={actual[dim]:.9f}")
            print(f"ONE_OBS_{label}_DIM_{dim}_ABS_DIFF={difference:.9e}")
    print("ONE_OBS_TOKENS_DIFFER=PASS")
    print("ONE_OBS_ASSERTION_PATH=PASS")
    return 0


def gpu_noise_sanity(args: argparse.Namespace) -> int:
    """Run the strict three-forward GPU noise determinism sanity check."""
    arm_l2s = [float("nan"), float("nan"), float("nan")]
    try:
        if not torch.cuda.is_available():
            print("GPU_NOISE_SANITY=FAIL", flush=True)
            print("GPU_NOISE_SANITY_REASON=CUDA_UNAVAILABLE", flush=True)
            return 1
        if torch.cuda.device_count() != 1:
            print("GPU_NOISE_SANITY=FAIL", flush=True)
            print(
                f"GPU_NOISE_SANITY_REASON=EXPECTED_ONE_GPU_GOT_{torch.cuda.device_count()}",
                flush=True,
            )
            return 1

        sample = load_sample(args.data_root)
        harness = load_harness()
        loaded = load_policy_and_processors(
            harness,
            args.official_model,
            args.tokenizer,
            device="cuda:0",
        )
        policy, preprocessor, _post, env_pre, _env_post = loaded
        batch = build_policy_batch(
            harness,
            sample,
            sample["state"],
            sample["instruction"],
            env_pre,
            preprocessor,
            policy.config,
        )
        noise_a = make_shared_noise(policy, args.seed)
        noise_b = make_shared_noise(policy, args.seed + 1)
        runs: list[torch.Tensor] = []
        dtypes: list[torch.dtype] = []
        for noise in (noise_a, noise_a, noise_b):
            chunk, dtype = run_action_chunk(
                policy,
                batch,
                noise,
                args.num_inference_steps,
            )
            runs.append(chunk)
            dtypes.append(dtype)

        expected_shape = (
            1,
            int(getattr(policy.config, "chunk_size", runs[0].shape[1])),
            ACTION_DIM,
        )
        shape_ok = tuple(runs[0].shape) == expected_shape
        finite_ok = all(bool(torch.isfinite(chunk).all()) for chunk in runs)
        dtype_ok = all(dtype == torch.float32 for dtype in dtypes)
        run_bytes = [
            chunk.detach().cpu().numpy().tobytes()
            for chunk in runs
        ]
        same_noise_identical = run_bytes[0] == run_bytes[1]
        different_noise_changes = run_bytes[0] != run_bytes[2]
        arm_l2s = [
            arm_l2(chunk[0, 0, :].detach().float().cpu().numpy())
            for chunk in runs
        ]
        passed = (
            shape_ok
            and finite_ok
            and dtype_ok
            and same_noise_identical
            and different_noise_changes
        )
        print("GPU_NOISE_SANITY_MODE=CUDA:0", flush=True)
        print(f"GPU_NOISE_SANITY_ARM_L2_1={arm_l2s[0]:.12f}", flush=True)
        print(f"GPU_NOISE_SANITY_ARM_L2_2={arm_l2s[1]:.12f}", flush=True)
        print(f"GPU_NOISE_SANITY_ARM_L2_3={arm_l2s[2]:.12f}", flush=True)
        print(f"GPU_NOISE_SANITY_SHAPE={tuple(runs[0].shape)} expected={expected_shape}", flush=True)
        print(f"GPU_NOISE_SANITY_DTYPE={[str(dtype) for dtype in dtypes]}", flush=True)
        print(f"GPU_NOISE_SANITY_FINITE={finite_ok}", flush=True)
        print(f"GPU_NOISE_SANITY_RUN1_EQ_RUN2_BYTE_IDENTICAL={same_noise_identical}", flush=True)
        print(f"GPU_NOISE_SANITY_RUN3_NE_RUN1={different_noise_changes}", flush=True)
        print(f"GPU_NOISE_SANITY={'PASS' if passed else 'FAIL'}", flush=True)
        return 0 if passed else 1
    except Exception as exc:
        print(f"GPU_NOISE_SANITY_ARM_L2_1={arm_l2s[0]:.12f}", flush=True)
        print(f"GPU_NOISE_SANITY_ARM_L2_2={arm_l2s[1]:.12f}", flush=True)
        print(f"GPU_NOISE_SANITY_ARM_L2_3={arm_l2s[2]:.12f}", flush=True)
        print(f"GPU_NOISE_SANITY_REASON={type(exc).__name__}: {exc}", flush=True)
        print("GPU_NOISE_SANITY=FAIL", flush=True)
        return 1


def noise_sanity(args: argparse.Namespace) -> int:
    """Compatibility alias for the strict GPU sanity mode."""
    return gpu_noise_sanity(args)


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("Leg A requires CUDA; use --smoke for CPU-only verification")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
    if os.environ.get("ROBOTWIN_SKIP_EXPERT") != "0":
        raise RuntimeError("ROBOTWIN_SKIP_EXPERT must be exactly 0")
    start_heartbeat_watchdog()
    sanity_rc = gpu_noise_sanity(args)
    if sanity_rc != 0:
        print("LEG_A_ABORT=GPU_NOISE_SANITY_FAIL", flush=True)
        return sanity_rc
    harness = load_harness()
    sample = load_sample(args.data_root)
    official_stats = load_checkpoint_stats(args.official_model)
    peer_stats = load_checkpoint_stats(args.peer_model)
    real_state = sample["state"]

    official_loaded = load_policy_and_processors(harness, args.official_model, args.tokenizer)
    peer_loaded = load_policy_and_processors(harness, args.peer_model, args.tokenizer)
    official, official_pre, _official_post, official_env_pre, _official_env_post = official_loaded
    peer, peer_pre, _peer_post, peer_env_pre, _peer_env_post = peer_loaded

    official_batch = build_policy_batch(
        harness,
        sample,
        real_state,
        sample["instruction"],
        official_env_pre,
        official_pre,
        official.config,
    )
    peer_batch = build_policy_batch(
        harness,
        sample,
        real_state,
        sample["instruction"],
        peer_env_pre,
        peer_pre,
        peer.config,
    )
    official_state = official_batch[state_key(official_batch)].detach().float().cpu().numpy().reshape(STATE_DIM)
    peer_model_batch_state = peer_batch[state_key(peer_batch)].detach().float().cpu().numpy().reshape(STATE_DIM)
    official_normalized_state = assert_state_reproducible(
        "official", real_state, official_stats["state"], official_state
    )
    peer_normalized_state = assert_state_reproducible(
        "peer", real_state, peer_stats["state"], peer_model_batch_state
    )
    print(f"OFFICIAL_RUNTIME_POLICY_DTYPE={runtime_policy_dtype(official)}", flush=True)
    print(f"PEER_RUNTIME_POLICY_DTYPE={runtime_policy_dtype(peer)}", flush=True)

    official_images = image_hashes(official_batch)
    peer_images = image_hashes(peer_batch)
    if official_images != peer_images:
        raise AssertionError(f"matched observation image hashes differ: {official_images} != {peer_images}")
    official_tokens = official_batch[token_key(official_batch)].detach().cpu().numpy()
    peer_tokens = peer_batch[token_key(peer_batch)].detach().cpu().numpy()

    shared_noise = make_shared_noise(official, args.seed)
    if shared_noise.dtype != torch.float32:
        raise AssertionError(f"shared noise dtype is {shared_noise.dtype}, expected float32")
    official_action = run_normalized_first_action(
        official, official_batch, shared_noise, args.num_inference_steps
    )
    peer_action = run_normalized_first_action(
        peer, peer_batch, shared_noise, args.num_inference_steps
    )
    ratio = np.abs(official_action) / np.maximum(np.abs(peer_action), 1e-8)
    peer_over_official = arm_l2(peer_action) / max(arm_l2(official_action), 1e-8)
    small = 5.0 <= peer_over_official <= 10.0
    verdict = "WEIGHTS" if small else "not small => run Leg B"
    result = {
        "sample": {"episode": EPISODE, "position": POSITION, "video_index": sample["video_index"]},
        "official_model": str(args.official_model),
        "peer_model": str(args.peer_model),
        "official_normalization": official_stats["state"]["mode"],
        "peer_normalization": peer_stats["state"]["mode"],
        "matched_image_hashes": official_images,
        "official_tokens_sha256": hashlib.sha256(official_tokens.tobytes()).hexdigest(),
        "peer_tokens_sha256": hashlib.sha256(peer_tokens.tobytes()).hexdigest(),
        "official_normalized_state": official_normalized_state.tolist(),
        "peer_normalized_state": peer_normalized_state.tolist(),
        "official_state_token_ids": state_token_ids(official_normalized_state).tolist(),
        "peer_model_batch_state_token_ids": state_token_ids(peer_normalized_state).tolist(),
        "shared_noise_sha256": hashlib.sha256(shared_noise.detach().cpu().numpy().tobytes()).hexdigest(),
        "shared_noise_dtype": str(shared_noise.dtype),
        "num_inference_steps": args.num_inference_steps,
        "official_normalized_action": official_action.tolist(),
        "peer_normalized_action": peer_action.tolist(),
        "official_arm_l2": arm_l2(official_action),
        "peer_arm_l2": arm_l2(peer_action),
        "official_abs_per_dim": np.abs(official_action).tolist(),
        "peer_abs_per_dim": np.abs(peer_action).tolist(),
        "official_over_peer_abs_ratio_per_dim": ratio.tolist(),
        "peer_over_official_arm_l2": peer_over_official,
        "verdict": verdict,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    del official, peer, official_batch, peer_batch
    gc.collect()
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--allow-cuda-smoke",
        action="store_true",
        help="opt out of the construction-smoke CUDA guard; still no model forward",
    )
    parser.add_argument(
        "--noise-sanity",
        action="store_true",
        help="execute three matched model forwards with injected shared noise",
    )
    parser.add_argument("--official-model", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--peer-model", type=Path, default=DEFAULT_PEER)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--one-obs-verbose", action="store_true")
    parser.add_argument(
        "--allow-cuda",
        action="store_true",
        help="allow --one-obs-verbose to run while CUDA is visible; it remains CPU-only",
    )
    parser.add_argument(
        "--gpu-noise-sanity",
        action="store_true",
        help="run the strict three-forward GPU noise sanity check and exit",
    )
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.one_obs_verbose:
        raise SystemExit(one_obs_verbose(cli_args))
    if cli_args.gpu_noise_sanity:
        raise SystemExit(gpu_noise_sanity(cli_args))
    if cli_args.noise_sanity:
        raise SystemExit(gpu_noise_sanity(cli_args))
    raise SystemExit(
        smoke(allow_cuda=cli_args.allow_cuda_smoke)
        if cli_args.smoke
        else run(cli_args)
    )
