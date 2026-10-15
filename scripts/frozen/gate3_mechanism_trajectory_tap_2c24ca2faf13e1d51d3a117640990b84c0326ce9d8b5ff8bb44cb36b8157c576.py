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
import types
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



class _TrajectoryTrace:
    def __init__(self, label: str):
        self.label = label
        self.x_inputs: list[torch.Tensor] = []
        self.velocities: list[torch.Tensor] = []
        self.times: list[torch.Tensor] = []
        self.suffix_out: list[torch.Tensor] = []
        self.prefix_emb: torch.Tensor | None = None


def _flat_cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    left = left.detach().float().reshape(-1)
    right = right.detach().float().reshape(-1)
    denom = torch.linalg.vector_norm(left) * torch.linalg.vector_norm(right)
    if float(denom) == 0.0:
        return float("nan")
    return float(torch.dot(left, right) / denom)


def _tensor_list(value: torch.Tensor) -> list:
    return value.detach().float().cpu().numpy().tolist()


def _attach_trace(policy: Any, trace: _TrajectoryTrace):
    model = policy.model
    original_embed_prefix = model.embed_prefix
    original_denoise_step = model.denoise_step

    def traced_embed_prefix(_self, *args, **kwargs):
        output = original_embed_prefix(*args, **kwargs)
        trace.prefix_emb = output[0].detach().float().cpu()
        return output

    def traced_denoise_step(_self, *args, **kwargs):
        x_t = kwargs.get("x_t")
        timestep = kwargs.get("timestep")
        if x_t is None:
            if len(args) < 3:
                raise AssertionError("denoise_step trace could not resolve x_t")
            x_t = args[2]
        if timestep is None:
            if len(args) < 4:
                raise AssertionError("denoise_step trace could not resolve timestep")
            timestep = args[3]
        trace.x_inputs.append(x_t.detach().float().cpu())
        trace.times.append(timestep.detach().float().cpu())
        velocity = original_denoise_step(*args, **kwargs)
        trace.velocities.append(velocity.detach().float().cpu())
        return velocity

    def capture_suffix_out(_module, inputs):
        if not inputs:
            raise AssertionError("action_out_proj received no suffix_out input")
        trace.suffix_out.append(inputs[0].detach().float().cpu())

    model.embed_prefix = types.MethodType(traced_embed_prefix, model)
    model.denoise_step = types.MethodType(traced_denoise_step, model)
    hook = model.action_out_proj.register_forward_pre_hook(capture_suffix_out)

    def cleanup():
        hook.remove()
        model.embed_prefix = original_embed_prefix
        model.denoise_step = original_denoise_step

    return cleanup


def _trace_record(
    trace: _TrajectoryTrace,
    noise: torch.Tensor,
    num_steps: int,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    if len(trace.velocities) != num_steps:
        raise AssertionError(
            f"{trace.label}: captured {len(trace.velocities)} velocity steps, expected {num_steps}"
        )
    if len(trace.suffix_out) != num_steps:
        raise AssertionError(
            f"{trace.label}: captured {len(trace.suffix_out)} suffix_out steps, expected {num_steps}"
        )
    if trace.prefix_emb is None:
        raise AssertionError(f"{trace.label}: prefix embedding was not captured")

    velocities = torch.stack(trace.velocities, dim=0)
    suffix_out = torch.stack(trace.suffix_out, dim=0)
    dt = -1.0 / num_steps
    x_values = [noise.detach().float().cpu()]
    deltas = []
    for velocity in velocities:
        delta = velocity * dt
        deltas.append(delta)
        x_values.append(x_values[-1] + delta)
    trajectory = torch.stack(x_values, dim=0)
    delta_tensor = torch.stack(deltas, dim=0)
    if trajectory.shape[0] != num_steps + 1:
        raise AssertionError(f"{trace.label}: trajectory has wrong step count")

    finite = bool(
        torch.isfinite(velocities).all()
        and torch.isfinite(suffix_out).all()
        and torch.isfinite(trajectory).all()
        and torch.isfinite(trace.prefix_emb).all()
    )
    if not finite:
        raise FloatingPointError(f"{trace.label}: non-finite tap")

    per_step_aggregate = torch.linalg.vector_norm(
        velocities.reshape(num_steps, -1), dim=1
    )
    per_dim_l2 = torch.linalg.vector_norm(velocities[:, 0], dim=1)
    per_dim_mean_abs = velocities[:, 0].abs().mean(dim=1)
    final = trajectory[-1, 0]
    initial = trajectory[0, 0]
    full_chunk_arm_l2 = torch.linalg.vector_norm(final[:, ARM_DIMS], dim=1)
    initial_arm_l2 = torch.linalg.vector_norm(initial[:, ARM_DIMS], dim=1)
    prefix_flat = trace.prefix_emb.reshape(-1)
    record = {
        "condition": trace.label,
        "finite": finite,
        "num_steps_captured": int(len(trace.velocities)),
        "dt": dt,
        "space": "model-normalized pre-postprocessor x/v space",
        "tap_shapes": {
            "velocity": list(velocities.shape),
            "velocity_10x_per_dim_summary": [num_steps, int(velocities.shape[-1])],
            "trajectory_x0_to_x10": list(trajectory.shape),
            "delta_dt_v": list(delta_tensor.shape),
            "suffix_out_pre_action_head": list(suffix_out.shape),
            "prefix_embed": list(trace.prefix_emb.shape),
        },
        "tap1_velocity": {
            "l2_over_full_50x32_per_step": _tensor_list(per_step_aggregate),
            "l2_per_dim_over_50_chunks_per_step": _tensor_list(per_dim_l2),
            "mean_abs_per_dim_over_50_chunks_per_step": _tensor_list(per_dim_mean_abs),
        },
        "tap1_trajectory_x0_to_x10": _tensor_list(trajectory),
        "tap1_delta_dt_v_per_step": _tensor_list(delta_tensor),
        "tap3_suffix_out_pre_action_head_per_step": _tensor_list(suffix_out),
        "magnitude_full_chunk": {
            "peak_abs_over_50x32": float(final.abs().max()),
            "mean_abs_over_50x32": float(final.abs().mean()),
            "peak_abs_per_dim_over_50_chunks": _tensor_list(final.abs().amax(dim=0)),
            "mean_abs_per_dim_over_50_chunks": _tensor_list(final.abs().mean(dim=0)),
            "arm_l2_peak_over_50_chunks": float(full_chunk_arm_l2.max()),
            "arm_l2_mean_over_50_chunks": float(full_chunk_arm_l2.mean()),
        },
        "scaling_read_pos0_only": {
            "noise_arm_l2": float(initial_arm_l2[0]),
            "x_final_arm_l2": float(full_chunk_arm_l2[0]),
            "x_final_over_noise": float(full_chunk_arm_l2[0] / max(float(initial_arm_l2[0]), 1e-12)),
            "interpretation": (
                "rotates"
                if float(full_chunk_arm_l2[0] / max(float(initial_arm_l2[0]), 1e-12)) >= 0.75
                else "collapses_toward_zero"
            ),
        },
        "tap4_prefix_embed_summary": {
            "norm": float(torch.linalg.vector_norm(prefix_flat)),
            "mean_abs": float(trace.prefix_emb.abs().mean()),
            "per_token_l2": _tensor_list(torch.linalg.vector_norm(trace.prefix_emb[0], dim=-1)),
        },
    }
    return record, velocities, trace.prefix_emb


def _run_traced_condition(
    policy: Any,
    batch: dict[str, Any],
    noise: torch.Tensor,
    num_steps: int,
    label: str,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    trace = _TrajectoryTrace(label)
    cleanup = _attach_trace(policy, trace)
    try:
        policy.reset()
        with torch.inference_mode():
            policy.predict_action_chunk(batch, noise=noise, num_steps=num_steps)
    finally:
        cleanup()
    return _trace_record(trace, noise, num_steps)


def _replace_language_tokens(
    target_batch: dict[str, Any],
    source_batch: dict[str, Any],
) -> dict[str, Any]:
    output = dict(target_batch)
    target_token_name = token_key(target_batch)
    source_token_name = token_key(source_batch)
    if target_token_name != source_token_name:
        raise AssertionError("token key differs between arms")
    target_mask_candidates = [
        key for key in target_batch
        if "language.attention_mask" in str(key)
    ]
    source_mask_candidates = [
        key for key in source_batch
        if "language.attention_mask" in str(key)
    ]
    if len(target_mask_candidates) != 1 or len(source_mask_candidates) != 1:
        raise AssertionError("expected one language attention-mask key per arm")
    target_tokens = target_batch[target_token_name]
    source_tokens = source_batch[source_token_name]
    target_mask = target_batch[target_mask_candidates[0]]
    source_mask = source_batch[source_mask_candidates[0]]
    if target_tokens.shape != source_tokens.shape or target_mask.shape != source_mask.shape:
        raise AssertionError("matched-token control requires equal token tensor shapes")
    output[target_token_name] = source_tokens.clone()
    output[target_mask_candidates[0]] = source_mask.clone()
    return output


def _token_strings(tokenizer_path: Path, batch: dict[str, Any]) -> list[str]:
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        local_files_only=True,
    )
    ids = batch[token_key(batch)].detach().cpu()
    return tokenizer.batch_decode(ids, skip_special_tokens=False)


def _direction_report(
    official_velocity: torch.Tensor,
    peer_velocity: torch.Tensor,
) -> dict[str, Any]:
    if official_velocity.shape != peer_velocity.shape:
        raise AssertionError("official and peer velocity taps have different shapes")
    difference = official_velocity - peer_velocity
    step_cosine = [
        _flat_cosine(official_velocity[step], peer_velocity[step])
        for step in range(int(official_velocity.shape[0]))
    ]
    return {
        "tap2_step_cosine_official_peer": step_cosine,
        "tap2_primary_step0_cosine_identical_noise_obs_t": step_cosine[0],
        "tap2_delta_v_t_official_minus_peer_signed_mean_over_50_per_dim": _tensor_list(
            difference.mean(dim=1)
        ),
        "tap2_delta_v_t_official_minus_peer_mean_abs_over_50_per_dim": _tensor_list(
            difference.abs().mean(dim=1)
        ),
        "tap2_delta_v_t_l2_over_full_50x32_per_step": _tensor_list(
            torch.linalg.vector_norm(difference.reshape(difference.shape[0], -1), dim=1)
        ),
    }


def _control_metric(record: dict[str, Any], velocity: torch.Tensor) -> dict[str, Any]:
    step0 = velocity[0, 0]
    return {
        "condition": record["condition"],
        "step0_velocity_l2_over_full_50x32": float(
            torch.linalg.vector_norm(velocity[0].reshape(-1))
        ),
        "step0_velocity_arm_l2_pos0": arm_l2(step0.numpy()),
        "tap_shapes": record["tap_shapes"],
        "finite": record["finite"],
        "num_steps_captured": record["num_steps_captured"],
    }


def _prefix_compare(
    official_prefix: torch.Tensor,
    peer_prefix: torch.Tensor,
) -> dict[str, Any]:
    if official_prefix.shape != peer_prefix.shape:
        raise AssertionError("official and peer prefix embeddings have different shapes")
    return {
        "official_shape": list(official_prefix.shape),
        "peer_shape": list(peer_prefix.shape),
        "official_norm": float(torch.linalg.vector_norm(official_prefix)),
        "peer_norm": float(torch.linalg.vector_norm(peer_prefix)),
        "official_peer_cosine": _flat_cosine(official_prefix, peer_prefix),
    }


class _SmokeTraceModel(torch.nn.Module):
    def __init__(self, seed: int):
        super().__init__()
        torch.manual_seed(seed)
        self.action_out_proj = torch.nn.Linear(16, 32, bias=False)
        self.context_scale = torch.nn.Parameter(torch.tensor(0.25))
        self._prefix_context = torch.tensor(0.0)

    def embed_prefix(self, images, img_masks, tokens, masks):
        context = tokens.float().mean(dim=1, keepdim=True)
        prefix = torch.tanh(
            context[:, :, None].expand(-1, 8, 16)
            + torch.arange(16, dtype=torch.float32)[None, None, :]
        )
        self._prefix_context = prefix.mean()
        pad = torch.ones(prefix.shape[:2], dtype=torch.bool)
        att = torch.zeros_like(pad)
        return prefix, pad, att

    def denoise_step(self, prefix_pad_masks, past_key_values, x_t, timestep):
        suffix_out = x_t[..., :16] + self._prefix_context * self.context_scale
        suffix_out = suffix_out + timestep[:, None, None]
        return self.action_out_proj(suffix_out)

    def sample_actions(self, images, img_masks, tokens, masks, noise=None, num_steps=10, **kwargs):
        prefix_embs, prefix_pad_masks, _prefix_att_masks = self.embed_prefix(
            images, img_masks, tokens, masks
        )
        del prefix_embs
        del _prefix_att_masks
        x_t = noise
        dt = -1.0 / num_steps
        past_key_values = object()
        for step in range(num_steps):
            timestep = torch.tensor(
                1.0 + step * dt,
                dtype=torch.float32,
                device=x_t.device,
            ).expand(x_t.shape[0])
            v_t = self.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            )
            x_t = x_t + dt * v_t
        return x_t


class _SmokeTracePolicy:
    def __init__(self, seed: int):
        self.model = _SmokeTraceModel(seed)

    def reset(self):
        return None

    def predict_action_chunk(self, batch, noise=None, num_steps=10):
        return self.model.sample_actions(
            None,
            None,
            batch["tokens"],
            batch["mask"],
            noise=noise,
            num_steps=num_steps,
        )


def _smoke_fixture_batch(tokens: list[int]) -> dict[str, torch.Tensor]:
    return {
        "tokens": torch.tensor([tokens], dtype=torch.long),
        "mask": torch.ones(1, len(tokens), dtype=torch.bool),
    }


def _check_record_shapes(label: str, record: dict[str, Any]) -> None:
    shapes = record["tap_shapes"]
    expected = {
        "velocity": [10, 1, 50, 32],
        "trajectory_x0_to_x10": [11, 1, 50, 32],
        "delta_dt_v": [10, 1, 50, 32],
        "suffix_out_pre_action_head": [10, 1, 50, 16],
        "prefix_embed": [1, 8, 16],
    }
    for name, shape in expected.items():
        if shapes.get(name) != shape:
            raise AssertionError(f"{label}: {name} shape={shapes.get(name)} expected={shape}")
    if record["num_steps_captured"] != 10 or not record["finite"]:
        raise AssertionError(f"{label}: finite/step-count contract failed")


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


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
    frozen_path = Path(__file__).resolve()
    if not frozen_path.exists():
        raise FileNotFoundError(f"frozen file missing: {frozen_path}")
    sha_before = _sha256_file(frozen_path)
    match = re.search(r"gate3_mechanism_trajectory_tap_([0-9a-f]{64})\.py$", frozen_path.name)
    if match is None or match.group(1) != sha_before:
        raise AssertionError(
            f"frozen filename/content SHA mismatch before smoke: {frozen_path.name} {sha_before}"
        )
    if torch.cuda.is_available() and not allow_cuda:
        raise RuntimeError('D244 smoke requires CUDA_VISIBLE_DEVICES=""')

    torch.manual_seed(SEED)
    official = _SmokeTracePolicy(seed=17)
    peer = _SmokeTracePolicy(seed=23)
    official_batch = _smoke_fixture_batch([101, 102, 103, 104])
    peer_batch = _smoke_fixture_batch([201, 202, 203, 204])
    shared_noise = torch.randn((1, 50, 32), dtype=torch.float32)
    official_peer_tokens = dict(official_batch)
    official_peer_tokens["tokens"] = peer_batch["tokens"].clone()
    official_peer_tokens["mask"] = peer_batch["mask"].clone()
    peer_official_tokens = dict(peer_batch)
    peer_official_tokens["tokens"] = official_batch["tokens"].clone()
    peer_official_tokens["mask"] = official_batch["mask"].clone()

    conditions = [
        ("official_native", official, official_batch),
        ("peer_native", peer, peer_batch),
        ("official_peer_tokens", official, official_peer_tokens),
        ("peer_official_tokens", peer, peer_official_tokens),
    ]
    records = []
    for label, policy, batch in conditions:
        record, _velocity, _prefix = _run_traced_condition(
            policy, batch, shared_noise.clone(), num_steps=10, label=label
        )
        _check_record_shapes(label, record)
        records.append(record)
        print(f"SMOKE_{label}_VELOCITY_SHAPE={record['tap_shapes']['velocity']}")
        print(f"SMOKE_{label}_TRAJECTORY_SHAPE={record['tap_shapes']['trajectory_x0_to_x10']}")
        print(f"SMOKE_{label}_COSINE_SHAPE=[10]")
        print(f"SMOKE_{label}_SUFFIX_OUT_SHAPE={record['tap_shapes']['suffix_out_pre_action_head']}")
        print(f"SMOKE_{label}_PREFIX_SHAPE={record['tap_shapes']['prefix_embed']}")
        print("SMOKE_%s_FINITE=%s" % (label, record["finite"]))
    token_shapes = {
        "official_token_arm": list(official_batch["tokens"].shape),
        "peer_token_arm": list(peer_batch["tokens"].shape),
    }
    if token_shapes["official_token_arm"] != [1, 4] or token_shapes["peer_token_arm"] != [1, 4]:
        raise AssertionError(f"token arm shape mismatch: {token_shapes}")
    sha_after = _sha256_file(frozen_path)
    if sha_before != sha_after:
        raise AssertionError(f"frozen file changed during smoke: {sha_before} != {sha_after}")
    print("SMOKE_CONSTRUCTS=PASS")
    print("SMOKE_FORWARDS=PASS count=4 conditions=official_native,peer_native,official_peer_tokens,peer_official_tokens")
    print("SMOKE_FINITE=PASS")
    print("SMOKE_TAP_SHAPES=PASS")
    print("SMOKE_STEPS_CAPTURED=10")
    print(f"SMOKE_TOKEN_SHAPES={json.dumps(token_shapes, sort_keys=True)}")
    print(f"SMOKE_FILE_SHA_BEFORE={sha_before}")
    print(f"SMOKE_FILE_SHA_AFTER={sha_after}")
    print("SMOKE_SHA_BEFORE_AFTER_MATCH=PASS")
    print("SMOKE_FILENAME_SHA_MATCH=PASS")
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
        raise RuntimeError("D244 full cell requires CUDA; use --smoke for CPU-only verification")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
    if os.environ.get("ROBOTWIN_SKIP_EXPERT") != "0":
        raise RuntimeError("ROBOTWIN_SKIP_EXPERT must be exactly 0")
    start_heartbeat_watchdog()

    harness = load_harness()
    sample = load_sample(args.data_root)
    official_stats = load_checkpoint_stats(args.official_model)
    peer_stats = load_checkpoint_stats(args.peer_model)
    raw_state = sample["state"]

    official_loaded = load_policy_and_processors(
        harness, args.official_model, args.tokenizer
    )
    peer_loaded = load_policy_and_processors(
        harness, args.peer_model, args.tokenizer
    )
    official, official_pre, _official_post, official_env_pre, _official_env_post = official_loaded
    peer, peer_pre, _peer_post, peer_env_pre, _peer_env_post = peer_loaded

    official_batch = build_policy_batch(
        harness, sample, raw_state, sample["instruction"],
        official_env_pre, official_pre, official.config,
    )
    peer_batch = build_policy_batch(
        harness, sample, raw_state, sample["instruction"],
        peer_env_pre, peer_pre, peer.config,
    )
    official_actual = (
        official_batch[state_key(official_batch)].detach().float().cpu().numpy().reshape(STATE_DIM)
    )
    peer_actual = (
        peer_batch[state_key(peer_batch)].detach().float().cpu().numpy().reshape(STATE_DIM)
    )
    official_normalized = assert_state_reproducible(
        "official", raw_state, official_stats["state"], official_actual
    )
    peer_normalized = assert_state_reproducible(
        "peer", raw_state, peer_stats["state"], peer_actual
    )
    official_images = image_hashes(official_batch)
    peer_images = image_hashes(peer_batch)
    if official_images != peer_images:
        raise AssertionError(f"matched observation image hashes differ: {official_images} != {peer_images}")

    official_token_string = _token_strings(args.tokenizer, official_batch)
    peer_token_string = _token_strings(args.tokenizer, peer_batch)
    official_peer_token_batch = _replace_language_tokens(official_batch, peer_batch)
    peer_official_token_batch = _replace_language_tokens(peer_batch, official_batch)

    shared_noise = make_shared_noise(official, args.seed)
    official_native, official_velocity, official_prefix = _run_traced_condition(
        official, official_batch, shared_noise, args.num_inference_steps, "official_native"
    )
    peer_native, peer_velocity, peer_prefix = _run_traced_condition(
        peer, peer_batch, shared_noise, args.num_inference_steps, "peer_native"
    )
    official_peer_tokens, official_peer_velocity, _official_peer_prefix = _run_traced_condition(
        official, official_peer_token_batch, shared_noise, args.num_inference_steps,
        "official_peer_tokens",
    )
    peer_official_tokens, peer_official_velocity, _peer_official_prefix = _run_traced_condition(
        peer, peer_official_token_batch, shared_noise, args.num_inference_steps,
        "peer_official_tokens",
    )

    direction = _direction_report(official_velocity, peer_velocity)
    prefix_compare = _prefix_compare(official_prefix, peer_prefix)
    control = {
        "controlled_diagnostic": True,
        "not_a_serving_correction": True,
        "official_token_string_for_both_models": {
            "token_string": official_token_string,
            "official_model": _control_metric(official_native, official_velocity),
            "peer_model": _control_metric(peer_official_tokens, peer_official_velocity),
        },
        "peer_token_string_for_both_models": {
            "token_string": peer_token_string,
            "official_model": _control_metric(official_peer_tokens, official_peer_velocity),
            "peer_model": _control_metric(peer_native, peer_velocity),
        },
    }
    noise_arm_l2 = arm_l2(shared_noise[0, 0].detach().float().cpu().numpy())
    official_final = torch.tensor(official_native["tap1_trajectory_x0_to_x10"][-1][0])
    peer_final = torch.tensor(peer_native["tap1_trajectory_x0_to_x10"][-1][0])
    official_pos0 = arm_l2(official_final[0].numpy())
    peer_pos0 = arm_l2(peer_final[0].numpy())
    official_ratio = official_pos0 / max(noise_arm_l2, 1e-12)
    peer_ratio = peer_pos0 / max(noise_arm_l2, 1e-12)
    scaling_yes = peer_ratio >= 0.75 and official_ratio <= 0.25

    result = {
        "probe": "D244 FINAL mechanism probe build",
        "n": 1,
        "n_disclosure": "n=1 matched frame per condition; static forward diagnostic, not a rate",
        "sample": {
            "episode": EPISODE,
            "position": POSITION,
            "video_index": sample["video_index"],
            "task": TASK,
        },
        "frozen_setup": {
            "base_probe": "gate3_shared_noise_nose2nose_2c15ea3.py",
            "official_normalization": official_stats["state"]["mode"],
            "peer_normalization": peer_stats["state"]["mode"],
            "own_normalizer_round_trip": True,
            "only_shared_tensor": "injected float32 noise",
            "shared_noise_dtype": str(shared_noise.dtype),
            "num_inference_steps": args.num_inference_steps,
            "matched_image_hashes": official_images,
            "official_normalized_state": official_normalized.tolist(),
            "peer_normalized_state": peer_normalized.tolist(),
            "shared_noise_sha256": hashlib.sha256(
                shared_noise.detach().cpu().numpy().tobytes()
            ).hexdigest(),
        },
        "model_records": {
            "official": {
                "native": official_native,
                "matched_token_control_peer_tokens": official_peer_tokens,
            },
            "peer": {
                "native": peer_native,
                "matched_token_control_official_tokens": peer_official_tokens,
            },
        },
        "tap2_direction": direction,
        "tap4_prefix_encoder_compare": prefix_compare,
        "tap5_matched_token_control": control,
        "scaling_consistency": {
            "space": "model-normalized pre-postprocessor x-space",
            "noise_arm_l2_pos0": noise_arm_l2,
            "official_x_final_arm_l2_pos0": official_pos0,
            "peer_x_final_arm_l2_pos0": peer_pos0,
            "official_x_final_over_noise": official_ratio,
            "peer_x_final_over_noise": peer_ratio,
            "yes_no": "YES" if scaling_yes else "NO",
            "rotate_vs_collapse_read": {
                "peer": "rotates" if peer_ratio >= 0.75 else "does_not_match_rotate_threshold",
                "official": "collapses_toward_zero" if official_ratio <= 0.25 else "does_not_match_collapse_threshold",
            },
        },
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
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
