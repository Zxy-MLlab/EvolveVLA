#!/usr/bin/env python3
"""Bounded PI05 flow/action-expert trace on one real dataset observation."""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
import types
from argparse import Namespace
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


REMOTE_ROOT = Path("/root/autodl-tmp/EvolveVLA")
SCRIPTS = REMOTE_ROOT / "scripts"
ROOTCAUSE_PATH = SCRIPTS / "gate3_conditioning_rootcause_probe.py"
CHECKPOINT = REMOTE_ROOT / "checkpoints/pi05_robotwin_lerobot"
DATA_ROOT = REMOTE_ROOT / "data/gate2_robotwin_cache/clean"
TOKENIZER = Path("/root/autodl-tmp/paligemma_tokenizer")
OUTPUT_LOG = REMOTE_ROOT / "results/gate3_action_expert_engagement_probe.log"
EPISODE = 50
POSITION = 31


def load_rootcause():
    spec = importlib.util.spec_from_file_location("gate3_rootcause", ROOTCAUSE_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {ROOTCAUSE_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def tensor_stats(value: torch.Tensor) -> dict[str, object]:
    value = value.detach().float()
    return {
        "shape": list(value.shape),
        "min": float(value.amin()),
        "max": float(value.amax()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "l2": float(value.norm()),
    }


def fmt(value: torch.Tensor, precision: int = 6) -> str:
    arr = value.detach().float().cpu().numpy().reshape(-1)
    return "[" + ", ".join(f"{float(item):.{precision}f}" for item in arr) + "]"


def load_real_batch(root, harness):
    data_path = DATA_ROOT / "data/chunk-000/file-000.parquet"
    data = pq.read_table(
        data_path,
        columns=[
            "observation.state",
            "action",
            "timestamp",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    ).to_pydict()
    tasks = pq.read_table(DATA_ROOT / "meta/tasks.parquet").to_pydict()
    task_text = dict(zip(tasks["task_index"], tasks["task"]))
    rows = [
        {key: data[key][row] for key in data}
        for row in range(len(data["episode_index"]))
        if data["episode_index"][row] == EPISODE
    ]
    if len(rows) <= POSITION:
        raise RuntimeError(f"episode {EPISODE} has only {len(rows)} rows")
    rows.sort(key=lambda row: row["frame_index"])
    row = rows[POSITION]
    video_index = int(row["index"])
    instruction = str(task_text[int(row["task_index"])])
    state = np.asarray(row["observation.state"], dtype=np.float32).reshape(14)
    recorded_action = np.asarray(row["action"], dtype=np.float32).reshape(14)

    camera_files = {
        "head_camera": DATA_ROOT
        / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
        "left_camera": DATA_ROOT
        / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
        "right_camera": DATA_ROOT
        / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
    }
    decoded = {
        camera: root.decode_frame(path, video_index)
        for camera, path in camera_files.items()
    }
    observation = root.make_observation(harness, decoded, state)

    args = Namespace(
        model_path=str(CHECKPOINT),
        task="beat_block_hammer",
        episode_steps=1200,
        tokenizer_path=str(TOKENIZER),
        task_config="demo_clean",
    )
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        harness.load_policy(args)
    )
    batch = root.make_batch(
        harness,
        observation,
        instruction,
        env_preprocessor,
        preprocessor,
        policy.config,
    )
    return {
        "policy": policy,
        "batch": batch,
        "postprocessor": postprocessor,
        "env_postprocessor": env_postprocessor,
        "instruction": instruction,
        "state": state,
        "recorded_action": recorded_action,
        "video_index": video_index,
    }


def checkpoint_match(expert, checkpoint_path: Path, selected_name: str) -> dict[str, object]:
    from safetensors.torch import load_file

    live = dict(expert.named_parameters())[selected_name].detach().float().cpu()
    state_dict = load_file(str(checkpoint_path / "model.safetensors"), device="cpu")
    matches = [key for key in state_dict if key.endswith(selected_name)]
    if not matches:
        return {
            "selected_name": selected_name,
            "checkpoint_key": None,
            "live_l2": float(live.norm()),
            "live_nonzero_fraction": float((live != 0).float().mean()),
            "exact_max_abs_diff": None,
        }
    checkpoint = state_dict[matches[0]].detach().float().cpu()
    return {
        "selected_name": selected_name,
        "checkpoint_key": matches[0],
        "live_l2": float(live.norm()),
        "checkpoint_l2": float(checkpoint.norm()),
        "live_nonzero_fraction": float((live != 0).float().mean()),
        "exact_max_abs_diff": float((live - checkpoint).abs().max()),
        "shape": list(live.shape),
    }


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))
    torch.manual_seed(0)

    root = load_rootcause()
    harness = root.load_harness()
    loaded = load_real_batch(root, harness)
    policy = loaded["policy"]
    batch = loaded["batch"]
    model = policy.model
    config = policy.config
    expert = model.paligemma_with_expert.gemma_expert.model
    steps = int(config.num_inference_steps)
    dt = -1.0 / steps

    print("=== GATE3 ACTION-EXPERT ENGAGEMENT PROBE ===")
    print(f"CHECKPOINT={CHECKPOINT}")
    print(f"HARNESS={root.HARNESS_PATH}")
    print(f"HARNESS_SHA256={hashlib.sha256(root.HARNESS_PATH.read_bytes()).hexdigest()}")
    print(f"MODEL_FILE_SHA256={hashlib.sha256((CHECKPOINT / 'model.safetensors').read_bytes()).hexdigest()}")
    print(f"DATASET={DATA_ROOT}")
    print(f"EPISODE={EPISODE}")
    print(f"POSITION={POSITION}")
    print(f"VIDEO_INDEX={loaded['video_index']}")
    print(f"INSTRUCTION={loaded['instruction']!r}")
    print(f"RECORDED_ACTION={fmt(torch.from_numpy(loaded['recorded_action']))}")
    print(f"CONFIG={json.dumps({key: getattr(config, key) for key in ('chunk_size', 'n_action_steps', 'max_action_dim', 'num_inference_steps', 'time_sampling_beta_alpha', 'time_sampling_beta_beta', 'time_sampling_scale', 'time_sampling_offset', 'min_period', 'max_period')}, sort_keys=True)}")
    print(f"RUNTIME_FLOW_STEPS={steps}")
    print(f"RUNTIME_FLOW_DT={dt}")
    print("RUNTIME_FLOW_TIMES=" + json.dumps([1.0 + step * dt for step in range(steps)]))
    print("RUNTIME_FLOW_SCHEDULE=forward Euler: x_t <- x_t + (-1/num_steps) * v_t; t=1+step*dt")
    print("RUNTIME_TRAINING_BETA_SAMPLER=declared for sample_time/training only; not called by sample_actions")

    expert_param_count = sum(parameter.numel() for parameter in expert.parameters())
    expert_sq_norm = sum(
        float(parameter.detach().float().pow(2).sum()) for parameter in expert.parameters()
    )
    expert_nonzero = sum(
        int((parameter.detach() != 0).sum()) for parameter in expert.parameters()
    )
    print(f"ACTION_EXPERT_PARAM_COUNT={expert_param_count}")
    print(f"ACTION_EXPERT_PARAM_L2={expert_sq_norm ** 0.5:.8f}")
    print(f"ACTION_EXPERT_NONZERO_FRACTION={expert_nonzero / expert_param_count:.8f}")
    print(
        "ACTION_EXPERT_CHECKPOINT_MATCH="
        + json.dumps(
            checkpoint_match(
                expert,
                CHECKPOINT,
                "layers.0.self_attn.q_proj.weight",
            ),
            sort_keys=True,
        )
    )

    traces: list[dict[str, object]] = []
    expert_calls: list[dict[str, object]] = []
    action_out_calls: list[dict[str, object]] = []
    action_in_calls: list[dict[str, object]] = []

    original_expert_forward = expert.forward

    def traced_expert_forward(*args, **kwargs):
        output = original_expert_forward(*args, **kwargs)
        hidden = getattr(output, "last_hidden_state", None)
        if hidden is not None:
            record = tensor_stats(hidden)
            expert_calls.append(record)
            print(f"ACTION_EXPERT_FORWARD call={len(expert_calls)} stats={json.dumps(record, sort_keys=True)}")
        return output

    expert.forward = traced_expert_forward

    def action_in_hook(_module, inputs, output):
        record = {
            "input": tensor_stats(inputs[0]),
            "output": tensor_stats(output),
        }
        action_in_calls.append(record)
        print(f"ACTION_IN_PROJ call={len(action_in_calls)} {json.dumps(record, sort_keys=True)}")

    def action_out_hook(_module, inputs, output):
        record = {
            "input": tensor_stats(inputs[0]),
            "output": tensor_stats(output),
        }
        action_out_calls.append(record)
        print(f"ACTION_OUT_PROJ call={len(action_out_calls)} {json.dumps(record, sort_keys=True)}")

    action_in_handle = model.action_in_proj.register_forward_hook(action_in_hook)
    action_out_handle = model.action_out_proj.register_forward_hook(action_out_hook)
    original_denoise = model.denoise_step

    def traced_denoise(*args, **kwargs):
        x_t = kwargs["x_t"]
        timestep = kwargs["timestep"]
        before = tensor_stats(x_t)
        velocity = original_denoise(*args, **kwargs)
        after = x_t + dt * velocity
        record = {
            "step": len(traces),
            "time": float(timestep[0].item()),
            "x_before": before,
            "velocity": tensor_stats(velocity),
            "delta": tensor_stats(dt * velocity),
            "x_after": tensor_stats(after),
            "sample_changed_l2": float((after - x_t).detach().float().norm()),
        }
        traces.append(record)
        print(f"FLOW_STEP {json.dumps(record, sort_keys=True)}")
        return velocity

    model.denoise_step = traced_denoise
    original_sample = model.sample_actions

    def traced_sample(*args, **kwargs):
        padded = original_sample(*args, **kwargs)
        print(f"SAMPLE_ACTIONS_PADDED_SHAPE={list(padded.shape)}")
        print(f"SAMPLE_ACTIONS_PADDED_STATS={json.dumps(tensor_stats(padded), sort_keys=True)}")
        return padded

    model.sample_actions = traced_sample

    try:
        with torch.inference_mode():
            chunk = policy.predict_action_chunk(batch)
    finally:
        action_in_handle.remove()
        action_out_handle.remove()
        expert.forward = original_expert_forward
        model.denoise_step = original_denoise
        model.sample_actions = original_sample

    print(f"PREDICT_ACTION_CHUNK_SHAPE={list(chunk.shape)}")
    print("PREDICT_ACTION_CHUNK_SLICE_VERDICT=32 padded dims were returned by model.sample_actions; policy.predict_action_chunk returned the first 14 dims")
    print(f"FLOW_STEP_COUNT={len(traces)}")
    print(f"ACTION_EXPERT_FORWARD_COUNT={len(expert_calls)}")
    print(f"ACTION_IN_PROJ_CALL_COUNT={len(action_in_calls)}")
    print(f"ACTION_OUT_PROJ_CALL_COUNT={len(action_out_calls)}")
    print(
        "ACTION_EXPERT_TO_HEAD_NORM_MATCH="
        + json.dumps(
            [
                {
                    "expert_output_l2": expert_calls[index]["l2"],
                    "action_out_input_l2": action_out_calls[index]["input"]["l2"],
                    "abs_diff": abs(
                        expert_calls[index]["l2"] - action_out_calls[index]["input"]["l2"]
                    ),
                }
                for index in range(min(len(expert_calls), len(action_out_calls)))
            ],
            sort_keys=True,
        )
    )
    print(
        "ACTION_EXPERT_ENGAGEMENT_VERDICT="
        + (
            "LOADED_AND_CONTRIBUTING"
            if len(expert_calls) == steps
            and len(action_out_calls) == steps
            and all(item["input"]["l2"] > 0 for item in action_out_calls)
            else "NOT_CONFIRMED"
        )
    )
    print(
        "FLOW_SAMPLE_MOVEMENT_VERDICT="
        + (
            "INTEGRATES"
            if len(traces) == steps
            and all(item["sample_changed_l2"] > 0 for item in traces)
            else "COLLAPSES_OR_INCOMPLETE"
        )
    )
    print("VERDICT=TRACE_COMPLETE; fault classification requires comparing this velocity path against the mean-collapse symptom.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
