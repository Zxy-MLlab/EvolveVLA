#!/usr/bin/env python3
"""Bounded PI05 expert-contribution and oracle-construction control probe."""

from __future__ import annotations

import gc
import hashlib
import importlib.util
import json
import os
import sys
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path("/root/autodl-tmp/EvolveVLA")
SCRIPTS = ROOT / "scripts"
DISCRIMINATOR_PATH = SCRIPTS / "gate3_two_checkpoint_discriminator.py"
ORACLE_PATH = SCRIPTS / "remote_oracle_action_reproduction.py"
HARNESS_PATH = SCRIPTS / "robotwin_eval_harness.py"
CHECKPOINT = ROOT / "checkpoints/pi05_robotwin_lerobot"
SIDNEY_CHECKPOINT = ROOT / "checkpoints/sidneyxie_pi05_robotwin"
TOKENIZER = Path("/root/autodl-tmp/paligemma_tokenizer")
REPORT_PATH = ROOT / "results/gate3_expert_contribution_and_oracle_control.md"
EPISODE = 50
POSITION = 31
TASK = "beat_block_hammer"
OUTPUT_DELTA_THRESHOLD = 1e-3


def start_heartbeat() -> None:
    heartbeat_path = os.environ.get("HEARTBEAT_LOG")
    if not heartbeat_path:
        return

    def beat() -> None:
        while True:
            time.sleep(30)
            try:
                os.utime(heartbeat_path, None)
            except OSError:
                return

    threading.Thread(target=beat, name="probe-log-heartbeat", daemon=True).start()


def load_module(path: Path, name: str) -> Any:
    if str(path.parent) not in sys.path:
        sys.path.insert(0, str(path.parent))
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def source_lines(path: Path, start: int, end: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [f"{index}:{lines[index - 1]}" for index in range(start, end + 1)]


def fmt(values: Any, precision: int = 8) -> str:
    if torch.is_tensor(values):
        array = values.detach().float().cpu().numpy()
    else:
        array = np.asarray(values)
    return "[" + ", ".join(f"{float(value):.{precision}f}" for value in array.reshape(-1)) + "]"


def tensor_max_abs_diff(left: torch.Tensor, right: torch.Tensor) -> float:
    return float((left.detach().float() - right.detach().float()).abs().max())


def build_raw_observation(sample: dict[str, Any]) -> dict[str, Any]:
    return {
        "observation": {
            camera: {"rgb": sample["decoded"][camera]}
            for camera in ("head_camera", "left_camera", "right_camera")
        },
        "joint_action": {"vector": sample["state"].tolist()},
    }


def load_policy_and_processors(
    harness: Any,
    checkpoint: Path,
) -> tuple[Any, Any, Any, Any, Any]:
    args = Namespace(
        model_path=str(checkpoint),
        task=TASK,
        episode_steps=1200,
        tokenizer_path=str(TOKENIZER),
        task_config="demo_clean",
    )
    return harness.load_policy(args)


def build_batch(
    harness: Any,
    policy: Any,
    preprocessor: Any,
    env_preprocessor: Any,
    sample: dict[str, Any],
) -> dict[str, torch.Tensor]:
    observation = harness.simulator_observation(build_raw_observation(sample))
    return harness.make_policy_batch(
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy.config,
    )


def fixed_noise(policy: Any) -> torch.Tensor:
    config = policy.config
    device = next(policy.model.parameters()).device
    torch.manual_seed(20261003)
    return policy.model.sample_noise(
        (1, int(config.chunk_size), int(config.max_action_dim)),
        device,
    ).detach().clone()


def run_raw_first_action(policy: Any, batch: dict[str, torch.Tensor], noise: torch.Tensor) -> np.ndarray:
    policy.reset()
    with torch.inference_mode():
        chunk = policy.predict_action_chunk(batch, noise=noise)
    return chunk[:, 0, :].detach().float().cpu().numpy().reshape(14)


def expert_parameters(policy: Any) -> list[torch.nn.Parameter]:
    expert = policy.model.paligemma_with_expert.gemma_expert.model
    return list(expert.parameters())


def run_expert_scale(
    policy: Any,
    batch: dict[str, torch.Tensor],
    noise: torch.Tensor,
    snapshots: list[torch.Tensor],
    scale: float,
) -> np.ndarray:
    parameters = expert_parameters(policy)
    if len(parameters) != len(snapshots):
        raise RuntimeError("expert parameter count changed during probe")
    try:
        for parameter, snapshot in zip(parameters, snapshots, strict=True):
            parameter.data.copy_(snapshot * scale)
        return run_raw_first_action(policy, batch, noise)
    finally:
        for parameter, snapshot in zip(parameters, snapshots, strict=True):
            parameter.data.copy_(snapshot)


def make_result(baseline: np.ndarray, output: np.ndarray, scale: float) -> dict[str, Any]:
    return {
        "scale": scale,
        "raw_action": output.tolist(),
        "raw_arm_l2": float(np.linalg.norm(output[:6])),
        "delta_from_baseline_l2": float(np.linalg.norm(output - baseline)),
        "arm_l2_delta": float(np.linalg.norm(output[:6] - baseline[:6])),
    }


def compare_batches(
    left: dict[str, torch.Tensor],
    right: dict[str, torch.Tensor],
) -> dict[str, Any]:
    left_keys = sorted(str(key) for key in left)
    right_keys = sorted(str(key) for key in right)
    common = sorted(set(left) & set(right), key=str)
    diffs: dict[str, float] = {}
    shapes: dict[str, Any] = {}
    for key in common:
        left_value = left[key]
        right_value = right[key]
        if torch.is_tensor(left_value) and torch.is_tensor(right_value):
            shapes[str(key)] = {
                "left": list(left_value.shape),
                "right": list(right_value.shape),
            }
            if left_value.shape == right_value.shape:
                diffs[str(key)] = tensor_max_abs_diff(left_value, right_value)
    return {
        "same_keys": left_keys == right_keys,
        "left_only": [str(key) for key in sorted(set(left) - set(right), key=str)],
        "right_only": [str(key) for key in sorted(set(right) - set(left), key=str)],
        "tensor_max_abs_diff": diffs,
        "max_tensor_diff": max(diffs.values(), default=0.0),
        "shapes": shapes,
    }


def oracle_control(
    discriminator: Any,
    harness: Any,
    sample: dict[str, Any],
    noise_seed: int = 20261003,
) -> dict[str, Any]:
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        load_policy_and_processors(harness, SIDNEY_CHECKPOINT)
    )
    del postprocessor, env_postprocessor

    # This is the construction used by remote_oracle_action_reproduction.py:
    # simulator_observation(raw) followed by make_policy_batch(...).
    oracle_batch = build_batch(
        harness,
        policy,
        preprocessor,
        env_preprocessor,
        sample,
    )
    discriminator_batch = build_batch(
        harness,
        policy,
        preprocessor,
        env_preprocessor,
        sample,
    )
    batch_diff = compare_batches(oracle_batch, discriminator_batch)

    torch.manual_seed(noise_seed)
    oracle_noise = fixed_noise(policy)
    discriminator_noise = oracle_noise.detach().clone()
    oracle_action = run_raw_first_action(policy, oracle_batch, oracle_noise)
    discriminator_action = run_raw_first_action(
        policy,
        discriminator_batch,
        discriminator_noise,
    )
    recorded = np.asarray(sample["recorded_action"], dtype=np.float32).reshape(14)
    result = {
        "checkpoint": str(SIDNEY_CHECKPOINT),
        "recorded_arm_l2": float(np.linalg.norm(recorded[:6])),
        "oracle_raw_action": oracle_action.tolist(),
        "oracle_raw_arm_l2": float(np.linalg.norm(oracle_action[:6])),
        "oracle_vs_recorded_arm_l2_error": float(
            np.linalg.norm(oracle_action[:6] - recorded[:6])
        ),
        "discriminator_raw_action": discriminator_action.tolist(),
        "discriminator_raw_arm_l2": float(np.linalg.norm(discriminator_action[:6])),
        "discriminator_vs_recorded_arm_l2_error": float(
            np.linalg.norm(discriminator_action[:6] - recorded[:6])
        ),
        "oracle_vs_discriminator_output_l2": float(
            np.linalg.norm(oracle_action - discriminator_action)
        ),
        "batch_diff": batch_diff,
    }
    del oracle_batch, discriminator_batch, policy, preprocessor, env_preprocessor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def main() -> int:
    os.environ["PYTHONUNBUFFERED"] = "1"
    start_heartbeat()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")

    discriminator = load_module(DISCRIMINATOR_PATH, "gate3_discriminator_reused")
    harness = discriminator.load_harness()
    sample = discriminator.load_sample()
    recorded = np.asarray(sample["recorded_action"], dtype=np.float32).reshape(14)

    official_policy, official_preprocessor, official_postprocessor, official_env_pre, official_env_post = (
        load_policy_and_processors(harness, CHECKPOINT)
    )
    del official_postprocessor, official_env_post
    official_batch = build_batch(
        harness,
        official_policy,
        official_preprocessor,
        official_env_pre,
        sample,
    )
    noise = fixed_noise(official_policy)
    baseline = run_raw_first_action(official_policy, official_batch, noise)
    official_expert_parameters = expert_parameters(official_policy)
    expert_parameter_count = sum(
        parameter.numel() for parameter in official_expert_parameters
    )
    snapshots = [parameter.detach().clone() for parameter in official_expert_parameters]

    scales = [0.0, 0.1, 1.0, 10.0]
    sweep: dict[str, dict[str, Any]] = {}
    for scale in scales:
        if scale == 1.0:
            output = baseline
        else:
            output = run_expert_scale(
                official_policy,
                official_batch,
                noise,
                snapshots,
                scale,
            )
        sweep[str(scale)] = make_result(baseline, output, scale)

    removal_delta = sweep["0.0"]["delta_from_baseline_l2"]
    scale_response = max(
        sweep[str(scale)]["delta_from_baseline_l2"]
        for scale in (0.1, 10.0)
    )
    expert_contributing = (
        removal_delta > OUTPUT_DELTA_THRESHOLD
        or scale_response > OUTPUT_DELTA_THRESHOLD
    )
    del official_batch, official_policy, official_preprocessor, official_env_pre
    del snapshots, official_expert_parameters
    gc.collect()
    torch.cuda.empty_cache()
    oracle = oracle_control(discriminator, harness, sample)

    payload = {
        "exp_id": "gate3_expert_contribution_and_oracle_control",
        "parent": "gate3_official_protocol_5x400_armfix",
        "dataset": str(discriminator.DATA_ROOT),
        "episode": EPISODE,
        "position": POSITION,
        "video_index": sample["video_index"],
        "instruction": sample["instruction"],
        "recorded_action": recorded.tolist(),
        "recorded_arm_l2": float(np.linalg.norm(recorded[:6])),
        "official_baseline_raw_action": baseline.tolist(),
        "official_baseline_raw_arm_l2": float(np.linalg.norm(baseline[:6])),
        "expert_parameter_count": expert_parameter_count,
        "expert_scale_sweep": sweep,
        "expert_contributing": expert_contributing,
        "oracle_control": oracle,
        "source_sha256": {
            "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "discriminator": hashlib.sha256(DISCRIMINATOR_PATH.read_bytes()).hexdigest(),
            "oracle": hashlib.sha256(ORACLE_PATH.read_bytes()).hexdigest(),
            "harness": hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest(),
        },
    }
    oracle_action_like = oracle["oracle_raw_arm_l2"] > 0.5
    if oracle_action_like:
        oracle_verdict = "ORACLE_PATH_CONTROL_PASSES: SidneyXie remains action-like"
    else:
        oracle_verdict = "ORACLE_PATH_CONTROL_COLLAPSES: diff construction paths"

    discriminator_lines = source_lines(DISCRIMINATOR_PATH, 198, 220)
    oracle_lines = source_lines(ORACLE_PATH, 122, 150)
    lines = [
        "# Gate3 Expert Contribution and Oracle Control",
        "",
        f"- Experiment: `{payload['exp_id']}`",
        f"- Parent: `{payload['parent']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Sample: episode `{EPISODE}`, position `{POSITION}`, video index `{sample['video_index']}`",
        f"- Recorded arm L2: `{payload['recorded_arm_l2']:.8f}`",
        f"- Source SHA256: `{json.dumps(payload['source_sha256'], sort_keys=True)}`",
        "",
        "## Part A: Official Action-Expert Contribution",
        "",
        f"- Expert target: `policy.model.paligemma_with_expert.gemma_expert.model`",
        f"- Expert parameter count: `{payload['expert_parameter_count']}`",
        f"- Fixed-noise baseline raw arm L2: `{payload['official_baseline_raw_arm_l2']:.8f}`",
        "",
        "| expert scale | raw arm L2 | output delta from baseline L2 | arm-vector delta L2 |",
        "|---:|---:|---:|---:|",
    ]
    for scale in scales:
        result = sweep[str(scale)]
        lines.append(
            f"| {scale:g} | {result['raw_arm_l2']:.8f} | "
            f"{result['delta_from_baseline_l2']:.8f} | "
            f"{result['arm_l2_delta']:.8f} |"
        )
    lines.extend(
        [
            "",
            f"- Expert-removed output delta: `{removal_delta:.8f}`",
            f"- Maximum non-unit scale response: `{scale_response:.8f}`",
            f"- **Part A verdict: `{'EXPERT_CONTRIBUTING' if expert_contributing else 'EXPERT_BYPASSED_OR_NOT_CONTRIBUTING'}`**",
            "",
            "## Part B: SidneyXie Through Oracle Construction",
            "",
            f"- Recorded arm L2: `{oracle['recorded_arm_l2']:.8f}`",
            f"- Oracle-path predicted arm L2: `{oracle['oracle_raw_arm_l2']:.8f}`",
            f"- Oracle-path predicted-vs-recorded arm L2 error: `{oracle['oracle_vs_recorded_arm_l2_error']:.8f}`",
            f"- Discriminator-path predicted arm L2: `{oracle['discriminator_raw_arm_l2']:.8f}`",
            f"- Oracle-vs-discriminator output L2: `{oracle['oracle_vs_discriminator_output_l2']:.8f}`",
            f"- Batch construction max tensor diff: `{oracle['batch_diff']['max_tensor_diff']:.8f}`",
            f"- **Part B verdict: `{oracle_verdict}`**",
            "",
            "### Construction-path source lines",
            "",
            "Discriminator path (`gate3_two_checkpoint_discriminator.py`):",
            "",
            "```text",
            *discriminator_lines,
            "```",
            "",
            "Oracle path (`remote_oracle_action_reproduction.py`):",
            "",
            "```text",
            *oracle_lines,
            "```",
            "",
            "## Overall Verdict",
            "",
            f"- (i) Expert branch: **{'contributing' if expert_contributing else 'bypassed/not contributing'}** under fixed-noise weight scaling.",
            f"- (ii) Oracle vs discriminator construction: **{'no material difference observed' if oracle_action_like else 'difference must be localized; see batch/source comparison above'}**.",
            "",
            "No simulator rollout was performed. Existing harness, discriminator, oracle, and checkpoint files were not modified.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT={REPORT_PATH}")
    print(f"RECORDED_ARM_L2={payload['recorded_arm_l2']:.8f}")
    print(f"OFFICIAL_BASELINE_RAW_ARM_L2={payload['official_baseline_raw_arm_l2']:.8f}")
    for scale in scales:
        result = sweep[str(scale)]
        print(
            f"EXPERT_SCALE={scale:g} RAW_ARM_L2={result['raw_arm_l2']:.8f} "
            f"OUTPUT_DELTA_L2={result['delta_from_baseline_l2']:.8f}"
        )
    print(f"EXPERT_CONTRIBUTING={expert_contributing}")
    print(f"SIDNEYXIE_ORACLE_RAW_ARM_L2={oracle['oracle_raw_arm_l2']:.8f}")
    print(f"SIDNEYXIE_DISCRIMINATOR_RAW_ARM_L2={oracle['discriminator_raw_arm_l2']:.8f}")
    print(f"ORACLE_VS_DISCRIMINATOR_OUTPUT_L2={oracle['oracle_vs_discriminator_output_l2']:.8f}")
    print(f"ORACLE_BATCH_MAX_TENSOR_DIFF={oracle['batch_diff']['max_tensor_diff']:.8f}")
    print(f"ORACLE_VERDICT={oracle_verdict}")
    print(f"REPORT_SHA256={hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
