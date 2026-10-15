#!/usr/bin/env python3
"""Bounded PI05 teacher-forcing discriminator.

This probe loads the production RoboTwin harness path, but never creates a
simulator and never calls select_action or performs a rollout. It holds the
recorded action at the denoiser input, compares the predicted velocity with
the flow-matching target, and runs the same measurement for the SidneyXie
checkpoint as a control.
"""

from __future__ import annotations

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

import numpy as np
import torch
import torch.nn.functional as F


ROOT = Path("/root/autodl-tmp/EvolveVLA")
SCRIPTS = ROOT / "scripts"
DISCRIMINATOR_PATH = SCRIPTS / "gate3_two_checkpoint_discriminator.py"
HARNESS_PATH = SCRIPTS / "robotwin_eval_harness.py"
DATA_ROOT = ROOT / "data/gate2_robotwin_cache/clean"
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
REPORT_PATH = ROOT / "results/gate3_teacher_forcing.md"
EPISODE = 50
POSITION = 31
TASK = "beat_block_hammer"
TIMES = (1.0, 0.5, 0.1)
SEED = 20261003

CHECKPOINTS = {
    "official": ROOT / "checkpoints/pi05_robotwin_lerobot",
    "SidneyXie": ROOT / "checkpoints/sidneyxie_pi05_robotwin",
}


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

    threading.Thread(target=beat, name="teacher-forcing-heartbeat", daemon=True).start()


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


def tensor_stats(value: torch.Tensor) -> dict[str, Any]:
    value = value.detach().float()
    return {
        "shape": list(value.shape),
        "l2": float(value.norm()),
        "mean": float(value.mean()),
        "std": float(value.std()),
        "min": float(value.min()),
        "max": float(value.max()),
    }


def fmt(value: Any, precision: int = 8) -> str:
    if torch.is_tensor(value):
        array = value.detach().float().cpu().numpy()
    else:
        array = np.asarray(value)
    return "[" + ", ".join(
        f"{float(item):.{precision}f}" for item in array.reshape(-1)
    ) + "]"


def checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for name in ("config.json", "policy_preprocessor.json", "norm_stats.json"):
        path = checkpoint / name
        if path.is_file():
            result[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def build_policy_args(checkpoint: Path) -> Namespace:
    return Namespace(
        model_path=str(checkpoint),
        task=TASK,
        episode_steps=1200,
        tokenizer_path=str(TOKENIZER_PATH),
        task_config="demo_clean",
    )


def make_observation(harness: Any, sample: dict[str, Any]) -> dict[str, Any]:
    return harness.simulator_observation(
        {
            "observation": {
                camera: {"rgb": sample["decoded"][camera]}
                for camera in ("head_camera", "left_camera", "right_camera")
            },
            "joint_action": {"vector": sample["state"].tolist()},
        }
    )


def normalize_action(
    preprocessor: Any,
    harness: Any,
    observation: dict[str, Any],
    instruction: str,
    state: np.ndarray,
    action: np.ndarray,
) -> torch.Tensor:
    from lerobot.utils.constants import ACTION

    payload: dict[str, Any] = {
        "observation.state": torch.from_numpy(state),
        "action": torch.from_numpy(action),
        "task": [instruction],
    }
    payload.update(
        {
            harness.POLICY_CAMERA_KEYS[camera]: pixels
            for camera, pixels in observation["pixels"].items()
        }
    )
    normalized = preprocessor(payload)[ACTION]
    if not torch.is_tensor(normalized):
        normalized = torch.as_tensor(normalized)
    normalized = normalized.detach().float()
    if normalized.numel() != 14:
        raise RuntimeError(f"normalized recorded action must have 14 values: {normalized.shape}")
    return normalized.reshape(1, 14)


def pad_action_chunk(action_14: torch.Tensor, model: Any) -> torch.Tensor:
    chunk_size = int(model.config.chunk_size)
    max_action_dim = int(model.config.max_action_dim)
    if max_action_dim < action_14.shape[-1]:
        raise RuntimeError(
            f"max_action_dim={max_action_dim} is smaller than action dim "
            f"{action_14.shape[-1]}"
        )
    repeated = action_14.unsqueeze(1).expand(-1, chunk_size, -1).contiguous()
    return F.pad(repeated, (0, max_action_dim - repeated.shape[-1]))


def build_prefix_cache(
    model: Any,
    policy: Any,
    batch: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor, Any, dict[str, Any]]:
    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )
    from lerobot.policies.common.vla_utils import (
        make_att_2d_masks,
        prepare_attention_masks_4d,
    )

    images, img_masks = policy._preprocess_images(batch)
    language_tokens = batch[OBS_LANGUAGE_TOKENS]
    language_mask = batch[OBS_LANGUAGE_ATTENTION_MASK].bool()
    prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
        images,
        img_masks,
        language_tokens,
        language_mask,
    )
    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
        "eager"
    )
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return prefix_pad_masks, past_key_values, prefix_embs, {
        "prefix_shape": list(prefix_embs.shape),
        "prefix_pad_shape": list(prefix_pad_masks.shape),
        "embed_suffix_signature": str(inspect.signature(model.embed_suffix)),
        "denoise_step_signature": str(inspect.signature(model.denoise_step)),
    }


def measure_checkpoint(
    label: str,
    checkpoint: Path,
    sample: dict[str, Any],
    harness: Any,
) -> dict[str, Any]:
    policy, preprocessor, _postprocessor, env_preprocessor, _env_postprocessor = (
        harness.load_policy(build_policy_args(checkpoint))
    )
    policy.eval()
    model = policy.model
    observation = make_observation(harness, sample)
    batch = harness.make_policy_batch(
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy.config,
    )
    action_norm_14 = normalize_action(
        preprocessor,
        harness,
        observation,
        sample["instruction"],
        sample["state"],
        sample["recorded_action"],
    )
    actions_norm = pad_action_chunk(action_norm_14, model).to(
        device=next(model.parameters()).device
    )
    prefix_pad_masks, past_key_values, prefix_embs, prefix_meta = build_prefix_cache(
        model, policy, batch
    )

    suffix_calls: list[dict[str, Any]] = []
    action_out_calls: list[dict[str, Any]] = []
    original_embed_suffix = model.embed_suffix

    def traced_embed_suffix(*args: Any, **kwargs: Any) -> Any:
        result = original_embed_suffix(*args, **kwargs)
        suffix_calls.append(
            {
                "args": [list(item.shape) for item in args if torch.is_tensor(item)],
                "kwargs": {
                    key: list(value.shape)
                    for key, value in kwargs.items()
                    if torch.is_tensor(value)
                },
            }
        )
        return result

    model.embed_suffix = traced_embed_suffix
    action_out_handle = model.action_out_proj.register_forward_hook(
        lambda _module, inputs, output: action_out_calls.append(
            {
                "input": tensor_stats(inputs[0]),
                "output": tensor_stats(output),
            }
        )
    )

    try:
        torch.manual_seed(SEED)
        noise = model.sample_noise(actions_norm.shape, actions_norm.device)
        noise = noise.detach().float()
        metrics: dict[str, Any] = {}
        with torch.inference_mode():
            for time_value in TIMES:
                timestep = torch.full(
                    (actions_norm.shape[0],),
                    time_value,
                    dtype=torch.float32,
                    device=actions_norm.device,
                )
                x_t = (
                    time_value * noise
                    + (1.0 - time_value) * actions_norm
                )
                velocity = model.denoise_step(
                    prefix_pad_masks=prefix_pad_masks,
                    past_key_values=past_key_values,
                    x_t=x_t,
                    timestep=timestep,
                )
                velocity = velocity.detach().float()
                target = noise - actions_norm
                diff = velocity - target
                flat_velocity = velocity.reshape(velocity.shape[0], -1)
                flat_target = target.reshape(target.shape[0], -1)
                metrics[f"{time_value:.1f}"] = {
                    "x_t": tensor_stats(x_t),
                    "target_u_t": tensor_stats(target),
                    "predicted_v_t": tensor_stats(velocity),
                    "mse": float(diff.pow(2).mean()),
                    "l2": float(diff.norm()),
                    "cosine": float(
                        F.cosine_similarity(flat_velocity, flat_target, dim=1).mean()
                    ),
                }
    finally:
        action_out_handle.remove()
        model.embed_suffix = original_embed_suffix

    result = {
        "checkpoint": str(checkpoint),
        "checkpoint_hashes": checkpoint_hashes(checkpoint),
        "normalization": {
            "config": {
                str(key): str(value)
                for key, value in (
                    getattr(policy.config, "normalization_mapping", {}) or {}
                ).items()
            },
        },
        "action_norm_14": action_norm_14.cpu().numpy().reshape(14).tolist(),
        "actions_norm_shape": list(actions_norm.shape),
        "actions_norm_stats": tensor_stats(actions_norm),
        "prefix": prefix_meta,
        "suffix_calls": suffix_calls,
        "action_out_calls": action_out_calls,
        "metrics": metrics,
        "finite": all(
            np.isfinite(item["mse"])
            and np.isfinite(item["l2"])
            and np.isfinite(item["cosine"])
            for item in metrics.values()
        ),
    }
    del batch, observation, policy, preprocessor, env_preprocessor
    del prefix_pad_masks, past_key_values, prefix_embs, actions_norm
    gc.collect()
    torch.cuda.empty_cache()
    return result


def compare_training_vs_eval_status() -> dict[str, Any]:
    """Only accept an on-disk eval observation with all required modalities."""
    candidates = [
        ROOT / "results",
        ROOT / "eval_obs",
        Path("/root/runs/zhouxueyangapexinai"),
    ]
    found = []
    for directory in candidates:
        if not directory.exists():
            continue
        for path in directory.glob("**/*"):
            name = path.name.lower()
            if path.is_file() and any(token in name for token in ("obs", "observation")):
                found.append(str(path))
    return {
        "status": "NOT_RUN",
        "reason": (
            "no complete three-camera eval observation with 14-D state and "
            "recorded action was available on disk; the retained eval dump is "
            "cam_high-only, so fabricating wrist inputs would invalidate the cell"
        ),
        "candidate_observation_files": found[:50],
    }


def load_sample(discriminator: Any) -> dict[str, Any]:
    sample = discriminator.load_sample()
    return sample


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    start_heartbeat()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")

    discriminator = load_module(
        DISCRIMINATOR_PATH, "gate3_teacher_forcing_discriminator"
    )
    harness = discriminator.load_harness()
    sample = load_sample(discriminator)
    results = {
        label: measure_checkpoint(label, checkpoint, sample, harness)
        for label, checkpoint in CHECKPOINTS.items()
    }
    official = results["official"]["metrics"]
    official_fit = all(
        item["cosine"] >= 0.9 and item["mse"] <= 1e-2
        for item in official.values()
    )
    if official_fit:
        verdict = "sampling/inference defect"
        named_fix = (
            "audit the Euler flow loop sign, timestep direction, and the "
            "select_action chunk-consumption path"
        )
    else:
        verdict = "conversion-semantics/training-target mismatch"
        named_fix = (
            "re-derive the official action-head/target semantics and obtain or "
            "reproduce the official inference recipe; do not treat this artifact "
            "as behaviorally equivalent to the leaderboard checkpoint"
        )

    payload = {
        "exp_id": "gate3_teacher_forcing",
        "parent": "gate3_official_protocol_5x400_armfix",
        "dataset": str(DATA_ROOT),
        "episode": EPISODE,
        "position": POSITION,
        "video_index": int(sample["video_index"]),
        "instruction": sample["instruction"],
        "recorded_action": np.asarray(sample["recorded_action"]).reshape(14).tolist(),
        "recorded_action_l2": float(
            np.linalg.norm(np.asarray(sample["recorded_action"]).reshape(14))
        ),
        "results": results,
        "training_vs_eval": compare_training_vs_eval_status(),
        "source_sha256": {
            "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "discriminator": hashlib.sha256(DISCRIMINATOR_PATH.read_bytes()).hexdigest(),
            "harness": hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest(),
        },
        "verdict": verdict,
        "named_fix": named_fix,
        "no_rollout": True,
    }

    lines = [
        "# Gate3 Teacher-Forcing Test",
        "",
        f"- Experiment: `{payload['exp_id']}`",
        f"- Parent: `{payload['parent']}`",
        f"- Dataset: `{DATA_ROOT}`",
        f"- Sample: episode `{EPISODE}`, position `{POSITION}`, video index `{sample['video_index']}`",
        f"- Instruction: `{sample['instruction']}`",
        f"- Recorded action: `{fmt(sample['recorded_action'])}`",
        f"- Recorded action L2: `{payload['recorded_action_l2']:.8f}`",
        f"- Source SHA256: `{json.dumps(payload['source_sha256'], sort_keys=True)}`",
        "- Harness was not modified; no rollout, simulator, perception, or resolution probe was run.",
        "",
        "## Target Construction",
        "",
        "- `actions_norm` is the recorded 14-D action normalized by each checkpoint's own preprocessor, repeated over `chunk_size`, then right-padded to `max_action_dim`.",
        "- `x_t = t * noise + (1 - t) * actions_norm`.",
        "- `u_t = noise - actions_norm`.",
        "- `v_t` is obtained through the model's own prefix cache, `denoise_step`/`embed_suffix`, and `action_out_proj`.",
        "",
        "## Teacher-Forcing Results",
        "",
    ]
    for label, result in results.items():
        lines.extend(
            [
                f"### {label}",
                f"- Checkpoint: `{result['checkpoint']}`",
                f"- Checkpoint file hashes: `{result['checkpoint_hashes']}`",
                f"- `actions_norm` shape: `{result['actions_norm_shape']}`",
                f"- `actions_norm` stats: `{result['actions_norm_stats']}`",
                f"- Prefix metadata: `{result['prefix']}`",
                f"- `embed_suffix` calls: `{len(result['suffix_calls'])}`",
                f"- `action_out_proj` calls: `{len(result['action_out_calls'])}`",
                "",
                "| t | MSE(v_t,u_t) | L2(v_t-u_t) | cosine(v_t,u_t) |",
                "|---:|---:|---:|---:|",
            ]
        )
        for time_value in TIMES:
            metric = result["metrics"][f"{time_value:.1f}"]
            lines.append(
                f"| {time_value:.1f} | {metric['mse']:.8f} | "
                f"{metric['l2']:.8f} | {metric['cosine']:.8f} |"
            )
        lines.extend(["", f"- Finite metrics: `{result['finite']}`", ""])

    lines.extend(
        [
            "## Training-vs-Eval Cell",
            "",
            f"- Status: `{payload['training_vs_eval']['status']}`",
            f"- Reason: {payload['training_vs_eval']['reason']}",
            f"- Candidate files inspected: `{payload['training_vs_eval']['candidate_observation_files']}`",
            "",
            "## Verdict",
            "",
            f"- Official fit criterion (`MSE <= 1e-2` and `cosine >= 0.9` at all three t): `{official_fit}`",
            f"- **VERDICT: `{verdict}`**",
            f"- Named fix: {named_fix}",
            "",
            "No simulator rollout was performed.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"REPORT={REPORT_PATH}")
    print(f"EXP_ID={payload['exp_id']}")
    print(f"HARNESS_SHA256={payload['source_sha256']['harness']}")
    print(f"PROBE_SHA256={payload['source_sha256']['probe']}")
    for label, result in results.items():
        print(f"{label.upper()}_ACTIONS_NORM_SHAPE={result['actions_norm_shape']}")
        for time_value in TIMES:
            metric = result["metrics"][f"{time_value:.1f}"]
            print(
                f"{label.upper()}_T={time_value:.1f} "
                f"MSE={metric['mse']:.8f} L2={metric['l2']:.8f} "
                f"COSINE={metric['cosine']:.8f}"
            )
    print(f"TRAINING_VS_EVAL={payload['training_vs_eval']['status']}")
    print(f"VERDICT={verdict}")
    print(f"NAMED_FIX={named_fix}")
    print(f"REPORT_SHA256={hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
