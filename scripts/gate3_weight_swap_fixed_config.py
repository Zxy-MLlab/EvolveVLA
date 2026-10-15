#!/usr/bin/env python3
# Runs one fixed real RoboTwin observation through the official PI05 config,
# then repeats it after swapping SidneyXie tensors into that policy. Inputs are
# the cached sample and both model.safetensors files; outputs are arm-L2/cosine
# measurements plus key audits in results/gate3_weight_swap_fixed_config.{md,json}.
# Key deps: PyTorch, safetensors, pyarrow/av, gate3_teacher_forcing, and the harness.
"""Bounded SidneyXie-weight swap under the official PI05 input path."""

from __future__ import annotations

import gc
import hashlib
import json
import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from safetensors import safe_open

import gate3_teacher_forcing as teacher_forcing
import gate3_two_checkpoint_discriminator as discriminator


ROOT = Path("/root/autodl-tmp/EvolveVLA")
HARNESS_PATH = ROOT / "scripts/robotwin_eval_harness.py"
DATA_ROOT = ROOT / "data/gate2_robotwin_cache/clean"
OFFICIAL = ROOT / "checkpoints/pi05_robotwin_lerobot"
SIDNEY = ROOT / "checkpoints/sidneyxie_pi05_robotwin"
REPORT_PATH = ROOT / "results/gate3_weight_swap_fixed_config.md"
JSON_PATH = ROOT / "results/gate3_weight_swap_fixed_config.json"
EPISODE = 50
POSITION = 31
TASK = "beat_block_hammer"
SEED = 20261003


def heartbeat(message: str) -> None:
    log_path = os.environ.get("HEARTBEAT_LOG")
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"HEARTBEAT {message}\n")
        handle.flush()


def tensor_metadata(path: Path) -> dict[str, tuple[int, ...]]:
    with safe_open(str(path), framework="pt", device="cpu") as handle:
        return {
            key: tuple(handle.get_slice(key).get_shape())
            for key in handle.keys()
        }


def audit_checkpoint_shapes() -> dict[str, Any]:
    official_meta = tensor_metadata(OFFICIAL / "model.safetensors")
    sidney_meta = tensor_metadata(SIDNEY / "model.safetensors")
    common = set(official_meta) & set(sidney_meta)
    same_shape = sorted(
        key for key in common if official_meta[key] == sidney_meta[key]
    )
    shape_mismatch = sorted(
        (
            key,
            list(official_meta[key]),
            list(sidney_meta[key]),
        )
        for key in common
        if official_meta[key] != sidney_meta[key]
    )
    official_only = sorted(set(official_meta) - set(sidney_meta))
    sidney_only = sorted(set(sidney_meta) - set(official_meta))
    return {
        "official_keys": len(official_meta),
        "sidney_keys": len(sidney_meta),
        "common_keys": len(common),
        "same_shape_keys": len(same_shape),
        "same_shape_key_list": same_shape,
        "shape_mismatch": [
            {"key": key, "official": official, "sidneyxie": sidney}
            for key, official, sidney in shape_mismatch
        ],
        "official_only": [
            {"key": key, "official": list(official_meta[key])}
            for key in official_only
        ],
        "sidney_only": [
            {"key": key, "sidneyxie": list(sidney_meta[key])}
            for key in sidney_only
        ],
        "_official_meta": official_meta,
        "_sidney_meta": sidney_meta,
    }


def write_report(payload: dict[str, Any], stopped: bool = False) -> None:
    audit = payload["weight_audit"]
    lines = [
        "# Gate3 Weight Swap Under Fixed Official Config",
        "",
        f"- Experiment: `gate3_weight_swap_fixed_config`",
        f"- Parent: `gate3_official_protocol_5x400_armfix`",
        f"- Harness SHA256: `{payload['harness_sha256']}`",
        f"- Dataset sample: episode `{EPISODE}`, position `{POSITION}`, "
        f"video index `{payload.get('video_index', 'not reached')}`",
        "- No simulator rollout was performed.",
        "",
        "## Fixed Configuration",
        "",
        "- Policy config and all processors were loaded from "
        f"`{OFFICIAL}` through `harness.load_policy`.",
        "- Official normalization was required to remain "
        "`VISUAL=IDENTITY, STATE=QUANTILES, ACTION=QUANTILES`.",
        f"- SidneyXie tensors came only from `{SIDNEY / 'model.safetensors'}`.",
        "- No key remapping was performed.",
        "",
        "## Weight-Key And Shape Audit",
        "",
        f"- Official checkpoint keys: `{audit['official_keys']}`",
        f"- SidneyXie checkpoint keys: `{audit['sidney_keys']}`",
        f"- Common keys: `{audit['common_keys']}`",
        f"- Same-shape keys loaded: `{audit['same_shape_keys']}`",
        f"- Common-key shape mismatches: `{len(audit['shape_mismatch'])}`",
        f"- Official-only keys skipped: `{len(audit['official_only'])}`",
        f"- SidneyXie-only keys skipped: `{len(audit['sidney_only'])}`",
        f"- Shape-mismatch fraction among common keys: "
        f"`{len(audit['shape_mismatch']) / max(audit['common_keys'], 1):.4f}`",
        "",
        "### Shape Mismatches",
        "",
    ]
    if audit["shape_mismatch"]:
        for item in audit["shape_mismatch"]:
            lines.append(
                f"- `{item['key']}`: official `{item['official']}` vs "
                f"SidneyXie `{item['sidneyxie']}`; **SKIP**"
            )
    else:
        lines.append("- None.")
    lines.extend(["", "### Official-Only Keys", ""])
    if audit["official_only"]:
        for item in audit["official_only"]:
            lines.append(f"- `{item['key']}` shape `{item['official']}`; **SKIP**")
    else:
        lines.append("- None.")
    lines.extend(["", "### SidneyXie-Only Keys", ""])
    if audit["sidney_only"]:
        for item in audit["sidney_only"]:
            lines.append(
                f"- `{item['key']}` shape `{item['sidneyxie']}`; **SKIP**"
            )
    else:
        lines.append("- None.")

    if stopped:
        lines.extend(
            [
                "",
                "## Pre-Forward Stop",
                "",
                "- **STOPPED** because most common keys disagreed in shape.",
                "- No model construction, oracle, or teacher-forcing forward was run.",
                "",
            ]
        )
    else:
        official_oracle = payload["official_oracle"]
        official_tf = payload["official_teacher_forcing"]
        sidney_oracle = payload["sidneyxie_oracle"]
        sidney_tf = payload["sidneyxie_teacher_forcing"]
        injection = payload["model_inject_audit"]
        lines.extend(
            [
                "",
                "## Model Injection Audit",
                "",
                f"- Official-config policy state keys: `{injection['model_state_keys']}`",
                f"- SidneyXie tensors copied into official-config policy: "
                f"`{injection['loaded_keys']}`",
                f"- Model-only keys left at official initialization: "
                f"`{len(injection['model_only_keys'])}`",
                f"- Model-vs-SidneyXie shape mismatches skipped: "
                f"`{len(injection['model_shape_mismatch'])}`",
                "",
                "### Model-Vs-SidneyXie Shape Mismatches",
                "",
            ]
        )
        if injection["model_shape_mismatch"]:
            for item in injection["model_shape_mismatch"]:
                lines.append(
                    f"- `{item['key']}`: model `{item['model']}` vs "
                    f"SidneyXie `{item['sidneyxie']}`; **SKIP**"
                )
        else:
            lines.append("- None beyond the checkpoint-file audit above.")
        lines.extend(
            [
                "",
                "## Fixed-Observation Measurements",
                "",
                "| weights | final arm L2 | raw arm L2 | t | cosine(v_t, u_t) | MSE(v_t, u_t) | L2(v_t-u_t) |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for label, oracle, tf in (
            ("official", official_oracle, official_tf),
            ("SidneyXie", sidney_oracle, sidney_tf),
        ):
            for time_value in ("0.5", "0.1"):
                metric = tf[time_value]
                lines.append(
                    f"| {label} | `{oracle['final_arm_l2']:.8f}` | "
                    f"`{oracle['raw_arm_l2']:.8f}` | {time_value} | "
                    f"`{metric['cosine']:.8f}` | `{metric['mse']:.8f}` | "
                    f"`{metric['l2']:.8f}` |"
                )
        lines.extend(
            [
                "",
                "## Verdict",
                "",
                f"- **{payload['verdict']}**",
                f"- Named next step: {payload['next_step']}",
                "",
            ]
        )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")


def inject_matching_weights(policy: Any, audit: dict[str, Any]) -> dict[str, Any]:
    model_state = policy.state_dict()
    model_only = sorted(set(model_state) - set(audit["_sidney_meta"]))
    model_shape_mismatch = []
    loaded = []
    with safe_open(
        str(SIDNEY / "model.safetensors"), framework="pt", device="cpu"
    ) as handle:
        for key in sorted(set(model_state) & set(audit["_sidney_meta"])):
            target = model_state[key]
            source_shape = tuple(handle.get_slice(key).get_shape())
            target_shape = tuple(target.shape)
            if source_shape != target_shape:
                model_shape_mismatch.append(
                    {
                        "key": key,
                        "model": list(target_shape),
                        "sidneyxie": list(source_shape),
                    }
                )
                continue
            source = handle.get_tensor(key)
            with torch.no_grad():
                target.copy_(source.to(device=target.device, dtype=target.dtype))
            loaded.append(key)
    return {
        "model_state_keys": len(model_state),
        "loaded_keys": len(loaded),
        "loaded_key_list": loaded,
        "model_only_keys": model_only,
        "model_shape_mismatch": model_shape_mismatch,
    }


def run_oracle(
    policy: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    batch: dict[str, Any],
) -> dict[str, Any]:
    from lerobot.utils.constants import ACTION

    policy.reset()
    heartbeat("oracle_forward_start")
    with torch.inference_mode():
        raw_action = policy.select_action(batch)
    final_action = postprocessor(raw_action)
    final_action = env_postprocessor({ACTION: final_action})[ACTION]
    raw_array = raw_action.detach().float().cpu().numpy().reshape(14)
    final_array = final_action.detach().float().cpu().numpy().reshape(14)
    heartbeat("oracle_forward_done")
    return {
        "raw_action": raw_array.tolist(),
        "final_action": final_array.tolist(),
        "raw_arm_l2": float(np.linalg.norm(raw_array[:6])),
        "final_arm_l2": float(np.linalg.norm(final_array[:6])),
        "finite": bool(np.isfinite(final_array).all()),
    }


def run_teacher_forcing(
    policy: Any,
    preprocessor: Any,
    harness: Any,
    observation: dict[str, Any],
    batch: dict[str, Any],
    sample: dict[str, Any],
) -> dict[str, Any]:
    model = policy.model
    action_norm_14 = teacher_forcing.normalize_action(
        preprocessor,
        harness,
        observation,
        sample["instruction"],
        sample["state"],
        sample["recorded_action"],
    )
    actions_norm = teacher_forcing.pad_action_chunk(action_norm_14, model).to(
        device=next(model.parameters()).device
    )
    prefix_pad_masks, past_key_values, prefix_embs, prefix_meta = (
        teacher_forcing.build_prefix_cache(model, policy, batch)
    )
    torch.manual_seed(SEED)
    noise = model.sample_noise(actions_norm.shape, actions_norm.device).detach().float()
    target = noise - actions_norm
    metrics: dict[str, Any] = {}
    heartbeat("teacher_forcing_prefix_ready")
    with torch.inference_mode():
        for time_value in (0.5, 0.1):
            timestep = torch.full(
                (actions_norm.shape[0],),
                time_value,
                dtype=torch.float32,
                device=actions_norm.device,
            )
            x_t = time_value * noise + (1.0 - time_value) * actions_norm
            velocity = model.denoise_step(
                prefix_pad_masks=prefix_pad_masks,
                past_key_values=past_key_values,
                x_t=x_t,
                timestep=timestep,
            ).detach().float()
            diff = velocity - target
            metrics[f"{time_value:.1f}"] = {
                "mse": float(diff.pow(2).mean()),
                "l2": float(diff.norm()),
                "cosine": float(
                    F.cosine_similarity(
                        velocity.reshape(velocity.shape[0], -1),
                        target.reshape(target.shape[0], -1),
                        dim=1,
                    ).mean()
                ),
            }
            heartbeat(f"teacher_forcing_t={time_value:.1f}_done")
    return {
        **metrics,
        "actions_norm_shape": list(actions_norm.shape),
        "prefix": prefix_meta,
    }


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    heartbeat("start")
    audit = audit_checkpoint_shapes()
    mismatch_fraction = len(audit["shape_mismatch"]) / max(
        audit["common_keys"], 1
    )
    if mismatch_fraction > 0.5:
        payload = {
            "weight_audit": audit,
            "harness_sha256": hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest(),
        }
        write_report(payload, stopped=True)
        print("STOP_MOST_KEYS_SHAPE_MISMATCH=True")
        print(f"SHAPE_MISMATCH_FRACTION={mismatch_fraction:.6f}")
        print(f"REPORT={REPORT_PATH}")
        return 2

    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("weight-swap job requires exactly one visible CUDA device")

    harness = discriminator.load_harness()
    sample = discriminator.load_sample()
    official_args = Namespace(
        model_path=str(OFFICIAL),
        task=TASK,
        episode_steps=400,
        tokenizer_path="/root/autodl-tmp/paligemma_tokenizer",
        task_config="demo_clean",
    )
    heartbeat("load_official_policy_start")
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        harness.load_policy(official_args)
    )
    heartbeat("load_official_policy_done")
    policy.eval()
    normalization = discriminator.normalization_map(policy, preprocessor)
    expected_normalization = {
        "config": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
        "processor": {"VISUAL": "IDENTITY", "STATE": "QUANTILES", "ACTION": "QUANTILES"},
    }
    if normalization != expected_normalization:
        raise RuntimeError(
            f"official processor path changed unexpectedly: {normalization}"
        )

    # FIX: the fixed-observation builder is exposed by gate3_teacher_forcing, not the discriminator.
    observation = teacher_forcing.make_observation(harness, sample)
    batch = harness.make_policy_batch(
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy.config,
    )
    official_oracle = run_oracle(policy, postprocessor, env_postprocessor, batch)
    official_teacher = run_teacher_forcing(
        policy, preprocessor, harness, observation, batch, sample
    )
    heartbeat("official_measurements_done")
    model_inject_audit = inject_matching_weights(policy, audit)
    sidney_oracle = run_oracle(policy, postprocessor, env_postprocessor, batch)
    sidney_teacher = run_teacher_forcing(
        policy, preprocessor, harness, observation, batch, sample
    )
    heartbeat("sidneyxie_measurements_done")
    if sidney_oracle["final_arm_l2"] >= 1.0:
        verdict = "H2w: WEIGHTS fault"
        next_step = (
            "Freeze the official processor/input-path conclusion; obtain or "
            "reconstruct behaviorally aligned official weights/conversion, then "
            "rerun the fixed-config oracle before any rollout."
        )
    elif sidney_oracle["final_arm_l2"] <= 0.5:
        verdict = "H1: OUR CONFIG/INPUT PATH fault"
        next_step = (
            "Audit the official processor/input path against the reference "
            "recipe, prioritizing state discretization and image/language "
            "assembly, then rerun the oracle."
        )
    else:
        verdict = "INCONCLUSIVE: neither H2w nor H1 threshold met"
        next_step = (
            "Resolve the intermediate arm magnitude with a repeated fixed-seed "
            "oracle before selecting a weights or input-path intervention."
        )
    payload = {
        "exp_id": "gate3_weight_swap_fixed_config",
        "parent": "gate3_official_protocol_5x400_armfix",
        "harness_sha256": hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest(),
        "dataset": str(DATA_ROOT),
        "episode": EPISODE,
        "position": POSITION,
        "video_index": int(sample["video_index"]),
        "instruction": sample["instruction"],
        "recorded_arm_l2": float(np.linalg.norm(sample["recorded_action"][:6])),
        "official_reference_arm_l2": 0.0598,
        "sidneyxie_own_config_arm_l2": 3.242,
        "sidneyxie_recorded_arm_l2": 3.294,
        "normalization": normalization,
        "weight_audit": {
            key: value
            for key, value in audit.items()
            if not key.startswith("_")
        },
        "model_inject_audit": model_inject_audit,
        "official_oracle": official_oracle,
        "official_teacher_forcing": official_teacher,
        "sidneyxie_oracle": sidney_oracle,
        "sidneyxie_teacher_forcing": sidney_teacher,
        "verdict": verdict,
        "next_step": next_step,
        "no_rollout": True,
    }
    JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    write_report(payload)
    print(f"EXP_ID={payload['exp_id']}")
    print(f"REPORT={REPORT_PATH}")
    print(f"JSON={JSON_PATH}")
    print(f"MODEL_LOADED_KEYS={model_inject_audit['loaded_keys']}")
    print(f"COMMON_SAME_SHAPE_KEYS={audit['same_shape_keys']}")
    print(f"SHAPE_MISMATCH_KEYS={len(audit['shape_mismatch'])}")
    print(f"OFFICIAL_CONFIG_NORMALIZATION={json.dumps(normalization, sort_keys=True)}")
    print(f"SIDNEYXIE_MISSING_KEYS={json.dumps(audit['official_only'], sort_keys=True)}")
    print(f"SIDNEYXIE_EXTRA_KEYS={json.dumps(audit['sidney_only'], sort_keys=True)}")
    for label, oracle, teacher in (
        ("OFFICIAL", official_oracle, official_teacher),
        ("SIDNEYXIE", sidney_oracle, sidney_teacher),
    ):
        print(f"{label}_ORACLE_FINAL_ARM_L2={oracle['final_arm_l2']:.8f}")
        for time_value in ("0.5", "0.1"):
            metric = teacher[time_value]
            print(
                f"{label}_TEACHER_T={time_value} COSINE={metric['cosine']:.8f} "
                f"MSE={metric['mse']:.8f} L2={metric['l2']:.8f}"
            )
    print(f"VERDICT={verdict}")
    print(f"NEXT_STEP={next_step}")
    print(f"REPORT_SHA256={hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()}")
    del batch, policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor
    gc.collect()
    torch.cuda.empty_cache()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
