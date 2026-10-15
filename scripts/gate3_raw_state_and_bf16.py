#!/usr/bin/env python3
"""Bounded PI05 raw-state-tokenizer and dtype probe.

This probe uses one cached RoboTwin observation (episode 50, position 31),
does not create a simulator, and does not perform a rollout.
"""

from __future__ import annotations

import sys, os; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import copy
import gc
import hashlib
import json
import os
import threading
import time
from argparse import Namespace
from pathlib import Path
from typing import Any

import numpy as np
import torch

import gate3_teacher_forcing as teacher_forcing
import gate3_two_checkpoint_discriminator as discriminator
import gate3_weight_swap_fixed_config as weight_swap


ROOT = Path("/root/autodl-tmp/EvolveVLA")
SCRIPTS = ROOT / "scripts"
HARNESS_PATH = ROOT / "scripts/robotwin_eval_harness.py"
DATA_ROOT = ROOT / "data/gate2_robotwin_cache/clean"
OFFICIAL = ROOT / "checkpoints/pi05_robotwin_lerobot"
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
REPORT_PATH = ROOT / "results/gate3_raw_state_and_bf16.md"
JSON_PATH = ROOT / "results/gate3_raw_state_and_bf16.json"
EPISODE = 50
POSITION = 31
TASK = "beat_block_hammer"
SEED = 20261003


def proof(message: str) -> None:
    """Write a progress-keyed log heartbeat for the managed watchdog."""
    log_path = os.environ.get("HEARTBEAT_LOG")
    if not log_path:
        return
    with open(log_path, "a", encoding="utf-8") as handle:
        handle.write(f"HEARTBEAT {message}\n")
        handle.flush()


def start_watchdog() -> None:
    """Fail if the managed log stops receiving progress for 120 seconds."""
    log_path = os.environ.get("HEARTBEAT_LOG")
    if not log_path:
        return
    Path(log_path).touch()

    def watch() -> None:
        while True:
            time.sleep(15)
            try:
                age = time.time() - Path(log_path).stat().st_mtime
            except OSError:
                continue
            if age > 120:
                print(f"WATCHDOG_FAIL log_mtime_age={age:.1f}", flush=True)
                os._exit(124)

    threading.Thread(target=watch, name="gate3-log-watchdog", daemon=True).start()


def assert_attrs() -> None:
    checks = [
        (discriminator, "load_harness"),
        (discriminator, "load_sample"),
        (teacher_forcing, "make_observation"),
        (weight_swap, "run_oracle"),
    ]
    for module, name in checks:
        if not hasattr(module, name):
            raise AttributeError(
                f"cross-module smoke failed: {module.__name__}.{name}"
            )
    proof("hasattr_checks_pass")


def checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    return {
        name: hashlib.sha256((checkpoint / name).read_bytes()).hexdigest()
        for name in ("config.json", "policy_preprocessor.json", "norm_stats.json")
        if (checkpoint / name).is_file()
    }


def policy_args() -> Namespace:
    return Namespace(
        model_path=str(OFFICIAL),
        task=TASK,
        episode_steps=400,
        tokenizer_path=str(TOKENIZER_PATH),
        task_config="demo_clean",
    )


def state_token_steps(preprocessor: Any) -> tuple[list[Any], int, int, int]:
    steps = list(preprocessor.steps)
    names = [step.__class__.__name__ for step in steps]

    def find(name: str) -> int:
        matches = [index for index, value in enumerate(names) if value == name]
        if len(matches) != 1:
            raise RuntimeError(f"expected one {name}, got {matches}; steps={names}")
        return matches[0]

    return (
        steps,
        find("NormalizerProcessorStep"),
        find("Pi05PrepareStateTokenizerProcessorStep"),
        find("TokenizerProcessorStep"),
    )


def make_policy_batch_variant(
    harness: Any,
    observation: dict[str, Any],
    instruction: str,
    env_preprocessor: Any,
    preprocessor: Any,
    policy_config: Any,
    raw_state_before_normalize: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Use the harness batch builder, optionally moving only state tokenization."""
    steps, norm_idx, prepare_idx, tokenizer_idx = state_token_steps(preprocessor)
    active_preprocessor = preprocessor
    if raw_state_before_normalize:
        if prepare_idx <= norm_idx:
            raise RuntimeError(
                "expected normalized production order with state tokenization "
                f"after normalization: norm={norm_idx}, prepare={prepare_idx}"
            )
        raw_steps = list(steps)
        state_token_step = raw_steps.pop(prepare_idx)
        raw_steps.insert(norm_idx, state_token_step)
        active_preprocessor = copy.copy(preprocessor)
        active_preprocessor.steps = raw_steps

    batch = harness.make_policy_batch(
        copy.deepcopy(observation),
        instruction,
        env_preprocessor,
        active_preprocessor,
        policy_config,
    )
    if "observation.language.tokens" not in batch:
        raise RuntimeError(
            "tokenizer output missing observation.language.tokens; "
            f"keys={sorted(map(str, batch))}"
        )
    return batch, {
        "source": "harness.make_policy_batch",
        "step_names": [step.__class__.__name__ for step in steps],
        "active_step_names": [
            step.__class__.__name__ for step in active_preprocessor.steps
        ],
        "normalizer_index": norm_idx,
        "state_tokenizer_index": prepare_idx,
        "tokenizer_index": tokenizer_idx,
        "raw_state_before_normalize": raw_state_before_normalize,
    }


def batch_token_ids(batch: dict[str, Any]) -> list[int]:
    tokens = batch["observation.language.tokens"]
    return tokens.detach().cpu().reshape(-1).tolist()


def measure_oracle(
    label: str,
    policy: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    batch: dict[str, Any],
) -> dict[str, Any]:
    proof(f"{label}_forward_start")
    torch.manual_seed(SEED)
    result = weight_swap.run_oracle(
        policy,
        postprocessor,
        env_postprocessor,
        batch,
    )
    result["label"] = label
    result["model_dtype"] = str(next(policy.model.parameters()).dtype)
    proof(f"{label}_forward_done")
    return result


def run_official_variants(
    harness: Any,
    sample: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], Any]:
    proof("load_fp32_start")
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        harness.load_policy(policy_args())
    )
    policy.eval()
    observation = teacher_forcing.make_observation(harness, sample)
    normalized_batch, normalized_trace = make_policy_batch_variant(
        harness,
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy.config,
        raw_state_before_normalize=False,
    )
    raw_batch, raw_trace = make_policy_batch_variant(
        harness,
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy.config,
        raw_state_before_normalize=True,
    )
    normalized = measure_oracle(
        "normalized_fp32",
        policy,
        postprocessor,
        env_postprocessor,
        normalized_batch,
    )
    raw = measure_oracle(
        "raw_state_fp32",
        policy,
        postprocessor,
        env_postprocessor,
        raw_batch,
    )
    normalized["construction"] = normalized_trace
    raw["construction"] = raw_trace
    normalized["language_token_ids"] = batch_token_ids(normalized_batch)
    raw["language_token_ids"] = batch_token_ids(raw_batch)
    normalized["token_ids_differ_from_raw"] = (
        normalized["language_token_ids"] != raw["language_token_ids"]
    )
    raw["token_ids_differ_from_normalized"] = normalized[
        "token_ids_differ_from_raw"
    ]
    raw["normalized_state_after_pipeline"] = (
        raw_batch["observation.state"].detach().float().cpu().reshape(-1).tolist()
    )
    normalized["normalized_state_after_pipeline"] = (
        normalized_batch["observation.state"]
        .detach()
        .float()
        .cpu()
        .reshape(-1)
        .tolist()
    )
    del normalized_batch, raw_batch, observation
    del policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor
    gc.collect()
    torch.cuda.empty_cache()
    proof("fp32_variants_complete")

    proof("load_bf16_start")
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        harness.load_policy(policy_args())
    )
    policy.model.to(dtype=torch.bfloat16)
    policy.eval()
    observation = teacher_forcing.make_observation(harness, sample)
    bf16_batch = harness.make_policy_batch(
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy.config,
    )
    bf16 = measure_oracle(
        "official_bf16",
        policy,
        postprocessor,
        env_postprocessor,
        bf16_batch,
    )
    bf16["construction"] = {
        "source": "harness.make_policy_batch",
        "state_tokenizer_semantics": "current normalized-state path",
        "model_dtype_override": "torch.bfloat16",
    }
    bf16["language_token_ids"] = batch_token_ids(bf16_batch)
    del bf16_batch, observation
    del policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor
    gc.collect()
    torch.cuda.empty_cache()
    proof("bf16_complete")
    return normalized, raw, bf16


def classify(value: float) -> str:
    return "action-scale" if value >= 2.5 else "not-action-scale"


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    start_watchdog()
    proof("start")
    assert_attrs()
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("probe requires exactly one visible CUDA device")
    if torch.cuda.current_device() != 0:
        raise RuntimeError("probe requires cuda:0")

    harness = discriminator.load_harness()
    sample = discriminator.load_sample()
    if not hasattr(harness, "load_policy"):
        raise AttributeError("cross-module smoke failed: harness.load_policy")
    if not hasattr(harness, "make_policy_batch"):
        raise AttributeError("cross-module smoke failed: harness.make_policy_batch")
    if not hasattr(harness, "simulator_observation"):
        raise AttributeError(
            "cross-module smoke failed: harness.simulator_observation"
        )
    if not hasattr(harness, "derive_camera_rename_map"):
        raise AttributeError(
            "cross-module smoke failed: harness.derive_camera_rename_map"
        )
    observation = teacher_forcing.make_observation(harness, sample)
    del observation
    proof("runtime_attr_checks_pass")

    normalized, raw, bf16 = run_official_variants(harness, sample)
    raw_scale = classify(raw["final_arm_l2"])
    bf16_scale = classify(bf16["final_arm_l2"])
    raw_restores = raw["final_arm_l2"] >= 2.5
    bf16_restores = bf16["final_arm_l2"] >= 2.5
    if raw_restores and bf16_restores:
        verdict = "RAW-STATE DISCRETIZATION AND BF16 BOTH RESTORE ACTION SCALE"
        next_step = (
            "Use the raw-state-before-normalization processor order as the "
            "official input-path fix, then rerun the fixed-seed oracle and "
            "the serialized 5x400 official protocol."
        )
    elif raw_restores:
        verdict = "RAW-STATE DISCRETIZATION RESTORES ACTION SCALE; BF16 DOES NOT"
        next_step = (
            "Adopt raw-state-before-normalization token construction, then "
            "rerun the fixed-seed oracle and serialized 5x400 official protocol."
        )
    elif bf16_restores:
        verdict = "BF16 RESTORES ACTION SCALE; RAW-STATE DISCRETIZATION DOES NOT"
        next_step = (
            "Make dtype match the OpenPI reference in the official checkpoint "
            "loader, then rerun the fixed-seed oracle before any rollout."
        )
    else:
        verdict = "NEITHER RAW-STATE DISCRETIZATION NOR BF16 RESTORES ACTION SCALE"
        next_step = (
            "Keep both H1 leads open; next isolate the official action-head "
            "conversion/inference semantics with the same raw-state construction."
        )

    payload = {
        "exp_id": "gate3_raw_state_and_bf16",
        "parent": "gate3_official_protocol_5x400_armfix",
        "dataset": str(DATA_ROOT),
        "episode": EPISODE,
        "position": POSITION,
        "video_index": int(sample["video_index"]),
        "instruction": sample["instruction"],
        "recorded_state": np.asarray(sample["state"]).reshape(14).tolist(),
        "recorded_action": np.asarray(sample["recorded_action"]).reshape(14).tolist(),
        "recorded_arm_l2": float(np.linalg.norm(sample["recorded_action"][:6])),
        "results": {
            "normalized_state_official_fp32": normalized,
            "raw_state_official_fp32": raw,
            "official_bf16": bf16,
        },
        "construction_source": {
            "working_weight_swap": str(SCRIPTS / "gate3_weight_swap_fixed_config.py"),
            "working_teacher_forcing": str(SCRIPTS / "gate3_teacher_forcing.py"),
            "harness": str(HARNESS_PATH),
            "processor_reference": (
                "/root/autodl-tmp/conda_envs/evolvevla_pi312/"
                "lib/python3.12/site-packages/lerobot/policies/pi05/processor_pi05.py"
            ),
            "current_lines": "processor_pi05.py:136-145",
            "discretization_lines": "processor_pi05.py:42-80",
            "harness_lines": "robotwin_eval_harness.py:215-235",
            "observation_lines": "gate3_teacher_forcing.py:119-128",
            "difference": (
                "Normalized arm: production order NormalizerProcessorStep -> "
                "Pi05PrepareStateTokenizerProcessorStep -> TokenizerProcessorStep. "
                "Raw arm: the same Pi05PrepareStateTokenizerProcessorStep is "
                "invoked on raw observation.state immediately before the normalizer; "
                "the normalizer then runs before the unchanged tokenizer/device "
                "steps. Images, instruction, tokenizer, model, seed, and "
                "postprocessing are unchanged."
            ),
        },
        "source_sha256": {
            "probe": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "teacher_forcing": hashlib.sha256(
                (SCRIPTS / "gate3_teacher_forcing.py").read_bytes()
            ).hexdigest(),
            "discriminator": hashlib.sha256(
                (SCRIPTS / "gate3_two_checkpoint_discriminator.py").read_bytes()
            ).hexdigest(),
            "weight_swap": hashlib.sha256(
                (SCRIPTS / "gate3_weight_swap_fixed_config.py").read_bytes()
            ).hexdigest(),
            "harness": hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest(),
        },
        "classification": {
            "normalized_state_official_fp32": classify(normalized["final_arm_l2"]),
            "raw_state_official_fp32": raw_scale,
            "official_bf16": bf16_scale,
        },
        "verdict": verdict,
        "next_step": next_step,
        "no_rollout": True,
        "no_perception_or_resolution_probe": True,
    }
    lines = [
        "# Gate3 Raw-State Discretization And BF16 Official Oracle",
        "",
        f"- Experiment: `{payload['exp_id']}`",
        f"- Parent: `{payload['parent']}`",
        f"- Sample: episode `{EPISODE}`, position `{POSITION}`, video index `{payload['video_index']}`",
        f"- Recorded arm L2: `{payload['recorded_arm_l2']:.8f}`",
        "- Same real frame for all three forwards; no rollout, perception, or resolution probe.",
        "",
        "## Mandatory Import Smoke",
        "",
        "- The managed launcher ran `python -m py_compile scripts/gate3_raw_state_and_bf16.py` before model load.",
        "- The managed launcher ran `hasattr` checks for every imported-module attribute called by the probe before model load.",
        "- Runtime checks also verified `harness.load_policy`, `harness.make_policy_batch`, and `harness.simulator_observation`.",
        "",
        "## Arm L2",
        "",
        "| construction | dtype | final arm L2 | classification |",
        "|---|---|---:|---|",
        f"| normalized-state official | `{normalized['model_dtype']}` | `{normalized['final_arm_l2']:.8f}` | `{classify(normalized['final_arm_l2'])}` |",
        f"| raw-state official | `{raw['model_dtype']}` | `{raw['final_arm_l2']:.8f}` | `{raw_scale}` |",
        f"| normalized-state official | `{bf16['model_dtype']}` | `{bf16['final_arm_l2']:.8f}` | `{bf16_scale}` |",
        "",
        "## Construction Difference",
        "",
        f"- Key construction: `{payload['construction_source']['current_lines']}` and `{payload['construction_source']['discretization_lines']}`.",
        f"- Observation/batch construction: `{payload['construction_source']['observation_lines']}` and `{payload['construction_source']['harness_lines']}`.",
        f"- Difference: {payload['construction_source']['difference']}",
        f"- Normalized processor order: `{normalized['construction']['active_step_names']}`",
        f"- Raw processor order: `{raw['construction']['active_step_names']}`",
        f"- Language token IDs differ: `{normalized['token_ids_differ_from_raw']}`",
        "",
        "## Provenance",
        "",
        f"- Source SHA256: `{json.dumps(payload['source_sha256'], sort_keys=True)}`",
        f"- Official checkpoint files: `{json.dumps(checkpoint_hashes(OFFICIAL), sort_keys=True)}`",
        "",
        "## Verdict",
        "",
        f"- Normalized-state official: `{normalized['final_arm_l2']:.8f}`.",
        f"- Raw-state official: `{raw['final_arm_l2']:.8f}`.",
        f"- BF16 official: `{bf16['final_arm_l2']:.8f}`.",
        f"- **VERDICT: `{verdict}`**",
        f"- Named next step: {next_step}",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    JSON_PATH.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    proof("report_written")
    print(f"SMOKE_PY_COMPILE=see launcher exit code")
    print(f"SMOKE_HASATTR=see launcher exit code")
    print(f"NORMALIZED_OFFICIAL_ARM_L2={normalized['final_arm_l2']:.8f}")
    print(f"RAW_STATE_OFFICIAL_ARM_L2={raw['final_arm_l2']:.8f}")
    print(f"BF16_OFFICIAL_ARM_L2={bf16['final_arm_l2']:.8f}")
    print(f"VERDICT={verdict}")
    print(f"REPORT={REPORT_PATH}")
    print(f"REPORT_SHA256={hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()}")
    return 0


def cpu_construction_dry_run() -> int:
    """Build both policy batches on CPU without constructing or loading a model."""
    if torch.cuda.is_available():
        raise RuntimeError(
            "CPU construction dry-run requires CUDA to be hidden; "
            "run with CUDA_VISIBLE_DEVICES empty"
        )

    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    harness = discriminator.load_harness()
    sample = discriminator.load_sample()
    env_cfg = RoboTwinEnvConfig(
        task=TASK,
        episode_length=400,
        obs_type="pixels_agent_pos",
        camera_names="head_camera,left_camera,right_camera",
        observation_height=harness.OBS_HEIGHT,
        observation_width=harness.OBS_WIDTH,
        action_mode="joint",
    )
    policy_cfg = PI05Config.from_pretrained(str(OFFICIAL), local_files_only=True)
    policy_cfg.device = "cpu"
    rename_map = harness.derive_camera_rename_map(policy_cfg)
    preprocessor, _ = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(OFFICIAL),
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(TOKENIZER_PATH)},
        },
    )
    env_preprocessor, _ = make_env_pre_post_processors(env_cfg, policy_cfg)
    observation = teacher_forcing.make_observation(harness, sample)
    normalized_batch, _ = make_policy_batch_variant(
        harness,
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy_cfg,
        raw_state_before_normalize=False,
    )
    raw_batch, _ = make_policy_batch_variant(
        harness,
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy_cfg,
        raw_state_before_normalize=True,
    )
    if not normalized_batch or not raw_batch:
        raise RuntimeError("CPU construction returned an empty batch")
    if normalized_batch["observation.language.tokens"].numel() == 0:
        raise RuntimeError("CPU construction returned empty language tokens")
    print("CPU_CONSTRUCTION_DRY_RUN=PASS")
    print(f"CPU_NORMALIZED_BATCH_KEYS={sorted(map(str, normalized_batch))}")
    print(f"CPU_RAW_BATCH_KEYS={sorted(map(str, raw_batch))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
