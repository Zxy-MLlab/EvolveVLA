#!/usr/bin/env python3
"""CPU loadability and single-observation smoke test for SidneyXie's pi05 checkpoint."""

from __future__ import annotations

import contextlib
import io
import json
from pathlib import Path
from typing import Any

import torch

from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.modeling_pi05 import PI05Policy


CHECKPOINT_DIR = Path(
    "/root/autodl-tmp/EvolveVLA/checkpoints/sidneyxie_pi05_robotwin"
)
TOKENIZER_DIR = Path("/root/autodl-tmp/paligemma_tokenizer")

# This is the inverse direction of robotwin_obs_bridge.py's dataset-to-wrapper map.
BRIDGE_CAMERA_KEYS = {
    "observation.images.cam_high": "observation.images.head_camera",
    "observation.images.cam_left_wrist": "observation.images.left_camera",
    "observation.images.cam_right_wrist": "observation.images.right_camera",
}


def feature_shape(feature: Any) -> tuple[int, ...]:
    shape = getattr(feature, "shape", None)
    if shape is None:
        raise ValueError(f"Input feature has no shape: {feature!r}")
    return tuple(int(dim) for dim in shape)


def describe_input_features(config: PI05Config) -> dict[str, list[int]]:
    described: dict[str, list[int]] = {}
    for key, feature in config.input_features.items():
        described[str(key)] = list(feature_shape(feature))
    return described


def build_synthetic_observation(config: PI05Config) -> dict[str, Any]:
    observation: dict[str, Any] = {
        "task": ["synthetic SidneyXie RobotWin load test"],
    }
    for raw_key, feature in config.input_features.items():
        key = str(raw_key)
        shape = feature_shape(feature)
        if key == "observation.state":
            if shape != (14,):
                raise ValueError(f"Expected state shape [14], got {shape} for {key}")
            observation[key] = torch.zeros(shape, dtype=torch.float32)
        elif key.startswith("observation.images."):
            if len(shape) != 3 or 3 not in (shape[0], shape[-1]):
                raise ValueError(f"Expected RGB image shape, got {shape} for {key}")
            if shape[0] == 3:
                chw_shape = shape
            else:
                chw_shape = (3, shape[0], shape[1])
            # LeRobot source: processor/batch_processor.py:116-119 adds the batch
            # dimension; policies/pi05/modeling_pi05.py:985-1001 expects CHW.
            observation[key] = torch.zeros(chw_shape, dtype=torch.float32)
        else:
            raise ValueError(f"Unsupported declared input feature: {key} {shape}")
    return observation


def print_load_diagnostics(captured: str) -> None:
    lines = [line.rstrip() for line in captured.splitlines() if line.strip()]
    diagnostic_lines = [
        line
        for line in lines
        if (
            "Missing keys" in line
            or "Unexpected keys" in line
            or "All keys loaded successfully" in line
            or "Could not load state dict" in line
            or "Returning model without loading pretrained weights" in line
            or line.startswith("  - ")
            or line.startswith("  ... and ")
        )
    ]
    print("LOAD_KEY_DIAGNOSTICS_BEGIN")
    if diagnostic_lines:
        print("\n".join(diagnostic_lines))
    else:
        print("No load-key diagnostics returned by PI05Policy.from_pretrained")
    print("LOAD_KEY_DIAGNOSTICS_END")


def main() -> bool:
    config_ok = False
    params_ok = False
    forward_ok = False
    policy = None
    config: PI05Config | None = None

    try:
        config = PI05Config.from_pretrained(CHECKPOINT_DIR, local_files_only=True)
        config_ok = True
        print(f"CONFIG_OK type={type(config).__name__}")
        print(
            "INPUT_FEATURES="
            + json.dumps(describe_input_features(config), sort_keys=True)
        )

        # The checkpoint was trained for CUDA/bfloat16 and requests compilation.
        # Override those runtime choices for this CPU-only load test.
        config.device = "cpu"
        config.dtype = "float32"
        config.compile_model = False
        config.gradient_checkpointing = False

        captured_stdout = io.StringIO()
        captured_stderr = io.StringIO()
        with contextlib.redirect_stdout(
            captured_stdout
        ), contextlib.redirect_stderr(captured_stderr):
            policy = PI05Policy.from_pretrained(
                CHECKPOINT_DIR,
                config=config,
                local_files_only=True,
            )
        print_load_diagnostics(captured_stdout.getvalue() + captured_stderr.getvalue())
        print("POLICY_LOADED_OK")

        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        params_ok = parameter_count > 0
        print(f"PARAM_COUNT={parameter_count}")

        preprocessor, _ = make_pre_post_processors(
            policy_cfg=config,
            pretrained_path=str(CHECKPOINT_DIR),
            preprocessor_overrides={
                "tokenizer_processor": {
                    "tokenizer_name": str(TOKENIZER_DIR),
                },
                "device_processor": {
                    "device": "cpu",
                },
            },
        )
        raw_observation = build_synthetic_observation(config)
        print("IMAGE_INPUT_LAYOUT=CHW_FLOAT32")
        print(
            "SYNTHETIC_OBSERVATION_KEYS="
            + json.dumps(sorted(raw_observation.keys()))
        )
        print(f"TOKENIZER_OVERRIDE={TOKENIZER_DIR}")

        batch = preprocessor(raw_observation)
        policy.reset()
        policy.eval()
        with torch.no_grad():
            actions = policy.select_action(batch)
        finite_actions = bool(torch.isfinite(actions).all().item())
        print(f"ACTION_SHAPE={tuple(actions.shape)}")
        print(f"ACTION_FINITE={finite_actions}")
        forward_ok = True
        print(f"FORWARD_OK={forward_ok}")
    except Exception as exc:
        print(f"LOAD_TEST_ERROR type={type(exc).__name__}: {exc}")
        print("FORWARD_OK=False")

    fully_loadable = config_ok and params_ok and forward_ok
    print(f"SIDNEYXIE_FULLY_LOADABLE={fully_loadable}")
    print(f"FORWARD_OK={forward_ok}")
    return fully_loadable


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
