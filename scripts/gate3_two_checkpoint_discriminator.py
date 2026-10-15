#!/usr/bin/env python3
"""Compare two PI05 checkpoints on one cached RoboTwin observation.

This is a bounded discriminator probe. It does not create a simulator, roll
out an episode, or modify the shipped harness. Each checkpoint is loaded with
its own config and processor files through the harness load path.
"""

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

import av
import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path("/root/autodl-tmp/EvolveVLA")
HARNESS_PATH = ROOT / "scripts/robotwin_eval_harness.py"
DATA_ROOT = ROOT / "data/gate2_robotwin_cache/clean"
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
REPORT_PATH = ROOT / "results/gate3_two_checkpoint_discriminator.md"
EPISODE = 50
POSITION = 31
TASK = "beat_block_hammer"

CHECKPOINTS = {
    "official": ROOT / "checkpoints/pi05_robotwin_lerobot",
    "SidneyXie": ROOT / "checkpoints/sidneyxie_pi05_robotwin",
}


def start_heartbeat() -> None:
    """Keep the managed log fresh during a long single forward."""
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

    threading.Thread(target=beat, name="log-heartbeat", daemon=True).start()

CAMERA_FILES = {
    "head_camera": DATA_ROOT
    / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
    "left_camera": DATA_ROOT
    / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
    "right_camera": DATA_ROOT
    / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
}


def load_harness():
    if str(HARNESS_PATH.parent) not in sys.path:
        sys.path.insert(0, str(HARNESS_PATH.parent))
    spec = importlib.util.spec_from_file_location(
        "robotwin_eval_harness", HARNESS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import harness from {HARNESS_PATH}")
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


def enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    if name:
        return str(name)
    text = str(value)
    return text.rsplit(".", 1)[-1]


def normalization_map(policy: Any, preprocessor: Any) -> dict[str, str]:
    config_map = getattr(policy.config, "normalization_mapping", {}) or {}
    config_map = {
        enum_name(key): enum_name(value)
        for key, value in config_map.items()
    }
    norm_step = next(
        (
            step
            for step in preprocessor.steps
            if step.__class__.__name__ == "NormalizerProcessorStep"
        ),
        None,
    )
    step_map = getattr(norm_step, "norm_map", {}) if norm_step else {}
    step_map = {
        enum_name(key): enum_name(value)
        for key, value in step_map.items()
    }
    return {
        "config": config_map,
        "processor": step_map,
    }


def checkpoint_hashes(checkpoint: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for name in ("config.json", "policy_preprocessor.json", "norm_stats.json"):
        path = checkpoint / name
        if path.is_file():
            hashes[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def load_sample() -> dict[str, Any]:
    data_path = DATA_ROOT / "data/chunk-000/file-000.parquet"
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

    tasks = pq.read_table(DATA_ROOT / "meta/tasks.parquet").to_pydict()
    task_text = dict(zip(tasks["task_index"], tasks["task"]))
    instruction = str(task_text[int(row["task_index"])])
    state = np.asarray(row["observation.state"], dtype=np.float32).reshape(14)
    action = np.asarray(row["action"], dtype=np.float32).reshape(14)
    video_index = int(row["index"])
    decoded = {
        camera: decode_frame(path, video_index)
        for camera, path in CAMERA_FILES.items()
    }
    return {
        "instruction": instruction,
        "state": state,
        "recorded_action": action,
        "video_index": video_index,
        "decoded": decoded,
        "row": row,
    }


def fmt(values: Any, precision: int = 8) -> str:
    array = (
        values.detach().float().cpu().numpy()
        if torch.is_tensor(values)
        else np.asarray(values)
    )
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in array.reshape(-1)) + "]"


def run_checkpoint(
    label: str,
    checkpoint: Path,
    sample: dict[str, Any],
    harness: Any,
) -> dict[str, Any]:
    args = Namespace(
        model_path=str(checkpoint),
        task=TASK,
        episode_steps=400,
        tokenizer_path=str(TOKENIZER_PATH),
        task_config="demo_clean",
    )
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        harness.load_policy(args)
    )
    policy.eval()
    observation = harness.simulator_observation(
        {
            "observation": {
                camera: {"rgb": sample["decoded"][camera]}
                for camera in ("head_camera", "left_camera", "right_camera")
            },
            "joint_action": {"vector": sample["state"].tolist()},
        }
    )
    batch = harness.make_policy_batch(
        observation,
        sample["instruction"],
        env_preprocessor,
        preprocessor,
        policy.config,
    )

    from lerobot.utils.constants import ACTION

    policy.reset()
    with torch.inference_mode():
        raw_action = policy.select_action(batch)
    final_action = postprocessor(raw_action)
    final_action = env_postprocessor({ACTION: final_action})[ACTION]
    raw_array = raw_action.detach().float().cpu().numpy().reshape(14)
    final_array = final_action.detach().float().cpu().numpy().reshape(14)
    normalization = normalization_map(policy, preprocessor)
    result = {
        "checkpoint": str(checkpoint),
        "normalization": normalization,
        "hashes": checkpoint_hashes(checkpoint),
        "raw_action": raw_array.tolist(),
        "final_action": final_array.tolist(),
        "raw_arm_l2": float(np.linalg.norm(raw_array[:6])),
        "final_arm_l2": float(np.linalg.norm(final_array[:6])),
        "finite": bool(np.isfinite(final_array).all()),
        "batch_shapes": {
            str(key): list(value.shape)
            for key, value in batch.items()
            if torch.is_tensor(value)
            and "observation.images." in str(key)
        },
    }
    del batch, observation, policy, preprocessor, postprocessor
    del env_preprocessor, env_postprocessor
    gc.collect()
    torch.cuda.empty_cache()
    return result


def verdict(results: dict[str, dict[str, Any]]) -> tuple[str, str]:
    official = results["official"]["final_arm_l2"]
    sidney = results["SidneyXie"]["final_arm_l2"]
    if official < 0.5 and sidney >= 0.5:
        return (
            "(B) official conversion/normalization fault",
            "SidneyXie is materially action-like while official remains near mean; "
            "re-derive the official checkpoint's true QUANTILES mapping or obtain "
            "its exact inference config.",
        )
    if official < 0.5 and sidney < 0.5:
        return (
            "(A) harness inference broken",
            "Both checkpoints remain near mean on the same observation; compare "
            "the harness batch and flow-sampling path against the expected input.",
        )
    return (
        "INCONCLUSIVE",
        "The fixed magnitude split was not observed; inspect the raw report before "
        "choosing the next intervention.",
    )


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    start_heartbeat()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(
            f"expected one visible GPU, got {torch.cuda.device_count()}"
        )
    harness = load_harness()
    sample = load_sample()
    recorded_arm_l2 = float(np.linalg.norm(sample["recorded_action"][:6]))
    results = {
        label: run_checkpoint(label, checkpoint, sample, harness)
        for label, checkpoint in CHECKPOINTS.items()
    }
    decision, next_fix = verdict(results)
    sidney_norm = results["SidneyXie"]["normalization"]
    sidney_mean_std = (
        sidney_norm["config"].get("STATE") == "MEAN_STD"
        and sidney_norm["config"].get("ACTION") == "MEAN_STD"
        and sidney_norm["processor"].get("STATE") == "MEAN_STD"
        and sidney_norm["processor"].get("ACTION") == "MEAN_STD"
    )
    payload = {
        "harness_sha256": hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest(),
        "dataset": str(DATA_ROOT),
        "episode": EPISODE,
        "position": POSITION,
        "video_index": sample["video_index"],
        "instruction": sample["instruction"],
        "recorded_action": sample["recorded_action"].tolist(),
        "recorded_arm_l2": recorded_arm_l2,
        "results": results,
        "sidneyxie_mean_std_correct": sidney_mean_std,
        "verdict": decision,
        "minimal_next_fix": next_fix,
    }
    lines = [
        "# Gate3 Two-Checkpoint Discriminator",
        "",
        f"- Harness: `{HARNESS_PATH}`",
        f"- Harness SHA256: `{payload['harness_sha256']}`",
        f"- Dataset: `{DATA_ROOT}`",
        f"- Sample: episode `{EPISODE}`, position `{POSITION}`, video index `{sample['video_index']}`",
        f"- Instruction: `{sample['instruction']}`",
        f"- Recorded action: `{fmt(sample['recorded_action'])}`",
        f"- Recorded arm L2: `{recorded_arm_l2:.8f}`",
        "",
        "## Checkpoint Results",
        "",
    ]
    for label, result in results.items():
        lines.extend(
            [
                f"### {label}",
                f"- Checkpoint: `{result['checkpoint']}`",
                f"- Config/pipeline normalization: `{result['normalization']}`",
                f"- Config/pipeline files: `{result['hashes']}`",
                f"- Batch image shapes: `{result['batch_shapes']}`",
                f"- Raw action: `{fmt(result['raw_action'])}`",
                f"- Final action: `{fmt(result['final_action'])}`",
                f"- Raw arm L2: `{result['raw_arm_l2']:.8f}`",
                f"- Final arm L2: `{result['final_arm_l2']:.8f}`",
                f"- Finite: `{result['finite']}`",
                "",
            ]
        )
    lines.extend(
        [
            "## Processor Check",
            "",
            f"- SidneyXie loaded with its own MEAN_STD processors: `{sidney_mean_std}`",
            "",
            "## Verdict",
            "",
            f"- **{decision}**",
            f"- Minimal next fix: {next_fix}",
            "",
            "No simulator rollout was performed.",
            "",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(f"REPORT={REPORT_PATH}")
    print(f"HARNESS_SHA256={payload['harness_sha256']}")
    print(f"VIDEO_INDEX={sample['video_index']}")
    print(f"RECORDED_ARM_L2={recorded_arm_l2:.8f}")
    for label, result in results.items():
        print(
            f"{label.upper()}_NORMALIZATION={json.dumps(result['normalization'], sort_keys=True)}"
        )
        print(f"{label.upper()}_FINAL_ARM_L2={result['final_arm_l2']:.8f}")
        print(f"{label.upper()}_FINAL_ACTION={fmt(result['final_action'])}")
    print(f"SIDNEYXIE_MEAN_STD_CORRECT={sidney_mean_std}")
    print(f"VERDICT={decision}")
    print(f"MINIMAL_NEXT_FIX={next_fix}")
    print(f"REPORT_SHA256={hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
