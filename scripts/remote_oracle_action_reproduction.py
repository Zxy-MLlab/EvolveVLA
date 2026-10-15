#!/usr/bin/env python3
"""Reproduce pi0.5 actions on real LeRobot frames through the eval harness path."""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch


REMOTE_ROOT = Path("/root/autodl-tmp/EvolveVLA")
HARNESS_PATH = REMOTE_ROOT / "scripts/robotwin_eval_harness.py"
DATA_ROOT = REMOTE_ROOT / "data/gate2_robotwin_cache/clean"
CHECKPOINT = REMOTE_ROOT / "checkpoints/pi05_robotwin_lerobot"
TOKENIZER = Path("/root/autodl-tmp/paligemma_tokenizer")
OUTPUT = REMOTE_ROOT / "results/gate3_oracle_action_reproduction.md"
EPISODE = 50
CAMERA_FILES = {
    "head_camera": DATA_ROOT / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
    "left_camera": DATA_ROOT / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
    "right_camera": DATA_ROOT / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
}


def load_harness():
    scripts_dir = str(HARNESS_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location("robotwin_eval_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import harness from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode_frames(video_path: Path, frame_indices: list[int]) -> dict[int, np.ndarray]:
    wanted = set(frame_indices)
    found: dict[int, np.ndarray] = {}
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index in wanted:
                found[index] = frame.to_rgb().to_ndarray()
            if len(found) == len(wanted):
                break
    missing = sorted(wanted - set(found))
    if missing:
        raise RuntimeError(f"{video_path}: missing video frames {missing}")
    return found


def source_lines(path: Path, start: int, end: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [f"{index}:{lines[index - 1]}" for index in range(start, end + 1)]


def fmt_vector(values: np.ndarray | list[float]) -> str:
    return "[" + ", ".join(f"{float(value): .5f}" for value in values) + "]"


def fmt_bool(value: bool) -> str:
    return "yes" if value else "no"


def main() -> int:
    harness = load_harness()

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
    task_data = pq.read_table(DATA_ROOT / "meta/tasks.parquet").to_pydict()
    task_text = dict(zip(task_data["task_index"], task_data["task"]))

    rows = [
        {
            key: data[key][row]
            for key in data
        }
        for row in range(len(data["episode_index"]))
        if data["episode_index"][row] == EPISODE
    ]
    if not rows:
        raise RuntimeError(f"episode {EPISODE} not found")
    rows.sort(key=lambda row: row["frame_index"])
    selected_positions = sorted(
        set([0, len(rows) // 4, len(rows) // 2, (3 * len(rows)) // 4, len(rows) - 1])
    )
    selected = [rows[position] for position in selected_positions]
    video_indices = [int(row["index"]) for row in selected]
    decoded = {
        camera: decode_frames(video_path, video_indices)
        for camera, video_path in CAMERA_FILES.items()
    }

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
    from lerobot.utils.constants import ACTION

    records: list[dict[str, object]] = []
    for row, position in zip(selected, selected_positions):
        video_index = int(row["index"])
        instruction = str(task_text[int(row["task_index"])])
        state = np.asarray(row["observation.state"], dtype=np.float32).reshape(14)
        recorded_action = np.asarray(row["action"], dtype=np.float32).reshape(14)
        raw = {
            "observation": {
                camera: {"rgb": decoded[camera][video_index]}
                for camera in ("head_camera", "left_camera", "right_camera")
            },
            "joint_action": {"vector": state.tolist()},
        }
        observation = harness.simulator_observation(raw)
        policy.reset()
        batch = harness.make_policy_batch(
            observation,
            instruction,
            env_preprocessor,
            preprocessor,
            policy.config,
        )
        with torch.inference_mode():
            predicted = policy.select_action(batch)
        predicted = postprocessor(predicted)
        predicted = env_postprocessor({ACTION: predicted})[ACTION]
        predicted_array = predicted.detach().float().cpu().numpy().reshape(14)
        abs_error = np.abs(predicted_array - recorded_action)
        records.append(
            {
                "position": int(position),
                "frame_index": int(row["frame_index"]),
                "video_index": video_index,
                "timestamp": float(row["timestamp"]),
                "task_index": int(row["task_index"]),
                "instruction": instruction,
                "state": state.tolist(),
                "recorded_action": recorded_action.tolist(),
                "predicted_action": predicted_array.tolist(),
                "abs_error": abs_error.tolist(),
                "l2": float(np.linalg.norm(predicted_array - recorded_action)),
                "camera_shapes": {
                    camera: list(decoded[camera][video_index].shape)
                    for camera in decoded
                },
            }
        )

    all_abs = np.asarray([record["abs_error"] for record in records], dtype=np.float64)
    all_l2 = np.asarray([record["l2"] for record in records], dtype=np.float64)
    gt = np.asarray([record["recorded_action"] for record in records], dtype=np.float64)
    pred = np.asarray([record["predicted_action"] for record in records], dtype=np.float64)
    global_action_min = np.asarray(
        json.loads((DATA_ROOT / "meta/stats.json").read_text(encoding="utf-8"))["action"]["min"],
        dtype=np.float64,
    )
    global_action_max = np.asarray(
        json.loads((DATA_ROOT / "meta/stats.json").read_text(encoding="utf-8"))["action"]["max"],
        dtype=np.float64,
    )
    in_global_range = np.logical_and(pred >= global_action_min, pred <= global_action_max)
    mean_mae = float(all_abs.mean())
    mean_l2 = float(all_l2.mean())
    max_mae = float(all_abs.max())
    range_fraction = float(in_global_range.mean())
    gripper_abs = all_abs[:, [6, 13]]
    gripper_pred = pred[:, [6, 13]]
    gripper_gt = gt[:, [6, 13]]

    # This is deliberately conservative: a broad, multi-step mismatch is the
    # conversion signal; near reproduction keeps the eval-loop hypothesis open.
    verdict = (
        "BROKEN — conversion/preprocessing suspect confirmed"
        if mean_mae > 0.50 or mean_l2 > 2.0
        else "OK — conversion/preprocessing is close; eval-loop/environment interaction remains suspect"
    )

    harness_action_lines = source_lines(HARNESS_PATH, 629, 645)
    harness_consume_lines = source_lines(HARNESS_PATH, 1046, 1052)
    model_path = Path(
        "/root/autodl-tmp/conda_envs/evolvevla_pi312/lib/python3.12/site-packages/"
        "lerobot/policies/pi05/modeling_pi05.py"
    )
    config_path = Path(
        "/root/autodl-tmp/conda_envs/evolvevla_pi312/lib/python3.12/site-packages/"
        "lerobot/policies/pi05/configuration_pi05.py"
    )
    model_proj_lines = source_lines(model_path, 427, 428)
    model_queue_lines = source_lines(model_path, 1032, 1038)
    config_chunk_lines = source_lines(config_path, 36, 37)
    config_max_dim_lines = source_lines(config_path, 41, 41)

    lines = [
        "# Gate-3 Oracle Action Reproduction",
        "",
        f"- Checkpoint: `{CHECKPOINT}`",
        f"- Harness: `{HARNESS_PATH}`",
        f"- Dataset: `{DATA_ROOT}` (LeRobot v3.0 cache)",
        f"- Episode: `{EPISODE}`; sampled positions: `{selected_positions}`",
        "- Policy calls were independently reset between sampled frames so each row "
        "compares the action predicted from that row's observation. The harness's "
        "persistent queue behavior is documented below.",
        "",
        "## Verdict",
        "",
        f"**{verdict}**",
        "",
        f"- Mean per-dimension absolute error: `{mean_mae:.6f}`",
        f"- Mean action L2 error: `{mean_l2:.6f}`",
        f"- Maximum per-dimension absolute error: `{max_mae:.6f}`",
        f"- Predicted values inside dataset global action range: `{range_fraction:.3%}`",
        f"- Dataset global action range test: `{fmt_bool(bool(in_global_range.all()))}`",
        "",
        "## Per-step comparison",
        "",
        "| pos | frame | video index | task index | instruction | recorded action (14-D) | predicted action (14-D) | abs error (14-D) | L2 |",
        "|---:|---:|---:|---:|---|---|---|---|---:|",
    ]
    for record in records:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(record["position"]),
                    str(record["frame_index"]),
                    str(record["video_index"]),
                    str(record["task_index"]),
                    str(record["instruction"]).replace("|", "\\|"),
                    fmt_vector(record["recorded_action"]),
                    fmt_vector(record["predicted_action"]),
                    fmt_vector(record["abs_error"]),
                    f"{float(record['l2']):.6f}",
                ]
            )
            + " |"
        )

    lines.extend(
        [
            "",
            "## Gripper dimensions",
            "",
            "| pos | recorded left/right gripper (dims 6,13) | predicted left/right gripper | abs error |",
            "|---:|---|---|---|",
        ]
    )
    for index, record in enumerate(records):
        lines.append(
            f"| {record['position']} | {fmt_vector(gripper_gt[index])} | "
            f"{fmt_vector(gripper_pred[index])} | {fmt_vector(gripper_abs[index])} |"
        )
    lines.extend(
        [
            "",
            "The gripper dimensions are the two final elements of each arm block: "
            "left gripper is dimension 6 and right gripper is dimension 13. "
            "The table reports whether predictions preserve the recorded binary-like "
            "range and timing at these sampled frames.",
            "",
            "## Dataset observations",
            "",
            "| pos | frame | timestamp | camera shapes | state (14-D) |",
            "|---:|---:|---:|---|---|",
        ]
    )
    for record in records:
        lines.append(
            f"| {record['position']} | {record['frame_index']} | "
            f"{float(record['timestamp']):.3f} | `{record['camera_shapes']}` | "
            f"{fmt_vector(record['state'])} |"
        )

    lines.extend(
        [
            "",
            "## Chunk semantics",
            "",
            "### Projection dimension",
            "",
            "The `32` in `action_out_proj=(32,1024)` is **not** the action horizon. "
            "It is `max_action_dim`, the padded action-vector width; the action horizon "
            "is `chunk_size=50`, and the configured number of executed action steps is "
            "`n_action_steps=50`.",
            "",
            "Verbatim source lines:",
            "",
            "```text",
            *model_proj_lines,
            *config_max_dim_lines,
            *config_chunk_lines,
            "```",
            "",
            "### Harness consumption",
            "",
            "The harness builds the batch, calls `select_action`, applies both "
            "postprocessors, reshapes the result to one 14-D action, and sends that "
            "single action to the simulator. It does not average a chunk and does not "
            "take a chunk tensor at the arm call site. Internally, `select_action` "
            "predicts a chunk only when its queue is empty, then returns one queued "
            "action per call.",
            "",
            "Harness call and postprocessing lines:",
            "",
            "```text",
            *harness_action_lines,
            "```",
            "",
            "Internal queue lines:",
            "",
            "```text",
            *model_queue_lines,
            "```",
            "",
            "Simulator action-consumption lines:",
            "",
            "```text",
            *harness_consume_lines,
            "```",
            "",
            "## Reproduction metadata",
            "",
            f"- Harness SHA-256: `{hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest()}`",
            f"- Data parquet: `{data_path}`",
            f"- Video frame indices used: `{video_indices}`",
            f"- Camera files: `{CAMERA_FILES}`",
            "- No harness or policy source file was modified by this diagnostic.",
        ]
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(OUTPUT), "verdict": verdict, "records": records}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
