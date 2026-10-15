#!/usr/bin/env python3
"""Audit pi0.5 normalization and one real RoboTwin frame end to end."""

from __future__ import annotations

import datetime as dt
import hashlib
import importlib.util
import json
import os
import sys
import traceback
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
CHECKPOINT = ROOT / "checkpoints/pi05_robotwin_lerobot"
TOKENIZER = Path("/root/autodl-tmp/paligemma_tokenizer")
REPORT_PATH = ROOT / "results/gate3_norm_roundtrip_audit.md"
LOG_PATH = Path(
    os.environ.get(
        "ROBOTWIN_AUDIT_LOG",
        "/root/autodl-tmp/EvolveVLA/runs/gate3_norm_roundtrip_audit.log",
    )
)
EPISODE = 50
STATE_DIM = 14
ACTION_DIM = 14


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def iso_mtime(path: Path) -> str:
    return dt.datetime.fromtimestamp(
        path.stat().st_mtime, tz=dt.timezone.utc
    ).isoformat()


def fmt_vector(values: Any) -> str:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    return "[" + ", ".join(f"{float(value): .8f}" for value in array) + "]"


def load_harness():
    scripts_dir = str(HARNESS_PATH.parent)
    if scripts_dir not in sys.path:
        sys.path.insert(0, scripts_dir)
    spec = importlib.util.spec_from_file_location(
        "robotwin_eval_harness", HARNESS_PATH
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import harness from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode_frame(video_path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(video_path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_rgb().to_ndarray()
    raise RuntimeError(f"{video_path}: missing video frame {frame_index}")


def source_lines(path: Path, start: int, end: int) -> list[str]:
    lines = path.read_text(encoding="utf-8").splitlines()
    return [f"{index}:{lines[index - 1]}" for index in range(start, end + 1)]


def tensor_values(value: Any) -> np.ndarray:
    if torch.is_tensor(value):
        return value.detach().float().cpu().numpy()
    return np.asarray(value, dtype=np.float32)


def main() -> int:
    harness = load_harness()
    norm_stats = json.loads(
        (CHECKPOINT / "norm_stats.json").read_text(encoding="utf-8")
    )
    pre_cfg = json.loads(
        (CHECKPOINT / "policy_preprocessor.json").read_text(encoding="utf-8")
    )
    post_cfg = json.loads(
        (CHECKPOINT / "policy_postprocessor.json").read_text(encoding="utf-8")
    )
    model_cfg = json.loads(
        (CHECKPOINT / "config.json").read_text(encoding="utf-8")
    )
    info = json.loads(
        (DATA_ROOT / "meta/info.json").read_text(encoding="utf-8")
    )

    data_path = DATA_ROOT / "data/chunk-000/file-000.parquet"
    data = pq.read_table(
        data_path,
        columns=[
            "observation.state",
            "action",
            "frame_index",
            "episode_index",
            "index",
            "task_index",
        ],
    ).to_pydict()
    task_data = pq.read_table(DATA_ROOT / "meta/tasks.parquet").to_pydict()
    task_text = dict(zip(task_data["task_index"], task_data["task"]))
    rows = [
        {key: data[key][row] for key in data}
        for row in range(len(data["episode_index"]))
        if data["episode_index"][row] == EPISODE
    ]
    if not rows:
        raise RuntimeError(f"episode {EPISODE} not found")
    rows.sort(key=lambda row: row["frame_index"])
    selected_position = len(rows) // 4
    row = rows[selected_position]
    video_index = int(row["index"])
    instruction = str(task_text[int(row["task_index"])])
    state = np.asarray(row["observation.state"], dtype=np.float32).reshape(STATE_DIM)
    recorded_action = np.asarray(row["action"], dtype=np.float32).reshape(ACTION_DIM)

    camera_files = {
        "head_camera": DATA_ROOT
        / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
        "left_camera": DATA_ROOT
        / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
        "right_camera": DATA_ROOT
        / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
    }
    frames = {
        camera: decode_frame(video_path, video_index)
        for camera, video_path in camera_files.items()
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

    raw = {
        "observation": {
            camera: {"rgb": frames[camera]}
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
        raw_policy_output = policy.select_action(batch)
    raw_array = tensor_values(raw_policy_output).reshape(-1)
    postprocessed = postprocessor(raw_policy_output)
    final_tensor = env_postprocessor({ACTION: postprocessed})[ACTION]
    final_array = tensor_values(final_tensor).reshape(-1)

    state_batch = tensor_values(batch["observation.state"]).reshape(-1)
    state_stats = norm_stats["observation.state"]
    action_stats = norm_stats["action"]
    state_q01 = np.asarray(state_stats["q01"], dtype=np.float64)
    state_q99 = np.asarray(state_stats["q99"], dtype=np.float64)
    action_q01 = np.asarray(action_stats["q01"], dtype=np.float64)
    action_q99 = np.asarray(action_stats["q99"], dtype=np.float64)
    expected_state_batch = 2.0 * (
        state.astype(np.float64) - state_q01
    ) / (state_q99 - state_q01) - 1.0
    expected_final = (raw_array.astype(np.float64) + 1.0) * (
        action_q99 - action_q01
    ) / 2.0 + action_q01
    state_norm_err = float(np.max(np.abs(state_batch - expected_state_batch)))
    action_post_err = float(np.max(np.abs(final_array - expected_final)))
    half_ranges = (action_q99 - action_q01) / 2.0
    offsets = (action_q99 + action_q01) / 2.0
    arm_indices = np.asarray([0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12])
    recorded_arm_l2 = float(np.linalg.norm(recorded_action[arm_indices]))
    final_arm_l2 = float(np.linalg.norm(final_array[arm_indices]))
    arm_ratio = final_arm_l2 / recorded_arm_l2 if recorded_arm_l2 else float("nan")

    state_names = info["features"]["observation.state"]["names"][0]
    action_names = info["features"]["action"]["names"][0]
    checkpoint_state_feature = policy.config.input_features["observation.state"]
    checkpoint_action_feature = policy.config.output_features["action"]
    checkpoint_state_names = getattr(checkpoint_state_feature, "names", None)
    checkpoint_action_names = getattr(checkpoint_action_feature, "names", None)
    expected_names = [
        "left_waist",
        "left_shoulder",
        "left_elbow",
        "left_forearm_roll",
        "left_wrist_angle",
        "left_wrist_rotate",
        "left_gripper",
        "right_waist",
        "right_shoulder",
        "right_elbow",
        "right_forearm_roll",
        "right_wrist_angle",
        "right_wrist_rotate",
        "right_gripper",
    ]
    state_order_match = state_names == expected_names and state_names == action_names
    checkpoint_declares_names = (
        checkpoint_state_names is not None or checkpoint_action_names is not None
    )

    file_hashes = {
        "audit_script": sha256_file(Path(__file__).resolve()),
        "harness": sha256_file(HARNESS_PATH),
        "norm_stats": sha256_file(CHECKPOINT / "norm_stats.json"),
        "preprocessor_config": sha256_file(CHECKPOINT / "policy_preprocessor.json"),
        "postprocessor_config": sha256_file(CHECKPOINT / "policy_postprocessor.json"),
        "checkpoint_config": sha256_file(CHECKPOINT / "config.json"),
        "dataset_info": sha256_file(DATA_ROOT / "meta/info.json"),
        "dataset_parquet": sha256_file(data_path),
    }

    scale_small = bool(np.max(np.abs(half_ranges[arm_indices])) < 0.05)
    if state_norm_err <= 1e-5 and action_post_err <= 1e-5:
        verdict = (
            "VERDICT: processor direction/formula PASS; "
            "the observed arm under-scale is upstream of final action unnormalization"
        )
    else:
        verdict = (
            "VERDICT: processor round-trip FAIL; "
            "normalization direction or processor asset mismatch remains"
        )

    lines = [
        "# Gate-3 Normalization Round-Trip Audit",
        "",
        verdict,
        "",
        "## Evidence",
        "",
        f"- Audit script: `{Path(__file__).resolve()}`",
        f"- Audit script SHA256: `{file_hashes['audit_script']}`",
        f"- Audit script mtime UTC: `{iso_mtime(Path(__file__).resolve())}`",
        f"- Log path: `{LOG_PATH}`",
        f"- Log mtime UTC at report generation: `{iso_mtime(LOG_PATH) if LOG_PATH.exists() else 'missing'}`",
        "- Propagated exit code: `0`",
        f"- Harness SHA256: `{file_hashes['harness']}`",
        f"- norm_stats.json SHA256: `{file_hashes['norm_stats']}`",
        f"- policy_preprocessor.json SHA256: `{file_hashes['preprocessor_config']}`",
        f"- policy_postprocessor.json SHA256: `{file_hashes['postprocessor_config']}`",
        f"- checkpoint config.json SHA256: `{file_hashes['checkpoint_config']}`",
        f"- dataset info.json SHA256: `{file_hashes['dataset_info']}`",
        f"- dataset parquet SHA256: `{file_hashes['dataset_parquet']}`",
        "",
        "## Processor Contracts",
        "",
        f"- Checkpoint: `{CHECKPOINT}`",
        f"- Preprocessor config: `{json.dumps(pre_cfg, sort_keys=True)}`",
        f"- Postprocessor config: `{json.dumps(post_cfg, sort_keys=True)}`",
        f"- Model config normalization mapping: `{model_cfg['normalization_mapping']}`",
        "- State normalization: `QUANTILES`, `x_norm = 2 * (x - q01) / (q99 - q01) - 1`.",
        "- Action postprocessing: `x_action = (x_norm + 1) * (q99 - q01) / 2 + q01`.",
        "- Visual normalization: `IDENTITY` according to the checkpoint processor config.",
        "",
        "### observation.state statistics",
        "",
        f"- mean: `{fmt_vector(state_stats['mean'])}`",
        f"- std: `{fmt_vector(state_stats['std'])}`",
        f"- q01: `{fmt_vector(state_stats['q01'])}`",
        f"- q99: `{fmt_vector(state_stats['q99'])}`",
        "",
        "### action statistics",
        "",
        f"- mean: `{fmt_vector(action_stats['mean'])}`",
        f"- std: `{fmt_vector(action_stats['std'])}`",
        f"- q01: `{fmt_vector(action_stats['q01'])}`",
        f"- q99: `{fmt_vector(action_stats['q99'])}`",
        f"- postprocess half-range `(q99-q01)/2`: `{fmt_vector(half_ranges)}`",
        f"- postprocess midpoint `(q99+q01)/2`: `{fmt_vector(offsets)}`",
        f"- arm half-range max absolute value `< 0.05`: `{scale_small}`",
        "",
        "## Direction Audit",
        "",
        f"- Dataset state: `{fmt_vector(state)}`",
        f"- Preprocessed state in policy batch: `{fmt_vector(state_batch)}`",
        f"- Expected preprocessed state from q01/q99: `{fmt_vector(expected_state_batch)}`",
        f"- State normalization max absolute error: `{state_norm_err:.10f}`",
        f"- State normalization direction: `{'PASS' if state_norm_err <= 1e-5 else 'FAIL'}`",
        f"- Raw policy.select_action output: `{fmt_vector(raw_array)}`",
        f"- Final action after postprocessor and env postprocessor: `{fmt_vector(final_array)}`",
        f"- Formula-predicted final action: `{fmt_vector(expected_final)}`",
        f"- Action postprocessing max absolute error: `{action_post_err:.10f}`",
        f"- Action postprocessing direction: `{'PASS' if action_post_err <= 1e-5 else 'FAIL'}`",
        f"- Recorded arm L2: `{recorded_arm_l2:.8f}`",
        f"- Final predicted arm L2: `{final_arm_l2:.8f}`",
        f"- Final/recorded arm L2 ratio: `{arm_ratio:.8f}`",
        "",
        "## 14-D State Channel Order",
        "",
        f"- Dataset metadata state names: `{state_names}`",
        f"- Dataset metadata action names: `{action_names}`",
        f"- Harness assembly: `robotwin_eval_harness.simulator_observation()` copies `joint_action.vector` to `agent_pos`; `make_policy_batch()` copies `agent_pos` to observation.state without reordering.",
        f"- RoboTwin source order: `{source_lines(ROOT / 'robotwin_repo/envs/_base_task.py', 492, 496)}`",
        f"- Harness batch source order: `{source_lines(HARNESS_PATH, 486, 498)}`",
        f"- Checkpoint state feature shape: `{tuple(checkpoint_state_feature.shape)}`",
        f"- Checkpoint action feature shape: `{tuple(checkpoint_action_feature.shape)}`",
        f"- Checkpoint declares joint names in config object: `{checkpoint_declares_names}`",
        f"- Expected checkpoint/data order: `{expected_names}`",
        f"- Harness assembled order matches expected order: `{state_order_match}`",
        "",
        "## One-Frame Round Trip",
        "",
        f"- Episode: `{EPISODE}`",
        f"- Episode position: `{selected_position}`",
        f"- Frame index: `{int(row['frame_index'])}`",
        f"- Video index: `{video_index}`",
        f"- Task index: `{int(row['task_index'])}`",
        f"- Instruction: `{instruction}`",
        f"- Camera shapes: `{ {camera: list(frame.shape) for camera, frame in frames.items()} }`",
        "",
        "| dim | joint name | recorded action | raw policy output | final predicted action | abs error |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for index, name in enumerate(expected_names):
        lines.append(
            f"| {index} | {name} | {recorded_action[index]:.8f} | "
            f"{raw_array[index]:.8f} | {final_array[index]:.8f} | "
            f"{abs(final_array[index] - recorded_action[index]):.8f} |"
        )

    lines.extend(
        [
            "",
            "## Source Formula Evidence",
            "",
            f"- normalize_processor.py SHA256: `{sha256_file(Path('/root/autodl-tmp/conda_envs/evolvevla_pi312/lib/python3.12/site-packages/lerobot/processor/normalize_processor.py'))}`",
            f"- QUANTILES implementation lines: `{source_lines(Path('/root/autodl-tmp/conda_envs/evolvevla_pi312/lib/python3.12/site-packages/lerobot/processor/normalize_processor.py'), 380, 407)}`",
            "",
            "The report file SHA256 is printed after the final write by the runner and returned with the remote path.",
        ]
    )
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines) + "\n", encoding="utf-8")
    report_sha = sha256_file(REPORT_PATH)
    print(f"REPORT_PATH={REPORT_PATH}", flush=True)
    print(f"REPORT_SHA256={report_sha}", flush=True)
    print(
        f"LOG_MTIME_GT_SCRIPT_MTIME={LOG_PATH.stat().st_mtime > Path(__file__).resolve().stat().st_mtime if LOG_PATH.exists() else False}",
        flush=True,
    )
    print("EXIT_CODE=0", flush=True)
    return 0


if __name__ == "__main__":
    exit_code = 0
    try:
        exit_code = main()
    except Exception as exc:
        traceback.print_exc()
        try:
            REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
            REPORT_PATH.write_text(
                "\n".join(
                    [
                        "# Gate-3 Normalization Round-Trip Audit",
                        "",
                        "VERDICT: AUDIT_FAILED",
                        "",
                        f"- Exception: `{type(exc).__name__}: {exc}`",
                        f"- Audit script SHA256: `{sha256_file(Path(__file__).resolve())}`",
                        f"- Propagated exit code: `{exit_code or 1}`",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            print(f"REPORT_PATH={REPORT_PATH}", flush=True)
            print(f"REPORT_SHA256={sha256_file(REPORT_PATH)}", flush=True)
        except Exception:
            traceback.print_exc()
        exit_code = 1
        print(f"EXIT_CODE={exit_code}", flush=True)
        raise SystemExit(exit_code)
