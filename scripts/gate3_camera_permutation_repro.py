#!/usr/bin/env python3
"""D119 action reproduction with camera-permutation arbitration.

This probe uses one cached demo_clean RoboTwin observation and its recorded
action chunk. It reuses the pinned harness observation bridge and
make_policy_batch helper, then drains the policy's own select_action queue.
No simulator is created and no rollout is performed.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.util
import json
import re
import sys
from pathlib import Path
from typing import Any

import av
import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path("/root/autodl-tmp/EvolveVLA")
SCRIPTS = ROOT / "scripts"
HARNESS_PATH = SCRIPTS / "robotwin_eval_harness.py"
DATA_ROOT = ROOT / "data/gate2_robotwin_cache/clean"
DATA_PATH = DATA_ROOT / "data/chunk-000/file-000.parquet"
TASKS_PATH = DATA_ROOT / "meta/tasks.parquet"
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
CHECKPOINT = ROOT / "checkpoints/pi05_robotwin_lerobot"
REPORT_PATH = ROOT / "results/gate3_camera_permutation_repro.md"
JSON_PATH = ROOT / "results/gate3_camera_permutation_repro.json"
COSINE_MARGIN_THRESHOLD = 0.01
NORM_GATE_RATIO = 0.5

TASK = "beat_block_hammer"
EPISODE = 50
POSITION = 31
ACTION_DIM = 14
ARM_INDICES = np.asarray(
    [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12],
    dtype=np.int64,
)

CAMERA_FILES = {
    "cam_high": DATA_ROOT
    / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
    "cam_left_wrist": DATA_ROOT
    / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
    "cam_right_wrist": DATA_ROOT
    / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
}

# Values are source PNG names assigned to each policy camera key.
CAMERA_MAPS = {
    "current": {
        "cam_high": "cam_high",
        "cam_left_wrist": "cam_left_wrist",
        "cam_right_wrist": "cam_right_wrist",
    },
    "swap_left_right_wrist": {
        "cam_high": "cam_high",
        "cam_left_wrist": "cam_right_wrist",
        "cam_right_wrist": "cam_left_wrist",
    },
    "swap_cam_high_left_wrist": {
        "cam_high": "cam_left_wrist",
        "cam_left_wrist": "cam_high",
        "cam_right_wrist": "cam_right_wrist",
    },
    "cycle_high_left_right": {
        "cam_high": "cam_left_wrist",
        "cam_left_wrist": "cam_right_wrist",
        "cam_right_wrist": "cam_high",
    },
}


def load_harness() -> Any:
    if str(SCRIPTS) not in sys.path:
        sys.path.insert(0, str(SCRIPTS))
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


def load_episode() -> dict[str, Any]:
    table = pq.read_table(
        DATA_PATH,
        columns=[
            "observation.state",
            "action",
            "episode_index",
            "index",
            "task_index",
            "frame_index",
            "timestamp",
        ],
    ).to_pydict()
    rows = [
        {key: table[key][index] for key in table}
        for index in range(len(table["episode_index"]))
        if int(table["episode_index"][index]) == EPISODE
    ]
    rows.sort(key=lambda row: int(row["frame_index"]))
    if len(rows) <= POSITION:
        raise RuntimeError(
            f"episode {EPISODE} has {len(rows)} rows; position {POSITION} is absent"
        )

    chunk_size = 50
    chunk_rows = rows[POSITION : POSITION + chunk_size]
    if len(chunk_rows) != chunk_size:
        raise RuntimeError(
            f"episode {EPISODE} has only {len(chunk_rows)} rows after "
            f"position {POSITION}; need {chunk_size}"
        )
    frame_indices = [int(row["frame_index"]) for row in chunk_rows]
    expected_frames = list(range(frame_indices[0], frame_indices[0] + chunk_size))
    if frame_indices != expected_frames:
        raise RuntimeError(
            f"recorded action chunk is not contiguous: {frame_indices[:4]}..."
        )

    tasks = pq.read_table(TASKS_PATH).to_pydict()
    task_text = dict(zip(tasks["task_index"], tasks["task"]))
    first = chunk_rows[0]
    instruction = str(task_text[int(first["task_index"])])
    if TASK not in instruction.lower() and "hammer" not in instruction.lower():
        raise RuntimeError(
            f"selected episode is not the requested hammer task: {instruction!r}"
        )

    state = np.asarray(first["observation.state"], dtype=np.float32).reshape(-1)
    if state.shape != (ACTION_DIM,) or not np.isfinite(state).all():
        raise RuntimeError(f"invalid observation.state shape/values: {state.shape}")
    recorded = np.asarray(
        [np.asarray(row["action"], dtype=np.float32).reshape(-1) for row in chunk_rows]
    )
    if recorded.shape != (chunk_size, ACTION_DIM) or not np.isfinite(recorded).all():
        raise RuntimeError(f"invalid recorded action chunk: {recorded.shape}")

    video_index = int(first["index"])
    frames = {
        camera: decode_frame(path, video_index)
        for camera, path in CAMERA_FILES.items()
    }
    if any(frame.shape[-1] != 3 for frame in frames.values()):
        raise RuntimeError(
            f"expected RGB frames, got { {key: value.shape for key, value in frames.items()} }"
        )

    return {
        "episode": EPISODE,
        "position": POSITION,
        "frame_index": int(first["frame_index"]),
        "video_index": video_index,
        "timestamp": float(first["timestamp"]),
        "instruction": instruction,
        "state": state,
        "recorded": recorded,
        "frames": frames,
        "frame_shapes": {key: list(value.shape) for key, value in frames.items()},
        "chunk_size": chunk_size,
        # demo_clean is the curated successful-demonstration split; this cache
        # slice has no separate success column, so validity is checked directly.
        "success_basis": (
            "demo_clean curated episode with contiguous finite recorded actions; "
            "the cached parquet has no explicit success field"
        ),
    }


def load_policy_cpu_or_cuda(
    harness: Any,
    model_path: Path,
    task: str,
    tokenizer_path: Path,
) -> tuple[Any, Any, Any, Any, Any]:
    """Mirror harness.load_policy while permitting the CPU preflight path."""
    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_policy, make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    env_cfg = RoboTwinEnvConfig(
        task=task,
        episode_length=1200,
        obs_type="pixels_agent_pos",
        camera_names="cam_high,cam_left_wrist,cam_right_wrist",
        observation_height=harness.OBS_HEIGHT,
        observation_width=harness.OBS_WIDTH,
        action_mode="joint",
    )
    policy_cfg = PI05Config.from_pretrained(
        str(model_path),
        local_files_only=True,
    )
    policy_cfg.device = device
    policy_cfg.pretrained_path = model_path
    rename_map = harness.derive_camera_rename_map(policy_cfg)
    policy = make_policy(cfg=policy_cfg, env_cfg=env_cfg)
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=str(model_path),
        preprocessor_overrides={
            "device_processor": {"device": device},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(tokenizer_path)},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy.config
    )
    expected_state = getattr(policy.config, "input_features", {}).get(
        "observation.state"
    )
    if expected_state is not None and tuple(expected_state.shape) != (ACTION_DIM,):
        raise RuntimeError(f"checkpoint state contract is {expected_state.shape}")
    expected_action = getattr(policy.config, "output_features", {}).get("action")
    if expected_action is not None and tuple(expected_action.shape) != (ACTION_DIM,):
        raise RuntimeError(f"checkpoint action contract is {expected_action.shape}")
    return policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor


def make_observation(
    harness: Any,
    frames: dict[str, np.ndarray],
    camera_map: dict[str, str],
    state: np.ndarray,
) -> dict[str, Any]:
    # simulator_observation owns the exact bridge and policy-key assembly. Only
    # the source image assigned to each simulated camera is changed here.
    raw = {
        "head_camera": {"rgb": frames[camera_map["cam_high"]]},
        "left_camera": {"rgb": frames[camera_map["cam_left_wrist"]]},
        "right_camera": {"rgb": frames[camera_map["cam_right_wrist"]]},
    }
    return harness.simulator_observation(
        {
            "observation": raw,
            "joint_action": {"vector": state.tolist()},
        }
    )


def predict_action_chunk(
    policy: Any,
    batch: dict[str, Any],
    postprocessor: Any,
    env_postprocessor: Any,
) -> np.ndarray:
    from lerobot.utils.constants import ACTION

    policy.reset()
    n_steps = int(
        getattr(
            policy.config,
            "n_action_steps",
            getattr(policy.config, "chunk_size", 50),
        )
    )
    if n_steps <= 0:
        raise RuntimeError(f"invalid action horizon: {n_steps}")
    predicted: list[np.ndarray] = []
    with torch.inference_mode():
        for _ in range(n_steps):
            action = policy.select_action(batch)
            action = postprocessor(action)
            action = env_postprocessor({ACTION: action})[ACTION]
            array = action.detach().float().cpu().numpy().reshape(-1)
            if array.shape != (ACTION_DIM,) or not np.isfinite(array).all():
                raise RuntimeError(f"invalid predicted action: {array.shape}")
            predicted.append(array)
    return np.stack(predicted, axis=0)


def score_chunk(
    predicted: np.ndarray,
    recorded: np.ndarray,
) -> dict[str, Any]:
    if predicted.shape != recorded.shape:
        raise RuntimeError(
            f"predicted/recorded chunk shape mismatch: {predicted.shape} vs "
            f"{recorded.shape}"
        )
    predicted_arm = predicted[:, ARM_INDICES]
    recorded_arm = recorded[:, ARM_INDICES]
    delta = predicted_arm - recorded_arm
    arm_l2 = np.linalg.norm(delta, axis=1)
    pred_norm = np.linalg.norm(predicted_arm, axis=1)
    rec_norm = np.linalg.norm(recorded_arm, axis=1)
    denominator = pred_norm * rec_norm
    cosines = np.full(len(predicted), np.nan, dtype=np.float64)
    valid = denominator > 1e-12
    cosines[valid] = (
        np.sum(predicted_arm[valid] * recorded_arm[valid], axis=1)
        / denominator[valid]
    )
    full_l2 = np.linalg.norm(predicted - recorded, axis=1)
    return {
        "mean_cosine": float(np.nanmean(cosines)),
        "mean_arm_l2_err": float(np.mean(arm_l2)),
        "mean_full_l2_err": float(np.mean(full_l2)),
        "mean_predicted_arm_norm": float(np.mean(pred_norm)),
        "final_predicted_arm_norm": float(pred_norm[-1]),
        "mean_recorded_arm_norm": float(np.mean(rec_norm)),
        "final_recorded_arm_norm": float(rec_norm[-1]),
        "cosine_valid_steps": int(np.isfinite(cosines).sum()),
        "per_step_cosine": cosines.tolist(),
        "per_step_arm_l2_err": arm_l2.tolist(),
        "per_step_full_l2_err": full_l2.tolist(),
        "predicted_action_min": float(predicted.min()),
        "predicted_action_max": float(predicted.max()),
    }


def fmt_vector(values: Any, precision: int = 5) -> str:
    array = np.asarray(values).reshape(-1)
    return "[" + ", ".join(f"{float(value): .{precision}f}" for value in array) + "]"


def write_report(
    payload: dict[str, Any],
    report_path: Path,
    json_path: Path,
) -> None:
    report_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=True) + "\n",
        encoding="utf-8",
    )
    lines = [
        "# D119 Camera-Permutation Action Reproduction",
        "",
        f"- Checkpoint: `{payload['checkpoint']}`",
        f"- Harness SHA256: `{payload['harness_sha256']}`",
        f"- Dataset: `{payload['dataset']}`",
        f"- Episode/position: `{payload['episode']}/{payload['position']}`",
        f"- Frame/video index: `{payload['frame_index']}/{payload['video_index']}`",
        f"- Instruction: `{payload['instruction']}`",
        f"- Observation frame shapes: `{payload['frame_shapes']}`",
        f"- Recorded action chunk shape: `{payload['recorded_action_chunk_shape']}`",
        f"- Success basis: {payload['success_basis']}",
        "",
        "The probe used `robotwin_eval_harness.simulator_observation`, "
        "`make_policy_batch`, `select_action`, the policy postprocessor, and the "
        "environment postprocessor. `select_action` was called repeatedly after "
        "one reset to drain the model's own action queue.",
        "",
        "## Camera Arbitration",
        "",
        "| mapping | cam_high source | cam_left_wrist source | cam_right_wrist source | mean cosine | mean arm-L2 error | mean predicted arm norm | final predicted arm norm | mean full-L2 error |",
        "|---|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for name, result in payload["results"].items():
        mapping = payload["camera_maps"][name]
        lines.append(
            f"| `{name}` | `{mapping['cam_high']}` | "
            f"`{mapping['cam_left_wrist']}` | `{mapping['cam_right_wrist']}` | "
            f"{result['mean_cosine']:.6f} | {result['mean_arm_l2_err']:.6f} | "
            f"{result['mean_predicted_arm_norm']:.6f} | "
            f"{result['final_predicted_arm_norm']:.6f} | "
            f"{result['mean_full_l2_err']:.6f} |"
        )
    lines.extend(
        [
            "",
            f"- `CAMMAP_BEST={payload['best_mapping']}`",
            f"- `CAMMAP_VERDICT={payload['verdict']}`",
            "",
            "## Per-Step Metrics",
            "",
        ]
    )
    for name, result in payload["results"].items():
        lines.extend(
            [
                f"### `{name}`",
                "",
                "| chunk step | cosine | arm-L2 error | full-L2 error |",
                "|---:|---:|---:|---:|",
            ]
        )
        for index, (cosine, arm_l2, full_l2) in enumerate(
            zip(
                result["per_step_cosine"],
                result["per_step_arm_l2_err"],
                result["per_step_full_l2_err"],
            )
        ):
            cosine_text = "nan" if not np.isfinite(cosine) else f"{cosine:.6f}"
            lines.append(
                f"| {index} | {cosine_text} | {arm_l2:.6f} | {full_l2:.6f} |"
            )
        lines.append("")
    report_path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", default=str(CHECKPOINT))
    parser.add_argument("--episode", type=int, default=EPISODE)
    parser.add_argument("--position", type=int, default=POSITION)
    parser.add_argument(
        "--report",
        default=None,
        help="report path; defaults to a checkpoint-specific filename",
    )
    parser.add_argument(
        "--json",
        default=None,
        help="JSON path; defaults to a checkpoint-specific filename",
    )
    parser.add_argument(
        "--dry-run-paths",
        action="store_true",
        help="print resolved report and JSON paths without loading the model",
    )
    return parser.parse_args()


def checkpoint_tag(model_path: Path) -> str:
    basename = model_path.name.lower()
    if basename == CHECKPOINT.name.lower() or "official" in basename:
        return "official"
    if "peer" in basename or "sidneyxie" in basename:
        return "peer"
    return re.sub(r"[^a-z0-9]+", "_", basename).strip("_")


def main() -> int:
    args = parse_args()
    if args.episode != EPISODE or args.position != POSITION:
        raise RuntimeError(
            "this bounded probe is pinned to the verified episode 50, position 31 "
            "sample; edit requires a new reviewed file"
        )
    model_path = Path(args.model_path)
    tag = checkpoint_tag(model_path)
    report_path = Path(args.report) if args.report else (
        ROOT / f"results/gate3_camera_permutation_repro_{tag}.md"
    )
    json_path = Path(args.json) if args.json else (
        ROOT / f"results/gate3_camera_permutation_repro_{tag}.json"
    )
    if args.dry_run_paths:
        print(f"REPORT={report_path}")
        print(f"JSON={json_path}")
        return 0
    harness = load_harness()
    sample = load_episode()
    policy, preprocessor, postprocessor, env_preprocessor, env_postprocessor = (
        load_policy_cpu_or_cuda(
            harness,
            model_path,
            TASK,
            TOKENIZER_PATH,
        )
    )
    policy.eval()
    results: dict[str, dict[str, Any]] = {}
    for name, camera_map in CAMERA_MAPS.items():
        observation = make_observation(
            harness,
            sample["frames"],
            camera_map,
            sample["state"],
        )
        batch = harness.make_policy_batch(
            observation,
            sample["instruction"],
            env_preprocessor,
            preprocessor,
            policy.config,
        )
        predicted = predict_action_chunk(
            policy,
            batch,
            postprocessor,
            env_postprocessor,
        )
        if len(predicted) != sample["chunk_size"]:
            raise RuntimeError(
                f"policy action horizon {len(predicted)} does not match the "
                f"recorded chunk size {sample['chunk_size']}"
            )
        recorded = sample["recorded"]
        results[name] = score_chunk(predicted, recorded)
        del batch, observation, predicted
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    best_mapping = max(
        results,
        key=lambda name: (
            results[name]["mean_cosine"],
            -results[name]["mean_arm_l2_err"],
        ),
    )
    cosine_margin = (
        results[best_mapping]["mean_cosine"]
        - results["current"]["mean_cosine"]
    )
    best_predicted_norm = results[best_mapping]["mean_predicted_arm_norm"]
    recorded_norm = results[best_mapping]["mean_recorded_arm_norm"]
    if best_predicted_norm < NORM_GATE_RATIO * recorded_norm:
        verdict = (
            "ALL MAPS REPRODUCE / arm magnitude too low for a cosine "
            "comparison / no wiring signal"
        )
    elif best_mapping != "current" and cosine_margin >= COSINE_MARGIN_THRESHOLD:
        verdict = (
            f"PERMUTATION_BETTER:{best_mapping}; mean-cosine margin "
            f"{cosine_margin:.6f} >= {COSINE_MARGIN_THRESHOLD:.3f}"
        )
    else:
        verdict = (
            "ALL MAPS REPRODUCE / no wiring sensitivity; mean-cosine margin "
            f"{cosine_margin:.6f} < {COSINE_MARGIN_THRESHOLD:.3f}"
        )

    payload = {
        "checkpoint": str(model_path),
        "harness_sha256": hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest(),
        "probe_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "dataset": str(DATA_ROOT),
        "episode": sample["episode"],
        "position": sample["position"],
        "frame_index": sample["frame_index"],
        "video_index": sample["video_index"],
        "timestamp": sample["timestamp"],
        "instruction": sample["instruction"],
        "frame_shapes": sample["frame_shapes"],
        "recorded_action_chunk_shape": list(sample["recorded"].shape),
        "success_basis": sample["success_basis"],
        "camera_maps": CAMERA_MAPS,
        "results": results,
        "best_mapping": best_mapping,
        "cosine_margin_vs_current": cosine_margin,
        "cosine_margin_threshold": COSINE_MARGIN_THRESHOLD,
        "verdict": verdict,
        "no_sim_rollout": True,
    }
    write_report(payload, report_path, json_path)
    print(f"EPISODE={sample['episode']} POSITION={sample['position']}")
    print(f"VIDEO_INDEX={sample['video_index']}")
    print(f"INSTRUCTION={sample['instruction']}")
    for name, result in results.items():
        print(
            f"CAMMAP={name} mean_cosine={result['mean_cosine']:.6f} "
            f"mean_armL2_err={result['mean_arm_l2_err']:.6f} "
            f"mean_pred_arm_norm={result['mean_predicted_arm_norm']:.6f} "
            f"final_pred_arm_norm={result['final_predicted_arm_norm']:.6f} "
            f"pred_action_min={result['predicted_action_min']:.6f} "
            f"pred_action_max={result['predicted_action_max']:.6f}"
        )
    print(f"CAMMAP_BEST={best_mapping}")
    print(f"CAMMAP_VERDICT={verdict}")
    print(f"REPORT={report_path}")
    print(f"JSON={json_path}")
    print(f"PROBE_SHA256={payload['probe_sha256']}")
    print(f"HARNESS_SHA256={payload['harness_sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
