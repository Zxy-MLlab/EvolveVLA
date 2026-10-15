#!/usr/bin/env python3
"""Paired CPU sweep of the official OpenPI path and the harness policy path."""

from __future__ import annotations

import argparse
import hashlib
import importlib
import os

# Keep this driver CPU-only before importing torch or any policy package.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path("/root/autodl-tmp/EvolveVLA")
DATA = ROOT / "data/gate2_robotwin_cache/clean"
DEFAULT_CHECKPOINT = ROOT / "checkpoints/pi05_robotwin_lerobot"
REFERENCE_CHECKPOINT = DEFAULT_CHECKPOINT
OUR_CHECKPOINT = DEFAULT_CHECKPOINT
TOKENIZER = Path("/root/autodl-tmp/paligemma_tokenizer")
OPENPI_SRC = Path("/root/autodl-tmp/openpi_feasibility_src/src")
REFERENCE_SCRIPT = ROOT / "scripts/frozen/gate3_reference_forward_a736d419.py"
REFERENCE_MODULE = "gate3_reference_forward_a736d419"
HARNESS_SCRIPT = ROOT / "scripts/robotwin_eval_harness.py"
FPS = 50
TASK = "Utilize the left arm to hold the grippy handle hammer and strike."
ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
EXPECTED_RAW_FRAME_HASHES = {
    "head_camera": "6813d9707236c51f5f65e2ecacd028c5c5ca21c863dc71586e74905071f66f04",
    "left_camera": "f7f91a608a5506567258e9b342b4555a2ec4b91ab306bf3407e5ec7c23bab707",
    "right_camera": "6b580e8fd7f49b1bf4a588cbede26aaa76b9bca27717e6ed09d1525460fbc705",
}

# Four positions from episode 50 plus two different episodes.
OBS_SPECS = (
    (50, 0),
    (50, 31),
    (50, 62),
    (50, 93),
    (0, 31),
    (1, 31),
)
REFERENCE_NORM_MODE = "quantiles"


def resolve_checkpoint(path: str | Path) -> Path:
    path = Path(path)
    if not path.is_absolute():
        path = ROOT / path
    return path


def reference_module():
    if str(OPENPI_SRC) not in sys.path:
        sys.path.insert(0, str(OPENPI_SRC))
    if str(REFERENCE_SCRIPT.parent) not in sys.path:
        sys.path.insert(0, str(REFERENCE_SCRIPT.parent))
    return importlib.import_module(REFERENCE_MODULE)


def load_observation_rows(specs: tuple[tuple[int, int], ...]) -> list[dict[str, Any]]:
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.parquet as pq

    episode_ids = sorted({episode for episode, _ in specs})
    episode_paths = sorted(
        (DATA / "meta/episodes").glob("chunk-*/file-*.parquet")
    )
    episode_rows: dict[int, dict[str, Any]] = {}
    for path in episode_paths:
        table = pq.read_table(path)
        for row in table.to_pylist():
            if int(row["episode_index"]) in episode_ids:
                episode_rows[int(row["episode_index"])] = row

    if set(episode_ids) != set(episode_rows):
        missing = sorted(set(episode_ids) - set(episode_rows))
        raise RuntimeError(f"missing episode metadata: {missing}")

    data_path = DATA / "data/chunk-000/file-000.parquet"
    table = pq.read_table(
        data_path,
        columns=["observation.state", "frame_index", "episode_index"],
    )
    mask = pc.is_in(
        table["episode_index"],
        value_set=pa.array(episode_ids, type=pa.int64()),
    )
    selected = table.filter(mask).to_pylist()
    state_rows = {
        (int(row["episode_index"]), int(row["frame_index"])): np.asarray(
            row["observation.state"], dtype=np.float32
        )
        for row in selected
    }

    observations: list[dict[str, Any]] = []
    for episode, position in specs:
        meta = episode_rows[episode]
        length = int(meta["length"])
        if position < 0 or position >= length:
            raise RuntimeError(
                f"episode {episode} position {position} outside length {length}"
            )
        key = (episode, position)
        if key not in state_rows:
            raise RuntimeError(f"missing state row for episode={episode} position={position}")
        video_index = int(
            meta["videos/observation.images.cam_high/file_index"]
        )
        chunk_index = int(
            meta["videos/observation.images.cam_high/chunk_index"]
        )
        from_timestamp = float(
            meta["videos/observation.images.cam_high/from_timestamp"]
        )
        video_frame = int(round(from_timestamp * FPS)) + position
        observations.append(
            {
                "episode": episode,
                "position": position,
                "state": state_rows[key],
                "video_frame": video_frame,
                "video_chunk": chunk_index,
                "video_index": video_index,
            }
        )
    return observations


def decode_selected(path: Path, indices: set[int]) -> dict[int, np.ndarray]:
    import av

    wanted = set(indices)
    frames: dict[int, np.ndarray] = {}
    with av.open(str(path)) as container:
        for index, decoded in enumerate(container.decode(video=0)):
            if index in wanted:
                frames[index] = decoded.to_ndarray(format="rgb24")
            if len(frames) == len(wanted):
                break
    missing = sorted(wanted - set(frames))
    if missing:
        raise RuntimeError(f"missing video frames in {path}: {missing}")
    return frames


def load_frames(observations: list[dict[str, Any]]) -> None:
    camera_paths = {
        "head_camera": DATA / "videos/observation.images.cam_high",
        "left_camera": DATA / "videos/observation.images.cam_left_wrist",
        "right_camera": DATA / "videos/observation.images.cam_right_wrist",
    }
    grouped: dict[tuple[str, int, int], set[int]] = {}
    for obs in observations:
        key = (str(obs["video_chunk"]), obs["video_index"], 0)
        grouped.setdefault(key, set()).add(int(obs["video_frame"]))

    decoded: dict[tuple[str, int, int], dict[int, np.ndarray]] = {}
    for camera, root in camera_paths.items():
        for key, indices in grouped.items():
            chunk_text, video_index, _ = key
            chunk = int(chunk_text)
            path = root / f"chunk-{chunk:03d}" / f"file-{video_index:03d}.mp4"
            if not path.exists():
                raise RuntimeError(f"video file is absent: {path}")
            decoded[(camera, chunk, video_index)] = decode_selected(path, indices)

    for obs in observations:
        key = (int(obs["video_chunk"]), int(obs["video_index"]))
        frame_index = int(obs["video_frame"])
        obs["frames"] = {
            camera: decoded[(camera, key[0], key[1])][frame_index]
            for camera in camera_paths
        }
        if frame_index == 7219:
            raw_hashes = {
                camera: hashlib.sha256(frame.tobytes()).hexdigest()
                for camera, frame in obs["frames"].items()
            }
            if raw_hashes != EXPECTED_RAW_FRAME_HASHES:
                raise AssertionError(
                    "raw decoded frame hash mismatch "
                    f"video_frame=7219 actual={raw_hashes} "
                    f"expected={EXPECTED_RAW_FRAME_HASHES}"
                )
            print(
                "RAW_FRAME_HASH_ASSERT "
                f"video_frame=7219 hashes={raw_hashes} "
                f"expected={EXPECTED_RAW_FRAME_HASHES}",
                flush=True,
            )
            print(
                "REFERENCE_INPUT=RAW decoded 7219 frame; "
                "preprocessing=OpenPI-owned (not OUR bridge)",
                flush=True,
            )


def load_reference_model() -> tuple[Any, Any, dict[str, Any], Any, Any]:
    import sentencepiece
    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    reference = reference_module()
    device = torch.device("cpu")
    model = PI0Pytorch(Pi0Config(pi05=True, dtype="float32"))
    loaded_keys = reference.load_weights(model, REFERENCE_CHECKPOINT)
    model = model.to(device).eval()
    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = 200
    with open(TOKENIZER / "tokenizer.model", "rb") as handle:
        tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=handle.read()
        )
    stats = reference.stats(REFERENCE_CHECKPOINT, REFERENCE_NORM_MODE)
    print(
        f"REFERENCE_MODEL_READY checkpoint={REFERENCE_CHECKPOINT} "
        f"norm_mode={REFERENCE_NORM_MODE} loaded_keys={loaded_keys}",
        flush=True,
    )
    return model, tokenizer, stats, device, reference


def reference_forward(
    model: Any,
    tokenizer: Any,
    stats: dict[str, Any],
    device: Any,
    reference: Any,
    state: np.ndarray,
    frames: dict[str, np.ndarray],
) -> float:
    import torch
    from openpi.models.model import Observation
    from openpi.shared.image_tools import resize_with_pad_torch

    normalized_state = reference.normalize_state(state, stats)
    tokens, token_mask = tokenizer.tokenize(TASK, normalized_state)
    image_names = {
        "head_camera": "base_0_rgb",
        "left_camera": "left_wrist_0_rgb",
        "right_camera": "right_wrist_0_rgb",
    }
    images: dict[str, torch.Tensor] = {}
    for camera, key in image_names.items():
        image = torch.from_numpy(frames[camera]).float() / 255.0 * 2.0 - 1.0
        image = image.unsqueeze(0)
        image = resize_with_pad_torch(image, 224, 224)
        images[key] = image.unsqueeze(0).permute(0, 3, 1, 2).to(device)

    obs = Observation.from_dict(
        {
            "image": images,
            "image_mask": {
                key: torch.ones(1, dtype=torch.bool, device=device)
                for key in images
            },
            "state": torch.from_numpy(normalized_state).view(1, -1).to(device),
            "tokenized_prompt": torch.from_numpy(tokens)
            .view(1, -1)
            .to(device=device, dtype=torch.long),
            "tokenized_prompt_mask": torch.from_numpy(token_mask)
            .view(1, -1)
            .to(device=device, dtype=torch.bool),
        }
    )
    torch.manual_seed(0)
    actions = model.sample_actions(device, obs, noise=None, num_steps=10)
    action = actions[0, 0, :14].detach().float().cpu().numpy()
    action = reference.unnormalize_action(action, stats)
    return float(np.linalg.norm(action[ARM_DIMS]))


def load_our_policy() -> tuple[Any, Any, Any, Any]:
    import robotwin_eval_harness as harness
    import torch
    from lerobot.envs.configs import RoboTwinEnvConfig
    from lerobot.envs.factory import make_env_pre_post_processors
    from lerobot.policies.factory import make_pre_post_processors
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.modeling_pi05 import PI05Policy

    checkpoint = OUR_CHECKPOINT
    env_cfg = RoboTwinEnvConfig(
        task="beat_block_hammer",
        episode_length=1200,
        obs_type="pixels_agent_pos",
        camera_names="cam_high,cam_left_wrist,cam_right_wrist",
        observation_height=harness.OBS_HEIGHT,
        observation_width=harness.OBS_WIDTH,
        action_mode="joint",
    )
    policy_cfg = PI05Config.from_pretrained(str(checkpoint), local_files_only=True)
    policy_cfg.device = "cpu"
    policy_cfg.pretrained_path = checkpoint
    rename_map = harness.derive_camera_rename_map(policy_cfg)
    policy = PI05Policy.from_pretrained(
        str(checkpoint),
        config=policy_cfg,
        local_files_only=True,
    )
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=policy_cfg,
        pretrained_path=checkpoint,
        preprocessor_overrides={
            "device_processor": {"device": "cpu"},
            "rename_observations_processor": {"rename_map": rename_map},
            "tokenizer_processor": {"tokenizer_name": str(TOKENIZER)},
        },
    )
    env_preprocessor, env_postprocessor = make_env_pre_post_processors(
        env_cfg, policy.config
    )
    policy.eval()
    print(f"OUR_MODEL_READY checkpoint={checkpoint}", flush=True)
    return policy, preprocessor, postprocessor, env_postprocessor


def our_forward(
    policy: Any,
    preprocessor: Any,
    postprocessor: Any,
    env_postprocessor: Any,
    state: np.ndarray,
    frames: dict[str, np.ndarray],
) -> float:
    import robotwin_eval_harness as harness
    import torch
    from lerobot.utils.constants import ACTION

    observation = {
        "agent_pos": state,
        "pixels": {
            camera: torch.from_numpy(frame)
            .permute(2, 0, 1)
            .contiguous()
            .to(dtype=torch.float32)
            .div(255.0)
            for camera, frame in frames.items()
        },
    }
    batch = harness.make_policy_batch(
        observation,
        TASK,
        None,
        preprocessor,
        policy.config,
    )
    policy.reset()
    torch.manual_seed(0)
    with torch.inference_mode():
        action = policy.select_action(batch)
    action = postprocessor(action)
    action = env_postprocessor({ACTION: action})[ACTION]
    action = action.detach().float().cpu().numpy().reshape(-1)[:14]
    return float(np.linalg.norm(action[ARM_DIMS]))


def summarize(values: list[float]) -> str:
    if not values:
        return "unavailable"
    return (
        f"{float(np.mean(values)):.8f} +/- "
        f"{float(np.min(values)):.8f}/{float(np.max(values)):.8f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-only", action="store_true")
    parser.add_argument("--our-only", action="store_true")
    parser.add_argument("--model-path", default=str(DEFAULT_CHECKPOINT))
    parser.add_argument("--norm-mode", choices=("quantiles", "mean_std"), default="quantiles")
    parser.add_argument("--our-model-path", default=str(DEFAULT_CHECKPOINT))
    args = parser.parse_args()

    global REFERENCE_CHECKPOINT, REFERENCE_NORM_MODE, OUR_CHECKPOINT
    REFERENCE_CHECKPOINT = resolve_checkpoint(args.model_path)
    REFERENCE_NORM_MODE = args.norm_mode
    OUR_CHECKPOINT = resolve_checkpoint(args.our_model_path)

    print(f"OBS_SPECS={json.dumps(OBS_SPECS)}", flush=True)
    print(
        f"REFERENCE_CONFIG checkpoint={REFERENCE_CHECKPOINT} "
        f"norm_mode={REFERENCE_NORM_MODE} script={REFERENCE_SCRIPT}",
        flush=True,
    )
    observations = load_observation_rows(OBS_SPECS)
    load_frames(observations)
    reference_model = None if args.our_only else load_reference_model()
    our_model = None
    if not args.reference_only:
        try:
            our_model = load_our_policy()
        except Exception:
            print("OUR_MODEL_BLOCKER", flush=True)
            print(traceback.format_exc(), end="", flush=True)

    rows: list[dict[str, Any]] = []
    for obs in observations:
        row: dict[str, Any] = {
            "episode": obs["episode"],
            "position": obs["position"],
            "reference": None,
            "our": None,
        }
        if reference_model is not None:
            try:
                row["reference"] = reference_forward(
                    *reference_model,
                    obs["state"],
                    obs["frames"],
                )
            except Exception as exc:
                print(
                    f"REFERENCE_TRACEBACK episode={obs['episode']} "
                    f"position={obs['position']}",
                    flush=True,
                )
                print(traceback.format_exc(), end="", flush=True)
                row["reference_error"] = f"{type(exc).__name__}: {exc}"
        if our_model is not None:
            try:
                row["our"] = our_forward(
                    *our_model,
                    obs["state"],
                    obs["frames"],
                )
            except Exception as exc:
                print(
                    f"OUR_TRACEBACK episode={obs['episode']} "
                    f"position={obs['position']}",
                    flush=True,
                )
                print(traceback.format_exc(), end="", flush=True)
                row["our_error"] = f"{type(exc).__name__}: {exc}"
        rows.append(row)
        print(
            f"ROW episode={row['episode']} position={row['position']} "
            f"REFERENCE_ARM_L2={row['reference']} "
            f"OUR_ARM_L2={row['our']}",
            flush=True,
        )

    print("TABLE episode | position | REFERENCE_ARM_L2 | OUR_ARM_L2", flush=True)
    for row in rows:
        reference = (
            f"{row['reference']:.8f}"
            if row["reference"] is not None
            else f"BLOCKER={row.get('reference_error', 'unknown')}"
        )
        our = (
            f"{row['our']:.8f}"
            if row["our"] is not None
            else f"BLOCKER={row.get('our_error', 'not_run')}"
        )
        print(
            f"{row['episode']} | {row['position']} | {reference} | {our}",
            flush=True,
        )
    print(
        "SUMMARY REFERENCE_ARM_L2 mean +/- min/max = "
        + summarize([row["reference"] for row in rows if row["reference"] is not None]),
        flush=True,
    )
    print(
        "SUMMARY OUR_ARM_L2 mean +/- min/max = "
        + summarize([row["our"] for row in rows if row["our"] is not None]),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
