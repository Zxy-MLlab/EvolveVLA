#!/usr/bin/env python3
import argparse
import hashlib
import importlib.util
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import numpy as np
import pandas as pd
import torch


ROOT = Path("/root/autodl-tmp/EvolveVLA")
REFERENCE_NAME = "gate3_reference_forward_4befcb36.py"
DEFAULT_FRAMES = "50:31:7219"
NUM_STEPS = 10
ACTION_DIMS = 14
EXPECTED_CHUNK = 50


@dataclass(frozen=True)
class FrameSpec:
    episode: int
    position: int
    video_index: int


@dataclass(frozen=True)
class ResolvedFrame:
    spec: FrameSpec
    dataset_from_index: int
    dataset_to_index: int
    length: int
    derived_video_index: int
    meta_source: str
    data_source: str
    state: np.ndarray
    task_index: int


def import_reference() -> Any:
    reference_path = Path(__file__).with_name(REFERENCE_NAME)
    spec = importlib.util.spec_from_file_location(
        "gate3_reference_forward_4befcb36", reference_path
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import reference helper: {reference_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def parse_frames(value: str) -> list[FrameSpec]:
    frames = []
    for raw in value.split(","):
        item = raw.strip()
        if not item:
            continue
        parts = item.split(":")
        if len(parts) != 3:
            raise ValueError(f"frame must be ep:pos:video_index, got {item!r}")
        frames.append(FrameSpec(*(int(part) for part in parts)))
    if not frames:
        raise ValueError("at least one frame is required")
    return frames


def load_episode_metadata(data_root: Path) -> pd.DataFrame:
    episode_dir = data_root / "meta/episodes/chunk-000"
    files = sorted(episode_dir.glob("file-*.parquet"))
    if not files:
        raise RuntimeError(f"missing episode metadata parquet files: {episode_dir}")
    return pd.concat(
        [pd.read_parquet(path).assign(_source=str(path)) for path in files],
        ignore_index=True,
    )


def load_frame_table(data_root: Path) -> tuple[pd.DataFrame, str]:
    path = data_root / "data/chunk-000/file-000.parquet"
    if not path.exists():
        raise RuntimeError(f"missing frame parquet: {path}")
    return pd.read_parquet(path), str(path)


def resolve_frame(
    spec: FrameSpec, episodes: pd.DataFrame, frames: pd.DataFrame, data_source: str
) -> ResolvedFrame:
    matches = episodes[episodes["episode_index"] == spec.episode]
    if len(matches) != 1:
        raise RuntimeError(f"episode {spec.episode} matched {len(matches)} rows")
    row = matches.iloc[0]
    length = int(row["length"])
    dataset_from_index = int(row["dataset_from_index"])
    dataset_to_index = int(row["dataset_to_index"])
    if not 0 <= spec.position < length:
        raise RuntimeError(
            f"position out of episode range: episode={spec.episode} "
            f"position={spec.position} length={length}"
        )
    derived_video_index = dataset_from_index + spec.position
    if spec.video_index != derived_video_index:
        raise RuntimeError(
            f"video_index mismatch: episode={spec.episode} "
            f"position={spec.position} supplied={spec.video_index} "
            f"derived={derived_video_index}"
        )
    if spec.video_index >= len(frames):
        raise RuntimeError(
            f"video_index out of frame table: {spec.video_index} >= {len(frames)}"
        )
    frame_row = frames.iloc[spec.video_index]
    if int(frame_row["episode_index"]) != spec.episode:
        raise RuntimeError(
            f"frame table episode mismatch: video_index={spec.video_index} "
            f"actual={int(frame_row['episode_index'])} expected={spec.episode}"
        )
    if int(frame_row["frame_index"]) != spec.position:
        raise RuntimeError(
            f"frame table position mismatch: video_index={spec.video_index} "
            f"actual={int(frame_row['frame_index'])} expected={spec.position}"
        )
    return ResolvedFrame(
        spec=spec,
        dataset_from_index=dataset_from_index,
        dataset_to_index=dataset_to_index,
        length=length,
        derived_video_index=derived_video_index,
        meta_source=str(row["_source"]),
        data_source=data_source,
        state=np.asarray(frame_row["observation.state"], dtype=np.float32),
        task_index=int(frame_row["task_index"]),
    )


def camera_paths(ref: Any) -> dict[str, Path]:
    return {
        "cam_high": ref.DATA
        / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
        "cam_left_wrist": ref.DATA
        / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
        "cam_right_wrist": ref.DATA
        / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
    }


def frame_hashes(ref: Any, video_index: int) -> dict[str, str]:
    hashes = {}
    for camera, path in camera_paths(ref).items():
        image = ref.frame(path, video_index)
        hashes[camera] = hashlib.sha256(image.tobytes()).hexdigest()
    return hashes


def configure_reference(
    ref: Any, resolved: ResolvedFrame, checkpoint: Path, norm_mode: str
) -> None:
    ref.CKPT = checkpoint
    ref.NORM_MODE = norm_mode
    ref.EPISODE_INDEX = resolved.spec.episode
    ref.POSITION = resolved.spec.position
    ref.VIDEO_INDEX = resolved.spec.video_index
    ref.STATE = resolved.state
    ref.EXPECTED_IMAGE_HASHES = frame_hashes(ref, resolved.spec.video_index)


def capture_action_chunk(ref: Any, st: dict[str, Any]) -> tuple[torch.Tensor, int]:
    if str(ref.OPENPI) not in sys.path:
        sys.path.insert(0, str(ref.OPENPI))
    from openpi.models_pytorch import pi0_pytorch

    captured: dict[str, torch.Tensor] = {}
    original = pi0_pytorch.PI0Pytorch.sample_actions

    def wrapped_sample_actions(self, device, obs, noise=None, num_steps=NUM_STEPS):
        if noise is not None:
            raise RuntimeError("probe requires noise=None")
        if int(num_steps) != NUM_STEPS:
            raise RuntimeError(f"probe requires num_steps={NUM_STEPS}")
        chunk = original(self, device, obs, noise=noise, num_steps=num_steps)
        captured["chunk"] = chunk.detach().float().cpu().contiguous()
        return chunk

    pi0_pytorch.PI0Pytorch.sample_actions = wrapped_sample_actions
    try:
        _arm_l2, loaded_keys = ref.openpi_forward(torch.device("cpu"), st)
    finally:
        pi0_pytorch.PI0Pytorch.sample_actions = original
    if "chunk" not in captured:
        raise RuntimeError("sample_actions did not return a captured chunk")
    return captured["chunk"], loaded_keys


def action_for_space(ref: Any, action: np.ndarray, st: dict[str, Any], space: str) -> np.ndarray:
    action14 = np.asarray(action[:ACTION_DIMS], dtype=np.float32)
    if space == "normalized":
        return action14
    return ref.unnormalize_action(action14, st)


def arm_norm_profile(
    ref: Any, chunk: torch.Tensor, st: dict[str, Any], space: str
) -> np.ndarray:
    if chunk.ndim != 3 or chunk.shape[0] != 1:
        raise RuntimeError(f"unexpected action chunk shape: {tuple(chunk.shape)}")
    if int(chunk.shape[1]) != EXPECTED_CHUNK:
        raise RuntimeError(
            f"unexpected chunk length: {int(chunk.shape[1])} != {EXPECTED_CHUNK}"
        )
    raw = chunk[0].numpy()
    values = []
    for position in range(raw.shape[0]):
        action = action_for_space(ref, raw[position], st, space)
        values.append(float(np.linalg.norm(action[ref.ARM_DIMS])))
    return np.asarray(values, dtype=np.float32)


def print_frame_result(
    ref: Any,
    resolved: ResolvedFrame,
    chunk: torch.Tensor,
    loaded_keys: int,
    norms: np.ndarray,
    space: str,
) -> None:
    spec = resolved.spec
    print(
        "FRAME_DERIVATION "
        f"episode={spec.episode} position={spec.position} "
        f"dataset_from_index={resolved.dataset_from_index} "
        f"derived_video_index={resolved.derived_video_index} "
        f"supplied_video_index={spec.video_index} "
        f"dataset_to_index={resolved.dataset_to_index} length={resolved.length} "
        f"task_index={resolved.task_index} source={resolved.meta_source}"
    )
    print(
        "FRAME_RESULT "
        f"episode={spec.episode} position={spec.position} "
        f"video_index={spec.video_index} space={space} "
        f"chunk_shape={list(chunk.shape)} loaded_keys={loaded_keys} "
        f"arm_dims={ref.ARM_DIMS}"
    )
    print(
        "ARM_NORM_VECTOR "
        f"episode={spec.episode} position={spec.position} "
        f"video_index={spec.video_index} space={space} "
        f"values={json.dumps([float(v) for v in norms])}"
    )
    print(
        "ARM_NORM_SUMMARY "
        f"episode={spec.episode} position={spec.position} "
        f"video_index={spec.video_index} space={space} "
        f"peak={float(np.max(norms)):.9f} argmax={int(np.argmax(norms))} "
        f"mean={float(np.mean(norms)):.9f} pos0={float(norms[0]):.9f}"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", required=True)
    parser.add_argument(
        "--norm-mode", required=True, choices=("quantiles", "mean_std")
    )
    parser.add_argument("--frames", default=DEFAULT_FRAMES)
    parser.add_argument(
        "--space", choices=("normalized", "physical"), default="physical"
    )
    args = parser.parse_args()

    torch.manual_seed(0)
    ref = import_reference()
    checkpoint = ref.resolve_checkpoint(args.model_path)
    episodes = load_episode_metadata(ref.DATA)
    frame_table, data_source = load_frame_table(ref.DATA)
    resolved_frames = [
        resolve_frame(spec, episodes, frame_table, data_source)
        for spec in parse_frames(args.frames)
    ]

    print(
        "FULL_CHUNK_PROFILE_CONFIG "
        f"checkpoint={checkpoint} norm_mode={args.norm_mode} space={args.space} "
        f"device=cpu cuda_visible_devices={os.environ.get('CUDA_VISIBLE_DEVICES', '')!r} "
        f"num_steps={NUM_STEPS} noise=None frames={args.frames}"
    )

    for resolved in resolved_frames:
        configure_reference(ref, resolved, checkpoint, args.norm_mode)
        st = ref.stats(checkpoint, args.norm_mode)
        print(f"NORMALIZATION_SOURCE={st.get('_source', 'unknown')}")
        print(f"NORMALIZATION_APPLIED={args.norm_mode.upper()}")
        chunk, loaded_keys = capture_action_chunk(ref, st)
        norms = arm_norm_profile(ref, chunk, st, args.space)
        print_frame_result(ref, resolved, chunk, loaded_keys, norms, args.space)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
