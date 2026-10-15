#!/usr/bin/env python3
"""Create a tiny smoke-only LeRobot dataset for the D013 pi05 training path."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np
from lerobot.datasets.lerobot_dataset import LeRobotDataset


IMAGE_KEYS = (
    "observation.images.robot0_agentview_left",
    "observation.images.robot0_agentview_right",
    "observation.images.robot0_eye_in_hand",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create a synthetic smoke-only LeRobot dataset.")
    parser.add_argument("--output", type=Path, default=Path("/root/autodl-tmp/EvolveVLA/data/smoke_lerobot"))
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--frames", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20261001)
    return parser.parse_args()


def features() -> dict:
    image_feature = {
        "dtype": "image",
        "shape": (3, 256, 256),
        "names": ("channel", "height", "width"),
    }
    return {
        "observation.state": {
            "dtype": "float32",
            "shape": (16,),
            "names": [f"state_{i}" for i in range(16)],
        },
        "action": {
            "dtype": "float32",
            "shape": (12,),
            "names": [f"action_{i}" for i in range(12)],
        },
        **{key: dict(image_feature) for key in IMAGE_KEYS},
    }


def make_image(rng: np.random.Generator, episode: int, frame: int, camera_idx: int) -> np.ndarray:
    base = np.zeros((256, 256, 3), dtype=np.uint8)
    base[:, :, 0] = (episode * 50 + camera_idx * 40) % 255
    base[:, :, 1] = (frame * 45 + camera_idx * 20) % 255
    base[:, :, 2] = rng.integers(0, 255, size=(256, 256), dtype=np.uint8)
    return base


def main() -> int:
    args = parse_args()
    if not str(args.output).startswith("/root/autodl-tmp/"):
        raise ValueError(f"Output must stay on data volume: {args.output}")
    if args.episodes < 2 or args.episodes > 5:
        raise ValueError("--episodes must be between 2 and 5 for this smoke dataset")
    if args.frames < 2:
        raise ValueError("--frames must be at least 2")

    shutil.rmtree(args.output, ignore_errors=True)
    rng = np.random.default_rng(args.seed)

    dataset = LeRobotDataset.create(
        repo_id="robocasa/d013_smoke_only",
        fps=10,
        features=features(),
        root=args.output,
        robot_type="synthetic_robocasa_smoke",
        use_videos=False,
        image_writer_processes=0,
        image_writer_threads=0,
    )
    task = "SMOKE_ONLY synthetic RoboCasa pipeline check; not real demonstrations"
    for ep in range(args.episodes):
        for frame_idx in range(args.frames):
            state = np.linspace(-1.0, 1.0, 16, dtype=np.float32) + ep * 0.01 + frame_idx * 0.001
            action = np.linspace(0.25, -0.25, 12, dtype=np.float32) + ep * 0.002 - frame_idx * 0.001
            frame = {
                "observation.state": state.astype(np.float32),
                "action": action.astype(np.float32),
                "task": task,
            }
            for cam_idx, key in enumerate(IMAGE_KEYS):
                frame[key] = make_image(rng, ep, frame_idx, cam_idx)
            dataset.add_frame(frame)
        dataset.save_episode(parallel_encoding=False)
    dataset.finalize()

    loaded = LeRobotDataset(
        repo_id="robocasa/d013_smoke_only",
        root=args.output,
        download_videos=False,
        return_uint8=True,
    )
    sample = loaded[0]
    summary = {
        "path": str(args.output),
        "episodes": loaded.num_episodes,
        "frames": loaded.num_frames,
        "state_shape": list(sample["observation.state"].shape),
        "action_shape": list(sample["action"].shape),
        "image_keys": IMAGE_KEYS,
        "image_shapes": {key: list(sample[key].shape) for key in IMAGE_KEYS},
        "meta_info_exists": (args.output / "meta/info.json").is_file(),
        "smoke_only": True,
    }
    print("SMOKE_DATASET_OK " + json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
