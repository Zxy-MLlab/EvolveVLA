#!/usr/bin/env python3
"""Verify one RoboTwin LeRobot v3.0 dataset slice without using a GPU.

The public Hub repository stores the LeRobot dataset below a ``clean`` or
``random`` directory. This script materializes one subset into an isolated
local LeRobot root, selects one episode by instruction, and loads it with the
installed LeRobotDataset implementation.

By default the check validates metadata and parquet loadability without
downloading the large AV1 video shards. Pass ``--decode-video`` to download
and decode the three video files referenced by the selected episode.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
from huggingface_hub import HfApi, hf_hub_download
from lerobot.datasets import LeRobotDataset


REPO_ID = "OpenMOSS-Team/robotwin2.0-lerobot-v3.0"
DEFAULT_SPLIT = "clean"
DEFAULT_CACHE_ROOT = Path("/root/autodl-tmp/EvolveVLA/data/gate2_robotwin_cache")
EXPECTED_STATE_KEY = "observation.state"
EXPECTED_ACTION_KEY = "action"
EXPECTED_CAMERA_KEYS = (
    "observation.images.cam_high",
    "observation.images.cam_left_wrist",
    "observation.images.cam_right_wrist",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--split",
        choices=("clean", "random"),
        default=DEFAULT_SPLIT,
        help="Nested dataset subset to verify.",
    )
    parser.add_argument(
        "--task-index",
        type=int,
        default=None,
        help="Natural-language task_index from tasks.parquet; default selects episode 0.",
    )
    parser.add_argument(
        "--cache-root",
        type=Path,
        default=DEFAULT_CACHE_ROOT,
        help="All downloaded material and the LeRobot root stay below this path.",
    )
    parser.add_argument(
        "--decode-video",
        action="store_true",
        help="Download the selected episode's three video shards and decode frame 0.",
    )
    return parser.parse_args()


def hub_download(repo_path: str, cache_root: Path) -> Path:
    """Download one public file without using an HF token."""
    downloaded = hf_hub_download(
        repo_id=REPO_ID,
        filename=repo_path,
        repo_type="dataset",
        local_dir=str(cache_root),
        token=False,
    )
    return Path(downloaded)


def ensure_symlink(link: Path, target: Path) -> None:
    link.parent.mkdir(parents=True, exist_ok=True)
    if link.is_symlink():
        if link.resolve() == target.resolve():
            return
        link.unlink()
    elif link.exists():
        raise RuntimeError(f"Refusing to replace existing path: {link}")
    link.symlink_to(target, target_is_directory=target.is_dir())


def shape_of(value: Any) -> list[int]:
    if hasattr(value, "shape"):
        return [int(dim) for dim in value.shape]
    return [int(dim) for dim in np.asarray(value).shape]


def get_episode_files(
    split: str,
    cache_root: Path,
    api: HfApi,
) -> tuple[dict[str, Any], list[dict[str, Any]], dict[int, str]]:
    split_root = cache_root / split
    info_path = hub_download(f"{split}/meta/info.json", cache_root)
    tasks_path = hub_download(f"{split}/meta/tasks.parquet", cache_root)
    stats_path = hub_download(f"{split}/meta/stats.json", cache_root)
    del stats_path

    info = json.loads(info_path.read_text(encoding="utf-8"))
    episode_paths = sorted(
        path
        for path in api.list_repo_files(REPO_ID, repo_type="dataset")
        if path.startswith(f"{split}/meta/episodes/") and path.endswith(".parquet")
    )
    if not episode_paths:
        raise RuntimeError(f"No episode metadata files found below {split}/meta/episodes/")
    for path in episode_paths:
        hub_download(path, cache_root)

    import pyarrow.parquet as pq

    tasks = pq.read_table(tasks_path).to_pylist()
    task_by_index = {int(row["task_index"]): str(row["task"]) for row in tasks}
    episodes: list[dict[str, Any]] = []
    for path in episode_paths:
        episodes.extend(pq.read_table(split_root / Path(path).relative_to(split)).to_pylist())
    episodes.sort(key=lambda row: int(row["episode_index"]))
    return info, episodes, task_by_index


def select_episode(
    episodes: list[dict[str, Any]],
    task_by_index: dict[int, str],
    task_index: int | None,
) -> tuple[dict[str, Any], str]:
    if task_index is None:
        episode = episodes[0]
        task_text = str(episode["tasks"][0])
        return episode, task_text

    if task_index not in task_by_index:
        raise ValueError(f"task_index={task_index} is not present in tasks.parquet")
    requested_text = task_by_index[task_index]
    for episode in episodes:
        if requested_text in [str(task) for task in episode["tasks"]]:
            return episode, requested_text
    raise ValueError(f"No episode references task_index={task_index}: {requested_text!r}")


def materialize_selected_files(
    split: str,
    info: dict[str, Any],
    episode: dict[str, Any],
    cache_root: Path,
    decode_video: bool,
) -> tuple[Path, list[Path]]:
    split_root = cache_root / split
    data_rel = info["data_path"].format(
        chunk_index=int(episode["data/chunk_index"]),
        file_index=int(episode["data/file_index"]),
    )
    hub_download(f"{split}/{data_rel}", cache_root)

    video_paths: list[Path] = []
    for camera_key in EXPECTED_CAMERA_KEYS:
        video_rel = info["video_path"].format(
            video_key=camera_key,
            chunk_index=int(episode[f"videos/{camera_key}/chunk_index"]),
            file_index=int(episode[f"videos/{camera_key}/file_index"]),
        )
        video_path = split_root / video_rel
        video_path.parent.mkdir(parents=True, exist_ok=True)
        if decode_video:
            if video_path.exists():
                video_path.unlink()
            hub_download(f"{split}/{video_rel}", cache_root)
        else:
            # LeRobot's cache probe requires the referenced paths to exist even
            # when the caller only wants raw parquet rows and metadata checks.
            video_path.touch(exist_ok=True)
        video_paths.append(video_path)

    return split_root, video_paths


def main() -> int:
    # This verifier is intentionally CPU-only. It never calls a CUDA API.
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
    os.environ.pop("HF_TOKEN", None)
    args = parse_args()

    result: dict[str, Any] = {
        "gate": "GATE 2",
        "repo_id": REPO_ID,
        "split": args.split,
        "status": "FAIL",
        "checks": {},
    }

    try:
        args.cache_root.mkdir(parents=True, exist_ok=True)
        api = HfApi(token=False)
        info, episodes, task_by_index = get_episode_files(args.split, args.cache_root, api)
        episode, task_text = select_episode(episodes, task_by_index, args.task_index)
        split_root, video_paths = materialize_selected_files(
            args.split,
            info,
            episode,
            args.cache_root,
            args.decode_video,
        )

        dataset_root = args.cache_root / f"lerobot_root_{args.split}"
        dataset_root.mkdir(parents=True, exist_ok=True)
        for dirname in ("meta", "data", "videos"):
            ensure_symlink(dataset_root / dirname, split_root / dirname)

        dataset = LeRobotDataset(
            REPO_ID,
            root=dataset_root,
            episodes=[int(episode["episode_index"])],
            download_videos=False,
            return_uint8=True,
            token=False,
        )
        raw_sample = dataset.get_raw_item(0)
        features = dataset.meta.features
        camera_keys = sorted(dataset.meta.camera_keys)
        state_shape = shape_of(raw_sample[EXPECTED_STATE_KEY])
        action_shape = shape_of(raw_sample[EXPECTED_ACTION_KEY])
        feature_state_shape = list(features[EXPECTED_STATE_KEY]["shape"])
        feature_action_shape = list(features[EXPECTED_ACTION_KEY]["shape"])
        codec_by_camera = {
            key: (features[key].get("info") or {}).get("video.codec")
            for key in EXPECTED_CAMERA_KEYS
        }

        result["selected_episode"] = int(episode["episode_index"])
        result["selected_task"] = task_text
        result["dataset_length"] = int(len(dataset))
        result["checks"] = {
            "state_shape": {
                "feature": feature_state_shape,
                "sample": state_shape,
                "pass": feature_state_shape == [14] and state_shape == [14],
            },
            "action_shape": {
                "feature": feature_action_shape,
                "sample": action_shape,
                "pass": feature_action_shape == [14] and action_shape == [14],
            },
            "camera_keys": {
                "expected": list(EXPECTED_CAMERA_KEYS),
                "actual": camera_keys,
                "pass": camera_keys == sorted(EXPECTED_CAMERA_KEYS),
            },
            "video_codec": {
                "codec_by_camera": codec_by_camera,
                "expected": "av1",
                "pass": set(codec_by_camera.values()) == {"av1"},
                "decode_attempted": args.decode_video,
                "files": [str(path) for path in video_paths],
            },
        }

        if args.decode_video:
            decoded_sample = dataset[0]
            result["checks"]["video_decode"] = {
                key: {
                    "shape": shape_of(decoded_sample[key]),
                    "pass": shape_of(decoded_sample[key])[-1] == 3,
                }
                for key in EXPECTED_CAMERA_KEYS
            }

        result["status"] = (
            "PASS"
            if all(check["pass"] for check in result["checks"].values() if isinstance(check, dict) and "pass" in check)
            and (
                not args.decode_video
                or all(item["pass"] for item in result["checks"]["video_decode"].values())
            )
            else "FAIL"
        )
    except Exception as exc:
        result["error"] = f"{type(exc).__name__}: {exc}"

    print(json.dumps(result, indent=2, sort_keys=True))
    print(f"{result['gate']} = {result['status']}")
    return 0 if result["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())
