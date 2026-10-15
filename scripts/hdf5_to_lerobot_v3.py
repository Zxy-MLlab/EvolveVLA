#!/usr/bin/env python3
"""Convert RoboTwin scripted-expert HDF5 episodes to a LeRobot v3 tree.

The checked-in RoboTwin collector currently writes the native XPolicyLab-style
layout (/state, /action, /vision). Older RoboTwin exports use
/observations/qpos and /action. This converter accepts both layouts and keeps
the source HDF5 directory parameterized; it never starts the simulator.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import h5py
import numpy as np
import pandas as pd

try:
    import pyarrow as pa
    import pyarrow.parquet as pq
except ImportError as exc:  # pragma: no cover - dependency guard
    raise SystemExit("pyarrow is required to write LeRobot v3 parquet files") from exc


CAMERA_FEATURES = {
    "cam_high": "observation.images.cam_high",
    "cam_left_wrist": "observation.images.cam_left_wrist",
    "cam_right_wrist": "observation.images.cam_right_wrist",
}

STATE_NAMES = [
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

CAMERA_CANDIDATES = {
    "cam_high": [
        "/vision/cam_head/colors",
        "/vision/cam_high/colors",
        "/observations/images/cam_high",
        "/observations/cam_high",
        "/observations/image/cam_high",
        "/camera/cam_high",
    ],
    "cam_left_wrist": [
        "/vision/cam_left_wrist/colors",
        "/observations/images/cam_left_wrist",
        "/observations/cam_left_wrist",
        "/observations/image/cam_left_wrist",
        "/camera/cam_left_wrist",
    ],
    "cam_right_wrist": [
        "/vision/cam_right_wrist/colors",
        "/observations/images/cam_right_wrist",
        "/observations/cam_right_wrist",
        "/observations/image/cam_right_wrist",
        "/camera/cam_right_wrist",
    ],
}


def _first_dataset(handle: h5py.File, paths: Iterable[str]) -> h5py.Dataset | None:
    for path in paths:
        if path in handle and isinstance(handle[path], h5py.Dataset):
            return handle[path]
    return None


def _as_time_matrix(value: Any, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 1:
        array = array[:, None]
    if array.ndim != 2:
        raise ValueError(f"{name} must be a rank-1/2 time series, got {array.shape}")
    return array


def _read_vector(handle: h5py.File, paths: Iterable[str], name: str) -> np.ndarray | None:
    dataset = _first_dataset(handle, paths)
    if dataset is None:
        return None
    return _as_time_matrix(dataset[()], name)


def _combine_native_joints(handle: h5py.File, group_name: str) -> np.ndarray | None:
    group = handle.get(group_name)
    if not isinstance(group, h5py.Group):
        return None
    fields = [
        "left_arm_joint_states",
        "left_ee_joint_states",
        "right_arm_joint_states",
        "right_ee_joint_states",
    ]
    values = []
    for field in fields:
        if field not in group:
            return None
        values.append(_as_time_matrix(group[field][()], f"/{group_name}/{field}"))
    lengths = {len(value) for value in values}
    if len(lengths) != 1:
        raise ValueError(f"/{group_name} joint streams have different lengths: {sorted(lengths)}")
    result = np.concatenate(values, axis=1)
    if result.shape[1] != 14:
        raise ValueError(f"/{group_name} native joint width is {result.shape[1]}, expected 14")
    return result


def _read_state(handle: h5py.File) -> tuple[np.ndarray, str]:
    native = _combine_native_joints(handle, "state")
    if native is not None:
        return native, "/state/*_joint_states"
    for path in ("/observations/qpos", "/observations/state", "/state/qpos"):
        if path in handle and isinstance(handle[path], h5py.Dataset):
            state = _as_time_matrix(handle[path][()], path)
            if state.shape[1] != 14:
                raise ValueError(f"{path} width is {state.shape[1]}, expected 14")
            return state, path
    raise KeyError(
        "No state stream found. Expected /state/{left_arm_joint_states,...}, "
        "/observations/qpos, /observations/state, or /state/qpos."
    )


def _read_action(handle: h5py.File) -> tuple[np.ndarray, str]:
    native = _combine_native_joints(handle, "action")
    if native is not None:
        return native, "/action/*_joint_states"
    for path in ("/action", "/actions"):
        if path in handle and isinstance(handle[path], h5py.Dataset):
            action = _as_time_matrix(handle[path][()], path)
            if action.shape[1] != 14:
                raise ValueError(f"{path} width is {action.shape[1]}, expected 14")
            return action, path
    raise KeyError(
        "No action stream found. Expected /action/{left_arm_joint_states,...}, "
        "/action, or /actions."
    )


def _decode_image_bit(value: Any) -> np.ndarray:
    """Decode RoboTwin/XPolicyLab encoded bytes, preserving RGB order."""
    try:
        robotwin_root = Path(__file__).resolve().parents[1] / "robotwin_repo"
        if str(robotwin_root) not in sys.path:
            sys.path.insert(0, str(robotwin_root))
        from data.decode_image_bit import decode_image_bit

        decoded = decode_image_bit(value)
        array = np.asarray(decoded)
        if array.ndim == 3:
            return array
    except (ImportError, ModuleNotFoundError):
        pass

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError(
            "OpenCV is required to decode encoded HDF5 camera buffers"
        ) from exc
    if isinstance(value, str):
        value = value.encode("utf-8")
    if isinstance(value, np.ndarray):
        value = value.tobytes() if value.dtype.kind in {"S", "U"} else value
    if isinstance(value, (bytes, bytearray, memoryview)):
        value = bytes(value).rstrip(b"\x00")
    image = cv2.imdecode(np.frombuffer(value, dtype=np.uint8), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError("OpenCV could not decode an HDF5 camera buffer")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _read_camera(dataset: h5py.Dataset, shape_dataset: h5py.Dataset | None) -> np.ndarray:
    raw = dataset[()]
    array = np.asarray(raw)
    if array.dtype.kind in {"S", "U", "O"} or (
        array.ndim == 1 and array.dtype == np.uint8
    ):
        frames = [_decode_image_bit(item) for item in array]
        result = np.stack(frames, axis=0)
    elif array.ndim == 4:
        if array.shape[-1] in (1, 3, 4):
            result = array
        elif array.shape[1] in (1, 3, 4):
            result = np.transpose(array, (0, 2, 3, 1))
        else:
            raise ValueError(f"Unsupported raw camera shape {array.shape}")
    else:
        raise ValueError(f"Unsupported camera dataset dtype/shape {array.dtype}/{array.shape}")

    if result.shape[-1] == 1:
        result = np.repeat(result, 3, axis=-1)
    if result.shape[-1] == 4:
        result = result[..., :3]
    if result.dtype != np.uint8:
        if np.issubdtype(result.dtype, np.floating) and float(np.nanmax(result)) <= 1.0:
            result = np.rint(result * 255.0)
        result = np.clip(result, 0, 255).astype(np.uint8)
    if result.ndim != 4 or result.shape[-1] != 3:
        raise ValueError(f"Decoded camera has invalid shape {result.shape}")
    if shape_dataset is not None and result.shape[1:] != tuple(shape_dataset[()].tolist()):
        raise ValueError(
            f"Decoded camera shape {result.shape[1:]} disagrees with stored shape "
            f"{tuple(shape_dataset[()].tolist())}"
        )
    return result


def _read_cameras(handle: h5py.File) -> dict[str, np.ndarray]:
    cameras = {}
    for camera_name, candidates in CAMERA_CANDIDATES.items():
        dataset = _first_dataset(handle, candidates)
        if dataset is None:
            raise KeyError(
                f"Missing required camera {camera_name}; tried {', '.join(candidates)}"
            )
        shape_dataset = None
        if dataset.name.startswith("/vision/"):
            parent = handle[dataset.name.rsplit("/", 1)[0]]
            if isinstance(parent, h5py.Group) and "shape" in parent:
                shape_dataset = parent["shape"]
        cameras[camera_name] = _read_camera(dataset, shape_dataset)
    return cameras


def _episode_index(path: Path, fallback: int) -> int:
    match = re.search(r"(?:episode[_-])?(\d+)", path.stem)
    return int(match.group(1)) if match else fallback


def _episode_files(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    candidates = sorted(
        {path for suffix in ("*.hdf5", "*.h5") for path in source.rglob(suffix)}
    )
    if not candidates:
        raise FileNotFoundError(f"No .hdf5 or .h5 episodes found under {source}")
    return candidates


@dataclass
class Episode:
    episode_index: int
    state: np.ndarray
    action: np.ndarray
    cameras: dict[str, np.ndarray]
    source: Path

    @property
    def length(self) -> int:
        return int(len(self.state))


def load_episode(path: Path, fallback_index: int) -> Episode:
    with h5py.File(path, "r") as handle:
        state, _ = _read_state(handle)
        action, _ = _read_action(handle)
        cameras = _read_cameras(handle)
    lengths = {"state": len(state), "action": len(action)}
    lengths.update({name: len(frames) for name, frames in cameras.items()})
    if len(set(lengths.values())) != 1:
        raise ValueError(f"{path} has mismatched stream lengths: {lengths}")
    return Episode(_episode_index(path, fallback_index), state, action, cameras, path)


class RunningStats:
    def __init__(self, width: int) -> None:
        self.count = 0
        self.minimum = np.full(width, np.inf, dtype=np.float64)
        self.maximum = np.full(width, -np.inf, dtype=np.float64)
        self.mean = np.zeros(width, dtype=np.float64)
        self.m2 = np.zeros(width, dtype=np.float64)

    def update(self, values: np.ndarray) -> None:
        values = np.asarray(values, dtype=np.float64)
        if values.ndim == 1:
            values = values[:, None]
        batch_count = len(values)
        if batch_count == 0:
            return
        self.minimum = np.minimum(self.minimum, values.min(axis=0))
        self.maximum = np.maximum(self.maximum, values.max(axis=0))
        batch_mean = values.mean(axis=0)
        batch_m2 = ((values - batch_mean) ** 2).sum(axis=0)
        if self.count == 0:
            self.count = batch_count
            self.mean = batch_mean
            self.m2 = batch_m2
            return
        delta = batch_mean - self.mean
        total = self.count + batch_count
        self.m2 += batch_m2 + delta**2 * self.count * batch_count / total
        self.mean += delta * batch_count / total
        self.count = total

    def as_json(self) -> dict[str, Any]:
        std = np.sqrt(self.m2 / self.count) if self.count else np.zeros_like(self.mean)
        return {
            "min": self.minimum.tolist(),
            "max": self.maximum.tolist(),
            "mean": self.mean.tolist(),
            "std": std.tolist(),
            "count": [int(self.count)],
        }


class ImageStats:
    def __init__(self) -> None:
        self.frame_count = 0
        self.pixel_count = 0
        self.minimum = np.full(3, np.inf, dtype=np.float64)
        self.maximum = np.full(3, -np.inf, dtype=np.float64)
        self.sum = np.zeros(3, dtype=np.float64)
        self.sum_sq = np.zeros(3, dtype=np.float64)

    def update(self, frames: np.ndarray) -> None:
        values = frames.astype(np.float64) / 255.0
        flat = values.reshape(-1, 3)
        self.minimum = np.minimum(self.minimum, flat.min(axis=0))
        self.maximum = np.maximum(self.maximum, flat.max(axis=0))
        self.sum += flat.sum(axis=0)
        self.sum_sq += np.square(flat).sum(axis=0)
        self.frame_count += len(frames)
        self.pixel_count += len(flat)

    def as_json(self) -> dict[str, Any]:
        pixel_count = max(self.pixel_count, 1)
        mean = self.sum / pixel_count
        std = np.sqrt(np.maximum(self.sum_sq / pixel_count - mean**2, 0.0))
        wrap = lambda values: [[[float(value)]] for value in values]
        return {
            "min": wrap(self.minimum),
            "max": wrap(self.maximum),
            "mean": wrap(mean),
            "std": wrap(std),
            "count": [int(self.frame_count)],
        }


def _video_writer(path: Path, fps: int):
    try:
        import imageio.v2 as imageio

        writer = imageio.get_writer(
            str(path),
            fps=fps,
            format="ffmpeg",
            codec="libx264",
            macro_block_size=None,
            pixelformat="yuv420p",
        )

        class ImageIOWriter:
            def append(self, frame: np.ndarray) -> None:
                writer.append_data(frame)

            def close(self) -> None:
                writer.close()

        return ImageIOWriter()
    except (ImportError, OSError, RuntimeError):
        pass

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("imageio[ffmpeg] or opencv-python is required for MP4 output") from exc

    first_shape: tuple[int, int] | None = None
    handle = {"writer": None}

    class OpenCVWriter:
        def append(self, frame: np.ndarray) -> None:
            nonlocal first_shape
            height, width = frame.shape[:2]
            if first_shape is None:
                first_shape = (height, width)
                handle["writer"] = cv2.VideoWriter(
                    str(path),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (width, height),
                )
                if not handle["writer"].isOpened():
                    raise RuntimeError(f"OpenCV could not open video writer for {path}")
            if (height, width) != first_shape:
                raise ValueError(f"Camera frame shape changed from {first_shape} to {(height, width)}")
            handle["writer"].write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

        def close(self) -> None:
            if handle["writer"] is not None:
                handle["writer"].release()

    return OpenCVWriter()


def _resize_camera_frames(
    frames: np.ndarray,
    resize: tuple[int, int] | None,
) -> np.ndarray:
    if resize is None:
        return frames
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - dependency guard
        raise RuntimeError("opencv-python is required when --resize is used") from exc
    height, width = resize
    return np.stack(
        [
            cv2.resize(frame, (width, height), interpolation=cv2.INTER_LINEAR)
            for frame in frames
        ]
    )


def _feature_info(fps: int, shape: tuple[int, int, int]) -> dict[str, Any]:
    height, width, channels = shape
    return {
        "dtype": "video",
        "shape": [height, width, channels],
        "names": ["height", "width", "rgb"],
        "info": {
            "video.height": height,
            "video.width": width,
            "video.codec": "avc1",
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": fps,
            "video.channels": channels,
            "has_audio": False,
        },
    }


def _write_table(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)


def _scalar_stats(values: np.ndarray | list[float] | list[int]) -> dict[str, Any]:
    array = np.asarray(values)
    return {
        "min": [float(array.min())],
        "max": [float(array.max())],
        "mean": [float(array.mean())],
        "std": [float(array.std())],
        "count": [int(array.size)],
    }


def convert(
    source: Path,
    output: Path,
    fps: int,
    resize: tuple[int, int] | None,
    task: str,
    task_index: int,
    overwrite: bool,
) -> None:
    if output.exists():
        if not overwrite:
            raise FileExistsError(f"Output exists; pass --overwrite: {output}")
        shutil.rmtree(output)
    output.mkdir(parents=True)

    episodes = [load_episode(path, i) for i, path in enumerate(_episode_files(source))]
    episodes.sort(key=lambda episode: episode.episode_index)
    if not episodes:
        raise ValueError("No episodes loaded")

    first_shape = episodes[0].cameras["cam_high"].shape[1:]
    video_shape = (resize[0], resize[1], first_shape[2]) if resize is not None else first_shape
    for episode in episodes:
        for name, frames in episode.cameras.items():
            if frames.shape[1:] != first_shape:
                raise ValueError(f"{episode.source}: {name} shape {frames.shape[1:]} != {first_shape}")

    data_rows: list[dict[str, Any]] = []
    episode_rows: list[dict[str, Any]] = []
    task_rows = [{"task_index": int(task_index), "task": task}]
    state_stats = RunningStats(14)
    action_stats = RunningStats(14)
    image_stats = {name: ImageStats() for name in CAMERA_FEATURES}
    global_index = 0
    video_frame = 0
    writers = {}
    try:
        for episode in episodes:
            length = episode.length
            data_start = global_index
            video_start = video_frame
            for camera_name in CAMERA_FEATURES:
                video_dir = output / "videos" / CAMERA_FEATURES[camera_name] / "chunk-000"
                video_dir.mkdir(parents=True, exist_ok=True)
                writers[camera_name] = writers.get(camera_name) or _video_writer(
                    video_dir / "file-000.mp4", fps
                )
            camera_frames = {
                camera_name: _resize_camera_frames(episode.cameras[camera_name], resize)
                for camera_name in CAMERA_FEATURES
            }
            state_stats.update(episode.state)
            action_stats.update(episode.action)
            for camera_name, frames in camera_frames.items():
                image_stats[camera_name].update(frames)
            for frame_index in range(length):
                data_rows.append(
                    {
                        "observation.state": episode.state[frame_index].astype(np.float32).tolist(),
                        "action": episode.action[frame_index].astype(np.float32).tolist(),
                        "timestamp": float(frame_index / fps),
                        "frame_index": int(frame_index),
                        "episode_index": int(episode.episode_index),
                        "index": int(global_index),
                        "task_index": int(task_index),
                    }
                )
                for camera_name in CAMERA_FEATURES:
                    writers[camera_name].append(camera_frames[camera_name][frame_index])
                global_index += 1
                video_frame += 1
            episode_row = {
                "episode_index": int(episode.episode_index),
                "tasks": [task],
                "length": int(length),
                "data/chunk_index": 0,
                "data/file_index": 0,
                "dataset_from_index": int(data_start),
                "dataset_to_index": int(global_index),
            }
            for camera_name, feature_name in CAMERA_FEATURES.items():
                episode_row[f"videos/{feature_name}/chunk_index"] = 0
                episode_row[f"videos/{feature_name}/file_index"] = 0
                episode_row[f"videos/{feature_name}/from_timestamp"] = float(video_start / fps)
                episode_row[f"videos/{feature_name}/to_timestamp"] = float(video_frame / fps)
            episode_row.update(
                {
                    "stats/observation.state/min": episode.state.min(axis=0).astype(float).tolist(),
                    "stats/observation.state/max": episode.state.max(axis=0).astype(float).tolist(),
                    "stats/observation.state/mean": episode.state.mean(axis=0).astype(float).tolist(),
                    "stats/observation.state/std": episode.state.std(axis=0).astype(float).tolist(),
                    "stats/observation.state/count": [int(length)],
                    "stats/action/min": episode.action.min(axis=0).astype(float).tolist(),
                    "stats/action/max": episode.action.max(axis=0).astype(float).tolist(),
                    "stats/action/mean": episode.action.mean(axis=0).astype(float).tolist(),
                    "stats/action/std": episode.action.std(axis=0).astype(float).tolist(),
                    "stats/action/count": [int(length)],
                    "stats/timestamp/min": [0.0],
                    "stats/timestamp/max": [float((length - 1) / fps)],
                    "stats/timestamp/mean": [float(np.mean(np.arange(length) / fps))],
                    "stats/timestamp/std": [float(np.std(np.arange(length) / fps))],
                    "stats/timestamp/count": [int(length)],
                    "stats/frame_index/min": [0],
                    "stats/frame_index/max": [int(length - 1)],
                    "stats/frame_index/mean": [float(np.mean(np.arange(length)))],
                    "stats/frame_index/std": [float(np.std(np.arange(length)))],
                    "stats/frame_index/count": [int(length)],
                    "stats/episode_index/min": [int(episode.episode_index)],
                    "stats/episode_index/max": [int(episode.episode_index)],
                    "stats/episode_index/mean": [float(episode.episode_index)],
                    "stats/episode_index/std": [0.0],
                    "stats/episode_index/count": [int(length)],
                    "stats/index/min": [int(data_start)],
                    "stats/index/max": [int(global_index - 1)],
                    "stats/index/mean": [float((data_start + global_index - 1) / 2)],
                    "stats/index/std": [float(np.std(np.arange(data_start, global_index)))],
                    "stats/index/count": [int(length)],
                    "stats/task_index/min": [int(task_index)],
                    "stats/task_index/max": [int(task_index)],
                    "stats/task_index/mean": [float(task_index)],
                    "stats/task_index/std": [0.0],
                    "stats/task_index/count": [int(length)],
                }
            )
            for camera_name, frames in episode.cameras.items():
                episode_image_stats = ImageStats()
                episode_image_stats.update(frames)
                feature_name = CAMERA_FEATURES[camera_name]
                episode_row[f"stats/{feature_name}/min"] = episode_image_stats.as_json()["min"]
                episode_row[f"stats/{feature_name}/max"] = episode_image_stats.as_json()["max"]
                episode_row[f"stats/{feature_name}/mean"] = episode_image_stats.as_json()["mean"]
                episode_row[f"stats/{feature_name}/std"] = episode_image_stats.as_json()["std"]
                episode_row[f"stats/{feature_name}/count"] = episode_image_stats.as_json()["count"]
            episode_rows.append(episode_row)
    finally:
        for writer in writers.values():
            writer.close()

    _write_table(output / "data/chunk-000/file-000.parquet", data_rows)
    _write_table(output / "meta/episodes/chunk-000/file-000.parquet", episode_rows)
    pd.DataFrame(task_rows).set_index("task").to_parquet(output / "meta/tasks.parquet")

    info = {
        "codebase_version": "v3.0",
        "robot_type": "aloha",
        "total_episodes": len(episodes),
        "total_frames": global_index,
        "total_tasks": len(task_rows),
        "total_videos": len(CAMERA_FEATURES),
        "chunks_size": max(1000, len(episodes)),
        "fps": fps,
        "splits": {"train": f"0:{len(episodes)}"},
        "data_path": "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet",
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {
            "observation.state": {
                "dtype": "float32",
                "shape": [14],
                "names": [STATE_NAMES],
            },
            "action": {
                "dtype": "float32",
                "shape": [14],
                "names": [STATE_NAMES],
            },
            **{
                f"observation.images.{camera_name}": _feature_info(fps, video_shape)
                for camera_name in CAMERA_FEATURES
            },
            "timestamp": {"dtype": "float32", "shape": [1], "names": None},
            "frame_index": {"dtype": "int64", "shape": [1], "names": None},
            "episode_index": {"dtype": "int64", "shape": [1], "names": None},
            "index": {"dtype": "int64", "shape": [1], "names": None},
            "task_index": {"dtype": "int64", "shape": [1], "names": None},
        },
        "data_files_size_in_mb": 100,
        "video_files_size_in_mb": 500,
        "video_shard_count": len(CAMERA_FEATURES),
    }
    (output / "meta").mkdir(exist_ok=True)
    (output / "meta/info.json").write_text(json.dumps(info, indent=2) + "\n", encoding="utf-8")
    timestamps = np.concatenate(
        [np.arange(episode.length, dtype=np.float64) / fps for episode in episodes]
    )
    frame_indices = np.concatenate(
        [np.arange(episode.length, dtype=np.int64) for episode in episodes]
    )
    episode_indices = np.concatenate(
        [
            np.full(episode.length, episode.episode_index, dtype=np.int64)
            for episode in episodes
        ]
    )
    indices = np.arange(global_index, dtype=np.int64)
    stats = {
        "observation.state": state_stats.as_json(),
        "action": action_stats.as_json(),
        "timestamp": _scalar_stats(timestamps),
        "frame_index": _scalar_stats(frame_indices),
        "episode_index": _scalar_stats(episode_indices),
        "index": _scalar_stats(indices),
        "task_index": _scalar_stats(np.full(global_index, task_index, dtype=np.int64)),
    }
    stats.update(
        {
            f"observation.images.{camera_name}": image_stats[camera_name].as_json()
            for camera_name in CAMERA_FEATURES
        }
    )
    (output / "meta/stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="HDF5 file or directory of episodes")
    parser.add_argument("--output", type=Path, required=True, help="LeRobot v3 output root")
    parser.add_argument("--fps", type=int, default=15, help="RoboTwin collector frequency; native cameras are 240x320 (H,W) unless --resize is set.")
    parser.add_argument("--resize", nargs=2, type=int, metavar=("H", "W"), default=None, help="Resize camera frames to H W before writing video.")
    parser.add_argument("--task", default="robotwin scripted expert")
    parser.add_argument("--task-index", type=int, default=0)
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if args.fps <= 0:
        parser.error("--fps must be positive")
    if args.resize is not None and any(value <= 0 for value in args.resize):
        parser.error("--resize dimensions must be positive")
    convert(
        source=args.source,
        output=args.output,
        fps=args.fps,
        resize=tuple(args.resize) if args.resize is not None else None,
        task=args.task,
        task_index=args.task_index,
        overwrite=args.overwrite,
    )


if __name__ == "__main__":
    main()
