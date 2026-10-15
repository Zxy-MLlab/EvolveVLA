"""Bridge RoboTwin LeRobot observations to the installed LeRobot wrapper.

Inputs are a flat observation mapping with dataset camera keys and HWC RGB
NumPy frames, normally:
    observation.images.cam_high
    observation.images.cam_left_wrist
    observation.images.cam_right_wrist

Outputs use the keys consumed by LeRobot's RoboTwin preprocessing path:
    observation.images.head_camera
    observation.images.left_camera
    observation.images.right_camera
with camera frames shaped H x W x C, specifically 240 x 320 x 3 by default.

Dependencies: Python standard library and NumPy only. No simulator, Gymnasium,
video decoder, or full dataset is needed.

Verified source of the target contract (inspected 2026-10-01):
  /root/autodl-tmp/conda_envs/evolvevla_pi312/lib/python3.12/site-packages/
  lerobot/envs/robotwin.py:47-51      raw camera names
  lerobot/envs/robotwin.py:172-173    default height=240, width=320
  lerobot/envs/robotwin.py:335         conversion to observation.images.<cam>
  lerobot/envs/robotwin.py:388,395     image shape=(height, width, 3)

FPS handling:
  The dataset is 50 FPS while the wrapper contract is 25 FPS. This module
  handles spatial key and frame-shape conversion only. Temporal decimation
  belongs at the sequence/window or data-loader boundary, before a temporal
  window is passed to the policy or rollout code. For the 50 -> 25 bridge,
  select every second frame; use decimate_frame_sequence() for that operation.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


CAMERA_KEY_RENAME: dict[str, str] = {
    "observation.images.cam_high": "observation.images.head_camera",
    "observation.images.cam_left_wrist": "observation.images.left_camera",
    "observation.images.cam_right_wrist": "observation.images.right_camera",
}

DEFAULT_OUTPUT_HEIGHT = 240
DEFAULT_OUTPUT_WIDTH = 320


def remap_camera_keys(observation: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy with dataset camera keys renamed for RoboTwin."""
    remapped = dict(observation)
    for source_key, target_key in CAMERA_KEY_RENAME.items():
        if source_key not in remapped:
            continue
        if target_key in remapped:
            raise ValueError(
                f"Observation contains both source key {source_key!r} "
                f"and target key {target_key!r}."
            )
        remapped[target_key] = remapped.pop(source_key)
    return remapped


def resize_camera_frame(
    frame: Any,
    *,
    output_height: int = DEFAULT_OUTPUT_HEIGHT,
    output_width: int = DEFAULT_OUTPUT_WIDTH,
) -> np.ndarray:
    """Resize one RGB frame from HWC input to the requested HWC output.

    The bridge uses deterministic nearest-neighbor sampling and preserves the
    input dtype. D435 sim cameras natively output [240, 320, 3], so the default
    [240, 320, 3] target is an IDENTITY (not a [480, 640, 3] downsample).
    The dimensions are H x W, not W x H.
    """
    array = np.asarray(frame)
    if array.ndim != 3 or array.shape[-1] != 3:
        raise ValueError(f"Expected an HWC RGB frame, got shape {array.shape}.")
    if output_height <= 0 or output_width <= 0:
        raise ValueError(
            f"Output dimensions must be positive, got "
            f"{output_height} x {output_width}."
        )

    source_height, source_width, _ = array.shape
    row_indices = (np.arange(output_height) * source_height) // output_height
    col_indices = (np.arange(output_width) * source_width) // output_width
    return array[row_indices[:, None], col_indices[None, :], :]


def bridge_observation(
    observation: Mapping[str, Any],
    *,
    output_height: int = DEFAULT_OUTPUT_HEIGHT,
    output_width: int = DEFAULT_OUTPUT_WIDTH,
) -> dict[str, Any]:
    """Rename dataset camera keys and resize all mapped camera frames."""
    bridged = remap_camera_keys(observation)
    for target_key in CAMERA_KEY_RENAME.values():
        if target_key in bridged:
            bridged[target_key] = resize_camera_frame(
                bridged[target_key],
                output_height=output_height,
                output_width=output_width,
            )
    return bridged


def decimate_frame_sequence(
    frames: Any,
    *,
    source_fps: int = 50,
    target_fps: int = 25,
) -> np.ndarray:
    """Decimate a time-major frame sequence at the sequence boundary.

    For the verified 50 -> 25 FPS bridge, this returns frames[::2]. It does
    not resize frames and should be applied before constructing temporal
    windows, not independently inside a single-observation transform.
    """
    if source_fps <= 0 or target_fps <= 0:
        raise ValueError(
            f"FPS values must be positive, got {source_fps} -> {target_fps}."
        )
    if source_fps % target_fps != 0:
        raise ValueError(
            f"FPS decimation must be an integer ratio, got "
            f"{source_fps} -> {target_fps}."
        )
    ratio = source_fps // target_fps
    return np.asarray(frames)[::ratio]


if __name__ == "__main__":
    synthetic_frame = np.arange(480 * 640 * 3, dtype=np.uint8).reshape(480, 640, 3)
    synthetic_observation = {
        "observation.images.cam_high": synthetic_frame,
        "observation.images.cam_left_wrist": synthetic_frame.copy(),
        "observation.images.cam_right_wrist": synthetic_frame.copy(),
        "observation.state": np.zeros(14, dtype=np.float32),
    }

    bridged = bridge_observation(synthetic_observation)
    expected_keys = {
        "observation.images.head_camera",
        "observation.images.left_camera",
        "observation.images.right_camera",
        "observation.state",
    }
    assert set(bridged) == expected_keys
    assert bridged["observation.images.head_camera"].shape == (240, 320, 3)
    assert bridged["observation.images.head_camera"].dtype == np.uint8
    assert np.array_equal(
        bridged["observation.images.head_camera"][1, 1],
        synthetic_frame[2, 2],
    )

    sequence = np.arange(4 * 2 * 2 * 3, dtype=np.uint8).reshape(4, 2, 2, 3)
    decimated = decimate_frame_sequence(sequence)
    assert decimated.shape == (2, 2, 2, 3)
    assert np.array_equal(decimated, sequence[[0, 2]])
    print("SELF-TEST PASS: camera rename, HWC resize, and 50->25 FPS decimation")
