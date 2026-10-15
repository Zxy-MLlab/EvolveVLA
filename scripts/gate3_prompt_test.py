#!/usr/bin/env python3
"""Test whether the task prompt moves the official PI05 reference output."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path("/root/autodl-tmp/EvolveVLA")
PINNED_PATH = ROOT / "scripts/gate3_reference_forward.py"
DATA_ROOT = ROOT / "data/gate2_robotwin_cache/clean"
EXPECTED_VIDEO_INDEX = 7219
EPISODE = 50
POSITION = 31
PROMPTS = [
    "Pick the hammer for nails and strike the block.",
    "Utilize the left arm to hold the grippy handle hammer and strike.",
    "With the left arm, grab the nail-driving hammer and beat the block.",
    "beat_block_hammer",
    "With the left arm, grab the block and beat the block.",
]


def load_pinned_module():
    spec = importlib.util.spec_from_file_location("m", str(PINNED_PATH))
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned module from {PINNED_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["m"] = module
    spec.loader.exec_module(module)
    return module


def load_aligned_row() -> dict:
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
    if int(row["frame_index"]) != POSITION:
        raise RuntimeError(
            f"unexpected aligned frame position={row['frame_index']} "
            f"expected={POSITION}"
        )
    if int(row["index"]) != EXPECTED_VIDEO_INDEX:
        raise RuntimeError(
            f"unexpected aligned video index={row['index']} "
            f"expected={EXPECTED_VIDEO_INDEX}"
        )
    return row


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    return value


def main() -> int:
    module = load_pinned_module()
    row = load_aligned_row()
    parquet_line = {
        key: json_value(value)
        for key, value in row.items()
    }
    parquet_state = np.asarray(row["observation.state"], dtype=np.float32).reshape(14)
    if not np.array_equal(parquet_state, module.STATE):
        raise RuntimeError("parquet aligned state differs from pinned STATE")

    print(f"ALIGNED_VIDEO_INDEX={int(row['index'])}")
    print(f"PARQUET_LINE={json.dumps(parquet_line, separators=(',', ':'))}")

    stats = module.stats()
    results = []
    loaded_keys = None
    for prompt in PROMPTS:
        module.TASK = prompt
        arm_l2, loaded_keys = module.openpi_forward(
            torch.device("cpu"),
            stats,
        )
        results.append((prompt, arm_l2))

    print(
        "OPENPI_PATH_OK "
        f"model=PI0Pytorch loaded_keys={loaded_keys} "
        "preprocessing=openpi.resize_with_pad_torch,"
        "openpi.PaligemmaTokenizer,openpi.Observation.from_dict"
    )
    for prompt, arm_l2 in results:
        print(f"PROMPT={prompt} REFERENCE_ARM_L2={arm_l2:.8f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
