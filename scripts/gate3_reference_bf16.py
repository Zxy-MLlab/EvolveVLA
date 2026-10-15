#!/usr/bin/env python3
"""Paired fp32/bfloat16 OpenPI reference forward on the aligned RoboTwin row."""

from __future__ import annotations

import importlib.util
import os
import sys
import traceback
from pathlib import Path

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


ROOT = Path("/root/autodl-tmp/EvolveVLA")
PINNED = ROOT / "scripts/gate3_reference_forward.py"
OPENPI = Path("/root/autodl-tmp/openpi_feasibility_src/src")
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer/tokenizer.model")
ALIGNED_VIDEO_INDEX = 7219
EPISODE = 50
POSITION = 31
ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def load_pinned_module():
    spec = importlib.util.spec_from_file_location(
        "gate3_reference_forward", PINNED
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load pinned module from {PINNED}")
    module = importlib.util.module_from_spec(spec)
    sys.modules["gate3_reference_forward"] = module
    spec.loader.exec_module(module)
    return module


def verify_aligned_observation(module) -> None:
    import pyarrow.parquet as pq

    data_path = module.DATA / "data/chunk-000/file-000.parquet"
    table = pq.read_table(
        data_path,
        columns=[
            "observation.state",
            "episode_index",
            "index",
            "frame_index",
        ],
    ).to_pylist()
    rows = [
        row
        for row in table
        if int(row["episode_index"]) == EPISODE
    ]
    rows.sort(key=lambda row: int(row["frame_index"]))
    if len(rows) <= POSITION:
        raise RuntimeError(
            f"episode {EPISODE} has only {len(rows)} rows; "
            f"cannot select position {POSITION}"
        )
    row = rows[POSITION]
    if int(row["frame_index"]) != POSITION:
        raise RuntimeError(
            f"unexpected frame_index={row['frame_index']} expected={POSITION}"
        )
    if int(row["index"]) != ALIGNED_VIDEO_INDEX:
        raise RuntimeError(
            f"unexpected video index={row['index']} "
            f"expected={ALIGNED_VIDEO_INDEX}"
        )
    state = np.asarray(row["observation.state"], dtype=np.float32)
    if not np.array_equal(state, module.STATE):
        raise RuntimeError("aligned parquet state differs from pinned STATE")
    print(f"ALIGNED_VIDEO_INDEX={ALIGNED_VIDEO_INDEX}", flush=True)
    print(f"ALIGNED_EPISODE={EPISODE} ALIGNED_POSITION={POSITION}", flush=True)


def make_tokenizer():
    import sentencepiece
    from openpi.models.tokenizer import PaligemmaTokenizer

    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = 200
    with TOKENIZER_PATH.open("rb") as handle:
        tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=handle.read()
        )
    return tokenizer


def make_observation(module, tokenizer, device, dtype):
    import torch
    from openpi.models.model import Observation
    from openpi.shared.image_tools import resize_with_pad_torch

    normalized_state = module.normalize_state(module.STATE, module.stats())
    tokens, token_mask = tokenizer.tokenize(module.TASK, normalized_state)
    image_paths = {
        "base_0_rgb": module.DATA
        / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
        "left_wrist_0_rgb": module.DATA
        / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
        "right_wrist_0_rgb": module.DATA
        / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
    }
    images = {}
    for key, path in image_paths.items():
        frame = module.frame(path, POSITION)
        image = torch.from_numpy(frame).float() / 255.0 * 2.0 - 1.0
        image = resize_with_pad_torch(image.unsqueeze(0), 224, 224)
        image = image.unsqueeze(0).permute(0, 3, 1, 2)
        images[key] = image.to(device=device, dtype=dtype)

    return Observation.from_dict(
        {
            "image": images,
            "image_mask": {
                key: torch.ones(1, dtype=torch.bool, device=device)
                for key in images
            },
            "state": torch.from_numpy(normalized_state)
            .view(1, -1)
            .to(device=device, dtype=dtype),
            "tokenized_prompt": torch.from_numpy(tokens)
            .view(1, -1)
            .to(device=device, dtype=torch.long),
            "tokenized_prompt_mask": torch.from_numpy(token_mask)
            .view(1, -1)
            .to(device=device, dtype=torch.bool),
        }
    )


def run_bf16(module, model_config_dtype: str = "bfloat16") -> float:
    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    device = torch.device("cpu")
    dtype = torch.bfloat16
    model_dtype = torch.bfloat16 if model_config_dtype == "bfloat16" else torch.float32
    model = PI0Pytorch(
        Pi0Config(
            pi05=True,
            dtype=model_config_dtype,
            pytorch_compile_mode=None,
        )
    )
    loaded_keys = module.load_weights(model)
    model = model.to(device=device, dtype=model_dtype).eval()

    # OpenPI's default sample_noise is float32 even for a bfloat16 config.
    model.sample_noise = lambda shape, target_device: torch.normal(
        mean=0.0,
        std=1.0,
        size=shape,
        dtype=dtype,
        device=target_device,
    )

    tokenizer = make_tokenizer()
    obs = make_observation(module, tokenizer, device, dtype)
    input_dtypes = sorted(
        {str(tensor.dtype) for tensor in obs.images.values()}
        | {str(obs.state.dtype)}
    )
    print(
        f"BF16_MODEL_READY model_config_dtype={model_config_dtype} "
        f"loaded_keys={loaded_keys} "
        f"input_dtypes={','.join(input_dtypes)}",
        flush=True,
    )
    torch.manual_seed(0)
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        actions = model.sample_actions(device, obs, noise=None, num_steps=10)
    stats = module.stats()
    q01 = np.asarray(stats["actions"]["q01"], dtype=np.float32)
    q99 = np.asarray(stats["actions"]["q99"], dtype=np.float32)
    action = actions[0, 0, :14].detach().float().cpu().numpy()
    action = (action + 1.0) * (q99 - q01) / 2.0 + q01
    return float(np.linalg.norm(action[ARM_DIMS]))


def main() -> int:
    module = load_pinned_module()
    if str(OPENPI) not in sys.path:
        sys.path.insert(0, str(OPENPI))

    print(f"OPENPI_PATH_OK path={OPENPI}", flush=True)
    verify_aligned_observation(module)

    import torch

    fp32, _ = module.openpi_forward(torch.device("cpu"), module.stats())
    print(f"DTYPE=float32 REFERENCE_ARM_L2={fp32:.8f}", flush=True)
    try:
        bf16 = run_bf16(module, model_config_dtype="bfloat16")
    except Exception as exc:
        traceback.print_exc()
        print(
            f"BF16_FAILED={type(exc).__name__}: {exc}",
            flush=True,
        )
        return 1

    print(f"DTYPE=bfloat16 REFERENCE_ARM_L2={bf16:.8f}", flush=True)
    print(
        f"BF16_DELTA_FROM_FP32={bf16 - fp32:.8f} "
        f"BF16_OFF_MEAN_VERDICT={'YES' if bf16 >= 1.0 else 'NO'}",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
