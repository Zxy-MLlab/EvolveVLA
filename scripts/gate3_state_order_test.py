#!/usr/bin/env python3
"""CPU-only contrastive state/image sensitivity test for the pinned PI05 path."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["JAX_PLATFORMS"] = "cpu"

import numpy as np

ROOT = Path("/root/autodl-tmp/EvolveVLA")
DATA = ROOT / "data/gate2_robotwin_cache/clean"
OPENPI_PATH = Path("/root/autodl-tmp/openpi_feasibility_src/src")
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
PINNED_PATH = ROOT / "scripts/gate3_reference_forward.py"
RESULTS_PATH = ROOT / "results/gate3_state_order_test.json"

ALIGNED_VIDEO_INDEX = 7219
EPISODE = 50
POSITION = 31
OFFICIAL_PROMPT = "With the left arm, grab the nail-driving hammer and beat the block."
ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]
STATE_PERMUTATION = np.asarray(
    [7, 8, 9, 10, 11, 12, 13, 0, 1, 2, 3, 4, 5, 6],
    dtype=np.int64,
)

IMAGE_PATHS = {
    "base_0_rgb": DATA / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
    "left_wrist_0_rgb": DATA
    / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
    "right_wrist_0_rgb": DATA
    / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
}


def load_reference_module() -> Any:
    if str(OPENPI_PATH) not in sys.path:
        sys.path.insert(0, str(OPENPI_PATH))
    spec = importlib.util.spec_from_file_location(
        "gate3_reference_forward_pinned",
        str(PINNED_PATH),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import pinned module from {PINNED_PATH}")
    reference = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = reference
    spec.loader.exec_module(reference)
    return reference


def load_aligned_state() -> np.ndarray:
    import pyarrow.parquet as pq

    data_path = DATA / "data/chunk-000/file-000.parquet"
    table = pq.read_table(
        data_path,
        columns=["observation.state", "episode_index", "index", "frame_index"],
    ).to_pydict()
    matches = [
        i
        for i, index in enumerate(table["index"])
        if int(index) == ALIGNED_VIDEO_INDEX
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one parquet row for index={ALIGNED_VIDEO_INDEX}, got {len(matches)}"
        )
    row_index = matches[0]
    episode = int(table["episode_index"][row_index])
    frame_index = int(table["frame_index"][row_index])
    if episode != EPISODE or frame_index != POSITION:
        raise RuntimeError(
            f"aligned row mismatch episode={episode} frame_index={frame_index}; "
            f"expected episode={EPISODE} frame_index={POSITION}"
        )
    state = np.asarray(
        table["observation.state"][row_index],
        dtype=np.float32,
    ).reshape(14)
    return state


def load_model(reference: Any) -> tuple[Any, Any, dict[str, Any], Any]:
    import sentencepiece
    import torch
    from openpi.models.pi0_config import Pi0Config
    from openpi.models.tokenizer import PaligemmaTokenizer
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    device = torch.device("cpu")
    model = PI0Pytorch(Pi0Config(pi05=True, dtype="float32"))
    loaded_keys = reference.load_weights(model)
    model = model.to(device).eval()

    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = 200
    with open(TOKENIZER_PATH / "tokenizer.model", "rb") as handle:
        tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=handle.read()
        )

    print(f"REFERENCE_MODEL_READY loaded_keys={loaded_keys}", flush=True)
    return model, tokenizer, reference.stats(), device


def load_true_images(reference: Any, device: Any) -> dict[str, Any]:
    import torch
    from openpi.shared.image_tools import resize_with_pad_torch

    images: dict[str, Any] = {}
    for key, path in IMAGE_PATHS.items():
        frame = reference.frame(path, ALIGNED_VIDEO_INDEX)
        if frame.shape != (480, 640, 3):
            raise RuntimeError(f"unexpected frame shape for {key}: {frame.shape}")
        image = torch.from_numpy(frame).float() / 255.0 * 2.0 - 1.0
        image = resize_with_pad_torch(image.unsqueeze(0), 224, 224)
        images[key] = image.unsqueeze(0).permute(0, 3, 1, 2).to(device)
    return images


def make_state_variants(true_state: np.ndarray) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {
        "TRUE_DATASET": true_state.copy(),
        "ZERO": np.zeros(14, dtype=np.float32),
        "RANDOM": rng.uniform(-1.0, 1.0, size=14).astype(np.float32),
        "SHUFFLED": true_state[STATE_PERMUTATION].copy(),
    }


def make_image_variants(true_images: dict[str, Any], device: Any) -> dict[str, dict[str, Any]]:
    import torch

    random_rng = np.random.default_rng(0)
    black_images = {
        key: torch.zeros_like(image, device=device)
        for key, image in true_images.items()
    }
    random_images = {
        key: torch.from_numpy(
            random_rng.uniform(-1.0, 1.0, size=tuple(image.shape)).astype(np.float32)
        ).to(device)
        for key, image in true_images.items()
    }
    return {
        "TRUE_FRAMES": true_images,
        "BLACK": black_images,
        "RANDOM": random_images,
    }


def forward_variant(
    reference: Any,
    model: Any,
    tokenizer: Any,
    stats: dict[str, Any],
    device: Any,
    images: dict[str, Any],
    state: np.ndarray,
) -> float:
    import torch
    from openpi.models.model import Observation

    normalized_state = reference.normalize_state(state, stats)
    tokens, token_mask = tokenizer.tokenize(OFFICIAL_PROMPT, normalized_state)
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
    q01 = np.asarray(stats["actions"]["q01"], dtype=np.float32)
    q99 = np.asarray(stats["actions"]["q99"], dtype=np.float32)
    action = actions[0, 0, :14].detach().float().cpu().numpy()
    action = (action + 1.0) * (q99 - q01) / 2.0 + q01
    return float(np.linalg.norm(action[ARM_DIMS]))


def as_json_vector(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.asarray(values).reshape(-1)]


def write_results(results: dict[str, Any]) -> None:
    RESULTS_PATH.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = RESULTS_PATH.with_suffix(".json.tmp")
    temporary_path.write_text(json.dumps(results, indent=2) + "\n")
    temporary_path.replace(RESULTS_PATH)


def main() -> int:
    reference = load_reference_module()
    dataset_state = load_aligned_state()
    pinned_state = np.asarray(reference.STATE, dtype=np.float32).reshape(14)
    harness_state = dataset_state.copy()

    print(f"OPENPI_PATH_OK path={OPENPI_PATH}", flush=True)
    print(f"ALIGNED_VIDEO_INDEX={ALIGNED_VIDEO_INDEX}", flush=True)
    print(f"EPISODE={EPISODE} POSITION={POSITION}", flush=True)
    print(f"OFFICIAL_PROMPT={OFFICIAL_PROMPT}", flush=True)
    print(f"STATE_PERMUTATION={STATE_PERMUTATION.tolist()}", flush=True)
    print(f"DATASET_state={json.dumps(as_json_vector(dataset_state))}", flush=True)
    print(f"OURS_state={json.dumps(as_json_vector(pinned_state))}", flush=True)
    print(
        "OURS_state_PINNED_STATE="
        + json.dumps(as_json_vector(pinned_state)),
        flush=True,
    )
    print(
        "OURS_state_HARNESS_make_policy_batch="
        + json.dumps(as_json_vector(harness_state)),
        flush=True,
    )
    print(
        "STATE_DIFF_DATASET_MINUS_OURS="
        + json.dumps(as_json_vector(dataset_state - pinned_state)),
        flush=True,
    )
    print(
        "STATE_DIFF_DATASET_MINUS_HARNESS="
        + json.dumps(as_json_vector(dataset_state - harness_state)),
        flush=True,
    )

    model, tokenizer, stats, device = load_model(reference)
    true_images = load_true_images(reference, device)
    state_variants = make_state_variants(dataset_state)
    image_variants = make_image_variants(true_images, device)

    results: dict[str, Any] = {
        "aligned_video_index": ALIGNED_VIDEO_INDEX,
        "episode": EPISODE,
        "position": POSITION,
        "official_prompt": OFFICIAL_PROMPT,
        "openpi_path": str(OPENPI_PATH),
        "dataset_state": as_json_vector(dataset_state),
        "ours_state": as_json_vector(pinned_state),
        "harness_state": as_json_vector(harness_state),
        "state_diff_dataset_minus_ours": as_json_vector(
            dataset_state - pinned_state
        ),
        "state_diff_dataset_minus_harness": as_json_vector(
            dataset_state - harness_state
        ),
        "state_permutation": STATE_PERMUTATION.tolist(),
        "state_results": {},
        "image_results": {},
    }
    write_results(results)

    for name, state in state_variants.items():
        arm_l2 = forward_variant(
            reference,
            model,
            tokenizer,
            stats,
            device,
            true_images,
            state,
        )
        results["state_results"][name] = {
            "reference_arm_l2": arm_l2,
            "state": as_json_vector(state),
        }
        write_results(results)
        print(f"STATE_VARIANT={name} REFERENCE_ARM_L2={arm_l2:.8f}", flush=True)

    for name, images in image_variants.items():
        if name == "TRUE_FRAMES":
            print(
                f"OPENPI_PATH_OK variant=TRUE_FRAMES path={OPENPI_PATH}",
                flush=True,
            )
        arm_l2 = forward_variant(
            reference,
            model,
            tokenizer,
            stats,
            device,
            images,
            dataset_state,
        )
        results["image_results"][name] = {
            "reference_arm_l2": arm_l2,
            "image_shape": {
                key: list(image.shape) for key, image in images.items()
            },
        }
        write_results(results)
        print(f"IMAGE_VARIANT={name} REFERENCE_ARM_L2={arm_l2:.8f}", flush=True)

    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        traceback.print_exc()
        print(f"REFERENCE_BLOCKER={type(exc).__name__}: {exc}", flush=True)
        raise
