#!/usr/bin/env python3
"""Run the SidneyXie checkpoint through the pinned OpenPI reference path."""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import traceback
from pathlib import Path
from typing import Any

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np


ROOT = Path("/root/autodl-tmp/EvolveVLA")
PINNED = ROOT / "scripts/gate3_reference_forward.py"
OPENPI = Path("/root/autodl-tmp/openpi_feasibility_src/src")
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer/tokenizer.model")
SIDNEY = ROOT / "checkpoints/sidneyxie_pi05_robotwin"
OFFICIAL = ROOT / "checkpoints/pi05_robotwin_lerobot"
ALIGNED_VIDEO_INDEX = 7219
EPISODE = 50
POSITION = 31
PROMPT = "With the left arm, grab the nail-driving hammer and beat the block."
ARM_DIMS = [0, 1, 2, 3, 4, 5, 7, 8, 9, 10, 11, 12]


def load_pinned_module() -> Any:
    spec = importlib.util.spec_from_file_location(
        "gate3_reference_forward_pinned",
        str(PINNED),
    )
    if spec is None or spec.loader is None:
        raise RuntimeError(f"unable to load pinned module from {PINNED}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def load_aligned_state(module: Any) -> np.ndarray:
    import pyarrow.parquet as pq

    table = pq.read_table(
        module.DATA / "data/chunk-000/file-000.parquet",
        columns=["observation.state", "episode_index", "index", "frame_index"],
    ).to_pydict()
    matches = [
        i
        for i, index in enumerate(table["index"])
        if int(index) == ALIGNED_VIDEO_INDEX
    ]
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one parquet row for index={ALIGNED_VIDEO_INDEX}, "
            f"got {len(matches)}"
        )
    row = matches[0]
    episode = int(table["episode_index"][row])
    position = int(table["frame_index"][row])
    if episode != EPISODE or position != POSITION:
        raise RuntimeError(
            f"aligned row mismatch episode={episode} frame_index={position}; "
            f"expected episode={EPISODE} frame_index={POSITION}"
        )
    state = np.asarray(table["observation.state"][row], dtype=np.float32)
    if state.shape != (14,):
        raise RuntimeError(f"aligned state shape={state.shape}, expected (14,)")
    if not np.array_equal(state, np.asarray(module.STATE, dtype=np.float32)):
        raise RuntimeError("aligned parquet state differs from pinned STATE")
    return state


def load_sidney_stats() -> dict[str, dict[str, np.ndarray]]:
    from safetensors.torch import load_file

    stats_path = (
        SIDNEY / "policy_preprocessor_step_3_normalizer_processor.safetensors"
    )
    tensors = load_file(str(stats_path), device="cpu")
    expected = {
        "observation.state.mean",
        "observation.state.std",
        "action.mean",
        "action.std",
    }
    missing = sorted(expected - set(tensors))
    if missing:
        raise RuntimeError(
            f"sidneyxie MEAN_STD stats missing keys={missing}; "
            f"available={sorted(tensors)}"
        )
    result = {
        "state": {
            "mean": tensors["observation.state.mean"].float().numpy(),
            "std": tensors["observation.state.std"].float().numpy(),
        },
        "action": {
            "mean": tensors["action.mean"].float().numpy(),
            "std": tensors["action.std"].float().numpy(),
        },
    }
    for feature, fields in result.items():
        for field, value in fields.items():
            if value.shape != (14,):
                raise RuntimeError(
                    f"sidneyxie {feature}.{field} shape={value.shape}, "
                    "expected (14,)"
                )
        if np.any(fields["std"] <= 0):
            raise RuntimeError(f"sidneyxie {feature}.std contains non-positive values")
    return result


def load_weights_with_audit(model: Any, checkpoint: Path) -> dict[str, Any]:
    from safetensors.torch import load_file

    path = checkpoint / "model.safetensors"
    checkpoint_state = load_file(str(path), device="cpu")
    model_keys = set(model.state_dict())
    checkpoint_keys = set(checkpoint_state)
    missing_before_load = sorted(model_keys - checkpoint_keys)
    unexpected_before_load = sorted(checkpoint_keys - model_keys)
    try:
        report = model.load_state_dict(checkpoint_state, strict=False)
    except Exception as exc:
        raise RuntimeError(
            f"key_mismatch checkpoint={checkpoint.name}: "
            f"{type(exc).__name__}: {exc}"
        ) from exc

    missing = sorted(report.missing_keys)
    unexpected = sorted(report.unexpected_keys)
    allowed_missing = {
        "paligemma_with_expert.gemma_expert.lm_head.weight",
        "paligemma_with_expert.paligemma.lm_head.weight",
    }
    if not set(missing).issubset(allowed_missing) or unexpected:
        raise RuntimeError(
            f"key_mismatch checkpoint={checkpoint.name}: "
            f"missing={missing} unexpected={unexpected}"
        )
    return {
        "checkpoint_keys": len(checkpoint_state),
        "model_keys": len(model_keys),
        "missing": missing,
        "unexpected": unexpected,
        "missing_before_load": missing_before_load,
        "unexpected_before_load": unexpected_before_load,
        "loaded_keys": len(checkpoint_state) - len(unexpected),
    }


def normalize_sidney_state(state: np.ndarray, stats: dict[str, Any]) -> np.ndarray:
    return (
        (state - stats["state"]["mean"]) / stats["state"]["std"]
    ).astype(np.float32)


def normalize_official_state(module: Any, state: np.ndarray) -> np.ndarray:
    return module.normalize_state(state, module.stats()).astype(np.float32)


def unnormalize_sidney_action(
    normalized_action: np.ndarray,
    stats: dict[str, Any],
) -> np.ndarray:
    return (
        normalized_action * stats["action"]["std"] + stats["action"]["mean"]
    ).astype(np.float32)


def make_tokenizer() -> Any:
    import sentencepiece
    from openpi.models.tokenizer import PaligemmaTokenizer

    tokenizer = PaligemmaTokenizer.__new__(PaligemmaTokenizer)
    tokenizer._max_len = 200
    with TOKENIZER_PATH.open("rb") as handle:
        tokenizer._tokenizer = sentencepiece.SentencePieceProcessor(
            model_proto=handle.read()
        )
    return tokenizer


def make_images(module: Any, device: Any) -> dict[str, Any]:
    import torch
    from openpi.shared.image_tools import resize_with_pad_torch

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
        frame = module.frame(path, ALIGNED_VIDEO_INDEX)
        if frame.shape != (480, 640, 3):
            raise RuntimeError(f"unexpected frame shape for {key}: {frame.shape}")
        image = torch.from_numpy(frame).float() / 255.0 * 2.0 - 1.0
        image = resize_with_pad_torch(image.unsqueeze(0), 224, 224)
        images[key] = image.unsqueeze(0).permute(0, 3, 1, 2).to(device)
    return images


def run_model(
    module: Any,
    checkpoint: Path,
    normalized_state: np.ndarray,
    images: dict[str, Any],
    tokenizer: Any,
    device: Any,
    action_postprocess: Any,
) -> dict[str, Any]:
    import torch
    from openpi.models.model import Observation
    from openpi.models.pi0_config import Pi0Config
    from openpi.models_pytorch.pi0_pytorch import PI0Pytorch

    model = PI0Pytorch(Pi0Config(pi05=True, dtype="float32"))
    audit = load_weights_with_audit(model, checkpoint)
    model = model.to(device).eval()
    tokens, token_mask = tokenizer.tokenize(PROMPT, normalized_state)
    observation = Observation.from_dict(
        {
            "image": images,
            "image_mask": {
                key: torch.ones(1, dtype=torch.bool, device=device)
                for key in images
            },
            "state": torch.from_numpy(normalized_state)
            .view(1, -1)
            .to(device),
            "tokenized_prompt": torch.from_numpy(tokens)
            .view(1, -1)
            .to(device=device, dtype=torch.long),
            "tokenized_prompt_mask": torch.from_numpy(token_mask)
            .view(1, -1)
            .to(device=device, dtype=torch.bool),
        }
    )
    torch.manual_seed(0)
    actions = model.sample_actions(device, observation, noise=None, num_steps=10)
    raw_action = actions[0, 0, :14].detach().float().cpu().numpy()
    postprocessed_action = action_postprocess(raw_action)
    return {
        "audit": audit,
        "raw_arm_l2": float(np.linalg.norm(raw_action[ARM_DIMS])),
        "arm_l2": float(np.linalg.norm(postprocessed_action[ARM_DIMS])),
        "raw_action": raw_action,
    }


def main() -> int:
    try:
        if str(OPENPI) not in sys.path:
            sys.path.insert(0, str(OPENPI))
        import torch

        if torch.cuda.is_available():
            raise RuntimeError("CPU-only probe unexpectedly sees CUDA")
        module = load_pinned_module()
        state = load_aligned_state(module)
        sidney_stats = load_sidney_stats()
        official_stats = module.stats()
        device = torch.device("cpu")
        tokenizer = make_tokenizer()
        images = make_images(module, device)

        official_state = normalize_official_state(module, state)
        sidney_state = normalize_sidney_state(state, sidney_stats)
        official = run_model(
            module,
            OFFICIAL,
            official_state,
            images,
            tokenizer,
            device,
            lambda raw: (
                (raw + 1.0)
                * (
                    np.asarray(official_stats["actions"]["q99"], dtype=np.float32)
                    - np.asarray(official_stats["actions"]["q01"], dtype=np.float32)
                )
                / 2.0
                + np.asarray(official_stats["actions"]["q01"], dtype=np.float32)
            ),
        )
        sidney = run_model(
            module,
            SIDNEY,
            sidney_state,
            images,
            tokenizer,
            device,
            lambda raw: unnormalize_sidney_action(raw, sidney_stats),
        )

        print(f"OPENPI_PATH_OK path={OPENPI}", flush=True)
        print(f"ALIGNED_VIDEO_INDEX={ALIGNED_VIDEO_INDEX}", flush=True)
        print(f"PROMPT={PROMPT}", flush=True)
        print(
            "MODEL=official raw_norm=QUANTILES "
            f"RAW_NORMALIZED_ARM_L2={official['raw_arm_l2']:.8f} "
            f"REFERENCE_ARM_L2={official['arm_l2']:.8f}",
            flush=True,
        )
        print(
            "MODEL=sidneyxie raw_norm=MEAN_STD "
            f"RAW_NORMALIZED_ARM_L2={sidney['raw_arm_l2']:.8f} "
            f"REFERENCE_ARM_L2={sidney['arm_l2']:.8f}",
            flush=True,
        )
        print(
            "OFFICIAL_PAIRED_EXPECTED_REFERENCE_ARM_L2_APPROX=0.0121",
            flush=True,
        )
        for label, result in (("official", official), ("sidneyxie", sidney)):
            audit = result["audit"]
            print(
                f"KEY_AUDIT={label} checkpoint_keys={audit['checkpoint_keys']} "
                f"model_keys={audit['model_keys']} loaded_keys={audit['loaded_keys']} "
                f"missing={json.dumps(audit['missing'], separators=(',', ':'))} "
                f"unexpected={json.dumps(audit['unexpected'], separators=(',', ':'))}",
                flush=True,
            )
        print(
            "VERDICT="
            + (
                "REFERENCE_PATH_SANE_OFFICIAL_UNDER_PRODUCES"
                if sidney["arm_l2"] >= 1.0
                else "REFERENCE_PATH_BUG_CANDIDATE_SIDNEYXIE_ALSO_NEAR_MEAN"
            ),
            flush=True,
        )
        return 0
    except Exception as exc:
        traceback.print_exc()
        print(f"REFERENCE_BLOCKER={type(exc).__name__}: {exc}", flush=True)
        return 0


if __name__ == "__main__":
    raise SystemExit(main())
