#!/usr/bin/env python3
"""Bounded PI05 prefix/KV conditioning probe.

This script does not roll out the simulator. It loads one recorded training
observation through the production harness, inspects the exact prefix passed to
the denoiser, and evaluates one t=1.0 velocity under three prefix conditions.
"""

from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import sys
from argparse import Namespace
from pathlib import Path

import av
import numpy as np
import pyarrow.parquet as pq
import torch


ROOT = Path("/root/autodl-tmp/EvolveVLA")
HARNESS_PATH = ROOT / "scripts/robotwin_eval_harness.py"
MODEL_PATH = ROOT / "checkpoints/pi05_robotwin_lerobot"
DATA_ROOT = ROOT / "data/gate2_robotwin_cache/clean"
TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
REPORT_PATH = ROOT / "results/gate3_conditioning_prefix_probe.md"
EPISODE = 50
POSITION = 31

CAMERA_FILES = {
    "head_camera": DATA_ROOT
    / "videos/observation.images.cam_high/chunk-000/file-000.mp4",
    "left_camera": DATA_ROOT
    / "videos/observation.images.cam_left_wrist/chunk-000/file-000.mp4",
    "right_camera": DATA_ROOT
    / "videos/observation.images.cam_right_wrist/chunk-000/file-000.mp4",
}


def load_harness():
    if str(HARNESS_PATH.parent) not in sys.path:
        sys.path.insert(0, str(HARNESS_PATH.parent))
    spec = importlib.util.spec_from_file_location("robotwin_eval_harness", HARNESS_PATH)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import harness from {HARNESS_PATH}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def decode_frame(path: Path, frame_index: int) -> np.ndarray:
    with av.open(str(path)) as container:
        for index, frame in enumerate(container.decode(video=0)):
            if index == frame_index:
                return frame.to_rgb().to_ndarray()
    raise RuntimeError(f"missing frame {frame_index} in {path}")


def tensor_stats(value: torch.Tensor) -> dict[str, object]:
    x = value.detach().float()
    flat = x.reshape(-1)
    token_norms = x.reshape(x.shape[0], x.shape[1], -1).norm(dim=-1)
    return {
        "shape": list(x.shape),
        "dtype": str(value.dtype),
        "l2": float(torch.linalg.vector_norm(flat)),
        "mean": float(x.mean()),
        "std": float(x.std()),
        "min": float(x.min()),
        "max": float(x.max()),
        "token_norm_mean": float(token_norms.mean()),
        "token_norm_std": float(token_norms.std()),
        "nonzero": bool(float(torch.linalg.vector_norm(flat)) > 1e-8),
        "nonconstant": bool(float(x.std()) > 1e-8),
    }


def fmt(value: object, precision: int = 6) -> str:
    if torch.is_tensor(value):
        values = value.detach().float().cpu().reshape(-1).tolist()
    else:
        values = np.asarray(value).reshape(-1).tolist()
    return "[" + ", ".join(f"{float(v):.{precision}f}" for v in values) + "]"


def get_normalizer(preprocessor):
    for step in preprocessor.steps:
        if step.__class__.__name__ == "NormalizerProcessorStep":
            return step
    raise RuntimeError("NormalizerProcessorStep not found")


def make_observation(harness, decoded: dict[str, np.ndarray], state: np.ndarray):
    return harness.simulator_observation(
        {
            "observation": {
                camera: {"rgb": decoded[camera]}
                for camera in ("head_camera", "left_camera", "right_camera")
            },
            "joint_action": {"vector": state.tolist()},
        }
    )


def make_batch(
    harness,
    observation,
    instruction: str,
    env_preprocessor,
    preprocessor,
    policy_config,
):
    return harness.make_policy_batch(
        observation,
        instruction,
        env_preprocessor,
        preprocessor,
        policy_config,
    )


def config_value(policy, name: str):
    for owner in (policy.config, policy, getattr(policy, "model", None)):
        if owner is not None and hasattr(owner, name):
            return getattr(owner, name)
    return None


def clone_with_prefix(
    prefix_embs: torch.Tensor,
    prefix_pad_masks: torch.Tensor,
    prefix_att_masks: torch.Tensor,
    keep_positions: torch.Tensor | None = None,
    zero: bool = False,
):
    if keep_positions is not None:
        prefix_embs = prefix_embs.index_select(1, keep_positions)
        prefix_pad_masks = prefix_pad_masks.index_select(1, keep_positions)
        prefix_att_masks = prefix_att_masks.index_select(1, keep_positions)
    if zero:
        prefix_embs = torch.zeros_like(prefix_embs)
    return prefix_embs, prefix_pad_masks, prefix_att_masks


def build_prefix_cache(model, prefix_embs, prefix_pad_masks, prefix_att_masks):
    from lerobot.policies.common.vla_utils import (
        make_att_2d_masks,
        prepare_attention_masks_4d,
    )

    prefix_att_2d_masks = make_att_2d_masks(prefix_pad_masks, prefix_att_masks)
    prefix_position_ids = torch.cumsum(prefix_pad_masks, dim=1) - 1
    prefix_att_masks_4d = prepare_attention_masks_4d(prefix_att_2d_masks)
    model.paligemma_with_expert.paligemma.model.language_model.config._attn_implementation = (
        "eager"
    )
    _, past_key_values = model.paligemma_with_expert.forward(
        attention_mask=prefix_att_masks_4d,
        position_ids=prefix_position_ids,
        past_key_values=None,
        inputs_embeds=[prefix_embs, None],
        use_cache=True,
    )
    return past_key_values


def velocity_at_t1(model, prefix_embs, prefix_pad_masks, prefix_att_masks, noise):
    past_key_values = build_prefix_cache(
        model, prefix_embs, prefix_pad_masks, prefix_att_masks
    )
    timestep = torch.ones(noise.shape[0], dtype=torch.float32, device=noise.device)
    with torch.inference_mode():
        return model.denoise_step(
            prefix_pad_masks=prefix_pad_masks,
            past_key_values=past_key_values,
            x_t=noise,
            timestep=timestep,
        )


def legacy_kv_layers(cache):
    if hasattr(cache, "to_legacy_cache"):
        cache = cache.to_legacy_cache()
    if hasattr(cache, "key_cache") and hasattr(cache, "value_cache"):
        return list(zip(cache.key_cache, cache.value_cache, strict=True))
    if isinstance(cache, (tuple, list)):
        return list(cache)
    try:
        return list(cache)
    except TypeError:
        return []


def kv_slice(value: torch.Tensor, positions: torch.Tensor, total_length: int):
    seq_dim = next(
        (dim for dim, size in enumerate(value.shape) if size == total_length), None
    )
    if seq_dim is None:
        return None
    return value.index_select(seq_dim, positions)


def kv_source_norms(cache, source_positions: dict[str, torch.Tensor], total_length: int):
    accum = {name: 0.0 for name in source_positions}
    layer_count = 0
    for layer in legacy_kv_layers(cache):
        if not isinstance(layer, (tuple, list)) or len(layer) < 2:
            continue
        layer_count += 1
        for value in layer[:2]:
            if not torch.is_tensor(value):
                continue
            for name, positions in source_positions.items():
                sliced = kv_slice(value, positions, total_length)
                if sliced is not None:
                    accum[name] += float(sliced.detach().float().pow(2).sum())
    return {
        name: {
            "l2": float(np.sqrt(value)),
            "nonzero": bool(value > 1e-16),
        }
        for name, value in accum.items()
    }, layer_count


def source_stats(prefix_embs, source_positions: dict[str, torch.Tensor]):
    result = {}
    for name, positions in source_positions.items():
        selected = prefix_embs.index_select(1, positions)
        result[name] = tensor_stats(selected)
    return result


def contiguous_positions(start: int, end: int, device):
    return torch.arange(start, end, dtype=torch.long, device=device)


def main() -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["PYTHONUNBUFFERED"] = "1"
    torch.set_num_threads(min(16, max(1, os.cpu_count() or 1)))

    harness = load_harness()
    data_path = DATA_ROOT / "data/chunk-000/file-000.parquet"
    table = pq.read_table(
        data_path,
        columns=["observation.state", "episode_index", "index", "task_index", "frame_index"],
    ).to_pydict()
    tasks = pq.read_table(DATA_ROOT / "meta/tasks.parquet").to_pydict()
    task_text = dict(zip(tasks["task_index"], tasks["task"]))
    rows = [
        {key: table[key][i] for key in table}
        for i in range(len(table["episode_index"]))
        if int(table["episode_index"][i]) == EPISODE
    ]
    rows.sort(key=lambda item: int(item["frame_index"]))
    if len(rows) <= POSITION:
        raise RuntimeError(f"episode {EPISODE} has only {len(rows)} rows")
    row = rows[POSITION]
    video_index = int(row["index"])
    instruction = str(task_text[int(row["task_index"])])
    state = np.asarray(row["observation.state"], dtype=np.float32).reshape(14)

    decoded = {
        camera: decode_frame(path, video_index)
        for camera, path in CAMERA_FILES.items()
    }
    args = Namespace(
        model_path=str(MODEL_PATH),
        task="beat_block_hammer",
        episode_steps=1200,
        tokenizer_path=str(TOKENIZER_PATH),
        task_config="demo_clean",
    )
    policy, preprocessor, _postprocessor, env_preprocessor, _env_postprocessor = (
        harness.load_policy(args)
    )
    model = policy.model
    policy.eval()

    observation_real = make_observation(harness, decoded, state)
    observation_zero = make_observation(
        harness, decoded, np.zeros(14, dtype=np.float32)
    )
    batch_real = make_batch(
        harness,
        observation_real,
        instruction,
        env_preprocessor,
        preprocessor,
        policy.config,
    )
    batch_zero = make_batch(
        harness,
        observation_zero,
        instruction,
        env_preprocessor,
        preprocessor,
        policy.config,
    )

    from lerobot.utils.constants import (
        OBS_LANGUAGE_ATTENTION_MASK,
        OBS_LANGUAGE_TOKENS,
    )

    real_tokens = batch_real[OBS_LANGUAGE_TOKENS]
    zero_tokens = batch_zero[OBS_LANGUAGE_TOKENS]
    language_mask = batch_real[OBS_LANGUAGE_ATTENTION_MASK].bool()
    state_token_positions = torch.where(
        (real_tokens[0] != zero_tokens[0]) & language_mask[0]
    )[0]
    if state_token_positions.numel() == 0:
        raise RuntimeError("state token positions could not be identified")

    with torch.inference_mode():
        images, img_masks = policy._preprocess_images(batch_real)
        prefix_embs, prefix_pad_masks, prefix_att_masks = model.embed_prefix(
            images,
            img_masks,
            real_tokens,
            language_mask,
        )
        image_lengths = [
            int(model.paligemma_with_expert.embed_image(image).shape[1])
            for image in images
        ]

    image_total = sum(image_lengths)
    language_length = int(real_tokens.shape[1])
    if prefix_embs.shape[1] != image_total + language_length:
        raise RuntimeError(
            f"prefix length mismatch: {prefix_embs.shape[1]} != "
            f"{image_total}+{language_length}"
        )
    device = prefix_embs.device
    image_positions = {}
    offset = 0
    for name, length in zip(("image_head", "image_left_wrist", "image_right_wrist"), image_lengths):
        image_positions[name] = contiguous_positions(offset, offset + length, device)
        offset += length
    language_positions = contiguous_positions(image_total, image_total + language_length, device)
    state_positions = image_total + state_token_positions.to(device)
    non_state_language_positions = language_positions[
        ~torch.isin(language_positions, state_positions)
    ]
    source_positions = {
        **image_positions,
        "language_non_state": non_state_language_positions,
        "language_all": language_positions,
        "state_tokens": state_positions,
    }

    prefix_source_stats = source_stats(prefix_embs, source_positions)
    state_token_ids = real_tokens[0, state_token_positions].detach().cpu().tolist()
    state_token_positions_list = state_token_positions.detach().cpu().tolist()

    torch.manual_seed(20261002)
    noise = model.sample_noise(
        (1, model.config.chunk_size, model.config.max_action_dim),
        device,
    )
    real_prefix = clone_with_prefix(
        prefix_embs, prefix_pad_masks, prefix_att_masks
    )
    zero_prefix = clone_with_prefix(
        prefix_embs, prefix_pad_masks, prefix_att_masks, zero=True
    )
    keep = torch.ones(prefix_embs.shape[1], dtype=torch.bool, device=device)
    keep[state_positions] = False
    state_removed_prefix = clone_with_prefix(
        prefix_embs,
        prefix_pad_masks,
        prefix_att_masks,
        keep_positions=torch.where(keep)[0],
    )

    with torch.inference_mode():
        real_cache = build_prefix_cache(model, *real_prefix)
        zero_cache = build_prefix_cache(model, *zero_prefix)
        removed_cache = build_prefix_cache(model, *state_removed_prefix)
        velocity_real = model.denoise_step(
            real_prefix[1],
            real_cache,
            noise,
            torch.ones(1, dtype=torch.float32, device=device),
        )
        velocity_zero = model.denoise_step(
            zero_prefix[1],
            zero_cache,
            noise,
            torch.ones(1, dtype=torch.float32, device=device),
        )
        velocity_removed = model.denoise_step(
            state_removed_prefix[1],
            removed_cache,
            noise,
            torch.ones(1, dtype=torch.float32, device=device),
        )

    velocity_map = {
        "real_prefix": velocity_real,
        "zeroed_prefix": velocity_zero,
        "state_removed": velocity_removed,
    }
    velocity_stats = {
        name: {
            "shape": list(value.shape),
            "l2_full_50x32": float(value.float().norm()),
            "l2_first14_dims": float(value[:, :, :14].float().norm()),
            "mean": float(value.float().mean()),
            "std": float(value.float().std()),
        }
        for name, value in velocity_map.items()
    }
    velocity_delta = {
        "real_minus_zeroed_l2": float((velocity_real - velocity_zero).float().norm()),
        "real_minus_state_removed_l2": float(
            (velocity_real - velocity_removed).float().norm()
        ),
        "zeroed_minus_state_removed_l2": float(
            (velocity_zero - velocity_removed).float().norm()
        ),
    }

    real_kv, real_layer_count = kv_source_norms(
        real_cache, source_positions, prefix_embs.shape[1]
    )
    zero_kv, _ = kv_source_norms(
        zero_cache,
        {
            "all_prefix": contiguous_positions(0, zero_prefix[0].shape[1], device),
        },
        zero_prefix[0].shape[1],
    )
    removed_kv, _ = kv_source_norms(
        removed_cache,
        {
            "all_prefix": contiguous_positions(
                0, state_removed_prefix[0].shape[1], device
            ),
        },
        state_removed_prefix[0].shape[1],
    )

    normalizer = get_normalizer(preprocessor)
    normalized_state = preprocessor.steps[3](
        {
            "observation.state": torch.from_numpy(state).reshape(1, 14),
        }
    )["observation.state"]
    normalized_state = normalized_state.detach().float().cpu().numpy().reshape(14)
    discretized_state = (
        np.digitize(normalized_state, bins=np.linspace(-1, 1, 257)[:-1]) - 1
    ).astype(int)
    cleaned_instruction = instruction.strip().replace("_", " ").replace("\n", " ")
    canonical_prompt = (
        f"Task: {cleaned_instruction}, "
        f"State: {' '.join(map(str, discretized_state))};\nAction: "
    )

    source_non_degenerate = {
        name: bool(value["nonzero"] and value["nonconstant"])
        for name, value in prefix_source_stats.items()
    }
    state_present = bool(state_token_positions.numel() > 0)
    conditioning_response = max(velocity_delta.values())
    if not state_present:
        verdict = "(b) state-token path is absent"
    elif conditioning_response < 1e-4:
        verdict = "(a) conditioning wiring is effectively ignored by the denoiser"
    else:
        verdict = (
            "(c) conditioning reaches the denoiser; the remaining fault is "
            "weak/behaviorally divergent checkpoint conditioning rather than a "
            "missing prefix path"
        )

    modeling_path = Path(
        "/root/autodl-tmp/conda_envs/evolvevla_pi312/lib/python3.12/site-packages/"
        "lerobot/policies/pi05/modeling_pi05.py"
    )
    processor_path = Path(
        "/root/autodl-tmp/conda_envs/evolvevla_pi312/lib/python3.12/site-packages/"
        "lerobot/policies/pi05/processor_pi05.py"
    )
    meta = json.loads((DATA_ROOT / "meta/info.json").read_text(encoding="utf-8"))
    state_order = meta["features"]["observation.state"]["names"][0]

    lines = [
        "# Gate3 Conditioning Prefix Probe",
        "",
        f"- Experiment: `gate3_conditioning_prefix_probe`",
        f"- Checkpoint: `{MODEL_PATH}`",
        f"- Harness: `{HARNESS_PATH}`",
        f"- Harness SHA256: `{hashlib.sha256(HARNESS_PATH.read_bytes()).hexdigest()}`",
        f"- LeRobot modeling_pi05 SHA256: `{hashlib.sha256(modeling_path.read_bytes()).hexdigest()}`",
        f"- LeRobot processor_pi05 SHA256: `{hashlib.sha256(processor_path.read_bytes()).hexdigest()}`",
        f"- Sample: training episode `{EPISODE}`, position `{POSITION}`, video index `{video_index}`",
        f"- Instruction: `{instruction}`",
        f"- State order: `{state_order}`",
        f"- CUDA: `{torch.cuda.current_device()}` ({torch.cuda.get_device_name(0)})",
        "",
        "## Prefix Layout",
        "",
        f"- Prefix shape: `{list(prefix_embs.shape)}`",
        f"- Image token lengths: `{image_lengths}`; language length: `{language_length}`",
        f"- State-token positions in language sequence: `{state_token_positions_list}`",
        f"- State-token IDs: `{state_token_ids}`",
        f"- State-token count: `{len(state_token_positions_list)}`",
        f"- Canonical prompt reconstructed from normalized state: `{canonical_prompt}`",
        "",
        "## Prefix Embedding Norms",
        "",
        "Non-degenerate means both nonzero L2 and nonzero elementwise standard deviation.",
        "",
        "```json",
        json.dumps(prefix_source_stats, indent=2, sort_keys=True),
        "```",
        f"- Non-degenerate flags: `{json.dumps(source_non_degenerate, sort_keys=True)}`",
        "",
        "## KV Norms",
        "",
        f"- KV layers observed: `{real_layer_count}`",
        "```json",
        json.dumps(
            {
                "real_prefix": real_kv,
                "zeroed_prefix_all": zero_kv,
                "state_removed_all": removed_kv,
            },
            indent=2,
            sort_keys=True,
        ),
        "```",
        "",
        "## Velocity At t=1.0",
        "",
        "The same sampled action noise is used for all three conditions; no rollout is run.",
        "",
        "```json",
        json.dumps(velocity_stats, indent=2, sort_keys=True),
        "```",
        f"- Pairwise velocity deltas: `{json.dumps(velocity_delta, sort_keys=True)}`",
        f"- Conditioning response verdict: `{'RESPONDS' if conditioning_response >= 1e-4 else 'UNCHANGED'}`",
        "",
        "## State-Token Path",
        "",
        "- Harness state enters as `joint_action.vector` -> `agent_pos` -> `observation.state` without reordering.",
        "- `robotwin_eval_harness.py:475-503` assembles the 14-D state and calls the production preprocessor.",
        "- `processor_pi05.py:44-73` reads normalized `observation.state` and discretizes it into 256 bins.",
        "- `processor_pi05.py:136-145` places `Pi05PrepareStateTokenizerProcessorStep` after normalization and before tokenization.",
        "- `modeling_pi05.py:483-518` embeds image and language prefixes; the state is present as language-token embeddings.",
        f"- Normalized state: `{fmt(normalized_state)}`",
        f"- Discretized state: `{discretized_state.tolist()}`",
        f"- State path verdict: `{'PRESENT: norm -> discretize -> embed' if state_present else 'ABSENT'}`",
        "",
        "## Training-vs-Eval Observation Sanity",
        "",
        "- Not run: the retained eval dump has only `cam_high` PNGs and no matching wrist frames plus 14-D state in one on-disk observation, so a valid three-camera eval comparison would require fabricating inputs.",
        "",
        "## Verdict",
        "",
        f"- `{verdict}`",
        "- This bounded result localizes the decision to prefix behavior. It does not establish leaderboard-checkpoint identity.",
        "",
    ]
    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    REPORT_PATH.write_text("\n".join(lines), encoding="utf-8")

    print(f"OUTPUT={REPORT_PATH}")
    print(f"RESULT_SHA256={hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()}")
    print(f"PREFIX_SHAPE={list(prefix_embs.shape)}")
    print(f"STATE_TOKEN_POSITIONS={state_token_positions_list}")
    print(f"STATE_TOKEN_COUNT={len(state_token_positions_list)}")
    print(f"PREFIX_NONDEGENERATE={json.dumps(source_non_degenerate, sort_keys=True)}")
    print(f"VELOCITY_STATS={json.dumps(velocity_stats, sort_keys=True)}")
    print(f"VELOCITY_DELTAS={json.dumps(velocity_delta, sort_keys=True)}")
    print(f"STATE_PATH={'PRESENT' if state_present else 'ABSENT'}")
    print(f"TRAINING_VS_EVAL=NOT_RUN_INSUFFICIENT_ON_DISK_OBSERVATION")
    print(f"VERDICT={verdict}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
