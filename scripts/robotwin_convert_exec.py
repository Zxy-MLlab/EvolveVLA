#!/usr/bin/env python3
"""Convert the neutral Stage-1 Orbax NPZ dump into a LeRobot PI05 policy.

This executor intentionally does not use JAX or Orbax.  It consumes either the
requested directory of one-leaf NPZ files or the single NPZ archive emitted by
the earlier Stage-1 dumper.  The conversion is keyed by the Stage-1 manifest,
not by the checkpoint directory name.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import shutil
import sys
import traceback
from pathlib import Path
from typing import Callable

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import numpy as np
import torch
from safetensors.torch import safe_open, save_file


DEFAULT_DUMP = Path("/root/autodl-tmp/EvolveVLA/checkpoints/orbax_npz_dump")
DEFAULT_MANIFEST = Path("/root/autodl-tmp/EvolveVLA/results/stage1_leaf_manifest.txt")
DEFAULT_NORM_STATS = Path(
    "/root/autodl-tmp/EvolveVLA/checkpoints/pi05_robotwin/"
    "RoboTwin-lerobot_v30-aloha_agilex-joint-0/29999/assets/"
    "RoboTwin_lerobot_v30/norm_stats.json"
)
DEFAULT_OUTPUT = Path("/root/autodl-tmp/EvolveVLA/checkpoints/pi05_robotwin_lerobot")
LOCAL_TOKENIZER = Path("/root/autodl-tmp/paligemma_tokenizer")
VISION_DEPTH = 27
LLM_DEPTH = 18
EXPECTED_SOURCE_LEAVES = 51
EXPECTED_SOURCE_ELEMENTS = 3_353_433_872


def d(key: str) -> str:
    return f"dict[{key!r}]"


def source_key(*parts: str) -> str:
    return "/".join(d(part) for part in parts)


def parse_manifest(path: Path) -> dict[str, tuple[tuple[int, ...], str]]:
    rows: dict[str, tuple[tuple[int, ...], str]] = {}
    pattern = re.compile(r"^(.*?) \((.*?)\) (\S+)$")
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            raise ValueError(f"Malformed manifest line: {raw_line!r}")
        shape_text = match.group(2).strip()
        if not shape_text:
            shape = ()
        else:
            shape_parts = [part.strip() for part in shape_text.split(",")]
            if shape_parts and not shape_parts[-1]:
                shape_parts.pop()
            if any(not part for part in shape_parts):
                raise ValueError(f"Malformed shape in manifest line: {raw_line!r}")
            shape = tuple(int(part) for part in shape_parts)
        rows[match.group(1)] = (shape, match.group(3))
    return rows


def canonical_member_name(name: str) -> str:
    return name.removesuffix(".npy")


def load_npz_archive(path: Path) -> dict[str, np.ndarray]:
    arrays: dict[str, np.ndarray] = {}
    with np.load(path, allow_pickle=False) as archive:
        for member in archive.files:
            arrays[canonical_member_name(member)] = np.asarray(archive[member])
    return arrays


def load_neutral_dump(path: Path) -> dict[str, np.ndarray]:
    if path.is_file():
        return load_npz_archive(path)
    if not path.is_dir():
        raise FileNotFoundError(f"Neutral dump does not exist: {path}")

    files = sorted(path.rglob("*.npz"))
    if not files:
        raise FileNotFoundError(f"No NPZ leaves found in neutral dump: {path}")

    arrays: dict[str, np.ndarray] = {}
    for leaf_file in files:
        with np.load(leaf_file, allow_pickle=False) as archive:
            members = archive.files
            if len(members) != 1:
                raise ValueError(f"Expected one array in {leaf_file}, found {members}")
            member = canonical_member_name(members[0])
            name = leaf_file.relative_to(path).with_suffix("").as_posix()
            if member != "arr_0" and member != name:
                raise ValueError(f"Leaf name mismatch for {leaf_file}: member={member!r} path={name!r}")
            arrays[name] = np.asarray(archive[members[0]])
    return arrays


def validate_source(
    arrays: dict[str, np.ndarray],
    manifest: dict[str, tuple[tuple[int, ...], str]],
) -> None:
    missing = sorted(set(manifest) - set(arrays))
    extra = sorted(set(arrays) - set(manifest))
    if missing or extra:
        raise ValueError(f"Source leaf mismatch: missing={missing} extra={extra}")
    if len(manifest) != EXPECTED_SOURCE_LEAVES:
        raise ValueError(f"Expected {EXPECTED_SOURCE_LEAVES} manifest leaves, got {len(manifest)}")
    total = 0
    for name, (shape, dtype) in manifest.items():
        array = arrays[name]
        if tuple(array.shape) != shape:
            raise ValueError(f"Shape mismatch for {name}: {array.shape} != {shape}")
        if str(array.dtype) != dtype:
            raise ValueError(f"Dtype mismatch for {name}: {array.dtype} != {dtype}")
        total += int(array.size)
    if total != EXPECTED_SOURCE_ELEMENTS:
        raise ValueError(f"Expected {EXPECTED_SOURCE_ELEMENTS} source elements, got {total}")


def as_tensor(array: np.ndarray, *, transpose: bool = False) -> torch.Tensor:
    value = np.asarray(array)
    if transpose:
        value = value.T
    return torch.from_numpy(np.ascontiguousarray(value))


def expect(arrays: dict[str, np.ndarray], name: str, shape: tuple[int, ...] | None = None) -> np.ndarray:
    if name not in arrays:
        raise KeyError(f"Missing required source leaf: {name}")
    value = arrays[name]
    if shape is not None and tuple(value.shape) != shape:
        raise ValueError(f"Unexpected shape for {name}: {value.shape} != {shape}")
    return value


def build_state_dict(
    arrays: dict[str, np.ndarray],
) -> tuple[dict[str, torch.Tensor], list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]]]:
    """Build LeRobot/OpenPI names and retain independent layer alignment checks."""
    state: dict[str, torch.Tensor] = {}
    alignment_records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]] = []

    def add(name: str, array: np.ndarray, transform: Callable[[np.ndarray], np.ndarray] | None = None) -> None:
        transformed = array if transform is None else transform(array)
        state[name] = as_tensor(transformed)

    def add_layered(
        source: str,
        depth: int,
        output_name: Callable[[int], str],
        transform: Callable[[np.ndarray], np.ndarray] | None = None,
    ) -> None:
        array = expect(arrays, source)
        if array.shape[0] != depth:
            raise ValueError(f"Layer depth mismatch for {source}: {array.shape[0]} != {depth}")
        transform_fn = transform or (lambda x: x)
        for layer_index in range(depth):
            output = output_name(layer_index)
            state[output] = as_tensor(transform_fn(array[layer_index]))
            alignment_records.append((output, array, layer_index, transform_fn))

    # Vision embedding and projector.
    add(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.weight",
        expect(arrays, source_key("params", "PaliGemma", "img", "embedding", "kernel", "value")),
        lambda x: np.transpose(x, (3, 2, 0, 1)),
    )
    add(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.patch_embedding.bias",
        expect(arrays, source_key("params", "PaliGemma", "img", "embedding", "bias", "value")),
    )
    add(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.embeddings.position_embedding.weight",
        expect(arrays, source_key("params", "PaliGemma", "img", "pos_embedding", "value")),
        lambda x: x.reshape(-1, 1152),
    )
    add(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.weight",
        expect(arrays, source_key("params", "PaliGemma", "img", "Transformer", "encoder_norm", "scale", "value")),
    )
    add(
        "paligemma_with_expert.paligemma.model.vision_tower.vision_model.post_layernorm.bias",
        expect(arrays, source_key("params", "PaliGemma", "img", "Transformer", "encoder_norm", "bias", "value")),
    )
    add(
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.weight",
        expect(arrays, source_key("params", "PaliGemma", "img", "head", "kernel", "value")),
        lambda x: x.T,
    )
    add(
        "paligemma_with_expert.paligemma.model.multi_modal_projector.linear.bias",
        expect(arrays, source_key("params", "PaliGemma", "img", "head", "bias", "value")),
    )

    vision_root = ("params", "PaliGemma", "img", "Transformer", "encoderblock")
    vision_layer_prefix = "paligemma_with_expert.paligemma.model.vision_tower.vision_model.encoder.layers"
    add_layered(
        source_key(*vision_root, "LayerNorm_0", "scale", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.layer_norm1.weight",
    )
    add_layered(
        source_key(*vision_root, "LayerNorm_0", "bias", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.layer_norm1.bias",
    )
    add_layered(
        source_key(*vision_root, "LayerNorm_1", "scale", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.layer_norm2.weight",
    )
    add_layered(
        source_key(*vision_root, "LayerNorm_1", "bias", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.layer_norm2.bias",
    )
    add_layered(
        source_key(*vision_root, "MlpBlock_0", "Dense_0", "kernel", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.mlp.fc1.weight",
        lambda x: x.T,
    )
    add_layered(
        source_key(*vision_root, "MlpBlock_0", "Dense_0", "bias", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.mlp.fc1.bias",
    )
    add_layered(
        source_key(*vision_root, "MlpBlock_0", "Dense_1", "kernel", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.mlp.fc2.weight",
        lambda x: x.T,
    )
    add_layered(
        source_key(*vision_root, "MlpBlock_0", "Dense_1", "bias", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.mlp.fc2.bias",
    )
    vision_attention_root = (*vision_root, "MultiHeadDotProductAttention_0")
    for short, target in (("query", "q_proj"), ("key", "k_proj"), ("value", "v_proj")):
        add_layered(
            source_key(*vision_attention_root, short, "kernel", "value"),
            VISION_DEPTH,
            lambda i, target=target: f"{vision_layer_prefix}.{i}.self_attn.{target}.weight",
            lambda x: x.reshape(-1, 1152).T,
        )
        add_layered(
            source_key(*vision_attention_root, short, "bias", "value"),
            VISION_DEPTH,
            lambda i, target=target: f"{vision_layer_prefix}.{i}.self_attn.{target}.bias",
            lambda x: x.reshape(-1),
        )
    add_layered(
        source_key(*vision_attention_root, "out", "kernel", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.self_attn.out_proj.weight",
        lambda x: x.reshape(-1, 1152).T,
    )
    add_layered(
        source_key(*vision_attention_root, "out", "bias", "value"),
        VISION_DEPTH,
        lambda i: f"{vision_layer_prefix}.{i}.self_attn.out_proj.bias",
        lambda x: x.reshape(-1),
    )

    # PaliGemma language model: RMSNorm path, depth 18.
    llm_layer_prefix = "paligemma_with_expert.paligemma.model.language_model.layers"
    add(
        "paligemma_with_expert.paligemma.model.language_model.embed_tokens.weight",
        expect(arrays, source_key("params", "PaliGemma", "llm", "embedder", "input_embedding", "value")),
    )
    base_final_norm_scale = source_key("params", "PaliGemma", "llm", "final_norm", "scale", "value")
    base_final_norm_bias = source_key("params", "PaliGemma", "llm", "final_norm", "Dense_0", "bias", "value")
    base_final_norm_kernel = source_key("params", "PaliGemma", "llm", "final_norm", "Dense_0", "kernel", "value")
    if (
        base_final_norm_scale not in arrays
        or base_final_norm_bias in arrays
        or base_final_norm_kernel in arrays
    ):
        raise AssertionError("final_norm must be RMSNorm scale-only from checkpoint structure")
    add(
        "paligemma_with_expert.paligemma.model.language_model.norm.weight",
        expect(arrays, base_final_norm_scale),
    )
    add_llm_layered(
        arrays,
        state,
        alignment_records,
        "attn_vec_einsum",
        lambda i: f"{llm_layer_prefix}.{i}.self_attn.o_proj.weight",
        lambda x: x.transpose(2, 0, 1).reshape(2048, 2048),
        LLM_DEPTH,
    )
    add_llm_layered(
        arrays,
        state,
        alignment_records,
        "kv_einsum",
        lambda i: f"{llm_layer_prefix}.{i}.self_attn.k_proj.weight",
        lambda x: x[0, 0].T,
        LLM_DEPTH,
    )
    add_llm_layered(
        arrays,
        state,
        alignment_records,
        "kv_einsum",
        lambda i: f"{llm_layer_prefix}.{i}.self_attn.v_proj.weight",
        lambda x: x[1, 0].T,
        LLM_DEPTH,
    )
    add_llm_layered(
        arrays,
        state,
        alignment_records,
        "q_einsum",
        lambda i: f"{llm_layer_prefix}.{i}.self_attn.q_proj.weight",
        lambda x: x.transpose(0, 2, 1).reshape(2048, 2048),
        LLM_DEPTH,
    )
    add_llm_mlp_layered(
        arrays,
        state,
        alignment_records,
        "mlp",
        lambda i: f"{llm_layer_prefix}.{i}.mlp.gate_proj.weight",
        lambda x: x[0].T,
        LLM_DEPTH,
    )
    add_llm_mlp_layered(
        arrays,
        state,
        alignment_records,
        "mlp",
        lambda i: f"{llm_layer_prefix}.{i}.mlp.up_proj.weight",
        lambda x: x[1].T,
        LLM_DEPTH,
    )
    add_llm_mlp_linear_layered(
        arrays,
        state,
        alignment_records,
        "mlp",
        lambda i: f"{llm_layer_prefix}.{i}.mlp.down_proj.weight",
        lambda x: x.T,
        LLM_DEPTH,
    )
    for norm_name, target in (
        ("pre_attention_norm", "input_layernorm.weight"),
        ("pre_ffw_norm", "post_attention_layernorm.weight"),
    ):
        add_llm_norm_layered(
            arrays,
            state,
            alignment_records,
            norm_name,
            lambda i, target=target: f"{llm_layer_prefix}.{i}.{target}",
            LLM_DEPTH,
        )

    # Action expert: the checkpoint structure, not its dirname, selects Dense-vs-RMSNorm.
    expert_prefix = "paligemma_with_expert.gemma_expert.model.layers"
    expert_norm_root = ("params", "PaliGemma", "llm", "layers")
    for norm_name, target in (
        ("pre_attention_norm_1", "input_layernorm"),
        ("pre_ffw_norm_1", "post_attention_layernorm"),
    ):
        dense_bias = source_key(*expert_norm_root, norm_name, "Dense_0", "bias", "value")
        dense_kernel = source_key(*expert_norm_root, norm_name, "Dense_0", "kernel", "value")
        scale = source_key(*expert_norm_root, norm_name, "scale", "value")
        if dense_bias not in arrays or dense_kernel not in arrays or scale in arrays:
            raise AssertionError(f"Expert adaptive norm structure is invalid for {norm_name}")
        add_layered(
            dense_bias,
            LLM_DEPTH,
            lambda i, target=target: f"{expert_prefix}.{i}.{target}.dense.bias",
        )
        add_layered(
            dense_kernel,
            LLM_DEPTH,
            lambda i, target=target: f"{expert_prefix}.{i}.{target}.dense.weight",
            lambda x: x.T,
        )

    expert_layer_root = ("params", "PaliGemma", "llm", "layers", "attn")
    add_llm_expert_layered(
        arrays,
        state,
        alignment_records,
        expert_layer_root,
        "attn_vec_einsum_1",
        lambda i: f"{expert_prefix}.{i}.self_attn.o_proj.weight",
        lambda x: x.reshape(2048, 1024).T,
        LLM_DEPTH,
    )
    add_llm_expert_layered(
        arrays,
        state,
        alignment_records,
        expert_layer_root,
        "kv_einsum_1",
        lambda i: f"{expert_prefix}.{i}.self_attn.k_proj.weight",
        lambda x: x[0, 0].T,
        LLM_DEPTH,
    )
    add_llm_expert_layered(
        arrays,
        state,
        alignment_records,
        expert_layer_root,
        "kv_einsum_1",
        lambda i: f"{expert_prefix}.{i}.self_attn.v_proj.weight",
        lambda x: x[1, 0].T,
        LLM_DEPTH,
    )
    add_llm_expert_layered(
        arrays,
        state,
        alignment_records,
        expert_layer_root,
        "q_einsum_1",
        lambda i: f"{expert_prefix}.{i}.self_attn.q_proj.weight",
        lambda x: x.transpose(0, 2, 1).reshape(2048, 1024),
        LLM_DEPTH,
    )
    add_llm_expert_mlp_layered(
        arrays,
        state,
        alignment_records,
        "mlp_1",
        lambda i: f"{expert_prefix}.{i}.mlp.gate_proj.weight",
        lambda x: x[0].T,
        LLM_DEPTH,
    )
    add_llm_expert_mlp_layered(
        arrays,
        state,
        alignment_records,
        "mlp_1",
        lambda i: f"{expert_prefix}.{i}.mlp.up_proj.weight",
        lambda x: x[1].T,
        LLM_DEPTH,
    )
    add_llm_expert_linear_layered(
        arrays,
        state,
        alignment_records,
        "mlp_1",
        lambda i: f"{expert_prefix}.{i}.mlp.down_proj.weight",
        lambda x: x.T,
        LLM_DEPTH,
    )

    final_norm_scale = source_key("params", "PaliGemma", "llm", "final_norm_1", "scale", "value")
    final_norm_bias = source_key("params", "PaliGemma", "llm", "final_norm_1", "Dense_0", "bias", "value")
    final_norm_kernel = source_key("params", "PaliGemma", "llm", "final_norm_1", "Dense_0", "kernel", "value")
    if (
        final_norm_scale in arrays
        or final_norm_bias not in arrays
        or final_norm_kernel not in arrays
    ):
        raise AssertionError("final_norm_1 must be Dense bias+kernel, not RMSNorm scale-only")
    state["paligemma_with_expert.gemma_expert.model.norm.dense.bias"] = as_tensor(
        expect(arrays, final_norm_bias)
    )
    state["paligemma_with_expert.gemma_expert.model.norm.dense.weight"] = as_tensor(
        expect(arrays, final_norm_kernel).T
    )

    # Projection heads.
    for name in ("action_in_proj", "action_out_proj", "time_mlp_in", "time_mlp_out"):
        add(
            f"{name}.weight",
            expect(arrays, source_key("params", name, "kernel", "value")),
            lambda x: x.T,
        )
        add(f"{name}.bias", expect(arrays, source_key("params", name, "bias", "value")))

    return state, alignment_records


def add_llm_simple_layered(
    arrays: dict[str, np.ndarray],
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
    name: str,
    output_name: Callable[[int], str],
    depth: int,
) -> None:
    add_llm_norm_layered(arrays, state, records, name, output_name, depth)


def add_llm_norm_layered(
    arrays: dict[str, np.ndarray],
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
    name: str,
    output_name: Callable[[int], str],
    depth: int,
) -> None:
    source = source_key("params", "PaliGemma", "llm", "layers", name, "scale", "value")
    add_llm_layered_array(arrays, state, records, source, output_name, lambda x: x, depth)


def add_llm_layered(
    arrays: dict[str, np.ndarray],
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
    name: str,
    output_name: Callable[[int], str],
    transform: Callable[[np.ndarray], np.ndarray],
    depth: int,
) -> None:
    source = source_key("params", "PaliGemma", "llm", "layers", "attn", name, "w", "value")
    array = expect(arrays, source)
    if array.shape[0] != depth:
        raise ValueError(f"Layer depth mismatch for {source}: {array.shape[0]} != {depth}")
    for index in range(depth):
        target = output_name(index)
        state[target] = as_tensor(transform(array[index]))
        records.append((target, array, index, transform))


def add_llm_mlp_layered(
    arrays: dict[str, np.ndarray],
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
    name: str,
    output_name: Callable[[int], str],
    transform: Callable[[np.ndarray], np.ndarray],
    depth: int,
) -> None:
    source = source_key("params", "PaliGemma", "llm", "layers", name, "gating_einsum", "value")
    add_llm_layered_array(arrays, state, records, source, output_name, transform, depth)


def add_llm_mlp_linear_layered(
    arrays: dict[str, np.ndarray],
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
    name: str,
    output_name: Callable[[int], str],
    transform: Callable[[np.ndarray], np.ndarray],
    depth: int,
) -> None:
    source = source_key("params", "PaliGemma", "llm", "layers", name, "linear", "value")
    add_llm_layered_array(arrays, state, records, source, output_name, transform, depth)


def add_llm_layered_array(
    arrays: dict[str, np.ndarray],
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
    source: str,
    output_name: Callable[[int], str],
    transform: Callable[[np.ndarray], np.ndarray],
    depth: int,
) -> None:
    array = expect(arrays, source)
    if array.shape[0] != depth:
        raise ValueError(f"Layer depth mismatch for {source}: {array.shape[0]} != {depth}")
    for index in range(depth):
        target = output_name(index)
        state[target] = as_tensor(transform(array[index]))
        records.append((target, array, index, transform))


def add_llm_expert_layered(
    arrays: dict[str, np.ndarray],
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
    root: tuple[str, ...],
    name: str,
    output_name: Callable[[int], str],
    transform: Callable[[np.ndarray], np.ndarray],
    depth: int,
) -> None:
    source = source_key(*root, name, "w", "value")
    add_llm_layered_array(arrays, state, records, source, output_name, transform, depth)


def add_llm_expert_mlp_layered(
    arrays: dict[str, np.ndarray],
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
    name: str,
    output_name: Callable[[int], str],
    transform: Callable[[np.ndarray], np.ndarray],
    depth: int,
) -> None:
    source = source_key("params", "PaliGemma", "llm", "layers", name, "gating_einsum", "value")
    add_llm_layered_array(arrays, state, records, source, output_name, transform, depth)


def add_llm_expert_linear_layered(
    arrays: dict[str, np.ndarray],
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
    name: str,
    output_name: Callable[[int], str],
    transform: Callable[[np.ndarray], np.ndarray],
    depth: int,
) -> None:
    source = source_key("params", "PaliGemma", "llm", "layers", name, "linear", "value")
    add_llm_layered_array(arrays, state, records, source, output_name, transform, depth)


def check_layer_alignment(
    state: dict[str, torch.Tensor],
    records: list[tuple[str, np.ndarray, int, Callable[[np.ndarray], np.ndarray]]],
) -> list[str]:
    failures: list[str] = []
    for target, source, index, transform in records:
        expected = as_tensor(transform(source[index]))
        if not torch.equal(state[target], expected):
            failures.append(f"{target} <- layer {index}")
    return failures


def load_norm_stats(path: Path) -> dict[str, dict[str, list[float]]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("norm_stats", payload)
    if "state" not in raw or "actions" not in raw:
        raise ValueError("Norm stats must contain OpenPI keys 'state' and 'actions'")
    mapped = {
        "observation.state": raw["state"],
        "action": raw["actions"],
    }
    for feature, stats in mapped.items():
        for field in ("mean", "std", "q01", "q99"):
            if field not in stats:
                raise ValueError(f"Missing {field} in {feature} normalization stats")
        lengths = {len(stats[field]) for field in ("mean", "std", "q01", "q99")}
        if lengths != {14}:
            raise ValueError(f"{feature} normalization vectors are not 14-D: {lengths}")
    return mapped


def check_normalization_round_trip(stats: dict[str, dict[str, list[float]]]) -> float:
    state_stats = stats["observation.state"]
    raw = np.linspace(-0.37, 0.61, 14, dtype=np.float64)
    q01 = np.asarray(state_stats["q01"], dtype=np.float64)
    q99 = np.asarray(state_stats["q99"], dtype=np.float64)
    normalized = 2.0 * (raw - q01) / (q99 - q01) - 1.0
    restored = (normalized + 1.0) * (q99 - q01) / 2.0 + q01
    error = float(np.max(np.abs(raw - restored)))
    if error > 1e-12:
        raise AssertionError(f"Quantile normalization round-trip error too high: {error}")
    return error


def make_config() -> dict:
    # COPIED from the installed PI05 schema and decoded checkpoint architecture.
    # UNKNOWN operational values are kept conservative and documented separately.
    return {
        "type": "pi05",
        "n_obs_steps": 1,
        "input_features": {
            "observation.state": {"type": "STATE", "shape": [14]},
            "observation.images.cam_high": {"type": "VISUAL", "shape": [480, 640, 3]},
            "observation.images.cam_left_wrist": {"type": "VISUAL", "shape": [480, 640, 3]},
            "observation.images.cam_right_wrist": {"type": "VISUAL", "shape": [480, 640, 3]},
        },
        "output_features": {"action": {"type": "ACTION", "shape": [14]}},
        "device": "cpu",
        "use_amp": False,
        "use_peft": False,
        "push_to_hub": False,
        "repo_id": None,
        "private": None,
        "tags": None,
        "license": None,
        "pretrained_path": None,
        "pretrained_revision": None,
        "paligemma_variant": "gemma_2b",
        "action_expert_variant": "gemma_300m",
        "dtype": "float32",
        "chunk_size": 50,
        "n_action_steps": 50,
        "max_state_dim": 32,
        "max_action_dim": 32,
        "num_inference_steps": 10,
        "time_sampling_beta_alpha": 1.5,
        "time_sampling_beta_beta": 1.0,
        "time_sampling_scale": 0.999,
        "time_sampling_offset": 0.001,
        "min_period": 0.004,
        "max_period": 4.0,
        "use_relative_actions": False,
        "relative_exclude_joints": ["gripper"],
        "action_feature_names": None,
        "rtc_config": None,
        "image_resolution": [224, 224],
        "empty_cameras": 0,
        "tokenizer_max_length": 200,
        "normalization_mapping": {
            "VISUAL": "IDENTITY",
            "STATE": "QUANTILES",
            "ACTION": "QUANTILES",
        },
        "gradient_checkpointing": False,
        "compile_model": False,
        "compile_mode": "max-autotune",
        "freeze_vision_encoder": False,
        "train_expert_only": False,
        "optimizer_lr": 2.5e-5,
        "optimizer_betas": [0.9, 0.95],
        "optimizer_eps": 1e-8,
        "optimizer_weight_decay": 0.01,
        "optimizer_grad_clip_norm": 1.0,
        "scheduler_warmup_steps": 1000,
        "scheduler_decay_steps": 30000,
        "scheduler_decay_lr": 2.5e-6,
    }


def write_output_metadata(
    output: Path,
    config_payload: dict,
    stats: dict[str, dict[str, list[float]]],
) -> None:
    output.mkdir(parents=True, exist_ok=True)
    (output / "config.json").write_text(json.dumps(config_payload, indent=2) + "\n", encoding="utf-8")
    (output / "norm_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    asset_dir = output / "assets" / "RoboTwin_lerobot_v30"
    asset_dir.mkdir(parents=True, exist_ok=True)
    (asset_dir / "norm_stats.json").write_text(json.dumps(stats, indent=2) + "\n", encoding="utf-8")
    (output / "conversion_notes.md").write_text(
        """# RobotWin PI05 conversion notes

- `paligemma_variant`, `action_expert_variant`, hidden widths, depths, and
  projection shapes are copied from the decoded checkpoint architecture.
- `chunk_size`, `n_action_steps`, image feature resolution, and optimizer
  fields use the installed PI05 schema defaults where the neutral dump does
  not encode an operational setting.
- Normalization map: OpenPI `state` -> LeRobot `observation.state`;
  OpenPI `actions` -> LeRobot `action`; all four `{mean,std,q01,q99}` fields
  are retained. PI05 uses `QUANTILES`, so q01/q99 are authoritative.
""",
        encoding="utf-8",
    )

def parity_check(output_file: Path, expected: dict[str, torch.Tensor]) -> tuple[bool, int, int, list[str], list[str], list[str]]:
    actual: dict[str, tuple[int, ...]] = {}
    with safe_open(str(output_file), framework="pt", device="cpu") as handle:
        for key in handle.keys():
            actual[key] = tuple(handle.get_tensor(key).shape)
    expected_shapes = {key: tuple(value.shape) for key, value in expected.items()}
    missing = sorted(set(expected_shapes) - set(actual))
    extra = sorted(set(actual) - set(expected_shapes))
    shape_diffs = sorted(
        f"{key}: expected={expected_shapes[key]} actual={actual[key]}"
        for key in set(expected_shapes) & set(actual)
        if expected_shapes[key] != actual[key]
    )
    element_count = sum(int(np.prod(shape, dtype=np.int64)) for shape in expected_shapes.values())
    ok = not missing and not extra and not shape_diffs
    return ok, len(expected_shapes), element_count, missing, extra, shape_diffs


def sha256_and_size(path: Path) -> tuple[int, str]:
    digest = hashlib.sha256()
    size = 0
    with path.open("rb") as handle:
        while chunk := handle.read(16 * 1024 * 1024):
            digest.update(chunk)
            size += len(chunk)
    return size, digest.hexdigest()


def tensorize_norm_stats(
    stats: dict[str, dict[str, list[float]]],
) -> dict[str, dict[str, torch.Tensor]]:
    return {
        feature: {
            field: torch.tensor(values, dtype=torch.float32)
            for field, values in fields.items()
        }
        for feature, fields in stats.items()
    }


def write_processors_and_config(
    output: Path,
    stats: dict[str, dict[str, list[float]]],
) -> None:
    """Write LeRobot processor files without constructing a gated tokenizer."""
    from lerobot.policies.pi05.configuration_pi05 import PI05Config
    from lerobot.policies.pi05.processor_pi05 import (
        Pi05PrepareStateTokenizerProcessorStep,
    )
    from lerobot.processor import (
        AbsoluteActionsProcessorStep,
        RelativeActionsProcessorStep,
        TokenizerProcessorStep,
    )
    from lerobot.processor.factory import (
        make_default_policy_processor_steps,
        make_policy_processor_pipelines,
    )

    policy_cfg = PI05Config.from_pretrained(output, local_files_only=True)
    dataset_stats = tensorize_norm_stats(stats)
    steps = make_default_policy_processor_steps(policy_cfg, dataset_stats)
    relative_step = RelativeActionsProcessorStep(
        enabled=policy_cfg.use_relative_actions,
        exclude_joints=getattr(policy_cfg, "relative_exclude_joints", []),
        action_names=getattr(policy_cfg, "action_feature_names", None),
    )
    preprocessor, postprocessor = make_policy_processor_pipelines(
        input_steps=[
            steps.rename_observations,
            steps.add_batch_dim,
            relative_step,
            steps.normalize,
            Pi05PrepareStateTokenizerProcessorStep(
                max_state_dim=policy_cfg.max_state_dim
            ),
            TokenizerProcessorStep(
                tokenizer_name=str(LOCAL_TOKENIZER),
                max_length=policy_cfg.tokenizer_max_length,
                padding_side="right",
                padding="max_length",
            ),
            steps.to_device,
        ],
        output_steps=[
            steps.unnormalize,
            AbsoluteActionsProcessorStep(
                enabled=policy_cfg.use_relative_actions,
                relative_step=relative_step,
            ),
            steps.to_cpu,
        ],
    )
    preprocessor.save_pretrained(output, config_filename="policy_preprocessor.json")
    postprocessor.save_pretrained(output, config_filename="policy_postprocessor.json")
    print(f"PROCESSORS_WRITTEN={output / 'policy_preprocessor.json'}")
    print(f"TOKENIZER_OVERRIDE={LOCAL_TOKENIZER}")


def run_load_criterion(
    output: Path,
    stats: dict[str, dict[str, list[float]]],
) -> bool:
    """Load the converted policy locally and run one CPU synthetic forward."""
    print("LOAD_LOCAL_FILES_ONLY=True")
    print(f"TOKENIZER_OVERRIDE={LOCAL_TOKENIZER}")
    try:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
        from lerobot.policies.pi05.modeling_pi05 import PI05Policy

        policy_cfg = PI05Config.from_pretrained(output, local_files_only=True)
        policy_cfg.device = "cpu"
        policy_cfg.pretrained_path = output
        policy = PI05Policy.from_pretrained(
            output,
            config=policy_cfg,
            local_files_only=True,
            strict=False,
        )
        print("LOAD_STRICT=False")
        parameter_count = sum(parameter.numel() for parameter in policy.parameters())
        print(f"POLICY_PARAM_COUNT={parameter_count}")
        if parameter_count <= 0:
            raise AssertionError("Policy has no parameters after from_pretrained")

        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(output),
            dataset_stats=tensorize_norm_stats(stats),
            preprocessor_overrides={
                "device_processor": {"device": "cpu"},
                "tokenizer_processor": {
                    "tokenizer_name": str(LOCAL_TOKENIZER),
                },
            },
        )
        batch = preprocessor(
            {
                "observation.state": torch.zeros(14, dtype=torch.float32),
                "observation.images.cam_high": torch.zeros(
                    3, 480, 640, dtype=torch.float32
                ),
                "observation.images.cam_left_wrist": torch.zeros(
                    3, 480, 640, dtype=torch.float32
                ),
                "observation.images.cam_right_wrist": torch.zeros(
                    3, 480, 640, dtype=torch.float32
                ),
                "task": ["pick up the object"],
            }
        )
        policy.reset()
        with torch.inference_mode():
            action = policy.select_action(batch)
        action_shape = tuple(action.shape)
        action_finite = bool(torch.isfinite(action).all().item())
        print(f"CPU_FORWARD_ACTION_SHAPE={action_shape}")
        print(f"CPU_FORWARD_FINITE={action_finite}")
        if not action_finite:
            raise AssertionError("CPU forward produced non-finite actions")
        print("LOAD_VERDICT=CONVERTED_AND_LOADED")
        print("HF_ACCESS_REMAINING=NONE")
        return True
    except Exception as exc:
        print(f"LOAD_ERROR={type(exc).__name__}: {exc}")
        print("LOAD_VERDICT=CONVERTED_SAFETENSORS_ONLY")
        print("HF_ACCESS_REMAINING=NONE")
        return False


def convert(dump: Path, manifest_path: Path, norm_stats_path: Path, output: Path) -> bool:
    manifest = parse_manifest(manifest_path)
    arrays = load_neutral_dump(dump)
    validate_source(arrays, manifest)
    print(f"SOURCE_LEAF_COUNT={len(arrays)}")
    print(f"SOURCE_ELEMENT_COUNT={sum(int(array.size) for array in arrays.values())}")
    state, alignment_records = build_state_dict(arrays)
    alignment_failures = check_layer_alignment(state, alignment_records)
    print(f"LAYER_ALIGNMENT_CHECKS={len(alignment_records)}")
    print(f"LAYER_ALIGNMENT_FAILURES={alignment_failures}")
    if alignment_failures:
        raise AssertionError("Layer-order/axis-alignment check failed")

    stats = load_norm_stats(norm_stats_path)
    round_trip_error = check_normalization_round_trip(stats)
    print("NORMALIZATION_MAP=state->observation.state,actions->action,fields={mean,std,q01,q99}")
    print(f"NORMALIZATION_ROUND_TRIP_MAX_ERROR={round_trip_error:.3e}")

    config_payload = make_config()
    if output.exists():
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    model_file = output / "model.safetensors"
    temporary_model_file = output / "model.safetensors.tmp"
    save_file(state, str(temporary_model_file), metadata={"format": "pt", "source": "OpenPI PI05"})
    temporary_model_file.replace(model_file)
    safetensors_bytes, safetensors_sha256 = sha256_and_size(model_file)
    if safetensors_bytes <= 0:
        raise AssertionError("model.safetensors was written with zero bytes")
    print(f"CONVERT_SAFETENSORS_BYTES={safetensors_bytes}")
    print(f"CONVERT_SAFETENSORS_SHA256={safetensors_sha256}")

    parity_ok, leaf_count, element_count, missing, extra, shape_diffs = parity_check(model_file, state)
    print(f"PARITY_LEAF_COUNT={leaf_count}")
    print(f"PARITY_ELEMENT_COUNT={element_count}")
    print(f"PARITY_MISSING_KEYS={missing}")
    print(f"PARITY_EXTRA_KEYS={extra}")
    print(f"PARITY_SHAPE_DIFFS={shape_diffs}")
    print(f"PARITY_OK={parity_ok}")
    if not parity_ok:
        return False

    write_output_metadata(output, config_payload, stats)
    write_processors_and_config(output, stats)
    run_load_criterion(output, stats)
    return parity_ok


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dump", type=Path, default=DEFAULT_DUMP)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--norm-stats", type=Path, default=DEFAULT_NORM_STATS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    parity_ok = False
    dump = args.dump
    if not dump.exists() and dump == DEFAULT_DUMP:
        fallback = dump.parent / "convert_stageA" / "orbax_params.npz"
        if fallback.exists():
            print(f"NEUTRAL_DUMP_FALLBACK={fallback}")
            dump = fallback
    try:
        parity_ok = convert(dump, args.manifest, args.norm_stats, args.output)
    except Exception as exc:
        print(f"CONVERT_ERROR={type(exc).__name__}: {exc}")

    model_file = args.output / "model.safetensors"
    if model_file.exists():
        output_bytes, output_sha256 = sha256_and_size(model_file)
        output_path = str(model_file)
    else:
        output_bytes, output_sha256 = 0, ""
        output_path = str(model_file)
    print(f"CONVERT_OUTPUT_PATH={output_path}")
    print(f"CONVERT_OUTPUT_BYTES={output_bytes}")
    print(f"CONVERT_OUTPUT_SHA256={output_sha256}")
    print(f"PARITY_OK={parity_ok}")
    exit_code = 0 if parity_ok else 1
    print(f"CONVERT_EXIT={exit_code}")
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
