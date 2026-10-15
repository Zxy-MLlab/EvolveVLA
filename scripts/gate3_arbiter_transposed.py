#!/usr/bin/env python3
"""Arbitrate converter transposed/einsum keys against the raw Orbax leaves."""

from __future__ import annotations

import numpy as np
from safetensors import safe_open


SOURCE_PATH = "/root/autodl-tmp/EvolveVLA/checkpoints/convert_stageA/orbax_params.npz"
OURS_PATH = (
    "/root/autodl-tmp/EvolveVLA/checkpoints/"
    "pi05_robotwin_lerobot/model.safetensors"
)

SOURCE_BASE = "dict['params']/dict['PaliGemma']"
VISION_PREFIX = (
    "paligemma_with_expert.paligemma.model.vision_tower."
    "vision_model.encoder.layers"
)
LLM_PREFIX = "paligemma_with_expert.paligemma.model.language_model.layers"


def source_leaf(*parts: str) -> str:
    return "/".join(
        [SOURCE_BASE]
        + [f"dict['{part}']" for part in parts]
        + ["dict['value']"]
    )


def max_abs_diff(left: np.ndarray, right: np.ndarray) -> float:
    if left.shape != right.shape:
        raise ValueError(f"shape mismatch: {left.shape} vs {right.shape}")
    return float(np.max(np.abs(left - right)))


def add_checks() -> list[dict]:
    checks: list[dict] = []
    vision_root = (
        "img",
        "Transformer",
        "encoderblock",
        "MultiHeadDotProductAttention_0",
    )
    for layer in (12, 21):
        for source_name, target_name in (
            ("key", "k_proj"),
            ("query", "q_proj"),
            ("value", "v_proj"),
        ):
            checks.append(
                {
                    "label": f"vision L{layer} {target_name}",
                    "leaf": source_leaf(*vision_root, source_name, "kernel"),
                    "slice": lambda array, layer=layer: array[layer],
                    "ours": (
                        f"{VISION_PREFIX}.{layer}.self_attn."
                        f"{target_name}.weight"
                    ),
                    "transform": (
                        "source.reshape(-1, source_dim).T "
                        "(Flax [in, heads, head_dim] -> PyTorch [out, in])"
                    ),
                    "transpose": lambda array, shape: array.reshape(
                        -1, array.shape[0]
                    ).T,
                }
            )
        checks.append(
            {
                "label": f"vision L{layer} out_proj",
                "leaf": source_leaf(*vision_root, "out", "kernel"),
                "slice": lambda array, layer=layer: array[layer],
                "ours": f"{VISION_PREFIX}.{layer}.self_attn.out_proj.weight",
                "transform": (
                    "source.reshape(-1, source_dim).T "
                    "(Flax [heads, head_dim, out] -> PyTorch [out, in])"
                ),
                "transpose": lambda array, shape: array.reshape(
                    -1, array.shape[-1]
                ).T,
            }
        )

    llm_root = ("llm", "layers", "attn")
    for layer in (0, 9):
        checks.append(
            {
                "label": f"LLM L{layer} q_proj",
                "leaf": source_leaf(*llm_root, "q_einsum", "w"),
                "slice": lambda array, layer=layer: array[layer],
                "ours": f"{LLM_PREFIX}.{layer}.self_attn.q_proj.weight",
                "transform": (
                    "source.transpose(0, 2, 1).reshape(2048, 2048)"
                ),
                "transpose": lambda array, shape: array.transpose(0, 2, 1).reshape(
                    shape
                ),
            }
        )
        checks.append(
            {
                "label": f"LLM L{layer} o_proj",
                "leaf": source_leaf(*llm_root, "attn_vec_einsum", "w"),
                "slice": lambda array, layer=layer: array[layer],
                "ours": f"{LLM_PREFIX}.{layer}.self_attn.o_proj.weight",
                "transform": (
                    "source.transpose(2, 0, 1).reshape(2048, 2048)"
                ),
                "transpose": lambda array, shape: array.transpose(2, 0, 1).reshape(
                    shape
                ),
            }
        )
        for packed_index, target_name in ((0, "k_proj"), (1, "v_proj")):
            checks.append(
                {
                    "label": (
                        f"LLM L{layer} {target_name} "
                        f"from kv_einsum[{packed_index},0]"
                    ),
                    "leaf": source_leaf(*llm_root, "kv_einsum", "w"),
                    "slice": lambda array, layer=layer, packed_index=packed_index: (
                        array[layer, packed_index, 0]
                    ),
                    "ours": f"{LLM_PREFIX}.{layer}.self_attn.{target_name}.weight",
                    "transform": (
                        f"source[{packed_index}, 0].T "
                        "(kv_einsum pack: [0,0]=K, [1,0]=V)"
                    ),
                    "transpose": lambda array, shape: array.T,
                }
            )
    return checks


def main() -> int:
    checks = add_checks()
    source_names: dict[str, tuple[int, ...]] = {}
    results: list[tuple[dict, bool, bool]] = []

    print(f"SOURCE_PATH={SOURCE_PATH}")
    print(f"OURS_PATH={OURS_PATH}")
    print("TRANSFORM_CONVENTION=compare direct flatten reshape against the")
    print("TRANSFORM_CONVENTION=converter's explicit axis transpose/reshape")
    print("TRANSFORM_CONVENTION=all exactness verdicts require np.array_equal")

    with np.load(SOURCE_PATH, allow_pickle=False) as source:
        with safe_open(OURS_PATH, framework="pt", device="cpu") as ours_file:
            print(f"SOURCE_ARCHIVE_LEAF_COUNT={len(source.files)}")
            for check in checks:
                leaf_array = np.asarray(source[check["leaf"]])
                source_names[check["leaf"]] = tuple(leaf_array.shape)
                source_value = np.asarray(check["slice"](leaf_array))
                ours_value = ours_file.get_tensor(check["ours"]).detach().cpu().numpy()
                if source_value.size != ours_value.size:
                    raise ValueError(
                        f"{check['label']}: element count mismatch "
                        f"{source_value.size} vs {ours_value.size}"
                    )

                direct_value = source_value.reshape(ours_value.shape)
                transposed_value = np.asarray(
                    check["transpose"](source_value, ours_value.shape)
                )
                direct_diff = max_abs_diff(ours_value, direct_value)
                transposed_diff = max_abs_diff(ours_value, transposed_value)
                direct_exact = np.array_equal(ours_value, direct_value)
                transposed_exact = np.array_equal(ours_value, transposed_value)
                if transposed_exact and direct_exact:
                    verdict = "MATCH-direct+transposed"
                elif transposed_exact:
                    verdict = "MATCH-transposed"
                elif direct_exact:
                    verdict = "MATCH-direct"
                else:
                    verdict = "MISMATCH"
                print(
                    f"{check['label']} | SOURCE shape={source_value.shape} | "
                    f"OURS shape={ours_value.shape} | "
                    f"max|OURS-SOURCE|={direct_diff:.9g} | "
                    f"max|OURS-SOURCE.T|={transposed_diff:.9g} | "
                    f"verdict={verdict}"
                )
                print(
                    f"  source_leaf={check['leaf']} | ours_key={check['ours']} | "
                    f"transform={check['transform']}"
                )
                results.append((check, direct_exact, transposed_exact))

    print("SOURCE_LEAVES_USED:")
    for leaf_name, shape in source_names.items():
        print(f"  {leaf_name} | shape={shape}")

    mismatch_count = sum(not direct and not transposed for _, direct, transposed in results)
    transposed_count = sum(transposed for _, _, transposed in results)
    print(f"CHECK_COUNT={len(results)}")
    print(f"TRANSPOSED_EXACT_COUNT={transposed_count}")
    print(f"MISMATCH_COUNT={mismatch_count}")
    if transposed_count == len(results) and mismatch_count == 0:
        print("OVERALL_VERDICT=SOURCE_EXACT_ON_TRANSPOSED_KEYS=yes")
        return 0
    print("OVERALL_VERDICT=SOURCE_EXACT_ON_TRANSPOSED_KEYS=no")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
