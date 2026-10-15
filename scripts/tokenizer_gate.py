#!/usr/bin/env python3
"""Persisted equivalence gate for the official pi0.5 tokenizer."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from safetensors import safe_open
from transformers import GemmaTokenizer, GemmaTokenizerFast


TOKENIZER_PATH = Path("/root/autodl-tmp/paligemma_tokenizer")
MODEL_PATH = Path(
    "/root/autodl-tmp/.cache/huggingface/hub/models--lerobot--pi05_robocasa"
)
MODEL_FILE = MODEL_PATH / "model.safetensors"
EMBEDDING_ROWS = 257152
PROMPTS = ("close the fridge", "open the cabinet", "turn off the stove")


def fail(message: str) -> None:
    print(f"TOKENIZER GUARD FAILED: {message}", file=sys.stderr)
    raise SystemExit(1)


def load_tokenizers():
    if not TOKENIZER_PATH.is_dir():
        fail(f"official tokenizer directory is missing: {TOKENIZER_PATH}")
    try:
        slow = GemmaTokenizer.from_pretrained(
            TOKENIZER_PATH, local_files_only=True
        )
        fast = GemmaTokenizerFast.from_pretrained(
            TOKENIZER_PATH, local_files_only=True
        )
    except Exception as exc:
        fail(f"could not load slow and fast tokenizers: {exc}")
    return slow, fast


def check_tokenizers(slow, fast) -> dict[str, int]:
    if len(slow) != EMBEDDING_ROWS:
        fail(f"slow tokenizer length {len(slow)} != {EMBEDDING_ROWS}")
    if len(fast) != EMBEDDING_ROWS:
        fail(f"fast tokenizer length {len(fast)} != {EMBEDDING_ROWS}")

    names = ("<loc0000>", "<seg000>", "<seg127>")
    slow_ids = {name: slow.convert_tokens_to_ids(name) for name in names}
    fast_ids = {name: fast.convert_tokens_to_ids(name) for name in names}
    if slow_ids != fast_ids:
        fail(f"slow/fast spatial token ids differ: slow={slow_ids} fast={fast_ids}")

    if slow_ids["<loc0000>"] == slow.unk_token_id:
        fail(f"<loc0000> is UNK ({slow.unk_token_id})")
    if slow_ids["<seg000>"] == slow.unk_token_id:
        fail(f"<seg000> is UNK ({slow.unk_token_id})")

    image_slow = slow.convert_tokens_to_ids("<image>")
    image_fast = fast.convert_tokens_to_ids("<image>")
    print(f"len(slow)={len(slow)}")
    print(f"len(fast)={len(fast)}")
    print(f"slow ids: loc0000={slow_ids['<loc0000>']} seg000={slow_ids['<seg000>']} seg127={slow_ids['<seg127>']}")
    print(f"fast ids: loc0000={fast_ids['<loc0000>']} seg000={fast_ids['<seg000>']} seg127={fast_ids['<seg127>']}")
    print(f"unk_token_id={slow.unk_token_id}")
    print(f"slow <image> id={image_slow}")
    print(f"fast <image> id={image_fast}")
    return {
        "slow_len": len(slow),
        "fast_len": len(fast),
        "loc0000": slow_ids["<loc0000>"],
        "seg000": slow_ids["<seg000>"],
        "seg127": slow_ids["<seg127>"],
    }


def check_embedding_rows() -> int:
    if not MODEL_FILE.is_file():
        fail(f"model file is missing: {MODEL_FILE}")
    try:
        with safe_open(MODEL_FILE, framework="pt", device="cpu") as tensors:
            keys = [key for key in tensors.keys() if key.endswith("embed_tokens.weight")]
            if len(keys) != 1:
                fail(f"expected one embed_tokens.weight key, found {keys}")
            shape = tuple(tensors.get_tensor(keys[0]).shape)
    except SystemExit:
        raise
    except Exception as exc:
        fail(f"could not inspect {MODEL_FILE}: {exc}")
    if shape[0] != EMBEDDING_ROWS:
        fail(f"{keys[0]} rows {shape[0]} != {EMBEDDING_ROWS}")
    print(f"embed_tokens.weight key={keys[0]}")
    print(f"embed_tokens.weight shape={shape}")
    print(f"embed_tokens rows={shape[0]}")
    return shape[0]


def check_actual_lerobot_processor() -> int:
    try:
        from lerobot.policies.factory import make_pre_post_processors
        from lerobot.policies.pi05.configuration_pi05 import PI05Config
    except Exception as exc:
        fail(f"could not import LeRobot pi0.5 processor: {exc}")

    try:
        policy_cfg = PI05Config.from_pretrained(
            MODEL_PATH, local_files_only=True
        )
        policy_cfg.device = "cpu"
        preprocessor, _ = make_pre_post_processors(
            policy_cfg=policy_cfg,
            pretrained_path=str(MODEL_PATH),
            preprocessor_overrides={
                "tokenizer_processor": {
                    "tokenizer_name": str(TOKENIZER_PATH)
                },
                "device_processor": {"device": "cpu"},
            },
        )
    except Exception as exc:
        fail(f"could not build actual LeRobot pi0.5 preprocessor: {exc}")

    max_prompt_id = -1
    token_keys: set[str] = set()
    for prompt in PROMPTS:
        try:
            batch = {
                "observation.state": torch.zeros(16, dtype=torch.float32),
                "task": [prompt],
            }
            processed = preprocessor(batch)
        except Exception as exc:
            fail(f"processor failed for prompt {prompt!r}: {exc}")

        candidates = {
            str(key): value
            for key, value in processed.items()
            if "token" in str(key).lower()
            and torch.is_tensor(value)
            and value.dtype in (torch.int8, torch.int16, torch.int32, torch.int64)
        }
        if not candidates:
            fail(
                f"processor produced no integer token tensor for prompt {prompt!r}; "
                f"keys={sorted(map(str, processed))}"
            )
        token_keys.update(candidates)
        prompt_max = max(int(value.max().item()) for value in candidates.values())
        prompt_min = min(int(value.min().item()) for value in candidates.values())
        max_prompt_id = max(max_prompt_id, prompt_max)
        print(
            f"prompt={prompt!r} token_keys={sorted(candidates)} "
            f"min_token_id={prompt_min} max_token_id={prompt_max}"
        )
        if prompt_max > EMBEDDING_ROWS - 1:
            fail(
                f"prompt {prompt!r} produced token id {prompt_max}; "
                f"every prompt token must be <= {EMBEDDING_ROWS - 1}"
            )

    print(f"processor token keys={sorted(token_keys)}")
    print(f"max prompt token id={max_prompt_id}")
    return max_prompt_id


def main() -> None:
    slow, fast = load_tokenizers()
    check_tokenizers(slow, fast)
    check_embedding_rows()
    check_actual_lerobot_processor()
    print("TOKENIZER GUARD PASSED")


if __name__ == "__main__":
    main()
