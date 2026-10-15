#!/usr/bin/env python3
"""D181 Leg B: official-checkpoint four-way state-conditioning intervention."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import torch

ROOT = Path("/root/autodl-tmp/EvolveVLA")
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from gate3_shared_noise_nose2nose import (
    ACTION_DIM,
    ARM_DIMS,
    DEFAULT_DATA_ROOT,
    DEFAULT_OFFICIAL,
    DEFAULT_TOKENIZER,
    EPISODE,
    POSITION,
    SEED,
    TASK,
    build_policy_batch,
    format_vector,
    image_hashes,
    load_checkpoint_stats,
    load_harness,
    load_policy_and_processors,
    load_sample,
    make_shared_noise,
    quantile_forward,
    quantile_inverse,
    raw_observation,
    noise_sanity,
    run_normalized_first_action,
    assert_state_reproducible,
    one_obs_verbose,
    start_heartbeat_watchdog,
    state_key,
    state_token_ids,
    token_key,
)


def subset_state_quantiles(data_root: Path) -> tuple[np.ndarray, np.ndarray]:
    # The "anomalous" reference is our subset-derived cached data (98,122
    # rows), not a confirmed defect in the full official training corpus.
    data_path = data_root / "data/chunk-000/file-000.parquet"
    table = pq.read_table(data_path, columns=["observation.state"]).to_pydict()
    states = np.asarray(table["observation.state"], dtype=np.float32).reshape(-1, ACTION_DIM)
    if states.shape[0] != 98122:
        print(f"DATA_SUBSET_ROWS_OBSERVED={states.shape[0]}", flush=True)
    return (
        np.quantile(states, 0.01, axis=0).astype(np.float32),
        np.quantile(states, 0.99, axis=0).astype(np.float32),
    )


def normalized_state_variants(
    raw_state: np.ndarray,
    official_state_stats: dict[str, Any],
    data_q01: np.ndarray,
    data_q99: np.ndarray,
) -> dict[str, np.ndarray]:
    real = quantile_forward(raw_state, official_state_stats)
    zeroed = np.zeros(ACTION_DIM, dtype=np.float32)
    rng = np.random.default_rng(SEED + 17)
    official_mean = np.asarray(official_state_stats["mean"], dtype=np.float32)
    official_std = np.asarray(official_state_stats["std"], dtype=np.float32)
    official_q01 = np.asarray(official_state_stats["q01"], dtype=np.float32)
    official_q99 = np.asarray(official_state_stats["q99"], dtype=np.float32)
    random_raw = np.clip(
        rng.normal(official_mean, official_std).astype(np.float32),
        official_q01,
        official_q99,
    )
    random_state = quantile_forward(random_raw, official_state_stats)
    data_stats_normalized = np.clip(
        2.0 * (raw_state - data_q01) / (data_q99 - data_q01) - 1.0,
        -1.0,
        1.0,
    ).astype(np.float32)
    return {
        "real": real,
        "zeroed": zeroed,
        "random": random_state,
        "data_stats_normalized": data_stats_normalized,
    }


def assert_tokens_otherwise_equal(
    real_tokens: np.ndarray,
    variant_tokens: np.ndarray,
    label: str,
) -> None:
    real = np.asarray(real_tokens).reshape(-1)
    variant = np.asarray(variant_tokens).reshape(-1)
    if real.shape != variant.shape:
        raise AssertionError(f"{label}: token shapes differ {real.shape} != {variant.shape}")
    differences = np.flatnonzero(real != variant)
    if differences.size == 0:
        raise AssertionError(f"{label}: state intervention produced no token change")
    first = int(differences[0])
    last = int(differences[-1])
    common_prefix = first
    common_suffix = int(real.size - 1 - last)
    if common_prefix < 8 or common_suffix < 2:
        raise AssertionError(
            f"{label}: token differences are not confined to the state substring "
            f"(common_prefix={common_prefix}, common_suffix={common_suffix})"
        )


def cosine(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    denom = np.linalg.norm(left) * np.linalg.norm(right)
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)


def arm_l2(action: np.ndarray) -> float:
    return float(np.linalg.norm(np.asarray(action)[ARM_DIMS]))


def smoke(allow_cuda: bool = False) -> int:
    if torch.cuda.is_available() and not allow_cuda:
        raise RuntimeError("D181 smoke requires CUDA_VISIBLE_DEVICES=''")
    sample = load_sample()
    official_stats = load_checkpoint_stats(DEFAULT_OFFICIAL)
    data_q01, data_q99 = subset_state_quantiles(DEFAULT_DATA_ROOT)
    variants = normalized_state_variants(
        sample["state"], official_stats["state"], data_q01, data_q99
    )
    token_sets = {name: state_token_ids(value).tolist() for name, value in variants.items()}
    if np.array_equal(token_sets["real"], token_sets["zeroed"]):
        raise AssertionError("real and zeroed state token IDs unexpectedly coincide")
    if np.array_equal(token_sets["real"], token_sets["random"]):
        raise AssertionError("real and random state token IDs unexpectedly coincide")
    if np.array_equal(token_sets["real"], token_sets["data_stats_normalized"]):
        raise AssertionError("real and data-stats-normalized token IDs unexpectedly coincide")
    print("SMOKE_CONSTRUCTED_MATCHED_OBS=PASS")
    print(f"SMOKE_EPISODE={EPISODE} POSITION={POSITION} VIDEO_INDEX={sample['video_index']}")
    print("SMOKE_NO_MODEL_FORWARD=PASS")
    print("SMOKE_NO_CUDA=PASS")
    for name in ("real", "zeroed", "random", "data_stats_normalized"):
        print(f"STATE_TOKEN_IDS[{name}]={token_sets[name]}")
    print("SMOKE_FOUR_STATE_TOKEN_SETS=PASS")
    print("SMOKE_NOISE_INPUT=DEFERRED_TO_LEG_A_SIGNATURE_CHECK")
    return 0


def run(args: argparse.Namespace) -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("Leg B requires CUDA; use --smoke for CPU-only verification")
    if torch.cuda.device_count() != 1:
        raise RuntimeError(f"expected one visible GPU, got {torch.cuda.device_count()}")
    if os.environ.get("ROBOTWIN_SKIP_EXPERT") != "0":
        raise RuntimeError("ROBOTWIN_SKIP_EXPERT must be exactly 0")
    start_heartbeat_watchdog()

    harness = load_harness()
    sample = load_sample(args.data_root)
    official_stats = load_checkpoint_stats(args.official_model)
    data_q01, data_q99 = subset_state_quantiles(args.data_root)
    variants = normalized_state_variants(
        sample["state"], official_stats["state"], data_q01, data_q99
    )

    # Do not inject the peer's state here: peer normalization is a checkpoint
    # confound, while Leg B isolates the official conditioning-input path.
    loaded = load_policy_and_processors(harness, args.official_model, args.tokenizer)
    policy, preprocessor, _post, env_pre, _env_post = loaded

    batches: dict[str, dict[str, Any]] = {}
    token_sets: dict[str, list[int]] = {}
    image_hash_set: dict[str, dict[str, str]] = {}
    real_tokens: np.ndarray | None = None
    for name, normalized in variants.items():
        raw_state = (
            sample["state"]
            if name == "real"
            else quantile_inverse(normalized, official_stats["state"])
        )
        batch = build_policy_batch(
            harness,
            sample,
            raw_state,
            sample["instruction"],
            env_pre,
            preprocessor,
            policy.config,
        )
        batches[name] = batch
        image_hash_set[name] = image_hashes(batch)
        token_array = batch[token_key(batch)].detach().cpu().numpy().reshape(-1)
        token_sets[name] = state_token_ids(normalized).tolist()
        if name == "real":
            real_tokens = token_array
        else:
            assert real_tokens is not None
            assert_tokens_otherwise_equal(real_tokens, token_array, name)
        observed_state = batch[state_key(batch)].detach().float().cpu().numpy().reshape(ACTION_DIM)
        assert_state_reproducible(
            f"Leg B {name}", raw_state, official_stats["state"], observed_state
        )

    if any(image_hash_set[name] != image_hash_set["real"] for name in variants):
        raise AssertionError("image hashes changed across state-only interventions")

    shared_noise = make_shared_noise(policy, args.seed)
    outputs = {
        name: run_normalized_first_action(
            policy, batches[name], shared_noise, args.num_inference_steps
        )
        for name in variants
    }
    action_stats = official_stats["action"]
    recorded_normalized = quantile_forward(sample["recorded_action"], action_stats)
    real_output = outputs["real"]
    real_l2 = arm_l2(real_output)
    recorded_l2 = arm_l2(recorded_normalized)
    metrics: dict[str, dict[str, Any]] = {}
    for name, output in outputs.items():
        metrics[name] = {
            "normalized_action": output.tolist(),
            "abs_per_dim": np.abs(output).tolist(),
            "arm_l2": arm_l2(output),
            "cos_vs_recorded_demo": cosine(output[ARM_DIMS], recorded_normalized[ARM_DIMS]),
            "cos_vs_real_state_output": cosine(output[ARM_DIMS], real_output[ARM_DIMS]),
            "delta_l2_vs_real": float(np.linalg.norm(output[ARM_DIMS] - real_output[ARM_DIMS])),
        }

    zero_alive = metrics["zeroed"]["delta_l2_vs_real"] > args.state_delta_threshold
    random_alive = metrics["random"]["delta_l2_vs_real"] > args.state_delta_threshold
    if zero_alive or random_alive:
        state_path_verdict = "state path ALIVE"
    else:
        state_path_verdict = "INCONCLUSIVE (conditioning OR weights)"
    data_error = abs(metrics["data_stats_normalized"]["arm_l2"] - recorded_l2)
    real_error = abs(real_l2 - recorded_l2)
    data_restores = (
        metrics["data_stats_normalized"]["arm_l2"] >= max(2.0 * real_l2, 0.5 * recorded_l2)
        and data_error < real_error
    )
    if data_restores:
        intervention_verdict = "STATE-STAT MIS-SCALING (conditioning-input bug)"
    else:
        intervention_verdict = state_path_verdict

    result = {
        "sample": {"episode": EPISODE, "position": POSITION, "video_index": sample["video_index"]},
        "official_model": str(args.official_model),
        "official_state_normalization": "QUANTILES",
        "cached_data_reference": {
            "path": str(args.data_root),
            "rows_expected": 98122,
            "q01": data_q01.tolist(),
            "q99": data_q99.tolist(),
            "caveat": "subset-derived candidate intervention, not a confirmed defect",
        },
        "state_token_ids": token_sets,
        "matched_image_hashes": image_hash_set,
        "shared_noise_sha256": hashlib.sha256(shared_noise.detach().cpu().numpy().tobytes()).hexdigest(),
        "shared_noise_dtype": str(shared_noise.dtype),
        "num_inference_steps": args.num_inference_steps,
        "recorded_normalized_action": recorded_normalized.tolist(),
        "recorded_arm_l2": recorded_l2,
        "conditions": metrics,
        "zero_or_random_state_path": state_path_verdict,
        "data_stats_normalized_restores_magnitude": data_restores,
        "verdict": intervention_verdict,
    }
    print(json.dumps(result, indent=2, sort_keys=True))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--allow-cuda-smoke",
        action="store_true",
        help="opt out of the construction-smoke CUDA guard; still no model forward",
    )
    parser.add_argument(
        "--noise-sanity",
        action="store_true",
        help="execute three matched model forwards with injected shared noise",
    )
    parser.add_argument("--official-model", type=Path, default=DEFAULT_OFFICIAL)
    parser.add_argument("--tokenizer", type=Path, default=DEFAULT_TOKENIZER)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--num-inference-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--state-delta-threshold", type=float, default=1e-4)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--one-obs-verbose", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    cli_args = parse_args()
    if cli_args.one_obs_verbose:
        raise SystemExit(one_obs_verbose(cli_args))
    if cli_args.noise_sanity:
        raise SystemExit(noise_sanity(cli_args))
    raise SystemExit(
        smoke(allow_cuda=cli_args.allow_cuda_smoke)
        if cli_args.smoke
        else run(cli_args)
    )
